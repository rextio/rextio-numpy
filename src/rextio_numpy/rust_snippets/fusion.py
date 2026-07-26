"""Rust helper-text for multi-op elementwise chain fusion.

Generates one helper per (tree, dtype, rank[, leaf-alias]) signature:

1. When every leaf is statically the same rank as the result, prefer an
   equal-shape standard-layout (C-order) fast path that is decided and entered
   *before* any LTR postorder ``__rxtnp_broadcast_shape`` ``Vec`` work: require
   equal leaf shapes and ``is_standard_layout``, load via ``as_slice``, and fill
   one output. F64 rank-1 fills a fresh NumPy-owned sink directly; other lanes
   use ``from_shape_fn``. This is a safe, layout-gated shortcut only (not a
   speed claim). F64 rank-1 preserves the Python view's layout, while other
   boundary ``to_owned()`` conversions do not guarantee C-contiguous layout;
   non-standard-layout leaves at helper entry therefore still take the generic
   path. Mixed-rank trees never emit this gate.
2. Otherwise (and on fast-path fallthrough) the generic path validates broadcast
   shapes for each internal binop in left-to-right postorder (same order as
   NumPy evaluation), raising NumPy's trailing-space ValueError on the first
   mismatch; then broadcasts each leaf view to the final static ``Ix1`` /
   ``Ix2`` shape (no intermediate owned ndarrays) and uses the same single
   output allocation + data pass. Errors, evaluation order, and exact messages
   are unchanged from the pre-fast-path helper.
3. When lower-time ``leaf_operands`` names prove that two leaf occurrences are
   the same binding, the helper takes one parameter per unique name and reuses
   loads/views. Alias patterns are encoded in the helper name so shared
   structural signatures cannot collide across different alias maps.
4. Scalar temps inside the element closure preserve AST evaluation order. No
   reassociation, constant folding, or FMA. i64 uses wrapping arithmetic at
   every intermediate node. No ``Zip`` arity dependency (ndarray's
   ZippableTuple tops out at six producers, while 8-binop trees need up to
   nine leaf occurrences).
"""

from __future__ import annotations

from typing import TypeAlias

from rextio_numpy.claim.fusion import OP_NAMES
from rextio_numpy.rust_snippets.array_repr import (
    F64_1D_OUTPUT_HELPER_NAME,
    array_rust_type,
    f64_1d_output_helper,
    is_python_backed,
    lifetime_decl,
    readonly_view_line,
)
from rextio_numpy.rust_snippets.elementwise import (
    _WRAP_OP,
    shared_broadcast_helpers,
)

_OP_SYMBOL = {"+": "+", "-": "-", "*": "*", "/": "/"}

# Side of a tree-plan node: leaf occurrence or intermediate temp index.
Side: TypeAlias = tuple[str, int]
TreeStep: TypeAlias = tuple[str, Side, Side]
# leaf occurrence index -> first occurrence index of the same binding.
LeafAliasMap: TypeAlias = tuple[int, ...]


def leaf_alias_map(leaf_operands: tuple[str, ...] | None, *, n_leaves: int) -> LeafAliasMap:
    """Map each leaf occurrence to the first index sharing its operand name.

    Without *leaf_operands*, the identity map is returned (no deduplication).
    Identity is proven only by exact lower-time rust operand-name equality.
    """
    if n_leaves < 0:
        raise ValueError(f"rextio-numpy fusion: bad n_leaves {n_leaves}")
    if leaf_operands is None:
        return tuple(range(n_leaves))
    if len(leaf_operands) != n_leaves:
        raise ValueError(
            "rextio-numpy fusion: leaf_operands length "
            f"{len(leaf_operands)} != n_leaves {n_leaves}"
        )
    first: dict[str, int] = {}
    out: list[int] = []
    for i, name in enumerate(leaf_operands):
        if name not in first:
            first[name] = i
        out.append(first[name])
    return tuple(out)


def fusion_call_name(
    signature: str,
    leaf_alias: LeafAliasMap | None = None,
) -> str:
    """Return the Rust helper name for a fusion signature.

    The signature is already a stable identifier-safe encoding of the tree
    (dtype, ranks, ops, leaf indexes) — never a Python hash. When *leaf_alias*
    is a non-identity map, a deterministic suffix is appended so specialized
    helpers cannot collide with the identity (or other alias) variants.
    """
    safe = (
        signature.replace("+", "p")
        .replace("-", "m")
        .replace("*", "t")
        .replace("/", "d")
        .replace(".", "_")
    )
    name = f"__rxtnp_echain_{safe}"
    if leaf_alias is None or leaf_alias == tuple(range(len(leaf_alias))):
        return name
    tag = "_".join(str(i) for i in leaf_alias)
    return f"{name}_al_{tag}"


def fusion_helper(
    *,
    signature: str,
    dtype: str,
    result_rank: int,
    leaf_ranks: tuple[int, ...],
    expression_ops_postorder: tuple[str, ...],
    tree_plan: list[TreeStep],
    leaf_operands: tuple[str, ...] | None = None,
) -> str:
    """Return the fused helper source for one expression tree.

    ``tree_plan`` is a LTR postorder list of ``(op, left, right)`` where each
    side is either ``("leaf", index)`` or ``("temp", index)`` identifying a
    previously emitted intermediate temp.

    When *leaf_operands* proves repeated names, parameters and loads are
    deduplicated via :func:`leaf_alias_map`.
    """
    if result_rank not in (1, 2):
        raise ValueError(f"rextio-numpy fusion: unsupported result rank {result_rank}")
    n_leaves = len(leaf_ranks)
    alias = leaf_alias_map(leaf_operands, n_leaves=n_leaves)
    # Unique parameter slots in first-occurrence order.
    unique_idxs = tuple(dict.fromkeys(alias))
    name = fusion_call_name(signature, alias)
    params = ", ".join(
        f"a{p}: &{array_rust_type(dtype, leaf_ranks[occ])}"
        for p, occ in enumerate(unique_idxs)
    )
    out_ty = array_rust_type(dtype, result_rank)
    python_output = is_python_backed(dtype, result_rank)
    helper_types = tuple((dtype, leaf_ranks[occ]) for occ in unique_idxs)
    lifetime = lifetime_decl(*helper_types, (dtype, result_rank))
    py_param = "py: pyo3::Python<'py>, " if python_output else ""
    view_lines = "".join(
        readonly_view_line(f"a{p}", dtype, leaf_ranks[occ])
        for p, occ in enumerate(unique_idxs)
    )

    # occurrence i -> parameter slot p (a{p})
    occ_to_param = {occ: p for p, occ in enumerate(unique_idxs)}
    param_of = [occ_to_param[alias[i]] for i in range(n_leaves)]

    # --- arithmetic body (shared by both paths; indent adjusted per path) ---
    arith_core: list[str] = []
    temp_count = 0
    for step, (op, left, right) in enumerate(tree_plan):
        left_val = _value_ref(left)
        right_val = _value_ref(right)
        expr = _arith_expr(dtype, op, left_val, right_val)
        is_last = step == len(tree_plan) - 1
        if is_last:
            arith_core.append(expr)
        else:
            arith_core.append(f"let t{temp_count} = {expr};")
            temp_count += 1
    # Fast path closure body is nested deeper than the generic path.
    body_arith_fast = "\n".join(f"                {line}" for line in arith_core)
    body_arith_generic = "\n".join(f"        {line}" for line in arith_core)
    direct_prefix = arith_core[:-1]
    direct_result = arith_core[-1]
    body_arith_fast_direct = "\n".join(
        [
            *(f"                    {line}" for line in direct_prefix),
            f"                    out[i] = {direct_result};",
        ]
    )
    body_arith_generic_direct = "\n".join(
        [
            *(f"            {line}" for line in direct_prefix),
            f"            out[i] = {direct_result};",
        ]
    )

    # Equal-shape fast path is only emitted when every leaf rank equals the
    # result rank (mixed-rank trees cannot be exact equal-shape).
    emit_fast = bool(leaf_ranks) and all(r == result_rank for r in leaf_ranks)

    # --- generic path: LTR postorder shape validation (unchanged semantics) ---
    shape_lines: list[str] = []
    node_shape_var: list[str] = []
    for step, (op, left, right) in enumerate(tree_plan):
        del op  # validation is shape-only; op used in arithmetic section.
        left_expr = _shape_ref(left, node_shape_var, param_of)
        right_expr = _shape_ref(right, node_shape_var, param_of)
        var = f"s{step}"
        shape_lines.append(f"    let {var} = __rxtnp_broadcast_shape({left_expr}, {right_expr})?;")
        node_shape_var.append(var)
    out_shape = node_shape_var[-1]

    n_params = len(unique_idxs)

    if result_rank == 1:
        dim_setup = (
            f"    if {out_shape}.len() != 1 {{\n"
            f"        return Err(pyo3::exceptions::PyValueError::new_err(\n"
            f'            "rextio-numpy: fused rank-1 result has non-1-D shape"\n'
            f"        ));\n"
            f"    }}\n"
            f"    let dim = numpy::ndarray::Ix1({out_shape}[0]);"
        )
        broadcast_lines = [
            f"    let v{p} = a{p}.broadcast(dim).ok_or_else(|| {{\n"
            f"        pyo3::exceptions::PyValueError::new_err(format!(\n"
            f'            "operands could not be broadcast together with shapes {{}} {{}} ",\n'
            f"            __rxtnp_fmt_shape(a{p}.shape()), __rxtnp_fmt_shape({out_shape}.as_slice())\n"
            f"        ))\n"
            f"    }})?;"
            for p in range(n_params)
        ]
        if python_output:
            generic_loads = _emit_loads(
                n_leaves=n_leaves,
                param_of=param_of,
                load_expr=lambda p: f"v{p}[i]",
                indent="            ",
            )
            generic_header = (
                f"    {F64_1D_OUTPUT_HELPER_NAME}(py, {out_shape}[0], |out| {{\n"
                "        for i in 0..out.len() {"
            )
            generic_tail = ("        }\n        Ok(())", "    })")
        else:
            generic_loads = _emit_loads(
                n_leaves=n_leaves,
                param_of=param_of,
                load_expr=lambda p: f"v{p}[i]",
                indent="        ",
            )
            generic_header = "    let out = numpy::ndarray::Array1::from_shape_fn(dim, |i| {"
            generic_tail = ("    });", "    Ok(out)")
        fast_body = (
            _emit_fast_path_rank1(
                n_params=n_params,
                n_leaves=n_leaves,
                param_of=param_of,
                body_arith_fast=body_arith_fast,
                body_arith_fast_direct=body_arith_fast_direct,
                python_output=python_output,
            )
            if emit_fast
            else ""
        )
    else:
        dim_setup = (
            f"    if {out_shape}.len() != 2 {{\n"
            f"        return Err(pyo3::exceptions::PyValueError::new_err(\n"
            f'            "rextio-numpy: fused rank-2 result has non-2-D shape"\n'
            f"        ));\n"
            f"    }}\n"
            f"    let dim = numpy::ndarray::Ix2({out_shape}[0], {out_shape}[1]);"
        )
        broadcast_lines = [
            f"    let v{p} = a{p}.broadcast(dim).ok_or_else(|| {{\n"
            f"        pyo3::exceptions::PyValueError::new_err(format!(\n"
            f'            "operands could not be broadcast together with shapes {{}} {{}} ",\n'
            f"            __rxtnp_fmt_shape(a{p}.shape()), __rxtnp_fmt_shape({out_shape}.as_slice())\n"
            f"        ))\n"
            f"    }})?;"
            for p in range(n_params)
        ]
        generic_loads = _emit_loads(
            n_leaves=n_leaves,
            param_of=param_of,
            load_expr=lambda p: f"v{p}[[i, j]]",
            indent="        ",
        )
        generic_header = (
            "    let out = numpy::ndarray::Array2::from_shape_fn("
            f"({out_shape}[0], {out_shape}[1]), |(i, j)| {{"
        )
        generic_tail = ("    });", "    Ok(out)")
        fast_body = (
            _emit_fast_path_rank2(
                n_params=n_params,
                n_leaves=n_leaves,
                param_of=param_of,
                body_arith_fast=body_arith_fast,
            )
            if emit_fast
            else ""
        )

    lines = [
        f"fn {name}{lifetime}({py_param}{params}) -> pyo3::PyResult<{out_ty}> {{",
    ]
    if view_lines:
        lines.append(view_lines.rstrip())
    if fast_body:
        lines.append(fast_body)
    lines.extend(shape_lines)
    lines.append(dim_setup)
    lines.extend(broadcast_lines)
    lines.append(generic_header)
    lines.append(generic_loads)
    lines.append(body_arith_generic_direct if python_output else body_arith_generic)
    lines.extend(generic_tail)
    lines.append("}")
    # Silence unused: expression_ops_postorder is validated by caller.
    del expression_ops_postorder
    return "\n".join(lines)


def fusion_helpers_bundle(
    *,
    signature: str,
    dtype: str,
    result_rank: int,
    leaf_ranks: tuple[int, ...],
    expression_ops_postorder: tuple[str, ...],
    tree_plan: list[TreeStep],
    leaf_operands: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Return shared broadcast helpers plus the fused helper."""
    helper = fusion_helper(
        signature=signature,
        dtype=dtype,
        result_rank=result_rank,
        leaf_ranks=leaf_ranks,
        expression_ops_postorder=expression_ops_postorder,
        tree_plan=tree_plan,
        leaf_operands=leaf_operands,
    )
    output_support = (
        (f64_1d_output_helper(),)
        if is_python_backed(dtype, result_rank)
        else ()
    )
    return (*output_support, *shared_broadcast_helpers(), helper)


def build_tree_plan(expression: object) -> list[TreeStep]:
    """Walk a ClaimExpr-like tree into LTR postorder (op, left, right) plan.

    Each side is ``("leaf", index)`` or ``("temp", temp_index)``.
    """
    plan: list[TreeStep] = []
    temp_i = [0]

    def walk(node: object) -> Side:
        kind = getattr(node, "kind", None)
        if kind == "leaf":
            idx = getattr(node, "leaf_index")
            if not isinstance(idx, int):
                raise ValueError(f"rextio-numpy fusion: bad leaf_index {idx!r}")
            return ("leaf", idx)
        if kind != "binop":
            raise ValueError(f"rextio-numpy fusion: unexpected node kind {kind!r}")
        children = getattr(node, "children", ())
        if not isinstance(children, tuple) or len(children) != 2:
            raise ValueError("rextio-numpy fusion: binop must have 2 children")
        left_node, right_node = children
        left = walk(left_node)
        right = walk(right_node)
        op = getattr(node, "target")
        if not isinstance(op, str):
            raise ValueError(f"rextio-numpy fusion: bad op {op!r}")
        # Emit this node; non-root results become temps for parent nodes.
        plan.append((op, left, right))
        tid = temp_i[0]
        temp_i[0] += 1
        return ("temp", tid)

    walk(expression)
    return plan


def _emit_fast_path_rank1(
    *,
    n_params: int,
    n_leaves: int,
    param_of: list[int],
    body_arith_fast: str,
    body_arith_fast_direct: str,
    python_output: bool,
) -> str:
    """Equal-shape standard-layout rank-1 gate placed before broadcast Vec work."""
    if n_params == 0:
        return ""
    equal_shape = " && ".join(f"a0.shape() == a{p}.shape()" for p in range(1, n_params))
    layouts = " && ".join(f"a{p}.is_standard_layout()" for p in range(n_params))
    if equal_shape:
        gate = f"{equal_shape} && {layouts}"
    else:
        gate = layouts
    slice_tuple = ", ".join(f"a{p}.as_slice()" for p in range(n_params))
    slice_pats = ", ".join(f"Some(sl{p})" for p in range(n_params))
    fast_loads = _emit_loads(
        n_leaves=n_leaves,
        param_of=param_of,
        load_expr=lambda p: f"sl{p}[i]",
        indent="                ",
    )
    if python_output:
        return (
            f"    // Equal-shape standard-layout fast path (before broadcast-shape Vec work).\n"
            f"    if {gate} {{\n"
            f"        if let ({slice_pats}) = ({slice_tuple}) {{\n"
            f"            return {F64_1D_OUTPUT_HELPER_NAME}("
            f"py, a0.shape()[0], |out| {{\n"
            f"                for i in 0..out.len() {{\n"
            f"{fast_loads}\n"
            f"{body_arith_fast_direct}\n"
            f"                }}\n"
            f"                Ok(())\n"
            f"            }});\n"
            f"        }}\n"
            f"    }}"
        )
    return (
        f"    // Equal-shape standard-layout fast path (before broadcast-shape Vec work).\n"
        f"    if {gate} {{\n"
        f"        if let ({slice_pats}) = ({slice_tuple}) {{\n"
        f"            let dim = numpy::ndarray::Ix1(a0.shape()[0]);\n"
        f"            let out = numpy::ndarray::Array1::from_shape_fn(dim, |i| {{\n"
        f"{fast_loads}\n"
        f"{body_arith_fast}\n"
        f"            }});\n"
        f"            return Ok(out);\n"
        f"        }}\n"
        f"    }}"
    )


def _emit_fast_path_rank2(
    *,
    n_params: int,
    n_leaves: int,
    param_of: list[int],
    body_arith_fast: str,
) -> str:
    """Equal-shape standard-layout rank-2 gate placed before broadcast Vec work."""
    if n_params == 0:
        return ""
    equal_shape = " && ".join(f"a0.shape() == a{p}.shape()" for p in range(1, n_params))
    layouts = " && ".join(f"a{p}.is_standard_layout()" for p in range(n_params))
    if equal_shape:
        gate = f"{equal_shape} && {layouts}"
    else:
        gate = layouts
    slice_tuple = ", ".join(f"a{p}.as_slice()" for p in range(n_params))
    slice_pats = ", ".join(f"Some(sl{p})" for p in range(n_params))
    fast_loads = _emit_loads(
        n_leaves=n_leaves,
        param_of=param_of,
        load_expr=lambda p: f"sl{p}[i * ncols + j]",
        indent="                ",
    )
    return (
        f"    // Equal-shape standard-layout fast path (before broadcast-shape Vec work).\n"
        f"    if {gate} {{\n"
        f"        if let ({slice_pats}) = ({slice_tuple}) {{\n"
        f"            let nrows = a0.shape()[0];\n"
        f"            let ncols = a0.shape()[1];\n"
        f"            let out = numpy::ndarray::Array2::from_shape_fn(\n"
        f"                (nrows, ncols),\n"
        f"                |(i, j)| {{\n"
        f"{fast_loads}\n"
        f"{body_arith_fast}\n"
        f"            }});\n"
        f"            return Ok(out);\n"
        f"        }}\n"
        f"    }}"
    )


def _emit_loads(
    *,
    n_leaves: int,
    param_of: list[int],
    load_expr,
    indent: str,
) -> str:
    """Emit per-occurrence loads, reusing a prior load when alias-proven equal."""
    lines: list[str] = []
    # First occurrence index that uses each parameter slot.
    first_occ_for_param: dict[int, int] = {}
    for i in range(n_leaves):
        p = param_of[i]
        if p not in first_occ_for_param:
            first_occ_for_param[p] = i
            lines.append(f"{indent}let x{i} = {load_expr(p)};")
        else:
            # Semantic identity: same lower-time operand name → reuse load.
            src = first_occ_for_param[p]
            lines.append(f"{indent}let x{i} = x{src};")
    return "\n".join(lines)


def _shape_ref(side: Side, node_shape_var: list[str], param_of: list[int]) -> str:
    tag, idx = side
    if tag == "leaf":
        return f"a{param_of[idx]}.shape()"
    if tag == "temp":
        return f"{node_shape_var[idx]}.as_slice()"
    raise ValueError(f"rextio-numpy fusion: bad shape side {side!r}")


def _value_ref(side: Side) -> str:
    tag, idx = side
    if tag == "leaf":
        return f"x{idx}"
    if tag == "temp":
        return f"t{idx}"
    raise ValueError(f"rextio-numpy fusion: bad value side {side!r}")


def _arith_expr(dtype: str, op: str, left: str, right: str) -> str:
    if dtype == "i64":
        name = OP_NAMES[op]
        wrap = _WRAP_OP[name]
        return f"{left}.{wrap}({right})"
    symbol = _OP_SYMBOL[op]
    return f"{left} {symbol} {right}"
