"""Rust helper-text for multi-op elementwise chain fusion.

Generates one helper per (tree, dtype, rank) signature:

1. Validate broadcast shapes for each internal binop in left-to-right
   postorder (same order as NumPy evaluation), raising NumPy's trailing-space
   ValueError on the first mismatch.
2. Broadcast each leaf view directly to the final static ``Ix1`` / ``Ix2``
   shape (no intermediate owned ndarrays).
3. One output allocation and one data pass via ``ArrayN::from_shape_fn`` —
   no ``Zip`` arity dependency (ndarray's ZippableTuple tops out at six
   producers, while 8-binop trees need up to nine leaf occurrences). Scalar
   temps inside the element closure preserve AST evaluation order. No
   reassociation, constant folding, or FMA. i64 uses wrapping arithmetic at
   every intermediate node.
"""

from __future__ import annotations

from typing import TypeAlias

from rextio_numpy.claim.fusion import OP_NAMES
from rextio_numpy.rust_snippets.elementwise import (
    _WRAP_OP,
    shared_broadcast_helpers,
)

_DTYPE_RUST = {"f64": "f64", "f32": "f32", "i64": "i64"}
_OP_SYMBOL = {"+": "+", "-": "-", "*": "*", "/": "/"}

# Side of a tree-plan node: leaf occurrence or intermediate temp index.
Side: TypeAlias = tuple[str, int]
TreeStep: TypeAlias = tuple[str, Side, Side]


def fusion_call_name(signature: str) -> str:
    """Return the Rust helper name for a fusion signature.

    The signature is already a stable identifier-safe encoding of the tree
    (dtype, ranks, ops, leaf indexes) — never a Python hash.
    """
    safe = (
        signature.replace("+", "p")
        .replace("-", "m")
        .replace("*", "t")
        .replace("/", "d")
        .replace(".", "_")
    )
    return f"__rxtnp_echain_{safe}"


def fusion_helper(
    *,
    signature: str,
    dtype: str,
    result_rank: int,
    leaf_ranks: tuple[int, ...],
    expression_ops_postorder: tuple[str, ...],
    tree_plan: list[TreeStep],
) -> str:
    """Return the fused helper source for one expression tree.

    ``tree_plan`` is a LTR postorder list of ``(op, left, right)`` where each
    side is either ``("leaf", index)`` or ``("temp", index)`` identifying a
    previously emitted intermediate temp.
    """
    if result_rank not in (1, 2):
        raise ValueError(f"rextio-numpy fusion: unsupported result rank {result_rank}")
    name = fusion_call_name(signature)
    rust_ty = _DTYPE_RUST[dtype]
    n_leaves = len(leaf_ranks)
    params = ", ".join(
        f"a{i}: &numpy::ndarray::Array{leaf_ranks[i]}<{rust_ty}>" for i in range(n_leaves)
    )
    out_ty = f"numpy::ndarray::Array{result_rank}<{rust_ty}>"

    # --- shape validation (LTR postorder) ---
    shape_lines: list[str] = []
    node_shape_var: list[str] = []
    for step, (op, left, right) in enumerate(tree_plan):
        del op  # validation is shape-only; op used in arithmetic section.
        left_expr = _shape_ref(left, node_shape_var)
        right_expr = _shape_ref(right, node_shape_var)
        var = f"s{step}"
        shape_lines.append(f"    let {var} = __rxtnp_broadcast_shape({left_expr}, {right_expr})?;")
        node_shape_var.append(var)
    out_shape = node_shape_var[-1]

    # --- broadcast leaves to final static IxN shape only ---
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
            f"    let v{i} = a{i}.broadcast(dim).ok_or_else(|| {{\n"
            f"        pyo3::exceptions::PyValueError::new_err(format!(\n"
            f'            "operands could not be broadcast together with shapes {{}} {{}} ",\n'
            f"            __rxtnp_fmt_shape(a{i}.shape()), __rxtnp_fmt_shape({out_shape}.as_slice())\n"
            f"        ))\n"
            f"    }})?;"
            for i in range(n_leaves)
        ]
        load_lines = [f"        let x{i} = v{i}[i];" for i in range(n_leaves)]
        closure_header = "    let out = numpy::ndarray::Array1::from_shape_fn(dim, |i| {"
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
            f"    let v{i} = a{i}.broadcast(dim).ok_or_else(|| {{\n"
            f"        pyo3::exceptions::PyValueError::new_err(format!(\n"
            f'            "operands could not be broadcast together with shapes {{}} {{}} ",\n'
            f"            __rxtnp_fmt_shape(a{i}.shape()), __rxtnp_fmt_shape({out_shape}.as_slice())\n"
            f"        ))\n"
            f"    }})?;"
            for i in range(n_leaves)
        ]
        load_lines = [f"        let x{i} = v{i}[[i, j]];" for i in range(n_leaves)]
        closure_header = (
            "    let out = numpy::ndarray::Array2::from_shape_fn("
            f"({out_shape}[0], {out_shape}[1]), |(i, j)| {{"
        )

    # --- postorder scalar temps (AST order) ---
    arith_lines: list[str] = []
    temp_count = 0
    for step, (op, left, right) in enumerate(tree_plan):
        left_val = _value_ref(left)
        right_val = _value_ref(right)
        expr = _arith_expr(dtype, op, left_val, right_val)
        is_last = step == len(tree_plan) - 1
        if is_last:
            arith_lines.append(f"        {expr}")
        else:
            arith_lines.append(f"        let t{temp_count} = {expr};")
            temp_count += 1

    body_arith = "\n".join(arith_lines)
    load_block = "\n".join(load_lines)
    lines = [
        f"fn {name}({params}) -> pyo3::PyResult<{out_ty}> {{",
        *shape_lines,
        dim_setup,
        *broadcast_lines,
        closure_header,
        load_block,
        body_arith,
        "    });",
        "    Ok(out)",
        "}",
    ]
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
) -> tuple[str, ...]:
    """Return shared broadcast helpers plus the fused helper."""
    helper = fusion_helper(
        signature=signature,
        dtype=dtype,
        result_rank=result_rank,
        leaf_ranks=leaf_ranks,
        expression_ops_postorder=expression_ops_postorder,
        tree_plan=tree_plan,
    )
    return (*shared_broadcast_helpers(), helper)


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


def _shape_ref(side: Side, node_shape_var: list[str]) -> str:
    tag, idx = side
    if tag == "leaf":
        return f"a{idx}.shape()"
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
