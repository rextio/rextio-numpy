"""Rust helper-text for elementwise binop lowering.

Error messages replicate what numpy 2.4.6 raises for the same inputs
byte-for-byte (the certification kit compares exception type *and* message):

- element-wise shape mismatch (note the trailing space, verbatim from NumPy):
  ``operands could not be broadcast together with shapes (N,) (M,) ``
  and the multi-dimensional form ``... shapes (R,C) (R2,C2) ``

Division by zero needs no special handling for floats: both NumPy and Rust
f64/f32 follow IEEE-754 (inf/nan), and NumPy does not raise. Integer true
division promotes to float64 (NumPy 2.4).

Integer ``+``/``-``/``*`` use wrapping arithmetic so release-mode NumPy
wraparound is matched even under debug Cargo builds.
"""

from __future__ import annotations

# Elementwise op name -> the Rust operator token used in the helper body.
OP_SYMBOLS: dict[str, str] = {"add": "+", "sub": "-", "mul": "*", "div": "/"}

_ARR1_F64 = "numpy::ndarray::Array1<f64>"
_ARR2_F64 = "numpy::ndarray::Array2<f64>"

_DTYPE_RUST = {"f64": "f64", "f32": "f32", "i64": "i64"}
_WRAP_OP = {
    "add": "wrapping_add",
    "sub": "wrapping_sub",
    "mul": "wrapping_mul",
}


def fmt_shape_helper() -> str:
    """Return the shared shape-formatter helper (NumPy-style tuples)."""
    return (
        "fn __rxtnp_fmt_shape(shape: &[usize]) -> String {\n"
        "    if shape.len() == 1 {\n"
        '        return format!("({},)", shape[0]);\n'
        "    }\n"
        "    let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();\n"
        '    format!("({})", parts.join(","))\n'
        "}"
    )


def broadcast_shape_helper() -> str:
    """Return the shared NumPy broadcasting shape helper (ranks 1–2)."""
    return (
        "fn __rxtnp_broadcast_shape(a: &[usize], b: &[usize]) "
        "-> pyo3::PyResult<Vec<usize>> {\n"
        "    let ndim = a.len().max(b.len());\n"
        "    let mut out = Vec::with_capacity(ndim);\n"
        "    for i in 0..ndim {\n"
        "        let da = if i + a.len() < ndim { 1 } else { a[i + a.len() - ndim] };\n"
        "        let db = if i + b.len() < ndim { 1 } else { b[i + b.len() - ndim] };\n"
        "        if da == db {\n"
        "            out.push(da);\n"
        "        } else if da == 1 {\n"
        "            out.push(db);\n"
        "        } else if db == 1 {\n"
        "            out.push(da);\n"
        "        } else {\n"
        "            return Err(pyo3::exceptions::PyValueError::new_err(format!(\n"
        '                "operands could not be broadcast together with shapes {} {} ",\n'
        "                __rxtnp_fmt_shape(a), __rxtnp_fmt_shape(b)\n"
        "            )));\n"
        "        }\n"
        "    }\n"
        "    Ok(out)\n"
        "}"
    )


def shared_broadcast_helpers() -> tuple[str, str]:
    """Return the (fmt_shape, broadcast_shape) helpers as a stable pair."""
    return (fmt_shape_helper(), broadcast_shape_helper())


def _arr(rank: int, dtype: str) -> str:
    return f"numpy::ndarray::Array{rank}<{_DTYPE_RUST[dtype]}>"


def _ix(rank: int) -> str:
    return f"numpy::ndarray::Ix{rank}"


def _float_bin(op: str, x: str, y: str) -> str:
    symbol = OP_SYMBOLS[op]
    return f"{x} {symbol} {y}"


def _int_bin(op: str, x: str, y: str) -> str:
    if op == "div":
        # True division: promote to f64 (NumPy int64 `/` → float64).
        return f"({x} as f64) / ({y} as f64)"
    return f"{x}.{_WRAP_OP[op]}({y})"


def _result_dtype(dtype: str, op: str) -> str:
    if op == "div" and dtype == "i64":
        return "f64"
    return dtype


def _result_rank(left_rank: int, right_rank: int) -> int:
    return max(left_rank, right_rank)


def _map_expr(dtype: str, op: str, left: str, right: str) -> str:
    if dtype == "i64":
        return _int_bin(op, left, right)
    return _float_bin(op, left, right)


# ---------------------------------------------------------------------------
# f64 rank-1 helpers (compat entry points; typed builders cover the full surface)
# ---------------------------------------------------------------------------


def elementwise_aa(op: str) -> str:
    """Return the array-array elementwise helper for f64 rank-1 ``op``.

    Implements NumPy's 1-D broadcasting: equal lengths operate pairwise, a
    length-1 operand broadcasts against the other side (in either position,
    preserving operand order for the non-commutative operators), and any
    other length mismatch raises NumPy's broadcast ValueError.
    """
    symbol = OP_SYMBOLS[op]
    return (
        f"fn __rxtnp_{op}1_aa(a: &{_ARR1_F64}, b: &{_ARR1_F64}) -> pyo3::PyResult<{_ARR1_F64}> {{\n"
        "    if a.len() == b.len() {\n"
        f"        return Ok(a {symbol} b);\n"
        "    }\n"
        "    if a.len() == 1 {\n"
        "        // broadcast: s comes from `a`, so it is the LEFT operand\n"
        "        let s = a[0];\n"
        f"        return Ok(b.mapv(|x| s {symbol} x));\n"
        "    }\n"
        "    if b.len() == 1 {\n"
        "        // broadcast: s comes from `b`, so it is the RIGHT operand\n"
        "        // (operand order matters for the non-commutative - and /)\n"
        "        let s = b[0];\n"
        f"        return Ok(a.mapv(|x| x {symbol} s));\n"
        "    }\n"
        "    Err(pyo3::exceptions::PyValueError::new_err(format!(\n"
        '        "operands could not be broadcast together with shapes ({},) ({},) ",\n'
        "        a.len(), b.len()\n"
        "    )))\n"
        "}"
    )


def elementwise_as(op: str) -> str:
    """Return the array-scalar elementwise helper for f64 rank-1 ``op``."""
    symbol = OP_SYMBOLS[op]
    return (
        f"fn __rxtnp_{op}1_as(a: &{_ARR1_F64}, s: f64) -> pyo3::PyResult<{_ARR1_F64}> {{\n"
        f"    Ok(a.mapv(|x| x {symbol} s))\n"
        "}"
    )


def elementwise_sa(op: str) -> str:
    """Return the scalar-array elementwise helper for f64 rank-1 ``op``."""
    symbol = OP_SYMBOLS[op]
    return (
        f"fn __rxtnp_{op}1_sa(s: f64, a: &{_ARR1_F64}) -> pyo3::PyResult<{_ARR1_F64}> {{\n"
        f"    Ok(a.mapv(|x| s {symbol} x))\n"
        "}"
    )


# ---------------------------------------------------------------------------
# General dtype × rank helpers (Wave 1)
# ---------------------------------------------------------------------------


def elementwise_aa_typed(op: str, dtype: str, left_rank: int, right_rank: int) -> str:
    """Return array-array helper for ``op`` at the given dtype and ranks.

    Result rank is ``max(left_rank, right_rank)``. Integer true division
    (``op == "div"`` and ``dtype == "i64"``) yields ``ArrayN<f64>``.
    """
    if dtype == "f64" and left_rank == 1 and right_rank == 1:
        return elementwise_aa(op)

    result_rank = _result_rank(left_rank, right_rank)
    result_dtype = _result_dtype(dtype, op)
    left_ty = _arr(left_rank, dtype)
    right_ty = _arr(right_rank, dtype)
    out_ty = _arr(result_rank, result_dtype)
    name = f"__rxtnp_{op}{left_rank}{right_rank}_aa_{dtype}"
    expr = _map_expr(dtype, op, "x", "y")
    return (
        f"fn {name}(a: &{left_ty}, b: &{right_ty}) -> pyo3::PyResult<{out_ty}> {{\n"
        "    let shape = __rxtnp_broadcast_shape(a.shape(), b.shape())?;\n"
        "    let a_b = a.broadcast(shape.as_slice()).ok_or_else(|| {\n"
        "        pyo3::exceptions::PyValueError::new_err(format!(\n"
        '            "operands could not be broadcast together with shapes {} {} ",\n'
        "            __rxtnp_fmt_shape(a.shape()), __rxtnp_fmt_shape(b.shape())\n"
        "        ))\n"
        "    })?;\n"
        "    let b_b = b.broadcast(shape.as_slice()).ok_or_else(|| {\n"
        "        pyo3::exceptions::PyValueError::new_err(format!(\n"
        '            "operands could not be broadcast together with shapes {} {} ",\n'
        "            __rxtnp_fmt_shape(a.shape()), __rxtnp_fmt_shape(b.shape())\n"
        "        ))\n"
        "    })?;\n"
        f"    let out = numpy::ndarray::Zip::from(&a_b).and(&b_b)"
        f".map_collect(|&x, &y| {expr});\n"
        f"    Ok(out.into_dimensionality::<{_ix(result_rank)}>().expect(\n"
        '        "rextio-numpy: broadcast rank mismatch"\n'
        "    ))\n"
        "}"
    )


def elementwise_as_typed(op: str, dtype: str, rank: int) -> str:
    """Return array-scalar helper for ``op`` at ``dtype``/``rank``."""
    if dtype == "f64" and rank == 1:
        return elementwise_as(op)

    result_dtype = _result_dtype(dtype, op)
    arr_ty = _arr(rank, dtype)
    out_ty = _arr(rank, result_dtype)
    name = f"__rxtnp_{op}{rank}_as_{dtype}"

    # Python float scalars arrive as f64; narrow to f32 for float32 arrays.
    if dtype == "f32":
        cast_line = "    let s = s as f32;\n"
        body_s = "s"
        param_s = "f64"
    elif dtype == "i64":
        cast_line = ""
        body_s = "s"
        param_s = "i64"
    else:
        cast_line = ""
        body_s = "s"
        param_s = "f64"

    if dtype == "i64" and op == "div":
        expr = f"(x as f64) / ({body_s} as f64)"
    elif dtype == "i64":
        expr = f"x.{_WRAP_OP[op]}({body_s})"
    else:
        symbol = OP_SYMBOLS[op]
        expr = f"x {symbol} {body_s}"

    return (
        f"fn {name}(a: &{arr_ty}, s: {param_s}) -> pyo3::PyResult<{out_ty}> {{\n"
        f"{cast_line}"
        f"    Ok(a.mapv(|x| {expr}))\n"
        "}"
    )


def elementwise_sa_typed(op: str, dtype: str, rank: int) -> str:
    """Return scalar-array helper for ``op`` at ``dtype``/``rank``."""
    if dtype == "f64" and rank == 1:
        return elementwise_sa(op)

    result_dtype = _result_dtype(dtype, op)
    arr_ty = _arr(rank, dtype)
    out_ty = _arr(rank, result_dtype)
    name = f"__rxtnp_{op}{rank}_sa_{dtype}"

    if dtype == "f32":
        cast_line = "    let s = s as f32;\n"
        body_s = "s"
        param_s = "f64"
    elif dtype == "i64":
        cast_line = ""
        body_s = "s"
        param_s = "i64"
    else:
        cast_line = ""
        body_s = "s"
        param_s = "f64"

    if dtype == "i64" and op == "div":
        expr = f"({body_s} as f64) / (x as f64)"
    elif dtype == "i64":
        expr = f"{body_s}.{_WRAP_OP[op]}(x)"
    else:
        symbol = OP_SYMBOLS[op]
        expr = f"{body_s} {symbol} x"

    return (
        f"fn {name}(s: {param_s}, a: &{arr_ty}) -> pyo3::PyResult<{out_ty}> {{\n"
        f"{cast_line}"
        f"    Ok(a.mapv(|x| {expr}))\n"
        "}"
    )


def elementwise_call_name_aa(op: str, dtype: str, left_rank: int, right_rank: int) -> str:
    """Return the Rust helper function name for array-array ``op``."""
    if dtype == "f64" and left_rank == 1 and right_rank == 1:
        return f"__rxtnp_{op}1_aa"
    return f"__rxtnp_{op}{left_rank}{right_rank}_aa_{dtype}"


def elementwise_call_name_as(op: str, dtype: str, rank: int) -> str:
    """Return the Rust helper function name for array-scalar ``op``."""
    if dtype == "f64" and rank == 1:
        return f"__rxtnp_{op}1_as"
    return f"__rxtnp_{op}{rank}_as_{dtype}"


def elementwise_call_name_sa(op: str, dtype: str, rank: int) -> str:
    """Return the Rust helper function name for scalar-array ``op``."""
    if dtype == "f64" and rank == 1:
        return f"__rxtnp_{op}1_sa"
    return f"__rxtnp_{op}{rank}_sa_{dtype}"
