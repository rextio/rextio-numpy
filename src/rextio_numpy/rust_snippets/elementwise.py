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

from rextio_numpy.rust_snippets.array_repr import (
    F64_1D_OUTPUT_HELPER_NAME,
    array_rust_type,
    is_python_backed,
    lifetime_decl,
    readonly_view_line,
)

# Elementwise op name -> the Rust operator token used in the helper body.
OP_SYMBOLS: dict[str, str] = {"add": "+", "sub": "-", "mul": "*", "div": "/"}

_ARR1_F64 = array_rust_type("f64", 1)
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
    return array_rust_type(dtype, rank)


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
    expr = _float_bin(op, "x", "y")
    return (
        f"fn __rxtnp_{op}1_aa<'py>(py: pyo3::Python<'py>, "
        f"a: &{_ARR1_F64}, b: &{_ARR1_F64}) -> pyo3::PyResult<{_ARR1_F64}> {{\n"
        "    let a = a.as_array();\n"
        "    let b = b.as_array();\n"
        "    let n = if a.len() == b.len() {\n"
        "        a.len()\n"
        "    } else if a.len() == 1 {\n"
        "        b.len()\n"
        "    } else if b.len() == 1 {\n"
        "        a.len()\n"
        "    } else {\n"
        "        return Err(pyo3::exceptions::PyValueError::new_err(format!(\n"
        '            "operands could not be broadcast together with shapes ({},) ({},) ",\n'
        "            a.len(), b.len()\n"
        "        )));\n"
        "    };\n"
        f"    {F64_1D_OUTPUT_HELPER_NAME}(py, n, |out| {{\n"
        "        for i in 0..n {\n"
        "            let x = if a.len() == 1 { a[0] } else { a[i] };\n"
        "            let y = if b.len() == 1 { b[0] } else { b[i] };\n"
        f"            out[i] = {expr};\n"
        "        }\n"
        "        Ok(())\n"
        "    })\n"
        "}"
    )


def elementwise_as(op: str) -> str:
    """Return the array-scalar elementwise helper for f64 rank-1 ``op``."""
    expr = _float_bin(op, "x", "s")
    return (
        f"fn __rxtnp_{op}1_as<'py>(py: pyo3::Python<'py>, "
        f"a: &{_ARR1_F64}, s: f64) -> pyo3::PyResult<{_ARR1_F64}> {{\n"
        "    let a = a.as_array();\n"
        f"    {F64_1D_OUTPUT_HELPER_NAME}(py, a.len(), |out| {{\n"
        "        for (i, value) in out.iter_mut().enumerate() {\n"
        "            let x = a[i];\n"
        f"            *value = {expr};\n"
        "        }\n"
        "        Ok(())\n"
        "    })\n"
        "}"
    )


def elementwise_sa(op: str) -> str:
    """Return the scalar-array elementwise helper for f64 rank-1 ``op``."""
    expr = _float_bin(op, "s", "x")
    return (
        f"fn __rxtnp_{op}1_sa<'py>(py: pyo3::Python<'py>, "
        f"s: f64, a: &{_ARR1_F64}) -> pyo3::PyResult<{_ARR1_F64}> {{\n"
        "    let a = a.as_array();\n"
        f"    {F64_1D_OUTPUT_HELPER_NAME}(py, a.len(), |out| {{\n"
        "        for (i, value) in out.iter_mut().enumerate() {\n"
        "            let x = a[i];\n"
        f"            *value = {expr};\n"
        "        }\n"
        "        Ok(())\n"
        "    })\n"
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
    python_output = is_python_backed(result_dtype, result_rank)
    lifetime = lifetime_decl(
        (dtype, left_rank),
        (dtype, right_rank),
        (result_dtype, result_rank),
    )
    py_param = "py: pyo3::Python<'py>, " if python_output else ""
    view_lines = (
        readonly_view_line("a", dtype, left_rank)
        + readonly_view_line("b", dtype, right_rank)
    )
    if python_output:
        output = (
            f"    {F64_1D_OUTPUT_HELPER_NAME}(py, shape[0], |out| {{\n"
            "        for i in 0..shape[0] {\n"
            "            let x = a_b[i];\n"
            "            let y = b_b[i];\n"
            f"            out[i] = {expr};\n"
            "        }\n"
            "        Ok(())\n"
            "    })\n"
        )
    else:
        output = (
            f"    let out = numpy::ndarray::Zip::from(&a_b).and(&b_b)"
            f".map_collect(|&x, &y| {expr});\n"
            f"    Ok(out.into_dimensionality::<{_ix(result_rank)}>().expect(\n"
            '        "rextio-numpy: broadcast rank mismatch"\n'
            "    ))\n"
        )
    return (
        f"fn {name}{lifetime}({py_param}a: &{left_ty}, b: &{right_ty}) "
        f"-> pyo3::PyResult<{out_ty}> {{\n"
        f"{view_lines}"
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
        f"{output}"
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
    python_output = is_python_backed(result_dtype, rank)

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

    lifetime = lifetime_decl((dtype, rank), (result_dtype, rank))
    py_param = "py: pyo3::Python<'py>, " if python_output else ""
    view_line = readonly_view_line("a", dtype, rank)
    if python_output:
        body = (
            f"    {F64_1D_OUTPUT_HELPER_NAME}(py, a.len(), |out| {{\n"
            "        for (i, value) in out.iter_mut().enumerate() {\n"
            "            let x = a[i];\n"
            f"            *value = {expr};\n"
            "        }\n"
            "        Ok(())\n"
            "    })\n"
        )
    else:
        body = f"    Ok(a.mapv(|x| {expr}))\n"
    return (
        f"fn {name}{lifetime}({py_param}a: &{arr_ty}, s: {param_s}) "
        f"-> pyo3::PyResult<{out_ty}> {{\n"
        f"{view_line}"
        f"{cast_line}"
        f"{body}"
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
    python_output = is_python_backed(result_dtype, rank)

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

    lifetime = lifetime_decl((dtype, rank), (result_dtype, rank))
    py_param = "py: pyo3::Python<'py>, " if python_output else ""
    view_line = readonly_view_line("a", dtype, rank)
    if python_output:
        body = (
            f"    {F64_1D_OUTPUT_HELPER_NAME}(py, a.len(), |out| {{\n"
            "        for (i, value) in out.iter_mut().enumerate() {\n"
            "            let x = a[i];\n"
            f"            *value = {expr};\n"
            "        }\n"
            "        Ok(())\n"
            "    })\n"
        )
    else:
        body = f"    Ok(a.mapv(|x| {expr}))\n"
    return (
        f"fn {name}{lifetime}({py_param}s: {param_s}, a: &{arr_ty}) "
        f"-> pyo3::PyResult<{out_ty}> {{\n"
        f"{view_line}"
        f"{cast_line}"
        f"{body}"
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
