"""Rust helper generators for bounded three-argument ``numpy.where``."""

from __future__ import annotations

_DTYPE_RUST = {"f64": "f64", "f32": "f32", "i64": "i64"}


def _arr(rank: int, elem: str) -> str:
    return f"numpy::ndarray::Array{rank}<{elem}>"


def _ix(rank: int) -> str:
    return f"numpy::ndarray::Ix{rank}"


def _scalar(dtype: str, name: str) -> tuple[str, str, str]:
    if dtype == "f32":
        return "f64", f"    let {name} = {name} as f32;\n", name
    if dtype == "i64":
        return "i64", "", name
    return "f64", "", name


def broadcast_shape3_helper() -> str:
    """Return NumPy-compatible three-operand broadcasting with exact errors."""
    return (
        "fn __rxtnp_broadcast_shape3(a: &[usize], b: &[usize], c: &[usize]) "
        "-> pyo3::PyResult<Vec<usize>> {\n"
        "    let ndim = a.len().max(b.len()).max(c.len());\n"
        "    let mut out = Vec::with_capacity(ndim);\n"
        "    for i in 0..ndim {\n"
        "        let da = if i + a.len() < ndim { 1 } else { a[i + a.len() - ndim] };\n"
        "        let db = if i + b.len() < ndim { 1 } else { b[i + b.len() - ndim] };\n"
        "        let dc = if i + c.len() < ndim { 1 } else { c[i + c.len() - ndim] };\n"
        "        let mut dimension = 1usize;\n"
        "        for candidate in [da, db, dc] {\n"
        "            if candidate == 1 {\n"
        "                continue;\n"
        "            }\n"
        "            if dimension == 1 {\n"
        "                dimension = candidate;\n"
        "            } else if dimension != candidate {\n"
        "                return Err(pyo3::exceptions::PyValueError::new_err(format!(\n"
        '                    "operands could not be broadcast together with shapes {} {} {} ",\n'
        "                    __rxtnp_fmt_shape(a), __rxtnp_fmt_shape(b), "
        "__rxtnp_fmt_shape(c)\n"
        "                )));\n"
        "            }\n"
        "        }\n"
        "        out.push(dimension);\n"
        "    }\n"
        "    Ok(out)\n"
        "}"
    )


def where_call_name_aa(
    condition_rank: int,
    dtype: str,
    yes_rank: int,
    no_rank: int,
) -> str:
    """Return the array-array branch helper name."""
    return f"__rxtnp_where{condition_rank}{yes_rank}{no_rank}_aa_{dtype}"


def where_call_name_as(condition_rank: int, dtype: str, yes_rank: int) -> str:
    """Return the array-scalar branch helper name."""
    return f"__rxtnp_where{condition_rank}{yes_rank}_as_{dtype}"


def where_call_name_sa(condition_rank: int, dtype: str, no_rank: int) -> str:
    """Return the scalar-array branch helper name."""
    return f"__rxtnp_where{condition_rank}{no_rank}_sa_{dtype}"


def where_aa_typed(
    condition_rank: int,
    dtype: str,
    yes_rank: int,
    no_rank: int,
) -> str:
    """Return a three-array where helper with a shared three-way shape."""
    elem = _DTYPE_RUST[dtype]
    result_rank = max(condition_rank, yes_rank, no_rank)
    name = where_call_name_aa(condition_rank, dtype, yes_rank, no_rank)
    return (
        f"fn {name}(condition: &{_arr(condition_rank, 'bool')}, "
        f"yes: &{_arr(yes_rank, elem)}, no: &{_arr(no_rank, elem)}) "
        f"-> pyo3::PyResult<{_arr(result_rank, elem)}> {{\n"
        "    let shape = __rxtnp_broadcast_shape3("
        "condition.shape(), yes.shape(), no.shape())?;\n"
        "    let condition_b = condition.broadcast(shape.as_slice()).ok_or_else(|| "
        "pyo3::exceptions::PyValueError::new_err(\"condition broadcast failed\"))?;\n"
        "    let yes_b = yes.broadcast(shape.as_slice()).ok_or_else(|| "
        "pyo3::exceptions::PyValueError::new_err(\"yes broadcast failed\"))?;\n"
        "    let no_b = no.broadcast(shape.as_slice()).ok_or_else(|| "
        "pyo3::exceptions::PyValueError::new_err(\"no broadcast failed\"))?;\n"
        "    let out = numpy::ndarray::Zip::from(&condition_b).and(&yes_b).and(&no_b)"
        ".map_collect(|mask, yes, no| if *mask { *yes } else { *no });\n"
        f"    Ok(out.into_dimensionality::<{_ix(result_rank)}>().expect(\n"
        '        "rextio-numpy: where broadcast rank mismatch"\n'
        "    ))\n"
        "}"
    )


def where_as_typed(condition_rank: int, dtype: str, yes_rank: int) -> str:
    """Return a where helper with an array yes branch and scalar no branch."""
    elem = _DTYPE_RUST[dtype]
    result_rank = max(condition_rank, yes_rank)
    scalar_type, cast_line, scalar = _scalar(dtype, "no")
    name = where_call_name_as(condition_rank, dtype, yes_rank)
    return (
        f"fn {name}(condition: &{_arr(condition_rank, 'bool')}, "
        f"yes: &{_arr(yes_rank, elem)}, no: {scalar_type}) "
        f"-> pyo3::PyResult<{_arr(result_rank, elem)}> {{\n"
        f"{cast_line}"
        "    let shape = __rxtnp_broadcast_shape3("
        "condition.shape(), yes.shape(), &[])?;\n"
        "    let condition_b = condition.broadcast(shape.as_slice()).ok_or_else(|| "
        "pyo3::exceptions::PyValueError::new_err(\"condition broadcast failed\"))?;\n"
        "    let yes_b = yes.broadcast(shape.as_slice()).ok_or_else(|| "
        "pyo3::exceptions::PyValueError::new_err(\"yes broadcast failed\"))?;\n"
        "    let out = numpy::ndarray::Zip::from(&condition_b).and(&yes_b)"
        f".map_collect(|mask, yes| if *mask {{ *yes }} else {{ {scalar} }});\n"
        f"    Ok(out.into_dimensionality::<{_ix(result_rank)}>().expect(\n"
        '        "rextio-numpy: where broadcast rank mismatch"\n'
        "    ))\n"
        "}"
    )


def where_sa_typed(condition_rank: int, dtype: str, no_rank: int) -> str:
    """Return a where helper with a scalar yes branch and array no branch."""
    elem = _DTYPE_RUST[dtype]
    result_rank = max(condition_rank, no_rank)
    scalar_type, cast_line, scalar = _scalar(dtype, "yes")
    name = where_call_name_sa(condition_rank, dtype, no_rank)
    return (
        f"fn {name}(condition: &{_arr(condition_rank, 'bool')}, "
        f"yes: {scalar_type}, no: &{_arr(no_rank, elem)}) "
        f"-> pyo3::PyResult<{_arr(result_rank, elem)}> {{\n"
        f"{cast_line}"
        "    let shape = __rxtnp_broadcast_shape3("
        "condition.shape(), &[], no.shape())?;\n"
        "    let condition_b = condition.broadcast(shape.as_slice()).ok_or_else(|| "
        "pyo3::exceptions::PyValueError::new_err(\"condition broadcast failed\"))?;\n"
        "    let no_b = no.broadcast(shape.as_slice()).ok_or_else(|| "
        "pyo3::exceptions::PyValueError::new_err(\"no broadcast failed\"))?;\n"
        "    let out = numpy::ndarray::Zip::from(&condition_b).and(&no_b)"
        f".map_collect(|mask, no| if *mask {{ {scalar} }} else {{ *no }});\n"
        f"    Ok(out.into_dimensionality::<{_ix(result_rank)}>().expect(\n"
        '        "rextio-numpy: where broadcast rank mismatch"\n'
        "    ))\n"
        "}"
    )


__all__ = [
    "broadcast_shape3_helper",
    "where_aa_typed",
    "where_as_typed",
    "where_call_name_aa",
    "where_call_name_as",
    "where_call_name_sa",
    "where_sa_typed",
]
