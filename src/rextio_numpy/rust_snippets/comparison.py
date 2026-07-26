"""Rust helper generators for bounded elementwise comparisons."""

from __future__ import annotations

from rextio_numpy.rust_snippets.array_repr import (
    array_rust_type,
    lifetime_decl,
    readonly_view_line,
)

COMPARE_SYMBOLS: dict[str, str] = {
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}

_DTYPE_RUST = {"f64": "f64", "f32": "f32", "i64": "i64"}


def _arr(rank: int, elem: str) -> str:
    if elem == "f64":
        return array_rust_type("f64", rank)
    return f"numpy::ndarray::Array{rank}<{elem}>"


def _ix(rank: int) -> str:
    return f"numpy::ndarray::Ix{rank}"


def _scalar(dtype: str) -> tuple[str, str, str]:
    if dtype == "f32":
        return "f64", "    let scalar = scalar as f32;\n", "scalar"
    if dtype == "i64":
        return "i64", "", "scalar"
    return "f64", "", "scalar"


def comparison_call_name_aa(op: str, dtype: str, left_rank: int, right_rank: int) -> str:
    """Return the array-array comparison helper name."""
    return f"__rxtnp_cmp_{op}{left_rank}{right_rank}_aa_{dtype}"


def comparison_call_name_as(op: str, dtype: str, rank: int) -> str:
    """Return the array-scalar comparison helper name."""
    return f"__rxtnp_cmp_{op}{rank}_as_{dtype}"


def comparison_call_name_sa(op: str, dtype: str, rank: int) -> str:
    """Return the scalar-array comparison helper name."""
    return f"__rxtnp_cmp_{op}{rank}_sa_{dtype}"


def comparison_aa_typed(op: str, dtype: str, left_rank: int, right_rank: int) -> str:
    """Return an array-array comparison helper with NumPy broadcasting."""
    symbol = COMPARE_SYMBOLS[op]
    elem = _DTYPE_RUST[dtype]
    result_rank = max(left_rank, right_rank)
    name = comparison_call_name_aa(op, dtype, left_rank, right_rank)
    lifetime = lifetime_decl((dtype, left_rank), (dtype, right_rank))
    view_lines = (
        readonly_view_line("left", dtype, left_rank)
        + readonly_view_line("right", dtype, right_rank)
    )
    return (
        f"fn {name}{lifetime}(left: &{_arr(left_rank, elem)}, "
        f"right: &{_arr(right_rank, elem)}) "
        f"-> pyo3::PyResult<{_arr(result_rank, 'bool')}> {{\n"
        f"{view_lines}"
        "    let shape = __rxtnp_broadcast_shape(left.shape(), right.shape())?;\n"
        "    let left_b = left.broadcast(shape.as_slice()).ok_or_else(|| {\n"
        "        pyo3::exceptions::PyValueError::new_err(format!(\n"
        '            "operands could not be broadcast together with shapes {} {} ",\n'
        "            __rxtnp_fmt_shape(left.shape()), __rxtnp_fmt_shape(right.shape())\n"
        "        ))\n"
        "    })?;\n"
        "    let right_b = right.broadcast(shape.as_slice()).ok_or_else(|| {\n"
        "        pyo3::exceptions::PyValueError::new_err(format!(\n"
        '            "operands could not be broadcast together with shapes {} {} ",\n'
        "            __rxtnp_fmt_shape(left.shape()), __rxtnp_fmt_shape(right.shape())\n"
        "        ))\n"
        "    })?;\n"
        "    let out = numpy::ndarray::Zip::from(&left_b).and(&right_b)"
        f".map_collect(|&left, &right| left {symbol} right);\n"
        f"    Ok(out.into_dimensionality::<{_ix(result_rank)}>().expect(\n"
        '        "rextio-numpy: comparison broadcast rank mismatch"\n'
        "    ))\n"
        "}"
    )


def comparison_as_typed(op: str, dtype: str, rank: int) -> str:
    """Return an array-scalar comparison helper."""
    symbol = COMPARE_SYMBOLS[op]
    elem = _DTYPE_RUST[dtype]
    scalar_type, cast_line, scalar = _scalar(dtype)
    name = comparison_call_name_as(op, dtype, rank)
    lifetime = lifetime_decl((dtype, rank))
    view_line = readonly_view_line("array", dtype, rank)
    return (
        f"fn {name}{lifetime}(array: &{_arr(rank, elem)}, scalar: {scalar_type}) "
        f"-> pyo3::PyResult<{_arr(rank, 'bool')}> {{\n"
        f"{view_line}"
        f"{cast_line}"
        f"    Ok(array.mapv(|value| value {symbol} {scalar}))\n"
        "}"
    )


def comparison_sa_typed(op: str, dtype: str, rank: int) -> str:
    """Return a scalar-array comparison helper while preserving operand order."""
    symbol = COMPARE_SYMBOLS[op]
    elem = _DTYPE_RUST[dtype]
    scalar_type, cast_line, scalar = _scalar(dtype)
    name = comparison_call_name_sa(op, dtype, rank)
    lifetime = lifetime_decl((dtype, rank))
    view_line = readonly_view_line("array", dtype, rank)
    return (
        f"fn {name}{lifetime}(scalar: {scalar_type}, array: &{_arr(rank, elem)}) "
        f"-> pyo3::PyResult<{_arr(rank, 'bool')}> {{\n"
        f"{view_line}"
        f"{cast_line}"
        f"    Ok(array.mapv(|value| {scalar} {symbol} value))\n"
        "}"
    )


__all__ = [
    "COMPARE_SYMBOLS",
    "comparison_aa_typed",
    "comparison_as_typed",
    "comparison_call_name_aa",
    "comparison_call_name_as",
    "comparison_call_name_sa",
    "comparison_sa_typed",
]
