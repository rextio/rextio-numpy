"""Rust helper generators for exact resident-boolean NumPy logical calls."""

from __future__ import annotations


def _arr(rank: int) -> str:
    return f"numpy::ndarray::Array{rank}<bool>"


def _ix(rank: int) -> str:
    return f"numpy::ndarray::Ix{rank}"


def logical_not_call_name(rank: int) -> str:
    """Return a deterministic unary resident-mask helper name."""
    return f"__rxtnp_logical_not_{rank}"


def logical_not_typed(rank: int) -> str:
    """Return exact elementwise logical-not lowering for one bool mask."""
    name = logical_not_call_name(rank)
    return (
        f"fn {name}(mask: &{_arr(rank)}) -> pyo3::PyResult<{_arr(rank)}> {{\n"
        "    Ok(mask.mapv(|value| !value))\n"
        "}"
    )


def logical_binary_call_name(op: str, left_rank: int, right_rank: int) -> str:
    """Return a deterministic binary resident-mask helper name."""
    return f"__rxtnp_logical_{op}{left_rank}{right_rank}"


def logical_binary_typed(op: str, left_rank: int, right_rank: int) -> str:
    """Return a NumPy-broadcasting boolean conjunction/disjunction helper."""
    if op not in {"and", "or"}:
        raise ValueError(f"unsupported resident logical binary op: {op!r}")
    symbol = "&&" if op == "and" else "||"
    result_rank = max(left_rank, right_rank)
    name = logical_binary_call_name(op, left_rank, right_rank)
    return (
        f"fn {name}(left: &{_arr(left_rank)}, right: &{_arr(right_rank)}) "
        f"-> pyo3::PyResult<{_arr(result_rank)}> {{\n"
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
        '        "rextio-numpy: logical broadcast rank mismatch"\n'
        "    ))\n"
        "}"
    )


def logical_reduction_call_name(op: str, rank: int) -> str:
    """Return a deterministic whole-mask reduction helper name."""
    return f"__rxtnp_logical_{op}_{rank}"


def logical_reduction_typed(op: str, rank: int) -> str:
    """Return exact whole-array ``all`` / ``any`` over a resident bool mask."""
    if op not in {"all", "any"}:
        raise ValueError(f"unsupported resident logical reduction op: {op!r}")
    iterator = "all" if op == "all" else "any"
    name = logical_reduction_call_name(op, rank)
    return (
        f"fn {name}(mask: &{_arr(rank)}) -> pyo3::PyResult<bool> {{\n"
        f"    Ok(mask.iter().{iterator}(|&value| value))\n"
        "}"
    )


__all__ = [
    "logical_binary_call_name",
    "logical_binary_typed",
    "logical_not_call_name",
    "logical_not_typed",
    "logical_reduction_call_name",
    "logical_reduction_typed",
]
