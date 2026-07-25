"""Rust helper-text for exact NumPy unary module calls."""

from __future__ import annotations


def unary_call_name(op: str, dtype: str, rank: int) -> str:
    """Return the deterministic helper name for a unary array operation."""
    return f"__rxtnp_{op}{rank}_{dtype}"


def unary_typed(op: str, dtype: str, rank: int) -> str:
    """Return a typed unary helper preserving NumPy's scalar arithmetic semantics."""
    if op not in {"negative", "absolute", "square"}:
        raise ValueError(f"unsupported unary operation: {op!r}")
    if dtype not in {"f64", "f32", "i64"} or rank not in {1, 2}:
        raise ValueError(f"unsupported unary dtype/rank: {dtype!r}/{rank}")
    array = f"numpy::ndarray::Array{rank}<{dtype}>"
    if dtype == "i64":
        expression = {
            "negative": "x.wrapping_neg()",
            "absolute": "x.wrapping_abs()",
            "square": "x.wrapping_mul(x)",
        }[op]
    else:
        expression = {"negative": "-x", "absolute": "x.abs()", "square": "x * x"}[op]
    name = unary_call_name(op, dtype, rank)
    return (
        f"fn {name}(a: &{array}) -> pyo3::PyResult<{array}> {{\n"
        f"    Ok(a.mapv(|x| {expression}))\n"
        "}"
    )
