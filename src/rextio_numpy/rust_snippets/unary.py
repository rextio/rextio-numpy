"""Rust helper-text for exact NumPy unary module calls."""

from __future__ import annotations

from rextio_numpy.rust_snippets.array_repr import (
    F64_1D_OUTPUT_HELPER_NAME,
    array_rust_type,
    is_python_backed,
    lifetime_decl,
    readonly_view_line,
)


def unary_call_name(op: str, dtype: str, rank: int) -> str:
    """Return the deterministic helper name for a unary array operation."""
    return f"__rxtnp_{op}{rank}_{dtype}"


def unary_typed(op: str, dtype: str, rank: int) -> str:
    """Return a typed unary helper preserving NumPy's scalar arithmetic semantics."""
    if op not in {"negative", "absolute", "square"}:
        raise ValueError(f"unsupported unary operation: {op!r}")
    if dtype not in {"f64", "f32", "i64"} or rank not in {1, 2}:
        raise ValueError(f"unsupported unary dtype/rank: {dtype!r}/{rank}")
    array = array_rust_type(dtype, rank)
    if dtype == "i64":
        expression = {
            "negative": "x.wrapping_neg()",
            "absolute": "x.wrapping_abs()",
            "square": "x.wrapping_mul(x)",
        }[op]
    else:
        expression = {"negative": "-x", "absolute": "x.abs()", "square": "x * x"}[op]
    name = unary_call_name(op, dtype, rank)
    python_output = is_python_backed(dtype, rank)
    lifetime = lifetime_decl((dtype, rank))
    py_param = "py: pyo3::Python<'py>, " if python_output else ""
    view_line = readonly_view_line("a", dtype, rank)
    if python_output:
        body = (
            f"    {F64_1D_OUTPUT_HELPER_NAME}(py, a.len(), |out| {{\n"
            "        for (i, value) in out.iter_mut().enumerate() {\n"
            "            let x = a[i];\n"
            f"            *value = {expression};\n"
            "        }\n"
            "        Ok(())\n"
            "    })\n"
        )
    else:
        body = f"    Ok(a.mapv(|x| {expression}))\n"
    return (
        f"fn {name}{lifetime}({py_param}a: &{array}) -> pyo3::PyResult<{array}> {{\n"
        f"{view_line}"
        f"{body}"
        "}"
    )
