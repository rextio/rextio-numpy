"""Rust helper-text for linear algebra lowering (dot)."""

from __future__ import annotations

_ARR = "numpy::ndarray::Array1<f64>"


def dot1() -> str:
    """Return the 1-D float64 dot-product helper (length-checked)."""
    return (
        f"fn __rxtnp_dot1(a: &{_ARR}, b: &{_ARR}) -> pyo3::PyResult<f64> {{\n"
        "    if a.len() != b.len() {\n"
        "        return Err(pyo3::exceptions::PyValueError::new_err(format!(\n"
        '            "shapes ({},) and ({},) not aligned: {} (dim 0) != {} (dim 0)",\n'
        "            a.len(), b.len(), a.len(), b.len()\n"
        "        )));\n"
        "    }\n"
        "    Ok(a.dot(b))\n"
        "}"
    )
