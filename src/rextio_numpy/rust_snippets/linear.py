"""Rust helper-text for linear algebra lowering (dot)."""

from __future__ import annotations

_ARR1 = {
    "f64": "numpy::ndarray::Array1<f64>",
    "f32": "numpy::ndarray::Array1<f32>",
    "i64": "numpy::ndarray::Array1<i64>",
}

_DOT_RET = {
    "f64": "f64",
    "f32": "f64",  # widen: native surface returns builtin float
    "i64": "i64",
}


def dot1() -> str:
    """Return the 1-D float64 dot-product helper (length-checked)."""
    return dot_typed("f64")


def dot_typed(dtype: str) -> str:
    """Return the 1-D same-dtype dot helper for ``dtype`` (f64/f32/i64)."""
    arr = _ARR1[dtype]
    ret = _DOT_RET[dtype]
    if dtype == "f64":
        name = "__rxtnp_dot1"
        body = "    Ok(a.dot(b))\n"
    elif dtype == "f32":
        name = "__rxtnp_dot1_f32"
        # Accumulate in f32 then widen so values match NumPy float32 dots.
        body = (
            "    let mut acc: f32 = 0.0;\n"
            "    for (x, y) in a.iter().zip(b.iter()) {\n"
            "        acc = acc + (*x) * (*y);\n"
            "    }\n"
            "    Ok(acc as f64)\n"
        )
    else:
        name = "__rxtnp_dot1_i64"
        body = (
            "    let mut acc: i64 = 0;\n"
            "    for (x, y) in a.iter().zip(b.iter()) {\n"
            "        acc = acc.wrapping_add(x.wrapping_mul(*y));\n"
            "    }\n"
            "    Ok(acc)\n"
        )
    return (
        f"fn {name}(a: &{arr}, b: &{arr}) -> pyo3::PyResult<{ret}> {{\n"
        "    if a.len() != b.len() {\n"
        "        return Err(pyo3::exceptions::PyValueError::new_err(format!(\n"
        '            "shapes ({},) and ({},) not aligned: {} (dim 0) != {} (dim 0)",\n'
        "            a.len(), b.len(), a.len(), b.len()\n"
        "        )));\n"
        "    }\n"
        f"{body}"
        "}"
    )


def dot_call_name(dtype: str) -> str:
    """Return the Rust helper function name for 1-D ``dtype`` dot."""
    if dtype == "f64":
        return "__rxtnp_dot1"
    return f"__rxtnp_dot1_{dtype}"
