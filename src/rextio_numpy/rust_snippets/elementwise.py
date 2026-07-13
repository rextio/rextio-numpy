"""Rust helper-text for elementwise binop lowering.

Error messages replicate what numpy 2.4.6 raises for the same inputs
byte-for-byte (the certification kit compares exception type *and* message):

- element-wise shape mismatch (note the trailing space, verbatim from NumPy):
  ``operands could not be broadcast together with shapes (N,) (M,) ``

Division by zero needs no special handling: both NumPy and Rust f64 follow
IEEE-754 (inf/nan), and NumPy does not raise.
"""

from __future__ import annotations

# Elementwise op name -> the Rust operator token used in the helper body.
OP_SYMBOLS: dict[str, str] = {"add": "+", "sub": "-", "mul": "*", "div": "/"}

_ARR = "numpy::ndarray::Array1<f64>"


def elementwise_aa(op: str) -> str:
    """Return the array-array elementwise helper for ``op``.

    Implements NumPy's 1-D broadcasting: equal lengths operate pairwise, a
    length-1 operand broadcasts against the other side (in either position,
    preserving operand order for the non-commutative operators), and any
    other length mismatch raises NumPy's broadcast ValueError.
    """
    symbol = OP_SYMBOLS[op]
    return (
        f"fn __rxtnp_{op}1_aa(a: &{_ARR}, b: &{_ARR}) -> pyo3::PyResult<{_ARR}> {{\n"
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
    """Return the array-scalar elementwise helper for ``op`` (no shape check)."""
    symbol = OP_SYMBOLS[op]
    return (
        f"fn __rxtnp_{op}1_as(a: &{_ARR}, s: f64) -> pyo3::PyResult<{_ARR}> {{\n"
        f"    Ok(a.mapv(|x| x {symbol} s))\n"
        "}"
    )


def elementwise_sa(op: str) -> str:
    """Return the scalar-array elementwise helper for ``op`` (no shape check).

    ``mapv`` keeps the operand order right for the non-commutative operators:
    the scalar is the LEFT operand (``s - x``, ``s / x``).
    """
    symbol = OP_SYMBOLS[op]
    return (
        f"fn __rxtnp_{op}1_sa(s: f64, a: &{_ARR}) -> pyo3::PyResult<{_ARR}> {{\n"
        f"    Ok(a.mapv(|x| s {symbol} x))\n"
        "}"
    )
