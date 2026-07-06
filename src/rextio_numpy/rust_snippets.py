"""Rust helper-text generators for the rextio-numpy lowering.

Pure string functions (no numpy import): each returns one module-level Rust
``fn`` item that core codegen splices into the generated crate, deduplicated
by exact text. All helpers return ``pyo3::PyResult<...>`` and use fully
qualified paths, so no ``use`` lines are required.

Error messages replicate what numpy 2.4.6 raises for the same inputs
byte-for-byte (the certification kit compares exception type *and* message):

- ``numpy.dot`` length mismatch:
  ``shapes (N,) and (M,) not aligned: N (dim 0) != M (dim 0)``
- element-wise shape mismatch (note the trailing space, verbatim from NumPy):
  ``operands could not be broadcast together with shapes (N,) (M,) ``

Division by zero needs no special handling: both NumPy and Rust f64 follow
IEEE-754 (inf/nan), and NumPy does not raise.

Array types are spelled ``numpy::ndarray::Array1<f64>`` — rust-numpy's own
re-export of the ndarray crate. The boundary conversion produces values of
exactly that version; naming the top-level ``ndarray`` crate instead can pick
a *second* incompatible copy when cargo resolves rust-numpy's ndarray range
(``>= 0.15, <= 0.17``) to a different version than the plugin's direct pin.
"""

from __future__ import annotations

# Elementwise op name -> the Rust operator token used in the helper body.
OP_SYMBOLS: dict[str, str] = {"add": "+", "sub": "-", "mul": "*", "div": "/"}

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


def elementwise_aa(op: str) -> str:
    """Return the array-array elementwise helper for ``op`` (shape-checked)."""
    symbol = OP_SYMBOLS[op]
    return (
        f"fn __rxtnp_{op}1_aa(a: &{_ARR}, b: &{_ARR}) -> pyo3::PyResult<{_ARR}> {{\n"
        "    if a.len() != b.len() {\n"
        "        return Err(pyo3::exceptions::PyValueError::new_err(format!(\n"
        '            "operands could not be broadcast together with shapes ({},) ({},) ",\n'
        "            a.len(), b.len()\n"
        "        )));\n"
        "    }\n"
        f"    Ok(a {symbol} b)\n"
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


def sum1() -> str:
    """Return the whole-array sum helper."""
    return f"fn __rxtnp_sum1(a: &{_ARR}) -> pyo3::PyResult<f64> {{ Ok(a.sum()) }}"


def mean1() -> str:
    """Return the whole-array mean helper.

    NumPy's mean of an empty array warns (RuntimeWarning) and returns nan;
    ``ndarray`` returns ``None`` from ``mean()`` on empty input, mapped to nan
    here. Value equivalence holds; the missing warning is the documented
    divergence recorded on rule ``rextio-numpy/reduction-sum-mean``.
    """
    return f"fn __rxtnp_mean1(a: &{_ARR}) -> pyo3::PyResult<f64> {{ Ok(a.mean().unwrap_or(f64::NAN)) }}"
