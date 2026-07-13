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

from rextio_numpy.rust_snippets.elementwise import (
    OP_SYMBOLS,
    elementwise_aa,
    elementwise_as,
    elementwise_sa,
)
from rextio_numpy.rust_snippets.linear import dot1
from rextio_numpy.rust_snippets.reductions import mean1, sum1

__all__ = [
    "OP_SYMBOLS",
    "dot1",
    "elementwise_aa",
    "elementwise_as",
    "elementwise_sa",
    "mean1",
    "sum1",
]
