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

Division by zero needs no special handling for floats: both NumPy and Rust
follow IEEE-754 (inf/nan), and NumPy does not raise. Integer true division
promotes to float64. Integer ``+``/``-``/``*``/sum/dot use wrapping
arithmetic so NumPy release-mode wraparound holds under debug Cargo.

Array types are spelled ``numpy::ndarray::ArrayN<T>`` — rust-numpy's own
re-export of the ndarray crate. The boundary conversion produces values of
exactly that version; naming the top-level ``ndarray`` crate instead can pick
a *second* incompatible copy when cargo resolves rust-numpy's ndarray range
(``>= 0.15, <= 0.17``) to a different version than the plugin's direct pin.
"""

from __future__ import annotations

from rextio_numpy.rust_snippets.elementwise import (
    OP_SYMBOLS,
    broadcast_shape_helper,
    elementwise_aa,
    elementwise_aa_typed,
    elementwise_as,
    elementwise_as_typed,
    elementwise_call_name_aa,
    elementwise_call_name_as,
    elementwise_call_name_sa,
    elementwise_sa,
    elementwise_sa_typed,
    fmt_shape_helper,
    shared_broadcast_helpers,
)
from rextio_numpy.rust_snippets.fusion import (
    build_tree_plan,
    fusion_call_name,
    fusion_helper,
    fusion_helpers_bundle,
)
from rextio_numpy.rust_snippets.linear import dot1, dot_call_name, dot_typed
from rextio_numpy.rust_snippets.reductions import (
    axis_call_name,
    axis_typed,
    extrema_call_name,
    extrema_typed,
    mean1,
    mean_call_name,
    mean_typed,
    op_from_target,
    sum1,
    sum_call_name,
    sum_typed,
)
from rextio_numpy.rust_snippets.unary import unary_call_name, unary_typed

__all__ = [
    "OP_SYMBOLS",
    "axis_call_name",
    "axis_typed",
    "broadcast_shape_helper",
    "build_tree_plan",
    "dot1",
    "dot_call_name",
    "dot_typed",
    "elementwise_aa",
    "elementwise_aa_typed",
    "elementwise_as",
    "elementwise_as_typed",
    "elementwise_call_name_aa",
    "elementwise_call_name_as",
    "elementwise_call_name_sa",
    "elementwise_sa",
    "elementwise_sa_typed",
    "extrema_call_name",
    "extrema_typed",
    "fmt_shape_helper",
    "fusion_call_name",
    "fusion_helper",
    "fusion_helpers_bundle",
    "mean1",
    "mean_call_name",
    "mean_typed",
    "op_from_target",
    "shared_broadcast_helpers",
    "sum1",
    "sum_call_name",
    "sum_typed",
    "unary_call_name",
    "unary_typed",
]
