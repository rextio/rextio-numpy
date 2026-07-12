# Changelog

## 0.1.0 — 2026-07-12

Initial release of the NumPy lowering plugin for Rextio (requires
`rextio >= 0.1.1, < 0.2`; plugin API 1.1, Experimental tier).

### Native surface (certified)

Lowers a narrow, certified float64 1-D surface to Rust (rust-numpy/ndarray):

- Element-wise `+`, `-`, `*`, `/` over `(F64_1D, F64_1D)`, `(F64_1D, float)`,
  and `(float, F64_1D)` operands, including NumPy's length-1 broadcasting on
  either side.
- `numpy.dot(a, b)` over two 1-D float64 arrays.
- Whole-array `numpy.sum(a)` and `numpy.mean(a)` (including the empty-array
  `nan` result and RuntimeWarning-free mean semantics documented below).

Functions are typed through the plugin's annotation vocabulary
(`rextio_numpy.types.F64Arr1`); claims are deterministic and made at analysis
time, lowering emits expression-level Rust at codegen time, and the pinned
crate dependencies (`numpy`, `ndarray`) are injected into the generated
Cargo.toml only when a plugin-lowered function exists.

### Equivalence and documented divergences

The whole native surface is certified against CPython NumPy with the core
certification kit under real cargo builds (dual-leg native/fallback
comparison, hypothesis property tests, exception-message equivalence — NumPy
error messages such as the dot length-mismatch and broadcast errors are
reproduced byte-for-byte). Documented divergences, recorded per rule:

- Native reductions return a builtin `float` where NumPy returns
  `numpy.float64`.
- Float summation order may differ (tolerance-based comparison for
  reductions over long mixed-magnitude arrays).
- NumPy `RuntimeWarning`s (e.g. mean of an empty slice) are not emitted on
  the native leg; values still match.

### Rule records and exclusions

Ships protocol-v2 rule records (`RXTP-NUMPY-*`) surfaced through
`rextio capabilities`: the certified native rules above (marked
`verified: true`) plus explicit exclusion records for the uncovered surface
(2-D/ND arrays, non-float64 dtypes, axis reductions, method-call forms),
which reject to the Python fallback with the plugin's own diagnostics.
