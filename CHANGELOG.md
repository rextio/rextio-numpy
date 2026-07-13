# Changelog

## 0.1.1 — 2026-07-13 (release candidate)

Release-candidate cut for package version **0.1.1** on this branch. This is an
**untagged / unuploaded** RC: it is **not** a PyPI publication claim. Published
PyPI **`rextio-numpy` 0.1.0** (2026-07-12) remains the last uploaded release;
this section documents the surface shipped on the `0.1.1` branch for the RC.

Requires **`rextio>=0.1.2,<0.2`** (plugin API **1.2**, Experimental tier). NumPy
is deliberately **not** a package dependency.

### Safe deployment order

1. **`rextio-lsp` 0.1.1** dual-map first, **or** simultaneous with core
2. **core `rextio` 0.1.2** (plugin API 1.2 claim metadata)
3. **`rextio-numpy` 0.1.1** only after that core dependency resolves

`rextio-numpy` **cannot** be published before its core dependency is available.

### Native surface (certified on this branch)

- **Element-wise `+ - * /`** on same-dtype **float64 / float32 / int64**, ranks
  **1–2**, covering the certified supported NumPy broadcasting cases (array–array,
  array–scalar, scalar–array); unsafe dtype/rank combinations remain fallback.
- **Literal-axis reductions** `numpy.sum|mean|max|min(a, axis=<int literal>)`
  for the certified dtype/rank matrix (see README / rule records).
- **Multi-op elementwise chain fusion** (2–8 pure array-name binops) via
  `operand_mode="leaves"`.
- Whole-array and 1-D linear ops carried forward and extended from 0.1.0 within
  the ranks/dtypes above (see rule records RXTP-NUMPY-001…005).

### Rank-2 matmul decision

Preregistered Wave-2 rank-2 f64 matmul research retains the product decision
**NO-GO / fallback-retained**. Rank-2 matmul / `@` stay unclaimed for this cut.

### Accepted release divergence: missing NumPy `RuntimeWarning`

**Certified acceptance surface:** values, dtypes, and exceptions remain the
equivalence contract under the core certification kit.

**Accepted divergence (signed off for this RC):** native empty-mean /
empty-axis-lane, divide-by-zero, invalid-value / invalid-reduction, elementwise,
fused-elementwise, and related covered paths **may omit** NumPy
`RuntimeWarning` emissions. Do **not** overclaim warning equivalence — only
values/dtypes/exceptions are certified.

Other documented divergences (e.g. builtin `float`/`int` vs NumPy scalar
subclasses for some reductions; tolerance-based float comparison) still apply
per rule.

### Rule records

Ships protocol-v2 rule records (`RXTP-NUMPY-*`) for the certified native surface
above (verified) plus explicit fallback exclusions (including matmul/`@`,
uncovered dtypes/forms, rank > 2).

---

## 0.1.0 — 2026-07-12

Initial **published** release of the NumPy lowering plugin for Rextio (requires
`rextio >= 0.1.1, < 0.2`; plugin API 1.1, Experimental tier). This is the last
PyPI-uploaded cut prior to the 0.1.1 RC above.

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
