# Changelog

## Unreleased — 0.1.3 candidate

Unreleased candidate work toward package version **0.1.3**. Not tagged, not
uploaded to PyPI, and not a publication claim. Latest published cut remains
**0.1.2** (2026-07-26). Requires the same **`rextio>=0.1.6,<0.2`** / plugin
API **1.5** surface as 0.1.2; no Core change and no certified-surface
broadening.

### Fusion contiguous equal-shape fast path

- Fused elementwise chain helpers (``__rxtnp_echain_*``) may take a **rank-1
  or rank-2 equal-shape standard-layout (C-order) fast path** that loads via
  ``as_slice`` after shape validation when every leaf already matches the
  final shape in standard layout at helper entry. The **generic** path
  handles leaves that remain non-standard-layout at helper entry and all
  broadcast cases (including length-1, zero-size, and mixed rank), with the
  same trailing-space ``ValueError`` messages and evaluation order.
- Boundary input conversion still uses ``as_array().to_owned()``: an owned
  Rust copy for the native frame. That does **not** guarantee every input
  becomes C-contiguous (contiguous input layout may be preserved;
  non-contiguous copy layout is unspecified). Do not claim that every strided
  Python input stays on the generic path, or that every ``to_owned`` copy is
  C-contiguous.
- Array results continue to use ``numpy::ToPyArray::to_pyarray`` so returned
  NumPy arrays keep ordinary NumPy ownership observables. The fast path is a
  layout-gated correctness-preserving shortcut only — not a published speed
  claim and not an unsafe reinterpretation of F-order storage.

### Explicit non-claims (unchanged product posture)

- Rank-2 ``dot`` / matmul / ``@`` remain **NO-GO / fallback-retained** and are
  not support or performance claims for this candidate.
- Certified surface, fail-closed lower-time ``ValueError`` guards, and the
  accepted missing-``RuntimeWarning`` divergence are unchanged from 0.1.2.

## 0.1.2 — 2026-07-26

Public Alpha release on PyPI. The package requires
**`rextio>=0.1.6,<0.2`**; its expanded resident comparison/mask/where surface
does not broaden the explicitly retained fallback cases below.

Requires **`rextio>=0.1.6,<0.2`** and advertises plugin API **1.5** for
non-chained comparison claim sites and resident-result propagation. It does not
advertise the optional standalone-artifact capability.

- Adds exact two-positional/no-keyword `numpy.add`, `numpy.subtract`,
  `numpy.multiply`, and `numpy.divide` call aliases over the existing
  operator dtype/rank/broadcast matrix. Optional ufunc arguments (`out`,
  `where`, `dtype`, `casting`, and all other extras), mixed dtypes, and
  unsupported ranks remain fail-closed fallback.
- Adds one positional signed-integer literal axis to the existing
  `sum`/`mean`/`min`/`max` module and ndarray-method reduction matrix.
  Dynamic, tuple, `None`, out-of-range, and additional option forms remain
  fallback; lower time independently revalidates the positional type, literal,
  arity, result type, and rendered-operand alignment.
- Adds no-argument whole-array `numpy.max` / `numpy.min` and `a.max()` /
  `a.min()` for int64 rank-1/rank-2 arrays. Empty arrays raise
  NumPy-compatible `ValueError`; floating extrema remain fallback because
  signed-zero and NaN selection varies across supported platform/SIMD
  profiles.
- Adds non-chained `==`, `!=`, `<`, `<=`, `>`, and `>=` over same-dtype
  f64/f32/i64 rank-1/rank-2 arrays, including every rank pairing and matching
  array↔scalar forms. Results are plugin-owned resident boolean arrays with no
  annotation spellings and no Python boundary conversion, so source cannot
  name or forge a condition type; returning one through an exported boundary
  is rejected by Core with `RXT092`.
- Adds exact resident-mask `numpy.logical_not`, `numpy.logical_and`, and
  `numpy.logical_or`. They consume only plugin-owned rank-1/rank-2 comparison
  or logical results; binary calls preserve NumPy rank-1/rank-2 broadcasting,
  including zero axes. Masks remain result-only and cannot gain an annotation
  spelling or cross a Python boundary.
- Adds exact three-positional-argument `numpy.where(condition, x, y)` for a
  resident comparison/logical condition and same-dtype numeric array branches, or one
  array plus a matching scalar. Independent three-way broadcasting includes
  zero axes and NumPy-compatible shape errors; condition-only, keyword,
  two-scalar, and mixed-dtype forms remain fallback. Core-canonicalized import
  aliases (`np.where`, `from numpy import where as choose`) remain supported;
  runtime assignment/rebinding aliases stay fallback.
- Certifies f32 weak Python scalars across huge finite values, NaN, infinities,
  and signed zero for comparisons and both `where` scalar branch positions,
  including selected-value bits. Integer scalar lanes inherit Core's signed-i64
  boundary; out-of-range Python integers are documented boundary
  type-contract violations and raise `OverflowError` before plugin lowering.
- Adds certified ndarray method parity for `a.dot(b)`, whole-array
  `a.sum()`/`a.mean()`, and literal-axis `a.sum/mean/max/min(axis=<int>)`, with
  the exact existing dtype/rank/axis matrix. Core evaluates a receiver exactly
  once before ordinary operands.
- Adds exact unary module calls `numpy.negative`, `numpy.absolute`/
  `numpy.abs`, and `numpy.square` for f64/f32/i64 rank-1/rank-2 arrays;
  float signed-zero/NaN/infinity behavior and wrapping int64 edge cases are
  covered.
- Keeps rank-2 dot/matmul/`@`, reshape/view, dynamic/tuple axes, floating
  extrema, float32
  dot/sum/mean, int64 mean, unary method forms, and ufunc overrides (`out`,
  `where`, dtype, etc.) on the Python fallback.
- Hardens every plugin-array native boundary with an exact base
  `numpy.ndarray` check. Nominal annotations cannot identify runtime subclasses,
  so `matrix`, `memmap`, and custom `__array_ufunc__` subclasses now raise a
  deterministic native-boundary `TypeError`; this is not an automatic static
  fallback. Exact base-ndarray views remain supported.
- Revalidates the certified dot and reduction dtype/rank matrices at lower
  time, including dot RHS equality, so forged/corrupted claims fail closed.
- Withdraws floating literal-axis `max`/`min` from native lowering. NumPy's
  NaN payload/sign and signed-zero tie behavior varies across supported
  platform/SIMD profiles; these routes now retain Python fallback. Int64
  literal-axis extrema remain native.

### Lower-time contract and CI hardening

- Every native lowerer now independently fail-closes with `ValueError` when
  its reconstructed `ClaimSite` / `LoweringContext` contract is inconsistent,
  including route rule/result, direct-versus-leaves operands, types/arity, and
  applicable literals, keywords, callables, expression, and receiver metadata.
  The guards are covered under `python -O` as well as normal execution; an
  omitted non-literal operand-literal tuple and Core's arity-matched nonliteral
  placeholders remain valid representations.
- Required CI installs the public Core **`rextio==0.1.6`** release, requires
  host plugin API 1.5 or later, and runs the complete real-Cargo certification
  suite without test selection; skipped certification cases fail the job.
- The current tree collects **969** tests in total and **151** tests in the
  real-Cargo certification module.

## 0.1.1 — 2026-07-14

Released cut for package version **0.1.1**, tagged and uploaded to PyPI on
2026-07-14. Published PyPI **`rextio-numpy` 0.1.0** (2026-07-12) was the prior
uploaded release; this section documents the surface shipped in the `0.1.1`
release.

Requires **`rextio>=0.1.2,<0.2`** (plugin API **1.2**, Experimental tier). NumPy
is deliberately **not** a package dependency.

### Safe deployment order

Strict sequential order only (do **not** ship these simultaneously):

1. **`rextio-lsp` 0.1.1** dual-map first
2. **core `rextio` 0.1.2** second (plugin API 1.2 claim metadata)
3. **`rextio-numpy` 0.1.1** third, only after that core dependency resolves

`rextio-numpy` **cannot** be published before its core dependency is available.

### Native surface (certified on this branch)

- **Element-wise `+ - * /`** on same-dtype **float64 / float32 / int64**, ranks
  **1–2**, covering the certified supported NumPy broadcasting cases (array–array,
  array–scalar, scalar–array); unsafe dtype/rank combinations remain fallback.
- **`numpy.dot(a, b)`** on same-dtype **1-D float64 and int64** (module-call);
  float32 1-D dots and rank-2 / `@` stay unclaimed.
- **Whole-array** `numpy.sum` (f64/i64 ranks 1–2) and `numpy.mean` (f64 ranks
  1–2); bare max/min and method forms stay fallback.
- **Literal-axis reductions** `numpy.sum|mean|max|min(a, axis=<int literal>)`
  for the certified dtype/rank matrix (see README / rule records).
- **Multi-op elementwise chain fusion** (2–8 pure array-name binop nodes) via
  `operand_mode="leaves"`: f64/f32 support `+ - * /`; i64 supports `+ - *`
  only.
- Whole-array and 1-D linear ops carried forward and extended from 0.1.0 within
  the ranks/dtypes above (see rule records RXTP-NUMPY-001…005).

### Optimization-safe lower validation

Only the **covered binop and reduction lower-time invariants** that previously
relied on `assert` were replaced with explicit **`ValueError`** guards. Those
guards remain active under `python -O` / `PYTHONOPTIMIZE=1` and fail closed for
the **covered** malformed `ClaimSite` / `LoweringContext` metadata. This does
**not** claim that all malformed metadata is rejected or that incorrect helpers
can never be emitted. Two real optimized-interpreter subprocess regressions —
one per lowerer (`tests/test_lower_binops.py`,
`tests/test_lower_reductions.py`) — protect that covered fail-closed path.

### Verified suite totals (this branch)

Repository evidence on this tree:

- **743** collected tests total from
  `.venv/bin/python -m pytest --collect-only -q`
- **119** collected real-Cargo certification cases from
  `.venv/bin/python -m pytest tests/test_certification_real_cargo.py --collect-only -q`
  (cargo-gated; may also skip via dependency `importorskip` conditions such as
  NumPy/Hypothesis)

### Rank-2 matmul decision

Preregistered Wave-2 rank-2 f64 matmul research retains the product decision
**NO-GO / fallback-retained**. Rank-2 matmul / `@` stay unclaimed for this cut.

### Accepted release divergence: missing NumPy `RuntimeWarning`

**Certified acceptance surface:** values, dtypes, and exceptions remain the
equivalence contract under the core certification kit.

**Accepted divergence (signed off for this release):** native empty-mean /
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

### Import under generated runtimes

- Package root and `rextio_numpy.types` import without requiring core
  `rextio.config` / analyzer / plugin host modules, so fallback wrappers can
  load annotation aliases when the built tree only ships a minimal `rextio`
  package (`__about__`, `__init__`, `runtime`).

---

## 0.1.0 — 2026-07-12

Initial **published** release of the NumPy lowering plugin for Rextio (requires
`rextio >= 0.1.1, < 0.2`; plugin API 1.1, Experimental tier). This is the last
PyPI-uploaded cut prior to the 0.1.1 release above.

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
