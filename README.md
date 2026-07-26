# rextio-numpy

**Rextio plugin that lowers eligible NumPy code to native Rust.**

The first-party [Rextio](https://github.com/rextio/rextio) plugin for NumPy.
It implements Rextio **plugin protocol v2** (`rextio.plugins.api`): the plugin
self-describes, as machine-readable rule records, which NumPy usage lowers to
Rust (via the `ndarray` crate) and which stays on the Python fallback —
following Rextio's core contract (CPython-equivalent semantics or fall back).

## Status: 0.1.3 unreleased candidate

`rextio-numpy` **0.1.3** is **unreleased candidate work** on the **0.1.3**
development/integration line. The latest published cut is
**`rextio-numpy` 0.1.2** (2026-07-26); prior published cuts include **0.1.1**
(2026-07-14) and **0.1.0** (2026-07-12). This section documents the in-tree
candidate, not a PyPI publication claim.

Implements **plugin API 1.5** end to end: the annotation vocabulary, the
deterministic `claim` pass (including keyword/literal axis metadata and
structured `ClaimExpr` trees from core API 1.2), `lower()` emission to Rust via
the `ndarray` crate, multi-op elementwise chain fusion via
`operand_mode="leaves"`, non-chained comparison claims with resident boolean
results, three-argument `numpy.where`, receiver metadata for certified ndarray
methods, and pinned crate injection (rust-numpy `numpy =0.29.0`; ndarray via
its re-export).

**Candidate focus (0.1.3):** fused elementwise chains may use a rank-1/rank-2
**equal-shape standard-layout** load path that is decided and entered **before**
LTR postorder broadcast-shape `Vec` work when every leaf is the same rank as
the result, shapes are equal, and every leaf is standard (C) layout at helper
entry. The generic path retains LTR broadcast validation and handles
non-standard-layout leaves and broadcast cases, with the same errors and
evaluation order. Statically proven repeated leaf names may share one helper
parameter and reuse loads. Boundary inputs still use `as_array().to_owned()`
(an owned Rust copy; not a guarantee that every input becomes C-contiguous).
Array returns continue to use `numpy::ToPyArray::to_pyarray` so results keep
ordinary NumPy ownership semantics. This is **not** a published speed claim.
Rank-2 dot/matmul/`@` remain **fallback-retained** and are not performance
claims.

**Experimental research harness (non-product):**
`benchmarks/boundary_allocation_poc/` is an isolated F64 rank-1
boundary-allocation PoC that compares owned-copy+`ToPyArray`,
borrowed-view+`ToPyArray`, and direct NumPy-owned sink fill strategies for
elementwise add. The three Rust strategies share one deterministic arithmetic
fill kernel and element order after their distinct boundary/output allocation
steps; timings remain fixed-order unpaired local diagnostics only. It does
**not** change production `BoundaryConversion` or lowering, forbids
`IntoPyArray`, records logical allocation formulas plus local wall times only,
and **must not** be cited as a published speedup or support claim. See that
directory’s README for build/run instructions.

**Dependency:** requires **`rextio>=0.1.6,<0.2`**. NumPy is deliberately **not**
a runtime dependency of this package — only the user-facing
`rextio_numpy.types` vocabulary imports NumPy in the **user** project.

### Core compatibility

This release requires **core `rextio>=0.1.6,<0.2`** and plugin API 1.5 for
comparison claim sites and resident-result propagation. It does not implement
or advertise the optional standalone-artifact capability.

### Annotation vocabulary (`rextio_numpy.types`)

| Annotation | Rank | dtype |
|---|---|---|
| `F64Arr1` / `F64Arr2` | 1 / 2 | float64 |
| `F32Arr1` / `F32Arr2` | 1 / 2 | float32 |
| `I64Arr1` / `I64Arr2` | 1 / 2 | int64 |

Plain runtime aliases of `numpy.ndarray` — no runtime validation. The
analyzer resolves them to plugin type keys when the plugin is enabled.

Comparison results use two additional plugin-owned resident types internally
(`rextio-numpy/bool-1d` and `rextio-numpy/bool-2d`). They deliberately have no
annotation spellings (`PluginType.annotations == ()`), no public
`BoolArr1` / `BoolArr2` aliases, and no Python boundary conversion. Source code
cannot name or forge them: a mask must be produced by a claimed comparison and
consumed inside generated native code.

### Exact base-ndarray boundary

The annotations are nominal, so static analysis cannot tell an exact
`numpy.ndarray` from `numpy.matrix`, `numpy.memmap`, or a custom ndarray
subclass. Every plugin-typed native parameter therefore applies NumPy's exact
C-level ndarray check before materializing an owned Rust copy
(`as_array().to_owned()`). If the native route is executed with a subclass, it
deterministically raises:

```text
TypeError: rextio-numpy native boundary requires exact numpy.ndarray; ndarray subclasses are unsupported
```

This is a runtime native-boundary rejection, not an automatic static fallback.
Exact base-ndarray views and strided arrays remain supported as **inputs**
(still converted via `to_owned()` at the boundary). That owned copy is for
the native frame; it does **not** guarantee every input becomes C-contiguous
(contiguous input layout may be preserved; non-contiguous copy layout is
unspecified). Convert with `numpy.asarray` before the typed hot path when
subclass behavior is irrelevant; when `matrix`, `__array_ufunc__`,
`__array_priority__`, or other subclass/subok semantics matter, keep the
enclosing function on Python fallback.

**Returns (array results):** plugin-typed array returns use rust-numpy
`ToPyArray::to_pyarray` so the result is a normal NumPy-owned array
(ordinary `OWNDATA` / `base is None` / resize observables), matching the
fallback leg's ownership model.

### Native surface (verified)

- **Element-wise `+ - * /`** on same-dtype **float64 / float32 / int64**
  arrays of **rank 1 or 2** — array–array under full supported NumPy
  broadcasting for ranks 1–2 (including 1-D↔2-D and zero-size axes),
  array–scalar, and scalar–array; operand order preserved for `-` and `/`.
  int64 true division (`/`) yields a **float64** array at the broadcast
  result rank (`RXTP-NUMPY-001`).
- **Exact binary ufunc calls** `numpy.add`, `numpy.subtract`,
  `numpy.multiply`, and `numpy.divide` with exactly two positional operands
  reuse that same dtype/rank/broadcast matrix (`RXTP-NUMPY-007`). Optional
  ufunc arguments such as `out`, `where`, `dtype`, and `casting`, extra
  operands, and mixed/unsupported dtypes remain on Python fallback.
- **Non-chained comparisons `== != < <= > >=`** over same-dtype
  **float64 / float32 / int64** rank-1/rank-2 arrays, including every
  rank-1↔rank-2 broadcast pairing and matching array↔Python-scalar forms.
  The result is a resident rank-1/rank-2 boolean array that cannot cross a
  Python parameter or return boundary; a direct exported return is rejected
  by Core with `RXT092` (`RXTP-NUMPY-009`). Chained, identity, and membership
  comparisons stay fallback.
- **Resident-mask logical composition:** exact `numpy.logical_not(mask)`,
  `numpy.logical_and(left, right)`, and `numpy.logical_or(left, right)` consume
  only resident rank-1/rank-2 boolean masks produced by supported comparisons
  or logical calls. Binary forms use NumPy-compatible rank-1/rank-2
  broadcasting, including zero axes; masks remain unnameable and cannot cross a
  Python boundary (`RXTP-NUMPY-015` / `016`).
- **Exact three-positional-argument `numpy.where(condition, x, y)`** where
  `condition` is one resident comparison/logical result and the branches
  are same-dtype numeric arrays, or one array plus its matching Python scalar.
  At least one branch must be an array. Condition-only, keyword, two-scalar,
  and mixed-dtype forms stay fallback. Core-canonicalized import aliases such
  as `np.where` and `from numpy import where as choose` are supported; runtime
  assignment/rebinding aliases stay fallback. Comparison and branch shapes
  are validated independently under NumPy broadcasting, including zero axes
  (`RXTP-NUMPY-014`).
- **`numpy.dot(a, b)` / `a.dot(b)`** on same-dtype **1-D float64 and int64**
  only. **float32** 1-D dots are **deliberately fallback**
  (sequential f32 accumulation diverges materially from NumPy pairwise
  summation; the plugin API has no enforceable runtime length gate).
  **2-D** operands and **`@` / matmul** stay unclaimed (`RXTP-NUMPY-002`).
  Rank-2 matmul research retained product decision **NO-GO /
  fallback-retained** for this cut — **not** a support or speed claim.
- **Whole-array reductions** (module-call form, **no** keywords):
  - `numpy.sum` on **float64 and int64**, ranks **1–2**
  - `numpy.mean` on **float64**, ranks **1–2**
  - `numpy.max` / `numpy.min` on **int64**, ranks **1–2**, including
    equivalent `a.max()` / `a.min()` method forms
  - **float32** sum/mean and **int64 mean** are fallback (material
    accumulation-order divergence; no runtime length gate).
  - Floating whole-array max/min stay fallback because NaN payload/sign and
    signed-zero tie behavior is platform/SIMD-dependent.
  - Equivalent `a.sum()` / `a.mean()` method forms are native under the
    current receiver-metadata contract (`RXTP-NUMPY-003`).
  - Empty int64 max/min raises NumPy-compatible `ValueError`
    (`RXTP-NUMPY-008`).
- **Literal-axis reductions** (`numpy.sum|mean|max|min(a, axis=<int literal>)`,
  `numpy.sum|mean|max|min(a, <int literal>)`, or the equivalent ndarray
  method — exactly one named or positional axis):
  - `sum`: f64/i64 ranks 1–2; `mean`: f64 ranks 1–2
  - `max`/`min`: **int64 ranks 1–2 only**. Float extrema remain fallback:
    NumPy's NaN payload/sign and signed-zero tie behavior varies by supported
    platform/SIMD profile and has no single stable native equivalent.
  - Negative axes are normalized at claim time; out-of-range, `None`, tuple,
    dynamic, `axis=+N` (when core does not extract `UAdd`), additional
    positional arguments, and extra-kw forms stay fallback
  - Rank-1 → core builtin `float`/`int` (not NumPy scalar subclasses);
    rank-2 single-axis → matching rank-1 plugin array
  - **f64 axis sum/mean**: NumPy-compatible pairwise (fast-stride) /
    sequential (slow-stride) dispatch from runtime strides — C and F layouts
  - Empty max/min reduced dimension → `ValueError` with NumPy-compatible text
  - Empty mean value semantics match; native leg omits NumPy's
    `RuntimeWarning` (documented divergence) (`RXTP-NUMPY-004`)
- **Elementwise chain fusion** (binary-op trees of **2–8** pure array-name
  binops, same dtype, ranks 1–2; f64/f32 `+ - * /`, i64 `+ - *` only):
  claimed with `operand_mode="leaves"` under
  `rextio-numpy/elementwise-chain-fusion` so core subsumes descendant
  per-op claims. One fused helper: optional rank-1/rank-2 **equal-shape
  standard-layout** load path decided before broadcast-shape `Vec` work
  (same-rank leaves only), else LTR postorder broadcast validation and
  the generic path for non-standard-layout leaves and broadcast cases;
  either path uses one output allocation/data pass with AST evaluation
  order preserved (i64 wrapping at every intermediate). Exact errors are
  unchanged. Out-of-scope trees keep ordinary per-op elementwise
  (`RXTP-NUMPY-005`).
- **Exact unary module calls** `numpy.negative(a)`, `numpy.absolute(a)`,
  `numpy.abs(a)`, and `numpy.square(a)` on f64/f32/i64 rank-1/rank-2 arrays
  (`RXTP-NUMPY-006`). No `out`, `where`, dtype override, or unary method form
  is claimed. Floating signed-zero/NaN/infinity values follow NumPy; int64
  negative/absolute/square use wraparound arithmetic, including `INT64_MIN`.

Shape/length mismatches raise `ValueError` with NumPy's exact messages.
int64 `+`, `-`, `*`, `sum`, and `dot` use **wraparound** arithmetic matching
NumPy **release** builds. Floating reductions/dots are certified
within-tolerance (not universal bit-equivalence). Whole-array float64
sum/mean may still differ from NumPy pairwise order within 1e-12; **literal-
axis f64 sum/mean** intentionally match NumPy's pairwise/sequential layout
rules.

Python scalar annotations inherit Core's native scalar domains. In particular,
`int` lowers to signed i64. Values in `[-2**63, 2**63-1]` are certified for
the comparison/`where` scalar lanes; an arbitrary Python integer outside that
range is a boundary type-contract violation and raises `OverflowError` before
the plugin helper runs (the ordinary NumPy fallback may accept it). float32
lanes apply NumPy 2.4 weak-scalar narrowing from Python float, including
overflow to infinity, NaN, infinities, and signed zero; comparison and both
scalar branch positions are bit-certified.

### Optimization-safe lower validation

Every native lowerer — elementwise binops and exact ufunc-call aliases, dot,
reductions, unary calls, and fusion — independently revalidates its
claim/context contract before emitting Rust. The explicit `ValueError` guards
remain active under `python -O` /
`PYTHONOPTIMIZE=1`: they verify the route rule and reconstructed result type,
operand mode and placement, operand arity/types, and the route's permitted
literals, keywords, callables, expression, and receiver metadata. Forged or
inconsistent metadata therefore fails closed instead of emitting a helper.
For non-literal operands, both Core's omitted `operand_literals` form and its
arity-matched `ClaimLiteral(is_literal=False)` placeholders are accepted;
populated slots require the route's exact count and lane-specific,
type-compatible literal metadata.

Required CI installs the public Core **`rextio==0.1.6`** release, requires host
plugin API 1.5 or later, runs the complete real-Cargo suite without test
selection, and rejects skipped certification cases.

### Accepted release divergence: missing NumPy `RuntimeWarning`

**Certified acceptance surface:** values, dtypes, and exceptions.

**Accepted for this release (do not overclaim warning equivalence):** native
empty-mean / empty-axis-lane, divide-by-zero, invalid-value /
invalid-reduction, elementwise, fused-elementwise, and related covered paths
**may omit** NumPy `RuntimeWarning` emissions. Values still match the certified
contract; warning parity is **not** part of the acceptance surface.

### Full rule surface

| Rule | Outcome | Code |
|---|---|---|
| Non-chained `== != < <= > >=` on same-dtype f64/f32/i64 ranks 1–2 (broadcasting, array↔scalar); resident bool result | native (verified) | RXTP-NUMPY-009 |
| `numpy.logical_not` over one resident bool mask | native (verified) | RXTP-NUMPY-015 |
| `numpy.logical_and` / `numpy.logical_or` over two resident bool masks (rank-1/rank-2 broadcast) | native (verified) | RXTP-NUMPY-016 |
| Exact three-positional `numpy.where(condition, x, y)` with resident comparison/logical condition and bounded same-dtype branches | native (verified) | RXTP-NUMPY-014 |
| Element-wise `+ - * /` on same-dtype f64/f32/i64 ranks 1–2 (broadcasting, array↔scalar) | native (verified) | RXTP-NUMPY-001 |
| Exact two-positional/no-keyword `numpy.add/subtract/multiply/divide(a, b)` over the same matrix | native (verified) | RXTP-NUMPY-007 |
| `numpy.dot(a, b)` / `a.dot(b)` on same-dtype 1-D f64/i64 (not f32, not 2-D, not `@`) | native (verified) | RXTP-NUMPY-002 |
| Whole-array `numpy.sum` / `a.sum` on f64/i64 ranks 1–2; `numpy.mean` / `a.mean` on f64 ranks 1–2 (no kwargs) | native (verified) | RXTP-NUMPY-003 |
| Whole-array `numpy.max/min(a)` / `a.max/min()` on i64 ranks 1–2 (no arguments/options) | native (verified) | RXTP-NUMPY-008 |
| Literal-axis module or ndarray-method `sum/mean/max/min` with one named or positional integer axis (see native surface) | native (verified) | RXTP-NUMPY-004 |
| Multi-op elementwise chain fusion (2–8 pure array-name binops; leaves mode) | native (verified) | RXTP-NUMPY-005 |
| Exact `numpy.negative/absolute/abs/square(a)` on f64/f32/i64 ranks 1–2 | native (verified) | RXTP-NUMPY-006 |
| Operand types outside the claimed set (incl. float extrema, f32 sum/mean/dots, i64 mean, mixed dtypes) | fallback | RXTP-NUMPY-010 |
| Rank > 2 or unknown rank | fallback | RXTP-NUMPY-011 |
| Mutating aliased views | fallback | RXTP-NUMPY-012 |
| Runtime ndarray subclasses / `matrix` / `__array_ufunc__` overrides at a native boundary | reject (runtime TypeError; not static fallback) | RXTP-NUMPY-013 |
| Any other NumPy API (unsupported method forms, non-literal/tuple axis, 2-D matmul/`@`, …) | fallback | RXTP-NUMPY-019 |

All rules are `experimental`. Codes RXTP-NUMPY-011/012/013/019 are
declarative-only: they document exclusion boundaries in the rule records but
are never attached to claim diagnostics. RXTP-NUMPY-013 documents a runtime
native-boundary rejection rather than fallback; other uncovered sites surface
as core's RXT030. Only RXTP-NUMPY-010 is actively emitted by `claim()`.

NumPy itself is deliberately **not** a dependency of the plugin — only the
user-facing `rextio_numpy.types` vocabulary module imports it, in the user's
project. A `@numba.*`-decorated function is always respected as the user's
opt-in to Numba's semantics and is never lowered by this plugin.

### Benchmarks

The public [honest benchmark suite](benchmarks/README.md) is
**repository / source-checkout tooling** (not a PyPI entry point). It measures
a **fixed F64Arr1** scenario subset (independent of the full released surface)
— fallback vs native wall latency, reporting **wins and losses** honestly.
A result below 1× is valid and rendered as such.

## Usage

```toml
# rextio.toml
[rust]
build_tool = "cargo"

[plugins]
enabled = ["rextio-numpy"]
```

```python
import numpy as np
from rextio_numpy.types import F64Arr1  # plain runtime alias of numpy.ndarray

def dot(a: F64Arr1, b: F64Arr1) -> float:
    return np.dot(a, b)
```

```bash
pip install rextio-numpy   # published 0.1.2 requires rextio >= 0.1.6
# candidate 0.1.3 is source-checkout work until tagged/uploaded
rextio capabilities --format json   # numpy rules appear under "rules"
rextio build .                      # lowered kernels compile via cargo
```

> **Note:** The 0.1.3 candidate (and published 0.1.2) surface requires a core
> that provides plugin API 1.5 (`rextio>=0.1.6`). To work against the surface
> from a source checkout, see Development below.

## Development

Core for this release requires **`rextio>=0.1.6,<0.2`**. For source
development against a sibling Core checkout that exposes plugin API 1.5,
install both trees editable:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e path/to/rextio
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python pytest ruff mypy
.venv/bin/python -m pytest
```

Benchmark suite (from a source checkout; see [benchmarks/README.md](benchmarks/README.md)):

```bash
python -m benchmarks --list
python -m benchmarks --output-dir /tmp/rextio-numpy-bench
```

### Verified suite totals (this candidate)

On this tree:

- `.venv/bin/python -m pytest --collect-only -q` reports **971** collected tests total.
- The focused collection command
  `.venv/bin/python -m pytest tests/test_certification_real_cargo.py --collect-only -q`
  reports **154** real-Cargo certification cases.

Those 154 cases are **cargo-gated** and may also skip via dependency
`importorskip` conditions (e.g. NumPy, Hypothesis). Re-collect after material
test changes; do not treat these numbers as a product API.

## License

MIT
