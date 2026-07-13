# rextio-numpy

**Rextio plugin that lowers eligible NumPy code to native Rust.**

The first-party [Rextio](https://github.com/rextio/rextio) plugin for NumPy.
It implements Rextio **plugin protocol v2** (`rextio.plugins.api`): the plugin
self-describes, as machine-readable rule records, which NumPy usage lowers to
Rust (via the `ndarray` crate) and which stays on the Python fallback —
following Rextio's core contract (CPython-equivalent semantics or fall back).

## Status: 0.1.1 release candidate

This repository branch is the **0.1.1 release candidate** for package version
`rextio-numpy` **0.1.1**. It is an **untagged / unuploaded** RC on this branch —
**not** a claim that 0.1.1 is published on PyPI. The last **published** cut is
**`rextio-numpy` 0.1.0** (2026-07-12 on PyPI).

Implements **plugin API 1.2** end to end: the annotation vocabulary, the
deterministic `claim` pass (including keyword/literal axis metadata and
structured `ClaimExpr` trees from core API 1.2), `lower()` emission to Rust via
the `ndarray` crate, multi-op elementwise chain fusion via
`operand_mode="leaves"`, and pinned crate injection (rust-numpy
`numpy =0.29.0`; ndarray via its re-export).

**Dependency:** requires **`rextio>=0.1.2,<0.2`**. NumPy is deliberately **not**
a runtime dependency of this package — only the user-facing
`rextio_numpy.types` vocabulary imports NumPy in the **user** project.

### Safe deployment order

Publish / deploy in this order (or ship lsp + core simultaneously for step 1–2):

1. **`rextio-lsp` 0.1.1** dual-map **first**, **or simultaneous** with core
2. **core `rextio` 0.1.2** (plugin API 1.2 claim metadata)
3. **`rextio-numpy` 0.1.1** only after core 0.1.2 resolves

**`rextio-numpy` cannot be published before its core dependency resolves.**

### Annotation vocabulary (`rextio_numpy.types`)

| Annotation | Rank | dtype |
|---|---|---|
| `F64Arr1` / `F64Arr2` | 1 / 2 | float64 |
| `F32Arr1` / `F32Arr2` | 1 / 2 | float32 |
| `I64Arr1` / `I64Arr2` | 1 / 2 | int64 |

Plain runtime aliases of `numpy.ndarray` — no runtime validation. The
analyzer resolves them to plugin type keys when the plugin is enabled.

### Native surface (verified)

- **Element-wise `+ - * /`** on same-dtype **float64 / float32 / int64**
  arrays of **rank 1 or 2** — array–array under full supported NumPy
  broadcasting for ranks 1–2 (including 1-D↔2-D and zero-size axes),
  array–scalar, and scalar–array; operand order preserved for `-` and `/`.
  int64 true division (`/`) yields a **float64** array at the broadcast
  result rank (`RXTP-NUMPY-001`).
- **`numpy.dot(a, b)`** on same-dtype **1-D float64 and int64** only
  (module-call form). **float32** 1-D dots are **deliberately fallback**
  (sequential f32 accumulation diverges materially from NumPy pairwise
  summation; the plugin API has no enforceable runtime length gate).
  **2-D** operands and **`@` / matmul** stay unclaimed (`RXTP-NUMPY-002`).
  Rank-2 matmul research retained product decision **NO-GO /
  fallback-retained** for this cut.
- **Whole-array reductions** (module-call form, **no** keywords):
  - `numpy.sum` on **float64 and int64**, ranks **1–2**
  - `numpy.mean` on **float64**, ranks **1–2**
  - **float32** sum/mean and **int64 mean** are fallback (material
    accumulation-order divergence; no runtime length gate).
  - Bare `numpy.max` / `numpy.min` (no `axis=`) stay fallback.
  - `a.sum()` / `a.mean()` method forms fallback (`RXTP-NUMPY-003`).
- **Literal-axis reductions** (module-call,
  `numpy.sum|mean|max|min(a, axis=<int literal>)` only — exactly one
  positional array and exactly one named `axis` keyword):
  - `sum`: f64/i64 ranks 1–2; `mean`: f64 ranks 1–2
  - `max`/`min`: f64/i64 ranks 1–2; **f32 rank 2 only** (rank-1 f32 stays
    fallback to preserve `numpy.float32` scalar semantics)
  - Negative axes are normalized at claim time; out-of-range, positional,
    `None`, tuple, dynamic, `axis=+N` (when core does not extract `UAdd`),
    and extra-kw forms stay fallback
  - Rank-1 → core builtin `float`/`int` (not NumPy scalar subclasses);
    rank-2 single-axis → matching rank-1 plugin array
  - **f64 axis sum/mean**: NumPy-compatible pairwise (fast-stride) /
    sequential (slow-stride) dispatch from runtime strides — C and F layouts
  - Float extrema: first NaN in logical order is preserved (sign/payload);
    `max(+0,-0)=+0`, `min(+0,-0)=-0`
  - Empty max/min reduced dimension → `ValueError` with NumPy-compatible text
  - Empty mean value semantics match; native leg omits NumPy's
    `RuntimeWarning` (documented divergence) (`RXTP-NUMPY-004`)
- **Elementwise chain fusion** (binary-op trees of **2–8** pure array-name
  binops, same dtype, ranks 1–2; f64/f32 `+ - * /`, i64 `+ - *` only):
  claimed with `operand_mode="leaves"` under
  `rextio-numpy/elementwise-chain-fusion` so core subsumes descendant
  per-op claims. One fused helper: LTR postorder broadcast validation,
  leaf views only, one output allocation/data pass, AST evaluation order
  preserved (i64 wrapping at every intermediate). Out-of-scope trees keep
  ordinary per-op elementwise (`RXTP-NUMPY-005`).

Shape/length mismatches raise `ValueError` with NumPy's exact messages.
int64 `+`, `-`, `*`, `sum`, and `dot` use **wraparound** arithmetic matching
NumPy **release** builds. Floating reductions/dots are certified
within-tolerance (not universal bit-equivalence). Whole-array float64
sum/mean may still differ from NumPy pairwise order within 1e-12; **literal-
axis f64 sum/mean** intentionally match NumPy's pairwise/sequential layout
rules.

### Accepted release divergence: missing NumPy `RuntimeWarning`

**Certified acceptance surface:** values, dtypes, and exceptions.

**Accepted for this RC (do not overclaim warning equivalence):** native
empty-mean / empty-axis-lane, divide-by-zero, invalid-value /
invalid-reduction, elementwise, fused-elementwise, and related covered paths
**may omit** NumPy `RuntimeWarning` emissions. Values still match the certified
contract; warning parity is **not** part of the acceptance surface.

### Full rule surface

| Rule | Outcome | Code |
|---|---|---|
| Element-wise `+ - * /` on same-dtype f64/f32/i64 ranks 1–2 (broadcasting, array↔scalar) | native (verified) | RXTP-NUMPY-001 |
| `numpy.dot(a, b)` on same-dtype 1-D f64/i64 (module-call; not f32, not 2-D, not `@`) | native (verified) | RXTP-NUMPY-002 |
| Whole-array `numpy.sum` on f64/i64 ranks 1–2; `numpy.mean` on f64 ranks 1–2 (module-call, no kwargs) | native (verified) | RXTP-NUMPY-003 |
| Literal-axis `numpy.sum/mean/max/min(a, axis=<int>)` (see native surface) | native (verified) | RXTP-NUMPY-004 |
| Multi-op elementwise chain fusion (2–8 pure array-name binops; leaves mode) | native (verified) | RXTP-NUMPY-005 |
| Operand types outside the claimed set (incl. f32 sum/mean/dots, rank-1 f32 max/min, i64 mean, mixed dtypes) | fallback | RXTP-NUMPY-010 |
| Rank > 2 or unknown rank | fallback | RXTP-NUMPY-011 |
| Mutating aliased views | fallback | RXTP-NUMPY-012 |
| Any other NumPy API (method forms, non-literal/tuple axis, 2-D matmul/`@`, …) | fallback | RXTP-NUMPY-019 |

All rules are `experimental`. Codes RXTP-NUMPY-011/012/019 are
declarative-only: they document fallback boundaries in the rule records but
are never attached to diagnostics — uncovered sites surface as core's RXT030
instead. Only RXTP-NUMPY-010 is actively emitted.

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
pip install rextio-numpy   # requires rextio >= 0.1.2 (once both are available)
rextio capabilities --format json   # numpy rules appear under "rules"
rextio build .                      # lowered kernels compile via cargo
```

> **Note:** a PyPI `rextio-numpy` install tracks the last **published** cut
> (**0.1.0**). This branch is the **0.1.1 RC** (untagged / unuploaded). To
> exercise the 0.1.1 surface, install from a source checkout (see Development)
> against a core that provides plugin API 1.2 (`rextio>=0.1.2`).

## Development

Core for this RC requires **`rextio>=0.1.2,<0.2`**. For day-to-day work on this
branch, install core from a build that exposes plugin API 1.2 (PyPI once
0.1.2 is available, or a sibling checkout when co-developing) and this package
editable without resolving a published `rextio-numpy` wheel over the tree:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "rextio>=0.1.2,<0.2"
# or, when co-developing core: uv pip install --python .venv/bin/python -e path/to/rextio
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python pytest ruff mypy
.venv/bin/python -m pytest
```

Benchmark suite (from a source checkout; see [benchmarks/README.md](benchmarks/README.md)):

```bash
python -m benchmarks --list
python -m benchmarks --output-dir /tmp/rextio-numpy-bench
```

## License

MIT
