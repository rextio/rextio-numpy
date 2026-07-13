# rextio-numpy

**Rextio plugin that lowers eligible NumPy code to native Rust.**

The first-party [Rextio](https://github.com/rextio/rextio) plugin for NumPy.
It implements Rextio **plugin protocol v2** (`rextio.plugins.api`): the plugin
self-describes, as machine-readable rule records, which NumPy usage lowers to
Rust (via the `ndarray` crate) and which stays on the Python fallback —
following Rextio's core contract (CPython-equivalent semantics or fall back).

## Status: expanded Wave 1 surface (this branch)

This repository branch is the **unreleased 0.1.1 development line** for
`rextio-numpy`. It implements **plugin API 1.1** end to end: the annotation
vocabulary, the deterministic `claim` pass, `lower()` emission to Rust via
the `ndarray` crate, and pinned crate injection (rust-numpy
`numpy =0.29.0`; ndarray via its re-export).

**Release boundary:** core Rextio **0.1.1 was released 2026-07-12 and is on
PyPI**. The expanded claim surface documented here lives on this development
branch only. Package metadata, changelog, and a PyPI `rextio-numpy` cut that
ships this surface remain a later **Wave 3** release-integration task — do
not assume an installed PyPI `rextio-numpy` wheel already exposes it.

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
- **Whole-array reductions** (module-call form, no `axis=` / kwargs):
  - `numpy.sum` on **float64 and int64**, ranks **1–2**
  - `numpy.mean` on **float64**, ranks **1–2**
  - **float32** sum/mean and **int64 mean** are fallback (material
    accumulation-order divergence; no runtime length gate).
  - `a.sum()` / `a.mean()` method forms fallback (`RXTP-NUMPY-003`).

Shape/length mismatches raise `ValueError` with NumPy's exact messages.
int64 `+`, `-`, `*`, `sum`, and `dot` use **wraparound** arithmetic matching
NumPy **release** builds. Floating reductions/dots are certified
within-tolerance (not bit-equivalent): float summation order may differ from
NumPy's pairwise summation. Native mean of an empty array returns nan without
NumPy's `RuntimeWarning`.

### Full rule surface

| Rule | Outcome | Code |
|---|---|---|
| Element-wise `+ - * /` on same-dtype f64/f32/i64 ranks 1–2 (broadcasting, array↔scalar) | native (verified) | RXTP-NUMPY-001 |
| `numpy.dot(a, b)` on same-dtype 1-D f64/i64 (module-call; not f32, not 2-D, not `@`) | native (verified) | RXTP-NUMPY-002 |
| Whole-array `numpy.sum` on f64/i64 ranks 1–2; `numpy.mean` on f64 ranks 1–2 (module-call) | native (verified) | RXTP-NUMPY-003 |
| Operand types outside the claimed set (incl. f32 dots/reductions, i64 mean, mixed dtypes) | fallback | RXTP-NUMPY-010 |
| Rank > 2 or unknown rank | fallback | RXTP-NUMPY-011 |
| Mutating aliased views | fallback | RXTP-NUMPY-012 |
| Any other NumPy API (`axis=`, method forms, 2-D matmul/`@`, …) | fallback | RXTP-NUMPY-019 |

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
**repository / source-checkout tooling** (not a PyPI entry point). It
measures fallback vs native wall latency and reports **wins and losses**
honestly — a result below 1× is valid and rendered as such.

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
pip install rextio-numpy   # requires rextio >= 0.1.1 (on PyPI)
rextio capabilities --format json   # numpy rules appear under "rules"
rextio build .                      # lowered kernels compile via cargo
```

> **Note:** a PyPI `rextio-numpy` install tracks the last published cut.
> To exercise the expanded surface on this branch, install from a source
> checkout (see Development).

## Development

Core rextio **0.1.1 is on PyPI**. For day-to-day work on this branch, install
core from PyPI (or a sibling checkout when co-developing) and this package
editable without resolving a published `rextio-numpy` wheel over the tree:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "rextio>=0.1.1,<0.2"
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
