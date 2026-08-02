# rextio-numpy

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio icon">
</p>

<p align="center"><strong>Bounded NumPy-to-Rust lowering for code that Rextio can prove safe.</strong></p>

<p align="center">
  English · <a href="https://github.com/rextio/rextio-numpy/blob/main/README.ko.md">한국어</a> · <a href="https://github.com/rextio/rextio-numpy/blob/main/README.zh-hans.md">简体中文</a> · <a href="https://github.com/rextio/rextio-numpy/blob/main/README.zh-hant.md">繁體中文</a> · <a href="https://github.com/rextio/rextio-numpy/blob/main/README.ja.md">日本語</a>
</p>

`rextio-numpy` lowers a deliberately limited set of typed NumPy operations to Rust through rust-numpy/`ndarray`. Eligible calls run natively. Expressions not claimed during analysis stay on Rextio's Python fallback; once a native route is selected, an exact runtime-boundary miss is rejected rather than silently deoptimized.

> **Public Alpha 0.1.3** (released 2026-07-27). Requires Python 3.11+, `rextio>=0.1.6,<0.2`, and plugin API 1.5. This release does not claim general zero-copy behavior, blanket speedups, or rank-2 matrix multiplication support.

## What is verified

| Area | Native surface |
| --- | --- |
| Elementwise | `+ - * /`, exact two-argument `numpy.add/subtract/multiply/divide`, and selected unary calls on same-dtype float64/float32/int64 arrays, ranks 1–2 |
| Comparisons | Non-chained `== != < <= > >=`; resident boolean masks may feed `logical_not/and/or` and exact three-argument `numpy.where` |
| Linear algebra | `numpy.dot(a, b)` and `a.dot(b)` for 1-D float64/int64 only |
| Reductions | Bounded whole-array and literal-axis `sum`, `mean`, `max`, and `min` forms; see the exact matrix below |
| Fusion | Pure 2–8-operation elementwise chains within the supported dtype/rank matrix |

Shape errors, integer wraparound, ordering, dtypes, and documented exception text are covered by real-Cargo native/fallback tests. Floating reductions and dot products use tolerance-based equivalence where operation order can differ. Native paths may omit NumPy `RuntimeWarning`; warning parity is not certified.

## How it works

1. Type annotations identify a bounded array dtype and rank.
2. The plugin claims only an exact supported expression and emits Rust using rust-numpy/`ndarray`.
3. Rextio compiles the accepted route with Cargo. Unclaimed NumPy code remains Python fallback.

For `F64Arr1`, exact base-`ndarray` inputs remain read-only rust-numpy views for the native frame, and rank-1 float64 results are filled into a fresh NumPy-owned output. Other supported dtype/rank lanes still copy inputs into owned Rust arrays and return through `ToPyArray`. This is an allocation-structure improvement—not general zero-copy and not a published speed claim.

## Quick start

```bash
python -m pip install "rextio-numpy==0.1.3" numpy
```

```toml
# rextio.toml
[rust]
build_tool = "cargo"

[plugins]
enabled = ["rextio-numpy"]
```

```python
import numpy as np
from rextio_numpy.types import F64Arr1

def dot(a: F64Arr1, b: F64Arr1) -> float:
    return np.dot(a, b)
```

```bash
rextio capabilities --format json
rextio build .
```

`F64Arr1` is a plain runtime alias of `numpy.ndarray`; the annotation guides static analysis and is not runtime validation by itself.

## Exact supported surface

### Annotation vocabulary

| Annotation | Rank | dtype |
| --- | ---: | --- |
| `F64Arr1`, `F64Arr2` | 1, 2 | float64 |
| `F32Arr1`, `F32Arr2` | 1, 2 | float32 |
| `I64Arr1`, `I64Arr2` | 1, 2 | int64 |

Resident boolean masks have no public annotation and cannot cross a Python parameter or return boundary.

### Operations and constraints

- Elementwise operators support rank-1/rank-2 NumPy broadcasting, array↔array and array↔matching Python scalar forms, including zero-size axes. Integer true division returns float64.
- Exact binary ufunc forms accept exactly two positional operands. `out`, ufunc `where`, `dtype`, `casting`, extra arguments, and mixed dtypes remain fallback.
- `numpy.where(condition, x, y)` requires a resident comparison/logical condition and same-dtype numeric branches, or one array plus its matching scalar. At least one branch must be an array; keyword, condition-only, two-scalar, and mixed-dtype forms remain fallback.
- `dot`: 1-D float64/int64 only. Float32 dot, all 2-D dot, `numpy.matmul`, and `@` remain fallback.
- Whole-array, no-keyword reductions: float64/int64 `sum`; float64 `mean`; int64 `max/min`; supported equivalent ndarray methods are included.
- Exactly one literal integer axis: float64/int64 `sum`, float64 `mean`, and int64 `max/min`, ranks 1–2. Negative axes are normalized. Dynamic/tuple/`None`/out-of-range axes and extra options remain fallback.
- Unary `numpy.negative`, `numpy.absolute`/`numpy.abs`, and `numpy.square` support float64/float32/int64 ranks 1–2 with no optional arguments or method form.
- Fusion accepts pure binary-op trees of 2–8 array-name operations: float64/float32 use `+ - * /`; int64 uses `+ - *`. Other trees keep ordinary per-operation handling.

## Boundaries and fallback

- Runtime inputs must be exact base `numpy.ndarray`. `numpy.matrix`, `numpy.memmap`, and custom subclasses raise a deterministic native-boundary `TypeError`; this is rejection, not automatic fallback.
- Exact ndarray views, read-only inputs, and positive/negative strides are supported. Only the `F64Arr1` lane preserves a borrowed input view; other lanes materialize owned copies.
- Array results are fresh NumPy-owned arrays with `OWNDATA`, `base is None`, and ordinary resize observables. No lane uses `IntoPyArray`.
- Float32 sum/mean/dot, int64 mean, and floating `max/min` stay fallback because stable NumPy-equivalent accumulation or NaN/signed-zero selection is not proven.
- Int64 `+ - *`, `sum`, and `dot` use release-build NumPy wraparound behavior. Python integer scalar lanes are limited to signed i64; out-of-range values raise `OverflowError` before the plugin helper.
- Chained, identity, and membership comparisons; mutating aliased views; ranks above 2; reshape/view operations; unsupported method forms; and other NumPy APIs stay fallback.
- Functions decorated with `@numba.*` are left to Numba.

## Evidence and non-claims

The [honest benchmark suite](benchmarks/README.md) measures a fixed `F64Arr1` subset and reports both wins and losses; the repository intentionally commits no result numbers. The [boundary-allocation PoC](benchmarks/boundary_allocation_poc/README.md) is research-only, and the [rank-2 matmul harness](benchmarks/matmul_wave2/README.md) retains a **NO-GO / fallback** product decision. None is evidence for a general speedup.

## Development

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "rextio>=0.1.6,<0.2"
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python pytest ruff mypy numpy hypothesis
.venv/bin/python -m pytest
```

See [CHANGELOG.md](CHANGELOG.md) for release history and the benchmark READMEs for reproducibility details.

## License

MIT
