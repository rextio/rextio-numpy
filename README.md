# rextio-numpy

**Rextio plugin that lowers eligible NumPy code to native Rust.**

The first-party [Rextio](https://github.com/rextio/rextio) plugin for NumPy.
It implements Rextio **plugin protocol v2** (`rextio.plugins.api`): the plugin
self-describes, as machine-readable rule records, which NumPy usage lowers to
Rust (via the `ndarray` crate) and which stays on the Python fallback —
following Rextio's core contract (CPython-equivalent semantics or fall back).

## Status: lowering implemented for the initial surface

The plugin now implements **plugin API 1.1** end to end: the annotation
vocabulary (`rextio_numpy.types.F64Arr1`), the deterministic `claim` pass,
`lower()` emission to Rust via the `ndarray` crate, and pinned crate
injection (rust-numpy `numpy =0.29.0`; ndarray via its re-export). The implemented
lowering surface, certified against CPython NumPy with the core plugin
certification kit (`rextio.plugins.testing`):

- Element-wise `+ - * /` on float64 1-D arrays — array-array, array-scalar,
  and scalar-array forms (`RXTP-NUMPY-001`, verified)
- `numpy.dot(a, b)` on float64 1-D arrays (`RXTP-NUMPY-002`, verified)
- Whole-array `numpy.sum` / `numpy.mean` reductions (`RXTP-NUMPY-003`,
  verified)

Shape/length mismatches raise `ValueError` with NumPy's exact messages.
Documented divergences (per rule `constraint`): dot/sum/mean float summation
order may differ from NumPy's pairwise summation, and the native mean of an
empty array returns nan without NumPy's `RuntimeWarning`.

Full rule surface (all `experimental`, codes `RXTP-NUMPY-NNN`):

| Rule | Outcome | Code |
|---|---|---|
| Element-wise `+ - * /` on 1-D float64 arrays (array-array incl. length-1 broadcasting, array-scalar, scalar-array) | native (verified) | RXTP-NUMPY-001 |
| `numpy.dot(a, b)` on 1-D float64 arrays (module-call form) | native (verified) | RXTP-NUMPY-002 |
| Whole-array `numpy.sum` / `numpy.mean` on 1-D float64 (module-call form) | native (verified) | RXTP-NUMPY-003 |
| Non-float64 dtypes / unsupported operand types | fallback | RXTP-NUMPY-010 |
| Rank > 2 or unknown rank | fallback | RXTP-NUMPY-011 |
| Mutating aliased views | fallback | RXTP-NUMPY-012 |
| Any other NumPy API | fallback | RXTP-NUMPY-019 |

NumPy itself is deliberately **not** a dependency of the plugin — only the
user-facing `rextio_numpy.types` vocabulary module imports it, in the user's
project. A `@numba.*`-decorated function is always respected as the user's
opt-in to Numba's semantics and is never lowered by this plugin.

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
pip install rextio-numpy   # requires rextio >= 0.1.1 (unreleased yet)
rextio capabilities --format json   # numpy rules appear under "rules"
rextio build .                      # lowered kernels compile via cargo
```

## Development

rextio 0.1.1 is not on PyPI yet, so install core from a checkout first and
this package without dependency resolution:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e path/to/rextio
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python pytest ruff mypy
.venv/bin/python -m pytest
```

## License

MIT
