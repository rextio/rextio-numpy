# rextio-numpy

**Rextio plugin that lowers eligible NumPy code to native Rust.**

The first-party [Rextio](https://github.com/rextio/rextio) plugin for NumPy.
It implements Rextio **plugin protocol v2** (`rextio.plugins.api`): the plugin
self-describes, as machine-readable rule records, which NumPy usage lowers to
Rust (via the `ndarray` crate) and which stays on the Python fallback —
following Rextio's core contract (CPython-equivalent semantics or fall back).

## Status: rule records first, lowering next

This release ships the **declarative contract only** — coverage and rule
records. They surface in `rextio capabilities`, power remediation guidance in
agent/IDE tooling, and enable the RXT091 plugin-lowerable hint on
`@numba.*`-decorated functions. **No code is lowered yet**: actual translation
activates once rextio core exposes the plugin `lower()` hook.

Initial rule surface (all `experimental`, codes `RXTP-NUMPY-NNN`):

| Rule | Outcome | Code |
|---|---|---|
| Element-wise `+ - * /` on float64 1-D/2-D arrays | native (planned) | RXTP-NUMPY-001 |
| `numpy.dot` / `@` on float64 1-D/2-D | native (planned) | RXTP-NUMPY-002 |
| Whole-array `sum` / `mean` reductions on float64 | native (planned) | RXTP-NUMPY-003 |
| Non-float64 dtypes | fallback | RXTP-NUMPY-010 |
| Rank > 2 or unknown rank | fallback | RXTP-NUMPY-011 |
| Mutating aliased views | fallback | RXTP-NUMPY-012 |
| Any other NumPy API | fallback | RXTP-NUMPY-019 |

NumPy itself is deliberately **not** a dependency — the plugin describes (and
later lowers) the *user project's* NumPy usage. A `@numba.*`-decorated
function is always respected as the user's opt-in to Numba's semantics and is
never lowered by this plugin.

## Usage

```toml
# rextio.toml
[plugins]
enabled = ["rextio-numpy"]
```

```bash
pip install rextio-numpy   # requires rextio >= 0.1.1 (unreleased yet)
rextio capabilities --format json   # numpy rules appear under "rules"
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
