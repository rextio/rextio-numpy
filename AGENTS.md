# rextio-numpy agent guide

This repository is the first-party **Rextio NumPy lowering plugin**.

Implements **plugin API 1.2** (`api_version = "1.2"`, protocol v2
describe/covers + claim/lower + type vocabulary + pinned crates). Requires
**`rextio>=0.1.2,<0.2`**. NumPy is **not** a package dependency. Package
version on this branch is **`0.1.1`**.

## Tag / upload gate (before publishing 0.1.1)

Before tagging or uploading **0.1.1** to PyPI, **remove or rewrite** transient
pre-release statements in **`README.md`**, **`CHANGELOG.md`**, and this
**`AGENTS.md`** that currently say any of:

- “release candidate” / “RC”
- “untagged” / “unuploaded”
- “not on PyPI” / “not a PyPI publication claim”
- “latest published is 0.1.0” / “last published cut is 0.1.0”

`README.md` is the package long description (`pyproject.toml` `readme`); leave
it stale and PyPI will ship RC language. Historical **0.1.0** changelog entries
stay as history; only rewrite the **0.1.1** header/body where it claims the cut
is still unreleased.

**Current release status:** `rextio-numpy` **0.1.1** is tagged and uploaded to
PyPI (2026-07-14). The prior published cut was **0.1.0** (2026-07-12).

## Safe deployment order (strict)

Do **not** ship these simultaneously:

1. **`rextio-lsp` 0.1.1** first
2. **core `rextio` 0.1.2** second
3. **`rextio-numpy` 0.1.1** third

## Scope (do not overclaim)

Certified native surface (values / dtypes / exceptions; see README + rule
records `RXTP-NUMPY-001`…`005`):

- Element-wise `+ - * /` on same-dtype f64/f32/i64, ranks 1–2 (broadcasting,
  array↔scalar)
- `numpy.dot` on same-dtype 1-D f64/i64 only (not f32, not 2-D, not `@`)
- Whole-array `sum` (f64/i64 ranks 1–2) and `mean` (f64 ranks 1–2)
- Literal-axis `sum|mean|max|min(a, axis=<int literal>)` for the certified
  dtype/rank matrix
- Multi-op elementwise chain fusion (2–8 pure array-name binop nodes,
  `operand_mode="leaves"`): f64/f32 support `+ - * /`; i64 supports `+ - *`
  only

Product decision for rank-2 matmul / `@`: **NO-GO / fallback-retained**.

Accepted divergence: native paths may omit NumPy `RuntimeWarning`; do not claim
warning equivalence.

## Optimization-safe lower validation

Only the **covered binop and reduction lower-time invariants** that previously
relied on `assert` were replaced with explicit **`ValueError`** guards. Those
guards remain active under `python -O` / `PYTHONOPTIMIZE=1` and fail closed for
the **covered** malformed `ClaimSite` / `LoweringContext` metadata. Do **not**
claim that all malformed metadata is rejected or that incorrect helpers can
never be emitted. Do **not** reintroduce bare `assert` for those covered
guards. Two real optimized-interpreter subprocess regressions — one per
lowerer — live in `tests/test_lower_binops.py` and
`tests/test_lower_reductions.py` (`test_fail_closed_under_python_optimize`).

## Layout

| Path | Role |
|---|---|
| `src/rextio_numpy/plugin.py` | Plugin facade (`api_version` 1.2) |
| `src/rextio_numpy/claim/` | Deterministic claim pass |
| `src/rextio_numpy/lower/` | Rust emission (covered binop/reduction ValueError guards) |
| `src/rextio_numpy/rules/` | Protocol-v2 rule records |
| `src/rextio_numpy/types.py` | User annotation vocabulary (imports NumPy) |
| `benchmarks/` | Honest F64Arr1 public suite (not a PyPI entry point) |
| `benchmarks/matmul_wave2/` | Research-only rank-2 matmul harness |
| `tests/test_certification_real_cargo.py` | Cargo-gated dual-leg certification |

## Development commands

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "rextio>=0.1.2,<0.2"
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python pytest ruff mypy numpy hypothesis
.venv/bin/python -m pytest
.venv/bin/python -m pytest --collect-only -q   # 661 collected on this branch
.venv/bin/python -m pytest tests/test_certification_real_cargo.py --collect-only -q  # 115 collected
```

Suite totals above are **verified collect counts for this branch**, not an API
guarantee. The 115 real-Cargo cases are **cargo-gated** and may also skip via
dependency `importorskip` conditions (e.g. NumPy, Hypothesis).

## Agent constraints

- Prefer docs edits that match code; never overstate the NumPy surface.
- Do not hand-edit generated `.rextio/` trees in user projects.
- Benchmark result numbers under `benchmarks/results/` (or temp dirs) are not
  committed. Public `multi_op_chain` is **statically labeled FUSED**; the
  fixture must prove fusion (`operand_mode=leaves` + `__rxtnp_echain_`) before
  timing; a FUSED label alone is not proof—successful measurement is.
- Changelog: preserve all historical entries; only extend the active unreleased
  section when documenting the current cut.
- Before publish: run the **Tag / upload gate** above so PyPI long description
  and changelog do not ship stale RC language.
