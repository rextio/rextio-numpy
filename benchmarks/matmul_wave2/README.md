# Wave 2 rank-2 f64 matmul research harness

Standalone, preregistered research harness for Wave 2 rank-2 matrix multiply.
Implements the frozen protocol `preregister-matmul-wave2-2026-07-13` without
modifying any product claim, lowering, rule, plugin, or type surface.

## Isolation

- Lives entirely under `benchmarks/matmul_wave2/`.
- Product code under `src/rextio_numpy/` is never imported for claims or rules.
- The Rust candidate is a research-only cdylib (not a product rule).
- Result artifacts are written only to a **user-selected output directory**
  (gitignored). Never commit result JSON/Markdown.

## Frozen matrix

- 27 shape cells: `n ∈ {2,4,8,16,32,64,128,256,512}`, `h = max(1, n//2)`
- Families: `square` `(n,n)×(n,n)`, `wide_tall` `(n,h)×(h,n)`, `tall_wide` `(h,n)×(n,h)`
- Per cell: one shared candidate + three NumPy spellings (`numpy.dot`,
  `numpy.matmul`, `a @ b`) as legs (not 81 product cells)
- Four fresh isolated timing subprocesses per cell, sequential
- Thread env set **before** NumPy import in every worker

## Modes

| Mode | Warmups | Samples | Performance conclusion |
|------|---------|---------|------------------------|
| `--smoke` | 1 | 2 | **Never** (`evidence=false`) |
| `--evidence` | ≥5 | ≥30 | Research gate only; product still NO-GO |

## Product verdict

Dispatchability is **hard-coded failed** (annotations expose rank/dtype, not
dimensions). Therefore the product verdict is always **NO-GO /
fallback-retained**, even if every timing cell “wins.”

## CLI

From the repository root (with NumPy available and `PYTHONPATH` including `.`):

```bash
# Smoke (harness validation only)
python -m benchmarks.matmul_wave2 \
  --output-dir /tmp/matmul-wave2-smoke \
  --smoke

# Single-cell smoke (faster; --cell-id is smoke/debug only)
python -m benchmarks.matmul_wave2 \
  --output-dir /tmp/matmul-wave2-smoke \
  --smoke --cell-id square_n2 --iterations 5

# Full evidence (long; requires all 27 cells — --cell-id is rejected)
python -m benchmarks.matmul_wave2 \
  --output-dir /tmp/matmul-wave2-evidence \
  --evidence
```


Requires: `cargo`, `rustc`, Python ≥3.11, NumPy, and a writable output dir.

## Candidate boundary

- `cargo build --release --locked` once per run
- `numpy =0.29.0`, `pyo3 =0.29.0` with `extension-module`
- `PyReadonlyArray2<f64>` → `as_array().to_owned()` → `Array2::dot` → `to_pyarray`
- No BLAS feature, no GIL-release timing, no product rule
- Staged under the run’s artifact dir; path + SHA-256 verified fail-closed

## Outputs

Under `--output-dir`:

- `matmul-wave2-report-<run_id>.json` — full machine-readable report + raw samples
- `matmul-wave2-report-<run_id>.md` — concise human summary
- `inputs/<run_id>/<cell_id>/{a,b,ref}.npy` — identical inputs for all legs
- `artifacts/<run_id>/matmul_wave2_candidate<EXT_SUFFIX>` — staged cdylib

## Tests

```bash
pytest tests/test_benchmarks_matmul_wave2.py -q
```

Most tests mock cargo/subprocess boundaries. One optional real-cargo one-cell
smoke is marked and skipped when `cargo` is absent.
