# rextio-numpy honest benchmark suite

Public, reproducible, **explicitly honest** fallback-vs-native benchmarks for a
fixed **F64Arr1** scenario subset of `rextio-numpy` (independent of the full
released surface).

This package is **not** a thin wrapper around core `rextio bench`. Core bench
cannot generate NumPy array arguments and reports a single in-process mean, so
it does **not** satisfy this suite's contract.

## Quick start

From the repository root, with the project virtualenv active (`.venv`):

```bash
# list scenarios
python -m benchmarks --list

# run the full suite; write reports to a temp / ignored directory
python -m benchmarks --output-dir /tmp/rextio-numpy-bench

# tighter / longer sampling
python -m benchmarks --output-dir /tmp/rextio-numpy-bench \
  --samples 7 --iterations 100 --warmups 5

# single scenario
python -m benchmarks --output-dir /tmp/rextio-numpy-bench \
  --scenario small_elementwise
```

**Requirements for a real run:** `cargo` and `rustc` on `PATH`, a working
`rextio` + `rextio-numpy` install (this checkout + editable install is fine),
and NumPy. Missing tools, missing native routes, or a failed native build
produce **explicit skipped/failed** scenario records and a **non-successful**
suite status — never a silent fallback timing labeled as native.

**Fixture kernel shape:** generated kernels intentionally omit function
docstrings (Rextio auto-native treats a docstring as an unsupported expression
statement → `not-candidate`) and use bare `np.dot` for the BLAS control
(`float(np.dot(...))` is RXT030). Scenario prose lives on the suite registry,
not in the fixture body.

## What is measured

Exactly four pre-registered scenarios (fixed F64Arr1 subset — not a full
released-surface inventory):

| id | intent |
|---|---|
| `small_elementwise` | Small-array elementwise workload (`a + b`) |
| `multi_op_chain` | Multi-op chain `(a + b) * (a - b)`, **statically labeled FUSED**; fixture construction must prove fusion before measurements proceed (see Reports) |
| `mixed_control_flow` | Loop + elementwise adds (control flow around array ops) |
| `large_dot_blas_control` | Large 1-D `numpy.dot` control — expected **BLAS-dominated on NumPy**, included to show **native losses** when they occur |

No scenario encodes or asserts an expected speedup. **A result below 1× is
valid and is rendered honestly.**

### Sample semantics (batch vs per-call wall latency)

Each measurement sample records the **raw batch elapsed** wall time for
`iterations` consecutive calls (`t1 - t0` around the inner loop). The suite
then derives **per-call wall latency**:

```
per_call_wall_s = batch_elapsed_s / iterations_per_sample
```

JSON legs expose both series explicitly:

| field | meaning |
|---|---|
| `batch_samples_s` | raw measured batch wall times (seconds) |
| `per_call_samples_s` | `batch_samples_s[i] / iterations_per_sample` |
| `iterations_per_sample` | denominator used for the derivation (auditable) |
| `summary` | median/mean/stdev/min/max/p95 over **per-call** samples only |

Schema version **2.0.0** introduced this split (older `samples_s` was ambiguous).
Schema **2.1.0** clarified package provenance (see below).

### How speedup is defined

```
speedup = fallback_per_call_median_s / native_per_call_median_s
```

- **> 1** → native faster
- **< 1** → fallback faster (native loss)
- mismatch / skip / failure → **no speedup claim**

## Method (honesty contract)

1. **Temporary fixture project** with `rextio-numpy` enabled and the four
   kernels above.
2. **Build** via core's certification path (`build_certification_project` /
   equivalent). The suite inspects `.rextio/reports/build.json` and
   `check.json`:
   - `native_build.status` must be `built`
   - every measured target must be natively served
     (`native_status=accepted` and `route` is `native-direct` or
     `native-plugin:…`)
3. **Separate fresh subprocesses** for fallback and native legs.
   - `REXTIO_NATIVE_MODE` is set **before** importing generated wrappers
   - native timing sets `REXTIO_DISABLE_BOUNDARY_FALLBACK=1`
4. Inputs are built **outside** timed regions (deterministic seeds).
5. Warmups → repeated iterations → multiple samples.
6. **Wall measurement** for both legs:
   - record **raw batch elapsed** per sample (`batch_samples_s`)
   - derive **per-call wall latency** (`per_call_samples_s = batch / iterations`)
   - summary stats (median, mean, stdev, min, max, **p95**) and speedup use
     **per-call** values only
   - **p95 method (nearest-rank):** for sorted ascending **per-call** samples
     of length `n`, `p95 = samples[ceil(0.95 * n) - 1]` (index clamped to
     `[0, n-1]`)
7. **Fail-closed generated-module path check:** the child worker resolves the
   imported module's `__file__` and requires it to lie under the fixture
   `build_python_dir`. Missing `__file__` or an out-of-tree import aborts the
   leg (`ok=false`, nonzero exit) — the suite never times an installed
   package by accident.
8. **Verification before accepting timings:**
   - arrays: exact equality including NaN (`numpy.array_equal(..., equal_nan=True)`)
   - scalars (dot): `float()`-coerced, NaN==NaN, `math.isclose` at `1e-12`
     (documented summation-order divergence)
   - mismatch → scenario `failed`, **no speedup claim**
9. **Build wall time** recorded separately from measurement samples.
10. **Optional metrics** (dispatch crossings, temporary allocations) are
    `unavailable` with an explicit reason unless reliably measurable — never
    inferred from timing.

## Reports

`--output-dir` receives:

- `benchmark-report.json` — versioned machine-readable report
  (`schema_version` **2.1.0**, full metadata, raw batch samples, per-call
  samples, stats over per-call wall latency)
- `benchmark-report.md` — human-readable summary that:
  - labels medians and stats as **per-call wall**
  - shows raw batch samples separately from per-call samples
  - explains `>1` / `<1` on per-call medians
  - lists skipped/failed scenarios
  - reflects the public `multi_op_chain` scenario as **statically labeled
    FUSED** in the registry; fixture construction must prove the exact fusion
    rule (`rextio-numpy/elementwise-chain-fusion` with
    `operand_mode=leaves` and a `__rxtnp_echain_` call in that function's
    generated body) **before measurements proceed**; missing proof fails
    closed before timing. Failed/skipped reports may still retain static
    scenario labels, so a **FUSED label alone is not proof** — successful
    measurement is the evidence that the gate passed
  - shows the BLAS control **without suppressing negative results**

**Do not commit result numbers.** Default to a temp path, or use
`benchmarks/results/` (gitignored via `benchmarks/.gitignore`).

## Reproducibility metadata

Each report records (with explicit `null` / `unavailable` when missing — never
invented):

- UTC timestamp
- platform / OS / architecture
- Python implementation, version, executable
- CPU info when discoverable
- Package provenance for NumPy / rextio / rextio-numpy (see below)
- cargo / rustc versions
- relevant BLAS / thread environment variables
- benchmark settings and scenario sizes
- git revision / dirty status when discoverable

### Package provenance (schema 2.1.0)

Editable installs and sibling source checkouts often diverge from the
installed distribution metadata recorded by `importlib.metadata`. The suite
therefore separates:

| field | meaning |
|---|---|
| `metadata.packages.*` | **Runtime module version** actually imported (`module.__version__`) |
| `metadata.package_distributions.*` | Installed distribution version (`importlib.metadata.version`) |
| `metadata.package_module_files.*` | Module origin path (`module.__file__`) |
| `metadata.package_version_mismatches.*` | `true` when both versions are known and differ; `false` when both known and equal; `null` when either side is unavailable |

`packages.*` is what the benchmark process executed. When a source checkout's
runtime `module.__version__` differs from the installed distribution version
recorded by `importlib.metadata` (for example an editable/sibling Core
checkout next to an older wheel), the report must set
`package_version_mismatches.<pkg> = true`, record both version fields, and
include the sibling `__file__` path. Missing imports or missing distributions
yield `null` — versions are never invented.


## Suite exit codes

| code | meaning |
|---|---|
| 0 | all scenarios `ok` |
| 1 | suite `failed` (zero successful scenarios) |
| 3 | suite `partial` (mix of ok and non-ok) |
| 2 | CLI usage / unknown scenario id |

## Layout

```
benchmarks/
  README.md          # this file
  __main__.py        # python -m benchmarks
  cli.py             # argparse entry
  scenarios.py       # fixed registry + fixture kernel source
  fixture.py         # temp project + native-build verification
  measure.py         # subprocess legs + env construction
  compare.py         # array / scalar equivalence
  stats.py           # median/mean/stdev/min/max/p95
  metadata.py        # reproducibility fields
  models.py          # report schema helpers
  report.py          # JSON + Markdown writers
  runner.py          # orchestration
```

## Tests

Deterministic unit tests live in `tests/test_benchmarks_*.py`. They mock
subprocess / clocks / metadata and use fixed samples — **no cargo**, no real
wall-clock speed assertions.

```bash
pytest tests/test_benchmarks_stats.py tests/test_benchmarks_suite.py -q
```
