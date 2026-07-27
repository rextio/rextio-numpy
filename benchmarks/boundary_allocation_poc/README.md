# Experimental F64 rank-1 NumPy boundary-allocation PoC

Standalone, research-only harness. Compares **three** boundary strategies for
float64 rank-1 elementwise add (`a + b`) plus an optional **Python/NumPy
reference lane**. `owned_topy` is the historical owned-boundary baseline; the
current product's F64 rank-1 lane separately uses the borrowed/direct-output
shape represented by `direct_sink`. The harness itself remains isolated from
product claim/lower/rule code and **makes no published speed claim**.

## Isolation

- Lives entirely under `benchmarks/boundary_allocation_poc/`.
- Product code under `src/rextio_numpy/` is never imported for claims or rules.
- The Rust candidate is a research-only cdylib (not a product rule).
- Result artifacts go only to a **user-selected output directory**. Never
  commit measured JSON/Markdown.

## Strategies (exact comparison)

| id | Boundary path |
|----|----------------|
| `owned_topy` | Historical baseline: exact ndarray check + **two owned input copies** + zero-init owned Rust `Array1` + shared `fill_add_views` + **`ToPyArray`** |
| `borrowed_topy` | Exact ndarray check + **borrowed `PyReadonlyArray` views** + zero-init owned Rust `Array1` + shared `fill_add_views` + **`ToPyArray`** |
| `direct_sink` | Exact ndarray check + borrowed views + zero-init **NumPy-owned** sink + shared `fill_add_views`, returned unchanged |
| `python_ref` | Ordinary NumPy `a + b` (reference lane only) |

**Shared arithmetic kernel:** after each Rust strategy performs only its
required boundary and output allocation steps, all three call the same
`fill_add_views` helper with identical element order (equal-length Zip;
length-1 broadcast manual loops). There is no alternate ndarray arithmetic
or `mapv` kernel. This isolates allocation policy from kernel implementation;
timings remain **fixed-order unpaired local diagnostics**, not causal speedups.

**Forbidden:** `IntoPyArray` (would transfer Rust ownership and break ordinary
`OWNDATA` / `base is None` / in-place resize observables).

### Output zero-initialization (all three Rust strategies)

Each Rust path allocates a zero-initialized output buffer before fill:

- `owned_topy` / `borrowed_topy`: `Array1::zeros(n)` then fill via contiguous
  mutable slice
- `direct_sink`: `PyArray::zeros` then fill the NumPy-owned sink

Zero-initialization is an **extra store pass** over N elements before the fill
overwrite. That traffic is not a second logical N-sized allocation, but it can
affect wall time versus an uninit+fill path. Aligning zero-store policy across
strategies avoids confounding allocation comparisons with init policy.

### Direct sink ownership contract

The direct-sink result must be:

- exact `numpy.ndarray` (not a subclass)
- `OWNDATA` true, `base is None`
- ordinary in-place `resize` when this NumPy build allows it on a fresh owned array
- every element initialized before return

**Never** `IntoPyArray`.

Hard semantic gates (all strategies including `python_ref`): OWNDATA true and
`base is None` on every accepted result; when this NumPy build supports
fresh-array `resize(..., refcheck=True)`, that resize must succeed on a fresh
result. Subclass inputs are rejected in **either** argument position for every
Rust strategy with the product exact-ndarray `TypeError` message.

## Logical allocation accounting (equal-length contiguous, length N)

Documented formulas (logical N-sized float64 buffers only):

| strategy | logical N-sized allocs | logical bytes |
|----------|------------------------|---------------|
| `owned_topy` | 4 | `4 * N * 8` |
| `borrowed_topy` | 2 | `2 * N * 8` |
| `direct_sink` | 1 | `1 * N * 8` |
| `python_ref` | 1 | `1 * N * 8` (typical result) |

**Caveats:** allocator internals and zero-initialization may differ from
logical accounting. Logical counts are **not** RSS or allocator-trace truth.
Arithmetic kernel and element order are shared across the three Rust
strategies; strategy timing is still **fixed-order and unpaired** — cache,
thermal, allocator, and order bias can affect relative times. Recorded
thread environment is **requested configuration**, not proof of effective
library thread state (especially if NumPy was already imported). Timing is
diagnostic only — **no speedup claim**.

## Predeclared cells

**Headline timing (contiguous equal-length):**

- `N ∈ {1_000, 100_000, 1_000_000, 10_000_000}`

**Correctness diagnostics only** (not headline speed claims):

- strided equal-length inputs
- length-1 broadcast either side
- subclass rejection, readonly inputs, same-object aliasing, zero length,
  NaN/Inf, incompatible broadcast error type/message (including NumPy's
  trailing space)

## Build and run (project venv / toolchain)

From the repository root, with the project virtualenv and a Rust toolchain:

```bash
# Environment (same pattern as AGENTS.md / public suite)
uv venv --python 3.11 .venv   # if needed
uv pip install --python .venv/bin/python "rextio>=0.1.6,<0.2"
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python pytest numpy

# Requires: cargo, rustc on PATH; PYTHONPATH includes repo root (cwd is fine
# for `python -m benchmarks...` when run from the repo root).

# Smoke: build candidate, semantics, tiny timing (not a speed claim)
.venv/bin/python -m benchmarks.boundary_allocation_poc \
  --output-dir /tmp/boundary-allocation-poc-smoke \
  --smoke

# Semantics only (no timing)
.venv/bin/python -m benchmarks.boundary_allocation_poc \
  --output-dir /tmp/boundary-allocation-poc-sem \
  --skip-timing

# Full local timing (still not a product claim; do not commit results)
.venv/bin/python -m benchmarks.boundary_allocation_poc \
  --output-dir /tmp/boundary-allocation-poc-full
```

Standalone candidate build (optional; the harness builds automatically):

```bash
cd benchmarks/boundary_allocation_poc/rust_candidate
PYO3_PYTHON="$(git rev-parse --show-toplevel)/.venv/bin/python" cargo build --release
```

Pins: `numpy =0.29.0`, `pyo3 =0.29.0` with `extension-module` (aligned with the
product plugin's rust-numpy line).

## Tests

```bash
.venv/bin/python -m pytest tests/test_benchmarks_boundary_allocation_poc.py -q
```

Most tests are pure-Python / static. One optional real-cargo smoke is skipped
when `cargo` is absent.

## Explicit non-claims

- The harness itself is not a production boundary implementation or a
  certified-surface change; the product F64 rank-1 change is documented in the
  repository README and changelog.
- Not a published speedup or “zero-copy product path” claim.
- Not BLAS / SIMD performance evidence.
- Fixed-order unpaired strategy timings are not causal speedups.
- Thread-env recording is not proof of effective BLAS/OpenMP thread state.
- Rank-2 matmul / `@` product posture is unchanged (separate harness).
