"""Preregistered constants for the boundary-allocation PoC.

Logical allocation accounting is documented here. Allocator internals and
zero-initialization cost may differ from logical counts; do not treat logical
bytes as measured RSS or allocator-trace truth. Arithmetic kernel and element
order are shared across the three Rust strategies (``fill_add_views``); timings
remain fixed-order unpaired local diagnostics only.
"""

from __future__ import annotations

from typing import Any, Final, Literal

PROTOCOL_ID: Final[str] = "boundary-allocation-poc-f64-r1-2026-07-27"
SCHEMA_VERSION: Final[str] = "boundary-allocation-poc-v1"

# Element size for float64 (logical accounting only).
F64_BYTES: Final[int] = 8

# Predeclared contiguous headline sizes (equal-length a, b).
CONTIGUOUS_SIZES: Final[tuple[int, ...]] = (
    1_000,
    100_000,
    1_000_000,
    10_000_000,
)

# Correctness diagnostics only — not headline speed cells.
DIAGNOSTIC_STRIDE: Final[int] = 2
DIAGNOSTIC_BASE_LEN: Final[int] = 1_024
DIAGNOSTIC_BROADCAST_N: Final[int] = 1_024

StrategyId = Literal["owned_topy", "borrowed_topy", "direct_sink", "python_ref"]
STRATEGIES: Final[tuple[StrategyId, ...]] = (
    "owned_topy",
    "borrowed_topy",
    "direct_sink",
    "python_ref",
)
RUST_STRATEGIES: Final[tuple[StrategyId, ...]] = (
    "owned_topy",
    "borrowed_topy",
    "direct_sink",
)
STRATEGY_FUNCTIONS: Final[dict[StrategyId, str]] = {
    "owned_topy": "add_owned_topy",
    "borrowed_topy": "add_borrowed_topy",
    "direct_sink": "add_direct_sink",
    "python_ref": "python_add",  # harness-side reference lane
}

# Exact product-matching subclass rejection text.
EXACT_NDARRAY_TYPEERROR: Final[str] = (
    "rextio-numpy native boundary requires exact numpy.ndarray; "
    "ndarray subclasses are unsupported"
)

# Candidate crate pins.
CANDIDATE_CRATE_NAME: Final[str] = "boundary_allocation_poc"
CANDIDATE_MODULE_NAME: Final[str] = "boundary_allocation_poc"
NUMPY_CRATE_VERSION: Final[str] = "=0.29.0"
PYO3_CRATE_VERSION: Final[str] = "=0.29.0"
CARGO_BUILD_ARGS: Final[tuple[str, ...]] = ("build", "--release", "--locked")

# Timing defaults (smoke is never a speed claim).
SMOKE_WARMUPS: Final[int] = 1
SMOKE_SAMPLES: Final[int] = 3
DEFAULT_WARMUPS: Final[int] = 5
DEFAULT_SAMPLES: Final[int] = 11
DEFAULT_MIN_BATCH_S: Final[float] = 0.05
DEFAULT_MAX_ITERATIONS: Final[int] = 10_000
DEFAULT_SEED: Final[int] = 2026_07_27

THREAD_ENV: Final[dict[str, str]] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}

# ---------------------------------------------------------------------------
# Logical allocation formulas (equal-length contiguous rank-1 add, length N)
# ---------------------------------------------------------------------------
#
# Count only N-sized float64 buffers that the strategy intentionally creates
# for the inputs/result path. Temporary broadcast bookkeeping, Python object
# headers, and allocator bucket rounding are out of scope.
#
#   owned_topy:
#     A_copy (N) + B_copy (N) + rust_out (N) + to_pyarray (N)  →  4
#   borrowed_topy:
#     rust_out (N) + to_pyarray (N)                            →  2
#   direct_sink:
#     numpy_out (N)                                           →  1
#   python_ref (NumPy a+b):
#     numpy_out (N)  (typical single result allocation)       →  1
#
# Caveats (must appear in reports):
# - Allocator internals may coalesce, over-allocate, or reuse arenas.
# - All three Rust strategies share one ``fill_add_views`` arithmetic kernel
#   and element order after their distinct boundary/output allocation steps.
# - Output buffers are zero-initialized before fill on all three Rust paths
#   (``Array1::zeros`` for owned/borrowed ToPyArray strategies;
#   ``PyArray::zeros`` for direct_sink). Zero-initialization is an extra store
#   pass over N elements before the fill overwrite. It is not an extra logical
#   N-sized allocation, but it is real memory traffic and can affect wall time
#   vs an uninit+fill path.
# - Strided inputs still pay full N for each logical ``to_owned`` copy, but
#   the resulting Rust layout is unspecified.
# - Length-1 broadcast: N is the *result* length; a length-1 leaf copy is
#   still counted as one N-sized unit only when the strategy copies that
#   leaf into a full-N buffer (owned path copies the leaf's true length;
#   see ``logical_allocs_for`` for the equal-length model used in headlines).

LOGICAL_ALLOC_FORMULAS: Final[dict[StrategyId, str]] = {
    "owned_topy": (
        "equal-length contiguous: 2×to_owned(N) + 1×Array1_zeros(N) + 1×ToPyArray(N) = 4×N×8 bytes"
    ),
    "borrowed_topy": (
        "equal-length contiguous: 0×input_copy + 1×Array1_zeros(N) + 1×ToPyArray(N) = 2×N×8 bytes"
    ),
    "direct_sink": (
        "equal-length contiguous: 0×input_copy + 1×NumPy_owned_zeros(N) = 1×N×8 bytes"
    ),
    "python_ref": (
        "equal-length contiguous: typical NumPy result allocation = 1×N×8 bytes (reference)"
    ),
}

LOGICAL_N_ALLOCS_EQUAL_CONTIG: Final[dict[StrategyId, int]] = {
    "owned_topy": 4,
    "borrowed_topy": 2,
    "direct_sink": 1,
    "python_ref": 1,
}

HONESTY_CAVEATS: Final[tuple[str, ...]] = (
    "Logical allocation counts are intentional buffer counts, not allocator traces.",
    "Allocator internals and zero-initialization may differ from logical accounting.",
    "All three Rust strategies execute one shared fill_add_views arithmetic "
    "kernel with identical element order after only their required boundary "
    "and output allocation steps; timings remain fixed-order unpaired local "
    "diagnostics and do not isolate kernel vs allocation effects beyond that.",
    "Every Rust strategy zero-initializes its output buffer before fill "
    "(Array1::zeros for owned_topy/borrowed_topy; PyArray::zeros for "
    "direct_sink): zero-initialization adds a store pass over the output "
    "buffer before fill; that traffic is not a second logical N-sized "
    "allocation but can affect wall time.",
    "Strategy timing is fixed-order and unpaired (strategies run sequentially "
    "in a declared order on shared inputs). Cache warm-up, thermal throttling, "
    "allocator reuse, and order bias can affect relative wall times; do not "
    "treat ratios as causal speedups.",
    "Recorded thread environment is the requested configuration applied by "
    "the harness, not proof of effective library thread state—especially if "
    "NumPy or a BLAS was already imported before those variables were set.",
    "This harness records local wall times for comparison only; it makes no "
    "published speedup claim and must not be cited as product performance.",
    "Production BoundaryConversion remains owned-copy + ToPyArray; this PoC "
    "does not change claim/lower/plugin behavior.",
    "IntoPyArray is forbidden in the candidate (breaks ordinary resize / ownership).",
)


def logical_allocs_for(strategy: StrategyId, n: int) -> dict[str, Any]:
    """Return logical N-sized allocation count and byte estimate for *strategy*.

    *n* is the equal-length contiguous result length used by headline cells.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    count = LOGICAL_N_ALLOCS_EQUAL_CONTIG[strategy]
    return {
        "strategy": strategy,
        "n": int(n),
        "logical_n_sized_allocs": count,
        "logical_bytes": int(count) * int(n) * F64_BYTES,
        "formula": LOGICAL_ALLOC_FORMULAS[strategy],
        "element_bytes": F64_BYTES,
    }


def protocol_manifest() -> dict[str, Any]:
    """Machine-readable protocol block for reports."""
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "contiguous_sizes": list(CONTIGUOUS_SIZES),
        "strategies": list(STRATEGIES),
        "rust_strategies": list(RUST_STRATEGIES),
        "strategy_functions": dict(STRATEGY_FUNCTIONS),
        "logical_n_allocs_equal_contig": dict(LOGICAL_N_ALLOCS_EQUAL_CONTIG),
        "logical_alloc_formulas": dict(LOGICAL_ALLOC_FORMULAS),
        "honesty_caveats": list(HONESTY_CAVEATS),
        "operation": "f64_rank1_add",
        "performance_claim": False,
        "product_surface_change": False,
    }
