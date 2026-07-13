"""Frozen Wave 2 matmul preregistration protocol constants and shape matrix.

This module freezes the decision matrix, thresholds, and honesty rules from
``preregister-matmul-wave2-2026-07-13``. Values here must not change after
measurement begins; results may not retroactively edit this contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, Literal, Sequence

PROTOCOL_ID: Final[str] = "preregister-matmul-wave2-2026-07-13"
SCHEMA_VERSION: Final[str] = "matmul-wave2-research-v1"

# Base sizes and derived rectangular height (frozen).
BASE_SIZES: Final[tuple[int, ...]] = (2, 4, 8, 16, 32, 64, 128, 256, 512)
ShapeFamily = Literal["square", "wide_tall", "tall_wide"]
SHAPE_FAMILIES: Final[tuple[ShapeFamily, ...]] = ("square", "wide_tall", "tall_wide")

# Three NumPy spellings share one research candidate; they are legs inside each
# cell, not independent product cells.
Spelling = Literal["dot", "matmul", "matmul_op"]
SPELLINGS: Final[tuple[Spelling, ...]] = ("dot", "matmul", "matmul_op")
SPELLING_LABELS: Final[dict[Spelling, str]] = {
    "dot": "numpy.dot(a, b)",
    "matmul": "numpy.matmul(a, b)",
    "matmul_op": "a @ b",
}

# Timing legs: one candidate + three NumPy spellings (four subprocesses / cell).
TimingLeg = Literal["candidate", "dot", "matmul", "matmul_op"]
TIMING_LEGS: Final[tuple[TimingLeg, ...]] = ("candidate", "dot", "matmul", "matmul_op")
LEG_ORDER: Final[tuple[TimingLeg, ...]] = TIMING_LEGS  # sequential order

# Evidence counts (minimums). Smoke uses lower counts and is never evidence.
EVIDENCE_WARMUPS: Final[int] = 5
EVIDENCE_SAMPLES: Final[int] = 30
SMOKE_WARMUPS: Final[int] = 1
SMOKE_SAMPLES: Final[int] = 2

# Correctness tolerances against the saved NumPy reference.
RTOL: Final[float] = 1e-12
ATOL: Final[float] = 1e-12
EQUAL_NAN: Final[bool] = True
EXPECTED_DTYPE: Final[str] = "float64"

# Statistics.
BOOTSTRAP_RESAMPLES: Final[int] = 20_000
BOOTSTRAP_ALPHA: Final[float] = 0.05  # two-sided 95% CI
PERCENTILE_METHOD: Final[str] = "nearest-rank"
PERCENTILE_METHOD_DOC: Final[str] = (
    "Nearest-rank: for sorted ascending samples of length n >= 1, "
    "p = samples[ceil(p * n) - 1] with index clamped to [0, n - 1]."
)
GEOMETRIC_MEAN_THRESHOLD: Final[float] = 1.25
CI_LOWER_THRESHOLD: Final[float] = 1.0

# Deterministic input seed base (recorded in every report).
BASE_SEED: Final[int] = 2026_07_13
# Cell seed = BASE_SEED + cell_index * CELL_SEED_STRIDE (stable, independent).
CELL_SEED_STRIDE: Final[int] = 1_000_003
# Bootstrap seed = BASE_SEED + 10_000_000 + cell_index * 100 + spelling_code.
BOOTSTRAP_SEED_BASE: Final[int] = BASE_SEED + 10_000_000
SPELLING_SEED_CODE: Final[dict[Spelling, int]] = {
    "dot": 1,
    "matmul": 2,
    "matmul_op": 3,
}

# Thread environment applied BEFORE importing NumPy in every timing subprocess.
THREAD_ENV: Final[dict[str, str]] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}

# Dispatchability is known-failed at preregistration (rank/dtype, no dimensions).
DISPATCHABILITY_STATUS: Final[str] = "failed"
DISPATCHABILITY_REASON: Final[str] = (
    "Current annotations expose rank and dtype but not matrix dimensions; "
    "a size-dependent native policy is not expressible without runtime-shape "
    "guesses. Dispatchability gate fails a priori."
)
PRODUCT_VERDICT: Final[str] = "NO-GO"
PRODUCT_VERDICT_DETAIL: Final[str] = "fallback-retained"

# Rust candidate crate pins (exact).
CANDIDATE_CRATE_NAME: Final[str] = "matmul_wave2_candidate"
CANDIDATE_MODULE_NAME: Final[str] = "matmul_wave2_candidate"
CANDIDATE_FUNCTION_NAME: Final[str] = "matmul_f64"
NUMPY_CRATE_VERSION: Final[str] = "=0.29.0"
PYO3_CRATE_VERSION: Final[str] = "=0.29.0"
CARGO_BUILD_ARGS: Final[tuple[str, ...]] = ("build", "--release", "--locked")

# Provenance invalidation (evidence runs only).
INVALID_BLAS_MARKERS: Final[tuple[str, ...]] = (
    "unknown",
    "unavailable",
    "unidentified",
    "",
)

# Concrete BLAS/LAPACK vendor tokens required for evidence identification.
# A generic occurrence of "blas"/"lapack" alone is NOT identification.
BLAS_VENDOR_TOKENS: Final[tuple[tuple[str, str], ...]] = (
    ("openblas", "openblas"),
    ("mkl", "mkl"),
    ("accelerate", "accelerate"),
    ("blis", "blis"),
    ("atlas", "atlas"),
    ("flexiblas", "flexiblas"),
    ("veclib", "veclib"),
    ("libflame", "libflame"),
)

# Python harness sources that define protocol/measurement/statistics.
# Hashed into evidence provenance in addition to git revision + candidate sources.
HARNESS_SOURCE_FILES: Final[tuple[str, ...]] = (
    "protocol.py",
    "runner.py",
    "worker.py",
    "stats.py",
    "report.py",
    "candidate.py",
)


@dataclass(frozen=True)
class ShapeCell:
    """One of the 27 mandatory preregistered shape cells."""

    cell_index: int
    family: ShapeFamily
    n: int
    h: int
    left_shape: tuple[int, int]
    right_shape: tuple[int, int]
    out_shape: tuple[int, int]

    @property
    def cell_id(self) -> str:
        """Stable cell identifier: ``{family}_n{n}``."""
        return f"{self.family}_n{self.n}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable cell description."""
        d = asdict(self)
        d["cell_id"] = self.cell_id
        d["left_shape"] = list(self.left_shape)
        d["right_shape"] = list(self.right_shape)
        d["out_shape"] = list(self.out_shape)
        return d


def half_size(n: int) -> int:
    """Return ``h = max(1, n // 2)`` for rectangular families."""
    return max(1, n // 2)


def shapes_for_family(
    family: ShapeFamily, n: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return ``(left, right, out)`` shapes for a family and base size ``n``."""
    h = half_size(n)
    if family == "square":
        left = (n, n)
        right = (n, n)
        out = (n, n)
    elif family == "wide_tall":
        # (n, h) @ (h, n) → (n, n)
        left = (n, h)
        right = (h, n)
        out = (n, n)
    elif family == "tall_wide":
        # (h, n) @ (n, h) → (h, h)
        left = (h, n)
        right = (n, h)
        out = (h, h)
    else:  # pragma: no cover - Literal exhaustiveness
        raise ValueError(f"unknown shape family: {family!r}")
    return left, right, out


def build_shape_cells(
    base_sizes: Sequence[int] = BASE_SIZES,
    families: Sequence[ShapeFamily] = SHAPE_FAMILIES,
) -> tuple[ShapeCell, ...]:
    """Build the frozen 27-cell matrix in deterministic order.

    Order: for each ``n`` in *base_sizes*, for each family in *families*.
    Cell index is the position in that product (0-based).
    """
    cells: list[ShapeCell] = []
    idx = 0
    for n in base_sizes:
        h = half_size(int(n))
        for family in families:
            left, right, out = shapes_for_family(family, int(n))
            cells.append(
                ShapeCell(
                    cell_index=idx,
                    family=family,
                    n=int(n),
                    h=h,
                    left_shape=left,
                    right_shape=right,
                    out_shape=out,
                )
            )
            idx += 1
    return tuple(cells)


# Frozen matrix used by the runner and tests.
SHAPE_CELLS: Final[tuple[ShapeCell, ...]] = build_shape_cells()
assert len(SHAPE_CELLS) == 27, f"expected 27 cells, got {len(SHAPE_CELLS)}"
# Exact frozen cell ids in stable order (evidence gate / schema / recompute).
EXPECTED_CELL_IDS: Final[tuple[str, ...]] = tuple(c.cell_id for c in SHAPE_CELLS)
assert len(EXPECTED_CELL_IDS) == 27
assert len(set(EXPECTED_CELL_IDS)) == 27


def is_exact_frozen_matrix(cells: Sequence[ShapeCell]) -> bool:
    """Return True iff *cells* is the exact frozen 27-cell matrix in order."""
    if len(cells) != len(SHAPE_CELLS):
        return False
    for got, exp in zip(cells, SHAPE_CELLS, strict=True):
        if (
            got.cell_index != exp.cell_index
            or got.family != exp.family
            or got.n != exp.n
            or got.h != exp.h
            or got.left_shape != exp.left_shape
            or got.right_shape != exp.right_shape
            or got.out_shape != exp.out_shape
            or got.cell_id != exp.cell_id
        ):
            return False
    return True


def require_exact_frozen_matrix(cells: Sequence[ShapeCell]) -> None:
    """Raise ``ValueError`` unless *cells* is the exact frozen 27-cell matrix.

    Evidence runs must measure every preregistered cell in stable order with
    identical ids/shapes. Subsets are smoke/debug only.
    """
    if is_exact_frozen_matrix(cells):
        return
    got_ids = [getattr(c, "cell_id", None) for c in cells]
    raise ValueError(
        "evidence mode requires the exact frozen 27 ShapeCell matrix in stable "
        f"order/ids/shapes; got {len(cells)} cells with ids={got_ids!r}, "
        f"expected ids={list(EXPECTED_CELL_IDS)!r}"
    )


def report_cell_ids_match_frozen(cell_ids: Sequence[str]) -> bool:
    """Return True iff *cell_ids* exactly equals :data:`EXPECTED_CELL_IDS`."""
    return tuple(str(x) for x in cell_ids) == EXPECTED_CELL_IDS


def cell_input_seed(cell_index: int, *, base_seed: int = BASE_SEED) -> int:
    """Deterministic RNG seed for cell inputs."""
    return int(base_seed) + int(cell_index) * CELL_SEED_STRIDE


def bootstrap_seed(
    cell_index: int,
    spelling: Spelling,
    *,
    base: int = BOOTSTRAP_SEED_BASE,
) -> int:
    """Deterministic PCG64 seed for a cell's spelling bootstrap."""
    return int(base) + int(cell_index) * 100 + SPELLING_SEED_CODE[spelling]


def run_counts(*, evidence: bool) -> tuple[int, int]:
    """Return ``(warmups, samples)`` for evidence or smoke mode."""
    if evidence:
        return EVIDENCE_WARMUPS, EVIDENCE_SAMPLES
    return SMOKE_WARMUPS, SMOKE_SAMPLES


def protocol_manifest() -> dict[str, Any]:
    """Machine-readable frozen protocol snapshot for reports and tests."""
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "base_sizes": list(BASE_SIZES),
        "shape_families": list(SHAPE_FAMILIES),
        "n_cells": len(SHAPE_CELLS),
        "expected_cell_ids": list(EXPECTED_CELL_IDS),
        "cells": [c.to_dict() for c in SHAPE_CELLS],
        "spellings": list(SPELLINGS),
        "spelling_labels": dict(SPELLING_LABELS),
        "timing_legs": list(TIMING_LEGS),
        "leg_order": list(LEG_ORDER),
        "evidence_warmups": EVIDENCE_WARMUPS,
        "evidence_samples": EVIDENCE_SAMPLES,
        "smoke_warmups": SMOKE_WARMUPS,
        "smoke_samples": SMOKE_SAMPLES,
        "rtol": RTOL,
        "atol": ATOL,
        "equal_nan": EQUAL_NAN,
        "expected_dtype": EXPECTED_DTYPE,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_alpha": BOOTSTRAP_ALPHA,
        "percentile_method": PERCENTILE_METHOD,
        "percentile_method_doc": PERCENTILE_METHOD_DOC,
        "geometric_mean_threshold": GEOMETRIC_MEAN_THRESHOLD,
        "ci_lower_threshold": CI_LOWER_THRESHOLD,
        "base_seed": BASE_SEED,
        "cell_seed_stride": CELL_SEED_STRIDE,
        "bootstrap_seed_base": BOOTSTRAP_SEED_BASE,
        "spelling_seed_code": dict(SPELLING_SEED_CODE),
        "thread_env": dict(THREAD_ENV),
        "dispatchability_status": DISPATCHABILITY_STATUS,
        "dispatchability_reason": DISPATCHABILITY_REASON,
        "product_verdict": PRODUCT_VERDICT,
        "product_verdict_detail": PRODUCT_VERDICT_DETAIL,
        "candidate_crate_name": CANDIDATE_CRATE_NAME,
        "candidate_module_name": CANDIDATE_MODULE_NAME,
        "candidate_function_name": CANDIDATE_FUNCTION_NAME,
        "numpy_crate_version": NUMPY_CRATE_VERSION,
        "pyo3_crate_version": PYO3_CRATE_VERSION,
        "cargo_build_args": list(CARGO_BUILD_ARGS),
        "blas_vendor_tokens": [label for _, label in BLAS_VENDOR_TOKENS],
        "harness_source_files": list(HARNESS_SOURCE_FILES),
    }
