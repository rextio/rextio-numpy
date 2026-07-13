"""Deterministic statistics for Wave 2 matmul research measurements.

Implements nearest-rank percentiles, independent two-sample percentile
bootstrap of median(NumPy)/median(candidate), cell-conservative aggregates,
and geometric mean of cell estimates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.matmul_wave2.protocol import (
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_RESAMPLES,
    CI_LOWER_THRESHOLD,
    EXPECTED_CELL_IDS,
    GEOMETRIC_MEAN_THRESHOLD,
    PERCENTILE_METHOD,
    PERCENTILE_METHOD_DOC,
    Spelling,
    report_cell_ids_match_frozen,
)


@dataclass(frozen=True)
class TimingSummary:
    """Summary over per-call wall latencies (seconds)."""

    median: float | None
    mean: float | None
    stdev: float | None
    min: float | None
    max: float | None
    count: int
    percentile_method: str = PERCENTILE_METHOD

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary."""
        return asdict(self)


@dataclass(frozen=True)
class BootstrapResult:
    """Independent two-sample percentile bootstrap of a median ratio."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    alpha: float
    n_resamples: int
    seed: int
    percentile_method: str
    percentile_method_doc: str
    numpy_median: float
    candidate_median: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable bootstrap result."""
        return asdict(self)


@dataclass(frozen=True)
class CellSpeedupAggregate:
    """Conservative cell aggregate across three NumPy spellings."""

    conservative_point: float
    conservative_ci_lower: float
    spelling_points: dict[str, float]
    spelling_ci_lowers: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable cell aggregate."""
        return asdict(self)


def nearest_rank_percentile(samples: Sequence[float], p: float) -> float:
    """Return the nearest-rank percentile of *samples*.

    For length ``n >= 1`` and ``0 < p <= 1``, rank is ``ceil(p * n)`` (1-based),
    so the 0-based index is ``ceil(p * n) - 1``, clamped to ``[0, n - 1]``.

    Raises:
        ValueError: if *samples* is empty or *p* is not in ``(0, 1]``.
    """
    if not samples:
        raise ValueError("nearest_rank_percentile requires at least one sample")
    if not (0.0 < p <= 1.0):
        raise ValueError(f"percentile p must be in (0, 1], got {p!r}")
    ordered = sorted(float(x) for x in samples)
    n = len(ordered)
    rank = max(1, min(n, math.ceil(p * n)))
    return ordered[rank - 1]


def summarize_per_call(samples: Sequence[float]) -> TimingSummary:
    """Summarize positive finite per-call wall samples (seconds)."""
    values = [float(x) for x in samples]
    n = len(values)
    if n == 0:
        return TimingSummary(
            median=None,
            mean=None,
            stdev=None,
            min=None,
            max=None,
            count=0,
        )
    ordered = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        median = ordered[mid]
    else:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])
    mean = math.fsum(values) / n
    if n >= 2:
        var = math.fsum((x - mean) ** 2 for x in values) / (n - 1)
        stdev: float | None = math.sqrt(var)
    else:
        stdev = None
    return TimingSummary(
        median=float(median),
        mean=float(mean),
        stdev=None if stdev is None else float(stdev),
        min=float(ordered[0]),
        max=float(ordered[-1]),
        count=n,
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(x) for x in values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def speedup_ratio(numpy_median: float, candidate_median: float) -> float:
    """Return ``median(NumPy) / median(candidate)``; ``>1`` means candidate faster."""
    if candidate_median <= 0.0:
        raise ValueError(f"candidate_median must be positive, got {candidate_median!r}")
    if numpy_median < 0.0:
        raise ValueError(f"numpy_median must be non-negative, got {numpy_median!r}")
    return float(numpy_median) / float(candidate_median)


def bootstrap_median_ratio(
    numpy_samples: Sequence[float],
    candidate_samples: Sequence[float],
    *,
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
) -> BootstrapResult:
    """Independent two-sample percentile bootstrap of median ratio.

    For each resample, draw with replacement independently from each sample
    series (sizes preserved), compute ``median(np*) / median(cand*)``, then
    form a two-sided ``(1 - alpha)`` CI via nearest-rank percentiles of the
    bootstrap distribution. Uses ``numpy.random.Generator(PCG64(seed))``.

    Raises:
        ValueError: on empty series, non-positive medians, or bad parameters.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples!r}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    np_vals = [float(x) for x in numpy_samples]
    cand_vals = [float(x) for x in candidate_samples]
    if not np_vals or not cand_vals:
        raise ValueError("bootstrap requires non-empty numpy and candidate samples")

    # Local import keeps the module importable without NumPy for pure-protocol
    # unit tests that never call bootstrap; the runner always has NumPy.
    import numpy as np

    np_arr = np.asarray(np_vals, dtype=np.float64)
    cand_arr = np.asarray(cand_vals, dtype=np.float64)
    np_med = float(np.median(np_arr))
    cand_med = float(np.median(cand_arr))
    if cand_med <= 0.0 or np_med < 0.0:
        raise ValueError(
            f"non-positive median in bootstrap inputs: numpy={np_med}, candidate={cand_med}"
        )
    point = speedup_ratio(np_med, cand_med)

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    n_np = np_arr.shape[0]
    n_cand = cand_arr.shape[0]
    ratios: list[float] = []
    for _ in range(int(n_resamples)):
        np_boot = rng.choice(np_arr, size=n_np, replace=True)
        cand_boot = rng.choice(cand_arr, size=n_cand, replace=True)
        b_np_med = float(np.median(np_boot))
        b_cand_med = float(np.median(cand_boot))
        if b_cand_med <= 0.0:
            # Degenerate resample: skip would bias; treat as +inf (candidate zero).
            # With positive wall times this branch is unreachable in practice.
            ratios.append(float("inf"))
            continue
        ratios.append(b_np_med / b_cand_med)

    # Finite ratios only for percentile CI; all-inf would be a hard failure.
    finite = [r for r in ratios if math.isfinite(r)]
    if len(finite) < n_resamples:
        raise ValueError(
            f"bootstrap produced {n_resamples - len(finite)} non-finite ratios; "
            "candidate samples may contain zeros"
        )
    lo_p = alpha / 2.0
    hi_p = 1.0 - (alpha / 2.0)
    # nearest-rank for p in (0, 1]: lo_p > 0 for alpha < 2.
    ci_lower = nearest_rank_percentile(finite, lo_p if lo_p > 0.0 else (1.0 / n_resamples))
    ci_upper = nearest_rank_percentile(finite, hi_p)

    return BootstrapResult(
        point_estimate=float(point),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        alpha=float(alpha),
        n_resamples=int(n_resamples),
        seed=int(seed),
        percentile_method=PERCENTILE_METHOD,
        percentile_method_doc=PERCENTILE_METHOD_DOC,
        numpy_median=np_med,
        candidate_median=cand_med,
    )


def cell_conservative_aggregate(
    spelling_bootstraps: dict[Spelling, BootstrapResult] | dict[str, BootstrapResult],
) -> CellSpeedupAggregate:
    """Conservative cell estimate = min of three spelling points and CI lowers."""
    if len(spelling_bootstraps) != 3:
        raise ValueError(f"expected 3 spelling bootstraps, got {len(spelling_bootstraps)}")
    points = {str(k): float(v.point_estimate) for k, v in spelling_bootstraps.items()}
    lowers = {str(k): float(v.ci_lower) for k, v in spelling_bootstraps.items()}
    return CellSpeedupAggregate(
        conservative_point=min(points.values()),
        conservative_ci_lower=min(lowers.values()),
        spelling_points=points,
        spelling_ci_lowers=lowers,
    )


def geometric_mean(values: Sequence[float]) -> float:
    """Return ``exp(fsum(log(x_i)) / n)`` for positive *values*.

    Raises:
        ValueError: if empty or any value is not strictly positive / finite.
    """
    vals = [float(x) for x in values]
    if not vals:
        raise ValueError("geometric_mean requires at least one value")
    for v in vals:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"geometric_mean requires positive finite values, got {v!r}")
    log_sum = math.fsum(math.log(v) for v in vals)
    return math.exp(log_sum / len(vals))


def performance_gate(
    cell_conservative_lowers: Sequence[float],
    cell_conservative_points: Sequence[float],
    *,
    cell_ids: Sequence[str] | None = None,
    ci_threshold: float = CI_LOWER_THRESHOLD,
    geom_threshold: float = GEOMETRIC_MEAN_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate the frozen performance go gate (research only).

    Passes only if:
    1. *cell_ids* is exactly the frozen 27-cell id list in stable order;
    2. every conservative CI lower bound is ``> ci_threshold``;
    3. the geometric mean of the 27 conservative point estimates is
       ``>= geom_threshold``.

    Length alone is not sufficient: a 27-long malformed of wrong ids cannot pass.
    """
    lowers = [float(x) for x in cell_conservative_lowers]
    points = [float(x) for x in cell_conservative_points]
    ids = None if cell_ids is None else [str(x) for x in cell_ids]
    n_expected = len(EXPECTED_CELL_IDS)
    if ids is None or not report_cell_ids_match_frozen(ids):
        return {
            "passed": False,
            "reason": (
                "performance gate requires the exact frozen 27 cell ids in stable "
                f"order; got {ids!r}"
            ),
            "geometric_mean": None,
            "min_ci_lower": min(lowers) if lowers else None,
            "all_ci_lowers_gt_one": False,
            "geometric_mean_ok": False,
            "matrix_ok": False,
        }
    if len(lowers) != n_expected or len(points) != n_expected:
        return {
            "passed": False,
            "reason": (
                f"expected {n_expected} cells, got lowers={len(lowers)} points={len(points)}"
            ),
            "geometric_mean": None,
            "min_ci_lower": min(lowers) if lowers else None,
            "all_ci_lowers_gt_one": False,
            "geometric_mean_ok": False,
            "matrix_ok": True,
        }
    all_ci = all(lo > ci_threshold for lo in lowers)
    try:
        gmean = geometric_mean(points)
    except ValueError as exc:
        return {
            "passed": False,
            "reason": f"geometric mean failed: {exc}",
            "geometric_mean": None,
            "min_ci_lower": min(lowers),
            "all_ci_lowers_gt_one": all_ci,
            "geometric_mean_ok": False,
            "matrix_ok": True,
        }
    gmean_ok = gmean >= geom_threshold
    passed = all_ci and gmean_ok
    reason_parts: list[str] = []
    if not all_ci:
        reason_parts.append(
            f"not all conservative CI lowers > {ci_threshold} (min={min(lowers)!r})"
        )
    if not gmean_ok:
        reason_parts.append(f"geometric mean {gmean!r} < {geom_threshold}")
    return {
        "passed": passed,
        "reason": None if passed else "; ".join(reason_parts),
        "geometric_mean": gmean,
        "min_ci_lower": min(lowers),
        "all_ci_lowers_gt_one": all_ci,
        "geometric_mean_ok": gmean_ok,
        "matrix_ok": True,
    }


def validate_positive_finite_samples(samples: Sequence[float], *, label: str) -> str | None:
    """Return an error message if any sample is missing, non-finite, or non-positive."""
    if not samples:
        return f"{label}: no samples"
    for i, raw in enumerate(samples):
        try:
            x = float(raw)
        except (TypeError, ValueError):
            return f"{label}: sample[{i}] is not a float ({raw!r})"
        if not math.isfinite(x):
            return f"{label}: sample[{i}] is non-finite ({x!r})"
        if x <= 0.0:
            return f"{label}: sample[{i}] is not positive ({x!r})"
    return None
