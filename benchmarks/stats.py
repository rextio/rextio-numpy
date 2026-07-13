"""Deterministic summary statistics for wall-time samples."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from benchmarks.models import PERCENTILE_METHOD, TimingSummary


def nearest_rank_percentile(samples: Sequence[float], p: float) -> float:
    """Return the nearest-rank percentile of *samples*.

    Samples are sorted ascending. For length ``n >= 1`` and ``0 < p <= 1``,
    the rank is ``ceil(p * n)`` (1-based), so the 0-based index is
    ``ceil(p * n) - 1``, clamped to ``[0, n - 1]``.

    Raises:
        ValueError: if *samples* is empty or *p* is not in ``(0, 1]``.
    """
    if not samples:
        raise ValueError("nearest_rank_percentile requires at least one sample")
    if not (0.0 < p <= 1.0):
        raise ValueError(f"percentile p must be in (0, 1], got {p!r}")
    ordered = sorted(float(x) for x in samples)
    n = len(ordered)
    # ceil(p * n) without importing math.ceil for floats carefully
    rank = max(1, min(n, math.ceil(p * n)))
    return ordered[rank - 1]


def summarize(samples: Sequence[float]) -> TimingSummary:
    """Compute median/mean/stdev/min/max/p95 for **per-call** wall samples (seconds).

    Callers must pass per-call wall latencies (batch elapsed / iterations), not
    raw batch elapsed times.

    Empty *samples* yields a summary with all statistics set to ``None`` and
    ``count == 0`` (callers should treat that as a measurement failure).
    """
    values = [float(x) for x in samples]
    n = len(values)
    if n == 0:
        return TimingSummary(
            median=None,
            mean=None,
            stdev=None,
            min=None,
            max=None,
            p95=None,
            count=0,
            percentile_method=PERCENTILE_METHOD,
        )
    median = statistics.median(values)
    mean = statistics.fmean(values)
    # population=False (sample stdev) for n >= 2; None for n == 1
    stdev: float | None
    if n >= 2:
        stdev = statistics.stdev(values)
    else:
        stdev = None
    return TimingSummary(
        median=float(median),
        mean=float(mean),
        stdev=None if stdev is None else float(stdev),
        min=float(min(values)),
        max=float(max(values)),
        p95=float(nearest_rank_percentile(values, 0.95)),
        count=n,
        percentile_method=PERCENTILE_METHOD,
    )


def speedup_ratio(
    fallback_median_s: float | None,
    native_median_s: float | None,
) -> float | None:
    """Return fallback/native **per-call** median ratio, or None if not computable.

    Both arguments must be per-call wall-latency medians (not raw batch elapsed).
    ``> 1`` means native was faster; ``< 1`` means fallback was faster.
    Zero or non-positive denominators yield ``None`` (no speedup claim).
    """
    if fallback_median_s is None or native_median_s is None:
        return None
    if native_median_s <= 0.0 or fallback_median_s < 0.0:
        return None
    return float(fallback_median_s) / float(native_median_s)
