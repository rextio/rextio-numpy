"""Deterministic unit tests for benchmark statistics and comparison rules."""

from __future__ import annotations

import math

import pytest

from benchmarks.compare import array_equals, results_equivalent, scalar_close
from benchmarks.measure import normalize_batch_to_per_call
from benchmarks.stats import nearest_rank_percentile, speedup_ratio, summarize


class TestNearestRankPercentile:
    def test_p95_known_sequence(self) -> None:
        # n=20 → ceil(0.95*20)=19 → index 18 → 19th value of 1..20 = 19
        samples = list(range(1, 21))
        assert nearest_rank_percentile(samples, 0.95) == 19.0

    def test_p95_unsorted(self) -> None:
        samples = [10.0, 1.0, 5.0, 3.0, 2.0, 9.0, 8.0, 7.0, 6.0, 4.0]
        # n=10 → ceil(9.5)=10 → index 9 → max = 10
        assert nearest_rank_percentile(samples, 0.95) == 10.0

    def test_single_sample(self) -> None:
        assert nearest_rank_percentile([3.5], 0.95) == 3.5

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            nearest_rank_percentile([], 0.95)

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError):
            nearest_rank_percentile([1.0], 0.0)
        with pytest.raises(ValueError):
            nearest_rank_percentile([1.0], 1.5)


class TestSummarize:
    def test_full_stats(self) -> None:
        samples = [1.0, 2.0, 3.0, 4.0, 100.0]
        s = summarize(samples)
        assert s.count == 5
        assert s.median == 3.0
        assert s.mean == pytest.approx(22.0)
        assert s.min == 1.0
        assert s.max == 100.0
        assert s.p95 == 100.0  # n=5 → ceil(4.75)=5 → index 4
        assert s.stdev is not None
        assert s.stdev == pytest.approx(
            math.sqrt(
                ((1 - 22) ** 2 + (2 - 22) ** 2 + (3 - 22) ** 2 + (4 - 22) ** 2 + (100 - 22) ** 2)
                / 4
            )
        )
        assert s.percentile_method == "nearest-rank"

    def test_empty_nulls(self) -> None:
        s = summarize([])
        assert s.count == 0
        assert s.median is None
        assert s.mean is None
        assert s.stdev is None
        assert s.min is None
        assert s.max is None
        assert s.p95 is None

    def test_single_sample_stdev_none(self) -> None:
        s = summarize([4.0])
        assert s.median == 4.0
        assert s.mean == 4.0
        assert s.stdev is None
        assert s.p95 == 4.0


class TestSpeedupRatio:
    def test_native_faster(self) -> None:
        assert speedup_ratio(2.0, 1.0) == pytest.approx(2.0)

    def test_fallback_faster_negative_speedup_honest(self) -> None:
        # <1 is valid — native loss
        assert speedup_ratio(1.0, 2.0) == pytest.approx(0.5)

    def test_none_on_missing_or_zero(self) -> None:
        assert speedup_ratio(None, 1.0) is None
        assert speedup_ratio(1.0, None) is None
        assert speedup_ratio(1.0, 0.0) is None
        assert speedup_ratio(-1.0, 1.0) is None

    def test_uses_per_call_medians_not_raw_batch(self) -> None:
        """Speedup is fallback_per_call_median / native_per_call_median."""
        iterations = 10
        # Raw batch times would yield a wrong ratio if used unnormalized.
        fb_batch = [0.20, 0.20, 0.20]  # 10 calls → 0.02 per call
        nt_batch = [0.10, 0.10, 0.10]  # 10 calls → 0.01 per call
        fb_med = summarize(normalize_batch_to_per_call(fb_batch, iterations)).median
        nt_med = summarize(normalize_batch_to_per_call(nt_batch, iterations)).median
        assert speedup_ratio(fb_med, nt_med) == pytest.approx(2.0)
        # Using raw batch medians would coincidentally match here, but if
        # iterations differed the ratio would be wrong — document the contract.
        assert fb_med == pytest.approx(0.02)
        assert nt_med == pytest.approx(0.01)


class TestNormalizeBatchToPerCall:
    def test_identity_when_iterations_is_one(self) -> None:
        assert normalize_batch_to_per_call([1.5, 2.5], 1) == pytest.approx([1.5, 2.5])

    def test_divides_when_iterations_gt_one(self) -> None:
        assert normalize_batch_to_per_call([1.0, 2.0, 3.0], 4) == pytest.approx([0.25, 0.5, 0.75])

    def test_rejects_non_positive_iterations(self) -> None:
        with pytest.raises(ValueError):
            normalize_batch_to_per_call([1.0], 0)
        with pytest.raises(ValueError):
            normalize_batch_to_per_call([1.0], -3)


class TestArrayEquals:
    def test_exact_match(self) -> None:
        np = pytest.importorskip("numpy")
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert array_equals(a, b)

    def test_nan_equals_nan(self) -> None:
        np = pytest.importorskip("numpy")
        a = np.array([1.0, float("nan"), 3.0])
        b = np.array([1.0, float("nan"), 3.0])
        assert array_equals(a, b)

    def test_mismatch(self) -> None:
        np = pytest.importorskip("numpy")
        assert not array_equals(np.array([1.0]), np.array([2.0]))
        assert not array_equals(np.array([1.0, 2.0]), np.array([1.0]))

    def test_dtype_mismatch_rejected(self) -> None:
        """Equal values after cast must not match when dtypes differ."""
        np = pytest.importorskip("numpy")
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([1.0, 2.0], dtype=np.float32)
        assert not array_equals(a, b)
        assert not results_equivalent(a, b, kind="array")


class TestScalarClose:
    def test_float_and_numpy_float(self) -> None:
        np = pytest.importorskip("numpy")
        assert scalar_close(1.0, np.float64(1.0))

    def test_nan_equals_nan(self) -> None:
        assert scalar_close(float("nan"), float("nan"))

    def test_within_tol(self) -> None:
        assert scalar_close(1.0, 1.0 + 1e-13)

    def test_outside_tol(self) -> None:
        assert not scalar_close(1.0, 1.0 + 1e-6)

    def test_results_equivalent_dispatch(self) -> None:
        np = pytest.importorskip("numpy")
        assert results_equivalent(np.array([1.0]), np.array([1.0]), kind="array")
        assert results_equivalent(1.0, 1.0, kind="scalar")
        with pytest.raises(ValueError):
            results_equivalent(1, 1, kind="nope")
