"""Focused lower coverage for whole-array sum/mean."""

from __future__ import annotations

import pytest

from rextio.plugins.api import ClaimSite, LoweringContext

from rextio_numpy.diagnostics import F64_1D, F64_2D, I64_1D
from rextio_numpy.lower import lower
from rextio_numpy.lower.reductions import try_lower

K = F64_1D


def site(target: str, operand_types: tuple[str, ...] = (K,)) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
    )


def ctx(*operands: str) -> LoweringContext:
    return LoweringContext(
        operands=tuple(operands),
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )


@pytest.mark.parametrize(
    ("target", "helper_name", "body"),
    [
        ("numpy.sum", "__rxtnp_sum1", "Ok(a.sum())"),
        ("numpy.mean", "__rxtnp_mean1", "Ok(a.mean().unwrap_or(f64::NAN))"),
    ],
)
def test_try_lower_reductions_f64_rank1(target: str, helper_name: str, body: str) -> None:
    lowered = try_lower(site(target), ctx("values"))
    assert lowered is not None
    assert lowered.rust == f"{helper_name}(&values)?"
    assert body in lowered.helpers[0]


def test_try_lower_i64_sum_wraps() -> None:
    lowered = try_lower(site("numpy.sum", (I64_1D,)), ctx("a"))
    assert lowered is not None
    assert lowered.rust == "__rxtnp_sum1_i64(&a)?"
    assert "wrapping_add" in lowered.helpers[0]
    assert "PyResult<i64>" in lowered.helpers[0]


def test_try_lower_i64_mean_helper_still_exists() -> None:
    """Lower helper remains for architecture symmetry; claim is the gate."""
    lowered = try_lower(site("numpy.mean", (I64_1D,)), ctx("a"))
    assert lowered is not None
    assert "as f64" in lowered.helpers[0]
    assert "PyResult<f64>" in lowered.helpers[0]


def test_try_lower_f64_rank2_sum() -> None:
    lowered = try_lower(site("numpy.sum", (F64_2D,)), ctx("a"))
    assert lowered is not None
    assert lowered.rust == "__rxtnp_sum2_f64(&a)?"
    assert "Ok(a.sum())" in lowered.helpers[0]


def test_try_lower_ignores_non_reduction() -> None:
    assert try_lower(site("numpy.dot"), ctx("a")) is None


def test_router_matches_try_lower() -> None:
    s = site("numpy.sum")
    c = ctx("a")
    assert lower(s, c) == try_lower(s, c)
