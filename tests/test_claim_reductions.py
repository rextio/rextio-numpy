"""Focused claim coverage for whole-array sum/mean reductions."""

from __future__ import annotations

import pytest

from rextio.config.schema import RextioConfig
from rextio.plugins.api import Claimed, ClaimSite, NotCovered, Rejected

from rextio_numpy.claim import claim
from rextio_numpy.claim.reductions import try_claim
from rextio_numpy.diagnostics import F32_1D, F32_2D, F64_1D, F64_2D, I64_1D, I64_2D

K = F64_1D
CONFIG = RextioConfig()

_F64_ARRAY_KEYS = [F64_1D, F64_2D]
_I64_ARRAY_KEYS = [I64_1D, I64_2D]
_F32_ARRAY_KEYS = [F32_1D, F32_2D]


def site(target: str, operand_types: tuple[str | None, ...]) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
    )


@pytest.mark.parametrize("key", _F64_ARRAY_KEYS)
@pytest.mark.parametrize("target", ["numpy.sum", "numpy.mean"])
def test_try_claim_f64_reductions(target: str, key: str) -> None:
    result = try_claim(site(target, (key,)))
    assert result == Claimed(rule_id="rextio-numpy/reduction-sum-mean", result_type="float")


@pytest.mark.parametrize("key", _I64_ARRAY_KEYS)
def test_try_claim_i64_sum_claimed_int(key: str) -> None:
    result = try_claim(site("numpy.sum", (key,)))
    assert result == Claimed(rule_id="rextio-numpy/reduction-sum-mean", result_type="int")


@pytest.mark.parametrize("key", _I64_ARRAY_KEYS)
def test_try_claim_i64_mean_rejected(key: str) -> None:
    """int64 mean must not lower: sequential i64→f64 sum diverges from NumPy."""
    result = try_claim(site("numpy.mean", (key,)))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"


@pytest.mark.parametrize("target", ["numpy.sum", "numpy.mean"])
@pytest.mark.parametrize("key", _F32_ARRAY_KEYS)
def test_try_claim_f32_reductions_rejected(target: str, key: str) -> None:
    """float32 sum/mean must not lower: no enforceable size gate for FP contract."""
    result = try_claim(site(target, (key,)))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"


def test_try_claim_ignores_non_reduction() -> None:
    assert try_claim(site("numpy.dot", (K, K))) is None


def test_try_claim_wrong_arity_not_covered() -> None:
    assert try_claim(site("numpy.sum", (K, K))) == NotCovered()
    assert try_claim(site("numpy.mean", (K, "int"))) == NotCovered()


def test_try_claim_bad_operand_rejected() -> None:
    result = try_claim(site("numpy.sum", ("list[float]",)))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"


def test_try_claim_unresolved_not_covered() -> None:
    assert try_claim(site("numpy.sum", (None,))) == NotCovered()


def test_router_matches_try_claim() -> None:
    s = site("numpy.mean", (K,))
    assert claim(s, CONFIG) == try_claim(s)
    s2 = site("numpy.sum", (I64_2D,))
    assert claim(s2, CONFIG) == try_claim(s2)
    s3 = site("numpy.mean", (I64_1D,))
    assert claim(s3, CONFIG) == try_claim(s3)
