"""Focused claim coverage for whole-array sum/mean reductions."""

from __future__ import annotations

import pytest

from rextio.config.schema import RextioConfig
from rextio.plugins.api import Claimed, ClaimSite, NotCovered, Rejected

from rextio_numpy.claim import claim
from rextio_numpy.claim.reductions import try_claim
from rextio_numpy.diagnostics import F64_1D

K = F64_1D
CONFIG = RextioConfig()


def site(target: str, operand_types: tuple[str | None, ...]) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
    )


@pytest.mark.parametrize("target", ["numpy.sum", "numpy.mean"])
def test_try_claim_supported_reductions(target: str) -> None:
    result = try_claim(site(target, (K,)))
    assert result == Claimed(rule_id="rextio-numpy/reduction-sum-mean", result_type="float")


def test_try_claim_ignores_non_reduction() -> None:
    assert try_claim(site("numpy.dot", (K, K))) is None


def test_try_claim_wrong_arity_not_covered() -> None:
    assert try_claim(site("numpy.sum", (K, K))) == NotCovered()
    assert try_claim(site("numpy.mean", (K, "int"))) == NotCovered()


def test_try_claim_bad_operand_rejected() -> None:
    result = try_claim(site("numpy.sum", ("list[float]",)))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"


def test_router_matches_try_claim() -> None:
    s = site("numpy.mean", (K,))
    assert claim(s, CONFIG) == try_claim(s)
