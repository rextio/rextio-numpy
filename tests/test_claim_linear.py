"""Focused claim coverage for numpy.dot."""

from __future__ import annotations

from rextio.config.schema import RextioConfig
from rextio.plugins.api import Claimed, ClaimSite, NotCovered, Rejected

from rextio_numpy.claim import claim
from rextio_numpy.claim.linear import try_claim
from rextio_numpy.diagnostics import F64_1D

K = F64_1D
CONFIG = RextioConfig()


def site(operand_types: tuple[str | None, ...], target: str = "numpy.dot") -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
    )


def test_try_claim_supported_dot() -> None:
    result = try_claim(site((K, K)))
    assert result == Claimed(rule_id="rextio-numpy/dot-float64", result_type="float")


def test_try_claim_ignores_non_dot() -> None:
    assert try_claim(site((K,), target="numpy.sum")) is None


def test_try_claim_wrong_arity_not_covered() -> None:
    assert try_claim(site((K,))) == NotCovered()
    assert try_claim(site((K, K, K))) == NotCovered()


def test_try_claim_unresolved_not_covered() -> None:
    assert try_claim(site((K, None))) == NotCovered()


def test_try_claim_bad_operand_rejected() -> None:
    result = try_claim(site((K, "int")))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"
    assert "numpy.dot" in result.diagnostic.message


def test_router_matches_try_claim() -> None:
    s = site((K, K))
    assert claim(s, CONFIG) == try_claim(s)
