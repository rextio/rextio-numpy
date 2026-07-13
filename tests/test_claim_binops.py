"""Focused claim coverage for elementwise binops."""

from __future__ import annotations

import pytest

from rextio.config.schema import RextioConfig
from rextio.plugins.api import Claimed, ClaimSite, NotCovered, Rejected

from rextio_numpy.claim import claim
from rextio_numpy.claim.binops import try_claim
from rextio_numpy.diagnostics import F64_1D

K = F64_1D
CONFIG = RextioConfig()


def site(target: str, operand_types: tuple[str | None, ...]) -> ClaimSite:
    return ClaimSite(
        kind="binop",
        target=target,
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
    )


@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
@pytest.mark.parametrize("operands", [(K, K), (K, "float"), ("float", K)])
def test_try_claim_supported_binops(op: str, operands: tuple[str, str]) -> None:
    result = try_claim(site(op, operands))
    assert result == Claimed(rule_id="rextio-numpy/elementwise-float64", result_type=K)


def test_try_claim_ignores_non_binop() -> None:
    call_site = ClaimSite(
        kind="call",
        target="numpy.dot",
        operand_types=(K, K),
        file_path="",
        line=0,
        column=0,
    )
    assert try_claim(call_site) is None


def test_try_claim_unsupported_op_is_none() -> None:
    assert try_claim(site("%", (K, K))) is None
    assert try_claim(site("@", (K, K))) is None


def test_try_claim_no_plugin_operand_is_not_covered() -> None:
    assert try_claim(site("+", ("float", "float"))) == NotCovered()


def test_try_claim_bad_operand_types_rejected() -> None:
    result = try_claim(site("+", (K, "int")))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"


def test_router_matches_try_claim() -> None:
    s = site("*", (K, "float"))
    assert claim(s, CONFIG) == try_claim(s)
