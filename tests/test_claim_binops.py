"""Focused claim coverage for elementwise binops."""

from __future__ import annotations

import pytest

from rextio.config.schema import RextioConfig
from rextio.plugins.api import Claimed, ClaimSite, NotCovered, Rejected

from rextio_numpy.claim import claim
from rextio_numpy.claim.binops import try_claim
from rextio_numpy.diagnostics import (
    F32_1D,
    F32_2D,
    F64_1D,
    F64_2D,
    I64_1D,
    I64_2D,
)

K = F64_1D
CONFIG = RextioConfig()

_ARRAY_KEYS = [F64_1D, F64_2D, F32_1D, F32_2D, I64_1D, I64_2D]
_SCALAR = {
    F64_1D: "float",
    F64_2D: "float",
    F32_1D: "float",
    F32_2D: "float",
    I64_1D: "int",
    I64_2D: "int",
}
_SAME_RESULT = {
    F64_1D: F64_1D,
    F64_2D: F64_2D,
    F32_1D: F32_1D,
    F32_2D: F32_2D,
    I64_1D: I64_1D,
    I64_2D: I64_2D,
}
_DIV_RESULT = {
    F64_1D: F64_1D,
    F64_2D: F64_2D,
    F32_1D: F32_1D,
    F32_2D: F32_2D,
    I64_1D: F64_1D,
    I64_2D: F64_2D,
}


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
@pytest.mark.parametrize("key", _ARRAY_KEYS)
def test_try_claim_same_dtype_array_array(op: str, key: str) -> None:
    result = try_claim(site(op, (key, key)))
    expected = _DIV_RESULT[key] if op == "/" else _SAME_RESULT[key]
    assert result == Claimed(rule_id="rextio-numpy/elementwise-float64", result_type=expected)


@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
@pytest.mark.parametrize("key", _ARRAY_KEYS)
def test_try_claim_array_scalar_and_scalar_array(op: str, key: str) -> None:
    scalar = _SCALAR[key]
    expected = _DIV_RESULT[key] if op == "/" else _SAME_RESULT[key]
    assert try_claim(site(op, (key, scalar))) == Claimed(
        rule_id="rextio-numpy/elementwise-float64", result_type=expected
    )
    assert try_claim(site(op, (scalar, key))) == Claimed(
        rule_id="rextio-numpy/elementwise-float64", result_type=expected
    )


@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
@pytest.mark.parametrize(
    ("left", "right", "result"),
    [
        (F64_1D, F64_2D, F64_2D),
        (F64_2D, F64_1D, F64_2D),
        (F32_1D, F32_2D, F32_2D),
        (I64_1D, I64_2D, I64_2D),
        (I64_2D, I64_1D, I64_2D),
    ],
)
def test_try_claim_mixed_rank_broadcast_result(op: str, left: str, right: str, result: str) -> None:
    expected = F64_2D if op == "/" and left.startswith("rextio-numpy/i64") else result
    # int64 true division always yields f64 at broadcast rank.
    if op == "/" and "i64" in left:
        expected = F64_2D
    assert try_claim(site(op, (left, right))) == Claimed(
        rule_id="rextio-numpy/elementwise-float64", result_type=expected
    )


def test_try_claim_legacy_float64_rank1() -> None:
    for op in ("+", "-", "*", "/"):
        for operands in ((K, K), (K, "float"), ("float", K)):
            assert try_claim(site(op, operands)) == Claimed(
                rule_id="rextio-numpy/elementwise-float64", result_type=K
            )


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
    assert try_claim(site("+", ("int", "int"))) == NotCovered()


def test_try_claim_mixed_array_dtypes_rejected() -> None:
    result = try_claim(site("+", (F64_1D, F32_1D)))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"


def test_try_claim_bad_operand_types_rejected() -> None:
    result = try_claim(site("+", (K, "int")))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"
    result = try_claim(site("+", (I64_1D, "float")))
    assert isinstance(result, Rejected)


def test_try_claim_unresolved_not_covered() -> None:
    assert try_claim(site("+", (K, None))) == NotCovered()
    assert try_claim(site("*", (None, I64_2D))) == NotCovered()


def test_router_matches_try_claim() -> None:
    s = site("*", (K, "float"))
    assert claim(s, CONFIG) == try_claim(s)
    s2 = site("/", (I64_1D, I64_2D))
    assert claim(s2, CONFIG) == try_claim(s2)
