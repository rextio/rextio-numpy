"""Claim coverage for certified ndarray method parity."""

from __future__ import annotations

import pytest

from rextio.config.schema import RextioConfig
from rextio.plugins.api import Claimed, ClaimLiteral, ClaimSite, KeywordArg, NotCovered, ReceiverMeta

from rextio_numpy.claim import claim
from rextio_numpy.claim.linear import try_claim as try_claim_dot
from rextio_numpy.claim.reductions import try_claim as try_claim_reduction
from rextio_numpy.diagnostics import F32_1D, F32_2D, F64_1D, F64_2D, I64_1D, I64_2D

CONFIG = RextioConfig()


def method_site(
    method: str,
    receiver_type: str,
    operand_types: tuple[str | None, ...] = (),
    *,
    keywords: tuple[KeywordArg, ...] = (),
) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=f"values.{method}",
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
        keywords=keywords,
        receiver=ReceiverMeta(arg_type=receiver_type, expr_kind="name", is_safe=True),
    )


def axis_kw(axis: int) -> tuple[KeywordArg, ...]:
    return (
        KeywordArg(
            name="axis",
            arg_type="int",
            literal=ClaimLiteral(is_literal=True, value=axis),
        ),
    )


@pytest.mark.parametrize(
    ("receiver", "argument", "result"),
    [(F64_1D, F64_1D, "float"), (I64_1D, I64_1D, "int")],
)
def test_claim_method_dot_same_as_module(receiver: str, argument: str, result: str) -> None:
    site = method_site("dot", receiver, (argument,))
    assert try_claim_dot(site) == Claimed("rextio-numpy/dot-float64", result)
    assert claim(site, CONFIG) == try_claim_dot(site)


@pytest.mark.parametrize(
    ("method", "receiver", "result"),
    [
        ("sum", F64_1D, "float"),
        ("sum", F64_2D, "float"),
        ("sum", I64_1D, "int"),
        ("mean", F64_1D, "float"),
        ("mean", F64_2D, "float"),
    ],
)
def test_claim_whole_array_methods(method: str, receiver: str, result: str) -> None:
    site = method_site(method, receiver)
    assert try_claim_reduction(site) == Claimed("rextio-numpy/reduction-sum-mean", result)
    assert claim(site, CONFIG) == try_claim_reduction(site)


@pytest.mark.parametrize(
    ("method", "receiver", "axis", "result"),
    [
        ("sum", F64_1D, -1, "float"),
        ("mean", F64_1D, 0, "float"),
        ("max", F64_1D, 0, "float"),
        ("min", I64_1D, 0, "int"),
        ("sum", F64_2D, 0, F64_1D),
        ("mean", F64_2D, 1, F64_1D),
        ("max", F32_2D, 0, F32_1D),
        ("min", I64_2D, -1, I64_1D),
    ],
)
def test_claim_literal_axis_methods(method: str, receiver: str, axis: int, result: str) -> None:
    assert try_claim_reduction(method_site(method, receiver, keywords=axis_kw(axis))) == Claimed(
        "rextio-numpy/reduction-axis", result
    )


@pytest.mark.parametrize(
    "site",
    [
        method_site("dot", F64_1D),
        method_site("dot", F64_1D, (F64_2D,)),
        method_site("dot", F32_1D, (F32_1D,)),
        method_site("sum", F32_1D),
        method_site("mean", I64_1D),
        method_site("max", F64_1D),
        method_site("min", F64_1D),
        method_site("sum", F64_1D, keywords=axis_kw(2)),
        method_site("sum", F64_1D, keywords=(KeywordArg(name="out", arg_type="None"),)),
    ],
)
def test_unsupported_method_forms_do_not_broaden_surface(site: ClaimSite) -> None:
    assert claim(site, CONFIG) == NotCovered() or claim(site, CONFIG).diagnostic.code == "RXTP-NUMPY-010"

