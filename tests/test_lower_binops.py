"""Focused lower coverage for elementwise binops."""

from __future__ import annotations

import pytest

from rextio.plugins.api import ClaimSite, LoweringContext

from rextio_numpy.diagnostics import F64_1D
from rextio_numpy.lower import lower
from rextio_numpy.lower.binops import try_lower

K = F64_1D


def site(target: str, operand_types: tuple[str, str]) -> ClaimSite:
    return ClaimSite(
        kind="binop",
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
    ("op", "name", "symbol"),
    [("+", "add", "+"), ("-", "sub", "-"), ("*", "mul", "*"), ("/", "div", "/")],
)
def test_try_lower_array_array(op: str, name: str, symbol: str) -> None:
    lowered = try_lower(site(op, (K, K)), ctx("a", "b"))
    assert lowered is not None
    assert lowered.rust == f"__rxtnp_{name}1_aa(&a, &b)?"
    assert f"Ok(a {symbol} b)" in lowered.helpers[0]


@pytest.mark.parametrize(
    ("op", "name"),
    [("+", "add"), ("-", "sub"), ("*", "mul"), ("/", "div")],
)
def test_try_lower_array_scalar_and_scalar_array(op: str, name: str) -> None:
    as_lowered = try_lower(site(op, (K, "float")), ctx("a", "s"))
    sa_lowered = try_lower(site(op, ("float", K)), ctx("s", "a"))
    assert as_lowered is not None
    assert sa_lowered is not None
    assert as_lowered.rust == f"__rxtnp_{name}1_as(&a, s)?"
    assert sa_lowered.rust == f"__rxtnp_{name}1_sa(s, &a)?"


def test_try_lower_ignores_non_binop() -> None:
    call_site = ClaimSite(
        kind="call",
        target="numpy.dot",
        operand_types=(K, K),
        file_path="",
        line=0,
        column=0,
    )
    assert try_lower(call_site, ctx("a", "b")) is None


def test_router_matches_try_lower() -> None:
    s = site("+", (K, K))
    c = ctx("x", "y")
    assert lower(s, c) == try_lower(s, c)
