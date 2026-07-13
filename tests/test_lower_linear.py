"""Focused lower coverage for numpy.dot."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweringContext

from rextio_numpy.diagnostics import F64_1D
from rextio_numpy.lower import lower
from rextio_numpy.lower.linear import try_lower

K = F64_1D


def site(target: str = "numpy.dot") -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=(K, K),
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


def test_try_lower_dot() -> None:
    lowered = try_lower(site(), ctx("a", "b"))
    assert lowered is not None
    assert lowered.rust == "__rxtnp_dot1(&a, &b)?"
    assert lowered.helpers[0].startswith(
        "fn __rxtnp_dot1(a: &numpy::ndarray::Array1<f64>, b: &numpy::ndarray::Array1<f64>)"
    )
    assert "not aligned: {} (dim 0) != {} (dim 0)" in lowered.helpers[0]


def test_try_lower_ignores_non_dot() -> None:
    assert try_lower(site("numpy.sum"), ctx("a")) is None


def test_router_matches_try_lower() -> None:
    s = site()
    c = ctx("x", "y")
    assert lower(s, c) == try_lower(s, c)
