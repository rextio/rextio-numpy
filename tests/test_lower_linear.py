"""Focused lower coverage for numpy.dot."""

from __future__ import annotations

import pytest

from rextio.plugins.api import ClaimSite, LoweringContext

from rextio_numpy.claim.linear import _DOT_RESULT, _DOT_RULE
from rextio_numpy.diagnostics import F32_1D, F64_1D, F64_2D, I64_1D, I64_2D, array_meta
from rextio_numpy.lower import lower
from rextio_numpy.lower.linear import try_lower

K = F64_1D


def site(
    target: str = "numpy.dot",
    operand_types: tuple[str | None, str | None] = (K, K),
) -> ClaimSite:
    meta = array_meta(operand_types[0])
    result_type = _DOT_RESULT[meta[0]] if meta is not None and meta[0] in _DOT_RESULT else "float"
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
        rule_id=_DOT_RULE,
        result_type=result_type,
    )


def ctx(*operands: str) -> LoweringContext:
    return LoweringContext(
        operands=tuple(operands),
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )


def test_try_lower_dot_f64() -> None:
    lowered = try_lower(site(), ctx("a", "b"))
    assert lowered is not None
    assert lowered.rust == "__rxtnp_dot1(&a, &b)?"
    assert lowered.helpers[0].startswith(
        "fn __rxtnp_dot1<'py>(a: &numpy::PyReadonlyArray1<'py, f64>, "
        "b: &numpy::PyReadonlyArray1<'py, f64>)"
    )
    assert "not aligned: {} (dim 0) != {} (dim 0)" in lowered.helpers[0]


def test_try_lower_dot_i64() -> None:
    lowered = try_lower(site(operand_types=(I64_1D, I64_1D)), ctx("a", "b"))
    assert lowered is not None
    assert lowered.rust == "__rxtnp_dot1_i64(&a, &b)?"
    assert "wrapping_mul" in lowered.helpers[0]


@pytest.mark.parametrize(
    "operand_types",
    [
        (F32_1D, F32_1D),
        (F64_2D, F64_2D),
        (F64_1D, I64_1D),
        (I64_1D, I64_2D),
        (F64_1D, "int"),
    ],
)
def test_try_lower_forged_dot_claim_fails_closed(operand_types: tuple[str, str]) -> None:
    with pytest.raises(ValueError, match="certified same-dtype rank-1 f64/i64"):
        try_lower(site(operand_types=operand_types), ctx("a", "b"))


def test_try_lower_ignores_non_dot() -> None:
    assert try_lower(site("numpy.sum"), ctx("a")) is None


def test_router_matches_try_lower() -> None:
    s = site()
    c = ctx("x", "y")
    assert lower(s, c) == try_lower(s, c)
