"""Lowering coverage for exact unary NumPy module calls."""

from __future__ import annotations

import pytest

from rextio.plugins.api import ClaimSite, LoweringContext

from rextio_numpy.diagnostics import F64_1D, I64_2D
from rextio_numpy.claim.unary import _RULE
from rextio_numpy.lower import lower


def site(target: str, operand_type: str) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=(operand_type,),
        file_path="",
        line=0,
        column=0,
        rule_id=_RULE,
        result_type=operand_type,
    )


def ctx() -> LoweringContext:
    return LoweringContext(operands=("values",), target_language="rust", fresh_name=lambda prefix: f"{prefix}_0")


@pytest.mark.parametrize(
    ("target", "helper", "fragment"),
    [
        ("numpy.negative", "__rxtnp_negative1_f64", "-x"),
        ("numpy.absolute", "__rxtnp_absolute1_f64", "x.abs()"),
        ("numpy.abs", "__rxtnp_absolute1_f64", "x.abs()"),
        ("numpy.square", "__rxtnp_square1_f64", "x * x"),
    ],
)
def test_lower_float_unary(target: str, helper: str, fragment: str) -> None:
    lowered = lower(site(target, F64_1D), ctx())
    assert lowered.rust == f"{helper}(py, &values)?"
    assert fragment in lowered.helpers[-1]
    assert "PyArray1::<f64>::zeros" in lowered.helpers[0]


@pytest.mark.parametrize("target", ["numpy.negative", "numpy.absolute", "numpy.square"])
def test_lower_i64_unary_wraps(target: str) -> None:
    lowered = lower(site(target, I64_2D), ctx())
    assert "wrapping_" in lowered.helpers[0]
