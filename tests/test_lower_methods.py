"""Lowering coverage for ndarray method parity."""

from __future__ import annotations

import pytest

from rextio.plugins.api import ClaimLiteral, ClaimSite, KeywordArg, LoweringContext, ReceiverMeta

from rextio_numpy.diagnostics import F32_1D, F64_1D, F64_2D, I64_1D, I64_2D
from rextio_numpy.claim.linear import _DOT_RESULT, _DOT_RULE
from rextio_numpy.claim.reductions import (
    _AXIS_RULE,
    _WHOLE_ARRAY_RULE,
    _WHOLE_EXTREMA_RULE,
    _axis_result_type,
    _whole_array_result_type,
)
from rextio_numpy.diagnostics import array_meta
from rextio_numpy.lower import lower


def method_site(
    method: str,
    receiver_type: str,
    operand_types: tuple[str, ...] = (),
    *,
    keywords: tuple[KeywordArg, ...] = (),
) -> ClaimSite:
    meta = array_meta(receiver_type)
    if method == "dot":
        rule_id = _DOT_RULE
        result_type = _DOT_RESULT[meta[0]] if meta is not None and meta[0] in _DOT_RESULT else "float"
    elif keywords:
        rule_id = _AXIS_RULE
        result_type = _axis_result_type(f"numpy.{method}", meta[0], meta[1]) if meta else "float"
    else:
        rule_id = _WHOLE_EXTREMA_RULE if method in {"max", "min"} else _WHOLE_ARRAY_RULE
        result_type = _whole_array_result_type(f"numpy.{method}", meta[0]) if meta else "float"
    return ClaimSite(
        kind="call", target=f"values.{method}", operand_types=operand_types, file_path="", line=0,
        column=0, keywords=keywords,
        receiver=ReceiverMeta(arg_type=receiver_type, expr_kind="attribute", is_safe=False),
        rule_id=rule_id,
        result_type=result_type,
    )


def ctx(receiver: str, *operands: str) -> LoweringContext:
    return LoweringContext(operands=operands, target_language="rust", fresh_name=lambda prefix: f"{prefix}_0", receiver=receiver)


def test_lower_method_dot_uses_once_evaluated_receiver() -> None:
    lowered = lower(method_site("dot", F64_1D, (F64_1D,)), ctx("__rextio_recv_0", "rhs"))
    assert lowered.rust == "__rxtnp_dot1(&__rextio_recv_0, &rhs)?"


def test_lower_method_sum_uses_receiver_not_positional_operand() -> None:
    lowered = lower(method_site("sum", F64_1D), ctx("__rextio_recv_0"))
    assert lowered.rust == "__rxtnp_sum1(&__rextio_recv_0)?"


@pytest.mark.parametrize(
    ("method", "receiver_type", "expected"),
    [
        ("max", I64_1D, "__rxtnp_max1_i64(&__rextio_recv_0)?"),
        ("min", I64_2D, "__rxtnp_min2_i64(&__rextio_recv_0)?"),
    ],
)
def test_lower_method_whole_i64_extrema(
    method: str,
    receiver_type: str,
    expected: str,
) -> None:
    lowered = lower(method_site(method, receiver_type), ctx("__rextio_recv_0"))
    assert lowered.rust == expected
    assert "which has no identity" in lowered.helpers[0]


def test_lower_method_axis_uses_receiver_not_positional_operand() -> None:
    keywords = (KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=-1)),)
    lowered = lower(method_site("mean", F64_2D, keywords=keywords), ctx("__rextio_recv_0"))
    assert lowered.rust == "__rxtnp_mean2_f64_axis1(py, &__rextio_recv_0)?"


def test_lower_forged_method_dot_rhs_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="certified same-dtype rank-1 f64/i64"):
        lower(method_site("dot", F64_1D, (I64_1D,)), ctx("receiver", "rhs"))


def test_lower_forged_method_reduction_dtype_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside certified dtype/rank matrix"):
        lower(method_site("sum", F32_1D), ctx("receiver"))
