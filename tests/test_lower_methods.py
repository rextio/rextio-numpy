"""Lowering coverage for ndarray method parity."""

from __future__ import annotations

import pytest

from rextio.plugins.api import ClaimLiteral, ClaimSite, KeywordArg, LoweringContext, ReceiverMeta

from rextio_numpy.diagnostics import F32_1D, F64_1D, F64_2D, I64_1D
from rextio_numpy.lower import lower


def method_site(
    method: str,
    receiver_type: str,
    operand_types: tuple[str, ...] = (),
    *,
    keywords: tuple[KeywordArg, ...] = (),
) -> ClaimSite:
    return ClaimSite(
        kind="call", target=f"values.{method}", operand_types=operand_types, file_path="", line=0,
        column=0, keywords=keywords,
        receiver=ReceiverMeta(arg_type=receiver_type, expr_kind="attribute", is_safe=False),
    )


def ctx(receiver: str, *operands: str) -> LoweringContext:
    return LoweringContext(operands=operands, target_language="rust", fresh_name=lambda prefix: f"{prefix}_0", receiver=receiver)


def test_lower_method_dot_uses_once_evaluated_receiver() -> None:
    lowered = lower(method_site("dot", F64_1D, (F64_1D,)), ctx("__rextio_recv_0", "rhs"))
    assert lowered.rust == "__rxtnp_dot1(&__rextio_recv_0, &rhs)?"


def test_lower_method_sum_uses_receiver_not_positional_operand() -> None:
    lowered = lower(method_site("sum", F64_1D), ctx("__rextio_recv_0"))
    assert lowered.rust == "__rxtnp_sum1(&__rextio_recv_0)?"


def test_lower_method_axis_uses_receiver_not_positional_operand() -> None:
    keywords = (KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=-1)),)
    lowered = lower(method_site("mean", F64_2D, keywords=keywords), ctx("__rextio_recv_0"))
    assert lowered.rust == "__rxtnp_mean2_f64_axis1(&__rextio_recv_0)?"


def test_lower_forged_method_dot_rhs_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="certified same-dtype rank-1 f64/i64"):
        lower(method_site("dot", F64_1D, (I64_1D,)), ctx("receiver", "rhs"))


def test_lower_forged_method_reduction_dtype_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside certified dtype/rank matrix"):
        lower(method_site("sum", F32_1D), ctx("receiver"))
