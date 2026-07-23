"""Claim/lower coverage for exact binary NumPy ufunc call aliases."""

from __future__ import annotations

import pytest

from rextio.config.schema import RextioConfig
from rextio.plugins.api import (
    Claimed,
    ClaimLiteral,
    ClaimSite,
    KeywordArg,
    LoweringContext,
    NotCovered,
    ReceiverMeta,
    Rejected,
)

from rextio_numpy.claim import claim
from rextio_numpy.claim.binops import (
    UFUNC_CALL_NAMES,
    _ELEMENTWISE_UFUNC_RULE,
    _result_array_type,
)
from rextio_numpy.diagnostics import F32_1D, F64_1D, F64_2D, I64_1D, I64_2D
from rextio_numpy.lower import lower

CONFIG = RextioConfig()


def site(
    target: str,
    operand_types: tuple[str | None, ...],
    *,
    keywords: tuple[KeywordArg, ...] = (),
    receiver: ReceiverMeta | None = None,
    result_type: str | None = None,
    operand_literals: tuple[ClaimLiteral, ...] = (),
) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
        receiver=receiver,
        keywords=keywords,
        operand_literals=operand_literals,
        rule_id=_ELEMENTWISE_UFUNC_RULE if result_type is not None else None,
        result_type=result_type,
    )


def ctx(*operands: str) -> LoweringContext:
    return LoweringContext(
        operands=tuple(operands),
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )


@pytest.mark.parametrize(
    ("target", "operator", "helper"),
    [
        ("numpy.add", "+", "add"),
        ("numpy.subtract", "-", "sub"),
        ("numpy.multiply", "*", "mul"),
        ("numpy.divide", "/", "div"),
    ],
)
def test_exact_ufunc_calls_claim_and_lower(
    target: str, operator: str, helper: str
) -> None:
    expected_type = _result_array_type("f64", 1, operator)
    candidate = site(target, (F64_1D, F64_1D))
    assert claim(candidate, CONFIG) == Claimed(
        rule_id=_ELEMENTWISE_UFUNC_RULE,
        result_type=expected_type,
    )
    lowered = lower(
        site(target, (F64_1D, F64_1D), result_type=expected_type),
        ctx("a", "b"),
    )
    assert lowered.rust == f"__rxtnp_{helper}1_aa(&a, &b)?"


@pytest.mark.parametrize("target", sorted(UFUNC_CALL_NAMES))
def test_ufunc_calls_reuse_array_scalar_and_broadcast_matrix(target: str) -> None:
    operator = UFUNC_CALL_NAMES[target]
    scalar_type = _result_array_type("f32", 1, operator)
    assert claim(site(target, (F32_1D, "float")), CONFIG) == Claimed(
        rule_id=_ELEMENTWISE_UFUNC_RULE,
        result_type=scalar_type,
    )
    broadcast_type = _result_array_type("f64", 2, operator)
    assert claim(site(target, (F64_1D, F64_2D)), CONFIG) == Claimed(
        rule_id=_ELEMENTWISE_UFUNC_RULE,
        result_type=broadcast_type,
    )


def test_i64_divide_call_promotes_to_float64_at_broadcast_rank() -> None:
    candidate = site("numpy.divide", (I64_1D, I64_2D))
    assert claim(candidate, CONFIG) == Claimed(
        rule_id=_ELEMENTWISE_UFUNC_RULE,
        result_type=F64_2D,
    )
    lowered = lower(
        site("numpy.divide", (I64_1D, I64_2D), result_type=F64_2D),
        ctx("a", "b"),
    )
    assert lowered.rust == "__rxtnp_div12_aa_i64(&a, &b)?"
    assert "as f64" in lowered.helpers[-1]


@pytest.mark.parametrize("name", ["out", "where", "dtype", "casting"])
def test_ufunc_optional_keywords_are_rejected(name: str) -> None:
    candidate = site(
        "numpy.add",
        (F64_1D, F64_1D),
        keywords=(KeywordArg(name=name, arg_type="None"),),
    )
    result = claim(candidate, CONFIG)
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"


def test_ufunc_receiver_extra_arity_mixed_dtype_and_unresolved_fail_closed() -> None:
    receiver = ReceiverMeta(arg_type=F64_1D, expr_kind="name", is_safe=True)
    assert isinstance(
        claim(site("numpy.add", (F64_1D, F64_1D), receiver=receiver), CONFIG),
        Rejected,
    )
    assert isinstance(
        claim(site("numpy.add", (F64_1D, F64_1D, F64_1D)), CONFIG),
        Rejected,
    )
    assert isinstance(claim(site("numpy.add", (F64_1D, F32_1D)), CONFIG), Rejected)
    assert claim(site("numpy.add", (F64_1D, None)), CONFIG) == NotCovered()


def test_ufunc_lower_revalidates_keywords_receiver_rule_and_literal_metadata() -> None:
    result_type = F64_1D
    with pytest.raises(ValueError, match="empty keywords"):
        lower(
            site(
                "numpy.add",
                (F64_1D, F64_1D),
                result_type=result_type,
                keywords=(KeywordArg(name="where", arg_type="bool"),),
            ),
            ctx("a", "b"),
        )
    receiver = ReceiverMeta(arg_type=F64_1D, expr_kind="name", is_safe=True)
    with pytest.raises(ValueError, match="no ctx.receiver|no ClaimSite.receiver"):
        lower(
            site(
                "numpy.add",
                (F64_1D, F64_1D),
                result_type=result_type,
                receiver=receiver,
            ),
            ctx("a", "b"),
        )
    forged_rule = site("numpy.add", (F64_1D, F64_1D), result_type=result_type)
    forged_rule = ClaimSite(**{**forged_rule.__dict__, "rule_id": "wrong"})
    with pytest.raises(ValueError, match="rule_id"):
        lower(forged_rule, ctx("a", "b"))
    with pytest.raises(ValueError, match="literal compatible"):
        lower(
            site(
                "numpy.multiply",
                (I64_1D, "int"),
                result_type=I64_1D,
                operand_literals=(
                    ClaimLiteral(),
                    ClaimLiteral(is_literal=True, value=True),
                ),
            ),
            ctx("a", "true"),
        )
