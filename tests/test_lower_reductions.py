"""Focused lower coverage for whole-array and literal-axis sum/mean/max/min."""

from __future__ import annotations

import subprocess
import sys

import pytest

from rextio.plugins.api import ClaimLiteral, ClaimSite, KeywordArg, LoweringContext, ReceiverMeta

from rextio_numpy.claim.reductions import _AXIS_RULE, _WHOLE_ARRAY_RULE, _axis_result_type, _whole_array_result_type
from rextio_numpy.diagnostics import F32_1D, F32_2D, F64_1D, F64_2D, I64_1D, I64_2D, array_meta
from rextio_numpy.lower import lower
from rextio_numpy.lower.reductions import try_lower

K = F64_1D


def site(
    target: str,
    operand_types: tuple[str | None, ...] = (K,),
    *,
    keywords: tuple[KeywordArg, ...] = (),
    operand_literals: tuple[ClaimLiteral, ...] = (),
) -> ClaimSite:
    meta = array_meta(operand_types[0]) if operand_types else None
    is_axis = bool(keywords) or len(operand_types) == 2
    if meta is None:
        result_type = "float"
    elif is_axis:
        result_type = _axis_result_type(target, meta[0], meta[1])
    else:
        result_type = _whole_array_result_type(target, meta[0])
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
        keywords=keywords,
        operand_literals=operand_literals,
        rule_id=_AXIS_RULE if is_axis else _WHOLE_ARRAY_RULE,
        result_type=result_type,
    )


def ctx(*operands: str) -> LoweringContext:
    return LoweringContext(
        operands=tuple(operands),
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )


def axis_kw(value: int) -> tuple[KeywordArg, ...]:
    return (
        KeywordArg(
            name="axis",
            arg_type="int",
            literal=ClaimLiteral(is_literal=True, value=value),
        ),
    )


def positional_site(
    target: str,
    operand_type: str,
    axis: object,
    *,
    axis_type: str = "int",
    is_literal: bool = True,
) -> ClaimSite:
    return site(
        target,
        (operand_type, axis_type),
        operand_literals=(
            ClaimLiteral(),
            ClaimLiteral(
                is_literal=is_literal,
                value=axis if is_literal else None,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("target", "helper_name", "body"),
    [
        ("numpy.sum", "__rxtnp_sum1", "Ok(a.sum())"),
        ("numpy.mean", "__rxtnp_mean1", "Ok(a.mean().unwrap_or(f64::NAN))"),
    ],
)
def test_try_lower_reductions_f64_rank1(target: str, helper_name: str, body: str) -> None:
    lowered = try_lower(site(target), ctx("values"))
    assert lowered is not None
    assert lowered.rust == f"{helper_name}(&values)?"
    assert body in lowered.helpers[0]


def test_try_lower_i64_sum_wraps() -> None:
    lowered = try_lower(site("numpy.sum", (I64_1D,)), ctx("a"))
    assert lowered is not None
    assert lowered.rust == "__rxtnp_sum1_i64(&a)?"
    assert "wrapping_add" in lowered.helpers[0]
    assert "PyResult<i64>" in lowered.helpers[0]


def test_try_lower_i64_mean_forged_claim_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside certified dtype/rank matrix"):
        try_lower(site("numpy.mean", (I64_1D,)), ctx("a"))


@pytest.mark.parametrize(
    ("target", "operand_types", "keywords"),
    [
        ("numpy.sum", (F32_2D,), ()),
        ("numpy.mean", (F32_1D,), ()),
        ("numpy.mean", (I64_2D,), axis_kw(0)),
        ("numpy.sum", (F32_1D,), axis_kw(0)),
        ("numpy.max", (F32_1D,), axis_kw(0)),
    ],
)
def test_try_lower_forged_reduction_claim_fails_closed(
    target: str,
    operand_types: tuple[str, ...],
    keywords: tuple[KeywordArg, ...],
) -> None:
    with pytest.raises(ValueError, match="outside certified dtype/rank matrix"):
        try_lower(site(target, operand_types, keywords=keywords), ctx("a"))


def test_try_lower_f64_rank2_sum() -> None:
    lowered = try_lower(site("numpy.sum", (F64_2D,)), ctx("a"))
    assert lowered is not None
    assert lowered.rust == "__rxtnp_sum2_f64(&a)?"
    assert "Ok(a.sum())" in lowered.helpers[0]


def test_try_lower_ignores_non_reduction() -> None:
    assert try_lower(site("numpy.dot"), ctx("a")) is None


def test_router_matches_try_lower() -> None:
    s = site("numpy.sum")
    c = ctx("a")
    assert lower(s, c) == try_lower(s, c)


# ---------------------------------------------------------------- axis form


def test_try_lower_axis_rank1_sum_encodes_normalized_axis() -> None:
    # axis=-1 on rank-1 normalizes to 0 in the helper identity.
    lowered = try_lower(site("numpy.sum", (K,), keywords=axis_kw(-1)), ctx("values"))
    assert lowered is not None
    assert lowered.rust == "__rxtnp_sum1_f64_axis0(&values)?"
    text = "\n".join(lowered.helpers)
    assert "fn __rxtnp_sum1_f64_axis0" in text
    # Axis f64 sum uses NumPy pairwise, not ndarray sequential sum.
    assert "__rxtnp_numpy_pairwise_sum_f64" in text
    assert "PW_BLOCKSIZE" in text


@pytest.mark.parametrize(
    ("target", "operand_type", "axis", "expected"),
    [
        ("numpy.sum", F64_1D, -1, "__rxtnp_sum1_f64_axis0(&values)?"),
        ("numpy.mean", F64_2D, 1, "__rxtnp_mean2_f64_axis1(&values)?"),
        ("numpy.max", I64_1D, 0, "__rxtnp_max1_i64_axis0(&values)?"),
        ("numpy.min", I64_2D, -2, "__rxtnp_min2_i64_axis0(&values)?"),
    ],
)
def test_try_lower_positional_axis_uses_static_literal_not_rendered_value(
    target: str,
    operand_type: str,
    axis: int,
    expected: str,
) -> None:
    lowered = try_lower(
        positional_site(target, operand_type, axis),
        ctx("values", str(axis)),
    )
    assert lowered is not None
    assert lowered.rust == expected


def test_try_lower_method_positional_axis() -> None:
    claimed = ClaimSite(
        kind="call",
        target="numpy.ndarray.sum",
        operand_types=("int",),
        file_path="",
        line=0,
        column=0,
        receiver=ReceiverMeta(arg_type=I64_2D, expr_kind="name", is_safe=True),
        operand_literals=(ClaimLiteral(is_literal=True, value=1),),
        rule_id=_AXIS_RULE,
        result_type=I64_1D,
    )
    lowered = try_lower(
        claimed,
        LoweringContext(
            operands=("1",),
            receiver="values",
            target_language="rust",
            fresh_name=lambda prefix: f"{prefix}_0",
        ),
    )
    assert lowered is not None
    assert lowered.rust == "__rxtnp_sum2_i64_axis1(&values)?"


def test_try_lower_axis_rank2_sum_axis0_and_axis1() -> None:
    lo0 = try_lower(site("numpy.sum", (F64_2D,), keywords=axis_kw(0)), ctx("a"))
    assert lo0 is not None
    assert lo0.rust == "__rxtnp_sum2_f64_axis0(&a)?"
    text0 = "\n".join(lo0.helpers)
    assert "__rxtnp_numpy_pairwise_sum_f64" in text0
    assert "Array1<f64>" in text0
    assert "nrows" in text0
    # Unit-stride → pairwise; non-unit → sequential (NumPy layout match).
    assert "sequential_sum" in text0
    assert "stride_of" in text0

    lo1 = try_lower(site("numpy.sum", (F64_2D,), keywords=axis_kw(-1)), ctx("a"))
    assert lo1 is not None
    assert lo1.rust == "__rxtnp_sum2_f64_axis1(&a)?"
    text1 = "\n".join(lo1.helpers)
    assert "__rxtnp_numpy_pairwise_sum_f64" in text1
    assert "ncols" in text1
    assert "sequential_sum" in text1


def test_try_lower_axis_i64_sum_wraps_per_add() -> None:
    lowered = try_lower(site("numpy.sum", (I64_2D,), keywords=axis_kw(0)), ctx("a"))
    assert lowered is not None
    assert lowered.rust == "__rxtnp_sum2_i64_axis0(&a)?"
    assert "wrapping_add" in lowered.helpers[-1]
    assert "Array1<i64>" in lowered.helpers[-1]


def test_try_lower_axis_mean_empty_lane_nan() -> None:
    lowered = try_lower(site("numpy.mean", (F64_2D,), keywords=axis_kw(0)), ctx("a"))
    assert lowered is not None
    text = "\n".join(lowered.helpers)
    assert "__rxtnp_numpy_pairwise_sum_f64" in text
    assert "f64::NAN" in text
    assert "nrows == 0" in text or "n == 0" in text
    assert "sequential_sum" in text


def test_try_lower_axis_i64_max() -> None:
    lowered = try_lower(site("numpy.max", (I64_1D,), keywords=axis_kw(0)), ctx("a"))
    assert lowered is not None
    assert "PyResult<i64>" in lowered.helpers[-1]
    assert "maximum which has no identity" in lowered.helpers[-1]


def test_router_matches_try_lower_axis() -> None:
    s = site("numpy.min", (I64_2D,), keywords=axis_kw(-2))
    c = ctx("m")
    assert lower(s, c) == try_lower(s, c)
    assert lower(s, c).rust == "__rxtnp_min2_i64_axis0(&m)?"


# ---------------------------------------------------------------- fail-closed


def test_fail_closed_none_operand_type() -> None:
    with pytest.raises(ValueError, match="non-None operand type"):
        try_lower(site("numpy.sum", (None,)), ctx("a"))


def test_fail_closed_non_array_operand_type() -> None:
    with pytest.raises(ValueError, match="array operand type"):
        try_lower(site("numpy.sum", ("float",)), ctx("a"))


def test_fail_closed_multiple_keywords() -> None:
    keywords = (
        KeywordArg(
            name="axis",
            arg_type="int",
            literal=ClaimLiteral(is_literal=True, value=0),
        ),
        KeywordArg(
            name="keepdims",
            arg_type="bool",
            literal=ClaimLiteral(),
        ),
    )
    with pytest.raises(ValueError, match="exactly one keyword"):
        try_lower(site("numpy.sum", (F64_2D,), keywords=keywords), ctx("a"))


def test_fail_closed_sole_wrong_keyword_name() -> None:
    """A sole keepdims=0 must not be treated as axis=0."""
    keywords = (
        KeywordArg(
            name="keepdims",
            arg_type="int",
            literal=ClaimLiteral(is_literal=True, value=0),
        ),
    )
    with pytest.raises(ValueError, match="keyword name 'axis'"):
        try_lower(site("numpy.sum", (F64_2D,), keywords=keywords), ctx("a"))


def test_fail_closed_non_literal_axis_metadata() -> None:
    """Default/non-literal ClaimLiteral must not lower as axis None/0."""
    keywords = (
        KeywordArg(
            name="axis",
            arg_type="int",
            literal=ClaimLiteral(),  # is_literal=False, value=None
        ),
    )
    with pytest.raises(ValueError, match="is_literal=True"):
        try_lower(site("numpy.sum", (F64_2D,), keywords=keywords), ctx("a"))


def test_fail_closed_non_int_axis_literal() -> None:
    keywords = (
        KeywordArg(
            name="axis",
            arg_type="int",
            literal=ClaimLiteral(is_literal=True, value=(0, 1)),
        ),
    )
    with pytest.raises(ValueError, match="int axis literal"):
        try_lower(site("numpy.sum", (F64_2D,), keywords=keywords), ctx("a"))


def test_fail_closed_extra_operand_type() -> None:
    with pytest.raises(ValueError, match="positional axis operand type"):
        try_lower(site("numpy.sum", (F64_1D, F64_1D)), ctx("a", "b"))


def test_fail_closed_wrong_ctx_arity() -> None:
    with pytest.raises(ValueError, match="one rendered ctx operand"):
        try_lower(site("numpy.sum", (F64_1D,)), ctx("a", "b"))
    with pytest.raises(ValueError, match="one rendered ctx operand"):
        try_lower(site("numpy.sum", (F64_2D,), keywords=axis_kw(0)), ctx())


@pytest.mark.parametrize(
    "candidate",
    [
        site("numpy.sum", (F64_1D, "int")),
        positional_site("numpy.sum", F64_1D, 0, is_literal=False),
        positional_site("numpy.sum", F64_1D, True),
        positional_site("numpy.sum", F64_1D, 0, axis_type="float"),
    ],
)
def test_fail_closed_positional_axis_metadata(candidate: ClaimSite) -> None:
    with pytest.raises(ValueError, match="operand_literals|is_literal|int axis|operand type"):
        try_lower(candidate, ctx("a", "axis"))


def test_fail_closed_named_axis_arg_type_mismatch() -> None:
    candidate = site(
        "numpy.sum",
        (F64_1D,),
        keywords=(
            KeywordArg(
                name="axis",
                arg_type="bool",
                literal=ClaimLiteral(is_literal=True, value=0),
            ),
        ),
    )
    with pytest.raises(ValueError, match="arg_type='int'"):
        try_lower(candidate, ctx("a"))


def test_fail_closed_axis_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        try_lower(site("numpy.sum", (F64_1D,), keywords=axis_kw(2)), ctx("a"))


def test_fail_closed_under_python_optimize() -> None:
    """Malformed axis lower must raise ValueError even under python -O.

    Assert-based guards are stripped by optimization mode; this subprocess
    proves the explicit exception path still rejects rather than emitting
    helper names like ``__rxtnp_sum1_f64_axisNone``.
    """
    script = r"""
from rextio.plugins.api import ClaimLiteral, ClaimSite, KeywordArg, LoweringContext
from rextio_numpy.diagnostics import F64_1D
from rextio_numpy.lower.reductions import try_lower

site = ClaimSite(
    kind="call",
    target="numpy.sum",
    operand_types=(F64_1D,),
    file_path="",
    line=0,
    column=0,
    keywords=(
        KeywordArg(
            name="axis",
            arg_type="int",
            literal=ClaimLiteral(is_literal=True, value=99),
        ),
    ),
)
ctx = LoweringContext(
    operands=("a",),
    target_language="rust",
    fresh_name=lambda prefix: f"{prefix}_0",
)
try:
    lowered = try_lower(site, ctx)
except ValueError as exc:
    msg = str(exc)
    if "out of range" in msg and "axisNone" not in msg:
        print("rejected")
    else:
        print(f"wrong-error:{msg!r}")
        raise SystemExit(2) from exc
else:
    print(f"leaked:{lowered!r}")
    raise SystemExit(3)
"""
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    assert completed.stdout.strip() == "rejected"
