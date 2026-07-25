"""Focused claim coverage for whole-array and literal-axis reductions."""

from __future__ import annotations

import pytest

from rextio.config.schema import RextioConfig
from rextio.plugins.api import (
    Claimed,
    ClaimLiteral,
    ClaimSite,
    KeywordArg,
    NotCovered,
    ReceiverMeta,
    Rejected,
)

from rextio_numpy.claim import claim
from rextio_numpy.claim.reductions import normalize_axis, try_claim
from rextio_numpy.diagnostics import F32_1D, F32_2D, F64_1D, F64_2D, I64_1D, I64_2D

K = F64_1D
CONFIG = RextioConfig()

_F64_ARRAY_KEYS = [F64_1D, F64_2D]
_I64_ARRAY_KEYS = [I64_1D, I64_2D]
_F32_ARRAY_KEYS = [F32_1D, F32_2D]


def site(
    target: str,
    operand_types: tuple[str | None, ...],
    *,
    keywords: tuple[KeywordArg, ...] = (),
    operand_literals: tuple[ClaimLiteral, ...] = (),
    receiver: ReceiverMeta | None = None,
) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
        keywords=keywords,
        operand_literals=operand_literals,
        receiver=receiver,
    )


def axis_kw(
    value: int | None | tuple[int, ...], *, is_literal: bool = True
) -> tuple[KeywordArg, ...]:
    return (
        KeywordArg(
            name="axis",
            arg_type="int" if isinstance(value, int) else "None",
            literal=ClaimLiteral(is_literal=is_literal, value=value if is_literal else None),
        ),
    )


def positional_axis(
    target: str,
    key: str,
    value: object,
    *,
    is_literal: bool = True,
) -> ClaimSite:
    return site(
        target,
        (key, "int"),
        operand_literals=(
            ClaimLiteral(),
            ClaimLiteral(
                is_literal=is_literal,
                value=value if is_literal else None,
            ),
        ),
    )


# ---------------------------------------------------------------- whole-array


@pytest.mark.parametrize("key", _F64_ARRAY_KEYS)
@pytest.mark.parametrize("target", ["numpy.sum", "numpy.mean"])
def test_try_claim_f64_reductions(target: str, key: str) -> None:
    result = try_claim(site(target, (key,)))
    assert result == Claimed(rule_id="rextio-numpy/reduction-sum-mean", result_type="float")


@pytest.mark.parametrize("key", _I64_ARRAY_KEYS)
def test_try_claim_i64_sum_claimed_int(key: str) -> None:
    result = try_claim(site("numpy.sum", (key,)))
    assert result == Claimed(rule_id="rextio-numpy/reduction-sum-mean", result_type="int")


@pytest.mark.parametrize("key", _I64_ARRAY_KEYS)
def test_try_claim_i64_mean_rejected(key: str) -> None:
    """int64 mean must not lower: sequential i64→f64 sum diverges from NumPy."""
    result = try_claim(site("numpy.mean", (key,)))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"


@pytest.mark.parametrize("target", ["numpy.sum", "numpy.mean"])
@pytest.mark.parametrize("key", _F32_ARRAY_KEYS)
def test_try_claim_f32_reductions_rejected(target: str, key: str) -> None:
    """float32 sum/mean must not lower: no enforceable size gate for FP contract."""
    result = try_claim(site(target, (key,)))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"


@pytest.mark.parametrize("target", ["numpy.max", "numpy.min"])
@pytest.mark.parametrize("key", _F64_ARRAY_KEYS + _F32_ARRAY_KEYS)
def test_try_claim_float_bare_max_min_not_covered(target: str, key: str) -> None:
    """Floating whole-array extrema stay on the honest fallback path."""
    assert try_claim(site(target, (key,))) == NotCovered()


@pytest.mark.parametrize("target", ["numpy.max", "numpy.min"])
@pytest.mark.parametrize("key", _I64_ARRAY_KEYS)
def test_try_claim_i64_whole_array_extrema(target: str, key: str) -> None:
    assert try_claim(site(target, (key,))) == Claimed(
        rule_id="rextio-numpy/reduction-whole-i64-extrema",
        result_type="int",
    )


def test_try_claim_ignores_non_reduction() -> None:
    assert try_claim(site("numpy.dot", (K, K))) is None


def test_try_claim_wrong_arity_not_covered() -> None:
    assert try_claim(site("numpy.sum", (K, K, K))) == NotCovered()
    assert try_claim(site("numpy.mean", (K, "int", "int"))) == NotCovered()


def test_try_claim_bad_operand_rejected() -> None:
    result = try_claim(site("numpy.sum", ("list[float]",)))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"


def test_try_claim_unresolved_not_covered() -> None:
    assert try_claim(site("numpy.sum", (None,))) == NotCovered()


# ---------------------------------------------------------------- axis form


@pytest.mark.parametrize(
    ("target", "key", "axis", "result_type"),
    [
        ("numpy.sum", F64_1D, 0, "float"),
        ("numpy.sum", F64_1D, -1, "float"),
        ("numpy.mean", F64_1D, 0, "float"),
        ("numpy.sum", I64_1D, 0, "int"),
        ("numpy.max", I64_1D, 0, "int"),
        ("numpy.min", I64_1D, -1, "int"),
        ("numpy.sum", F64_2D, 0, F64_1D),
        ("numpy.sum", F64_2D, 1, F64_1D),
        ("numpy.sum", F64_2D, -1, F64_1D),
        ("numpy.sum", F64_2D, -2, F64_1D),
        ("numpy.mean", F64_2D, 0, F64_1D),
        ("numpy.mean", F64_2D, -1, F64_1D),
        ("numpy.sum", I64_2D, 0, I64_1D),
        ("numpy.max", I64_2D, 1, I64_1D),
        ("numpy.min", I64_2D, -1, I64_1D),
    ],
)
def test_try_claim_axis_literal_admitted(
    target: str, key: str, axis: int, result_type: str
) -> None:
    result = try_claim(site(target, (key,), keywords=axis_kw(axis)))
    assert result == Claimed(rule_id="rextio-numpy/reduction-axis", result_type=result_type)


@pytest.mark.parametrize(
    ("target", "key", "axis", "result_type"),
    [
        ("numpy.sum", F64_1D, 0, "float"),
        ("numpy.mean", F64_2D, -1, F64_1D),
        ("numpy.sum", I64_2D, 0, I64_1D),
        ("numpy.max", I64_1D, -1, "int"),
        ("numpy.min", I64_2D, 1, I64_1D),
    ],
)
def test_try_claim_positional_axis_literal_admitted(
    target: str,
    key: str,
    axis: int,
    result_type: str,
) -> None:
    assert try_claim(positional_axis(target, key, axis)) == Claimed(
        rule_id="rextio-numpy/reduction-axis",
        result_type=result_type,
    )


def test_try_claim_method_positional_axis_literal_admitted() -> None:
    receiver = ReceiverMeta(arg_type=I64_2D, expr_kind="name", is_safe=True)
    candidate = site(
        "numpy.ndarray.max",
        ("int",),
        receiver=receiver,
        operand_literals=(ClaimLiteral(is_literal=True, value=1),),
    )
    assert try_claim(candidate) == Claimed(
        rule_id="rextio-numpy/reduction-axis",
        result_type=I64_1D,
    )


@pytest.mark.parametrize(
    ("target", "key", "axis"),
    [
        ("numpy.sum", F64_1D, 1),
        ("numpy.sum", F64_1D, -2),
        ("numpy.mean", F64_2D, 2),
        ("numpy.max", F64_2D, -3),
        ("numpy.min", I64_2D, 99),
    ],
)
def test_try_claim_axis_out_of_range_not_covered(target: str, key: str, axis: int) -> None:
    assert try_claim(site(target, (key,), keywords=axis_kw(axis))) == NotCovered()


@pytest.mark.parametrize(
    "keywords",
    [
        axis_kw(None),  # axis=None
        axis_kw((0, 1)),  # tuple axis
        axis_kw(0, is_literal=False),  # dynamic axis (if ever offered)
        (
            KeywordArg(
                name="axis",
                arg_type="int",
                literal=ClaimLiteral(is_literal=True, value=0),
            ),
            KeywordArg(
                name="keepdims",
                arg_type="bool",
                literal=ClaimLiteral(is_literal=True, value=None),
            ),
        ),
        (
            KeywordArg(
                name="dtype",
                arg_type="None",
                literal=ClaimLiteral(is_literal=True, value=None),
            ),
        ),
        (
            KeywordArg(
                name="out",
                arg_type="None",
                literal=ClaimLiteral(is_literal=True, value=None),
            ),
        ),
    ],
)
def test_try_claim_axis_unsupported_keyword_forms_not_covered(
    keywords: tuple[KeywordArg, ...],
) -> None:
    assert try_claim(site("numpy.sum", (K,), keywords=keywords)) == NotCovered()
    assert try_claim(site("numpy.max", (F64_2D,), keywords=keywords)) == NotCovered()


@pytest.mark.parametrize(
    "candidate",
    [
        site("numpy.sum", (F64_1D, "int")),
        positional_axis("numpy.sum", F64_1D, 0, is_literal=False),
        positional_axis("numpy.sum", F64_1D, None),
        positional_axis("numpy.sum", F64_2D, (0, 1)),
        site(
            "numpy.sum",
            (F64_1D, "float"),
            operand_literals=(
                ClaimLiteral(),
                ClaimLiteral(is_literal=True, value=0),
            ),
        ),
    ],
)
def test_try_claim_positional_axis_requires_aligned_int_literal(
    candidate: ClaimSite,
) -> None:
    assert try_claim(candidate) == NotCovered()


def test_try_claim_named_axis_requires_int_arg_type() -> None:
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
    assert try_claim(candidate) == NotCovered()


@pytest.mark.parametrize("target", ["numpy.sum", "numpy.mean"])
@pytest.mark.parametrize("key", _F32_ARRAY_KEYS)
def test_try_claim_axis_f32_sum_mean_rejected(target: str, key: str) -> None:
    result = try_claim(site(target, (key,), keywords=axis_kw(0)))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"


@pytest.mark.parametrize("key", _I64_ARRAY_KEYS)
def test_try_claim_axis_i64_mean_rejected(key: str) -> None:
    result = try_claim(site("numpy.mean", (key,), keywords=axis_kw(0)))
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-NUMPY-010"


@pytest.mark.parametrize("target", ["numpy.max", "numpy.min"])
@pytest.mark.parametrize("key", _F64_ARRAY_KEYS + _F32_ARRAY_KEYS)
def test_try_claim_axis_float_max_min_not_covered(target: str, key: str) -> None:
    assert try_claim(site(target, (key,), keywords=axis_kw(0))) == NotCovered()


def test_try_claim_amax_amin_not_this_lane() -> None:
    assert try_claim(site("numpy.amax", (K,), keywords=axis_kw(0))) is None
    assert try_claim(site("numpy.amin", (K,), keywords=axis_kw(0))) is None


@pytest.mark.parametrize(
    ("axis", "rank", "expected"),
    [
        (0, 1, 0),
        (-1, 1, 0),
        (0, 2, 0),
        (1, 2, 1),
        (-1, 2, 1),
        (-2, 2, 0),
        (2, 2, None),
        (-3, 2, None),
        (1, 1, None),
    ],
)
def test_normalize_axis(axis: int, rank: int, expected: int | None) -> None:
    assert normalize_axis(axis, rank) == expected


def test_router_matches_try_claim() -> None:
    s = site("numpy.mean", (K,))
    assert claim(s, CONFIG) == try_claim(s)
    s2 = site("numpy.sum", (I64_2D,))
    assert claim(s2, CONFIG) == try_claim(s2)
    s3 = site("numpy.mean", (I64_1D,))
    assert claim(s3, CONFIG) == try_claim(s3)
    s4 = site("numpy.max", (F64_2D,), keywords=axis_kw(-1))
    assert claim(s4, CONFIG) == try_claim(s4)
