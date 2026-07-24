"""Contract tests for API-1.5 comparisons and three-argument ``numpy.where``."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rextio.config.schema import RextioConfig
from rextio.plugins.api import (
    Claimed,
    ClaimLiteral,
    ClaimSite,
    KeywordArg,
    LoweringContext,
    NotCovered,
    Rejected,
)

from rextio_numpy.diagnostics import (
    F32_1D,
    F32_2D,
    F64_1D,
    F64_2D,
    I64_1D,
    I64_2D,
)
from rextio_numpy.plugin import RextioNumpyPlugin
from rextio_numpy.plugin_types import plugin_type, plugin_type_keys

BOOL_1D = "rextio-numpy/bool-1d"
BOOL_2D = "rextio-numpy/bool-2d"
COMPARE_RULE = "rextio-numpy/elementwise-compare"
WHERE_RULE = "rextio-numpy/where-three-argument"
LOGICAL_NOT_RULE = "rextio-numpy/resident-logical-not"
LOGICAL_BINARY_RULE = "rextio-numpy/resident-logical-binary"
LOGICAL_REDUCTION_RULE = "rextio-numpy/resident-logical-reduction"

PLUGIN = RextioNumpyPlugin()
CONFIG = RextioConfig()


def _site(
    kind: str,
    target: str,
    operand_types: tuple[str | None, ...],
    *,
    operand_literals: tuple[ClaimLiteral, ...] = (),
    keywords: tuple[KeywordArg, ...] = (),
) -> ClaimSite:
    return ClaimSite(
        kind=kind,
        target=target,
        operand_types=operand_types,
        operand_literals=operand_literals,
        keywords=keywords,
        file_path="",
        line=0,
        column=0,
    )


def _claimed(site: ClaimSite) -> ClaimSite:
    result = PLUGIN.claim(site, CONFIG)
    assert isinstance(result, Claimed)
    return replace(site, rule_id=result.rule_id, result_type=result.result_type)


def _ctx(*operands: str) -> LoweringContext:
    return LoweringContext(
        operands=operands,
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )


def test_bool_results_are_plugin_owned_resident_types_without_boundaries() -> None:
    assert {BOOL_1D, BOOL_2D}.issubset(plugin_type_keys())
    for key, rank in ((BOOL_1D, 1), (BOOL_2D, 2)):
        resident = plugin_type(key)
        assert resident.annotations == ()
        assert resident.rust_type == f"numpy::ndarray::Array{rank}<bool>"
        assert resident.conversion is None
        assert resident.is_resident is True


@pytest.mark.parametrize("operator", ("==", "!=", "<", "<=", ">", ">="))
@pytest.mark.parametrize(
    ("operands", "result_type"),
    (
        ((F64_1D, F64_1D), BOOL_1D),
        ((F32_2D, F32_1D), BOOL_2D),
        ((I64_1D, I64_2D), BOOL_2D),
        ((F64_1D, "float"), BOOL_1D),
        (("float", F32_2D), BOOL_2D),
        ((I64_1D, "int"), BOOL_1D),
        (("int", I64_2D), BOOL_2D),
    ),
)
def test_claims_bounded_non_chained_comparison_matrix(
    operator: str,
    operands: tuple[str, str],
    result_type: str,
) -> None:
    assert PLUGIN.claim(_site("compare", operator, operands), CONFIG) == Claimed(
        rule_id=COMPARE_RULE,
        result_type=result_type,
    )


@pytest.mark.parametrize(
    ("target", "operands", "expected_type"),
    (
        ("<", (F64_1D, F32_1D), Rejected),
        ("==", (F64_1D, "int"), Rejected),
        (">", (F64_1D, F64_1D, F64_1D), Rejected),
        ("<", (BOOL_1D, BOOL_1D), Rejected),
        ("is", (F64_1D, F64_1D), NotCovered),
        ("in", (F64_1D, F64_1D), NotCovered),
    ),
)
def test_comparison_near_misses_fail_closed(
    target: str,
    operands: tuple[str, ...],
    expected_type: type[Rejected] | type[NotCovered],
) -> None:
    result = PLUGIN.claim(_site("compare", target, operands), CONFIG)
    assert isinstance(result, expected_type)


def test_comparison_rejects_hidden_metadata() -> None:
    result = PLUGIN.claim(
        _site(
            "compare",
            ">",
            (F64_1D, "float"),
            keywords=(
                KeywordArg(
                    name="out",
                    arg_type=F64_1D,
                    literal=ClaimLiteral(is_literal=False),
                ),
            ),
        ),
        CONFIG,
    )
    assert isinstance(result, Rejected)


@pytest.mark.parametrize(
    ("target", "operands", "helper_name"),
    (
        (">", (F64_1D, "float"), "__rxtnp_cmp_gt1_as_f64"),
        ("<=", ("int", I64_2D), "__rxtnp_cmp_le2_sa_i64"),
        ("!=", (F32_2D, F32_1D), "__rxtnp_cmp_ne21_aa_f32"),
    ),
)
def test_lowers_comparison_with_resident_bool_result(
    target: str,
    operands: tuple[str, str],
    helper_name: str,
) -> None:
    claimed = _claimed(_site("compare", target, operands))
    lowered = PLUGIN.lower(claimed, _ctx("left", "right"))
    assert lowered.rust == f"{helper_name}(&left, right)?" if "_as_" in helper_name else (
        f"{helper_name}(left, &right)?" if "_sa_" in helper_name else
        f"{helper_name}(&left, &right)?"
    )
    helper = "\n".join(lowered.helpers)
    assert f"fn {helper_name}" in helper
    assert "Array" in helper and "<bool>" in helper


def test_comparison_lower_revalidates_result_type_and_literals() -> None:
    claimed = _claimed(_site("compare", ">", (I64_1D, "int")))
    with pytest.raises(ValueError, match="result_type"):
        PLUGIN.lower(replace(claimed, result_type=I64_1D), _ctx("values", "limit"))
    with pytest.raises(ValueError, match="i64 literal"):
        PLUGIN.lower(
            replace(
                claimed,
                operand_literals=(
                    ClaimLiteral(is_literal=False),
                    ClaimLiteral(is_literal=True, value=2**63),
                ),
            ),
            _ctx("values", "limit"),
        )


@pytest.mark.parametrize(
    ("target", "operands", "rule_id", "result_type"),
    (
        ("numpy.logical_not", (BOOL_1D,), LOGICAL_NOT_RULE, BOOL_1D),
        ("numpy.logical_not", (BOOL_2D,), LOGICAL_NOT_RULE, BOOL_2D),
        ("numpy.logical_and", (BOOL_1D, BOOL_2D), LOGICAL_BINARY_RULE, BOOL_2D),
        ("numpy.logical_or", (BOOL_2D, BOOL_1D), LOGICAL_BINARY_RULE, BOOL_2D),
        ("numpy.all", (BOOL_1D,), LOGICAL_REDUCTION_RULE, "bool"),
        ("numpy.any", (BOOL_2D,), LOGICAL_REDUCTION_RULE, "bool"),
    ),
)
def test_claims_exact_resident_logical_surface(
    target: str,
    operands: tuple[str, ...],
    rule_id: str,
    result_type: str,
) -> None:
    assert PLUGIN.claim(_site("call", target, operands), CONFIG) == Claimed(
        rule_id=rule_id,
        result_type=result_type,
    )


@pytest.mark.parametrize(
    ("target", "operands", "expected_type"),
    (
        ("numpy.logical_not", (F64_1D,), Rejected),
        ("numpy.logical_and", (BOOL_1D, F64_1D), Rejected),
        ("numpy.logical_or", (BOOL_1D,), Rejected),
        ("numpy.all", (F64_1D,), Rejected),
        ("numpy.any", (BOOL_1D, BOOL_1D), Rejected),
        ("numpy.logical_not", (None,), NotCovered),
    ),
)
def test_resident_logical_near_misses_fail_closed(
    target: str,
    operands: tuple[str | None, ...],
    expected_type: type[Rejected] | type[NotCovered],
) -> None:
    assert isinstance(PLUGIN.claim(_site("call", target, operands), CONFIG), expected_type)


def test_resident_logical_options_and_lower_metadata_fail_closed() -> None:
    option = PLUGIN.claim(
        _site(
            "call",
            "numpy.all",
            (BOOL_1D,),
            keywords=(
                KeywordArg(
                    name="axis",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=0),
                ),
            ),
        ),
        CONFIG,
    )
    assert isinstance(option, Rejected)

    claimed = _claimed(_site("call", "numpy.logical_and", (BOOL_1D, BOOL_2D)))
    with pytest.raises(ValueError, match="result_type"):
        PLUGIN.lower(replace(claimed, result_type=BOOL_1D), _ctx("left", "right"))
    with pytest.raises(ValueError, match="operand_literals"):
        PLUGIN.lower(
            replace(
                claimed,
                operand_literals=(
                    ClaimLiteral(is_literal=True, value=True),
                    ClaimLiteral(is_literal=False),
                ),
            ),
            _ctx("left", "right"),
        )


@pytest.mark.parametrize(
    ("target", "operands", "helper_name", "rust_call"),
    (
        (
            "numpy.logical_not",
            (BOOL_1D,),
            "__rxtnp_logical_not_1",
            "__rxtnp_logical_not_1(&mask)?",
        ),
        (
            "numpy.logical_and",
            (BOOL_1D, BOOL_2D),
            "__rxtnp_logical_and12",
            "__rxtnp_logical_and12(&left, &right)?",
        ),
        (
            "numpy.logical_or",
            (BOOL_2D, BOOL_1D),
            "__rxtnp_logical_or21",
            "__rxtnp_logical_or21(&left, &right)?",
        ),
        (
            "numpy.all",
            (BOOL_2D,),
            "__rxtnp_logical_all_2",
            "__rxtnp_logical_all_2(&mask)?",
        ),
        (
            "numpy.any",
            (BOOL_1D,),
            "__rxtnp_logical_any_1",
            "__rxtnp_logical_any_1(&mask)?",
        ),
    ),
)
def test_lowers_exact_resident_logical_surface(
    target: str,
    operands: tuple[str, ...],
    helper_name: str,
    rust_call: str,
) -> None:
    rendered = ("mask",) if len(operands) == 1 else ("left", "right")
    lowered = PLUGIN.lower(_claimed(_site("call", target, operands)), _ctx(*rendered))
    assert lowered.rust == rust_call
    helpers = "\n".join(lowered.helpers)
    assert f"fn {helper_name}" in helpers
    if target in {"numpy.logical_and", "numpy.logical_or"}:
        assert "__rxtnp_broadcast_shape" in helpers
        assert "numpy::ndarray::Zip::from" in helpers
    if target in {"numpy.all", "numpy.any"}:
        assert ".iter()." in helpers


@pytest.mark.parametrize(
    ("operands", "result_type"),
    (
        ((BOOL_1D, F64_1D, F64_1D), F64_1D),
        ((BOOL_2D, F32_2D, F32_1D), F32_2D),
        ((BOOL_1D, I64_1D, "int"), I64_1D),
        ((BOOL_2D, "float", F64_1D), F64_2D),
    ),
)
def test_claims_three_argument_where_matrix(
    operands: tuple[str, str, str],
    result_type: str,
) -> None:
    assert PLUGIN.claim(_site("call", "numpy.where", operands), CONFIG) == Claimed(
        rule_id=WHERE_RULE,
        result_type=result_type,
    )


@pytest.mark.parametrize(
    "operands",
    (
        (F64_1D, F64_1D, F64_1D),
        (BOOL_1D, F64_1D, F32_1D),
        (BOOL_1D, F64_1D, "int"),
        (BOOL_1D, "float", "float"),
        (BOOL_1D, F64_1D),
        (BOOL_1D, F64_1D, F64_1D, F64_1D),
        (None, F64_1D, F64_1D),
    ),
)
def test_where_near_misses_fail_closed(operands: tuple[str | None, ...]) -> None:
    result = PLUGIN.claim(_site("call", "numpy.where", operands), CONFIG)
    if any(operand is None for operand in operands):
        assert isinstance(result, NotCovered)
    else:
        assert isinstance(result, Rejected)


def test_where_rejects_keywords_and_other_numpy_apis() -> None:
    keyword = PLUGIN.claim(
        _site(
            "call",
            "numpy.where",
            (BOOL_1D, F64_1D, F64_1D),
            keywords=(
                KeywordArg(
                    name="x",
                    arg_type=F64_1D,
                    literal=ClaimLiteral(is_literal=False),
                ),
            ),
        ),
        CONFIG,
    )
    assert isinstance(keyword, Rejected)
    assert isinstance(
        PLUGIN.claim(
            _site("call", "numpy.select", (BOOL_1D, F64_1D, F64_1D)),
            CONFIG,
        ),
        NotCovered,
    )


@pytest.mark.parametrize(
    ("operands", "helper_name", "rust_call"),
    (
        (
            (BOOL_1D, F64_1D, F64_1D),
            "__rxtnp_where111_aa_f64",
            "__rxtnp_where111_aa_f64(&mask, &yes, &no)?",
        ),
        (
            (BOOL_2D, F32_2D, "float"),
            "__rxtnp_where22_as_f32",
            "__rxtnp_where22_as_f32(&mask, &yes, no)?",
        ),
        (
            (BOOL_2D, "int", I64_1D),
            "__rxtnp_where21_sa_i64",
            "__rxtnp_where21_sa_i64(&mask, yes, &no)?",
        ),
    ),
)
def test_lowers_where_with_three_way_broadcast_contract(
    operands: tuple[str, str, str],
    helper_name: str,
    rust_call: str,
) -> None:
    lowered = PLUGIN.lower(
        _claimed(_site("call", "numpy.where", operands)),
        _ctx("mask", "yes", "no"),
    )
    assert lowered.rust == rust_call
    helpers = "\n".join(lowered.helpers)
    assert f"fn {helper_name}" in helpers
    assert "__rxtnp_broadcast_shape3" in helpers
    assert "numpy::ndarray::Zip::from" in helpers
    assert "if *mask" in helpers
    assert "operands could not be broadcast together with shapes {} {} {} " in helpers


def test_where_lower_revalidates_condition_and_result_metadata() -> None:
    claimed = _claimed(_site("call", "numpy.where", (BOOL_1D, F64_1D, F64_1D)))
    with pytest.raises(ValueError, match="condition"):
        PLUGIN.lower(
            replace(claimed, operand_types=(F64_1D, F64_1D, F64_1D)),
            _ctx("mask", "yes", "no"),
        )
    with pytest.raises(ValueError, match="result_type"):
        PLUGIN.lower(
            replace(claimed, result_type=F64_2D),
            _ctx("mask", "yes", "no"),
        )
