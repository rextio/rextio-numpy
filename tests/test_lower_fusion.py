"""Lower-time fusion coverage: helpers, allocation shape, fail-closed metadata."""

from __future__ import annotations

import pytest

from rextio.plugins.api import ClaimExpr, ClaimSite, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.claim.fusion import FUSION_RULE, try_match
from rextio_numpy.diagnostics import F64_1D, F64_2D, I64_1D
from rextio_numpy.lower import lower
from rextio_numpy.lower.fusion import try_lower


def leaf(index: int, type_key: str) -> ClaimExpr:
    return ClaimExpr(
        kind="leaf",
        result_type=type_key,
        leaf_index=index,
        leaf_kind="name",
    )


def binop(op: str, left: ClaimExpr, right: ClaimExpr, result_type: str | None = None) -> ClaimExpr:
    return ClaimExpr(
        kind="binop",
        target=op,
        result_type=result_type or left.result_type,
        children=(left, right),
    )


def fusion_site(expr: ClaimExpr) -> ClaimSite:
    match = try_match(expr)
    assert match is not None
    return ClaimSite(
        kind="binop",
        target=expr.target,
        operand_types=match.root_child_types,
        file_path="",
        line=0,
        column=0,
        rule_id=FUSION_RULE,
        result_type=match.result_type,
        expression=expr,
    )


def leaves_ctx(*names: str) -> LoweringContext:
    return LoweringContext(
        operands=(),
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
        leaf_operands=tuple(names),
    )


def multi_op_expr() -> ClaimExpr:
    # (a + b) * (a - b)
    return binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        binop("-", leaf(2, F64_1D), leaf(3, F64_1D)),
    )


def test_try_lower_emits_one_fused_helper_call() -> None:
    expr = multi_op_expr()
    match = try_match(expr)
    assert match is not None
    lowered = try_lower(fusion_site(expr), leaves_ctx("a", "b", "a", "b"))
    assert lowered is not None
    name = rust_snippets.fusion_call_name(match.signature)
    assert lowered.rust == f"{name}(&a, &b, &a, &b)?"
    text = "\n".join(lowered.helpers)
    assert f"fn {name}" in text
    # Fast path + generic path each allocate once via from_shape_fn (runtime
    # takes exactly one branch; no Zip / map_collect).
    assert text.count("from_shape_fn") == 2
    assert "is_standard_layout()" in text
    assert "as_slice()" in text
    assert "Zip::" not in text
    assert "map_collect" not in text
    # No ordinary per-op elementwise helper names.
    for ordinary in (
        "__rxtnp_add1_aa",
        "__rxtnp_sub1_aa",
        "__rxtnp_mul1_aa",
        "__rxtnp_add11_aa_f64",
    ):
        assert ordinary not in text
    # Shared broadcast helpers present once each via bundle.
    assert "__rxtnp_broadcast_shape" in text
    assert "__rxtnp_fmt_shape" in text


def test_helper_has_zero_intermediate_ndarray_allocations() -> None:
    expr = multi_op_expr()
    lowered = try_lower(fusion_site(expr), leaves_ctx("a", "b", "a", "b"))
    assert lowered is not None
    helper = lowered.helpers[-1]
    # Owned intermediate arrays would look like these patterns.
    assert "to_owned()" not in helper
    assert "Array::zeros" not in helper
    assert helper.count("from_shape_fn") == 2
    assert "Zip::" not in helper
    # Scalar temps for internal nodes (2 of 3 binops bind temps; root returns).
    # Both paths emit the same AST-order temps.
    assert helper.count("let t0") >= 1
    assert helper.count("let t1") >= 1


def test_nine_leaf_helper_uses_from_shape_fn_not_zip() -> None:
    """8-binop / 9-leaf trees must not depend on Zip arity (max 6 producers)."""
    expr: ClaimExpr = leaf(0, F64_1D)
    for i in range(8):
        expr = binop("+", expr, leaf(i + 1, F64_1D), F64_1D)
    match = try_match(expr)
    assert match is not None
    assert match.binop_count == 8
    assert match.leaf_count == 9
    names = tuple(f"a{i}" for i in range(9))
    lowered = try_lower(fusion_site(expr), leaves_ctx(*names))
    assert lowered is not None
    helper = lowered.helpers[-1]
    assert "from_shape_fn" in helper
    assert "Zip::" not in helper
    assert helper.count("let v") == 9 or helper.count("broadcast(dim)") == 9


def test_i64_wrapping_at_every_intermediate() -> None:
    expr = binop(
        "*",
        binop("+", leaf(0, I64_1D), leaf(1, I64_1D), I64_1D),
        binop("-", leaf(2, I64_1D), leaf(3, I64_1D), I64_1D),
        I64_1D,
    )
    lowered = try_lower(fusion_site(expr), leaves_ctx("a", "b", "c", "d"))
    assert lowered is not None
    helper = lowered.helpers[-1]
    assert "wrapping_add" in helper
    assert "wrapping_sub" in helper
    assert "wrapping_mul" in helper
    assert " as f64" not in helper


def test_mixed_rank_result() -> None:
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_2D), F64_2D),
        leaf(2, F64_1D),
        F64_2D,
    )
    match = try_match(expr)
    assert match is not None
    assert match.result_rank == 2
    lowered = try_lower(fusion_site(expr), leaves_ctx("a", "b", "c"))
    assert lowered is not None
    assert "Array2<f64>" in lowered.helpers[-1]
    assert "Array2::from_shape_fn" in lowered.helpers[-1]


def test_broadcast_validation_before_allocation() -> None:
    expr = multi_op_expr()
    helper = try_lower(fusion_site(expr), leaves_ctx("a", "b", "a", "b")).helpers[-1]  # type: ignore[union-attr]
    # Shape lets appear before any from_shape_fn allocation (fast or generic).
    assert helper.index("__rxtnp_broadcast_shape") < helper.index("from_shape_fn")
    # Generic broadcast path remains after the equal-shape layout gate.
    assert helper.index("is_standard_layout()") < helper.index(".broadcast(dim)")
    assert helper.index(".broadcast(dim)") < helper.rindex("from_shape_fn")


def test_contiguous_equal_shape_fast_path_emitted_for_rank1_and_rank2() -> None:
    """Both ranks emit a standard-layout equal-shape gate without unsafe."""
    r1 = try_lower(fusion_site(multi_op_expr()), leaves_ctx("a", "b", "a", "b"))
    assert r1 is not None
    h1 = r1.helpers[-1]
    assert "is_standard_layout()" in h1
    assert "as_slice()" in h1
    assert "unsafe" not in h1

    expr2 = binop(
        "*",
        binop("+", leaf(0, F64_2D), leaf(1, F64_2D), F64_2D),
        binop("-", leaf(2, F64_2D), leaf(3, F64_2D), F64_2D),
        F64_2D,
    )
    r2 = try_lower(fusion_site(expr2), leaves_ctx("a", "b", "a", "b"))
    assert r2 is not None
    h2 = r2.helpers[-1]
    assert "is_standard_layout()" in h2
    assert "ncols" in h2
    assert "sl0[i * ncols + j]" in h2
    # Slice bindings must not shadow LTR shape locals (s0, s1, ...).
    assert "Some(sl0)" in h2
    assert "Some(s0)" not in h2.split("is_standard_layout()")[-1]
    assert "unsafe" not in h2
    # Generic path still present for strided/broadcast fallthrough.
    assert ".broadcast(dim)" in h2


def test_fail_closed_missing_expression() -> None:
    site = ClaimSite(
        kind="binop",
        target="*",
        operand_types=(F64_1D, F64_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=FUSION_RULE,
        result_type=F64_1D,
        expression=None,
    )
    with pytest.raises(ValueError, match="requires ClaimSite.expression"):
        try_lower(site, leaves_ctx("a", "b", "a", "b"))


def test_fail_closed_wrong_leaf_count() -> None:
    expr = multi_op_expr()
    with pytest.raises(ValueError, match="leaf_operands length"):
        try_lower(fusion_site(expr), leaves_ctx("a", "b"))


def test_fail_closed_nonempty_operands() -> None:
    expr = multi_op_expr()
    ctx = LoweringContext(
        operands=("x", "y"),
        target_language="rust",
        fresh_name=lambda p: f"{p}_0",
        leaf_operands=("a", "b", "a", "b"),
    )
    with pytest.raises(ValueError, match="empty ctx.operands"):
        try_lower(fusion_site(expr), ctx)


def test_fail_closed_opaque_leaf_at_lower() -> None:
    expr = binop(
        "*",
        binop(
            "+",
            ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=0, leaf_kind="opaque"),
            leaf(1, F64_1D),
        ),
        leaf(2, F64_1D),
    )
    # Force rule_id even though match would fail.
    site = ClaimSite(
        kind="binop",
        target="*",
        operand_types=(F64_1D, F64_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=FUSION_RULE,
        result_type=F64_1D,
        expression=expr,
    )
    with pytest.raises(ValueError, match="failed fusion match|non-name"):
        try_lower(site, leaves_ctx("a", "b", "c"))


def test_fail_closed_gapped_leaf_indexes() -> None:
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        binop("-", leaf(2, F64_1D), leaf(4, F64_1D)),  # gap
    )
    site = ClaimSite(
        kind="binop",
        target="*",
        operand_types=(F64_1D, F64_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=FUSION_RULE,
        result_type=F64_1D,
        expression=expr,
    )
    with pytest.raises(ValueError, match="failed fusion match|leaf_index"):
        try_lower(site, leaves_ctx("a", "b", "c", "d"))


def test_fail_closed_wrong_internal_result_type() -> None:
    # Intermediate claims F64_2D but children are both rank-1 → implied F64_1D.
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D), F64_2D),
        leaf(2, F64_1D),
        F64_1D,
    )
    site = ClaimSite(
        kind="binop",
        target="*",
        operand_types=(F64_2D, F64_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=FUSION_RULE,
        result_type=F64_1D,
        expression=expr,
    )
    with pytest.raises(ValueError, match="failed fusion match"):
        try_lower(site, leaves_ctx("a", "b", "c"))


def test_fail_closed_site_target_mismatch() -> None:
    expr = multi_op_expr()
    match = try_match(expr)
    assert match is not None
    site = ClaimSite(
        kind="binop",
        target="+",  # wrong; expression root is *
        operand_types=match.root_child_types,
        file_path="",
        line=0,
        column=0,
        rule_id=FUSION_RULE,
        result_type=match.result_type,
        expression=expr,
    )
    with pytest.raises(ValueError, match="site.target"):
        try_lower(site, leaves_ctx("a", "b", "a", "b"))


def test_fail_closed_operand_types_mismatch() -> None:
    expr = multi_op_expr()
    match = try_match(expr)
    assert match is not None
    site = ClaimSite(
        kind="binop",
        target=expr.target,
        operand_types=(F64_2D, F64_2D),  # wrong for rank-1 children
        file_path="",
        line=0,
        column=0,
        rule_id=FUSION_RULE,
        result_type=match.result_type,
        expression=expr,
    )
    with pytest.raises(ValueError, match="inconsistent"):
        try_lower(site, leaves_ctx("a", "b", "a", "b"))


def test_fail_closed_result_type_none_at_lower() -> None:
    expr = multi_op_expr()
    match = try_match(expr)
    assert match is not None
    site = ClaimSite(
        kind="binop",
        target=expr.target,
        operand_types=match.root_child_types,
        file_path="",
        line=0,
        column=0,
        rule_id=FUSION_RULE,
        result_type=None,
        expression=expr,
    )
    with pytest.raises(ValueError, match="non-None ClaimSite.result_type"):
        try_lower(site, leaves_ctx("a", "b", "a", "b"))


def test_fail_closed_operand_types_none_at_lower() -> None:
    expr = multi_op_expr()
    match = try_match(expr)
    assert match is not None
    site = ClaimSite(
        kind="binop",
        target=expr.target,
        operand_types=(None, None),
        file_path="",
        line=0,
        column=0,
        rule_id=FUSION_RULE,
        result_type=match.result_type,
        expression=expr,
    )
    with pytest.raises(ValueError, match="inconsistent"):
        try_lower(site, leaves_ctx("a", "b", "a", "b"))


def test_router_matches_try_lower() -> None:
    expr = multi_op_expr()
    site = fusion_site(expr)
    ctx = leaves_ctx("a", "b", "a", "b")
    assert lower(site, ctx) == try_lower(site, ctx)


def test_try_lower_ignores_non_fusion_rule() -> None:
    expr = multi_op_expr()
    site = ClaimSite(
        kind="binop",
        target="*",
        operand_types=(F64_1D, F64_1D),
        file_path="",
        line=0,
        column=0,
        rule_id="rextio-numpy/elementwise-float64",
        result_type=F64_1D,
        expression=expr,
    )
    assert try_lower(site, leaves_ctx("a", "b", "a", "b")) is None
