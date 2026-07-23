"""Lowering for multi-op elementwise chain fusion (Wave 2)."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.claim.fusion import FUSION_RULE, site_consistent_with_match, try_match
from rextio_numpy.lower.contracts import (
    require_no_hidden_site_metadata,
    require_rust_pyo3_context,
)


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Return a fused lowering, or None if this site is not a fusion claim."""
    if claimed.rule_id != FUSION_RULE:
        return None

    require_rust_pyo3_context(ctx, "fusion")

    # Fail closed on malformed lower-time metadata rather than emitting
    # incorrect fused code.
    if claimed.kind != "binop":
        raise ValueError(f"rextio-numpy fusion lower requires kind='binop', got {claimed.kind!r}")
    if claimed.expression is None:
        raise ValueError("rextio-numpy fusion lower requires ClaimSite.expression")
    if claimed.target != claimed.expression.target:
        raise ValueError(
            "rextio-numpy fusion lower: site.target "
            f"{claimed.target!r} != expression.target {claimed.expression.target!r}"
        )
    if claimed.keywords:
        raise ValueError(
            "rextio-numpy fusion lower requires empty keywords on binop sites; "
            f"got {claimed.keywords!r}"
        )
    require_no_hidden_site_metadata(
        claimed,
        "fusion",
        allow_expression=True,
        expected_operand_literals=2,
    )
    if claimed.receiver is not None:
        raise ValueError("rextio-numpy fusion lower requires no ClaimSite.receiver")
    if ctx.receiver is not None:
        raise ValueError("rextio-numpy fusion lower requires no ctx.receiver")
    if claimed.result_type is None:
        raise ValueError(
            "rextio-numpy fusion lower requires non-None ClaimSite.result_type "
            "matching the fused root result"
        )
    if ctx.operands:
        raise ValueError(
            "rextio-numpy fusion lower requires empty ctx.operands "
            f"(operand_mode=leaves); got {ctx.operands!r}"
        )

    match = try_match(claimed.expression)
    if match is None:
        raise ValueError(
            "rextio-numpy fusion lower: expression failed fusion match "
            "(out of scope or malformed ClaimExpr tree)"
        )
    if not site_consistent_with_match(claimed, match, require_result_type=True):
        raise ValueError(
            "rextio-numpy fusion lower: ClaimSite metadata inconsistent with "
            "matched fusion tree (result_type / operand_types / root operator)"
        )

    if len(ctx.leaf_operands) != match.leaf_count:
        raise ValueError(
            "rextio-numpy fusion lower: leaf_operands length "
            f"{len(ctx.leaf_operands)} != expected {match.leaf_count}"
        )

    # Re-check dense leaf indexes against the frozen tree.
    _assert_leaf_indexes(claimed.expression, match.leaf_count)

    from rextio_numpy.rust_snippets.fusion import build_tree_plan, fusion_call_name

    tree_plan = build_tree_plan(claimed.expression)
    if len(tree_plan) != match.binop_count:
        raise ValueError(
            "rextio-numpy fusion lower: tree_plan length "
            f"{len(tree_plan)} != binop_count {match.binop_count}"
        )

    name = fusion_call_name(match.signature)
    args = ", ".join(f"&{leaf}" for leaf in ctx.leaf_operands)
    helpers = rust_snippets.fusion_helpers_bundle(
        signature=match.signature,
        dtype=match.dtype,
        result_rank=match.result_rank,
        leaf_ranks=match.leaf_ranks,
        expression_ops_postorder=match.postorder_ops,
        tree_plan=tree_plan,
    )
    return LoweredExpr(
        rust=f"{name}({args})?",
        helpers=helpers,
    )


def _assert_leaf_indexes(expression: object, leaf_count: int) -> None:
    """Fail closed when leaf_index metadata is not a dense LTR 0..n-1 sequence."""
    seen: list[int] = []

    def walk(node: object) -> None:
        kind = getattr(node, "kind", None)
        if kind == "leaf":
            idx = getattr(node, "leaf_index", None)
            leaf_kind = getattr(node, "leaf_kind", None)
            if leaf_kind != "name":
                raise ValueError(f"rextio-numpy fusion lower: non-name leaf_kind {leaf_kind!r}")
            if idx is None or idx < 0:
                raise ValueError(f"rextio-numpy fusion lower: bad leaf_index {idx!r}")
            seen.append(idx)
            return
        if kind == "binop":
            children = getattr(node, "children", ())
            if not isinstance(children, tuple) or len(children) != 2:
                raise ValueError("rextio-numpy fusion lower: binop must have 2 children")
            left, right = children
            walk(left)
            walk(right)
            return
        raise ValueError(f"rextio-numpy fusion lower: unexpected expression kind {kind!r}")

    walk(expression)
    if seen != list(range(leaf_count)):
        raise ValueError(
            "rextio-numpy fusion lower: leaf_index sequence "
            f"{seen!r} is not dense 0..{leaf_count - 1}"
        )
