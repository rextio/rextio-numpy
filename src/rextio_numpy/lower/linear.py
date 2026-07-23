"""Lowering for linear algebra sites (numpy.dot)."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.claim.linear import _DOT_RESULT, _DOT_RULE
from rextio_numpy.diagnostics import array_meta
from rextio_numpy.lower.contracts import (
    require_direct_context,
    require_no_hidden_site_metadata,
    require_result_type,
    require_rule_id,
    require_site_expression_matches_claim,
)


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Return a lowered expression for a certified 1-D dot site, or None."""
    is_module = claimed.target == "numpy.dot"
    is_method = not is_module and claimed.target.rpartition(".")[2] == "dot"
    if claimed.kind != "call" or not (is_module or is_method):
        return None
    if is_module and claimed.receiver is not None:
        raise ValueError("rextio-numpy module dot lower requires no ClaimSite.receiver")
    if is_method and claimed.receiver is None:
        raise ValueError("rextio-numpy method dot lower requires ClaimSite.receiver")
    require_rule_id(claimed, _DOT_RULE, "dot")
    require_direct_context(ctx, "dot", receiver=is_method)
    require_no_hidden_site_metadata(
        claimed,
        "dot",
        allow_expression=True,
        expected_operand_literals=1 if is_method else 2,
    )
    require_site_expression_matches_claim(claimed, "dot")
    if claimed.keywords:
        raise ValueError("rextio-numpy dot lower does not support keyword arguments")
    if is_method:
        if len(claimed.operand_types) != 1:
            raise ValueError(
                "rextio-numpy method dot lower requires exactly one operand type; "
                f"got {len(claimed.operand_types)}"
            )
        if ctx.receiver is None:
            raise ValueError("rextio-numpy method dot lower requires ctx.receiver")
        if len(ctx.operands) != 1:
            raise ValueError(
                "rextio-numpy method dot lower requires exactly one ctx.operands entry; "
                f"got {len(ctx.operands)}"
            )
        left = claimed.receiver.arg_type
        right = claimed.operand_types[0]
        left_expr = ctx.receiver
        right_expr = ctx.operands[0]
    else:
        if len(claimed.operand_types) != 2:
            raise ValueError(
                "rextio-numpy dot lower requires exactly two operand types; "
                f"got {len(claimed.operand_types)}"
            )
        if len(ctx.operands) != 2:
            raise ValueError(
                "rextio-numpy dot lower requires exactly two ctx.operands entries; "
                f"got {len(ctx.operands)}"
            )
        left = claimed.operand_types[0]
        right = claimed.operand_types[1]
        left_expr, right_expr = ctx.operands
    if left is None:
        raise ValueError("rextio-numpy dot lower requires non-None left operand type")
    meta = array_meta(left)
    if meta is None:
        raise ValueError(f"rextio-numpy dot lower requires array left operand type, got {left!r}")
    if right is None:
        raise ValueError("rextio-numpy dot lower requires non-None right operand type")
    right_meta = array_meta(right)
    dtype, rank = meta
    if right_meta != meta or dtype not in {"f64", "i64"} or rank != 1:
        raise ValueError(
            "rextio-numpy dot lower requires certified same-dtype rank-1 f64/i64 "
            f"operands; got left={left!r}, right={right!r}"
        )
    require_result_type(claimed, _DOT_RESULT[dtype], "dot")
    name = rust_snippets.dot_call_name(dtype)
    helper = rust_snippets.dot_typed(dtype)
    return LoweredExpr(
        rust=f"{name}(&{left_expr}, &{right_expr})?",
        helpers=(helper,),
    )
