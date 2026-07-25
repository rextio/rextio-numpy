"""Defensive lowering for resident-mask logical composition."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.claim.logical import (
    LOGICAL_BINARY_RULE,
    LOGICAL_BINARY_TARGETS,
    LOGICAL_NOT_RULE,
    LOGICAL_NOT_TARGET,
)
from rextio_numpy.diagnostics import bool_rank, bool_type_for, is_bool_type
from rextio_numpy.lower.contracts import (
    require_direct_context,
    require_no_hidden_site_metadata,
    require_result_type,
    require_rule_id,
    require_site_expression_matches_claim,
)


def _require_mask(mask: str | None, lane: str) -> tuple[str, int]:
    """Reconstruct one resident bool mask key and its fixed rank."""
    if mask is None or not is_bool_type(mask):
        raise ValueError(f"rextio-numpy {lane} lower requires a resident bool mask, got {mask!r}")
    rank = bool_rank(mask)
    if rank is None:
        raise ValueError(f"rextio-numpy {lane} lower received an unknown mask rank: {mask!r}")
    return mask, rank


def _lower_not(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
    lane = "resident logical_not"
    require_rule_id(claimed, LOGICAL_NOT_RULE, lane)
    require_direct_context(ctx, lane, receiver=False)
    require_no_hidden_site_metadata(
        claimed, lane, allow_expression=True, expected_operand_literals=1
    )
    require_site_expression_matches_claim(claimed, lane)
    if claimed.receiver is not None or claimed.keywords or len(claimed.operand_types) != 1:
        raise ValueError("rextio-numpy resident logical_not lower requires one positional mask")
    if len(ctx.operands) != 1:
        raise ValueError("rextio-numpy resident logical_not lower requires one rendered operand")
    mask, rank = _require_mask(claimed.operand_types[0], lane)
    require_result_type(claimed, mask, lane)
    name = rust_snippets.logical_not_call_name(rank)
    return LoweredExpr(
        rust=f"{name}(&{ctx.operands[0]})?",
        helpers=(rust_snippets.logical_not_typed(rank),),
    )


def _lower_binary(claimed: ClaimSite, ctx: LoweringContext, op: str) -> LoweredExpr:
    lane = f"resident logical_{op}"
    require_rule_id(claimed, LOGICAL_BINARY_RULE, lane)
    require_direct_context(ctx, lane, receiver=False)
    require_no_hidden_site_metadata(
        claimed, lane, allow_expression=True, expected_operand_literals=2
    )
    require_site_expression_matches_claim(claimed, lane)
    if claimed.receiver is not None or claimed.keywords or len(claimed.operand_types) != 2:
        raise ValueError(f"rextio-numpy {lane} lower requires exactly two positional masks")
    if len(ctx.operands) != 2:
        raise ValueError(f"rextio-numpy {lane} lower requires exactly two rendered operands")
    _left, left_rank = _require_mask(claimed.operand_types[0], lane)
    _right, right_rank = _require_mask(claimed.operand_types[1], lane)
    result_rank = max(left_rank, right_rank)
    require_result_type(claimed, bool_type_for(result_rank), lane)
    name = rust_snippets.logical_binary_call_name(op, left_rank, right_rank)
    return LoweredExpr(
        rust=f"{name}(&{ctx.operands[0]}, &{ctx.operands[1]})?",
        helpers=(*rust_snippets.shared_broadcast_helpers(), rust_snippets.logical_binary_typed(op, left_rank, right_rank)),
    )


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower exact logical resident-mask calls, or return ``None`` for other sites."""
    if claimed.kind != "call":
        return None
    if claimed.target == LOGICAL_NOT_TARGET:
        return _lower_not(claimed, ctx)
    op = LOGICAL_BINARY_TARGETS.get(claimed.target)
    if op is not None:
        return _lower_binary(claimed, ctx, op)
    return None


__all__ = ["try_lower"]
