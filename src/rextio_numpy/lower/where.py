"""Defensive lowering for bounded three-argument ``numpy.where``."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.claim.where import WHERE_RULE, WHERE_TARGET, _branch_contract
from rextio_numpy.diagnostics import (
    SCALAR_FOR_DTYPE,
    array_meta,
    bool_rank,
    is_array_type,
    is_bool_type,
    type_key_for,
)
from rextio_numpy.lower.binops import _require_literal_metadata
from rextio_numpy.lower.contracts import (
    require_direct_context,
    require_no_hidden_site_metadata,
    require_result_type,
    require_rule_id,
    require_site_expression_matches_claim,
)


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower exact ``numpy.where(condition, yes, no)`` or return ``None``."""
    if claimed.kind != "call" or claimed.target != WHERE_TARGET:
        return None
    lane = "three-argument where"
    require_rule_id(claimed, WHERE_RULE, lane)
    require_direct_context(ctx, lane, receiver=False)
    require_no_hidden_site_metadata(
        claimed,
        lane,
        allow_expression=True,
        expected_operand_literals=3,
        allowed_literal_positions=frozenset({1, 2}),
    )
    require_site_expression_matches_claim(claimed, lane)
    if claimed.receiver is not None or claimed.keywords:
        raise ValueError(
            "rextio-numpy three-argument where lower requires no receiver or keywords"
        )
    if len(claimed.operand_types) != 3 or len(ctx.operands) != 3:
        raise ValueError(
            "rextio-numpy three-argument where lower requires exactly three operands"
        )
    _require_literal_metadata(claimed, lane)
    condition_type, yes_type, no_type = claimed.operand_types
    if (
        condition_type is None
        or yes_type is None
        or no_type is None
        or not is_bool_type(condition_type)
    ):
        raise ValueError(
            "rextio-numpy three-argument where lower requires a resident bool condition"
        )
    condition_rank = bool_rank(condition_type)
    branch = _branch_contract(yes_type, no_type)
    if condition_rank is None or branch is None:
        raise ValueError(
            "rextio-numpy three-argument where lower received unsupported branch metadata"
        )
    dtype, branch_rank = branch
    require_result_type(
        claimed,
        type_key_for(dtype, max(condition_rank, branch_rank)),
        lane,
    )
    condition, yes, no = ctx.operands
    common_helpers = (
        rust_snippets.fmt_shape_helper(),
        rust_snippets.broadcast_shape3_helper(),
    )

    if is_array_type(yes_type) and is_array_type(no_type):
        yes_meta = array_meta(yes_type)
        no_meta = array_meta(no_type)
        if yes_meta is None or no_meta is None or yes_meta[0] != no_meta[0]:
            raise ValueError(
                "rextio-numpy three-argument where lower requires same-dtype arrays"
            )
        name = rust_snippets.where_call_name_aa(
            condition_rank,
            dtype,
            yes_meta[1],
            no_meta[1],
        )
        helper = rust_snippets.where_aa_typed(
            condition_rank,
            dtype,
            yes_meta[1],
            no_meta[1],
        )
        return LoweredExpr(
            rust=f"{name}(&{condition}, &{yes}, &{no})?",
            helpers=(*common_helpers, helper),
        )

    if is_array_type(yes_type):
        yes_meta = array_meta(yes_type)
        if yes_meta is None or no_type != SCALAR_FOR_DTYPE[dtype]:
            raise ValueError(
                "rextio-numpy three-argument where lower requires a matching no scalar"
            )
        name = rust_snippets.where_call_name_as(
            condition_rank,
            dtype,
            yes_meta[1],
        )
        return LoweredExpr(
            rust=f"{name}(&{condition}, &{yes}, {no})?",
            helpers=(
                *common_helpers,
                rust_snippets.where_as_typed(condition_rank, dtype, yes_meta[1]),
            ),
        )

    no_meta = array_meta(no_type)
    if no_meta is None or yes_type != SCALAR_FOR_DTYPE[dtype]:
        raise ValueError(
            "rextio-numpy three-argument where lower requires a matching yes scalar"
        )
    name = rust_snippets.where_call_name_sa(condition_rank, dtype, no_meta[1])
    return LoweredExpr(
        rust=f"{name}(&{condition}, {yes}, &{no})?",
        helpers=(
            *common_helpers,
            rust_snippets.where_sa_typed(condition_rank, dtype, no_meta[1]),
        ),
    )


__all__ = ["try_lower"]
