"""Defensive lowering for API-1.5 elementwise comparisons."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.claim.compare import COMPARE_NAMES, COMPARE_RULE
from rextio_numpy.diagnostics import (
    SCALAR_FOR_DTYPE,
    array_meta,
    bool_type_for,
    is_array_type,
)
from rextio_numpy.lower.binops import _require_literal_metadata
from rextio_numpy.lower.contracts import (
    require_direct_context,
    require_no_hidden_site_metadata,
    require_result_type,
    require_rule_id,
)


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower one exact non-chained comparison or return ``None``."""
    if claimed.kind != "compare" or claimed.target not in COMPARE_NAMES:
        return None
    lane = "comparisons"
    require_rule_id(claimed, COMPARE_RULE, lane)
    require_direct_context(ctx, lane, receiver=False)
    require_no_hidden_site_metadata(
        claimed,
        lane,
        expected_operand_literals=2,
        allowed_literal_positions=frozenset({0, 1}),
    )
    if claimed.receiver is not None or claimed.keywords:
        raise ValueError(
            "rextio-numpy comparisons lower requires no receiver or keywords"
        )
    if len(claimed.operand_types) != 2 or len(ctx.operands) != 2:
        raise ValueError(
            "rextio-numpy comparisons lower requires exactly two typed/rendered operands"
        )
    _require_literal_metadata(claimed, lane)
    left, right = claimed.operand_types
    first, second = ctx.operands
    op = COMPARE_NAMES[claimed.target]

    if is_array_type(left) and is_array_type(right):
        if left is None or right is None:
            raise ValueError("rextio-numpy comparisons lower requires resolved arrays")
        left_meta = array_meta(left)
        right_meta = array_meta(right)
        if left_meta is None or right_meta is None or left_meta[0] != right_meta[0]:
            raise ValueError(
                "rextio-numpy comparisons lower requires same-dtype numeric arrays"
            )
        dtype, left_rank = left_meta
        right_rank = right_meta[1]
        require_result_type(
            claimed,
            bool_type_for(max(left_rank, right_rank)),
            lane,
        )
        name = rust_snippets.comparison_call_name_aa(
            op,
            dtype,
            left_rank,
            right_rank,
        )
        helper = rust_snippets.comparison_aa_typed(
            op,
            dtype,
            left_rank,
            right_rank,
        )
        return LoweredExpr(
            rust=f"{name}(&{first}, &{second})?",
            helpers=(*rust_snippets.shared_broadcast_helpers(), helper),
        )

    if is_array_type(left):
        if left is None:
            raise ValueError("rextio-numpy comparisons lower requires a left array")
        meta = array_meta(left)
        if meta is None:
            raise ValueError("rextio-numpy comparisons lower requires a numeric left array")
        dtype, rank = meta
        if right != SCALAR_FOR_DTYPE[dtype]:
            raise ValueError(
                "rextio-numpy comparisons lower requires a matching right scalar"
            )
        require_result_type(claimed, bool_type_for(rank), lane)
        name = rust_snippets.comparison_call_name_as(op, dtype, rank)
        return LoweredExpr(
            rust=f"{name}(&{first}, {second})?",
            helpers=(rust_snippets.comparison_as_typed(op, dtype, rank),),
        )

    if right is None:
        raise ValueError("rextio-numpy comparisons lower requires a right array")
    meta = array_meta(right)
    if meta is None:
        raise ValueError("rextio-numpy comparisons lower requires a numeric right array")
    dtype, rank = meta
    if left != SCALAR_FOR_DTYPE[dtype]:
        raise ValueError(
            "rextio-numpy comparisons lower requires a matching left scalar"
        )
    require_result_type(claimed, bool_type_for(rank), lane)
    name = rust_snippets.comparison_call_name_sa(op, dtype, rank)
    return LoweredExpr(
        rust=f"{name}({first}, &{second})?",
        helpers=(rust_snippets.comparison_sa_typed(op, dtype, rank),),
    )


__all__ = ["try_lower"]
