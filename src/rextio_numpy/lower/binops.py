"""Lowering for elementwise binops (+, -, *, /)."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.claim.binops import BINOP_NAMES
from rextio_numpy.diagnostics import array_meta, is_array_type


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Return a lowered expression for elementwise binops, or None if not this lane."""
    if claimed.kind != "binop" or claimed.target not in BINOP_NAMES:
        return None
    op = BINOP_NAMES[claimed.target]
    left, right = claimed.operand_types
    first, second = ctx.operands

    if is_array_type(left) and is_array_type(right):
        assert left is not None and right is not None
        left_meta = array_meta(left)
        right_meta = array_meta(right)
        assert left_meta is not None and right_meta is not None
        dtype, left_rank = left_meta
        right_dtype, right_rank = right_meta
        assert dtype == right_dtype
        name = rust_snippets.elementwise_call_name_aa(op, dtype, left_rank, right_rank)
        helper = rust_snippets.elementwise_aa_typed(op, dtype, left_rank, right_rank)
        helpers: tuple[str, ...]
        if dtype == "f64" and left_rank == 1 and right_rank == 1:
            helpers = (helper,)
        else:
            helpers = (*rust_snippets.shared_broadcast_helpers(), helper)
        return LoweredExpr(
            rust=f"{name}(&{first}, &{second})?",
            helpers=helpers,
        )

    if is_array_type(left):
        assert left is not None
        meta = array_meta(left)
        assert meta is not None
        dtype, rank = meta
        name = rust_snippets.elementwise_call_name_as(op, dtype, rank)
        helper = rust_snippets.elementwise_as_typed(op, dtype, rank)
        return LoweredExpr(
            rust=f"{name}(&{first}, {second})?",
            helpers=(helper,),
        )

    assert right is not None
    meta = array_meta(right)
    assert meta is not None
    dtype, rank = meta
    name = rust_snippets.elementwise_call_name_sa(op, dtype, rank)
    helper = rust_snippets.elementwise_sa_typed(op, dtype, rank)
    return LoweredExpr(
        rust=f"{name}({first}, &{second})?",
        helpers=(helper,),
    )
