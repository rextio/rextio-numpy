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
    if len(claimed.operand_types) != 2:
        raise ValueError(
            "rextio-numpy binops lower requires exactly two operand types; "
            f"got {len(claimed.operand_types)}"
        )
    left, right = claimed.operand_types
    if len(ctx.operands) != 2:
        raise ValueError(
            f"rextio-numpy binops lower requires exactly two ctx.operands; got {len(ctx.operands)}"
        )
    first, second = ctx.operands

    # Fail closed on malformed lower-time metadata rather than emitting
    # incorrect elementwise code (asserts are stripped under PYTHONOPTIMIZE=1).
    if is_array_type(left) and is_array_type(right):
        if left is None or right is None:
            raise ValueError("rextio-numpy binops lower requires non-None array operand types")
        left_meta = array_meta(left)
        right_meta = array_meta(right)
        if left_meta is None or right_meta is None:
            raise ValueError(
                "rextio-numpy binops lower requires array operand types, "
                f"got {left!r} and {right!r}"
            )
        dtype, left_rank = left_meta
        right_dtype, right_rank = right_meta
        if dtype != right_dtype:
            raise ValueError(
                "rextio-numpy binops lower requires matching array dtypes, "
                f"got {dtype!r} and {right_dtype!r}"
            )
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
        if left is None:
            raise ValueError("rextio-numpy binops lower requires non-None left array operand type")
        meta = array_meta(left)
        if meta is None:
            raise ValueError(
                f"rextio-numpy binops lower requires array left operand type, got {left!r}"
            )
        dtype, rank = meta
        name = rust_snippets.elementwise_call_name_as(op, dtype, rank)
        helper = rust_snippets.elementwise_as_typed(op, dtype, rank)
        return LoweredExpr(
            rust=f"{name}(&{first}, {second})?",
            helpers=(helper,),
        )

    if right is None:
        raise ValueError("rextio-numpy binops lower requires non-None right array operand type")
    meta = array_meta(right)
    if meta is None:
        raise ValueError(
            f"rextio-numpy binops lower requires array right operand type, got {right!r}"
        )
    dtype, rank = meta
    name = rust_snippets.elementwise_call_name_sa(op, dtype, rank)
    helper = rust_snippets.elementwise_sa_typed(op, dtype, rank)
    return LoweredExpr(
        rust=f"{name}({first}, &{second})?",
        helpers=(helper,),
    )
