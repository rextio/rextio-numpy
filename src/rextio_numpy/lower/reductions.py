"""Lowering for whole-array reductions (numpy.sum / numpy.mean)."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.diagnostics import array_meta


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Return a lowered expression for sum/mean, or None if not this lane."""
    if claimed.kind != "call":
        return None
    if claimed.target not in ("numpy.sum", "numpy.mean"):
        return None
    operand = claimed.operand_types[0]
    assert operand is not None
    meta = array_meta(operand)
    assert meta is not None
    dtype, rank = meta
    if claimed.target == "numpy.sum":
        name = rust_snippets.sum_call_name(dtype, rank)
        helper = rust_snippets.sum_typed(dtype, rank)
    else:
        name = rust_snippets.mean_call_name(dtype, rank)
        helper = rust_snippets.mean_typed(dtype, rank)
    return LoweredExpr(
        rust=f"{name}(&{ctx.operands[0]})?",
        helpers=(helper,),
    )
