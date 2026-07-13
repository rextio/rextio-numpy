"""Lowering for whole-array reductions (numpy.sum / numpy.mean)."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Return a lowered expression for sum/mean, or None if not this lane."""
    if claimed.kind != "call":
        return None
    if claimed.target == "numpy.sum":
        return LoweredExpr(
            rust=f"__rxtnp_sum1(&{ctx.operands[0]})?",
            helpers=(rust_snippets.sum1(),),
        )
    if claimed.target == "numpy.mean":
        return LoweredExpr(
            rust=f"__rxtnp_mean1(&{ctx.operands[0]})?",
            helpers=(rust_snippets.mean1(),),
        )
    return None
