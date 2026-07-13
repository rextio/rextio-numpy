"""Lowering for linear algebra sites (numpy.dot)."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Return a lowered expression for ``numpy.dot``, or None if not this lane."""
    if claimed.kind != "call" or claimed.target != "numpy.dot":
        return None
    return LoweredExpr(
        rust=f"__rxtnp_dot1(&{ctx.operands[0]}, &{ctx.operands[1]})?",
        helpers=(rust_snippets.dot1(),),
    )
