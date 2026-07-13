"""Lowering for linear algebra sites (numpy.dot)."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.diagnostics import array_meta


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Return a lowered expression for ``numpy.dot``, or None if not this lane."""
    if claimed.kind != "call" or claimed.target != "numpy.dot":
        return None
    left = claimed.operand_types[0]
    assert left is not None
    meta = array_meta(left)
    assert meta is not None
    dtype, _rank = meta
    name = rust_snippets.dot_call_name(dtype)
    helper = rust_snippets.dot_typed(dtype)
    return LoweredExpr(
        rust=f"{name}(&{ctx.operands[0]}, &{ctx.operands[1]})?",
        helpers=(helper,),
    )
