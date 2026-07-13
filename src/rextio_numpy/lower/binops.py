"""Lowering for elementwise binops (+, -, *, /)."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.claim.binops import BINOP_NAMES
from rextio_numpy.diagnostics import F64_1D


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Return a lowered expression for elementwise binops, or None if not this lane."""
    if claimed.kind != "binop" or claimed.target not in BINOP_NAMES:
        return None
    op = BINOP_NAMES[claimed.target]
    left, right = claimed.operand_types
    first, second = ctx.operands
    if left == F64_1D and right == F64_1D:
        return LoweredExpr(
            rust=f"__rxtnp_{op}1_aa(&{first}, &{second})?",
            helpers=(rust_snippets.elementwise_aa(op),),
        )
    if left == F64_1D:
        return LoweredExpr(
            rust=f"__rxtnp_{op}1_as(&{first}, {second})?",
            helpers=(rust_snippets.elementwise_as(op),),
        )
    return LoweredExpr(
        rust=f"__rxtnp_{op}1_sa({first}, &{second})?",
        helpers=(rust_snippets.elementwise_sa(op),),
    )
