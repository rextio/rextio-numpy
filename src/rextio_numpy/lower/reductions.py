"""Lowering for whole-array and literal-axis reductions."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.claim.reductions import normalize_axis
from rextio_numpy.diagnostics import array_meta
from rextio_numpy.rust_snippets.reductions import axis_call_name, axis_typed, op_from_target

_WHOLE_ARRAY_TARGETS = ("numpy.sum", "numpy.mean")
_AXIS_TARGETS = ("numpy.sum", "numpy.mean", "numpy.max", "numpy.min")


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Return a lowered expression for sum/mean/max/min, or None if not this lane."""
    if claimed.kind != "call":
        return None
    if claimed.target not in _AXIS_TARGETS:
        return None

    operand = claimed.operand_types[0]
    assert operand is not None
    meta = array_meta(operand)
    assert meta is not None
    dtype, rank = meta

    if not claimed.keywords:
        # Whole-array sum/mean (bare max/min are never claimed).
        if claimed.target not in _WHOLE_ARRAY_TARGETS:
            return None
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

    # Literal-axis path: re-normalize axis for deterministic helper identity.
    assert len(claimed.keywords) == 1
    raw = claimed.keywords[0].literal.value
    assert isinstance(raw, int)
    axis = normalize_axis(raw, rank)
    assert axis is not None
    op = op_from_target(claimed.target)
    name = axis_call_name(op, dtype, rank, axis)
    helpers = axis_typed(op, dtype, rank, axis)
    return LoweredExpr(
        rust=f"{name}(&{ctx.operands[0]})?",
        helpers=helpers,
    )
