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

    # Fail closed on malformed lower-time metadata rather than emitting
    # incorrect reduction code (asserts are stripped under PYTHONOPTIMIZE=1).
    if len(claimed.operand_types) != 1:
        raise ValueError(
            "rextio-numpy reductions lower requires exactly one operand type; "
            f"got {len(claimed.operand_types)}"
        )
    operand = claimed.operand_types[0]
    if operand is None:
        raise ValueError("rextio-numpy reductions lower requires non-None operand type")
    meta = array_meta(operand)
    if meta is None:
        raise ValueError(
            f"rextio-numpy reductions lower requires array operand type, got {operand!r}"
        )
    dtype, rank = meta

    if not claimed.keywords:
        # Whole-array sum/mean (bare max/min are never claimed).
        if claimed.target not in _WHOLE_ARRAY_TARGETS:
            return None
        if len(ctx.operands) != 1:
            raise ValueError(
                "rextio-numpy reductions lower requires exactly one ctx.operands entry; "
                f"got {len(ctx.operands)}"
            )
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
    if len(claimed.keywords) != 1:
        raise ValueError(
            "rextio-numpy reductions lower requires exactly one keyword (axis=); "
            f"got {len(claimed.keywords)}"
        )
    keyword = claimed.keywords[0]
    if keyword.name != "axis":
        raise ValueError(
            f"rextio-numpy reductions lower requires keyword name 'axis', got {keyword.name!r}"
        )
    literal = keyword.literal
    if not literal.is_literal:
        raise ValueError(
            "rextio-numpy reductions lower requires axis ClaimLiteral with is_literal=True"
        )
    raw = literal.value
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"rextio-numpy reductions lower requires int axis literal, got {raw!r}")
    axis = normalize_axis(raw, rank)
    if axis is None:
        raise ValueError(
            f"rextio-numpy reductions lower: axis {raw!r} out of range for rank {rank}"
        )
    if len(ctx.operands) != 1:
        raise ValueError(
            "rextio-numpy reductions lower requires exactly one ctx.operands entry; "
            f"got {len(ctx.operands)}"
        )
    op = op_from_target(claimed.target)
    name = axis_call_name(op, dtype, rank, axis)
    helpers = axis_typed(op, dtype, rank, axis)
    return LoweredExpr(
        rust=f"{name}(&{ctx.operands[0]})?",
        helpers=helpers,
    )
