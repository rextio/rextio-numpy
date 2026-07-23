"""Lowering for whole-array and literal-axis reductions."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.claim.reductions import (
    _AXIS_RULE,
    _WHOLE_ARRAY_RULE,
    _axis_result_type,
    _dtype_allowed,
    _whole_array_result_type,
    normalize_axis,
)
from rextio_numpy.diagnostics import array_meta
from rextio_numpy.lower.contracts import (
    require_direct_context,
    require_no_hidden_site_metadata,
    require_result_type,
    require_rule_id,
    require_site_expression_matches_claim,
)
from rextio_numpy.rust_snippets.reductions import axis_call_name, axis_typed, op_from_target

_WHOLE_ARRAY_TARGETS = ("numpy.sum", "numpy.mean")
_AXIS_TARGETS = ("numpy.sum", "numpy.mean", "numpy.max", "numpy.min")


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Return a lowered expression for sum/mean/max/min, or None if not this lane."""
    if claimed.kind != "call":
        return None
    method_name = claimed.target.rpartition(".")[2]
    is_module = claimed.target in _AXIS_TARGETS
    is_method = not is_module and method_name in {
        "sum",
        "mean",
        "max",
        "min",
    }
    if not (is_module or is_method):
        return None
    if is_module and claimed.receiver is not None:
        raise ValueError("rextio-numpy module reductions lower requires no ClaimSite.receiver")
    if is_method and claimed.receiver is None:
        raise ValueError("rextio-numpy method reductions lower requires ClaimSite.receiver")
    target = f"numpy.{method_name}" if is_method else claimed.target
    require_direct_context(ctx, "reductions", receiver=is_method)
    require_no_hidden_site_metadata(
        claimed,
        "reductions",
        allow_expression=True,
        expected_operand_literals=0 if is_method else 1,
    )
    require_site_expression_matches_claim(claimed, "reductions")

    # Fail closed on malformed lower-time metadata rather than emitting
    # incorrect reduction code (asserts are stripped under PYTHONOPTIMIZE=1).
    expected_arity = 0 if is_method else 1
    if len(claimed.operand_types) != expected_arity:
        requirement = (
            "exactly zero operand types"
            if is_method
            else "exactly one operand type"
        )
        raise ValueError(
            f"rextio-numpy reductions lower requires {requirement}; "
            f"got {len(claimed.operand_types)}"
        )
    if is_method:
        if ctx.receiver is None:
            raise ValueError("rextio-numpy method reductions lower requires ctx.receiver")
        operand = claimed.receiver.arg_type
        operand_expr = ctx.receiver
        expected_ctx_arity = 0
    else:
        operand = claimed.operand_types[0]
        operand_expr = ctx.operands[0] if ctx.operands else ""
        expected_ctx_arity = 1
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
        if target not in _WHOLE_ARRAY_TARGETS:
            raise ValueError(
                f"rextio-numpy whole-array reduction lower does not certify {target!r}"
            )
        if not _dtype_allowed(target, dtype, rank, axis=False):
            raise ValueError(
                "rextio-numpy reduction lower operand is outside certified dtype/rank matrix: "
                f"target={target!r}, dtype={dtype!r}, rank={rank}, axis=False"
            )
        require_rule_id(claimed, _WHOLE_ARRAY_RULE, "reductions")
        require_result_type(claimed, _whole_array_result_type(target, dtype), "reductions")
        if len(ctx.operands) != expected_ctx_arity:
            requirement = (
                "exactly zero ctx.operands entries"
                if is_method
                else "exactly one ctx.operands entry"
            )
            raise ValueError(
                f"rextio-numpy reductions lower requires {requirement}; "
                f"got {len(ctx.operands)}"
            )
        if target == "numpy.sum":
            name = rust_snippets.sum_call_name(dtype, rank)
            helper = rust_snippets.sum_typed(dtype, rank)
        else:
            name = rust_snippets.mean_call_name(dtype, rank)
            helper = rust_snippets.mean_typed(dtype, rank)
        return LoweredExpr(
            rust=f"{name}(&{operand_expr})?",
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
    if not _dtype_allowed(target, dtype, rank, axis=True):
        raise ValueError(
            "rextio-numpy reduction lower operand is outside certified dtype/rank matrix: "
            f"target={target!r}, dtype={dtype!r}, rank={rank}, axis=True"
        )
    require_rule_id(claimed, _AXIS_RULE, "reductions")
    require_result_type(claimed, _axis_result_type(target, dtype, rank), "reductions")
    if len(ctx.operands) != expected_ctx_arity:
        requirement = (
            "exactly zero ctx.operands entries"
            if is_method
            else "exactly one ctx.operands entry"
        )
        raise ValueError(
            f"rextio-numpy reductions lower requires {requirement}; "
            f"got {len(ctx.operands)}"
        )
    op = op_from_target(target)
    name = axis_call_name(op, dtype, rank, axis)
    helpers = axis_typed(op, dtype, rank, axis)
    return LoweredExpr(
        rust=f"{name}(&{operand_expr})?",
        helpers=helpers,
    )
