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
    base_arity = 0 if is_method else 1
    is_whole = len(claimed.operand_types) == base_arity and not claimed.keywords
    is_named_axis = len(claimed.operand_types) == base_arity and bool(claimed.keywords)
    is_positional_axis = (
        len(claimed.operand_types) == base_arity + 1 and not claimed.keywords
    )
    if not (is_whole or is_named_axis or is_positional_axis):
        raise ValueError(
            "rextio-numpy reductions lower requires a whole-array form or exactly "
            "one named/positional axis"
        )

    require_direct_context(ctx, "reductions", receiver=is_method)
    require_no_hidden_site_metadata(
        claimed,
        "reductions",
        allow_expression=True,
        expected_operand_literals=len(claimed.operand_types),
        allowed_literal_positions=(
            frozenset({base_arity}) if is_positional_axis else frozenset()
        ),
    )
    require_site_expression_matches_claim(claimed, "reductions")

    # Fail closed on malformed lower-time metadata rather than emitting
    # incorrect reduction code (asserts are stripped under PYTHONOPTIMIZE=1).
    if is_method:
        if ctx.receiver is None:
            raise ValueError("rextio-numpy method reductions lower requires ctx.receiver")
        operand = claimed.receiver.arg_type
        operand_expr = ctx.receiver
    else:
        if not claimed.operand_types:
            raise ValueError(
                "rextio-numpy module reductions lower requires an array operand type"
            )
        operand = claimed.operand_types[0]
        operand_expr = ctx.operands[0] if ctx.operands else ""
    if len(ctx.operands) != len(claimed.operand_types):
        raise ValueError(
            "rextio-numpy reductions lower requires one rendered ctx operand per "
            f"positional argument; got {len(ctx.operands)} for "
            f"{len(claimed.operand_types)} claimed operands"
        )
    if operand is None:
        raise ValueError("rextio-numpy reductions lower requires non-None operand type")
    meta = array_meta(operand)
    if meta is None:
        raise ValueError(
            f"rextio-numpy reductions lower requires array operand type, got {operand!r}"
        )
    dtype, rank = meta

    if is_whole:
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

    # Literal-axis path: independently reconstruct named or positional metadata.
    if is_positional_axis:
        axis_index = base_arity
        if claimed.operand_types[axis_index] != "int":
            raise ValueError(
                "rextio-numpy reductions lower requires positional axis operand "
                f"type 'int', got {claimed.operand_types[axis_index]!r}"
            )
        if len(claimed.operand_literals) != len(claimed.operand_types):
            raise ValueError(
                "rextio-numpy reductions lower requires arity-aligned operand_literals "
                "for a positional axis"
            )
        literal = claimed.operand_literals[axis_index]
    else:
        if len(claimed.keywords) != 1:
            raise ValueError(
                "rextio-numpy reductions lower requires exactly one keyword (axis=); "
                f"got {len(claimed.keywords)}"
            )
        keyword = claimed.keywords[0]
        if keyword.name != "axis":
            raise ValueError(
                "rextio-numpy reductions lower requires keyword name 'axis', "
                f"got {keyword.name!r}"
            )
        if keyword.arg_type != "int":
            raise ValueError(
                "rextio-numpy reductions lower requires axis keyword arg_type='int', "
                f"got {keyword.arg_type!r}"
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
    op = op_from_target(target)
    name = axis_call_name(op, dtype, rank, axis)
    helpers = axis_typed(op, dtype, rank, axis)
    return LoweredExpr(
        rust=f"{name}(&{operand_expr})?",
        helpers=helpers,
    )
