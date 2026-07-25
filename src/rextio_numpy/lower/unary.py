"""Lowering for exact one-argument NumPy unary module calls."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.claim.unary import _RULE
from rextio_numpy.diagnostics import array_meta
from rextio_numpy.lower.contracts import (
    require_direct_context,
    require_no_hidden_site_metadata,
    require_result_type,
    require_rule_id,
    require_site_expression_matches_claim,
)

_OPS = {
    "numpy.negative": "negative",
    "numpy.absolute": "absolute",
    "numpy.abs": "absolute",
    "numpy.square": "square",
}


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Return a lowered unary module call, or None if not this lane."""
    op = _OPS.get(claimed.target)
    if claimed.kind != "call" or op is None:
        return None
    if claimed.receiver is not None:
        raise ValueError("rextio-numpy unary lower requires no ClaimSite.receiver")
    require_rule_id(claimed, _RULE, "unary")
    require_direct_context(ctx, "unary", receiver=False)
    require_no_hidden_site_metadata(
        claimed,
        "unary",
        allow_expression=True,
        expected_operand_literals=1,
    )
    require_site_expression_matches_claim(claimed, "unary")
    if len(claimed.operand_types) != 1:
        raise ValueError(
            "rextio-numpy unary lower requires exactly one operand type; "
            f"got {len(claimed.operand_types)}"
        )
    if len(ctx.operands) != 1:
        raise ValueError(
            "rextio-numpy unary lower requires exactly one ctx.operands entry; "
            f"got {len(ctx.operands)}"
        )
    if claimed.keywords:
        raise ValueError("rextio-numpy unary lower does not support keyword arguments")
    operand = claimed.operand_types[0]
    if operand is None:
        raise ValueError("rextio-numpy unary lower requires non-None operand type")
    meta = array_meta(operand)
    if meta is None:
        raise ValueError(f"rextio-numpy unary lower requires array operand type, got {operand!r}")
    dtype, rank = meta
    require_result_type(claimed, operand, "unary")
    name = rust_snippets.unary_call_name(op, dtype, rank)
    helper = rust_snippets.unary_typed(op, dtype, rank)
    return LoweredExpr(rust=f"{name}(&{ctx.operands[0]})?", helpers=(helper,))
