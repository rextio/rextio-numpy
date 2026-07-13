"""Claim decisions for elementwise binops (+, -, *, /)."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_numpy.diagnostics import (
    SCALAR_FOR_DTYPE,
    array_meta,
    is_array_type,
    not_covered_or_rejected,
    type_key_for,
)

# Binary operator token -> the elementwise helper op name (shared with lower).
BINOP_NAMES = {"+": "add", "-": "sub", "*": "mul", "/": "div"}

_ELEMENTWISE_RULE = "rextio-numpy/elementwise-float64"


def _result_array_type(dtype: str, rank: int, op: str) -> str:
    """Return the result plugin type key for an elementwise op on ``dtype``."""
    # NumPy true division of integers yields float64 at the broadcast rank.
    if op == "/" and dtype == "i64":
        return type_key_for("f64", rank)
    return type_key_for(dtype, rank)


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Return a claim result for elementwise binops, or None if not this lane."""
    if site.kind != "binop" or site.target not in BINOP_NAMES:
        return None
    if len(site.operand_types) != 2:
        return not_covered_or_rejected(site)
    left, right = site.operand_types
    left_arr = is_array_type(left)
    right_arr = is_array_type(right)

    if not left_arr and not right_arr:
        # No plugin-typed operand: not this plugin's business.
        return NotCovered()

    # Unresolved handled uniformly (NotCovered) before dtype checks.
    if left is None or right is None:
        return not_covered_or_rejected(site)

    op = site.target

    if left_arr and right_arr:
        assert left is not None and right is not None
        left_meta = array_meta(left)
        right_meta = array_meta(right)
        assert left_meta is not None and right_meta is not None
        left_dtype, left_rank = left_meta
        right_dtype, right_rank = right_meta
        if left_dtype != right_dtype:
            # Mixed array dtypes are known-unsupported.
            return not_covered_or_rejected(site)
        result_rank = max(left_rank, right_rank)
        return Claimed(
            rule_id=_ELEMENTWISE_RULE,
            result_type=_result_array_type(left_dtype, result_rank, op),
        )

    # Exactly one operand is an array; the other must be the matching scalar.
    if left_arr:
        assert left is not None
        meta = array_meta(left)
        assert meta is not None
        dtype, rank = meta
        if right != SCALAR_FOR_DTYPE[dtype]:
            return not_covered_or_rejected(site)
        return Claimed(
            rule_id=_ELEMENTWISE_RULE,
            result_type=_result_array_type(dtype, rank, op),
        )

    assert right is not None
    meta = array_meta(right)
    assert meta is not None
    dtype, rank = meta
    if left != SCALAR_FOR_DTYPE[dtype]:
        return not_covered_or_rejected(site)
    return Claimed(
        rule_id=_ELEMENTWISE_RULE,
        result_type=_result_array_type(dtype, rank, op),
    )
