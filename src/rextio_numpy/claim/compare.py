"""Claims for API-1.5 non-chained elementwise comparison expressions."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_numpy.diagnostics import (
    SCALAR_FOR_DTYPE,
    array_meta,
    bool_type_for,
    is_array_type,
    not_covered_or_rejected,
)

COMPARE_NAMES: dict[str, str] = {
    "==": "eq",
    "!=": "ne",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
}
COMPARE_RULE = "rextio-numpy/elementwise-compare"


def _claim_pair(site: ClaimSite) -> ClaimResult:
    if len(site.operand_types) != 2:
        return not_covered_or_rejected(site)
    left, right = site.operand_types
    left_array = is_array_type(left)
    right_array = is_array_type(right)
    if not left_array and not right_array:
        return not_covered_or_rejected(site)
    if left is None or right is None:
        return NotCovered()

    if left_array and right_array:
        left_meta = array_meta(left)
        right_meta = array_meta(right)
        if left_meta is None or right_meta is None:
            return not_covered_or_rejected(site)
        left_dtype, left_rank = left_meta
        right_dtype, right_rank = right_meta
        if left_dtype != right_dtype:
            return not_covered_or_rejected(site)
        return Claimed(
            rule_id=COMPARE_RULE,
            result_type=bool_type_for(max(left_rank, right_rank)),
        )

    if left_array:
        left_meta = array_meta(left)
        if left_meta is None:
            return not_covered_or_rejected(site)
        dtype, rank = left_meta
        if right != SCALAR_FOR_DTYPE[dtype]:
            return not_covered_or_rejected(site)
        return Claimed(rule_id=COMPARE_RULE, result_type=bool_type_for(rank))

    right_meta = array_meta(right)
    if right_meta is None:
        return not_covered_or_rejected(site)
    dtype, rank = right_meta
    if left != SCALAR_FOR_DTYPE[dtype]:
        return not_covered_or_rejected(site)
    return Claimed(rule_id=COMPARE_RULE, result_type=bool_type_for(rank))


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim exact API-1.5 comparison tokens over the bounded numeric matrix."""
    if site.kind != "compare" or site.target not in COMPARE_NAMES:
        return None
    if site.receiver is not None or site.keywords or site.callables:
        return not_covered_or_rejected(site)
    return _claim_pair(site)


__all__ = ["COMPARE_NAMES", "COMPARE_RULE", "try_claim"]
