"""Claims for resident-boolean NumPy logical composition.

The boolean ndarray keys owned by this plugin are result-only resident types:
they can be produced only by an already claimed expression and cannot cross a
Python boundary.  These routes deliberately keep that property: logical calls
accept only resident masks and preserve the result-only property.
"""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite

from rextio_numpy.diagnostics import (
    bool_rank,
    bool_type_for,
    is_bool_type,
    not_covered_or_rejected,
)

LOGICAL_NOT_TARGET = "numpy.logical_not"
LOGICAL_BINARY_TARGETS: dict[str, str] = {
    "numpy.logical_and": "and",
    "numpy.logical_or": "or",
}
LOGICAL_NOT_RULE = "rextio-numpy/resident-logical-not"
LOGICAL_BINARY_RULE = "rextio-numpy/resident-logical-binary"


def _reject_or_uncovered(site: ClaimSite) -> ClaimResult:
    """Keep unresolved sites fallback-only, but guide known bad operand types."""
    return not_covered_or_rejected(site)


def _claim_not(site: ClaimSite) -> ClaimResult:
    if len(site.operand_types) != 1 or site.receiver is not None or site.keywords or site.callables:
        return _reject_or_uncovered(site)
    (mask,) = site.operand_types
    if not is_bool_type(mask):
        return _reject_or_uncovered(site)
    return Claimed(rule_id=LOGICAL_NOT_RULE, result_type=mask)


def _claim_binary(site: ClaimSite) -> ClaimResult:
    if len(site.operand_types) != 2 or site.receiver is not None or site.keywords or site.callables:
        return _reject_or_uncovered(site)
    left, right = site.operand_types
    if not is_bool_type(left) or not is_bool_type(right):
        return _reject_or_uncovered(site)
    if left is None or right is None:
        return _reject_or_uncovered(site)
    left_rank = bool_rank(left)
    right_rank = bool_rank(right)
    if left_rank is None or right_rank is None:
        return _reject_or_uncovered(site)
    return Claimed(
        rule_id=LOGICAL_BINARY_RULE,
        result_type=bool_type_for(max(left_rank, right_rank)),
    )


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim only exact resident-mask logical calls."""
    if site.kind != "call":
        return None
    if site.target == LOGICAL_NOT_TARGET:
        return _claim_not(site)
    if site.target in LOGICAL_BINARY_TARGETS:
        return _claim_binary(site)
    return None


__all__ = [
    "LOGICAL_BINARY_RULE",
    "LOGICAL_BINARY_TARGETS",
    "LOGICAL_NOT_RULE",
    "LOGICAL_NOT_TARGET",
    "try_claim",
]
