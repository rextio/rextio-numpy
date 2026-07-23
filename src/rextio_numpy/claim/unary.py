"""Claim decisions for exact one-argument NumPy unary module calls."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_numpy.diagnostics import is_array_type, not_covered_or_rejected

_TARGETS = frozenset({"numpy.negative", "numpy.absolute", "numpy.abs", "numpy.square"})
_RULE = "rextio-numpy/unary-module"


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim a certified exact unary module call, or return None for other lanes."""
    if site.kind != "call" or site.target not in _TARGETS or site.receiver is not None:
        return None
    if len(site.operand_types) != 1 or site.keywords:
        return NotCovered()
    (operand,) = site.operand_types
    if not is_array_type(operand):
        return not_covered_or_rejected(site)
    assert operand is not None
    return Claimed(rule_id=_RULE, result_type=operand)
