"""Claim decisions for whole-array reductions (numpy.sum / numpy.mean)."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_numpy.diagnostics import F64_1D, not_covered_or_rejected

_REDUCTION_TARGETS = ("numpy.sum", "numpy.mean")


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Return a claim result for sum/mean, or None if not this lane."""
    if site.kind != "call" or site.target not in _REDUCTION_TARGETS:
        return None
    operands = site.operand_types
    if len(operands) != 1:
        return NotCovered()
    if operands == (F64_1D,):
        return Claimed(rule_id="rextio-numpy/reduction-sum-mean", result_type="float")
    return not_covered_or_rejected(site)
