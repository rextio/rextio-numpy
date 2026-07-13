"""Claim decisions for linear algebra sites (numpy.dot)."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_numpy.diagnostics import F64_1D, not_covered_or_rejected


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Return a claim result for ``numpy.dot``, or None if not this lane."""
    if site.kind != "call" or site.target != "numpy.dot":
        return None
    operands = site.operand_types
    if len(operands) != 2:
        # Wrong arity is an unsupported call SHAPE, not an operand-type
        # problem; hand it back so core's RXT030 names the real cause
        # instead of the dtype-oriented RXTP-NUMPY-010 (council round 8).
        return NotCovered()
    if operands == (F64_1D, F64_1D):
        return Claimed(rule_id="rextio-numpy/dot-float64", result_type="float")
    return not_covered_or_rejected(site)
