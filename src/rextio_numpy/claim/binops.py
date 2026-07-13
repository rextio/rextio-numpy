"""Claim decisions for elementwise binops (+, -, *, /)."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_numpy.diagnostics import F64_1D, not_covered_or_rejected

# Binary operator token -> the elementwise helper op name (shared with lower).
BINOP_NAMES = {"+": "add", "-": "sub", "*": "mul", "/": "div"}


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Return a claim result for elementwise binops, or None if not this lane."""
    if site.kind != "binop" or site.target not in BINOP_NAMES:
        return None
    operands = site.operand_types
    if operands in ((F64_1D, F64_1D), (F64_1D, "float"), ("float", F64_1D)):
        return Claimed(rule_id="rextio-numpy/elementwise-float64", result_type=F64_1D)
    if F64_1D not in operands:
        # No plugin-typed operand: not this plugin's business.
        return NotCovered()
    return not_covered_or_rejected(site)
