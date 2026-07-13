"""Claim router: dispatch analysis sites to feature-owned claim modules."""

from __future__ import annotations

from rextio.config.schema import RextioConfig
from rextio.plugins.api import ClaimResult, ClaimSite, NotCovered

from rextio_numpy.claim import binops, linear, reductions

__all__ = ["claim"]


def claim(site: ClaimSite, config: RextioConfig) -> ClaimResult:
    """Decide, at analysis time, whether this plugin lowers the site.

    Deterministic by contract: the decision is a pure function of
    ``(site.kind, site.target, site.operand_types)``. Covered targets with
    unresolved operands return :class:`NotCovered`; covered targets with
    known-but-unsupported operand types return :class:`Rejected` with
    RXTP-NUMPY-010 guidance; everything else is :class:`NotCovered`.
    """
    del config
    for handler in (linear.try_claim, reductions.try_claim, binops.try_claim):
        result = handler(site)
        if result is not None:
            return result
    return NotCovered()
