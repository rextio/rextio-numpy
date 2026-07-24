"""Claim router: dispatch analysis sites to feature-owned claim modules."""

from __future__ import annotations

from rextio.config.schema import RextioConfig
from rextio.plugins.api import ClaimResult, ClaimSite, NotCovered

from rextio_numpy.claim import binops, compare, fusion, linear, logical, reductions, unary, where

__all__ = ["claim"]


def claim(site: ClaimSite, config: RextioConfig) -> ClaimResult:
    """Decide, at analysis time, whether this plugin lowers the site.

    Deterministic by contract: the decision is a pure function of
    ``(site.kind, site.target, site.operand_types, site.keywords,
    site.operand_literals, site.expression)``. Covered targets with
    unresolved operands return :class:`NotCovered`; covered targets with
    known-but-unsupported operand types return :class:`Rejected` with
    RXTP-NUMPY-010 guidance; multi-op pure-array trees may claim as
    leaves-mode fusion (``rextio-numpy/elementwise-chain-fusion``) before the
    ordinary per-op elementwise path; unsupported call shapes return
    :class:`NotCovered`; everything else is :class:`NotCovered`.
    """
    del config
    # Fusion must run before ordinary elementwise so eligible multi-op roots
    # claim with operand_mode="leaves" and subsume descendant binops.
    for handler in (
        compare.try_claim,
        logical.try_claim,
        where.try_claim,
        linear.try_claim,
        reductions.try_claim,
        unary.try_claim,
        fusion.try_claim,
        binops.try_claim,
    ):
        result = handler(site)
        if result is not None:
            return result
    return NotCovered()
