"""Lower router: dispatch claimed sites to feature-owned lower modules."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy.lower import binops, compare, fusion, linear, reductions, unary, where

__all__ = ["lower"]


def lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
    """Emit the Rust expression for a previously claimed site.

    Every emitted expression is fallible (``pyo3::PyResult``) and ends
    with ``?``; the helper ``fn`` items travel in ``helpers`` and are
    deduplicated by exact text in core codegen. Fusion claims
    (``rextio-numpy/elementwise-chain-fusion``) consume ``ctx.leaf_operands``
    and the frozen ``ClaimExpr`` tree.
    """
    for handler in (
        compare.try_lower,
        where.try_lower,
        linear.try_lower,
        reductions.try_lower,
        unary.try_lower,
        fusion.try_lower,
        binops.try_lower,
    ):
        result = handler(claimed, ctx)
        if result is not None:
            return result
    raise ValueError(f"rextio-numpy cannot lower unclaimed site: {claimed.kind} {claimed.target!r}")
