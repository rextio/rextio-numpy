"""Claim decisions for whole-array reductions (numpy.sum / numpy.mean)."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_numpy.diagnostics import array_meta, is_array_type, not_covered_or_rejected

_REDUCTION_TARGETS = ("numpy.sum", "numpy.mean")
_REDUCTION_RULE = "rextio-numpy/reduction-sum-mean"

# float32 whole-array sum/mean are intentionally unclaimed: sequential f32
# accumulation (ndarray) diverges materially from NumPy pairwise summation on
# long/mixed-magnitude inputs, and the plugin claim API has no runtime length
# gate or fallback hook. Elementwise f32 remains claimed elsewhere.
#
# int64 mean is also unclaimed: sequential i64→f64 cast-and-sum diverges from
# NumPy pairwise mean on large integers near the float64 mantissa boundary
# (e.g. tile([2**53, 1, -2**53], n) → NumPy ~0.11 vs sequential 0.0). int64
# sum remains claimed (exact wraparound). Elementwise i64 remains claimed.
_REDUCTION_SUM_DTYPES = frozenset({"f64", "i64"})
_REDUCTION_MEAN_DTYPES = frozenset({"f64"})


def _result_type(target: str, dtype: str) -> str:
    """Return the core scalar result type for a whole-array reduction.

    Matches NumPy 2.4 practical dtypes: int64 sum stays int; float64
    reductions surface as core ``float`` (native returns a builtin float).
    int64 mean is claim-rejected (not lowered).
    """
    if target == "numpy.sum" and dtype == "i64":
        return "int"
    return "float"


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Return a claim result for sum/mean, or None if not this lane."""
    if site.kind != "call" or site.target not in _REDUCTION_TARGETS:
        return None
    operands = site.operand_types
    if len(operands) != 1:
        return NotCovered()
    (operand,) = operands
    if not is_array_type(operand):
        return not_covered_or_rejected(site)
    assert operand is not None
    meta = array_meta(operand)
    assert meta is not None
    dtype, _rank = meta
    if site.target == "numpy.sum":
        allowed = _REDUCTION_SUM_DTYPES
    else:
        allowed = _REDUCTION_MEAN_DTYPES
    if dtype not in allowed:
        # Known array type outside the verified reduction surface
        # (e.g. f32 sum/mean, i64 mean).
        return not_covered_or_rejected(site)
    return Claimed(
        rule_id=_REDUCTION_RULE,
        result_type=_result_type(site.target, dtype),
    )
