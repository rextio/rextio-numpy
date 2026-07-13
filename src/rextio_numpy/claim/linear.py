"""Claim decisions for linear algebra sites (numpy.dot)."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_numpy.diagnostics import array_meta, is_array_type, not_covered_or_rejected

_DOT_RULE = "rextio-numpy/dot-float64"

# Same-dtype rank-1 only; 2-D and matmul stay forbidden for the benchmark gate.
# float32 is intentionally excluded: sequential f32 accumulation diverges from
# NumPy on long/mixed-magnitude inputs, and claim metadata cannot enforce a
# size bound or runtime fallback (see reduction claim lane for the same gate).
_DOT_RESULT: dict[str, str] = {
    "f64": "float",
    "i64": "int",
}


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
    left, right = operands
    if not is_array_type(left) and not is_array_type(right):
        # No plugin-typed operand at all.
        if left is None or right is None:
            return NotCovered()
        return NotCovered()
    if left is None or right is None:
        return not_covered_or_rejected(site)
    if not is_array_type(left) or not is_array_type(right):
        return not_covered_or_rejected(site)
    assert left is not None and right is not None
    left_meta = array_meta(left)
    right_meta = array_meta(right)
    assert left_meta is not None and right_meta is not None
    left_dtype, left_rank = left_meta
    right_dtype, right_rank = right_meta
    if left_dtype != right_dtype or left_rank != 1 or right_rank != 1:
        return not_covered_or_rejected(site)
    if left_dtype not in _DOT_RESULT:
        # Known array type outside the verified 1-D dot surface (e.g. f32).
        return not_covered_or_rejected(site)
    return Claimed(rule_id=_DOT_RULE, result_type=_DOT_RESULT[left_dtype])
