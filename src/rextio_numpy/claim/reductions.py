"""Claim decisions for whole-array and literal-axis reductions.

Whole-array surface (no keywords):
  * ``numpy.sum`` on float64/int64 ranks 1–2
  * ``numpy.mean`` on float64 ranks 1–2

Literal-axis surface (exactly ``axis=<int literal>``):
  * ``numpy.sum`` / ``numpy.mean`` — same dtype matrix as whole-array
  * ``numpy.max`` / ``numpy.min`` — float64/int64 ranks 1–2; float32 rank 2 only

Bare ``max``/``min`` without ``axis=``, positional axis, ``axis=None``, tuple
axis, dynamic axis, extra kwargs, method forms, and ``amax``/``amin`` stay
unclaimed (``NotCovered`` / honest fallback). float32 sum/mean and int64 mean
remain RXTP-NUMPY-010 rejections.
"""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_numpy.diagnostics import (
    array_meta,
    is_array_type,
    not_covered_or_rejected,
    type_key_for,
)

_WHOLE_ARRAY_TARGETS = frozenset({"numpy.sum", "numpy.mean"})
_AXIS_TARGETS = frozenset({"numpy.sum", "numpy.mean", "numpy.max", "numpy.min"})
_ALL_TARGETS = _WHOLE_ARRAY_TARGETS | _AXIS_TARGETS

_WHOLE_ARRAY_RULE = "rextio-numpy/reduction-sum-mean"
_AXIS_RULE = "rextio-numpy/reduction-axis"

# float32 whole-array / axis sum/mean are intentionally unclaimed: sequential
# f32 accumulation diverges materially from NumPy pairwise summation.
# int64 mean is unclaimed: sequential i64→f64 cast-and-sum diverges from NumPy
# pairwise mean on large integers near the float64 mantissa boundary.
# float32 max/min rank-1 stays unclaimed: a core float scalar would lose
# numpy.float32 scalar semantics; rank-2 axis max/min return F32Arr1.
_SUM_DTYPES = frozenset({"f64", "i64"})
_MEAN_DTYPES = frozenset({"f64"})
_EXTREMA_DTYPES = frozenset({"f64", "i64", "f32"})


def normalize_axis(axis: int, rank: int) -> int | None:
    """Return the normalized non-negative axis, or None if out of range.

    Negative axes are resolved against ``rank`` at claim time (and again at
    lower time for helper identity). Matches NumPy's range rule for a single
    integer axis: ``-rank <= axis < rank``.
    """
    if not isinstance(axis, int) or isinstance(axis, bool):
        return None
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        return None
    return normalized


def _whole_array_result_type(target: str, dtype: str) -> str:
    """Return the core scalar result type for a whole-array reduction."""
    if target == "numpy.sum" and dtype == "i64":
        return "int"
    return "float"


def _axis_result_type(target: str, dtype: str, rank: int) -> str:
    """Return scalar or rank-1 plugin array type for a single-axis reduction."""
    if rank == 1:
        # Fully reduces to a core scalar (same as whole-array).
        if target in ("numpy.sum", "numpy.max", "numpy.min") and dtype == "i64":
            return "int"
        return "float"
    # Rank-2 single-axis → matching rank-1 plugin array.
    return type_key_for(dtype, 1)


def _dtype_allowed(target: str, dtype: str, rank: int, *, axis: bool) -> bool:
    """Report whether ``(target, dtype, rank)`` is on the certified surface."""
    if target == "numpy.sum":
        return dtype in _SUM_DTYPES
    if target == "numpy.mean":
        return dtype in _MEAN_DTYPES
    # max / min
    if dtype == "f32":
        # Rank-1 f32 extrema would surface as core float (losing float32 scalar).
        return axis and rank == 2
    return dtype in {"f64", "i64"}


def _axis_literal(site: ClaimSite) -> int | None:
    """Extract a single signed-int ``axis=`` literal, else None (unsupported form)."""
    if len(site.keywords) != 1:
        return None
    kw = site.keywords[0]
    if kw.name != "axis":
        return None
    lit = kw.literal
    if not lit.is_literal:
        return None
    value = lit.value
    # Literal None and int-tuples are offered by API 1.2 but not claimed here.
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Return a claim result for sum/mean/max/min, or None if not this lane."""
    if site.kind != "call" or site.target not in _ALL_TARGETS:
        return None
    operands = site.operand_types
    if len(operands) != 1:
        # Wrong arity (incl. positional axis) — leave to core RXT030.
        return NotCovered()
    (operand,) = operands

    if not site.keywords:
        # Whole-array path: sum/mean only. Bare max/min stay fallback.
        if site.target not in _WHOLE_ARRAY_TARGETS:
            return NotCovered()
        if not is_array_type(operand):
            return not_covered_or_rejected(site)
        assert operand is not None
        meta = array_meta(operand)
        assert meta is not None
        dtype, _rank = meta
        if not _dtype_allowed(site.target, dtype, _rank, axis=False):
            return not_covered_or_rejected(site)
        return Claimed(
            rule_id=_WHOLE_ARRAY_RULE,
            result_type=_whole_array_result_type(site.target, dtype),
        )

    # Axis path: exactly one named keyword ``axis=<int literal>``.
    if site.target not in _AXIS_TARGETS:
        return NotCovered()
    raw_axis = _axis_literal(site)
    if raw_axis is None:
        return NotCovered()
    if not is_array_type(operand):
        return not_covered_or_rejected(site)
    assert operand is not None
    meta = array_meta(operand)
    assert meta is not None
    dtype, rank = meta
    if normalize_axis(raw_axis, rank) is None:
        return NotCovered()
    if not _dtype_allowed(site.target, dtype, rank, axis=True):
        return not_covered_or_rejected(site)
    return Claimed(
        rule_id=_AXIS_RULE,
        result_type=_axis_result_type(site.target, dtype, rank),
    )
