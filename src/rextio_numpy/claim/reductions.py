"""Claim decisions for whole-array and literal-axis reductions.

Whole-array surface (no keywords):
  * ``numpy.sum`` on float64/int64 ranks 1–2
  * ``numpy.mean`` on float64 ranks 1–2
  * ``numpy.max`` / ``numpy.min`` on int64 ranks 1–2

Literal-axis surface (named ``axis=<int literal>`` or one positional literal):
  * ``numpy.sum`` / ``numpy.mean`` — same dtype matrix as whole-array
  * ``numpy.max`` / ``numpy.min`` — int64 ranks 1–2 only

Float ``max``/``min``, ``axis=None``, tuple axis, dynamic axis, extra
positional/keyword options, and ``amax``/``amin`` stay unclaimed
(``NotCovered`` / honest fallback). Certified ``ndarray`` method forms use
plugin API 1.3 receiver metadata and share the exact module-call matrix.
float32 sum/mean and int64 mean remain RXTP-NUMPY-010 rejections.
"""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_numpy.diagnostics import (
    array_meta,
    is_array_type,
    not_covered_or_rejected,
    type_key_for,
)

_WHOLE_ARRAY_TARGETS = frozenset(
    {"numpy.sum", "numpy.mean", "numpy.max", "numpy.min"}
)
_AXIS_TARGETS = frozenset({"numpy.sum", "numpy.mean", "numpy.max", "numpy.min"})
_ALL_TARGETS = _WHOLE_ARRAY_TARGETS | _AXIS_TARGETS

_WHOLE_ARRAY_RULE = "rextio-numpy/reduction-sum-mean"
_WHOLE_EXTREMA_RULE = "rextio-numpy/reduction-whole-i64-extrema"
_AXIS_RULE = "rextio-numpy/reduction-axis"

# float32 whole-array / axis sum/mean are intentionally unclaimed: sequential
# f32 accumulation diverges materially from NumPy pairwise summation.
# int64 mean is unclaimed: sequential i64→f64 cast-and-sum diverges from NumPy
# pairwise mean on large integers near the float64 mantissa boundary.
# Float max/min stay unclaimed: NumPy NaN payload/sign and signed-zero tie
# behavior differs across supported platform/SIMD profiles.
_SUM_DTYPES = frozenset({"f64", "i64"})
_MEAN_DTYPES = frozenset({"f64"})


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
    if target in {"numpy.sum", "numpy.max", "numpy.min"} and dtype == "i64":
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
    # Float extrema reduction details (NaN payload/sign and signed-zero ties)
    # vary across NumPy's supported platform/SIMD implementations. They cannot
    # be reproduced by one stable native helper, so retain only exact i64.
    return dtype == "i64"


def _axis_literal(site: ClaimSite, base_arity: int) -> int | None:
    """Extract one named or positional signed-int axis literal."""
    if len(site.operand_types) == base_arity and len(site.keywords) == 1:
        kw = site.keywords[0]
        if kw.name != "axis" or kw.arg_type != "int":
            return None
        lit = kw.literal
    elif len(site.operand_types) == base_arity + 1 and not site.keywords:
        axis_index = base_arity
        if site.operand_types[axis_index] != "int":
            return None
        if len(site.operand_literals) != len(site.operand_types):
            return None
        lit = site.operand_literals[axis_index]
    else:
        return None
    if not lit.is_literal:
        return None
    value = lit.value
    # Literal None and int-tuples are offered by API 1.2 but not claimed here.
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Return a claim result for sum/mean/max/min, or None if not this lane."""
    is_method = site.receiver is not None and site.target.rpartition(".")[2] in {
        "sum",
        "mean",
        "max",
        "min",
    }
    target = f"numpy.{site.target.rpartition('.')[2]}" if is_method else site.target
    if site.kind != "call" or target not in _ALL_TARGETS:
        return None
    operands = site.operand_types
    expected_arity = 0 if is_method else 1
    if len(operands) not in {expected_arity, expected_arity + 1}:
        # Wrong arity beyond the optional one positional axis.
        return NotCovered()
    operand = site.receiver.arg_type if is_method else operands[0]

    if len(operands) == expected_arity and not site.keywords:
        # Whole-array path. Floating extrema remain ordinary fallback because
        # signed-zero/NaN tie behavior varies across NumPy platform profiles.
        if not is_array_type(operand):
            return not_covered_or_rejected(site)
        assert operand is not None
        meta = array_meta(operand)
        assert meta is not None
        dtype, _rank = meta
        if target in {"numpy.max", "numpy.min"} and dtype != "i64":
            return NotCovered()
        if not _dtype_allowed(target, dtype, _rank, axis=False):
            return not_covered_or_rejected(site)
        return Claimed(
            rule_id=(
                _WHOLE_EXTREMA_RULE
                if target in {"numpy.max", "numpy.min"}
                else _WHOLE_ARRAY_RULE
            ),
            result_type=_whole_array_result_type(target, dtype),
        )

    # Axis path: one named or positional ``axis=<int literal>``.
    if target not in _AXIS_TARGETS:
        return NotCovered()
    raw_axis = _axis_literal(site, expected_arity)
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
    if target in {"numpy.max", "numpy.min"} and dtype != "i64":
        # Leave platform/SIMD-dependent float extrema on ordinary fallback;
        # this is an exclusion, not a type error in user code.
        return NotCovered()
    if not _dtype_allowed(target, dtype, rank, axis=True):
        return not_covered_or_rejected(site)
    return Claimed(
        rule_id=_AXIS_RULE,
        result_type=_axis_result_type(target, dtype, rank),
    )
