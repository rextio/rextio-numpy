"""Result-equivalence rules for fallback vs native verification.

Array comparison is exact and pointwise, with NaN == NaN.
Scalar comparison (dot / reductions) is tolerant: float()-coerced, NaN == NaN,
``math.isclose`` at ``rel_tol=1e-12`` / ``abs_tol=1e-12`` to absorb documented
summation-order divergence between native and NumPy.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

# Documented tolerant threshold for dot/summation-order divergence.
SCALAR_REL_TOL = 1e-12
SCALAR_ABS_TOL = 1e-12


def array_equals(left: Any, right: Any) -> bool:
    """Exact pointwise array equality (NaN == NaN) with dtype equality.

    Both sides must be array-like with matching shape **and** dtype. A wrong
    native dtype (e.g. float32 vs float64 with equal values after cast) must
    not count as matched. Falls back to ``==`` for non-array values so
    mixed-type mismatches return False.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is required for real runs
        return left == right

    left_arr = np.asarray(left) if not isinstance(left, np.ndarray) else left
    right_arr = np.asarray(right) if not isinstance(right, np.ndarray) else right
    if left_arr.shape != right_arr.shape:
        return False
    if left_arr.dtype != right_arr.dtype:
        return False
    if left_arr.dtype.kind == "f" or right_arr.dtype.kind == "f":
        return bool(np.array_equal(left_arr, right_arr, equal_nan=True))
    return bool(np.array_equal(left_arr, right_arr))


def scalar_close(left: Any, right: Any) -> bool:
    """Tolerant scalar equality for dot/summation-order divergence.

    Coerces both sides via ``float()`` so ``numpy.float64`` and plain
    ``float`` compare equal. NaN matches NaN. Uses
    ``math.isclose(..., rel_tol=1e-12, abs_tol=1e-12)``.
    """
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        left_f, right_f = float(left), float(right)
        if math.isnan(left_f) and math.isnan(right_f):
            return True
        return math.isclose(left_f, right_f, rel_tol=SCALAR_REL_TOL, abs_tol=SCALAR_ABS_TOL)
    return left == right


def results_equivalent(left: Any, right: Any, *, kind: str) -> bool:
    """Dispatch to the appropriate comparator for *kind*.

    *kind* is ``"array"`` or ``"scalar"``. Unknown kinds raise ``ValueError``.
    """
    if kind == "array":
        return array_equals(left, right)
    if kind == "scalar":
        return scalar_close(left, right)
    raise ValueError(f"unknown comparison kind: {kind!r}")
