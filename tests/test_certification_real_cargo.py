"""The real vertical slice: certification of the lowering surface under cargo.

Builds a fixture project once per module through the core certification kit
(``rextio.plugins.testing``) with the REAL entry-point-discoverable plugin —
no monkeypatching — then asserts native<->fallback equivalence for every
lowered kernel, including exception equivalence and hypothesis-driven inputs.

Wave 1 Lane A expands the claimed surface to float64/float32/int64 × ranks
1–2 elementwise, float64/int64 sum, float64 mean, and 1-D float64/int64 dot.
float32 whole-array sum/mean/dot and int64 mean are intentionally **not**
claimed: sequential f32 accumulation diverges from NumPy pairwise summation
on long/mixed-magnitude inputs; sequential i64→f64 cast-and-sum diverges from
NumPy pairwise mean on large integers near 2**53; the plugin claim API has no
runtime length gate. Those kernels must stay on the Python fallback (not
natively served). F64 rank-1 kernels and f32/i64 elementwise must remain green.

Comparator notes (empirically verified): the native leg returns plain Python
``float``/``int`` for scalar results while the fallback (NumPy) leg returns
numpy scalars; both custom comparators coerce before comparing. Elementwise
ops are pointwise IEEE (or wrapping int64), so arrays must match exactly.
float64 dot/sum/mean use ``scalar_close`` at 1e-12.
"""

from __future__ import annotations

import copy
import math
import numbers
import shutil
import warnings
from typing import Any, Callable

import pytest

np = pytest.importorskip("numpy")
hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings, strategies as st  # noqa: E402
from hypothesis.extra import numpy as npst  # noqa: E402

from rextio.plugins.testing import (  # noqa: E402
    CertificationError,
    CertifiedProject,
    build_certification_project,
    default_equals,
)

pytestmark = pytest.mark.skipif(
    shutil.which("cargo") is None, reason="real-cargo certification requires cargo on PATH"
)

KERNELS = """
import numpy as np
from rextio_numpy.types import (
    F32Arr1,
    F32Arr2,
    F64Arr1,
    F64Arr2,
    I64Arr1,
    I64Arr2,
)


def dot(a: F64Arr1, b: F64Arr1) -> float:
    return np.dot(a, b)


def add(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return a + b


def sub(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return a - b


def mul(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return a * b


def div(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return a / b


def ufunc_add_f64_2d(a: F64Arr2, b: F64Arr2) -> F64Arr2:
    return np.add(a, b)


def ufunc_sub_f64_2d_1d(a: F64Arr2, b: F64Arr1) -> F64Arr2:
    return np.subtract(a, b)


def ufunc_mul_f32_scalar(a: F32Arr1, factor: float) -> F32Arr1:
    return np.multiply(a, factor)


def ufunc_div_i64_1d(a: I64Arr1, b: I64Arr1) -> F64Arr1:
    return np.divide(a, b)


def scale(a: F64Arr1, factor: float) -> F64Arr1:
    return a * factor


def rsub(offset: float, a: F64Arr1) -> F64Arr1:
    return offset - a


def identity(a: F64Arr1) -> F64Arr1:
    return a


def peek(a: F64Arr1) -> float:
    return 0.0


def total(a: F64Arr1) -> float:
    return np.sum(a)


def average(a: F64Arr1) -> float:
    return np.mean(a)


def method_dot(a: F64Arr1, b: F64Arr1) -> float:
    return a.dot(b)


def method_total(a: F64Arr1) -> float:
    return a.sum()


def method_average_axis(a: F64Arr2) -> F64Arr1:
    return a.mean(axis=1)


def method_max_axis(a: F32Arr2) -> F32Arr1:
    return a.max(axis=0)


def unary_negative_f64(a: F64Arr1) -> F64Arr1:
    return np.negative(a)


def unary_absolute_f32(a: F32Arr2) -> F32Arr2:
    return np.absolute(a)


def unary_abs_i64(a: I64Arr1) -> I64Arr1:
    return np.abs(a)


def unary_square_i64(a: I64Arr2) -> I64Arr2:
    return np.square(a)


def accumulate(a: F64Arr1, b: F64Arr1, n: int) -> F64Arr1:
    c = a + b
    for i in range(n):
        c = c + b
    return c


# --- Wave 1 post-integration surface (requires plugin type_vocabulary wiring) ---

def add_f64_2d(a: F64Arr2, b: F64Arr2) -> F64Arr2:
    return a + b


def add_f64_1_2(a: F64Arr1, b: F64Arr2) -> F64Arr2:
    return a + b


def add_f64_2_1(a: F64Arr2, b: F64Arr1) -> F64Arr2:
    return a + b


def scale_f64_2d(a: F64Arr2, factor: float) -> F64Arr2:
    return a * factor


def rsub_f64_2d(offset: float, a: F64Arr2) -> F64Arr2:
    return offset - a


def total_f64_2d(a: F64Arr2) -> float:
    return np.sum(a)


def average_f64_2d(a: F64Arr2) -> float:
    return np.mean(a)


def add_f32_1d(a: F32Arr1, b: F32Arr1) -> F32Arr1:
    return a + b


def sub_f32_1d(a: F32Arr1, b: F32Arr1) -> F32Arr1:
    return a - b


def mul_f32_1d(a: F32Arr1, b: F32Arr1) -> F32Arr1:
    return a * b


def div_f32_1d(a: F32Arr1, b: F32Arr1) -> F32Arr1:
    return a / b


def scale_f32_1d(a: F32Arr1, factor: float) -> F32Arr1:
    return a * factor


def rsub_f32_1d(offset: float, a: F32Arr1) -> F32Arr1:
    return offset - a


def add_f32_2d(a: F32Arr2, b: F32Arr2) -> F32Arr2:
    return a + b


def add_f32_1_2(a: F32Arr1, b: F32Arr2) -> F32Arr2:
    return a + b


def total_f32_1d(a: F32Arr1) -> float:
    return np.sum(a)


def average_f32_1d(a: F32Arr1) -> float:
    return np.mean(a)


def total_f32_2d(a: F32Arr2) -> float:
    return np.sum(a)


def average_f32_2d(a: F32Arr2) -> float:
    return np.mean(a)


def dot_f32(a: F32Arr1, b: F32Arr1) -> float:
    return np.dot(a, b)


def add_i64_1d(a: I64Arr1, b: I64Arr1) -> I64Arr1:
    return a + b


def sub_i64_1d(a: I64Arr1, b: I64Arr1) -> I64Arr1:
    return a - b


def mul_i64_1d(a: I64Arr1, b: I64Arr1) -> I64Arr1:
    return a * b


def div_i64_1d(a: I64Arr1, b: I64Arr1) -> F64Arr1:
    return a / b


def scale_i64_1d(a: I64Arr1, factor: int) -> I64Arr1:
    return a * factor


def rsub_i64_1d(offset: int, a: I64Arr1) -> I64Arr1:
    return offset - a


def add_i64_2d(a: I64Arr2, b: I64Arr2) -> I64Arr2:
    return a + b


def add_i64_1_2(a: I64Arr1, b: I64Arr2) -> I64Arr2:
    return a + b


def div_i64_2d(a: I64Arr2, b: I64Arr2) -> F64Arr2:
    return a / b


def total_i64_1d(a: I64Arr1) -> int:
    return np.sum(a)


def average_i64_1d(a: I64Arr1) -> float:
    return np.mean(a)


def total_i64_2d(a: I64Arr2) -> int:
    return np.sum(a)


def whole_max_i64_1d(a: I64Arr1) -> int:
    return np.max(a)


def whole_min_i64_2d(a: I64Arr2) -> int:
    return np.min(a)


def method_whole_max_i64_2d(a: I64Arr2) -> int:
    return a.max()


def average_i64_2d(a: I64Arr2) -> float:
    return np.mean(a)


def dot_i64(a: I64Arr1, b: I64Arr1) -> int:
    return np.dot(a, b)


# --- Wave 2 literal-axis reduction surface ---

def sum_f64_1d_axis0(a: F64Arr1) -> float:
    return np.sum(a, axis=0)


def sum_f64_1d_axis_neg1(a: F64Arr1) -> float:
    return np.sum(a, axis=-1)


def mean_f64_1d_axis0(a: F64Arr1) -> float:
    return np.mean(a, axis=0)


def max_f64_1d_axis0(a: F64Arr1) -> float:
    return np.max(a, axis=0)


def min_f64_1d_axis0(a: F64Arr1) -> float:
    return np.min(a, axis=0)


def sum_f64_2d_axis0(a: F64Arr2) -> F64Arr1:
    return np.sum(a, axis=0)


def sum_f64_2d_axis1(a: F64Arr2) -> F64Arr1:
    return np.sum(a, axis=1)


def sum_f64_2d_axis_neg1(a: F64Arr2) -> F64Arr1:
    return np.sum(a, axis=-1)


def sum_f64_2d_axis_neg2(a: F64Arr2) -> F64Arr1:
    return np.sum(a, axis=-2)


def mean_f64_2d_axis0(a: F64Arr2) -> F64Arr1:
    return np.mean(a, axis=0)


def mean_f64_2d_axis1(a: F64Arr2) -> F64Arr1:
    return np.mean(a, axis=1)


def max_f64_2d_axis0(a: F64Arr2) -> F64Arr1:
    return np.max(a, axis=0)


def max_f64_2d_axis1(a: F64Arr2) -> F64Arr1:
    return np.max(a, axis=1)


def min_f64_2d_axis0(a: F64Arr2) -> F64Arr1:
    return np.min(a, axis=0)


def min_f64_2d_axis1(a: F64Arr2) -> F64Arr1:
    return np.min(a, axis=1)


def sum_i64_1d_axis0(a: I64Arr1) -> int:
    return np.sum(a, axis=0)


def sum_i64_2d_axis0(a: I64Arr2) -> I64Arr1:
    return np.sum(a, axis=0)


def sum_i64_2d_axis1(a: I64Arr2) -> I64Arr1:
    return np.sum(a, axis=1)


def max_i64_2d_axis0(a: I64Arr2) -> I64Arr1:
    return np.max(a, axis=0)


def min_i64_2d_axis1(a: I64Arr2) -> I64Arr1:
    return np.min(a, axis=1)


def sum_f64_2d_pos_axis0(a: F64Arr2) -> F64Arr1:
    return np.sum(a, 0)


def mean_f64_1d_pos_neg1(a: F64Arr1) -> float:
    return np.mean(a, -1)


def max_i64_1d_pos_axis0(a: I64Arr1) -> int:
    return np.max(a, 0)


def method_min_i64_2d_pos_axis1(a: I64Arr2) -> I64Arr1:
    return a.min(1)


def max_f32_2d_axis0(a: F32Arr2) -> F32Arr1:
    return np.max(a, axis=0)


def min_f32_2d_axis1(a: F32Arr2) -> F32Arr1:
    return np.min(a, axis=1)


# Intentionally unclaimed Wave-2 cells (must stay fallback).
def bare_max_f64(a: F64Arr1) -> float:
    return np.max(a)


def bare_min_f64(a: F64Arr1) -> float:
    return np.min(a)


def max_f32_1d_axis0(a: F32Arr1) -> float:
    return np.max(a, axis=0)


def sum_f32_2d_axis0(a: F32Arr2) -> F32Arr1:
    return np.sum(a, axis=0)


def mean_i64_2d_axis0(a: I64Arr2) -> F64Arr1:
    return np.mean(a, axis=0)


# --- Wave 2 elementwise chain fusion surface ---

def fuse_multi_op(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return (a + b) * (a - b)


def fuse_chain_add(a: F64Arr1, b: F64Arr1, c: F64Arr1) -> F64Arr1:
    return a + b + c


def fuse_noncomm(a: F64Arr1, b: F64Arr1, c: F64Arr1) -> F64Arr1:
    return (a - b) / (c + a)


def fuse_f32(a: F32Arr1, b: F32Arr1) -> F32Arr1:
    return (a + b) * (a - b)


def fuse_i64(a: I64Arr1, b: I64Arr1) -> I64Arr1:
    return (a + b) * (a - b)


def fuse_mixed_rank(a: F64Arr1, b: F64Arr2) -> F64Arr2:
    return (a + b) * (a - b)


def fuse_rank2(a: F64Arr2, b: F64Arr2) -> F64Arr2:
    return (a + b) * (a - b)


def fuse_balanced(a: F64Arr1, b: F64Arr1, c: F64Arr1, d: F64Arr1) -> F64Arr1:
    return (a + b) * (c - d)


def fuse_max_bound_f64(
    a0: F64Arr1,
    a1: F64Arr1,
    a2: F64Arr1,
    a3: F64Arr1,
    a4: F64Arr1,
    a5: F64Arr1,
    a6: F64Arr1,
    a7: F64Arr1,
    a8: F64Arr1,
) -> F64Arr1:
    # Certified maximum: 8 binops / 9 name-leaf occurrences (no Zip path).
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8


def fuse_max_bound_i64(
    a0: I64Arr1,
    a1: I64Arr1,
    a2: I64Arr1,
    a3: I64Arr1,
    a4: I64Arr1,
    a5: I64Arr1,
    a6: I64Arr1,
    a7: I64Arr1,
    a8: I64Arr1,
) -> I64Arr1:
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8


# --- API 1.5 resident comparison -> three-argument where surface ---

def where_eq_f64_11(
    left: F64Arr1,
    right: F64Arr1,
    yes: F64Arr1,
    no: F64Arr1,
) -> F64Arr1:
    return np.where(left == right, yes, no)


def where_ne_f64_12(
    left: F64Arr1,
    right: F64Arr2,
    yes: F64Arr2,
    no: F64Arr1,
) -> F64Arr2:
    return np.where(left != right, yes, no)


def where_lt_f64_21(
    left: F64Arr2,
    right: F64Arr1,
    yes: F64Arr1,
    no: F64Arr2,
) -> F64Arr2:
    return np.where(left < right, yes, no)


def where_le_f64_22(
    left: F64Arr2,
    right: F64Arr2,
    yes: F64Arr2,
    no: F64Arr2,
) -> F64Arr2:
    return np.where(left <= right, yes, no)


def where_gt_f32_scalar(
    values: F32Arr1,
    threshold: float,
    yes: F32Arr1,
    no: F32Arr1,
) -> F32Arr1:
    return np.where(values > threshold, yes, no)


def where_scalar_ge_f32(
    threshold: float,
    values: F32Arr2,
    yes: F32Arr2,
    no: F32Arr2,
) -> F32Arr2:
    return np.where(threshold >= values, yes, no)


def where_f32_array_scalar(values: F32Arr1, no: float) -> F32Arr1:
    return np.where(values >= 0.0, values, no)


def where_f32_scalar_array(yes: float, values: F32Arr1) -> F32Arr1:
    return np.where(values >= 0.0, yes, values)


def where_f64_array_scalar(values: F64Arr1, no: float) -> F64Arr1:
    return np.where(values >= 0.0, values, no)


def where_eq_i64_scalar(
    values: I64Arr1,
    target: int,
    yes: I64Arr1,
    no: I64Arr1,
) -> I64Arr1:
    return np.where(values == target, yes, no)


# --- API 1.5 resident bool composition/reduction surface ---

def logical_where_not_f64(values: F64Arr1, yes: F64Arr1, no: F64Arr1) -> F64Arr1:
    return np.where(np.logical_not(values > 0.0), yes, no)


def logical_where_and_f64_12(
    left: F64Arr1,
    right: F64Arr2,
    yes: F64Arr2,
    no: F64Arr1,
) -> F64Arr2:
    return np.where(np.logical_and(left > 0.0, right < 0.0), yes, no)


def logical_where_or_f64_21(
    left: F64Arr2,
    right: F64Arr1,
    yes: F64Arr1,
    no: F64Arr2,
) -> F64Arr2:
    return np.where(np.logical_or(left > 0.0, right < 0.0), yes, no)


def all_positive_f64(values: F64Arr1) -> bool:
    return np.all(values > 0.0)


def any_positive_f64(values: F64Arr1) -> bool:
    return np.any(values > 0.0)


def any_positive_branch_f64(values: F64Arr1, fallback: F64Arr1) -> F64Arr1:
    if np.any(values > 0.0):
        return values + 0.0
    return fallback + 0.0
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> CertifiedProject:
    root = tmp_path_factory.mktemp("np_certification")
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-numpy"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "np_app"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNELS, encoding="utf-8")
    return build_certification_project(root)


def checker(project: CertifiedProject, name: str, equals=None, args_equals=None, copy_args=None):
    return project.equivalence_checker(
        f"np_app.kernels.{name}", equals=equals, args_equals=args_equals, copy_args=copy_args
    )


def array_equals(left: object, right: object) -> bool:
    """Exact pointwise equality for arrays (NaN == NaN); kit default otherwise."""
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right, equal_nan=True))
    return default_equals(left, right)


# Wave-0 float64 certification contract (unchanged).
SCALAR_REL_TOL = 1e-12
SCALAR_ABS_TOL = 1e-12

# Adversarial float32 reduction/dot pattern: length 30000 of [1e8, 1, -1e8].
# NumPy pairwise sum ≈ 224; sequential f32 accumulation → 0. Used to prove
# why f32 sum/mean/dot must not be natively claimed without a runtime gate.
F32_ADVERSARIAL_LEN = 30_000

# Adversarial int64 mean pattern: tile([2**53, 1, -2**53], 10000) as int64.
# NumPy pairwise mean ≈ 0.1113; sequential i64→f64 cast-and-sum mean → 0.0.
# Proves why int64 mean must not be natively claimed without a runtime gate.
I64_MEAN_ADVERSARIAL_REPEATS = 10_000


def scalar_close(left: object, right: object) -> bool:
    """Tolerant scalar equality: float()-coerced, NaN == NaN, isclose at 1e-12.

    The native leg returns plain ``float`` while the fallback returns
    ``numpy.float64`` — a type-strict comparator would flag every call.
    This is the float64 Wave-0/Wave-1 contract only (int64 mean is unclaimed).
    """
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        left_f, right_f = float(left), float(right)
        if math.isnan(left_f) and math.isnan(right_f):
            return True
        return math.isclose(left_f, right_f, rel_tol=SCALAR_REL_TOL, abs_tol=SCALAR_ABS_TOL)
    return default_equals(left, right)


def scalar_int_equal(left: object, right: object) -> bool:
    """Exact integer scalar equality (numpy.int64 vs builtin int)."""
    if isinstance(left, numbers.Integral) and isinstance(right, numbers.Integral):
        return int(left) == int(right)
    return default_equals(left, right)


def scalar_bool_equal(left: object, right: object) -> bool:
    """Compare Python ``bool`` native output to NumPy's ``bool_`` by value."""
    return isinstance(left, (bool, np.bool_)) and isinstance(right, (bool, np.bool_)) and bool(left) is bool(right)


def _sequential_f32_sum(values: np.ndarray) -> float:
    """Naive left-to-right f32 accumulation (models unclaimed native f32 sum)."""
    acc = np.float32(0.0)
    for x in np.asarray(values, dtype=np.float32).ravel(order="C"):
        acc = np.float32(acc + x)
    return float(acc)


def _f32_adversarial_array(n: int = F32_ADVERSARIAL_LEN) -> np.ndarray:
    """Build the long cancellation pattern that breaks sequential f32 sum."""
    if n % 3 != 0:
        raise ValueError(f"adversarial length must be a multiple of 3, got {n}")
    pat = np.array([1e8, 1.0, -1e8], dtype=np.float32)
    return np.tile(pat, n // 3)


def _i64_mean_adversarial_array(repeats: int = I64_MEAN_ADVERSARIAL_REPEATS) -> np.ndarray:
    """Build the large-integer cancellation pattern that breaks sequential i64 mean."""
    pat = np.array([2**53, 1, -(2**53)], dtype=np.int64)
    return np.tile(pat, repeats)


def _sequential_i64_mean_as_f64(values: np.ndarray) -> float:
    """Naive left-to-right i64→f64 cast-and-sum mean (models unclaimed native path)."""
    arr = np.asarray(values, dtype=np.int64).ravel(order="C")
    if arr.size == 0:
        return float("nan")
    acc = 0.0
    for x in arr:
        acc += float(x)
    return acc / float(arr.size)


ARRAY = np.array([1.0, -2.5, 3.25, 0.5])
OTHER = np.array([4.0, 0.125, -6.5, 2.0])


# ---------------------------------------------------------------------------
# Wave-0 F64 rank-1 surface (must stay green without plugin.py integration)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["add", "sub", "mul", "div"])
def test_elementwise_array_array_exact(project: CertifiedProject, name: str) -> None:
    # args_equals (council T12): neither leg may mutate its argument arrays.
    check = checker(project, name, equals=array_equals, args_equals=array_equals)
    result = check(ARRAY, OTHER)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_exact_ufunc_call_aliases_are_natively_served(project: CertifiedProject) -> None:
    """Four exact call spellings reuse the certified operator matrix."""
    add = _require_native(project, "ufunc_add_f64_2d")
    sub = _require_native(project, "ufunc_sub_f64_2d_1d")
    mul = _require_native(project, "ufunc_mul_f32_scalar")
    div = _require_native(project, "ufunc_div_i64_1d")

    matrix = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float64)
    other = np.array([[0.5, 4.0, -1.0], [2.0, -3.0, 8.0]], dtype=np.float64)
    vector = np.array([0.25, -0.5, 2.0], dtype=np.float64)
    f32 = np.array([1.5, -2.0, 0.0], dtype=np.float32)
    left_i64 = np.array([1, -2, 0], dtype=np.int64)
    right_i64 = np.array([2, 4, 0], dtype=np.int64)

    np.testing.assert_array_equal(add(matrix, other), np.add(matrix, other))
    np.testing.assert_array_equal(sub(matrix, vector), np.subtract(matrix, vector))
    np.testing.assert_array_equal(mul(f32, 2.5), np.multiply(f32, 2.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = div(left_i64, right_i64)
        expected = np.divide(left_i64, right_i64)
    assert np.array_equal(result, expected, equal_nan=True)
    assert result.dtype == np.float64


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_elementwise_division_by_zero_is_ieee(project: CertifiedProject) -> None:
    check = checker(project, "div", equals=array_equals, args_equals=array_equals)
    result = check(np.array([1.0, -1.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert np.isposinf(result[0]) and np.isneginf(result[1]) and np.isnan(result[2])


def test_array_scalar_and_scalar_array_forms(project: CertifiedProject) -> None:
    scale = checker(project, "scale", equals=array_equals, args_equals=array_equals)
    rsub = checker(project, "rsub", equals=array_equals, args_equals=array_equals)
    scaled = scale(ARRAY, 2.5)
    np.testing.assert_array_equal(scaled, ARRAY * 2.5)
    shifted = rsub(10.0, ARRAY)
    np.testing.assert_array_equal(shifted, 10.0 - ARRAY)


def test_dot_sum_mean_close(project: CertifiedProject) -> None:
    dot = checker(project, "dot", equals=scalar_close, args_equals=array_equals)
    total = checker(project, "total", equals=scalar_close, args_equals=array_equals)
    average = checker(project, "average", equals=scalar_close, args_equals=array_equals)
    assert float(dot(ARRAY, OTHER)) == pytest.approx(float(np.dot(ARRAY, OTHER)), rel=1e-12)
    assert float(total(ARRAY)) == pytest.approx(float(np.sum(ARRAY)), rel=1e-12)
    assert float(average(ARRAY)) == pytest.approx(float(np.mean(ARRAY)), rel=1e-12)


def test_method_parity_is_natively_served(project: CertifiedProject) -> None:
    """API-1.3 receiver paths use the same certified helpers as module calls."""
    dot = _require_native_scalar(project, "method_dot", scalar_close)
    total = _require_native_scalar(project, "method_total", scalar_close)
    mean_axis = _require_native(project, "method_average_axis")
    a = np.array([1.0, -2.5, 3.25])
    b = np.array([0.5, 4.0, -1.0])
    matrix = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
    assert float(dot(a, b)) == pytest.approx(float(a.dot(b)), rel=SCALAR_REL_TOL)
    assert float(total(a)) == pytest.approx(float(a.sum()), rel=SCALAR_REL_TOL)
    np.testing.assert_allclose(mean_axis(matrix), matrix.mean(axis=1))


def test_unary_module_calls_are_natively_served(project: CertifiedProject) -> None:
    negative = _require_native(project, "unary_negative_f64")
    absolute = _require_native(project, "unary_absolute_f32")
    abs_i64 = _require_native(project, "unary_abs_i64")
    square_i64 = _require_native(project, "unary_square_i64")
    f64 = np.array([-0.0, np.inf, -np.inf, np.nan], dtype=np.float64)
    f32 = np.array([[-0.0, -3.5], [np.inf, np.nan]], dtype=np.float32)
    i64 = np.array([np.iinfo(np.int64).min, -3, 0, 4], dtype=np.int64)
    i64_2d = np.array([[np.iinfo(np.int64).max, 2], [-3, 4]], dtype=np.int64)
    negative_result = negative(f64)
    absolute_result = absolute(f32)
    assert array_equals(negative_result, np.negative(f64))
    assert array_equals(absolute_result, np.absolute(f32))
    assert bool(np.signbit(negative_result[0])) == bool(np.signbit(np.negative(f64)[0]))
    assert bool(np.signbit(absolute_result[0, 0])) == bool(np.signbit(np.absolute(f32)[0, 0]))
    np.testing.assert_array_equal(abs_i64(i64), np.abs(i64))
    np.testing.assert_array_equal(square_i64(i64_2d), np.square(i64_2d))


@pytest.mark.filterwarnings("ignore:the matrix subclass.*:PendingDeprecationWarning")
def test_ndarray_subclasses_are_rejected_before_method_semantics(project: CertifiedProject) -> None:
    """Matrix axis semantics must never be silently normalized to base ndarray."""
    check = checker(project, "method_average_axis", equals=array_equals)
    matrix = np.matrix([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    native, _ = check._run("native", (matrix,))
    fallback, _ = check._run("fallback", (matrix,))
    assert native[0] == "raised"
    assert isinstance(native[1], TypeError)
    assert str(native[1]) == (
        "rextio-numpy native boundary requires exact numpy.ndarray; "
        "ndarray subclasses are unsupported"
    )
    assert fallback[0] == "returned"
    assert type(fallback[1]) is np.matrix
    assert fallback[1].shape == (2, 1)


def test_array_ufunc_override_is_rejected_before_unary_helper(project: CertifiedProject) -> None:
    """A custom __array_ufunc__ result remains fallback-only, never erased natively."""

    class OverrideArray(np.ndarray):
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            del ufunc, method, inputs, kwargs
            return "override-result"

    values = np.array([1.0, -2.0], dtype=np.float64).view(OverrideArray)
    check = checker(project, "unary_negative_f64", equals=array_equals)
    native, _ = check._run("native", (values,))
    fallback, _ = check._run("fallback", (values,))
    assert native[0] == "raised"
    assert isinstance(native[1], TypeError)
    assert str(native[1]) == (
        "rextio-numpy native boundary requires exact numpy.ndarray; "
        "ndarray subclasses are unsupported"
    )
    assert fallback == ("returned", "override-result")


def test_dot_length_mismatch_raises_equivalently(project: CertifiedProject) -> None:
    dot = checker(project, "dot", equals=scalar_close, args_equals=array_equals)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    try:
        np.dot(a, b)
    except ValueError as exc:
        expected = str(exc)
    else:  # pragma: no cover - numpy always raises here
        pytest.fail("numpy.dot did not raise for mismatched lengths")
    # The kit certifies both legs raised equivalently, then re-raises.
    with pytest.raises(ValueError) as excinfo:
        dot(a, b)
    assert str(excinfo.value) == expected


def test_broadcast_mismatch_on_add_raises_equivalently(project: CertifiedProject) -> None:
    add = checker(project, "add", equals=array_equals, args_equals=array_equals)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    try:
        a + b
    except ValueError as exc:
        expected = str(exc)
    else:  # pragma: no cover - numpy always raises here
        pytest.fail("numpy broadcasting did not raise for mismatched 1-D shapes")
    with pytest.raises(ValueError) as excinfo:
        add(a, b)
    assert str(excinfo.value) == expected


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_mean_of_empty_array_is_nan_on_both_legs(project: CertifiedProject) -> None:
    average = checker(project, "average", equals=scalar_close, args_equals=array_equals)
    with warnings.catch_warnings():
        # The fallback leg emits NumPy's "mean of empty slice" RuntimeWarning;
        # the native leg does not (documented divergence on the rule record).
        warnings.simplefilter("ignore", RuntimeWarning)
        result = average(np.zeros(0))
    assert math.isnan(float(result))


ELEMENTS = st.floats(
    allow_nan=False, allow_infinity=False, width=64, min_value=-1e100, max_value=1e100
)


@st.composite
def paired_arrays(draw: st.DrawFn) -> tuple[object, object]:
    length = draw(st.integers(min_value=0, max_value=32))
    shape = (length,)
    a = draw(npst.arrays(dtype=np.float64, shape=shape, elements=ELEMENTS))
    b = draw(npst.arrays(dtype=np.float64, shape=shape, elements=ELEMENTS))
    return a, b


@settings(max_examples=25, deadline=None)
@given(pair=paired_arrays())
def test_hypothesis_dot_equivalence(project: CertifiedProject, pair) -> None:
    a, b = pair
    dot = checker(project, "dot", equals=scalar_close, args_equals=array_equals)
    dot(a, b)


@settings(max_examples=25, deadline=None)
@given(pair=paired_arrays())
def test_hypothesis_add_equivalence(project: CertifiedProject, pair) -> None:
    a, b = pair
    add = checker(project, "add", equals=array_equals, args_equals=array_equals)
    add(a, b)


@pytest.mark.parametrize("name", ["add", "sub", "mul", "div"])
def test_elementwise_length_one_broadcasting_exact(project: CertifiedProject, name: str) -> None:
    # Council M7: NumPy broadcasts a length-1 operand in either position;
    # the native helpers must match exactly (pointwise IEEE), including for
    # the non-commutative operators.
    op = checker(project, name, equals=array_equals, args_equals=array_equals)
    single = np.array([2.0])
    many = np.array([1.0, 4.0, 9.0])
    op(single, many)
    op(many, single)
    # Broadcasting against an empty array yields an empty result on both legs.
    op(single, np.array([], dtype=np.float64))
    op(np.array([], dtype=np.float64), single)
    # Two length-1 arrays are the equal-length path, not the broadcast path.
    op(single, np.array([5.0]))


def test_signature_only_plugin_function_round_trips(project: CertifiedProject) -> None:
    # Council round-2 R17: a plugin-typed function with NO claimed body site
    # exercises the param conversion alone (a distinct codegen path). The
    # return-conversion-only variant (`return a`) is rejected by core since
    # round 3 (T1, alias divergence), so this covers the parameter side; the
    # claimed kernels above cover the return conversion.
    peek = checker(project, "peek", equals=scalar_close, args_equals=array_equals)
    assert float(peek(np.array([1.5, -0.0, 2.0**53]))) == 0.0
    assert float(peek(np.array([], dtype=np.float64))) == 0.0


def test_alias_returning_function_is_not_natively_served(project: CertifiedProject) -> None:
    # Council T1 (round 3): `return a` returns the caller's own object on the
    # fallback but a fresh copy natively — core rejects it to the fallback,
    # and the kit must refuse to certify it (both legs would run Python).
    with pytest.raises(CertificationError, match="not natively served"):
        checker(project, "identity", equals=array_equals)


def test_reduction_tolerance_is_exercised_by_long_mixed_magnitude_arrays(
    project: CertifiedProject,
) -> None:
    # Council round 5 (glm): the hypothesis arrays (length <= 32) cannot
    # produce summation-order divergence anywhere near the documented 1e-12
    # tolerance, so the bound was never actually tested. Long arrays with
    # interleaved large/small magnitudes DO diverge between naive (ndarray)
    # and pairwise (NumPy) summation - this certifies the divergence stays
    # within the rule records' documented tolerance.
    rng = np.random.default_rng(20260707)
    a = rng.uniform(-1.0, 1.0, 4096)
    a[::2] *= 1e12  # interleave large and small magnitudes
    b = rng.uniform(-1.0, 1.0, 4096)
    dot = checker(project, "dot", equals=scalar_close, args_equals=array_equals)
    total = checker(project, "total", equals=scalar_close, args_equals=array_equals)
    average = checker(project, "average", equals=scalar_close, args_equals=array_equals)
    dot(a, b)
    total(a)
    average(a)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_nonfinite_reductions_and_dot_match_numpy(project: CertifiedProject) -> None:
    # Council round 7 (kimi/glm/codex; round-2 R7 promoted): the hypothesis
    # strategies exclude NaN/inf, so the non-finite reduction behavior was
    # never certified despite verified=True. IEEE propagation must agree on
    # both legs: inf - inf and inf * 0 are NaN, NaN propagates.

    dot = checker(project, "dot", equals=scalar_close, args_equals=array_equals)
    total = checker(project, "total", equals=scalar_close, args_equals=array_equals)
    average = checker(project, "average", equals=scalar_close, args_equals=array_equals)

    inf = float("inf")
    cases = (
        np.array([inf, 1.0, -2.0]),
        np.array([inf, -inf]),  # sum -> nan
        np.array([float("nan"), 1.0]),  # nan propagates
        np.array([-0.0, -0.0]),
        np.array([1e308, 1e308]),  # overflow -> inf
    )
    for values in cases:
        total(values)
        average(values)
    dot(np.array([inf, 0.0]), np.array([0.0, 1.0]))  # inf * 0 -> nan
    dot(np.array([inf, 1.0]), np.array([1.0, inf]))  # inf + inf -> inf
    dot(np.array([], dtype=np.float64), np.array([], dtype=np.float64))  # length-0 -> 0.0


def test_claimed_sites_inside_loops_accumulate_like_numpy(project: CertifiedProject) -> None:
    # Council round 7 (claude): every certified kernel was straight-line;
    # claims inside loop bodies exercise a distinct codegen context
    # (statement rendering in loop scope, per-iteration rebinding).
    accumulate = checker(project, "accumulate", equals=array_equals, args_equals=array_equals)
    a = np.array([1.0, -2.5, 3.25])
    b = np.array([0.5, 4.0, -1.0])
    result = accumulate(a, b, 3)
    np.testing.assert_array_equal(result, a + b + b + b + b)
    result = accumulate(a, b, 0)
    np.testing.assert_array_equal(result, a + b)


def _stride_preserving_copy(args: tuple[object, ...]) -> tuple[object, ...]:
    """Per-leg copier that keeps non-contiguous views non-contiguous.

    The kit's default ``copy.deepcopy`` materializes a strided view into a
    fresh C-contiguous array, so the boundary would never see the strides
    (council round 4: the original T19 test was vacuous because of this).
    """
    copied = []
    for arg in args:
        if isinstance(arg, np.ndarray) and not arg.flags["C_CONTIGUOUS"]:
            base = arg.base
            assert base is not None
            copied.append(base.copy()[::2])
        else:
            copied.append(copy.deepcopy(arg))
    return tuple(copied)


def test_non_contiguous_arguments(project: CertifiedProject) -> None:
    # Council T19 (round 3) + round-4 fix: a strided view (a[::2]) is a
    # legitimate float64 1-D ndarray at runtime. The custom copier keeps the
    # per-leg copies strided, so the native leg's PyReadonlyArray1 really
    # receives a non-contiguous array. Empirically rust-numpy accepts it and
    # as_array() honors the strides, so both legs agree on the values -
    # certified behavior (spec section 4).
    add = checker(
        project,
        "add",
        equals=array_equals,
        args_equals=array_equals,
        copy_args=_stride_preserving_copy,
    )
    base = np.array([1.0, 99.0, 2.0, 99.0, 3.0, 99.0])
    strided = base[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    copied_strided = _stride_preserving_copy((strided,))[0]
    assert isinstance(copied_strided, np.ndarray)
    assert not copied_strided.flags["C_CONTIGUOUS"]
    other = np.array([10.0, 20.0, 30.0])
    result = add(strided, other)
    np.testing.assert_array_equal(result, strided + other)


# ---------------------------------------------------------------------------
# Wave 1 post-integration certification matrix
# ---------------------------------------------------------------------------
# These kernels use the new annotation aliases. Until plugin.py consumes
# plugin_types(), the analyzer cannot resolve them to plugin types and the
# certification kit reports "not natively served". That failure mode is the
# expected integration-boundary signal — do not skip/xfail these tests.


def _require_native(project: CertifiedProject, name: str):
    """Return a checker, or fail clearly if the kernel is not natively served."""
    try:
        return checker(project, name, equals=array_equals, args_equals=array_equals)
    except CertificationError as exc:
        pytest.fail(
            f"Wave-1 kernel {name!r} is not natively served — likely waiting on "
            f"plugin.py type_vocabulary integration (plugin_types()): {exc}"
        )


def _require_native_scalar(project: CertifiedProject, name: str, equals):
    try:
        return checker(project, name, equals=equals, args_equals=array_equals)
    except CertificationError as exc:
        pytest.fail(
            f"Wave-1 kernel {name!r} is not natively served — likely waiting on "
            f"plugin.py type_vocabulary integration (plugin_types()): {exc}"
        )


def test_api15_resident_logical_composition_and_whole_reductions(
    project: CertifiedProject,
) -> None:
    """Certify native compare→logical→where and compare→all/any scalar flow."""
    logical_not = _require_native(project, "logical_where_not_f64")
    logical_and = _require_native(project, "logical_where_and_f64_12")
    logical_or = _require_native(project, "logical_where_or_f64_21")
    all_positive = _require_native_scalar(project, "all_positive_f64", scalar_bool_equal)
    any_positive = _require_native_scalar(project, "any_positive_f64", scalar_bool_equal)
    any_branch = _require_native(project, "any_positive_branch_f64")

    values = np.array([-1.0, 0.0, 2.0])
    yes = np.array([10.0, 20.0, 30.0])
    no = np.array([-10.0, -20.0, -30.0])
    np.testing.assert_array_equal(
        logical_not(values, yes, no),
        np.where(np.logical_not(values > 0.0), yes, no),
    )

    left_1d = np.array([1.0, -1.0, 2.0])
    right_2d = np.array([[-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]])
    yes_2d = np.arange(6.0).reshape(2, 3)
    no_1d = np.array([-1.0, -2.0, -3.0])
    np.testing.assert_array_equal(
        logical_and(left_1d, right_2d, yes_2d, no_1d),
        np.where(np.logical_and(left_1d > 0.0, right_2d < 0.0), yes_2d, no_1d),
    )
    np.testing.assert_array_equal(
        logical_or(right_2d, left_1d, no_1d, yes_2d),
        np.where(np.logical_or(right_2d > 0.0, left_1d < 0.0), no_1d, yes_2d),
    )

    assert all_positive(np.array([1.0, 2.0])) is True
    assert all_positive(np.array([1.0, 0.0])) is False
    assert all_positive(np.array([], dtype=np.float64)) is True
    assert any_positive(np.array([0.0, 2.0])) is True
    assert any_positive(np.array([0.0, -2.0])) is False
    assert any_positive(np.array([], dtype=np.float64)) is False
    np.testing.assert_array_equal(
        any_branch(np.array([0.0, 1.0]), no),
        np.array([0.0, 1.0]),
    )
    np.testing.assert_array_equal(
        any_branch(np.array([0.0, -1.0]), no),
        no,
    )


@pytest.mark.parametrize(
    ("name", "a", "b"),
    [
        (
            "add_f64_2d",
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]]),
        ),
        (
            "add_f64_1_2",
            np.array([10.0, 20.0, 30.0]),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ),
        (
            "add_f64_2_1",
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([10.0, 20.0, 30.0]),
        ),
        (
            "add_f32_1d",
            np.array([1.0, -2.5, 3.25], dtype=np.float32),
            np.array([0.5, 4.0, -1.0], dtype=np.float32),
        ),
        (
            "add_f32_2d",
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
        ),
        (
            "add_f32_1_2",
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        ),
        (
            "add_i64_1d",
            np.array([1, -2, 3], dtype=np.int64),
            np.array([4, 5, -6], dtype=np.int64),
        ),
        (
            "add_i64_2d",
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            np.array([[5, 6], [7, 8]], dtype=np.int64),
        ),
        (
            "add_i64_1_2",
            np.array([1, 2], dtype=np.int64),
            np.array([[3, 4], [5, 6]], dtype=np.int64),
        ),
    ],
)
def test_wave1_elementwise_array_array(
    project: CertifiedProject, name: str, a: Any, b: Any
) -> None:
    check = _require_native(project, name)
    result = check(a, b)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, a + b)


@pytest.mark.parametrize(
    ("name", "arr", "scalar"),
    [
        ("scale_f64_2d", np.array([[1.0, 2.0], [3.0, 4.0]]), 2.5),
        ("rsub_f64_2d", np.array([[1.0, 2.0], [3.0, 4.0]]), 10.0),
        ("scale_f32_1d", np.array([1.0, 2.0, 3.0], dtype=np.float32), 1.5),
        ("rsub_f32_1d", np.array([1.0, 2.0, 3.0], dtype=np.float32), 10.0),
        ("scale_i64_1d", np.array([1, 2, 3], dtype=np.int64), 4),
        ("rsub_i64_1d", np.array([1, 2, 3], dtype=np.int64), 10),
    ],
)
def test_wave1_scalar_array_forms(
    project: CertifiedProject, name: str, arr: Any, scalar: float | int
) -> None:
    check = _require_native(project, name)
    if name.startswith("rsub"):
        result = check(scalar, arr)
        expected = scalar - arr
    else:
        result = check(arr, scalar)
        expected = arr * scalar
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("add_f64_2d", np.float64),
        ("add_f32_2d", np.float32),
        ("add_i64_2d", np.int64),
    ],
)
def test_wave1_broadcast_zero_and_length_one(
    project: CertifiedProject, name: str, dtype: Any
) -> None:
    check = _require_native(project, name)
    # length-1 along an axis
    a = np.array([[1], [2], [3]], dtype=dtype)
    b = np.array([[10, 20, 30]], dtype=dtype)
    # reshape to broadcast-compatible (3,1) and (1,3)
    a = a.reshape(3, 1)
    b = b.reshape(1, 3)
    check(a, b)
    # zero-size axes
    check(np.zeros((0, 3), dtype=dtype), np.zeros((1, 3), dtype=dtype))
    check(np.zeros((2, 0), dtype=dtype), np.zeros((2, 1), dtype=dtype))


@pytest.mark.parametrize(
    ("name", "a", "b"),
    [
        (
            "add_f64_2d",
            np.ones((2, 3)),
            np.ones((2, 4)),
        ),
        (
            "add_f64_1_2",
            np.ones(4),
            np.ones((2, 3)),
        ),
        (
            "add_i64_1d",
            np.ones(3, dtype=np.int64),
            np.ones(4, dtype=np.int64),
        ),
    ],
)
def test_wave1_broadcast_mismatch_raises_equivalently(
    project: CertifiedProject, name: str, a: Any, b: Any
) -> None:
    check = _require_native(project, name)
    try:
        a + b
    except ValueError as exc:
        expected = str(exc)
    else:  # pragma: no cover
        pytest.fail("numpy broadcasting did not raise")
    with pytest.raises(ValueError) as excinfo:
        check(a, b)
    assert str(excinfo.value) == expected


def test_wave1_int64_true_division_dtype(project: CertifiedProject) -> None:
    check = _require_native(project, "div_i64_1d")
    a = np.array([3, 5, 7], dtype=np.int64)
    b = np.array([2, 2, 2], dtype=np.int64)
    result = check(a, b)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, a / b)

    check2 = _require_native(project, "div_i64_2d")
    a2 = np.array([[3, 5], [7, 9]], dtype=np.int64)
    b2 = np.array([[2, 2], [2, 2]], dtype=np.int64)
    result2 = check2(a2, b2)
    assert result2.dtype == np.float64
    np.testing.assert_array_equal(result2, a2 / b2)


def test_wave1_int64_overflow_wraparound(project: CertifiedProject) -> None:
    add = _require_native(project, "add_i64_1d")
    mul = _require_native(project, "mul_i64_1d")
    big = np.array([2**62], dtype=np.int64)
    np.testing.assert_array_equal(add(big, big), big + big)
    np.testing.assert_array_equal(
        mul(big, np.array([4], dtype=np.int64)), big * np.array([4], dtype=np.int64)
    )

    total = _require_native_scalar(project, "total_i64_1d", scalar_int_equal)
    assert int(total(np.array([2**63 - 1, 1], dtype=np.int64))) == int(
        np.sum(np.array([2**63 - 1, 1], dtype=np.int64))
    )

    dot = _require_native_scalar(project, "dot_i64", scalar_int_equal)
    assert int(dot(np.array([2**62], dtype=np.int64), np.array([4], dtype=np.int64))) == int(
        np.dot(np.array([2**62], dtype=np.int64), np.array([4], dtype=np.int64))
    )


@pytest.mark.parametrize(
    ("name", "arr", "equals"),
    [
        ("total_f64_2d", np.array([[1.0, 2.0], [3.0, 4.0]]), scalar_close),
        ("average_f64_2d", np.array([[1.0, 2.0], [3.0, 4.0]]), scalar_close),
        ("total_i64_1d", np.array([1, 2, 3], dtype=np.int64), scalar_int_equal),
        ("total_i64_2d", np.array([[1, 2], [3, 4]], dtype=np.int64), scalar_int_equal),
        ("dot_i64", np.array([1, 2, 3], dtype=np.int64), scalar_int_equal),
    ],
)
def test_wave1_reductions_and_dot(
    project: CertifiedProject, name: str, arr: Any, equals: Callable[..., bool]
) -> None:
    check = _require_native_scalar(project, name, equals)
    if name.startswith("dot"):
        result = check(arr, arr)
        expected = np.dot(arr, arr)
    elif name.startswith("total"):
        result = check(arr)
        expected = np.sum(arr)
    else:
        result = check(arr)
        expected = np.mean(arr)
    if equals is scalar_int_equal:
        assert int(result) == int(expected)
    else:
        assert float(result) == pytest.approx(
            float(expected), rel=SCALAR_REL_TOL, abs=SCALAR_ABS_TOL
        )


def test_whole_array_i64_extrema_and_empty_errors(
    project: CertifiedProject,
) -> None:
    maximum = _require_native_scalar(project, "whole_max_i64_1d", scalar_int_equal)
    minimum = _require_native_scalar(project, "whole_min_i64_2d", scalar_int_equal)
    method_max = _require_native_scalar(
        project,
        "method_whole_max_i64_2d",
        scalar_int_equal,
    )

    vector = np.array([3, -7, 4, np.iinfo(np.int64).max], dtype=np.int64)
    matrix = np.array([[3, -7, 4], [12, 0, -5]], dtype=np.int64)
    assert int(maximum(vector)) == int(np.max(vector))
    assert int(minimum(matrix)) == int(np.min(matrix))
    assert int(method_max(matrix)) == int(matrix.max())

    for check, empty, operation in (
        (maximum, np.array([], dtype=np.int64), np.max),
        (minimum, np.empty((0, 2), dtype=np.int64), np.min),
    ):
        with pytest.raises(ValueError) as numpy_error:
            operation(empty)
        with pytest.raises(ValueError) as native_error:
            check(empty)
        assert str(native_error.value) == str(numpy_error.value)


@pytest.mark.parametrize(
    "name",
    [
        "total_f32_1d",
        "average_f32_1d",
        "total_f32_2d",
        "average_f32_2d",
        "dot_f32",
        "average_i64_1d",
        "average_i64_2d",
    ],
)
def test_wave1_unclaimed_reductions_and_dot_not_natively_served(
    project: CertifiedProject, name: str
) -> None:
    """float32 sum/mean/dot and int64 mean must stay fallback-only."""
    with pytest.raises(CertificationError):
        checker(project, name, equals=scalar_close, args_equals=array_equals)


def test_wave1_f32_adversarial_long_input_not_silently_native(
    project: CertifiedProject,
) -> None:
    """Regression: long adversarial f32 input must not get divergent native results.

    Demonstrates the NumPy-vs-sequential gap that motivated the claim narrowing,
    then proves the certification project does not natively serve f32
    sum/mean/dot kernels (so the adversarial input cannot be silently wrong).
    """
    a = _f32_adversarial_array()
    assert a.size == F32_ADVERSARIAL_LEN
    np_sum = float(np.sum(a))
    seq_sum = _sequential_f32_sum(a)
    # Material divergence: NumPy pairwise ~224, sequential f32 cancels to 0.
    assert abs(np_sum - seq_sum) > 100.0
    assert seq_sum == 0.0
    assert np_sum == pytest.approx(224.0, abs=1.0)

    for name in ("total_f32_1d", "average_f32_1d", "dot_f32"):
        with pytest.raises(CertificationError):
            checker(project, name, equals=scalar_close, args_equals=array_equals)


def test_wave1_i64_mean_adversarial_not_silently_native(
    project: CertifiedProject,
) -> None:
    """Regression: adversarial int64 mean must not get divergent native results.

    Confirmed Cargo failure mode: tile([2**53, 1, -2**53], 10000) as int64 —
    NumPy mean ≈ 0.1113, sequential i64→f64 cast-and-sum mean → 0.0. Claim
    reject (RXTP-NUMPY-010) keeps the kernel on the Python fallback.
    """
    a = _i64_mean_adversarial_array()
    assert a.dtype == np.int64
    assert a.size == 3 * I64_MEAN_ADVERSARIAL_REPEATS
    np_mean = float(np.mean(a))
    seq_mean = _sequential_i64_mean_as_f64(a)
    # Material divergence: NumPy pairwise ≈ 0.1113, sequential cancels to 0.
    assert abs(np_mean - seq_mean) > 0.05
    assert seq_mean == 0.0
    assert np_mean == pytest.approx(0.1113, abs=0.01)

    for name in ("average_i64_1d", "average_i64_2d"):
        with pytest.raises(CertificationError):
            checker(project, name, equals=scalar_close, args_equals=array_equals)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    ("name", "empty"),
    [
        ("average_f64_2d", np.zeros((0, 3))),
    ],
)
def test_wave1_mean_empty_is_nan(project: CertifiedProject, name: str, empty: Any) -> None:
    check = _require_native_scalar(project, name, scalar_close)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = check(empty)
    assert math.isnan(float(result))


def test_wave1_f32_elementwise_ops(project: CertifiedProject) -> None:
    a = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    b = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    for name, expected in [
        ("add_f32_1d", a + b),
        ("sub_f32_1d", a - b),
        ("mul_f32_1d", a * b),
        ("div_f32_1d", a / b),
    ]:
        check = _require_native(project, name)
        np.testing.assert_array_equal(check(a, b), expected)


def test_wave1_i64_elementwise_noncommutative_scalar_order(
    project: CertifiedProject,
) -> None:
    a = np.array([1, 2, 3], dtype=np.int64)
    rsub = _require_native(project, "rsub_i64_1d")
    np.testing.assert_array_equal(rsub(10, a), 10 - a)
    scale = _require_native(project, "scale_i64_1d")
    np.testing.assert_array_equal(scale(a, 4), a * 4)


# ---------------------------------------------------------------------------
# Wave 2 literal-axis reduction certification matrix
# ---------------------------------------------------------------------------


def _signed_zero_equal(left: object, right: object) -> bool:
    """Array/scalar equality that distinguishes +0.0 from -0.0 and treats NaN==NaN."""
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        # bitwise for floats catches signed-zero; equal_nan for NaN payload.
        if np.issubdtype(left.dtype, np.floating):
            return bool(
                np.array_equal(left, right, equal_nan=True)
                and np.array_equal(np.signbit(left), np.signbit(right))
            )
        return bool(np.array_equal(left, right))
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        lf, rf = float(left), float(right)
        if math.isnan(lf) and math.isnan(rf):
            return True
        if lf == 0.0 and rf == 0.0:
            return math.copysign(1.0, lf) == math.copysign(1.0, rf)
        return math.isclose(lf, rf, rel_tol=SCALAR_REL_TOL, abs_tol=SCALAR_ABS_TOL)
    return default_equals(left, right)


@pytest.mark.parametrize(
    ("name", "arr", "equals"),
    [
        ("sum_f64_1d_axis0", np.array([1.0, -2.5, 3.25]), scalar_close),
        ("sum_f64_1d_axis_neg1", np.array([1.0, -2.5, 3.25]), scalar_close),
        ("mean_f64_1d_axis0", np.array([1.0, -2.5, 3.25]), scalar_close),
        ("sum_i64_1d_axis0", np.array([1, 2, 3], dtype=np.int64), scalar_int_equal),
    ],
)
def test_wave2_rank1_axis_scalar(
    project: CertifiedProject, name: str, arr: Any, equals: Callable[..., bool]
) -> None:
    check = _require_native_scalar(project, name, equals)
    result = check(arr)
    if name.startswith("sum"):
        expected = np.sum(arr, axis=0 if "neg" not in name else -1)
    elif name.startswith("mean"):
        expected = np.mean(arr, axis=0)
    elif name.startswith("max"):
        expected = np.max(arr, axis=0)
    else:
        expected = np.min(arr, axis=0)
    if equals is scalar_int_equal:
        assert int(result) == int(expected)
    else:
        assert float(result) == pytest.approx(
            float(expected), rel=SCALAR_REL_TOL, abs=SCALAR_ABS_TOL
        )


@pytest.mark.parametrize(
    ("name", "axis"),
    [
        ("sum_f64_2d_axis0", 0),
        ("sum_f64_2d_axis1", 1),
        ("sum_f64_2d_axis_neg1", -1),
        ("sum_f64_2d_axis_neg2", -2),
        ("mean_f64_2d_axis0", 0),
        ("mean_f64_2d_axis1", 1),
    ],
)
def test_wave2_f64_rank2_axis(project: CertifiedProject, name: str, axis: int) -> None:
    a = np.array([[1.0, -2.0, 0.5], [3.0, 4.0, -1.0]])
    check = _require_native(project, name)
    result = check(a)
    if name.startswith("sum"):
        expected = np.sum(a, axis=axis)
    elif name.startswith("mean"):
        expected = np.mean(a, axis=axis)
    elif name.startswith("max"):
        expected = np.max(a, axis=axis)
    else:
        expected = np.min(a, axis=axis)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == expected.shape
    if name.startswith("sum") or name.startswith("mean"):
        np.testing.assert_allclose(result, expected, rtol=SCALAR_REL_TOL, atol=SCALAR_ABS_TOL)
    else:
        np.testing.assert_array_equal(result, expected)


def test_wave2_i64_axis_wraparound(project: CertifiedProject) -> None:
    total = _require_native_scalar(project, "sum_i64_1d_axis0", scalar_int_equal)
    assert int(total(np.array([2**63 - 1, 1], dtype=np.int64))) == int(
        np.sum(np.array([2**63 - 1, 1], dtype=np.int64), axis=0)
    )
    a = np.array([[2**63 - 1, 1], [1, 2**63 - 1]], dtype=np.int64)
    s0 = _require_native(project, "sum_i64_2d_axis0")
    s1 = _require_native(project, "sum_i64_2d_axis1")
    np.testing.assert_array_equal(s0(a), np.sum(a, axis=0))
    np.testing.assert_array_equal(s1(a), np.sum(a, axis=1))


def test_wave2_i64_axis_max_min(project: CertifiedProject) -> None:
    a = np.array([[1, -2, 3], [4, 0, -5]], dtype=np.int64)
    mx = _require_native(project, "max_i64_2d_axis0")
    mn = _require_native(project, "min_i64_2d_axis1")
    np.testing.assert_array_equal(mx(a), np.max(a, axis=0))
    np.testing.assert_array_equal(mn(a), np.min(a, axis=1))


def test_positional_literal_axis_module_and_method_forms(
    project: CertifiedProject,
) -> None:
    """Static positional axes use the same certified helpers as axis=."""
    f64_matrix = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]], dtype=np.float64)
    f64_vector = np.array([1.0, -2.5, 3.25], dtype=np.float64)
    i64_vector = np.array([1, -7, 4], dtype=np.int64)
    i64_matrix = np.array([[1, -2, 3], [4, 0, -5]], dtype=np.int64)

    total = _require_native(project, "sum_f64_2d_pos_axis0")
    average = _require_native_scalar(project, "mean_f64_1d_pos_neg1", scalar_close)
    maximum = _require_native_scalar(project, "max_i64_1d_pos_axis0", scalar_int_equal)
    minimum = _require_native(project, "method_min_i64_2d_pos_axis1")

    np.testing.assert_allclose(
        total(f64_matrix),
        np.sum(f64_matrix, 0),
        rtol=SCALAR_REL_TOL,
        atol=SCALAR_ABS_TOL,
    )
    assert float(average(f64_vector)) == pytest.approx(
        float(np.mean(f64_vector, -1)),
        rel=SCALAR_REL_TOL,
        abs=SCALAR_ABS_TOL,
    )
    assert int(maximum(i64_vector)) == int(np.max(i64_vector, 0))
    np.testing.assert_array_equal(minimum(i64_matrix), i64_matrix.min(1))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_wave2_empty_sum_mean_value_semantics(project: CertifiedProject) -> None:
    s0 = _require_native(project, "sum_f64_2d_axis0")
    s1 = _require_native(project, "sum_f64_2d_axis1")
    m0 = _require_native(project, "mean_f64_2d_axis0")
    m1 = _require_native(project, "mean_f64_2d_axis1")
    # (0, n) / (n, 0) / (0, 0)
    for shape in ((0, 3), (3, 0), (0, 0)):
        a = np.zeros(shape)
        np.testing.assert_array_equal(s0(a), np.sum(a, axis=0))
        np.testing.assert_array_equal(s1(a), np.sum(a, axis=1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r0 = m0(a)
            r1 = m1(a)
            e0 = np.mean(a, axis=0)
            e1 = np.mean(a, axis=1)
        assert r0.shape == e0.shape and r1.shape == e1.shape
        assert np.allclose(r0, e0, equal_nan=True) or (r0.size == 0 and e0.size == 0)
        assert np.allclose(r1, e1, equal_nan=True) or (r1.size == 0 and e1.size == 0)


def test_wave2_axis_does_not_mutate_input(project: CertifiedProject) -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    for name in ("sum_f64_2d_axis0", "mean_f64_2d_axis0"):
        check = _require_native(project, name)
        check(a)
    np.testing.assert_array_equal(a, np.array([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.parametrize(
    "name",
    [
        "bare_max_f64",
        "bare_min_f64",
        "method_max_axis",
        "max_f32_1d_axis0",
        "max_f64_1d_axis0",
        "min_f64_1d_axis0",
        "max_f64_2d_axis0",
        "max_f64_2d_axis1",
        "min_f64_2d_axis0",
        "min_f64_2d_axis1",
        "max_f32_2d_axis0",
        "min_f32_2d_axis1",
        "sum_f32_2d_axis0",
        "mean_i64_2d_axis0",
    ],
)
def test_wave2_excluded_cells_not_natively_served(project: CertifiedProject, name: str) -> None:
    with pytest.raises(CertificationError):
        checker(project, name, equals=array_equals, args_equals=array_equals)


@settings(max_examples=20, deadline=None)
@given(
    a=npst.arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(0, 8), st.integers(0, 8)),
        elements=st.floats(
            allow_nan=False, allow_infinity=False, width=64, min_value=-1e6, max_value=1e6
        ),
    )
)
def test_wave2_hypothesis_axis_sum_mean(project: CertifiedProject, a) -> None:
    s0 = _require_native(project, "sum_f64_2d_axis0")
    s1 = _require_native(project, "sum_f64_2d_axis1")
    m0 = _require_native(project, "mean_f64_2d_axis0")
    m1 = _require_native(project, "mean_f64_2d_axis1")
    s0(a)
    s1(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m0(a)
        m1(a)


# ---------------------------------------------------------------------------
# Wave 2 f64 axis pairwise-sum regression (NumPy cancellation pattern)
# ---------------------------------------------------------------------------
# Confirmed blocker: tile([1e16, 1.0, -1e16], 10000) as float64 —
# sequential/ndarray sum → 0 or ~3, NumPy pairwise sum → 476.0.
# Axis f64 sum/mean must track NumPy within the certified 1e-12 tolerance.


F64_AXIS_PAIRWISE_REPEATS = 10_000
F64_AXIS_PAIRWISE_PATTERN = np.array([1e16, 1.0, -1e16], dtype=np.float64)


def _f64_axis_pairwise_vector(repeats: int = F64_AXIS_PAIRWISE_REPEATS) -> np.ndarray:
    return np.tile(F64_AXIS_PAIRWISE_PATTERN, repeats)


def test_wave2_f64_axis_pairwise_adversarial_rank1(project: CertifiedProject) -> None:
    """Rank-1 axis sum/mean must match NumPy pairwise on cancellation input."""
    a = _f64_axis_pairwise_vector()
    assert a.dtype == np.float64 and a.size == 3 * F64_AXIS_PAIRWISE_REPEATS
    np_sum = float(np.sum(a, axis=0))
    np_mean = float(np.mean(a, axis=0))
    # Guard: sequential accumulation loses the 1.0 contributions entirely.
    seq = 0.0
    for x in a:
        seq += float(x)
    assert abs(np_sum - seq) > 100.0
    assert np_sum == pytest.approx(476.0, abs=0.0)

    total = _require_native_scalar(project, "sum_f64_1d_axis0", scalar_close)
    average = _require_native_scalar(project, "mean_f64_1d_axis0", scalar_close)
    # Equivalence checker compares native vs fallback (NumPy); also pin absolute.
    assert float(total(a)) == pytest.approx(np_sum, rel=SCALAR_REL_TOL, abs=SCALAR_ABS_TOL)
    assert float(average(a)) == pytest.approx(np_mean, rel=SCALAR_REL_TOL, abs=SCALAR_ABS_TOL)
    # Negative axis route shares the same pairwise helper identity (axis0).
    total_neg = _require_native_scalar(project, "sum_f64_1d_axis_neg1", scalar_close)
    assert float(total_neg(a)) == pytest.approx(np_sum, rel=SCALAR_REL_TOL, abs=SCALAR_ABS_TOL)


def test_wave2_f64_axis_pairwise_adversarial_rank2_orientations(
    project: CertifiedProject,
) -> None:
    """Cancellation on unit-stride axis 0 and axis 1 must match NumPy pairwise.

    NumPy only applies pairwise when the reduced axis is contiguous:
    * axis=1 on C-order (row-contiguous) → pairwise (476)
    * axis=0 on F-order (column-contiguous) → pairwise (476)
    * axis=0 on C-order multi-column → sequential (loses cancellation; both legs
      agree near 0 — covered separately)
    """
    v = _f64_axis_pairwise_vector()
    # Axis 0 unit-stride: single column is always contiguous along axis 0.
    col = np.ascontiguousarray(v.reshape(-1, 1))
    # Multi-column axis-0 pairwise: F-order so each column is contiguous.
    mat_f = np.asfortranarray(np.column_stack([v, v, np.flip(v)]))
    assert mat_f.flags["F_CONTIGUOUS"]
    # Axis 1 unit-stride: pattern along rows (C-order).
    row = np.ascontiguousarray(v.reshape(1, -1))
    wide = np.ascontiguousarray(np.vstack([v, np.flip(v), v]))
    assert wide.flags["C_CONTIGUOUS"]

    s0 = _require_native(project, "sum_f64_2d_axis0")
    s1 = _require_native(project, "sum_f64_2d_axis1")
    m0 = _require_native(project, "mean_f64_2d_axis0")
    m1 = _require_native(project, "mean_f64_2d_axis1")
    s_neg1 = _require_native(project, "sum_f64_2d_axis_neg1")
    s_neg2 = _require_native(project, "sum_f64_2d_axis_neg2")

    for arr in (col, mat_f):
        np.testing.assert_allclose(
            s0(arr), np.sum(arr, axis=0), rtol=SCALAR_REL_TOL, atol=SCALAR_ABS_TOL
        )
        np.testing.assert_allclose(
            m0(arr), np.mean(arr, axis=0), rtol=SCALAR_REL_TOL, atol=SCALAR_ABS_TOL
        )
        np.testing.assert_allclose(
            s_neg2(arr), np.sum(arr, axis=-2), rtol=SCALAR_REL_TOL, atol=SCALAR_ABS_TOL
        )

    for arr in (row, wide):
        np.testing.assert_allclose(
            s1(arr), np.sum(arr, axis=1), rtol=SCALAR_REL_TOL, atol=SCALAR_ABS_TOL
        )
        np.testing.assert_allclose(
            m1(arr), np.mean(arr, axis=1), rtol=SCALAR_REL_TOL, atol=SCALAR_ABS_TOL
        )
        np.testing.assert_allclose(
            s_neg1(arr), np.sum(arr, axis=-1), rtol=SCALAR_REL_TOL, atol=SCALAR_ABS_TOL
        )

    # Absolute anchors: NumPy pairwise result on unit-stride lanes.
    assert float(np.sum(col, axis=0)[0]) == pytest.approx(476.0, abs=0.0)
    assert float(s0(col)[0]) == pytest.approx(476.0, abs=SCALAR_ABS_TOL)
    assert float(s0(mat_f)[0]) == pytest.approx(476.0, abs=SCALAR_ABS_TOL)
    assert float(s1(row)[0]) == pytest.approx(476.0, abs=SCALAR_ABS_TOL)
    assert float(m0(col)[0]) == pytest.approx(
        float(np.mean(col, axis=0)[0]), rel=SCALAR_REL_TOL, abs=SCALAR_ABS_TOL
    )


def test_wave2_f64_axis_c_order_axis0_matches_numpy_sequential(
    project: CertifiedProject,
) -> None:
    """C-order multi-column axis=0 stays NumPy-equivalent (sequential, not pairwise).

    On C-contiguous multi-column arrays NumPy's axis=0 reduction is non-unit
    stride and loses the adversarial 1.0 contributions (sum → 0). Native must
    match that fallback result — not invent a more accurate column-wise sum.
    """
    v = _f64_axis_pairwise_vector()
    mat_c = np.ascontiguousarray(np.column_stack([v, v, np.flip(v)]))
    assert mat_c.flags["C_CONTIGUOUS"] and not mat_c.flags["F_CONTIGUOUS"]
    np_sum = np.sum(mat_c, axis=0)
    # NumPy sequential path cancels; pairwise-on-columns would be ~476.
    assert np.allclose(np_sum, 0.0)
    s0 = _require_native(project, "sum_f64_2d_axis0")
    m0 = _require_native(project, "mean_f64_2d_axis0")
    np.testing.assert_allclose(s0(mat_c), np_sum, rtol=SCALAR_REL_TOL, atol=SCALAR_ABS_TOL)
    np.testing.assert_allclose(
        m0(mat_c), np.mean(mat_c, axis=0), rtol=SCALAR_REL_TOL, atol=SCALAR_ABS_TOL
    )


def test_wave2_f64_axis_pairwise_helper_text_in_generated_source() -> None:
    """Generated-source guard: axis f64 helpers embed NumPy pairwise (+ stride dispatch)."""
    from rextio_numpy import rust_snippets

    for rank, axis in ((1, 0), (2, 0), (2, 1)):
        helpers = rust_snippets.axis_typed("sum", "f64", rank, axis)
        text = "\n".join(helpers)
        assert "__rxtnp_numpy_pairwise_sum_f64" in text
        assert "PW_BLOCKSIZE" in text
        assert "sum_axis" not in text
        if rank == 2:
            assert "sequential_sum" in text
            assert "stride_of" in text
        mean_helpers = rust_snippets.axis_typed("mean", "f64", rank, axis)
        mean_text = "\n".join(mean_helpers)
        assert "__rxtnp_numpy_pairwise_sum_f64" in mean_text
        assert "mean_axis" not in mean_text
    # Whole-array route stays on ndarray sum (not rewritten by this fix).
    assert "Ok(a.sum())" in rust_snippets.sum_typed("f64", 1)
    assert "__rxtnp_numpy_pairwise_sum_f64" not in rust_snippets.sum_typed("f64", 1)


def test_wave2_rank1_builtin_scalar_not_numpy_subclass(project: CertifiedProject) -> None:
    """Documented divergence: rank-1 native results are builtin float/int."""
    total = _require_native_scalar(project, "sum_f64_1d_axis0", scalar_close)
    isum = _require_native_scalar(project, "sum_i64_1d_axis0", scalar_int_equal)
    f = total(np.array([1.0, 2.0, 3.0]))
    i = isum(np.array([1, 2, 3], dtype=np.int64))
    assert not isinstance(f, np.ndarray)
    assert not isinstance(i, np.ndarray)
    assert isinstance(float(f), float)
    assert isinstance(int(i), int)


def test_wave2_strided_axis0_matches_numpy(project: CertifiedProject) -> None:
    """Non-contiguous axis-0 view matches NumPy (typically sequential)."""
    v = _f64_axis_pairwise_vector()
    tall = np.zeros((v.size * 2, 2), dtype=np.float64)
    tall[::2, 0] = v
    tall[::2, 1] = np.flip(v)
    strided = tall[::2, :]
    assert not strided.flags["C_CONTIGUOUS"]
    s0 = _require_native(project, "sum_f64_2d_axis0")
    m0 = _require_native(project, "mean_f64_2d_axis0")
    np.testing.assert_allclose(
        s0(strided), np.sum(strided, axis=0), rtol=SCALAR_REL_TOL, atol=SCALAR_ABS_TOL
    )
    np.testing.assert_allclose(
        m0(strided), np.mean(strided, axis=0), rtol=SCALAR_REL_TOL, atol=SCALAR_ABS_TOL
    )


# ---------------------------------------------------------------------------
# Wave 2 elementwise chain fusion certification
# ---------------------------------------------------------------------------


def test_wave2_fusion_f64_multi_op_exact(project: CertifiedProject) -> None:
    check = _require_native(project, "fuse_multi_op")
    a = np.array([1.0, -2.5, 3.25, 0.5])
    b = np.array([4.0, 0.125, -6.5, 2.0])
    result = check(a, b)
    np.testing.assert_array_equal(result, (a + b) * (a - b))
    # No input mutation.
    np.testing.assert_array_equal(a, np.array([1.0, -2.5, 3.25, 0.5]))
    np.testing.assert_array_equal(b, np.array([4.0, 0.125, -6.5, 2.0]))


def test_wave2_fusion_f32_and_i64(project: CertifiedProject) -> None:
    f32 = _require_native(project, "fuse_f32")
    i64 = _require_native(project, "fuse_i64")
    a32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b32 = np.array([0.5, -1.0, 4.0], dtype=np.float32)
    np.testing.assert_array_equal(f32(a32, b32), (a32 + b32) * (a32 - b32))
    a64 = np.array([10, -3, 7], dtype=np.int64)
    b64 = np.array([2, 5, -1], dtype=np.int64)
    np.testing.assert_array_equal(i64(a64, b64), (a64 + b64) * (a64 - b64))


def test_wave2_fusion_mixed_rank_and_rank2(project: CertifiedProject) -> None:
    mixed = _require_native(project, "fuse_mixed_rank")
    r2 = _require_native(project, "fuse_rank2")
    a1 = np.array([1.0, 2.0, 3.0])
    b2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(mixed(a1, b2), (a1 + b2) * (a1 - b2))
    a2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    b2b = np.array([[0.5, 1.5], [-1.0, 2.0]])
    np.testing.assert_array_equal(r2(a2, b2b), (a2 + b2b) * (a2 - b2b))


def test_wave2_fusion_length_one_and_zero(project: CertifiedProject) -> None:
    check = _require_native(project, "fuse_multi_op")
    single = np.array([2.0])
    many = np.array([1.0, 4.0, 9.0])
    check(single, many)
    check(many, single)
    empty = np.array([], dtype=np.float64)
    out = check(empty, empty)
    assert isinstance(out, np.ndarray) and out.shape == (0,)
    # length-1 vs empty broadcast
    check(single, empty)
    check(empty, single)


def test_wave2_fusion_balanced_and_noncommutative(project: CertifiedProject) -> None:
    bal = _require_native(project, "fuse_balanced")
    non = _require_native(project, "fuse_noncomm")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    c = np.array([0.5, 1.5, 2.5])
    d = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(bal(a, b, c, d), (a + b) * (c - d))
    np.testing.assert_array_equal(non(a, b, c), (a - b) / (c + a))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_wave2_fusion_nan_inf_signed_zero(project: CertifiedProject) -> None:
    check = _require_native(project, "fuse_multi_op")
    a = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0])
    b = np.array([1.0, 2.0, 3.0, -0.0, 0.0])
    result = check(a, b)
    expected = (a + b) * (a - b)
    assert np.array_equal(result, expected, equal_nan=True)


def test_wave2_fusion_i64_overflow_intermediate(project: CertifiedProject) -> None:
    """i64 wrapping at intermediate nodes (e.g. near max int64)."""
    check = _require_native(project, "fuse_i64")
    big = np.array([2**62, -(2**62), 2**63 - 1], dtype=np.int64)
    other = np.array([2**62, 2**62, 2], dtype=np.int64)
    # NumPy release builds wrap; both legs must agree.
    result = check(big, other)
    np.testing.assert_array_equal(result, (big + other) * (big - other))


def test_wave2_fusion_broadcast_mismatch_positions(project: CertifiedProject) -> None:
    """Three mismatch positions: left child, right child, root — priority order."""
    # (a + b) * (c - d) with independent shapes so each internal op can fail first.
    bal = _require_native(project, "fuse_balanced")
    # Left child mismatch: a+b fails first.
    a = np.zeros(3)
    b = np.zeros(4)
    c = np.zeros(3)
    d = np.zeros(3)
    try:
        (a + b) * (c - d)
    except ValueError as exc:
        expected_left = str(exc)
    else:  # pragma: no cover
        pytest.fail("numpy did not raise on left child mismatch")
    with pytest.raises(ValueError) as excinfo:
        bal(a, b, c, d)
    assert str(excinfo.value) == expected_left

    # Right child mismatch (left ok): c-d fails.
    a = np.zeros(3)
    b = np.zeros(3)
    c = np.zeros(3)
    d = np.zeros(5)
    try:
        (a + b) * (c - d)
    except ValueError as exc:
        expected_right = str(exc)
    else:  # pragma: no cover
        pytest.fail("numpy did not raise on right child mismatch")
    with pytest.raises(ValueError) as excinfo:
        bal(a, b, c, d)
    assert str(excinfo.value) == expected_right

    # Root mismatch: children succeed, root fails.
    # (a+b) shape (3,), (c-d) shape (4,) when a,b len 3 and c,d len 4.
    a = np.zeros(3)
    b = np.zeros(3)
    c = np.zeros(4)
    d = np.zeros(4)
    try:
        (a + b) * (c - d)
    except ValueError as exc:
        expected_root = str(exc)
    else:  # pragma: no cover
        pytest.fail("numpy did not raise on root mismatch")
    with pytest.raises(ValueError) as excinfo:
        bal(a, b, c, d)
    assert str(excinfo.value) == expected_root
    # Trailing space is part of NumPy's message.
    assert str(excinfo.value).endswith(" ")


def test_wave2_fusion_strided_inputs(project: CertifiedProject) -> None:
    check = checker(
        project,
        "fuse_multi_op",
        equals=array_equals,
        args_equals=array_equals,
        copy_args=_stride_preserving_copy,
    )
    base_a = np.array([1.0, 99.0, 2.0, 99.0, 3.0, 99.0])
    base_b = np.array([4.0, 99.0, 5.0, 99.0, 6.0, 99.0])
    a = base_a[::2]
    b = base_b[::2]
    assert not a.flags["C_CONTIGUOUS"]
    result = check(a, b)
    np.testing.assert_array_equal(result, (a + b) * (a - b))


@settings(max_examples=20, deadline=None)
@given(pair=paired_arrays())
def test_wave2_fusion_hypothesis_f64(project: CertifiedProject, pair) -> None:
    a, b = pair
    check = _require_native(project, "fuse_multi_op")
    check(a, b)


@settings(max_examples=15, deadline=None)
@given(
    a=npst.arrays(
        dtype=np.int64,
        shape=st.integers(0, 16),
        elements=st.integers(-1000, 1000),
    ),
    b=npst.arrays(
        dtype=np.int64,
        shape=st.integers(0, 16),
        elements=st.integers(-1000, 1000),
    ),
)
def test_wave2_fusion_hypothesis_i64(project: CertifiedProject, a, b) -> None:
    # Only equal-length pairs exercise the happy path without broadcast errors
    # dominating the example budget.
    if a.shape != b.shape:
        return
    check = _require_native(project, "fuse_i64")
    check(a, b)


def test_wave2_fusion_helper_allocation_evidence() -> None:
    """Generated helper text: one from_shape_fn pass, zero intermediate ndarrays, no Zip."""
    from rextio.plugins.api import ClaimExpr
    from rextio_numpy.claim.fusion import try_match
    from rextio_numpy.rust_snippets.fusion import build_tree_plan, fusion_helper

    def _leaf(i: int) -> ClaimExpr:
        return ClaimExpr(
            kind="leaf",
            result_type="rextio-numpy/f64-1d",
            leaf_index=i,
            leaf_kind="name",
        )

    expr = ClaimExpr(
        kind="binop",
        target="*",
        result_type="rextio-numpy/f64-1d",
        children=(
            ClaimExpr(
                kind="binop",
                target="+",
                result_type="rextio-numpy/f64-1d",
                children=(_leaf(0), _leaf(1)),
            ),
            ClaimExpr(
                kind="binop",
                target="-",
                result_type="rextio-numpy/f64-1d",
                children=(_leaf(2), _leaf(3)),
            ),
        ),
    )
    match = try_match(expr)
    assert match is not None
    helper = fusion_helper(
        signature=match.signature,
        dtype=match.dtype,
        result_rank=match.result_rank,
        leaf_ranks=match.leaf_ranks,
        expression_ops_postorder=match.postorder_ops,
        tree_plan=build_tree_plan(expr),
    )
    assert helper.count("from_shape_fn") == 1
    assert "Zip::" not in helper
    assert "map_collect" not in helper
    assert "to_owned()" not in helper
    assert "Array::zeros" not in helper
    # Broadcast views only for leaves (one per leaf).
    assert helper.count(".broadcast(") == match.leaf_count
    # Scalar temps for internal non-root nodes.
    assert "let t0" in helper and "let t1" in helper


def test_wave2_fusion_max_bound_f64_compiles_and_matches(project: CertifiedProject) -> None:
    """8-binop / 9-leaf f64 tree: real Cargo native path matches NumPy."""
    check = _require_native(project, "fuse_max_bound_f64")
    arrays = [np.full(8, float(i + 1), dtype=np.float64) for i in range(9)]
    result = check(*arrays)
    expected = arrays[0]
    for other in arrays[1:]:
        expected = expected + other
    np.testing.assert_array_equal(result, expected)
    # Zero-sized max-bound output.
    empty = [np.zeros(0, dtype=np.float64) for _ in range(9)]
    out = check(*empty)
    assert isinstance(out, np.ndarray) and out.shape == (0,)


def test_wave2_fusion_max_bound_i64_overflow(project: CertifiedProject) -> None:
    """8-binop / 9-leaf i64 tree: wrapping intermediates match NumPy release builds."""
    check = _require_native(project, "fuse_max_bound_i64")
    # Near i64 extremes so intermediate wrapping is observable.
    arrays = [
        np.array([2**62, -(2**62), 2**63 - 1], dtype=np.int64),
        np.array([2**62, 2**62, 2], dtype=np.int64),
        np.array([1, -1, 3], dtype=np.int64),
        np.array([7, 8, 9], dtype=np.int64),
        np.array([-3, 4, -5], dtype=np.int64),
        np.array([11, -12, 13], dtype=np.int64),
        np.array([0, 1, -1], dtype=np.int64),
        np.array([5, 5, 5], dtype=np.int64),
        np.array([-2, 2, -2], dtype=np.int64),
    ]
    result = check(*arrays)
    expected = arrays[0]
    for other in arrays[1:]:
        expected = expected + other
    np.testing.assert_array_equal(result, expected)


def test_wave2_fusion_max_bound_routes_and_generated_source(project: CertifiedProject) -> None:
    """Max-bound kernels route through fusion and emit from_shape_fn (not Zip)."""
    import json

    root = project.project_root
    check_path = root / ".rextio" / "reports" / "check.json"
    assert check_path.is_file()
    report = json.loads(check_path.read_text(encoding="utf-8"))
    found_fusion = False
    for module in report.get("modules", ()) or ():
        for function in module.get("functions", ()) or ():
            q = function.get("qualname") or ""
            if not q.endswith("fuse_max_bound_f64"):
                continue
            claims = function.get("plugin_claims") or ()
            assert any(
                c.get("rule_id") == "rextio-numpy/elementwise-chain-fusion"
                and (c.get("operand_mode") or "direct") == "leaves"
                for c in claims
            ), claims
            found_fusion = True
    assert found_fusion, "fuse_max_bound_f64 missing from check report"

    import re

    echain_bodies: list[str] = []
    for path in (root / ".rextio").rglob("*.rs"):
        text = path.read_text(encoding="utf-8")
        if "__rxtnp_echain_" not in text:
            continue
        # Extract only fused helper fn bodies (module also holds ordinary Zip helpers).
        for match in re.finditer(
            r"fn (__rxtnp_echain_\w+)\([^)]*\)[^{]*\{",
            text,
        ):
            start = match.start()
            # Naive brace match from the opening '{' of this fn.
            brace_at = text.find("{", match.end() - 1)
            depth = 0
            end = brace_at
            for i in range(brace_at, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            body = text[start:end]
            echain_bodies.append(body)
            assert "from_shape_fn" in body, match.group(1)
            assert "Zip::" not in body, match.group(1)
            assert "map_collect" not in body, match.group(1)
    assert echain_bodies, "no __rxtnp_echain_ helpers found in generated Rust"
    # Max-bound tree needs 9 leaf broadcasts on at least one helper.
    assert any(body.count("broadcast(dim)") >= 9 for body in echain_bodies)


@pytest.mark.parametrize(
    ("name", "args"),
    (
        (
            "where_eq_f64_11",
            (
                np.array([1.0, 2.0, 3.0]),
                np.array([2.0]),
                np.array([10.0]),
                np.array([20.0, 30.0, 40.0]),
            ),
        ),
        (
            "where_ne_f64_12",
            (
                np.array([1.0, 2.0, 3.0]),
                np.array([[1.0], [3.0]]),
                np.array([[10.0, 11.0, 12.0]]),
                np.array([20.0, 21.0, 22.0]),
            ),
        ),
        (
            "where_lt_f64_21",
            (
                np.array([[1.0], [4.0]]),
                np.array([2.0, 3.0, 4.0]),
                np.array([10.0, 11.0, 12.0]),
                np.array([[20.0], [21.0]]),
            ),
        ),
        (
            "where_le_f64_22",
            (
                np.array([[1.0], [4.0]]),
                np.array([[2.0, 3.0, 4.0]]),
                np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
                np.array([[20.0, 21.0, 22.0]]),
            ),
        ),
    ),
)
def test_compare_where_all_rank_pair_broadcasts(
    project: CertifiedProject,
    name: str,
    args: tuple[np.ndarray, ...],
) -> None:
    """Rank 1/2 × rank 1/2 comparisons and three-way where broadcast natively."""
    result = _require_native(project, name)(*args)
    operations = {
        "where_eq_f64_11": np.equal,
        "where_ne_f64_12": np.not_equal,
        "where_lt_f64_21": np.less,
        "where_le_f64_22": np.less_equal,
    }
    expected = np.where(operations[name](args[0], args[1]), args[2], args[3])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("name", "args", "operation"),
    (
        (
            "where_eq_f64_11",
            (
                np.ones(2),
                np.ones(3),
                np.ones(3),
                np.zeros(3),
            ),
            np.equal,
        ),
        (
            "where_ne_f64_12",
            (
                np.ones(2),
                np.ones((2, 3)),
                np.ones((2, 3)),
                np.zeros(3),
            ),
            np.not_equal,
        ),
        (
            "where_lt_f64_21",
            (
                np.ones((2, 3)),
                np.ones(2),
                np.ones(3),
                np.zeros((2, 3)),
            ),
            np.less,
        ),
        (
            "where_le_f64_22",
            (
                np.ones((2, 3)),
                np.ones((3, 2)),
                np.ones((2, 3)),
                np.zeros((2, 3)),
            ),
            np.less_equal,
        ),
    ),
)
def test_compare_where_all_rank_pair_mismatches(
    project: CertifiedProject,
    name: str,
    args: tuple[np.ndarray, ...],
    operation: Callable[..., object],
) -> None:
    """Every admitted static rank pair retains NumPy's incompatible-shape error."""
    with pytest.raises(ValueError) as expected:
        operation(args[0], args[1])
    with pytest.raises(ValueError) as actual:
        _require_native(project, name)(*args)
    assert str(actual.value) == str(expected.value)


def test_where_independent_three_way_shape_error(project: CertifiedProject) -> None:
    """Compatible comparison operands do not bypass branch broadcast validation."""
    left = np.array([1.0, 2.0])
    right = np.array([1.0, 0.0])
    yes = np.ones(3)
    no = np.zeros(3)
    with pytest.raises(ValueError) as expected:
        np.where(left == right, yes, no)
    with pytest.raises(ValueError) as actual:
        _require_native(project, "where_eq_f64_11")(left, right, yes, no)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize(
    ("name", "args"),
    (
        (
            "where_eq_f64_11",
            (
                np.empty((0,), dtype=np.float64),
                np.ones((1,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.ones((1,), dtype=np.float64),
            ),
        ),
        (
            "where_le_f64_22",
            (
                np.empty((0, 3), dtype=np.float64),
                np.ones((1, 3), dtype=np.float64),
                np.empty((0, 1), dtype=np.float64),
                np.ones((1, 3), dtype=np.float64),
            ),
        ),
    ),
)
def test_compare_where_zero_sized_broadcasts(
    project: CertifiedProject,
    name: str,
    args: tuple[np.ndarray, ...],
) -> None:
    """Length-one axes broadcast to zero axes exactly as NumPy does."""
    operations = {
        "where_eq_f64_11": np.equal,
        "where_le_f64_22": np.less_equal,
    }
    expected = np.where(operations[name](args[0], args[1]), args[2], args[3])
    result = _require_native(project, name)(*args)
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("threshold", (1e40, -1e40, float("nan"), float("inf"), -0.0))
def test_f32_weak_scalar_compare_and_where_special_values(
    project: CertifiedProject,
    threshold: float,
) -> None:
    """Rust's f64→f32 narrowing matches NumPy 2.4 weak-scalar comparisons."""
    values = np.array(
        [np.nan, -np.inf, -0.0, 0.0, 1.0, np.inf],
        dtype=np.float32,
    )
    yes = np.array(
        [0.0, -0.0, np.nan, np.inf, -np.inf, np.float32(7.0)],
        dtype=np.float32,
    )
    no = np.array(
        [-0.0, 0.0, np.float32(3.0), -np.inf, np.inf, np.nan],
        dtype=np.float32,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected_gt = np.where(values > threshold, yes, no)
        expected_ge = np.where(threshold >= values.reshape(2, 3), yes.reshape(2, 3), no.reshape(2, 3))
        actual_gt = _require_native(project, "where_gt_f32_scalar")(
            values,
            threshold,
            yes,
            no,
        )
        actual_ge = _require_native(project, "where_scalar_ge_f32")(
            threshold,
            values.reshape(2, 3),
            yes.reshape(2, 3),
            no.reshape(2, 3),
        )
    np.testing.assert_array_equal(actual_gt.view(np.uint32), expected_gt.view(np.uint32))
    np.testing.assert_array_equal(actual_ge.view(np.uint32), expected_ge.view(np.uint32))


@pytest.mark.parametrize("scalar", (1e40, -1e40, float("nan"), float("inf"), -0.0))
def test_f32_where_branch_weak_scalars_preserve_selected_bits(
    project: CertifiedProject,
    scalar: float,
) -> None:
    """Both scalar branch positions use NumPy 2.4's f64-to-f32 narrowing."""
    values = np.array([-1.0, -0.0, 0.0, 1.0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected_as = np.where(values >= 0.0, values, scalar)
        expected_sa = np.where(values >= 0.0, scalar, values)
        actual_as = _require_native(project, "where_f32_array_scalar")(values, scalar)
        actual_sa = _require_native(project, "where_f32_scalar_array")(scalar, values)
    np.testing.assert_array_equal(actual_as.view(np.uint32), expected_as.view(np.uint32))
    np.testing.assert_array_equal(actual_sa.view(np.uint32), expected_sa.view(np.uint32))


@pytest.mark.parametrize(
    "scalar",
    (
        np.finfo(np.float64).max,
        -np.finfo(np.float64).max,
        float("nan"),
        float("inf"),
        -0.0,
    ),
)
def test_f64_where_branch_scalars_preserve_selected_bits(
    project: CertifiedProject,
    scalar: float,
) -> None:
    values = np.array([-1.0, -0.0, 0.0, 1.0], dtype=np.float64)
    expected = np.where(values >= 0.0, values, scalar)
    result = _require_native(project, "where_f64_array_scalar")(values, scalar)
    np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("target", (-(2**63), -1, 2**63 - 1))
def test_i64_compare_where_scalar_and_inputs_unchanged(
    project: CertifiedProject,
    target: int,
) -> None:
    values = np.array([-(2**63), -1, 0, 2**63 - 1], dtype=np.int64)
    yes = np.array([11, 12, 13, 14], dtype=np.int64)
    no = np.array([-11, -12, -13, -14], dtype=np.int64)
    before = tuple(array.copy() for array in (values, yes, no))
    result = _require_native(project, "where_eq_i64_scalar")(values, target, yes, no)
    np.testing.assert_array_equal(result, np.where(values == target, yes, no))
    for array, snapshot in zip((values, yes, no), before, strict=True):
        np.testing.assert_array_equal(array, snapshot)


@pytest.mark.parametrize("target", (-(2**63) - 1, 2**63))
def test_i64_scalar_outside_core_boundary_is_a_contract_violation(
    project: CertifiedProject,
    target: int,
) -> None:
    """Python ``int`` lowers to Core i64; values outside it never reach the helper."""
    values = np.array([-(2**63), -1, 0, 2**63 - 1], dtype=np.int64)
    yes = np.array([11, 12, 13, 14], dtype=np.int64)
    no = np.array([-11, -12, -13, -14], dtype=np.int64)
    check = checker(
        project,
        "where_eq_i64_scalar",
        equals=array_equals,
        args_equals=array_equals,
    )
    native, _ = check._run("native", (values, target, yes, no))
    fallback, _ = check._run("fallback", (values, target, yes, no))
    assert native[0] == "raised"
    assert isinstance(native[1], OverflowError)
    assert fallback[0] == "returned"
    np.testing.assert_array_equal(
        fallback[1],
        np.where(values == target, yes, no),
    )
