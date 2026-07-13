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


def average_i64_2d(a: I64Arr2) -> float:
    return np.mean(a)


def dot_i64(a: I64Arr1, b: I64Arr1) -> int:
    return np.dot(a, b)
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
