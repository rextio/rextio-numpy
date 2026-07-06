"""The real vertical slice: certification of the lowering surface under cargo.

Builds a fixture project once per module through the core certification kit
(``rextio.plugins.testing``) with the REAL entry-point-discoverable plugin —
no monkeypatching — then asserts native<->fallback equivalence for every
lowered kernel, including exception equivalence and hypothesis-driven inputs.

Comparator notes (empirically verified): the native leg returns plain Python
``float`` for scalar results while the fallback (NumPy) leg returns
``numpy.float64``; both custom comparators coerce real numbers via ``float()``
before comparing. Elementwise ops are pointwise IEEE, so arrays must match
exactly; dot/sum/mean tolerate summation-order divergence (documented per
rule) via ``math.isclose`` at 1e-12.
"""

from __future__ import annotations

import copy
import math
import numbers
import shutil
import warnings

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

KERNELS = '''
import numpy as np
from rextio_numpy.types import F64Arr1


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
'''


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


def scalar_close(left: object, right: object) -> bool:
    """Tolerant scalar equality: float()-coerced, NaN == NaN, isclose at 1e-12.

    The native leg returns plain ``float`` while the fallback returns
    ``numpy.float64`` — a type-strict comparator would flag every call.
    """
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        left_f, right_f = float(left), float(right)
        if math.isnan(left_f) and math.isnan(right_f):
            return True
        return math.isclose(left_f, right_f, rel_tol=1e-12, abs_tol=1e-12)
    return default_equals(left, right)


ARRAY = np.array([1.0, -2.5, 3.25, 0.5])
OTHER = np.array([4.0, 0.125, -6.5, 2.0])


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
    assert not _stride_preserving_copy((strided,))[0].flags["C_CONTIGUOUS"]
    other = np.array([10.0, 20.0, 30.0])
    result = add(strided, other)
    np.testing.assert_array_equal(result, strided + other)
