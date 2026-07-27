"""Correctness / ownership diagnostics for the boundary-allocation PoC.

These checks are not headline speed claims. Strided and length-1 broadcast
cases live here as diagnostics only.

Ownership is a hard gate: every strategy result (all three Rust strategies and
the Python reference) must report OWNDATA true and ``base is None``. When this
NumPy build supports in-place ``resize`` on a fresh owned array of the same
shape/dtype, a fresh result must also resize successfully under ``refcheck``.
Failures raise; they are never recorded as soft ``ok``/false flags.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import DTypeLike

from benchmarks.boundary_allocation_poc.protocol import (
    DIAGNOSTIC_BASE_LEN,
    DIAGNOSTIC_BROADCAST_N,
    DIAGNOSTIC_STRIDE,
    EXACT_NDARRAY_TYPEERROR,
    RUST_STRATEGIES,
    STRATEGY_FUNCTIONS,
    StrategyId,
)


def ndarray_owns_data(arr: np.ndarray) -> bool:
    """Return True when the array owns its data buffer."""
    flags = arr.flags
    if hasattr(flags, "owndata"):
        return bool(flags.owndata)
    return bool(flags["OWNDATA"])


def ordinary_numpy_supports_inplace_resize(
    shape: tuple[int, ...], dtype: DTypeLike
) -> bool:
    """Return True when this NumPy build allows in-place resize on a fresh owned array."""
    probe = np.empty(shape, dtype=dtype)
    try:
        probe.resize(int(np.prod(shape, dtype=int)) + 1, refcheck=True)
    except (ValueError, SystemError, TypeError, AttributeError):
        return False
    return True


def python_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return ordinary NumPy elementwise add (reference lane)."""
    return a + b


def resolve_strategy_fn(mod: Any, strategy: StrategyId) -> Callable[..., Any]:
    """Resolve a strategy id to a callable on *mod* (or the Python reference)."""
    if strategy == "python_ref":
        return python_add
    name = STRATEGY_FUNCTIONS[strategy]
    return getattr(mod, name)


def assert_result_values(
    result: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
) -> None:
    """Assert exact ndarray type, dtype, shape, and element values."""
    assert isinstance(result, np.ndarray), f"{label}: expected ndarray, got {type(result)!r}"
    assert type(result) is np.ndarray, f"{label}: expected exact ndarray type, got {type(result)!r}"
    assert result.dtype == np.float64, f"{label}: dtype {result.dtype}"
    assert result.shape == expected.shape, f"{label}: shape {result.shape} != {expected.shape}"
    np.testing.assert_array_equal(result, expected, err_msg=label)


def assert_ownership(result: np.ndarray, *, label: str) -> None:
    """Hard-fail unless ordinary NumPy ownership (and resize when supported) holds.

    Requires OWNDATA true and ``base is None``. When a fresh owned array of the
    same shape/dtype supports in-place ``resize(..., refcheck=True)`` on this
    NumPy build, *result* must also resize successfully.

    **Call with a single live reference** (NumPy ``refcheck`` fails if another
    Python name aliases the same array), e.g. ``assert_ownership(fn(a, b),
    label=...)`` rather than ``x = fn(...); assert_ownership(x)``. Does not
    soft-record failures.
    """
    if not ndarray_owns_data(result):
        raise AssertionError(f"{label}: OWNDATA must be true")
    if result.base is not None:
        raise AssertionError(f"{label}: base must be None (got {result.base!r})")
    shape = tuple(int(x) for x in result.shape)
    dtype = result.dtype
    if ordinary_numpy_supports_inplace_resize(shape, dtype):
        n = int(result.size)
        try:
            result.resize(n + 1, refcheck=True)
        except (ValueError, SystemError, TypeError, AttributeError) as exc:
            raise AssertionError(
                f"{label}: in-place resize must succeed when NumPy supports it "
                f"on a fresh owned array (single live reference required); "
                f"got {type(exc).__name__}: {exc}"
            ) from exc
        if result.shape != (n + 1,):
            raise AssertionError(
                f"{label}: resize shape {result.shape!r} != expected {(n + 1,)!r}"
            )
        # result is consumed/resized; callers that need values must re-call.


def require_subclass_rejection(
    fn: Callable[..., Any],
    *,
    exact_a: np.ndarray,
    exact_b: np.ndarray,
    sub_a: np.ndarray,
    sub_b: np.ndarray,
    strategy: StrategyId,
) -> list[dict[str, Any]]:
    """Hard-fail unless *fn* rejects subclass in either input position.

    Expects the product exact-ndarray TypeError message for left-only and
    right-only subclass inputs. Returns ok records only after both succeed.
    """
    records: list[dict[str, Any]] = []
    cases = (
        ("left", sub_a, exact_b),
        ("right", exact_a, sub_b),
    )
    for position, left, right in cases:
        raised: str | None = None
        try:
            fn(left, right)
        except TypeError as exc:
            raised = str(exc)
        if raised != EXACT_NDARRAY_TYPEERROR:
            raise AssertionError(
                f"{strategy}/subclass_{position}: expected TypeError "
                f"{EXACT_NDARRAY_TYPEERROR!r}, got {raised!r}"
            )
        records.append(
            {
                "case": "subclass_rejection",
                "strategy": strategy,
                "position": position,
                "ok": True,
                "message": raised,
            }
        )
    return records


def run_semantic_suite(mod: Any) -> list[dict[str, Any]]:
    """Run correctness diagnostics for every Rust strategy (+ python_ref).

    Returns a list of per-case records. Raises on hard failure, including
    ownership / resize / subclass gates (never soft-fails those).
    """
    records: list[dict[str, Any]] = []
    strategies: list[StrategyId] = list(RUST_STRATEGIES) + ["python_ref"]

    # --- exact ndarray acceptance / ownership hard gate / subclass rejection ---
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    expected = a + b

    class SubND(np.ndarray):
        pass

    a_sub = a.view(SubND)
    b_sub = b.view(SubND)
    resize_supported = ordinary_numpy_supports_inplace_resize(
        expected.shape, expected.dtype
    )

    for sid in strategies:
        fn = resolve_strategy_fn(mod, sid)
        out = fn(a, b)
        assert_result_values(out, expected, label=f"{sid}/exact")
        # Fresh call; pass temporary so resize refcheck sees a single live ref.
        assert_ownership(fn(a, b), label=f"{sid}/ownership")
        records.append(
            {
                "case": "exact_acceptance_ownership",
                "strategy": sid,
                "ok": True,
                "owndata": True,
                "base_is_none": True,
                "resize_required": resize_supported,
                "resize_ok": True if resize_supported else None,
            }
        )

        if sid == "python_ref":
            # Python reference does not enforce exact-ndarray TypeError.
            continue
        records.extend(
            require_subclass_rejection(
                fn,
                exact_a=a,
                exact_b=b,
                sub_a=a_sub,
                sub_b=b_sub,
                strategy=sid,
            )
        )

    # --- readonly inputs ---
    a_ro = a.copy()
    a_ro.setflags(write=False)
    b_ro = b.copy()
    b_ro.setflags(write=False)
    for sid in strategies:
        fn = resolve_strategy_fn(mod, sid)
        out = fn(a_ro, b_ro)
        assert_result_values(out, expected, label=f"{sid}/readonly")
        assert_ownership(fn(a_ro, b_ro), label=f"{sid}/readonly_ownership")
        records.append({"case": "readonly_inputs", "strategy": sid, "ok": True})

    # --- same-object aliasing ---
    for sid in strategies:
        fn = resolve_strategy_fn(mod, sid)
        out = fn(a, a)
        assert_result_values(out, a + a, label=f"{sid}/alias")
        assert_ownership(fn(a, a), label=f"{sid}/alias_ownership")
        records.append({"case": "same_object_aliasing", "strategy": sid, "ok": True})

    # --- zero length ---
    z = np.array([], dtype=np.float64)
    for sid in strategies:
        fn = resolve_strategy_fn(mod, sid)
        out = fn(z, z)
        assert_result_values(out, z + z, label=f"{sid}/zero")
        assert_ownership(fn(z, z), label=f"{sid}/zero_ownership")
        records.append({"case": "zero_length", "strategy": sid, "ok": True})

    # --- NaN / Inf ---
    a_nf = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float64)
    b_nf = np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float64)
    exp_nf = a_nf + b_nf
    for sid in strategies:
        fn = resolve_strategy_fn(mod, sid)
        out = fn(a_nf, b_nf)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.shape == exp_nf.shape
        np.testing.assert_array_equal(out, exp_nf)
        assert_ownership(fn(a_nf, b_nf), label=f"{sid}/nan_inf_ownership")
        records.append({"case": "nan_inf", "strategy": sid, "ok": True})

    # --- length-1 broadcast (diagnostic) ---
    leaf = np.array([2.0], dtype=np.float64)
    vec = np.arange(DIAGNOSTIC_BROADCAST_N, dtype=np.float64)
    for sid in strategies:
        fn = resolve_strategy_fn(mod, sid)
        out_l = fn(leaf, vec)
        out_r = fn(vec, leaf)
        assert_result_values(out_l, leaf + vec, label=f"{sid}/bcast_left")
        assert_result_values(out_r, vec + leaf, label=f"{sid}/bcast_right")
        assert_ownership(fn(leaf, vec), label=f"{sid}/bcast_left_ownership")
        assert_ownership(fn(vec, leaf), label=f"{sid}/bcast_right_ownership")
        records.append(
            {
                "case": "length1_broadcast",
                "strategy": sid,
                "ok": True,
                "n": DIAGNOSTIC_BROADCAST_N,
                "diagnostic_only": True,
            }
        )

    # --- incompatible broadcast error type / order / exact trailing space ---
    bad_a = np.ones(3, dtype=np.float64)
    bad_b = np.ones(4, dtype=np.float64)
    try:
        bad_a + bad_b
    except ValueError as np_exc:
        np_msg = str(np_exc)
    else:  # pragma: no cover
        raise AssertionError("NumPy should have raised ValueError")

    for sid in strategies:
        fn = resolve_strategy_fn(mod, sid)
        try:
            fn(bad_a, bad_b)
            raise AssertionError(f"{sid}: expected ValueError for (3,)+(4,)")
        except ValueError as exc:
            msg = str(exc)
            assert type(exc) is ValueError
            assert msg == np_msg, f"{sid}: message {msg!r} != NumPy {np_msg!r}"
            assert msg.endswith(" "), f"{sid}: missing trailing space: {msg!r}"
            records.append(
                {
                    "case": "incompatible_broadcast",
                    "strategy": sid,
                    "ok": True,
                    "message": msg,
                }
            )

    # --- strided diagnostic (correctness only) ---
    base = np.arange(DIAGNOSTIC_BASE_LEN * DIAGNOSTIC_STRIDE, dtype=np.float64)
    a_s = base[::DIAGNOSTIC_STRIDE]
    b_s = base[::-DIAGNOSTIC_STRIDE]
    # Match lengths for equal-length strided add.
    n = min(a_s.shape[0], b_s.shape[0])
    a_s = a_s[:n]
    b_s = b_s[:n]
    exp_s = a_s + b_s
    for sid in strategies:
        fn = resolve_strategy_fn(mod, sid)
        out = fn(a_s, b_s)
        assert_result_values(out, exp_s, label=f"{sid}/strided")
        # Single-ref temporary for hard ownership/resize gate.
        assert_ownership(fn(a_s, b_s), label=f"{sid}/strided_ownership")
        records.append(
            {
                "case": "strided_equal_length",
                "strategy": sid,
                "ok": True,
                "n": int(n),
                "stride": DIAGNOSTIC_STRIDE,
                "diagnostic_only": True,
            }
        )

    return records
