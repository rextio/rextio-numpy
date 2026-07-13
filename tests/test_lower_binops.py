"""Focused lower coverage for elementwise binops."""

from __future__ import annotations

import subprocess
import sys

import pytest

from rextio.plugins.api import ClaimSite, LoweringContext

from rextio_numpy.diagnostics import F32_1D, F64_1D, F64_2D, I64_1D, I64_2D
from rextio_numpy.lower import lower
from rextio_numpy.lower.binops import try_lower

K = F64_1D


def site(target: str, operand_types: tuple[str | None, str | None]) -> ClaimSite:
    return ClaimSite(
        kind="binop",
        target=target,
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
    )


def ctx(*operands: str) -> LoweringContext:
    return LoweringContext(
        operands=tuple(operands),
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )


@pytest.mark.parametrize(
    ("op", "name", "symbol"),
    [("+", "add", "+"), ("-", "sub", "-"), ("*", "mul", "*"), ("/", "div", "/")],
)
def test_try_lower_array_array_f64_rank1(op: str, name: str, symbol: str) -> None:
    lowered = try_lower(site(op, (K, K)), ctx("a", "b"))
    assert lowered is not None
    assert lowered.rust == f"__rxtnp_{name}1_aa(&a, &b)?"
    assert f"Ok(a {symbol} b)" in lowered.helpers[0]
    assert len(lowered.helpers) == 1


@pytest.mark.parametrize(
    ("op", "name"),
    [("+", "add"), ("-", "sub"), ("*", "mul"), ("/", "div")],
)
def test_try_lower_array_scalar_and_scalar_array_f64(op: str, name: str) -> None:
    as_lowered = try_lower(site(op, (K, "float")), ctx("a", "s"))
    sa_lowered = try_lower(site(op, ("float", K)), ctx("s", "a"))
    assert as_lowered is not None
    assert sa_lowered is not None
    assert as_lowered.rust == f"__rxtnp_{name}1_as(&a, s)?"
    assert sa_lowered.rust == f"__rxtnp_{name}1_sa(s, &a)?"


def test_try_lower_broadcast_mixed_rank_includes_shape_helpers() -> None:
    lowered = try_lower(site("+", (F64_1D, F64_2D)), ctx("a", "b"))
    assert lowered is not None
    assert lowered.rust == "__rxtnp_add12_aa_f64(&a, &b)?"
    assert any("__rxtnp_broadcast_shape" in h for h in lowered.helpers)
    assert any("__rxtnp_fmt_shape" in h for h in lowered.helpers)
    assert any("__rxtnp_add12_aa_f64" in h for h in lowered.helpers)


def test_try_lower_i64_wrapping_and_true_div() -> None:
    add = try_lower(site("+", (I64_1D, I64_1D)), ctx("a", "b"))
    assert add is not None
    assert "wrapping_add" in add.helpers[-1]
    div = try_lower(site("/", (I64_1D, I64_1D)), ctx("a", "b"))
    assert div is not None
    assert "as f64" in div.helpers[-1]
    assert "Array1<f64>" in div.helpers[-1]


def test_try_lower_i64_scalar_order() -> None:
    as_lowered = try_lower(site("-", (I64_2D, "int")), ctx("a", "s"))
    sa_lowered = try_lower(site("-", ("int", I64_2D)), ctx("s", "a"))
    assert as_lowered is not None and sa_lowered is not None
    assert "wrapping_sub" in as_lowered.helpers[0]
    assert as_lowered.rust == "__rxtnp_sub2_as_i64(&a, s)?"
    assert sa_lowered.rust == "__rxtnp_sub2_sa_i64(s, &a)?"


def test_try_lower_f32_scalar_casts() -> None:
    lowered = try_lower(site("*", (F32_1D, "float")), ctx("a", "s"))
    assert lowered is not None
    assert "s as f32" in lowered.helpers[0]


def test_try_lower_ignores_non_binop() -> None:
    call_site = ClaimSite(
        kind="call",
        target="numpy.dot",
        operand_types=(K, K),
        file_path="",
        line=0,
        column=0,
    )
    assert try_lower(call_site, ctx("a", "b")) is None


def test_router_matches_try_lower() -> None:
    s = site("+", (K, K))
    c = ctx("x", "y")
    assert lower(s, c) == try_lower(s, c)


# ---------------------------------------------------------------- fail-closed


def test_fail_closed_dtype_mismatch() -> None:
    with pytest.raises(ValueError, match="matching array dtypes"):
        try_lower(site("+", (F64_1D, I64_1D)), ctx("a", "b"))


def test_fail_closed_none_array_operand() -> None:
    with pytest.raises(ValueError, match="non-None right array operand type"):
        try_lower(site("+", ("float", None)), ctx("s", "a"))


def test_fail_closed_non_array_scalar_array_path() -> None:
    # Neither operand is a plugin array type → scalar-array path sees non-array right.
    with pytest.raises(ValueError, match="array right operand type"):
        try_lower(site("+", ("float", "float")), ctx("s", "t"))


def test_fail_closed_wrong_operand_arity() -> None:
    bad = ClaimSite(
        kind="binop",
        target="+",
        operand_types=(K,),
        file_path="",
        line=0,
        column=0,
    )
    with pytest.raises(ValueError, match="exactly two operand types"):
        try_lower(bad, ctx("a", "b"))


def test_fail_closed_wrong_ctx_operands() -> None:
    with pytest.raises(ValueError, match="exactly two ctx.operands"):
        try_lower(site("+", (K, K)), ctx("a"))


def test_fail_closed_under_python_optimize() -> None:
    """Mismatched-dtype binop lower must raise ValueError even under python -O.

    With asserts stripped, dtype mismatch would otherwise emit a helper keyed
    only by the left dtype and silently generate incorrect Rust.
    """
    script = r"""
from rextio.plugins.api import ClaimSite, LoweringContext
from rextio_numpy.diagnostics import F64_1D, I64_1D
from rextio_numpy.lower.binops import try_lower

site = ClaimSite(
    kind="binop",
    target="+",
    operand_types=(F64_1D, I64_1D),
    file_path="",
    line=0,
    column=0,
)
ctx = LoweringContext(
    operands=("a", "b"),
    target_language="rust",
    fresh_name=lambda prefix: f"{prefix}_0",
)
try:
    lowered = try_lower(site, ctx)
except ValueError as exc:
    msg = str(exc)
    if "matching array dtypes" in msg:
        print("rejected")
    else:
        print(f"wrong-error:{msg!r}")
        raise SystemExit(2) from exc
else:
    # Silent emission under -O would look like a normal lowered helper.
    print(f"leaked:{getattr(lowered, 'rust', lowered)!r}")
    raise SystemExit(3)
"""
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    assert completed.stdout.strip() == "rejected"
