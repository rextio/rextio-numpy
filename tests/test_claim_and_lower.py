"""Cargo-free tests of the claim decision table and the lower() emission."""

from __future__ import annotations

import pytest

from rextio.config.schema import RextioConfig
from rextio.plugins.api import Claimed, ClaimSite, LoweringContext, NotCovered, Rejected

from rextio_numpy.plugin import F64_1D, RextioNumpyPlugin

K = F64_1D
CONFIG = RextioConfig()
PLUGIN = RextioNumpyPlugin()


def site(kind: str, target: str, operand_types: tuple[str | None, ...]) -> ClaimSite:
    return ClaimSite(
        kind=kind,
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


# ---------------------------------------------------------------- claim ----


def test_claim_dot_both_arrays() -> None:
    result = PLUGIN.claim(site("call", "numpy.dot", (K, K)), CONFIG)
    assert result == Claimed(rule_id="rextio-numpy/dot-float64", result_type="float")


@pytest.mark.parametrize("target", ["numpy.sum", "numpy.mean"])
def test_claim_reductions(target: str) -> None:
    result = PLUGIN.claim(site("call", target, (K,)), CONFIG)
    assert result == Claimed(rule_id="rextio-numpy/reduction-sum-mean", result_type="float")


@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
@pytest.mark.parametrize("operands", [(K, K), (K, "float"), ("float", K)])
def test_claim_elementwise_binops(op: str, operands: tuple[str, str]) -> None:
    result = PLUGIN.claim(site("binop", op, operands), CONFIG)
    assert result == Claimed(rule_id="rextio-numpy/elementwise-float64", result_type=K)


@pytest.mark.parametrize(
    ("kind", "target", "operands"),
    [
        ("call", "numpy.dot", (K, None)),
        ("call", "numpy.dot", (None, None)),
        ("call", "numpy.sum", (None,)),
        ("binop", "+", (K, None)),
        ("binop", "/", (None, K)),
    ],
)
def test_claim_unresolved_operand_is_not_covered(
    kind: str, target: str, operands: tuple[str | None, ...]
) -> None:
    assert PLUGIN.claim(site(kind, target, operands), CONFIG) == NotCovered()


@pytest.mark.parametrize(
    ("kind", "target", "operands"),
    [
        ("call", "numpy.zeros", ("int",)),
        ("call", "numpy.linalg.solve", (K, K)),
        ("binop", "%", (K, K)),
        ("binop", "@", (K, K)),
        ("binop", "+", ("float", "float")),
    ],
)
def test_claim_uncovered_sites_are_not_covered(
    kind: str, target: str, operands: tuple[str | None, ...]
) -> None:
    assert PLUGIN.claim(site(kind, target, operands), CONFIG) == NotCovered()


@pytest.mark.parametrize(
    ("kind", "target", "operands"),
    [
        ("call", "numpy.dot", (K, "int")),
        ("call", "numpy.dot", ("list[float]", K)),
        ("call", "numpy.dot", (K,)),
        ("call", "numpy.sum", ("list[float]",)),
        ("call", "numpy.mean", (K, "int")),
        ("binop", "+", (K, "int")),
        ("binop", "-", ("int", K)),
        ("binop", "*", (K, "list[float]")),
    ],
)
def test_claim_covered_but_unsupported_operands_are_rejected(
    kind: str, target: str, operands: tuple[str | None, ...]
) -> None:
    result = PLUGIN.claim(site(kind, target, operands), CONFIG)
    assert isinstance(result, Rejected)
    diagnostic = result.diagnostic
    assert diagnostic.code == "RXTP-NUMPY-010"
    assert diagnostic.severity == "error"
    assert target in diagnostic.message
    for operand in operands:
        assert operand is not None
        assert operand in diagnostic.message
    assert diagnostic.suggestion
    assert "float64" in diagnostic.suggestion
    # Core re-stamps the location; the plugin leaves it blank.
    assert (diagnostic.file_path, diagnostic.line, diagnostic.column) == ("", 0, 0)


def test_claim_is_deterministic() -> None:
    sites = [
        site("call", "numpy.dot", (K, K)),
        site("call", "numpy.sum", (K,)),
        site("call", "numpy.dot", (K, None)),
        site("call", "numpy.dot", (K, "int")),
        site("binop", "*", (K, "float")),
        site("binop", "%", (K, K)),
    ]
    for claim_site in sites:
        first = PLUGIN.claim(claim_site, CONFIG)
        second = PLUGIN.claim(claim_site, CONFIG)
        assert first == second


# ---------------------------------------------------------------- lower ----


def test_lower_dot() -> None:
    lowered = PLUGIN.lower(site("call", "numpy.dot", (K, K)), ctx("a", "b"))
    assert lowered.rust == "__rxtnp_dot1(&a, &b)?"
    assert lowered.uses == ()
    assert len(lowered.helpers) == 1
    helper = lowered.helpers[0]
    assert helper.startswith(
        "fn __rxtnp_dot1(a: &numpy::ndarray::Array1<f64>, b: &numpy::ndarray::Array1<f64>) -> pyo3::PyResult<f64>"
    )
    assert "not aligned: {} (dim 0) != {} (dim 0)" in helper
    assert "pyo3::exceptions::PyValueError::new_err" in helper


@pytest.mark.parametrize(
    ("target", "helper_name", "body"),
    [
        ("numpy.sum", "__rxtnp_sum1", "Ok(a.sum())"),
        ("numpy.mean", "__rxtnp_mean1", "Ok(a.mean().unwrap_or(f64::NAN))"),
    ],
)
def test_lower_reductions(target: str, helper_name: str, body: str) -> None:
    lowered = PLUGIN.lower(site("call", target, (K,)), ctx("values"))
    assert lowered.rust == f"{helper_name}(&values)?"
    assert len(lowered.helpers) == 1
    assert f"fn {helper_name}(a: &numpy::ndarray::Array1<f64>) -> pyo3::PyResult<f64>" in lowered.helpers[0]
    assert body in lowered.helpers[0]


@pytest.mark.parametrize(
    ("op", "name", "symbol"),
    [("+", "add", "+"), ("-", "sub", "-"), ("*", "mul", "*"), ("/", "div", "/")],
)
def test_lower_binop_array_array(op: str, name: str, symbol: str) -> None:
    lowered = PLUGIN.lower(site("binop", op, (K, K)), ctx("a", "b"))
    assert lowered.rust == f"__rxtnp_{name}1_aa(&a, &b)?"
    helper = lowered.helpers[0]
    assert f"fn __rxtnp_{name}1_aa(" in helper
    assert "operands could not be broadcast together with shapes ({},) ({},) " in helper
    assert f"Ok(a {symbol} b)" in helper


@pytest.mark.parametrize(
    ("op", "name", "symbol"),
    [("+", "add", "+"), ("-", "sub", "-"), ("*", "mul", "*"), ("/", "div", "/")],
)
def test_lower_binop_array_scalar(op: str, name: str, symbol: str) -> None:
    lowered = PLUGIN.lower(site("binop", op, (K, "float")), ctx("a", "s"))
    assert lowered.rust == f"__rxtnp_{name}1_as(&a, s)?"
    helper = lowered.helpers[0]
    assert f"fn __rxtnp_{name}1_as(a: &numpy::ndarray::Array1<f64>, s: f64)" in helper
    # No shape check on the scalar form.
    assert "broadcast" not in helper
    assert f"Ok(a.mapv(|x| x {symbol} s))" in helper


@pytest.mark.parametrize(
    ("op", "name", "symbol"),
    [("+", "add", "+"), ("-", "sub", "-"), ("*", "mul", "*"), ("/", "div", "/")],
)
def test_lower_binop_scalar_array(op: str, name: str, symbol: str) -> None:
    lowered = PLUGIN.lower(site("binop", op, ("float", K)), ctx("s", "a"))
    assert lowered.rust == f"__rxtnp_{name}1_sa(s, &a)?"
    helper = lowered.helpers[0]
    assert f"fn __rxtnp_{name}1_sa(s: f64, a: &numpy::ndarray::Array1<f64>)" in helper
    # mapv keeps the scalar as the LEFT operand for sub/div.
    assert f"Ok(a.mapv(|x| s {symbol} x))" in helper


def test_lower_expressions_are_fallible() -> None:
    lowered_all = [
        PLUGIN.lower(site("call", "numpy.dot", (K, K)), ctx("a", "b")),
        PLUGIN.lower(site("call", "numpy.sum", (K,)), ctx("a")),
        PLUGIN.lower(site("call", "numpy.mean", (K,)), ctx("a")),
        PLUGIN.lower(site("binop", "+", (K, K)), ctx("a", "b")),
        PLUGIN.lower(site("binop", "-", (K, "float")), ctx("a", "s")),
        PLUGIN.lower(site("binop", "/", ("float", K)), ctx("s", "a")),
    ]
    for lowered in lowered_all:
        assert lowered.rust.endswith("?")
        assert all("pyo3::PyResult<" in helper for helper in lowered.helpers)


def test_lower_rejects_unclaimed_sites() -> None:
    with pytest.raises(ValueError, match="unclaimed site"):
        PLUGIN.lower(site("call", "numpy.zeros", ("int",)), ctx("n"))
