"""Analyzer → codegen integration for elementwise chain fusion.

Proves the outer leaves claim subsumes descendants: generated Rust contains
one fused helper call, no ordinary intermediate elementwise helper calls,
LTR leaf order, and malformed expression/index metadata fails closed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rextio.analyzer.project_scanner import analyze_project
from rextio.codegen.rust.generator import generate_rust_module
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.ir.lowering import PluginTypeMaps, lower_project
from rextio.ir.types import RxtPluginType
from rextio.plugins.api import ClaimExpr, ClaimSite, LoweringContext
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_numpy.claim.fusion import FUSION_RULE
from rextio_numpy.diagnostics import F64_1D
from rextio_numpy.plugin import RextioNumpyPlugin, plugin
from rextio_numpy.plugin_types import plugin_types

pytest.importorskip("numpy")


class FakeEntryPoint:
    name = "rextio-numpy"

    def load(self):
        return plugin


def _write_module(tmp_path: Path, body: str) -> Path:
    root = tmp_path
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-numpy"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "myapp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(body, encoding="utf-8")
    return root


def _registry():
    return load_plugin_registry(
        PluginConfig(enabled=("rextio-numpy",)),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )


def _function(analysis, qualname: str):
    for module in analysis.modules:
        for function in module.functions:
            if function.qualname == qualname:
                return function
    raise AssertionError(f"function {qualname!r} not found")


def _type_maps() -> PluginTypeMaps:
    by_key: dict[str, RxtPluginType] = {}
    by_spelling: dict[str, RxtPluginType] = {}
    for pt in plugin_types():
        rxt = RxtPluginType(
            key=pt.key,
            native_rust=pt.rust_type,
            param_rust=pt.conversion.param_rust,
            param_expr=pt.conversion.param_expr,
            return_rust=pt.conversion.return_rust,
            return_expr=pt.conversion.return_expr,
        )
        by_key[pt.key] = rxt
        for spelling in pt.annotations:
            by_spelling[spelling] = rxt
    return PluginTypeMaps(by_key=by_key, by_spelling=by_spelling)


def test_outer_leaves_claim_subsumes_descendants(tmp_path: Path) -> None:
    root = _write_module(
        tmp_path,
        """
from rextio_numpy.types import F64Arr1

def multi_op_chain(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return (a + b) * (a - b)
""",
    )
    registry = _registry()
    assert registry.active[0].api_version == "1.2"
    analysis = analyze_project(
        root,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )
    fn = _function(analysis, "myapp.kernels.multi_op_chain")
    fusion_claims = [c for c in fn.plugin_claims if c.rule_id == FUSION_RULE]
    assert len(fusion_claims) == 1
    outer = fusion_claims[0]
    assert outer.operand_mode == "leaves"
    assert outer.expression is not None
    assert outer.expression.kind == "binop"
    assert outer.expression.target == "*"

    # Descendant elementwise claims may still be recorded; codegen must not
    # lower them when the outer leaves claim is rendered.
    elementwise = [c for c in fn.plugin_claims if c.rule_id == "rextio-numpy/elementwise-float64"]
    # Inner + and - are depth-1; they may be claimed for typing.
    assert all(c.operand_mode == "direct" for c in elementwise)

    module_ir = lower_project(analysis, plugin_types=_type_maps())
    provider = RextioNumpyPlugin()
    source = generate_rust_module(
        module_ir,
        plugin_providers={"rextio-numpy": provider},
        plugin_types_by_key=_type_maps().by_key,
    )
    assert "__rxtnp_echain_" in source
    # Exactly one fused helper definition.
    assert source.count("fn __rxtnp_echain_") == 1
    # No ordinary intermediate elementwise helper *calls* for add/sub/mul of
    # the nested form. The fused body uses scalar ops, not __rxtnp_add1_aa.
    for banned in (
        "__rxtnp_add1_aa(",
        "__rxtnp_sub1_aa(",
        "__rxtnp_mul1_aa(",
        "__rxtnp_add11_aa_f64(",
        "__rxtnp_sub11_aa_f64(",
        "__rxtnp_mul11_aa_f64(",
    ):
        assert banned not in source, banned
    # Leaf order LTR: a, b, a, b
    assert "&a, &b, &a, &b" in source.replace(" ", "") or ("&a,&b,&a,&b" in source.replace(" ", ""))


def test_single_binop_not_fused(tmp_path: Path) -> None:
    root = _write_module(
        tmp_path,
        """
from rextio_numpy.types import F64Arr1

def add(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return a + b
""",
    )
    registry = _registry()
    analysis = analyze_project(
        root,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )
    fn = _function(analysis, "myapp.kernels.add")
    assert not any(c.rule_id == FUSION_RULE for c in fn.plugin_claims)
    assert any(c.rule_id == "rextio-numpy/elementwise-float64" for c in fn.plugin_claims)


def test_malformed_expression_fails_closed_at_lower() -> None:
    plugin_obj = RextioNumpyPlugin()
    # Rule forces fusion path, but expression is a single binop (out of scope).
    expr = ClaimExpr(
        kind="binop",
        target="+",
        result_type=F64_1D,
        children=(
            ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=0, leaf_kind="name"),
            ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=1, leaf_kind="name"),
        ),
    )
    site = ClaimSite(
        kind="binop",
        target="+",
        operand_types=(F64_1D, F64_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=FUSION_RULE,
        result_type=F64_1D,
        expression=expr,
    )
    ctx = LoweringContext(
        operands=(),
        target_language="rust",
        fresh_name=lambda p: f"{p}_0",
        leaf_operands=("a", "b"),
    )
    with pytest.raises(ValueError, match="failed fusion match"):
        plugin_obj.lower(site, ctx)


def test_malformed_leaf_index_metadata_fails_closed() -> None:
    plugin_obj = RextioNumpyPlugin()
    expr = ClaimExpr(
        kind="binop",
        target="*",
        result_type=F64_1D,
        children=(
            ClaimExpr(
                kind="binop",
                target="+",
                result_type=F64_1D,
                children=(
                    ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=0, leaf_kind="name"),
                    ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=1, leaf_kind="name"),
                ),
            ),
            ClaimExpr(
                kind="binop",
                target="-",
                result_type=F64_1D,
                children=(
                    ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=2, leaf_kind="name"),
                    ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=5, leaf_kind="name"),
                ),
            ),
        ),
    )
    site = ClaimSite(
        kind="binop",
        target="*",
        operand_types=(F64_1D, F64_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=FUSION_RULE,
        result_type=F64_1D,
        expression=expr,
    )
    ctx = LoweringContext(
        operands=(),
        target_language="rust",
        fresh_name=lambda p: f"{p}_0",
        leaf_operands=("a", "b", "c", "d"),
    )
    with pytest.raises(ValueError, match="failed fusion match|leaf_index"):
        plugin_obj.lower(site, ctx)
