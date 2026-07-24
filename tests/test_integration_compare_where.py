"""Core-local API-1.5 analyzer → compare → where → Rust vertical slice."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rextio.analyzer.plugin_claims import ClaimEngine
from rextio.analyzer.project_scanner import analyze_project
from rextio.codegen.rust.generator import generate_rust_module
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.ir.lowering import PluginTypeMaps, lower_project
from rextio.ir.types import RxtPluginType
from rextio.plugins.api import PLUGIN_API_VERSION
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_numpy.plugin import plugin
from rextio_numpy.plugin_types import plugin_types

pytest.importorskip("numpy")


class FakeEntryPoint:
    """Load the source-tree plugin through Core's real registry path."""

    name = "rextio-numpy"

    def load(self):
        """Return the plugin entry-point factory."""
        return plugin


def _registry():
    return load_plugin_registry(
        PluginConfig(enabled=("rextio-numpy",)),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )


def _type_maps() -> PluginTypeMaps:
    by_key: dict[str, RxtPluginType] = {}
    by_spelling: dict[str, RxtPluginType] = {}
    for plugin_type in plugin_types():
        conversion = plugin_type.conversion
        if conversion is None:
            lowered = RxtPluginType(
                key=plugin_type.key,
                native_rust=plugin_type.rust_type,
                resident=True,
            )
        else:
            lowered = RxtPluginType(
                key=plugin_type.key,
                native_rust=plugin_type.rust_type,
                param_rust=conversion.param_rust,
                param_expr=conversion.param_expr,
                return_rust=conversion.return_rust,
                return_expr=conversion.return_expr,
            )
        by_key[plugin_type.key] = lowered
        for spelling in plugin_type.annotations:
            by_spelling[spelling] = lowered
    return PluginTypeMaps(by_key=by_key, by_spelling=by_spelling)


def _write_project(tmp_path: Path) -> Path:
    (tmp_path / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-numpy"]\n',
        encoding="utf-8",
    )
    package = tmp_path / "src" / "myapp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(
        """
import numpy as np
import rextio
from numpy import where as choose
from rextio_numpy.types import F64Arr1

where_rebound = np.where

def choose_positive(values: F64Arr1, fallback: F64Arr1) -> F64Arr1:
    mask = values > 0.0
    return np.where(mask, values, fallback)

def choose_positive_import_alias(values: F64Arr1, fallback: F64Arr1) -> F64Arr1:
    return choose(values > 0.0, values, fallback)

def choose_composed_masks(values: F64Arr1, fallback: F64Arr1) -> F64Arr1:
    mask = np.logical_or(np.logical_not(values > 0.0), fallback < 0.0)
    return np.where(mask, values, fallback)

def all_positive(values: F64Arr1) -> bool:
    return np.all(values > 0.0)

def any_positive_branch(values: F64Arr1, fallback: F64Arr1) -> F64Arr1:
    if np.any(values > 0.0):
        return values + 0.0
    return fallback + 0.0

def chained_fallback(a: F64Arr1, b: F64Arr1, c: F64Arr1) -> F64Arr1:
    return np.where(a < b < c, a, c)

def identity_fallback(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return np.where(a is b, a, b)

def condition_boundary(mask: np.ndarray, a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return np.where(mask, a, b)

@rextio.native
def comparison_boundary(values: F64Arr1):
    return values > 0.0

def runtime_rebound_where(
    values: F64Arr1,
    fallback: F64Arr1,
) -> F64Arr1:
    return where_rebound(values > 0.0, values, fallback)
""",
        encoding="utf-8",
    )
    return tmp_path


def _function(analysis, name: str):
    qualname = f"myapp.kernels.{name}"
    for module in analysis.modules:
        for function in module.functions:
            if function.qualname == qualname:
                return function
    raise AssertionError(f"{qualname!r} not found")


def test_result_only_bool_type_cannot_be_forged_from_source() -> None:
    engine = ClaimEngine(_registry(), RextioConfig())
    direct = ast.parse("rextio_numpy.types.BoolArr1", mode="eval").body
    imported = ast.parse("BoolArr1", mode="eval").body

    assert engine.resolve_annotation(direct, {}) is None
    assert engine.resolve_annotation(
        imported,
        {"BoolArr1": "rextio_numpy.types.BoolArr1"},
    ) is None
    assert engine.is_plugin_type("rextio-numpy/bool-1d") is True
    assert engine.is_resident_type("rextio-numpy/bool-1d") is True


def test_api_15_compare_result_flows_into_where_and_codegen(tmp_path: Path) -> None:
    host_major, host_minor = (int(part) for part in PLUGIN_API_VERSION.split(".", 1))
    assert host_major == 1
    assert host_minor >= 5
    registry = _registry()
    assert registry.active[0].api_version == "1.5"
    analysis = analyze_project(
        _write_project(tmp_path),
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )

    choose = _function(analysis, "choose_positive")
    assert choose.accepted is True
    assert choose.route == "native-plugin:rextio-numpy"
    assert [
        (claim.kind, claim.target, claim.rule_id, claim.result_type)
        for claim in choose.plugin_claims
    ] == [
        (
            "compare",
            ">",
            "rextio-numpy/elementwise-compare",
            "rextio-numpy/bool-1d",
        ),
        (
            "call",
            "numpy.where",
            "rextio-numpy/where-three-argument",
            "rextio-numpy/f64-1d",
        ),
    ]

    type_maps = _type_maps()
    source = generate_rust_module(
        lower_project(analysis, plugin_types=type_maps),
        plugin_providers={"rextio-numpy": registry.providers[0].provider},
        plugin_types_by_key=type_maps.by_key,
    )
    assert "__rxtnp_cmp_gt1_as_f64(&values, 0.0)?" in source
    assert "__rxtnp_where111_aa_f64(&" in source
    assert "Array1<bool>" in source

    composed = _function(analysis, "choose_composed_masks")
    assert composed.accepted is True
    assert [claim.rule_id for claim in composed.plugin_claims] == [
        "rextio-numpy/elementwise-compare",
        "rextio-numpy/resident-logical-not",
        "rextio-numpy/elementwise-compare",
        "rextio-numpy/resident-logical-binary",
        "rextio-numpy/where-three-argument",
    ]
    assert "__rxtnp_logical_not_1" in source
    assert "__rxtnp_logical_or11" in source

    all_positive = _function(analysis, "all_positive")
    assert all_positive.accepted is True
    assert [claim.rule_id for claim in all_positive.plugin_claims] == [
        "rextio-numpy/elementwise-compare",
        "rextio-numpy/resident-logical-reduction",
    ]
    assert "__rxtnp_logical_all_1" in source

    any_positive_branch = _function(analysis, "any_positive_branch")
    assert any_positive_branch.accepted is True
    assert [claim.rule_id for claim in any_positive_branch.plugin_claims[:2]] == [
        "rextio-numpy/elementwise-compare",
        "rextio-numpy/resident-logical-reduction",
    ]
    assert "__rxtnp_logical_any_1" in source

    imported_alias = _function(analysis, "choose_positive_import_alias")
    assert imported_alias.accepted is True
    assert [
        (claim.kind, claim.target, claim.rule_id)
        for claim in imported_alias.plugin_claims
    ] == [
        ("compare", ">", "rextio-numpy/elementwise-compare"),
        ("call", "numpy.where", "rextio-numpy/where-three-argument"),
    ]

    for name in ("chained_fallback", "identity_fallback", "condition_boundary"):
        function = _function(analysis, name)
        assert function.accepted is False, (
            name,
            function.route,
            [(claim.kind, claim.target, claim.rule_id) for claim in function.plugin_claims],
        )
        assert not any(
            claim.rule_id
            in {
                "rextio-numpy/elementwise-compare",
                "rextio-numpy/where-three-argument",
            }
            for claim in function.plugin_claims
        )

    comparison_boundary = _function(analysis, "comparison_boundary")
    assert comparison_boundary.accepted is False
    assert "RXT092" in comparison_boundary.rejection_codes
    assert any(
        claim.rule_id == "rextio-numpy/elementwise-compare"
        and claim.result_type == "rextio-numpy/bool-1d"
        for claim in comparison_boundary.plugin_claims
    )

    rebound = _function(analysis, "runtime_rebound_where")
    assert rebound.accepted is False
    assert not any(
        claim.rule_id == "rextio-numpy/where-three-argument"
        for claim in rebound.plugin_claims
    )
