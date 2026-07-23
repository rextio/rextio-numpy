"""Core-local API-1.5 analyzer → compare → where → Rust vertical slice."""

from __future__ import annotations

from pathlib import Path

import pytest

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
from rextio_numpy.types import F64Arr1
from rextio_numpy.types._resident import BoolArr1

def choose_positive(values: F64Arr1, fallback: F64Arr1) -> F64Arr1:
    return np.where(values > 0.0, values, fallback)

def chained_fallback(a: F64Arr1, b: F64Arr1, c: F64Arr1) -> F64Arr1:
    return np.where(a < b < c, a, c)

def identity_fallback(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return np.where(a is b, a, b)

def condition_boundary(mask: np.ndarray, a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return np.where(mask, a, b)

@rextio.native
def resident_bool_boundary(mask: BoolArr1) -> int:
    return 1
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


def test_api_15_compare_result_flows_into_where_and_codegen(tmp_path: Path) -> None:
    assert PLUGIN_API_VERSION == "1.5"
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

    resident_boundary = _function(analysis, "resident_bool_boundary")
    assert resident_boundary.has_resident_signature is True
    assert resident_boundary.accepted is False
    assert "RXT092" in resident_boundary.rejection_codes
