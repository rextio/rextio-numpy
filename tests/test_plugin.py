from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rextio.analyzer.project_scanner import analyze_project
from rextio.cli.capabilities_cmd import build_manifest
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.api import CoverageDecl, RuleRecord
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec
from rextio.targets.plan import TargetPlan

from rextio_numpy import __version__
from rextio_numpy.plugin import RextioNumpyPlugin, plugin
from rextio_numpy.rules import COVERAGE, numpy_rule_records


class FakeEntryPoint:
    name = "rextio-numpy"

    def load(self) -> Any:
        return plugin


def load_registry(enabled: tuple[str, ...] = ("rextio-numpy",)):
    return load_plugin_registry(
        PluginConfig(enabled=enabled),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )


def test_plugin_object_satisfies_protocol_v2() -> None:
    instance = RextioNumpyPlugin()
    assert instance.plugin_id == "rextio-numpy"
    assert instance.api_version == "1.1"
    assert isinstance(instance.covers(), CoverageDecl)
    records = instance.describe(RextioConfig())
    assert records and all(isinstance(record, RuleRecord) for record in records)


def test_core_loader_accepts_the_plugin() -> None:
    registry = load_registry()

    active = registry.active[0]
    assert active.id == "rextio-numpy"
    assert active.rules_provided is True
    assert active.lowering_provided is True
    assert active.api_version == "1.1"
    assert active.packages == ("numpy",)
    assert __version__ in active.name

    assert registry.coverages[0].coverage == COVERAGE
    assert [record.id for record in registry.rule_records] == [
        record.id for record in numpy_rule_records()
    ]
    assert all(record.provider == "rextio-numpy" for record in registry.rule_records)


def test_core_loader_registers_the_type_vocabulary() -> None:
    registry = load_registry()

    assert len(registry.types) == 1
    binding = registry.types[0]
    assert binding.plugin_id == "rextio-numpy"
    plugin_type = binding.plugin_type
    assert plugin_type.key == "rextio-numpy/f64-1d"
    assert plugin_type.annotations == ("rextio_numpy.types.F64Arr1",)
    assert plugin_type.rust_type == "numpy::ndarray::Array1<f64>"
    conversion = plugin_type.conversion
    assert conversion.param_rust == "numpy::PyReadonlyArray1<'py, f64>"
    assert conversion.param_expr == "{param}.as_array().to_owned()"
    assert conversion.return_rust == "pyo3::Bound<'py, numpy::PyArray1<f64>>"
    assert conversion.return_expr == "numpy::ToPyArray::to_pyarray(&{value}, py)"


def test_core_loader_registers_the_pinned_crates() -> None:
    registry = load_registry()

    pins = [
        (binding.plugin_id, binding.dependency.name, binding.dependency.version)
        for binding in registry.crate_dependencies
    ]
    assert pins == [
        ("rextio-numpy", "numpy", "=0.29.0"),
    ]


def test_annotation_vocabulary_is_a_plain_runtime_alias() -> None:
    numpy = pytest.importorskip("numpy")
    from rextio_numpy import types

    assert types.F64Arr1 is numpy.ndarray


def test_rule_records_shape() -> None:
    records = numpy_rule_records()
    ids = [record.id for record in records]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    assert all(record.id.startswith("rextio-numpy/") for record in records)
    codes = [record.diagnostic_code for record in records if record.diagnostic_code]
    assert len(codes) == len(set(codes))
    assert all(code.startswith("RXTP-NUMPY-") for code in codes)
    assert all(record.stability == "experimental" for record in records)
    assert all(record.constraint.strip() and record.guidance.strip() for record in records)
    # The implemented lowering surface: elementwise, dot, and sum/mean
    # reductions — all certified with the core kit (verified=True).
    native_ids = {record.id for record in records if record.outcome == "native"}
    assert native_ids == {
        "rextio-numpy/elementwise-float64",
        "rextio-numpy/dot-float64",
        "rextio-numpy/reduction-sum-mean",
    }
    assert all(record.verified is True for record in records if record.outcome == "native")
    assert all(record.verified is None for record in records if record.outcome != "native")


def test_capabilities_manifest_merges_plugin_rules(tmp_path: Path) -> None:
    registry = load_registry()
    manifest = build_manifest(
        tmp_path,
        RextioConfig(plugins=PluginConfig(enabled=("rextio-numpy",))),
        TargetPlan(spec=TargetSpec(), plugins=registry),
    )

    plugin_entries = manifest["plugins"]
    assert plugin_entries[0]["id"] == "rextio-numpy"
    assert plugin_entries[0]["rules_provided"] is True
    plugin_rules = [rule for rule in manifest["rules"] if rule["provider"] == "rextio-numpy"]
    assert len(plugin_rules) == len(numpy_rule_records())
    assert any(rule["provider"] == "core" for rule in manifest["rules"])


def test_numba_decorated_numpy_function_gets_rxt091_hint(tmp_path: Path) -> None:
    module = tmp_path / "src" / "myapp" / "kernels.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        """
import numba
import numpy as np

@numba.njit
def scale(x: float) -> float:
    return x * 2.0
""",
        encoding="utf-8",
    )
    registry = load_registry()
    analysis = analyze_project(tmp_path, active_plugins=registry.active)

    function = next(
        function
        for module_analysis in analysis.modules
        for function in module_analysis.functions
        if function.qualname == "myapp.kernels.scale"
    )
    assert function.route == "fallback-accelerated:numba"
    hints = [d for d in function.diagnostics if d.code == "RXT091"]
    assert len(hints) == 1
    assert "rextio-numpy" in hints[0].message
