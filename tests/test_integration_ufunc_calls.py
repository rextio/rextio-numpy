"""Analyzer-to-lower integration for exact binary NumPy ufunc calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from rextio.analyzer.project_scanner import analyze_project
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_numpy.plugin import plugin

pytest.importorskip("numpy")


class FakeEntryPoint:
    name = "rextio-numpy"

    def load(self):
        return plugin


def _function(analysis, qualname: str):
    for module in analysis.modules:
        for function in module.functions:
            if function.qualname == qualname:
                return function
    raise AssertionError(f"function {qualname!r} not found")


def test_analyzer_claims_exact_ufunc_calls_and_rejects_optional_forms(
    tmp_path: Path,
) -> None:
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
from rextio_numpy.types import F64Arr1

def add(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return np.add(a, b)

def subtract(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return np.subtract(a, b)

def with_out(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return np.add(a, b, out=a)
""",
        encoding="utf-8",
    )
    registry = load_plugin_registry(
        PluginConfig(enabled=("rextio-numpy",)),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )
    analysis = analyze_project(
        tmp_path,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )

    for qualname, target in (
        ("myapp.kernels.add", "numpy.add"),
        ("myapp.kernels.subtract", "numpy.subtract"),
    ):
        function = _function(analysis, qualname)
        claim = next(candidate for candidate in function.plugin_claims if candidate.target == target)
        assert claim.rule_id == "rextio-numpy/elementwise-ufunc-call"
        assert claim.result_type == "rextio-numpy/f64-1d"

    rejected = _function(analysis, "myapp.kernels.with_out")
    assert not any(
        claim.rule_id == "rextio-numpy/elementwise-ufunc-call"
        for claim in rejected.plugin_claims
    )
