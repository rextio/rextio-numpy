"""Boundary return ownership: IntoPyArray transfer, input still copies.

Static checks that generated conversion templates and lowered/generated Rust
use ownership transfer for array results without claiming input zero-copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rextio.analyzer.project_scanner import analyze_project
from rextio.codegen.rust.generator import generate_rust_module
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.ir.lowering import PluginTypeMaps, lower_project
from rextio.ir.types import RxtPluginType
from rextio.plugins.api import ClaimExpr
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_numpy.diagnostics import F32_1D, F32_2D, F64_1D, F64_2D, I64_1D, I64_2D
from rextio_numpy.plugin import RextioNumpyPlugin, plugin
from rextio_numpy.plugin_types import plugin_type, plugin_types
from rextio_numpy.rust_snippets.fusion import build_tree_plan, fusion_helper


class FakeEntryPoint:
    name = "rextio-numpy"

    def load(self):
        return plugin


_RETURN_EXPR = "numpy::IntoPyArray::into_pyarray({value}, py)"
_LEGACY_COPY_RETURN = "numpy::ToPyArray::to_pyarray(&{value}, py)"


def test_all_array_boundaries_use_into_pyarray_not_to_pyarray() -> None:
    """Every supported array materialization transfers owned storage on return."""
    keys = (F64_1D, F64_2D, F32_1D, F32_2D, I64_1D, I64_2D)
    for key in keys:
        conv = plugin_type(key).conversion
        assert conv is not None
        assert conv.return_expr == _RETURN_EXPR
        assert conv.return_expr != _LEGACY_COPY_RETURN
        # Ownership transfer: consume value by move, not &borrow copy.
        assert "{value}" in conv.return_expr
        assert "&{value}" not in conv.return_expr
        # Inputs remain an explicit owned copy after the exact-ndarray check.
        assert "to_owned()" in conv.param_expr
        assert "is_exact_instance_of" in conv.param_expr


def test_return_expr_is_str_format_safe() -> None:
    """return_expr must be a str.format template with only {value}."""
    rendered = _RETURN_EXPR.format(value="out")
    assert rendered == "numpy::IntoPyArray::into_pyarray(out, py)"
    # Doubled braces are not required here; ensure no accidental format holes.
    assert "{" not in rendered and "}" not in rendered


def _type_maps() -> PluginTypeMaps:
    by_key: dict[str, RxtPluginType] = {}
    by_spelling: dict[str, RxtPluginType] = {}
    for pt in plugin_types():
        conversion = pt.conversion
        if conversion is None:
            rxt = RxtPluginType(
                key=pt.key,
                native_rust=pt.rust_type,
                resident=True,
            )
        else:
            rxt = RxtPluginType(
                key=pt.key,
                native_rust=pt.rust_type,
                param_rust=conversion.param_rust,
                param_expr=conversion.param_expr,
                return_rust=conversion.return_rust,
                return_expr=conversion.return_expr,
            )
        by_key[pt.key] = rxt
        for spelling in pt.annotations:
            by_spelling[spelling] = rxt
    return PluginTypeMaps(by_key=by_key, by_spelling=by_spelling)


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


def test_generated_rust_uses_into_pyarray_for_array_return(tmp_path: Path) -> None:
    """Codegen embeds IntoPyArray on array-returning functions."""
    pytest.importorskip("numpy")
    root = _write_module(
        tmp_path,
        """
from rextio_numpy.types import F64Arr1

def add(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return a + b
""",
    )
    registry = load_plugin_registry(
        PluginConfig(enabled=("rextio-numpy",)),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )
    analysis = analyze_project(
        root,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )
    module_ir = lower_project(analysis, plugin_types=_type_maps())
    source = generate_rust_module(
        module_ir,
        plugin_providers={"rextio-numpy": RextioNumpyPlugin()},
        plugin_types_by_key=_type_maps().by_key,
    )
    assert "IntoPyArray::into_pyarray" in source
    assert "ToPyArray::to_pyarray" not in source
    # Input path still materializes owned copies.
    assert "to_owned()" in source
    assert "is_exact_instance_of" in source


def test_fusion_helper_retains_generic_broadcast_after_fast_gate() -> None:
    """Fast path is optional; generic broadcast + errors remain in the helper."""

    def leaf(i: int) -> ClaimExpr:
        return ClaimExpr(
            kind="leaf",
            result_type=F64_1D,
            leaf_index=i,
            leaf_kind="name",
        )

    expr = ClaimExpr(
        kind="binop",
        target="*",
        result_type=F64_1D,
        children=(
            ClaimExpr(
                kind="binop",
                target="+",
                result_type=F64_1D,
                children=(leaf(0), leaf(1)),
            ),
            ClaimExpr(
                kind="binop",
                target="-",
                result_type=F64_1D,
                children=(leaf(2), leaf(3)),
            ),
        ),
    )
    helper = fusion_helper(
        signature="test-sig",
        dtype="f64",
        result_rank=1,
        leaf_ranks=(1, 1, 1, 1),
        expression_ops_postorder=("+", "-", "*"),
        tree_plan=build_tree_plan(expr),
    )
    assert "is_standard_layout()" in helper
    assert "as_slice()" in helper
    assert ".broadcast(dim)" in helper
    assert "operands could not be broadcast together with shapes" in helper
    assert "unsafe" not in helper
