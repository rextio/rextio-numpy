"""End-to-end analyzer → ClaimSite keyword literal → IR claim integration.

Verifies that core API 1.2 offers ``axis=<int literal>`` metadata to this
plugin (``api_version`` 1.2) and that the plugin claims/rejects the Wave-2
literal-axis surface correctly. Lower emission is checked on the claimed
sites so the analyzer→claim→lower path is covered without Cargo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rextio.analyzer.project_scanner import analyze_project
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.api import Claimed, ClaimLiteral, LoweringContext
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_numpy.diagnostics import F64_1D, F64_2D
from rextio_numpy.plugin import RextioNumpyPlugin, plugin

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


def test_analyzer_offers_axis_literal_and_plugin_claims(tmp_path: Path) -> None:
    root = _write_module(
        tmp_path,
        """
import numpy as np
from rextio_numpy.types import F64Arr1, F64Arr2

def row_sums(a: F64Arr2) -> F64Arr1:
    return np.sum(a, axis=1)

def col_max(a: F64Arr2) -> F64Arr1:
    return np.max(a, axis=0)

def neg_axis_mean(a: F64Arr2) -> F64Arr1:
    return np.mean(a, axis=-1)

def whole_sum(a: F64Arr1) -> float:
    return np.sum(a)
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

    row = _function(analysis, "myapp.kernels.row_sums")
    assert any(
        c.target == "numpy.sum" and c.keywords and c.keywords[0].literal.value == 1
        for c in row.plugin_claims
    )
    sum_claim = next(c for c in row.plugin_claims if c.target == "numpy.sum")
    assert sum_claim.keywords[0].name == "axis"
    assert sum_claim.keywords[0].literal == ClaimLiteral(is_literal=True, value=1)
    assert sum_claim.rule_id == "rextio-numpy/reduction-axis"
    assert sum_claim.result_type == F64_1D

    col = _function(analysis, "myapp.kernels.col_max")
    max_claim = next(c for c in col.plugin_claims if c.target == "numpy.max")
    assert max_claim.keywords[0].literal.value == 0
    assert max_claim.rule_id == "rextio-numpy/reduction-axis"

    neg = _function(analysis, "myapp.kernels.neg_axis_mean")
    mean_claim = next(c for c in neg.plugin_claims if c.target == "numpy.mean")
    assert mean_claim.keywords[0].literal.value == -1
    assert mean_claim.rule_id == "rextio-numpy/reduction-axis"

    whole = _function(analysis, "myapp.kernels.whole_sum")
    whole_claim = next(c for c in whole.plugin_claims if c.target == "numpy.sum")
    assert whole_claim.keywords == ()
    assert whole_claim.rule_id == "rextio-numpy/reduction-sum-mean"

    # lower() consumes the claimed site with keywords intact.
    plugin_obj = RextioNumpyPlugin()
    ctx = LoweringContext(
        operands=("a",),
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )
    # Rebuild a lower-time ClaimSite from analyzer claim fields we care about.
    from rextio.plugins.api import ClaimSite, KeywordArg

    lower_site = ClaimSite(
        kind="call",
        target="numpy.sum",
        operand_types=(F64_2D,),
        file_path="",
        line=0,
        column=0,
        rule_id=sum_claim.rule_id,
        result_type=sum_claim.result_type,
        keywords=(
            KeywordArg(
                name="axis",
                arg_type="int",
                literal=ClaimLiteral(is_literal=True, value=1),
            ),
        ),
    )
    lowered = plugin_obj.lower(lower_site, ctx)
    assert lowered.rust == "__rxtnp_sum2_f64_axis1(&a)?"
    assert "Axis(1)" in "\n".join(lowered.helpers)
    assert "__rxtnp_numpy_pairwise_sum_f64" in "\n".join(lowered.helpers)


def test_analyzer_does_not_claim_unsupported_axis_forms(tmp_path: Path) -> None:
    root = _write_module(
        tmp_path,
        """
import numpy as np
from rextio_numpy.types import F64Arr1, F64Arr2

def bare_max(a: F64Arr1) -> float:
    return np.max(a)

def axis_none(a: F64Arr1) -> float:
    return np.sum(a, axis=None)

def tuple_axis(a: F64Arr2) -> float:
    return np.sum(a, axis=(0, 1))

def dynamic_axis(a: F64Arr1, axis: int) -> float:
    return np.sum(a, axis=axis)

def keepdims(a: F64Arr2) -> F64Arr2:
    return np.sum(a, axis=0, keepdims=True)
""",
    )
    registry = _registry()
    analysis = analyze_project(
        root,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )

    bare = _function(analysis, "myapp.kernels.bare_max")
    # Bare max may be offered, but must not become a successful plugin claim
    # with a native reduction rule.
    assert not any(
        getattr(c, "rule_id", None)
        in {
            "rextio-numpy/reduction-axis",
            "rextio-numpy/reduction-sum-mean",
        }
        for c in bare.plugin_claims
    )

    for qualname in (
        "myapp.kernels.axis_none",
        "myapp.kernels.tuple_axis",
        "myapp.kernels.dynamic_axis",
        "myapp.kernels.keepdims",
    ):
        fn = _function(analysis, qualname)
        # No admitted axis-reduction claim on unsupported forms.
        assert not any(
            getattr(c, "rule_id", None) == "rextio-numpy/reduction-axis" for c in fn.plugin_claims
        )


def test_plugin_claim_on_analyzer_shaped_site_matches_unit_table(tmp_path: Path) -> None:
    """Direct claim() on an analyzer-shaped site stays deterministic."""
    del tmp_path
    from rextio.plugins.api import ClaimSite, KeywordArg

    site = ClaimSite(
        kind="call",
        target="numpy.min",
        operand_types=(F64_2D,),
        file_path="kernels.py",
        line=3,
        column=11,
        keywords=(
            KeywordArg(
                name="axis",
                arg_type="int",
                literal=ClaimLiteral(is_literal=True, value=-2),
            ),
        ),
    )
    result = RextioNumpyPlugin().claim(site, RextioConfig())
    assert result == Claimed(rule_id="rextio-numpy/reduction-axis", result_type=F64_1D)


def test_unsupported_axis_forms_stay_fallback_not_merely_no_claim(tmp_path: Path) -> None:
    """Positional/None/tuple/dynamic/keepdims/UAdd axis must not natively serve."""
    root = _write_module(
        tmp_path,
        """
import numpy as np
from rextio_numpy.types import F64Arr1, F64Arr2

def positional(a: F64Arr1) -> float:
    return np.sum(a, 0)

def axis_none(a: F64Arr1) -> float:
    return np.sum(a, axis=None)

def tuple_axis(a: F64Arr2) -> float:
    return np.sum(a, axis=(0, 1))

def dynamic(a: F64Arr1, axis: int) -> float:
    return np.sum(a, axis=axis)

def keepdims(a: F64Arr2) -> F64Arr2:
    return np.sum(a, axis=0, keepdims=True)

def dtype_kw(a: F64Arr1) -> float:
    return np.sum(a, axis=0, dtype=None)

def plus_one(a: F64Arr2) -> F64Arr1:
    return np.sum(a, axis=+1)

def out_of_range(a: F64Arr1) -> float:
    return np.sum(a, axis=2)

def ok_axis(a: F64Arr2) -> F64Arr1:
    return np.sum(a, axis=1)
""",
    )
    registry = _registry()
    analysis = analyze_project(
        root,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )
    # Control: admitted axis form is natively claimed.
    ok = _function(analysis, "myapp.kernels.ok_axis")
    assert any(c.rule_id == "rextio-numpy/reduction-axis" for c in ok.plugin_claims)
    assert ok.route.startswith("native-plugin")

    for qualname in (
        "myapp.kernels.positional",
        "myapp.kernels.axis_none",
        "myapp.kernels.tuple_axis",
        "myapp.kernels.dynamic",
        "myapp.kernels.keepdims",
        "myapp.kernels.dtype_kw",
        "myapp.kernels.plus_one",
        "myapp.kernels.out_of_range",
    ):
        fn = _function(analysis, qualname)
        assert not any(
            getattr(c, "rule_id", None) == "rextio-numpy/reduction-axis" for c in fn.plugin_claims
        ), qualname
        # Actual fallback routing — not merely an empty claim list on a native fn.
        assert fn.route in {"fallback-python", "fallback-accelerated:numba"} or (
            fn.native_status in {"rejected", "not-candidate"}
        ), (qualname, fn.route, fn.native_status)
