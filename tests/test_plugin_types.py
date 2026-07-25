"""Feature-level tests for the plugin type registry (Wave 1 Lane A)."""

from __future__ import annotations

from rextio.plugins.api import BoundaryConversion, PluginType

from rextio_numpy.diagnostics import (
    BOOL_1D,
    BOOL_2D,
    F32_1D,
    F32_2D,
    F64_1D,
    F64_2D,
    I64_1D,
    I64_2D,
)
from rextio_numpy.plugin_types import (
    PLUGIN_TYPES,
    plugin_type,
    plugin_type_keys,
    plugin_types,
)


def test_plugin_types_stable_api() -> None:
    assert plugin_types() is PLUGIN_TYPES
    assert len(PLUGIN_TYPES) == 8
    assert plugin_type_keys() == {
        F64_1D,
        F64_2D,
        F32_1D,
        F32_2D,
        I64_1D,
        I64_2D,
        BOOL_1D,
        BOOL_2D,
    }


def test_plugin_types_order_and_keys() -> None:
    keys = [t.key for t in PLUGIN_TYPES]
    assert keys == [
        F64_1D,
        F64_2D,
        F32_1D,
        F32_2D,
        I64_1D,
        I64_2D,
        BOOL_1D,
        BOOL_2D,
    ]
    assert len(keys) == len(set(keys))


def test_f64_1d_matches_wave0_boundary() -> None:
    pt = plugin_type(F64_1D)
    assert isinstance(pt, PluginType)
    assert pt.annotations == ("rextio_numpy.types.F64Arr1",)
    assert pt.rust_type == "numpy::ndarray::Array1<f64>"
    conv = pt.conversion
    assert isinstance(conv, BoundaryConversion)
    assert conv.param_rust == "numpy::PyReadonlyArray1<'py, f64>"
    assert "is_exact_instance_of::<numpy::PyArray1<f64>>" in conv.param_expr
    assert "requires exact numpy.ndarray" in conv.param_expr
    assert conv.return_rust == "pyo3::Bound<'py, numpy::PyArray1<f64>>"
    assert conv.return_expr == "numpy::ToPyArray::to_pyarray(&{value}, py)"


def test_all_rank_dtype_conversions() -> None:
    expected = {
        F64_1D: (1, "f64", "F64Arr1"),
        F64_2D: (2, "f64", "F64Arr2"),
        F32_1D: (1, "f32", "F32Arr1"),
        F32_2D: (2, "f32", "F32Arr2"),
        I64_1D: (1, "i64", "I64Arr1"),
        I64_2D: (2, "i64", "I64Arr2"),
    }
    for key, (rank, elem, ann) in expected.items():
        pt = plugin_type(key)
        assert pt.rust_type == f"numpy::ndarray::Array{rank}<{elem}>"
        assert pt.annotations == (f"rextio_numpy.types.{ann}",)
        assert pt.conversion.param_rust == f"numpy::PyReadonlyArray{rank}<'py, {elem}>"
        assert pt.conversion.return_rust == (f"pyo3::Bound<'py, numpy::PyArray{rank}<{elem}>>")
        assert (
            f"is_exact_instance_of::<numpy::PyArray{rank}<{elem}>>"
            in pt.conversion.param_expr
        )
        assert "ndarray subclasses are unsupported" in pt.conversion.param_expr
        assert pt.conversion.return_expr == "numpy::ToPyArray::to_pyarray(&{value}, py)"
