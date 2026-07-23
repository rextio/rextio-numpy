"""Focused checks for the annotation vocabulary (Wave 1 Lane A)."""

from __future__ import annotations

import pytest

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
from rextio_numpy.plugin import F64_1D as PLUGIN_F64_1D
from rextio_numpy.plugin import RextioNumpyPlugin
from rextio_numpy.plugin_types import PLUGIN_TYPES, plugin_types


def test_f64_1d_reexported_from_plugin() -> None:
    assert F64_1D == "rextio-numpy/f64-1d"
    assert PLUGIN_F64_1D is F64_1D or PLUGIN_F64_1D == F64_1D


def test_type_key_constants() -> None:
    assert F64_2D == "rextio-numpy/f64-2d"
    assert F32_1D == "rextio-numpy/f32-1d"
    assert F32_2D == "rextio-numpy/f32-2d"
    assert I64_1D == "rextio-numpy/i64-1d"
    assert I64_2D == "rextio-numpy/i64-2d"


def test_type_vocabulary_surface_via_plugin_still_wave0() -> None:
    """Plugin facade exposes six boundary types plus two resident bool types.

    Keeps the historical test name so existing failure reports stay stable;
    assertions now match director integration of plugin_types into plugin.py.
    """
    types = RextioNumpyPlugin().type_vocabulary()
    assert types is PLUGIN_TYPES or types == PLUGIN_TYPES
    assert [t.key for t in types] == [
        F64_1D,
        F64_2D,
        F32_1D,
        F32_2D,
        I64_1D,
        I64_2D,
        BOOL_1D,
        BOOL_2D,
    ]
    assert [t.annotations for t in types] == [
        ("rextio_numpy.types.F64Arr1",),
        ("rextio_numpy.types.F64Arr2",),
        ("rextio_numpy.types.F32Arr1",),
        ("rextio_numpy.types.F32Arr2",),
        ("rextio_numpy.types.I64Arr1",),
        ("rextio_numpy.types.I64Arr2",),
        (),
        (),
    ]

    # Wave-0 F64 rank-1 compatibility surface remains first and unchanged.
    f64_r1 = types[0]
    assert f64_r1.key == F64_1D
    assert f64_r1.annotations == ("rextio_numpy.types.F64Arr1",)
    assert f64_r1.rust_type == "numpy::ndarray::Array1<f64>"
    conv = f64_r1.conversion
    assert conv.param_rust == "numpy::PyReadonlyArray1<'py, f64>"
    assert "is_exact_instance_of::<numpy::PyArray1<f64>>" in conv.param_expr
    assert "requires exact numpy.ndarray" in conv.param_expr
    assert conv.param_expr.endswith("{param}.as_array().to_owned() }}")
    rendered = conv.param_expr.format(param="values")
    assert rendered.startswith("{ if !values.is_exact_instance_of")
    assert rendered.endswith("values.as_array().to_owned() }")
    assert conv.return_rust == "pyo3::Bound<'py, numpy::PyArray1<f64>>"
    assert conv.return_expr == "numpy::ToPyArray::to_pyarray(&{value}, py)"


def test_feature_owned_registry_is_complete() -> None:
    assert plugin_types() is PLUGIN_TYPES
    assert {t.key for t in PLUGIN_TYPES} == {
        F64_1D,
        F64_2D,
        F32_1D,
        F32_2D,
        I64_1D,
        I64_2D,
        BOOL_1D,
        BOOL_2D,
    }


def test_runtime_aliases() -> None:
    numpy = pytest.importorskip("numpy")
    from rextio_numpy import types

    assert types.F64Arr1 is numpy.ndarray
    assert types.F64Arr2 is numpy.ndarray
    assert types.F32Arr1 is numpy.ndarray
    assert types.F32Arr2 is numpy.ndarray
    assert types.I64Arr1 is numpy.ndarray
    assert types.I64Arr2 is numpy.ndarray
    assert not hasattr(types, "BoolArr1")
    assert not hasattr(types, "BoolArr2")
