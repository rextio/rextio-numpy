"""Focused checks for the annotation vocabulary (unchanged in Wave 0)."""

from __future__ import annotations

import pytest

from rextio_numpy.diagnostics import F64_1D
from rextio_numpy.plugin import F64_1D as PLUGIN_F64_1D
from rextio_numpy.plugin import RextioNumpyPlugin


def test_f64_1d_reexported_from_plugin() -> None:
    assert F64_1D == "rextio-numpy/f64-1d"
    assert PLUGIN_F64_1D is F64_1D or PLUGIN_F64_1D == F64_1D


def test_type_vocabulary_surface() -> None:
    types = RextioNumpyPlugin().type_vocabulary()
    assert len(types) == 1
    plugin_type = types[0]
    assert plugin_type.key == F64_1D
    assert plugin_type.annotations == ("rextio_numpy.types.F64Arr1",)
    assert plugin_type.rust_type == "numpy::ndarray::Array1<f64>"


def test_runtime_alias() -> None:
    numpy = pytest.importorskip("numpy")
    from rextio_numpy import types

    assert types.F64Arr1 is numpy.ndarray
