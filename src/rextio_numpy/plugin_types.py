"""Feature-owned plugin type registry for the rextio-numpy array surface.

Holds the complete ``PluginType`` / ``BoundaryConversion`` definitions for
float64, float32, and int64 at ranks 1 and 2 plus two unnameable resident
boolean result-only types. ``plugin.py`` exposes this registry via
``type_vocabulary()``.
"""

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

# rust-numpy 0.29 element type tokens used in ArrayN / PyArrayN paths.
_RUST_ELEM = {
    "f64": "f64",
    "f32": "f32",
    "i64": "i64",
}


def _boundary(rank: int, elem: str) -> BoundaryConversion:
    """Return the exact-base-ndarray rust-numpy boundary for rank/elem.

    ``PyReadonlyArray`` deliberately accepts ndarray subclasses. Those carry
    observable method/ufunc dispatch (``matrix`` shape rules,
    ``__array_ufunc__``, ``__array_priority__``), which the owned ndarray copy
    cannot preserve. The conversion therefore checks NumPy's C-level exact
    array predicate before materializing. This is a deterministic native
    boundary rejection, not a static claim-time fallback.

    Parameters materialize an owned Rust copy via ``as_array().to_owned()``.
    That is an owned buffer for the native frame; it does **not** guarantee
    every input becomes C-contiguous (contiguous input layout may be
    preserved; non-contiguous copy layout is unspecified). Array results use
    ``ToPyArray::to_pyarray`` so returned NumPy arrays keep ordinary NumPy
    ownership observables (``OWNDATA``, ``base is None``, resize behavior).
    """
    rust_elem = _RUST_ELEM[elem]
    exact_type = f"numpy::PyArray{rank}<{rust_elem}>"
    param_expr = (
        "{{ "
        f"if !{{param}}.is_exact_instance_of::<{exact_type}>() {{{{ "
        "return Err(pyo3::exceptions::PyTypeError::new_err("
        '"rextio-numpy native boundary requires exact numpy.ndarray; '
        'ndarray subclasses are unsupported")); '
        "}} "
        "{param}.as_array().to_owned() }}"
    )
    return BoundaryConversion(
        param_rust=f"numpy::PyReadonlyArray{rank}<'py, {rust_elem}>",
        param_expr=param_expr,
        return_rust=f"pyo3::Bound<'py, numpy::PyArray{rank}<{rust_elem}>>",
        # Ordinary NumPy-owned result: copy into a Python-heap buffer (UFCS).
        return_expr="numpy::ToPyArray::to_pyarray(&{value}, py)",
    )


def _array_type(
    *,
    key: str,
    annotation: str,
    rank: int,
    elem: str,
) -> PluginType:
    """Build one array PluginType with Array{rank}<elem> native representation."""
    rust_elem = _RUST_ELEM[elem]
    return PluginType(
        key=key,
        annotations=(f"rextio_numpy.types.{annotation}",),
        rust_type=f"numpy::ndarray::Array{rank}<{rust_elem}>",
        conversion=_boundary(rank, elem),
    )


def _resident_bool_type(*, key: str, rank: int) -> PluginType:
    """Build an unnameable result-only boolean array type."""
    return PluginType(
        key=key,
        annotations=(),
        rust_type=f"numpy::ndarray::Array{rank}<bool>",
        conversion=None,
    )


# Stable registry: six materialized numeric types, then two result-only bool types.
PLUGIN_TYPES: tuple[PluginType, ...] = (
    _array_type(key=F64_1D, annotation="F64Arr1", rank=1, elem="f64"),
    _array_type(key=F64_2D, annotation="F64Arr2", rank=2, elem="f64"),
    _array_type(key=F32_1D, annotation="F32Arr1", rank=1, elem="f32"),
    _array_type(key=F32_2D, annotation="F32Arr2", rank=2, elem="f32"),
    _array_type(key=I64_1D, annotation="I64Arr1", rank=1, elem="i64"),
    _array_type(key=I64_2D, annotation="I64Arr2", rank=2, elem="i64"),
    _resident_bool_type(key=BOOL_1D, rank=1),
    _resident_bool_type(key=BOOL_2D, rank=2),
)

_PLUGIN_TYPES_BY_KEY: dict[str, PluginType] = {t.key: t for t in PLUGIN_TYPES}


def plugin_types() -> tuple[PluginType, ...]:
    """Return six boundary types plus two unnameable resident result types."""
    return PLUGIN_TYPES


def plugin_type(key: str) -> PluginType:
    """Return the ``PluginType`` for ``key``, or raise ``KeyError``."""
    return _PLUGIN_TYPES_BY_KEY[key]


def plugin_type_keys() -> frozenset[str]:
    """Return the set of type keys this registry owns."""
    return frozenset(_PLUGIN_TYPES_BY_KEY)


__all__ = [
    "PLUGIN_TYPES",
    "plugin_type",
    "plugin_type_keys",
    "plugin_types",
]
