"""Shared Rust representation helpers for plugin-owned array snippets.

Only ``F64_1D`` is Python-backed. Every other dtype/rank keeps the historical
owned ``ndarray::ArrayN`` representation.
"""

from __future__ import annotations

F64_1D_RUST = "numpy::PyReadonlyArray1<'py, f64>"
F64_1D_OUTPUT_HELPER_NAME = "__rxtnp_f64_1d_output"
F64_1D_RETURN_HELPER_NAME = "__rxtnp_release_f64_1d"

_DTYPE_RUST = {"f64": "f64", "f32": "f32", "i64": "i64"}


def is_python_backed(dtype: str, rank: int) -> bool:
    """Whether ``dtype``/``rank`` uses a GIL-bound read-only NumPy owner."""
    return dtype == "f64" and rank == 1


def array_rust_type(dtype: str, rank: int) -> str:
    """Return the native Rust representation for one numeric array type."""
    if is_python_backed(dtype, rank):
        return F64_1D_RUST
    return f"numpy::ndarray::Array{rank}<{_DTYPE_RUST[dtype]}>"


def lifetime_decl(*types: tuple[str, int], force: bool = False) -> str:
    """Return ``<'py>`` when a helper owns a Python-backed argument/result."""
    if force or any(is_python_backed(dtype, rank) for dtype, rank in types):
        return "<'py>"
    return ""


def readonly_view_line(name: str, dtype: str, rank: int, *, indent: str = "    ") -> str:
    """Return a shadowing read-only ndarray view line for Python-backed input."""
    if not is_python_backed(dtype, rank):
        return ""
    return f"{indent}let {name} = {name}.as_array();\n"


def f64_1d_output_helper() -> str:
    """Allocate one zeroed NumPy-owned F64_1D sink and expose it only after fill."""
    return """\
fn __rxtnp_f64_1d_output<'py, F>(
    py: pyo3::Python<'py>,
    len: usize,
    fill: F,
) -> pyo3::PyResult<numpy::PyReadonlyArray1<'py, f64>>
where
    F: FnOnce(&mut [f64]) -> pyo3::PyResult<()>,
{
    let out = numpy::PyArray1::<f64>::zeros(py, len, false);
    {
        let mut writable = numpy::PyArrayMethods::try_readwrite(&out).map_err(|error| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "rextio-numpy: fresh f64 rank-1 output borrow failed: {error}"
            ))
        })?;
        let values = writable.as_slice_mut().map_err(|error| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "rextio-numpy: fresh f64 rank-1 output is not contiguous: {error}"
            ))
        })?;
        fill(values)?;
    }
    numpy::PyArrayMethods::try_into_readonly(out).map_err(|error| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "rextio-numpy: fresh f64 rank-1 output freeze failed: {error}"
        ))
    })
}"""


def f64_1d_return_helper() -> str:
    """Release the native read-only borrow before returning its NumPy owner."""
    return """\
fn __rxtnp_release_f64_1d<'py>(
    value: numpy::PyReadonlyArray1<'py, f64>,
) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<f64>>> {
    let owner = (*value).clone();
    drop(value);
    Ok(owner)
}"""


def f64_1d_output_support() -> tuple[str, ...]:
    """Return exact-text support required by any F64_1D-producing claim."""
    return (f64_1d_output_helper(),)


__all__ = [
    "F64_1D_OUTPUT_HELPER_NAME",
    "F64_1D_RETURN_HELPER_NAME",
    "F64_1D_RUST",
    "array_rust_type",
    "f64_1d_output_helper",
    "f64_1d_output_support",
    "f64_1d_return_helper",
    "is_python_backed",
    "lifetime_decl",
    "readonly_view_line",
]
