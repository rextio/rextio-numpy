//! Standalone research candidate for F64 rank-1 NumPy boundary-allocation PoC.
//!
//! Compares three elementwise-add strategies. None of these are product rules;
//! production BoundaryConversion / lowering is not modified by this crate.
//!
//! Shared boundary gates (all strategies):
//! - exact `numpy.ndarray` check via `is_exact_instance_of::<PyArray1<f64>>`
//! - reject ndarray subclasses with the product-matching TypeError text
//!
//! Strategies:
//! 1. `add_owned_topy` — two owned input copies + owned Rust output + `ToPyArray`
//! 2. `add_borrowed_topy` — borrowed `PyReadonlyArray` views + owned Rust output + `ToPyArray`
//! 3. `add_direct_sink` — borrowed views + fill a NumPy-owned output buffer, return unchanged
//!
//! **Never** uses `IntoPyArray` (would transfer Rust ownership and break ordinary
//! resize / OWNDATA observables).

use numpy::ndarray::{Array1, ArrayView1, Zip};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods; // is_exact_instance_of

const EXACT_NDARRAY_MSG: &str = "rextio-numpy native boundary requires exact numpy.ndarray; \
     ndarray subclasses are unsupported";

/// Fail closed unless *arr* is exactly `numpy.ndarray` (not a subclass).
fn require_exact_ndarray1(arr: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
    // PyReadonlyArray derefs to Bound<PyArray1>; check the exact Python type.
    if !arr.is_exact_instance_of::<PyArray1<f64>>() {
        return Err(PyTypeError::new_err(EXACT_NDARRAY_MSG));
    }
    Ok(())
}

/// NumPy-compatible rank-1 broadcast shape (equal lengths, length-1 either side).
fn broadcast_len(a_len: usize, b_len: usize) -> PyResult<usize> {
    if a_len == b_len {
        return Ok(a_len);
    }
    if a_len == 1 {
        return Ok(b_len);
    }
    if b_len == 1 {
        return Ok(a_len);
    }
    Err(PyValueError::new_err(format!(
        "operands could not be broadcast together with shapes ({},) ({},) ",
        a_len, b_len
    )))
}

/// Elementwise add into a newly allocated owned `Array1<f64>` from two views.
fn add_views_owned(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> PyResult<Array1<f64>> {
    let _n = broadcast_len(a.len(), b.len())?;
    if a.len() == b.len() {
        // Equal-length path (including both length-0 and both length-1).
        return Ok(&a + &b);
    }
    if a.len() == 1 {
        let s = a[0];
        return Ok(b.mapv(|x| s + x));
    }
    // b.len() == 1 (guaranteed by broadcast_len)
    let s = b[0];
    Ok(a.mapv(|x| x + s))
}

/// Fill *out* (length n) from broadcast-compatible views. Every element written.
fn fill_add_views(out: &mut [f64], a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> PyResult<()> {
    let n = broadcast_len(a.len(), b.len())?;
    if out.len() != n {
        return Err(PyValueError::new_err(format!(
            "boundary-allocation-poc: sink length {} != broadcast length {}",
            out.len(),
            n
        )));
    }
    if a.len() == b.len() {
        // Equal length (contiguous or strided views); Zip indexes each view.
        Zip::from(out)
            .and(a)
            .and(b)
            .for_each(|o, &x, &y| *o = x + y);
        return Ok(());
    }
    if a.len() == 1 {
        let s = a[0];
        for (i, o) in out.iter_mut().enumerate() {
            *o = s + b[i];
        }
        return Ok(());
    }
    // b.len() == 1
    let s = b[0];
    for (i, o) in out.iter_mut().enumerate() {
        *o = a[i] + s;
    }
    Ok(())
}

/// Strategy 1: current product-style boundary (owned copies + ToPyArray).
///
/// Logical N-sized allocations (equal-length contiguous, output length N):
///   2 input `to_owned` + 1 Rust result + 1 `ToPyArray` Python buffer = **4**.
#[pyfunction]
fn add_owned_topy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_exact_ndarray1(&*a)?;
    require_exact_ndarray1(&*b)?;
    let a_owned: Array1<f64> = a.as_array().to_owned();
    let b_owned: Array1<f64> = b.as_array().to_owned();
    let out = add_views_owned(a_owned.view(), b_owned.view())?;
    // Explicit ToPyArray UFCS (never IntoPyArray).
    Ok(ToPyArray::to_pyarray(&out, py))
}

/// Strategy 2: borrow inputs; still materialize owned Rust result + ToPyArray.
///
/// Logical N-sized allocations (equal-length contiguous, output length N):
///   0 input copies + 1 Rust result + 1 `ToPyArray` Python buffer = **2**.
#[pyfunction]
fn add_borrowed_topy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_exact_ndarray1(&*a)?;
    require_exact_ndarray1(&*b)?;
    let a_view = a.as_array();
    let b_view = b.as_array();
    let out = add_views_owned(a_view, b_view)?;
    Ok(ToPyArray::to_pyarray(&out, py))
}

/// Strategy 3: borrow inputs; allocate one NumPy-owned buffer; fill; return it.
///
/// Logical N-sized allocations (equal-length contiguous, output length N):
///   0 input copies + 1 NumPy-owned output = **1**.
///
/// Ownership contract (must hold for ordinary NumPy results):
/// - exact `numpy.ndarray` (not a subclass)
/// - `OWNDATA` true, `base is None`
/// - ordinary in-place `resize` when NumPy allows it on a fresh owned array
/// - every element initialized before return (no uninit escape)
///
/// Deliberately avoids `IntoPyArray` (Rust-owned base breaks resize).
#[pyfunction]
fn add_direct_sink<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_exact_ndarray1(&*a)?;
    require_exact_ndarray1(&*b)?;
    let a_view = a.as_array();
    let b_view = b.as_array();
    let n = broadcast_len(a_view.len(), b_view.len())?;

    // NumPy-owned allocation (Python heap). Prefer `zeros` over `new` so
    // partial failure paths never observe uninit. Zero-initialization is an
    // extra store pass over N elements before the fill overwrite below; it is
    // not a second logical N-sized allocation, but it is real memory traffic.
    // Every element is still overwritten before return. Never use IntoPyArray.
    let out = PyArray1::<f64>::zeros(py, n, false);
    {
        let mut rw = out.readwrite();
        // Contiguous C-order sink from zeros: as_slice_mut is available.
        let slice = rw.as_slice_mut().map_err(|e| {
            PyValueError::new_err(format!(
                "boundary-allocation-poc: direct sink not contiguous: {e}"
            ))
        })?;
        fill_add_views(slice, a_view, b_view)?;
    }
    // Return the same NumPy-owned object unchanged (no ToPyArray copy, no IntoPyArray).
    Ok(out)
}

#[pymodule]
fn boundary_allocation_poc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_owned_topy, m)?)?;
    m.add_function(wrap_pyfunction!(add_borrowed_topy, m)?)?;
    m.add_function(wrap_pyfunction!(add_direct_sink, m)?)?;
    // Documented strategy ids for the harness (import-time constants).
    m.add("STRATEGY_OWNED_TOPY", "owned_topy")?;
    m.add("STRATEGY_BORROWED_TOPY", "borrowed_topy")?;
    m.add("STRATEGY_DIRECT_SINK", "direct_sink")?;
    Ok(())
}
