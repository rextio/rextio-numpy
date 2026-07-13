//! Standalone research candidate for Wave 2 rank-2 f64 matmul.
//!
//! Boundary contract (frozen protocol):
//! - `PyReadonlyArray2<f64>` for both arguments
//! - `as_array().to_owned()` for each input (pay the ownership boundary cost)
//! - `numpy::ndarray::Array2::dot` for the multiply
//! - `ToPyArray` / `to_pyarray` for the output
//!
//! No BLAS feature, no GIL-release timing tricks, no product rule registration.

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

/// Rank-2 f64 matrix product with an explicit ownership boundary.
///
/// Both inputs are copied into owned `Array2<f64>` values before `dot`,
/// matching the end-to-end call cost a product native path would pay.
#[pyfunction]
fn matmul_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let a_owned: Array2<f64> = a.as_array().to_owned();
    let b_owned: Array2<f64> = b.as_array().to_owned();
    let c: Array2<f64> = a_owned.dot(&b_owned);
    c.to_pyarray(py)
}

/// Python module entry point. Module name is fixed; the harness places the
/// built artifact under a run-specific directory on `sys.path`.
#[pymodule]
fn matmul_wave2_candidate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul_f64, m)?)?;
    Ok(())
}
