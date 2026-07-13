"""Rust helper-text for whole-array reduction lowering (sum/mean)."""

from __future__ import annotations

_ARR = "numpy::ndarray::Array1<f64>"


def sum1() -> str:
    """Return the whole-array sum helper."""
    return f"fn __rxtnp_sum1(a: &{_ARR}) -> pyo3::PyResult<f64> {{ Ok(a.sum()) }}"


def mean1() -> str:
    """Return the whole-array mean helper.

    NumPy's mean of an empty array warns (RuntimeWarning) and returns nan;
    ``ndarray`` returns ``None`` from ``mean()`` on empty input, mapped to nan
    here. Value equivalence holds; the missing warning is the documented
    divergence recorded on rule ``rextio-numpy/reduction-sum-mean``.
    """
    return f"fn __rxtnp_mean1(a: &{_ARR}) -> pyo3::PyResult<f64> {{ Ok(a.mean().unwrap_or(f64::NAN)) }}"
