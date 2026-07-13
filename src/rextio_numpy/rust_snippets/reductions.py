"""Rust helper-text for whole-array reduction lowering (sum/mean)."""

from __future__ import annotations

_ARR = {
    ("f64", 1): "numpy::ndarray::Array1<f64>",
    ("f64", 2): "numpy::ndarray::Array2<f64>",
    ("f32", 1): "numpy::ndarray::Array1<f32>",
    ("f32", 2): "numpy::ndarray::Array2<f32>",
    ("i64", 1): "numpy::ndarray::Array1<i64>",
    ("i64", 2): "numpy::ndarray::Array2<i64>",
}


def sum1() -> str:
    """Return the whole-array sum helper for f64 rank-1 (compat entry point)."""
    return sum_typed("f64", 1)


def mean1() -> str:
    """Return the whole-array mean helper for f64 rank-1 (compat entry point).

    NumPy's mean of an empty array warns (RuntimeWarning) and returns nan;
    ``ndarray`` returns ``None`` from ``mean()`` on empty input, mapped to nan
    here. Value equivalence holds; the missing warning is the documented
    divergence recorded on rule ``rextio-numpy/reduction-sum-mean``.
    """
    return mean_typed("f64", 1)


def sum_typed(dtype: str, rank: int) -> str:
    """Return whole-array sum helper for ``dtype`` at ``rank``."""
    arr = _ARR[(dtype, rank)]
    if dtype == "f64" and rank == 1:
        name = "__rxtnp_sum1"
    else:
        name = f"__rxtnp_sum{rank}_{dtype}"

    if dtype == "i64":
        # Match NumPy release-mode wraparound under debug Cargo.
        body = "    Ok(a.iter().fold(0i64, |acc, &x| acc.wrapping_add(x)))\n"
        ret = "i64"
    elif dtype == "f32":
        # Accumulate in f32 then widen to the core float result type.
        body = "    Ok(a.sum() as f64)\n"
        ret = "f64"
    else:
        body = "    Ok(a.sum())\n"
        ret = "f64"

    return f"fn {name}(a: &{arr}) -> pyo3::PyResult<{ret}> {{\n{body}}}"


def mean_typed(dtype: str, rank: int) -> str:
    """Return whole-array mean helper for ``dtype`` at ``rank``."""
    arr = _ARR[(dtype, rank)]
    if dtype == "f64" and rank == 1:
        name = "__rxtnp_mean1"
    else:
        name = f"__rxtnp_mean{rank}_{dtype}"

    if dtype == "i64":
        # Dead lower path retained for architecture symmetry: claim rejects
        # int64 mean (RXTP-NUMPY-010) because sequential i64→f64 cast-and-sum
        # diverges from NumPy pairwise mean on large integers. Do not re-claim
        # without a correct pairwise/compensated algorithm.
        body = (
            "    if a.len() == 0 {\n"
            "        return Ok(f64::NAN);\n"
            "    }\n"
            "    let sum: f64 = a.iter().map(|&x| x as f64).sum();\n"
            "    Ok(sum / (a.len() as f64))\n"
        )
    elif dtype == "f32":
        body = "    Ok(a.mean().unwrap_or(f32::NAN) as f64)\n"
    else:
        body = "    Ok(a.mean().unwrap_or(f64::NAN))\n"

    return f"fn {name}(a: &{arr}) -> pyo3::PyResult<f64> {{\n{body}}}"


def sum_call_name(dtype: str, rank: int) -> str:
    """Return the Rust helper function name for sum."""
    if dtype == "f64" and rank == 1:
        return "__rxtnp_sum1"
    return f"__rxtnp_sum{rank}_{dtype}"


def mean_call_name(dtype: str, rank: int) -> str:
    """Return the Rust helper function name for mean."""
    if dtype == "f64" and rank == 1:
        return "__rxtnp_mean1"
    return f"__rxtnp_mean{rank}_{dtype}"
