"""Rust helper-text for whole-array and literal-axis reduction lowering.

Whole-array sum/mean keep the historical ndarray helpers (Wave 1).

Literal-axis f64 sum/mean match NumPy's axis-reduction accumulation:

* **unit-stride** lanes along the reduced axis (rank-1 slices; C-order
  rows / F-order columns) use NumPy's partial pairwise sum
  (``PW_BLOCKSIZE = 128``, 8-way unroll, recursive half-split);
* **non-unit-stride** lanes use left-to-right sequential sum — the same
  order NumPy's multi-column C-order ``sum(axis=0)`` uses (pairwise only
  when the reduced axis is contiguous).

Mean is (compatible sum) / length. Empty sum is 0.0; empty mean is nan
(no RuntimeWarning on the native leg).
"""

from __future__ import annotations

_ARR = {
    ("f64", 1): "numpy::ndarray::Array1<f64>",
    ("f64", 2): "numpy::ndarray::Array2<f64>",
    ("f32", 1): "numpy::ndarray::Array1<f32>",
    ("f32", 2): "numpy::ndarray::Array2<f32>",
    ("i64", 1): "numpy::ndarray::Array1<i64>",
    ("i64", 2): "numpy::ndarray::Array2<i64>",
}

_SCALAR = {
    "f64": "f64",
    "f32": "f32",
    "i64": "i64",
}

# NumPy-compatible empty-extrema messages (exact text, ValueError).
_MAX_EMPTY_MSG = "zero-size array to reduction operation maximum which has no identity"
_MIN_EMPTY_MSG = "zero-size array to reduction operation minimum which has no identity"


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
    """Return the Rust helper function name for whole-array sum."""
    if dtype == "f64" and rank == 1:
        return "__rxtnp_sum1"
    return f"__rxtnp_sum{rank}_{dtype}"


def mean_call_name(dtype: str, rank: int) -> str:
    """Return the Rust helper function name for whole-array mean."""
    if dtype == "f64" and rank == 1:
        return "__rxtnp_mean1"
    return f"__rxtnp_mean{rank}_{dtype}"


def extrema_call_name(op: str, dtype: str, rank: int) -> str:
    """Return the Rust helper name for a whole-array integer extremum."""
    if op not in {"max", "min"} or dtype != "i64" or rank not in {1, 2}:
        raise ValueError(
            f"unsupported whole-array extremum: {op!r}/{dtype!r}/rank-{rank}"
        )
    return f"__rxtnp_{op}{rank}_{dtype}"


def extrema_typed(op: str, dtype: str, rank: int) -> str:
    """Return an exact whole-array int64 min/max helper."""
    name = extrema_call_name(op, dtype, rank)
    arr = _ARR[(dtype, rank)]
    iterator_op = "max" if op == "max" else "min"
    message = _MAX_EMPTY_MSG if op == "max" else _MIN_EMPTY_MSG
    return (
        f"fn {name}(a: &{arr}) -> pyo3::PyResult<i64> {{\n"
        f"    a.iter().copied().{iterator_op}().ok_or_else(|| "
        f'pyo3::exceptions::PyValueError::new_err("{message}"))\n'
        "}"
    )


def axis_call_name(op: str, dtype: str, rank: int, axis: int) -> str:
    """Return the Rust helper name for a normalized single-axis reduction.

    ``axis`` must already be normalized to ``0 .. rank-1``. The normalized
    axis is part of the helper identity so lower() is deterministic.
    """
    return f"__rxtnp_{op}{rank}_{dtype}_axis{axis}"


def _numpy_pairwise_sum_helpers() -> tuple[str, str]:
    """Return (contiguous pairwise, strided pairwise) NumPy f64 sum helpers.

    Faithful port of NumPy's ``DOUBLE_pairwise_sum`` (``PW_BLOCKSIZE = 128``,
    8-way unrolled block, recursive half-split floored to a multiple of 8).
    Not Kahan/Neumaier. Empty length returns ``0.0``.
    """
    contig = """\
fn __rxtnp_numpy_pairwise_sum_f64(data: &[f64]) -> f64 {
    const PW_BLOCKSIZE: usize = 128;
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    if n < 8 {
        let mut res = -0.0f64;
        for &x in data {
            res += x;
        }
        return res;
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; 8];
        r[0] = data[0];
        r[1] = data[1];
        r[2] = data[2];
        r[3] = data[3];
        r[4] = data[4];
        r[5] = data[5];
        r[6] = data[6];
        r[7] = data[7];
        let mut i: usize = 8;
        let lim = n - (n % 8);
        while i < lim {
            r[0] += data[i];
            r[1] += data[i + 1];
            r[2] += data[i + 2];
            r[3] += data[i + 3];
            r[4] += data[i + 4];
            r[5] += data[i + 5];
            r[6] += data[i + 6];
            r[7] += data[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += data[i];
            i += 1;
        }
        return res;
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        return __rxtnp_numpy_pairwise_sum_f64(&data[..n2])
            + __rxtnp_numpy_pairwise_sum_f64(&data[n2..]);
    }
}"""
    # Strided logical-index pairwise: same recursion/block structure, loads via
    # a closure so non-unit ndarray strides still match NumPy's stride walk.
    strided = """\
fn __rxtnp_numpy_pairwise_sum_f64_strided(n: usize, get: &dyn Fn(usize) -> f64) -> f64 {
    const PW_BLOCKSIZE: usize = 128;
    if n == 0 {
        return 0.0;
    }
    if n < 8 {
        let mut res = -0.0f64;
        for i in 0..n {
            res += get(i);
        }
        return res;
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; 8];
        for k in 0..8 {
            r[k] = get(k);
        }
        let mut i: usize = 8;
        let lim = n - (n % 8);
        while i < lim {
            for k in 0..8 {
                r[k] += get(i + k);
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += get(i);
            i += 1;
        }
        return res;
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        let left = __rxtnp_numpy_pairwise_sum_f64_strided(n2, get);
        let right = __rxtnp_numpy_pairwise_sum_f64_strided(n - n2, &|i| get(i + n2));
        return left + right;
    }
}

fn __rxtnp_numpy_sequential_sum_f64(n: usize, get: &dyn Fn(usize) -> f64) -> f64 {
    // Matches NumPy's non-unit-stride axis reduction order (not pairwise).
    if n == 0 {
        return 0.0;
    }
    let mut res = -0.0f64;
    for i in 0..n {
        res += get(i);
    }
    res
}"""
    return contig, strided


def _float_pairwise_helper(dtype: str, op: str) -> str:
    """Return a pairwise NumPy-semantics max/min helper for ``f32``/``f64``.

    Preserves the **first** NaN encountered in left-fold order (sign and
    payload): if ``a`` is NaN return ``a``, else if ``b`` is NaN return ``b``.
    ``max(+0, -0) = +0`` and ``min(+0, -0) = -0``. Does not use Rust's
    ``f32``/``f64`` ``min``/``max``.
    """
    ty = _SCALAR[dtype]
    name = f"__rxtnp_numpy_{op}_{dtype}"
    if op == "max":
        zero_branch = (
            f"        if a.is_sign_positive() || b.is_sign_positive() {{\n"
            f"            0.0{'' if dtype == 'f64' else '_f32'}\n"
            f"        }} else {{\n"
            f"            -0.0{'' if dtype == 'f64' else '_f32'}\n"
            f"        }}\n"
        )
        cmp_branch = "        if a >= b { a } else { b }\n"
    else:
        zero_branch = (
            f"        if a.is_sign_negative() || b.is_sign_negative() {{\n"
            f"            -0.0{'' if dtype == 'f64' else '_f32'}\n"
            f"        }} else {{\n"
            f"            0.0{'' if dtype == 'f64' else '_f32'}\n"
            f"        }}\n"
        )
        cmp_branch = "        if a <= b { a } else { b }\n"
    return (
        f"fn {name}(a: {ty}, b: {ty}) -> {ty} {{\n"
        f"    if a.is_nan() {{\n"
        f"        a\n"
        f"    }} else if b.is_nan() {{\n"
        f"        b\n"
        f"    }} else if a == 0.0 && b == 0.0 {{\n"
        f"{zero_branch}"
        f"    }} else {{\n"
        f"{cmp_branch}"
        f"    }}\n"
        f"}}"
    )


def _empty_extrema_err(op: str) -> str:
    msg = _MAX_EMPTY_MSG if op == "max" else _MIN_EMPTY_MSG
    return (
        "        return Err(pyo3::exceptions::PyValueError::new_err(\n"
        f'            "{msg}"\n'
        "        ));\n"
    )


def _rank1_f64_sum_body() -> str:
    return (
        "    if let Some(sl) = a.as_slice() {\n"
        "        Ok(__rxtnp_numpy_pairwise_sum_f64(sl))\n"
        "    } else {\n"
        "        let n = a.len();\n"
        "        Ok(__rxtnp_numpy_pairwise_sum_f64_strided(n, &|i| a[i]))\n"
        "    }\n"
    )


def _rank1_f64_mean_body() -> str:
    return (
        "    let n = a.len();\n"
        "    if n == 0 {\n"
        "        return Ok(f64::NAN);\n"
        "    }\n"
        "    let s = if let Some(sl) = a.as_slice() {\n"
        "        __rxtnp_numpy_pairwise_sum_f64(sl)\n"
        "    } else {\n"
        "        __rxtnp_numpy_pairwise_sum_f64_strided(n, &|i| a[i])\n"
        "    };\n"
        "    Ok(s / (n as f64))\n"
    )


def _rank1_axis_body(op: str, dtype: str) -> tuple[str, str, tuple[str, ...]]:
    """Return (body, ret_type, extra_helpers) for rank-1 axis reduction → scalar."""
    if op == "sum":
        if dtype == "i64":
            return (
                "    Ok(a.iter().fold(0i64, |acc, &x| acc.wrapping_add(x)))\n",
                "i64",
                (),
            )
        if dtype == "f64":
            contig, strided = _numpy_pairwise_sum_helpers()
            return (_rank1_f64_sum_body(), "f64", (contig, strided))
        return ("    Ok(a.sum())\n", "f64", ())
    if op == "mean":
        if dtype == "f64":
            contig, strided = _numpy_pairwise_sum_helpers()
            return (_rank1_f64_mean_body(), "f64", (contig, strided))
        return ("    Ok(a.mean().unwrap_or(f64::NAN))\n", "f64", ())
    # max / min
    if dtype == "i64":
        fold = "max" if op == "max" else "min"
        body = (
            "    if a.is_empty() {\n"
            f"{_empty_extrema_err(op)}"
            "    }\n"
            f'    Ok(*a.iter().{fold}().expect("non-empty"))\n'
        )
        return (body, "i64", ())
    pair = _float_pairwise_helper(dtype, op)
    pair_name = f"__rxtnp_numpy_{op}_{dtype}"
    if dtype == "f64":
        body = (
            "    if a.is_empty() {\n"
            f"{_empty_extrema_err(op)}"
            "    }\n"
            f"    let mut acc = a[0];\n"
            f"    for &x in a.iter().skip(1) {{\n"
            f"        acc = {pair_name}(acc, x);\n"
            f"    }}\n"
            f"    Ok(acc)\n"
        )
        return (body, "f64", (pair,))
    body = (
        "    if a.is_empty() {\n"
        f"{_empty_extrema_err(op)}"
        "    }\n"
        f"    let mut acc = a[0];\n"
        f"    for &x in a.iter().skip(1) {{\n"
        f"        acc = {pair_name}(acc, x);\n"
        f"    }}\n"
        f"    Ok(acc as f64)\n"
    )
    return (body, "f64", (pair,))


def _rank2_f64_lane_sum(axis: int) -> str:
    """Emit body that fills ``out`` with per-lane f64 sums along ``axis``.

    NumPy uses partial pairwise when the reduced axis is unit-stride (fast)
    and left-to-right sequential when the reduced axis is slow-stride.
    Runtime ``stride_of(Axis(axis)) == 1`` selects the path so C- and
    F-contiguous inputs match (boundary conversion preserves layout).
    """
    if axis == 0:
        # Reduce rows → one value per column. Fast when column stride is 1 (F).
        return (
            "    let nrows = a.nrows();\n"
            "    let ncols = a.ncols();\n"
            "    let mut out = numpy::ndarray::Array1::<f64>::zeros(ncols);\n"
            "    let fast = a.stride_of(numpy::ndarray::Axis(0)) == 1;\n"
            "    for j in 0..ncols {\n"
            "        let s = if fast {\n"
            "            __rxtnp_numpy_pairwise_sum_f64_strided(nrows, &|i| a[[i, j]])\n"
            "        } else {\n"
            "            __rxtnp_numpy_sequential_sum_f64(nrows, &|i| a[[i, j]])\n"
            "        };\n"
            "        out[j] = s;\n"
            "    }\n"
            "    Ok(out)\n"
        )
    # axis == 1: reduce cols → one value per row. Fast when row stride is 1 (C).
    return (
        "    let nrows = a.nrows();\n"
        "    let ncols = a.ncols();\n"
        "    let mut out = numpy::ndarray::Array1::<f64>::zeros(nrows);\n"
        "    let fast = a.stride_of(numpy::ndarray::Axis(1)) == 1;\n"
        "    for i in 0..nrows {\n"
        "        let s = if fast {\n"
        "            __rxtnp_numpy_pairwise_sum_f64_strided(ncols, &|j| a[[i, j]])\n"
        "        } else {\n"
        "            __rxtnp_numpy_sequential_sum_f64(ncols, &|j| a[[i, j]])\n"
        "        };\n"
        "        out[i] = s;\n"
        "    }\n"
        "    Ok(out)\n"
    )


def _rank2_f64_lane_mean(axis: int) -> str:
    """Emit body: compatible sum along ``axis`` divided by reduced length."""
    if axis == 0:
        return (
            "    let nrows = a.nrows();\n"
            "    let ncols = a.ncols();\n"
            "    if nrows == 0 {\n"
            "        return Ok(numpy::ndarray::Array1::from_elem(ncols, f64::NAN));\n"
            "    }\n"
            "    let mut out = numpy::ndarray::Array1::<f64>::zeros(ncols);\n"
            "    let fast = a.stride_of(numpy::ndarray::Axis(0)) == 1;\n"
            "    let denom = nrows as f64;\n"
            "    for j in 0..ncols {\n"
            "        let s = if fast {\n"
            "            __rxtnp_numpy_pairwise_sum_f64_strided(nrows, &|i| a[[i, j]])\n"
            "        } else {\n"
            "            __rxtnp_numpy_sequential_sum_f64(nrows, &|i| a[[i, j]])\n"
            "        };\n"
            "        out[j] = s / denom;\n"
            "    }\n"
            "    Ok(out)\n"
        )
    return (
        "    let nrows = a.nrows();\n"
        "    let ncols = a.ncols();\n"
        "    if ncols == 0 {\n"
        "        return Ok(numpy::ndarray::Array1::from_elem(nrows, f64::NAN));\n"
        "    }\n"
        "    let mut out = numpy::ndarray::Array1::<f64>::zeros(nrows);\n"
        "    let fast = a.stride_of(numpy::ndarray::Axis(1)) == 1;\n"
        "    let denom = ncols as f64;\n"
        "    for i in 0..nrows {\n"
        "        let s = if fast {\n"
        "            __rxtnp_numpy_pairwise_sum_f64_strided(ncols, &|j| a[[i, j]])\n"
        "        } else {\n"
        "            __rxtnp_numpy_sequential_sum_f64(ncols, &|j| a[[i, j]])\n"
        "        };\n"
        "        out[i] = s / denom;\n"
        "    }\n"
        "    Ok(out)\n"
    )


def _rank2_sum_body(dtype: str, axis: int) -> tuple[str, str, tuple[str, ...]]:
    """Return (body, ret_array_type, extras) for rank-2 sum along ``axis``."""
    ret = _ARR[(dtype, 1)]
    if dtype == "i64":
        if axis == 0:
            body = (
                "    let mut out = numpy::ndarray::Array1::<i64>::zeros(a.ncols());\n"
                "    for j in 0..a.ncols() {\n"
                "        let mut acc: i64 = 0;\n"
                "        for i in 0..a.nrows() {\n"
                "            acc = acc.wrapping_add(a[[i, j]]);\n"
                "        }\n"
                "        out[j] = acc;\n"
                "    }\n"
                "    Ok(out)\n"
            )
        else:
            body = (
                "    let mut out = numpy::ndarray::Array1::<i64>::zeros(a.nrows());\n"
                "    for i in 0..a.nrows() {\n"
                "        let mut acc: i64 = 0;\n"
                "        for j in 0..a.ncols() {\n"
                "            acc = acc.wrapping_add(a[[i, j]]);\n"
                "        }\n"
                "        out[i] = acc;\n"
                "    }\n"
                "    Ok(out)\n"
            )
        return body, ret, ()
    # f64 NumPy-compatible
    contig, strided = _numpy_pairwise_sum_helpers()
    return _rank2_f64_lane_sum(axis), ret, (contig, strided)


def _rank2_mean_body(axis: int) -> tuple[str, str, tuple[str, ...]]:
    """Return (body, ret_array_type, extras) for rank-2 f64 mean along ``axis``."""
    ret = _ARR[("f64", 1)]
    contig, strided = _numpy_pairwise_sum_helpers()
    return _rank2_f64_lane_mean(axis), ret, (contig, strided)


def _rank2_extrema_body(op: str, dtype: str, axis: int) -> tuple[str, str, tuple[str, ...]]:
    """Return (body, ret_type, extra_helpers) for rank-2 max/min along ``axis``."""
    ret = _ARR[(dtype, 1)]
    ty = _SCALAR[dtype]
    empty_check = "a.nrows() == 0" if axis == 0 else "a.ncols() == 0"
    out_len = "a.ncols()" if axis == 0 else "a.nrows()"
    if dtype == "i64":
        cmp = ">=" if op == "max" else "<="
        if axis == 0:
            body = (
                f"    if {empty_check} {{\n"
                f"{_empty_extrema_err(op)}"
                f"    }}\n"
                f"    let mut out = numpy::ndarray::Array1::<i64>::zeros({out_len});\n"
                f"    for j in 0..a.ncols() {{\n"
                f"        let mut acc = a[[0, j]];\n"
                f"        for i in 1..a.nrows() {{\n"
                f"            let x = a[[i, j]];\n"
                f"            if x {cmp} acc {{\n"
                f"                acc = x;\n"
                f"            }}\n"
                f"        }}\n"
                f"        out[j] = acc;\n"
                f"    }}\n"
                f"    Ok(out)\n"
            )
        else:
            body = (
                f"    if {empty_check} {{\n"
                f"{_empty_extrema_err(op)}"
                f"    }}\n"
                f"    let mut out = numpy::ndarray::Array1::<i64>::zeros({out_len});\n"
                f"    for i in 0..a.nrows() {{\n"
                f"        let mut acc = a[[i, 0]];\n"
                f"        for j in 1..a.ncols() {{\n"
                f"            let x = a[[i, j]];\n"
                f"            if x {cmp} acc {{\n"
                f"                acc = x;\n"
                f"            }}\n"
                f"        }}\n"
                f"        out[i] = acc;\n"
                f"    }}\n"
                f"    Ok(out)\n"
            )
        return body, ret, ()

    pair = _float_pairwise_helper(dtype, op)
    pair_name = f"__rxtnp_numpy_{op}_{dtype}"
    if axis == 0:
        body = (
            f"    if {empty_check} {{\n"
            f"{_empty_extrema_err(op)}"
            f"    }}\n"
            f"    let mut out = numpy::ndarray::Array1::<{ty}>::zeros({out_len});\n"
            f"    for j in 0..a.ncols() {{\n"
            f"        let mut acc = a[[0, j]];\n"
            f"        for i in 1..a.nrows() {{\n"
            f"            acc = {pair_name}(acc, a[[i, j]]);\n"
            f"        }}\n"
            f"        out[j] = acc;\n"
            f"    }}\n"
            f"    Ok(out)\n"
        )
    else:
        body = (
            f"    if {empty_check} {{\n"
            f"{_empty_extrema_err(op)}"
            f"    }}\n"
            f"    let mut out = numpy::ndarray::Array1::<{ty}>::zeros({out_len});\n"
            f"    for i in 0..a.nrows() {{\n"
            f"        let mut acc = a[[i, 0]];\n"
            f"        for j in 1..a.ncols() {{\n"
            f"            acc = {pair_name}(acc, a[[i, j]]);\n"
            f"        }}\n"
            f"        out[i] = acc;\n"
            f"    }}\n"
            f"    Ok(out)\n"
        )
    return body, ret, (pair,)


def axis_typed(op: str, dtype: str, rank: int, axis: int) -> tuple[str, ...]:
    """Return helper ``fn`` items for a single-axis reduction (main + support).

    ``axis`` must be normalized. Returns one or more module-level items; the
    **last** is the call target named by :func:`axis_call_name` (support
    helpers precede it so they are defined before use in Rust).
    """
    name = axis_call_name(op, dtype, rank, axis)
    arr = _ARR[(dtype, rank)]
    if rank == 1:
        body, ret, extras = _rank1_axis_body(op, dtype)
        main = f"fn {name}(a: &{arr}) -> pyo3::PyResult<{ret}> {{\n{body}}}"
        return (*extras, main)
    # rank == 2
    if op == "sum":
        body, ret, extras = _rank2_sum_body(dtype, axis)
        main = f"fn {name}(a: &{arr}) -> pyo3::PyResult<{ret}> {{\n{body}}}"
        return (*extras, main)
    if op == "mean":
        body, ret, extras = _rank2_mean_body(axis)
        main = f"fn {name}(a: &{arr}) -> pyo3::PyResult<{ret}> {{\n{body}}}"
        return (*extras, main)
    body, ret, extras = _rank2_extrema_body(op, dtype, axis)
    main = f"fn {name}(a: &{arr}) -> pyo3::PyResult<{ret}> {{\n{body}}}"
    return (*extras, main)


def op_from_target(target: str) -> str:
    """Map a module-call target to the short op token used in helper names."""
    return {
        "numpy.sum": "sum",
        "numpy.mean": "mean",
        "numpy.max": "max",
        "numpy.min": "min",
    }[target]
