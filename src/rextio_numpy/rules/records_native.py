"""Native (verified) rule records for the rextio-numpy lowering surface."""

from __future__ import annotations

from rextio.plugins.api import RuleRecord, RuleScope

NATIVE_RECORDS: tuple[RuleRecord, ...] = (
    RuleRecord(
        id="rextio-numpy/elementwise-float64",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="binop",
            pattern=(
                "element-wise +, -, *, / on float64/float32/int64 arrays of rank 1 or 2 "
                "(same-dtype array-array with NumPy broadcasting, array-scalar, and "
                "scalar-array forms; operand order preserved for - and /)"
            ),
        ),
        constraint=(
            "Element-wise arithmetic on same-dtype float64/float32/int64 ndarrays of "
            "rank 1 or 2 (array-array under full NumPy broadcasting for ranks 1–2 "
            "including zero-size axes, or array-scalar/scalar-array with a matching "
            "Python scalar — float for floating arrays, int for int64 — operand order "
            "preserved for - and /) maps to ndarray-crate operations. Floating ops "
            "follow IEEE-754; int64 +,-,* use wraparound arithmetic matching NumPy "
            "release builds; int64 true division (/) yields float64 arrays at the "
            "broadcast result rank. Mixed array dtypes and other ranks/dtypes are "
            "rejected. Documented divergence: the native lowering emits no NumPy "
            "RuntimeWarnings (e.g. divide-by-zero or invalid-value warnings); result "
            "values match."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-001",
        guidance=(
            "Keep hot array math to same-dtype float64/float32/int64 arrays of rank "
            "1 or 2 and plain element-wise operators; annotate array parameters and "
            "returns so shapes and dtypes resolve statically."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-numpy/dot-float64",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern=(
                "numpy.dot(a, b) on same-dtype 1-D float64/int64 arrays "
                "(module-call form; float32, 2-D, and matmul/@ are not claimed)"
            ),
        ),
        constraint=(
            "Vector-vector dot products on same-dtype 1-D float64/int64 arrays lower "
            "natively for any length (claim metadata cannot encode a size bound). "
            "int64 dots use wraparound accumulation matching NumPy release builds. "
            "float32 1-D dots are rejected at claim time (RXTP-NUMPY-010) and stay on "
            "the Python fallback: sequential f32 accumulation diverges materially from "
            "NumPy pairwise summation on long/mixed-magnitude inputs, and the plugin "
            "API has no enforceable runtime length gate or fallback hook. 2-D "
            "operands, mixed dtypes, and the @ operator stay unclaimed. Documented "
            "divergences for claimed float64 dots: float summation order may differ "
            "from NumPy's pairwise summation (verified means within-tolerance, not "
            "bit-equivalence); certified within 1e-12 relative/absolute tolerance. "
            "The native leg returns a builtin float (or int for int64) where NumPy "
            "returns a numpy scalar subclass (type()/repr/.dtype observably differ)."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-002",
        guidance=(
            "Use numpy.dot directly on same-dtype 1-D float64/int64 arrays; float32 "
            "dots, the @ operator, and 2-D matmul forms are not lowered by "
            "rextio-numpy, so they stay on the fallback. Avoid dtype-mixing operands "
            "(cast explicitly first)."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-numpy/reduction-sum-mean",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern=(
                "numpy.sum over a whole float64/int64 array of rank 1 or 2, or "
                "numpy.mean over a whole float64 array of rank 1 or 2 "
                "(module-call form with no keywords; float32 whole-array reductions, "
                "int64 mean, ndarray method forms, bare max/min, and non-literal "
                "axis forms are not claimed under this rule)"
            ),
        ),
        constraint=(
            "Full-array sum reductions on float64/int64 arrays of rank 1 or 2, and "
            "full-array mean on float64 arrays of rank 1 or 2, lower natively for any "
            "length (claim metadata cannot encode a size bound). float32 whole-array "
            "sum/mean and int64 mean are rejected at claim time (RXTP-NUMPY-010) and "
            "stay on the Python fallback: sequential f32 accumulation diverges "
            "materially from NumPy pairwise summation on long/mixed-magnitude inputs; "
            "int64 mean's sequential i64→f64 cast-and-sum diverges from NumPy pairwise "
            "mean on large integers near the float64 mantissa boundary (e.g. "
            "tile([2**53, 1, -2**53], n) → NumPy ~0.11 vs sequential 0.0), and the "
            "plugin API has no enforceable runtime length gate or fallback hook. "
            "Literal single-axis reductions live under rextio-numpy/reduction-axis "
            "(RXTP-NUMPY-004). The a.sum()/a.mean() method forms stay on the fallback. "
            "Result dtypes follow NumPy 2.4 practical semantics: int64 sum is int64 "
            "(wraparound under overflow); float64 sum/mean return a builtin Python "
            "float (not a NumPy scalar subclass — type()/repr/.dtype observably "
            "differ). Empty-array mean returns nan on both legs without NumPy's "
            "RuntimeWarning on the native leg. Documented divergences for claimed "
            "float64 whole-array reductions: float summation order may differ from "
            "NumPy's pairwise summation (verified means within-tolerance, not "
            "bit-equivalence); certified within 1e-12 relative/absolute tolerance. "
            "Native reductions also omit NumPy RuntimeWarnings for invalid ops such "
            "as +inf + -inf."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-003",
        guidance=(
            "Prefer whole-array float64 sum/mean and float64/int64 sum in hot paths; "
            "cast float32 arrays to float64 at the boundary if a native reduction is "
            "required, or keep float32 reductions and int64 mean on the Python "
            "fallback. For single-axis reductions use numpy.sum/mean/max/min(a, "
            "axis=<int literal>) — see rextio-numpy/reduction-axis."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-numpy/reduction-axis",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern=(
                "numpy.sum/mean/max/min(a, axis=<int literal>) with exactly one "
                "positional array and exactly one named axis keyword "
                "(module-call form; ranks 1–2; see constraint for dtype matrix)"
            ),
        ),
        constraint=(
            "Single-axis reductions with a static signed integer axis literal "
            "(normalized against rank at claim time; out-of-range axes are not "
            "claimed). Accepted forms: exactly one positional array argument and "
            "exactly the keyword axis=<int>; no dtype/out/keepdims/initial/where, "
            "no positional axis, no axis=None, no tuple axis, no dynamic axis, no "
            "duplicate/extra keywords, no method forms, no numpy.amax/amin. "
            "Dtype/rank matrix: sum on float64/int64 ranks 1–2; mean on float64 "
            "ranks 1–2; max/min on float64/int64 ranks 1–2 and float32 rank 2 only "
            "(rank-1 float32 max/min stay fallback so a core float scalar cannot "
            "erase numpy.float32 scalar semantics). float32 sum/mean and int64 mean "
            "remain RXTP-NUMPY-010. Rank-1 admitted reductions return a core scalar; "
            "rank-2 single-axis reductions return the matching rank-1 plugin array. "
            "int64 sum wraps at every addition. Literal-axis float64 sum/mean use a "
            "NumPy-compatible reduction: rank-1 and rank-2 fast-stride lanes "
            "(element stride of the reduced axis equals 1) use NumPy's partial "
            "pairwise summation (PW_BLOCKSIZE=128, 8-way unroll, recursive split); "
            "rank-2 slow-stride lanes use left-to-right sequential summation. "
            "Dispatch is decided at runtime from ndarray strides so C-/F-contiguous "
            "and other strided layouts match NumPy; mean is compatible-sum / length. "
            "Cancellation patterns such as tile([1e16, 1.0, -1e16], n) match NumPy "
            "within 1e-12 on every supported axis orientation. Whole-array sum/mean "
            "keep the separate ndarray route. Float extrema preserve the first NaN "
            "encountered in logical order (sign and payload) and implement "
            "max(+0,-0)=+0 and min(+0,-0)=-0 (not Rust f32/f64 min/max). max/min over "
            "an empty reduced dimension raise PyValueError with NumPy-compatible "
            "'maximum/minimum which has no identity' text. Empty sum/mean lanes match "
            "existing value semantics; native mean of an empty reduced lane returns "
            "nan without NumPy's RuntimeWarning. Rank-1 native scalar results are "
            "builtin float/int, not NumPy scalar subclasses. Native reductions do not "
            "reproduce NumPy RuntimeWarning behavior (empty mean, invalid ops such as "
            "+inf + -inf). Certification is within 1e-12 relative/absolute tolerance "
            "for float values — not a claim of universal bit-equivalence. Inputs are "
            "not mutated."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-004",
        guidance=(
            "Write numpy.sum/mean/max/min(a, axis=<int literal>) with a static "
            "integer axis on float64/int64 ranks 1–2 (or float32 rank-2 max/min). "
            "Cast float32 sum/mean and int64 mean to float64 at the boundary if a "
            "native reduction is required. Keep method forms, amax/amin, tuple "
            "axes, and keepdims/out kwargs on the Python fallback."
        ),
        stability="experimental",
        verified=True,
    ),
)
