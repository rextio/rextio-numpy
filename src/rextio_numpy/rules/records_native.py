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
                "(module-call form; float32 whole-array reductions, int64 mean, "
                "ndarray method forms, and axis=/kwargs are not claimed)"
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
            "axis= keyword reductions and the a.sum()/a.mean() method forms stay on "
            "the fallback. Result dtypes follow NumPy 2.4 practical semantics: int64 "
            "sum is int64 (wraparound under overflow); float64 sum/mean return a "
            "builtin float. Empty-array mean returns nan on both legs without NumPy's "
            "RuntimeWarning on the native leg. Documented divergences for claimed "
            "float64 reductions: float summation order may differ from NumPy's "
            "pairwise summation (verified means within-tolerance, not bit-equivalence); "
            "certified within 1e-12 relative/absolute tolerance."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-003",
        guidance=(
            "Prefer whole-array float64 sum/mean and float64/int64 sum in hot paths; "
            "cast float32 arrays to float64 at the boundary if a native reduction is "
            "required, or keep float32 reductions and int64 mean on the Python "
            "fallback. Hoist axis-wise reductions out of native candidates until axis "
            "support lands."
        ),
        stability="experimental",
        verified=True,
    ),
)
