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
                "numpy.dot(a, b) or a.dot(b) on same-dtype 1-D float64/int64 arrays "
                "(float32, 2-D, keywords, and matmul/@ are not claimed)"
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
            "operands, mixed dtypes, keyword forms, and the @ operator stay unclaimed. "
            "The equivalent ndarray method form a.dot(b) is admitted only through "
            "plugin API 1.3 receiver metadata; core evaluates the receiver exactly "
            "once before b. Documented "
            "divergences for claimed float64 dots: float summation order may differ "
            "from NumPy's pairwise summation (verified means within-tolerance, not "
            "bit-equivalence); certified within 1e-12 relative/absolute tolerance. "
            "The native leg returns a builtin float (or int for int64) where NumPy "
            "returns a numpy scalar subclass (type()/repr/.dtype observably differ)."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-002",
        guidance=(
            "Use numpy.dot(a, b) or a.dot(b) on same-dtype 1-D float64/int64 arrays; float32 "
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
                "numpy.sum(a)/a.sum() over a whole float64/int64 array of rank 1 or 2, "
                "or numpy.mean(a)/a.mean() over a whole float64 array of rank 1 or 2 "
                "(no keywords; float32 whole-array reductions, int64 mean, bare max/min, "
                "and non-literal axis forms are not claimed under this rule)"
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
            "(RXTP-NUMPY-004). The equivalent a.sum()/a.mean() method forms are "
            "admitted through plugin API 1.3 receiver metadata, evaluated exactly "
            "once by core before any positional operands. "
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
                "numpy.sum/mean/max/min(a, axis=<int literal>) or "
                "a.sum/mean/max/min(axis=<int literal>) with exactly one named axis "
                "keyword (ranks 1–2; see constraint for dtype matrix)"
            ),
        ),
        constraint=(
            "Single-axis reductions with a static signed integer axis literal "
            "(normalized against rank at claim time; out-of-range axes are not "
            "claimed). Accepted forms: module calls with exactly one positional array "
            "argument, or ndarray method calls with that array as the API-1.3 receiver, "
            "and in both forms exactly the keyword axis=<int>; no dtype/out/keepdims/initial/where, "
            "no positional axis, no axis=None, no tuple axis, no dynamic axis, no "
            "duplicate/extra keywords, no numpy.amax/amin. Core evaluates a method "
            "receiver exactly once before call operands. "
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
            "Write numpy.sum/mean/max/min(a, axis=<int literal>) or "
            "a.sum/mean/max/min(axis=<int literal>) with a static "
            "integer axis on float64/int64 ranks 1–2 (or float32 rank-2 max/min). "
            "Cast float32 sum/mean and int64 mean to float64 at the boundary if a "
            "native reduction is required. Keep amax/amin, tuple "
            "axes, and keepdims/out kwargs on the Python fallback."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-numpy/unary-module",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern=(
                "exact module calls numpy.negative(a), numpy.absolute(a), numpy.abs(a), "
                "or numpy.square(a) on float64/float32/int64 arrays of rank 1 or 2"
            ),
        ),
        constraint=(
            "The exact one-positional-array module-call forms lower elementwise for "
            "the existing float64/float32/int64 rank-1/rank-2 matrix. No out, where, "
            "dtype, casting, order, subok, signature, or other keyword/positional "
            "overrides are claimed, and ndarray method unary forms remain fallback. "
            "Floating operations preserve NumPy value semantics for signed zero, NaN, "
            "and infinity. int64 negative, absolute (including INT64_MIN), and square "
            "use wrapping arithmetic matching NumPy release builds. Inputs are not "
            "mutated; native paths omit NumPy RuntimeWarnings where NumPy may emit them."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-006",
        guidance=(
            "Use numpy.negative/absolute/abs/square with exactly one typed float64, "
            "float32, or int64 rank-1/rank-2 ndarray and no optional arguments; keep "
            "other ufunc forms on the Python fallback."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-numpy/elementwise-chain-fusion",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="binop",
            pattern=(
                "binary-op trees of 2–8 element-wise array-array binops whose "
                "leaves are simple names of same-dtype float64/float32/int64 "
                "arrays of rank 1 or 2 (ClaimExpr leaf_kind=name; mixed ranks "
                "under NumPy broadcasting; f64/f32 may use +,-,*,/; i64 may "
                "use +,-,* only; no scalars/literals/calls/attributes/"
                "subscripts). Claimed with operand_mode=leaves so core subsumes "
                "descendant per-op claims."
            ),
        ),
        constraint=(
            "A pure array-array binary-op tree with 2 through 8 inclusive binop "
            "nodes, every leaf a simple name with a resolved rextio-numpy array "
            "type, one identical dtype, and ranks in {1,2}, lowers as one fused "
            "helper under plugin API 1.2 ClaimExpr + operand_mode=leaves. "
            "f64/f32 trees may use +,-,*,/; i64 trees may use +,-,* only (i64 "
            "true division stays on ordinary per-op elementwise, which promotes "
            "to float64). Out-of-scope trees (1 or ≥9 binops, literals, Python "
            "scalars, calls, attributes, subscripts, opaque/unresolved leaves, "
            "mixed dtypes, other ranks) retain ordinary direct per-op "
            "claim/lowering — a failed fusion match is never a product fallback "
            "and does not broaden scalar handling. Result type/rank match "
            "current binop semantics (max leaf rank, same dtype). At lower time "
            "the plugin consumes ctx.leaf_operands in leaf_index order and the "
            "frozen ClaimExpr, re-validates invariants, and fails closed on "
            "malformed metadata. Helper identities are deterministic from a "
            "canonical tree/dtype/rank signature (never Python hash()). "
            "Evaluation and rounding order are preserved: one typed temporary "
            "per internal arithmetic node in left-to-right postorder inside the "
            "element closure; no reassociation, constant-folding, or FMA; i64 "
            "uses wrapping_add/wrapping_sub/wrapping_mul at every intermediate. "
            "Broadcast shapes are validated for each internal binop in the same "
            "LTR postorder as NumPy, before allocation/arithmetic, so the first "
            "mismatch and exact trailing-space ValueError shape message match "
            "existing behavior (including zero dimensions). After validation, "
            "only leaf views are broadcast; exactly one output ndarray is "
            "allocated and filled in one data pass (zero intermediate "
            "ndarrays/collections); zero-sized results are supported. Inputs "
            "are not mutated. Native elementwise continues to omit NumPy "
            "RuntimeWarnings (documented divergence)."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-005",
        guidance=(
            "Write multi-op element-wise chains as pure array-name binary trees "
            "(2–8 ops) on same-dtype float64/float32/int64 ranks 1–2 so "
            "rextio-numpy can fuse them under elementwise-chain-fusion; keep "
            "scalars, subscripts, and longer/shorter trees on the ordinary "
            "per-op elementwise path."
        ),
        stability="experimental",
        verified=True,
    ),
)
