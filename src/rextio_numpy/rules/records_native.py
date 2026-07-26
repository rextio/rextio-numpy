"""Native (verified) rule records for the rextio-numpy lowering surface."""

from __future__ import annotations

from rextio.plugins.api import RuleRecord, RuleScope

NATIVE_RECORDS: tuple[RuleRecord, ...] = (
    RuleRecord(
        id="rextio-numpy/elementwise-compare",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="compare",
            pattern=(
                "non-chained ==, !=, <, <=, >, >= over same-dtype "
                "float64/float32/int64 rank-1/rank-2 arrays, including "
                "array-array broadcasting and matching scalar forms"
            ),
        ),
        constraint=(
            "Plugin API 1.5 offers only non-chained comparison sites. Exactly "
            "two operands are accepted: same-dtype numeric arrays at ranks 1–2 "
            "under NumPy broadcasting, or one such array plus its matching "
            "Python scalar type (float for floating arrays, int for int64). "
            "The result is a plugin-owned resident bool rank-1/rank-2 array and "
            "cannot cross a Python function boundary. Rust comparison semantics "
            "match NumPy boolean results for finite values, NaN, infinities, and "
            "signed zero; float32 weak Python scalars are narrowed to float32 and "
            "an attempted exported return is rejected by Core with RXT092. "
            "int scalars are limited by the Core signed-i64 boundary. A Python int "
            "outside [-2**63, 2**63-1] is a native-boundary type-contract violation "
            "that raises OverflowError before this helper runs; ordinary NumPy "
            "fallback may accept it. Chained comparisons, is/is not, in/not in, "
            "mixed dtypes, higher ranks, and hidden metadata remain fallback/rejected."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-009",
        guidance=(
            "Use one non-chained comparison between same-dtype rank-1/rank-2 "
            "numeric arrays, or an array and its matching Python scalar, and "
            "consume the resident bool result immediately in a supported plugin call."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-numpy/resident-logical-not",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern="exact numpy.logical_not(mask) over one plugin-owned resident bool rank-1/rank-2 mask",
        ),
        constraint=(
            "Exactly one positional resident bool mask and no keywords, optional "
            "arguments, receiver, or callable metadata. The mask must have been "
            "produced inside the native expression graph by a claimed comparison or "
            "another claimed resident logical operation; it has no source annotation "
            "or Python boundary conversion. The result has the same resident rank and "
            "can only feed another claimed native expression. Elementwise boolean "
            "negation preserves shape, including zero axes, and does not mutate input."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-015",
        guidance=(
            "Use numpy.logical_not immediately on a resident mask produced by a "
            "supported NumPy comparison; do not pass masks across a Python boundary."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-numpy/resident-logical-binary",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern=(
                "exact numpy.logical_and(mask_a, mask_b) or numpy.logical_or(mask_a, mask_b) "
                "over two plugin-owned resident bool rank-1/rank-2 masks"
            ),
        ),
        constraint=(
            "Exactly two positional resident bool masks and no keywords, optional "
            "arguments, receiver, or callable metadata. Both operands stay inside the "
            "native expression graph; NumPy rank-1/rank-2 broadcasting, including "
            "zero axes, determines the resident bool result rank. Mismatched shapes "
            "raise the established NumPy-compatible broadcast ValueError. A mask may "
            "be bound only as a fresh resident local inside the native graph; it cannot "
            "be materialized, annotated, returned, or supplied through a Python boundary."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-016",
        guidance=(
            "Compose supported comparison masks with exact two-argument "
            "numpy.logical_and/or calls, then consume the result in numpy.where."
        ),
        stability="experimental",
        verified=True,
    ),
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
        id="rextio-numpy/elementwise-ufunc-call",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern=(
                "exact numpy.add/subtract/multiply/divide(a, b) calls with two "
                "positional operands and no keywords, over the same dtype/rank/"
                "broadcast matrix as element-wise +, -, *, /"
            ),
        ),
        constraint=(
            "Exact two-positional-operand NumPy ufunc calls reuse the certified "
            "operator lowering without broadening it: same-dtype float64/float32/"
            "int64 arrays of rank 1 or 2 under rank-1/rank-2 broadcasting, or one "
            "such array and the matching Python scalar. Subtract and divide preserve "
            "operand order; int64 divide yields float64 at the broadcast result rank; "
            "int64 add/subtract/multiply wrap as NumPy release builds do. Calls with "
            "out, where, dtype, casting, order, subok, signature, extobj, or any "
            "other keyword/extra argument are rejected and retain Python fallback. "
            "Mixed array dtypes, mismatched scalar types, unresolved operands, and "
            "other ranks remain outside the native surface. As for the operator "
            "route, native floating operations omit NumPy RuntimeWarnings while "
            "preserving certified values, dtypes, shapes, exceptions, and inputs."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-007",
        guidance=(
            "Use numpy.add/subtract/multiply/divide with exactly two positional "
            "operands from the documented same-dtype rank-1/rank-2 array/scalar "
            "matrix and no optional ufunc arguments; otherwise keep the call on "
            "Python fallback."
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
            "the current receiver metadata contract; core evaluates the receiver exactly "
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
                "(no keywords; float32 whole-array reductions, int64 mean, whole-array "
                "max/min, and non-literal axis forms are not claimed under this rule)"
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
            "(RXTP-NUMPY-004), and whole-array int64 max/min under "
            "rextio-numpy/reduction-whole-i64-extrema (RXTP-NUMPY-008). The "
            "equivalent a.sum()/a.mean() method forms are "
            "admitted through the current receiver metadata contract, evaluated exactly "
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
        id="rextio-numpy/reduction-whole-i64-extrema",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern=(
                "numpy.max(a), numpy.min(a), a.max(), or a.min() over a whole "
                "int64 rank-1/rank-2 ndarray with no arguments or keywords"
            ),
        ),
        constraint=(
            "Whole-array max/min lower only for exact int64 rank-1/rank-2 plugin "
            "arrays. The helper traverses every element without mutation and returns "
            "a builtin int; empty inputs raise ValueError with NumPy-compatible "
            "'maximum/minimum which has no identity' text. Float extrema remain "
            "ordinary Python fallback because NaN payload/sign and signed-zero tie "
            "behavior varies across supported NumPy platform/SIMD profiles. No axis, "
            "out, where, initial, keepdims, dtype, or other option is accepted by "
            "this rule; literal-axis int64 extrema remain separately covered by "
            "RXTP-NUMPY-004."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-008",
        guidance=(
            "Use numpy.max/min(a) or a.max/min() with no arguments on an exact "
            "int64 rank-1/rank-2 array. Keep float extrema and every optional form "
            "on Python fallback."
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
                "numpy.sum/mean/max/min(a, <int literal>), with equivalent ndarray "
                "method forms and exactly one named or positional axis "
                "(ranks 1–2; see constraint for dtype matrix)"
            ),
        ),
        constraint=(
            "Single-axis reductions with a static signed integer axis literal "
            "(normalized against rank at claim time; out-of-range axes are not "
            "claimed). Accepted forms: module calls with one positional array plus "
            "either named axis=<int> or one positional integer axis, and ndarray method "
            "calls with the array as the API-1.3 receiver plus the same named/positional "
            "axis alternatives; no dtype/out/keepdims/initial/where, "
            "no axis=None, no tuple axis, no dynamic axis, no "
            "duplicate/extra keywords, no numpy.amax/amin. Core evaluates a method "
            "receiver exactly once before call operands. "
            "Dtype/rank matrix: sum on float64/int64 ranks 1–2; mean on float64 "
            "ranks 1–2; max/min on int64 ranks 1–2 only; float extrema stay fallback "
            "because NumPy NaN payload/sign and signed-zero tie behavior varies by "
            "supported platform/SIMD profile. float32 sum/mean and int64 mean "
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
            "keep the separate ndarray route. Literal-axis max/min are int64 only: "
            "float extrema stay fallback because NumPy NaN payload/sign and signed-zero "
            "tie behavior varies by supported platform/SIMD profile. Max/min over "
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
            "Write numpy.sum/mean/max/min(a, axis=<int literal>), "
            "numpy.sum/mean/max/min(a, <int literal>), or the equivalent ndarray "
            "method form with a static "
            "integer axis on float64/int64 ranks 1–2 for sum, float64 ranks 1–2 "
            "for mean, or int64 ranks 1–2 for max/min. Float extrema, float32 "
            "sum/mean, and int64 mean stay on the Python fallback. Keep amax/amin, tuple "
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
        id="rextio-numpy/where-three-argument",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern=(
                "exact numpy.where(condition, x, y) with a resident comparison "
                "mask and same-dtype numeric rank-1/rank-2 array branches"
            ),
        ),
        constraint=(
            "Exactly three positional operands and no keywords. condition must "
            "be a plugin-owned resident bool rank-1/rank-2 result; condition "
            "has no annotation spelling and cannot be forged in source or cross "
            "a parameter/return boundary. x/y must be "
            "same-dtype float64/float32/int64 rank-1/rank-2 arrays, or one array "
            "and a matching Python scalar; two scalar branches, dtype mixing, "
            "coercion ambiguity, condition-only where, out/keyword forms, and "
            "higher ranks are excluded. Core-canonicalized import aliases such as "
            "np.where and 'from numpy import where as choose' are admitted; runtime "
            "assignment/rebinding aliases are excluded. The helper independently computes the "
            "three-way NumPy broadcast shape (including zero axes), raises "
            "NumPy-compatible shape errors, and copies the selected branch value "
            "without arithmetic, preserving NaN/inf classes and signed-zero bits. "
            "Float32 Python scalar branches use NumPy 2.4 weak-scalar narrowing, "
            "including overflow to infinity. Integer branches inherit Core's signed-"
            "i64 boundary; out-of-range Python ints are boundary type-contract "
            "violations and may differ from ordinary fallback NumPy behavior."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-014",
        guidance=(
            "Call numpy.where(mask, x, y) with a resident comparison mask and "
            "same-dtype supported numeric branches, at least one of which is an "
            "annotated array; use three positional arguments and no options."
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
            "When every leaf is the same rank as the result, an equal-shape "
            "standard-layout (C-order) fast path may be decided and entered "
            "before any LTR postorder broadcast-shape Vec work: equal leaf "
            "shapes plus is_standard_layout, then contiguous-slice loads. "
            "Otherwise the generic path validates broadcast shapes for each "
            "internal binop in the same LTR postorder as NumPy before "
            "allocation/arithmetic, so the first mismatch and exact "
            "trailing-space ValueError shape message match existing behavior "
            "(including zero dimensions); that path also handles leaves that "
            "remain non-standard-layout at helper entry and all broadcast "
            "cases (leaf views broadcast as before). Mixed-rank trees emit "
            "only the generic path. Either path allocates exactly one output "
            "ndarray and fills it in one data pass (zero intermediate "
            "ndarrays/collections); zero-sized results are supported. When "
            "lower-time leaf operand names prove repeated bindings, the helper "
            "may take one parameter per unique name and reuse loads. The "
            "contiguous equal-shape path is a layout-gated "
            "correctness-preserving shortcut, not a published speed claim. "
            "Inputs are not mutated. Native elementwise continues to omit "
            "NumPy RuntimeWarnings (documented divergence)."
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
