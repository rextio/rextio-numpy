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
                "element-wise +, -, *, / on 1-D float64 arrays "
                "(array-array, array-scalar, and scalar-array forms)"
            ),
        ),
        constraint=(
            "Element-wise arithmetic on 1-D float64 ndarrays (array-array of equal "
            "length or with a length-1 operand broadcast NumPy-style, or "
            "array-scalar/scalar-array with a float scalar, operand order preserved for - and /) maps to ndarray-crate "
            "operations with IEEE-754 semantics matching NumPy. Documented "
            "divergence: the native lowering emits no NumPy RuntimeWarnings "
            "(e.g. divide-by-zero or invalid-value warnings); result values match."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-001",
        guidance=(
            "Keep hot array math to float64 arrays and plain element-wise operators; "
            "annotate array parameters and returns so shapes and dtypes resolve statically."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-numpy/dot-float64",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern="numpy.dot(a, b) on 1-D float64 arrays (module-call form)",
        ),
        constraint=(
            "Vector-vector dot products on 1-D float64 arrays lower to ndarray dot "
            "products. Documented divergences: float summation order may differ from "
            "NumPy's pairwise summation (certified within 1e-12 relative/absolute "
            "tolerance, so verified means within-tolerance, not bit-equivalence), and "
            "the native leg returns a builtin float where NumPy returns numpy.float64 "
            "(a float subclass; type()/repr/.dtype observably differ)."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-002",
        guidance=(
            "Use numpy.dot directly on float64 arrays; the @ operator is offered "
            "to plugins but not lowered by rextio-numpy, so it stays on the "
            "fallback. Avoid dtype-mixing operands (cast explicitly first)."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-numpy/reduction-sum-mean",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern="numpy.sum / numpy.mean over a whole 1-D float64 array (module-call form; ndarray method forms are not claimed)",
        ),
        constraint=(
            "Full-array sum and mean reductions on 1-D float64 arrays lower natively; "
            "axis= keyword reductions and the a.sum()/a.mean() method forms stay on the "
            "fallback in the initial surface. Documented divergences: float summation "
            "order may differ from NumPy's pairwise summation (certified within 1e-12 "
            "relative/absolute tolerance, so verified means within-tolerance), the "
            "native leg returns a builtin float where NumPy returns numpy.float64, and "
            "the mean of an empty array returns nan on both legs without NumPy's "
            "RuntimeWarning on the native leg."
        ),
        outcome="native",
        diagnostic_code="RXTP-NUMPY-003",
        guidance=(
            "Prefer whole-array reductions in hot paths; hoist axis-wise reductions out of "
            "native candidates or keep the function on the fallback until axis support lands."
        ),
        stability="experimental",
        verified=True,
    ),
)
