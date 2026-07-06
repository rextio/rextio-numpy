"""The rule records rextio-numpy describes to Rextio core.

L2 rule records per the Rextio tooling contract: each states a pattern, the
constraint that decides it, the RXTP-NUMPY diagnostic code it will fire as,
and remediation guidance. The rule set covers the implemented first lowering
surface — float64 1-D element-wise arithmetic (array-array, array-scalar,
scalar-array), ``numpy.dot``, and whole-array ``sum``/``mean`` reductions via
the Rust ``ndarray`` crate — plus the explicit exclusions around it.

All records are ``experimental`` (plugin API 1.1, rextio 0.1.1 line). Records
with outcome ``native`` carry ``verified=True``: their lowering passed the
core plugin certification kit (``rextio.plugins.testing``) against CPython
NumPy, with the divergences documented per rule in ``constraint``. Records
with outcome ``fallback`` are exclusions that keep code on the Python
fallback.

RXTP-NUMPY-011 (rank), RXTP-NUMPY-012 (view aliasing), and RXTP-NUMPY-019
(uncovered API) are **declarative-only**: they document why code stays on the
fallback but are never emitted by ``claim()`` — sites outside the covered
surface return ``NotCovered`` and core reports its own diagnostic (RXT030).
Only RXTP-NUMPY-010 is actively emitted, via ``Rejected``.
"""

from __future__ import annotations

from rextio.plugins.api import CoverageDecl, RuleRecord, RuleScope

# ``symbols`` is DESCRIPTIVE (it appears in the capability manifest); the claim
# pass routes sites by package + operand-type ownership, not by this list. In
# particular ``numpy.ndarray`` denotes the type this plugin lowers -- it is NOT
# a claimed call target: ndarray METHOD forms (a.dot(b), a.sum()) are never
# claimed (they stay on the fallback). ``numpy.dot/sum/mean`` are the covered
# module-call forms (council round 8: clarify the dual meaning).
COVERAGE = CoverageDecl(
    packages=("numpy",),
    modules=("numpy",),
    symbols=(
        "numpy.ndarray",
        "numpy.dot",
        "numpy.sum",
        "numpy.mean",
    ),
)


def numpy_rule_records() -> tuple[RuleRecord, ...]:
    """Return the plugin's rule records, ordered by rule id."""
    return _RULES


_RULES: tuple[RuleRecord, ...] = tuple(
    sorted(
        (
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
            RuleRecord(
                id="rextio-numpy/unsupported-operand-types",
                provider="rextio-numpy",
                scope=RuleScope(
                    kind="call",
                    pattern=(
                        "covered numpy.dot/sum/mean call or elementwise +/-/*// binop whose "
                        "resolved operand types are outside the float64 1-D surface"
                    ),
                ),
                constraint=(
                    "A covered numpy operation whose operand types are known but outside the "
                    "supported set (float64 1-D arrays, plus float scalars for elementwise binops) "
                    "is rejected here so the plugin's guidance is delivered. Unresolved operands and "
                    "wrong-arity/unsupported call shapes are NotCovered instead, so core's own "
                    "diagnostic fires. Emitted from both call and binop sites -- the code is the "
                    "operand-type rejection, not a dtype-annotation rule."
                ),
                outcome="fallback",
                diagnostic_code="RXTP-NUMPY-010",
                guidance=(
                    "Cast operands to float64 1-D arrays (x.astype(np.float64)) at the boundary of "
                    "the hot path, or keep the function on the Python fallback."
                ),
                stability="experimental",
            ),
            RuleRecord(
                id="rextio-numpy/unsupported-ndim",
                provider="rextio-numpy",
                scope=RuleScope(
                    kind="type",
                    pattern="ndarray with more than 1 dimension, or a dynamically unknown rank",
                ),
                constraint=(
                    "The initial surface covers statically known 1-D arrays only (2-D support "
                    "is planned but not implemented); higher ranks and rank-polymorphic code "
                    "stay on the Python fallback."
                ),
                outcome="fallback",
                diagnostic_code="RXTP-NUMPY-011",
                guidance=(
                    "Reshape or split higher-rank work into 1-D/2-D kernels, or keep the function on "
                    "the fallback."
                ),
                stability="experimental",
            ),
            RuleRecord(
                id="rextio-numpy/view-aliasing",
                provider="rextio-numpy",
                scope=RuleScope(
                    kind="call",
                    pattern="mutating a slice/view that aliases another live array (b = a[1:]; b[0] = ...)",
                ),
                constraint=(
                    "NumPy views alias their base array; Rust ownership cannot reproduce mutation "
                    "through an alias without diverging, so mutating aliased views keeps the "
                    "function on the Python fallback."
                ),
                outcome="fallback",
                diagnostic_code="RXTP-NUMPY-012",
                guidance=(
                    "Operate on the base array directly, or take explicit copies (a[1:].copy()) when "
                    "aliasing is not required."
                ),
                stability="experimental",
            ),
            RuleRecord(
                id="rextio-numpy/unsupported-api",
                provider="rextio-numpy",
                scope=RuleScope(
                    kind="call",
                    pattern="any numpy API outside the covered symbols (fancy indexing, broadcasting beyond scalars, ufunc kwargs, random, linalg, ...)",
                ),
                constraint=(
                    "APIs outside the covered surface have no verified Rust lowering and keep the "
                    "surrounding candidate on the Python fallback — Rextio never guesses."
                ),
                outcome="fallback",
                diagnostic_code="RXTP-NUMPY-019",
                guidance=(
                    "Isolate covered array math into its own typed function and leave the rest of "
                    "the NumPy usage on the fallback (or under Numba, which stays a valid choice for "
                    "kernels this plugin does not cover)."
                ),
                stability="experimental",
            ),
        ),
        key=lambda record: record.id,
    )
)
