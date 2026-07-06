"""The rule records rextio-numpy describes to Rextio core.

L2 rule records per the Rextio tooling contract: each states a pattern, the
constraint that decides it, the RXTP-NUMPY diagnostic code it will fire as,
and remediation guidance. The initial rule set covers the planned first
lowering surface — float64 1-D/2-D element-wise arithmetic, ``dot``, and
``sum``/``mean`` reductions via the Rust ``ndarray`` crate — plus the explicit
exclusions around it.

All records are ``experimental``: lowering itself activates only when rextio
core exposes the plugin ``lower()`` hook, and the rule surface may change
until then. Records with outcome ``native`` describe the coverage this plugin
is building toward; records with outcome ``fallback`` are exclusions that will
keep code on the Python fallback even after lowering lands.
"""

from __future__ import annotations

from rextio.plugins.api import CoverageDecl, RuleRecord, RuleScope

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
                    kind="call",
                    pattern="element-wise +, -, *, / on float64 arrays of 1 or 2 dimensions",
                ),
                constraint=(
                    "Element-wise arithmetic on float64 ndarrays (array-array of equal shape, "
                    "or array-scalar) maps to ndarray-crate operations with IEEE-754 semantics "
                    "matching NumPy."
                ),
                outcome="native",
                diagnostic_code="RXTP-NUMPY-001",
                guidance=(
                    "Keep hot array math to float64 arrays and plain element-wise operators; "
                    "annotate array parameters and returns so shapes and dtypes resolve statically."
                ),
                stability="experimental",
            ),
            RuleRecord(
                id="rextio-numpy/dot-float64",
                provider="rextio-numpy",
                scope=RuleScope(
                    kind="call",
                    pattern="numpy.dot(a, b) or a @ b on float64 arrays of 1 or 2 dimensions",
                ),
                constraint=(
                    "Vector-vector, matrix-vector, and matrix-matrix products on float64 lower to "
                    "ndarray dot products; float summation order may differ from NumPy's pairwise "
                    "summation, a documented divergence."
                ),
                outcome="native",
                diagnostic_code="RXTP-NUMPY-002",
                guidance=(
                    "Use numpy.dot or @ directly on float64 arrays; avoid dtype-mixing operands "
                    "(cast explicitly first)."
                ),
                stability="experimental",
            ),
            RuleRecord(
                id="rextio-numpy/reduction-sum-mean",
                provider="rextio-numpy",
                scope=RuleScope(
                    kind="call",
                    pattern="numpy.sum / numpy.mean (or ndarray.sum/.mean) over a whole float64 array",
                ),
                constraint=(
                    "Full-array sum and mean reductions on float64 lower natively; axis= keyword "
                    "reductions stay on the fallback in the initial surface. Float summation order "
                    "may differ from NumPy's pairwise summation, a documented divergence."
                ),
                outcome="native",
                diagnostic_code="RXTP-NUMPY-003",
                guidance=(
                    "Prefer whole-array reductions in hot paths; hoist axis-wise reductions out of "
                    "native candidates or keep the function on the fallback until axis support lands."
                ),
                stability="experimental",
            ),
            RuleRecord(
                id="rextio-numpy/unsupported-dtype",
                provider="rextio-numpy",
                scope=RuleScope(
                    kind="type",
                    pattern="ndarray with a dtype other than float64",
                ),
                constraint=(
                    "Only float64 arrays are covered by the initial lowering surface; other dtypes "
                    "(int arrays, float32, complex, object, structured) keep the function on the "
                    "Python fallback."
                ),
                outcome="fallback",
                diagnostic_code="RXTP-NUMPY-010",
                guidance=(
                    "Cast to float64 at the boundary of the hot path (x.astype(np.float64)) when the "
                    "extra precision is acceptable, or keep the function on the fallback."
                ),
                stability="experimental",
            ),
            RuleRecord(
                id="rextio-numpy/unsupported-ndim",
                provider="rextio-numpy",
                scope=RuleScope(
                    kind="type",
                    pattern="ndarray with more than 2 dimensions, or a dynamically unknown rank",
                ),
                constraint=(
                    "The initial surface covers statically known 1-D and 2-D arrays only; higher "
                    "ranks and rank-polymorphic code stay on the Python fallback."
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
