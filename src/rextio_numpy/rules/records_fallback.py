"""Fallback (exclusion) rule records for the rextio-numpy surface."""

from __future__ import annotations

from rextio.plugins.api import RuleRecord, RuleScope

FALLBACK_RECORDS: tuple[RuleRecord, ...] = (
    RuleRecord(
        id="rextio-numpy/unsupported-operand-types",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern=(
                "covered numpy.dot/sum/mean/max/min/unary module (including certified ndarray method) "
                "call, exact numpy.add/subtract/multiply/divide call, or elementwise +/-/*// binop "
                "whose resolved operand types are outside the float64/float32/int64 "
                "rank-1/2 surface (including excluded reduction dtype cells)"
            ),
        ),
        constraint=(
            "A covered numpy operation whose operand types are known but outside the "
            "supported set is rejected here so the plugin's guidance is delivered. "
            "Elementwise covers same-dtype float64/float32/int64 arrays of rank 1 or 2 "
            "plus matching float/int scalars. Whole-array and literal-axis sum cover "
            "float64/int64 ranks 1–2; mean covers float64 ranks 1–2 only (float32 "
            "sum/mean and int64 mean are excluded). Whole-array and literal-axis "
            "max/min cover int64 ranks 1–2 only; float extrema stay fallback because "
            "NumPy NaN payload/sign "
            "and signed-zero tie behavior varies by supported platform/SIMD profile. "
            "1-D dot covers same-dtype float64/int64 only (float32 dots "
            "are excluded). Unresolved operands and wrong-arity/unsupported call "
            "shapes (including float bare max/min, non-literal axis, tuple axis, extra "
            "kwargs) are NotCovered instead, except exact arithmetic ufunc calls with "
            "optional/extra arguments, which fail closed through this fallback "
            "diagnostic. Emitted "
            "from both call and binop sites — the code is the operand-type rejection, "
            "not a dtype-annotation rule."
        ),
        outcome="fallback",
        diagnostic_code="RXTP-NUMPY-010",
        guidance=(
            "Cast operands to a supported dtype and rank at the boundary of the hot "
            "path (float64/float32/int64 for elementwise; float64/int64 for sum; "
            "float64 for mean; float64/int64 for dot; and int64 for max/min). Float "
            "extrema, float32 sum/mean/dots, and int64 mean stay on the fallback; keep "
            "array dtypes uniform, or keep the "
            "function on the Python fallback."
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
            "The implemented surface covers statically known 1-D and 2-D arrays only; "
            "higher ranks and rank-polymorphic code stay on the Python fallback."
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
        id="rextio-numpy/ndarray-subclass-boundary",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="type",
            pattern=(
                "numpy.ndarray subclass (including numpy.matrix or a custom "
                "__array_ufunc__ override) passed to a plugin-typed native boundary"
            ),
        ),
        constraint=(
            "The annotation vocabulary is nominal and static analysis cannot distinguish "
            "an exact base numpy.ndarray from a runtime subclass. Claims therefore remain "
            "unchanged. Every materialized plugin-array parameter performs NumPy's exact "
            "C-level ndarray type check before copying; a subclass deterministically raises "
            "TypeError('rextio-numpy native boundary requires exact numpy.ndarray; ndarray "
            "subclasses are unsupported') when the native route executes. This is a runtime "
            "native-boundary rejection, not an automatic claim-time fallback. Exact base "
            "ndarray views/strided arrays remain admitted."
        ),
        outcome="reject",
        diagnostic_code="RXTP-NUMPY-013",
        guidance=(
            "Convert to an exact base array with numpy.asarray before entering the typed hot "
            "path when subclass behavior is unnecessary. If matrix/subclass dispatch, "
            "__array_ufunc__, or subok behavior is required, keep the enclosing function on "
            "the Python fallback."
        ),
        stability="experimental",
    ),
    RuleRecord(
        id="rextio-numpy/unsupported-api",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern=(
                "any numpy API outside the covered symbols (fancy indexing, unsupported "
                "method forms, non-literal/tuple/None axis, keepdims/out kwargs, 2-D "
                "matmul/@, unsupported ufuncs or optional ufunc arguments, random, "
                "linalg, amax/amin, ...)"
            ),
        ),
        constraint=(
            "APIs outside the covered surface have no verified Rust lowering and keep the "
            "surrounding candidate on the Python fallback — Rextio never guesses. Literal "
            "single-axis sum/mean/max/min are covered under RXTP-NUMPY-004; other axis "
            "forms remain fallback."
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
)
