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
                "covered numpy.dot/sum/mean call or elementwise +/-/*// binop whose "
                "resolved operand types are outside the float64/float32/int64 rank-1/2 surface"
            ),
        ),
        constraint=(
            "A covered numpy operation whose operand types are known but outside the "
            "supported set is rejected here so the plugin's guidance is delivered. "
            "Elementwise covers same-dtype float64/float32/int64 arrays of rank 1 or 2 "
            "plus matching float/int scalars. Whole-array sum covers float64/int64 "
            "ranks 1–2; mean covers float64 ranks 1–2 only (float32 reductions and "
            "int64 mean are excluded). 1-D dot covers same-dtype float64/int64 only "
            "(float32 dots are excluded). Unresolved operands and wrong-arity/"
            "unsupported call shapes are NotCovered instead, so core's own diagnostic "
            "fires. Emitted from both call and binop sites — the code is the "
            "operand-type rejection, not a dtype-annotation rule."
        ),
        outcome="fallback",
        diagnostic_code="RXTP-NUMPY-010",
        guidance=(
            "Cast operands to a supported dtype and rank at the boundary of the hot "
            "path (float64/float32/int64 for elementwise; float64 for sum/mean and "
            "float64/int64 for sum/dot — float32 whole-array reductions/dots and "
            "int64 mean stay on the fallback, so cast those to float64 if native "
            "lowering is required), keep array dtypes uniform, or keep the function "
            "on the Python fallback."
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
        id="rextio-numpy/unsupported-api",
        provider="rextio-numpy",
        scope=RuleScope(
            kind="call",
            pattern=(
                "any numpy API outside the covered symbols (fancy indexing, axis= reductions, "
                "2-D matmul/@, ufunc kwargs, random, linalg, ...)"
            ),
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
)
