"""The rule records rextio-numpy describes to Rextio core.

L2 rule records per the Rextio tooling contract: each states a pattern, the
constraint that decides it, the RXTP-NUMPY diagnostic code it will fire as,
and remediation guidance. The rule set covers the implemented Wave-1/Wave-2
lowering surface — float64/float32/int64 ranks 1–2 element-wise arithmetic
(array-array with NumPy broadcasting, array-scalar, scalar-array), 1-D
float64/int64 ``numpy.dot``, whole-array float64/int64 ``sum`` and whole-array
float64 ``mean``, plus literal-axis ``sum``/``mean``/``max``/``min``
(``axis=<int literal>``) via the Rust ``ndarray`` crate (float32 sum/mean/dot,
rank-1 float32 max/min, and int64 mean stay fallback) — plus the explicit
exclusions around it.

All records are ``experimental`` (plugin API 1.2, rextio 0.1.1 line with core
API 1.2 claim metadata). Records with outcome ``native`` carry
``verified=True``: their lowering is certified via the core plugin
certification kit (``rextio.plugins.testing``) against CPython NumPy, with the
divergences documented per rule in ``constraint``. Records with outcome
``fallback`` are exclusions that keep code on the Python fallback.

RXTP-NUMPY-011 (rank), RXTP-NUMPY-012 (view aliasing), and RXTP-NUMPY-019
(uncovered API) are **declarative-only**: they document why code stays on the
fallback but are never emitted by ``claim()`` — sites outside the covered
surface return ``NotCovered`` and core reports its own diagnostic (RXT030).
Only RXTP-NUMPY-010 is actively emitted, via ``Rejected``.
"""

from __future__ import annotations

from rextio.plugins.api import RuleRecord

from rextio_numpy.rules.coverage import COVERAGE
from rextio_numpy.rules.records_fallback import FALLBACK_RECORDS
from rextio_numpy.rules.records_native import NATIVE_RECORDS

__all__ = ["COVERAGE", "numpy_rule_records"]


def numpy_rule_records() -> tuple[RuleRecord, ...]:
    """Return the plugin's rule records, ordered by rule id."""
    return _RULES


_RULES: tuple[RuleRecord, ...] = tuple(
    sorted(
        (*NATIVE_RECORDS, *FALLBACK_RECORDS),
        key=lambda record: record.id,
    )
)
