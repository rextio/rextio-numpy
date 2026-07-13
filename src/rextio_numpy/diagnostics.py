"""Shared rejection guidance helpers for rextio-numpy claim decisions.

Holds the plugin type key and the RXTP-NUMPY-010 rejection builder used by
the claim routers. Guidance text is taken from the matching rule record so
message/suggestion stay synchronized with the described surface.
"""

from __future__ import annotations

from rextio.analyzer.diagnostics import Diagnostic
from rextio.plugins.api import ClaimResult, ClaimSite, NotCovered, Rejected

from rextio_numpy.rules import numpy_rule_records

#: The plugin type key for 1-D float64 arrays.
F64_1D = "rextio-numpy/f64-1d"

# The remediation guidance for claim rejections comes from the rule record
# that owns diagnostic code RXTP-NUMPY-010 (unsupported dtype/operand types).
_REJECTION_GUIDANCE = next(
    record.guidance
    for record in numpy_rule_records()
    if record.diagnostic_code == "RXTP-NUMPY-010"
)


def not_covered_or_rejected(site: ClaimSite) -> ClaimResult:
    """Resolve a covered-target miss: NotCovered when unresolved, else Rejected."""
    if any(operand is None for operand in site.operand_types):
        return NotCovered()
    named = ", ".join(str(operand) for operand in site.operand_types)
    return Rejected(
        diagnostic=Diagnostic(
            code="RXTP-NUMPY-010",
            severity="error",
            message=(
                f"rextio-numpy cannot lower {site.target!r}: operand types "
                f"({named}) are outside the float64 1-D surface "
                f"({F64_1D} and float scalars only)"
            ),
            file_path="",
            line=0,
            column=0,
            suggestion=_REJECTION_GUIDANCE,
        )
    )
