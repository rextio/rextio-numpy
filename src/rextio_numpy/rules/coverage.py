"""Coverage declaration for the rextio-numpy plugin surface."""

from __future__ import annotations

from rextio.plugins.api import CoverageDecl

# ``symbols`` is DESCRIPTIVE (it appears in the capability manifest); the claim
# pass routes sites by package + operand-type ownership, not by this list. In
# particular ``numpy.ndarray`` denotes the type this plugin lowers -- it is NOT
# a claimed call target: ndarray METHOD forms (a.dot(b), a.sum()) are never
# claimed (they stay on the fallback). ``numpy.dot/sum/mean/max/min`` are the
# covered module-call forms (council round 8: clarify the dual meaning).
COVERAGE = CoverageDecl(
    packages=("numpy",),
    modules=("numpy",),
    symbols=(
        "numpy.ndarray",
        "numpy.dot",
        "numpy.sum",
        "numpy.mean",
        "numpy.max",
        "numpy.min",
    ),
)
