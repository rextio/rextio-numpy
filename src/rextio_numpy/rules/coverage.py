"""Coverage declaration for the rextio-numpy plugin surface."""

from __future__ import annotations

from rextio.plugins.api import CoverageDecl

# ``symbols`` is DESCRIPTIVE (it appears in the capability manifest); the claim
# pass routes sites by package + operand-type ownership, not by this list. In
# particular ``numpy.ndarray`` denotes both the type this plugin lowers and the
# receiver type for the certified method-parity surface. ``numpy.dot/sum/mean/
# max/min`` name the equivalent module-call forms.
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
        "numpy.ndarray.dot",
        "numpy.ndarray.sum",
        "numpy.ndarray.mean",
        "numpy.ndarray.max",
        "numpy.ndarray.min",
        "numpy.add",
        "numpy.subtract",
        "numpy.multiply",
        "numpy.divide",
        "numpy.negative",
        "numpy.absolute",
        "numpy.abs",
        "numpy.square",
        "numpy.logical_not",
        "numpy.logical_and",
        "numpy.logical_or",
        "numpy.where",
    ),
)
