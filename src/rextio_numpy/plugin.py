"""The rextio-numpy plugin object and entry-point factory.

Implements plugin API 1.1 (``rextio.plugins.api.RextioLoweringPlugin``): the
protocol-v2 describe/covers surface plus the lowering members — annotation
vocabulary, the deterministic claim pass, expression lowering, and pinned
crate dependencies. The plugin module itself never imports numpy; only the
user-facing :mod:`rextio_numpy.types` vocabulary module does.
"""

from __future__ import annotations

from rextio.analyzer.diagnostics import Diagnostic
from rextio.config.schema import RextioConfig
from rextio.plugins.api import (
    BoundaryConversion,
    ClaimResult,
    Claimed,
    ClaimSite,
    CoverageDecl,
    CrateDependency,
    LoweredExpr,
    LoweringContext,
    NotCovered,
    PluginType,
    Rejected,
    RuleRecord,
)
from rextio.plugins.models import RextioPlugin

from rextio_numpy import rust_snippets
from rextio_numpy.__about__ import __version__
from rextio_numpy.rules import COVERAGE, numpy_rule_records

PLUGIN_ID = "rextio-numpy"

#: The plugin type key for 1-D float64 arrays.
F64_1D = "rextio-numpy/f64-1d"

# The proven boundary conversion (compiled and certified under cargo with
# pyo3 0.29 + rust-numpy =0.29.0). The native type is rust-numpy's ndarray
# RE-EXPORT: rust-numpy pins its own compatible ndarray, so a direct ndarray
# dependency would be a second, type-incompatible copy of the crate.
_F64_1D_TYPE = PluginType(
    key=F64_1D,
    annotations=("rextio_numpy.types.F64Arr1",),
    rust_type="numpy::ndarray::Array1<f64>",
    conversion=BoundaryConversion(
        param_rust="numpy::PyReadonlyArray1<'py, f64>",
        param_expr="{param}.as_array().to_owned()",
        return_rust="pyo3::Bound<'py, numpy::PyArray1<f64>>",
        return_expr="numpy::ToPyArray::to_pyarray(&{value}, py)",
    ),
)

# Binary operator token -> the elementwise helper op name.
_BINOP_NAMES = {"+": "add", "-": "sub", "*": "mul", "/": "div"}

_REDUCTION_TARGETS = ("numpy.sum", "numpy.mean")

# The remediation guidance for claim rejections comes from the rule record
# that owns diagnostic code RXTP-NUMPY-010 (unsupported dtype/operand types).
_REJECTION_GUIDANCE = next(
    record.guidance
    for record in numpy_rule_records()
    if record.diagnostic_code == "RXTP-NUMPY-010"
)


class RextioNumpyPlugin:
    """Plugin API 1.1: describes AND lowers eligible NumPy usage to Rust."""

    plugin_id = PLUGIN_ID
    api_version = "1.1"

    def to_rextio_plugin(self) -> RextioPlugin:
        """Return the v1 metadata Rextio core registers this plugin under."""
        return RextioPlugin(
            id=PLUGIN_ID,
            name=f"NumPy to Rust (rextio-numpy {__version__})",
            source_language="python",
            target_language="rust",
            packages=COVERAGE.packages,
        )

    def covers(self) -> CoverageDecl:
        """Return the packages, modules, and symbols this plugin covers."""
        return COVERAGE

    def describe(self, config: RextioConfig) -> tuple[RuleRecord, ...]:
        """Return the rule records for the resolved project configuration.

        The rule surface is currently config-independent; the parameter is part
        of the protocol so future rules can vary with (for example) import
        policies or target versions.
        """
        del config
        return numpy_rule_records()

    def type_vocabulary(self) -> tuple[PluginType, ...]:
        """Return the annotation vocabulary this plugin adds to the analyzer."""
        return (_F64_1D_TYPE,)

    def claim(self, site: ClaimSite, config: RextioConfig) -> ClaimResult:
        """Decide, at analysis time, whether this plugin lowers the site.

        Deterministic by contract: the decision is a pure function of
        ``(site.kind, site.target, site.operand_types)``. Covered targets with
        unresolved operands return :class:`NotCovered`; covered targets with
        known-but-unsupported operand types return :class:`Rejected` with
        RXTP-NUMPY-010 guidance; everything else is :class:`NotCovered`.
        """
        del config
        kind, target, operands = site.kind, site.target, site.operand_types
        if kind == "call" and target == "numpy.dot":
            if operands == (F64_1D, F64_1D):
                return Claimed(rule_id="rextio-numpy/dot-float64", result_type="float")
            return self._not_covered_or_rejected(site)
        if kind == "call" and target in _REDUCTION_TARGETS:
            if operands == (F64_1D,):
                return Claimed(rule_id="rextio-numpy/reduction-sum-mean", result_type="float")
            return self._not_covered_or_rejected(site)
        if kind == "binop" and target in _BINOP_NAMES:
            if operands in ((F64_1D, F64_1D), (F64_1D, "float"), ("float", F64_1D)):
                return Claimed(rule_id="rextio-numpy/elementwise-float64", result_type=F64_1D)
            if F64_1D not in operands:
                # No plugin-typed operand: not this plugin's business.
                return NotCovered()
            return self._not_covered_or_rejected(site)
        return NotCovered()

    @staticmethod
    def _not_covered_or_rejected(site: ClaimSite) -> ClaimResult:
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

    def lower(self, claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
        """Emit the Rust expression for a previously claimed site.

        Every emitted expression is fallible (``pyo3::PyResult``) and ends
        with ``?``; the helper ``fn`` items travel in ``helpers`` and are
        deduplicated by exact text in core codegen.
        """
        kind, target = claimed.kind, claimed.target
        if kind == "call" and target == "numpy.dot":
            return LoweredExpr(
                rust=f"__rxtnp_dot1(&{ctx.operands[0]}, &{ctx.operands[1]})?",
                helpers=(rust_snippets.dot1(),),
            )
        if kind == "call" and target == "numpy.sum":
            return LoweredExpr(
                rust=f"__rxtnp_sum1(&{ctx.operands[0]})?",
                helpers=(rust_snippets.sum1(),),
            )
        if kind == "call" and target == "numpy.mean":
            return LoweredExpr(
                rust=f"__rxtnp_mean1(&{ctx.operands[0]})?",
                helpers=(rust_snippets.mean1(),),
            )
        if kind == "binop" and target in _BINOP_NAMES:
            return self._lower_binop(claimed, ctx)
        raise ValueError(f"rextio-numpy cannot lower unclaimed site: {kind} {target!r}")

    @staticmethod
    def _lower_binop(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
        """Dispatch an elementwise binop on which operand carries the array."""
        op = _BINOP_NAMES[claimed.target]
        left, right = claimed.operand_types
        first, second = ctx.operands
        if left == F64_1D and right == F64_1D:
            return LoweredExpr(
                rust=f"__rxtnp_{op}1_aa(&{first}, &{second})?",
                helpers=(rust_snippets.elementwise_aa(op),),
            )
        if left == F64_1D:
            return LoweredExpr(
                rust=f"__rxtnp_{op}1_as(&{first}, {second})?",
                helpers=(rust_snippets.elementwise_as(op),),
            )
        return LoweredExpr(
            rust=f"__rxtnp_{op}1_sa({first}, &{second})?",
            helpers=(rust_snippets.elementwise_sa(op),),
        )

    def crate_dependencies(self) -> tuple[CrateDependency, ...]:
        """Return the pinned crates the generated helpers depend on."""
        return (
            CrateDependency(name="numpy", version="=0.29.0"),
        )


def plugin() -> RextioNumpyPlugin:
    """Entry-point factory for the ``rextio.plugins`` group."""
    return RextioNumpyPlugin()
