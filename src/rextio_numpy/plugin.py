"""The rextio-numpy plugin object and entry-point factory.

Implements plugin API 1.2 (``rextio.plugins.api.RextioLoweringPlugin``): the
protocol-v2 describe/covers surface plus the lowering members — annotation
vocabulary, the deterministic claim pass (including keyword/literal axis
metadata), expression lowering, and pinned crate dependencies. The plugin
module itself never imports numpy; only the user-facing
:mod:`rextio_numpy.types` vocabulary module does.

Claim and lower logic live in :mod:`rextio_numpy.claim` and
:mod:`rextio_numpy.lower`; this module is a thin facade.

Import-time contract: this module (and therefore the package root and
:mod:`rextio_numpy.types`) must load without analyzer/config/plugin modules
from core. Generated project runtimes ship a minimal ``rextio`` package
(``__about__``, ``__init__``, ``runtime`` only); fallback wrappers still
import annotation aliases from this package. Core types are imported lazily
inside methods that only run under a full analyzer/plugin host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rextio_numpy.__about__ import __version__

if TYPE_CHECKING:
    from rextio.config.schema import RextioConfig
    from rextio.plugins.api import (
        ClaimResult,
        ClaimSite,
        CoverageDecl,
        CrateDependency,
        LoweredExpr,
        LoweringContext,
        PluginType,
        RuleRecord,
    )
    from rextio.plugins.models import RextioPlugin

PLUGIN_ID = "rextio-numpy"

# Stable Wave-0 re-export (same value as rextio_numpy.diagnostics.F64_1D).
# Defined here as a literal so importing this module never pulls analyzer/
# diagnostics under a minimal generated-runtime rextio package.
F64_1D = "rextio-numpy/f64-1d"

# Re-export for existing test and internal imports.
__all__ = ["F64_1D", "PLUGIN_ID", "RextioNumpyPlugin", "plugin"]


class RextioNumpyPlugin:
    """Plugin API 1.2: describes AND lowers eligible NumPy usage to Rust."""

    plugin_id = PLUGIN_ID
    api_version = "1.2"

    def to_rextio_plugin(self) -> RextioPlugin:
        """Return the v1 metadata Rextio core registers this plugin under."""
        from rextio.plugins.models import RextioPlugin

        from rextio_numpy.rules import COVERAGE

        return RextioPlugin(
            id=PLUGIN_ID,
            name=f"NumPy to Rust (rextio-numpy {__version__})",
            source_language="python",
            target_language="rust",
            packages=COVERAGE.packages,
        )

    def covers(self) -> CoverageDecl:
        """Return the packages, modules, and symbols this plugin covers."""
        from rextio_numpy.rules import COVERAGE

        return COVERAGE

    def describe(self, config: RextioConfig) -> tuple[RuleRecord, ...]:
        """Return the rule records for the resolved project configuration.

        The rule surface is currently config-independent; the parameter is part
        of the protocol so future rules can vary with (for example) import
        policies or target versions.
        """
        from rextio_numpy.rules import numpy_rule_records

        del config
        return numpy_rule_records()

    def type_vocabulary(self) -> tuple[PluginType, ...]:
        """Return the annotation vocabulary this plugin adds to the analyzer."""
        from rextio_numpy.plugin_types import plugin_types

        return plugin_types()

    def claim(self, site: ClaimSite, config: RextioConfig) -> ClaimResult:
        """Decide, at analysis time, whether this plugin lowers the site.

        Deterministic by contract: the decision is a pure function of
        ``(site.kind, site.target, site.operand_types, site.keywords,
        site.operand_literals, site.expression)``. Covered targets with
        unresolved operands return :class:`NotCovered`; covered targets with
        known-but-unsupported operand types return :class:`Rejected` with
        RXTP-NUMPY-010 guidance; multi-op pure-array trees may claim as
        leaves-mode fusion; unsupported call shapes (bare max/min, non-literal
        axis, extra kwargs) return :class:`NotCovered`; everything else is
        :class:`NotCovered`.
        """
        from rextio_numpy.claim import claim as claim_site

        return claim_site(site, config)

    def lower(self, claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
        """Emit the Rust expression for a previously claimed site.

        Every emitted expression is fallible (``pyo3::PyResult``) and ends
        with ``?``; the helper ``fn`` items travel in ``helpers`` and are
        deduplicated by exact text in core codegen. Normalized axis values are
        encoded in helper identity for literal-axis reductions. Fusion claims
        consume ``ctx.leaf_operands`` and the frozen ClaimExpr tree.
        """
        from rextio_numpy.lower import lower as lower_site

        return lower_site(claimed, ctx)

    def crate_dependencies(self) -> tuple[CrateDependency, ...]:
        """Return the pinned crates the generated helpers depend on."""
        from rextio.plugins.api import CrateDependency

        return (CrateDependency(name="numpy", version="=0.29.0"),)


def plugin() -> RextioNumpyPlugin:
    """Entry-point factory for the ``rextio.plugins`` group."""
    return RextioNumpyPlugin()
