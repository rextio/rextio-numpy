"""The rextio-numpy plugin object and entry-point factory."""

from __future__ import annotations

from rextio.config.schema import RextioConfig
from rextio.plugins.api import CoverageDecl, RuleRecord
from rextio.plugins.models import RextioPlugin

from rextio_numpy.__about__ import __version__
from rextio_numpy.rules import COVERAGE, numpy_rule_records

PLUGIN_ID = "rextio-numpy"


class RextioNumpyPlugin:
    """Protocol-v2 plugin: self-describes NumPy lowering rules to Rextio core."""

    plugin_id = PLUGIN_ID
    api_version = "1.0"

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


def plugin() -> RextioNumpyPlugin:
    """Entry-point factory for the ``rextio.plugins`` group."""
    return RextioNumpyPlugin()
