"""rextio-numpy: the first-party Rextio plugin for NumPy.

Implements Rextio plugin protocol v2 (``rextio.plugins.api``): the plugin
self-describes the rules for lowering eligible NumPy usage to native Rust.
In this release the plugin ships **rule records and coverage only** — the
declarative contract surfaced by ``rextio capabilities`` and the RXT091
plugin-lowerable hint. Actual lowering activates once rextio core exposes the
plugin ``lower()`` hook.
"""

from rextio_numpy.__about__ import __version__
from rextio_numpy.plugin import RextioNumpyPlugin, plugin

__all__ = ["RextioNumpyPlugin", "__version__", "plugin"]
