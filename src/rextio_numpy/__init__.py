"""rextio-numpy: the first-party Rextio plugin for NumPy.

Implements Rextio plugin API 1.2 (``rextio.plugins.api``): the plugin
self-describes its rules AND lowers the Wave-1/Wave-2 surface — float64/
float32/int64 ranks 1–2 element-wise arithmetic, ``numpy.dot``, whole-array
``sum``/``mean``, and literal-axis ``sum``/``mean``/``max``/``min``
(``axis=<int literal>``) — to Rust via the ``ndarray`` crate, with pinned
crate injection and the ``rextio_numpy.types`` annotation vocabulary. The
lowering is certified against CPython NumPy with the core plugin
certification kit (``rextio.plugins.testing``).
"""

from rextio_numpy.__about__ import __version__
from rextio_numpy.plugin import RextioNumpyPlugin, plugin

__all__ = ["RextioNumpyPlugin", "__version__", "plugin"]
