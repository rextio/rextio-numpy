"""rextio-numpy: the first-party Rextio plugin for NumPy.

Implements Rextio plugin API 1.1 (``rextio.plugins.api``): the plugin
self-describes its rules AND lowers the initial surface — float64 1-D
element-wise arithmetic, ``numpy.dot``, and whole-array ``sum``/``mean``
reductions — to Rust via the ``ndarray`` crate, with pinned crate injection
and the ``rextio_numpy.types`` annotation vocabulary. The lowering is
certified against CPython NumPy with the core plugin certification kit
(``rextio.plugins.testing``).
"""

from rextio_numpy.__about__ import __version__
from rextio_numpy.plugin import RextioNumpyPlugin, plugin

__all__ = ["RextioNumpyPlugin", "__version__", "plugin"]
