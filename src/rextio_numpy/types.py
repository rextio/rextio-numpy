"""The user-facing annotation vocabulary of rextio-numpy.

Only *user projects* import this module (it imports numpy); the plugin object
itself never does. At Python runtime the names here are plain aliases of
``numpy.ndarray``, so annotated code runs unchanged on the CPython fallback
and without Rextio entirely. The Rextio analyzer resolves the dotted spelling
``rextio_numpy.types.F64Arr1`` to the plugin type ``rextio-numpy/f64-1d``
when the plugin is enabled (docs/specs/plugin-lowering.md section 1).
"""

import numpy

F64Arr1 = numpy.ndarray
"""A 1-D ``float64`` NumPy array.

A plain runtime alias of ``numpy.ndarray`` — it adds no runtime behavior and
no validation. Its value is static: the Rextio analyzer resolves a parameter
or return annotated ``F64Arr1`` to the plugin type ``rextio-numpy/f64-1d``,
which makes the function a native candidate whose array arguments cross the
boundary as read-only C-contiguous float64 1-D arrays.
"""
