"""The user-facing annotation vocabulary of rextio-numpy.

Only *user projects* import this module (it imports numpy); the plugin object
itself never does. At Python runtime the names here are plain aliases of
``numpy.ndarray``, so annotated code runs unchanged on the CPython fallback
and without Rextio entirely. The Rextio analyzer resolves the dotted spellings
(e.g. ``rextio_numpy.types.F64Arr1``) to the matching plugin type keys
(e.g. ``rextio-numpy/f64-1d``) when the plugin is enabled
(docs/specs/plugin-lowering.md section 1).
"""

import numpy

F64Arr1 = numpy.ndarray
"""A 1-D ``float64`` NumPy array.

A plain runtime alias of ``numpy.ndarray`` — it adds no runtime behavior and
no validation. Its value is static: the Rextio analyzer resolves a parameter
or return annotated ``F64Arr1`` to the plugin type ``rextio-numpy/f64-1d``,
which makes the function a native candidate whose array arguments cross the
boundary as read-only float64 1-D arrays. Non-contiguous (strided) views are
accepted — the **input** conversion materializes an owned contiguous copy at
the boundary (certified; still a copy, not zero-copy residency). Array
**returns** transfer ownership of the owned Rust buffer into a NumPy array
via rust-numpy ``IntoPyArray`` (no second elementwise copy on the return
path). A wrong dtype or rank raises the PyO3 conversion error in native mode.
"""

F64Arr2 = numpy.ndarray
"""A 2-D ``float64`` NumPy array (plugin type ``rextio-numpy/f64-2d``)."""

F32Arr1 = numpy.ndarray
"""A 1-D ``float32`` NumPy array (plugin type ``rextio-numpy/f32-1d``)."""

F32Arr2 = numpy.ndarray
"""A 2-D ``float32`` NumPy array (plugin type ``rextio-numpy/f32-2d``)."""

I64Arr1 = numpy.ndarray
"""A 1-D ``int64`` NumPy array (plugin type ``rextio-numpy/i64-1d``)."""

I64Arr2 = numpy.ndarray
"""A 2-D ``int64`` NumPy array (plugin type ``rextio-numpy/i64-2d``)."""
