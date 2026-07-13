"""Honest, reproducible Rextio/NumPy public benchmark suite.

This package is intentionally independent of the core ``rextio bench`` CLI.
Core bench cannot generate NumPy array arguments and reports a single
in-process mean, so it does not satisfy the requirements of this suite.

Run from the repository root with the project virtualenv active::

    python -m benchmarks --output-dir /tmp/rextio-numpy-bench

See ``benchmarks/README.md`` for full usage, honesty policy, and scenario
descriptions.
"""

from __future__ import annotations

__all__ = ["__version__", "REPORT_SCHEMA_VERSION"]

__version__ = "0.1.0"
# 2.0.0: legs store raw batch elapsed samples and derived per-call wall samples
# separately; summary/speedup always use per-call wall latency.
# 2.1.0: metadata.packages.* is the runtime module version actually imported;
# package_distributions / package_module_files / package_version_mismatches
# record distribution versions, module origins, and explicit mismatches.
REPORT_SCHEMA_VERSION = "2.1.0"
