"""Experimental F64 rank-1 NumPy boundary-allocation PoC harness.

Research-only. Does not modify production BoundaryConversion, claim, lower,
or plugin types. Makes no published speed claim.

See ``benchmarks/boundary_allocation_poc/README.md``.
"""

from __future__ import annotations

PROTOCOL_ID = "boundary-allocation-poc-f64-r1-2026-07-27"
SCHEMA_VERSION = "boundary-allocation-poc-v1"

__all__ = ["PROTOCOL_ID", "SCHEMA_VERSION"]
