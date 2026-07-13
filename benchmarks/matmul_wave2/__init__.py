"""Standalone Wave 2 rank-2 f64 matmul research harness.

Isolated from the product claim/lower/rule/plugin surface. Implements the
frozen preregistration protocol (2026-07-13) for research measurement only.
Product verdict is always NO-GO because dispatchability fails a priori.
"""

from __future__ import annotations

__all__ = ["PROTOCOL_ID", "SCHEMA_VERSION"]

PROTOCOL_ID = "preregister-matmul-wave2-2026-07-13"
SCHEMA_VERSION = "matmul-wave2-research-v1"
