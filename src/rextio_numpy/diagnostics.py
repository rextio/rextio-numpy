"""Shared rejection guidance helpers for rextio-numpy claim decisions.

Holds the plugin type keys and the RXTP-NUMPY-010 rejection builder used by
the claim routers. Guidance text is taken from the matching rule record so
message/suggestion stay synchronized with the described surface.
"""

from __future__ import annotations

from rextio.analyzer.diagnostics import Diagnostic
from rextio.plugins.api import ClaimResult, ClaimSite, NotCovered, Rejected

from rextio_numpy.rules import numpy_rule_records

#: Plugin type keys for the Wave-1 array surface (dtype × rank).
F64_1D = "rextio-numpy/f64-1d"
F64_2D = "rextio-numpy/f64-2d"
F32_1D = "rextio-numpy/f32-1d"
F32_2D = "rextio-numpy/f32-2d"
I64_1D = "rextio-numpy/i64-1d"
I64_2D = "rextio-numpy/i64-2d"
BOOL_1D = "rextio-numpy/bool-1d"
BOOL_2D = "rextio-numpy/bool-2d"

#: All array type keys this plugin owns.
ARRAY_TYPE_KEYS: frozenset[str] = frozenset({F64_1D, F64_2D, F32_1D, F32_2D, I64_1D, I64_2D})
BOOL_TYPE_KEYS: frozenset[str] = frozenset({BOOL_1D, BOOL_2D})
PLUGIN_ARRAY_TYPE_KEYS: frozenset[str] = ARRAY_TYPE_KEYS | BOOL_TYPE_KEYS

# dtype token -> (rank -> type key)
_TYPE_KEY_BY_DTYPE_RANK: dict[str, dict[int, str]] = {
    "f64": {1: F64_1D, 2: F64_2D},
    "f32": {1: F32_1D, 2: F32_2D},
    "i64": {1: I64_1D, 2: I64_2D},
}

# type key -> (dtype token, rank)
_ARRAY_META: dict[str, tuple[str, int]] = {
    F64_1D: ("f64", 1),
    F64_2D: ("f64", 2),
    F32_1D: ("f32", 1),
    F32_2D: ("f32", 2),
    I64_1D: ("i64", 1),
    I64_2D: ("i64", 2),
}

_BOOL_RANK: dict[str, int] = {
    BOOL_1D: 1,
    BOOL_2D: 2,
}

# dtype token -> core scalar type name used in claim operand_types
SCALAR_FOR_DTYPE: dict[str, str] = {
    "f64": "float",
    "f32": "float",
    "i64": "int",
}


def array_meta(type_key: str) -> tuple[str, int] | None:
    """Return ``(dtype, rank)`` for a plugin array key, else None."""
    return _ARRAY_META.get(type_key)


def type_key_for(dtype: str, rank: int) -> str:
    """Return the plugin type key for ``dtype`` at ``rank``."""
    return _TYPE_KEY_BY_DTYPE_RANK[dtype][rank]


def is_array_type(type_key: str | None) -> bool:
    """Report whether ``type_key`` is one of this plugin's numeric array types."""
    return type_key is not None and type_key in ARRAY_TYPE_KEYS


def is_bool_type(type_key: str | None) -> bool:
    """Report whether ``type_key`` is a resident boolean array result type."""
    return type_key is not None and type_key in BOOL_TYPE_KEYS


def bool_rank(type_key: str) -> int | None:
    """Return the resident boolean-array rank, else ``None``."""
    return _BOOL_RANK.get(type_key)


def bool_type_for(rank: int) -> str:
    """Return the resident boolean-array key for rank 1 or 2."""
    if rank == 1:
        return BOOL_1D
    if rank == 2:
        return BOOL_2D
    raise KeyError(rank)


# The remediation guidance for claim rejections comes from the rule record
# that owns diagnostic code RXTP-NUMPY-010 (unsupported dtype/operand types).
_REJECTION_GUIDANCE = next(
    record.guidance for record in numpy_rule_records() if record.diagnostic_code == "RXTP-NUMPY-010"
)


def not_covered_or_rejected(site: ClaimSite) -> ClaimResult:
    """Resolve a covered-target miss: NotCovered when unresolved, else Rejected."""
    if any(operand is None for operand in site.operand_types):
        return NotCovered()
    named = ", ".join(str(operand) for operand in site.operand_types)
    return Rejected(
        diagnostic=Diagnostic(
            code="RXTP-NUMPY-010",
            severity="error",
            message=(
                f"rextio-numpy cannot lower {site.target!r}: operand types "
                f"({named}) are outside the supported array surface "
                f"(float64/float32/int64 ranks 1–2, resident bool masks, plus "
                f"matching scalars for elementwise operations; keys "
                f"{sorted(PLUGIN_ARRAY_TYPE_KEYS)})"
            ),
            file_path="",
            line=0,
            column=0,
            suggestion=_REJECTION_GUIDANCE,
        )
    )
