"""Pre-registered honest benchmark scenarios (F64Arr1 vocabulary only).

Scenarios intentionally use only the currently released
``rextio_numpy.types.F64Arr1`` annotation so the suite does not depend on
future type counts from other lanes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CompareKind = Literal["array", "scalar"]

# Package / module layout inside the temporary fixture project.
FIXTURE_PACKAGE = "np_bench"
FIXTURE_MODULE = "kernels"
FIXTURE_QUAL_PREFIX = f"{FIXTURE_PACKAGE}.{FIXTURE_MODULE}"

# Source of the fixture kernels — only F64Arr1 + numpy.dot / elementwise.
#
# Shape constraints (empirically verified against rextio check; do not relax):
# - No function docstrings. Rextio's auto-native path treats a leading string
#   expression statement as unsupported syntax, so a docstring makes the
#   function ``not-candidate`` with empty ``plugin_type_keys`` (silent — no
#   rejection code on the check report). Scenario descriptions live on
#   ScenarioSpec, not in the fixture body.
# - No ``from __future__ import annotations`` required; certification KERNELS
#   also omit it. Postponed annotations alone are *not* the rejection cause.
# - ``large_dot`` must return bare ``np.dot(a, b)``. Wrapping with ``float(...)``
#   is rejected as RXT030 (unsupported unresolved call: float).
KERNELS_SOURCE = '''\
"""Benchmark kernels for the rextio-numpy honest suite (F64Arr1 only)."""

import numpy as np
from rextio_numpy.types import F64Arr1


def small_elementwise(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    return a + b


def multi_op_chain(a: F64Arr1, b: F64Arr1) -> F64Arr1:
    # FUSED route: (a + b) * (a - b) via rextio-numpy/elementwise-chain-fusion.
    return (a + b) * (a - b)


def mixed_control_flow(a: F64Arr1, b: F64Arr1, n: int) -> F64Arr1:
    # Loop + elementwise adds in one typed candidate (mixed control flow).
    c = a + b
    for _i in range(n):
        c = c + b
    return c


def large_dot(a: F64Arr1, b: F64Arr1) -> float:
    # BLAS control; bare np.dot (float() wrapper is RXT030).
    return np.dot(a, b)
'''

REXTIO_TOML = """\
[rust]
build_tool = "cargo"

[plugins]
enabled = ["rextio-numpy"]
"""


@dataclass(frozen=True)
class ScenarioSpec:
    """Static registration of one honest scenario."""

    id: str
    name: str
    description: str
    function_name: str
    compare_kind: CompareKind
    # Deterministic input sizes / parameters (not wall-clock).
    size: dict[str, Any]
    labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Seed for numpy RNG when building inputs.
    seed: int = 0

    @property
    def qualname(self) -> str:
        """Fully-qualified function name in the fixture project."""
        return f"{FIXTURE_QUAL_PREFIX}.{self.function_name}"


def registered_scenarios() -> list[ScenarioSpec]:
    """Return the fixed public scenario registry (order is stable)."""
    return [
        ScenarioSpec(
            id="small_elementwise",
            name="Small-array elementwise",
            description=(
                "Single elementwise add on a small 1-D float64 pair. "
                "Measures basic native-vs-fallback elementwise work."
            ),
            function_name="small_elementwise",
            compare_kind="array",
            size={"n": 64},
            labels=["elementwise", "small"],
            notes=["Uses F64Arr1 only."],
            seed=1,
        ),
        ScenarioSpec(
            id="multi_op_chain",
            name="Multi-op elementwise chain (FUSED)",
            description=(
                "Expression (a + b) * (a - b) on 1-D float64 arrays. "
                "FUSED via rextio-numpy/elementwise-chain-fusion when the fixture "
                "route/generated-source assertion confirms the fusion rule."
            ),
            function_name="multi_op_chain",
            compare_kind="array",
            size={"n": 4096},
            labels=["elementwise", "chain", "fused"],
            notes=[
                "FUSED — fixture asserts check-report claim "
                "rextio-numpy/elementwise-chain-fusion (operand_mode=leaves) "
                "and a __rxtnp_echain_ call inside multi_op_chain's generated body.",
                "No speedup claim is encoded; low-sample runs remain descriptive only.",
            ],
            seed=2,
        ),
        ScenarioSpec(
            id="mixed_control_flow",
            name="Mixed control flow around array ops",
            description=(
                "Elementwise adds inside an explicit range loop on 1-D float64 "
                "arrays. Captures control flow around array operations."
            ),
            function_name="mixed_control_flow",
            compare_kind="array",
            size={"n": 256, "loop_iters": 8},
            labels=["control-flow", "mixed"],
            notes=[
                "Loop + elementwise ops in one typed candidate.",
                "Not a pure single-op kernel.",
            ],
            seed=3,
        ),
        ScenarioSpec(
            id="large_dot_blas_control",
            name="Large 1-D numpy.dot (BLAS control)",
            description=(
                "Large 1-D float64 dot product. On NumPy this path is typically "
                "BLAS-dominated; included to show native losses when they occur. "
                "Sub-1x speedup is valid and reported honestly."
            ),
            function_name="large_dot",
            compare_kind="scalar",
            size={"n": 1_000_000},
            labels=["dot", "blas-control", "large"],
            notes=[
                "BLAS-dominated control on NumPy.",
                "Native may be slower; results below 1x are not suppressed.",
            ],
            seed=4,
        ),
    ]


def scenario_by_id(scenario_id: str) -> ScenarioSpec:
    """Look up a scenario by id; raise KeyError if missing."""
    for spec in registered_scenarios():
        if spec.id == scenario_id:
            return spec
    raise KeyError(scenario_id)
