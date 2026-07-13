"""Data models and schema helpers for the benchmark suite.

Status vocabulary
-----------------
``ok``
    Scenario measured; fallback and native results matched; timings recorded.
``failed``
    Scenario attempted but verification failed, measurement failed, or an
    unexpected error occurred. No speedup claim is emitted.
``skipped``
    Scenario not measured because a hard prerequisite is missing (cargo,
    rustc, plugin, native route, native artifact, etc.). Explicit, never a
    silent fallback timing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ScenarioStatus = Literal["ok", "failed", "skipped"]
SuiteStatus = Literal["ok", "failed", "partial"]

# Percentile method used by :func:`benchmarks.stats.summarize`.
# Nearest-rank: after sorting ascending samples of length n >= 1,
# p95 is samples[ceil(0.95 * n) - 1] (0-based index, clamped to [0, n-1]).
PERCENTILE_METHOD = "nearest-rank"
PERCENTILE_METHOD_DOC = (
    "Nearest-rank: for sorted ascending samples of length n, "
    "p = samples[ceil(p * n) - 1] with index clamped to [0, n-1]. "
    "Defined for n >= 1; empty samples yield null statistics."
)


@dataclass(frozen=True)
class TimingSummary:
    """Deterministic summary statistics over **per-call** wall-time samples (seconds).

    Callers must pass per-call wall latencies (batch elapsed / iterations), never
    raw batch elapsed times, so median/mean/p95 are comparable across iteration
    settings.
    """

    median: float | None
    mean: float | None
    stdev: float | None
    min: float | None
    max: float | None
    p95: float | None
    count: int
    percentile_method: str = PERCENTILE_METHOD

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)


@dataclass
class LegTiming:
    """One measured leg (fallback or native).

    Measurement records one **raw batch elapsed** time per sample (wall clock
    for ``iterations_per_sample`` calls). Per-call wall latency is derived as
    ``batch_elapsed / iterations_per_sample``. Summary statistics and speedup
    use the per-call series only.
    """

    mode: Literal["fallback", "native"]
    # Raw measured wall time for each sample batch (seconds), unnormalized.
    batch_samples_s: list[float]
    # Per-call wall latency samples: batch_samples_s[i] / iterations_per_sample.
    per_call_samples_s: list[float]
    # Summary over *per_call_samples_s* (never over raw batch elapsed).
    summary: TimingSummary
    iterations_per_sample: int
    warmups: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict with unambiguous sample field names."""
        return {
            "mode": self.mode,
            "batch_samples_s": list(self.batch_samples_s),
            "per_call_samples_s": list(self.per_call_samples_s),
            "summary": self.summary.to_dict(),
            "iterations_per_sample": self.iterations_per_sample,
            "warmups": self.warmups,
        }


@dataclass
class ScenarioResult:
    """One scenario outcome (ok / failed / skipped)."""

    id: str
    name: str
    description: str
    status: ScenarioStatus
    qualname: str
    size: dict[str, Any]
    labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reason: str | None = None
    fallback: LegTiming | None = None
    native: LegTiming | None = None
    # fallback_median / native_median; >1 means native faster; <1 means fallback faster.
    # None when status is not ok or when either median is missing/zero.
    speedup: float | None = None
    verification: dict[str, Any] | None = None
    optional_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "qualname": self.qualname,
            "size": dict(self.size),
            "labels": list(self.labels),
            "notes": list(self.notes),
            "reason": self.reason,
            "fallback": None if self.fallback is None else self.fallback.to_dict(),
            "native": None if self.native is None else self.native.to_dict(),
            "speedup": self.speedup,
            "verification": self.verification,
            "optional_metrics": dict(self.optional_metrics),
        }


@dataclass
class SuiteReport:
    """Versioned top-level benchmark report."""

    schema_version: str
    suite_status: SuiteStatus
    build_wall_s: float | None
    metadata: dict[str, Any]
    settings: dict[str, Any]
    scenarios: list[ScenarioResult]
    honesty: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "suite_status": self.suite_status,
            "build_wall_s": self.build_wall_s,
            "metadata": dict(self.metadata),
            "settings": dict(self.settings),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "honesty": dict(self.honesty),
        }


def compute_suite_status(scenarios: list[ScenarioResult]) -> SuiteStatus:
    """Derive suite status from per-scenario outcomes.

    * ``ok`` — every scenario is ``ok``.
    * ``partial`` — at least one ``ok`` and at least one non-ok.
    * ``failed`` — zero ``ok`` scenarios (all skipped/failed, or empty).
    """
    if not scenarios:
        return "failed"
    statuses = {s.status for s in scenarios}
    if statuses == {"ok"}:
        return "ok"
    if "ok" in statuses:
        return "partial"
    return "failed"


def default_honesty() -> dict[str, Any]:
    """Return static honesty policy statements embedded in every report."""
    return {
        "speedup_interpretation": (
            "speedup = fallback_per_call_median_s / native_per_call_median_s. "
            "Each leg stores raw batch_samples_s (measured wall for "
            "iterations_per_sample calls) and per_call_samples_s "
            "(batch_elapsed / iterations_per_sample). Summary stats and speedup "
            "use per-call wall latency only. "
            "Values > 1 mean native was faster; values < 1 mean fallback was faster. "
            "A result below 1x is valid and is rendered honestly — never suppressed."
        ),
        "sample_semantics": (
            "batch_samples_s: raw measured wall seconds for each sample batch "
            "(iterations_per_sample calls between t0 and t1). "
            "per_call_samples_s: batch_samples_s[i] / iterations_per_sample. "
            "summary.*: statistics over per_call_samples_s."
        ),
        "no_expected_speedup": (
            "This suite never encodes or asserts an expected speedup. "
            "Native losses (e.g. BLAS-dominated large dot) are first-class outcomes."
        ),
        "fused_chain": (
            "The public multi_op_chain scenario is statically labeled FUSED in "
            "the registry. Measurement proceeds only after the fixture proves "
            "the fusion rule (check-report claim "
            "rextio-numpy/elementwise-chain-fusion with operand_mode=leaves and "
            "the multi_op_chain generated Rust function body calls "
            "__rxtnp_echain_). A helper definition elsewhere is insufficient. "
            "The static label alone is not proof; failed/skipped reports may "
            "retain the label — status and reason are authoritative. "
            "No speedup is asserted from low-sample runs."
        ),
        "blas_control": (
            "The large 1-D numpy.dot scenario is a BLAS-dominated control on NumPy "
            "and is included to surface native losses when they occur."
        ),
        "silent_fallback_forbidden": (
            "Missing cargo/rustc/plugin/native route/artifact yields an explicit "
            "skipped or failed record and a non-successful suite outcome; "
            "the suite never silently times a fallback-only path as 'native'."
        ),
        "core_bench_not_used": (
            "Core rextio bench cannot generate NumPy array arguments and measures "
            "one in-process mean; this suite does not present core bench as a substitute."
        ),
        "optional_metrics": (
            "Dispatch crossings and temporary allocations are recorded only when "
            "reliably measurable; otherwise structured unavailable entries with reasons."
        ),
        "percentile_method": PERCENTILE_METHOD_DOC,
    }


def unavailable(reason: str) -> dict[str, Any]:
    """Structured 'unavailable' optional-metric entry."""
    return {"status": "unavailable", "reason": reason}
