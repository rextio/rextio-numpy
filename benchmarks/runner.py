"""Orchestrate fixture build, per-scenario measurement, and reporting."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from benchmarks import REPORT_SCHEMA_VERSION, __version__ as suite_version
from benchmarks.fixture import FixtureBuildResult, build_fixture
from benchmarks.measure import measure_leg, verify_leg_results
from benchmarks.metadata import collect_metadata
from benchmarks.models import (
    ScenarioResult,
    SuiteReport,
    compute_suite_status,
    default_honesty,
    unavailable,
)
from benchmarks.report import write_json_report, write_markdown_report
from benchmarks.scenarios import ScenarioSpec, registered_scenarios
from benchmarks.stats import speedup_ratio


@dataclass
class RunnerConfig:
    """User-facing run settings."""

    output_dir: Path
    samples: int = 5
    iterations: int = 50
    warmups: int = 3
    scenario_ids: list[str] | None = None
    python_executable: str | None = None
    repo_root: Path | None = None
    keep_fixture: bool = False
    fixture_dir: Path | None = None


@dataclass
class RunnerHooks:
    """Injectable collaborators for deterministic unit tests."""

    build_fixture: Callable[..., FixtureBuildResult] | None = None
    measure_leg: Callable[..., Any] | None = None
    collect_metadata: Callable[..., dict[str, Any]] | None = None


def _select_scenarios(ids: list[str] | None) -> list[ScenarioSpec]:
    all_specs = registered_scenarios()
    if not ids:
        return list(all_specs)
    wanted = set(ids)
    selected = [s for s in all_specs if s.id in wanted]
    missing = wanted - {s.id for s in selected}
    if missing:
        raise KeyError(f"unknown scenario id(s): {sorted(missing)}")
    return selected


def _optional_metrics() -> dict[str, Any]:
    """Dispatch crossings / temp allocations — unavailable unless measurable."""
    return {
        "dispatch_crossings": unavailable(
            "not reliably measurable in this suite version without invasive "
            "instrumentation of generated wrappers"
        ),
        "temporary_allocations": unavailable(
            "not reliably measurable without a memory profiler attached to each subprocess leg"
        ),
    }


def _skipped_all(
    scenarios: list[ScenarioSpec],
    reason: str,
    *,
    build_wall_s: float | None,
    metadata: dict[str, Any],
    settings: dict[str, Any],
) -> SuiteReport:
    results = [
        ScenarioResult(
            id=s.id,
            name=s.name,
            description=s.description,
            status="skipped",
            qualname=s.qualname,
            size=dict(s.size),
            labels=list(s.labels),
            notes=list(s.notes),
            reason=reason,
            optional_metrics=_optional_metrics(),
        )
        for s in scenarios
    ]
    return SuiteReport(
        schema_version=REPORT_SCHEMA_VERSION,
        suite_status=compute_suite_status(results),
        build_wall_s=build_wall_s,
        metadata=metadata,
        settings=settings,
        scenarios=results,
        honesty=default_honesty(),
    )


def _failed_all(
    scenarios: list[ScenarioSpec],
    reason: str,
    *,
    build_wall_s: float | None,
    metadata: dict[str, Any],
    settings: dict[str, Any],
    routes: dict[str, dict[str, Any]] | None = None,
) -> SuiteReport:
    results: list[ScenarioResult] = []
    for s in scenarios:
        route = (routes or {}).get(s.qualname)
        results.append(
            ScenarioResult(
                id=s.id,
                name=s.name,
                description=s.description,
                status="failed",
                qualname=s.qualname,
                size=dict(s.size),
                labels=list(s.labels),
                notes=list(s.notes),
                reason=reason,
                verification={"route": route} if route else None,
                optional_metrics=_optional_metrics(),
            )
        )
    return SuiteReport(
        schema_version=REPORT_SCHEMA_VERSION,
        suite_status=compute_suite_status(results),
        build_wall_s=build_wall_s,
        metadata=metadata,
        settings=settings,
        scenarios=results,
        honesty=default_honesty(),
    )


def run_suite(config: RunnerConfig, hooks: RunnerHooks | None = None) -> SuiteReport:
    """Execute the full suite and return an in-memory report (no I/O of results)."""
    hooks = hooks or RunnerHooks()
    scenarios = _select_scenarios(config.scenario_ids)
    python_exe = config.python_executable or sys.executable
    repo_root = config.repo_root
    settings = {
        "suite_version": suite_version,
        "schema_version": REPORT_SCHEMA_VERSION,
        "samples": config.samples,
        "iterations": config.iterations,
        "warmups": config.warmups,
        "scenario_ids": [s.id for s in scenarios],
        "python_executable": python_exe,
        "output_dir": str(config.output_dir),
    }
    meta_fn = hooks.collect_metadata or collect_metadata
    metadata = meta_fn(repo_root=repo_root)

    # Fixture lifecycle:
    # - explicit ``fixture_dir``: user-owned path, never auto-deleted
    # - ``keep_fixture`` without fixture_dir: ``mkdtemp`` (survives GC; reported)
    # - otherwise: TemporaryDirectory cleaned up on all exit paths
    tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
    fixture_root: Path
    if config.fixture_dir is not None:
        fixture_root = Path(config.fixture_dir)
        fixture_root.mkdir(parents=True, exist_ok=True)
    elif config.keep_fixture:
        # mkdtemp is not auto-deleted on object finalization (unlike
        # TemporaryDirectory); that is the only durable keep-fixture path.
        fixture_root = Path(tempfile.mkdtemp(prefix="rextio-numpy-bench-"))
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="rextio-numpy-bench-")
        fixture_root = Path(tmp_ctx.name)

    settings["fixture_dir"] = str(fixture_root)
    settings["keep_fixture"] = bool(config.keep_fixture or config.fixture_dir is not None)

    build_fn = hooks.build_fixture or build_fixture
    try:
        fixture: FixtureBuildResult = build_fn(fixture_root, scenarios)
    except Exception as exc:  # noqa: BLE001
        report = _failed_all(
            scenarios,
            f"fixture build raised {type(exc).__name__}: {exc}",
            build_wall_s=None,
            metadata=metadata,
            settings=settings,
        )
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        return report

    if fixture.status == "skipped":
        report = _skipped_all(
            scenarios,
            fixture.reason or "fixture prerequisites missing",
            build_wall_s=fixture.build_wall_s,
            metadata=metadata,
            settings=settings,
        )
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        return report

    if fixture.status != "ok" or fixture.build_python_dir is None:
        report = _failed_all(
            scenarios,
            fixture.reason or "fixture build failed",
            build_wall_s=fixture.build_wall_s,
            metadata=metadata,
            settings=settings,
            routes=fixture.native_routes,
        )
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        return report

    measure = hooks.measure_leg or measure_leg
    results: list[ScenarioResult] = []
    try:
        for spec in scenarios:
            results.append(
                _run_one_scenario(
                    spec,
                    build_python_dir=fixture.build_python_dir,
                    python_exe=python_exe,
                    samples=config.samples,
                    iterations=config.iterations,
                    warmups=config.warmups,
                    measure=measure,
                    route=fixture.native_routes.get(spec.qualname),
                )
            )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    return SuiteReport(
        schema_version=REPORT_SCHEMA_VERSION,
        suite_status=compute_suite_status(results),
        build_wall_s=fixture.build_wall_s,
        metadata=metadata,
        settings=settings,
        scenarios=results,
        honesty=default_honesty(),
    )


def _run_one_scenario(
    spec: ScenarioSpec,
    *,
    build_python_dir: Path,
    python_exe: str,
    samples: int,
    iterations: int,
    warmups: int,
    measure: Callable[..., Any],
    route: dict[str, Any] | None,
) -> ScenarioResult:
    """Measure one scenario with separate fallback/native subprocesses."""
    base_kwargs = dict(
        python_executable=python_exe,
        build_python_dir=build_python_dir,
        spec=spec,
        warmups=warmups,
        iterations=iterations,
        samples=samples,
    )
    try:
        fb_timing, fb_raw = measure(mode="fallback", **base_kwargs)
        nt_timing, nt_raw = measure(mode="native", **base_kwargs)
    except Exception as exc:  # noqa: BLE001
        return ScenarioResult(
            id=spec.id,
            name=spec.name,
            description=spec.description,
            status="failed",
            qualname=spec.qualname,
            size=dict(spec.size),
            labels=list(spec.labels),
            notes=list(spec.notes),
            reason=f"measurement raised {type(exc).__name__}: {exc}",
            verification={"route": route} if route else None,
            optional_metrics=_optional_metrics(),
        )

    if fb_timing is None or nt_timing is None:
        err_parts = []
        if fb_timing is None:
            err_parts.append(f"fallback: {fb_raw.get('error') or fb_raw}")
        if nt_timing is None:
            err_parts.append(f"native: {nt_raw.get('error') or nt_raw}")
        return ScenarioResult(
            id=spec.id,
            name=spec.name,
            description=spec.description,
            status="failed",
            qualname=spec.qualname,
            size=dict(spec.size),
            labels=list(spec.labels),
            notes=list(spec.notes),
            reason="; ".join(err_parts),
            verification={"route": route} if route else None,
            optional_metrics=_optional_metrics(),
        )

    verification = verify_leg_results(fb_raw, nt_raw, compare_kind=spec.compare_kind)
    if route:
        verification = {**verification, "route": route}

    if not verification.get("matched"):
        return ScenarioResult(
            id=spec.id,
            name=spec.name,
            description=spec.description,
            status="failed",
            qualname=spec.qualname,
            size=dict(spec.size),
            labels=list(spec.labels),
            notes=list(spec.notes),
            reason=verification.get("reason") or "result mismatch",
            fallback=fb_timing,
            native=nt_timing,
            speedup=None,  # mismatch → no speedup claim
            verification=verification,
            optional_metrics=_optional_metrics(),
        )

    ratio = speedup_ratio(fb_timing.summary.median, nt_timing.summary.median)
    return ScenarioResult(
        id=spec.id,
        name=spec.name,
        description=spec.description,
        status="ok",
        qualname=spec.qualname,
        size=dict(spec.size),
        labels=list(spec.labels),
        notes=list(spec.notes),
        reason=None,
        fallback=fb_timing,
        native=nt_timing,
        speedup=ratio,
        verification=verification,
        optional_metrics=_optional_metrics(),
    )


def write_reports(report: SuiteReport, output_dir: Path) -> tuple[Path, Path]:
    """Write JSON + Markdown under *output_dir*; return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json_report(report, output_dir / "benchmark-report.json")
    md_path = write_markdown_report(report, output_dir / "benchmark-report.md")
    return json_path, md_path
