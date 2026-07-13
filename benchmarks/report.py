"""JSON and human-readable Markdown report writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.models import ScenarioResult, SuiteReport


def write_json_report(report: SuiteReport, path: Path) -> Path:
    """Write the versioned JSON report; return *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1.0:
        return f"{value:.6f} s"
    return f"{value * 1000.0:.3f} ms"


def _fmt_speedup(value: float | None) -> str:
    if value is None:
        return "n/a (no speedup claim)"
    return f"{value:.4f}x"


def render_markdown(report: SuiteReport) -> str:
    """Render a human-readable Markdown report (honest about losses)."""
    lines: list[str] = []
    lines.append("# rextio-numpy honest benchmark report")
    lines.append("")
    lines.append(f"**Suite status:** `{report.suite_status}`")
    lines.append(f"**Schema version:** `{report.schema_version}`")
    if report.build_wall_s is not None:
        lines.append(f"**Build wall time:** {_fmt_seconds(report.build_wall_s)}")
    else:
        lines.append("**Build wall time:** unavailable")
    lines.append("")

    lines.append("## How to read speedups")
    lines.append("")
    lines.append(
        "- **speedup = fallback_per_call_median / native_per_call_median** (per-call wall latency)."
    )
    lines.append(
        "- Each sample records **raw batch elapsed** wall time for "
        "`iterations_per_sample` calls; **per-call wall latency** is "
        "`batch_elapsed / iterations_per_sample`. Summary stats and speedup "
        "use per-call values only."
    )
    lines.append("- **> 1** means native was **faster** than the Python/NumPy fallback.")
    lines.append(
        "- **< 1** means the fallback was **faster** (native loss). "
        "Sub-1x results are valid and are **not** suppressed."
    )
    lines.append(
        "- This suite **never asserts an expected speedup**. "
        "Native losses (especially the BLAS control) are first-class outcomes."
    )
    lines.append("")

    honesty = report.honesty or {}
    if honesty.get("unfused_chain"):
        lines.append("## Fusion policy")
        lines.append("")
        lines.append(str(honesty["unfused_chain"]))
        lines.append("")
    if honesty.get("blas_control"):
        lines.append("## BLAS control")
        lines.append("")
        lines.append(str(honesty["blas_control"]))
        lines.append("")

    lines.append("## Scenarios")
    lines.append("")

    ok = [s for s in report.scenarios if s.status == "ok"]
    failed = [s for s in report.scenarios if s.status == "failed"]
    skipped = [s for s in report.scenarios if s.status == "skipped"]

    if skipped:
        lines.append("### Skipped")
        lines.append("")
        for s in skipped:
            lines.append(f"- **{s.id}** (`{s.name}`): {s.reason or 'no reason given'}")
        lines.append("")
    if failed:
        lines.append("### Failed")
        lines.append("")
        for s in failed:
            lines.append(f"- **{s.id}** (`{s.name}`): {s.reason or 'no reason given'}")
            if s.verification and s.verification.get("matched") is False:
                lines.append(f"  - verification: {s.verification.get('reason') or 'mismatch'}")
        lines.append("")
    if not ok and not failed and not skipped:
        lines.append("_No scenarios were recorded._")
        lines.append("")

    for s in report.scenarios:
        lines.extend(_render_scenario_section(s))

    lines.append("## Metadata (summary)")
    lines.append("")
    meta = report.metadata or {}
    lines.append(f"- timestamp (UTC): `{meta.get('timestamp_utc')}`")
    py = meta.get("python") or {}
    lines.append(
        f"- Python: `{py.get('implementation')} {py.get('version')}` (`{py.get('executable')}`)"
    )
    plat = meta.get("platform") or {}
    lines.append(f"- platform: `{plat.get('system')} {plat.get('release')} {plat.get('machine')}`")
    pkgs = meta.get("packages") or {}
    lines.append(
        f"- packages (runtime module): numpy=`{pkgs.get('numpy')}`, "
        f"rextio=`{pkgs.get('rextio')}`, rextio-numpy=`{pkgs.get('rextio-numpy')}`"
    )
    dists = meta.get("package_distributions") or {}
    if dists:
        lines.append(
            f"- package distributions (installed): numpy=`{dists.get('numpy')}`, "
            f"rextio=`{dists.get('rextio')}`, rextio-numpy=`{dists.get('rextio-numpy')}`"
        )
    mismatches = meta.get("package_version_mismatches") or {}
    if mismatches:
        lines.append(
            f"- package version mismatches: numpy=`{mismatches.get('numpy')}`, "
            f"rextio=`{mismatches.get('rextio')}`, "
            f"rextio-numpy=`{mismatches.get('rextio-numpy')}`"
        )
    origins = meta.get("package_module_files") or {}
    if origins:
        lines.append(
            f"- package module files: numpy=`{origins.get('numpy')}`, "
            f"rextio=`{origins.get('rextio')}`, "
            f"rextio-numpy=`{origins.get('rextio-numpy')}`"
        )
    tool = meta.get("toolchain") or {}
    lines.append(f"- cargo: `{tool.get('cargo_version')}`, rustc: `{tool.get('rustc_version')}`")
    git = meta.get("git") or {}
    lines.append(
        f"- git: revision=`{git.get('revision')}`, dirty=`{git.get('dirty')}`, "
        f"status=`{git.get('status')}`"
    )
    lines.append("")
    lines.append(
        "_Full metadata (CPU, BLAS/thread env, settings) is in the companion JSON report._"
    )
    lines.append("")
    return "\n".join(lines)


def _render_scenario_section(s: ScenarioResult) -> list[str]:
    lines: list[str] = []
    lines.append(f"### `{s.id}` — {s.name}")
    lines.append("")
    lines.append(f"- **status:** `{s.status}`")
    lines.append(f"- **qualname:** `{s.qualname}`")
    lines.append(f"- **size:** `{s.size}`")
    if s.labels:
        lines.append(f"- **labels:** {', '.join(f'`{x}`' for x in s.labels)}")
    if "unfused" in s.labels or any("UNFUSED" in n for n in s.notes):
        lines.append("- **fusion:** CURRENTLY UNFUSED (measured as-is; no fusion claim).")
    if "blas-control" in s.labels:
        lines.append(
            "- **role:** BLAS-dominated NumPy control — native losses are expected "
            "to appear here and are reported without suppression."
        )
    for note in s.notes:
        lines.append(f"- note: {note}")
    if s.reason:
        lines.append(f"- **reason:** {s.reason}")
    if s.status == "ok":
        fb_leg = s.fallback
        nt_leg = s.native
        fb = fb_leg.summary if fb_leg else None
        nt = nt_leg.summary if nt_leg else None
        lines.append(
            f"- **fallback median (per-call wall):** {_fmt_seconds(fb.median if fb else None)}"
        )
        lines.append(
            f"- **native median (per-call wall):** {_fmt_seconds(nt.median if nt else None)}"
        )
        lines.append(f"- **speedup (fallback/native per-call median):** {_fmt_speedup(s.speedup)}")
        if s.speedup is not None and s.speedup < 1.0:
            lines.append(
                "- **interpretation:** native was **slower** than fallback "
                f"({_fmt_speedup(s.speedup)})."
            )
        elif s.speedup is not None and s.speedup > 1.0:
            lines.append(
                "- **interpretation:** native was **faster** than fallback "
                f"({_fmt_speedup(s.speedup)})."
            )
        elif s.speedup is not None:
            lines.append(
                "- **interpretation:** native and fallback per-call medians matched (~1x)."
            )
        if fb_leg is not None:
            lines.append(
                f"- iterations per sample: `{fb_leg.iterations_per_sample}` "
                f"(per-call wall = raw batch elapsed / iterations)"
            )
            lines.append(
                f"- fallback raw batch samples (s): `{json.dumps(fb_leg.batch_samples_s)}`"
            )
            lines.append(
                f"- fallback per-call wall samples (s): `{json.dumps(fb_leg.per_call_samples_s)}`"
            )
        if nt_leg is not None:
            lines.append(f"- native raw batch samples (s): `{json.dumps(nt_leg.batch_samples_s)}`")
            lines.append(
                f"- native per-call wall samples (s): `{json.dumps(nt_leg.per_call_samples_s)}`"
            )
        if fb:
            lines.append(
                f"- fallback per-call stats: mean={_fmt_seconds(fb.mean)}, "
                f"stdev={_fmt_seconds(fb.stdev)}, min={_fmt_seconds(fb.min)}, "
                f"max={_fmt_seconds(fb.max)}, p95={_fmt_seconds(fb.p95)} "
                f"(method={fb.percentile_method})"
            )
        if nt:
            lines.append(
                f"- native per-call stats: mean={_fmt_seconds(nt.mean)}, "
                f"stdev={_fmt_seconds(nt.stdev)}, min={_fmt_seconds(nt.min)}, "
                f"max={_fmt_seconds(nt.max)}, p95={_fmt_seconds(nt.p95)} "
                f"(method={nt.percentile_method})"
            )
    opt = s.optional_metrics or {}
    if opt:
        lines.append(f"- **optional metrics:** `{json.dumps(opt, sort_keys=True)}`")
    lines.append("")
    return lines


def write_markdown_report(report: SuiteReport, path: Path) -> Path:
    """Write the Markdown report; return *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def report_from_dict(data: dict[str, Any]) -> SuiteReport:
    """Rehydrate a :class:`SuiteReport` from a JSON dict (for tests/roundtrip)."""
    from benchmarks.models import LegTiming, TimingSummary

    def _summary(d: dict[str, Any] | None) -> TimingSummary | None:
        if not d:
            return None
        return TimingSummary(
            median=d.get("median"),
            mean=d.get("mean"),
            stdev=d.get("stdev"),
            min=d.get("min"),
            max=d.get("max"),
            p95=d.get("p95"),
            count=int(d.get("count") or 0),
            percentile_method=str(d.get("percentile_method") or "nearest-rank"),
        )

    def _leg(d: dict[str, Any] | None) -> LegTiming | None:
        if not d:
            return None
        summary = _summary(d.get("summary"))
        if summary is None:
            return None
        iterations = int(d.get("iterations_per_sample") or 0)
        batch = list(d.get("batch_samples_s") or d.get("samples_s") or [])
        per_call = list(d.get("per_call_samples_s") or [])
        if not per_call and batch and iterations >= 1:
            from benchmarks.measure import normalize_batch_to_per_call

            per_call = normalize_batch_to_per_call(batch, iterations)
        return LegTiming(
            mode=d["mode"],
            batch_samples_s=[float(x) for x in batch],
            per_call_samples_s=[float(x) for x in per_call],
            summary=summary,
            iterations_per_sample=iterations,
            warmups=int(d.get("warmups") or 0),
        )

    scenarios: list[ScenarioResult] = []
    for raw in data.get("scenarios") or []:
        scenarios.append(
            ScenarioResult(
                id=str(raw["id"]),
                name=str(raw["name"]),
                description=str(raw.get("description") or ""),
                status=raw["status"],
                qualname=str(raw.get("qualname") or ""),
                size=dict(raw.get("size") or {}),
                labels=list(raw.get("labels") or []),
                notes=list(raw.get("notes") or []),
                reason=raw.get("reason"),
                fallback=_leg(raw.get("fallback")),
                native=_leg(raw.get("native")),
                speedup=raw.get("speedup"),
                verification=raw.get("verification"),
                optional_metrics=dict(raw.get("optional_metrics") or {}),
            )
        )
    return SuiteReport(
        schema_version=str(data.get("schema_version") or ""),
        suite_status=data.get("suite_status") or "failed",
        build_wall_s=data.get("build_wall_s"),
        metadata=dict(data.get("metadata") or {}),
        settings=dict(data.get("settings") or {}),
        scenarios=scenarios,
        honesty=dict(data.get("honesty") or {}),
    )
