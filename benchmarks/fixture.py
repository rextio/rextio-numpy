"""Temporary fixture project creation and native-build verification.

Uses the same layout as the certification tests: a small package with
``rextio.toml`` enabling ``rextio-numpy``, built via core's
``build_certification_project`` when available, or an equivalent
``rextio build`` invocation. Generated check/build reports are inspected so
every measured target is confirmed natively served — missing tools, plugins,
routes, or artifacts become explicit skipped/failed records, never silent
fallback timings labeled as native.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from benchmarks.scenarios import (
    FIXTURE_PACKAGE,
    KERNELS_SOURCE,
    REXTIO_TOML,
    ScenarioSpec,
)

# Injectable hooks for unit tests (no real cargo).
BuildFn = Callable[[Path], dict[str, Any]]
WhichFn = Callable[[str], str | None]


@dataclass
class FixtureBuildResult:
    """Outcome of creating and building the temporary fixture project."""

    project_root: Path | None
    build_python_dir: Path | None
    build_wall_s: float | None
    status: str  # "ok" | "failed" | "skipped"
    reason: str | None = None
    build_report: dict[str, Any] | None = None
    check_report: dict[str, Any] | None = None
    native_routes: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the fixture is ready for measurement."""
        return self.status == "ok"


def write_fixture_project(root: Path) -> Path:
    """Materialize the fixture project under *root* and return *root*."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "rextio.toml").write_text(REXTIO_TOML, encoding="utf-8")
    package = root / "src" / FIXTURE_PACKAGE
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNELS_SOURCE, encoding="utf-8")
    return root


def _default_build(project_root: Path) -> dict[str, Any]:
    """Build via core certification kit (asserts native_build == built)."""
    from rextio.plugins.testing import CertificationError, build_certification_project

    try:
        certified = build_certification_project(project_root)
    except CertificationError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "project_root": str(project_root),
        }
    except Exception as exc:  # noqa: BLE001 - surface any build failure honestly
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "project_root": str(project_root),
        }
    return {
        "ok": True,
        "project_root": str(certified.project_root),
        "build_python_dir": str(certified.build_python_dir),
    }


def load_json_report(path: Path) -> dict[str, Any] | None:
    """Load a JSON report file, or None if missing/invalid."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def extract_function_routes(check_report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Map qualname -> {native_status, route} from a check report."""
    routes: dict[str, dict[str, Any]] = {}
    if not check_report:
        return routes
    for module in check_report.get("modules", ()) or ():
        for function in module.get("functions", ()) or ():
            qual = function.get("qualname")
            if not qual:
                continue
            routes[str(qual)] = {
                "native_status": function.get("native_status"),
                "route": function.get("route"),
            }
    return routes


def is_natively_served(route_info: dict[str, Any] | None) -> bool:
    """Return whether the check report marks the function as natively served."""
    if not route_info:
        return False
    route = str(route_info.get("route") or "")
    status = route_info.get("native_status")
    return status == "accepted" and (route == "native-direct" or route.startswith("native-plugin:"))


def verify_native_targets(
    check_report: dict[str, Any] | None,
    scenarios: list[ScenarioSpec],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return (routes, list of problem strings) for requested scenarios."""
    routes = extract_function_routes(check_report)
    problems: list[str] = []
    if check_report is None:
        problems.append("missing or unreadable .rextio/reports/check.json")
        return routes, problems
    for spec in scenarios:
        info = routes.get(spec.qualname)
        if info is None:
            problems.append(f"{spec.qualname} not found in check report")
            continue
        if not is_natively_served(info):
            problems.append(
                f"{spec.qualname} is not natively served "
                f"(native_status={info.get('native_status')!r}, "
                f"route={info.get('route')!r})"
            )
    return routes, problems


def preflight_tools(which: WhichFn | None = None) -> list[str]:
    """Return human-readable problems for missing cargo/rustc (empty if ok)."""
    which_fn = which or shutil.which
    problems: list[str] = []
    if which_fn("cargo") is None:
        problems.append("cargo not found on PATH")
    if which_fn("rustc") is None:
        problems.append("rustc not found on PATH")
    return problems


def build_fixture(
    root: Path,
    scenarios: list[ScenarioSpec],
    *,
    which: WhichFn | None = None,
    build_fn: BuildFn | None = None,
    clock: Callable[[], float] | None = None,
) -> FixtureBuildResult:
    """Write, build, and verify the fixture project.

    *which* and *build_fn* are injectable for deterministic unit tests.
    """
    which_fn = which or shutil.which
    build = build_fn or _default_build
    now = clock or time.perf_counter

    tool_problems = preflight_tools(which_fn)
    if tool_problems:
        return FixtureBuildResult(
            project_root=None,
            build_python_dir=None,
            build_wall_s=None,
            status="skipped",
            reason="; ".join(tool_problems),
        )

    project_root = write_fixture_project(root)
    t0 = now()
    build_outcome = build(project_root)
    build_wall = float(now() - t0)

    if not build_outcome.get("ok"):
        # Still try to load reports for diagnostics.
        build_report = load_json_report(project_root / ".rextio" / "reports" / "build.json")
        check_report = load_json_report(project_root / ".rextio" / "reports" / "check.json")
        native_build = (build_report or {}).get("native_build") or {}
        reason = str(build_outcome.get("error") or "build failed")
        if native_build and native_build.get("status") != "built":
            reason = (
                f"native artifact not built (native_build.status="
                f"{native_build.get('status')!r}): {reason}"
            )
        return FixtureBuildResult(
            project_root=project_root,
            build_python_dir=None,
            build_wall_s=build_wall,
            status="failed",
            reason=reason,
            build_report=build_report,
            check_report=check_report,
        )

    build_python = Path(
        str(
            build_outcome.get("build_python_dir") or (project_root / ".rextio" / "build" / "python")
        )
    )
    build_report = load_json_report(project_root / ".rextio" / "reports" / "build.json")
    check_report = load_json_report(project_root / ".rextio" / "reports" / "check.json")

    # Explicit native_build gate: missing or non-built reports fail closed
    # (never treat a fallback-only tree as a successful native fixture).
    if build_report is None:
        return FixtureBuildResult(
            project_root=project_root,
            build_python_dir=None,
            build_wall_s=build_wall,
            status="failed",
            reason="missing or unreadable .rextio/reports/build.json after build",
            build_report=None,
            check_report=check_report,
        )
    native_build = build_report.get("native_build") or {}
    if native_build.get("status") != "built":
        return FixtureBuildResult(
            project_root=project_root,
            build_python_dir=None,
            build_wall_s=build_wall,
            status="failed",
            reason=(
                f"native artifact not built (native_build.status={native_build.get('status')!r})"
            ),
            build_report=build_report,
            check_report=check_report,
        )

    routes, problems = verify_native_targets(check_report, scenarios)
    if problems:
        return FixtureBuildResult(
            project_root=project_root,
            build_python_dir=build_python if build_python.is_dir() else None,
            build_wall_s=build_wall,
            status="failed",
            reason="; ".join(problems),
            build_report=build_report,
            check_report=check_report,
            native_routes=routes,
        )

    if not build_python.is_dir():
        return FixtureBuildResult(
            project_root=project_root,
            build_python_dir=None,
            build_wall_s=build_wall,
            status="failed",
            reason=f"missing build python dir: {build_python}",
            build_report=build_report,
            check_report=check_report,
            native_routes=routes,
        )

    return FixtureBuildResult(
        project_root=project_root,
        build_python_dir=build_python,
        build_wall_s=build_wall,
        status="ok",
        reason=None,
        build_report=build_report,
        check_report=check_report,
        native_routes=routes,
    )
