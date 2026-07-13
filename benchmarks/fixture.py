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


FUSION_RULE_ID = "rextio-numpy/elementwise-chain-fusion"
FUSION_HELPER_PREFIX = "__rxtnp_echain_"


def function_has_fusion_claim(check_report: dict[str, Any] | None, qualname: str) -> bool:
    """Return whether *qualname* has an outer leaves-mode fusion claim."""
    if not check_report:
        return False
    for module in check_report.get("modules", ()) or ():
        for function in module.get("functions", ()) or ():
            if function.get("qualname") != qualname:
                continue
            for claim in function.get("plugin_claims", ()) or ():
                if claim.get("rule_id") == FUSION_RULE_ID:
                    mode = claim.get("operand_mode") or "direct"
                    return mode == "leaves"
            return False
    return False


def scenario_fusion_label_state(labels: list[str], notes: list[str]) -> str:
    """Return authoritative fusion label state for honesty validation.

    Returns one of: ``fused``, ``unfused``, ``none``, ``conflict``.

    Labels are exclusive: fused only when label ``fused`` is present and
    ``unfused`` is absent. Notes use ``UNFUSED`` before ``FUSED`` so that
    the substring ``UNFUSED`` never counts as a fused note.
    """
    has_fused_label = "fused" in labels
    has_unfused_label = "unfused" in labels
    if has_fused_label and has_unfused_label:
        return "conflict"

    note_fused = False
    note_unfused = False
    for note in notes:
        if "UNFUSED" in note:
            note_unfused = True
        elif "FUSED" in note:
            note_fused = True
    if note_fused and note_unfused:
        return "conflict"
    if has_fused_label and note_unfused:
        return "conflict"
    if has_unfused_label and note_fused:
        return "conflict"

    if has_fused_label or (note_fused and not has_unfused_label):
        # Authoritative fused: label fused without unfused, or fused note alone.
        if has_unfused_label:
            return "conflict"
        return "fused"
    if has_unfused_label or note_unfused:
        return "unfused"
    return "none"


def rust_function_name_for_qualname(qualname: str) -> str:
    """Core mangling: ``pkg.mod.fn`` → ``pkg__mod__fn``."""
    return qualname.replace(".", "__")


def _extract_balanced_fn_body(source: str, fn_name: str) -> str | None:
    """Return the body of ``fn {fn_name}...`` including braces, or None if ambiguous."""
    import re

    # Match `fn name` not as a prefix of a longer identifier.
    pattern = re.compile(rf"\bfn\s+{re.escape(fn_name)}\b")
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        return None  # missing or ambiguous
    m = matches[0]
    brace_at = source.find("{", m.end())
    if brace_at < 0:
        return None
    depth = 0
    for i in range(brace_at, len(source)):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[brace_at : i + 1]
    return None


def function_body_calls_fusion_helper(
    project_root: Path,
    *,
    qualname: str,
    function_name: str,
) -> bool:
    """Report whether the generated Rust fn for *qualname* calls ``__rxtnp_echain_``.

    Locates the mangled function body and requires an echain *call* inside it.
    A helper definition elsewhere (or a call in another function) is not enough.
    Fails closed on missing/ambiguous function location.
    """
    mangled = rust_function_name_for_qualname(qualname)
    # Also accept a trailing-only match used by some layouts.
    candidates = (mangled, function_name)
    root = Path(project_root) / ".rextio"
    search_roots = (root / "generated", root / "build")
    bodies_found = 0
    call_found = False
    for rust_root in search_roots:
        if not rust_root.is_dir():
            continue
        for path in rust_root.rglob("*.rs"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for name in candidates:
                body = _extract_balanced_fn_body(text, name)
                if body is None:
                    continue
                bodies_found += 1
                # Require a call site, not merely a string in a comment-free form.
                if f"{FUSION_HELPER_PREFIX}" in body and "(" in body:
                    # e.g. __rxtnp_echain_...(
                    import re

                    if re.search(rf"{re.escape(FUSION_HELPER_PREFIX)}\w*\s*\(", body):
                        call_found = True
    # Exactly one located body across the tree is ideal; if multiple candidate
    # names hit, still accept only when a call was found and no pure ambiguity
    # without a call. Fail closed when nothing located.
    if bodies_found == 0:
        return False
    return call_found


def generated_source_has_fusion_helper(project_root: Path) -> bool:
    """Scan for any echain helper in the project (not function-scoped).

    Prefer :func:`function_body_calls_fusion_helper` for honesty gates.
    Kept for debugging only; honesty validation does not use this alone.
    """
    root = Path(project_root) / ".rextio"
    search_roots = (root / "generated", root / "build")
    for rust_root in search_roots:
        if not rust_root.is_dir():
            continue
        for path in rust_root.rglob("*.rs"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if FUSION_HELPER_PREFIX in text:
                return True
    return False


def verify_multi_op_chain_fusion(
    check_report: dict[str, Any] | None,
    project_root: Path,
    scenarios: list[ScenarioSpec],
) -> list[str]:
    """Fail closed when multi_op_chain is labeled FUSED without fusion evidence.

    Requires a check-report fusion claim and an ``__rxtnp_echain_`` *call*
    inside the generated Rust function body for that scenario (not merely a
    helper definition elsewhere).
    """
    problems: list[str] = []
    chain_specs = [
        s for s in scenarios if s.id == "multi_op_chain" or s.function_name == "multi_op_chain"
    ]
    if not chain_specs:
        return problems
    for spec in chain_specs:
        state = scenario_fusion_label_state(list(spec.labels), list(spec.notes))
        if state == "conflict":
            problems.append(
                f"{spec.qualname} has contradictory fusion labels/notes "
                f"(labels={list(spec.labels)!r}, notes={list(spec.notes)!r})"
            )
            continue
        if state != "fused":
            continue
        if not function_has_fusion_claim(check_report, spec.qualname):
            problems.append(
                f"{spec.qualname} is labeled FUSED but check report lacks "
                f"rule_id={FUSION_RULE_ID!r} with operand_mode=leaves"
            )
        if not function_body_calls_fusion_helper(
            project_root,
            qualname=spec.qualname,
            function_name=spec.function_name,
        ):
            problems.append(
                f"{spec.qualname} is labeled FUSED but its generated Rust "
                f"function body does not call {FUSION_HELPER_PREFIX!r} "
                "(helper definition elsewhere is not sufficient)"
            )
    return problems


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
    fusion_problems = verify_multi_op_chain_fusion(
        check_report=check_report,
        project_root=project_root,
        scenarios=scenarios,
    )
    problems.extend(fusion_problems)
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
