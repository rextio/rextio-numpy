"""Deterministic suite tests: schema, status, subprocess env, reports, failures.

No cargo, no real wall-clock speed assertions. Clocks, subprocesses, metadata,
and fixture builds are injected or fixed.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from benchmarks import REPORT_SCHEMA_VERSION
from benchmarks.fixture import (
    FixtureBuildResult,
    extract_function_routes,
    is_natively_served,
    preflight_tools,
    verify_native_targets,
    write_fixture_project,
)
from benchmarks.measure import (
    SubprocessSpec,
    build_leg_subprocess_spec,
    measure_leg,
    normalize_batch_to_per_call,
    parse_result_dtype,
    reconstruct_result,
    validate_module_under_build_tree,
    verify_leg_results,
)
from benchmarks.metadata import (
    collect_metadata,
    collect_package_provenance,
    resolve_package_provenance,
)
from benchmarks.models import (
    LegTiming,
    ScenarioResult,
    SuiteReport,
    compute_suite_status,
    default_honesty,
)
from benchmarks.report import render_markdown, report_from_dict, write_json_report
from benchmarks.runner import RunnerConfig, RunnerHooks, run_suite, write_reports
from benchmarks.scenarios import (
    FIXTURE_QUAL_PREFIX,
    KERNELS_SOURCE,
    registered_scenarios,
    scenario_by_id,
)
from benchmarks.stats import summarize


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _leg(
    mode: str,
    per_call_samples: list[float],
    *,
    iterations_per_sample: int = 10,
    batch_samples: list[float] | None = None,
) -> LegTiming:
    """Build a leg from per-call samples; reconstruct batch as per_call * iters."""
    if batch_samples is None:
        batch_samples = [x * iterations_per_sample for x in per_call_samples]
    return LegTiming(
        mode=mode,  # type: ignore[arg-type]
        batch_samples_s=list(batch_samples),
        per_call_samples_s=list(per_call_samples),
        summary=summarize(per_call_samples),
        iterations_per_sample=iterations_per_sample,
        warmups=1,
    )


def _ok_fixture(root: Path) -> FixtureBuildResult:
    build_python = root / ".rextio" / "build" / "python"
    build_python.mkdir(parents=True, exist_ok=True)
    routes = {
        s.qualname: {"native_status": "accepted", "route": "native-plugin:rextio-numpy"}
        for s in registered_scenarios()
    }
    return FixtureBuildResult(
        project_root=root,
        build_python_dir=build_python,
        build_wall_s=1.25,
        status="ok",
        native_routes=routes,
        build_report={"native_build": {"status": "built"}},
        check_report={
            "modules": [
                {
                    "functions": [
                        {
                            "qualname": q,
                            "native_status": info["native_status"],
                            "route": info["route"],
                        }
                        for q, info in routes.items()
                    ]
                }
            ]
        },
    )


def _measure_ok(mode: str, spec: Any, **_kwargs: Any) -> tuple[LegTiming, dict[str, Any]]:
    # Deterministic fake timings: fallback slower so speedup > 1 for most tests.
    # Values are per-call wall latencies; batch = per_call * iterations.
    iterations = int(_kwargs.get("iterations", 10))
    result: dict[str, Any]
    if mode == "fallback":
        per_call = [0.020, 0.021, 0.019, 0.022, 0.020]
        result = {"kind": "array", "data": [1.0, 2.0], "dtype": "float64"}
        if spec.compare_kind == "scalar":
            result = {"kind": "scalar", "data": 42.0}
    else:
        per_call = [0.010, 0.011, 0.009, 0.010, 0.010]
        result = {"kind": "array", "data": [1.0, 2.0], "dtype": "float64"}
        if spec.compare_kind == "scalar":
            result = {"kind": "scalar", "data": 42.0}
    timing = _leg(mode, per_call, iterations_per_sample=iterations)
    raw = {
        "ok": True,
        "mode": mode,
        "batch_samples_s": timing.batch_samples_s,
        "result": result,
    }
    return timing, raw


def _measure_mismatch(mode: str, spec: Any, **_kwargs: Any) -> tuple[LegTiming, dict[str, Any]]:
    iterations = int(_kwargs.get("iterations", 10))
    per_call = [0.01, 0.01, 0.01]
    if mode == "fallback":
        result: dict[str, Any] = {"kind": "array", "data": [1.0, 2.0], "dtype": "float64"}
        if spec.compare_kind == "scalar":
            result = {"kind": "scalar", "data": 1.0}
    else:
        result = {"kind": "array", "data": [9.0, 9.0], "dtype": "float64"}
        if spec.compare_kind == "scalar":
            result = {"kind": "scalar", "data": 999.0}
    timing = _leg(mode, per_call, iterations_per_sample=iterations)
    return timing, {
        "ok": True,
        "mode": mode,
        "batch_samples_s": timing.batch_samples_s,
        "result": result,
    }


def _measure_native_loss(mode: str, spec: Any, **_kwargs: Any) -> tuple[LegTiming, dict[str, Any]]:
    """Native slower than fallback → speedup < 1 (honest loss)."""
    iterations = int(_kwargs.get("iterations", 10))
    if mode == "fallback":
        per_call = [0.010, 0.010, 0.010]
    else:
        per_call = [0.040, 0.041, 0.039]
    result: dict[str, Any]
    if spec.compare_kind == "scalar":
        result = {"kind": "scalar", "data": 7.0}
    else:
        result = {"kind": "array", "data": [0.0], "dtype": "float64"}
    timing = _leg(mode, per_call, iterations_per_sample=iterations)
    return timing, {
        "ok": True,
        "mode": mode,
        "batch_samples_s": timing.batch_samples_s,
        "result": result,
    }


def _fixed_meta(**_kwargs: Any) -> dict[str, Any]:
    return {
        "timestamp_utc": "2020-01-02T03:04:05Z",
        "platform": {
            "system": "TestOS",
            "release": "1",
            "version": "v",
            "machine": "x",
            "architecture": "64bit",
        },
        "python": {"implementation": "CPython", "version": "3.11.0", "executable": "/tmp/python"},
        "cpu": {"model_name": None, "source": "unavailable"},
        "packages": {"numpy": "1.0", "rextio": "0.1.1", "rextio-numpy": "0.1.0"},
        "package_distributions": {
            "numpy": "1.0",
            "rextio": "0.1.0",
            "rextio-numpy": "0.1.0",
        },
        "package_module_files": {
            "numpy": "/site-packages/numpy/__init__.py",
            "rextio": "/src/rextio/__init__.py",
            "rextio-numpy": "/src/rextio_numpy/__init__.py",
        },
        "package_version_mismatches": {
            "numpy": False,
            "rextio": True,
            "rextio-numpy": False,
        },
        "toolchain": {
            "cargo_path": None,
            "cargo_version": None,
            "rustc_path": None,
            "rustc_version": None,
        },
        "blas_thread_env": {"OMP_NUM_THREADS": None},
        "git": {"revision": None, "dirty": None, "status": "unavailable"},
    }


# ---------------------------------------------------------------------------
# scenarios registry
# ---------------------------------------------------------------------------


class TestScenarioRegistry:
    def test_exactly_four_scenarios(self) -> None:
        specs = registered_scenarios()
        assert len(specs) == 4
        ids = [s.id for s in specs]
        assert ids == [
            "small_elementwise",
            "multi_op_chain",
            "mixed_control_flow",
            "large_dot_blas_control",
        ]

    def test_unfused_and_blas_labels(self) -> None:
        chain = scenario_by_id("multi_op_chain")
        assert "unfused" in chain.labels
        assert any("UNFUSED" in n for n in chain.notes)
        dot = scenario_by_id("large_dot_blas_control")
        assert "blas-control" in dot.labels

    def test_kernels_source_uses_f64arr1_only(self) -> None:
        assert "F64Arr1" in KERNELS_SOURCE
        # No other plugin array types.
        assert "F32Arr" not in KERNELS_SOURCE
        assert "F64Arr2" not in KERNELS_SOURCE

    def test_kernels_source_has_no_function_docstrings(self) -> None:
        """Regression: function docstrings make rextio auto-native not-candidate.

        Rextio treats a leading string Expr as unsupported syntax in native
        bodies, so auto-discovery silently leaves the function on
        fallback-python with empty plugin_type_keys. Certification KERNELS
        have no function docstrings; the public fixture must match that shape.
        """
        tree = ast.parse(KERNELS_SOURCE)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        assert functions, "KERNELS_SOURCE must define benchmark kernels"
        for node in functions:
            assert ast.get_docstring(node) is None, (
                f"{node.name} has a function docstring (breaks rextio auto-native)"
            )

    def test_kernels_source_large_dot_is_bare_np_dot(self) -> None:
        """Regression: float(np.dot(...)) is RXT030; bare np.dot is claimable."""
        assert "float(np.dot" not in KERNELS_SOURCE
        tree = ast.parse(KERNELS_SOURCE)
        large_dot = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "large_dot"
        )
        returns = [n for n in large_dot.body if isinstance(n, ast.Return)]
        assert len(returns) == 1
        ret = returns[0].value
        assert isinstance(ret, ast.Call)
        # np.dot(...) — Attribute(Name('np'), 'dot')
        assert isinstance(ret.func, ast.Attribute)
        assert ret.func.attr == "dot"
        assert isinstance(ret.func.value, ast.Name)
        assert ret.func.value.id == "np"

    def test_kernels_source_omits_postponed_annotations_import(self) -> None:
        """Fixture matches certification KERNELS: no future annotations import.

        Postponed annotations alone are *not* the not-candidate cause (verified),
        but the public fixture stays aligned with the certified kernel shape.
        """
        assert "from __future__ import annotations" not in KERNELS_SOURCE


# ---------------------------------------------------------------------------
# fixture source route (real rextio check; no cargo)
# ---------------------------------------------------------------------------


class TestFixtureKernelRoutes:
    """Fail-closed regression: written KERNELS_SOURCE must be natively accepted."""

    def test_written_fixture_kernels_are_native_candidates(self, tmp_path: Path) -> None:
        rextio = shutil.which("rextio")
        if rextio is None:
            # Editable installs often expose the CLI only via the venv path.
            candidate = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "rextio"
            rextio = str(candidate) if candidate.is_file() else None
        if rextio is None:
            pytest.skip("rextio CLI not available")

        root = write_fixture_project(tmp_path / "fixture")
        proc = subprocess.run(
            [rextio, "check", "--format", "json", str(root)],
            capture_output=True,
            text=True,
            check=False,
        )
        check_path = root / ".rextio" / "reports" / "check.json"
        assert check_path.is_file(), (
            f"rextio check did not write check.json (exit={proc.returncode}): "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        report = json.loads(check_path.read_text(encoding="utf-8"))
        routes = extract_function_routes(report)
        expected = {
            f"{FIXTURE_QUAL_PREFIX}.small_elementwise",
            f"{FIXTURE_QUAL_PREFIX}.multi_op_chain",
            f"{FIXTURE_QUAL_PREFIX}.mixed_control_flow",
            f"{FIXTURE_QUAL_PREFIX}.large_dot",
        }
        missing = expected - set(routes)
        assert not missing, f"functions missing from check report: {sorted(missing)}"
        for qual in sorted(expected):
            info = routes[qual]
            assert is_natively_served(info), (
                f"{qual} not natively served: native_status={info.get('native_status')!r} "
                f"route={info.get('route')!r} (full={info!r})"
            )


# ---------------------------------------------------------------------------
# schema / suite status
# ---------------------------------------------------------------------------


class TestSuiteStatus:
    def test_all_ok(self) -> None:
        scenarios = [
            ScenarioResult(
                id="a",
                name="A",
                description="",
                status="ok",
                qualname="m.a",
                size={},
                speedup=1.5,
            )
        ]
        assert compute_suite_status(scenarios) == "ok"

    def test_partial(self) -> None:
        scenarios = [
            ScenarioResult(id="a", name="A", description="", status="ok", qualname="m.a", size={}),
            ScenarioResult(
                id="b",
                name="B",
                description="",
                status="failed",
                qualname="m.b",
                size={},
                reason="x",
            ),
        ]
        assert compute_suite_status(scenarios) == "partial"

    def test_all_failed_or_skipped(self) -> None:
        scenarios = [
            ScenarioResult(
                id="a",
                name="A",
                description="",
                status="skipped",
                qualname="m.a",
                size={},
                reason="no cargo",
            ),
            ScenarioResult(
                id="b",
                name="B",
                description="",
                status="failed",
                qualname="m.b",
                size={},
                reason="x",
            ),
        ]
        assert compute_suite_status(scenarios) == "failed"

    def test_empty_failed(self) -> None:
        assert compute_suite_status([]) == "failed"


# ---------------------------------------------------------------------------
# fixture / native route verification
# ---------------------------------------------------------------------------


class TestFixtureVerification:
    def test_write_fixture_project(self, tmp_path: Path) -> None:
        root = write_fixture_project(tmp_path / "fx")
        assert (root / "rextio.toml").is_file()
        assert "rextio-numpy" in (root / "rextio.toml").read_text(encoding="utf-8")
        assert (root / "src" / "np_bench" / "kernels.py").is_file()

    def test_preflight_missing_tools(self) -> None:
        problems = preflight_tools(which=lambda _name: None)
        assert any("cargo" in p for p in problems)
        assert any("rustc" in p for p in problems)

    def test_preflight_ok(self) -> None:
        assert preflight_tools(which=lambda name: f"/usr/bin/{name}") == []

    def test_is_natively_served(self) -> None:
        assert is_natively_served(
            {"native_status": "accepted", "route": "native-plugin:rextio-numpy"}
        )
        assert is_natively_served({"native_status": "accepted", "route": "native-direct"})
        assert not is_natively_served({"native_status": "accepted", "route": "fallback-python"})
        assert not is_natively_served({"native_status": "rejected", "route": "native-direct"})
        assert not is_natively_served(None)

    def test_verify_native_targets_missing_route(self) -> None:
        check = {
            "modules": [
                {
                    "functions": [
                        {
                            "qualname": "np_bench.kernels.small_elementwise",
                            "native_status": "accepted",
                            "route": "native-plugin:rextio-numpy",
                        }
                    ]
                }
            ]
        }
        routes, problems = verify_native_targets(check, registered_scenarios())
        assert "np_bench.kernels.small_elementwise" in routes
        assert problems  # other scenarios missing
        assert extract_function_routes(None) == {}


class TestBuildFixtureInjected:
    def test_skipped_when_tools_missing(self, tmp_path: Path) -> None:
        from benchmarks.fixture import build_fixture

        result = build_fixture(
            tmp_path,
            registered_scenarios(),
            which=lambda _n: None,
            build_fn=lambda _p: {"ok": True},
        )
        assert result.status == "skipped"
        assert result.reason is not None
        assert "cargo" in result.reason

    def test_failed_when_native_not_built(self, tmp_path: Path) -> None:
        from benchmarks.fixture import build_fixture

        def fake_build(project_root: Path) -> dict[str, Any]:
            reports = project_root / ".rextio" / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            (reports / "build.json").write_text(
                json.dumps({"native_build": {"status": "skipped"}}),
                encoding="utf-8",
            )
            return {"ok": False, "error": "native not built"}

        result = build_fixture(
            tmp_path,
            registered_scenarios(),
            which=lambda n: f"/bin/{n}",
            build_fn=fake_build,
            clock=lambda: 0.0,
        )
        assert result.status == "failed"
        assert result.reason is not None

    def test_failed_when_build_report_missing(self, tmp_path: Path) -> None:
        from benchmarks.fixture import build_fixture

        def fake_build(project_root: Path) -> dict[str, Any]:
            build_python = project_root / ".rextio" / "build" / "python"
            build_python.mkdir(parents=True, exist_ok=True)
            # Intentionally no reports/build.json
            return {
                "ok": True,
                "project_root": str(project_root),
                "build_python_dir": str(build_python),
            }

        result = build_fixture(
            tmp_path,
            registered_scenarios(),
            which=lambda n: f"/bin/{n}",
            build_fn=fake_build,
            clock=lambda: 1.0,
        )
        assert result.status == "failed"
        assert "build.json" in (result.reason or "")

    def test_failed_when_route_not_native(self, tmp_path: Path) -> None:
        from benchmarks.fixture import build_fixture

        def fake_build(project_root: Path) -> dict[str, Any]:
            reports = project_root / ".rextio" / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            (reports / "build.json").write_text(
                json.dumps({"native_build": {"status": "built"}}),
                encoding="utf-8",
            )
            (reports / "check.json").write_text(
                json.dumps(
                    {
                        "modules": [
                            {
                                "functions": [
                                    {
                                        "qualname": s.qualname,
                                        "native_status": "rejected",
                                        "route": "fallback-python",
                                    }
                                    for s in registered_scenarios()
                                ]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            build_python = project_root / ".rextio" / "build" / "python"
            build_python.mkdir(parents=True, exist_ok=True)
            return {
                "ok": True,
                "project_root": str(project_root),
                "build_python_dir": str(build_python),
            }

        result = build_fixture(
            tmp_path,
            registered_scenarios(),
            which=lambda n: f"/bin/{n}",
            build_fn=fake_build,
            clock=lambda: 1.0,
        )
        assert result.status == "failed"
        assert "not natively served" in (result.reason or "")


# ---------------------------------------------------------------------------
# subprocess construction
# ---------------------------------------------------------------------------


class TestSubprocessConstruction:
    def test_fallback_env(self) -> None:
        spec = scenario_by_id("small_elementwise")
        sub = build_leg_subprocess_spec(
            python_executable="/usr/bin/python3",
            build_python_dir="/tmp/build/python",
            spec=spec,
            mode="fallback",
            warmups=2,
            iterations=10,
            samples=3,
            base_env={"PATH": "/usr/bin", "REXTIO_DISABLE_BOUNDARY_FALLBACK": "1"},
        )
        assert isinstance(sub, SubprocessSpec)
        assert sub.argv[0] == "/usr/bin/python3"
        assert sub.argv[1] == "-c"
        assert sub.env["REXTIO_NATIVE_MODE"] == "fallback"
        assert "REXTIO_DISABLE_BOUNDARY_FALLBACK" not in sub.env
        assert sub.stdin_payload["mode"] == "fallback"
        assert sub.stdin_payload["disable_boundary_fallback"] is False
        assert sub.stdin_payload["module_name"] == "np_bench.kernels"
        assert sub.stdin_payload["function_name"] == "small_elementwise"
        assert sub.stdin_payload["build_python_dir"] == "/tmp/build/python"

    def test_native_env_disables_boundary_fallback(self) -> None:
        spec = scenario_by_id("large_dot_blas_control")
        sub = build_leg_subprocess_spec(
            python_executable="python",
            build_python_dir="/build",
            spec=spec,
            mode="native",
            warmups=1,
            iterations=5,
            samples=2,
            base_env={},
        )
        assert sub.env["REXTIO_NATIVE_MODE"] == "native"
        assert sub.env["REXTIO_DISABLE_BOUNDARY_FALLBACK"] == "1"
        assert sub.stdin_payload["disable_boundary_fallback"] is True
        assert sub.stdin_payload["scenario_id"] == "large_dot_blas_control"


# ---------------------------------------------------------------------------
# verification / reconstruct
# ---------------------------------------------------------------------------


class TestVerification:
    def test_matched_array(self) -> None:
        fb = {
            "ok": True,
            "result": {"kind": "array", "data": [1.0, float("nan")], "dtype": "float64"},
        }
        nt = {
            "ok": True,
            "result": {"kind": "array", "data": [1.0, float("nan")], "dtype": "float64"},
        }
        v = verify_leg_results(fb, nt, compare_kind="array")
        assert v["matched"] is True

    def test_mismatched_scalar_no_speedup_path(self) -> None:
        fb = {"ok": True, "result": {"kind": "scalar", "data": 1.0}}
        nt = {"ok": True, "result": {"kind": "scalar", "data": 2.0}}
        v = verify_leg_results(fb, nt, compare_kind="scalar")
        assert v["matched"] is False

    def test_reconstruct_scalar(self) -> None:
        assert reconstruct_result({"kind": "scalar", "data": 3.5}) == 3.5

    def test_reconstruct_preserves_dtype(self) -> None:
        np = pytest.importorskip("numpy")
        for name, expected in (
            ("float64", np.float64),
            ("float32", np.float32),
            ("int64", np.int64),
        ):
            arr = reconstruct_result({"kind": "array", "data": [1, 2], "dtype": name})
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.dtype(expected)

    def test_reconstruct_rejects_missing_or_unsupported_dtype(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            reconstruct_result({"kind": "array", "data": [1.0]})
        with pytest.raises(ValueError, match="unsupported result dtype"):
            reconstruct_result({"kind": "array", "data": [1.0], "dtype": "complex128"})
        with pytest.raises(ValueError, match="unsupported result dtype"):
            parse_result_dtype("object")

    def test_dtype_mismatch_not_matched(self) -> None:
        """Wrong native dtype must not earn matched=true (even if values cast equal)."""
        fb = {
            "ok": True,
            "result": {"kind": "array", "data": [1.0, 2.0], "dtype": "float64"},
        }
        nt = {
            "ok": True,
            "result": {"kind": "array", "data": [1.0, 2.0], "dtype": "float32"},
        }
        v = verify_leg_results(fb, nt, compare_kind="array")
        assert v["matched"] is False
        assert "diverged" in (v.get("reason") or "")


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


class _FakeModule:
    """Minimal module stand-in for provenance tests (no real import)."""

    def __init__(self, version: str | None, file: str | None) -> None:
        if version is not None:
            self.__version__ = version
        if file is not None:
            self.__file__ = file


class TestMetadata:
    def test_clock_injection_and_nulls(self) -> None:
        fixed = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        meta = collect_metadata(repo_root=None, clock=lambda: fixed)
        assert meta["timestamp_utc"] == "2020-01-02T03:04:05Z"
        assert meta["git"]["status"] == "unavailable"
        assert meta["git"]["revision"] is None
        # packages may or may not be installed — keys must exist
        assert "numpy" in meta["packages"]
        assert "rextio" in meta["packages"]
        assert "rextio-numpy" in meta["packages"]
        assert set(meta["packages"]) == set(meta["package_distributions"])
        assert set(meta["packages"]) == set(meta["package_module_files"])
        assert set(meta["packages"]) == set(meta["package_version_mismatches"])
        assert "blas_thread_env" in meta
        assert "OMP_NUM_THREADS" in meta["blas_thread_env"]

    def test_provenance_match(self) -> None:
        mod = _FakeModule("1.2.3", "/site-packages/pkg/__init__.py")
        info = resolve_package_provenance(
            import_name="pkg",
            distribution_name="pkg-dist",
            import_module=lambda _n: mod,
            distribution_version=lambda _n: "1.2.3",
        )
        assert info["runtime_version"] == "1.2.3"
        assert info["distribution_version"] == "1.2.3"
        assert info["module_file"] == "/site-packages/pkg/__init__.py"
        assert info["mismatch"] is False

    def test_provenance_mismatch_source_checkout(self) -> None:
        # Mirrors the confirmed defect: dist metadata lags sibling source.
        mod = _FakeModule(
            "0.1.1",
            "/Volumes/Data/workspace/rextio/rextio/src/rextio/__init__.py",
        )
        info = resolve_package_provenance(
            import_name="rextio",
            distribution_name="rextio",
            import_module=lambda _n: mod,
            distribution_version=lambda _n: "0.1.0",
        )
        assert info["runtime_version"] == "0.1.1"
        assert info["distribution_version"] == "0.1.0"
        assert info["module_file"] is not None
        assert info["module_file"].endswith("rextio/__init__.py")
        assert info["mismatch"] is True

    def test_provenance_missing_module(self) -> None:
        def _raise(_name: str) -> Any:
            raise ModuleNotFoundError(_name)

        info = resolve_package_provenance(
            import_name="missing_pkg",
            distribution_name="missing-pkg",
            import_module=_raise,
            distribution_version=lambda _n: "9.9.9",
        )
        assert info["runtime_version"] is None
        assert info["distribution_version"] == "9.9.9"
        assert info["module_file"] is None
        assert info["mismatch"] is None

    def test_provenance_missing_distribution(self) -> None:
        mod = _FakeModule("0.2.0", "/src/pkg/__init__.py")
        info = resolve_package_provenance(
            import_name="pkg",
            distribution_name="pkg",
            import_module=lambda _n: mod,
            distribution_version=lambda _n: None,
        )
        assert info["runtime_version"] == "0.2.0"
        assert info["distribution_version"] is None
        assert info["module_file"] == "/src/pkg/__init__.py"
        assert info["mismatch"] is None

    def test_provenance_missing_both(self) -> None:
        def _raise(_name: str) -> Any:
            raise ModuleNotFoundError(_name)

        info = resolve_package_provenance(
            import_name="gone",
            distribution_name="gone",
            import_module=_raise,
            distribution_version=lambda _n: None,
        )
        assert info["runtime_version"] is None
        assert info["distribution_version"] is None
        assert info["module_file"] is None
        assert info["mismatch"] is None

    def test_provenance_missing_version_attr(self) -> None:
        # Module imports but has no __version__ — do not invent one.
        class _NoVersion:
            __file__ = "/src/pkg/__init__.py"

        info = resolve_package_provenance(
            import_name="pkg",
            distribution_name="pkg",
            import_module=lambda _n: _NoVersion(),  # type: ignore[return-value]
            distribution_version=lambda _n: "1.0.0",
        )
        assert info["runtime_version"] is None
        assert info["distribution_version"] == "1.0.0"
        assert info["module_file"] == "/src/pkg/__init__.py"
        assert info["mismatch"] is None

    def test_collect_package_provenance_maps(self) -> None:
        modules = {
            "numpy": _FakeModule("2.4.6", "/sp/numpy/__init__.py"),
            "rextio": _FakeModule("0.1.1", "/src/rextio/__init__.py"),
            "rextio_numpy": _FakeModule("0.1.0", "/src/rextio_numpy/__init__.py"),
        }
        dists = {
            "numpy": "2.4.6",
            "rextio": "0.1.0",
            "rextio-numpy": "0.1.0",
        }
        block = collect_package_provenance(
            import_module=lambda name: modules[name],
            distribution_version=lambda name: dists[name],
        )
        assert block["packages"]["numpy"] == "2.4.6"
        assert block["packages"]["rextio"] == "0.1.1"
        assert block["packages"]["rextio-numpy"] == "0.1.0"
        assert block["package_distributions"]["rextio"] == "0.1.0"
        assert block["package_version_mismatches"]["numpy"] is False
        assert block["package_version_mismatches"]["rextio"] is True
        assert block["package_version_mismatches"]["rextio-numpy"] is False
        assert block["package_module_files"]["rextio"].endswith("rextio/__init__.py")

    def test_collect_metadata_uses_injected_probes(self) -> None:
        fixed = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        modules = {
            "numpy": _FakeModule("1.0", "/n.py"),
            "rextio": _FakeModule("0.1.1", "/r.py"),
            "rextio_numpy": _FakeModule("0.1.0", "/rn.py"),
        }
        dists = {"numpy": "1.0", "rextio": "0.1.0", "rextio-numpy": "0.1.0"}
        meta = collect_metadata(
            repo_root=None,
            clock=lambda: fixed,
            import_module=lambda name: modules[name],
            distribution_version=lambda name: dists[name],
        )
        assert meta["packages"]["rextio"] == "0.1.1"
        assert meta["package_distributions"]["rextio"] == "0.1.0"
        assert meta["package_version_mismatches"]["rextio"] is True
        assert meta["package_module_files"]["rextio"] == "/r.py"


# ---------------------------------------------------------------------------
# JSON / Markdown reports
# ---------------------------------------------------------------------------


class TestSampleNormalization:
    def test_iterations_one_identity(self) -> None:
        batch = [0.05, 0.06, 0.04]
        assert normalize_batch_to_per_call(batch, 1) == batch

    def test_iterations_gt_one_divides(self) -> None:
        batch = [0.20, 0.40, 0.10]
        per_call = normalize_batch_to_per_call(batch, 10)
        assert per_call == pytest.approx([0.02, 0.04, 0.01])

    def test_iterations_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="iterations_per_sample"):
            normalize_batch_to_per_call([1.0], 0)

    def test_measure_leg_preserves_batch_and_derives_per_call(self) -> None:
        """Injected worker returns raw batch times; parent normalizes."""
        iterations = 5
        batch = [0.10, 0.15, 0.20]

        def fake_run(_spec: SubprocessSpec) -> dict[str, Any]:
            return {
                "ok": True,
                "mode": "fallback",
                "batch_samples_s": batch,
                "result": {"kind": "scalar", "data": 1.0},
            }

        timing, raw = measure_leg(
            python_executable="python",
            build_python_dir="/tmp/build/python",
            spec=scenario_by_id("large_dot_blas_control"),
            mode="fallback",
            warmups=0,
            iterations=iterations,
            samples=3,
            run_subprocess=fake_run,
        )
        assert timing is not None
        assert timing.batch_samples_s == pytest.approx(batch)
        assert timing.per_call_samples_s == pytest.approx([0.02, 0.03, 0.04])
        assert timing.summary.median == pytest.approx(0.03)
        assert timing.iterations_per_sample == iterations
        assert raw["batch_samples_s"] == batch


class TestModulePathValidation:
    def test_missing_file_fails_closed(self) -> None:
        err = validate_module_under_build_tree(None, "/tmp/build/python")
        assert err is not None
        assert "no __file__" in err
        assert validate_module_under_build_tree("", "/tmp/build/python") is not None

    def test_under_tree_ok(self, tmp_path: Path) -> None:
        build = tmp_path / "build" / "python"
        mod = build / "pkg" / "mod.py"
        mod.parent.mkdir(parents=True)
        mod.write_text("x = 1\n", encoding="utf-8")
        assert validate_module_under_build_tree(str(mod), build) is None
        # Relative path that resolves under build also ok.
        assert validate_module_under_build_tree(str(mod.resolve()), str(build)) is None

    def test_outside_tree_rejected(self, tmp_path: Path) -> None:
        build = tmp_path / "build" / "python"
        build.mkdir(parents=True)
        outside = tmp_path / "elsewhere" / "mod.py"
        outside.parent.mkdir(parents=True)
        outside.write_text("x = 1\n", encoding="utf-8")
        err = validate_module_under_build_tree(str(outside), build)
        assert err is not None
        assert "outside build_python_dir" in err

    def test_worker_rejects_out_of_tree_module(self, tmp_path: Path) -> None:
        """Real worker subprocess: module on PYTHONPATH but outside build tree."""
        import os
        import subprocess
        import sys

        from benchmarks.measure import _WORKER_SOURCE

        build = tmp_path / "build" / "python"
        build.mkdir(parents=True)
        outside = tmp_path / "site" / "out_of_tree_bench_mod.py"
        outside.parent.mkdir(parents=True)
        outside.write_text(
            "def small_elementwise(a, b):\n    return a\n",
            encoding="utf-8",
        )

        env = dict(os.environ)
        env["PYTHONPATH"] = str(outside.parent) + os.pathsep + env.get("PYTHONPATH", "")
        # Minimal size so input construction works if validation were skipped.
        payload = {
            "mode": "fallback",
            "build_python_dir": str(build),
            "module_name": "out_of_tree_bench_mod",
            "function_name": "small_elementwise",
            "scenario_id": "small_elementwise",
            "size": {"n": 4},
            "seed": 0,
            "warmups": 0,
            "iterations": 1,
            "samples": 1,
            "disable_boundary_fallback": False,
        }
        completed = subprocess.run(
            [sys.executable, "-c", _WORKER_SOURCE],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert completed.returncode != 0
        data = json.loads(completed.stdout)
        assert data["ok"] is False
        assert "outside build_python_dir" in data["error"]
        # Must not have produced timings for the wrong module.
        assert "batch_samples_s" not in data
        assert "samples_s" not in data

    def test_worker_accepts_in_tree_module(self, tmp_path: Path) -> None:
        """Worker times a module that lives under build_python_dir."""
        import os
        import subprocess
        import sys

        from benchmarks.measure import _WORKER_SOURCE

        build = tmp_path / "build" / "python"
        build.mkdir(parents=True)
        mod = build / "in_tree_bench_mod.py"
        mod.write_text(
            "def small_elementwise(a, b):\n    return a + b\n",
            encoding="utf-8",
        )
        payload = {
            "mode": "fallback",
            "build_python_dir": str(build),
            "module_name": "in_tree_bench_mod",
            "function_name": "small_elementwise",
            "scenario_id": "small_elementwise",
            "size": {"n": 4},
            "seed": 0,
            "warmups": 0,
            "iterations": 2,
            "samples": 2,
            "disable_boundary_fallback": False,
        }
        completed = subprocess.run(
            [sys.executable, "-c", _WORKER_SOURCE],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        data = json.loads(completed.stdout)
        assert data["ok"] is True
        assert len(data["batch_samples_s"]) == 2
        assert data["iterations"] == 2
        # Path must resolve under build tree (same check as production validator).
        assert validate_module_under_build_tree(data["loaded_from"], build) is None

    def test_worker_pins_installed_rextio_before_build_path(self) -> None:
        """Regression: partial build-tree rextio must not shadow rextio.config.

        Generated trees ship only ``rextio.runtime``. Inserting that path first
        without pre-importing the full install makes ``rextio_numpy`` fail on
        ``rextio.config`` when fallback kernels import annotation aliases.
        """
        from benchmarks.measure import _WORKER_SOURCE

        pin = _WORKER_SOURCE.index("import rextio")
        path_insert = _WORKER_SOURCE.index("sys.path.insert(0, build_python_dir)")
        assert pin < path_insert
        assert "import rextio.runtime" in _WORKER_SOURCE

    def test_worker_serializes_numpy_scalar_as_scalar_kind(self, tmp_path: Path) -> None:
        """Regression: bare np.dot returns numpy.float64 (has tolist + ndim 0).

        Must not be JSON-encoded as kind=array, or scalar compare_kind
        verification falsely reports fallback/native divergence.
        """
        import os
        import subprocess
        import sys

        from benchmarks.measure import _WORKER_SOURCE

        build = tmp_path / "build" / "python"
        build.mkdir(parents=True)
        (build / "dot_mod.py").write_text(
            "import numpy as np\ndef large_dot(a, b):\n    return np.dot(a, b)\n",
            encoding="utf-8",
        )
        payload = {
            "mode": "fallback",
            "build_python_dir": str(build),
            "module_name": "dot_mod",
            "function_name": "large_dot",
            "scenario_id": "large_dot_blas_control",
            "size": {"n": 8},
            "seed": 0,
            "warmups": 0,
            "iterations": 1,
            "samples": 1,
            "disable_boundary_fallback": False,
        }
        completed = subprocess.run(
            [sys.executable, "-c", _WORKER_SOURCE],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        data = json.loads(completed.stdout)
        assert data["ok"] is True
        assert data["result"]["kind"] == "scalar"
        assert isinstance(data["result"]["data"], float)


class TestReports:
    def test_json_roundtrip(self, tmp_path: Path) -> None:
        iterations = 10
        fb = _leg("fallback", [0.02, 0.02], iterations_per_sample=iterations)
        nt = _leg("native", [0.01, 0.01], iterations_per_sample=iterations)
        report = SuiteReport(
            schema_version=REPORT_SCHEMA_VERSION,
            suite_status="ok",
            build_wall_s=2.5,
            metadata=_fixed_meta(),
            settings={"samples": 5, "iterations": iterations},
            scenarios=[
                ScenarioResult(
                    id="small_elementwise",
                    name="Small",
                    description="d",
                    status="ok",
                    qualname="np_bench.kernels.small_elementwise",
                    size={"n": 64},
                    labels=["elementwise"],
                    fallback=fb,
                    native=nt,
                    speedup=2.0,
                    verification={"matched": True},
                    optional_metrics={
                        "dispatch_crossings": {"status": "unavailable", "reason": "x"}
                    },
                )
            ],
            honesty=default_honesty(),
        )
        path = write_json_report(report, tmp_path / "r.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == REPORT_SCHEMA_VERSION
        assert data["schema_version"] == "2.1.0"
        leg = data["scenarios"][0]["fallback"]
        assert "batch_samples_s" in leg
        assert "per_call_samples_s" in leg
        assert "samples_s" not in leg
        assert leg["batch_samples_s"] == pytest.approx([0.2, 0.2])
        assert leg["per_call_samples_s"] == pytest.approx([0.02, 0.02])
        assert leg["iterations_per_sample"] == iterations
        assert data["scenarios"][0]["speedup"] == 2.0
        restored = report_from_dict(data)
        assert restored.suite_status == "ok"
        assert restored.scenarios[0].speedup == 2.0
        assert restored.scenarios[0].fallback is not None
        assert restored.scenarios[0].fallback.summary.median == pytest.approx(0.02)
        assert restored.scenarios[0].fallback.batch_samples_s == pytest.approx([0.2, 0.2])
        assert restored.scenarios[0].fallback.per_call_samples_s == pytest.approx([0.02, 0.02])

    def test_markdown_distinguishes_batch_and_per_call(self) -> None:
        report = SuiteReport(
            schema_version=REPORT_SCHEMA_VERSION,
            suite_status="ok",
            build_wall_s=1.0,
            metadata=_fixed_meta(),
            settings={},
            scenarios=[
                ScenarioResult(
                    id="small_elementwise",
                    name="Small",
                    description="d",
                    status="ok",
                    qualname="np_bench.kernels.small_elementwise",
                    size={"n": 64},
                    fallback=_leg(
                        "fallback",
                        [0.02, 0.02],
                        iterations_per_sample=5,
                        batch_samples=[0.10, 0.10],
                    ),
                    native=_leg(
                        "native",
                        [0.01, 0.01],
                        iterations_per_sample=5,
                        batch_samples=[0.05, 0.05],
                    ),
                    speedup=2.0,
                    verification={"matched": True},
                )
            ],
            honesty=default_honesty(),
        )
        md = render_markdown(report)
        assert "per-call wall" in md
        assert "per-call median" in md
        assert "raw batch samples" in md
        assert "per-call wall samples" in md
        assert "fallback per-call stats" in md
        # Raw batch values appear distinctly from per-call.
        assert "0.1" in md or "0.10" in md
        assert "speedup = fallback_per_call_median" in md

    def test_markdown_negative_speedup_rendered(self) -> None:
        report = SuiteReport(
            schema_version=REPORT_SCHEMA_VERSION,
            suite_status="ok",
            build_wall_s=1.0,
            metadata=_fixed_meta(),
            settings={},
            scenarios=[
                ScenarioResult(
                    id="large_dot_blas_control",
                    name="Large 1-D numpy.dot (BLAS control)",
                    description="blas",
                    status="ok",
                    qualname="np_bench.kernels.large_dot",
                    size={"n": 1_000_000},
                    labels=["dot", "blas-control", "large"],
                    notes=["BLAS-dominated control on NumPy."],
                    fallback=_leg("fallback", [0.01, 0.01]),
                    native=_leg("native", [0.04, 0.04]),
                    speedup=0.25,
                    verification={"matched": True},
                ),
                ScenarioResult(
                    id="multi_op_chain",
                    name="Multi-op elementwise chain (CURRENTLY UNFUSED)",
                    description="chain",
                    status="ok",
                    qualname="np_bench.kernels.multi_op_chain",
                    size={"n": 4096},
                    labels=["elementwise", "chain", "unfused"],
                    notes=["CURRENTLY UNFUSED — no fusion claim is made or asserted."],
                    fallback=_leg("fallback", [0.02, 0.02]),
                    native=_leg("native", [0.01, 0.01]),
                    speedup=2.0,
                ),
                ScenarioResult(
                    id="missing_tool",
                    name="Skipped example",
                    description="",
                    status="skipped",
                    qualname="x",
                    size={},
                    reason="cargo not found on PATH",
                ),
                ScenarioResult(
                    id="bad",
                    name="Failed example",
                    description="",
                    status="failed",
                    qualname="y",
                    size={},
                    reason="fallback and native results diverged",
                    verification={
                        "matched": False,
                        "reason": "fallback and native results diverged",
                    },
                ),
            ],
            honesty=default_honesty(),
        )
        md = render_markdown(report)
        assert "> 1" in md or ">1" in md.replace(" ", "")
        assert "< 1" in md or "<1" in md.replace(" ", "")
        assert "0.2500x" in md
        assert "native was **slower**" in md
        assert "CURRENTLY UNFUSED" in md
        assert "BLAS" in md
        assert "Skipped" in md
        assert "cargo not found" in md
        assert "Failed" in md
        assert "results diverged" in md
        # negative result not suppressed
        assert "0.2500x" in md


# ---------------------------------------------------------------------------
# full runner with injected hooks
# ---------------------------------------------------------------------------


class TestRunnerInjected:
    def test_happy_path_all_ok(self, tmp_path: Path) -> None:
        def build(root: Path, scenarios: list[Any], **_k: Any) -> FixtureBuildResult:
            return _ok_fixture(root)

        report = run_suite(
            RunnerConfig(
                output_dir=tmp_path / "out",
                samples=5,
                iterations=10,
                warmups=1,
                repo_root=None,
            ),
            hooks=RunnerHooks(
                build_fixture=build,
                measure_leg=_measure_ok,
                collect_metadata=_fixed_meta,
            ),
        )
        assert report.suite_status == "ok"
        assert report.schema_version == REPORT_SCHEMA_VERSION
        assert report.build_wall_s == pytest.approx(1.25)
        assert len(report.scenarios) == 4
        for s in report.scenarios:
            assert s.status == "ok"
            assert s.speedup is not None
            assert s.speedup > 1.0
            assert s.optional_metrics["dispatch_crossings"]["status"] == "unavailable"
            assert s.optional_metrics["temporary_allocations"]["status"] == "unavailable"
        json_path, md_path = write_reports(report, tmp_path / "out")
        assert json_path.is_file()
        assert md_path.is_file()

    def test_native_loss_speedup_below_one(self, tmp_path: Path) -> None:
        def build(root: Path, scenarios: list[Any], **_k: Any) -> FixtureBuildResult:
            return _ok_fixture(root)

        report = run_suite(
            RunnerConfig(output_dir=tmp_path / "out", scenario_ids=["large_dot_blas_control"]),
            hooks=RunnerHooks(
                build_fixture=build,
                measure_leg=_measure_native_loss,
                collect_metadata=_fixed_meta,
            ),
        )
        assert report.suite_status == "ok"
        s = report.scenarios[0]
        assert s.speedup is not None
        assert s.speedup < 1.0
        md = render_markdown(report)
        assert "slower" in md.lower() or s.speedup < 1.0

    def test_mismatch_fails_without_speedup(self, tmp_path: Path) -> None:
        def build(root: Path, scenarios: list[Any], **_k: Any) -> FixtureBuildResult:
            return _ok_fixture(root)

        report = run_suite(
            RunnerConfig(output_dir=tmp_path / "out", scenario_ids=["small_elementwise"]),
            hooks=RunnerHooks(
                build_fixture=build,
                measure_leg=_measure_mismatch,
                collect_metadata=_fixed_meta,
            ),
        )
        assert report.suite_status == "failed"
        s = report.scenarios[0]
        assert s.status == "failed"
        assert s.speedup is None
        assert s.verification is not None
        assert s.verification.get("matched") is False

    def test_missing_tools_skips_all(self, tmp_path: Path) -> None:
        def build(_root: Path, scenarios: list[Any], **_k: Any) -> FixtureBuildResult:
            return FixtureBuildResult(
                project_root=None,
                build_python_dir=None,
                build_wall_s=None,
                status="skipped",
                reason="cargo not found on PATH; rustc not found on PATH",
            )

        report = run_suite(
            RunnerConfig(output_dir=tmp_path / "out"),
            hooks=RunnerHooks(
                build_fixture=build,
                measure_leg=_measure_ok,
                collect_metadata=_fixed_meta,
            ),
        )
        assert report.suite_status == "failed"
        assert all(s.status == "skipped" for s in report.scenarios)
        assert all("cargo" in (s.reason or "") for s in report.scenarios)

    def test_native_artifact_failure(self, tmp_path: Path) -> None:
        def build(_root: Path, scenarios: list[Any], **_k: Any) -> FixtureBuildResult:
            return FixtureBuildResult(
                project_root=tmp_path,
                build_python_dir=None,
                build_wall_s=0.5,
                status="failed",
                reason="native artifact not built (native_build.status='skipped')",
            )

        report = run_suite(
            RunnerConfig(output_dir=tmp_path / "out"),
            hooks=RunnerHooks(
                build_fixture=build,
                measure_leg=_measure_ok,
                collect_metadata=_fixed_meta,
            ),
        )
        assert report.suite_status == "failed"
        assert all(s.status == "failed" for s in report.scenarios)
        assert all("native artifact" in (s.reason or "") for s in report.scenarios)

    def test_measure_leg_failure(self, tmp_path: Path) -> None:
        def build(root: Path, scenarios: list[Any], **_k: Any) -> FixtureBuildResult:
            return _ok_fixture(root)

        def measure_fail(mode: str, **_k: Any) -> tuple[None, dict[str, Any]]:
            return None, {"ok": False, "error": f"{mode} boom"}

        report = run_suite(
            RunnerConfig(output_dir=tmp_path / "out", scenario_ids=["small_elementwise"]),
            hooks=RunnerHooks(
                build_fixture=build,
                measure_leg=measure_fail,
                collect_metadata=_fixed_meta,
            ),
        )
        assert report.scenarios[0].status == "failed"
        assert "fallback" in (report.scenarios[0].reason or "")

    def test_keep_fixture_preserves_directory(self, tmp_path: Path) -> None:
        """--keep-fixture must leave the fixture on disk after run_suite returns."""
        seen: dict[str, Path] = {}

        def build(root: Path, scenarios: list[Any], **_k: Any) -> FixtureBuildResult:
            seen["root"] = root
            marker = root / "KEEP_ME"
            marker.write_text("alive", encoding="utf-8")
            return _ok_fixture(root)

        report = run_suite(
            RunnerConfig(
                output_dir=tmp_path / "out",
                scenario_ids=["small_elementwise"],
                keep_fixture=True,
            ),
            hooks=RunnerHooks(
                build_fixture=build,
                measure_leg=_measure_ok,
                collect_metadata=_fixed_meta,
            ),
        )
        assert report.suite_status == "ok"
        fixture_root = seen["root"]
        assert report.settings.get("fixture_dir") == str(fixture_root)
        assert report.settings.get("keep_fixture") is True
        assert fixture_root.is_dir()
        assert (fixture_root / "KEEP_ME").read_text(encoding="utf-8") == "alive"
        # Clean up the intentionally preserved temp dir after the assertion.
        shutil.rmtree(fixture_root, ignore_errors=True)

    def test_default_temp_fixture_is_cleaned(self, tmp_path: Path) -> None:
        """Without keep_fixture, TemporaryDirectory cleanup must remove the path."""
        seen: dict[str, Path] = {}

        def build(root: Path, scenarios: list[Any], **_k: Any) -> FixtureBuildResult:
            seen["root"] = root
            (root / "TEMP").write_text("x", encoding="utf-8")
            return _ok_fixture(root)

        report = run_suite(
            RunnerConfig(
                output_dir=tmp_path / "out",
                scenario_ids=["small_elementwise"],
                keep_fixture=False,
            ),
            hooks=RunnerHooks(
                build_fixture=build,
                measure_leg=_measure_ok,
                collect_metadata=_fixed_meta,
            ),
        )
        assert report.suite_status == "ok"
        assert report.settings.get("keep_fixture") is False
        # Path was cleaned (may still exist briefly on some platforms only if
        # cleanup failed — TemporaryDirectory.cleanup should remove it).
        assert not seen["root"].exists()

    def test_explicit_fixture_dir_is_preserved(self, tmp_path: Path) -> None:
        fixture_dir = tmp_path / "explicit-fixture"

        def build(root: Path, scenarios: list[Any], **_k: Any) -> FixtureBuildResult:
            assert root == fixture_dir
            (root / "USER").write_text("owned", encoding="utf-8")
            return _ok_fixture(root)

        report = run_suite(
            RunnerConfig(
                output_dir=tmp_path / "out",
                scenario_ids=["small_elementwise"],
                fixture_dir=fixture_dir,
                keep_fixture=False,  # explicit dir still preserved
            ),
            hooks=RunnerHooks(
                build_fixture=build,
                measure_leg=_measure_ok,
                collect_metadata=_fixed_meta,
            ),
        )
        assert report.suite_status == "ok"
        assert report.settings.get("fixture_dir") == str(fixture_dir)
        assert report.settings.get("keep_fixture") is True
        assert (fixture_dir / "USER").read_text(encoding="utf-8") == "owned"


# ---------------------------------------------------------------------------
# CLI smoke (no cargo)
# ---------------------------------------------------------------------------


class TestCli:
    def test_list_exit_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        from benchmarks.cli import main

        code = main(["--list"])
        assert code == 0
        out = capsys.readouterr().out
        assert "small_elementwise" in out
        assert "large_dot_blas_control" in out
        assert "multi_op_chain" in out

    def test_requires_output_dir(self) -> None:
        from benchmarks.cli import main

        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 2
