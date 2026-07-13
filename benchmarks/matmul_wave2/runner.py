"""CLI runner for the Wave 2 matmul research harness.

Orchestrates: provenance → candidate build (once) → per-cell inputs → four
isolated timing subprocesses → bootstrap → aggregates → always-NO-GO verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from benchmarks.matmul_wave2 import SCHEMA_VERSION
from benchmarks.matmul_wave2.candidate import (
    CandidateArtifact,
    build_candidate,
    ensure_importable,
    python_executable,
)
from benchmarks.matmul_wave2.protocol import (
    ATOL,
    BASE_SEED,
    BLAS_VENDOR_TOKENS,
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_RESAMPLES,
    DISPATCHABILITY_REASON,
    DISPATCHABILITY_STATUS,
    EQUAL_NAN,
    EVIDENCE_SAMPLES,
    EVIDENCE_WARMUPS,
    EXPECTED_CELL_IDS,
    HARNESS_SOURCE_FILES,
    LEG_ORDER,
    PRODUCT_VERDICT,
    PRODUCT_VERDICT_DETAIL,
    PROTOCOL_ID,
    RTOL,
    SHAPE_CELLS,
    SPELLINGS,
    THREAD_ENV,
    ShapeCell,
    bootstrap_seed,
    cell_input_seed,
    is_exact_frozen_matrix,
    protocol_manifest,
    report_cell_ids_match_frozen,
    require_exact_frozen_matrix,
    run_counts,
)
from benchmarks.matmul_wave2.report import (
    apply_evidence_integrity,
    validate_report_schema,
    write_json_report,
    write_markdown_report,
)
from benchmarks.matmul_wave2.stats import (
    bootstrap_median_ratio,
    cell_conservative_aggregate,
    performance_gate,
    summarize_per_call,
    validate_positive_finite_samples,
)

RunWorkerFn = Callable[[dict[str, Any]], dict[str, Any]]
BuildCandidateFn = Callable[..., CandidateArtifact]


@dataclass
class RunSettings:
    """Resolved run settings."""

    evidence: bool
    warmups: int
    samples: int
    base_seed: int
    bootstrap_resamples: int
    output_dir: str
    run_id: str
    cells: tuple[ShapeCell, ...]
    calibrate: bool
    min_batch_s: float
    max_iterations: int
    performance_conclusion_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON settings block."""
        return {
            "evidence": self.evidence,
            "warmups": self.warmups,
            "samples": self.samples,
            "base_seed": self.base_seed,
            "bootstrap_resamples": self.bootstrap_resamples,
            "output_dir": self.output_dir,
            "run_id": self.run_id,
            "n_cells": len(self.cells),
            "cell_ids": [c.cell_id for c in self.cells],
            "calibrate": self.calibrate,
            "min_batch_s": self.min_batch_s,
            "max_iterations": self.max_iterations,
            "performance_conclusion_allowed": self.performance_conclusion_allowed,
            "leg_order": list(LEG_ORDER),
            "thread_env": dict(THREAD_ENV),
            "smoke_subset": len(self.cells) != len(SHAPE_CELLS)
            or not is_exact_frozen_matrix(self.cells),
            "run_kind": (
                "evidence"
                if self.evidence
                else (
                    "smoke_subset"
                    if (
                        len(self.cells) != len(SHAPE_CELLS)
                        or not is_exact_frozen_matrix(self.cells)
                    )
                    else "smoke"
                )
            ),
        }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_text(argv: list[str], *, timeout: float = 10.0) -> str | None:
    """Run a command; return stripped stdout, or None on failure/empty."""
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    text = (completed.stdout or "").strip()
    return text or None


def _run_stdout_preserve_empty(argv: list[str], *, timeout: float = 10.0) -> str | None:
    """Run a command; return stdout (possibly empty) on success, else None.

    Unlike :func:`_run_text`, a successful empty stdout (e.g. clean
    ``git status --porcelain``) is preserved as an empty string rather than
    becoming ``None`` (which would be misread as command failure).
    """
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout if completed.stdout is not None else ""


def _git_info(repo_root: Path) -> dict[str, Any]:
    """Local git provenance without remote URLs."""
    if shutil.which("git") is None:
        return {"revision": None, "dirty": None, "status": "unavailable"}
    rev = _run_text(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    if rev is None:
        return {"revision": None, "dirty": None, "status": "unavailable"}
    # Clean repos produce empty porcelain stdout; must not treat as unavailable.
    porcelain = _run_stdout_preserve_empty(["git", "-C", str(repo_root), "status", "--porcelain"])
    if porcelain is None:
        return {"revision": rev, "dirty": None, "status": "unavailable"}
    return {"revision": rev, "dirty": bool(porcelain.strip()), "status": "ok"}


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform_processor": platform.processor() or None,
        "platform_machine": platform.machine() or None,
        "logical_cores": os.cpu_count(),
        "model_name": None,
        "source": "unavailable",
    }
    brand = _run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
    if brand:
        info["model_name"] = brand
        info["source"] = "sysctl machdep.cpu.brand_string"
        return info
    # Apple Silicon / some macOS configs omit brand_string; hw.model still works.
    hw_model = _run_text(["sysctl", "-n", "hw.model"])
    if hw_model:
        info["model_name"] = hw_model
        info["source"] = "sysctl hw.model"
        return info
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            text = cpuinfo.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for line in text.splitlines():
            if line.lower().startswith("model name") and ":" in line:
                info["model_name"] = line.split(":", 1)[1].strip()
                info["source"] = "/proc/cpuinfo"
                break
    return info


def _power_thermal_status() -> dict[str, Any]:
    """Best-effort OS power/thermal observability (macOS/Linux)."""
    status: dict[str, Any] = {
        "power_source": None,
        "thermal_pressure": None,
        "thermal_anomaly": False,
        "notes": [],
        "observable": False,
    }
    # macOS: pmset + thermal levels via sysctl / powermetrics (often root).
    batt = _run_text(["pmset", "-g", "batt"])
    if batt:
        status["observable"] = True
        status["power_source"] = batt.splitlines()[0] if batt else None
        low = "low power" in batt.lower() or "battery power" in batt.lower()
        if low:
            status["notes"].append("on battery or low-power indicators present")
    thermal = _run_text(["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"])
    if thermal is not None:
        status["observable"] = True
        status["thermal_pressure"] = thermal
        try:
            if int(thermal) > 0:
                status["thermal_anomaly"] = True
                status["notes"].append(f"cpu_thermal_level={thermal}")
        except ValueError:
            status["notes"].append(f"unparsed thermal level: {thermal}")
    # Linux thermal zones (best-effort).
    thermal_root = Path("/sys/class/thermal")
    if thermal_root.is_dir():
        status["observable"] = True
        temps: list[float] = []
        for zone in sorted(thermal_root.glob("thermal_zone*/temp")):
            try:
                raw = zone.read_text(encoding="utf-8").strip()
                # Usually millidegrees C.
                temps.append(float(raw) / 1000.0 if float(raw) > 200 else float(raw))
            except (OSError, ValueError):
                continue
        if temps:
            status["thermal_pressure"] = {"max_c": max(temps), "readings_c": temps[:16]}
            if max(temps) >= 95.0:
                status["thermal_anomaly"] = True
                status["notes"].append(f"high thermal zone temp max_c={max(temps)}")
    return status


def _numpy_blas_config() -> dict[str, Any]:
    """Identify NumPy BLAS/LAPACK configuration; unknown is explicit."""
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return {
            "identified": False,
            "error": str(exc),
            "build_info": None,
            "blas_libs": None,
            "lapack_libs": None,
        }
    build_info: dict[str, Any] | None = None
    blas_libs: Any = None
    lapack_libs: Any = None
    try:
        if hasattr(np, "__config__") and hasattr(np.__config__, "show"):
            # np.__config__.show() prints; use get_info when available.
            pass
        cfg = getattr(np, "__config__", None)
        if cfg is not None and hasattr(cfg, "CONFIG"):
            build_info = dict(getattr(cfg, "CONFIG"))
        elif cfg is not None:
            # Fallback: stringify show_config capture is awkward; use dir keys.
            build_info = {"repr": repr(cfg)}
    except Exception as exc:  # noqa: BLE001
        build_info = {"error": str(exc)}

    # NumPy 2.x exposes build config differently.
    try:
        from numpy import __config__ as npc

        if hasattr(npc, "CONFIG"):
            build_info = npc.CONFIG  # type: ignore[assignment]
        for attr in ("blas_ilp64_opt_info", "blas_opt_info", "lapack_opt_info", "openblas_info"):
            if hasattr(npc, attr):
                info = getattr(npc, attr)
                if isinstance(info, dict) and info:
                    if "blas" in attr:
                        blas_libs = info
                    if "lapack" in attr:
                        lapack_libs = info
    except Exception:
        pass

    # Concrete vendor tokens only — generic "blas"/"lapack" is NOT identification.
    identified = False
    label = "unidentified"
    parts: list[str] = []
    if build_info is not None:
        parts.append(json.dumps(build_info, default=str).lower())
    if blas_libs is not None:
        parts.append(json.dumps(blas_libs, default=str).lower())
    if lapack_libs is not None:
        parts.append(json.dumps(lapack_libs, default=str).lower())
    blob = " ".join(parts)
    for token, vendor_label in BLAS_VENDOR_TOKENS:
        if token in blob:
            identified = True
            label = vendor_label
            break
    return {
        "identified": identified,
        "label": label if identified else "unidentified",
        "build_info": build_info,
        "blas_libs": blas_libs,
        "lapack_libs": lapack_libs,
        "numpy_version": str(getattr(np, "__version__", None)),
        "numpy_file": str(getattr(np, "__file__", None)),
        "vendor_tokens_checked": [label for _, label in BLAS_VENDOR_TOKENS],
    }


def harness_source_hashes(*, package_dir: Path | None = None) -> dict[str, str]:
    """SHA-256 of harness Python sources that define protocol/measurement/stats."""
    root = package_dir if package_dir is not None else Path(__file__).resolve().parent
    out: dict[str, str] = {}
    for name in HARNESS_SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"harness source missing: {path}")
        out[name] = _sha256_file(path)
    return out


def collect_provenance(*, repo_root: Path, run_id: str) -> dict[str, Any]:
    """Collect full reproducibility provenance for the research report."""
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    packages: dict[str, Any] = {}
    for report_key, import_name, dist_name in (
        ("numpy", "numpy", "numpy"),
        ("rextio", "rextio", "rextio"),
        ("rextio-numpy", "rextio_numpy", "rextio-numpy"),
    ):
        runtime_version = None
        module_file = None
        dist_version = None
        try:
            mod = __import__(import_name)
            runtime_version = getattr(mod, "__version__", None)
            module_file = getattr(mod, "__file__", None)
        except Exception:
            pass
        try:
            from importlib.metadata import PackageNotFoundError, version

            try:
                dist_version = version(dist_name)
            except PackageNotFoundError:
                dist_version = None
        except Exception:
            dist_version = None
        packages[report_key] = {
            "runtime_version": runtime_version,
            "distribution_version": dist_version,
            "module_file": module_file,
        }

    return {
        "timestamp_utc": now,
        "run_id": run_id,
        "platform": {
            "system": platform.system() or None,
            "release": platform.release() or None,
            "version": platform.version() or None,
            "machine": platform.machine() or None,
            "architecture": platform.architecture()[0] if platform.architecture() else None,
            "python_platform": platform.platform() or None,
        },
        "python": {
            "implementation": platform.python_implementation() or None,
            "version": platform.python_version() or None,
            "executable": sys.executable or None,
        },
        "cpu": _cpu_info(),
        # Callers of evidence runs should also set power_thermal_before/after
        # around measurements; this snapshot is the collection-time baseline.
        "power_thermal": _power_thermal_status(),
        "packages": packages,
        "blas": _numpy_blas_config(),
        "thread_env": dict(THREAD_ENV),
        "toolchain": {
            "cargo_path": shutil.which("cargo"),
            "cargo_version": _run_text(["cargo", "--version"]),
            "rustc_path": shutil.which("rustc"),
            "rustc_version": _run_text(["rustc", "--version"]),
            "rustc_verbose": _run_text(["rustc", "-vV"]),
        },
        "git": _git_info(repo_root),
        "harness_source_hashes": harness_source_hashes(),
        # Deliberately omit remote URLs.
    }


def validate_evidence_provenance(provenance: dict[str, Any]) -> list[str]:
    """Return invalidation reasons for an evidence run (empty if ok).

    Full evidence requires a clean committed harness revision, identified BLAS
    vendor, and no thermal anomaly before or after measurements. Any failure
    invalidates the run and requires a full rerun (not cherry-picked cells).
    """
    problems: list[str] = []
    git = provenance.get("git") or {}
    if git.get("status") != "ok":
        problems.append(
            "git status unavailable; evidence requires a clean committed harness "
            "revision — full rerun required"
        )
    if not git.get("revision"):
        problems.append(
            "missing git revision; evidence requires a clean committed harness "
            "revision — full rerun required"
        )
    if git.get("dirty") is True:
        problems.append(
            "working tree dirty=true; evidence requires a clean committed harness "
            "revision — full rerun required"
        )
    if git.get("dirty") is not False and git.get("status") == "ok":
        # dirty is None with status ok should not happen, but fail closed.
        if git.get("dirty") is None:
            problems.append(
                "git dirty flag unknown; evidence requires dirty=false — full rerun required"
            )

    blas = provenance.get("blas") or {}
    if not blas.get("identified"):
        problems.append(
            "NumPy BLAS/LAPACK vendor unidentified (concrete vendor token required; "
            "generic 'blas'/'lapack' is insufficient) — full rerun required"
        )
    label = str(blas.get("label") or "").lower()
    if blas.get("identified") and label in {"", "unidentified", "unknown", "unavailable"}:
        problems.append("BLAS label not a concrete vendor — full rerun required")

    for key, when in (
        ("power_thermal_before", "before measurements"),
        ("power_thermal_after", "after measurements"),
        # Fallback: single snapshot still checked if dual not present yet.
        ("power_thermal", "at provenance collection"),
    ):
        thermal = provenance.get(key)
        if thermal is None:
            if key in ("power_thermal_before", "power_thermal_after"):
                # Only required when the dual keys are expected; caller may still
                # be mid-run. validate_report_schema enforces both for final reports.
                continue
            continue
        if thermal.get("thermal_anomaly"):
            problems.append(
                f"thermal anomaly observed {when}; evidence run is invalid — full rerun required"
            )

    cpu = provenance.get("cpu") or {}
    if not cpu.get("model_name") and not cpu.get("platform_machine"):
        problems.append("CPU model/arch unidentified — full rerun required")

    hashes = provenance.get("harness_source_hashes") or {}
    for name in HARNESS_SOURCE_FILES:
        h = hashes.get(name)
        if not h or not isinstance(h, str) or len(h) != 64:
            problems.append(f"missing/invalid harness_source_hashes[{name}] — full rerun required")
    return problems


def generate_cell_inputs(
    cell: ShapeCell,
    inputs_dir: Path,
    *,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    """Create deterministic finite f64 inputs + NumPy reference; persist .npy."""
    import numpy as np

    seed = cell_input_seed(cell.cell_index, base_seed=base_seed)
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(cell.left_shape, dtype=np.float64)
    b = rng.standard_normal(cell.right_shape, dtype=np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise RuntimeError(f"non-finite inputs generated for {cell.cell_id}")
    # Reference: numpy.matmul (same as matmul spelling); all spellings must match.
    ref = np.matmul(a, b)
    if tuple(ref.shape) != cell.out_shape:
        raise RuntimeError(
            f"reference shape {ref.shape} != expected {cell.out_shape} for {cell.cell_id}"
        )
    cell_dir = inputs_dir / cell.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    a_path = cell_dir / "a.npy"
    b_path = cell_dir / "b.npy"
    ref_path = cell_dir / "ref.npy"
    np.save(a_path, a)
    np.save(b_path, b)
    np.save(ref_path, ref)
    # Verify all three spellings agree with the saved reference before timing.
    for name, got in (
        ("dot", np.dot(a, b)),
        ("matmul", np.matmul(a, b)),
        ("matmul_op", a @ b),
    ):
        if not np.allclose(got, ref, rtol=RTOL, atol=ATOL, equal_nan=EQUAL_NAN):
            raise RuntimeError(f"spelling {name} disagrees with reference for {cell.cell_id}")
        if str(got.dtype) != "float64":
            raise RuntimeError(f"spelling {name} dtype {got.dtype} != float64")
    return {
        "cell_id": cell.cell_id,
        "cell_index": cell.cell_index,
        "seed": seed,
        "a_path": str(a_path.resolve()),
        "b_path": str(b_path.resolve()),
        "ref_path": str(ref_path.resolve()),
        "a_sha256": _sha256_file(a_path),
        "b_sha256": _sha256_file(b_path),
        "ref_sha256": _sha256_file(ref_path),
        "left_shape": list(cell.left_shape),
        "right_shape": list(cell.right_shape),
        "out_shape": list(cell.out_shape),
    }


def calibrate_iterations(
    *,
    a_path: str,
    b_path: str,
    min_batch_s: float = 0.05,
    max_iterations: int = 100_000,
) -> int:
    """Pick iterations so one batch is roughly >= min_batch_s (NumPy matmul)."""
    import numpy as np
    import time

    a = np.load(a_path)
    b = np.load(b_path)
    # Warm once.
    _ = a @ b
    iterations = 1
    while iterations < max_iterations:
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = a @ b
        elapsed = time.perf_counter() - t0
        if elapsed >= min_batch_s:
            return iterations
        # Growth factor.
        if elapsed <= 0.0:
            iterations *= 10
        else:
            scale = max(2, int(min_batch_s / elapsed) + 1)
            iterations = min(max_iterations, iterations * scale)
    return max_iterations


def default_run_worker_subprocess(cfg: dict[str, Any], *, timeout: float = 600.0) -> dict[str, Any]:
    """Spawn an isolated worker process for one timing leg."""
    env = os.environ.copy()
    # Ensure repo root import of benchmarks.matmul_wave2 works in the child.
    # Thread env is applied inside the worker before NumPy import; do not
    # pre-import NumPy here.
    for k, v in THREAD_ENV.items():
        env[k] = v
    proc = subprocess.run(
        [python_executable(), "-m", "benchmarks.matmul_wave2.worker"],
        input=json.dumps(cfg),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0 and not (proc.stdout or "").strip():
        return {
            "ok": False,
            "error": f"worker exit {proc.returncode}: {proc.stderr[-2000:]}",
            "stderr": proc.stderr,
        }
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error": f"worker returned non-JSON: {exc}; stdout={proc.stdout[:500]!r}",
            "stderr": proc.stderr,
        }


def run_cell(
    cell: ShapeCell,
    *,
    input_meta: dict[str, Any],
    artifact: CandidateArtifact,
    warmups: int,
    samples: int,
    iterations: int,
    run_worker: RunWorkerFn,
) -> dict[str, Any]:
    """Run four sequential isolated legs and bootstrap three spelling ratios.

    All four legs are always attempted and recorded. Early leg failures still
    invalidate the cell, but do not skip remaining legs.
    """
    legs: dict[str, Any] = {}
    failure_reasons: list[str] = []
    for leg in LEG_ORDER:
        cfg: dict[str, Any] = {
            "leg": leg,
            "a_path": input_meta["a_path"],
            "b_path": input_meta["b_path"],
            "ref_path": input_meta["ref_path"],
            "warmups": warmups,
            "samples": samples,
            "iterations": iterations,
            "expected_shape": list(cell.out_shape),
            "rtol": RTOL,
            "atol": ATOL,
            "equal_nan": EQUAL_NAN,
            "thread_env": dict(THREAD_ENV),
        }
        if leg == "candidate":
            cfg.update(
                {
                    "candidate_load_dir": artifact.load_dir,
                    "candidate_module_name": artifact.module_name,
                    "candidate_function_name": artifact.function_name,
                    "expected_module_path": artifact.module_path,
                    "expected_sha256": artifact.artifact_sha256,
                }
            )
        result = run_worker(cfg)
        process = result.get("process") or {}
        base_leg: dict[str, Any] = {
            "warmups": warmups,
            "samples": samples,
            "iterations": iterations,
            "meta": result.get("meta") or {},
            "process": process,
            "thread_env": result.get("thread_env") or process.get("thread_env_applied"),
            "numpy_version": result.get("numpy_version") or process.get("numpy_version"),
            "ok": bool(result.get("ok")),
            "error": result.get("error"),
        }
        if not result.get("ok"):
            failure_reasons.append(f"leg {leg} failed: {result.get('error')}")
            legs[leg] = base_leg
            continue
        per_call = list(result.get("per_call_samples_s") or [])
        sample_err = validate_positive_finite_samples(per_call, label=f"{cell.cell_id}:{leg}")
        if sample_err:
            failure_reasons.append(sample_err)
            base_leg["per_call_samples_s"] = per_call
            base_leg["batch_samples_s"] = list(result.get("batch_samples_s") or [])
            base_leg["ok"] = False
            base_leg["error"] = sample_err
            legs[leg] = base_leg
            continue
        if len(per_call) != samples:
            msg = f"leg {leg}: expected {samples} samples, got {len(per_call)}"
            failure_reasons.append(msg)
            base_leg["per_call_samples_s"] = per_call
            base_leg["batch_samples_s"] = list(result.get("batch_samples_s") or [])
            base_leg["ok"] = False
            base_leg["error"] = msg
            legs[leg] = base_leg
            continue
        summary = summarize_per_call(per_call)
        legs[leg] = {
            **base_leg,
            "batch_samples_s": list(result["batch_samples_s"]),
            "per_call_samples_s": list(per_call),
            "summary": summary.to_dict(),
            "ok": True,
            "error": None,
        }

    base_out: dict[str, Any] = {
        "cell_id": cell.cell_id,
        "cell_index": cell.cell_index,
        "family": cell.family,
        "n": cell.n,
        "h": cell.h,
        "left_shape": list(cell.left_shape),
        "right_shape": list(cell.right_shape),
        "out_shape": list(cell.out_shape),
        "inputs": input_meta,
        "iterations": iterations,
        "legs": legs,
        "leg_attempts": len(legs),
    }
    if failure_reasons or len(legs) != 4:
        return {
            **base_out,
            "status": "failed",
            "reason": "; ".join(failure_reasons) if failure_reasons else "incomplete legs",
            "spellings": {},
            "conservative_point": None,
            "conservative_ci_lower": None,
        }

    cand_samples = legs["candidate"]["per_call_samples_s"]
    spellings_out: dict[str, Any] = {}
    boot_map = {}
    for spelling in SPELLINGS:
        seed = bootstrap_seed(cell.cell_index, spelling)
        boot = bootstrap_median_ratio(
            legs[spelling]["per_call_samples_s"],
            cand_samples,
            seed=seed,
            n_resamples=BOOTSTRAP_RESAMPLES,
            alpha=BOOTSTRAP_ALPHA,
        )
        boot_map[spelling] = boot
        spellings_out[spelling] = {
            "label": {
                "dot": "numpy.dot(a, b)",
                "matmul": "numpy.matmul(a, b)",
                "matmul_op": "a @ b",
            }[spelling],
            "bootstrap": boot.to_dict(),
        }
    agg = cell_conservative_aggregate(boot_map)
    return {
        **base_out,
        "status": "ok",
        "reason": None,
        "spellings": spellings_out,
        "conservative_point": agg.conservative_point,
        "conservative_ci_lower": agg.conservative_ci_lower,
        "conservative_aggregate": agg.to_dict(),
    }


def build_verdicts(
    *,
    cells: list[dict[str, Any]],
    evidence: bool,
    provenance_problems: list[str],
) -> dict[str, Any]:
    """Construct research + product verdicts. Product is always NO-GO.

    Performance conclusions require the exact frozen 27 cell ids in order;
    subsets and wrong-id 27-length lists cannot pass the research gate.
    """
    cell_ids = [str(c.get("cell_id")) for c in cells]
    matrix_ok = report_cell_ids_match_frozen(cell_ids)
    ok_cells = [c for c in cells if c.get("status") == "ok"]
    all_valid = (
        matrix_ok
        and len(ok_cells) == len(EXPECTED_CELL_IDS)
        and all(c.get("status") == "ok" for c in cells)
    )
    perf_reason: str | None
    gmean: float | None
    min_lo: float | None
    if not evidence:
        perf_passed = False
        perf_reason = "smoke mode: performance conclusion permanently disallowed (evidence=false)"
        gmean = None
        min_lo = None
        gate: dict[str, Any] = {
            "passed": False,
            "reason": perf_reason,
            "geometric_mean": None,
            "min_ci_lower": None,
            "all_ci_lowers_gt_one": False,
            "geometric_mean_ok": False,
            "matrix_ok": matrix_ok,
        }
    elif not matrix_ok:
        perf_passed = False
        perf_reason = (
            "performance conclusion requires the exact frozen 27 cell ids in stable "
            f"order; got {cell_ids!r}"
        )
        gmean = None
        min_lo = None
        gate = {
            "passed": False,
            "reason": perf_reason,
            "geometric_mean": None,
            "min_ci_lower": None,
            "all_ci_lowers_gt_one": False,
            "geometric_mean_ok": False,
            "matrix_ok": False,
        }
    elif provenance_problems:
        perf_passed = False
        perf_reason = "; ".join(provenance_problems)
        gmean = None
        min_lo = None
        gate = {
            "passed": False,
            "reason": perf_reason,
            "geometric_mean": None,
            "min_ci_lower": None,
            "all_ci_lowers_gt_one": False,
            "geometric_mean_ok": False,
            "matrix_ok": True,
        }
    elif not all_valid:
        perf_passed = False
        perf_reason = f"not all cells valid: {len(ok_cells)}/{len(EXPECTED_CELL_IDS)} ok"
        gmean = None
        min_lo = None
        gate = {
            "passed": False,
            "reason": perf_reason,
            "geometric_mean": None,
            "min_ci_lower": None,
            "all_ci_lowers_gt_one": False,
            "geometric_mean_ok": False,
            "matrix_ok": True,
        }
    else:
        # Order by frozen matrix so gate cannot be spoofed by reordering.
        by_id = {str(c.get("cell_id")): c for c in cells}
        ordered = [by_id[cid] for cid in EXPECTED_CELL_IDS]
        points = [float(c["conservative_point"]) for c in ordered]
        lowers = [float(c["conservative_ci_lower"]) for c in ordered]
        gate = performance_gate(lowers, points, cell_ids=list(EXPECTED_CELL_IDS))
        perf_passed = bool(gate["passed"])
        reason_raw = gate.get("reason")
        perf_reason = None if reason_raw is None else str(reason_raw)
        gmean_raw = gate.get("geometric_mean")
        gmean = None if gmean_raw is None else float(gmean_raw)
        min_lo_raw = gate.get("min_ci_lower")
        min_lo = None if min_lo_raw is None else float(min_lo_raw)

    # Dispatchability hard-coded failed → product always NO-GO.
    return {
        "dispatchability_status": DISPATCHABILITY_STATUS,
        "dispatchability_reason": DISPATCHABILITY_REASON,
        "performance_passed": bool(perf_passed) if evidence and matrix_ok else False,
        "performance_reason": perf_reason,
        "performance_gate": gate,
        "product_verdict": PRODUCT_VERDICT,
        "product_verdict_detail": PRODUCT_VERDICT_DETAIL,
        "product_reason": (
            "Dispatchability gate failed a priori (rank/dtype without dimensions). "
            "Product claim remains NO-GO / fallback-retained regardless of timing."
        ),
        "aggregate_geometric_mean": gmean,
        "aggregate_min_ci_lower": min_lo,
        "all_cells_valid": all_valid,
        "matrix_ok": matrix_ok,
    }


def run_harness(
    *,
    output_dir: Path | str,
    evidence: bool = False,
    base_seed: int = BASE_SEED,
    cells: Sequence[ShapeCell] | None = None,
    run_id: str | None = None,
    build_fn: BuildCandidateFn | None = None,
    run_worker: RunWorkerFn | None = None,
    repo_root: Path | str | None = None,
    calibrate: bool = True,
    min_batch_s: float = 0.05,
    max_iterations: int = 100_000,
    fixed_iterations: int | None = None,
    skip_ensure_importable: bool = False,
) -> dict[str, Any]:
    """Execute the research harness and write reports under *output_dir*.

    Evidence mode requires the exact frozen 27-cell matrix, real cargo build,
    real worker subprocesses, frozen base seed, and calibration before any
    measurement. Injected build/worker, subsets, fixed iterations, and disabled
    calibration are rejected before build. Smoke mode may inject for tests.
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rid = run_id or uuid.uuid4().hex
    warmups, samples = run_counts(evidence=evidence)
    selected = tuple(cells) if cells is not None else SHAPE_CELLS

    if evidence:
        # Fail closed before cargo build / worker subprocesses.
        if build_fn is not None:
            raise ValueError(
                "evidence mode rejects injected build_fn; use the real candidate builder"
            )
        if run_worker is not None:
            raise ValueError(
                "evidence mode rejects injected run_worker; use real subprocess isolation"
            )
        if skip_ensure_importable:
            raise ValueError("evidence mode rejects skip_ensure_importable")
        if int(base_seed) != int(BASE_SEED):
            raise ValueError(
                f"evidence mode requires frozen base_seed={BASE_SEED}, got {base_seed!r}"
            )
        require_exact_frozen_matrix(selected)
        if fixed_iterations is not None:
            raise ValueError("evidence mode rejects fixed_iterations; calibration is required")
        if not calibrate:
            raise ValueError("evidence mode rejects disabled calibration")

    full_matrix = is_exact_frozen_matrix(selected)
    # Performance conclusions require evidence AND the full frozen matrix.
    performance_conclusion_allowed = bool(evidence) and full_matrix
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]

    settings = RunSettings(
        evidence=evidence,
        warmups=warmups,
        samples=samples,
        base_seed=base_seed,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        output_dir=str(out),
        run_id=rid,
        cells=selected,
        calibrate=calibrate and fixed_iterations is None,
        min_batch_s=min_batch_s,
        max_iterations=max_iterations,
        performance_conclusion_allowed=performance_conclusion_allowed,
    )

    provenance = collect_provenance(repo_root=root, run_id=rid)
    power_before = _power_thermal_status()
    provenance["power_thermal_before"] = power_before
    # Keep power_thermal as the latest snapshot for generic readers.
    provenance["power_thermal"] = power_before

    # Stage dirs.
    artifacts_dir = out / "artifacts" / rid
    inputs_dir = out / "inputs" / rid
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    builder = build_fn or build_candidate
    try:
        artifact = builder(artifacts_dir)
        if not skip_ensure_importable:
            ensure_importable(artifact)
    except Exception:
        # Fail closed without claiming evidence if build cannot proceed.
        if evidence:
            power_after = _power_thermal_status()
            provenance["power_thermal_after"] = power_after
            provenance["power_thermal"] = power_after
        raise

    worker = run_worker or default_run_worker_subprocess

    cell_results: list[dict[str, Any]] = []
    try:
        for cell in selected:
            input_meta = generate_cell_inputs(cell, inputs_dir, base_seed=base_seed)
            if fixed_iterations is not None:
                iterations = int(fixed_iterations)
            elif calibrate:
                iterations = calibrate_iterations(
                    a_path=input_meta["a_path"],
                    b_path=input_meta["b_path"],
                    min_batch_s=min_batch_s,
                    max_iterations=max_iterations,
                )
            else:
                iterations = 1
            cell_result = run_cell(
                cell,
                input_meta=input_meta,
                artifact=artifact,
                warmups=warmups,
                samples=samples,
                iterations=iterations,
                run_worker=worker,
            )
            cell_results.append(cell_result)
    finally:
        power_after = _power_thermal_status()
        provenance["power_thermal_after"] = power_after
        provenance["power_thermal"] = power_after

    provenance_problems = validate_evidence_provenance(provenance) if evidence else []

    verdicts = build_verdicts(
        cells=cell_results,
        evidence=evidence,
        provenance_problems=provenance_problems,
    )
    ok_cells = [c for c in cell_results if c.get("status") == "ok"]
    aggregate = {
        "n_cells": len(cell_results),
        "n_ok": len(ok_cells),
        "geometric_mean": verdicts.get("aggregate_geometric_mean"),
        "min_ci_lower": verdicts.get("aggregate_min_ci_lower"),
        "performance_gate": verdicts.get("performance_gate"),
    }

    honesty = {
        "smoke_not_evidence": not evidence,
        "performance_conclusion_allowed": settings.performance_conclusion_allowed,
        "dispatchability_hardcoded_failed": True,
        "product_always_nogo": True,
        "kernel_only_timing_forbidden_for_gate": True,
        "no_blas_feature_on_candidate": True,
        "no_product_rule_added": True,
        "four_legs_per_cell_not_81_product_cells": True,
        "evidence_invalid_if_unidentified_blas_or_thermal": True,
        "evidence_requires_clean_git": True,
        "provenance_problems": provenance_problems,
    }

    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": rid,
        "settings": settings.to_dict(),
        "provenance": provenance,
        "protocol": protocol_manifest(),
        "candidate": artifact.to_dict() if hasattr(artifact, "to_dict") else asdict(artifact),
        "cells": cell_results,
        "aggregate": aggregate,
        "verdicts": verdicts,
        "honesty": honesty,
    }

    schema_errors = validate_report_schema(report)
    apply_evidence_integrity(
        report,
        schema_errors=schema_errors,
        provenance_problems=provenance_problems,
    )

    json_path = out / f"matmul-wave2-report-{rid}.json"
    md_path = out / f"matmul-wave2-report-{rid}.md"
    write_json_report(json_path, report)
    write_markdown_report(md_path, report)
    report["artifact_paths"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "artifacts_dir": str(artifacts_dir),
        "inputs_dir": str(inputs_dir),
    }
    # Rewrite JSON with artifact_paths included.
    write_json_report(json_path, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.matmul_wave2",
        description=(
            "Wave 2 rank-2 f64 matmul research harness (preregistered protocol). "
            "Does not modify product claims. Product verdict is always NO-GO."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="User-selected output directory (gitignored results; never stage artifacts).",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Low-sample harness validation; evidence=false; no performance conclusion.",
    )
    mode.add_argument(
        "--evidence",
        action="store_true",
        help=f"Full evidence run (>= {EVIDENCE_WARMUPS} warmups, >= {EVIDENCE_SAMPLES} samples).",
    )
    p.add_argument("--base-seed", type=int, default=BASE_SEED, help="Base RNG seed for inputs.")
    p.add_argument("--run-id", default=None, help="Optional fixed run id (default: random hex).")
    p.add_argument(
        "--cell-id",
        action="append",
        default=None,
        help=(
            "Optional cell_id filter (repeatable; smoke/debug only). "
            "Incompatible with --evidence (frozen protocol requires all 27 cells)."
        ),
    )
    p.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable iteration calibration (use 1 iteration per sample batch).",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Fixed iterations per sample batch (skips calibration).",
    )
    p.add_argument(
        "--min-batch-s",
        type=float,
        default=0.05,
        help="Calibration target minimum batch wall time (seconds).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    evidence = bool(args.evidence)
    selected: Sequence[ShapeCell] | None = None
    if args.cell_id:
        if evidence:
            parser.error(
                "--cell-id is smoke/debug only and cannot be combined with --evidence; "
                "the frozen protocol requires all 27 cells for evidence runs"
            )
        wanted = set(args.cell_id)
        selected = tuple(c for c in SHAPE_CELLS if c.cell_id in wanted)
        missing = wanted - {c.cell_id for c in selected}
        if missing:
            parser.error(f"unknown cell id(s): {sorted(missing)}")
        if not selected:
            parser.error("no cells selected")
    report = run_harness(
        output_dir=args.output_dir,
        evidence=evidence,
        base_seed=int(args.base_seed),
        cells=selected,
        run_id=args.run_id,
        calibrate=not args.no_calibrate,
        min_batch_s=float(args.min_batch_s),
        fixed_iterations=args.iterations,
    )
    print(f"Wrote {report['artifact_paths']['json']}")
    print(f"Wrote {report['artifact_paths']['markdown']}")
    print(f"Product verdict: {report['verdicts']['product_verdict']}")
    print(f"Dispatchability: {report['verdicts']['dispatchability_status']}")
    if evidence:
        status = report.get("evidence_status", "UNKNOWN")
        print(f"Evidence status: {status}")
        print(f"Performance research gate: {report['verdicts']['performance_passed']}")
        if status == "INVALID":
            reasons = report.get("invalid_reasons") or report.get("schema_errors") or []
            print("Evidence INVALID:")
            for reason in reasons[:20]:
                print(f"  - {reason}")
            # Schema/integrity invalidity is fatal for the process exit code.
            # A valid measured slowdown (VALID + performance_passed=false) returns 0.
            return 1
    else:
        print("Smoke mode: no performance conclusion.")
    return 0
