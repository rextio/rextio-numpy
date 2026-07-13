"""Reproducibility metadata collection.

Unavailable fields are explicit ``None`` / ``"unavailable"`` — never invented.

Package provenance
------------------
``metadata.packages.*`` records the **runtime module version** actually
imported in this process (what the suite executes), not the installed
distribution metadata alone. Editable / source-checkout installs often
diverge: e.g. ``importlib.metadata.version("rextio")`` may report an older
wheel while ``rextio.__version__`` and ``rextio.__file__`` point at a sibling
source tree.

Additional parallel maps keep the installed distribution version, module
origin path, and an explicit mismatch flag for each tracked package.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

# Report key → (import name, distribution name for importlib.metadata).
_PACKAGE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("numpy", "numpy", "numpy"),
    ("rextio", "rextio", "rextio"),
    ("rextio-numpy", "rextio_numpy", "rextio-numpy"),
)

ImportModuleFn = Callable[[str], ModuleType]
DistributionVersionFn = Callable[[str], str | None]


def _run_text(argv: list[str], *, timeout: float = 5.0) -> str | None:
    """Run a short command; return stripped stdout or None on any failure."""
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


def _distribution_version(name: str) -> str | None:
    """Return the installed distribution version, or None if not found."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _load_module(
    import_name: str,
    *,
    import_module: ImportModuleFn | None = None,
) -> ModuleType | None:
    """Load a module for provenance without inventing a different import path.

    When *import_module* is provided (tests), call it exclusively so probes
    are not shadowed by already-imported real packages.

    In production (no injector): prefer an already-imported ``sys.modules``
    entry so metadata observes the same object the process is already
    running, and only import when missing.
    """
    if import_module is not None:
        try:
            return import_module(import_name)
        except Exception:
            return None
    existing = sys.modules.get(import_name)
    if existing is not None:
        return existing
    try:
        return importlib.import_module(import_name)
    except Exception:
        return None


def _module_version(mod: ModuleType) -> str | None:
    raw = getattr(mod, "__version__", None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _module_file(mod: ModuleType) -> str | None:
    raw = getattr(mod, "__file__", None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _versions_mismatch(runtime: str | None, distribution: str | None) -> bool | None:
    """Return True when both versions are known and differ; None if either is unknown."""
    if runtime is None or distribution is None:
        return None
    return runtime != distribution


def resolve_package_provenance(
    *,
    import_name: str,
    distribution_name: str,
    import_module: ImportModuleFn | None = None,
    distribution_version: DistributionVersionFn | None = None,
) -> dict[str, Any]:
    """Resolve runtime vs distribution provenance for one package.

    Returns keys:
    - ``runtime_version``: module ``__version__`` (or None)
    - ``distribution_version``: ``importlib.metadata`` version (or None)
    - ``module_file``: module ``__file__`` origin path (or None)
    - ``mismatch``: True/False when both versions known; None otherwise
    """
    dist_fn = distribution_version or _distribution_version
    dist_ver = dist_fn(distribution_name)
    mod = _load_module(import_name, import_module=import_module)
    if mod is None:
        return {
            "runtime_version": None,
            "distribution_version": dist_ver,
            "module_file": None,
            "mismatch": _versions_mismatch(None, dist_ver),
        }
    runtime_ver = _module_version(mod)
    module_file = _module_file(mod)
    return {
        "runtime_version": runtime_ver,
        "distribution_version": dist_ver,
        "module_file": module_file,
        "mismatch": _versions_mismatch(runtime_ver, dist_ver),
    }


def collect_package_provenance(
    *,
    import_module: ImportModuleFn | None = None,
    distribution_version: DistributionVersionFn | None = None,
) -> dict[str, Any]:
    """Collect package maps for the report metadata block.

    * ``packages`` — runtime module versions (what actually executed)
    * ``package_distributions`` — installed distribution versions
    * ``package_module_files`` — ``__file__`` origins for visibility of
      editable / source-checkout imports
    * ``package_version_mismatches`` — explicit True/False/None per package
    """
    packages: dict[str, str | None] = {}
    distributions: dict[str, str | None] = {}
    module_files: dict[str, str | None] = {}
    mismatches: dict[str, bool | None] = {}
    for report_key, import_name, dist_name in _PACKAGE_SPECS:
        info = resolve_package_provenance(
            import_name=import_name,
            distribution_name=dist_name,
            import_module=import_module,
            distribution_version=distribution_version,
        )
        packages[report_key] = info["runtime_version"]
        distributions[report_key] = info["distribution_version"]
        module_files[report_key] = info["module_file"]
        mismatches[report_key] = info["mismatch"]
    return {
        "packages": packages,
        "package_distributions": distributions,
        "package_module_files": module_files,
        "package_version_mismatches": mismatches,
    }


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform_processor": platform.processor() or None,
        "platform_machine": platform.machine() or None,
        "python_platform": platform.platform() or None,
    }
    # Linux
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            text = cpuinfo.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        model = None
        for line in text.splitlines():
            if line.lower().startswith("model name") and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
        info["model_name"] = model
        info["source"] = "/proc/cpuinfo"
        return info
    # macOS
    brand = _run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
    if brand is not None:
        info["model_name"] = brand
        info["source"] = "sysctl machdep.cpu.brand_string"
        return info
    info["model_name"] = None
    info["source"] = "unavailable"
    return info


def _git_info(repo_root: Path | None) -> dict[str, Any]:
    if repo_root is None:
        return {"revision": None, "dirty": None, "status": "unavailable"}
    if shutil.which("git") is None:
        return {"revision": None, "dirty": None, "status": "unavailable"}
    rev = _run_text(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    if rev is None:
        return {"revision": None, "dirty": None, "status": "unavailable"}
    porcelain = _run_text(["git", "-C", str(repo_root), "status", "--porcelain"])
    # empty stdout => clean; None => unavailable
    if porcelain is None:
        dirty: bool | None = None
        status = "unavailable"
    else:
        dirty = bool(porcelain.strip())
        status = "ok"
    return {"revision": rev, "dirty": dirty, "status": status}


def _blas_thread_env() -> dict[str, Any]:
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "NPY_NUM_BUILD_JOBS",
    )
    return {k: os.environ.get(k) for k in keys}


def collect_metadata(
    *,
    repo_root: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    import_module: ImportModuleFn | None = None,
    distribution_version: DistributionVersionFn | None = None,
) -> dict[str, Any]:
    """Collect reproducibility metadata; missing fields are explicit nulls.

    *clock* is injectable for deterministic tests (defaults to ``datetime.now``
    in UTC).

    *import_module* and *distribution_version* are injectable so tests can
    exercise match / mismatch / missing cases without altering the process
    import graph. Production callers leave them unset so provenance reflects
    the same modules this interpreter would execute.
    """
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    cargo = shutil.which("cargo")
    rustc = shutil.which("rustc")
    cargo_version = _run_text(["cargo", "--version"]) if cargo else None
    rustc_version = _run_text(["rustc", "--version"]) if rustc else None

    package_block = collect_package_provenance(
        import_module=import_module,
        distribution_version=distribution_version,
    )

    return {
        "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
        "platform": {
            "system": platform.system() or None,
            "release": platform.release() or None,
            "version": platform.version() or None,
            "machine": platform.machine() or None,
            "architecture": platform.architecture()[0] if platform.architecture() else None,
        },
        "python": {
            "implementation": platform.python_implementation() or None,
            "version": platform.python_version() or None,
            "executable": sys.executable or None,
        },
        "cpu": _cpu_info(),
        "packages": package_block["packages"],
        "package_distributions": package_block["package_distributions"],
        "package_module_files": package_block["package_module_files"],
        "package_version_mismatches": package_block["package_version_mismatches"],
        "toolchain": {
            "cargo_path": cargo,
            "cargo_version": cargo_version,
            "rustc_path": rustc,
            "rustc_version": rustc_version,
        },
        "blas_thread_env": _blas_thread_env(),
        "git": _git_info(repo_root),
    }
