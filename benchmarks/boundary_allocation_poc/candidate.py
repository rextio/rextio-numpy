"""Build and stage the boundary-allocation PoC research cdylib."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from benchmarks.boundary_allocation_poc.protocol import (
    CANDIDATE_CRATE_NAME,
    CANDIDATE_MODULE_NAME,
    CARGO_BUILD_ARGS,
    RUST_STRATEGIES,
    STRATEGY_FUNCTIONS,
)

RUST_CANDIDATE_DIR = Path(__file__).resolve().parent / "rust_candidate"

RunCommandFn = Callable[..., subprocess.CompletedProcess[str]]

_BUILD_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "CARGO_TERM_COLOR",
        "CARGO_BUILD_TARGET",
        "CARGO_TARGET_DIR",
        "CARGO_HOME",
        "CARGO_INCREMENTAL",
        "RUSTC",
        "RUSTFLAGS",
        "RUST_BACKTRACE",
        "RUST_LOG",
        "RUSTC_WRAPPER",
        "RUSTC_LINKER",
        "PYO3_PYTHON",
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "PATH",
        "MACOSX_DEPLOYMENT_TARGET",
        "SDKROOT",
    }
)

_SECRET_KEY_MARKERS: tuple[str, ...] = (
    "TOKEN",
    "PASSWORD",
    "SECRET",
    "CREDENTIAL",
    "AUTH",
    "PRIVATE_KEY",
    "PRIVATEKEY",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
)


@dataclass(frozen=True)
class CandidateArtifact:
    """Staged native candidate ready for import."""

    module_name: str
    load_dir: str
    module_path: str
    artifact_sha256: str
    build_wall_s: float
    cargo_stdout: str
    cargo_stderr: str
    crate_dir: str
    cargo_toml_sha256: str
    cargo_lock_sha256: str
    lib_rs_sha256: str
    cargo_config_sha256: str
    rustc_verbose: str | None
    cargo_version: str | None
    profile: str
    build_env: dict[str, str]
    strategy_functions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact dict."""
        return asdict(self)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ext_suffix() -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("sysconfig EXT_SUFFIX is unavailable; cannot stage candidate")
    return str(suffix)


def is_secret_env_key(key: str) -> bool:
    """Return True if *key* looks like a credential-bearing environment variable."""
    upper = key.upper()
    return any(marker in upper for marker in _SECRET_KEY_MARKERS)


def record_build_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return a secret-safe allowlisted snapshot of build environment keys."""
    recorded: dict[str, str] = {}
    for key in sorted(env):
        if is_secret_env_key(key):
            continue
        if key == "PATH":
            recorded["PATH"] = "(set)"
            continue
        allowed = key in _BUILD_ENV_ALLOWLIST or (
            key.startswith(("CARGO_", "RUST", "PYO3_")) and not is_secret_env_key(key)
        )
        if key.startswith("CARGO_REGISTRIES_"):
            continue
        if not allowed:
            continue
        recorded[key] = str(env[key])
    return recorded


def _find_built_library(crate_dir: Path, crate_name: str) -> Path:
    target_release = crate_dir / "target" / "release"
    candidates = [
        target_release / f"lib{crate_name}.so",
        target_release / f"lib{crate_name}.dylib",
        target_release / f"{crate_name}.dll",
        target_release / f"lib{crate_name}.dll",
        target_release / f"{crate_name}.so",
        target_release / f"{crate_name}.dylib",
    ]
    deps = target_release / "deps"
    if deps.is_dir():
        for p in sorted(deps.iterdir()):
            if crate_name in p.name and p.suffix in {".so", ".dylib", ".dll"}:
                candidates.append(p)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"built cdylib for {crate_name!r} not found under {target_release}"
    )


def _run_text(argv: list[str], *, timeout: float = 30.0) -> str | None:
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


def validate_loaded_module(
    module_file: str | None,
    *,
    expected_path: str | Path,
    expected_sha256: str,
) -> str | None:
    """Return None when the loaded module path and hash match expectations."""
    if module_file is None or module_file == "":
        return "imported candidate module has no __file__ (fail-closed)"
    try:
        loaded = Path(module_file).resolve()
        expected = Path(expected_path).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return f"failed to resolve candidate paths: {exc}"
    if loaded != expected:
        return (
            f"imported candidate path {str(loaded)!r} != expected "
            f"{str(expected)!r} (fail-closed)"
        )
    if not loaded.is_file():
        return f"imported candidate path is not a file: {str(loaded)!r}"
    actual_sha = _sha256_file(loaded)
    if actual_sha != expected_sha256:
        return (
            f"candidate artifact SHA-256 mismatch: got {actual_sha}, "
            f"expected {expected_sha256} (fail-closed)"
        )
    return None


def build_candidate(
    stage_dir: Path | str,
    *,
    crate_dir: Path | str | None = None,
    run_command: RunCommandFn | None = None,
    cargo_bin: str | None = None,
    env: dict[str, str] | None = None,
    profile: str = "release",
) -> CandidateArtifact:
    """Build the research cdylib once and stage it under *stage_dir*."""
    crate = Path(crate_dir) if crate_dir is not None else RUST_CANDIDATE_DIR
    crate = crate.resolve()
    stage = Path(stage_dir).resolve()
    stage.mkdir(parents=True, exist_ok=True)

    cargo_toml = crate / "Cargo.toml"
    cargo_lock = crate / "Cargo.lock"
    lib_rs = crate / "src" / "lib.rs"
    cargo_config = crate / ".cargo" / "config.toml"
    for required in (cargo_toml, lib_rs, cargo_config):
        if not required.is_file():
            raise FileNotFoundError(f"candidate crate missing required file: {required}")
    # Cargo.lock is generated on first build if missing; prefer committed lock.
    if not cargo_lock.is_file() and run_command is None:
        # Allow first local build to create the lock via cargo.
        pass

    cargo = cargo_bin or shutil.which("cargo")
    if cargo is None and run_command is None:
        raise FileNotFoundError("cargo not found on PATH; cannot build research candidate")

    build_env = os.environ.copy()
    if env:
        build_env.update(env)
    build_env.setdefault("CARGO_TERM_COLOR", "never")
    build_env.setdefault("PYO3_PYTHON", sys.executable)

    # Prefer --locked when lock exists; otherwise plain release build once.
    build_args = list(CARGO_BUILD_ARGS)
    if not cargo_lock.is_file():
        build_args = [a for a in build_args if a != "--locked"]

    argv = [cargo or "cargo", *build_args]
    runner = run_command or subprocess.run
    t0 = time.perf_counter()
    completed = runner(
        argv,
        cwd=str(crate),
        check=False,
        capture_output=True,
        text=True,
        env=build_env,
    )
    build_wall = time.perf_counter() - t0
    stdout = (completed.stdout or "")[-8000:]
    stderr = (completed.stderr or "")[-8000:]
    if getattr(completed, "returncode", 1) != 0:
        raise RuntimeError(
            f"cargo build failed (code={getattr(completed, 'returncode', None)}): "
            f"{stderr[-2000:] or stdout[-2000:]}"
        )

    # Refresh lock hash after possible first generation.
    if not cargo_lock.is_file():
        cargo_lock = crate / "Cargo.lock"
    if not cargo_lock.is_file():
        raise FileNotFoundError(f"Cargo.lock missing after build: {cargo_lock}")

    ext = _ext_suffix()
    staged_path = stage / f"{CANDIDATE_MODULE_NAME}{ext}"
    try:
        built = _find_built_library(crate, CANDIDATE_CRATE_NAME)
        shutil.copy2(built, staged_path)
    except FileNotFoundError:
        if not staged_path.is_file():
            raise

    if not staged_path.is_file():
        raise RuntimeError(f"staged candidate missing: {staged_path}")

    # Widen StrategyId keys to str for the JSON-friendly artifact field type.
    rust_fn_map: dict[str, str] = {
        str(sid): STRATEGY_FUNCTIONS[sid] for sid in RUST_STRATEGIES
    }
    return CandidateArtifact(
        module_name=CANDIDATE_MODULE_NAME,
        load_dir=str(stage),
        module_path=str(staged_path.resolve()),
        artifact_sha256=_sha256_file(staged_path),
        build_wall_s=float(build_wall),
        cargo_stdout=stdout,
        cargo_stderr=stderr,
        crate_dir=str(crate),
        cargo_toml_sha256=_sha256_file(cargo_toml),
        cargo_lock_sha256=_sha256_file(cargo_lock),
        lib_rs_sha256=_sha256_file(lib_rs),
        cargo_config_sha256=_sha256_file(cargo_config),
        rustc_verbose=_run_text(["rustc", "-vV"]),
        cargo_version=_run_text(["cargo", "--version"]),
        profile=profile,
        build_env=record_build_env(build_env),
        strategy_functions=rust_fn_map,
    )


def ensure_importable(artifact: CandidateArtifact) -> Any:
    """Import staged module, verify path/hash, return module object."""
    import importlib
    import sys as _sys

    load_dir = str(Path(artifact.load_dir).resolve())
    if load_dir not in _sys.path:
        _sys.path.insert(0, load_dir)
    _sys.modules.pop(artifact.module_name, None)
    mod = importlib.import_module(artifact.module_name)
    err = validate_loaded_module(
        getattr(mod, "__file__", None),
        expected_path=artifact.module_path,
        expected_sha256=artifact.artifact_sha256,
    )
    if err is not None:
        raise RuntimeError(err)
    for sid in RUST_STRATEGIES:
        name = STRATEGY_FUNCTIONS[sid]
        if not hasattr(mod, name):
            raise RuntimeError(f"candidate module missing function {name!r}")
    return mod


def candidate_source_hashes(crate_dir: Path | str | None = None) -> dict[str, str]:
    """Return SHA-256 hashes of committed crate source and build-config files."""
    crate = Path(crate_dir) if crate_dir is not None else RUST_CANDIDATE_DIR
    crate = crate.resolve()
    out = {
        "Cargo.toml": _sha256_file(crate / "Cargo.toml"),
        "src/lib.rs": _sha256_file(crate / "src" / "lib.rs"),
        ".cargo/config.toml": _sha256_file(crate / ".cargo" / "config.toml"),
    }
    lock = crate / "Cargo.lock"
    if lock.is_file():
        out["Cargo.lock"] = _sha256_file(lock)
    return out


def python_executable() -> str:
    """Return the interpreter used for local harness runs."""
    return sys.executable
