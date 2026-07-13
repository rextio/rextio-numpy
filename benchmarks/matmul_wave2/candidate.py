"""Build and stage the standalone Wave 2 matmul research cdylib.

Builds exactly once per run with ``cargo build --release --locked``, then
copies the artifact into a run-specific directory under the user-selected
output dir using the interpreter's ``EXT_SUFFIX``. Load path and SHA-256 are
verified fail-closed before any timing leg proceeds.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from benchmarks.matmul_wave2.protocol import (
    CANDIDATE_CRATE_NAME,
    CANDIDATE_FUNCTION_NAME,
    CANDIDATE_MODULE_NAME,
    CARGO_BUILD_ARGS,
)

# Harness-owned crate root (committed with Cargo.lock).
RUST_CANDIDATE_DIR = Path(__file__).resolve().parent / "rust_candidate"

RunCommandFn = Callable[..., subprocess.CompletedProcess[str]]

# Build-env keys that are safe to record (values still scrubbed if secret-like).
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

# Substrings that mark a key as secret; never record raw values.
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
    """Staged native candidate ready for import in a worker subprocess."""

    module_name: str
    function_name: str
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
    linker: str | None
    profile: str
    build_env: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable artifact provenance."""
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
        # Fail-closed: never invent a platform suffix.
        raise RuntimeError("sysconfig EXT_SUFFIX is unavailable; cannot stage candidate")
    return str(suffix)


def is_secret_env_key(key: str) -> bool:
    """Return True if *key* looks like a credential-bearing environment variable."""
    upper = key.upper()
    return any(marker in upper for marker in _SECRET_KEY_MARKERS)


def record_build_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return a secret-safe allowlisted snapshot of build environment keys.

    - Only keys in the allowlist (or safe non-secret ``CARGO_*`` / ``RUST*`` /
      ``PYO3_*`` build-control keys) are considered.
    - Any key matching secret markers is omitted entirely.
    - ``PATH`` is recorded only as the marker string ``(set)``.
    """
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
        # Still drop registry credential-style CARGO keys that might not match
        # markers if naming is unusual — require allowlist for CARGO_REGISTRIES_*.
        if key.startswith("CARGO_REGISTRIES_"):
            continue
        if not allowed:
            continue
        # Redact values that look like embedded secrets even on allowlisted keys.
        value = str(env[key])
        if is_secret_env_key(key):
            continue
        recorded[key] = value
    return recorded


def scrub_secrets_from_text(text: str, env: Mapping[str, str]) -> str:
    """Remove known secret env values from cargo/tool output before recording."""
    result = text
    for key, raw in env.items():
        if not is_secret_env_key(key):
            continue
        value = str(raw)
        if len(value) < 4:
            continue
        result = result.replace(value, "[REDACTED]")
    # Belt-and-suspenders: redact common token assignment forms in logs.
    result = re.sub(
        r"(?i)(token|password|secret|credential|api[_-]?key)\s*[=:]\s*\S+",
        r"\1=[REDACTED]",
        result,
    )
    return result


def _find_built_library(crate_dir: Path, crate_name: str) -> Path:
    """Locate the release cdylib produced by cargo for *crate_name*."""
    target_release = crate_dir / "target" / "release"
    # Common layouts: libNAME.so / libNAME.dylib / NAME.dll / libNAME.dll
    candidates = [
        target_release / f"lib{crate_name}.so",
        target_release / f"lib{crate_name}.dylib",
        target_release / f"{crate_name}.dll",
        target_release / f"lib{crate_name}.dll",
        target_release / f"{crate_name}.so",
        target_release / f"{crate_name}.dylib",
    ]
    # Also scan deps/ in case of unusual layouts.
    deps = target_release / "deps"
    if deps.is_dir():
        for p in sorted(deps.iterdir()):
            name = p.name
            if crate_name in name and p.suffix in {".so", ".dylib", ".dll"}:
                candidates.append(p)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"built cdylib for {crate_name!r} not found under {target_release}; "
        f"looked for: {[str(c) for c in candidates[:6]]}"
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
    """Fail-closed path + content hash check for the imported candidate.

    Returns ``None`` on success, otherwise an error message.
    """
    if module_file is None or module_file == "":
        return "imported candidate module has no __file__ (fail-closed)"
    try:
        loaded = Path(module_file).resolve()
        expected = Path(expected_path).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return f"failed to resolve candidate paths: {exc}"
    if loaded != expected:
        return (
            f"imported candidate path {str(loaded)!r} != expected {str(expected)!r} (fail-closed)"
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
    """Build the research cdylib once and stage it under *stage_dir*.

    Parameters:
        stage_dir: Run-specific directory that will hold the importable module.
        crate_dir: Optional override of the harness crate root (tests).
        run_command: Injectable subprocess runner (tests mock cargo).
        cargo_bin: Optional cargo executable path.
        env: Extra environment for the cargo process.
        profile: Build profile name recorded in provenance (default release).

    Raises:
        FileNotFoundError: cargo missing or artifact not produced.
        RuntimeError: cargo build failed or staging checks failed.
    """
    import time

    crate = Path(crate_dir) if crate_dir is not None else RUST_CANDIDATE_DIR
    crate = crate.resolve()
    stage = Path(stage_dir).resolve()
    stage.mkdir(parents=True, exist_ok=True)

    cargo_toml = crate / "Cargo.toml"
    cargo_lock = crate / "Cargo.lock"
    lib_rs = crate / "src" / "lib.rs"
    cargo_config = crate / ".cargo" / "config.toml"
    for required in (cargo_toml, cargo_lock, lib_rs, cargo_config):
        if not required.is_file():
            raise FileNotFoundError(f"candidate crate missing required file: {required}")

    cargo = cargo_bin or shutil.which("cargo")
    if cargo is None and run_command is None:
        raise FileNotFoundError("cargo not found on PATH; cannot build research candidate")

    build_env = os.environ.copy()
    if env:
        build_env.update(env)
    # Keep the build honest: no accidental BLAS injection via env features.
    build_env.setdefault("CARGO_TERM_COLOR", "never")
    # Point PyO3 at the same interpreter that will import the cdylib.
    build_env.setdefault("PYO3_PYTHON", sys.executable)

    argv = [cargo or "cargo", *CARGO_BUILD_ARGS]
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
    stdout_raw = completed.stdout or ""
    stderr_raw = completed.stderr or ""
    # Never persist secret env values that cargo/tools might echo.
    stdout = scrub_secrets_from_text(stdout_raw, build_env)[-8000:]
    stderr = scrub_secrets_from_text(stderr_raw, build_env)[-8000:]
    if getattr(completed, "returncode", 1) != 0:
        raise RuntimeError(
            f"cargo build failed (code={getattr(completed, 'returncode', None)}): "
            f"{stderr[-2000:] or stdout[-2000:]}"
        )

    # When tests inject run_command they may also pre-stage a fake .so; if the
    # real target is missing but stage already has the module, accept that path.
    ext = _ext_suffix()
    staged_name = f"{CANDIDATE_MODULE_NAME}{ext}"
    staged_path = stage / staged_name

    try:
        built = _find_built_library(crate, CANDIDATE_CRATE_NAME)
        shutil.copy2(built, staged_path)
    except FileNotFoundError:
        if not staged_path.is_file():
            raise
        # Mocked build path: artifact pre-placed by the test harness.

    if not staged_path.is_file():
        raise RuntimeError(f"staged candidate missing: {staged_path}")

    artifact_sha = _sha256_file(staged_path)
    rustc_verbose = _run_text(["rustc", "-vV"])
    cargo_version = _run_text(["cargo", "--version"])
    # Linker is best-effort from rustc -vV host line / env (allowlisted keys only).
    linker = build_env.get("RUSTC_LINKER")
    if linker is None and rustc_verbose:
        for line in rustc_verbose.splitlines():
            if line.lower().startswith("host:"):
                linker = f"default-for-{line.split(':', 1)[1].strip()}"
                break

    recorded_env = record_build_env(build_env)

    return CandidateArtifact(
        module_name=CANDIDATE_MODULE_NAME,
        function_name=CANDIDATE_FUNCTION_NAME,
        load_dir=str(stage),
        module_path=str(staged_path.resolve()),
        artifact_sha256=artifact_sha,
        build_wall_s=float(build_wall),
        cargo_stdout=stdout,
        cargo_stderr=stderr,
        crate_dir=str(crate),
        cargo_toml_sha256=_sha256_file(cargo_toml),
        cargo_lock_sha256=_sha256_file(cargo_lock),
        lib_rs_sha256=_sha256_file(lib_rs),
        cargo_config_sha256=_sha256_file(cargo_config),
        rustc_verbose=rustc_verbose,
        cargo_version=cargo_version,
        linker=linker,
        profile=profile,
        build_env=recorded_env,
    )


def ensure_importable(artifact: CandidateArtifact) -> None:
    """Import the staged module in-process and verify path + hash (fail-closed).

    Used by the parent after build and by tests. Timing workers re-check in
    their own process before measurement.
    """
    import importlib
    import sys as _sys

    load_dir = str(Path(artifact.load_dir).resolve())
    if load_dir not in _sys.path:
        _sys.path.insert(0, load_dir)
    # Drop a previously loaded copy so path checks see the staged file.
    _sys.modules.pop(artifact.module_name, None)
    mod = importlib.import_module(artifact.module_name)
    err = validate_loaded_module(
        getattr(mod, "__file__", None),
        expected_path=artifact.module_path,
        expected_sha256=artifact.artifact_sha256,
    )
    if err is not None:
        raise RuntimeError(err)
    if not hasattr(mod, artifact.function_name):
        raise RuntimeError(f"candidate module missing function {artifact.function_name!r}")


def candidate_source_hashes(crate_dir: Path | str | None = None) -> dict[str, str]:
    """Return SHA-256 hashes of committed crate source and build-config files."""
    crate = Path(crate_dir) if crate_dir is not None else RUST_CANDIDATE_DIR
    crate = crate.resolve()
    return {
        "Cargo.toml": _sha256_file(crate / "Cargo.toml"),
        "Cargo.lock": _sha256_file(crate / "Cargo.lock"),
        "src/lib.rs": _sha256_file(crate / "src" / "lib.rs"),
        ".cargo/config.toml": _sha256_file(crate / ".cargo" / "config.toml"),
    }


def python_executable() -> str:
    """Interpreter used for worker subprocesses (same as parent)."""
    return sys.executable
