"""Subprocess-isolated fallback/native measurement legs.

Each leg runs in a **fresh subprocess**. ``REXTIO_NATIVE_MODE`` is set
**before** importing generated wrappers. Native timing also sets
``REXTIO_DISABLE_BOUNDARY_FALLBACK=1`` so the measured path stays native.
Inputs are constructed outside timed regions. Warmups, repeated iterations,
and multiple samples are performed inside the worker.

Sample semantics
----------------
The worker measures **raw batch elapsed** wall time per sample (``t1 - t0``
around ``iterations`` calls). The parent derives **per-call wall latency** as
``batch_elapsed / iterations``. Summary statistics and speedup use per-call
latencies only; raw batch samples are preserved for auditability.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from benchmarks.compare import results_equivalent
from benchmarks.models import LegTiming
from benchmarks.scenarios import ScenarioSpec
from benchmarks.stats import summarize

Mode = Literal["fallback", "native"]


def normalize_batch_to_per_call(
    batch_samples_s: Sequence[float],
    iterations_per_sample: int,
) -> list[float]:
    """Derive per-call wall latencies from raw batch elapsed times.

    ``per_call[i] = batch_samples_s[i] / iterations_per_sample``.

    Raises:
        ValueError: if ``iterations_per_sample < 1``.
    """
    if iterations_per_sample < 1:
        raise ValueError(f"iterations_per_sample must be >= 1, got {iterations_per_sample!r}")
    denom = float(iterations_per_sample)
    return [float(x) / denom for x in batch_samples_s]


def validate_module_under_build_tree(
    module_file: str | None,
    build_python_dir: str | Path,
) -> str | None:
    """Fail-closed check: module ``__file__`` must resolve under *build_python_dir*.

    Returns:
        ``None`` if the path is valid; otherwise an error message string.
        Missing ``__file__`` and out-of-tree paths both fail (never soft-pass).
    """
    if module_file is None or module_file == "":
        return (
            "imported module has no __file__; refusing to time a module with "
            "unknown origin (fail-closed)"
        )
    try:
        loaded = Path(module_file).resolve()
        root = Path(build_python_dir).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return f"failed to resolve module/build paths: {exc}"
    try:
        loaded.relative_to(root)
    except ValueError:
        return (
            f"imported module file {str(loaded)!r} is outside build_python_dir "
            f"{str(root)!r}; refusing to time a non-generated module (fail-closed)"
        )
    return None


# Worker script: pure stdlib + numpy; receives a JSON config on stdin, prints
# a JSON result on stdout. Kept as a string so the suite is self-contained
# under benchmarks/** without requiring installation.
_WORKER_SOURCE = textwrap.dedent(
    r"""
    from __future__ import annotations

    import json
    import os
    import sys
    import time
    from pathlib import Path


    def _validate_module_under_build_tree(module_file, build_python_dir):
        if module_file is None or module_file == "":
            return (
                "imported module has no __file__; refusing to time a module with "
                "unknown origin (fail-closed)"
            )
        try:
            loaded = Path(module_file).resolve()
            root = Path(build_python_dir).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            return f"failed to resolve module/build paths: {exc}"
        try:
            loaded.relative_to(root)
        except ValueError:
            return (
                f"imported module file {str(loaded)!r} is outside build_python_dir "
                f"{str(root)!r}; refusing to time a non-generated module (fail-closed)"
            )
        return None


    def main() -> int:
        cfg = json.load(sys.stdin)
        mode = cfg["mode"]
        build_python_dir = cfg["build_python_dir"]
        module_name = cfg["module_name"]
        function_name = cfg["function_name"]
        scenario_id = cfg["scenario_id"]
        size = cfg["size"]
        seed = int(cfg["seed"])
        warmups = int(cfg["warmups"])
        iterations = int(cfg["iterations"])
        samples = int(cfg["samples"])
        disable_boundary = bool(cfg.get("disable_boundary_fallback", False))

        # Mode MUST be set before importing generated wrappers.
        os.environ["REXTIO_NATIVE_MODE"] = mode
        if disable_boundary:
            os.environ["REXTIO_DISABLE_BOUNDARY_FALLBACK"] = "1"
        else:
            os.environ.pop("REXTIO_DISABLE_BOUNDARY_FALLBACK", None)

        # Pin the *installed* rextio package before the build tree is on
        # sys.path. The generated tree ships a partial ``rextio/`` (runtime
        # helpers only). If that partial package is discovered first,
        # ``import rextio_numpy.types`` (via package __init__ / plugin) cannot
        # resolve ``rextio.config`` and the leg dies with ModuleNotFoundError.
        # Pre-importing matches how in-process certification works: full
        # rextio is already in sys.modules before the build dir is inserted.
        import rextio  # noqa: F401
        import rextio.runtime  # noqa: F401

        # Prefer the generated build tree for fixture packages + native .so.
        sys.path.insert(0, build_python_dir)

        import numpy as np

        def build_inputs():
            rng = np.random.default_rng(seed)
            n = int(size["n"])
            a = rng.standard_normal(n, dtype=np.float64)
            b = rng.standard_normal(n, dtype=np.float64)
            if scenario_id == "mixed_control_flow":
                return (a, b, int(size.get("loop_iters", 8)))
            return (a, b)

        # Construct inputs outside the timed region.
        args = build_inputs()

        import importlib

        mod = importlib.import_module(module_name)
        loaded_from = getattr(mod, "__file__", None)
        path_error = _validate_module_under_build_tree(loaded_from, build_python_dir)
        if path_error is not None:
            json.dump(
                {
                    "ok": False,
                    "error": path_error,
                    "loaded_from": loaded_from,
                    "build_python_dir": build_python_dir,
                },
                sys.stdout,
            )
            return 1

        fn = getattr(mod, function_name)

        # One untimed reference call for verification payload.
        # Order matters: numpy scalar results (e.g. bare np.dot → numpy.float64)
        # expose both ``tolist`` and ``ndim == 0``. Treat them as scalars so
        # native (plain float) and fallback (numpy scalar) share one payload
        # shape; do not route 0-d values through the array path.
        result = fn(*args)
        if isinstance(result, (bool, int, float)):
            result_payload = {"kind": "scalar", "data": float(result)}
        elif hasattr(result, "ndim") and int(getattr(result, "ndim")) == 0:
            result_payload = {"kind": "scalar", "data": float(result)}
        elif hasattr(result, "tolist"):
            # Convert multi-element arrays to lists for JSON.
            result_payload = {
                "kind": "array",
                "data": result.tolist(),
                "dtype": str(getattr(result, "dtype", "float64")),
            }
        else:
            result_payload = {"kind": "scalar", "data": float(result)}

        for _ in range(warmups):
            fn(*args)

        # Raw batch elapsed times (wall for `iterations` calls each sample).
        batch_sample_times = []
        for _ in range(samples):
            t0 = time.perf_counter()
            for _i in range(iterations):
                fn(*args)
            t1 = time.perf_counter()
            batch_sample_times.append(t1 - t0)

        out = {
            "ok": True,
            "mode": mode,
            "batch_samples_s": batch_sample_times,
            "result": result_payload,
            "loaded_from": loaded_from,
            "iterations": iterations,
            "warmups": warmups,
        }
        json.dump(out, sys.stdout)
        return 0


    if __name__ == "__main__":
        try:
            raise SystemExit(main())
        except Exception as exc:
            json.dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, sys.stdout)
            raise SystemExit(1)
    """
).lstrip()


@dataclass(frozen=True)
class SubprocessSpec:
    """Fully constructed subprocess invocation (command + env + stdin)."""

    argv: list[str]
    env: dict[str, str]
    stdin_payload: dict[str, Any]


def build_leg_subprocess_spec(
    *,
    python_executable: str,
    build_python_dir: Path | str,
    spec: ScenarioSpec,
    mode: Mode,
    warmups: int,
    iterations: int,
    samples: int,
    base_env: dict[str, str] | None = None,
) -> SubprocessSpec:
    """Construct argv/env/stdin for one measurement leg (no execution).

    Environment for the child includes ``REXTIO_NATIVE_MODE`` and, for native,
    ``REXTIO_DISABLE_BOUNDARY_FALLBACK=1``. The worker *also* re-sets these
    before import as a belt-and-suspenders guard.
    """
    env = dict(base_env if base_env is not None else os.environ)
    env["REXTIO_NATIVE_MODE"] = mode
    if mode == "native":
        env["REXTIO_DISABLE_BOUNDARY_FALLBACK"] = "1"
    else:
        env.pop("REXTIO_DISABLE_BOUNDARY_FALLBACK", None)

    module_name = spec.qualname.rsplit(".", 1)[0]
    function_name = spec.function_name
    payload = {
        "mode": mode,
        "build_python_dir": str(build_python_dir),
        "module_name": module_name,
        "function_name": function_name,
        "scenario_id": spec.id,
        "size": dict(spec.size),
        "seed": spec.seed,
        "warmups": warmups,
        "iterations": iterations,
        "samples": samples,
        "disable_boundary_fallback": mode == "native",
    }
    # ``python -c`` worker keeps the suite free of generated helper files.
    argv = [python_executable, "-c", _WORKER_SOURCE]
    return SubprocessSpec(argv=argv, env=env, stdin_payload=payload)


RunSubprocessFn = Callable[[SubprocessSpec], dict[str, Any]]


def _default_run_subprocess(spec: SubprocessSpec, *, timeout: float = 600.0) -> dict[str, Any]:
    """Execute a measurement subprocess and parse its JSON stdout."""
    try:
        completed = subprocess.run(
            spec.argv,
            input=json.dumps(spec.stdin_payload),
            capture_output=True,
            text=True,
            env=spec.env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": f"subprocess timed out: {exc}"}
    except OSError as exc:
        return {"ok": False, "error": f"subprocess OSError: {exc}"}

    stdout = completed.stdout or ""
    try:
        data = json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": (
                f"non-JSON worker stdout (exit {completed.returncode}): "
                f"{stdout[:500]!r} stderr={(completed.stderr or '')[:500]!r}"
            ),
        }
    if not isinstance(data, dict):
        return {"ok": False, "error": f"worker returned non-object: {type(data).__name__}"}
    # Fail-closed: any nonzero worker exit is a failed leg (even if JSON says ok).
    if completed.returncode != 0:
        data["ok"] = False
        data.setdefault(
            "error",
            f"worker exit {completed.returncode}: {(completed.stderr or '')[:500]}",
        )
    return data


# Worker emits ``str(ndarray.dtype)``; only these names are accepted when
# reconstructing array payloads (fail-closed — no arbitrary dtype constructors).
_SUPPORTED_RESULT_DTYPES: frozenset[str] = frozenset(
    {
        "float64",
        "float32",
        "int64",
        "int32",
        "bool",
    }
)


def parse_result_dtype(dtype_name: object) -> Any:
    """Map a validated dtype string to a NumPy dtype.

    Raises:
        ValueError: if *dtype_name* is missing or not in the allowlist.
    """
    import numpy as np

    if not isinstance(dtype_name, str) or not dtype_name:
        raise ValueError(f"array result payload missing or invalid dtype: {dtype_name!r}")
    if dtype_name not in _SUPPORTED_RESULT_DTYPES:
        raise ValueError(
            f"unsupported result dtype {dtype_name!r}; allowed: {sorted(_SUPPORTED_RESULT_DTYPES)}"
        )
    return np.dtype(dtype_name)


def reconstruct_result(payload: dict[str, Any]) -> Any:
    """Rebuild a Python value from the worker's JSON result payload.

    Array payloads must carry a supported ``dtype`` string; the reconstructed
    ndarray preserves that dtype (never forced to float64).
    """
    kind = payload.get("kind")
    data = payload.get("data")
    if kind == "array":
        import numpy as np

        dtype = parse_result_dtype(payload.get("dtype"))
        return np.asarray(data, dtype=dtype)
    if kind == "scalar":
        if data is None:
            raise ValueError("scalar result payload missing data")
        return float(data)
    raise ValueError(f"unknown result kind: {kind!r}")


def measure_leg(
    *,
    python_executable: str,
    build_python_dir: Path | str,
    spec: ScenarioSpec,
    mode: Mode,
    warmups: int,
    iterations: int,
    samples: int,
    run_subprocess: RunSubprocessFn | None = None,
) -> tuple[LegTiming | None, dict[str, Any]]:
    """Run one measurement leg; return (timing or None, raw worker dict).

    Raw worker ``batch_samples_s`` values are preserved on :class:`LegTiming`.
    Per-call wall latencies and the summary are derived as
    ``batch / iterations``.
    """
    runner = run_subprocess or _default_run_subprocess
    sub = build_leg_subprocess_spec(
        python_executable=python_executable,
        build_python_dir=build_python_dir,
        spec=spec,
        mode=mode,
        warmups=warmups,
        iterations=iterations,
        samples=samples,
    )
    raw = runner(sub)
    if not raw.get("ok"):
        return None, raw
    # Prefer explicit batch field; accept legacy samples_s only as raw batch.
    batch_list = [float(x) for x in (raw.get("batch_samples_s") or raw.get("samples_s") or [])]
    if not batch_list:
        raw = {**raw, "ok": False, "error": "worker returned no batch samples"}
        return None, raw
    try:
        per_call_list = normalize_batch_to_per_call(batch_list, iterations)
    except ValueError as exc:
        raw = {**raw, "ok": False, "error": str(exc)}
        return None, raw
    timing = LegTiming(
        mode=mode,
        batch_samples_s=batch_list,
        per_call_samples_s=per_call_list,
        summary=summarize(per_call_list),
        iterations_per_sample=iterations,
        warmups=warmups,
    )
    return timing, raw


def verify_leg_results(
    fallback_raw: dict[str, Any],
    native_raw: dict[str, Any],
    *,
    compare_kind: str,
) -> dict[str, Any]:
    """Compare fallback/native result payloads; return a verification record."""
    if not fallback_raw.get("ok") or not native_raw.get("ok"):
        return {
            "matched": False,
            "reason": (
                f"leg not ok: fallback_ok={fallback_raw.get('ok')!r} "
                f"native_ok={native_raw.get('ok')!r} "
                f"fallback_error={fallback_raw.get('error')!r} "
                f"native_error={native_raw.get('error')!r}"
            ),
        }
    try:
        fb = reconstruct_result(fallback_raw["result"])
        nt = reconstruct_result(native_raw["result"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"matched": False, "reason": f"result reconstruction failed: {exc}"}
    matched = results_equivalent(fb, nt, kind=compare_kind)
    return {
        "matched": bool(matched),
        "compare_kind": compare_kind,
        "reason": None if matched else "fallback and native results diverged",
    }
