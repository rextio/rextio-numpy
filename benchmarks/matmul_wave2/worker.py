"""Subprocess worker for one timing leg of one shape cell.

Thread environment variables are set **before** importing NumPy. Each leg
loads identical ``.npy`` inputs, validates correctness against a saved
reference, and records positive finite per-call wall samples.

Process-isolation evidence is always recorded: PID, per-process UUID nonce,
whether NumPy was pre-imported (fail-closed if true), applied thread env, and
NumPy module identity.

Invoked as::

    python -m benchmarks.matmul_wave2.worker <config.json>

or with config on stdin when argv has no path. Prints one JSON object to stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any


def _apply_thread_env(thread_env: dict[str, str]) -> dict[str, str]:
    """Apply *thread_env* and return the post-application os.environ snapshot."""
    for key, value in thread_env.items():
        os.environ[str(key)] = str(value)
    return {str(k): str(os.environ.get(str(k), "")) for k in thread_env}


def _validate_result(
    result: Any,
    *,
    reference: Any,
    expected_shape: tuple[int, ...],
    rtol: float,
    atol: float,
    equal_nan: bool,
    np: Any,
) -> str | None:
    if not hasattr(result, "dtype") or not hasattr(result, "shape"):
        return f"result is not an array-like with dtype/shape: {type(result)!r}"
    if str(result.dtype) != "float64" and result.dtype != np.float64:
        return f"result dtype is {result.dtype!r}, expected float64"
    if tuple(int(x) for x in result.shape) != tuple(int(x) for x in expected_shape):
        return f"result shape {tuple(result.shape)!r} != expected {expected_shape!r}"
    if not bool(np.isfinite(result).all()):
        return "result contains non-finite values"
    if not bool(np.isfinite(reference).all()):
        return "reference contains non-finite values"
    if not bool(np.allclose(result, reference, rtol=rtol, atol=atol, equal_nan=equal_nan)):
        max_abs = float(np.max(np.abs(result.astype(np.float64) - reference)))
        return f"allclose failed (max_abs_diff={max_abs})"
    return None


def _resolve_callable(
    leg: str,
    *,
    np: Any,
    candidate_load_dir: str | None,
    candidate_module_name: str | None,
    candidate_function_name: str | None,
    expected_module_path: str | None,
    expected_sha256: str | None,
) -> tuple[Any, dict[str, Any]]:
    meta: dict[str, Any] = {"leg": leg}
    if leg == "candidate":
        if not candidate_load_dir or not candidate_module_name or not candidate_function_name:
            raise RuntimeError("candidate leg missing load_dir/module/function in config")
        if not expected_module_path or not expected_sha256:
            raise RuntimeError("candidate leg missing expected path/sha256 in config")
        load_dir = str(Path(candidate_load_dir).resolve())
        if load_dir not in sys.path:
            sys.path.insert(0, load_dir)
        sys.modules.pop(candidate_module_name, None)
        import hashlib
        import importlib

        mod = importlib.import_module(candidate_module_name)
        module_file = getattr(mod, "__file__", None)
        meta["module_file"] = module_file
        if module_file is None:
            raise RuntimeError("candidate module has no __file__ (fail-closed)")
        loaded = Path(module_file).resolve()
        expected = Path(expected_module_path).resolve()
        if loaded != expected:
            raise RuntimeError(f"candidate path mismatch: {str(loaded)!r} != {str(expected)!r}")
        h = hashlib.sha256()
        with loaded.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        actual_sha = h.hexdigest()
        meta["artifact_sha256"] = actual_sha
        meta["expected_module_path"] = str(expected)
        meta["expected_sha256"] = expected_sha256
        if actual_sha != expected_sha256:
            raise RuntimeError(f"candidate SHA-256 mismatch: {actual_sha} != {expected_sha256}")
        fn = getattr(mod, candidate_function_name)
        return fn, meta

    if leg == "dot":
        return np.dot, meta
    if leg == "matmul":
        return np.matmul, meta
    if leg == "matmul_op":

        def matmul_op(a: Any, b: Any) -> Any:
            return a @ b

        return matmul_op, meta
    raise RuntimeError(f"unknown timing leg: {leg!r}")


def _process_base() -> dict[str, Any]:
    """Capture isolation fields available before thread env / NumPy import."""
    return {
        "pid": int(os.getpid()),
        "nonce": str(uuid.uuid4()),
        "numpy_preimported": "numpy" in sys.modules,
    }


def run_worker(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute one timing leg; return a JSON-serializable result dict."""
    process = _process_base()
    thread_env = dict(cfg.get("thread_env") or {})

    # Fail closed if NumPy was already imported (thread env would be too late).
    if process["numpy_preimported"]:
        return {
            "ok": False,
            "error": (
                "numpy already present in sys.modules before thread env application "
                "(fail-closed process isolation)"
            ),
            "leg": str(cfg.get("leg")),
            "process": process,
            "thread_env_requested": thread_env,
        }

    # Thread env BEFORE NumPy import.
    thread_env_applied = _apply_thread_env(thread_env)
    process["thread_env_applied"] = thread_env_applied

    import numpy as np

    process["numpy_module"] = "numpy"
    process["numpy_version"] = str(getattr(np, "__version__", None))
    process["numpy_file"] = str(getattr(np, "__file__", None))
    process["numpy_preimported"] = False

    leg = str(cfg["leg"])
    a_path = Path(cfg["a_path"])
    b_path = Path(cfg["b_path"])
    ref_path = Path(cfg["ref_path"])
    warmups = int(cfg["warmups"])
    samples = int(cfg["samples"])
    iterations = int(cfg["iterations"])
    expected_shape = tuple(int(x) for x in cfg["expected_shape"])
    rtol = float(cfg["rtol"])
    atol = float(cfg["atol"])
    equal_nan = bool(cfg["equal_nan"])

    if warmups < 0 or samples < 1 or iterations < 1:
        return {
            "ok": False,
            "error": f"invalid counts warmups={warmups} samples={samples} iterations={iterations}",
            "leg": leg,
            "process": process,
            "thread_env": thread_env_applied,
        }

    a = np.load(str(a_path))
    b = np.load(str(b_path))
    reference = np.load(str(ref_path))

    try:
        fn, meta = _resolve_callable(
            leg,
            np=np,
            candidate_load_dir=cfg.get("candidate_load_dir"),
            candidate_module_name=cfg.get("candidate_module_name"),
            candidate_function_name=cfg.get("candidate_function_name"),
            expected_module_path=cfg.get("expected_module_path"),
            expected_sha256=cfg.get("expected_sha256"),
        )
    except Exception as exc:  # noqa: BLE001 - surface to parent as cell failure
        return {
            "ok": False,
            "error": f"resolve callable failed: {exc}",
            "traceback": traceback.format_exc(),
            "leg": leg,
            "process": process,
            "thread_env": thread_env_applied,
        }

    # Correctness gate before timing.
    try:
        probe = fn(a, b)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"correctness call crashed: {exc}",
            "traceback": traceback.format_exc(),
            "meta": meta,
            "leg": leg,
            "process": process,
            "thread_env": thread_env_applied,
        }
    cerr = _validate_result(
        probe,
        reference=reference,
        expected_shape=expected_shape,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        np=np,
    )
    if cerr is not None:
        return {
            "ok": False,
            "error": f"correctness failed: {cerr}",
            "meta": meta,
            "leg": leg,
            "process": process,
            "thread_env": thread_env_applied,
        }

    # Unreported warmups.
    for _ in range(warmups):
        fn(a, b)

    batch_samples_s: list[float] = []
    per_call_samples_s: list[float] = []
    for _ in range(samples):
        t0 = time.perf_counter()
        for _ in range(iterations):
            fn(a, b)
        elapsed = time.perf_counter() - t0
        if not (elapsed > 0.0) or elapsed != elapsed or elapsed == float("inf"):
            return {
                "ok": False,
                "error": f"non-positive or non-finite batch elapsed: {elapsed!r}",
                "meta": meta,
                "leg": leg,
                "process": process,
                "thread_env": thread_env_applied,
            }
        batch_samples_s.append(float(elapsed))
        per_call = float(elapsed) / float(iterations)
        if not (per_call > 0.0):
            return {
                "ok": False,
                "error": f"non-positive per-call sample: {per_call!r}",
                "meta": meta,
                "leg": leg,
                "process": process,
                "thread_env": thread_env_applied,
            }
        per_call_samples_s.append(per_call)

    return {
        "ok": True,
        "leg": leg,
        "batch_samples_s": batch_samples_s,
        "per_call_samples_s": per_call_samples_s,
        "warmups": warmups,
        "samples": samples,
        "iterations": iterations,
        "thread_env": thread_env_applied,
        "thread_env_requested": thread_env,
        "numpy_version": process["numpy_version"],
        "meta": meta,
        "process": process,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry: load config, run worker, print JSON."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args:
            cfg = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        else:
            cfg = json.load(sys.stdin)
        result = run_worker(cfg)
    except Exception as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "error": f"worker crash: {exc}",
            "traceback": traceback.format_exc(),
            "process": {
                "pid": int(os.getpid()),
                "nonce": str(uuid.uuid4()),
                "numpy_preimported": "numpy" in sys.modules,
            },
        }
    json.dump(result, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
