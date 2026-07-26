"""CLI runner for the experimental boundary-allocation PoC harness.

Records logical allocation accounting and calibrated steady-state wall times.
Does **not** publish a speed claim. Result artifacts go only to a user-selected
output directory (never commit measured results).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from benchmarks.boundary_allocation_poc.candidate import (
    CandidateArtifact,
    build_candidate,
    ensure_importable,
)
from benchmarks.boundary_allocation_poc.protocol import (
    CONTIGUOUS_SIZES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MIN_BATCH_S,
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    DEFAULT_WARMUPS,
    HONESTY_CAVEATS,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    SMOKE_SAMPLES,
    SMOKE_WARMUPS,
    STRATEGIES,
    THREAD_ENV,
    StrategyId,
    logical_allocs_for,
    protocol_manifest,
)
from benchmarks.boundary_allocation_poc.semantics import (
    python_add,
    resolve_strategy_fn,
    run_semantic_suite,
)

BuildCandidateFn = Callable[..., CandidateArtifact]


def _apply_thread_env() -> dict[str, str]:
    for k, v in THREAD_ENV.items():
        os.environ[str(k)] = str(v)
    return {k: str(os.environ.get(k, "")) for k in THREAD_ENV}


def _calibrate_iterations(
    fn: Callable[..., Any],
    a: Any,
    b: Any,
    *,
    min_batch_s: float,
    max_iterations: int,
) -> int:
    """Grow iterations until one batch exceeds *min_batch_s* (capped)."""
    iters = 1
    while iters < max_iterations:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(a, b)
        elapsed = time.perf_counter() - t0
        if elapsed >= min_batch_s:
            return iters
        # Scale toward target with a small safety factor.
        if elapsed <= 0:
            iters = min(max_iterations, iters * 2)
            continue
        scale = max(2, int(min_batch_s / elapsed * 1.2))
        iters = min(max_iterations, max(iters + 1, iters * scale))
    return max_iterations


def _measure_strategy(
    fn: Callable[..., Any],
    a: Any,
    b: Any,
    *,
    warmups: int,
    samples: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        fn(a, b)
    batch_samples: list[float] = []
    for _ in range(samples):
        t0 = time.perf_counter()
        for _ in range(iterations):
            fn(a, b)
        batch_samples.append(time.perf_counter() - t0)
    per_call = [b / iterations for b in batch_samples]
    return {
        "warmups": warmups,
        "samples": samples,
        "iterations": iterations,
        "batch_samples_s": batch_samples,
        "per_call_samples_s": per_call,
        "summary": {
            "median_s": float(statistics.median(per_call)),
            "mean_s": float(statistics.fmean(per_call)),
            "min_s": float(min(per_call)),
            "max_s": float(max(per_call)),
            "count": len(per_call),
        },
    }


def run_timing(
    mod: Any,
    *,
    sizes: Sequence[int],
    strategies: Sequence[StrategyId],
    warmups: int,
    samples: int,
    seed: int,
    calibrate: bool,
    min_batch_s: float,
    max_iterations: int,
    fixed_iterations: int | None,
) -> list[dict[str, Any]]:
    """Measure calibrated steady-state wall times for contiguous equal-length cells."""
    import numpy as np

    rng = np.random.default_rng(seed)
    cells: list[dict[str, Any]] = []
    for n in sizes:
        a = rng.standard_normal(int(n), dtype=np.float64)
        b = rng.standard_normal(int(n), dtype=np.float64)
        # Correctness gate before timing.
        ref = python_add(a, b)
        cell: dict[str, Any] = {
            "kind": "contiguous_equal",
            "n": int(n),
            "headline": True,
            "logical_accounting": {
                sid: logical_allocs_for(sid, int(n)) for sid in strategies
            },
            "strategies": {},
        }
        for sid in strategies:
            fn = resolve_strategy_fn(mod, sid)
            out = fn(a, b)
            if not np.array_equal(out, ref, equal_nan=True):
                raise RuntimeError(f"value mismatch for {sid} at n={n}")
            if calibrate and fixed_iterations is None:
                iters = _calibrate_iterations(
                    fn,
                    a,
                    b,
                    min_batch_s=min_batch_s,
                    max_iterations=max_iterations,
                )
            else:
                iters = int(fixed_iterations or 1)
            timing = _measure_strategy(
                fn, a, b, warmups=warmups, samples=samples, iterations=iters
            )
            cell["strategies"][sid] = {
                "timing": timing,
                "logical": logical_allocs_for(sid, int(n)),
            }
        cells.append(cell)
    return cells


def run_harness(
    *,
    output_dir: Path | str,
    smoke: bool = False,
    sizes: Sequence[int] | None = None,
    strategies: Sequence[StrategyId] | None = None,
    warmups: int | None = None,
    samples: int | None = None,
    seed: int = DEFAULT_SEED,
    calibrate: bool = True,
    min_batch_s: float = DEFAULT_MIN_BATCH_S,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    fixed_iterations: int | None = None,
    run_id: str | None = None,
    skip_semantics: bool = False,
    skip_timing: bool = False,
    build_fn: BuildCandidateFn | None = None,
    crate_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build candidate, run semantics + optional timing, write reports."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rid = run_id or uuid.uuid4().hex[:12]
    stage = out_dir / "artifacts" / rid
    stage.mkdir(parents=True, exist_ok=True)

    if smoke:
        warmups = SMOKE_WARMUPS if warmups is None else warmups
        samples = SMOKE_SAMPLES if samples is None else samples
        if sizes is None:
            sizes = (CONTIGUOUS_SIZES[0],)  # 1_000 only for smoke
        if fixed_iterations is None and not calibrate:
            fixed_iterations = 1
    else:
        warmups = DEFAULT_WARMUPS if warmups is None else warmups
        samples = DEFAULT_SAMPLES if samples is None else samples
        if sizes is None:
            sizes = CONTIGUOUS_SIZES

    strategies = tuple(strategies or STRATEGIES)
    thread_env = _apply_thread_env()

    builder = build_fn or build_candidate
    artifact = builder(stage, crate_dir=crate_dir)
    mod = ensure_importable(artifact)

    semantic_records: list[dict[str, Any]] = []
    if not skip_semantics:
        semantic_records = run_semantic_suite(mod)

    timing_cells: list[dict[str, Any]] = []
    if not skip_timing:
        timing_cells = run_timing(
            mod,
            sizes=sizes,
            strategies=strategies,
            warmups=warmups,
            samples=samples,
            seed=seed,
            calibrate=calibrate and fixed_iterations is None,
            min_batch_s=min_batch_s,
            max_iterations=max_iterations,
            fixed_iterations=fixed_iterations,
        )

    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": rid,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "speedup_claim": None,
        "honesty_caveats": list(HONESTY_CAVEATS),
        "protocol": protocol_manifest(),
        "settings": {
            "smoke": smoke,
            "sizes": list(sizes),
            "strategies": list(strategies),
            "warmups": warmups,
            "samples": samples,
            "seed": seed,
            "calibrate": calibrate and fixed_iterations is None,
            "min_batch_s": min_batch_s,
            "max_iterations": max_iterations,
            "fixed_iterations": fixed_iterations,
            "skip_semantics": skip_semantics,
            "skip_timing": skip_timing,
            "thread_env": thread_env,
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "candidate": artifact.to_dict(),
        "semantics": semantic_records,
        "timing_cells": timing_cells,
        "notes": [
            "Measured times are local wall samples only; do not commit as product evidence.",
            "Logical bytes use explicit formulas in protocol.logical_alloc_formulas.",
        ],
    }

    json_path = out_dir / f"boundary-allocation-poc-{rid}.json"
    md_path = out_dir / f"boundary-allocation-poc-{rid}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    report["paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Boundary allocation PoC report (`{report['run_id']}`)",
        "",
        f"- Protocol: `{report['protocol_id']}`",
        f"- Schema: `{report['schema_version']}`",
        "- Performance claim: **false** (no speedup claim)",
        "",
        "## Honesty",
        "",
    ]
    for c in report.get("honesty_caveats") or []:
        lines.append(f"- {c}")
    lines.extend(["", "## Logical allocation formulas", ""])
    formulas = (report.get("protocol") or {}).get("logical_alloc_formulas") or {}
    for sid, formula in formulas.items():
        lines.append(f"- **{sid}**: {formula}")
    lines.extend(["", "## Semantics", ""])
    sem = report.get("semantics") or []
    ok = sum(1 for r in sem if r.get("ok"))
    lines.append(f"- Records: {len(sem)} (ok={ok})")
    lines.extend(["", "## Timing cells (local only; not a product claim)", ""])
    for cell in report.get("timing_cells") or []:
        n = cell.get("n")
        lines.append(f"### N = {n}")
        for sid, body in (cell.get("strategies") or {}).items():
            med = ((body.get("timing") or {}).get("summary") or {}).get("median_s")
            logical = body.get("logical") or {}
            lines.append(
                f"- `{sid}`: median_per_call={med!r}s; "
                f"logical_allocs={logical.get('logical_n_sized_allocs')}; "
                f"logical_bytes={logical.get('logical_bytes')}"
            )
        lines.append("")
    lines.append(
        "Do not treat ratios between strategies as a published speedup. "
        "Allocator internals, SIMD, and zero-initialization may differ from "
        "logical accounting."
    )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the PoC harness."""
    p = argparse.ArgumentParser(
        description=(
            "Experimental F64 rank-1 NumPy boundary-allocation PoC. "
            "Research-only; no product speed claim."
        )
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Writable directory for reports (gitignored / temp; never commit results).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Bounded smoke: small N, few samples; never a performance conclusion.",
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=None,
        help=f"Override contiguous sizes (default: {list(CONTIGUOUS_SIZES)}).",
    )
    p.add_argument("--warmups", type=int, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable iteration calibration (use --iterations).",
    )
    p.add_argument("--iterations", type=int, default=None, help="Fixed iterations per sample.")
    p.add_argument("--min-batch-s", type=float, default=DEFAULT_MIN_BATCH_S)
    p.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    p.add_argument("--skip-semantics", action="store_true")
    p.add_argument("--skip-timing", action="store_true")
    p.add_argument("--run-id", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: run the harness and print a small JSON status object."""
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    report = run_harness(
        output_dir=args.output_dir,
        smoke=bool(args.smoke),
        sizes=tuple(args.sizes) if args.sizes else None,
        warmups=args.warmups,
        samples=args.samples,
        seed=args.seed,
        calibrate=not args.no_calibrate,
        min_batch_s=args.min_batch_s,
        max_iterations=args.max_iterations,
        fixed_iterations=args.iterations,
        run_id=args.run_id,
        skip_semantics=bool(args.skip_semantics),
        skip_timing=bool(args.skip_timing),
    )
    print(json.dumps({"ok": True, "paths": report.get("paths"), "run_id": report["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
