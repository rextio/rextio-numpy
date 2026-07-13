"""JSON + Markdown report writers and artifact recompute validators."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from benchmarks.matmul_wave2.protocol import (
    BASE_SEED,
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_RESAMPLES,
    CI_LOWER_THRESHOLD,
    DISPATCHABILITY_REASON,
    DISPATCHABILITY_STATUS,
    EVIDENCE_SAMPLES,
    EVIDENCE_WARMUPS,
    EXPECTED_CELL_IDS,
    GEOMETRIC_MEAN_THRESHOLD,
    LEG_ORDER,
    PERCENTILE_METHOD,
    PRODUCT_VERDICT,
    PRODUCT_VERDICT_DETAIL,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    SHAPE_CELLS,
    SPELLINGS,
    THREAD_ENV,
    TIMING_LEGS,
    bootstrap_seed,
    cell_input_seed,
    report_cell_ids_match_frozen,
)
from benchmarks.matmul_wave2.stats import (
    bootstrap_median_ratio,
    cell_conservative_aggregate,
    geometric_mean,
    performance_gate,
    validate_positive_finite_samples,
)


def write_json_report(path: Path | str, report: dict[str, Any]) -> None:
    """Write the machine-readable research report."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_markdown_report(path: Path | str, report: dict[str, Any]) -> None:
    """Write a concise human-readable Markdown summary."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    evidence = bool(report.get("settings", {}).get("evidence", False))
    evidence_status = report.get("evidence_status")
    if evidence and evidence_status == "INVALID":
        mode = "EVIDENCE INVALID"
    elif evidence:
        mode = "EVIDENCE"
    else:
        mode = "SMOKE (not performance evidence)"
    verdict = report.get("verdicts", {})
    lines = [
        "# Wave 2 Matmul Research Report",
        "",
        f"- Protocol: `{report.get('protocol_id', PROTOCOL_ID)}`",
        f"- Schema: `{report.get('schema_version', SCHEMA_VERSION)}`",
        f"- Run id: `{report.get('run_id', '')}`",
        f"- Mode: **{mode}**",
        f"- UTC: `{report.get('provenance', {}).get('timestamp_utc', '')}`",
        "",
        "## Verdicts",
        "",
        f"- Dispatchability: **{verdict.get('dispatchability_status', DISPATCHABILITY_STATUS)}**",
        f"  - {verdict.get('dispatchability_reason', DISPATCHABILITY_REASON)}",
        f"- Performance gate (research): **{'PASS' if verdict.get('performance_passed') else 'FAIL/N-A'}**",
        f"  - {verdict.get('performance_reason') or 'n/a'}",
        f"- Product decision: **{verdict.get('product_verdict', PRODUCT_VERDICT)}** "
        f"({verdict.get('product_verdict_detail', PRODUCT_VERDICT_DETAIL)})",
        "",
        "## Aggregate",
        "",
    ]
    if evidence_status:
        lines.insert(8, f"- Evidence status: **{evidence_status}**")
    agg = report.get("aggregate", {})
    lines.append(f"- Geometric mean (conservative points): `{agg.get('geometric_mean')}`")
    lines.append(f"- Min conservative CI lower: `{agg.get('min_ci_lower')}`")
    lines.append(
        f"- Thresholds: CI lower > {CI_LOWER_THRESHOLD}, geom mean >= {GEOMETRIC_MEAN_THRESHOLD}"
    )
    inv = report.get("invalid_reasons") or []
    if inv:
        lines.append("")
        lines.append("## Invalid reasons")
        lines.append("")
        for reason in inv:
            lines.append(f"- {reason}")
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    lines.append("| cell_id | status | cons. point | cons. CI lo | notes |")
    lines.append("|---------|--------|-------------|-------------|-------|")
    for cell in report.get("cells", []):
        lines.append(
            f"| {cell.get('cell_id')} | {cell.get('status')} | "
            f"{cell.get('conservative_point')} | {cell.get('conservative_ci_lower')} | "
            f"{(cell.get('reason') or '')[:60]} |"
        )
    lines.append("")
    lines.append("## Honesty")
    lines.append("")
    honesty = report.get("honesty", {})
    for k, v in honesty.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)


def recompute_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute bootstraps/aggregates/gates from raw samples using frozen constants.

    Does **not** trust report-stored bootstrap seed, resample count, alpha,
    percentile method, or cell ids. Seeds are derived via
    :func:`bootstrap_seed` for the frozen cell index/spelling; resamples and
    alpha are the protocol constants. Stored fields must match exactly or the
    recompute reports a mismatch.
    """
    mismatches: list[str] = []
    recomputed_cells: list[dict[str, Any]] = []
    points: list[float] = []
    lowers: list[float] = []

    cells = report.get("cells", [])
    if not isinstance(cells, list):
        return {
            "matches": False,
            "mismatches": ["cells is not a list"],
            "recomputed_cells": [],
            "performance_gate": {"passed": False, "reason": "cells is not a list"},
            "geometric_mean": None,
            "matrix_ok": False,
        }

    cell_ids_seen = [str(c.get("cell_id")) for c in cells]
    matrix_ok = report_cell_ids_match_frozen(cell_ids_seen)
    if not matrix_ok:
        mismatches.append(
            f"cell ids do not match frozen matrix: got {cell_ids_seen!r}, "
            f"expected {list(EXPECTED_CELL_IDS)!r}"
        )

    # Walk in frozen matrix order; do not trust report order alone for seeds.
    by_id = {str(c.get("cell_id")): c for c in cells}
    for expected in SHAPE_CELLS:
        cell = by_id.get(expected.cell_id)
        if cell is None:
            mismatches.append(f"missing cell {expected.cell_id}")
            recomputed_cells.append(
                {"cell_id": expected.cell_id, "status": "missing", "skipped": True}
            )
            continue
        if cell.get("status") != "ok":
            recomputed_cells.append(
                {
                    "cell_id": expected.cell_id,
                    "status": cell.get("status"),
                    "skipped": True,
                }
            )
            continue
        # Prefer frozen cell_index for seed derivation.
        cell_index = int(expected.cell_index)
        if "cell_index" in cell and int(cell["cell_index"]) != cell_index:
            mismatches.append(
                f"{expected.cell_id}.cell_index stored={cell['cell_index']!r} expected={cell_index}"
            )
        try:
            cand = cell["legs"]["candidate"]["per_call_samples_s"]
        except (KeyError, TypeError) as exc:
            mismatches.append(f"{expected.cell_id}: missing candidate samples ({exc})")
            continue
        spelling_bs: dict[str, Any] = {}
        for spelling in SPELLINGS:
            frozen_seed = bootstrap_seed(cell_index, spelling)
            try:
                stored = cell["spellings"][spelling]["bootstrap"]
                samples = cell["legs"][spelling]["per_call_samples_s"]
            except (KeyError, TypeError) as exc:
                mismatches.append(f"{expected.cell_id}.{spelling}: missing data ({exc})")
                continue
            # Stored bootstrap params must exactly match frozen protocol.
            if int(stored.get("seed", -1)) != frozen_seed:
                mismatches.append(
                    f"{expected.cell_id}.{spelling}.seed stored={stored.get('seed')!r} "
                    f"frozen={frozen_seed}"
                )
            if int(stored.get("n_resamples", -1)) != BOOTSTRAP_RESAMPLES:
                mismatches.append(
                    f"{expected.cell_id}.{spelling}.n_resamples stored="
                    f"{stored.get('n_resamples')!r} frozen={BOOTSTRAP_RESAMPLES}"
                )
            if float(stored.get("alpha", -1.0)) != float(BOOTSTRAP_ALPHA):
                mismatches.append(
                    f"{expected.cell_id}.{spelling}.alpha stored={stored.get('alpha')!r} "
                    f"frozen={BOOTSTRAP_ALPHA}"
                )
            if stored.get("percentile_method") != PERCENTILE_METHOD:
                mismatches.append(
                    f"{expected.cell_id}.{spelling}.percentile_method stored="
                    f"{stored.get('percentile_method')!r} frozen={PERCENTILE_METHOD!r}"
                )
            # Always recompute with frozen seed/counts/alpha (never trust stored).
            boot = bootstrap_median_ratio(
                samples,
                cand,
                seed=frozen_seed,
                n_resamples=BOOTSTRAP_RESAMPLES,
                alpha=BOOTSTRAP_ALPHA,
            )
            for key in ("point_estimate", "ci_lower", "ci_upper"):
                a = float(getattr(boot, key))
                b = float(stored[key])
                if not _close(a, b):
                    mismatches.append(
                        f"{expected.cell_id}.{spelling}.{key}: recomputed={a} stored={b}"
                    )
            spelling_bs[spelling] = boot
        if len(spelling_bs) != 3:
            continue
        agg = cell_conservative_aggregate(spelling_bs)
        if not _close(agg.conservative_point, float(cell["conservative_point"])):
            mismatches.append(
                f"{expected.cell_id}.conservative_point: "
                f"{agg.conservative_point} vs {cell['conservative_point']}"
            )
        if not _close(agg.conservative_ci_lower, float(cell["conservative_ci_lower"])):
            mismatches.append(
                f"{expected.cell_id}.conservative_ci_lower: "
                f"{agg.conservative_ci_lower} vs {cell['conservative_ci_lower']}"
            )
        points.append(agg.conservative_point)
        lowers.append(agg.conservative_ci_lower)
        recomputed_cells.append(
            {
                "cell_id": expected.cell_id,
                "status": "ok",
                "conservative_point": agg.conservative_point,
                "conservative_ci_lower": agg.conservative_ci_lower,
            }
        )

    n_ok = len(points)
    gmean = geometric_mean(points) if n_ok == 27 and matrix_ok else None
    gate = (
        performance_gate(lowers, points, cell_ids=list(EXPECTED_CELL_IDS))
        if n_ok == 27 and matrix_ok
        else {
            "passed": False,
            "reason": (
                f"only {n_ok}/27 ok cells"
                if matrix_ok
                else "cell ids do not match frozen 27-cell matrix"
            ),
            "geometric_mean": gmean,
            "matrix_ok": matrix_ok,
        }
    )

    # Compare every stored derived performance field / research verdict.
    stored_gmean = report.get("aggregate", {}).get("geometric_mean")
    if gmean is not None and stored_gmean is not None:
        if not _close(float(gmean), float(stored_gmean)):
            mismatches.append(f"geometric_mean: recomputed={gmean} stored={stored_gmean}")
    stored_min = report.get("aggregate", {}).get("min_ci_lower")
    if gate.get("min_ci_lower") is not None and stored_min is not None:
        if not _close(float(gate["min_ci_lower"]), float(stored_min)):  # type: ignore[arg-type]
            mismatches.append(
                f"min_ci_lower: recomputed={gate['min_ci_lower']} stored={stored_min}"
            )
    stored_gate = (report.get("verdicts") or {}).get("performance_gate") or report.get(
        "aggregate", {}
    ).get("performance_gate")
    if isinstance(stored_gate, dict) and "passed" in stored_gate:
        if bool(stored_gate.get("passed")) != bool(gate.get("passed")):
            mismatches.append(
                f"performance_gate.passed: recomputed={gate.get('passed')} "
                f"stored={stored_gate.get('passed')}"
            )
    stored_passed = (report.get("verdicts") or {}).get("performance_passed")
    if stored_passed is not None and matrix_ok and n_ok == 27:
        # Only compare when evidence path could have concluded; smoke always false.
        if report.get("settings", {}).get("evidence") is True:
            if bool(stored_passed) != bool(gate.get("passed")):
                # Evidence may force false due to provenance/schema invalidity.
                # If report is INVALID, stored performance_passed must be false.
                if report.get("evidence_status") == "INVALID":
                    if stored_passed is not False:
                        mismatches.append(
                            "INVALID evidence report must have performance_passed=false"
                        )
                elif bool(stored_passed) != bool(gate.get("passed")):
                    mismatches.append(
                        f"performance_passed: recomputed={gate.get('passed')} "
                        f"stored={stored_passed}"
                    )

    # Product verdict must always be NO-GO; dispatchability failed.
    product = report.get("verdicts", {}).get("product_verdict")
    if product != PRODUCT_VERDICT:
        mismatches.append(f"product_verdict stored as {product!r}, expected {PRODUCT_VERDICT!r}")
    disp = report.get("verdicts", {}).get("dispatchability_status")
    if disp != DISPATCHABILITY_STATUS:
        mismatches.append(
            f"dispatchability_status stored as {disp!r}, expected {DISPATCHABILITY_STATUS!r}"
        )

    return {
        "matches": len(mismatches) == 0,
        "mismatches": mismatches,
        "recomputed_cells": recomputed_cells,
        "performance_gate": gate,
        "geometric_mean": gmean,
        "matrix_ok": matrix_ok,
    }


def _validate_leg_process(
    leg_name: str,
    leg: dict[str, Any],
    *,
    cell_id: str,
    expected_thread_env: dict[str, str],
    candidate_path: str | None,
    candidate_sha: str | None,
) -> list[str]:
    """Validate process-isolation evidence for one leg."""
    errors: list[str] = []
    process = leg.get("process") or {}
    if process.get("pid") is None:
        errors.append(f"{cell_id}.{leg_name}: missing process.pid")
    nonce = process.get("nonce")
    if not nonce or not isinstance(nonce, str):
        errors.append(f"{cell_id}.{leg_name}: missing process.nonce")
    if process.get("numpy_preimported") is not False:
        errors.append(
            f"{cell_id}.{leg_name}: numpy_preimported must be false "
            f"(got {process.get('numpy_preimported')!r})"
        )
    applied = process.get("thread_env_applied") or leg.get("thread_env") or {}
    for key, expected in expected_thread_env.items():
        if str(applied.get(key)) != str(expected):
            errors.append(
                f"{cell_id}.{leg_name}: thread env {key}={applied.get(key)!r} "
                f"!= expected {expected!r}"
            )
    if not process.get("numpy_version") and not leg.get("numpy_version"):
        errors.append(f"{cell_id}.{leg_name}: missing numpy version evidence")
    if leg_name == "candidate":
        meta = leg.get("meta") or {}
        module_file = meta.get("module_file")
        # module_file is required evidence, not optional-when-present.
        if not module_file:
            errors.append(f"{cell_id}.candidate: missing module_file in meta")
        elif candidate_path:
            try:
                if Path(str(module_file)).resolve() != Path(candidate_path).resolve():
                    errors.append(
                        f"{cell_id}.candidate: module_file mismatch "
                        f"{module_file!r} vs {candidate_path!r}"
                    )
            except (OSError, ValueError):
                errors.append(f"{cell_id}.candidate: cannot resolve module_file")
        if not meta.get("artifact_sha256"):
            errors.append(f"{cell_id}.candidate: missing artifact_sha256 in meta")
        elif candidate_sha and meta.get("artifact_sha256") != candidate_sha:
            errors.append(
                f"{cell_id}.candidate: artifact_sha256 mismatch "
                f"{meta.get('artifact_sha256')!r} vs {candidate_sha!r}"
            )
    return errors


def validate_process_isolation(report: dict[str, Any]) -> list[str]:
    """Validate 27×4 process isolation evidence for a full evidence report."""
    errors: list[str] = []
    cells = report.get("cells")
    if not isinstance(cells, list) or not report_cell_ids_match_frozen(
        [str(c.get("cell_id")) for c in cells]
    ):
        return ["process isolation requires exact frozen 27-cell matrix"]

    candidate = report.get("candidate") or {}
    cand_path = candidate.get("module_path")
    cand_sha = candidate.get("artifact_sha256")
    nonces: list[str] = []
    leg_count = 0
    for cell in cells:
        cell_id = str(cell.get("cell_id"))
        legs = cell.get("legs") or {}
        for leg_name in LEG_ORDER:
            if leg_name not in legs:
                errors.append(f"{cell_id}: missing leg {leg_name}")
                continue
            leg_count += 1
            leg = legs[leg_name]
            process = leg.get("process") or {}
            nonce = process.get("nonce")
            if isinstance(nonce, str) and nonce:
                nonces.append(nonce)
            errors.extend(
                _validate_leg_process(
                    leg_name,
                    leg,
                    cell_id=cell_id,
                    expected_thread_env=dict(THREAD_ENV),
                    candidate_path=cand_path,
                    candidate_sha=cand_sha,
                )
            )
    if leg_count != 108:
        errors.append(f"expected 108 leg attempts (27×4), got {leg_count}")
    if len(nonces) == leg_count and leg_count > 0:
        if len(set(nonces)) != len(nonces):
            errors.append(
                f"process nonces must be distinct across legs; "
                f"unique={len(set(nonces))} total={len(nonces)}"
            )
    return errors


def validate_report_schema(report: dict[str, Any]) -> list[str]:
    """Return schema problems (empty if ok).

    Smoke/debug subsets are allowed when ``evidence`` is false and settings
    label the run as non-protocol-complete (incomplete matrix is not fatal for
    smoke). Evidence reports require the full frozen matrix, four legs per
    cell, sample counts, bootstrap frozen fields, process isolation, and
    provenance requirements.
    """
    errors: list[str] = []
    for key in (
        "protocol_id",
        "schema_version",
        "run_id",
        "settings",
        "provenance",
        "protocol",
        "candidate",
        "cells",
        "aggregate",
        "verdicts",
        "honesty",
    ):
        if key not in report:
            errors.append(f"missing top-level key: {key}")
    if report.get("protocol_id") != PROTOCOL_ID:
        errors.append(f"protocol_id mismatch: {report.get('protocol_id')!r}")
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version mismatch: {report.get('schema_version')!r}")

    settings = report.get("settings") or {}
    evidence = settings.get("evidence") is True
    cells = report.get("cells")
    matrix_ok = False
    cell_ids: list[str] = []

    if not isinstance(cells, list):
        errors.append("cells must be a list")
    else:
        cell_ids = [str(c.get("cell_id")) for c in cells]
        matrix_ok = report_cell_ids_match_frozen(cell_ids)

    verdicts = report.get("verdicts") or {}
    if verdicts.get("product_verdict") != PRODUCT_VERDICT:
        errors.append("product_verdict must be NO-GO")
    if verdicts.get("dispatchability_status") != DISPATCHABILITY_STATUS:
        errors.append("dispatchability_status must be failed")

    pca = settings.get("performance_conclusion_allowed")
    if not evidence:
        # Smoke/debug: incomplete matrix is allowed; never allow performance conclusion.
        if pca is not False:
            errors.append("smoke mode must set performance_conclusion_allowed=false")
        if settings.get("evidence") is not False:
            errors.append("smoke/debug reports must set evidence=false")
        # Explicit label when not full matrix.
        if not matrix_ok:
            label = settings.get("run_kind") or settings.get("smoke_subset")
            if label not in (True, "smoke", "smoke_subset", "debug") and not settings.get(
                "smoke_subset", False
            ):
                # Accept if evidence is false — treat as smoke/debug by evidence flag.
                pass
        return errors

    # ---- Evidence mode: authoritative full protocol schema ----
    if not matrix_ok:
        errors.append(
            f"evidence requires exact frozen 27 cell ids in stable order; got {cell_ids!r}"
        )
    if pca is not True:
        errors.append("full-matrix evidence mode must set performance_conclusion_allowed=true")
    if int(settings.get("warmups", -1)) < EVIDENCE_WARMUPS:
        errors.append(
            f"evidence warmups must be >= {EVIDENCE_WARMUPS}, got {settings.get('warmups')!r}"
        )
    if int(settings.get("samples", -1)) < EVIDENCE_SAMPLES:
        errors.append(
            f"evidence samples must be >= {EVIDENCE_SAMPLES}, got {settings.get('samples')!r}"
        )
    if int(settings.get("base_seed", -1)) != BASE_SEED:
        errors.append(f"evidence base_seed must be {BASE_SEED}, got {settings.get('base_seed')!r}")
    if int(settings.get("bootstrap_resamples", -1)) != BOOTSTRAP_RESAMPLES:
        errors.append(
            f"evidence bootstrap_resamples must be {BOOTSTRAP_RESAMPLES}, "
            f"got {settings.get('bootstrap_resamples')!r}"
        )

    provenance = report.get("provenance") or {}
    git = provenance.get("git") or {}
    if git.get("status") != "ok":
        errors.append("evidence provenance requires git status=ok")
    if not git.get("revision"):
        errors.append("evidence provenance requires git revision")
    if git.get("dirty") is not False:
        errors.append("evidence provenance requires dirty=false (clean committed revision)")
    harness_hashes = provenance.get("harness_source_hashes") or {}
    for name in (
        "protocol.py",
        "runner.py",
        "worker.py",
        "stats.py",
        "report.py",
        "candidate.py",
    ):
        h = harness_hashes.get(name)
        if not h or not isinstance(h, str) or len(h) != 64:
            errors.append(f"evidence provenance missing harness_source_hashes[{name}]")
    if not (provenance.get("power_thermal_before") and provenance.get("power_thermal_after")):
        errors.append("evidence provenance requires power_thermal_before and power_thermal_after")
    blas = provenance.get("blas") or {}
    if not blas.get("identified"):
        errors.append("evidence provenance requires identified BLAS vendor")

    candidate = report.get("candidate") or {}
    for key in (
        "artifact_sha256",
        "cargo_toml_sha256",
        "cargo_lock_sha256",
        "lib_rs_sha256",
        "cargo_config_sha256",
        "module_path",
    ):
        if not candidate.get(key):
            errors.append(f"evidence candidate missing {key}")

    if isinstance(cells, list) and matrix_ok:
        for cell, expected in zip(cells, SHAPE_CELLS, strict=True):
            cell_id = expected.cell_id
            for shape_key, exp_shape in (
                ("left_shape", list(expected.left_shape)),
                ("right_shape", list(expected.right_shape)),
                ("out_shape", list(expected.out_shape)),
            ):
                if list(cell.get(shape_key, [])) != exp_shape:
                    errors.append(f"{cell_id}.{shape_key}={cell.get(shape_key)!r} != {exp_shape!r}")
            if int(cell.get("n", -1)) != expected.n:
                errors.append(f"{cell_id}.n mismatch")
            if cell.get("family") != expected.family:
                errors.append(f"{cell_id}.family mismatch")
            if int(cell.get("cell_index", -1)) != expected.cell_index:
                errors.append(f"{cell_id}.cell_index mismatch")

            inputs = cell.get("inputs") or {}
            if (
                not inputs.get("a_sha256")
                or not inputs.get("b_sha256")
                or not inputs.get("ref_sha256")
            ):
                errors.append(f"{cell_id}: missing input hashes")
            expected_input_seed = cell_input_seed(expected.cell_index)
            if "seed" not in inputs:
                errors.append(f"{cell_id}: missing input seed")
            elif int(inputs.get("seed", -1)) != expected_input_seed:
                errors.append(
                    f"{cell_id}: input seed must be frozen derived "
                    f"{expected_input_seed}, got {inputs.get('seed')!r}"
                )

            legs = cell.get("legs") or {}
            for leg_name in TIMING_LEGS:
                if leg_name not in legs:
                    errors.append(f"{cell_id}: missing leg {leg_name}")
                    continue
                leg = legs[leg_name]
                warmups = int(leg.get("warmups", -1))
                if warmups < EVIDENCE_WARMUPS:
                    errors.append(f"{cell_id}.{leg_name}: warmups {warmups} < {EVIDENCE_WARMUPS}")
                samples = leg.get("per_call_samples_s") or []
                if len(samples) < EVIDENCE_SAMPLES:
                    errors.append(
                        f"{cell_id}.{leg_name}: need >= {EVIDENCE_SAMPLES} samples, "
                        f"got {len(samples)}"
                    )
                sample_err = validate_positive_finite_samples(
                    samples, label=f"{cell_id}.{leg_name}"
                )
                if sample_err:
                    errors.append(sample_err)
                if int(leg.get("iterations", 0)) <= 0:
                    errors.append(f"{cell_id}.{leg_name}: iterations must be > 0")
                if not (leg.get("process") or {}).get("nonce"):
                    errors.append(f"{cell_id}.{leg_name}: missing process isolation nonce")

            if cell.get("status") == "ok":
                spellings = cell.get("spellings") or {}
                for spelling in SPELLINGS:
                    boot = (spellings.get(spelling) or {}).get("bootstrap") or {}
                    expected_seed = bootstrap_seed(expected.cell_index, spelling)
                    if int(boot.get("seed", -1)) != expected_seed:
                        errors.append(
                            f"{cell_id}.{spelling}: bootstrap seed must be frozen "
                            f"{expected_seed}, got {boot.get('seed')!r}"
                        )
                    if int(boot.get("n_resamples", -1)) != BOOTSTRAP_RESAMPLES:
                        errors.append(
                            f"{cell_id}.{spelling}: n_resamples must be {BOOTSTRAP_RESAMPLES}"
                        )
                    if float(boot.get("alpha", -1.0)) != float(BOOTSTRAP_ALPHA):
                        errors.append(f"{cell_id}.{spelling}: alpha must be {BOOTSTRAP_ALPHA}")
                    if boot.get("percentile_method") != PERCENTILE_METHOD:
                        errors.append(
                            f"{cell_id}.{spelling}: percentile_method must be {PERCENTILE_METHOD!r}"
                        )

        errors.extend(validate_process_isolation(report))

    # Evidence integrity: performance_passed cannot be true with schema errors.
    # (Caller also forces this; double-check stored report honesty.)
    if errors and verdicts.get("performance_passed") is True:
        errors.append("malformed evidence report cannot claim performance_passed=true")

    return errors


def apply_evidence_integrity(
    report: dict[str, Any],
    *,
    schema_errors: list[str],
    provenance_problems: list[str],
) -> dict[str, Any]:
    """Force INVALID status and performance_passed=false when evidence is broken.

    Mutates *report* in place and returns it. Valid measured slowdowns remain
    VALID evidence with performance_passed=false (gate failure is not INVALID).
    """
    settings = report.get("settings") or {}
    if settings.get("evidence") is not True:
        report["evidence_status"] = "SMOKE"
        report["schema_errors"] = list(schema_errors)
        report["invalid_reasons"] = []
        return report

    invalid_reasons = list(schema_errors) + list(provenance_problems)
    report["schema_errors"] = list(schema_errors)
    report["provenance_problems"] = list(provenance_problems)
    if invalid_reasons:
        report["evidence_status"] = "INVALID"
        report["invalid_reasons"] = invalid_reasons
        verdicts = report.setdefault("verdicts", {})
        verdicts["performance_passed"] = False
        prior = verdicts.get("performance_reason")
        reason = "evidence INVALID: " + "; ".join(invalid_reasons[:8])
        if len(invalid_reasons) > 8:
            reason += f" (+{len(invalid_reasons) - 8} more)"
        if prior and prior not in reason:
            verdicts["performance_reason"] = f"{reason} | prior={prior}"
        else:
            verdicts["performance_reason"] = reason
        honesty = report.setdefault("honesty", {})
        honesty["evidence_invalid"] = True
        honesty["invalid_reasons"] = invalid_reasons
    else:
        report["evidence_status"] = "VALID"
        report["invalid_reasons"] = []
    return report
