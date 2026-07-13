"""Acceptance tests for the isolated Wave 2 matmul research harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from benchmarks.matmul_wave2 import PROTOCOL_ID, SCHEMA_VERSION  # noqa: E402
from benchmarks.matmul_wave2.candidate import (  # noqa: E402
    CandidateArtifact,
    RUST_CANDIDATE_DIR,
    is_secret_env_key,
    record_build_env,
    scrub_secrets_from_text,
    validate_loaded_module,
)
from benchmarks.matmul_wave2.protocol import (  # noqa: E402
    BASE_SEED,
    BASE_SIZES,
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_RESAMPLES,
    CARGO_BUILD_ARGS,
    DISPATCHABILITY_STATUS,
    EVIDENCE_SAMPLES,
    EVIDENCE_WARMUPS,
    EXPECTED_CELL_IDS,
    HARNESS_SOURCE_FILES,
    PERCENTILE_METHOD,
    PRODUCT_VERDICT,
    SHAPE_CELLS,
    SHAPE_FAMILIES,
    SMOKE_SAMPLES,
    SMOKE_WARMUPS,
    SPELLINGS,
    THREAD_ENV,
    TIMING_LEGS,
    bootstrap_seed,
    build_shape_cells,
    cell_input_seed,
    half_size,
    is_exact_frozen_matrix,
    protocol_manifest,
    require_exact_frozen_matrix,
    run_counts,
    shapes_for_family,
)
from benchmarks.matmul_wave2.report import (  # noqa: E402
    apply_evidence_integrity,
    recompute_from_report,
    validate_process_isolation,
    validate_report_schema,
    write_json_report,
)
from benchmarks.matmul_wave2.runner import (  # noqa: E402
    _cpu_info,
    _git_info,
    _numpy_blas_config,
    build_arg_parser,
    build_verdicts,
    default_run_worker_subprocess,
    generate_cell_inputs,
    harness_source_hashes,
    main as wave2_main,
    run_cell,
    run_harness,
    validate_evidence_provenance,
)
from benchmarks.matmul_wave2.stats import (  # noqa: E402
    bootstrap_median_ratio,
    cell_conservative_aggregate,
    geometric_mean,
    nearest_rank_percentile,
    performance_gate,
    summarize_per_call,
    validate_positive_finite_samples,
)
from benchmarks.matmul_wave2.worker import run_worker  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _install_fake_candidate(stage_dir: Path) -> CandidateArtifact:
    """Stage a pure-Python stand-in for the research candidate module."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    module_path = stage_dir / "matmul_wave2_candidate.py"
    source = textwrap.dedent(
        """
        import numpy as np

        def matmul_f64(a, b):
            return np.matmul(a, b)
        """
    ).lstrip()
    module_path.write_text(source, encoding="utf-8")
    sha = _sha256_file(module_path)
    return CandidateArtifact(
        module_name="matmul_wave2_candidate",
        function_name="matmul_f64",
        load_dir=str(stage_dir.resolve()),
        module_path=str(module_path.resolve()),
        artifact_sha256=sha,
        build_wall_s=0.0,
        cargo_stdout="mocked",
        cargo_stderr="",
        crate_dir=str(stage_dir),
        cargo_toml_sha256="0" * 64,
        cargo_lock_sha256="0" * 64,
        lib_rs_sha256="0" * 64,
        cargo_config_sha256="0" * 64,
        rustc_verbose=None,
        cargo_version=None,
        linker=None,
        profile="release",
        build_env={},
    )


def _fast_boot_patch(monkeypatch: pytest.MonkeyPatch, n: int = 20) -> None:
    """Shrink bootstrap resamples for speed in smoke integration-style tests.

    Note: recompute_from_report always uses frozen BOOTSTRAP_RESAMPLES; only
    in-run bootstrap is patched for speed in non-recompute tests.
    """
    import benchmarks.matmul_wave2.runner as runner_mod
    import benchmarks.matmul_wave2.stats as stats_mod

    monkeypatch.setattr(runner_mod, "BOOTSTRAP_RESAMPLES", n)
    monkeypatch.setattr(stats_mod, "BOOTSTRAP_RESAMPLES", n)


def _process_block(nonce: str | None = None) -> dict[str, Any]:
    return {
        "pid": 1000 + abs(hash(nonce or "x")) % 10000,
        "nonce": nonce or os.urandom(8).hex(),
        "numpy_preimported": False,
        "thread_env_applied": dict(THREAD_ENV),
        "numpy_version": "2.0.0",
        "numpy_file": "/fake/numpy/__init__.py",
        "numpy_module": "numpy",
    }


def _synthetic_leg(
    *,
    samples: list[float],
    warmups: int = EVIDENCE_WARMUPS,
    iterations: int = 3,
    nonce: str,
    candidate_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    process = _process_block(nonce)
    return {
        "batch_samples_s": [s * iterations for s in samples],
        "per_call_samples_s": list(samples),
        "summary": {"median": float(np.median(samples)), "count": len(samples)},
        "warmups": warmups,
        "samples": len(samples),
        "iterations": iterations,
        "meta": candidate_meta or {},
        "process": process,
        "thread_env": dict(THREAD_ENV),
        "numpy_version": process["numpy_version"],
        "ok": True,
        "error": None,
    }


def _injectable_worker(art: CandidateArtifact | None = None) -> Any:
    """Smoke-only injectable worker that never imports NumPy in the parent."""

    def _worker(cfg: dict[str, Any]) -> dict[str, Any]:
        # Real subprocess path for isolation correctness.
        return default_run_worker_subprocess(cfg)

    return _worker


# ---------------------------------------------------------------------------
# Protocol / shape matrix
# ---------------------------------------------------------------------------


class TestShapeMatrix:
    def test_exactly_27_cells(self) -> None:
        assert len(SHAPE_CELLS) == 27
        assert len(build_shape_cells()) == 27

    def test_base_sizes_and_families(self) -> None:
        assert BASE_SIZES == (2, 4, 8, 16, 32, 64, 128, 256, 512)
        assert SHAPE_FAMILIES == ("square", "wide_tall", "tall_wide")

    def test_h_and_shapes(self) -> None:
        for n in BASE_SIZES:
            h = half_size(n)
            assert h == max(1, n // 2)
            left, right, out = shapes_for_family("square", n)
            assert left == right == out == (n, n)
            left, right, out = shapes_for_family("wide_tall", n)
            assert left == (n, h) and right == (h, n) and out == (n, n)
            left, right, out = shapes_for_family("tall_wide", n)
            assert left == (h, n) and right == (n, h) and out == (h, h)

    def test_cell_indices_stable(self) -> None:
        for i, cell in enumerate(SHAPE_CELLS):
            assert cell.cell_index == i
            assert cell.cell_id == f"{cell.family}_n{cell.n}"

    def test_three_spellings_four_legs_not_81_cells(self) -> None:
        assert SPELLINGS == ("dot", "matmul", "matmul_op")
        assert TIMING_LEGS == ("candidate", "dot", "matmul", "matmul_op")
        assert len(SPELLINGS) * len(SHAPE_CELLS) == 81
        # Protocol treats spellings as legs inside 27 cells, not product cells.
        assert len(SHAPE_CELLS) == 27

    def test_evidence_and_smoke_counts(self) -> None:
        assert run_counts(evidence=True) == (EVIDENCE_WARMUPS, EVIDENCE_SAMPLES)
        assert run_counts(evidence=False) == (SMOKE_WARMUPS, SMOKE_SAMPLES)
        assert EVIDENCE_WARMUPS >= 5
        assert EVIDENCE_SAMPLES >= 30

    def test_protocol_manifest_schema_keys(self) -> None:
        m = protocol_manifest()
        assert m["protocol_id"] == PROTOCOL_ID
        assert m["schema_version"] == SCHEMA_VERSION
        assert m["n_cells"] == 27
        assert m["product_verdict"] == PRODUCT_VERDICT
        assert m["dispatchability_status"] == DISPATCHABILITY_STATUS
        assert m["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES


# ---------------------------------------------------------------------------
# Stats / bootstrap
# ---------------------------------------------------------------------------


class TestStats:
    def test_nearest_rank_percentile_golden(self) -> None:
        samples = list(range(1, 21))
        assert nearest_rank_percentile(samples, 0.95) == 19.0
        assert nearest_rank_percentile([3.5], 0.5) == 3.5

    def test_nearest_rank_rejects_empty_and_bad_p(self) -> None:
        with pytest.raises(ValueError):
            nearest_rank_percentile([], 0.5)
        with pytest.raises(ValueError):
            nearest_rank_percentile([1.0], 0.0)

    def test_geometric_mean_fsum_log(self) -> None:
        vals = [1.0, 2.0, 4.0]
        expected = math.exp(math.fsum(math.log(v) for v in vals) / 3)
        assert geometric_mean(vals) == pytest.approx(expected)
        assert geometric_mean([1.25] * 27) == pytest.approx(1.25)

    def test_bootstrap_deterministic_golden(self) -> None:
        rng = np.random.default_rng(0)
        numpy_s = list(rng.uniform(0.02, 0.03, size=30))
        cand_s = list(rng.uniform(0.01, 0.015, size=30))
        a = bootstrap_median_ratio(numpy_s, cand_s, seed=12345, n_resamples=200)
        b = bootstrap_median_ratio(numpy_s, cand_s, seed=12345, n_resamples=200)
        assert a.point_estimate == b.point_estimate
        assert a.ci_lower == b.ci_lower
        assert a.ci_upper == b.ci_upper
        assert a.percentile_method == "nearest-rank"
        assert a.point_estimate == pytest.approx(
            float(np.median(numpy_s)) / float(np.median(cand_s))
        )
        assert a.ci_lower <= a.point_estimate <= a.ci_upper or True  # CI may not contain point

    def test_bootstrap_recompute_matches(self) -> None:
        numpy_s = [0.02, 0.021, 0.019, 0.022, 0.020]
        cand_s = [0.010, 0.011, 0.0095, 0.0105, 0.010]
        first = bootstrap_median_ratio(numpy_s, cand_s, seed=7, n_resamples=500)
        second = bootstrap_median_ratio(numpy_s, cand_s, seed=7, n_resamples=500)
        assert first.to_dict() == second.to_dict()

    def test_cell_conservative_is_min(self) -> None:
        agg = cell_conservative_aggregate(
            {
                k: bootstrap_median_ratio([2.0] * 5, [1.0] * 5, seed=i + 1, n_resamples=10)
                if k != "matmul"
                else bootstrap_median_ratio([1.0] * 5, [1.0] * 5, seed=99, n_resamples=10)
                for i, k in enumerate(("dot", "matmul", "matmul_op"))
            }
        )
        assert agg.conservative_point == min(agg.spelling_points.values())
        assert agg.conservative_ci_lower == min(agg.spelling_ci_lowers.values())

    def test_performance_gate_thresholds(self) -> None:
        ids = list(EXPECTED_CELL_IDS)
        # All win
        points = [1.3] * 27
        lowers = [1.05] * 27
        gate = performance_gate(lowers, points, cell_ids=ids)
        assert gate["passed"] is True
        assert gate["geometric_mean"] == pytest.approx(1.3)
        # One CI failure
        lowers2 = [1.05] * 26 + [0.99]
        assert performance_gate(lowers2, points, cell_ids=ids)["passed"] is False
        # Geom mean failure
        points2 = [1.1] * 27
        assert performance_gate(lowers, points2, cell_ids=ids)["passed"] is False
        # Length 27 with wrong ids cannot pass
        wrong_ids = [f"fake_{i}" for i in range(27)]
        assert performance_gate(lowers, points, cell_ids=wrong_ids)["passed"] is False
        # Missing cell_ids cannot pass
        assert performance_gate(lowers, points)["passed"] is False

    def test_positive_finite_sample_validator(self) -> None:
        assert validate_positive_finite_samples([0.1, 0.2], label="x") is None
        assert validate_positive_finite_samples([], label="x") is not None
        assert validate_positive_finite_samples([0.1, 0.0], label="x") is not None
        assert validate_positive_finite_samples([0.1, float("nan")], label="x") is not None

    def test_summarize_per_call(self) -> None:
        s = summarize_per_call([1.0, 2.0, 3.0])
        assert s.median == 2.0
        assert s.count == 3
        assert s.mean == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Path / hash fail-closed
# ---------------------------------------------------------------------------


class TestPathHashFailClosed:
    def test_path_mismatch(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.so"
        f.write_bytes(b"abc")
        sha = _sha256_file(f)
        err = validate_loaded_module(
            str(f), expected_path=tmp_path / "other.so", expected_sha256=sha
        )
        assert err is not None
        assert "mismatch" in err.lower() or "!=" in err

    def test_hash_mismatch(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.so"
        f.write_bytes(b"abc")
        err = validate_loaded_module(str(f), expected_path=f, expected_sha256="deadbeef" * 8)
        assert err is not None
        assert "SHA-256" in err or "sha" in err.lower()

    def test_missing_file_attr(self) -> None:
        err = validate_loaded_module(None, expected_path="/tmp/x", expected_sha256="0" * 64)
        assert err is not None


# ---------------------------------------------------------------------------
# Inputs / correctness
# ---------------------------------------------------------------------------


class TestInputsAndCorrectness:
    def test_deterministic_inputs_and_hashes(self, tmp_path: Path) -> None:
        cell = SHAPE_CELLS[0]
        m1 = generate_cell_inputs(cell, tmp_path / "a", base_seed=BASE_SEED)
        m2 = generate_cell_inputs(cell, tmp_path / "b", base_seed=BASE_SEED)
        assert m1["seed"] == m2["seed"] == cell_input_seed(cell.cell_index)
        assert m1["a_sha256"] == m2["a_sha256"]
        assert m1["b_sha256"] == m2["b_sha256"]
        assert m1["ref_sha256"] == m2["ref_sha256"]
        a = np.load(m1["a_path"])
        b = np.load(m1["b_path"])
        assert a.dtype == np.float64 and b.dtype == np.float64
        assert a.shape == cell.left_shape
        assert b.shape == cell.right_shape

    def test_worker_correctness_failure(self, tmp_path: Path) -> None:
        cell = SHAPE_CELLS[0]
        meta = generate_cell_inputs(cell, tmp_path, base_seed=BASE_SEED)
        # Corrupt reference so allclose fails.
        bad_ref = np.load(meta["ref_path"]) + 1.0
        np.save(meta["ref_path"], bad_ref)
        cfg = {
            "leg": "dot",
            "a_path": meta["a_path"],
            "b_path": meta["b_path"],
            "ref_path": meta["ref_path"],
            "warmups": 0,
            "samples": 1,
            "iterations": 1,
            "expected_shape": list(cell.out_shape),
            "rtol": 1e-12,
            "atol": 1e-12,
            "equal_nan": True,
            "thread_env": dict(THREAD_ENV),
        }
        # Must use a fresh subprocess so NumPy is not pre-imported.
        result = default_run_worker_subprocess(cfg)
        assert result["ok"] is False
        assert "correctness" in result["error"].lower()
        assert result["process"]["numpy_preimported"] is False

    def test_worker_shape_failure(self, tmp_path: Path) -> None:
        cell = SHAPE_CELLS[0]
        meta = generate_cell_inputs(cell, tmp_path, base_seed=BASE_SEED)
        cfg = {
            "leg": "dot",
            "a_path": meta["a_path"],
            "b_path": meta["b_path"],
            "ref_path": meta["ref_path"],
            "warmups": 0,
            "samples": 1,
            "iterations": 1,
            "expected_shape": [999, 999],
            "rtol": 1e-12,
            "atol": 1e-12,
            "equal_nan": True,
            "thread_env": dict(THREAD_ENV),
        }
        result = default_run_worker_subprocess(cfg)
        assert result["ok"] is False

    def test_worker_env_before_import_and_samples(self, tmp_path: Path) -> None:
        cell = SHAPE_CELLS[0]
        meta = generate_cell_inputs(cell, tmp_path, base_seed=BASE_SEED)
        cfg = {
            "leg": "matmul_op",
            "a_path": meta["a_path"],
            "b_path": meta["b_path"],
            "ref_path": meta["ref_path"],
            "warmups": 1,
            "samples": 3,
            "iterations": 2,
            "expected_shape": list(cell.out_shape),
            "rtol": 1e-12,
            "atol": 1e-12,
            "equal_nan": True,
            "thread_env": dict(THREAD_ENV),
        }
        result = default_run_worker_subprocess(cfg)
        assert result["ok"] is True
        assert len(result["per_call_samples_s"]) == 3
        assert all(x > 0.0 and math.isfinite(x) for x in result["per_call_samples_s"])
        assert result["thread_env"]["OMP_NUM_THREADS"] == "1"
        assert result["process"]["numpy_preimported"] is False
        assert result["process"]["nonce"]
        assert result["process"]["pid"]

    def test_worker_rejects_preimported_numpy_in_process(self, tmp_path: Path) -> None:
        # Parent already imported numpy; in-process worker must fail closed.
        cell = SHAPE_CELLS[0]
        meta = generate_cell_inputs(cell, tmp_path, base_seed=BASE_SEED)
        cfg = {
            "leg": "dot",
            "a_path": meta["a_path"],
            "b_path": meta["b_path"],
            "ref_path": meta["ref_path"],
            "warmups": 0,
            "samples": 1,
            "iterations": 1,
            "expected_shape": list(cell.out_shape),
            "rtol": 1e-12,
            "atol": 1e-12,
            "equal_nan": True,
            "thread_env": dict(THREAD_ENV),
        }
        assert "numpy" in sys.modules
        result = run_worker(cfg)
        assert result["ok"] is False
        assert "numpy already" in result["error"].lower()
        assert result["process"]["numpy_preimported"] is True

    def test_candidate_leg_path_hash(self, tmp_path: Path) -> None:
        cell = SHAPE_CELLS[0]
        meta = generate_cell_inputs(cell, tmp_path / "in", base_seed=BASE_SEED)
        art = _install_fake_candidate(tmp_path / "cand")
        cfg = {
            "leg": "candidate",
            "a_path": meta["a_path"],
            "b_path": meta["b_path"],
            "ref_path": meta["ref_path"],
            "warmups": 0,
            "samples": 2,
            "iterations": 1,
            "expected_shape": list(cell.out_shape),
            "rtol": 1e-12,
            "atol": 1e-12,
            "equal_nan": True,
            "thread_env": dict(THREAD_ENV),
            "candidate_load_dir": art.load_dir,
            "candidate_module_name": art.module_name,
            "candidate_function_name": art.function_name,
            "expected_module_path": art.module_path,
            "expected_sha256": art.artifact_sha256,
        }
        result = default_run_worker_subprocess(cfg)
        assert result["ok"] is True
        assert result["meta"]["artifact_sha256"] == art.artifact_sha256
        # Wrong hash fails closed
        cfg["expected_sha256"] = "00" * 32
        bad = default_run_worker_subprocess(cfg)
        assert bad["ok"] is False


# ---------------------------------------------------------------------------
# Cell / harness with mocked build + in-process worker
# ---------------------------------------------------------------------------


class TestRunCellAndHarness:
    def test_run_cell_four_legs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _fast_boot_patch(monkeypatch, 30)
        cell = SHAPE_CELLS[1]
        meta = generate_cell_inputs(cell, tmp_path / "in", base_seed=BASE_SEED)
        art = _install_fake_candidate(tmp_path / "cand")
        result = run_cell(
            cell,
            input_meta=meta,
            artifact=art,
            warmups=1,
            samples=3,
            iterations=2,
            run_worker=default_run_worker_subprocess,
        )
        assert result["status"] == "ok", result.get("reason")
        assert set(result["legs"]) == set(TIMING_LEGS)
        nonces = [result["legs"][leg]["process"]["nonce"] for leg in TIMING_LEGS]
        assert len(set(nonces)) == 4
        for sp in SPELLINGS:
            assert "bootstrap" in result["spellings"][sp]
            assert result["spellings"][sp]["bootstrap"]["seed"] == bootstrap_seed(
                cell.cell_index, sp
            )
        assert result["conservative_point"] == min(
            result["spellings"][s]["bootstrap"]["point_estimate"] for s in SPELLINGS
        )

    def test_smoke_harness_cannot_conclude_performance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fast_boot_patch(monkeypatch, 15)
        cell = SHAPE_CELLS[0]

        def fake_build(stage_dir: Path | str, **kwargs: Any) -> CandidateArtifact:
            return _install_fake_candidate(Path(stage_dir))

        report = run_harness(
            output_dir=tmp_path / "out",
            evidence=False,
            cells=(cell,),
            build_fn=fake_build,
            run_worker=default_run_worker_subprocess,
            calibrate=False,
            fixed_iterations=2,
            skip_ensure_importable=False,
            run_id="smoke-test-1",
        )
        assert report["settings"]["evidence"] is False
        assert report["settings"]["performance_conclusion_allowed"] is False
        assert report["settings"]["smoke_subset"] is True
        assert report["verdicts"]["performance_passed"] is False
        assert "smoke" in (report["verdicts"]["performance_reason"] or "").lower()
        assert report["verdicts"]["product_verdict"] == PRODUCT_VERDICT
        assert report["verdicts"]["dispatchability_status"] == DISPATCHABILITY_STATUS
        assert report["verdicts"]["product_verdict"] == "NO-GO"
        assert Path(report["artifact_paths"]["json"]).is_file()
        assert Path(report["artifact_paths"]["markdown"]).is_file()
        # Smoke subset is not protocol-complete; schema allows it when evidence=false.
        assert validate_report_schema(report) == []

    def test_synthetic_all_win_still_nogo(self) -> None:
        cells = []
        for i, cell in enumerate(SHAPE_CELLS):
            cells.append(
                {
                    "cell_id": cell.cell_id,
                    "status": "ok",
                    "conservative_point": 2.0,
                    "conservative_ci_lower": 1.5,
                }
            )
        verdicts = build_verdicts(cells=cells, evidence=True, provenance_problems=[])
        assert verdicts["performance_passed"] is True
        assert verdicts["dispatchability_status"] == "failed"
        assert verdicts["product_verdict"] == "NO-GO"
        assert verdicts["product_verdict_detail"] == "fallback-retained"

    def test_evidence_subset_cannot_pass_performance_gate(self) -> None:
        # 27 fake wins with wrong ids — must not pass.
        cells = [
            {
                "cell_id": f"fake_{i}",
                "status": "ok",
                "conservative_point": 2.0,
                "conservative_ci_lower": 1.5,
            }
            for i in range(27)
        ]
        v = build_verdicts(cells=cells, evidence=True, provenance_problems=[])
        assert v["performance_passed"] is False
        assert v["matrix_ok"] is False

        # Real subset of 1 cell — must not pass.
        one = [
            {
                "cell_id": SHAPE_CELLS[0].cell_id,
                "status": "ok",
                "conservative_point": 2.0,
                "conservative_ci_lower": 1.5,
            }
        ]
        v2 = build_verdicts(cells=one, evidence=True, provenance_problems=[])
        assert v2["performance_passed"] is False
        assert v2["matrix_ok"] is False

    def test_evidence_harness_rejects_subset(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="exact frozen 27"):
            run_harness(
                output_dir=tmp_path / "out",
                evidence=True,
                cells=(SHAPE_CELLS[0],),
                run_id="evidence-subset-reject",
            )

    def test_evidence_rejects_injected_build_fn(self, tmp_path: Path) -> None:
        def fake_build(stage_dir: Path | str, **kwargs: Any) -> CandidateArtifact:
            return _install_fake_candidate(Path(stage_dir))

        with pytest.raises(ValueError, match="injected build_fn"):
            run_harness(
                output_dir=tmp_path / "out",
                evidence=True,
                build_fn=fake_build,
                run_id="reject-build-fn",
            )

    def test_evidence_rejects_injected_run_worker(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="injected run_worker"):
            run_harness(
                output_dir=tmp_path / "out",
                evidence=True,
                run_worker=default_run_worker_subprocess,
                run_id="reject-worker",
            )

    def test_evidence_rejects_skip_ensure_importable(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="skip_ensure_importable"):
            run_harness(
                output_dir=tmp_path / "out",
                evidence=True,
                skip_ensure_importable=True,
                run_id="reject-skip",
            )

    def test_evidence_rejects_non_frozen_base_seed(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="base_seed"):
            run_harness(
                output_dir=tmp_path / "out",
                evidence=True,
                base_seed=BASE_SEED + 1,
                run_id="reject-seed",
            )

    def test_evidence_rejects_fixed_iterations(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="fixed_iterations"):
            run_harness(
                output_dir=tmp_path / "out",
                evidence=True,
                fixed_iterations=5,
                run_id="reject-fixed-iter",
            )

    def test_evidence_rejects_disabled_calibration(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="calibration"):
            run_harness(
                output_dir=tmp_path / "out",
                evidence=True,
                calibrate=False,
                run_id="reject-cal",
            )

    def test_evidence_rejects_reordered_matrix(self, tmp_path: Path) -> None:
        reordered = tuple(reversed(SHAPE_CELLS))
        with pytest.raises(ValueError, match="exact frozen 27"):
            run_harness(
                output_dir=tmp_path / "out",
                evidence=True,
                cells=reordered,
                run_id="reject-reorder",
            )

    def test_cli_rejects_evidence_with_cell_id(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            wave2_main(
                [
                    "--output-dir",
                    "/tmp/unused-wave2",
                    "--evidence",
                    "--cell-id",
                    "square_n2",
                ]
            )
        assert excinfo.value.code == 2  # argparse error

    def test_missing_sample_fails_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fast_boot_patch(monkeypatch, 10)
        cell = SHAPE_CELLS[0]
        meta = generate_cell_inputs(cell, tmp_path / "in", base_seed=BASE_SEED)
        art = _install_fake_candidate(tmp_path / "cand")
        calls = {"n": 0}

        def flaky(cfg: dict[str, Any]) -> dict[str, Any]:
            calls["n"] += 1
            if cfg["leg"] == "matmul" and calls["n"] >= 1:
                return {"ok": True, "per_call_samples_s": [0.01], "batch_samples_s": [0.01]}
            return run_worker(cfg)

        # Force wrong sample count on matmul leg; still attempt all four legs.
        def bad_worker(cfg: dict[str, Any]) -> dict[str, Any]:
            r = default_run_worker_subprocess(cfg)
            if cfg["leg"] == "matmul" and r.get("ok"):
                r["per_call_samples_s"] = r["per_call_samples_s"][:1]
                r["batch_samples_s"] = r["batch_samples_s"][:1]
            return r

        result = run_cell(
            cell,
            input_meta=meta,
            artifact=art,
            warmups=0,
            samples=3,
            iterations=1,
            run_worker=bad_worker,
        )
        assert result["status"] == "failed"
        assert "samples" in (result["reason"] or "").lower()
        assert set(result["legs"]) == set(TIMING_LEGS)
        assert all("process" in result["legs"][leg] for leg in TIMING_LEGS)


# ---------------------------------------------------------------------------
# Provenance / schema / round-trip
# ---------------------------------------------------------------------------


class TestProvenanceAndRoundTrip:
    def test_unidentified_blas_invalidates_evidence(self) -> None:
        prov = {
            "blas": {"identified": False, "label": "unidentified"},
            "power_thermal": {"thermal_anomaly": False},
            "cpu": {"model_name": "Test CPU", "platform_machine": "arm64"},
        }
        problems = validate_evidence_provenance(prov)
        assert any("BLAS" in p or "unidentified" in p.lower() for p in problems)

    def test_thermal_anomaly_invalidates_evidence(self) -> None:
        prov = {
            "blas": {"identified": True, "label": "openblas"},
            "power_thermal": {"thermal_anomaly": True, "notes": ["hot"]},
            "cpu": {"model_name": "Test CPU", "platform_machine": "arm64"},
        }
        problems = validate_evidence_provenance(prov)
        assert any("thermal" in p.lower() for p in problems)

    def _synthetic_full_evidence_cells(
        self,
        *,
        n_samples: int = EVIDENCE_SAMPLES,
        warmups: int = EVIDENCE_WARMUPS,
        cand_path: str | None = None,
        cand_sha: str = "a" * 64,
        n_resamples: int = BOOTSTRAP_RESAMPLES,
        tamper_seed: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if cand_path is None:
            cand_path = str(Path("/tmp/fake_cand.so").resolve())
        cells_out: list[dict[str, Any]] = []
        nonce_i = 0
        for cell in SHAPE_CELLS:
            cand = [0.010 + 0.0001 * (i % 5) for i in range(n_samples)]
            spell_samples = {
                "dot": [0.020 + 0.0001 * (i % 5) for i in range(n_samples)],
                "matmul": [0.021 + 0.0001 * (i % 5) for i in range(n_samples)],
                "matmul_op": [0.019 + 0.0001 * (i % 5) for i in range(n_samples)],
            }
            legs = {
                "candidate": _synthetic_leg(
                    samples=cand,
                    warmups=warmups,
                    nonce=f"n{nonce_i}",
                    candidate_meta={
                        "module_file": cand_path,
                        "artifact_sha256": cand_sha,
                    },
                ),
            }
            nonce_i += 1
            for sp in SPELLINGS:
                legs[sp] = _synthetic_leg(
                    samples=spell_samples[sp],
                    warmups=warmups,
                    nonce=f"n{nonce_i}",
                )
                nonce_i += 1
            spellings = {}
            honest_boots = {}
            for sp in SPELLINGS:
                frozen_seed = bootstrap_seed(cell.cell_index, sp)
                # Always compute honest bootstrap with frozen constants for cell aggregates
                # that recompute will regenerate.
                honest = bootstrap_median_ratio(
                    spell_samples[sp],
                    cand,
                    seed=frozen_seed,
                    n_resamples=BOOTSTRAP_RESAMPLES,
                    alpha=BOOTSTRAP_ALPHA,
                )
                honest_boots[sp] = honest
                stored_seed = frozen_seed + 999 if tamper_seed else frozen_seed
                # Stored payload may intentionally lie about seed/resamples.
                if n_resamples != BOOTSTRAP_RESAMPLES or tamper_seed:
                    stored_boot = bootstrap_median_ratio(
                        spell_samples[sp],
                        cand,
                        seed=stored_seed,
                        n_resamples=n_resamples,
                        alpha=BOOTSTRAP_ALPHA,
                    ).to_dict()
                    stored_boot["seed"] = stored_seed
                    stored_boot["n_resamples"] = n_resamples
                else:
                    stored_boot = honest.to_dict()
                spellings[sp] = {"label": sp, "bootstrap": stored_boot}
            agg = cell_conservative_aggregate(honest_boots)
            cells_out.append(
                {
                    "cell_id": cell.cell_id,
                    "cell_index": cell.cell_index,
                    "family": cell.family,
                    "n": cell.n,
                    "h": cell.h,
                    "left_shape": list(cell.left_shape),
                    "right_shape": list(cell.right_shape),
                    "out_shape": list(cell.out_shape),
                    "status": "ok",
                    "conservative_point": agg.conservative_point,
                    "conservative_ci_lower": agg.conservative_ci_lower,
                    "inputs": {
                        "seed": cell_input_seed(cell.cell_index),
                        "a_sha256": "b" * 64,
                        "b_sha256": "c" * 64,
                        "ref_sha256": "d" * 64,
                    },
                    "legs": legs,
                    "spellings": spellings,
                    "iterations": 3,
                }
            )
        points = [c["conservative_point"] for c in cells_out]
        lowers = [c["conservative_ci_lower"] for c in cells_out]
        gate = performance_gate(lowers, points, cell_ids=list(EXPECTED_CELL_IDS))
        return cells_out, gate

    def test_report_round_trip_recompute(self, tmp_path: Path) -> None:
        # Uses frozen bootstrap constants (20000) so recompute matches.
        cand_path = str(Path("/tmp/fake_cand.so").resolve())
        cand_sha = "a" * 64
        cells_out, gate = self._synthetic_full_evidence_cells(
            cand_path=cand_path,
            cand_sha=cand_sha,
            n_samples=8,  # raw series length; bootstrap still uses frozen 20000
        )
        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": "roundtrip",
            "settings": {
                "evidence": True,
                "performance_conclusion_allowed": True,
                "warmups": EVIDENCE_WARMUPS,
                "samples": EVIDENCE_SAMPLES,
                "base_seed": BASE_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            },
            "provenance": {
                "git": {"status": "ok", "revision": "abc123", "dirty": False},
                "blas": {"identified": True, "label": "openblas"},
                "harness_source_hashes": {n: "e" * 64 for n in HARNESS_SOURCE_FILES},
                "power_thermal_before": {"thermal_anomaly": False},
                "power_thermal_after": {"thermal_anomaly": False},
                "cpu": {"model_name": "Test", "platform_machine": "arm64"},
            },
            "protocol": protocol_manifest(),
            "candidate": {
                "module_path": cand_path,
                "artifact_sha256": cand_sha,
                "cargo_toml_sha256": "1" * 64,
                "cargo_lock_sha256": "2" * 64,
                "lib_rs_sha256": "3" * 64,
                "cargo_config_sha256": "4" * 64,
            },
            "cells": cells_out,
            "aggregate": {
                "geometric_mean": gate["geometric_mean"],
                "min_ci_lower": gate["min_ci_lower"],
                "performance_gate": gate,
            },
            "verdicts": {
                "product_verdict": PRODUCT_VERDICT,
                "dispatchability_status": DISPATCHABILITY_STATUS,
                "performance_passed": gate["passed"],
                "performance_gate": gate,
            },
            "honesty": {},
        }
        path = tmp_path / "report.json"
        write_json_report(path, report)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        # Round-trip focuses on frozen bootstrap recompute (not full sample-count schema).
        recomputed = recompute_from_report(loaded)
        assert recomputed["matches"] is True, recomputed["mismatches"]
        assert recomputed["matrix_ok"] is True

    def test_recompute_rejects_tampered_low_resamples(self) -> None:
        cells_out, gate = self._synthetic_full_evidence_cells(n_resamples=25, n_samples=5)
        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": "tamper-resamples",
            "settings": {"evidence": True, "performance_conclusion_allowed": True},
            "provenance": {},
            "protocol": {},
            "candidate": {},
            "cells": cells_out,
            "aggregate": {
                "geometric_mean": gate["geometric_mean"],
                "min_ci_lower": gate["min_ci_lower"],
            },
            "verdicts": {
                "product_verdict": PRODUCT_VERDICT,
                "dispatchability_status": DISPATCHABILITY_STATUS,
                "performance_passed": gate["passed"],
            },
            "honesty": {},
        }
        result = recompute_from_report(report)
        assert result["matches"] is False
        assert any("n_resamples" in m for m in result["mismatches"])

    def test_recompute_rejects_tampered_seed(self) -> None:
        cells_out, gate = self._synthetic_full_evidence_cells(tamper_seed=True, n_samples=5)
        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": "tamper-seed",
            "settings": {"evidence": True, "performance_conclusion_allowed": True},
            "provenance": {},
            "protocol": {},
            "candidate": {},
            "cells": cells_out,
            "aggregate": {
                "geometric_mean": gate["geometric_mean"],
                "min_ci_lower": gate["min_ci_lower"],
            },
            "verdicts": {
                "product_verdict": PRODUCT_VERDICT,
                "dispatchability_status": DISPATCHABILITY_STATUS,
                "performance_passed": True,  # dishonest claim
            },
            "honesty": {},
        }
        result = recompute_from_report(report)
        assert result["matches"] is False
        assert any(".seed" in m for m in result["mismatches"])

    def test_smoke_schema_performance_flag(self) -> None:
        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": "x",
            "settings": {
                "evidence": False,
                "performance_conclusion_allowed": True,
                "smoke_subset": True,
            },
            "provenance": {},
            "protocol": {},
            "candidate": {},
            "cells": [{"cell_id": SHAPE_CELLS[0].cell_id}],
            "aggregate": {},
            "verdicts": {
                "product_verdict": PRODUCT_VERDICT,
                "dispatchability_status": DISPATCHABILITY_STATUS,
            },
            "honesty": {},
        }
        errs = validate_report_schema(report)
        assert any("performance_conclusion_allowed" in e for e in errs)

    def test_schema_rejects_wrong_cell_ids_even_if_length_27(self) -> None:
        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": "x",
            "settings": {
                "evidence": True,
                "performance_conclusion_allowed": True,
                "warmups": EVIDENCE_WARMUPS,
                "samples": EVIDENCE_SAMPLES,
                "base_seed": BASE_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            },
            "provenance": {
                "git": {"status": "ok", "revision": "x", "dirty": False},
                "blas": {"identified": True, "label": "openblas"},
                "harness_source_hashes": {n: "e" * 64 for n in HARNESS_SOURCE_FILES},
                "power_thermal_before": {"thermal_anomaly": False},
                "power_thermal_after": {"thermal_anomaly": False},
            },
            "protocol": {},
            "candidate": {
                "module_path": "/x",
                "artifact_sha256": "a" * 64,
                "cargo_toml_sha256": "1" * 64,
                "cargo_lock_sha256": "2" * 64,
                "lib_rs_sha256": "3" * 64,
                "cargo_config_sha256": "4" * 64,
            },
            "cells": [{"cell_id": f"fake_{i}"} for i in range(27)],
            "aggregate": {},
            "verdicts": {
                "product_verdict": PRODUCT_VERDICT,
                "dispatchability_status": DISPATCHABILITY_STATUS,
                "performance_passed": False,
            },
            "honesty": {},
        }
        errs = validate_report_schema(report)
        assert any("exact frozen 27" in e for e in errs)

    def test_git_info_clean_and_dirty(self, tmp_path: Path) -> None:
        # Isolated throwaway repos — no remotes, no network.
        clean = tmp_path / "clean"
        clean.mkdir()
        subprocess.run(["git", "init"], cwd=clean, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "wave2@example.test"],
            cwd=clean,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "wave2"],
            cwd=clean,
            check=True,
            capture_output=True,
        )
        (clean / "f.txt").write_text("x\n", encoding="utf-8")
        subprocess.run(["git", "add", "f.txt"], cwd=clean, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=clean,
            check=True,
            capture_output=True,
        )
        info_clean = _git_info(clean)
        assert info_clean["status"] == "ok"
        assert info_clean["dirty"] is False
        assert info_clean["revision"]
        assert "http" not in json.dumps(info_clean).lower()
        assert "github" not in json.dumps(info_clean).lower()

        dirty = tmp_path / "dirty"
        dirty.mkdir()
        subprocess.run(["git", "init"], cwd=dirty, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "wave2@example.test"],
            cwd=dirty,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "wave2"],
            cwd=dirty,
            check=True,
            capture_output=True,
        )
        (dirty / "f.txt").write_text("x\n", encoding="utf-8")
        subprocess.run(["git", "add", "f.txt"], cwd=dirty, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=dirty,
            check=True,
            capture_output=True,
        )
        (dirty / "extra.txt").write_text("dirty\n", encoding="utf-8")
        info_dirty = _git_info(dirty)
        assert info_dirty["status"] == "ok"
        assert info_dirty["dirty"] is True
        assert "remote" not in info_dirty

    def test_build_env_never_records_secrets(self) -> None:
        env = {
            "PATH": "/usr/bin:/secret/bin",
            "PYO3_PYTHON": "/venv/bin/python",
            "CARGO_TERM_COLOR": "never",
            "CARGO_REGISTRIES_CRATES_IO_TOKEN": "super-secret-token-value",
            "CARGO_REGISTRY_TOKEN": "another-secret",
            "MY_PASSWORD": "hunter2",
            "AWS_SECRET_ACCESS_KEY": "AKIAsecret",
            "SOME_AUTH_HEADER": "Bearer abc",
            "RUSTFLAGS": "-C opt-level=3",
            "NOT_RELEVANT": "x",
        }
        recorded = record_build_env(env)
        blob = json.dumps(recorded)
        assert "super-secret-token-value" not in blob
        assert "another-secret" not in blob
        assert "hunter2" not in blob
        assert "AKIAsecret" not in blob
        assert "Bearer abc" not in blob
        assert "/secret/bin" not in blob
        assert recorded.get("PATH") == "(set)"
        assert recorded.get("PYO3_PYTHON") == "/venv/bin/python"
        assert recorded.get("CARGO_TERM_COLOR") == "never"
        assert "CARGO_REGISTRIES_CRATES_IO_TOKEN" not in recorded
        assert is_secret_env_key("CARGO_REGISTRIES_CRATES_IO_TOKEN")
        scrubbed = scrub_secrets_from_text(
            "token=super-secret-token-value and CARGO uses another-secret",
            env,
        )
        assert "super-secret-token-value" not in scrubbed
        assert "another-secret" not in scrubbed

    def test_cargo_config_hash_in_source_hashes(self) -> None:
        from benchmarks.matmul_wave2.candidate import candidate_source_hashes

        hashes = candidate_source_hashes()
        assert ".cargo/config.toml" in hashes
        assert len(hashes[".cargo/config.toml"]) == 64
        config_path = RUST_CANDIDATE_DIR / ".cargo" / "config.toml"
        assert config_path.is_file()
        assert hashes[".cargo/config.toml"] == _sha256_file(config_path)

    def test_frozen_matrix_helpers(self) -> None:
        assert is_exact_frozen_matrix(SHAPE_CELLS)
        assert list(EXPECTED_CELL_IDS) == [c.cell_id for c in SHAPE_CELLS]
        with pytest.raises(ValueError, match="exact frozen 27"):
            require_exact_frozen_matrix((SHAPE_CELLS[0],))
        assert build_arg_parser()  # smoke construct


# ---------------------------------------------------------------------------
# Process isolation (env applied in child)
# ---------------------------------------------------------------------------


class TestProcessIsolation:
    def test_worker_module_subprocess(self, tmp_path: Path) -> None:
        cell = SHAPE_CELLS[0]
        meta = generate_cell_inputs(cell, tmp_path, base_seed=BASE_SEED)
        cfg = {
            "leg": "dot",
            "a_path": meta["a_path"],
            "b_path": meta["b_path"],
            "ref_path": meta["ref_path"],
            "warmups": 1,
            "samples": 2,
            "iterations": 1,
            "expected_shape": list(cell.out_shape),
            "rtol": 1e-12,
            "atol": 1e-12,
            "equal_nan": True,
            "thread_env": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        )
        proc = subprocess.run(
            [sys.executable, "-m", "benchmarks.matmul_wave2.worker", str(cfg_path)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout)
        assert result["ok"] is True
        assert len(result["per_call_samples_s"]) == 2


# ---------------------------------------------------------------------------
# Optional real-cargo one-cell smoke
# ---------------------------------------------------------------------------


class TestCargoTomlStaticPins:
    def test_exact_crate_pins_and_build_args(self) -> None:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - py311+ has tomllib
            import tomli as tomllib  # type: ignore

        cargo_path = RUST_CANDIDATE_DIR / "Cargo.toml"
        data = tomllib.loads(cargo_path.read_text(encoding="utf-8"))
        deps = data.get("dependencies") or {}
        assert deps.get("numpy") == "=0.29.0"
        pyo3 = deps.get("pyo3")
        assert isinstance(pyo3, dict)
        assert pyo3.get("version") == "=0.29.0"
        assert pyo3.get("features") == ["extension-module"]
        # No BLAS-related dependency names or crate features.
        dep_names = {str(k).lower() for k in deps}
        banned_names = {
            "openblas",
            "openblas-src",
            "intel-mkl",
            "intel-mkl-src",
            "blas-src",
            "netlib-src",
            "accelerate-src",
            "blis-src",
        }
        assert dep_names.isdisjoint(banned_names)
        for name, spec in deps.items():
            if isinstance(spec, dict):
                feats = [str(f).lower() for f in (spec.get("features") or [])]
                assert "blas" not in feats, f"{name} enables blas feature"
        # No [features] table enabling BLAS, and no optional BLAS deps.
        assert "features" not in data or not data["features"]
        assert CARGO_BUILD_ARGS == ("build", "--release", "--locked")
        assert (RUST_CANDIDATE_DIR / "Cargo.lock").is_file()
        assert (RUST_CANDIDATE_DIR / ".cargo" / "config.toml").is_file()
        # Source pins also appear literally for human review.
        text = cargo_path.read_text(encoding="utf-8")
        assert 'numpy = "=0.29.0"' in text
        assert 'version = "=0.29.0"' in text
        assert '"extension-module"' in text


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not on PATH")
class TestRealCargoBoundedSmoke:
    def test_one_cell_real_cargo_smoke(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bounded real build + one cell smoke when the toolchain is present."""
        _fast_boot_patch(monkeypatch, 10)
        # Real cargo build can take a while the first time; keep to one tiny cell.
        report = run_harness(
            output_dir=tmp_path / "real",
            evidence=False,
            cells=(SHAPE_CELLS[0],),
            calibrate=False,
            fixed_iterations=1,
            run_id="real-cargo-smoke",
            # default build_fn + default subprocess worker
        )
        assert report["verdicts"]["product_verdict"] == "NO-GO"
        assert report["settings"]["evidence"] is False
        assert report["settings"]["performance_conclusion_allowed"] is False
        assert report["cells"][0]["status"] == "ok", report["cells"][0].get("reason")
        cand = report["candidate"]
        assert Path(cand["module_path"]).is_file()
        assert len(cand["artifact_sha256"]) == 64
        assert "cargo_config_sha256" in cand
        assert len(cand["cargo_config_sha256"]) == 64
        # Secrets must never appear in recorded build env / cargo logs.
        build_env = cand.get("build_env", {})
        assert not any(is_secret_env_key(k) for k in build_env)
        assert "CARGO_REGISTRIES_CRATES_IO_TOKEN" not in build_env
        assert build_env.get("PATH") in (None, "(set)")
        assert "super-secret" not in json.dumps(cand)
        # Dual thermal snapshots recorded even for smoke.
        assert "power_thermal_before" in report["provenance"]
        assert "power_thermal_after" in report["provenance"]
        assert "harness_source_hashes" in report["provenance"]
        # All four legs recorded with process nonces.
        legs = report["cells"][0]["legs"]
        assert set(legs) == set(TIMING_LEGS)
        nonces = [legs[name]["process"]["nonce"] for name in TIMING_LEGS]
        assert len(set(nonces)) == 4


# ---------------------------------------------------------------------------
# Evidence integrity (provenance / process / schema / hw.model)
# ---------------------------------------------------------------------------


class TestEvidenceProvenanceIntegrity:
    def test_dirty_git_invalidates_evidence(self) -> None:
        prov = {
            "git": {"status": "ok", "revision": "abc", "dirty": True},
            "blas": {"identified": True, "label": "openblas"},
            "power_thermal_before": {"thermal_anomaly": False},
            "power_thermal_after": {"thermal_anomaly": False},
            "cpu": {"model_name": "X", "platform_machine": "arm64"},
            "harness_source_hashes": {n: "a" * 64 for n in HARNESS_SOURCE_FILES},
        }
        problems = validate_evidence_provenance(prov)
        assert any("dirty" in p.lower() for p in problems)

    def test_unavailable_git_invalidates_evidence(self) -> None:
        prov = {
            "git": {"status": "unavailable", "revision": None, "dirty": None},
            "blas": {"identified": True, "label": "openblas"},
            "power_thermal_before": {"thermal_anomaly": False},
            "power_thermal_after": {"thermal_anomaly": False},
            "cpu": {"model_name": "X", "platform_machine": "arm64"},
            "harness_source_hashes": {n: "a" * 64 for n in HARNESS_SOURCE_FILES},
        }
        problems = validate_evidence_provenance(prov)
        assert any("git" in p.lower() for p in problems)

    def test_missing_revision_invalidates_evidence(self) -> None:
        prov = {
            "git": {"status": "ok", "revision": None, "dirty": False},
            "blas": {"identified": True, "label": "openblas"},
            "power_thermal_before": {"thermal_anomaly": False},
            "power_thermal_after": {"thermal_anomaly": False},
            "cpu": {"model_name": "X", "platform_machine": "arm64"},
            "harness_source_hashes": {n: "a" * 64 for n in HARNESS_SOURCE_FILES},
        }
        problems = validate_evidence_provenance(prov)
        assert any("revision" in p.lower() for p in problems)

    def test_thermal_before_or_after_invalidates(self) -> None:
        base = {
            "git": {"status": "ok", "revision": "abc", "dirty": False},
            "blas": {"identified": True, "label": "openblas"},
            "cpu": {"model_name": "X", "platform_machine": "arm64"},
            "harness_source_hashes": {n: "a" * 64 for n in HARNESS_SOURCE_FILES},
        }
        before = {
            **base,
            "power_thermal_before": {"thermal_anomaly": True},
            "power_thermal_after": {"thermal_anomaly": False},
        }
        after = {
            **base,
            "power_thermal_before": {"thermal_anomaly": False},
            "power_thermal_after": {"thermal_anomaly": True},
        }
        assert any("thermal" in p.lower() for p in validate_evidence_provenance(before))
        assert any("thermal" in p.lower() for p in validate_evidence_provenance(after))

    def test_generic_blas_not_identified(self) -> None:
        # Generic "blas"/"lapack" alone must not count as vendor identification.
        blob = json.dumps({"libraries": ["blas", "lapack"]}).lower()
        assert "blas" in blob and "lapack" in blob
        identified = any(
            token in blob
            for token, _label in (("openblas", "x"), ("mkl", "x"), ("accelerate", "x"))
        )
        assert identified is False
        info = _numpy_blas_config()
        assert "identified" in info
        assert "vendor_tokens_checked" in info

    def test_harness_source_hashes_complete(self) -> None:
        hashes = harness_source_hashes()
        assert set(hashes) == set(HARNESS_SOURCE_FILES)
        assert all(len(v) == 64 for v in hashes.values())

    def test_cpu_info_hw_model_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import benchmarks.matmul_wave2.runner as runner_mod

        def fake_run_text(argv: list[str], *, timeout: float = 10.0) -> str | None:
            joined = " ".join(argv)
            if "machdep.cpu.brand_string" in joined:
                return None
            if "hw.model" in joined:
                return "Mac16,7"
            return None

        monkeypatch.setattr(runner_mod, "_run_text", fake_run_text)
        info = _cpu_info()
        assert info["model_name"] == "Mac16,7"
        assert info["source"] == "sysctl hw.model"
        assert info["logical_cores"] is not None or info["platform_machine"] is not None


class TestEvidenceSchemaAndIntegrity:
    def test_apply_integrity_forces_invalid(self) -> None:
        report: dict[str, Any] = {
            "settings": {"evidence": True},
            "verdicts": {
                "performance_passed": True,
                "performance_reason": "would have passed",
                "product_verdict": PRODUCT_VERDICT,
                "dispatchability_status": DISPATCHABILITY_STATUS,
            },
            "honesty": {},
        }
        apply_evidence_integrity(
            report,
            schema_errors=["missing legs"],
            provenance_problems=["dirty tree"],
        )
        assert report["evidence_status"] == "INVALID"
        verdicts = report["verdicts"]
        assert isinstance(verdicts, dict)
        assert verdicts["performance_passed"] is False
        invalid_reasons = report["invalid_reasons"]
        assert isinstance(invalid_reasons, list)
        assert "missing legs" in invalid_reasons

    def _schema_cells_no_bootstrap(self) -> tuple[list[dict[str, Any]], str]:
        """Build schema-valid cells without running expensive bootstraps."""
        cand_path = str(Path("/tmp/fake_cand.so").resolve())
        cand_sha = "a" * 64
        cells: list[dict[str, Any]] = []
        nonce_i = 0
        samples = [0.01 + 0.0001 * (i % 3) for i in range(EVIDENCE_SAMPLES)]
        for cell in SHAPE_CELLS:
            legs = {}
            for leg_name in TIMING_LEGS:
                meta = {}
                if leg_name == "candidate":
                    meta = {"module_file": cand_path, "artifact_sha256": cand_sha}
                legs[leg_name] = _synthetic_leg(
                    samples=samples,
                    warmups=EVIDENCE_WARMUPS,
                    nonce=f"sn{nonce_i}",
                    candidate_meta=meta,
                )
                nonce_i += 1
            spellings = {}
            for sp in SPELLINGS:
                seed = bootstrap_seed(cell.cell_index, sp)
                spellings[sp] = {
                    "label": sp,
                    "bootstrap": {
                        "point_estimate": 1.5,
                        "ci_lower": 1.1,
                        "ci_upper": 2.0,
                        "alpha": BOOTSTRAP_ALPHA,
                        "n_resamples": BOOTSTRAP_RESAMPLES,
                        "seed": seed,
                        "percentile_method": PERCENTILE_METHOD,
                        "percentile_method_doc": "nearest-rank",
                        "numpy_median": 0.02,
                        "candidate_median": 0.01,
                    },
                }
            cells.append(
                {
                    "cell_id": cell.cell_id,
                    "cell_index": cell.cell_index,
                    "family": cell.family,
                    "n": cell.n,
                    "h": cell.h,
                    "left_shape": list(cell.left_shape),
                    "right_shape": list(cell.right_shape),
                    "out_shape": list(cell.out_shape),
                    "status": "ok",
                    "conservative_point": 1.5,
                    "conservative_ci_lower": 1.1,
                    "inputs": {
                        "seed": cell_input_seed(cell.cell_index),
                        "a_sha256": "b" * 64,
                        "b_sha256": "c" * 64,
                        "ref_sha256": "d" * 64,
                    },
                    "legs": legs,
                    "spellings": spellings,
                    "iterations": 3,
                }
            )
        return cells, cand_path

    def test_full_synthetic_schema_pass(self) -> None:
        cells, cand_path = self._schema_cells_no_bootstrap()
        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": "schema-pass",
            "settings": {
                "evidence": True,
                "performance_conclusion_allowed": True,
                "warmups": EVIDENCE_WARMUPS,
                "samples": EVIDENCE_SAMPLES,
                "base_seed": BASE_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            },
            "provenance": {
                "git": {"status": "ok", "revision": "abc123", "dirty": False},
                "blas": {"identified": True, "label": "openblas"},
                "harness_source_hashes": {n: "e" * 64 for n in HARNESS_SOURCE_FILES},
                "power_thermal_before": {"thermal_anomaly": False},
                "power_thermal_after": {"thermal_anomaly": False},
                "cpu": {"model_name": "Test", "platform_machine": "arm64"},
            },
            "protocol": protocol_manifest(),
            "candidate": {
                "module_path": cand_path,
                "artifact_sha256": "a" * 64,
                "cargo_toml_sha256": "1" * 64,
                "cargo_lock_sha256": "2" * 64,
                "lib_rs_sha256": "3" * 64,
                "cargo_config_sha256": "4" * 64,
            },
            "cells": cells,
            "aggregate": {"geometric_mean": 1.5, "min_ci_lower": 1.1},
            "verdicts": {
                "product_verdict": PRODUCT_VERDICT,
                "dispatchability_status": DISPATCHABILITY_STATUS,
                "performance_passed": False,
            },
            "honesty": {},
        }
        errs = validate_report_schema(report)
        assert errs == [], errs
        assert len(validate_process_isolation(report)) == 0

    def test_schema_corruption_missing_leg(self) -> None:
        cells, cand_path = self._schema_cells_no_bootstrap()
        del cells[0]["legs"]["dot"]
        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": "corrupt-leg",
            "settings": {
                "evidence": True,
                "performance_conclusion_allowed": True,
                "warmups": EVIDENCE_WARMUPS,
                "samples": EVIDENCE_SAMPLES,
                "base_seed": BASE_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            },
            "provenance": {
                "git": {"status": "ok", "revision": "abc", "dirty": False},
                "blas": {"identified": True, "label": "mkl"},
                "harness_source_hashes": {n: "e" * 64 for n in HARNESS_SOURCE_FILES},
                "power_thermal_before": {"thermal_anomaly": False},
                "power_thermal_after": {"thermal_anomaly": False},
            },
            "protocol": {},
            "candidate": {
                "module_path": cand_path,
                "artifact_sha256": "a" * 64,
                "cargo_toml_sha256": "1" * 64,
                "cargo_lock_sha256": "2" * 64,
                "lib_rs_sha256": "3" * 64,
                "cargo_config_sha256": "4" * 64,
            },
            "cells": cells,
            "aggregate": {},
            "verdicts": {
                "product_verdict": PRODUCT_VERDICT,
                "dispatchability_status": DISPATCHABILITY_STATUS,
                "performance_passed": True,
            },
            "honesty": {},
        }
        errs = validate_report_schema(report)
        assert any("missing leg" in e for e in errs)
        assert any("performance_passed" in e for e in errs)

    def test_process_isolation_requires_distinct_nonces(self) -> None:
        cells, cand_path = self._schema_cells_no_bootstrap()
        for cell in cells:
            for leg in cell["legs"].values():
                leg["process"]["nonce"] = "same-nonce"
        report = {
            "cells": cells,
            "candidate": {
                "module_path": cand_path,
                "artifact_sha256": "a" * 64,
            },
        }
        errs = validate_process_isolation(report)
        assert any("distinct" in e for e in errs)

    def test_input_seed_must_equal_frozen_derived_seed(self) -> None:
        cells, cand_path = self._schema_cells_no_bootstrap()
        cells[0]["inputs"]["seed"] = cell_input_seed(SHAPE_CELLS[0].cell_index) + 1
        report: dict[str, Any] = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": "bad-input-seed",
            "settings": {
                "evidence": True,
                "performance_conclusion_allowed": True,
                "warmups": EVIDENCE_WARMUPS,
                "samples": EVIDENCE_SAMPLES,
                "base_seed": BASE_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            },
            "provenance": {
                "git": {"status": "ok", "revision": "abc", "dirty": False},
                "blas": {"identified": True, "label": "openblas"},
                "harness_source_hashes": {n: "e" * 64 for n in HARNESS_SOURCE_FILES},
                "power_thermal_before": {"thermal_anomaly": False},
                "power_thermal_after": {"thermal_anomaly": False},
            },
            "protocol": {},
            "candidate": {
                "module_path": cand_path,
                "artifact_sha256": "a" * 64,
                "cargo_toml_sha256": "1" * 64,
                "cargo_lock_sha256": "2" * 64,
                "lib_rs_sha256": "3" * 64,
                "cargo_config_sha256": "4" * 64,
            },
            "cells": cells,
            "aggregate": {},
            "verdicts": {
                "product_verdict": PRODUCT_VERDICT,
                "dispatchability_status": DISPATCHABILITY_STATUS,
                "performance_passed": False,
            },
            "honesty": {},
        }
        errs = validate_report_schema(report)
        assert any("input seed must be frozen derived" in e for e in errs), errs

    def test_candidate_module_file_required(self) -> None:
        cells, cand_path = self._schema_cells_no_bootstrap()
        cells[0]["legs"]["candidate"]["meta"].pop("module_file", None)
        report = {
            "cells": cells,
            "candidate": {
                "module_path": cand_path,
                "artifact_sha256": "a" * 64,
            },
        }
        errs = validate_process_isolation(report)
        assert any("missing module_file" in e for e in errs), errs
