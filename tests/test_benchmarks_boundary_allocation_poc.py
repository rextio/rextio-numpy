"""Tests for the experimental F64 rank-1 boundary-allocation PoC harness."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from benchmarks.boundary_allocation_poc import PROTOCOL_ID, SCHEMA_VERSION  # noqa: E402
from benchmarks.boundary_allocation_poc.candidate import (  # noqa: E402
    CandidateArtifact,
    RUST_CANDIDATE_DIR,
    is_secret_env_key,
    record_build_env,
    validate_loaded_module,
)
from benchmarks.boundary_allocation_poc.protocol import (  # noqa: E402
    CONTIGUOUS_SIZES,
    EXACT_NDARRAY_TYPEERROR,
    F64_BYTES,
    HONESTY_CAVEATS,
    LOGICAL_ALLOC_FORMULAS,
    LOGICAL_N_ALLOCS_EQUAL_CONTIG,
    RUST_STRATEGIES,
    STRATEGY_FUNCTIONS,
    STRATEGIES,
    logical_allocs_for,
    protocol_manifest,
)
from benchmarks.boundary_allocation_poc.runner import (  # noqa: E402
    build_arg_parser,
    run_harness,
)
from benchmarks.boundary_allocation_poc.semantics import (  # noqa: E402
    assert_ownership,
    ndarray_owns_data,
    python_add,
    require_subclass_rejection,
    resolve_strategy_fn,
    run_semantic_suite,
)


# ---------------------------------------------------------------------------
# Static protocol / honesty
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_protocol_ids_stable(self) -> None:
        assert PROTOCOL_ID == "boundary-allocation-poc-f64-r1-2026-07-27"
        assert SCHEMA_VERSION == "boundary-allocation-poc-v1"

    def test_predeclared_contiguous_sizes(self) -> None:
        assert CONTIGUOUS_SIZES == (1_000, 100_000, 1_000_000, 10_000_000)

    def test_strategies_include_three_rust_and_python_ref(self) -> None:
        assert STRATEGIES == (
            "owned_topy",
            "borrowed_topy",
            "direct_sink",
            "python_ref",
        )
        assert RUST_STRATEGIES == ("owned_topy", "borrowed_topy", "direct_sink")

    def test_logical_alloc_formulas_and_counts(self) -> None:
        assert LOGICAL_N_ALLOCS_EQUAL_CONTIG["owned_topy"] == 4
        assert LOGICAL_N_ALLOCS_EQUAL_CONTIG["borrowed_topy"] == 2
        assert LOGICAL_N_ALLOCS_EQUAL_CONTIG["direct_sink"] == 1
        assert LOGICAL_N_ALLOCS_EQUAL_CONTIG["python_ref"] == 1
        for sid in STRATEGIES:
            assert sid in LOGICAL_ALLOC_FORMULAS
            rec = logical_allocs_for(sid, 1000)
            assert rec["logical_bytes"] == LOGICAL_N_ALLOCS_EQUAL_CONTIG[sid] * 1000 * F64_BYTES
            assert "N" in rec["formula"] or "N" in LOGICAL_ALLOC_FORMULAS[sid]

    def test_honesty_caveats_mention_allocator_and_no_speedup(self) -> None:
        blob = " ".join(HONESTY_CAVEATS).lower()
        assert "allocator" in blob
        assert "zero-initialization" in blob or "zero-initializes" in blob
        assert "speedup" in blob or "speed" in blob
        assert "intopyarray" in blob
        # Shared kernel honesty (benchmark-validity: allocation vs arithmetic).
        assert "fill_add_views" in blob
        assert "shared" in blob and "element order" in blob
        # Fixed-order / unpaired timing and thread-env honesty (review finding 4).
        assert "fixed-order" in blob
        assert "unpaired" in blob
        assert "cache" in blob
        assert "thermal" in blob
        assert "order bias" in blob
        assert "requested configuration" in blob
        assert "already imported" in blob
        # Zero-init is common to all three Rust strategies (not direct_sink-only).
        assert "zeros" in blob and "store pass" in blob
        assert "array1::zeros" in blob
        assert "pyarray::zeros" in blob
        # The PoC's owned lane is now a historical baseline, not a description
        # of the current product F64 rank-1 boundary.
        assert "historical owned-copy" in blob
        assert "current product f64 rank-1" in blob
        assert "production boundaryconversion remains owned-copy" not in blob

    def test_protocol_manifest_flags(self) -> None:
        m = protocol_manifest()
        assert m["performance_claim"] is False
        assert m["product_surface_change"] is False
        assert m["contiguous_sizes"] == list(CONTIGUOUS_SIZES)
        caveats = " ".join(m["honesty_caveats"]).lower()
        assert "fixed-order" in caveats
        assert "requested configuration" in caveats

    def test_exact_ndarray_message_matches_product(self) -> None:
        assert "exact numpy.ndarray" in EXACT_NDARRAY_TYPEERROR
        assert "subclasses are unsupported" in EXACT_NDARRAY_TYPEERROR


class TestCargoTomlStatic:
    def test_pins_and_no_intopyarray_in_source(self) -> None:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore

        cargo_path = RUST_CANDIDATE_DIR / "Cargo.toml"
        data = tomllib.loads(cargo_path.read_text(encoding="utf-8"))
        deps = data.get("dependencies") or {}
        assert deps.get("numpy") == "=0.29.0"
        pyo3 = deps.get("pyo3")
        assert isinstance(pyo3, dict)
        assert pyo3.get("version") == "=0.29.0"
        assert pyo3.get("features") == ["extension-module"]
        lib_rs = (RUST_CANDIDATE_DIR / "src" / "lib.rs").read_text(encoding="utf-8")
        assert "IntoPyArray" not in lib_rs or "never" in lib_rs.lower()
        # Stronger: no use of into_pyarray symbol call.
        assert "into_pyarray" not in lib_rs
        assert "ToPyArray" in lib_rs
        assert "add_owned_topy" in lib_rs
        assert "add_borrowed_topy" in lib_rs
        assert "add_direct_sink" in lib_rs
        # Zero-init output on all three strategies (aligned store policy).
        assert "Array1::<f64>::zeros" in lib_rs
        assert "PyArray1::<f64>::zeros" in lib_rs
        assert "store pass" in lib_rs.lower() or "extra store" in lib_rs.lower()
        assert (RUST_CANDIDATE_DIR / ".cargo" / "config.toml").is_file()


class TestSharedFillKernelStructural:
    """Benchmark-validity gate: one arithmetic kernel for all Rust strategies.

    owned_topy / borrowed_topy must not keep ndarray ``+`` / ``mapv`` paths that
    bypass ``fill_add_views`` while direct_sink uses Zip/manual loops.
    """

    def _lib_rs(self) -> str:
        return (RUST_CANDIDATE_DIR / "src" / "lib.rs").read_text(encoding="utf-8")

    def test_single_fill_add_views_definition(self) -> None:
        import re

        lib_rs = self._lib_rs()
        defs = re.findall(
            r"^\s*fn\s+fill_add_views\s*\(",
            lib_rs,
            flags=re.MULTILINE,
        )
        assert len(defs) == 1, f"expected exactly one fill_add_views definition, got {len(defs)}"

    def test_every_exported_strategy_calls_fill_add_views(self) -> None:
        lib_rs = self._lib_rs()
        # Count the call token only (trailing '('). Doc/module comments mention
        # fill_add_views in backticks without '(', so they cannot inflate this.
        # Exactly four: one definition (fn fill_add_views(...)) + three strategy
        # call sites. A comment-only mention must not satisfy this gate.
        token = "fill_add_views("
        assert lib_rs.count(token) == 4, (
            f"expected exactly 4 fill_add_views( occurrences "
            f"(1 def + 3 calls), got {lib_rs.count(token)}"
        )
        # Exact call forms (actual Rust spacing/syntax in lib.rs).
        owned_call = "fill_add_views(slice, a_owned.view(), b_owned.view())"
        borrow_or_sink_call = "fill_add_views(slice, a_view, b_view)"
        assert lib_rs.count(owned_call) == 1, (
            "owned_topy must call fill_add_views(slice, a_owned.view(), "
            "b_owned.view()) exactly once"
        )
        assert lib_rs.count(borrow_or_sink_call) == 2, (
            "borrowed_topy and direct_sink must each call "
            "fill_add_views(slice, a_view, b_view) (exactly two total)"
        )
        # Three exact call forms + one definition consume all four tokens.
        assert lib_rs.count(owned_call) + lib_rs.count(borrow_or_sink_call) == 3

    def test_no_alternate_arithmetic_or_mapv_kernel(self) -> None:
        lib_rs = self._lib_rs()
        assert "add_views_owned" not in lib_rs
        assert ".mapv(" not in lib_rs
        assert "mapv(" not in lib_rs
        # Forbid owned-array / operator-add kernel forms used previously.
        for banned in (
            "&a + &b",
            "&a_owned + &b_owned",
            "Ok(&a + &b)",
            "return Ok(&a + &b)",
            "a.mapv",
            "b.mapv",
        ):
            assert banned not in lib_rs


class TestCandidateHelpers:
    def test_secret_env_and_path_recording(self) -> None:
        assert is_secret_env_key("CARGO_REGISTRIES_CRATES_IO_TOKEN")
        rec = record_build_env(
            {
                "PATH": "/secret/bin",
                "PYO3_PYTHON": "/venv/bin/python",
                "CARGO_REGISTRIES_CRATES_IO_TOKEN": "super-secret",
            }
        )
        assert rec.get("PATH") == "(set)"
        assert rec.get("PYO3_PYTHON") == "/venv/bin/python"
        assert "CARGO_REGISTRIES_CRATES_IO_TOKEN" not in rec

    def test_validate_loaded_module_path_mismatch(self, tmp_path: Path) -> None:
        p = tmp_path / "mod.so"
        p.write_bytes(b"x")
        err = validate_loaded_module(
            str(p),
            expected_path=tmp_path / "other.so",
            expected_sha256="0" * 64,
        )
        assert err is not None
        assert "mismatch" in err.lower() or "!=" in err


def _install_fake_candidate(stage_dir: Path) -> CandidateArtifact:
    """Pure-Python stand-in implementing the three strategies + ownership."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    module_path = stage_dir / "boundary_allocation_poc.py"
    source = textwrap.dedent(
        '''
        import numpy as np

        _MSG = (
            "rextio-numpy native boundary requires exact numpy.ndarray; "
            "ndarray subclasses are unsupported"
        )

        def _exact(a, b):
            if type(a) is not np.ndarray or type(b) is not np.ndarray:
                raise TypeError(_MSG)

        def _add(a, b):
            return a + b

        def add_owned_topy(a, b):
            _exact(a, b)
            # Simulate owned copies + result + to_pyarray materialization.
            aa = np.array(a, copy=True)
            bb = np.array(b, copy=True)
            return np.array(_add(aa, bb), copy=True)

        def add_borrowed_topy(a, b):
            _exact(a, b)
            return np.array(_add(a, b), copy=True)

        def add_direct_sink(a, b):
            _exact(a, b)
            # Single NumPy-owned buffer filled in place. Let a+b raise the
            # exact NumPy broadcast ValueError (trailing space included).
            values = _add(a, b)
            out = np.empty(values.shape, dtype=np.float64)
            out[:] = values
            return out
        '''
    ).lstrip()
    module_path.write_text(source, encoding="utf-8")
    import hashlib

    sha = hashlib.sha256(module_path.read_bytes()).hexdigest()
    return CandidateArtifact(
        module_name="boundary_allocation_poc",
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
        profile="release",
        build_env={},
        strategy_functions={
            str(sid): STRATEGY_FUNCTIONS[sid] for sid in RUST_STRATEGIES
        },
    )


class TestOwnershipHardGates:
    def test_assert_ownership_hard_fails_on_view(self) -> None:
        base = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        view = base[:]  # shares buffer; not ordinary owned result
        assert view.base is not None or not ndarray_owns_data(view)
        with pytest.raises(AssertionError, match="OWNDATA|base must be None"):
            assert_ownership(view, label="view")

    def test_assert_ownership_hard_fails_without_owndata(self) -> None:
        base = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        view = base[::2]
        if ndarray_owns_data(view) and view.base is None:
            pytest.skip("this NumPy/view layout still looks fully owned")
        with pytest.raises(AssertionError, match="OWNDATA|base must be None"):
            assert_ownership(view, label="no-own")

    def test_assert_ownership_accepts_fresh_owned(self) -> None:
        # Pass a temporary so resize refcheck sees a single live reference.
        assert_ownership(
            np.array([1.0, 2.0], dtype=np.float64),
            label="fresh",
        )


class TestSemanticsWithFakeCandidate:
    def test_semantic_suite_covers_required_cases(self, tmp_path: Path) -> None:
        art = _install_fake_candidate(tmp_path / "stage")
        from benchmarks.boundary_allocation_poc.candidate import ensure_importable

        mod = ensure_importable(art)
        records = run_semantic_suite(mod)
        cases = {r["case"] for r in records}
        required = {
            "exact_acceptance_ownership",
            "subclass_rejection",
            "readonly_inputs",
            "same_object_aliasing",
            "zero_length",
            "nan_inf",
            "length1_broadcast",
            "incompatible_broadcast",
            "strided_equal_length",
        }
        assert required.issubset(cases)
        assert all(r.get("ok") for r in records)
        # Hard ownership gate: every acceptance record must assert true ownership.
        own_recs = [r for r in records if r["case"] == "exact_acceptance_ownership"]
        assert len(own_recs) == len(STRATEGIES)
        for r in own_recs:
            assert r["owndata"] is True
            assert r["base_is_none"] is True
            assert r["ok"] is True
            # Never soft-record a failed resize.
            assert r["resize_ok"] in (True, None)

    def test_ownership_records_all_strategies_including_python_ref(
        self, tmp_path: Path
    ) -> None:
        art = _install_fake_candidate(tmp_path / "stage")
        from benchmarks.boundary_allocation_poc.candidate import ensure_importable

        mod = ensure_importable(art)
        records = run_semantic_suite(mod)
        by_sid = {
            r["strategy"]: r
            for r in records
            if r["case"] == "exact_acceptance_ownership"
        }
        for sid in STRATEGIES:
            assert sid in by_sid
            assert by_sid[sid]["owndata"] is True
            assert by_sid[sid]["base_is_none"] is True

    def test_direct_sink_ownership_and_resize(self, tmp_path: Path) -> None:
        art = _install_fake_candidate(tmp_path / "stage")
        from benchmarks.boundary_allocation_poc.candidate import ensure_importable

        mod = ensure_importable(art)
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([3.0, 4.0], dtype=np.float64)
        out = mod.add_direct_sink(a, b)
        assert type(out) is np.ndarray
        assert ndarray_owns_data(out)
        assert out.base is None
        # Temporary-only call for single-ref resize gate.
        assert_ownership(mod.add_direct_sink(a, b), label="direct_sink")

    def test_subclass_rejection_either_position_all_rust_strategies(
        self, tmp_path: Path
    ) -> None:
        art = _install_fake_candidate(tmp_path / "stage")
        from benchmarks.boundary_allocation_poc.candidate import ensure_importable

        mod = ensure_importable(art)

        class Sub(np.ndarray):
            pass

        exact_a = np.array([1.0, 2.0], dtype=np.float64)
        exact_b = np.array([3.0, 4.0], dtype=np.float64)
        sub_a = exact_a.view(Sub)
        sub_b = exact_b.view(Sub)
        for sid in RUST_STRATEGIES:
            fn = resolve_strategy_fn(mod, sid)
            recs = require_subclass_rejection(
                fn,
                exact_a=exact_a,
                exact_b=exact_b,
                sub_a=sub_a,
                sub_b=sub_b,
                strategy=sid,
            )
            assert {r["position"] for r in recs} == {"left", "right"}
            for r in recs:
                assert r["message"] == EXACT_NDARRAY_TYPEERROR

        # Semantic suite must also record both positions per Rust strategy.
        suite = run_semantic_suite(mod)
        sub_recs = [r for r in suite if r["case"] == "subclass_rejection"]
        assert len(sub_recs) == len(RUST_STRATEGIES) * 2
        for sid in RUST_STRATEGIES:
            positions = {
                r["position"] for r in sub_recs if r["strategy"] == sid
            }
            assert positions == {"left", "right"}

    def test_broadcast_error_trailing_space(self, tmp_path: Path) -> None:
        art = _install_fake_candidate(tmp_path / "stage")
        from benchmarks.boundary_allocation_poc.candidate import ensure_importable

        mod = ensure_importable(art)
        a = np.ones(3, dtype=np.float64)
        b = np.ones(4, dtype=np.float64)
        try:
            a + b
        except ValueError as np_exc:
            np_msg = str(np_exc)
        for name in ("add_owned_topy", "add_borrowed_topy", "add_direct_sink"):
            with pytest.raises(ValueError) as ei:
                getattr(mod, name)(a, b)
            assert str(ei.value) == np_msg
            assert str(ei.value).endswith(" ")

    def test_python_ref_lane(self) -> None:
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([3.0, 4.0], dtype=np.float64)
        np.testing.assert_array_equal(python_add(a, b), a + b)
        fn = resolve_strategy_fn(object(), "python_ref")
        np.testing.assert_array_equal(fn(a, b), a + b)
        assert_ownership(python_add(a, b), label="python_ref")  # temporary


class TestHarnessSmokeMocked:
    def test_run_harness_smoke_with_fake_build(self, tmp_path: Path) -> None:
        def fake_build(stage_dir: Any, **kwargs: Any) -> CandidateArtifact:
            return _install_fake_candidate(Path(stage_dir))

        report = run_harness(
            output_dir=tmp_path / "out",
            smoke=True,
            sizes=(32,),
            calibrate=False,
            fixed_iterations=1,
            warmups=0,
            samples=2,
            build_fn=fake_build,
            run_id="mock-smoke",
        )
        assert report["performance_claim"] is False
        assert report["speedup_claim"] is None
        assert report["protocol_id"] == PROTOCOL_ID
        assert len(report["semantics"]) > 0
        # Ownership hard gate must have passed for every strategy.
        own = [
            r
            for r in report["semantics"]
            if r["case"] == "exact_acceptance_ownership"
        ]
        assert len(own) == len(STRATEGIES)
        assert all(r["owndata"] is True and r["base_is_none"] is True for r in own)
        caveats = " ".join(report["honesty_caveats"]).lower()
        assert "fixed-order" in caveats
        assert "requested configuration" in caveats
        assert len(report["timing_cells"]) == 1
        cell = report["timing_cells"][0]
        assert cell["n"] == 32
        assert cell["headline"] is True
        for sid in STRATEGIES:
            assert sid in cell["strategies"]
            assert "logical" in cell["strategies"][sid]
            assert "timing" in cell["strategies"][sid]
        paths = report["paths"]
        assert Path(paths["json"]).is_file()
        assert Path(paths["markdown"]).is_file()
        md = Path(paths["markdown"]).read_text(encoding="utf-8")
        assert "Performance claim: **false**" in md
        assert "no speedup claim" in md.lower() or "Performance claim" in md

    def test_arg_parser_requires_output_dir(self) -> None:
        p = build_arg_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not on PATH")
class TestRealCargoBoundedSmoke:
    def test_real_cargo_semantics_and_tiny_timing(
        self, tmp_path: Path
    ) -> None:
        """Bounded real build + semantics + N=1000 smoke when cargo is present."""
        report = run_harness(
            output_dir=tmp_path / "real",
            smoke=True,
            sizes=(1_000,),
            calibrate=False,
            fixed_iterations=1,
            warmups=1,
            samples=2,
            run_id="real-cargo-bap",
        )
        assert report["performance_claim"] is False
        assert all(r.get("ok") for r in report["semantics"])
        own = [
            r
            for r in report["semantics"]
            if r["case"] == "exact_acceptance_ownership"
        ]
        assert len(own) == len(STRATEGIES)
        assert all(r["owndata"] is True and r["base_is_none"] is True for r in own)
        sub = [r for r in report["semantics"] if r["case"] == "subclass_rejection"]
        assert len(sub) == len(RUST_STRATEGIES) * 2
        assert report["timing_cells"][0]["n"] == 1_000
        cand = report["candidate"]
        assert Path(cand["module_path"]).is_file()
        assert len(cand["artifact_sha256"]) == 64
        lib_rs = (RUST_CANDIDATE_DIR / "src" / "lib.rs").read_text(encoding="utf-8")
        assert "into_pyarray" not in lib_rs
        assert "PyArray1::<f64>::zeros" in lib_rs
        assert "Array1::<f64>::zeros" in lib_rs
        assert "fill_add_views(" in lib_rs
        assert ".mapv(" not in lib_rs
        # Direct sink ownership on a real build.
        from benchmarks.boundary_allocation_poc.candidate import ensure_importable
        from benchmarks.boundary_allocation_poc.candidate import CandidateArtifact as CA

        # Re-import from staged path recorded in report.
        art = CA(
            module_name=cand["module_name"],
            load_dir=cand["load_dir"],
            module_path=cand["module_path"],
            artifact_sha256=cand["artifact_sha256"],
            build_wall_s=cand["build_wall_s"],
            cargo_stdout=cand.get("cargo_stdout", ""),
            cargo_stderr=cand.get("cargo_stderr", ""),
            crate_dir=cand["crate_dir"],
            cargo_toml_sha256=cand["cargo_toml_sha256"],
            cargo_lock_sha256=cand["cargo_lock_sha256"],
            lib_rs_sha256=cand["lib_rs_sha256"],
            cargo_config_sha256=cand["cargo_config_sha256"],
            rustc_verbose=cand.get("rustc_verbose"),
            cargo_version=cand.get("cargo_version"),
            profile=cand.get("profile", "release"),
            build_env=cand.get("build_env") or {},
            strategy_functions=cand.get("strategy_functions") or {},
        )
        mod = ensure_importable(art)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        out = mod.add_direct_sink(a, b)
        assert type(out) is np.ndarray
        assert ndarray_owns_data(out)
        assert out.base is None
        np.testing.assert_array_equal(out, a + b)
        assert_ownership(mod.add_direct_sink(a, b), label="real_direct_sink")

        # Runtime value parity across all three strategies + NumPy reference
        # (shared fill kernel must not diverge by allocation policy).
        ref = a + b
        outs = {
            "owned_topy": mod.add_owned_topy(a, b),
            "borrowed_topy": mod.add_borrowed_topy(a, b),
            "direct_sink": mod.add_direct_sink(a, b),
        }
        for sid, arr in outs.items():
            np.testing.assert_array_equal(arr, ref, err_msg=f"{sid} vs numpy")
        np.testing.assert_array_equal(outs["owned_topy"], outs["borrowed_topy"])
        np.testing.assert_array_equal(outs["borrowed_topy"], outs["direct_sink"])
        # Ownership gates need a single live reference (may resize).
        assert_ownership(mod.add_owned_topy(a, b), label="parity_owned_topy")
        assert_ownership(mod.add_borrowed_topy(a, b), label="parity_borrowed_topy")
        assert_ownership(mod.add_direct_sink(a, b), label="parity_direct_sink")

        # Broadcast / strided / NaN-Inf parity under shared kernel.
        a1 = np.array([2.0], dtype=np.float64)
        b_n = np.array([1.0, 3.0, 5.0], dtype=np.float64)
        ref_bc = a1 + b_n
        for name in ("add_owned_topy", "add_borrowed_topy", "add_direct_sink"):
            np.testing.assert_array_equal(getattr(mod, name)(a1, b_n), ref_bc)
            np.testing.assert_array_equal(getattr(mod, name)(b_n, a1), ref_bc)
        base = np.arange(0.0, 12.0, dtype=np.float64)
        a_str = base[::2]
        b_str = base[1 : 1 + 2 * a_str.shape[0] : 2]
        assert a_str.shape == b_str.shape
        ref_st = a_str + b_str
        for name in ("add_owned_topy", "add_borrowed_topy", "add_direct_sink"):
            np.testing.assert_array_equal(getattr(mod, name)(a_str, b_str), ref_st)
        a_nan = np.array([np.nan, 1.0, np.inf, -np.inf], dtype=np.float64)
        b_nan = np.array([0.0, np.nan, 1.0, 2.0], dtype=np.float64)
        ref_nan = a_nan + b_nan
        for name in ("add_owned_topy", "add_borrowed_topy", "add_direct_sink"):
            np.testing.assert_array_equal(getattr(mod, name)(a_nan, b_nan), ref_nan)

        # Subclass either position for every real Rust strategy.

        class Sub(np.ndarray):
            pass

        a_sub = a.view(Sub)
        b_sub = b.view(Sub)
        for sid in RUST_STRATEGIES:
            fn = resolve_strategy_fn(mod, sid)
            require_subclass_rejection(
                fn,
                exact_a=a,
                exact_b=b,
                sub_a=a_sub,
                sub_b=b_sub,
                strategy=sid,
            )
