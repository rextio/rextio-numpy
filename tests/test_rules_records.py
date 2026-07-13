"""Focused checks for the rules package boundary and public imports."""

from __future__ import annotations

from rextio_numpy import rust_snippets
from rextio_numpy.rules import COVERAGE, numpy_rule_records
from rextio_numpy.rules.coverage import COVERAGE as COVERAGE_DIRECT
from rextio_numpy.rules.records_fallback import FALLBACK_RECORDS
from rextio_numpy.rules.records_native import NATIVE_RECORDS


def test_public_coverage_import() -> None:
    assert COVERAGE is COVERAGE_DIRECT
    assert COVERAGE.packages == ("numpy",)
    assert COVERAGE.modules == ("numpy",)
    assert COVERAGE.symbols == (
        "numpy.ndarray",
        "numpy.dot",
        "numpy.sum",
        "numpy.mean",
        "numpy.max",
        "numpy.min",
    )


def test_numpy_rule_records_order_and_ids() -> None:
    records = numpy_rule_records()
    ids = [record.id for record in records]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    # Combined native + fallback, sorted by id.
    combined = sorted(
        (*NATIVE_RECORDS, *FALLBACK_RECORDS),
        key=lambda record: record.id,
    )
    assert [r.id for r in records] == [r.id for r in combined]
    assert [r.diagnostic_code for r in records] == [r.diagnostic_code for r in combined]
    assert [r.outcome for r in records] == [r.outcome for r in combined]
    assert [r.guidance for r in records] == [r.guidance for r in combined]
    assert [r.constraint for r in records] == [r.constraint for r in combined]


def test_native_and_fallback_split() -> None:
    assert {r.outcome for r in NATIVE_RECORDS} == {"native"}
    assert all(r.verified is True for r in NATIVE_RECORDS)
    assert {r.outcome for r in FALLBACK_RECORDS} == {"fallback"}
    assert all(r.verified is None for r in FALLBACK_RECORDS)


def test_native_records_broadened_but_ids_stable() -> None:
    by_id = {r.id: r for r in NATIVE_RECORDS}
    assert set(by_id) == {
        "rextio-numpy/elementwise-float64",
        "rextio-numpy/dot-float64",
        "rextio-numpy/reduction-sum-mean",
        "rextio-numpy/reduction-axis",
        "rextio-numpy/elementwise-chain-fusion",
    }
    elem = by_id["rextio-numpy/elementwise-float64"]
    assert elem.diagnostic_code == "RXTP-NUMPY-001"
    assert "rank 1 or 2" in elem.scope.pattern
    assert "int64" in elem.constraint
    assert "broadcast" in elem.constraint.lower()
    dot = by_id["rextio-numpy/dot-float64"]
    assert dot.diagnostic_code == "RXTP-NUMPY-002"
    assert "2-D" in dot.scope.pattern or "2-D" in dot.constraint
    # float32 is elementwise-only; sum/mean/dot claims must not cover it.
    assert "float32" in dot.scope.pattern or "float32" in dot.constraint
    assert "rejected" in dot.constraint.lower() or "not claimed" in dot.scope.pattern
    red = by_id["rextio-numpy/reduction-sum-mean"]
    assert red.diagnostic_code == "RXTP-NUMPY-003"
    assert "int64 sum" in red.constraint or "float64/int64" in red.constraint
    assert "float32" in red.scope.pattern or "float32" in red.constraint
    assert "int64 mean" in red.scope.pattern or "int64 mean" in red.constraint
    assert "rejected" in red.constraint.lower() or "not claimed" in red.scope.pattern
    # Native rule must not advertise verified int64 mean.
    assert "int64 mean is float64" not in red.constraint
    axis = by_id["rextio-numpy/reduction-axis"]
    assert axis.diagnostic_code == "RXTP-NUMPY-004"
    assert "axis=" in axis.scope.pattern
    assert "max" in axis.scope.pattern and "min" in axis.scope.pattern
    assert "RuntimeWarning" in axis.constraint
    assert "no identity" in axis.constraint
    assert axis.verified is True
    fusion = by_id["rextio-numpy/elementwise-chain-fusion"]
    assert fusion.diagnostic_code == "RXTP-NUMPY-005"
    assert "2–8" in fusion.scope.pattern or "2-8" in fusion.scope.pattern
    assert "operand_mode" in fusion.scope.pattern or "leaves" in fusion.scope.pattern
    assert "wrapping" in fusion.constraint.lower()
    assert fusion.verified is True


def test_fallback_ndim_is_rank_gt_2() -> None:
    ndim = next(r for r in FALLBACK_RECORDS if r.id == "rextio-numpy/unsupported-ndim")
    assert ndim.diagnostic_code == "RXTP-NUMPY-011"
    assert "more than 2" in ndim.scope.pattern


def test_rust_snippets_package_public_api() -> None:
    # Same import path as the pre-split module.
    assert callable(rust_snippets.dot1)
    assert callable(rust_snippets.sum1)
    assert callable(rust_snippets.mean1)
    assert callable(rust_snippets.elementwise_aa)
    assert callable(rust_snippets.elementwise_as)
    assert callable(rust_snippets.elementwise_sa)
    assert callable(rust_snippets.fusion_call_name)
    assert callable(rust_snippets.fusion_helpers_bundle)
    assert rust_snippets.OP_SYMBOLS == {"add": "+", "sub": "-", "mul": "*", "div": "/"}
    assert "fn __rxtnp_dot1" in rust_snippets.dot1()
    assert "fn __rxtnp_sum1" in rust_snippets.sum1()
    assert "fn __rxtnp_mean1" in rust_snippets.mean1()
    assert "fn __rxtnp_add1_aa" in rust_snippets.elementwise_aa("add")
    assert "wrapping_add" in rust_snippets.elementwise_aa_typed("add", "i64", 1, 1)
    assert "broadcast" in rust_snippets.broadcast_shape_helper()
    assert rust_snippets.axis_call_name("sum", "f64", 2, 1) == "__rxtnp_sum2_f64_axis1"
    helpers = rust_snippets.axis_typed("max", "f64", 2, 0)
    joined = "\n".join(helpers)
    assert "__rxtnp_max2_f64_axis0" in joined
    assert "maximum which has no identity" in joined
    assert rust_snippets.op_from_target("numpy.min") == "min"
