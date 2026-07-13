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


def test_rust_snippets_package_public_api() -> None:
    # Same import path as the pre-split module.
    assert callable(rust_snippets.dot1)
    assert callable(rust_snippets.sum1)
    assert callable(rust_snippets.mean1)
    assert callable(rust_snippets.elementwise_aa)
    assert callable(rust_snippets.elementwise_as)
    assert callable(rust_snippets.elementwise_sa)
    assert rust_snippets.OP_SYMBOLS == {"add": "+", "sub": "-", "mul": "*", "div": "/"}
    assert "fn __rxtnp_dot1" in rust_snippets.dot1()
    assert "fn __rxtnp_sum1" in rust_snippets.sum1()
    assert "fn __rxtnp_mean1" in rust_snippets.mean1()
    assert "fn __rxtnp_add1_aa" in rust_snippets.elementwise_aa("add")
