"""Forged lower-time metadata must fail closed for every native NumPy lane."""

from __future__ import annotations

from dataclasses import replace
import subprocess
import sys

import pytest

from rextio.plugins.api import (
    CallableMeta,
    ClaimExpr,
    ClaimLiteral,
    ClaimSite,
    LoweringContext,
    ReceiverMeta,
)

from rextio_numpy.claim.fusion import FUSION_RULE
from rextio_numpy.diagnostics import F64_1D
from rextio_numpy.lower import lower

_CONTEXT_FIELDS = [("target_language", "wasm")]
if "backend" in LoweringContext.__dataclass_fields__:
    _CONTEXT_FIELDS.append(("backend", "abi3"))


def _ctx(*, operands: tuple[str, ...], leaves: tuple[str, ...] = ()) -> LoweringContext:
    return LoweringContext(
        operands=operands,
        leaf_operands=leaves,
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )


def _fusion_site() -> ClaimSite:
    expression = ClaimExpr(
        kind="binop",
        target="*",
        result_type=F64_1D,
        children=(
            ClaimExpr(
                kind="binop",
                target="+",
                result_type=F64_1D,
                children=(
                    ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=0, leaf_kind="name"),
                    ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=1, leaf_kind="name"),
                ),
            ),
            ClaimExpr(
                kind="binop",
                target="-",
                result_type=F64_1D,
                children=(
                    ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=2, leaf_kind="name"),
                    ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=3, leaf_kind="name"),
                ),
            ),
        ),
    )
    return ClaimSite(
        kind="binop",
        target="*",
        operand_types=(F64_1D, F64_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=FUSION_RULE,
        result_type=F64_1D,
        expression=expression,
    )


@pytest.mark.parametrize(
    ("site", "ctx"),
    [
        (
            ClaimSite(
                kind="binop",
                target="+",
                operand_types=(F64_1D, F64_1D),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/elementwise-float64",
                result_type=F64_1D,
            ),
            _ctx(operands=("a", "b")),
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.dot",
                operand_types=(F64_1D, F64_1D),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/dot-float64",
                result_type="float",
            ),
            _ctx(operands=("a", "b")),
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.sum",
                operand_types=(F64_1D,),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/reduction-sum-mean",
                result_type="float",
            ),
            _ctx(operands=("a",)),
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.negative",
                operand_types=(F64_1D,),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/unary-module",
                result_type=F64_1D,
            ),
            _ctx(operands=("a",)),
        ),
        (_fusion_site(), _ctx(operands=(), leaves=("a", "b", "c", "d"))),
    ],
)
def test_lower_rejects_missing_claim_rule(
    site: ClaimSite,
    ctx: LoweringContext,
) -> None:
    with pytest.raises(ValueError, match="rule_id"):
        lower(replace(site, rule_id=None), ctx)


@pytest.mark.parametrize(
    ("site", "ctx"),
    [
        (
            ClaimSite(
                kind="binop",
                target="+",
                operand_types=(F64_1D, F64_1D),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/elementwise-float64",
                result_type=F64_1D,
            ),
            _ctx(operands=("a", "b"), leaves=("a", "b")),
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.dot",
                operand_types=(F64_1D, F64_1D),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/dot-float64",
                result_type="float",
            ),
            _ctx(operands=("a", "b"), leaves=("a", "b")),
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.sum",
                operand_types=(F64_1D,),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/reduction-sum-mean",
                result_type="float",
            ),
            _ctx(operands=("a",), leaves=("a",)),
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.negative",
                operand_types=(F64_1D,),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/unary-module",
                result_type=F64_1D,
            ),
            _ctx(operands=("a",), leaves=("a",)),
        ),
    ],
)
def test_direct_lower_rejects_leaves_context(site: ClaimSite, ctx: LoweringContext) -> None:
    with pytest.raises(ValueError, match="leaf_operands"):
        lower(site, ctx)


def test_fusion_lower_rejects_direct_only_metadata() -> None:
    site = replace(_fusion_site(), operand_literals=(ClaimLiteral(is_literal=True, value=0),))
    with pytest.raises(ValueError, match="operand_literals"):
        lower(site, _ctx(operands=(), leaves=("a", "b", "c", "d")))


@pytest.mark.parametrize(
    "site",
    [
        ClaimSite(
            kind="binop",
            target="+",
            operand_types=(F64_1D, F64_1D),
            file_path="",
            line=0,
            column=0,
            rule_id="rextio-numpy/elementwise-float64",
            result_type="float",
        ),
        ClaimSite(
            kind="call",
            target="numpy.dot",
            operand_types=(F64_1D, F64_1D),
            file_path="",
            line=0,
            column=0,
            rule_id="rextio-numpy/dot-float64",
            result_type="int",
        ),
        ClaimSite(
            kind="call",
            target="numpy.sum",
            operand_types=(F64_1D,),
            file_path="",
            line=0,
            column=0,
            rule_id="rextio-numpy/reduction-sum-mean",
            result_type=F64_1D,
        ),
        ClaimSite(
            kind="call",
            target="numpy.negative",
            operand_types=(F64_1D,),
            file_path="",
            line=0,
            column=0,
            rule_id="rextio-numpy/unary-module",
            result_type="float",
        ),
    ],
)
def test_lower_rejects_forged_result_type(site: ClaimSite) -> None:
    with pytest.raises(ValueError, match="result_type"):
        lower(site, _ctx(operands=("a", "b") if site.target == "numpy.dot" or site.kind == "binop" else ("a",)))


def test_unary_lower_rejects_forged_receiver_metadata() -> None:
    site = ClaimSite(
        kind="call",
        target="numpy.negative",
        operand_types=(F64_1D,),
        file_path="",
        line=0,
        column=0,
        rule_id="rextio-numpy/unary-module",
        result_type=F64_1D,
        receiver=ReceiverMeta(arg_type=F64_1D, expr_kind="attribute", is_safe=False),
    )
    with pytest.raises(ValueError, match="receiver"):
        lower(site, _ctx(operands=("a",)))


@pytest.mark.parametrize(
    ("site", "ctx"),
    [
        (
            ClaimSite(
                kind="binop",
                target="+",
                operand_types=(F64_1D, F64_1D),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/elementwise-float64",
                result_type=F64_1D,
            ),
            _ctx(operands=("a", "b")),
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.dot",
                operand_types=(F64_1D, F64_1D),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/dot-float64",
                result_type="float",
            ),
            _ctx(operands=("a", "b")),
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.sum",
                operand_types=(F64_1D,),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/reduction-sum-mean",
                result_type="float",
            ),
            _ctx(operands=("a",)),
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.negative",
                operand_types=(F64_1D,),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/unary-module",
                result_type=F64_1D,
            ),
            _ctx(operands=("a",)),
        ),
    ],
)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operand_literals", (ClaimLiteral(is_literal=True, value=1),)),
        ("callables", (CallableMeta(arg_index=0, qualname="forged"),)),
    ],
)
def test_direct_lower_rejects_unconsumed_site_metadata(
    site: ClaimSite,
    ctx: LoweringContext,
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        lower(replace(site, **{field: value}), ctx)


def test_binop_lower_rejects_inconsistent_expression_metadata() -> None:
    site = ClaimSite(
        kind="binop",
        target="+",
        operand_types=(F64_1D, F64_1D),
        file_path="",
        line=0,
        column=0,
        rule_id="rextio-numpy/elementwise-float64",
        result_type=F64_1D,
        expression=ClaimExpr(
            kind="binop",
            target="-",
            result_type=F64_1D,
            children=(
                ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=0, leaf_kind="name"),
                ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=1, leaf_kind="name"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="ClaimSite.expression"):
        lower(site, _ctx(operands=("a", "b")))


def test_direct_lower_accepts_core_nonliteral_slots_but_rejects_wrong_count() -> None:
    site = ClaimSite(
        kind="call",
        target="numpy.dot",
        operand_types=(F64_1D, F64_1D),
        file_path="",
        line=0,
        column=0,
        rule_id="rextio-numpy/dot-float64",
        result_type="float",
        operand_literals=(ClaimLiteral(), ClaimLiteral()),
    )
    assert lower(site, _ctx(operands=("a", "b"))).rust == "__rxtnp_dot1(&a, &b)?"
    with pytest.raises(ValueError, match="operand_literals.*2 slots"):
        lower(replace(site, operand_literals=(ClaimLiteral(),)), _ctx(operands=("a", "b")))


@pytest.mark.parametrize(
    ("site", "ctx", "message"),
    [
        (
            ClaimSite(
                kind="call",
                target="values.dot",
                operand_types=(F64_1D,),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/dot-float64",
                result_type="float",
            ),
            _ctx(operands=("rhs",)),
            "ClaimSite.receiver",
        ),
        (
            ClaimSite(
                kind="call",
                target="values.sum",
                operand_types=(),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/reduction-sum-mean",
                result_type="float",
            ),
            _ctx(operands=()),
            "ClaimSite.receiver",
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.dot",
                operand_types=(F64_1D,),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/dot-float64",
                result_type="float",
                receiver=ReceiverMeta(arg_type=F64_1D, expr_kind="attribute", is_safe=False),
            ),
            _ctx(operands=("rhs",), leaves=()),
            "no ClaimSite.receiver",
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.sum",
                operand_types=(),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/reduction-sum-mean",
                result_type="float",
                receiver=ReceiverMeta(arg_type=F64_1D, expr_kind="attribute", is_safe=False),
            ),
            _ctx(operands=()),
            "no ClaimSite.receiver",
        ),
    ],
)
def test_method_and_module_targets_reject_forged_receiver_placement(
    site: ClaimSite,
    ctx: LoweringContext,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        lower(site, ctx)


@pytest.mark.parametrize(
    ("site", "ctx"),
    [
        (
            ClaimSite(
                kind="binop",
                target="+",
                operand_types=(F64_1D, F64_1D),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/elementwise-float64",
                result_type=F64_1D,
            ),
            _ctx(operands=("a", "b")),
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.dot",
                operand_types=(F64_1D, F64_1D),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/dot-float64",
                result_type="float",
            ),
            _ctx(operands=("a", "b")),
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.sum",
                operand_types=(F64_1D,),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/reduction-sum-mean",
                result_type="float",
            ),
            _ctx(operands=("a",)),
        ),
        (
            ClaimSite(
                kind="call",
                target="numpy.negative",
                operand_types=(F64_1D,),
                file_path="",
                line=0,
                column=0,
                rule_id="rextio-numpy/unary-module",
                result_type=F64_1D,
            ),
            _ctx(operands=("a",)),
        ),
        (_fusion_site(), _ctx(operands=(), leaves=("a", "b", "c", "d"))),
    ],
)
@pytest.mark.parametrize(
    ("field", "value"),
    _CONTEXT_FIELDS,
)
def test_every_lower_rejects_non_pyo3_rust_context(
    site: ClaimSite,
    ctx: LoweringContext,
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        lower(site, replace(ctx, **{field: value}))


@pytest.mark.parametrize(
    "script",
    [
        r'''
from rextio.plugins.api import ClaimSite, LoweringContext
from rextio_numpy.diagnostics import F64_1D
from rextio_numpy.lower.linear import try_lower
site = ClaimSite(kind="call", target="numpy.dot", operand_types=(F64_1D, F64_1D), file_path="", line=0, column=0, result_type="float")
ctx = LoweringContext(operands=("a", "b"), target_language="rust", fresh_name=lambda prefix: prefix)
try:
    try_lower(site, ctx)
except ValueError as exc:
    if "rule_id" not in str(exc):
        raise SystemExit(2) from exc
else:
    raise SystemExit(3)
''',
        r'''
from rextio.plugins.api import ClaimSite, LoweringContext
from rextio_numpy.diagnostics import F64_1D
from rextio_numpy.lower.unary import try_lower
site = ClaimSite(kind="call", target="numpy.negative", operand_types=(F64_1D,), file_path="", line=0, column=0, rule_id="rextio-numpy/unary-module", result_type="float")
ctx = LoweringContext(operands=("a",), target_language="rust", fresh_name=lambda prefix: prefix)
try:
    try_lower(site, ctx)
except ValueError as exc:
    if "result_type" not in str(exc):
        raise SystemExit(2) from exc
else:
    raise SystemExit(3)
''',
        r'''
from rextio.plugins.api import ClaimExpr, ClaimSite, LoweringContext
from rextio_numpy.claim.fusion import FUSION_RULE
from rextio_numpy.diagnostics import F64_1D
from rextio_numpy.lower.fusion import try_lower
def leaf(index):
    return ClaimExpr(kind="leaf", result_type=F64_1D, leaf_index=index, leaf_kind="name")
expr = ClaimExpr(kind="binop", target="*", result_type=F64_1D, children=(
    ClaimExpr(kind="binop", target="+", result_type=F64_1D, children=(leaf(0), leaf(1))),
    ClaimExpr(kind="binop", target="-", result_type=F64_1D, children=(leaf(2), leaf(3))),
))
site = ClaimSite(kind="binop", target="*", operand_types=(F64_1D, F64_1D), file_path="", line=0, column=0, rule_id=FUSION_RULE, result_type=F64_1D, expression=expr)
ctx = LoweringContext(operands=(), leaf_operands=("a", "b", "c", "d"), receiver="forged", target_language="rust", fresh_name=lambda prefix: prefix)
try:
    try_lower(site, ctx)
except ValueError as exc:
    if "ctx.receiver" not in str(exc):
        raise SystemExit(2) from exc
else:
    raise SystemExit(3)
''',
    ],
)
def test_lowerers_without_prior_optimize_regressions_fail_closed_under_python_optimize(
    script: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
