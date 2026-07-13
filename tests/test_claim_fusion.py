"""Matcher unit tables for elementwise chain fusion (Wave 2)."""

from __future__ import annotations

import pytest

from rextio.config.schema import RextioConfig
from rextio.plugins.api import Claimed, ClaimExpr, ClaimLiteral, ClaimSite

from rextio_numpy.claim import claim
from rextio_numpy.claim.binops import try_claim as try_claim_binops
from rextio_numpy.claim.fusion import FUSION_RULE, try_claim, try_match
from rextio_numpy.diagnostics import (
    F32_1D,
    F32_2D,
    F64_1D,
    F64_2D,
    I64_1D,
    I64_2D,
)

CONFIG = RextioConfig()


def leaf(index: int, type_key: str, *, kind: str = "name") -> ClaimExpr:
    return ClaimExpr(
        kind="leaf",
        result_type=type_key,
        leaf_index=index,
        leaf_kind=kind,
    )


def binop(op: str, left: ClaimExpr, right: ClaimExpr, result_type: str | None = None) -> ClaimExpr:
    return ClaimExpr(
        kind="binop",
        target=op,
        result_type=result_type or left.result_type,
        children=(left, right),
    )


def chain_add(n_binops: int, type_key: str = F64_1D) -> ClaimExpr:
    """Left-skewed chain of *n_binops* additions over distinct leaves."""
    expr: ClaimExpr = leaf(0, type_key)
    for i in range(n_binops):
        expr = binop("+", expr, leaf(i + 1, type_key), type_key)
    return expr


def site_for(expr: ClaimExpr, op: str | None = None) -> ClaimSite:
    target = op if op is not None else expr.target
    return ClaimSite(
        kind="binop",
        target=target,
        operand_types=(
            expr.children[0].result_type if expr.children else None,
            expr.children[1].result_type if len(expr.children) > 1 else None,
        ),
        file_path="",
        line=0,
        column=0,
        expression=expr,
    )


# ---------------------------------------------------------------------------
# Node-count bounds
# ---------------------------------------------------------------------------


def test_match_rejects_one_binop() -> None:
    expr = binop("+", leaf(0, F64_1D), leaf(1, F64_1D))
    assert try_match(expr) is None
    assert try_claim(site_for(expr)) is None
    # Ordinary path still claims.
    assert isinstance(try_claim_binops(site_for(expr)), Claimed)


def test_match_accepts_two_and_eight_binops() -> None:
    for n in (2, 8):
        expr = chain_add(n)
        match = try_match(expr)
        assert match is not None, n
        assert match.binop_count == n
        assert match.leaf_count == n + 1
        claimed = try_claim(site_for(expr))
        assert claimed == Claimed(
            rule_id=FUSION_RULE,
            result_type=F64_1D,
            operand_mode="leaves",
        )


def test_match_rejects_nine_binops() -> None:
    expr = chain_add(9)
    assert try_match(expr) is None
    assert try_claim(site_for(expr)) is None


# ---------------------------------------------------------------------------
# Operators / dtypes / ranks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
@pytest.mark.parametrize("key", [F64_1D, F64_2D, F32_1D, F32_2D])
def test_match_float_all_ops(op: str, key: str) -> None:
    expr = binop(op, binop("+", leaf(0, key), leaf(1, key), key), leaf(2, key), key)
    match = try_match(expr)
    assert match is not None
    assert match.dtype in {"f64", "f32"}
    assert match.result_type == key


@pytest.mark.parametrize("op", ["+", "-", "*"])
@pytest.mark.parametrize("key", [I64_1D, I64_2D])
def test_match_i64_allowed_ops(op: str, key: str) -> None:
    expr = binop(op, binop("+", leaf(0, key), leaf(1, key), key), leaf(2, key), key)
    match = try_match(expr)
    assert match is not None
    assert match.dtype == "i64"


def test_match_i64_rejects_division() -> None:
    expr = binop(
        "/",
        binop("+", leaf(0, I64_1D), leaf(1, I64_1D), I64_1D),
        leaf(2, I64_1D),
        F64_1D,
    )
    assert try_match(expr) is None


def test_match_mixed_rank_broadcast_result() -> None:
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_2D), F64_2D),
        leaf(2, F64_1D),
        F64_2D,
    )
    match = try_match(expr)
    assert match is not None
    assert match.result_type == F64_2D
    assert match.result_rank == 2


# ---------------------------------------------------------------------------
# Tree shapes and repeated names
# ---------------------------------------------------------------------------


def test_match_balanced_tree() -> None:
    # (a + b) * (c - d)
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        binop("-", leaf(2, F64_1D), leaf(3, F64_1D)),
    )
    match = try_match(expr)
    assert match is not None
    assert match.binop_count == 3
    assert match.leaf_count == 4
    assert match.postorder_ops == ("+", "-", "*")


def test_match_skewed_tree() -> None:
    expr = chain_add(4)
    match = try_match(expr)
    assert match is not None
    assert match.binop_count == 4
    assert match.postorder_ops == ("+", "+", "+", "+")


def test_match_repeated_names_separate_leaf_indexes() -> None:
    # (a + b) * (a - b) — four leaf occurrences
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        binop("-", leaf(2, F64_1D), leaf(3, F64_1D)),
    )
    match = try_match(expr)
    assert match is not None
    assert match.leaf_count == 4
    # Signature must mention all four leaf indexes.
    assert "L0" in match.signature and "L3" in match.signature


def test_match_rejects_non_dense_leaf_indexes() -> None:
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        binop("-", leaf(0, F64_1D), leaf(1, F64_1D)),  # duplicate indexes
    )
    assert try_match(expr) is None


# ---------------------------------------------------------------------------
# Excluded leaf / node forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_leaf",
    [
        leaf(0, F64_1D, kind="opaque"),
        ClaimExpr(
            kind="literal",
            result_type="int",
            literal=ClaimLiteral(is_literal=True, value=1),
        ),
    ],
)
def test_match_rejects_bad_leaves(bad_leaf: ClaimExpr) -> None:
    if bad_leaf.kind == "leaf":
        # opaque leaf as left of outer with a nested add on the right so
        # binop_count would be 2 if accepted.
        expr = binop(
            "*",
            bad_leaf,
            binop("+", leaf(1, F64_1D), leaf(2, F64_1D)),
        )
    else:
        expr = binop(
            "*",
            bad_leaf,
            binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        )
    assert try_match(expr) is None


def test_match_rejects_scalar_leaf_type() -> None:
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        ClaimExpr(kind="leaf", result_type="float", leaf_index=2, leaf_kind="name"),
    )
    assert try_match(expr) is None


def test_match_rejects_call_node() -> None:
    call = ClaimExpr(
        kind="call",
        target="numpy.abs",
        result_type=F64_1D,
        children=(leaf(0, F64_1D),),
    )
    expr = binop("*", call, binop("+", leaf(1, F64_1D), leaf(2, F64_1D)))
    assert try_match(expr) is None


def test_match_rejects_mixed_dtypes() -> None:
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        leaf(2, F32_1D),
    )
    assert try_match(expr) is None


def test_match_rejects_unresolved_leaf() -> None:
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        ClaimExpr(kind="leaf", result_type=None, leaf_index=2, leaf_kind="name"),
    )
    assert try_match(expr) is None


def test_match_none_expression() -> None:
    assert try_match(None) is None
    site = ClaimSite(
        kind="binop",
        target="+",
        operand_types=(F64_1D, F64_1D),
        file_path="",
        line=0,
        column=0,
        expression=None,
    )
    assert try_claim(site) is None


def test_router_prefers_fusion_over_elementwise() -> None:
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        binop("-", leaf(2, F64_1D), leaf(3, F64_1D)),
    )
    result = claim(site_for(expr), CONFIG)
    assert result == Claimed(
        rule_id=FUSION_RULE,
        result_type=F64_1D,
        operand_mode="leaves",
    )


def test_router_single_binop_stays_elementwise() -> None:
    expr = binop("+", leaf(0, F64_1D), leaf(1, F64_1D))
    result = claim(site_for(expr), CONFIG)
    assert result == Claimed(
        rule_id="rextio-numpy/elementwise-float64",
        result_type=F64_1D,
    )


def test_signature_deterministic_no_hash() -> None:
    expr = chain_add(3)
    a = try_match(expr)
    b = try_match(expr)
    assert a is not None and b is not None
    assert a.signature == b.signature
    assert "hash" not in a.signature.lower()


# ---------------------------------------------------------------------------
# Internal ClaimExpr metadata consistency
# ---------------------------------------------------------------------------


def test_match_rejects_wrong_internal_binop_result_type() -> None:
    # Intermediate node claims rank-2 but both children are rank-1.
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D), F64_2D),
        leaf(2, F64_1D),
        F64_1D,
    )
    assert try_match(expr) is None
    assert try_claim(site_for(expr)) is None
    # Ordinary elementwise still owns the outer op when offered flat.
    flat = ClaimSite(
        kind="binop",
        target="*",
        operand_types=(F64_2D, F64_1D),
        file_path="",
        line=0,
        column=0,
    )
    assert isinstance(try_claim_binops(flat), Claimed) or try_claim_binops(flat) is not None


def test_claim_rejects_site_target_mismatch_falls_through() -> None:
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        leaf(2, F64_1D),
    )
    site = site_for(expr, op="+")  # wrong root operator
    assert try_claim(site) is None


def test_claim_rejects_operand_types_mismatch_falls_through() -> None:
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        leaf(2, F64_1D),
    )
    site = ClaimSite(
        kind="binop",
        target="*",
        operand_types=(F64_2D, F64_1D),  # left child is F64_1D intermediate
        file_path="",
        line=0,
        column=0,
        expression=expr,
    )
    assert try_claim(site) is None


def test_claim_rejects_unresolved_operand_types_none_none() -> None:
    """Reproduced blocker: (None, None) must not fuse; fall through."""
    expr = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        binop("-", leaf(2, F64_1D), leaf(3, F64_1D)),
    )
    assert try_match(expr) is not None
    site = ClaimSite(
        kind="binop",
        target="*",
        operand_types=(None, None),
        file_path="",
        line=0,
        column=0,
        expression=expr,
    )
    assert try_claim(site) is None
    # Router falls through to ordinary elementwise (not fusion).
    result = claim(site, CONFIG)
    assert not (isinstance(result, Claimed) and result.rule_id == FUSION_RULE)


def test_match_rejects_missing_binop_result_type() -> None:
    # Build a tree where an intermediate binop has result_type=None via replace.
    from dataclasses import replace

    good = binop(
        "*",
        binop("+", leaf(0, F64_1D), leaf(1, F64_1D)),
        leaf(2, F64_1D),
    )
    left = replace(good.children[0], result_type=None)
    bad = replace(good, children=(left, good.children[1]))
    assert try_match(bad) is None
