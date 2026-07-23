"""Lowering for elementwise operators and exact NumPy ufunc aliases."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_numpy import rust_snippets
from rextio_numpy.claim.binops import (
    BINOP_NAMES,
    UFUNC_CALL_NAMES,
    _ELEMENTWISE_RULE,
    _ELEMENTWISE_UFUNC_RULE,
    _result_array_type,
)
from rextio_numpy.diagnostics import SCALAR_FOR_DTYPE, array_meta, is_array_type
from rextio_numpy.lower.contracts import (
    require_direct_context,
    require_no_hidden_site_metadata,
    require_result_type,
    require_rule_id,
    require_site_expression_matches_claim,
)


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Return a lowered expression for elementwise operators or ufunc calls."""
    is_binop = claimed.kind == "binop" and claimed.target in BINOP_NAMES
    is_ufunc_call = claimed.kind == "call" and claimed.target in UFUNC_CALL_NAMES
    if not is_binop and not is_ufunc_call:
        return None
    lane = "binops" if is_binop else "ufunc calls"
    expected_rule = _ELEMENTWISE_RULE if is_binop else _ELEMENTWISE_UFUNC_RULE
    op_token = claimed.target if is_binop else UFUNC_CALL_NAMES[claimed.target]
    op = BINOP_NAMES[op_token]

    require_rule_id(claimed, expected_rule, lane)
    require_direct_context(ctx, lane, receiver=False)
    require_no_hidden_site_metadata(
        claimed,
        lane,
        allow_expression=True,
        expected_operand_literals=2,
        allowed_literal_positions=frozenset({0, 1}),
    )
    require_site_expression_matches_claim(claimed, lane)
    if claimed.receiver is not None:
        raise ValueError(f"rextio-numpy {lane} lower requires no ClaimSite.receiver")
    if claimed.keywords:
        raise ValueError(f"rextio-numpy {lane} lower requires empty keywords")
    if len(claimed.operand_types) != 2:
        raise ValueError(
            f"rextio-numpy {lane} lower requires exactly two operand types; "
            f"got {len(claimed.operand_types)}"
        )
    left, right = claimed.operand_types
    _require_literal_metadata(claimed, lane)
    if len(ctx.operands) != 2:
        raise ValueError(
            f"rextio-numpy {lane} lower requires exactly two ctx.operands; "
            f"got {len(ctx.operands)}"
        )
    first, second = ctx.operands

    # Fail closed on malformed lower-time metadata rather than emitting
    # incorrect elementwise code (asserts are stripped under PYTHONOPTIMIZE=1).
    if is_array_type(left) and is_array_type(right):
        if left is None or right is None:
            raise ValueError(
                f"rextio-numpy {lane} lower requires non-None array operand types"
            )
        left_meta = array_meta(left)
        right_meta = array_meta(right)
        if left_meta is None or right_meta is None:
            raise ValueError(
                f"rextio-numpy {lane} lower requires array operand types, "
                f"got {left!r} and {right!r}"
            )
        dtype, left_rank = left_meta
        right_dtype, right_rank = right_meta
        if dtype != right_dtype:
            raise ValueError(
                f"rextio-numpy {lane} lower requires matching array dtypes, "
                f"got {dtype!r} and {right_dtype!r}"
            )
        require_result_type(
            claimed,
            _result_array_type(dtype, max(left_rank, right_rank), op_token),
            lane,
        )
        name = rust_snippets.elementwise_call_name_aa(op, dtype, left_rank, right_rank)
        helper = rust_snippets.elementwise_aa_typed(op, dtype, left_rank, right_rank)
        helpers: tuple[str, ...]
        if dtype == "f64" and left_rank == 1 and right_rank == 1:
            helpers = (helper,)
        else:
            helpers = (*rust_snippets.shared_broadcast_helpers(), helper)
        return LoweredExpr(
            rust=f"{name}(&{first}, &{second})?",
            helpers=helpers,
        )

    if is_array_type(left):
        if left is None:
            raise ValueError(
                f"rextio-numpy {lane} lower requires non-None left array operand type"
            )
        meta = array_meta(left)
        if meta is None:
            raise ValueError(
                f"rextio-numpy {lane} lower requires array left operand type, got {left!r}"
            )
        dtype, rank = meta
        if right != SCALAR_FOR_DTYPE[dtype]:
            raise ValueError(
                f"rextio-numpy {lane} lower requires the matching scalar type for "
                f"{dtype!r}; got {right!r}"
            )
        require_result_type(claimed, _result_array_type(dtype, rank, op_token), lane)
        name = rust_snippets.elementwise_call_name_as(op, dtype, rank)
        helper = rust_snippets.elementwise_as_typed(op, dtype, rank)
        return LoweredExpr(
            rust=f"{name}(&{first}, {second})?",
            helpers=(helper,),
        )

    if right is None:
        raise ValueError(
            f"rextio-numpy {lane} lower requires non-None right array operand type"
        )
    meta = array_meta(right)
    if meta is None:
        raise ValueError(
            f"rextio-numpy {lane} lower requires array right operand type, got {right!r}"
        )
    dtype, rank = meta
    if left != SCALAR_FOR_DTYPE[dtype]:
        raise ValueError(
            f"rextio-numpy {lane} lower requires the matching scalar type for "
            f"{dtype!r}; got {left!r}"
        )
    require_result_type(claimed, _result_array_type(dtype, rank, op_token), lane)
    name = rust_snippets.elementwise_call_name_sa(op, dtype, rank)
    helper = rust_snippets.elementwise_sa_typed(op, dtype, rank)
    return LoweredExpr(
        rust=f"{name}({first}, &{second})?",
        helpers=(helper,),
    )


def _require_literal_metadata(claimed: ClaimSite, lane: str = "binops") -> None:
    """Validate literal values against the exact scalar type Core claimed."""
    if not claimed.operand_literals:
        return
    for index, (operand_type, literal) in enumerate(
        zip(claimed.operand_types, claimed.operand_literals, strict=True)
    ):
        if not literal.is_literal:
            continue
        value = literal.value
        if operand_type == "int" and isinstance(value, int) and not isinstance(value, bool):
            if -(2**63) <= value <= 2**63 - 1:
                continue
            raise ValueError(
                f"rextio-numpy {lane} lower requires an i64 literal in "
                f"[-2**63, 2**63 - 1] at operand_literals[{index}]; got {value!r}"
            )
        if operand_type == "float" and isinstance(value, float):
            continue
        raise ValueError(
            f"rextio-numpy {lane} lower requires a literal compatible with "
            f"operand type {operand_type!r} at operand_literals[{index}]; got {literal!r}"
        )
