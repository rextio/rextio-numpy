"""Shared fail-closed checks for lower-time claim and context metadata."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweringContext


def require_rust_pyo3_context(ctx: LoweringContext, lane: str) -> None:
    """Reject a context this host-extension provider cannot lower safely."""
    if ctx.target_language != "rust":
        raise ValueError(
            f"rextio-numpy {lane} lower requires target_language='rust', "
            f"got {ctx.target_language!r}"
        )
    backend = getattr(ctx, "backend", "pyo3")
    if backend != "pyo3":
        raise ValueError(
            f"rextio-numpy {lane} lower requires backend='pyo3', got {backend!r}"
        )


def require_rule_id(claimed: ClaimSite, expected: str, lane: str) -> None:
    """Require the exact claim rule that owns one lowering lane."""
    if claimed.rule_id != expected:
        raise ValueError(
            f"rextio-numpy {lane} lower requires rule_id={expected!r}, "
            f"got {claimed.rule_id!r}"
        )


def require_result_type(claimed: ClaimSite, expected: str, lane: str) -> None:
    """Require the result key reconstructed from the complete site metadata."""
    if claimed.result_type != expected:
        raise ValueError(
            f"rextio-numpy {lane} lower requires result_type={expected!r}, "
            f"got {claimed.result_type!r}"
        )


def require_direct_context(ctx: LoweringContext, lane: str, *, receiver: bool) -> None:
    """Validate direct-operand mode and its receiver placement."""
    require_rust_pyo3_context(ctx, lane)
    if ctx.leaf_operands:
        raise ValueError(
            f"rextio-numpy {lane} lower requires empty ctx.leaf_operands "
            f"(direct operand mode); got {ctx.leaf_operands!r}"
        )
    if receiver and ctx.receiver is None:
        raise ValueError(f"rextio-numpy {lane} lower requires ctx.receiver")
    if not receiver and ctx.receiver is not None:
        raise ValueError(
            f"rextio-numpy {lane} lower requires no ctx.receiver; got {ctx.receiver!r}"
        )


def require_no_hidden_site_metadata(
    claimed: ClaimSite,
    lane: str,
    *,
    allow_expression: bool = False,
    expected_operand_literals: int | None = None,
    allowed_literal_positions: frozenset[int] = frozenset(),
) -> None:
    """Reject fields that no certified direct NumPy rule consumes."""
    if (
        expected_operand_literals is not None
        and claimed.operand_literals
        and len(claimed.operand_literals) != expected_operand_literals
    ):
        raise ValueError(
            f"rextio-numpy {lane} lower requires either empty operand_literals "
            f"or {expected_operand_literals} slots; got {len(claimed.operand_literals)}"
        )
    for index, literal in enumerate(claimed.operand_literals):
        if literal.is_literal:
            if index not in allowed_literal_positions:
                raise ValueError(
                    f"rextio-numpy {lane} lower does not accept a literal at "
                    f"operand_literals[{index}]; got {literal!r}"
                )
        elif literal.value is not None:
            raise ValueError(
                f"rextio-numpy {lane} lower requires non-literal operand_literals "
                f"to have value=None; got {literal!r}"
            )
    if claimed.callables:
        raise ValueError(
            f"rextio-numpy {lane} lower requires empty callables; got {claimed.callables!r}"
        )
    if not allow_expression and claimed.expression is not None:
        raise ValueError(f"rextio-numpy {lane} lower requires no ClaimSite.expression")


def require_site_expression_matches_claim(claimed: ClaimSite, lane: str) -> None:
    """Accept Core's optional expression tree only when its root matches the site."""
    expression = claimed.expression
    if expression is None:
        return
    if (
        expression.kind != claimed.kind
        or expression.target != claimed.target
        or expression.result_type != claimed.result_type
    ):
        raise ValueError(
            f"rextio-numpy {lane} lower requires ClaimSite.expression to match "
            "the site kind, target, and result_type"
        )
