"""Claim decisions for multi-op elementwise chain fusion (Wave 2).

Matches binary-op trees of 2–8 binop nodes whose leaves are all simple
array names of one dtype (ranks 1–2), and returns a leaves-mode claim so
core subsumes descendant elementwise claims.

Internal ClaimExpr metadata is validated against the types implied by each
node's children and operator. Malformed or unresolved metadata fails the
match (claim falls through to ordinary per-op lowering); lower-time
re-validation raises.
"""

from __future__ import annotations

from dataclasses import dataclass

from rextio.plugins.api import Claimed, ClaimExpr, ClaimResult, ClaimSite

from rextio_numpy.diagnostics import array_meta, is_array_type, type_key_for

FUSION_RULE = "rextio-numpy/elementwise-chain-fusion"

MIN_BINOPS = 2
MAX_BINOPS = 8

# Operator token -> short name used in signatures / Rust.
OP_NAMES: dict[str, str] = {"+": "add", "-": "sub", "*": "mul", "/": "div"}

# Allowed operators by dtype token.
_OPS_FLOAT: frozenset[str] = frozenset(OP_NAMES)
_OPS_I64: frozenset[str] = frozenset({"+", "-", "*"})


@dataclass(frozen=True)
class FusionMatch:
    """Validated fusion plan derived from a ClaimExpr tree."""

    dtype: str
    result_rank: int
    result_type: str
    binop_count: int
    leaf_count: int
    leaf_ranks: tuple[int, ...]
    # Deterministic structural signature (safe for Rust identifiers).
    signature: str
    # LTR postorder list of op tokens for codegen.
    postorder_ops: tuple[str, ...]
    # Root children's result types (for site.operand_types consistency).
    root_child_types: tuple[str, str]


def try_match(expression: ClaimExpr | None) -> FusionMatch | None:
    """Return a fusion plan when *expression* is an eligible pure-array tree.

    Nested binop nodes must carry a non-None ``result_type`` equal to the
    plugin array type implied by children+op. The root may be ``None`` at
    claim time (Core fills it after ``Claimed``); if present it must match.
    Returns None for any out-of-scope or metadata-inconsistent tree so the
    ordinary per-op claim path remains responsible.
    """
    if expression is None or expression.kind != "binop":
        return None
    if expression.target not in OP_NAMES:
        return None

    leaves: list[tuple[int, str, int]] = []  # (leaf_index, dtype, rank)
    postorder_ops: list[str] = []
    binop_count = 0

    def walk(node: ClaimExpr, *, is_root: bool) -> tuple[str, int, str] | None:
        """Return ``(dtype, rank, type_key)`` for *node*, or None if ineligible.

        Every non-root binop must carry a non-None ``result_type`` equal to the
        implied plugin array type. Core leaves the *root* ``result_type`` as
        None at claim time and fills it after ``Claimed``; the root may
        therefore be None, but if present it must match the implied type.
        """
        nonlocal binop_count
        if node.kind == "leaf":
            if node.leaf_kind != "name":
                return None
            if node.leaf_index is None or node.leaf_index < 0:
                return None
            if not is_array_type(node.result_type):
                return None
            assert node.result_type is not None
            meta = array_meta(node.result_type)
            if meta is None:
                return None
            dtype, rank = meta
            if rank not in (1, 2):
                return None
            leaves.append((node.leaf_index, dtype, rank))
            return dtype, rank, node.result_type
        if node.kind == "literal":
            return None
        if node.kind == "call":
            return None
        if node.kind != "binop":
            return None
        if node.target not in OP_NAMES:
            return None
        if len(node.children) != 2:
            return None
        left = walk(node.children[0], is_root=False)
        if left is None:
            return None
        right = walk(node.children[1], is_root=False)
        if right is None:
            return None
        left_dtype, left_rank, _left_key = left
        right_dtype, right_rank, _right_key = right
        if left_dtype != right_dtype:
            return None
        # Fusion is array–array only; i64 `/` promotes and is excluded below.
        dtype = left_dtype
        if dtype == "i64" and node.target == "/":
            return None
        result_rank = max(left_rank, right_rank)
        implied_key = type_key_for(dtype, result_rank)
        if is_root:
            # Claim-time Core trees leave root result_type unset until Claimed.
            if node.result_type is not None and node.result_type != implied_key:
                return None
        else:
            # Nested binops must already be fully typed (post descendant claims).
            if node.result_type is None or node.result_type != implied_key:
                return None
        binop_count += 1
        postorder_ops.append(node.target)
        return dtype, result_rank, implied_key

    root = walk(expression, is_root=True)
    if root is None:
        return None
    dtype, result_rank, result_type = root

    if binop_count < MIN_BINOPS or binop_count > MAX_BINOPS:
        return None

    # Leaf indexes must be a dense LTR sequence 0..n-1 (Core contract).
    if not leaves:
        return None
    indexes = [idx for idx, _dtype, _rank in leaves]
    if indexes != list(range(len(indexes))):
        return None
    if any(leaf_dtype != dtype for _idx, leaf_dtype, _rank in leaves):
        return None

    allowed = _OPS_I64 if dtype == "i64" else _OPS_FLOAT
    if any(op not in allowed for op in postorder_ops):
        return None

    left_child, right_child = expression.children
    left_info = _node_type_key(left_child)
    right_info = _node_type_key(right_child)
    if left_info is None or right_info is None:
        return None

    leaf_ranks = tuple(rank for _idx, _dtype, rank in leaves)
    signature = _canonical_signature(expression, dtype=dtype, result_rank=result_rank)
    return FusionMatch(
        dtype=dtype,
        result_rank=result_rank,
        result_type=result_type,
        binop_count=binop_count,
        leaf_count=len(leaves),
        leaf_ranks=leaf_ranks,
        signature=signature,
        postorder_ops=tuple(postorder_ops),
        root_child_types=(left_info, right_info),
    )


def _node_type_key(node: ClaimExpr) -> str | None:
    """Return the array type key for a leaf or binop (requires non-None metadata)."""
    if node.kind == "leaf":
        if not is_array_type(node.result_type):
            return None
        return node.result_type
    if node.kind == "binop":
        # try_match already required exact non-None result_type on binops.
        if node.result_type is None or not is_array_type(node.result_type):
            return None
        return node.result_type
    return None


def site_consistent_with_match(
    site: ClaimSite,
    match: FusionMatch,
    *,
    require_result_type: bool,
) -> bool:
    """Whether claim/lower site fields agree with a matched fusion tree.

    * ``operand_types`` must *exactly* equal the two computed root-child
      result types (``None`` is not accepted).
    * ``target`` must equal the expression root operator.
    * Binop sites must have empty ``keywords``.
    * Claim time: ``result_type`` may be ``None``; if set it must equal the
      match. Lower time: ``result_type`` must be non-None and exact.
    """
    expression = site.expression
    if expression is None or expression.kind != "binop":
        return False
    if site.target != expression.target:
        return False
    if site.kind != "binop":
        return False
    if site.keywords:
        return False
    if len(site.operand_types) != 2:
        return False
    # Exact equality — unresolved (None, None) must not fuse.
    if site.operand_types != match.root_child_types:
        return False
    if require_result_type:
        if site.result_type is None or site.result_type != match.result_type:
            return False
    elif site.result_type is not None and site.result_type != match.result_type:
        return False
    return True


def _canonical_signature(expression: ClaimExpr, *, dtype: str, result_rank: int) -> str:
    """Build a stable, collision-free structural signature (no Python hash)."""

    def encode(node: ClaimExpr) -> str:
        if node.kind == "leaf":
            meta = array_meta(node.result_type) if node.result_type else None
            rank = meta[1] if meta is not None else 0
            # Include leaf_index so repeated names stay distinct occurrences.
            return f"L{node.leaf_index}r{rank}"
        if node.kind == "binop":
            op = OP_NAMES.get(node.target, "op")
            left = encode(node.children[0])
            right = encode(node.children[1])
            return f"B{op}_{left}_{right}"
        # Unreachable for matched trees; keep deterministic for fail-closed paths.
        return f"X{node.kind}"

    return f"{dtype}_r{result_rank}_{encode(expression)}"


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Return a leaves-mode fusion claim, or None if not this lane."""
    if site.kind != "binop" or site.target not in OP_NAMES:
        return None
    if site.expression is None:
        return None
    # Root operator on the site must match the frozen expression root.
    if site.target != site.expression.target:
        return None
    match = try_match(site.expression)
    if match is None:
        return None
    if not site_consistent_with_match(site, match, require_result_type=False):
        return None
    return Claimed(
        rule_id=FUSION_RULE,
        result_type=match.result_type,
        operand_mode="leaves",
    )
