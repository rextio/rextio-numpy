"""Claims for bounded three-positional-argument ``numpy.where``."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_numpy.diagnostics import (
    SCALAR_FOR_DTYPE,
    array_meta,
    bool_rank,
    is_array_type,
    is_bool_type,
    not_covered_or_rejected,
    type_key_for,
)

WHERE_TARGET = "numpy.where"
WHERE_RULE = "rextio-numpy/where-three-argument"


def _branch_contract(
    yes_type: str,
    no_type: str,
) -> tuple[str, int] | None:
    yes_array = is_array_type(yes_type)
    no_array = is_array_type(no_type)
    if not yes_array and not no_array:
        return None

    if yes_array and no_array:
        yes_meta = array_meta(yes_type)
        no_meta = array_meta(no_type)
        if yes_meta is None or no_meta is None or yes_meta[0] != no_meta[0]:
            return None
        return yes_meta[0], max(yes_meta[1], no_meta[1])

    if yes_array:
        yes_meta = array_meta(yes_type)
        if yes_meta is None or no_type != SCALAR_FOR_DTYPE[yes_meta[0]]:
            return None
        return yes_meta

    no_meta = array_meta(no_type)
    if no_meta is None or yes_type != SCALAR_FOR_DTYPE[no_meta[0]]:
        return None
    return no_meta


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim exact three-argument ``numpy.where(condition, x, y)`` calls."""
    if site.kind != "call" or site.target != WHERE_TARGET:
        return None
    if site.receiver is not None or site.keywords or site.callables:
        return not_covered_or_rejected(site)
    if len(site.operand_types) != 3:
        return not_covered_or_rejected(site)
    condition_type, yes_type, no_type = site.operand_types
    if condition_type is None or yes_type is None or no_type is None:
        return NotCovered()
    if not is_bool_type(condition_type):
        return not_covered_or_rejected(site)
    condition_rank = bool_rank(condition_type)
    branch = _branch_contract(yes_type, no_type)
    if condition_rank is None or branch is None:
        return not_covered_or_rejected(site)
    dtype, branch_rank = branch
    return Claimed(
        rule_id=WHERE_RULE,
        result_type=type_key_for(dtype, max(condition_rank, branch_rank)),
    )


__all__ = ["WHERE_RULE", "WHERE_TARGET", "try_claim"]
