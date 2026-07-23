"""Claim coverage for exact unary NumPy module calls."""

from __future__ import annotations

import pytest

from rextio.config.schema import RextioConfig
from rextio.plugins.api import Claimed, ClaimSite, KeywordArg, NotCovered

from rextio_numpy.claim import claim
from rextio_numpy.claim.unary import try_claim
from rextio_numpy.diagnostics import F32_1D, F32_2D, F64_1D, F64_2D, I64_1D, I64_2D

CONFIG = RextioConfig()


def site(
    target: str, operand_types: tuple[str | None, ...], *, keywords: tuple[KeywordArg, ...] = ()
) -> ClaimSite:
    return ClaimSite(kind="call", target=target, operand_types=operand_types, file_path="", line=0, column=0, keywords=keywords)


@pytest.mark.parametrize("target", ["numpy.negative", "numpy.absolute", "numpy.abs", "numpy.square"])
@pytest.mark.parametrize("array_type", [F64_1D, F64_2D, F32_1D, F32_2D, I64_1D, I64_2D])
def test_claim_exact_unary_module_calls(target: str, array_type: str) -> None:
    expected = Claimed("rextio-numpy/unary-module", array_type)
    assert try_claim(site(target, (array_type,))) == expected
    assert claim(site(target, (array_type,)), CONFIG) == expected


@pytest.mark.parametrize(
    "bad_site",
    [
        site("numpy.negative", (F64_1D, F64_1D)),
        site("numpy.abs", (F64_1D,), keywords=(KeywordArg(name="out", arg_type="None"),)),
        site("numpy.absolute", (F64_1D,), keywords=(KeywordArg(name="where", arg_type="bool"),)),
        site("numpy.square", (F64_1D,), keywords=(KeywordArg(name="dtype", arg_type="str"),)),
        site("numpy.negative", (None,)),
        site("numpy.negative", ("list[float]",)),
        site("numpy.sqrt", (F64_1D,)),
    ],
)
def test_unary_rejects_only_known_bad_operand_types_and_falls_back_for_forms(
    bad_site: ClaimSite,
) -> None:
    result = claim(bad_site, CONFIG)
    if bad_site.target == "numpy.negative" and bad_site.operand_types == ("list[float]",):
        assert result.diagnostic.code == "RXTP-NUMPY-010"
    else:
        assert result == NotCovered()
