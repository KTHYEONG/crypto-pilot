"""Strategy research configuration for MHS golden fixtures and unit suites.

``research_baseline`` reproduces the pre-unification ``MhsDiagnosticRequest()``
defaults (the configuration every golden fixture was captured under) on top of
the unified production-default request type. Test-only; never imported from
``src/``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final
from collections.abc import Mapping

from src.lab.mhs.contracts import MhsDiagnosticRequest
from src.core.params import (COMMITTEE_TARGET_GROSS, GROWTH_ENVELOPE_DEFAULT)
from src.lab.mhs.params import COMMITTEE_MEMBER_SET_INERT

RESEARCH_BASELINE: Final[Mapping[str, object]] = MappingProxyType({
    "execution_universe_size": 30,
    "pnl_vol_target_mode": "median_relative",
    "committee_kelly_sizing": False,
    "committee_capital": False,
    "committee_member_set": COMMITTEE_MEMBER_SET_INERT,
    "committee_regime_adaptive_tranche": False,
    "committee_evidence_weighting": False,
    "funding_carry_sleeve": False,
    "funding_carry_weight": 0.0,
    "exposure_scale_two_sided": False,
    "growth_envelope": GROWTH_ENVELOPE_DEFAULT,
})


def research_baseline(**overrides: object) -> MhsDiagnosticRequest:
    """The strategy research configuration the golden fixtures and unit suites were captured under.

    Equals the pre-unification ``MhsDiagnosticRequest()`` defaults. When
    ``committee_target_gross`` is not overridden it resolves exactly as the
    retired unset sentinel did: ``COMMITTEE_TARGET_GROSS`` if the resulting
    request has committee capital on, else ``None``. Overrides are applied on
    top and validated by the request itself.
    """
    params = dict(RESEARCH_BASELINE)
    if "committee_target_gross" not in overrides:
        capital = overrides.get("committee_capital", params["committee_capital"])
        params["committee_target_gross"] = COMMITTEE_TARGET_GROSS if capital else None
    params.update(overrides)
    return MhsDiagnosticRequest(**params)  # type: ignore[arg-type]
