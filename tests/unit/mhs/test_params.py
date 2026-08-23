"""Tests for the MHS params module."""

from __future__ import annotations

import pytest

from src.mhs.params import (
    COMMITTEE_DEFAULT_MEMBER_SET,
    COMMITTEE_GROWTH_MAX_DRAWDOWN,
    COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB,
    COMMITTEE_GROWTH_MAX_RUIN_PROB,
    COMMITTEE_GROWTH_HORIZON_YEARS,
    COMMITTEE_GROWTH_RUIN_FRACTION,
    COMMITTEE_MEMBER_SETS,
    COMMITTEE_MEMBERS,
    GROWTH_ENVELOPE_DEFAULT,
    GROWTH_RISK_ENVELOPES,
    GrowthRiskEnvelope,
)


def test_committee_member_sets_keys() -> None:
    assert set(COMMITTEE_MEMBER_SETS.keys()) == {"flow_momentum", "risk_premia"}


def test_default_member_set_is_flow_momentum() -> None:
    # risk_premia was measured non-adopted (full 3m replay breached the
    # registered drawdown budget); see ADR_20260820_MHS_COMPOUNDING_ALPHA_AXES.
    assert COMMITTEE_DEFAULT_MEMBER_SET == "flow_momentum"
    assert COMMITTEE_MEMBER_SETS["flow_momentum"] == COMMITTEE_MEMBERS


def test_risk_premia_has_five_members() -> None:
    assert len(COMMITTEE_MEMBER_SETS["risk_premia"]) == 5


def test_flow_momentum_has_five_members() -> None:
    assert len(COMMITTEE_MEMBER_SETS["flow_momentum"]) == 5


# SCENARIO_GROWTH_ENVELOPE_CONSERVATIVE_IS_BYTE_IDENTICAL
def test_growth_envelope_conservative_is_byte_identical() -> None:
    conservative = GROWTH_RISK_ENVELOPES["conservative"]
    assert conservative.max_drawdown == COMMITTEE_GROWTH_MAX_DRAWDOWN
    assert conservative.max_drawdown_prob == COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB
    assert conservative.ruin_fraction == COMMITTEE_GROWTH_RUIN_FRACTION
    assert conservative.max_ruin_prob == COMMITTEE_GROWTH_MAX_RUIN_PROB
    assert conservative.horizon_years == COMMITTEE_GROWTH_HORIZON_YEARS
    assert conservative.leverage_ceiling == 1.0
    assert GROWTH_ENVELOPE_DEFAULT == "conservative"
    assert sorted(GROWTH_RISK_ENVELOPES) == [
        "balanced", "conservative", "growth", "growth_moderate",
    ]
    for env in GROWTH_RISK_ENVELOPES.values():
        assert env.ruin_fraction == COMMITTEE_GROWTH_RUIN_FRACTION
        assert env.max_ruin_prob == COMMITTEE_GROWTH_MAX_RUIN_PROB


# SCENARIO_GROWTH_ENVELOPE_FAIL_CLOSED_BOUNDS
def test_growth_envelope_fail_closed_bounds() -> None:
    base = {
        "name": "test", "max_drawdown": 0.25, "max_drawdown_prob": 0.10,
        "ruin_fraction": 0.60, "max_ruin_prob": 0.01, "horizon_years": 3.0,
        "leverage_ceiling": 1.0,
    }
    with pytest.raises(ValueError, match="leverage_ceiling"):
        GrowthRiskEnvelope(**{**base, "leverage_ceiling": 0.99})
    with pytest.raises(ValueError, match="max_drawdown"):
        GrowthRiskEnvelope(**{**base, "max_drawdown": 0.0})
    with pytest.raises(ValueError, match="max_drawdown_prob"):
        GrowthRiskEnvelope(**{**base, "max_drawdown_prob": 0.0})
    with pytest.raises(ValueError, match="max_drawdown_prob"):
        GrowthRiskEnvelope(**{**base, "max_drawdown_prob": 1.5})
    with pytest.raises(ValueError, match="ruin_fraction"):
        GrowthRiskEnvelope(**{**base, "ruin_fraction": 1.0})
    with pytest.raises(ValueError, match="horizon_years"):
        GrowthRiskEnvelope(**{**base, "horizon_years": 0.0})
    # Valid construction with leverage_ceiling=2.0 succeeds and is frozen
    env = GrowthRiskEnvelope(**{**base, "leverage_ceiling": 2.0})
    assert env.leverage_ceiling == 2.0
    with pytest.raises(AttributeError):
        env.max_drawdown = 0.30  # type: ignore[misc]


# SCENARIO_MHS_EXPOSURE_CEILING_05
def test_scenario_mhs_exposure_ceiling_05_growth_moderate_rung_matches_growth() -> None:
    moderate = GROWTH_RISK_ENVELOPES["growth_moderate"]
    growth = GROWTH_RISK_ENVELOPES["growth"]
    assert sorted(GROWTH_RISK_ENVELOPES) == [
        "balanced", "conservative", "growth", "growth_moderate",
    ]
    assert moderate.leverage_ceiling == 1.5
    assert moderate.max_drawdown == growth.max_drawdown
    assert moderate.max_drawdown_prob == growth.max_drawdown_prob
    assert moderate.ruin_fraction == growth.ruin_fraction
    assert moderate.max_ruin_prob == growth.max_ruin_prob
    assert moderate.horizon_years == growth.horizon_years
    for envelope in GROWTH_RISK_ENVELOPES.values():
        assert envelope.ruin_fraction == COMMITTEE_GROWTH_RUIN_FRACTION
        assert envelope.max_ruin_prob == COMMITTEE_GROWTH_MAX_RUIN_PROB
    assert GROWTH_ENVELOPE_DEFAULT == "conservative"
