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
        "balanced", "conservative", "growth", "growth_extreme", "growth_moderate",
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
        "balanced", "conservative", "growth", "growth_extreme", "growth_moderate",
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


# SCENARIO_MHS_KELLY_TWO_SIDED_05
def test_scenario_mhs_kelly_two_sided_05_registered_roster_cap_and_extreme_rung() -> None:
    from src.mhs.types import REGISTERED_POLICY_THRESHOLDS

    assert REGISTERED_POLICY_THRESHOLDS == {
        "cap_60_roster": 60.0, "primary_annual_return": 0.05,
        "deflated_sharpe_ratio": 0.95, "max_drawdown_budget_ceiling": 0.60,
    }
    assert "cap_30_roster" not in REGISTERED_POLICY_THRESHOLDS
    extreme = GROWTH_RISK_ENVELOPES["growth_extreme"]
    assert extreme.leverage_ceiling == 3.0
    assert sorted(GROWTH_RISK_ENVELOPES) == [
        "balanced", "conservative", "growth", "growth_extreme", "growth_moderate",
    ]
    for env in GROWTH_RISK_ENVELOPES.values():
        assert env.ruin_fraction == COMMITTEE_GROWTH_RUIN_FRACTION
        assert env.max_ruin_prob == COMMITTEE_GROWTH_MAX_RUIN_PROB


# SCENARIO_MHS_CONSTANT_RISK_PARAMS_REGISTERED
def test_constant_risk_params_registered() -> None:
    from src.mhs.params import (
        CONSTANT_RISK_CAP_BINDING_QUANTILE,
        CONSTANT_RISK_EWMA_HALFLIFE_DAYS,
        CONSTANT_RISK_MIN_PERIODS_DAYS,
        CONSTANT_RISK_TARGET_ANNUAL_VOL,
        FOLD_REALIZED_RISK_PARITY_TOLERANCE,
        PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS,
    )

    assert CONSTANT_RISK_EWMA_HALFLIFE_DAYS == 90
    assert CONSTANT_RISK_MIN_PERIODS_DAYS == 45
    assert CONSTANT_RISK_TARGET_ANNUAL_VOL == 0.40
    assert 0.0 < CONSTANT_RISK_CAP_BINDING_QUANTILE <= 0.25
    assert FOLD_REALIZED_RISK_PARITY_TOLERANCE == 0.35
    # 기존 모드용 EWMA 반감기는 불변이다.
    assert PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS == 20


# SCENARIO_MHS_EVIDENCE_GATE_ALPHA_REGISTERED
def test_evidence_gate_alpha_registered() -> None:
    import src.mhs.params as params_module
    from src.mhs.params import (
        EVIDENCE_GATE_ALPHA,
        NULL_BOOTSTRAP_MEAN_BLOCK_DAYS,
        NULL_BOOTSTRAP_MIN_ROWS,
        NULL_BOOTSTRAP_TRIALS,
        NULL_BOOTSTRAP_SEED,
    )

    assert EVIDENCE_GATE_ALPHA == 0.05
    assert 0.0 < EVIDENCE_GATE_ALPHA < 0.5
    assert NULL_BOOTSTRAP_MEAN_BLOCK_DAYS == 20
    assert NULL_BOOTSTRAP_TRIALS >= 1000
    assert NULL_BOOTSTRAP_MIN_ROWS >= 250
    assert NULL_BOOTSTRAP_SEED > 0
    # 유도 근거 없는 고정 임계값은 선언형 alpha로 교체되어 삭제되다.
    assert not hasattr(params_module, "FOLD_GROWTH_CONCENTRATION_MAX_SHARE")


# SCENARIO_MHS_DSR_06_REGISTERED_THRESHOLD_UNCHANGED
def test_SCENARIO_MHS_DSR_06_REGISTERED_THRESHOLD_UNCHANGED() -> None:
    """A4 guard: the registered DSR pass line and the drawdown-budget ceiling
    stay at their preregistered values -- no future statistical loosening."""
    from src.mhs.types import REGISTERED_POLICY_THRESHOLDS

    assert REGISTERED_POLICY_THRESHOLDS["deflated_sharpe_ratio"] == 0.95
    assert REGISTERED_POLICY_THRESHOLDS["max_drawdown_budget_ceiling"] == 0.60


def test_SCENARIO_MHS_EVID_02_SELECTION_OVERLAP_IS_DISCLOSED() -> None:
    """SCENARIO_MHS_EVID_02_SELECTION_OVERLAP_IS_DISCLOSED: the fraction of the
    report window inside the defaults' selection window is 1.0 when identical,
    0.0 when disjoint, fractional when partially overlapping, and fails closed
    on an inverted window."""
    import pandas as pd

    from src.mhs.evidence import selection_overlap_fraction
    from src.mhs.params import DEFAULT_SELECTION_WINDOW

    registered_window = DEFAULT_SELECTION_WINDOW
    assert registered_window == (
        pd.Timestamp("2021-01-01", tz="UTC"),
        pd.Timestamp("2025-12-31", tz="UTC"),
    )

    full = selection_overlap_fraction(
        pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")
    )
    assert full == 1.0

    disjoint = selection_overlap_fraction(
        pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")
    )
    assert disjoint == 0.0

    window_start, window_end = DEFAULT_SELECTION_WINDOW
    partial = selection_overlap_fraction(window_start, window_end + (window_end - window_start))
    assert 0.0 < partial < 1.0
    assert partial == pytest.approx(0.5)

    with pytest.raises(ValueError, match="report_end"):
        selection_overlap_fraction(
            pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")
        )

    zero_length = selection_overlap_fraction(
        pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")
    )
    assert zero_length == 0.0
