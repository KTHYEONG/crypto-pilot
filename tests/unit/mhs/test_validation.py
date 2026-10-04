"""Tests for the MHS application validation module."""

from __future__ import annotations

import re

import pytest

from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.validation import validate_request
from src.mhs.params import COMMITTEE_TARGET_GROSS_UNSET


def test_validate_request_committee_member_set() -> None:
    """committee_member_set validation accepts valid choices."""
    req = MhsDiagnosticRequest(
        committee_capital=True, committee_member_set="risk_premia"
    )
    # Should not raise
    validate_request(req, COMMITTEE_TARGET_GROSS_UNSET)


def test_validate_request_pnl_vol_target_mode_growth_budget() -> None:
    """pnl_vol_target_mode accepts 'growth_budget'."""
    req = MhsDiagnosticRequest(pnl_vol_target_mode="growth_budget")
    # Should not raise
    validate_request(req, COMMITTEE_TARGET_GROSS_UNSET)


def test_request_rejects_unknown_data_policy() -> None:
    import pytest

    from src.mhs.contracts import MhsDiagnosticRequest

    with pytest.raises(ValueError, match="data_policy"):
        MhsDiagnosticRequest(data_policy="zombie_mask_v9")  # type: ignore[arg-type]


def test_bogus_liquidity_model_rejected() -> None:
    from src.mhs.pipeline.config import MhsRunConfig

    with pytest.raises(ValueError, match=re.escape("unknown liquidity_cost_model 'bogus'")):
        MhsDiagnosticRequest(liquidity_cost_model="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=re.escape("unknown liquidity_cost_model 'bogus'")):
        MhsRunConfig(liquidity_cost_model="bogus")  # type: ignore[arg-type]
    MhsDiagnosticRequest(liquidity_cost_model="corwin_schultz")
    MhsRunConfig(liquidity_cost_model="corwin_schultz")


def test_run_config_is_validated_at_construction() -> None:
    from src.mhs.pipeline.config import MhsRunConfig

    cases = [
        ({"committee_capital": False}, "committee_kelly_sizing requires committee_book=True or committee_capital=True"),
        (
            {"pnl_vol_target_mode": "median_relative"},
            "exposure_scale_two_sided requires pnl_vol_target_mode='exante_target', 'growth_budget', or 'constant_risk'",
        ),
        ({"trend_sleeve_gross": 5.0}, "trend_sleeve_gross must be in [0.0, 1.0]"),
        ({"execution_universe_size": 1}, "execution_universe_size must be >= 8"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=re.escape(message)):
            MhsRunConfig(**kwargs)  # type: ignore[arg-type]


def test_replace_revalidates() -> None:
    import dataclasses

    from src.mhs.pipeline.config import MhsRunConfig

    with pytest.raises(ValueError, match=">= 8"):
        dataclasses.replace(MhsRunConfig(), execution_universe_size=1)


def test_inactive_member_set_must_be_canonical() -> None:
    with pytest.raises(
        ValueError, match=re.escape("committee_member_set requires committee_capital=True")
    ):
        MhsDiagnosticRequest(committee_capital=False, committee_member_set="flow_momentum")
    MhsDiagnosticRequest(committee_capital=False, committee_member_set="risk_premia")
    MhsDiagnosticRequest(committee_capital=True, committee_member_set="flow_momentum")


def test_rule_table_parity() -> None:
    from src.mhs.validation import INERT_DEPENDENT_RULES

    samples = {
        "committee_member_set": "flow_momentum",
        "committee_tranche_count": 5,
        "committee_target_gross": 1.0,
        "funding_carry_weight": 0.25,
        "trend_sleeve_gross": 0.15,
    }
    for rule in INERT_DEPENDENT_RULES:
        kwargs: dict = {
            "committee_capital": False,
            "committee_tranche_smoothing": False,
            "committee_regime_adaptive_tranche": False,
            "funding_carry_sleeve": False,
            "trend_sleeve": False,
            rule.field: samples[rule.field],
        }
        if rule.field == "committee_target_gross":
            kwargs["committee_target_gross"] = samples[rule.field]
        with pytest.raises(ValueError, match=re.escape(rule.message)):
            MhsDiagnosticRequest(**kwargs)  # type: ignore[arg-type]


def test_canonical_inert_constant_matches_identity_baseline() -> None:
    from src.mhs.params import COMMITTEE_MEMBER_SET_INERT, COMMITTEE_TRANCHE_COUNT
    from src.mhs.run_history import TRIAL_IDENTITY_BASELINE

    assert TRIAL_IDENTITY_BASELINE["committee_member_set"] == COMMITTEE_MEMBER_SET_INERT
    assert TRIAL_IDENTITY_BASELINE["committee_tranche_count"] == COMMITTEE_TRANCHE_COUNT
