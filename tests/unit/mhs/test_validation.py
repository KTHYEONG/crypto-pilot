"""Tests for the MHS application validation module."""

from __future__ import annotations

from tests.fixtures.mhs_requests import research_baseline
import re

import pytest

from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.validation import validate_request


def test_validate_request_committee_member_set() -> None:
    """committee_member_set validation accepts valid choices."""
    req = research_baseline(
        committee_capital=True, committee_member_set="risk_premia"
    )
    # Should not raise
    validate_request(req)


def test_validate_request_pnl_vol_target_mode_growth_budget() -> None:
    """pnl_vol_target_mode accepts 'growth_budget'."""
    req = research_baseline(pnl_vol_target_mode="growth_budget")
    # Should not raise
    validate_request(req)


def test_request_rejects_unknown_data_policy() -> None:
    import pytest


    with pytest.raises(ValueError, match="data_policy"):
        research_baseline(data_policy="zombie_mask_v9")  # type: ignore[arg-type]


def test_bogus_liquidity_model_rejected() -> None:

    with pytest.raises(ValueError, match=re.escape("unknown liquidity_cost_model 'bogus'")):
        research_baseline(liquidity_cost_model="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=re.escape("unknown liquidity_cost_model 'bogus'")):
        MhsDiagnosticRequest(liquidity_cost_model="bogus")  # type: ignore[arg-type]
    research_baseline(liquidity_cost_model="corwin_schultz")
    MhsDiagnosticRequest(liquidity_cost_model="corwin_schultz")


def test_run_config_is_validated_at_construction() -> None:

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
            MhsDiagnosticRequest(**kwargs)  # type: ignore[arg-type]


def test_replace_revalidates() -> None:
    import dataclasses


    with pytest.raises(ValueError, match=">= 8"):
        dataclasses.replace(MhsDiagnosticRequest(), execution_universe_size=1)


def test_inactive_member_set_must_be_canonical() -> None:
    with pytest.raises(
        ValueError, match=re.escape("committee_member_set requires committee_capital=True")
    ):
        research_baseline(committee_capital=False, committee_member_set="flow_momentum")
    research_baseline(committee_capital=False, committee_member_set="risk_premia")
    research_baseline(committee_capital=True, committee_member_set="flow_momentum")


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
            research_baseline(**kwargs)  # type: ignore[arg-type]


def test_canonical_inert_constant_matches_identity_baseline() -> None:
    from src.mhs.params import COMMITTEE_MEMBER_SET_INERT, COMMITTEE_TRANCHE_COUNT
    from src.mhs.run_history import TRIAL_IDENTITY_BASELINE

    assert TRIAL_IDENTITY_BASELINE["committee_member_set"] == COMMITTEE_MEMBER_SET_INERT
    assert TRIAL_IDENTITY_BASELINE["committee_tranche_count"] == COMMITTEE_TRANCHE_COUNT


def test_choice_errors_name_registered_set() -> None:
    with pytest.raises(ValueError, match=re.escape("unknown growth_envelope 'nope'")) as exc_info:
        MhsDiagnosticRequest(growth_envelope="nope")  # type: ignore[arg-type]
    assert "growth_extreme_budgeted" in str(exc_info.value)
    assert "registered:" in str(exc_info.value)


def test_production_envelope_accepted() -> None:
    from src.mhs.params import GROWTH_RISK_ENVELOPES

    for g in GROWTH_RISK_ENVELOPES:
        validate_request(MhsDiagnosticRequest(growth_envelope=g))


def test_bounds_helper_gone() -> None:
    import src.mhs.validation as validation

    assert not hasattr(validation, "_validate_field_bounds")


@pytest.mark.parametrize("field", ["placebo_diagnostic", "phase_diagnostic", "signal_48h_diagnostic", "bootstrap_ci_diagnostic", "reference_books_diagnostic", "patient_reference_diagnostic"])
def test_non_bool_report_flag_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a bool"):
        research_baseline(**{field: 1})  # type: ignore[arg-type]
