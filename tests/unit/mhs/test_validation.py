"""Tests for the MHS application validation module."""

from __future__ import annotations

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
