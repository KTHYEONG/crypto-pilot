"""Tests for the MHS params module."""

from __future__ import annotations

from src.mhs.params import (
    COMMITTEE_DEFAULT_MEMBER_SET,
    COMMITTEE_MEMBER_SETS,
    COMMITTEE_MEMBERS,
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
