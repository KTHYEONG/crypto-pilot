"""Tests for the MHS params module."""

from __future__ import annotations

from src.mhs.params import (
    MHS_COMMITTEE_DEFAULT_MEMBER_SET,
    MHS_COMMITTEE_MEMBER_SETS,
    MHS_COMMITTEE_MEMBERS,
)


def test_committee_member_sets_keys() -> None:
    assert set(MHS_COMMITTEE_MEMBER_SETS.keys()) == {"flow_momentum_v1", "risk_premia_v2"}


def test_default_member_set_is_flow_momentum_v1() -> None:
    # risk_premia_v2 was measured non-adopted (full 3m replay breached the
    # registered drawdown budget); see ADR_20260820_MHS_COMPOUNDING_ALPHA_AXES.
    assert MHS_COMMITTEE_DEFAULT_MEMBER_SET == "flow_momentum_v1"
    assert MHS_COMMITTEE_MEMBER_SETS["flow_momentum_v1"] == MHS_COMMITTEE_MEMBERS


def test_risk_premia_v2_has_five_members() -> None:
    assert len(MHS_COMMITTEE_MEMBER_SETS["risk_premia_v2"]) == 5


def test_flow_momentum_v1_has_five_members() -> None:
    assert len(MHS_COMMITTEE_MEMBER_SETS["flow_momentum_v1"]) == 5
