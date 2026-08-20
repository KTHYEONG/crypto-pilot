"""Tests for the MHS Research-GO gate module."""

from __future__ import annotations

import pytest

from src.application.research.mhs.contracts import MhsDiagnosticRequest
from src.application.research.mhs.research_go import _resolved_committee_members
from src.mhs.params import MHS_COMMITTEE_MEMBER_SETS


def test_resolved_committee_members_risk_premia_v2() -> None:
    req = MhsDiagnosticRequest(committee_capital=True, committee_member_set="risk_premia_v2")
    result = _resolved_committee_members(req)
    assert result == MHS_COMMITTEE_MEMBER_SETS["risk_premia_v2"]
    assert len(result) == 5


def test_resolved_committee_members_flow_momentum_v1() -> None:
    req = MhsDiagnosticRequest(committee_capital=True, committee_member_set="flow_momentum_v1")
    result = _resolved_committee_members(req)
    assert result == MHS_COMMITTEE_MEMBER_SETS["flow_momentum_v1"]
    assert len(result) == 5


def test_resolved_committee_members_unregistered_raises() -> None:
    req = MhsDiagnosticRequest(committee_capital=True, committee_member_set="risk_premia_v2")
    # Simulate an unregistered key by replacing the field (bypassing validation)
    object.__setattr__(req, "committee_member_set", "unregistered")
    with pytest.raises(ValueError, match="unknown committee_member_set"):
        _resolved_committee_members(req)
