"""Tests for the MHS Research-GO gate module."""

from __future__ import annotations

import pytest

from src.application.research.mhs.contracts import MhsDiagnosticRequest
from src.application.research.mhs.research_go import (
    _drawdown_budget_reasons,
    _resolved_committee_members,
    _resolved_growth_envelope,
)
from src.mhs.params import COMMITTEE_MEMBER_SETS, GROWTH_RISK_ENVELOPES


def test_resolved_committee_members_risk_premia() -> None:
    req = MhsDiagnosticRequest(committee_capital=True, committee_member_set="risk_premia")
    result = _resolved_committee_members(req)
    assert result == COMMITTEE_MEMBER_SETS["risk_premia"]
    assert len(result) == 5


def test_resolved_committee_members_flow_momentum() -> None:
    req = MhsDiagnosticRequest(committee_capital=True, committee_member_set="flow_momentum")
    result = _resolved_committee_members(req)
    assert result == COMMITTEE_MEMBER_SETS["flow_momentum"]
    assert len(result) == 5


def test_resolved_committee_members_unregistered_raises() -> None:
    req = MhsDiagnosticRequest(committee_capital=True, committee_member_set="risk_premia")
    # Simulate an unregistered key by replacing the field (bypassing validation)
    object.__setattr__(req, "committee_member_set", "unregistered")
    with pytest.raises(ValueError, match="unknown committee_member_set"):
        _resolved_committee_members(req)


# SCENARIO_DRAWDOWN_GATE_READS_ENVELOPE_NOT_CONSTANT
class TestDrawdownGateReadsEnvelope:
    def test_conservative_blocks_over_budget(self) -> None:
        reasons = _drawdown_budget_reasons(
            -0.30, GROWTH_RISK_ENVELOPES["conservative"].max_drawdown,
        )
        assert reasons == ("PRIMARY_MAX_DRAWDOWN_OVER_BUDGET",)

    def test_balanced_allows_same_drawdown(self) -> None:
        reasons = _drawdown_budget_reasons(
            -0.30, GROWTH_RISK_ENVELOPES["balanced"].max_drawdown,
        )
        assert reasons == ()

    def test_none_primary_returns_empty(self) -> None:
        for env in GROWTH_RISK_ENVELOPES.values():
            assert _drawdown_budget_reasons(None, env.max_drawdown) == ()

    def test_nan_primary_returns_empty(self) -> None:
        for env in GROWTH_RISK_ENVELOPES.values():
            assert _drawdown_budget_reasons(float("nan"), env.max_drawdown) == ()


class TestResolvedGrowthEnvelope:
    def test_default_returns_conservative(self) -> None:
        req = MhsDiagnosticRequest()
        env = _resolved_growth_envelope(req)
        assert env is GROWTH_RISK_ENVELOPES["conservative"]

    def test_balanced_returns_balanced(self) -> None:
        req = MhsDiagnosticRequest(growth_envelope="balanced")
        env = _resolved_growth_envelope(req)
        assert env is GROWTH_RISK_ENVELOPES["balanced"]

    def test_unregistered_key_raises(self) -> None:
        req = MhsDiagnosticRequest()
        object.__setattr__(req, "growth_envelope", "unregistered")
        with pytest.raises(ValueError, match="unknown growth_envelope"):
            _resolved_growth_envelope(req)
