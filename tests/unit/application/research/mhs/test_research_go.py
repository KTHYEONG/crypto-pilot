"""Tests for the MHS Research-GO gate module."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.application.research.mhs.contracts import (
    MhsDiagnosticRequest,
    MhsFoldReport,
)
from src.application.research.mhs.research_go import (
    GO_REASON_DRAWDOWN_BUDGET_NON_BINDING,
    _drawdown_budget_reasons,
    _mhs_research_go,
    _pooled_level_gate_reasons,
    _resolved_committee_members,
    _resolved_growth_envelope,
)
from src.mhs.params import COMMITTEE_MEMBER_SETS, GROWTH_RISK_ENVELOPES
from src.mhs.types import REGISTERED_POLICY_THRESHOLDS


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
            if env.max_drawdown > REGISTERED_POLICY_THRESHOLDS["max_drawdown_budget_ceiling"]:
                # A non-binding budget blocks regardless of the observed value.
                assert _drawdown_budget_reasons(None, env.max_drawdown) == (
                    GO_REASON_DRAWDOWN_BUDGET_NON_BINDING,
                )
            else:
                assert _drawdown_budget_reasons(None, env.max_drawdown) == ()

    def test_nan_primary_returns_empty(self) -> None:
        for env in GROWTH_RISK_ENVELOPES.values():
            if env.max_drawdown > REGISTERED_POLICY_THRESHOLDS["max_drawdown_budget_ceiling"]:
                assert _drawdown_budget_reasons(float("nan"), env.max_drawdown) == (
                    GO_REASON_DRAWDOWN_BUDGET_NON_BINDING,
                )
            else:
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


# SCENARIO_MHS_DSR_05_NON_BINDING_BUDGET_BLOCKS_GO
def test_SCENARIO_MHS_DSR_05_NON_BINDING_BUDGET_BLOCKS_GO() -> None:
    # A budget at/above 1.0 can never bind (-100% is capital extinction), so
    # it must block instead of silently passing -- regardless of the observed
    # drawdown value.
    assert _drawdown_budget_reasons(-0.389, max_drawdown=1.0) == (
        GO_REASON_DRAWDOWN_BUDGET_NON_BINDING,
    )
    assert _drawdown_budget_reasons(-0.95, max_drawdown=1.0) == (
        GO_REASON_DRAWDOWN_BUDGET_NON_BINDING,
    )
    # A binding budget keeps the original breach / pass semantics.
    assert _drawdown_budget_reasons(-0.389, max_drawdown=0.25) == (
        "PRIMARY_MAX_DRAWDOWN_OVER_BUDGET",
    )
    assert _drawdown_budget_reasons(-0.10, max_drawdown=0.25) == ()
    with pytest.raises(ValueError, match="max_drawdown"):
        _drawdown_budget_reasons(-0.10, max_drawdown=0.0)


# SCENARIO_MHS_DSR_05_NON_BINDING_BUDGET_BLOCKS_GO (ceiling boundary)
def test_non_binding_code_fires_only_above_registered_ceiling() -> None:
    ceiling = REGISTERED_POLICY_THRESHOLDS["max_drawdown_budget_ceiling"]
    assert ceiling == 0.60
    # At the ceiling the budget can still bind: no non-binding code.
    assert _drawdown_budget_reasons(-0.10, max_drawdown=ceiling) == ()
    # Strictly above it: always blocked.
    assert _drawdown_budget_reasons(-0.10, max_drawdown=ceiling + 1e-9) == (
        GO_REASON_DRAWDOWN_BUDGET_NON_BINDING,
    )


# SCENARIO_MHS_POOLED_LEVEL_GATE_REPLACES_MIN_FOLD
class TestPooledLevelGateReasons:
    def test_strong_pooled_evidence_passes(self) -> None:
        payload: dict[str, object] = {
            "n_measured_folds": 4,
            "pooled_sharpe_lcb": 1.28,
            "pooled_stress_sharpe_lcb": 0.5,
            "pooled_annual_log_return": 0.3,
        }
        assert _pooled_level_gate_reasons(payload) == ()

    def test_low_primary_sharpe_lcb_flags_only_primary_sharpe(self) -> None:
        payload: dict[str, object] = {
            "n_measured_folds": 4,
            "pooled_sharpe_lcb": 0.4,
            "pooled_stress_sharpe_lcb": 0.5,
            "pooled_annual_log_return": 0.3,
        }
        assert _pooled_level_gate_reasons(payload) == ("PRIMARY_AUTOCORR_SHARPE_BELOW_0_6",)

    def test_negative_stress_sharpe_lcb_flags_only_stress(self) -> None:
        payload: dict[str, object] = {
            "n_measured_folds": 4,
            "pooled_sharpe_lcb": 1.28,
            "pooled_stress_sharpe_lcb": -0.1,
            "pooled_annual_log_return": 0.3,
        }
        assert _pooled_level_gate_reasons(payload) == ("STRESS_SHARPE_NOT_POSITIVE",)

    def test_below_return_floor_flags_return_code(self) -> None:
        payload: dict[str, object] = {
            "n_measured_folds": 4,
            "pooled_sharpe_lcb": 1.28,
            "pooled_stress_sharpe_lcb": 0.5,
            "pooled_annual_log_return": 0.01,
        }
        codes = _pooled_level_gate_reasons(payload)
        assert codes == ("PRIMARY_ANNUAL_RETURN_BELOW_FLOOR",)

    def test_fewer_than_two_measured_folds_defers_to_incomplete_fold(self) -> None:
        payload: dict[str, object] = {
            "n_measured_folds": 1,
            "pooled_sharpe_lcb": -1.0,
            "pooled_stress_sharpe_lcb": -1.0,
            "pooled_annual_log_return": -1.0,
        }
        assert _pooled_level_gate_reasons(payload) == ()

    def test_explicit_return_floor_overrides_registered_policy(self) -> None:
        payload: dict[str, object] = {
            "n_measured_folds": 4,
            "pooled_sharpe_lcb": 1.28,
            "pooled_stress_sharpe_lcb": 0.5,
            "pooled_annual_log_return": 0.3,
        }
        assert _pooled_level_gate_reasons(payload, return_floor=0.5) == (
            "PRIMARY_ANNUAL_RETURN_BELOW_FLOOR",
        )


def _gate_fold(
    failures: tuple[str, ...],
    primary_valid: bool = True,
) -> MhsFoldReport:
    return MhsFoldReport(
        fold_index=0,
        validation_start="2021-02-10",
        validation_end="2021-04-19",
        strict=SimpleNamespace(),  # 완결 fold: INCOMPLETE 코드 유발하지 않는 최소 스텁
        stress=None,
        primary_valid=primary_valid,
        primary_autocorr_sharpe=0.1,
        primary_naive_sharpe=0.1,
        primary_net_ann=0.01,
        primary_geometric_cagr=0.01,
        primary_max_drawdown=-0.1,
        stress_naive_sharpe=0.0,
        decision_intents=0,
        termination_counts={},
        failures=failures,
        strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
    )


# SCENARIO_MHS_FOLD_LEVEL_CODES_NO_LONGER_PER_FOLD
def test_level_codes_no_longer_derived_per_fold() -> None:
    # 경제하한 미달(autocorr 0.1 < 0.6) fold라도 무결성 문제가 없으면
    # fold별 level 코드가 붙지 않는다(level은 pooled 게이트 소유).
    result = _mhs_research_go((_gate_fold(()),), deflated_sharpe_ratio=0.96)
    assert result.reason_codes == ()
    assert result.eligible is True


# SCENARIO_MHS_FOLD_LEVEL_CODES_NO_LONGER_PER_FOLD
def test_invalid_primary_integrity_code_still_blocks_fold() -> None:
    result = _mhs_research_go(
        (_gate_fold(("INVALID_PRIMARY_LEDGER",), primary_valid=False),),
    )
    assert "INVALID_PRIMARY_LEDGER" in result.reason_codes
    assert "PRIMARY_AUTOCORR_SHARPE_BELOW_0_6" not in result.reason_codes
