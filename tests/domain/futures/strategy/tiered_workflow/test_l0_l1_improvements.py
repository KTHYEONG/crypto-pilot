import numpy as np
import pytest

from src.domain.futures.alpha_foundry.contracts import (
    CausalFeedbackError,
    L1CausalFeedback,
    SignalHypothesisKey,
    resolve_l1_feedback_multiplier,
)
from src.domain.futures.strategy.candidate_contracts import Layer1FoldReadiness
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.tiered_workflow.evidence_policy import assess_fold_evidence
from src.domain.futures.strategy.tiered_workflow.signal_selection import _compute_pooled_probe_lcb


def test_dynamic_cost_injection() -> None:
    # Scenario 1 Verification
    gross = np.array([10.0, 15.0, 20.0])
    exec_cost = np.array([2.0, 2.5, 3.0])
    funding_cost = np.array([0.5, 0.5, 0.5])

    readiness = Layer1FoldReadiness(
        fold_id=1,
        registry_source_end_idx=100,
        outer_oos_start_idx=10,
        outer_oos_end_idx=20,
        ready_symbols=("BTCUSDT",),
        probe_series_bps=tuple(gross),
        dynamic_funding_cost_bps=tuple(funding_cost),
        dynamic_execution_cost_bps=tuple(exec_cost),
        funding_observed=(True, True, True),
        cost_observed=(True, True, True),
    )

    assert readiness.dynamic_funding_cost_bps == (0.5, 0.5, 0.5)
    assert readiness.dynamic_execution_cost_bps == (2.0, 2.5, 3.0)

    assessment = assess_fold_evidence(
        fold_id=readiness.fold_id,
        gross_series_bps=np.asarray(readiness.probe_series_bps, dtype=np.float64),
        execution_cost_bps=np.asarray(readiness.dynamic_execution_cost_bps, dtype=np.float64),
        funding_cost_bps=np.asarray(readiness.dynamic_funding_cost_bps, dtype=np.float64),
        matched_event_count=3,
        unmatched_event_count=0,
        decision_count=3,
        effective_symbol_count=1.0,
        cost_observed=np.asarray(readiness.cost_observed, dtype=np.bool_),
        funding_observed=np.asarray(readiness.funding_observed, dtype=np.bool_),
        min_matched_events=1,
        min_match_wilson_lcb=0.0,
        min_decision_count=1,
        max_cost_fallback_ratio=1.0,
        min_funding_coverage_ratio=0.0,
        block_bars=6,
        n_bootstrap=10,
        seed=42,
    )

    expected_net = gross - exec_cost - funding_cost
    assert np.allclose(assessment.net_series_bps, expected_net)


def test_causal_cutoff_violation() -> None:
    # Scenario 2 Verification
    key = SignalHypothesisKey("test_fam", "test_var", "2h")
    fb = L1CausalFeedback(
        key=key,
        outcome="deployable",
        evidence_end_ns=1000,
        effective_n=10.0,
        survival_successes=5,
        survival_trials=10,
        pooled_net_lcb_bps=5.0,
        positive_fold_ratio=0.8,
    )
    with pytest.raises(CausalFeedbackError):
        resolve_l1_feedback_multiplier(feedback=fb, current_evidence_start_ns=1000)


def test_incomplete_cost_mask() -> None:
    # Scenario 3 Verification
    gross = np.array([10.0, 15.0, 20.0])
    exec_cost = np.array([2.0, 2.5, 3.0])
    funding_cost = np.array([0.5, 0.5, 0.5])

    assessment = assess_fold_evidence(
        fold_id=1,
        gross_series_bps=gross,
        execution_cost_bps=exec_cost,
        funding_cost_bps=funding_cost,
        matched_event_count=3,
        unmatched_event_count=0,
        decision_count=3,
        effective_symbol_count=1.0,
        cost_observed=np.array([True, False, False]),
        funding_observed=np.array([True, True, True]),
        min_matched_events=1,
        min_match_wilson_lcb=0.0,
        min_decision_count=1,
        max_cost_fallback_ratio=0.50,
        min_funding_coverage_ratio=0.0,
        block_bars=6,
        n_bootstrap=10,
        seed=42,
    )
    assert "cost_data_incomplete" in assessment.blockers
    assert assessment.state == "insufficient_support"


def test_integration_pooled_probe_lcb() -> None:
    # Scenario 4: Integration Wiring Verification
    # _compute_pooled_probe_lcb가 readiness에 지정된 동적 비용을 감안해 올바른 LCB를 계산하는지 통합 검증
    gross = (10.0, 15.0, 20.0)
    exec_cost = (2.0, 2.0, 2.0)
    funding_cost = (0.5, 0.5, 0.5)

    readiness = Layer1FoldReadiness(
        fold_id=1,
        registry_source_end_idx=100,
        outer_oos_start_idx=10,
        outer_oos_end_idx=20,
        ready_symbols=("BTCUSDT",),
        probe_series_bps=gross,
        dynamic_funding_cost_bps=funding_cost,
        dynamic_execution_cost_bps=exec_cost,
        funding_observed=(True, True, True),
        cost_observed=(True, True, True),
        matched_event_count=3,
        unique_decision_count=3,
        effective_symbol_count=1.0,
        passed=True,
    )

    cfg = CandidateStrategyConfig(
        l1_breakeven_floor_bps=0.0,
        l1_bootstrap_samples=10,
    )

    lcb = _compute_pooled_probe_lcb(
        fold_reports=(readiness,),
        cfg=cfg,
        seed=42,
    )

    # 동적 비용이 연동되었다면 LCB는 12.5 이하(대략 12.5 근처)여야 하며, 연동되지 않았다면 15.0 근처일 것임
    # 따라서 LCB가 14.0 미만임을 어서트하여 동적 비용 반영을 강제 검증함.
    assert lcb < 14.0
