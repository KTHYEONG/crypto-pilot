# tests/unit/domain/futures/strategy/test_signal_selection_incremental.py

from __future__ import annotations

from dataclasses import replace

import pytest

from src.domain.futures.strategy.candidate_contracts import (
    Layer1FoldReadiness,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    _compute_effective_sym_n,
    evaluate_layer1_readiness,
)


def test_scenario_1_reform_pass() -> None:
    fold_reports = (
        Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=10,
            outer_oos_start_idx=11,
            outer_oos_end_idx=20,
            ready_symbols=("BTC", "ETH", "SOL"),
            matched_event_count=50,
            unmatched_event_count=0,
            realized_match_ratio=1.0,
            unique_decision_count=10,
            prediction_unique_count=5,
            opportunity_ic=0.05,
            opportunity_ic_tstat=2.0,
            probe_bps=52.0,
            probe_lcb_bps=-5.0,
            probe_series_bps=(12.0, 8.0, 25.0, 40.0, 7.0),
            effective_symbol_count=3.0,
            passed=True,
            blockers=(),
        ),
        Layer1FoldReadiness(
            fold_id=1,
            registry_source_end_idx=20,
            outer_oos_start_idx=21,
            outer_oos_end_idx=30,
            ready_symbols=("BTC", "ETH", "SOL", "BNB"),
            matched_event_count=40,
            unmatched_event_count=0,
            realized_match_ratio=1.0,
            unique_decision_count=8,
            prediction_unique_count=4,
            opportunity_ic=0.03,
            opportunity_ic_tstat=1.5,
            probe_bps=6.73,
            probe_lcb_bps=-30.0,
            probe_series_bps=(3.0, 10.0, 5.0, 12.0),
            effective_symbol_count=4.0,
            passed=True,
            blockers=(),
        ),
        Layer1FoldReadiness(
            fold_id=2,
            registry_source_end_idx=30,
            outer_oos_start_idx=31,
            outer_oos_end_idx=40,
            ready_symbols=("BTC", "ETH", "SOL", "BNB"),
            matched_event_count=60,
            unmatched_event_count=0,
            realized_match_ratio=1.0,
            unique_decision_count=12,
            prediction_unique_count=6,
            opportunity_ic=0.08,
            opportunity_ic_tstat=2.5,
            probe_bps=203.8,
            probe_lcb_bps=80.0,
            probe_series_bps=(150.0, 200.0, 250.0),
            effective_symbol_count=4.0,
            passed=True,
            blockers=(),
        ),
        Layer1FoldReadiness(
            fold_id=3,
            registry_source_end_idx=40,
            outer_oos_start_idx=41,
            outer_oos_end_idx=50,
            ready_symbols=("BTC", "ETH", "SOL", "BNB", "XRP"),
            matched_event_count=45,
            unmatched_event_count=0,
            realized_match_ratio=1.0,
            unique_decision_count=9,
            prediction_unique_count=5,
            opportunity_ic=0.04,
            opportunity_ic_tstat=1.8,
            probe_bps=20.3,
            probe_lcb_bps=-10.0,
            probe_series_bps=(5.0, 15.0, 30.0, 40.0),
            effective_symbol_count=5.0,
            passed=True,
            blockers=(),
        ),
    )

    cfg = CandidateStrategyConfig()
    cfg = replace(
        cfg,
        l1_sym_count_mode="effective_n",
        l1_min_effective_sym_n=3.0,
        l1_min_fold_ratio=0.50,
        l1_min_realized_match_ratio=0.90,
        l1_probe_lcb_pooled=True,
        l1_min_probe_bps=0.0,
    )

    report = evaluate_layer1_readiness(
        fold_reports=fold_reports,
        fold_cov=1.0,
        trade_scope_count=57,
        cfg=cfg,
        seed=42,
    )

    assert report.passed is True
    assert len(report.blockers) == 0

    eff_n = _compute_effective_sym_n(fold_reports)
    assert eff_n == pytest.approx(4.413793, abs=1e-3)


def test_scenario_2_garbage_block() -> None:
    fold_reports = (
        Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=10,
            outer_oos_start_idx=11,
            outer_oos_end_idx=20,
            ready_symbols=("BTC", "ETH"),
            matched_event_count=50,
            unmatched_event_count=0,
            realized_match_ratio=1.0,
            unique_decision_count=10,
            prediction_unique_count=5,
            opportunity_ic=0.0,
            opportunity_ic_tstat=0.0,
            probe_bps=-10.0,
            probe_lcb_bps=-20.0,
            probe_series_bps=(-12.0, -8.0, -10.0),
            effective_symbol_count=2.0,
            passed=False,
            blockers=("non_positive_gross_edge",),
        ),
        Layer1FoldReadiness(
            fold_id=1,
            registry_source_end_idx=20,
            outer_oos_start_idx=21,
            outer_oos_end_idx=30,
            ready_symbols=("BTC", "ETH"),
            matched_event_count=40,
            unmatched_event_count=0,
            realized_match_ratio=1.0,
            unique_decision_count=8,
            prediction_unique_count=4,
            opportunity_ic=0.0,
            opportunity_ic_tstat=0.0,
            probe_bps=-5.0,
            probe_lcb_bps=-15.0,
            probe_series_bps=(-3.0, -10.0, -2.0),
            effective_symbol_count=2.0,
            passed=False,
            blockers=("non_positive_gross_edge",),
        ),
    )

    cfg = CandidateStrategyConfig()
    cfg = replace(
        cfg,
        l1_sym_count_mode="effective_n",
        l1_min_effective_sym_n=3.0,
        l1_min_fold_ratio=0.50,
        l1_probe_lcb_pooled=True,
        l1_min_probe_bps=0.0,
    )

    report = evaluate_layer1_readiness(
        fold_reports=fold_reports,
        fold_cov=1.0,
        trade_scope_count=57,
        cfg=cfg,
        seed=42,
    )

    assert report.passed is False
    assert any("fold_ratio" in b for b in report.blockers)
    assert any("probe_lcb_bps" in b for b in report.blockers)


def test_scenario_3_hhi_concentration() -> None:
    fold_reports = (
        Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=10,
            outer_oos_start_idx=11,
            outer_oos_end_idx=20,
            ready_symbols=("BTC",),
            matched_event_count=50,
            unmatched_event_count=0,
            realized_match_ratio=1.0,
            unique_decision_count=10,
            prediction_unique_count=5,
            opportunity_ic=0.05,
            opportunity_ic_tstat=2.0,
            probe_bps=52.0,
            probe_lcb_bps=10.0,
            probe_series_bps=(50.0, 52.0, 54.0),
            effective_symbol_count=1.0,
            passed=True,
            blockers=(),
        ),
        Layer1FoldReadiness(
            fold_id=1,
            registry_source_end_idx=20,
            outer_oos_start_idx=21,
            outer_oos_end_idx=30,
            ready_symbols=("BTC",),
            matched_event_count=40,
            unmatched_event_count=0,
            realized_match_ratio=1.0,
            unique_decision_count=8,
            prediction_unique_count=4,
            opportunity_ic=0.03,
            opportunity_ic_tstat=1.5,
            probe_bps=60.0,
            probe_lcb_bps=12.0,
            probe_series_bps=(58.0, 60.0, 62.0),
            effective_symbol_count=1.0,
            passed=True,
            blockers=(),
        ),
    )

    cfg = CandidateStrategyConfig()
    cfg = replace(
        cfg,
        l1_sym_count_mode="effective_n",
        l1_min_effective_sym_n=3.0,
        l1_min_fold_ratio=0.50,
        l1_probe_lcb_pooled=True,
        l1_min_probe_bps=0.0,
    )

    report = evaluate_layer1_readiness(
        fold_reports=fold_reports,
        fold_cov=1.0,
        trade_scope_count=57,
        cfg=cfg,
        seed=42,
    )

    assert report.passed is False
    assert any("sym_count" in b for b in report.blockers)


def test_scenario_4_backward_compat() -> None:
    fold_reports = (
        Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=10,
            outer_oos_start_idx=11,
            outer_oos_end_idx=20,
            ready_symbols=("BTC", "ETH"),
            matched_event_count=50,
            unmatched_event_count=0,
            realized_match_ratio=1.0,
            unique_decision_count=10,
            prediction_unique_count=5,
            opportunity_ic=0.05,
            opportunity_ic_tstat=2.0,
            probe_bps=52.0,
            probe_lcb_bps=10.0,
            probe_series_bps=(50.0, 52.0, 54.0),
            effective_symbol_count=2.0,
            passed=True,
            blockers=(),
        ),
    )

    cfg = CandidateStrategyConfig()
    cfg = replace(
        cfg,
        l1_sym_count_mode="count",
        l1_min_sym_count=6,
        l1_min_sym_ratio=0.30,
        l1_min_fold_ratio=0.50,
        l1_probe_lcb_pooled=False,
        l1_min_probe_bps=0.0,
    )

    report = evaluate_layer1_readiness(
        fold_reports=fold_reports,
        fold_cov=1.0,
        trade_scope_count=10,
        cfg=cfg,
        seed=42,
    )

    assert report.passed is False
    assert any("sym_count" in b for b in report.blockers)
