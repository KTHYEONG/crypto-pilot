from __future__ import annotations

import types
from dataclasses import replace

import pytest

from src.domain.futures.strategy.candidate_contracts import Layer1FoldReadiness
from src.domain.futures.strategy.config import CandidateStrategyConfig, apply_tf_gate_overrides
from src.domain.futures.strategy.tiered_workflow.metrics import _bars_per_year_for_tf
from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    _resolve_block_bars_eff,
    evaluate_layer1_readiness,
)

# ── Fix 1: _resolve_block_bars_eff ──────────────────────────────────────────


def test_resolve_block_bars_eff_scales_with_holding_period() -> None:
    cfg = CandidateStrategyConfig(l1_bootstrap_block_bars=6, max_holding_bars=20)
    block_bars_eff = _resolve_block_bars_eff(cfg)
    assert block_bars_eff == 40


def test_resolve_block_bars_eff_floors_at_base_when_holding_short() -> None:
    cfg = CandidateStrategyConfig(l1_bootstrap_block_bars=6, max_holding_bars=1)
    block_bars_eff = _resolve_block_bars_eff(cfg)
    assert block_bars_eff == 6


def test_resolve_block_bars_eff_fallback_on_missing_holding_bars() -> None:
    mock_cfg = types.SimpleNamespace(l1_bootstrap_block_bars=6)
    block_bars_eff = _resolve_block_bars_eff(mock_cfg)  # type: ignore[arg-type]
    assert block_bars_eff == 6, "fallback to default max_holding_bars=1"


# ── Fix 3: probe_lcb_bps breakeven gate ─────────────────────────────────────


def test_probe_lcb_breakeven_blocks_when_below_breakeven() -> None:
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
            probe_lcb_bps=5.0,
            probe_series_bps=(12.0, 8.0, 25.0, 40.0, 7.0),
            effective_symbol_count=3.0,
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
        l1_probe_lcb_pooled=False,
        l1_min_probe_bps=0.0,
        l1_breakeven_floor_bps=7.5,
    )

    report = evaluate_layer1_readiness(
        fold_reports=fold_reports,
        fold_cov=1.0,
        trade_scope_count=57,
        cfg=cfg,
        seed=42,
    )

    assert report.passed is False
    assert any("probe_lcb_bps" in b for b in report.blockers)


def test_probe_lcb_breakeven_backward_compat_custom_threshold() -> None:
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
            probe_lcb_bps=12.0,
            probe_series_bps=(12.0, 8.0, 25.0, 40.0, 7.0),
            effective_symbol_count=3.0,
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
        l1_probe_lcb_pooled=False,
        l1_min_probe_bps=10.0,
        l1_breakeven_floor_bps=7.5,
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


# ── Fix 2: per-TF effective_sym_n override ──────────────────────────────────


def test_apply_tf_gate_overrides_effective_sym_n_2h() -> None:
    cfg = CandidateStrategyConfig()
    overridden = apply_tf_gate_overrides(cfg, "2h")
    assert overridden.l1_min_effective_sym_n == 5.0


def test_apply_tf_gate_overrides_effective_sym_n_6h_fallback() -> None:
    cfg = CandidateStrategyConfig()
    overridden = apply_tf_gate_overrides(cfg, "6h")
    assert overridden.l1_min_effective_sym_n == 3.0


def test_layer1_fold_readiness_diagnostic_fields_defaults() -> None:
    report = Layer1FoldReadiness(
        fold_id=0,
        registry_source_end_idx=10,
        outer_oos_start_idx=0,
        outer_oos_end_idx=100,
        ready_symbols=("BTC", "ETH"),
    )
    assert report.bars_per_fold_native == 0
    assert report.decision_points_per_calendar_year == 0.0


def test_layer1_fold_readiness_diagnostic_fields_populated() -> None:
    report = Layer1FoldReadiness(
        fold_id=0,
        registry_source_end_idx=10,
        outer_oos_start_idx=0,
        outer_oos_end_idx=100,
        ready_symbols=("BTC", "ETH"),
        bars_per_fold_native=100,
        decision_points_per_calendar_year=50.0,
    )
    assert report.bars_per_fold_native == 100
    assert report.decision_points_per_calendar_year == 50.0


class TestBarsPerYearForTf:
    """_bars_per_year_for_tf는 tiered_workflow.metrics의 SSOT를 재사용(중복 정의 금지)."""

    def test_returns_8760_for_1h(self) -> None:
        assert _bars_per_year_for_tf("1h") == pytest.approx(8760.0, rel=1e-9)

    def test_returns_4380_for_2h(self) -> None:
        assert _bars_per_year_for_tf("2h") == pytest.approx(4380.0, rel=1e-9)

    def test_returns_2190_for_4h(self) -> None:
        assert _bars_per_year_for_tf("4h") == pytest.approx(2190.0, rel=1e-9)

    def test_ratio_1h_to_4h_is_4(self) -> None:
        r1 = _bars_per_year_for_tf("1h")
        r4 = _bars_per_year_for_tf("4h")
        assert r1 / r4 == pytest.approx(4.0, rel=1e-9)

    def test_ratio_1h_to_2h_is_2(self) -> None:
        r1 = _bars_per_year_for_tf("1h")
        r2 = _bars_per_year_for_tf("2h")
        assert r1 / r2 == pytest.approx(2.0, rel=1e-9)

    def test_ratio_2h_to_4h_is_2(self) -> None:
        r2 = _bars_per_year_for_tf("2h")
        r4 = _bars_per_year_for_tf("4h")
        assert r2 / r4 == pytest.approx(2.0, rel=1e-9)
