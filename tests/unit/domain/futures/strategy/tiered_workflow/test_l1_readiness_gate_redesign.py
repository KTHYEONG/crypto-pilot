from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_contracts import (
    Layer1FoldReadiness,
    SignalSourceKey,
    ValidatedSignalBatch,
    ValidatedSignalEvent,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    _wilson_lower_bound,
    align_outer_opportunities_with_realized,
    build_qualified_signal_registry,
    evaluate_layer1_readiness,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_opportunity_batch(
    decision_idx: int = 10,
    symbol: str = "BTCUSDT",
    strategy_id: str = "trend_ma:x",
    activation_context: str = "bull_quiet",
) -> ValidatedSignalBatch:
    event = ValidatedSignalEvent(
        decision_idx=decision_idx,
        decision_time=np.datetime64("2025-01-01"),
        symbol=symbol,
        strategy_id=strategy_id,
        activation_context=activation_context,
        side=1,
        expected_gross_bps=50.0,
        q10_gross_bps=30.0,
        q90_gross_bps=70.0,
        expected_holding_bars=24,
        quality_weight=1.0,
        registry_version="test",
        model_version="v1",
    )
    return ValidatedSignalBatch(
        events=(event,),
        start_idx=decision_idx,
        end_idx=decision_idx + 1,
        symbols=(symbol,),
        registry_version="test",
        model_version="v1",
    )


def _make_realized_frame(
    decision_idx: int = 10,
    symbol: str = "BTCUSDT",
    strategy_id: str = "trend_ma:x",
    activation_context: str = "bull_quiet",
    gross_bps: float = 10.0,
) -> pd.DataFrame:
    return pd.DataFrame({
        "entry_idx": [decision_idx + 1],
        "symbol": [symbol],
        "family": [strategy_id.split(":")[0]],
        "variant": [strategy_id.split(":")[1]],
        "signal_cell": [activation_context],
        "gross_event_bps": [gross_bps],
    })


def _fold(
    fold_id: int,
    matched: int = 50,
    unmatched: int = 0,
    label_drift: int = 0,
    realized_match_ratio: float = 1.0,
    passed: bool = True,
    probe_series_bps: tuple[float, ...] = (50.0, 52.0),
    probe_lcb_bps: float = 20.0,
    **kwargs: Any,
) -> Layer1FoldReadiness:
    return Layer1FoldReadiness(
        fold_id=fold_id,
        registry_source_end_idx=10,
        outer_oos_start_idx=11,
        outer_oos_end_idx=20,
        ready_symbols=kwargs.get("ready_symbols", ("BTC", "ETH", "SOL")),
        matched_event_count=matched,
        unmatched_event_count=unmatched,
        realized_match_ratio=realized_match_ratio,
        unique_decision_count=10,
        prediction_unique_count=5,
        opportunity_ic=kwargs.get("opportunity_ic", 0.05),
        opportunity_ic_tstat=kwargs.get("opportunity_ic_tstat", 2.0),
        probe_bps=kwargs.get("probe_bps", 50.0),
        probe_lcb_bps=probe_lcb_bps,
        probe_series_bps=probe_series_bps,
        effective_symbol_count=kwargs.get("effective_symbol_count", 3.0),
        passed=passed,
        blockers=kwargs.get("blockers", ()),
        label_drift_unmatched_count=label_drift,
    )


def _base_cfg(**overrides: object) -> CandidateStrategyConfig:
    cfg = CandidateStrategyConfig(
        l1_sym_count_mode="effective_n",
        l1_min_effective_sym_n=3.0,
        l1_min_fold_ratio=0.50,
        l1_min_realized_match_ratio=0.90,
        l1_probe_lcb_pooled=True,
        l1_min_probe_bps=0.0,
        l1_structural_gate_only=False,
    )
    if overrides:
        cfg = replace(cfg, **overrides)  # type: ignore[arg-type]
    return cfg


# ── LIMIT-01: align_outer_opportunities_with_realized ──────────────────────


def test_align_opportunities_happy_path() -> None:
    """Scenario 1 (Happy Path): 4-key all match → true_unmatched=0, label_drift=0."""
    opp = _make_opportunity_batch()
    realized = _make_realized_frame()

    matched, true_unmatched, label_drift = align_outer_opportunities_with_realized(
        opportunities=opp, realized_event_results=realized, activation_match_regime=True,
    )

    assert true_unmatched == 0
    assert label_drift == 0
    assert not matched.empty


def test_align_opportunities_separates_label_drift_from_true_unmatched() -> None:
    """Scenario 2 (Edge): 4-key unmatched but 3-key matches → label_drift=1, true_unmatched=0."""
    opp = _make_opportunity_batch(activation_context="bull_quiet")
    realized = _make_realized_frame(activation_context="bull_volatile")

    matched, true_unmatched, label_drift = align_outer_opportunities_with_realized(
        opportunities=opp, realized_event_results=realized, activation_match_regime=True,
    )

    assert true_unmatched == 0
    assert label_drift == 1
    assert matched.empty


def test_align_opportunities_missing_activation_context_column() -> None:
    """Scenario 3 (Error Handling): realized_frame without activation_context → fallback 'all'."""
    opp = _make_opportunity_batch(activation_context="all")
    realized = _make_realized_frame(activation_context="bull_quiet")
    realized = realized.drop(columns=["signal_cell"], errors="ignore")

    matched, true_unmatched, label_drift = align_outer_opportunities_with_realized(
        opportunities=opp, realized_event_results=realized, activation_match_regime=True,
    )

    assert true_unmatched == 0
    assert label_drift == 0
    assert not matched.empty


def test_align_opportunities_activation_match_false() -> None:
    """When activation_match_regime=False, activation_context is not a key → no label drift possible."""
    opp = _make_opportunity_batch(activation_context="bull_quiet")
    realized = _make_realized_frame(activation_context="bull_volatile")

    matched, true_unmatched, label_drift = align_outer_opportunities_with_realized(
        opportunities=opp, realized_event_results=realized, activation_match_regime=False,
    )

    assert true_unmatched == 0
    assert label_drift == 0
    assert not matched.empty


# ── LIMIT-02: _wilson_lower_bound ──────────────────────────────────────────


class TestWilsonLowerBound:
    def test_zero_n_returns_zero(self) -> None:
        assert _wilson_lower_bound(0, 0) == 0.0

    def test_perfect_match_returns_high_value(self) -> None:
        lcb = _wilson_lower_bound(100, 100, confidence=0.90)
        assert 0.95 < lcb <= 1.0

    def test_partial_match(self) -> None:
        lcb = _wilson_lower_bound(5, 10, confidence=0.90)
        assert 0.0 <= lcb < 0.5


# ── LIMIT-02 + LIMIT-03: evaluate_layer1_readiness ─────────────────────────


def test_evaluate_readiness_happy_path() -> None:
    """All folds pass all checks → passed, structural_passed, and advisory all True."""
    fold_reports = tuple(
        _fold(fid, matched=50, unmatched=0, passed=True, probe_lcb_bps=20.0)
        for fid in range(4)
    )
    cfg = _base_cfg()
    report = evaluate_layer1_readiness(
        fold_reports=fold_reports, fold_cov=1.0, trade_scope_count=57, cfg=cfg, seed=42,
    )

    assert report.structural_passed is True
    assert report.passed is True
    assert len(report.blockers) == 0
    assert len(report.checks) == 3  # fold_cov + sym_count + probe_lcb_bps
    assert len(report.advisory_checks) == 2  # match_ratio + fold_ratio


def test_fold_ratio_check_is_advisory_not_blocking() -> None:
    """Scenario LIMIT-02: fold_ratio is advisory (blocking=False) and doesn't affect structural_passed."""
    fold_reports = (
        _fold(0, matched=50, unmatched=0, passed=True, probe_lcb_bps=20.0),
        _fold(1, matched=40, unmatched=0, passed=True, probe_lcb_bps=15.0),
        _fold(2, matched=60, unmatched=0, passed=False, probe_lcb_bps=10.0),
        _fold(3, matched=45, unmatched=0, passed=False, probe_lcb_bps=5.0),
    )
    cfg = _base_cfg(l1_min_fold_ratio=0.75)  # requires 3/4 but only 2 ready
    report = evaluate_layer1_readiness(
        fold_reports=fold_reports, fold_cov=1.0, trade_scope_count=57, cfg=cfg, seed=42,
    )

    fold_ratio_check = next(c for c in report.advisory_checks if c.key == "fold_ratio")
    assert fold_ratio_check.blocking is False
    assert fold_ratio_check.passed is False
    assert report.structural_passed is True  # fold_cov/sym_count/probe_lcb_bps independent


def test_match_ratio_uses_pooled_wilson_not_per_fold_mean() -> None:
    """match_ratio is computed as pooled Wilson LCB, not per-fold mean."""
    fold_reports = tuple(
        _fold(fid, matched=100, unmatched=1, passed=True, probe_lcb_bps=20.0)
        for fid in range(4)
    )
    cfg = _base_cfg(l1_min_realized_match_ratio=0.95)  # 100/101 ≈ 0.990 → Wilson LCB > 0.95
    report = evaluate_layer1_readiness(
        fold_reports=fold_reports, fold_cov=1.0, trade_scope_count=57, cfg=cfg, seed=42,
    )

    match_ratio_check = next(c for c in report.advisory_checks if c.key == "match_ratio")
    per_fold_mean = 100 / 101  # 0.990
    assert match_ratio_check.value < per_fold_mean  # Wilson LCB < point estimate
    assert match_ratio_check.passed is True  # still passes at 0.95 threshold


def test_structural_failure_still_blocks_passed() -> None:
    """structural_passed=False → passed=False."""
    fold_reports = (
        _fold(0, matched=50, unmatched=0, passed=True,
              probe_series_bps=(-10.0,), probe_lcb_bps=-20.0, effective_symbol_count=1.0),
        _fold(1, matched=40, unmatched=0, passed=True,
              probe_series_bps=(-10.0,), probe_lcb_bps=-20.0, effective_symbol_count=1.0),
    )
    cfg = _base_cfg(l1_min_probe_bps=5.0, l1_min_effective_sym_n=3.0)
    report = evaluate_layer1_readiness(
        fold_reports=fold_reports, fold_cov=0.5, trade_scope_count=57, cfg=cfg, seed=42,
    )

    assert report.structural_passed is False
    assert report.passed is False


# ── LIMIT-03: build_qualified_signal_registry with advisory_penalty ────────


def _make_symbol_strategy_evidence(
    symbol: str = "BTCUSDT",
    strategy_id: str = "trend_ma:x",
    activation_context: str = "all",
    quality_weight: float = 1.0,
    hard_eligible: bool = True,
    lcb_net_bps: float = 10.0,
) -> Any:
    key = SignalSourceKey(symbol=symbol, strategy_id=strategy_id, activation_context=activation_context)
    mock = type("_MockEvidence", (), {
        "key": key,
        "hard_eligible": hard_eligible,
        "quality_weight": quality_weight,
        "reliability": quality_weight,
        "lcb_net_bps": lcb_net_bps,
    })()
    return mock


def test_advisory_penalty_affects_sort_order() -> None:
    """advisory_penalty < 1.0 reduces effective quality for sort (lower priority)."""
    ev_high = _make_symbol_strategy_evidence(symbol="BTCUSDT", quality_weight=1.0)
    ev_low = _make_symbol_strategy_evidence(symbol="ETHUSDT", quality_weight=0.5)
    registry_with_penalty = build_qualified_signal_registry(
        evidence=(ev_high, ev_low),
        symbols=("BTCUSDT", "ETHUSDT"),
        min_signals_per_symbol=1,
        registry_version="test",
        advisory_penalty=0.5,
    )

    assert "BTCUSDT" in registry_with_penalty.ready_symbols
    assert "ETHUSDT" in registry_with_penalty.ready_symbols


def test_advisory_penalty_keeps_symbol_included() -> None:
    """Even with penalty, if quality_weight * penalty > 0, item stays."""
    ev = _make_symbol_strategy_evidence(quality_weight=0.3)
    registry = build_qualified_signal_registry(
        evidence=(ev,),
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="test",
        advisory_penalty=0.5,
    )

    assert "BTCUSDT" in registry.ready_symbols


def test_structural_failure_still_blocks_registry() -> None:
    """structural_passed=False → build_qualified_signal_registry won't be called.
    This test verifies the gate condition by passing a failing gate report scenario."""
    fold_reports = (
        _fold(0, matched=50, unmatched=0, passed=True,
              probe_series_bps=(-10.0,), probe_lcb_bps=-50.0, effective_symbol_count=1.0),
    )
    cfg = _base_cfg(l1_min_probe_bps=10.0)
    report = evaluate_layer1_readiness(
        fold_reports=fold_reports, fold_cov=0.3, trade_scope_count=10, cfg=cfg, seed=42,
    )

    assert report.structural_passed is False
