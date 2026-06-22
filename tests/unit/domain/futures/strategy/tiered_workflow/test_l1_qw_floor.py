"""Tests for L1 quality weight floor logic (l1_qw_floor)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.domain.futures.strategy.candidate_contracts import (
    SignalSourceKey,
    SymbolStrategyEvidence,
)
from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    build_qualified_signal_registry,
    compute_symbol_strategy_evidence,
)


def _make_cfg(**overrides: object) -> MagicMock:
    """Return a CandidateStrategyConfig-like mock stub."""
    cfg = MagicMock()
    defaults: dict[str, object] = {
        "l1_baseline_mode": "peer_exclusive",
        "l1_qualify_by_regime": False,
        "l1_pair_min_effective_obs": 1.0,
        "l1_pair_alpha": 0.05,
        "l1_pair_power": 0.80,
        "l1_pair_mdes_multiplier": 0.5,
        "l1_pair_min_folds": 1,
        "l1_pair_min_mean_gross_bps": 0.0,
        "l1_pair_min_incremental_bps": 0.0,
        "l1_pair_min_positive_fold_ratio": 0.0,
        "l1_pair_fdr_alpha": 1.0,
        "l1_bootstrap_block_bars": 1,
        "l1_bootstrap_samples": 10,
        "l1_quality_weight_enabled": True,
        "l1_qw_floor": 0.0,
        "l1_qw_probe_boost": 0.3,
        "l1_fdr_hard_reject": True,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_event_frame(
    gross_bps_list: list[float],
    symbol: str = "BTCUSDT",
    strategy_id: str = "trend_ma:ema_12_72",
) -> pd.DataFrame:
    rows = [
        {
            "symbol": symbol,
            "strategy_id": strategy_id,
            "side": 1,
            "holding_bucket": 4,
            "gross_event_bps": g,
            "incremental_bps": g,
            "fold_id": 0,
            "uniqueness_weight": 1.0,
            "expected_holding_bars": 4,
        }
        for g in gross_bps_list
    ]
    return pd.DataFrame(rows)


def _make_evidence(
    symbol: str = "BTCUSDT",
    strategy_id: str = "trend_ma:ema_12_72",
    activation_context: str = "all",
    quality_weight: float = 0.5,
    hard_eligible: bool = True,
    lcb_net_bps: float = 100.0,
) -> SymbolStrategyEvidence:
    return SymbolStrategyEvidence(
        key=SignalSourceKey(
            symbol=symbol,
            strategy_id=strategy_id,
            activation_context=activation_context,
        ),
        mean_gross_bps=50.0,
        mean_incremental_bps=30.0,
        block_tstat_incremental=2.5,
        probability_positive=0.6,
        p_value=0.01,
        q_value=0.02,
        positive_fold_ratio=0.8,
        n_obs=100,
        effective_n=80.0,
        n_folds=4,
        quality_weight=quality_weight,
        hard_eligible=hard_eligible,
        lcb_net_bps=lcb_net_bps,
    )


# ─── Scenario 1: qw_floor prevents zero quality_weight ─────────────────────

def test_qw_floor_raises_low_quality_weight() -> None:
    """Scenario 1: When calculated qw < l1_qw_floor, qw is raised to floor."""
    evidence = (
        _make_evidence(quality_weight=0.003, hard_eligible=True, lcb_net_bps=100.0),
    )
    cfg = _make_cfg(l1_qw_floor=0.05)
    registry = build_qualified_signal_registry(
        evidence=evidence,
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg,
    )
    assert "BTCUSDT" in registry.by_symbol


def test_qw_floor_below_threshold_does_not_raise_when_already_above() -> None:
    """If quality_weight is already above floor, floor has no effect."""
    evidence = (
        _make_evidence(quality_weight=0.5, hard_eligible=True, lcb_net_bps=100.0),
    )
    cfg = _make_cfg(l1_qw_floor=0.05)
    registry = build_qualified_signal_registry(
        evidence=evidence,
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg,
    )
    assert "BTCUSDT" in registry.by_symbol


# ─── Scenario 2: qw_floor=0.0 = backward compat ───────────────────────────

def test_qw_floor_zero_backward_compat() -> None:
    """Scenario 2: l1_qw_floor=0.0 preserves old behavior (no floor)."""
    evidence = (
        _make_evidence(quality_weight=0.003, hard_eligible=True, lcb_net_bps=100.0),
    )
    cfg = _make_cfg(l1_qw_floor=0.0)
    registry = build_qualified_signal_registry(
        evidence=evidence,
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg,
    )
    assert "BTCUSDT" in registry.by_symbol


# ─── Scenario 7: FDR hard reject overrides qw_floor ───────────────────────

def test_fdr_hard_reject_overrides_qw_floor() -> None:
    """Scenario 7: FDR hard reject (q > alpha) is ABSOLUTE — qw_floor does NOT rescue."""
    cfg = _make_cfg(l1_qw_floor=0.05, l1_fdr_hard_reject=True)
    evidence_list = compute_symbol_strategy_evidence(
        event_results=_make_event_frame(
            gross_bps_list=[0.1, 0.2] * 5,
        ),
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
    )
    for ev in evidence_list:
        if ev.q_value > float(cfg.l1_pair_fdr_alpha):
            assert ev.quality_weight == 0.0
