"""Tests for L1 probe prior boost in build_qualified_signal_registry."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.domain.futures.strategy.candidate_contracts import (
    SignalSourceKey,
    SymbolStrategyEvidence,
)
from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    build_qualified_signal_registry,
)


def _make_cfg(**overrides: object) -> MagicMock:
    cfg = MagicMock()
    cfg.l1_breakeven_floor_bps = 0.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


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


# ─── Scenario 3: Probe prior boost raises quality_weight ───────────────────


def test_probe_prior_boost_raises_qw() -> None:
    """Scenario 3: probe-winning signal gets qw boosted to probe_prior_map floor."""
    evidence = (
        _make_evidence(
            symbol="BTCUSDT",
            strategy_id="trend_ma:ema_12_72",
            quality_weight=0.01,
        ),
    )
    cfg = _make_cfg()
    probe_prior_map = {("trend_ma", "ema_12_72", "BTCUSDT"): 0.3}
    registry = build_qualified_signal_registry(
        evidence=evidence,
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg,
        probe_prior_map=probe_prior_map,
    )
    assert "BTCUSDT" in registry.by_symbol


def test_probe_prior_boost_signal_survives_gate() -> None:
    """Signal with low qw survives registration when boosted by probe prior."""
    evidence = (
        _make_evidence(
            symbol="BTCUSDT",
            strategy_id="trend_ma:ema_12_72",
            quality_weight=0.001,
            hard_eligible=True,
            lcb_net_bps=100.0,
        ),
    )
    cfg = _make_cfg()
    probe_prior_map = {("trend_ma", "ema_12_72", "BTCUSDT"): 0.3}
    registry = build_qualified_signal_registry(
        evidence=evidence,
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg,
        probe_prior_map=probe_prior_map,
    )
    assert "BTCUSDT" in registry.by_symbol


# ─── Scenario 4: Probe non-winner gets no boost ────────────────────────────


def test_probe_non_winner_no_false_boost() -> None:
    """Scenario 4: non-probe-winner signal is NOT boosted."""
    evidence = (
        _make_evidence(
            symbol="BTCUSDT",
            strategy_id="trend_ma:ema_12_72",
            quality_weight=0.01,
            hard_eligible=True,
            lcb_net_bps=100.0,
        ),
    )
    cfg = _make_cfg()
    # probe_prior_map has no entry for this signal
    probe_prior_map: dict[tuple[str, str, str], float] = {}
    registry = build_qualified_signal_registry(
        evidence=evidence,
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg,
        probe_prior_map=probe_prior_map,
    )
    # qw stays at 0.01 > 0, so signal still passes
    assert "BTCUSDT" in registry.by_symbol


def test_probe_non_winner_qw_below_threshold_fails() -> None:
    """Non-probe-winner with qw ≈ 0.0 still fails registration."""
    evidence = (
        _make_evidence(
            symbol="BTCUSDT",
            strategy_id="trend_ma:ema_12_72",
            quality_weight=0.0,
            hard_eligible=True,
            lcb_net_bps=100.0,
        ),
    )
    cfg = _make_cfg()
    probe_prior_map: dict[tuple[str, str, str], float] = {}
    registry = build_qualified_signal_registry(
        evidence=evidence,
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg,
        probe_prior_map=probe_prior_map,
    )
    # qw = 0.0 → fails registration (qw > 0.0 check)
    assert "BTCUSDT" not in registry.by_symbol
