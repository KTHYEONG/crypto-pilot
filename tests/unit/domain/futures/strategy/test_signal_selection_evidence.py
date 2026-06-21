"""Tests for adaptive t-statistic thresholding, MDES filter, and
activation floor boundary checks in signal selection.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import (
    CandidateModelOutput,
    EdgeSource,
    QualifiedSignalRegistry,
    SignalSourceKey,
    SymbolStrategyEvidence,
)
from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    _candidate_output_to_signal_batch,
    build_qualified_signal_registry,
    compute_symbol_strategy_evidence,
)


def _make_cfg(**overrides: object) -> MagicMock:
    """Return a CandidateStrategyConfig-like mock stub."""
    cfg = MagicMock()
    defaults: dict[str, object] = {
        "l1_baseline_mode": "peer_exclusive",
        "l1_qualify_by_regime": False,
        "l1_pair_min_effective_obs": 5.0,
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
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_event_frame(
    gross_bps_list: list[float],
    symbol: str = "BTCUSDT",
    strategy_id: str = "trend:fast",
) -> pd.DataFrame:
    """Helper to construct an event DataFrame for evidence computation."""
    rows = [
        {
            "symbol": symbol,
            "strategy_id": strategy_id,
            "side": 1,
            "holding_bucket": 4,
            "gross_event_bps": g,
            "incremental_bps": g,  # Will be recalculated but provides seed values
            "fold_id": 0,
            "uniqueness_weight": 1.0,
            "expected_holding_bars": 4,
        }
        for g in gross_bps_list
    ]
    df = pd.DataFrame(rows)
    return df


# ─── Scenario 1: Insufficient Effective Obs ───────────────────────────────

def test_insufficient_effective_obs_rejection() -> None:
    """If effective_n < l1_pair_min_effective_obs, reject with insufficient_effective_obs."""
    # Given: effective_obs = 3, but threshold is 5.0
    cfg = _make_cfg(l1_pair_min_effective_obs=5.0)
    df = _make_event_frame(gross_bps_list=[5.0, 4.0, 6.0])

    # When
    evidence = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
    )

    # Then
    assert len(evidence) == 1
    assert not evidence[0].qualified
    assert "insufficient_effective_obs" in evidence[0].rejection_reasons


# ─── Scenario 2: Adaptive t-statistic and MDES Filtering ──────────────────

def test_adaptive_tstat_and_mdes_filtering() -> None:
    """Verify adaptive t-threshold and MDES filtering."""
    # N=10, df=9. For alpha=0.05, t_crit is approx 1.833.
    # Mean = 0.15, Std = 0.0527. t_stat = 0.15 / (0.0527 / sqrt(10)) = 9.0 (Far above 1.833).
    gross_values = [0.1, 0.2] * 5
    df = _make_event_frame(gross_bps_list=gross_values)

    # Case A: MDES multiplier is small (0.1), should pass since mean (0.15) > mdes_bps * multiplier
    cfg_pass = _make_cfg(
        l1_pair_min_effective_obs=5.0,
        l1_pair_alpha=0.05,
        l1_pair_power=0.80,
        l1_pair_mdes_multiplier=0.1,
    )
    evidence_pass = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg_pass,
        seed=0,
        registry_as_of_idx=999,
    )
    assert len(evidence_pass) == 1
    assert "insufficient_effect_size" not in evidence_pass[0].rejection_reasons
    assert "weak_tstat" not in evidence_pass[0].rejection_reasons

    # Case B: MDES multiplier is large (10.0), should fail with insufficient_effect_size
    cfg_fail = _make_cfg(
        l1_pair_min_effective_obs=5.0,
        l1_pair_alpha=0.05,
        l1_pair_power=0.80,
        l1_pair_mdes_multiplier=10.0,
    )
    evidence_fail = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg_fail,
        seed=0,
        registry_as_of_idx=999,
    )
    assert len(evidence_fail) == 1
    assert "insufficient_effect_size" in evidence_fail[0].rejection_reasons
    assert evidence_fail[0].qualified


# ─── Scenario 3 & 4: Activation Floor Boundary Condition and Zero-Prediction Fallback ─

def test_candidate_output_to_signal_batch_zero_bps_retained() -> None:
    """Verify that pred == 0.0 is not discarded when activation_floor_bps == 0.0 (Strict comparison <)."""
    # Arrange
    events = pd.DataFrame([
        {
            "entry_idx": 1,
            "symbol": "BTCUSDT",
            "family": "trend",
            "variant": "fast",
            "entry_regime": "bull",
            "side": 1,
            "expected_holding_bars": 3,
        }
    ])
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.asarray([1.0], dtype=np.float64),
        edge_source=EdgeSource.PRIOR_ONLY,
        expected_gross_bps=np.asarray([0.0], dtype=np.float64), # 0.0 prediction
        q10_gross_bps=np.asarray([-2.0], dtype=np.float64),
        q90_gross_bps=np.asarray([2.0], dtype=np.float64),
    )
    evidence = SymbolStrategyEvidence(
        key=SignalSourceKey("BTCUSDT", "trend:fast", "bull"),
        mean_gross_bps=5.0,
        mean_incremental_bps=2.0,
        bootstrap_tstat_incremental=2.1,
        p_value=0.02,
        q_value=0.03,
        positive_fold_ratio=1.0,
        n_obs=12,
        effective_n=12.0,
        n_folds=3,
        reliability=0.8,
        qualified=True,
        rejection_reasons=(),
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (evidence,)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="deployment",
    )
    datetimes = np.array(
        [
            np.datetime64("2026-06-13T00:00:00"),
            np.datetime64("2026-06-13T04:00:00"),
            np.datetime64("2026-06-13T08:00:00"),
        ],
        dtype="datetime64[ns]",
    )

    # Act: activation_floor_bps = 0.0. Under BEFORE(<=), pred=0.0 is dropped. Under AFTER(<), it must be retained.
    batch = _candidate_output_to_signal_batch(
        model_output=model_output,
        registry=registry,
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        model_version="m1",
        activation_floor_bps=0.0,
    )

    # Assert
    assert len(batch.events) == 1
    assert batch.events[0].symbol == "BTCUSDT"


def test_build_qualified_signal_registry_prefers_quality_weight_over_input_order() -> None:
    strong = SimpleNamespace(
        key=SignalSourceKey("BTCUSDT", "trend:fast", "bull"),
        mean_gross_bps=4.0,
        mean_incremental_bps=2.0,
        bootstrap_tstat_incremental=2.5,
        p_value=0.01,
        q_value=0.02,
        positive_fold_ratio=1.0,
        n_obs=10,
        effective_n=10.0,
        n_folds=3,
        reliability=0.7,
        quality_weight=0.95,
        qualified=True,
        rejection_reasons=(),
    )
    weak = SimpleNamespace(
        key=SignalSourceKey("BTCUSDT", "trend:slow", "bull"),
        mean_gross_bps=4.0,
        mean_incremental_bps=2.0,
        bootstrap_tstat_incremental=2.5,
        p_value=0.01,
        q_value=0.02,
        positive_fold_ratio=1.0,
        n_obs=10,
        effective_n=10.0,
        n_folds=3,
        reliability=0.7,
        quality_weight=0.45,
        qualified=True,
        rejection_reasons=(),
    )

    builder: Any = build_qualified_signal_registry
    registry: Any = builder(
        evidence=(weak, strong),
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="v1",
    )

    assert registry.by_symbol["BTCUSDT"][0].quality_weight == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# P3: Adaptive Evidence Gate (snapshot_index)
# ---------------------------------------------------------------------------

def _make_event_frame_3events() -> pd.DataFrame:
    """Return a 3-row event DataFrame with 1 fold, effective_n ≈ 3.0.

    Uses the same columns as the existing `_make_event_frame` helper (no exit_idx
    so maturity filter is skipped). entry_idx is included for bootstrap grouping.
    """
    return pd.DataFrame({
        "symbol": ["BTCUSDT"] * 3,
        "family": ["trend_ma"] * 3,
        "variant": ["ema_12_72"] * 3,
        "strategy_id": ["trend_ma:ema_12_72"] * 3,
        "activation_context": ["all"] * 3,
        "fold_id": [0] * 3,
        "gross_event_bps": [5.0, 6.0, 4.0],
        "side": [1] * 3,
        "expected_holding_bars": [24] * 3,
        "uniqueness_weight": [1.0] * 3,
        "entry_idx": [0, 1, 2],
    })


def test_adaptive_gate_early_snapshot_relaxed() -> None:
    """P3.1: Early snapshot (index=1 < early_snapshots=2) uses relaxed thresholds."""
    cfg = _make_cfg(
        l1_pair_min_effective_obs=5.0,       # strict
        l1_pair_min_folds=2,                  # strict
        l1_evidence_early_snapshots=2,
        l1_pair_min_effective_obs_early=2.0,  # relaxed
        l1_pair_min_folds_early=1,            # relaxed
        l1_pair_min_mean_gross_bps=0.0,
        l1_pair_min_incremental_bps=0.0,
        l1_quality_weight_enabled=False,      # simplify: quality_weight controlled by hard_eligible only
    )
    df = _make_event_frame_3events()

    evidence = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
        snapshot_index=1,
    )
    # With relaxed gates (eff_obs>=2.0, folds>=1), all 3 events pass
    assert len(evidence) == 1
    assert evidence[0].hard_eligible, "early snapshot with relaxed gates should be hard_eligible"


def test_adaptive_gate_late_snapshot_strict() -> None:
    """P3.2: Late snapshot (index=3 >= early_snapshots=2) uses strict thresholds."""
    cfg = _make_cfg(
        l1_pair_min_effective_obs=5.0,       # strict — 3 < 5
        l1_pair_min_folds=2,                  # strict — 1 < 2
        l1_evidence_early_snapshots=2,
        l1_pair_min_effective_obs_early=2.0,
        l1_pair_min_folds_early=1,
    )
    df = _make_event_frame_3events()

    evidence = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
        snapshot_index=3,
    )
    assert len(evidence) == 1
    assert not evidence[0].hard_eligible
    # Should fail on both insufficient_effective_obs (3<5) AND insufficient_folds (1<2)
    reasons = set(evidence[0].structural_reasons)
    assert "insufficient_effective_obs" in reasons
    assert "insufficient_folds" in reasons


def test_adaptive_gate_disabled_always_strict() -> None:
    """P3.3: l1_evidence_early_snapshots=0 → strict gates regardless of snapshot_index."""
    cfg = _make_cfg(
        l1_pair_min_effective_obs=5.0,
        l1_pair_min_folds=2,
        l1_evidence_early_snapshots=0,       # disabled
        l1_pair_min_effective_obs_early=2.0,  # would be relaxed but never used
        l1_pair_min_folds_early=1,
    )
    df = _make_event_frame_3events()

    # Even with snapshot_index=0, fails because early_snapshots=0 means never relaxed
    evidence_early = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
        snapshot_index=0,
    )
    assert len(evidence_early) == 1
    assert not evidence_early[0].hard_eligible

    # snapshot_index=1 also fails
    evidence_late = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
        snapshot_index=1,
    )
    assert not evidence_late[0].hard_eligible
