"""Tests for peer-exclusive incremental_bps in compute_symbol_strategy_evidence."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    _compute_incremental_bps,
    compute_symbol_strategy_evidence,
)


def _make_cfg(**overrides: object) -> MagicMock:
    """Return a minimal CandidateStrategyConfig-like stub."""
    cfg = MagicMock()
    defaults: dict[str, object] = {
        "l1_baseline_mode": "peer_exclusive",
        "l1_qualify_by_regime": False,
        "l1_pair_min_effective_obs": 1.0,
        "l1_pair_min_folds": 1,
        "l1_pair_min_mean_gross_bps": 0.0,
        "l1_pair_min_incremental_bps": 0.0,
        "l1_pair_min_incremental_tstat": 1.96,
        "l1_pair_min_positive_fold_ratio": 0.0,
        "l1_pair_fdr_alpha": 1.0,
        "l1_bootstrap_block_bars": 1,
        "l1_bootstrap_samples": 10,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a minimal event DataFrame."""
    df = pd.DataFrame(rows)
    if "holding_bucket" not in df.columns:
        df["holding_bucket"] = 4
    return df


# ─── S1: peer-exclusive — A(gross=20) vs B(gross=2) in same bucket ──────────

def test_compute_incremental_bps_peer_exclusive_two_strategies() -> None:
    """A: peer mean=2 → incremental≈+18; B: peer mean=20 → incremental≈-18."""
    # Arrange
    n = 50
    rows_a = [
        {"symbol": "BTC", "side": 1, "holding_bucket": 4, "strategy_id": "A", "gross_event_bps": 20.0}
    ] * n
    rows_b = [
        {"symbol": "BTC", "side": 1, "holding_bucket": 4, "strategy_id": "B", "gross_event_bps": 2.0}
    ] * n
    frame = _make_frame(rows_a + rows_b)

    # Act
    result = _compute_incremental_bps(frame, mode="peer_exclusive")

    # Assert — A: peer mean=2.0, incremental≈+18; B: peer mean=20.0, incremental≈-18
    inc_a = result.iloc[:n]
    inc_b = result.iloc[n:]
    assert inc_a.mean() == pytest.approx(18.0, abs=1e-6)
    assert inc_b.mean() == pytest.approx(-18.0, abs=1e-6)


# ─── S2: single strategy in bucket → absolute fallback (incremental == gross) ─

def test_compute_incremental_bps_single_strategy_absolute_fallback() -> None:
    """No peers in bucket → baseline=0 → incremental == gross (no zero-collapse)."""
    # Arrange
    frame = _make_frame([
        {"symbol": "ETH", "side": 1, "holding_bucket": 4, "strategy_id": "A", "gross_event_bps": 15.0}
    ] * 30)

    # Act
    result = _compute_incremental_bps(frame, mode="peer_exclusive")

    # Assert — absolute fallback: incremental == gross
    assert result.mean() == pytest.approx(15.0, abs=1e-6)


# ─── S3: zero-division safety ─────────────────────────────────────────────────

def test_compute_incremental_bps_no_nan_or_inf_when_single_strategy() -> None:
    """Single strategy bucket must produce finite incremental values, no NaN/inf."""
    # Arrange
    frame = _make_frame([
        {"symbol": "SOL", "side": -1, "holding_bucket": 8, "strategy_id": "X", "gross_event_bps": 5.0}
    ] * 20)

    # Act
    result = _compute_incremental_bps(frame, mode="peer_exclusive")

    # Assert — all finite, no division-by-zero artifacts
    assert result.notna().all()
    assert np.isfinite(result.to_numpy()).all()


# ─── S4: absolute mode — incremental == gross ─────────────────────────────────

def test_compute_incremental_bps_absolute_mode_equals_gross() -> None:
    """mode='absolute' must return gross unchanged regardless of peer count."""
    # Arrange
    rows = [
        {"symbol": "BTC", "side": 1, "holding_bucket": 4, "strategy_id": "A", "gross_event_bps": 12.0},
        {"symbol": "BTC", "side": 1, "holding_bucket": 4, "strategy_id": "B", "gross_event_bps": 7.0},
    ] * 25
    frame = _make_frame(rows)

    # Act
    result = _compute_incremental_bps(frame, mode="absolute")

    # Assert — no peer adjustment applied
    expected = frame["gross_event_bps"].to_numpy(dtype=np.float64)
    np.testing.assert_array_almost_equal(result.to_numpy(), expected)


# ─── S4b: regime exclusion invariance ───────────────────────────────────────

def test_compute_symbol_strategy_evidence_regime_exclusion_pooled() -> None:
    """l1_qualify_by_regime=False → single 'all' evidence row, no regime split."""
    # Arrange — same symbol+strategy, different entry_regime_code values
    common: dict[str, object] = {
        "symbol": "BTC",
        "strategy_id": "A",
        "side": 1,
        "gross_event_bps": 10.0,
        "fold_id": 0,
        "expected_holding_bars": 4,
    }
    rows = [{**common, "entry_regime_code": i} for i in range(3)] * 30
    frame = pd.DataFrame(rows)
    cfg = _make_cfg(l1_qualify_by_regime=False)

    # Act
    evidence = compute_symbol_strategy_evidence(
        event_results=frame,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
    )

    # Assert — single row with activation_context="all", not 3 separate regime cells
    assert len(evidence) == 1
    assert evidence[0].key.activation_context == "all"


# ─── S5: fold_id propagation ──────────────────────────────────────────────────

def test_evaluate_outer_signal_opportunities_fold_id_propagated() -> None:
    """fold_id=2 must appear in Layer1FoldReadiness, not hardcoded 0."""
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        evaluate_outer_signal_opportunities,
    )

    # Arrange — empty batch forces early-return path
    batch = ValidatedSignalBatch(
        events=(),
        start_idx=0,
        end_idx=0,
        symbols=(),
        registry_version="v0",
        model_version="m0",
    )
    fold = MagicMock()
    fold.fit_end = 100
    fold.oos_start = 100
    fold.oos_end = 200
    cfg = _make_cfg()
    vols = np.zeros((200, 1), dtype=np.float64)

    # Act
    result = evaluate_outer_signal_opportunities(
        opportunities=batch,
        realized_event_results=pd.DataFrame(),
        volatility_2d=vols,
        aligned_symbols=(),
        fold=fold,
        fold_id=2,
        cfg=cfg,
        seed=0,
    )

    # Assert — fold_id must not be hardcoded 0
    assert result.fold_id == 2


def test_evaluate_outer_signal_opportunities_preserves_side_adjusted_gross_sign() -> None:
    from src.domain.futures.strategy.candidate_contracts import (
        ValidatedSignalBatch,
        ValidatedSignalEvent,
    )
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        evaluate_outer_signal_opportunities,
    )
    from src.domain.futures.strategy.walk_forward import WFFold

    batch = ValidatedSignalBatch(
        events=(
            ValidatedSignalEvent(
                decision_idx=10,
                decision_time=np.datetime64("2024-01-01T00:00:00"),
                symbol="BTC",
                strategy_id="strat:v1",
                activation_context="all",
                side=-1,
                expected_gross_bps=12.0,
                q10_gross_bps=10.0,
                q90_gross_bps=14.0,
                expected_holding_bars=4,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
        ),
        start_idx=10,
        end_idx=11,
        symbols=("BTC",),
        registry_version="test",
        model_version="test",
    )
    realized = pd.DataFrame(
        {
            "entry_idx": [11],
            "symbol": ["BTC"],
            "strategy_id": ["strat:v1"],
            "activation_context": ["all"],
            "realized_side_adjusted_gross_bps": [12.0],
            "exit_idx": [14],
        }
    )
    fold = WFFold(fit_start=0, fit_end=10, cal_start=10, cal_end=10, oos_start=10, oos_end=20)
    cfg = _make_cfg(
        l1_opp_ic_mode="cross_section",
        l1_min_cross_section=1,
        l1_probe_top_k=1,
        l1_min_sym_count=1,
        l1_min_opportunity_timestamps=1,
        l1_min_fold_ratio=0.0,
    )
    vol = np.ones((20, 1), dtype=np.float64)

    result = evaluate_outer_signal_opportunities(
        opportunities=batch,
        realized_event_results=realized,
        volatility_2d=vol,
        aligned_symbols=("BTC",),
        fold=fold,
        fold_id=0,
        cfg=cfg,
        seed=0,
    )

    assert result.probe_bps == pytest.approx(12.0)
    assert "non_positive_probe" not in result.blockers


def test_evaluate_outer_signal_opportunities_uses_aligned_symbol_volatility_order() -> None:
    from src.domain.futures.strategy.candidate_contracts import (
        ValidatedSignalBatch,
        ValidatedSignalEvent,
    )
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        evaluate_outer_signal_opportunities,
    )
    from src.domain.futures.strategy.walk_forward import WFFold

    symbols = ("SOL", "BTC", "ETH")
    batch = ValidatedSignalBatch(
        events=(
            ValidatedSignalEvent(
                decision_idx=10,
                decision_time=np.datetime64("2024-01-01T00:00:00"),
                symbol="SOL",
                strategy_id="strat:v1",
                activation_context="all",
                side=1,
                expected_gross_bps=1.0,
                q10_gross_bps=0.8,
                q90_gross_bps=1.2,
                expected_holding_bars=4,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=10,
                decision_time=np.datetime64("2024-01-01T00:00:00"),
                symbol="BTC",
                strategy_id="strat:v1",
                activation_context="all",
                side=1,
                expected_gross_bps=1.0,
                q10_gross_bps=0.8,
                q90_gross_bps=1.2,
                expected_holding_bars=4,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=10,
                decision_time=np.datetime64("2024-01-01T00:00:00"),
                symbol="ETH",
                strategy_id="strat:v1",
                activation_context="all",
                side=1,
                expected_gross_bps=1.0,
                q10_gross_bps=0.8,
                q90_gross_bps=1.2,
                expected_holding_bars=4,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
        ),
        start_idx=10,
        end_idx=11,
        symbols=symbols,
        registry_version="test",
        model_version="test",
    )
    realized = pd.DataFrame(
        {
            "entry_idx": [11, 11, 11],
            "symbol": ["SOL", "BTC", "ETH"],
            "strategy_id": ["strat:v1"] * 3,
            "activation_context": ["all"] * 3,
            "realized_side_adjusted_gross_bps": [9.0, 1.0, 5.0],
            "exit_idx": [14, 14, 14],
        }
    )
    fold = WFFold(fit_start=0, fit_end=10, cal_start=10, cal_end=10, oos_start=10, oos_end=20)
    cfg = _make_cfg(
        l1_opp_ic_mode="cross_section",
        l1_min_cross_section=3,
        l1_probe_top_k=1,
        l1_min_sym_count=1,
        l1_min_opportunity_timestamps=1,
        l1_min_fold_ratio=0.0,
    )
    vol = np.zeros((20, 3), dtype=np.float64)
    vol[10] = np.asarray([0.01, 0.2, 0.1], dtype=np.float64)

    result = evaluate_outer_signal_opportunities(
        opportunities=batch,
        realized_event_results=realized,
        volatility_2d=vol,
        aligned_symbols=("BTC", "ETH", "SOL"),
        fold=fold,
        fold_id=0,
        cfg=cfg,
        seed=0,
    )

    assert result.probe_bps == pytest.approx(1.0)
