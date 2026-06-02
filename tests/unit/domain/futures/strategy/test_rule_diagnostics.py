from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_labels import label_candidate_events
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.rule_diagnostics import compute_rule_diagnostics


def _make_aligned() -> AlignedMarketData:
    t = 40
    n = 1
    close = np.linspace(100.0, 130.0, t, dtype=np.float64).reshape(t, n)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.zeros((t, n), dtype=np.float64),
    )


def _make_events(aligned: AlignedMarketData) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": [
                aligned.datetimes[10],
                aligned.datetimes[14],
                aligned.datetimes[10],
                aligned.datetimes[14],
            ],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "family": [
                "trend_ma",
                "trend_ma",
                "rsi_reversion",
                "rsi_reversion",
            ],
            "variant": [
                "ema_12_72",
                "ema_6_36",
                "rsi_14",
                "rsi_6",
            ],
            "side": [1, 1, -1, -1],
            "raw_score": [0.9, 0.2, -0.8, -0.3],
            "score_z": [0.9, 0.2, -0.8, -0.3],
            "entry_idx": [11, 15, 11, 15],
            "expected_holding_bars": [4, 4, 4, 4],
            "min_holding_bars": [1, 1, 1, 1],
            "stop_atr_mult": [50.0, 50.0, 50.0, 50.0],
            "take_profit_atr_mult": [50.0, 50.0, 50.0, 50.0],
            "turnover_proxy": [0.1, 0.1, 0.1, 0.1],
            "cost_floor_bps": [0.0, 0.0, 0.0, 0.0],
            "hurdle_bps": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_compute_rule_diagnostics_detects_keep_and_side_flip_candidates() -> None:
    aligned = _make_aligned()
    labeled = label_candidate_events(
        events=_make_events(aligned),
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
    )

    result = compute_rule_diagnostics(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
        min_obs=1,
    )

    assert result.by_family.shape[0] == 2
    assert result.by_variant.shape[0] == 4
    assert result.by_family_side.shape[0] == 2
    assert np.isfinite(
        float(result.by_family.loc[result.by_family["group"] == "family=trend_ma", "spearman_score_edge"].iloc[0])
    )
    assert (
        result.by_family.loc[result.by_family["group"] == "family=trend_ma", "candidate_action"].iloc[0]
        == "KEEP_CANDIDATE"
    )
    assert (
        result.by_family.loc[result.by_family["group"] == "family=rsi_reversion", "candidate_action"].iloc[0]
        == "SIDE_FLIP_CANDIDATE"
    )
    assert result.decision["keep"] == 2
    assert result.decision["flip"] == 2
    assert result.decision["best_group"] == "variant=trend_ma:ema_12_72"

    flip_row = result.side_flip.loc[result.side_flip["group"] == "family=rsi_reversion"]
    assert not flip_row.empty
    assert flip_row.iloc[0]["candidate_action"] == "SIDE_FLIP_CANDIDATE"
    assert float(flip_row.iloc[0]["delta_mean_edge_bps"]) > 25.0


def test_compute_rule_diagnostics_marks_insufficient_obs() -> None:
    aligned = _make_aligned()
    events = _make_events(aligned).iloc[[0]].copy().reset_index(drop=True)
    labeled = label_candidate_events(
        events=events,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
    )

    result = compute_rule_diagnostics(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
        min_obs=2,
    )

    assert result.by_family.iloc[0]["candidate_action"] == "INSUFFICIENT_OBS"
    assert result.side_flip.iloc[0]["candidate_action"] == "INSUFFICIENT_OBS"
