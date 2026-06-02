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


def test_compute_rule_diagnostics_keeps_positive_expectancy_low_hit_rate_variant() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [
                aligned.datetimes[33],
                aligned.datetimes[35],
                aligned.datetimes[37],
            ],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "family": ["trend_donchian", "trend_donchian", "trend_donchian"],
            "variant": ["donchian_72", "donchian_72", "donchian_72"],
            "side": [1, 1, 1],
            "raw_score": [0.9, 0.5, 0.4],
            "score_z": [1.2, 0.8, 0.6],
            "entry_idx": [34, 36, 38],
            "expected_holding_bars": [2, 2, 2],
            "min_holding_bars": [1, 1, 1],
            "stop_atr_mult": [50.0, 50.0, 50.0],
            "take_profit_atr_mult": [50.0, 50.0, 50.0],
            "turnover_proxy": [0.1, 0.1, 0.1],
            "cost_floor_bps": [0.0, 0.0, 0.0],
            "hurdle_bps": [0.0, 0.0, 0.0],
            "profitable_after_hurdle_label": [1, 0, 0],
            "edge_after_hurdle_bps": [300.0, -10.0, -10.0],
            "mae_bps": [-20.0, -20.0, -20.0],
            "mfe_bps": [400.0, 15.0, 15.0],
        }
    )
    labeled["triple_barrier_label"] = labeled["profitable_after_hurdle_label"]

    cfg = CandidateStrategyConfig(
        min_rule_hit_rate=0.50,
        min_variant_oos_obs=1,
        min_variant_oos_hit_rate=0.50,
        min_variant_oos_payoff_ratio=3.0,
        max_variant_oos_q10_fail_rate=0.50,
        max_expected_shortfall_bps=80.0,
    )

    result = compute_rule_diagnostics(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        min_obs=1,
    )

    variant_row = result.by_variant.loc[result.by_variant["group"] == "variant=trend_donchian:donchian_72"].iloc[0]
    assert float(variant_row["oos_pct_edge_pos"]) < 0.50
    assert float(variant_row["oos_payoff_ratio"]) >= 3.0
    assert variant_row["candidate_action"] == "KEEP_CANDIDATE"
    assert result.recommended_keep_variants == ("variant=trend_donchian:donchian_72",)
