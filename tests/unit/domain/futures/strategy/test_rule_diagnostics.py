from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_labels import label_candidate_events
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.rule_diagnostics import (
    _failed_recommendation_checks,
    _meets_recommendation_thresholds,
    compute_rule_diagnostics,
    summarize_recommendation_gate_failures,
)


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


def _make_regime_aligned() -> AlignedMarketData:
    t = 260
    n = 1
    low_vol = np.linspace(100.0, 140.0, 130, dtype=np.float64)
    high_vol_base = np.linspace(140.5, 190.0, 130, dtype=np.float64)
    high_vol_noise = np.array([(-1.0) ** i * 3.0 for i in range(130)], dtype=np.float64)
    close = np.concatenate([low_vol, high_vol_base + high_vol_noise]).reshape(t, n)
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


def test_compute_rule_diagnostics_keeps_positive_expectancy_variant_under_stricter_gates() -> None:
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
            "edge_after_hurdle_bps": [300.0, 20.0, -10.0],
            "mae_bps": [-20.0, -20.0, -20.0],
            "mfe_bps": [400.0, 25.0, 15.0],
        }
    )
    labeled["triple_barrier_label"] = labeled["profitable_after_hurdle_label"]

    cfg = CandidateStrategyConfig(
        min_rule_hit_rate=0.50,
        min_variant_oos_obs=1,
        min_variant_oos_hit_rate=0.50,
        min_variant_oos_payoff_ratio=3.0,
        max_variant_oos_q10_fail_rate=0.50,
        max_variant_event_fraction_per_bar=0.50,
        max_expected_shortfall_bps=80.0,
        regime_diagnostic_enabled=False,
    )

    result = compute_rule_diagnostics(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        min_obs=1,
    )

    variant_row = result.by_variant.loc[result.by_variant["group"] == "variant=trend_donchian:donchian_72"].iloc[0]
    assert float(variant_row["oos_pct_edge_pos"]) >= 0.50
    assert float(variant_row["oos_payoff_ratio"]) >= 3.0
    assert variant_row["candidate_action"] == "KEEP_CANDIDATE"
    assert float(variant_row["oos_rank_ic"]) >= cfg.min_oos_rank_ic
    assert result.recommended_keep_variants == ("trend_donchian:donchian_72",)


def test_compute_rule_diagnostics_promotes_signal_cells_when_present() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[33], aligned.datetimes[35]],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "family": ["trend_donchian", "trend_donchian"],
            "variant": ["donchian_72", "donchian_72"],
            "signal_cell": [
                "trend_donchian:donchian_72:trend_grind:bull_quiet",
                "trend_donchian:donchian_72:trend_grind:bull_quiet",
            ],
            "exit_policy_id": ["trend_grind", "trend_grind"],
            "entry_regime": ["bull_quiet", "bull_quiet"],
            "archetype": ["trend_continuation", "trend_continuation"],
            "side": [1, 1],
            "raw_score": [0.9, 0.8],
            "score_z": [1.2, 1.0],
            "entry_idx": [34, 36],
            "expected_holding_bars": [2, 2],
            "min_holding_bars": [1, 1],
            "stop_atr_mult": [50.0, 50.0],
            "take_profit_atr_mult": [50.0, 50.0],
            "turnover_proxy": [0.1, 0.1],
            "cost_floor_bps": [0.0, 0.0],
            "hurdle_bps": [0.0, 0.0],
            "profitable_after_hurdle_label": [1, 1],
            "edge_after_hurdle_bps": [300.0, 120.0],
            "mae_bps": [-20.0, -10.0],
            "mfe_bps": [400.0, 160.0],
            "triple_barrier_label": [1, 1],
        }
    )
    cfg = CandidateStrategyConfig(
        min_variant_oos_obs=1,
        min_signal_cell_oos_obs=1,
        max_variant_event_fraction_per_bar=0.50,
        max_signal_cell_event_fraction_per_bar=0.50,
        regime_diagnostic_enabled=False,
        # Use signal_cell promotion to test per-cell granularity explicitly.
        promotion_level="signal_cell",
    )

    result = compute_rule_diagnostics(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        min_obs=1,
    )

    assert result.recommended_keep_signal_cells == ("trend_donchian:donchian_72:trend_grind:bull_quiet",)


def test_rule_recommendations_use_explicit_recommendation_window_not_report_window() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [
                aligned.datetimes[9],
                aligned.datetimes[11],
                aligned.datetimes[21],
                aligned.datetimes[23],
                aligned.datetimes[33],
                aligned.datetimes[35],
            ],
            "symbol": ["BTCUSDT"] * 6,
            "family": ["trend_ma"] * 6,
            "variant": ["ema_12_72"] * 6,
            "side": [1] * 6,
            "raw_score": [0.5, 0.4, 0.9, 0.8, 0.1, 0.2],
            "score_z": [0.5, 0.4, 0.9, 0.8, 0.1, 0.2],
            "entry_idx": [10, 12, 22, 24, 34, 36],
            "expected_holding_bars": [2] * 6,
            "min_holding_bars": [1] * 6,
            "stop_atr_mult": [50.0] * 6,
            "take_profit_atr_mult": [50.0] * 6,
            "turnover_proxy": [0.1] * 6,
            "cost_floor_bps": [0.0] * 6,
            "hurdle_bps": [0.0] * 6,
            "profitable_after_hurdle_label": [0, 0, 1, 1, 0, 0],
            "edge_after_hurdle_bps": [-20.0, -10.0, 40.0, 30.0, -30.0, -25.0],
            "mae_bps": [-10.0] * 6,
            "mfe_bps": [15.0, 12.0, 55.0, 45.0, 10.0, 9.0],
        }
    )
    labeled["triple_barrier_label"] = labeled["profitable_after_hurdle_label"]

    cfg = CandidateStrategyConfig(
        min_variant_oos_obs=2,
        min_variant_oos_edge_bps=1.0,
        min_variant_oos_hit_rate=0.5,
        min_variant_oos_payoff_ratio=1.0,
        max_variant_oos_q10_fail_rate=1.0,
        regime_diagnostic_enabled=False,
    )

    result = compute_rule_diagnostics(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        min_obs=1,
        recommendation_start=20,
        recommendation_end=30,
        report_start=32,
        report_end=40,
    )

    variant_row = result.by_variant.loc[result.by_variant["group"] == "variant=trend_ma:ema_12_72"].iloc[0]
    assert float(variant_row["oos_mean_edge_bps"]) < 0.0
    assert result.recommended_keep_variants == ("trend_ma:ema_12_72",)
    assert result.recommendation_basis == "fit_calibration"
    assert result.recommendation_split == (20, 30)
    assert result.report_split == (32, 40)


def test_meets_recommendation_thresholds_rejects_median_below_floor() -> None:
    # min_variant_oos_median_edge_bps=-100; median=-101 should still fail.
    cfg = CandidateStrategyConfig(min_variant_oos_obs=10, min_variant_oos_median_edge_bps=-100.0)
    row = pd.Series(
        {
            "oos_n": 10,
            "oos_mean_edge_bps": 5.0,
            "oos_median_edge_bps": -101.0,
            "oos_p10_edge_bps": -50.0,
            "oos_q10_shortfall_fail_rate": 0.2,
            "event_fraction_per_bar": 0.05,
            "edge_stability_bps": 0.0,
            "oos_pct_edge_pos": 0.6,
            "oos_payoff_ratio": 1.3,
        }
    )
    assert not _meets_recommendation_thresholds(row, cfg)
    assert "median_edge" in _failed_recommendation_checks(row, cfg)


def test_summarize_recommendation_gate_failures_reports_pass_and_fail_reasons() -> None:
    cfg = CandidateStrategyConfig(
        min_variant_oos_obs=2,
        min_variant_oos_edge_bps=5.0,
        max_variant_event_fraction_per_bar=0.10,
    )
    summary = pd.DataFrame(
        [
            {
                "group": "variant=trend_ma:ema_12_72",
                "candidate_action": "KEEP_CANDIDATE",
                "oos_n": 3,
                "oos_mean_edge_bps": 8.0,
                "oos_median_edge_bps": 1.0,
                "oos_p10_edge_bps": -20.0,
                "oos_q10_shortfall_fail_rate": 0.10,
                "event_fraction_per_bar": 0.05,
                "edge_stability_bps": 0.0,
                "oos_pct_edge_pos": 0.60,
                "oos_payoff_ratio": 1.30,
                "archetype": "mean_reversion",
                "exit_policy_id": "",
            },
            {
                "group": "variant=trend_ma:ema_6_36",
                "candidate_action": "DROP_OR_REWORK",
                "oos_n": 3,
                "oos_mean_edge_bps": 2.0,
                "oos_median_edge_bps": 1.0,
                "oos_p10_edge_bps": -20.0,
                "oos_q10_shortfall_fail_rate": 0.10,
                "event_fraction_per_bar": 0.05,
                "edge_stability_bps": 0.0,
                "oos_pct_edge_pos": 0.60,
                "oos_payoff_ratio": 1.30,
                "archetype": "mean_reversion",
                "exit_policy_id": "",
            },
            {
                "group": "variant=rsi_reversion:rsi_14",
                "candidate_action": "DROP_OR_REWORK",
                "oos_n": 3,
                "oos_mean_edge_bps": 8.0,
                "oos_median_edge_bps": 1.0,
                "oos_p10_edge_bps": -20.0,
                "oos_q10_shortfall_fail_rate": 0.10,
                "event_fraction_per_bar": 0.25,
                "edge_stability_bps": 0.0,
                "oos_pct_edge_pos": 0.60,
                "oos_payoff_ratio": 1.30,
                "archetype": "mean_reversion",
                "exit_policy_id": "",
            },
        ]
    )

    out = summarize_recommendation_gate_failures(summary, cfg)

    keep_row = out.loc[out["group"] == "variant=trend_ma:ema_12_72"].iloc[0]
    edge_row = out.loc[out["group"] == "variant=trend_ma:ema_6_36"].iloc[0]
    density_row = out.loc[out["group"] == "variant=rsi_reversion:rsi_14"].iloc[0]

    assert bool(keep_row["recommended"]) is True
    assert keep_row["failed_checks"] == ()
    assert bool(edge_row["recommended"]) is False
    assert "mean_edge" in edge_row["failed_checks"]
    assert bool(density_row["recommended"]) is False
    assert "event_density" in density_row["failed_checks"]


def test_meets_recommendation_thresholds_rejects_bad_p10_tail() -> None:
    cfg = CandidateStrategyConfig(min_variant_oos_obs=10, min_variant_oos_p10_edge_bps=-150.0)
    row = pd.Series(
        {
            "oos_n": 10,
            "oos_mean_edge_bps": 6.0,
            "oos_median_edge_bps": 4.0,
            "oos_p10_edge_bps": -220.0,
            "oos_q10_shortfall_fail_rate": 0.2,
            "event_fraction_per_bar": 0.05,
            "edge_stability_bps": 0.0,
            "oos_pct_edge_pos": 0.6,
            "oos_payoff_ratio": 1.3,
        }
    )
    assert not _meets_recommendation_thresholds(row, cfg)
    assert "p10_edge" in _failed_recommendation_checks(row, cfg)


def test_meets_recommendation_thresholds_rejects_over_dense_variant() -> None:
    cfg = CandidateStrategyConfig(min_variant_oos_obs=10, max_variant_event_fraction_per_bar=0.10)
    row = pd.Series(
        {
            "oos_n": 10,
            "oos_mean_edge_bps": 6.0,
            "oos_median_edge_bps": 4.0,
            "oos_p10_edge_bps": -50.0,
            "oos_q10_shortfall_fail_rate": 0.2,
            "event_fraction_per_bar": 0.25,
            "edge_stability_bps": 0.0,
            "oos_pct_edge_pos": 0.6,
            "oos_payoff_ratio": 1.3,
        }
    )
    assert not _meets_recommendation_thresholds(row, cfg)
    assert "event_density" in _failed_recommendation_checks(row, cfg)


def test_meets_recommendation_thresholds_accepts_when_all_gates_pass() -> None:
    cfg = CandidateStrategyConfig(min_variant_oos_obs=10)
    row = pd.Series(
        {
            "oos_n": 12,
            "oos_mean_edge_bps": 7.0,
            "oos_median_edge_bps": 6.0,
            "oos_p10_edge_bps": -40.0,
            "oos_q10_shortfall_fail_rate": 0.2,
            "event_fraction_per_bar": 0.05,
            "regime_pass": True,
            "edge_stability_bps": -10.0,
            "oos_pct_edge_pos": 0.6,
            "oos_payoff_ratio": 1.3,
        }
    )
    assert _meets_recommendation_thresholds(row, cfg)


def test_compute_rule_diagnostics_rejects_variant_without_positive_regime_edge() -> None:
    aligned = _make_regime_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[idx] for idx in (110, 120, 130, 210, 220, 230)],
            "symbol": ["BTCUSDT"] * 6,
            "family": ["trend_ma"] * 6,
            "variant": ["ema_12_72"] * 6,
            "side": [1] * 6,
            "raw_score": [0.6] * 6,
            "score_z": [0.7] * 6,
            "entry_idx": [110, 120, 130, 210, 220, 230],
            "expected_holding_bars": [2] * 6,
            "min_holding_bars": [1] * 6,
            "stop_atr_mult": [50.0] * 6,
            "take_profit_atr_mult": [50.0] * 6,
            "turnover_proxy": [0.1] * 6,
            "cost_floor_bps": [0.0] * 6,
            "hurdle_bps": [0.0] * 6,
            "profitable_after_hurdle_label": [1, 1, 1, 1, 1, 1],
            "edge_after_hurdle_bps": [1.5, 1.6, 1.7, 1.6, 1.7, 1.8],
            "mae_bps": [-10.0] * 6,
            "mfe_bps": [20.0] * 6,
        }
    )
    labeled["triple_barrier_label"] = labeled["profitable_after_hurdle_label"]
    cfg = CandidateStrategyConfig(
        min_variant_oos_obs=3,
        min_variant_oos_edge_bps=1.0,
        min_variant_oos_median_edge_bps=0.0,
        min_variant_oos_p10_edge_bps=-50.0,
        min_variant_oos_hit_rate=0.5,
        min_variant_oos_payoff_ratio=1.0,
        max_variant_oos_q10_fail_rate=1.0,
        max_variant_event_fraction_per_bar=1.0,
        min_regime_variant_oos_obs=1,
        min_regime_variant_oos_edge_bps=2.0,
        # signal_cell promotion: regime_edge check is active (as in legacy behaviour)
        promotion_level="signal_cell",
    )

    result = compute_rule_diagnostics(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        min_obs=1,
        recommendation_start=100,
        recommendation_end=240,
        report_start=100,
        report_end=240,
    )

    variant_row = result.by_variant.loc[result.by_variant["group"] == "variant=trend_ma:ema_12_72"].iloc[0]
    rec_row = pd.Series(
        {
            "oos_n": 6,
            "oos_mean_edge_bps": 1.65,
            "oos_median_edge_bps": 1.65,
            "oos_p10_edge_bps": 1.5,
            "oos_q10_shortfall_fail_rate": 0.0,
            "event_fraction_per_bar": 6.0 / 140.0,
            "regime_pass": False,
            "edge_stability_bps": 0.0,
            "oos_pct_edge_pos": 1.0,
            "oos_payoff_ratio": 2.0,
        }
    )
    assert variant_row["candidate_action"] == "KEEP_CANDIDATE"
    assert result.recommended_keep_variants == ()
    assert "regime_edge" in _failed_recommendation_checks(rec_row, cfg)


def test_compute_rule_diagnostics_accepts_variant_with_positive_regime_edge() -> None:
    aligned = _make_regime_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[idx] for idx in (110, 120, 130, 210, 220, 230)],
            "symbol": ["BTCUSDT"] * 6,
            "family": ["trend_ma"] * 6,
            "variant": ["ema_12_72"] * 6,
            "side": [1] * 6,
            "raw_score": [0.6] * 6,
            "score_z": [0.7] * 6,
            "entry_idx": [110, 120, 130, 210, 220, 230],
            "expected_holding_bars": [2] * 6,
            "min_holding_bars": [1] * 6,
            "stop_atr_mult": [50.0] * 6,
            "take_profit_atr_mult": [50.0] * 6,
            "turnover_proxy": [0.1] * 6,
            "cost_floor_bps": [0.0] * 6,
            "hurdle_bps": [0.0] * 6,
            "profitable_after_hurdle_label": [1, 1, 1, 1, 1, 1],
            "edge_after_hurdle_bps": [3.1, 3.0, 3.2, 1.0, 1.1, 1.2],
            "mae_bps": [-10.0] * 6,
            "mfe_bps": [20.0] * 6,
        }
    )
    labeled["triple_barrier_label"] = labeled["profitable_after_hurdle_label"]
    cfg = CandidateStrategyConfig(
        min_variant_oos_obs=3,
        min_variant_oos_edge_bps=1.0,
        min_variant_oos_median_edge_bps=0.0,
        min_variant_oos_p10_edge_bps=-50.0,
        min_variant_oos_hit_rate=0.5,
        min_variant_oos_payoff_ratio=1.0,
        max_variant_oos_q10_fail_rate=1.0,
        max_variant_event_fraction_per_bar=1.0,
        min_regime_variant_oos_obs=1,
        min_regime_variant_oos_edge_bps=2.0,
    )

    result = compute_rule_diagnostics(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        min_obs=1,
        recommendation_start=100,
        recommendation_end=240,
        report_start=100,
        report_end=240,
    )

    assert result.recommended_keep_variants == ("trend_ma:ema_12_72",)
