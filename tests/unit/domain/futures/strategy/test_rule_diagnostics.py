from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_labels import label_candidate_events
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.rule_diagnostics import (
    _failed_recommendation_checks,
    _meets_recommendation_thresholds,
    _regime_cell_admission,
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
        # IC t-stat gate requires N>=3; bypass here since test uses only 2 events
        min_ic_tstat=0.0,
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
        # IC t-stat requires N>=3; bypass to test recommendation-window logic in isolation
        min_ic_tstat=0.0,
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
                "oos_n": 300,
                "oos_mean_edge_bps": 8.0,
                "oos_median_edge_bps": 1.0,
                "oos_p10_edge_bps": -20.0,
                "oos_q10_shortfall_fail_rate": 0.10,
                "event_fraction_per_bar": 0.05,
                "edge_stability_bps": 0.0,
                "oos_pct_edge_pos": 0.60,
                "oos_payoff_ratio": 1.30,
                "breakeven_hard_pass": True,
                "oos_rank_ic": 0.06,
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
                "breakeven_hard_pass": True,
                "oos_rank_ic": 0.05,
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
                "breakeven_hard_pass": True,
                "oos_rank_ic": 0.05,
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
            "breakeven_hard_pass": True,
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
            "breakeven_hard_pass": True,
        }
    )
    assert not _meets_recommendation_thresholds(row, cfg)
    assert "event_density" in _failed_recommendation_checks(row, cfg)


def test_meets_recommendation_thresholds_accepts_when_all_gates_pass() -> None:
    # oos_n=300 ensures ic_tstat gate passes: t = 0.06 * sqrt(298) / sqrt(1-0.06²) ≈ 1.04 > 0.8
    cfg = CandidateStrategyConfig(min_variant_oos_obs=10)
    row = pd.Series(
        {
            "oos_n": 300,
            "oos_mean_edge_bps": 7.0,
            "oos_median_edge_bps": 6.0,
            "oos_p10_edge_bps": -40.0,
            "oos_q10_shortfall_fail_rate": 0.2,
            "event_fraction_per_bar": 0.05,
            "regime_pass": True,
            "edge_stability_bps": -10.0,
            "oos_pct_edge_pos": 0.6,
            "oos_payoff_ratio": 1.3,
            "breakeven_hard_pass": True,
            "oos_rank_ic": 0.06,
        }
    )
    assert _meets_recommendation_thresholds(row, cfg)


def test_ic_tstat_gate_uses_oos_ic_n_not_inflated_is_n() -> None:
    # Arrange: IC=0.0375 with true OOS N=309 gives t≈0.66 < 0.8 (FAIL),
    # but the IS-window oos_n=772 would inflate t to ≈1.04 (false PASS).
    # The gate must honor oos_ic_n (true OOS sample) over oos_n.
    cfg = CandidateStrategyConfig(min_variant_oos_obs=10, min_ic_tstat=0.8)
    base = {
        "oos_n": 772,
        "oos_mean_edge_bps": 7.0,
        "oos_median_edge_bps": 6.0,
        "oos_p10_edge_bps": -40.0,
        "oos_q10_shortfall_fail_rate": 0.2,
        "event_fraction_per_bar": 0.05,
        "regime_pass": True,
        "edge_stability_bps": -10.0,
        "oos_pct_edge_pos": 0.6,
        "oos_payoff_ratio": 1.3,
        "breakeven_hard_pass": True,
        "oos_rank_ic": 0.0375,
    }

    # Act: with correct OOS N the t-stat fails; without oos_ic_n it would falsely pass.
    rejected = _meets_recommendation_thresholds(pd.Series({**base, "oos_ic_n": 309}), cfg)
    inflated = _meets_recommendation_thresholds(pd.Series(base), cfg)

    # Assert
    assert rejected is False
    assert inflated is True


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
        min_oos_rank_ic=0.0,
        standalone_breakeven_hard_gate_enabled=False,
        # IC t-stat gate bypassed to test regime-edge logic in isolation
        min_ic_tstat=0.0,
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


def test_breakeven_hard_gate_excludes_subbreakeven_variant() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[33], aligned.datetimes[35], aligned.datetimes[37]],
            "symbol": ["BTCUSDT"] * 3,
            "family": ["trend_ma"] * 3,
            "variant": ["ema_12_72"] * 3,
            "archetype": ["trend_continuation"] * 3,
            "side": [1] * 3,
            "raw_score": [0.9, 0.8, 0.7],
            "score_z": [1.2, 1.1, 1.0],
            "entry_idx": [34, 36, 38],
            "expected_holding_bars": [4] * 3,
            "min_holding_bars": [1] * 3,
            "stop_atr_mult": [50.0] * 3,
            "take_profit_atr_mult": [50.0] * 3,
            "turnover_proxy": [0.1] * 3,
            "cost_floor_bps": [0.0] * 3,
            "hurdle_bps": [0.0] * 3,
            "profitable_after_hurdle_label": [1, 1, 0],
            "edge_after_hurdle_bps": [100.0, -90.0, 20.0],
            "mae_bps": [-10.0] * 3,
            "mfe_bps": [120.0, 20.0, 30.0],
            "triple_barrier_label": [1, 1, 0],
        }
    )
    cfg = CandidateStrategyConfig(
        min_variant_oos_obs=1,
        min_variant_oos_edge_bps=1.0,
        min_variant_oos_hit_rate=0.5,
        min_variant_oos_payoff_ratio=1.0,
        max_variant_oos_q10_fail_rate=1.0,
        max_variant_event_fraction_per_bar=1.0,
        regime_diagnostic_enabled=False,
        standalone_breakeven_hard_gate_enabled=True,
        min_rule_ir_t=1.0,
    )

    result = compute_rule_diagnostics(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        min_obs=1,
        recommendation_start=32,
        recommendation_end=40,
        report_start=32,
        report_end=40,
    )

    assert result.recommended_keep_variants == ()
    assert result.recommendation_failure_report is not None
    assert "breakeven_hard_gate" in result.recommendation_failure_report["rows"][0]["failed_checks"]


# ---------------------------------------------------------------------------
# Regime-cell conditional admission tests
# ---------------------------------------------------------------------------

def _make_cell_admission_cfg(**overrides: object) -> CandidateStrategyConfig:
    defaults: dict[str, object] = {
        "regime_cell_admission_enabled": True,
        "min_regime_cell_oos_obs": 60,
        "min_regime_cell_edge_bps": 8.0,
        "min_regime_cell_tstat": 1.0,
        "max_admitted_cells_per_variant": 2,
        "min_variant_oos_obs": 10,
        "max_variant_oos_q10_fail_rate": 0.65,
        "max_variant_event_fraction_per_bar": 1.0,
    }
    defaults.update(overrides)
    return CandidateStrategyConfig(**defaults)  # type: ignore[arg-type]


def _make_regime_arrays(
    n_bars: int = 300,
    n_per_cell: int = 90,
    cell_code: int = 0,
    cell_edge: float = 14.0,
    noise_std: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build entry_idx, edge, and regime_code arrays for admission tests.

    Bar layout: first half = cell_code, second half = code 1 (different regime).
    Returns (rec_entry_idx, rec_edge, regime_code).
    """
    rng = np.random.default_rng(42)
    half = n_bars // 2
    other_code = 1 if cell_code != 1 else 2
    regime_code = np.full(n_bars, other_code, dtype=np.int64)
    regime_code[:half] = cell_code
    # n_per_cell events in target cell (bars 0..half-1), same count in other cell
    entry_idx = np.concatenate([
        np.arange(0, n_per_cell, dtype=np.int64),
        np.arange(half, half + n_per_cell, dtype=np.int64),
    ])
    edge = np.concatenate([
        rng.normal(cell_edge, noise_std, n_per_cell),
        rng.normal(2.0, noise_std, n_per_cell),
    ])
    return entry_idx, edge, regime_code


def test_regime_cell_admission_happy_path_promotes_cell_specialist() -> None:
    # Arrange: bull_quiet cell has strong edge; global average is weak.
    cfg = _make_cell_admission_cfg()
    regime_names = ["bull_quiet", "bear_quiet", "bull_volatile", "bear_volatile", "transition", "crash"]
    entry_idx, edge, regime_code = _make_regime_arrays(cell_edge=14.0, cell_code=0)

    # Act
    result = _regime_cell_admission(entry_idx, edge, regime_code, regime_names, cfg)

    # Assert
    assert result["admitted"] is True
    assert "bull_quiet" in result["admitted_cells"]


def test_regime_cell_admission_rejects_insufficient_obs() -> None:
    # Arrange: only 30 events in cell — below min_regime_cell_oos_obs=60.
    cfg = _make_cell_admission_cfg(min_regime_cell_oos_obs=60)
    regime_names = ["bull_quiet", "bear_quiet"]
    entry_idx, edge, regime_code = _make_regime_arrays(n_per_cell=30, cell_edge=20.0, cell_code=0)

    # Act
    result = _regime_cell_admission(entry_idx, edge, regime_code, regime_names, cfg)

    # Assert: cell is ignored due to insufficient obs; no admission from that cell
    assert "bull_quiet" not in result["admitted_cells"]


def test_regime_cell_admission_caps_at_max_cells() -> None:
    # Arrange: 4 cells each pass, but max_admitted_cells_per_variant=2.
    cfg = _make_cell_admission_cfg(max_admitted_cells_per_variant=2)
    regime_names = ["bull_quiet", "bear_quiet", "bull_volatile", "bear_volatile"]
    rng = np.random.default_rng(0)
    n_bars, obs_per_cell = 600, 80
    regime_code = np.repeat(np.arange(4, dtype=np.int64), n_bars // 4)
    entry_idx = np.arange(obs_per_cell * 4, dtype=np.int64)
    # Edges: 25, 20, 15, 10 bps — all above threshold=8
    edge = np.concatenate([rng.normal(e, 30.0, obs_per_cell) for e in (25.0, 20.0, 15.0, 10.0)])

    # Act
    result = _regime_cell_admission(entry_idx, edge, regime_code, regime_names, cfg)

    # Assert
    assert len(result["admitted_cells"]) == 2
    # Top-2 by edge: bull_quiet (25bps) and bear_quiet (20bps)
    assert result["admitted_cells"][0] == "bull_quiet"
    assert result["admitted_cells"][1] == "bear_quiet"


def test_regime_cell_admission_safety_gate_blocks_high_q10_fail() -> None:
    # Arrange: cell passes edge/tstat, but q10_shortfall_fail_rate=0.80 > 0.65.
    cfg = _make_cell_admission_cfg()
    row = pd.Series({
        "oos_n": 150,
        "oos_mean_edge_bps": 4.0,            # fails global mean_edge gate
        "oos_median_edge_bps": -5.0,
        "oos_p10_edge_bps": -200.0,
        "oos_q10_shortfall_fail_rate": 0.80,  # FAILS safety gate
        "event_fraction_per_bar": 0.05,
        "regime_pass": False,
        "edge_stability_bps": float("nan"),
        "oos_pct_edge_pos": 0.4,
        "oos_payoff_ratio": 1.1,
        "breakeven_hard_pass": False,
        "oos_rank_ic": 0.01,
        "regime_cell_admitted": True,         # cell path would normally override
    })

    # Act
    result = _meets_recommendation_thresholds(row, cfg)

    # Assert: q10_fail safety gate rejects despite cell_admitted=True
    assert result is False
    checks = _failed_recommendation_checks(row, cfg)
    assert "q10_fail" in checks


def test_regime_cell_admission_default_off_is_bit_identical() -> None:
    # Arrange: default config (regime_cell_admission_enabled=False) with a row
    # that would be admitted via cell path if enabled.
    cfg_off = CandidateStrategyConfig(
        min_variant_oos_obs=10,
        regime_cell_admission_enabled=False,
    )
    cfg_on = CandidateStrategyConfig(
        min_variant_oos_obs=10,
        regime_cell_admission_enabled=True,
        min_regime_cell_oos_obs=60,
        min_regime_cell_edge_bps=8.0,
        min_regime_cell_tstat=1.0,
        max_admitted_cells_per_variant=2,
    )
    row_no_admission = pd.Series({
        "oos_n": 150,
        "oos_mean_edge_bps": 4.0,
        "oos_median_edge_bps": 3.0,
        "oos_p10_edge_bps": -200.0,
        "oos_q10_shortfall_fail_rate": 0.40,
        "event_fraction_per_bar": 0.05,
        "regime_pass": False,
        "edge_stability_bps": float("nan"),
        "oos_pct_edge_pos": 0.4,
        "oos_payoff_ratio": 1.1,
        "breakeven_hard_pass": False,
        "oos_rank_ic": 0.005,
        "regime_cell_admitted": False,  # not admitted even if enabled
    })

    # Act
    result_off = _meets_recommendation_thresholds(row_no_admission, cfg_off)
    result_on = _meets_recommendation_thresholds(row_no_admission, cfg_on)

    # Assert: both paths reject identically when no cell_admitted
    assert result_off is False
    assert result_on is False


def test_regime_cell_admission_zero_std_guard_prevents_nan_tstat() -> None:
    # Arrange: all edges in cell are identical → std=0.
    cfg = _make_cell_admission_cfg()
    regime_names = ["bull_quiet", "bear_quiet"]
    n_bars = 300
    regime_code = np.zeros(n_bars, dtype=np.int64)
    entry_idx = np.arange(80, dtype=np.int64)
    edge = np.full(80, 15.0, dtype=np.float64)  # std=0 for all 80 events

    # Act
    result = _regime_cell_admission(entry_idx, edge, regime_code, regime_names, cfg)

    # Assert: tstat is finite (no inf/nan from zero division)
    assert np.isfinite(result["cell_tstats"].get("bull_quiet", float("nan")))
    assert result["admitted"] is True
