from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_dataset import CandidateDataset, build_candidate_dataset
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig


def _make_aligned() -> AlignedMarketData:
    t = 60
    n = 2
    x = np.linspace(100.0, 160.0, t, dtype=np.float64)
    close = np.stack([x, x * 1.02], axis=1)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
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
        execution_cost_bps_2d=np.full((t, n), 4.0, dtype=np.float64),
        vol_30d_1d=np.array([0.35, 0.55], dtype=np.float32),
        friction_score_1d=np.array([0.8, 0.6], dtype=np.float32),
        alpha_capacity_score_1d=np.array([0.7, 0.9], dtype=np.float32),
        diversification_score_1d=np.array([0.4, 0.5], dtype=np.float32),
        tradeable_score_1d=np.array([0.75, 0.72], dtype=np.float32),
        cluster_id_1d=np.array([3.0, 7.0], dtype=np.float32),
        beta_vs_market_1d=np.array([1.1, 0.4], dtype=np.float32),
        cluster_size_1d=np.array([2.0, 5.0], dtype=np.float32),
        anchor_cluster_1d=np.array([1.0, 0.0], dtype=np.float32),
    )


def test_build_candidate_dataset_shapes_and_types() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25], aligned.datetimes[30]],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": [1, -1],
            "entry_idx": [26, 31],
            "raw_score": [0.5, -0.4],
            "score_z": [1.2, -0.8],
            "turnover_proxy": [0.1, 0.2],
            "triple_barrier_label": [1, 0],
            "profitable_after_hurdle_label": [0, 1],
            "edge_after_hurdle_bps": [12.0, -5.0],
            "sl_thr_bps": [25.0, 30.0],
            "mae_bps": [-6.0, -8.0],
            "mfe_bps": [18.0, 5.0],
            "ex_ante_cost_bps": [4.0, 4.0],
        }
    )

    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
        split_start=20,
        split_end=40,
    )

    assert isinstance(ds, CandidateDataset)
    assert ds.X.shape[0] == 2
    assert ds.X.shape[1] == len(ds.feature_names)
    assert ds.y_gate.dtype == np.int8
    assert ds.y_edge_bps.dtype == np.float32
    assert ds.y_mfe_bps.dtype == np.float32
    assert ds.groups.dtype == np.int32
    assert ds.y_gate.tolist() == [0, 1]
    assert ds.y_q10_bps.tolist() == [-10.0, -12.0]
    assert ds.y_mfe_bps.tolist() == [14.0, 1.0]
    assert ds.feature_schema_version == "candidate_v3"
    assert "sl_thr_bps" in ds.feature_names
    assert "universe_vol_30d" in ds.feature_names
    assert "universe_friction_score" in ds.feature_names
    assert "universe_alpha_capacity_score" in ds.feature_names
    assert "universe_diversification_score" in ds.feature_names
    assert "universe_tradeable_score" in ds.feature_names
    assert "universe_cluster_id" in ds.feature_names
    assert "universe_beta_vs_market" in ds.feature_names
    assert "universe_cluster_size" in ds.feature_names
    assert "universe_anchor_cluster_member" in ds.feature_names
    assert ds.X[0, ds.feature_names.index("sl_thr_bps")] == pytest.approx(25.0)


def test_build_candidate_dataset_split_filter() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[10], aligned.datetimes[50]],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": [1, 1],
            "entry_idx": [11, 51],
            "raw_score": [0.5, 0.6],
            "score_z": [1.0, 1.1],
            "turnover_proxy": [0.1, 0.1],
            "triple_barrier_label": [1, 1],
            "edge_after_hurdle_bps": [10.0, 10.0],
            "mae_bps": [-5.0, -5.0],
            "mfe_bps": [15.0, 15.0],
            "ex_ante_cost_bps": [4.0, 4.0],
        }
    )

    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
        split_start=20,
        split_end=40,
    )
    assert ds.X.shape[0] == 0
    assert ds.X.shape[1] == len(ds.feature_names)


def test_build_candidate_dataset_uses_configured_gate_label_column() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25], aligned.datetimes[30]],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": [1, -1],
            "entry_idx": [26, 31],
            "raw_score": [0.5, -0.4],
            "score_z": [1.2, -0.8],
            "turnover_proxy": [0.1, 0.2],
            "barrier_first_label": [1, 0],
            "triple_barrier_label": [1, 0],
            "edge_after_hurdle_bps": [12.0, -5.0],
            "mae_bps": [-6.0, -8.0],
            "mfe_bps": [18.0, 5.0],
            "ex_ante_cost_bps": [4.0, 4.0],
        }
    )

    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(gate_label_column="barrier_first_label"),
        split_start=20,
        split_end=40,
    )

    assert ds.y_gate.tolist() == [1, 0]


def test_build_candidate_dataset_raises_when_configured_gate_label_is_missing() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [26],
            "raw_score": [0.5],
            "score_z": [1.2],
            "turnover_proxy": [0.1],
            "triple_barrier_label": [1],
            "edge_after_hurdle_bps": [12.0],
            "mae_bps": [-6.0],
            "mfe_bps": [18.0],
            "ex_ante_cost_bps": [4.0],
        }
    )

    with pytest.raises(ValueError, match="missing configured gate label column"):
        build_candidate_dataset(
            labeled_events=labeled,
            aligned=aligned,
            cfg=CandidateStrategyConfig(gate_label_column="profitable_after_hurdle_label"),
            split_start=20,
            split_end=40,
        )


def test_build_candidate_dataset_builds_net_q10_and_mfe_targets_with_hurdle() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [26],
            "raw_score": [0.5],
            "score_z": [1.2],
            "turnover_proxy": [0.1],
            "triple_barrier_label": [1],
            "profitable_after_hurdle_label": [1],
            "edge_after_hurdle_bps": [12.0],
            "mae_bps": [-6.0],
            "mfe_bps": [18.0],
            "ex_ante_cost_bps": [4.0],
            "hurdle_bps": [2.0],
        }
    )

    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
        split_start=20,
        split_end=40,
    )

    assert ds.y_q10_bps.tolist() == [-12.0]
    assert ds.y_mfe_bps.tolist() == [12.0]


def test_build_candidate_dataset_keeps_feature_schema_stable_across_splits() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25], aligned.datetimes[45]],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "family": ["trend_ma", "trend_donchian"],
            "variant": ["ema_12_72", "donchian_36"],
            "side": [1, -1],
            "entry_idx": [26, 46],
            "raw_score": [0.5, -0.4],
            "score_z": [1.2, -0.8],
            "turnover_proxy": [0.1, 0.2],
            "expected_holding_bars": [12, 24],
            "min_holding_bars": [4, 8],
            "stop_atr_mult": [2.0, 2.0],
            "take_profit_atr_mult": [4.0, 4.0],
            "triple_barrier_label": [1, 0],
            "profitable_after_hurdle_label": [1, 0],
            "edge_after_hurdle_bps": [12.0, -5.0],
            "mae_bps": [-6.0, -8.0],
            "mfe_bps": [18.0, 5.0],
            "ex_ante_cost_bps": [4.0, 4.0],
        }
    )

    train = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
        split_start=20,
        split_end=40,
    )
    valid = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
        split_start=40,
        split_end=60,
    )

    assert train.feature_names == valid.feature_names
    assert train.feature_schema_version == "candidate_v3"
    assert valid.feature_schema_version == "candidate_v3"
    assert "family=trend_ma" in train.feature_names
    assert "variant=trend_donchian:donchian_36" in train.feature_names


def test_build_candidate_dataset_includes_universe_metadata_features() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [26],
            "raw_score": [0.5],
            "score_z": [1.2],
            "turnover_proxy": [0.1],
            "triple_barrier_label": [1],
            "profitable_after_hurdle_label": [1],
            "edge_after_hurdle_bps": [12.0],
            "mae_bps": [-6.0],
            "mfe_bps": [18.0],
            "ex_ante_cost_bps": [4.0],
        }
    )

    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
        split_start=20,
        split_end=40,
    )

    by_name = {name: idx for idx, name in enumerate(ds.feature_names)}
    row = ds.X[0]
    assert row[by_name["universe_vol_30d"]] == pytest.approx(0.35)
    assert row[by_name["universe_friction_score"]] == pytest.approx(0.8)
    assert row[by_name["universe_alpha_capacity_score"]] == pytest.approx(0.7)
    assert row[by_name["universe_diversification_score"]] == pytest.approx(0.4)
    assert row[by_name["universe_tradeable_score"]] == pytest.approx(0.75)
    assert row[by_name["universe_cluster_id"]] == 3.0
    assert row[by_name["universe_beta_vs_market"]] == 1.1
    assert row[by_name["universe_cluster_size"]] == 2.0
    assert row[by_name["universe_anchor_cluster_member"]] == 1.0
