from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_dataset import (
    CandidateDataset,
    build_candidate_dataset,
    fit_candidate_feature_schema,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.market_regime import compute_market_regime_context


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
            "exit_idx": [27, 32],
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
        schema=fit_candidate_feature_schema(
            labeled_events=labeled,
            cfg=CandidateStrategyConfig(),
            split_start=20,
            split_end=40,
        ),
        split_start=20,
        split_end=40,
    )

    assert isinstance(ds, CandidateDataset)
    assert ds.X.shape[0] == 2
    assert ds.X.shape[1] == len(ds.feature_names)
    assert ds.y_gate.dtype == np.int8
    assert ds.y_edge_bps.dtype == np.float32
    assert ds.y_return_r is not None
    assert ds.y_return_bps is not None
    assert ds.y_mae_r is not None
    assert ds.risk_unit_bps is not None
    assert ds.groups.dtype == np.int32
    assert ds.y_gate.tolist() == [0, 1]
    assert ds.y_return_bps.tolist() == [12.0, -5.0]
    assert ds.y_q10_bps.tolist() == [12.0, -5.0]
    assert ds.y_mfe_bps.tolist() == [12.0, -5.0]
    assert ds.feature_schema_version == "candidate_v6"
    assert np.all(ds.gate_weight > 0.0)
    assert np.all(ds.edge_weight > 0.0)
    assert ds.effective_sample_size > 0.0
    assert "sl_thr_bps" in ds.feature_names
    assert "universe_vol_30d" not in ds.feature_names
    assert ds.X[0, ds.feature_names.index("sl_thr_bps")] == pytest.approx(25.0)
    assert "edge_after_hurdle_bps" not in ds.feature_names
    assert "profitable_after_hurdle_label" not in ds.feature_names
    assert "edge_after_hurdle_bps" in ds.event_index.columns
    assert "profitable_after_hurdle_label" in ds.event_index.columns


def test_build_candidate_dataset_split_filter() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[10], aligned.datetimes[50]],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": [1, 1],
            "entry_idx": [11, 51],
            "exit_idx": [12, 52],
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
        schema=fit_candidate_feature_schema(
            labeled_events=labeled,
            cfg=CandidateStrategyConfig(),
            split_start=20,
            split_end=40,
        ),
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
            "exit_idx": [27, 32],
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
        schema=fit_candidate_feature_schema(
            labeled_events=labeled,
            cfg=CandidateStrategyConfig(gate_label_column="barrier_first_label"),
            split_start=20,
            split_end=40,
        ),
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
            "exit_idx": [27],
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
            schema=fit_candidate_feature_schema(
                labeled_events=labeled,
                cfg=CandidateStrategyConfig(gate_label_column="profitable_after_hurdle_label"),
                split_start=20,
                split_end=40,
            ),
            split_start=20,
            split_end=40,
        )


def test_build_candidate_dataset_builds_risk_unit_return_targets() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [26],
            "exit_idx": [27],
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
            "net_event_bps": [12.0],
            "risk_unit_bps": [25.0],
            "net_return_r": [0.48],
            "mae_r": [-0.24],
        }
    )

    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
        schema=fit_candidate_feature_schema(
            labeled_events=labeled,
            cfg=CandidateStrategyConfig(),
            split_start=20,
            split_end=40,
        ),
        split_start=20,
        split_end=40,
    )

    assert ds.y_return_r is not None
    assert ds.y_return_bps is not None
    assert ds.y_mae_r is not None
    assert ds.risk_unit_bps is not None
    assert ds.y_return_r[0] == pytest.approx(0.48)
    assert ds.y_return_bps[0] == pytest.approx(12.0)
    assert ds.y_mae_r[0] == pytest.approx(-0.24)
    assert ds.risk_unit_bps[0] == pytest.approx(25.0)
    assert ds.y_q10_bps.tolist() == [12.0]
    assert ds.y_mfe_bps.tolist() == [12.0]


def test_build_candidate_dataset_prefers_gross_targets_over_legacy_net_targets() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [26],
            "exit_idx": [27],
            "raw_score": [0.5],
            "score_z": [1.2],
            "turnover_proxy": [0.1],
            "triple_barrier_label": [1],
            "profitable_after_hurdle_label": [1],
            "gross_event_bps": [25.0],
            "gross_return_r": [1.0],
            "net_event_bps": [5.0],
            "edge_after_hurdle_bps": [5.0],
            "mae_bps": [-6.0],
            "mfe_bps": [18.0],
            "ex_ante_cost_bps": [4.0],
            "risk_unit_bps": [25.0],
        }
    )

    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
        schema=fit_candidate_feature_schema(
            labeled_events=labeled,
            cfg=CandidateStrategyConfig(),
            split_start=20,
            split_end=40,
        ),
        split_start=20,
        split_end=40,
    )

    assert ds.y_return_bps is not None
    assert ds.y_return_r is not None
    assert ds.y_return_bps.tolist() == [25.0]
    assert ds.y_return_r.tolist() == [1.0]
    assert ds.y_gross_return_bps is not None
    assert ds.y_gross_return_r is not None
    assert ds.y_gross_return_bps.tolist() == [25.0]
    assert ds.y_gross_return_r.tolist() == [1.0]


def test_build_candidate_dataset_supports_label_free_inference_without_exit_idx() -> None:
    aligned = _make_aligned()
    unlabeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25], aligned.datetimes[30]],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": [1, -1],
            "entry_idx": [26, 31],
            "raw_score": [0.5, -0.4],
            "score_z": [1.2, -0.8],
            "turnover_proxy": [0.1, 0.2],
            "expected_holding_bars": [3, 5],
            "min_holding_bars": [1, 1],
            "stop_atr_mult": [2.0, 2.0],
            "take_profit_atr_mult": [4.0, 4.0],
            "family": ["trend_ma", "trend_ma"],
            "variant": ["fast", "slow"],
            "archetype": ["trend_continuation", "trend_continuation"],
            "ex_ante_cost_bps": [4.0, 4.0],
        }
    )
    cfg = CandidateStrategyConfig()
    schema = fit_candidate_feature_schema(
        labeled_events=unlabeled,
        cfg=cfg,
        split_start=20,
        split_end=40,
    )

    ds = build_candidate_dataset(
        labeled_events=unlabeled,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=20,
        split_end=40,
        require_label_within_split=False,
    )

    assert ds.X.shape[0] == 2
    assert ds.y_gate.tolist() == [0, 0]
    assert ds.y_return_bps is not None
    assert ds.y_return_bps.tolist() == [0.0, 0.0]
    assert ds.y_gross_return_bps is not None
    assert ds.y_gross_return_bps.tolist() == [0.0, 0.0]
    assert ds.groups.tolist() == [25, 30]
    assert ds.effective_sample_size > 0.0


def test_build_candidate_dataset_imputes_missing_values_without_dropping_rows() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25], aligned.datetimes[26]],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": [1, -1],
            "entry_idx": [26, 27],
            "exit_idx": [27, 28],
            "raw_score": [0.5, np.nan],
            "score_z": [1.0, np.nan],
            "turnover_proxy": [0.1, np.nan],
            "triple_barrier_label": [1, 0],
            "profitable_after_hurdle_label": [1, 0],
            "edge_after_hurdle_bps": [12.0, -5.0],
            "mae_bps": [-6.0, -8.0],
            "mfe_bps": [18.0, 5.0],
            "ex_ante_cost_bps": [4.0, 4.0],
        }
    )

    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
        schema=fit_candidate_feature_schema(
            labeled_events=labeled,
            cfg=CandidateStrategyConfig(),
            split_start=20,
            split_end=40,
        ),
        split_start=20,
        split_end=40,
    )

    assert ds.X.shape[0] == 2
    assert np.isfinite(ds.X).all()


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
            "exit_idx": [27, 47],
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

    cfg = CandidateStrategyConfig()
    schema = fit_candidate_feature_schema(
        labeled_events=labeled,
        cfg=cfg,
        split_start=20,
        split_end=40,
    )

    train = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=20,
        split_end=40,
    )
    valid = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=40,
        split_end=60,
    )

    assert train.feature_names == valid.feature_names
    assert train.feature_schema_version == "candidate_v6"
    assert valid.feature_schema_version == "candidate_v6"
    assert "family=trend_ma" in train.feature_names
    assert "variant=trend_donchian:donchian_36" not in train.feature_names


def test_build_candidate_dataset_includes_universe_metadata_features() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [26],
            "exit_idx": [27],
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

    cfg = CandidateStrategyConfig(static_universe_features_enabled=True)
    schema = fit_candidate_feature_schema(
        labeled_events=labeled,
        cfg=cfg,
        split_start=20,
        split_end=40,
    )
    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
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


def test_build_candidate_dataset_preserves_realized_diagnostics_only_in_event_index() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [26],
            "exit_idx": [27],
            "raw_score": [0.5],
            "score_z": [1.2],
            "turnover_proxy": [0.1],
            "triple_barrier_label": [1],
            "profitable_after_hurdle_label": [1],
            "edge_after_hurdle_bps": [12.0],
            "mae_bps": [-6.0],
            "mfe_bps": [18.0],
            "ex_ante_cost_bps": [4.0],
            "exit_reason": ["take_profit"],
        }
    )

    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=CandidateStrategyConfig(),
        schema=fit_candidate_feature_schema(
            labeled_events=labeled,
            cfg=CandidateStrategyConfig(),
            split_start=20,
            split_end=40,
        ),
        split_start=20,
        split_end=40,
    )

    assert "exit_reason" in ds.event_index.columns
    assert ds.event_index.loc[0, "exit_reason"] == "take_profit"
    assert "exit_reason" not in ds.feature_names


def test_signal_prequalify_excludes_negative_edge_variants() -> None:
    """Layer 0: is_fit_split=True zeros out edge_weight for variants with IS mean_edge < 0."""
    aligned = _make_aligned()
    # Two variants: "good" variant with positive edge, "bad" variant with negative edge
    labeled = pd.DataFrame(
        {
            "datetime": [
                aligned.datetimes[25], aligned.datetimes[26],
                aligned.datetimes[27], aligned.datetimes[28],
                aligned.datetimes[29], aligned.datetimes[30],
            ],
            "symbol": ["BTCUSDT"] * 6,
            "family": ["trend_ma"] * 3 + ["trend_ma"] * 3,
            "variant": ["good_v"] * 3 + ["bad_v"] * 3,
            "side": [1] * 6,
            "entry_idx": [26, 27, 28, 29, 30, 31],
            "exit_idx": [27, 28, 29, 30, 31, 32],
            "raw_score": [0.5] * 6,
            "score_z": [1.0] * 6,
            "turnover_proxy": [0.1] * 6,
            "profitable_after_hurdle_label": [1, 1, 1, 0, 0, 0],
            "edge_after_hurdle_bps": [20.0, 25.0, 30.0, -10.0, -15.0, -20.0],
            "sl_thr_bps": [25.0] * 6,
            "mae_bps": [-5.0] * 6,
            "mfe_bps": [20.0] * 6,
            "ex_ante_cost_bps": [4.0] * 6,
        }
    )
    cfg = CandidateStrategyConfig(signal_prequalify_min_obs=2)
    schema = fit_candidate_feature_schema(
        labeled_events=labeled, cfg=cfg, split_start=20, split_end=40
    )

    # Arrange: is_fit_split=True should zero out bad_v
    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=20,
        split_end=40,
        is_fit_split=True,
    )

    # Assert: bad_v events have edge_weight=0, good_v events have edge_weight>0
    event_idx = ds.event_index
    good_mask = event_idx["variant"] == "good_v"
    bad_mask = event_idx["variant"] == "bad_v"
    assert ds.edge_weight[good_mask.to_numpy(dtype=bool)].sum() > 0.0
    assert ds.edge_weight[bad_mask.to_numpy(dtype=bool)].sum() == pytest.approx(0.0)


def test_signal_prequalify_not_applied_when_is_fit_split_false() -> None:
    """Non-fit splits must not apply signal pre-qualification."""
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25], aligned.datetimes[26]],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "family": ["trend_ma", "trend_ma"],
            "variant": ["bad_v", "bad_v"],
            "side": [1, 1],
            "entry_idx": [26, 27],
            "exit_idx": [27, 28],
            "raw_score": [0.5, 0.5],
            "score_z": [1.0, 1.0],
            "turnover_proxy": [0.1, 0.1],
            "profitable_after_hurdle_label": [0, 0],
            "edge_after_hurdle_bps": [-10.0, -20.0],
            "sl_thr_bps": [25.0, 25.0],
            "mae_bps": [-5.0, -5.0],
            "mfe_bps": [5.0, 5.0],
            "ex_ante_cost_bps": [4.0, 4.0],
        }
    )
    cfg = CandidateStrategyConfig(signal_prequalify_min_obs=2)
    schema = fit_candidate_feature_schema(
        labeled_events=labeled, cfg=cfg, split_start=20, split_end=40
    )

    # Act: is_fit_split=False (default) — bad_v should retain its weight
    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=20,
        split_end=40,
        is_fit_split=False,
    )

    # Assert: edge_weight is not zeroed out even though mean_edge < 0
    assert ds.edge_weight.sum() > 0.0


def test_prequalify_bootstrap_disqualifies_insignificant_variant() -> None:
    aligned = _make_aligned()
    n_events = 33
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25 + idx] for idx in range(n_events)],
            "symbol": ["BTCUSDT"] * n_events,
            "family": ["trend_ma"] * n_events,
            "variant": ["noisy_v"] * n_events,
            "side": [1] * n_events,
            "entry_idx": np.arange(26, 26 + n_events),
            "exit_idx": np.arange(27, 27 + n_events),
            "expected_holding_bars": [3] * n_events,
            "raw_score": [0.5] * n_events,
            "score_z": [1.0] * n_events,
            "turnover_proxy": [0.1] * n_events,
            "profitable_after_hurdle_label": [1] * n_events,
            "edge_after_hurdle_bps": ([1.0, -1.0] * 16) + [0.0],
            "sl_thr_bps": [25.0] * n_events,
            "mae_bps": [-5.0] * n_events,
            "mfe_bps": [5.0] * n_events,
            "ex_ante_cost_bps": [4.0] * n_events,
        }
    )
    cfg = CandidateStrategyConfig(
        signal_prequalify_method="block_bootstrap",
        signal_prequalify_min_obs=30,
        signal_prequalify_min_tstat=3.0,
        signal_prequalify_bootstrap_n=100,
        seed=19,
    )
    schema = fit_candidate_feature_schema(
        labeled_events=labeled,
        cfg=cfg,
        split_start=20,
        split_end=60,
    )

    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=20,
        split_end=60,
        is_fit_split=True,
    )

    assert ds.X.shape[0] == n_events
    assert ds.edge_weight.sum() == pytest.approx(0.0)


def test_prequalify_mean_keeps_positive_mean_variant_without_tstat_gate() -> None:
    aligned = _make_aligned()
    n_events = 30
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25 + idx] for idx in range(n_events)],
            "symbol": ["BTCUSDT"] * n_events,
            "family": ["trend_ma"] * n_events,
            "variant": ["legacy_mean_v"] * n_events,
            "side": [1] * n_events,
            "entry_idx": np.arange(26, 26 + n_events),
            "exit_idx": np.arange(27, 27 + n_events),
            "expected_holding_bars": [3] * n_events,
            "raw_score": [0.5] * n_events,
            "score_z": [1.0] * n_events,
            "turnover_proxy": [0.1] * n_events,
            "profitable_after_hurdle_label": [1] * n_events,
            "edge_after_hurdle_bps": ([5.0, -4.0] * 15),
            "sl_thr_bps": [25.0] * n_events,
            "mae_bps": [-5.0] * n_events,
            "mfe_bps": [5.0] * n_events,
            "ex_ante_cost_bps": [4.0] * n_events,
        }
    )
    cfg = CandidateStrategyConfig(
        signal_prequalify_method="mean",
        signal_prequalify_min_obs=30,
        signal_prequalify_min_tstat=100.0,
        seed=23,
    )
    schema = fit_candidate_feature_schema(
        labeled_events=labeled,
        cfg=cfg,
        split_start=20,
        split_end=60,
    )

    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=20,
        split_end=60,
        is_fit_split=True,
    )

    assert ds.X.shape[0] == n_events
    assert ds.edge_weight.sum() > 0.0


def test_regime_code_from_entry_minus_one() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25]],
            "symbol": ["BTCUSDT"],
            "family": ["trend_ma"],
            "variant": ["ema_12_72"],
            "side": [1],
            "entry_idx": [26],
            "exit_idx": [27],
            "expected_holding_bars": [3],
            "raw_score": [0.5],
            "score_z": [1.2],
            "turnover_proxy": [0.1],
            "profitable_after_hurdle_label": [1],
            "edge_after_hurdle_bps": [12.0],
            "sl_thr_bps": [25.0],
            "mae_bps": [-6.0],
            "mfe_bps": [18.0],
            "ex_ante_cost_bps": [4.0],
        }
    )
    cfg = CandidateStrategyConfig()
    schema = fit_candidate_feature_schema(
        labeled_events=labeled,
        cfg=cfg,
        split_start=20,
        split_end=40,
    )

    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=20,
        split_end=40,
    )
    regime = compute_market_regime_context(aligned=aligned)

    assert int(ds.event_index.loc[0, "entry_regime_code"]) == int(regime.code_1d[25])
    assert ds.event_index.loc[0, "entry_regime"] == regime.name_by_code[int(regime.code_1d[25])]


# ─── Signal Context Feature Tests (candidate_v6) ─────────────────────────────


def _make_labeled_with_variants(aligned: AlignedMarketData) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25], aligned.datetimes[30], aligned.datetimes[35]],
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT"],
            "side": [1, -1, 1],
            "entry_idx": [26, 31, 36],
            "exit_idx": [27, 32, 37],
            "raw_score": [0.5, -0.4, 0.8],
            "score_z": [1.2, -0.8, 1.5],
            "turnover_proxy": [0.1, 0.2, 0.1],
            "triple_barrier_label": [1, 0, 1],
            "profitable_after_hurdle_label": [1, 0, 1],
            "edge_after_hurdle_bps": [12.0, -5.0, 20.0],
            "sl_thr_bps": [25.0, 30.0, 25.0],
            "mae_bps": [-6.0, -8.0, -4.0],
            "mfe_bps": [18.0, 5.0, 25.0],
            "ex_ante_cost_bps": [4.0, 4.0, 4.0],
            "family": ["trend_ma", "mean_rev", "trend_ma"],
            "variant": ["ema_12_72", "rsi_14", "ema_12_72"],
            "archetype": ["trend_continuation", "mean_reversion", "trend_continuation"],
        }
    )


def test_signal_context_features_present_in_schema() -> None:
    aligned = _make_aligned()
    labeled = _make_labeled_with_variants(aligned)
    cfg = CandidateStrategyConfig(signal_context_features_enabled=True)
    schema = fit_candidate_feature_schema(
        labeled_events=labeled, cfg=cfg, split_start=20, split_end=40
    )
    ctx_features = {
        "overlay_mult_entry",
        "crisis_active_entry",
        "funding_side_alignment",
        "score_pct_variant_hist_90d",
        "archetype_regime_match",
        "n_same_dir_variants_log",
    }
    assert ctx_features.issubset(set(schema.feature_names))


def test_signal_context_features_absent_when_disabled() -> None:
    aligned = _make_aligned()
    labeled = _make_labeled_with_variants(aligned)
    cfg = CandidateStrategyConfig(signal_context_features_enabled=False)
    schema = fit_candidate_feature_schema(
        labeled_events=labeled, cfg=cfg, split_start=20, split_end=40
    )
    ctx_features = {
        "overlay_mult_entry",
        "crisis_active_entry",
        "funding_side_alignment",
        "score_pct_variant_hist_90d",
        "archetype_regime_match",
        "n_same_dir_variants_log",
    }
    assert ctx_features.isdisjoint(set(schema.feature_names))


def test_score_pct_variant_hist_causal_monotone() -> None:
    from src.domain.futures.strategy.candidate_dataset import _compute_score_pct_variant_hist

    events = pd.DataFrame(
        {
            "entry_idx": [10, 20, 30, 40, 50, 60],
            "family": ["trend_ma"] * 6,
            "variant": ["ema_12"] * 6,
            "raw_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    result = _compute_score_pct_variant_hist(events, window_bars=1000)

    # First 4 events should be 0.5 (insufficient history < 5)
    assert result[0] == pytest.approx(0.5)
    assert result[4] == pytest.approx(0.5)
    # 6th event (index 5) has 5 prior events with lower scores → pct = 1.0
    assert result[5] == pytest.approx(1.0)


def test_funding_side_alignment_direction() -> None:
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25], aligned.datetimes[30]],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": [1, -1],
            "entry_idx": [26, 31],
            "exit_idx": [27, 32],
            "raw_score": [0.5, -0.4],
            "score_z": [1.2, -0.8],
            "turnover_proxy": [0.1, 0.2],
            "triple_barrier_label": [1, 0],
            "profitable_after_hurdle_label": [1, 0],
            "edge_after_hurdle_bps": [12.0, -5.0],
            "sl_thr_bps": [25.0, 30.0],
            "mae_bps": [-6.0, -8.0],
            "mfe_bps": [18.0, 5.0],
            "ex_ante_cost_bps": [4.0, 4.0],
        }
    )

    from src.domain.futures.strategy.common.alignment import AlignedMarketData

    t = 60
    n = 2
    # Increasing funding: rolling z-score at entry bars (25, 30) will be positive.
    funding_pos = np.zeros((t, n), dtype=np.float64)
    funding_pos[:, 0] = np.linspace(0.0, 0.1, t)
    funding_pos[:, 1] = np.linspace(0.0, 0.1, t)
    x = np.linspace(100.0, 160.0, t, dtype=np.float64)
    close = np.stack([x, x * 1.02], axis=1)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    aligned_pos_funding = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=funding_pos,
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), 4.0, dtype=np.float64),
    )
    cfg = CandidateStrategyConfig(signal_context_features_enabled=True)
    schema = fit_candidate_feature_schema(
        labeled_events=labeled, cfg=cfg, split_start=20, split_end=40
    )
    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned_pos_funding,
        cfg=cfg,
        schema=schema,
        split_start=20,
        split_end=40,
    )
    fsa_idx = ds.feature_names.index("funding_side_alignment")
    # long (side=1) with positive funding → tanh(positive * 1) = positive alignment
    # short (side=-1) with positive funding → tanh(positive * -1) = negative alignment
    assert ds.X[0, fsa_idx] > 0.0
    assert ds.X[1, fsa_idx] < 0.0


def test_archetype_regime_match_lookup() -> None:
    from src.domain.futures.strategy.candidate_dataset import _ARCHETYPE_REGIME_AFFINITY

    assert _ARCHETYPE_REGIME_AFFINITY[("trend_continuation", "bull_quiet")] == pytest.approx(1.0)
    assert _ARCHETYPE_REGIME_AFFINITY[("mean_reversion", "bull_volatile")] == pytest.approx(0.8)
    assert _ARCHETYPE_REGIME_AFFINITY[("trend_continuation", "crash")] == pytest.approx(-1.0)
    assert _ARCHETYPE_REGIME_AFFINITY[("position_unwind", "crash")] == pytest.approx(1.0)


def test_n_same_dir_variants_log_confluence() -> None:
    aligned = _make_aligned()
    # 3 events at same entry_idx=26, same symbol, same side → confluence 3
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25]] * 3,
            "symbol": ["BTCUSDT"] * 3,
            "side": [1, 1, 1],
            "entry_idx": [26, 26, 26],
            "exit_idx": [27, 27, 27],
            "raw_score": [0.5, 0.6, 0.7],
            "score_z": [1.2, 1.3, 1.4],
            "turnover_proxy": [0.1, 0.1, 0.1],
            "triple_barrier_label": [1, 1, 1],
            "profitable_after_hurdle_label": [1, 1, 1],
            "edge_after_hurdle_bps": [12.0, 14.0, 16.0],
            "sl_thr_bps": [25.0, 25.0, 25.0],
            "mae_bps": [-6.0, -6.0, -6.0],
            "mfe_bps": [18.0, 18.0, 18.0],
            "ex_ante_cost_bps": [4.0, 4.0, 4.0],
            "family": ["trend_ma", "dual_momentum", "trend_donchian"],
            "variant": ["ema_12_72", "dm_12_48", "donchian_36"],
            "archetype": ["trend_continuation", "time_series_momentum", "trend_continuation"],
        }
    )
    cfg = CandidateStrategyConfig(signal_context_features_enabled=True)
    schema = fit_candidate_feature_schema(
        labeled_events=labeled, cfg=cfg, split_start=20, split_end=40
    )
    ds = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=20,
        split_end=40,
    )
    import math

    n_idx = ds.feature_names.index("n_same_dir_variants_log")
    # 3 events in the same group → log1p(3-1) = log1p(2) ≈ 1.099
    assert ds.X[0, n_idx] == pytest.approx(math.log1p(2), rel=1e-4)


def test_build_candidate_dataset_features_cached() -> None:
    from src.domain.futures.strategy.candidate_dataset import _ALIGNED_FEATURE_CACHE
    aligned = _make_aligned()
    labeled = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[25]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [26],
            "exit_idx": [27],
            "raw_score": [0.5],
            "score_z": [1.2],
            "turnover_proxy": [0.1],
            "triple_barrier_label": [1],
            "profitable_after_hurdle_label": [1],
            "edge_after_hurdle_bps": [12.0],
            "sl_thr_bps": [25.0],
            "mae_bps": [-6.0],
            "mfe_bps": [18.0],
            "ex_ante_cost_bps": [4.0],
            "family": ["trend_ma"],
            "variant": ["ema_12_72"],
            "archetype": ["trend_continuation"],
        }
    )
    cfg = CandidateStrategyConfig(
        signal_context_features_enabled=True,
        exclude_immediate_return_features=False,
    )
    schema = fit_candidate_feature_schema(
        labeled_events=labeled, cfg=cfg, split_start=20, split_end=40
    )

    # 캐시 비우기
    aligned_id = id(aligned)
    if aligned_id in _ALIGNED_FEATURE_CACHE:
        del _ALIGNED_FEATURE_CACHE[aligned_id]

    # Act 1 (최초 계산)
    _ = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=20,
        split_end=40,
    )
    assert aligned_id in _ALIGNED_FEATURE_CACHE
    assert "sym_ret_1" in _ALIGNED_FEATURE_CACHE[aligned_id]
    assert "overlay_ctx" in _ALIGNED_FEATURE_CACHE[aligned_id]
    assert "regime_ctx" in _ALIGNED_FEATURE_CACHE[aligned_id]

    # 캐시 값을 수정하여 캐시가 사용되는지 확인
    cached_ret = _ALIGNED_FEATURE_CACHE[aligned_id]["sym_ret_1"]
    _ALIGNED_FEATURE_CACHE[aligned_id]["sym_ret_1"] = np.ones_like(cached_ret) * 999.0

    # Act 2 (캐시 재사용)
    ds2 = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=20,
        split_end=40,
    )

    # Assert
    idx = ds2.feature_names.index("sym_ret_1")
    assert ds2.X[0, idx] == pytest.approx(999.0)


def test_compute_bootstrap_means_numba() -> None:
    from src.domain.futures.strategy.candidate_dataset import _compute_bootstrap_means_numba
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    # 2개의 bootstrap 샘플, 각 샘플당 3개의 블록 시작 인덱스
    # block=2 로 테스트
    start_idxs = np.array([
        [0, 2, 4],  # 블록들: [1,2], [3,4], [5] -> 복사 후 sx=[1,2,3,4,5]
        [1, 3, 0]   # 블록들: [2,3], [4,5], [1,2] -> 복사 후 sx=[2,3,4,5,1]
    ], dtype=np.int64)

    means = _compute_bootstrap_means_numba(x, w, start_idxs, block=2)
    assert means.shape == (2,)
    assert means[0] == pytest.approx(3.0)
    assert means[1] == pytest.approx(3.0)


def test_compute_uniqueness_weights_numba() -> None:
    from src.domain.futures.strategy.candidate_dataset import _compute_uniqueness_weights_numba
    starts = np.array([0, 1, 2], dtype=np.int64)
    ends = np.array([1, 2, 3], dtype=np.int64)
    inv_active = np.array([1.0, 0.5, 0.333, 0.25], dtype=np.float64)

    weights = _compute_uniqueness_weights_numba(starts, ends, inv_active)
    assert weights.shape == (3,)
    assert weights[0] == pytest.approx(0.75, rel=1e-3)
    assert weights[1] == pytest.approx(0.4165, rel=1e-3)
