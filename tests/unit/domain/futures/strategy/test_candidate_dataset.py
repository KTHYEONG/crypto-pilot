from __future__ import annotations

import numpy as np
import pandas as pd

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


def test_build_candidate_dataset_falls_back_to_triple_barrier_label() -> None:
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
        split_start=20,
        split_end=40,
    )

    assert ds.y_gate.tolist() == [1, 0]
