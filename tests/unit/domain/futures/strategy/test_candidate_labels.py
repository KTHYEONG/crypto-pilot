from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_labels import label_candidate_events
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig


def _make_aligned() -> AlignedMarketData:
    t = 30
    n = 1
    base = np.linspace(100.0, 130.0, t, dtype=np.float64).reshape(t, n)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=base.copy(),
        high_2d=base * 1.01,
        low_2d=base * 0.99,
        close_2d=base.copy(),
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), 5.0, dtype=np.float64),
    )


def test_label_candidate_events_t_plus_one_entry_and_columns() -> None:
    aligned = _make_aligned()
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[10]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [11],
            "expected_holding_bars": [5],
            "min_holding_bars": [1],
            "stop_atr_mult": [1.0],
            "take_profit_atr_mult": [1.0],
            "cost_floor_bps": [5.0],
        }
    )

    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())

    assert out.shape[0] == 1
    assert int(out.loc[0, "entry_idx"]) == 11
    for col in (
        "gross_fwd_bps",
        "ex_ante_cost_bps",
        "edge_after_hurdle_bps",
        "triple_barrier_label",
        "time_to_exit_bars",
        "mae_bps",
        "mfe_bps",
        "realized_vol_bps",
    ):
        assert col in out.columns
    assert np.isfinite(float(out.loc[0, "gross_fwd_bps"]))


def test_label_candidate_events_uses_future_window_only_for_targets() -> None:
    aligned = _make_aligned()
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[8]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [9],
            "expected_holding_bars": [4],
            "min_holding_bars": [1],
            "stop_atr_mult": [50.0],
            "take_profit_atr_mult": [50.0],
            "cost_floor_bps": [0.0],
        }
    )
    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())
    assert int(out.loc[0, "time_to_exit_bars"]) == 4
