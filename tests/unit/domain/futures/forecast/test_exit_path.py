from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from src.domain.futures.forecast.contracts import ExitPathRequest
from src.domain.futures.forecast.exit_path import label_exit_paths


def _request(*, high: float = 101.0, low: float = 99.0, stop: float = 1.0, target: float = 2.0) -> ExitPathRequest:
    open_ = np.array([[100.0], [100.0], [100.0], [100.0]], dtype=np.float64)
    high_2d = np.full((4, 1), high, dtype=np.float64)
    low_2d = np.full((4, 1), low, dtype=np.float64)
    return ExitPathRequest(
        decision_idx=np.array([0]), entry_idx=np.array([1]), side=np.array([1]),
        horizon_bars=np.array([2]), stop_atr_mult=np.array([stop]), target_atr_mult=np.array([target]),
        min_hold_bars=np.array([1]), symbol_idx=np.array([0]), open_2d=open_,
        high_2d=high_2d, low_2d=low_2d, close_2d=open_.copy(), atr_2d=np.ones((4, 1)),
        cost_bps_2d=np.zeros((4, 1)), funding_2d=np.zeros((4, 1)),
        cost_floor_bps=np.array([np.nan]), hurdle_bps=np.array([0.0]), taker_round_trip_bps=0.0,
    )


def test_label_exit_paths_long_short_and_time_exit() -> None:
    result = label_exit_paths(_request(high=101.0, low=99.5))
    assert result.exit_reason[0] == "time_exit"
    short = _request(high=100.5, low=99.5)
    short = replace(short, side=np.array([-1]))
    assert label_exit_paths(short).exit_reason[0] == "time_exit"


def test_label_exit_paths_is_next_open_causal() -> None:
    with pytest.raises(ValueError, match="entry_idx"):
        label_exit_paths(replace(_request(), entry_idx=np.array([0])))


def test_label_exit_paths_gap_and_same_bar_collision_are_conservative() -> None:
    gap = label_exit_paths(_request(high=101.0, low=90.0))
    assert gap.exit_reason[0] == "stop_loss"
    collision = label_exit_paths(_request(high=106.0, low=94.0, target=1.5))
    assert collision.exit_reason[0] == "stop_loss"
    assert int(collision.same_bar_collision[0]) == 1
