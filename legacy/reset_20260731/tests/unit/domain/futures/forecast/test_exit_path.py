from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from src.domain.futures.forecast.contracts import ExitPathRequest
from src.domain.futures.forecast.exit_path import (
    _label_kernel,
    _label_kernel_python,
    label_exit_paths,
)


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


def test_label_kernel_numba_python_equivalence() -> None:
    req = _request(high=105.0, low=95.0, stop=1.5, target=3.0)
    numba_result = _label_kernel(req)
    python_result = _label_kernel_python(req)
    for arr_n, arr_p in zip(numba_result, python_result, strict=True):
        assert arr_n.dtype == arr_p.dtype
        np.testing.assert_allclose(arr_n, arr_p, rtol=1e-12, atol=1e-12)


def test_label_kernel_zero_events_returns_empty() -> None:
    req = _request(high=105.0, low=95.0)
    zero_req = replace(req, entry_idx=np.array([], dtype=np.int64),
                       decision_idx=np.array([], dtype=np.int64),
                       side=np.array([], dtype=np.float64),
                       horizon_bars=np.array([], dtype=np.int64),
                       stop_atr_mult=np.array([], dtype=np.float64),
                       target_atr_mult=np.array([], dtype=np.float64),
                       min_hold_bars=np.array([], dtype=np.int64),
                       symbol_idx=np.array([], dtype=np.int64),
                       cost_floor_bps=np.array([], dtype=np.float64),
                       hurdle_bps=np.array([], dtype=np.float64))
    numba_result = _label_kernel(zero_req)
    python_result = _label_kernel_python(zero_req)
    for arr_n, arr_p in zip(numba_result, python_result, strict=True):
        assert arr_n.shape == (0,)
        assert arr_p.shape == (0,)
        assert arr_n.dtype == arr_p.dtype


def test_label_kernel_numba_multi_event_equivalence() -> None:
    n = 10
    ts = 20
    rng = np.random.default_rng(42)
    open_2d = 100.0 + rng.standard_normal((ts, 3)).cumsum(axis=0) * 0.5
    close_2d = open_2d + rng.standard_normal((ts, 3)) * 0.1
    high_2d = np.maximum(open_2d, close_2d) + rng.random((ts, 3)) * 2.0
    low_2d = np.minimum(open_2d, close_2d) - rng.random((ts, 3)) * 2.0
    req = ExitPathRequest(
        decision_idx=np.full(n, 0, dtype=np.int64),
        entry_idx=np.full(n, 1, dtype=np.int64),
        side=np.where(rng.random(n) > 0.5, 1.0, -1.0),
        horizon_bars=np.full(n, 5, dtype=np.int64),
        stop_atr_mult=np.full(n, 2.0, dtype=np.float64),
        target_atr_mult=np.full(n, 3.0, dtype=np.float64),
        min_hold_bars=np.full(n, 1, dtype=np.int64),
        symbol_idx=rng.integers(0, 3, size=n).astype(np.int64),
        open_2d=open_2d.astype(np.float64),
        high_2d=high_2d.astype(np.float64),
        low_2d=low_2d.astype(np.float64),
        close_2d=close_2d.astype(np.float64),
        atr_2d=np.full((ts, 3), 0.5, dtype=np.float64),
        cost_bps_2d=np.full((ts, 3), 10.0, dtype=np.float64),
        funding_2d=np.zeros((ts, 3), dtype=np.float64),
        cost_floor_bps=np.full(n, np.nan, dtype=np.float64),
        hurdle_bps=np.zeros(n, dtype=np.float64),
        taker_round_trip_bps=5.0,
    )
    numba_result = _label_kernel(req)
    python_result = _label_kernel_python(req)
    for arr_n, arr_p in zip(numba_result, python_result, strict=True):
        np.testing.assert_allclose(arr_n, arr_p, rtol=1e-12, atol=1e-12)
