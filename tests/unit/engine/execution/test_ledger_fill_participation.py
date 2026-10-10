"""Fill bar quote-volume evidence for capacity evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import dataclasses

from src.engine.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
from src.engine.execution import fill_terms as fill_terms_mod


def _window(grid: pd.DatetimeIndex, *, quote: float) -> ExecutionReplayWindow:
    px = pd.DataFrame({"BTCUSDT": np.full(len(grid), 100.0)}, index=grid)
    qv = pd.DataFrame({"BTCUSDT": np.full(len(grid), quote)}, index=grid)
    index = pd.DatetimeIndex([grid[0]])
    weights = pd.DataFrame({"BTCUSDT": [0.1]}, index=index)
    return ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=("BTCUSDT",), symbols=("BTCUSDT",),
        minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
        target_weights=weights, signal_available_at=index, quote_volumes=qv,
        funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(minutes=3),
    )


def test_fill_carries_bar_quote_volume() -> None:
    """A fill on a 5 000-quote bar records 5 000; other fill columns are unchanged."""
    grid = pd.date_range("2025-01-01 00:00", "2025-01-01 02:00", freq="3min", tz="UTC")
    result = replay_execution_windows((_window(grid, quote=5000.0),), 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    fills = result.simulated_fills
    assert len(fills) == 1
    assert fills["bar_quote_volume"].iloc[0] == 5000.0
    assert fills["quantity_delta"].iloc[0] == 1.0
    assert fills["fill_price"].iloc[0] == 100.0
    assert str(fills["bar_quote_volume"].dtype) == "float64"


def test_degenerate_quote_volume_maps_to_nan_never_inf() -> None:
    """Zero, unknown, negative, and infinite volumes all record NaN."""
    track: list[float] = []
    for quote in (0.0, float("nan"), -5.0, float("inf")):
        fill_terms_mod.book_fill_quote_volume(
            track, ["BTCUSDT"], np.array([[quote]]), 0, "BTCUSDT",
        )
    assert len(track) == 4
    assert all(isinstance(v, float) and np.isnan(v) for v in track)


def test_unfilled_run_keeps_typed_quote_volume_column() -> None:
    """A zero-volume bar books no fill, but the column stays float64."""
    grid = pd.date_range("2025-01-01 00:00", "2025-01-01 02:00", freq="3min", tz="UTC")
    result = replay_execution_windows((_window(grid, quote=0.0),), 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.simulated_fills.empty
    assert str(result.simulated_fills["bar_quote_volume"].dtype) == "float64"


def test_unrecorded_volume_is_not_legacy_tradability_default() -> None:
    grid = pd.date_range("2025-01-01 00:00", "2025-01-01 02:00", freq="3min", tz="UTC")
    window = _window(grid, quote=5000.0)
    baseline = replay_execution_windows((window,), 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    for quote in (None, pd.DataFrame(index=grid)):
        result = replay_execution_windows(
            (dataclasses.replace(window, quote_volumes=quote),), 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
        )
        assert result.simulated_fills["bar_quote_volume"].isna().all()
        pd.testing.assert_frame_equal(
            result.simulated_fills.drop(columns="bar_quote_volume"),
            baseline.simulated_fills.drop(columns="bar_quote_volume"),
        )
        pd.testing.assert_series_equal(result.ledger.equity, baseline.ledger.equity)
