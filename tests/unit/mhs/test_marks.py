"""Tests for the MHS observed-funding / PIT-roster / minute-frame loading module."""

from __future__ import annotations

import pandas as pd

from src.mhs import marks


def test_pit_execution_mask_entry_and_exit_hysteresis() -> None:
    """A member kept past the entry rank is dropped once it exits the exit band."""
    idx = pd.date_range("2022-01-01", periods=730, freq="h", tz="UTC")
    symbols = ["A", "B"]
    volume = pd.DataFrame(1.0, index=idx, columns=symbols)
    # A stays top-ranked throughout; B decays under the exit band on the last bar
    # only, so hysteresis should keep it held through most of the tail.
    volume.loc[idx[-1], "B"] = 0.0
    eligible = pd.DataFrame(True, index=idx, columns=symbols)

    mask = marks._pit_execution_mask(volume, eligible, universe_size=1)

    assert mask["A"].iloc[-1]
    assert mask.index.equals(idx)
    assert list(mask.columns) == symbols


def test_pit_execution_mask_no_eligible_never_holds() -> None:
    """An always-ineligible symbol is never selected regardless of volume."""
    idx = pd.date_range("2022-01-01", periods=730, freq="h", tz="UTC")
    volume = pd.DataFrame({"A": 100.0}, index=idx)
    eligible = pd.DataFrame({"A": False}, index=idx)

    mask = marks._pit_execution_mask(volume, eligible, universe_size=1)

    assert not mask["A"].any()


def test_replay_window_reads_no_historical_mark(tmp_path, monkeypatch) -> None:
    """A replay window load opens no Mark file."""
    import src.market_data.services.futures_collection as fc

    def _boom(symbol: str, timeframe: str):
        raise AssertionError("no Mark file may be read")

    monkeypatch.setattr(fc, "_mark_price_path", _boom)
    assert not hasattr(marks, "_contemporaneous_mark_close_panel")
    assert not hasattr(marks, "_fill_mark_parity_eligibility")
    grid = pd.date_range("2022-01-01", periods=4, freq="3min", tz="UTC")
    frames = marks._load_window_minute_frames(str(tmp_path), ["A"], grid[0], grid[-1], "3m")
    assert frames == {}


def test_replay_window_ohlcv_valuation_identity(tmp_path) -> None:
    """A completed 3m close emits marks None under OHLCV close valuation."""
    from src.mhs.execution.window_stream import _materialize_execution_piece
    from src.mhs.resources import MhsExecutionAllocation

    grid = pd.date_range("2022-01-01", periods=4, freq="3min", tz="UTC")
    ts_ms = [int(ts.value // 1_000_000) for ts in grid]
    frame = pd.DataFrame(
        {"timestamp": ts_ms, "high": 101.0, "low": 99.0, "close": 100.0, "quote_vol": 10.0},
    )
    root = tmp_path / "mkt"
    (root / "3m").mkdir(parents=True)
    frame.to_parquet(root / "3m" / "A.parquet")
    weights = pd.DataFrame(0.0, index=pd.DatetimeIndex([grid[0]]), columns=["A"])
    window = _materialize_execution_piece(
        piece_grid=grid,
        piece_weights=weights,
        piece_signals=pd.DatetimeIndex([grid[0]]),
        roster=["A"],
        columns=("A",),
        root=str(root),
        timeframe="3m",
        funding_by_symbol={},
        funding_failures=None,
        allocation=MhsExecutionAllocation(fixed_bytes=1, bytes_per_bar=1, decoder_bytes=1),
        budget_bytes=None,
        reserve_bytes=None,
        window_start=grid[0],
        window_end=grid[-1] + pd.Timedelta(minutes=3),
        logical_partition=(0, 1),
    )
    assert window.marks is None
    assert float(window.closes["A"].iloc[0]) == 100.0


def test_replay_window_missing_trade_close_invalid(tmp_path) -> None:
    """An unavailable 3m bar stays NaN under the existing coverage gate."""
    from src.mhs.execution.window_stream import _materialize_execution_piece
    from src.mhs.resources import MhsExecutionAllocation

    grid = pd.date_range("2022-01-01", periods=4, freq="3min", tz="UTC")
    weights = pd.DataFrame(0.0, index=pd.DatetimeIndex([grid[0]]), columns=["A"])
    window = _materialize_execution_piece(
        piece_grid=grid,
        piece_weights=weights,
        piece_signals=pd.DatetimeIndex([grid[0]]),
        roster=["A"],
        columns=("A",),
        root=str(tmp_path / "empty"),
        timeframe="3m",
        funding_by_symbol={},
        funding_failures=None,
        allocation=MhsExecutionAllocation(fixed_bytes=1, bytes_per_bar=1, decoder_bytes=1),
        budget_bytes=None,
        reserve_bytes=None,
        window_start=grid[0],
        window_end=grid[-1] + pd.Timedelta(minutes=3),
        logical_partition=(0, 1),
    )
    assert window.marks is None
    assert window.closes["A"].isna().all()


def test_align_minute_frames_empty_returns_none() -> None:
    """An empty frame dict yields None rather than an empty aligned panel."""
    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2022-01-01T01:00:00", tz="UTC")

    assert marks._align_minute_frames({}, "3m", start, end) is None


def test_align_minute_frames_degenerate_window_returns_none() -> None:
    """start >= end yields None even with non-empty frames."""
    idx = pd.date_range("2022-01-01", periods=3, freq="3min", tz="UTC")
    frame = pd.DataFrame({"high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    same = pd.Timestamp("2022-01-01", tz="UTC")

    assert marks._align_minute_frames({"A": frame}, "3m", same, same) is None


def test_align_minute_frames_reindexes_to_requested_grid() -> None:
    """The aligned high/low/close panels use the requested grid, not the source index."""
    idx = pd.date_range("2022-01-01", periods=5, freq="3min", tz="UTC")
    frame = pd.DataFrame(
        {"high": 2.0, "low": 1.0, "close": 1.5}, index=idx,
    )
    start = idx[0]
    end = idx[-1]

    result = marks._align_minute_frames({"A": frame}, "3m", start, end)

    assert result is not None
    highs, lows, closes = result
    assert highs.index.equals(pd.date_range(start, end, freq="3min", tz="UTC"))
    assert (highs["A"] == 2.0).all()
    assert (lows["A"] == 1.0).all()
    assert (closes["A"] == 1.5).all()


def test_build_window_frames_no_roster_data_returns_none() -> None:
    """A roster whose symbols are absent from symbol_frames yields None."""
    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2022-01-01T00:05:00", tz="UTC")
    grid = pd.date_range(start, end, freq="3min", tz="UTC")

    result = marks._build_window_frames({}, ["A"], start, end, grid, "3m")

    assert result is None


def test_build_window_frames_slices_and_reindexes() -> None:
    """Per-symbol frames are sliced to the window and reindexed onto the minute grid."""
    idx = pd.date_range("2022-01-01", periods=10, freq="3min", tz="UTC")
    frame = pd.DataFrame(
        {"high": 2.0, "low": 1.0, "close": 1.5}, index=idx,
    )
    start = idx[2]
    end = idx[5]
    grid = pd.date_range(start, end, freq="3min", tz="UTC")

    result = marks._build_window_frames(
        {"A": frame}, ["A"], start, end, grid, "3m",
    )

    assert result is not None
    highs, lows, closes = result
    assert highs.index.equals(grid)
    assert (highs["A"] == 2.0).all()
    assert (lows["A"] == 1.0).all()
    assert (closes["A"] == 1.5).all()


def test_load_funding_series_missing_and_loaded(tmp_path, monkeypatch) -> None:
    """Missing paths are dropped with reason 'missing'; loaded series pass through."""
    idx = pd.date_range("2022-01-01", periods=3, freq="h", tz="UTC")
    loaded = pd.Series([0.0001, 0.0002, 0.0003], index=idx)

    def fake_funding_path(symbol: str):
        return tmp_path / f"{symbol}.parquet"

    def fake_load_funding_rates(path: str) -> pd.Series:
        return loaded

    monkeypatch.setattr(marks, "funding_path", fake_funding_path)
    monkeypatch.setattr(marks, "load_funding_rates", fake_load_funding_rates)
    (tmp_path / "B.parquet").touch()

    series, dropped = marks._load_funding_series(["A", "B"])

    assert dropped["A"] == "missing"
    assert "B" in series
    assert series["B"].equals(loaded)


def test_clear_market_data_caches_keeps_retained_loaders_stateless(tmp_path, monkeypatch) -> None:
    """Retired mark caches are gone; retained loaders always read the lake file."""
    import pandas as pd

    assert not hasattr(marks, "_cached_mark_panel")
    assert not hasattr(marks, "_get_symbol_mark_frame")
    assert not hasattr(marks, "_compact_mark_series_for_path")

    frame = pd.DataFrame(
        {"timestamp": [1640995200000], "high": [1.0], "low": [1.0], "close": [100.0]},
    )
    path = tmp_path / "AUSDT.parquet"
    frame.to_parquet(path)
    sym, first = marks._load_symbol_minute_frame(
        str(path), "AUSDT", 0, 4102444800000,
        pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-01-02", tz="UTC"),
    )
    assert first is not None
    assert float(first["close"].iloc[0]) == 100.0
    # Replacing the file is observed immediately with no cache to invalidate.
    frame.assign(close=[200.0]).to_parquet(path)
    marks.clear_mhs_market_data_caches()
    _, second = marks._load_symbol_minute_frame(
        str(path), "AUSDT", 0, 4102444800000,
        pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-01-02", tz="UTC"),
    )
    assert second is not None
    assert float(second["close"].iloc[0]) == 200.0
