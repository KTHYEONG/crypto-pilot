"""Tests for the MHS mark-price / minute-frame loading module."""

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


def test_fill_mark_parity_eligibility_disabled_returns_unchanged() -> None:
    """With enabled=False the eligibility mask passes through with no census."""
    idx = pd.date_range("2022-01-01", periods=5, freq="h", tz="UTC")
    close = pd.DataFrame({"A": 1.0}, index=idx)
    eligible = pd.DataFrame({"A": True}, index=idx)

    result_eligible, census = marks._fill_mark_parity_eligibility(
        close, eligible, enabled=False,
    )

    assert result_eligible.equals(eligible)
    assert census is None


def test_fill_mark_parity_eligibility_removes_diverged_cells() -> None:
    """Cells where mark diverges beyond the log-band are excluded, with a census."""
    idx = pd.date_range("2022-01-01", periods=3, freq="h", tz="UTC")
    close = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    mark_close = pd.DataFrame({"A": [1.0, 10.0, 1.0]}, index=idx)
    eligible = pd.DataFrame({"A": [True, True, True]}, index=idx)

    result_eligible, census = marks._fill_mark_parity_eligibility(
        close, eligible, enabled=True, mark_close=mark_close,
    )

    assert not result_eligible["A"].iloc[1]
    assert result_eligible["A"].iloc[0]
    assert result_eligible["A"].iloc[2]
    assert census is not None
    assert census["cells_over_band"] == 1
    assert census["eligible_cells_removed"] == 1
    assert "A" in census["symbols"]


def test_align_minute_frames_empty_returns_none() -> None:
    """An empty frame dict yields None rather than an empty aligned panel."""
    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2022-01-01T01:00:00", tz="UTC")

    assert marks._align_minute_frames({}, "1m", start, end) is None


def test_align_minute_frames_degenerate_window_returns_none() -> None:
    """start >= end yields None even with non-empty frames."""
    idx = pd.date_range("2022-01-01", periods=3, freq="1min", tz="UTC")
    frame = pd.DataFrame({"high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    same = pd.Timestamp("2022-01-01", tz="UTC")

    assert marks._align_minute_frames({"A": frame}, "1m", same, same) is None


def test_align_minute_frames_reindexes_to_requested_grid() -> None:
    """The aligned high/low/close panels use the requested grid, not the source index."""
    idx = pd.date_range("2022-01-01", periods=5, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {"high": 2.0, "low": 1.0, "close": 1.5}, index=idx,
    )
    start = idx[0]
    end = idx[-1]

    result = marks._align_minute_frames({"A": frame}, "1m", start, end)

    assert result is not None
    highs, lows, closes = result
    assert highs.index.equals(pd.date_range(start, end, freq="1min", tz="UTC"))
    assert (highs["A"] == 2.0).all()
    assert (lows["A"] == 1.0).all()
    assert (closes["A"] == 1.5).all()


def test_build_window_frames_no_roster_data_returns_none() -> None:
    """A roster whose symbols are absent from symbol_frames yields None."""
    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2022-01-01T00:05:00", tz="UTC")
    grid = pd.date_range(start, end, freq="1min", tz="UTC")

    result = marks._build_window_frames({}, ["A"], start, end, grid, "1m")

    assert result is None


def test_build_window_frames_slices_and_reindexes() -> None:
    """Per-symbol frames are sliced to the window and reindexed onto the minute grid."""
    idx = pd.date_range("2022-01-01", periods=10, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {"high": 2.0, "low": 1.0, "close": 1.5}, index=idx,
    )
    start = idx[2]
    end = idx[5]
    grid = pd.date_range(start, end, freq="1min", tz="UTC")

    result = marks._build_window_frames(
        {"A": frame}, ["A"], start, end, grid, "1m",
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
