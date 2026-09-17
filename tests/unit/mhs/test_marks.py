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


def test_clear_market_data_caches_reloads_replaced_file(monkeypatch) -> None:
    import pandas as pd
    import src.mhs.marks as marks
    state = {'close': 100.0}
    def frame():
        return pd.DataFrame({'datetime': [pd.Timestamp('2025-01-01', tz='UTC')], 'close': [state['close']]})
    monkeypatch.setattr(marks.DataCollector, '_load_mark_price_cache', staticmethod(lambda path: frame()))
    monkeypatch.setattr(marks._futures_collection, '_mark_price_path', lambda symbol, timeframe: '/tmp/fake.parquet')  # noqa: S108
    first = marks._get_symbol_mark_frame('BTCUSDT', '1h')['close'].iloc[0]
    state['close'] = 200.0
    marks.clear_mhs_market_data_caches()
    second = marks._get_symbol_mark_frame('BTCUSDT', '1h')['close'].iloc[0]
    assert (first, second) == (100.0, 200.0)


def test_compact_mark_series_missing_file_returns_typed_empty(tmp_path) -> None:
    """A missing source yields correctly typed empty arrays."""
    import numpy as np

    marks._compact_mark_series_for_path.cache_clear()
    avail, close = marks._compact_mark_series_for_path(
        "MISUSDT", "1h", str(tmp_path / "missing.parquet"),
    )
    assert avail.dtype == np.dtype("int64")
    assert close.dtype == np.dtype("float64")
    assert avail.size == 0
    assert close.size == 0
    marks._compact_mark_series_for_path.cache_clear()


def test_compact_mark_series_schema_without_close_raises(tmp_path) -> None:
    """A parquet without the close column fails closed."""
    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError

    pd.DataFrame({"timestamp": [1640995200000]}).to_parquet(tmp_path / "noclose.parquet")
    marks._compact_mark_series_for_path.cache_clear()
    with pytest.raises(DataIntegrityError):
        marks._compact_mark_series_for_path("NCUSDT", "1h", str(tmp_path / "noclose.parquet"))
    marks._compact_mark_series_for_path.cache_clear()


def test_compact_mark_series_corrupt_file_raises(tmp_path) -> None:
    """An unreadable parquet fails closed instead of returning empty."""
    import pytest

    from src.common.errors import DataIntegrityError

    (tmp_path / "corrupt.parquet").write_bytes(b"not a parquet file")
    marks._compact_mark_series_for_path.cache_clear()
    with pytest.raises(DataIntegrityError):
        marks._compact_mark_series_for_path("COUSDT", "1h", str(tmp_path / "corrupt.parquet"))
    marks._compact_mark_series_for_path.cache_clear()


def test_compact_mark_series_read_failure_raises(tmp_path, monkeypatch) -> None:
    """A schema-valid file that fails on read fails closed."""
    import pandas as pd
    import pytest

    import pyarrow.parquet as _pq

    from src.common.errors import DataIntegrityError

    stamps = pd.date_range("2022-01-01", periods=2, freq="1h", tz="UTC")
    epoch_ms = (stamps - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    pd.DataFrame({"timestamp": epoch_ms.to_numpy(), "close": [10.0, 20.0]}).to_parquet(
        tmp_path / "readfail.parquet",
    )
    real_read_table = _pq.read_table

    def _fail_first(path, *args, **kwargs):
        if str(path).endswith("readfail.parquet") and kwargs.get("columns"):
            raise RuntimeError("simulated read failure")
        return real_read_table(path, *args, **kwargs)

    monkeypatch.setattr(_pq, "read_table", _fail_first)
    marks._compact_mark_series_for_path.cache_clear()
    with pytest.raises(DataIntegrityError):
        marks._compact_mark_series_for_path("RFUSDT", "1h", str(tmp_path / "readfail.parquet"))
    marks._compact_mark_series_for_path.cache_clear()


def test_compact_mark_series_without_datetime_derives_availability(tmp_path) -> None:
    """A timestamp/close-only file maps availability to timestamp + 1h."""
    import numpy as np
    import pandas as pd

    stamps = pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC")
    epoch_ms = (stamps - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    pd.DataFrame({"timestamp": epoch_ms.to_numpy(), "close": [10.0, 20.0, 30.0]}).to_parquet(
        tmp_path / "nodt.parquet",
    )
    marks._compact_mark_series_for_path.cache_clear()
    avail, close = marks._compact_mark_series_for_path("NDUSDT", "1h", str(tmp_path / "nodt.parquet"))
    assert np.array_equal(avail, (stamps + pd.Timedelta(hours=1)).as_unit("ns").asi8)
    assert np.array_equal(close, np.array([10.0, 20.0, 30.0]))
    marks._compact_mark_series_for_path.cache_clear()


def test_compact_mark_series_root_isolation(tmp_path) -> None:
    """Two roots for one symbol never share cached values."""
    import pandas as pd

    stamps = pd.date_range("2022-01-01", periods=2, freq="1h", tz="UTC")
    epoch_ms = (stamps - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    for root, price in (("a", 11.0), ("b", 22.0)):
        d = tmp_path / root
        d.mkdir()
        pd.DataFrame(
            {"timestamp": epoch_ms.to_numpy(), "close": [price, price], "datetime": stamps},
        ).to_parquet(d / "ISOUSDT.parquet")
    marks._compact_mark_series_for_path.cache_clear()
    _, close_a = marks._compact_mark_series_for_path("ISOUSDT", "1h", str(tmp_path / "a" / "ISOUSDT.parquet"))
    _, close_b = marks._compact_mark_series_for_path("ISOUSDT", "1h", str(tmp_path / "b" / "ISOUSDT.parquet"))
    assert (close_a[0], close_b[0]) == (11.0, 22.0)
    marks._compact_mark_series_for_path.cache_clear()
