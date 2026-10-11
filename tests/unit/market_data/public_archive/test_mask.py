from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.mask import allowed_days, load_daily_census, seat_day_mask
from src.strategy.universe import build_pit_roster


def _census(n: int = 120, symbols: tuple[str, ...] = ("AUSDT", "BUSDT", "CUSDT")) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC"), tz="UTC")
    vols = {"AUSDT": 10_000_000.0, "BUSDT": 5_000_000.0, "CUSDT": 2_000_000.0}
    close = pd.DataFrame(dict.fromkeys(symbols, 100.0), index=index, dtype="float64")
    qv = pd.DataFrame({s: vols.get(s, 3_000_000.0) for s in symbols}, index=index, dtype="float64")  # noqa: C420
    return close, qv


def _write_lake(path: Path, symbol: str, days: list[date]) -> None:
    rows: list[tuple[int, float, float]] = []
    for day in days:
        base = pd.Timestamp(day, tz="UTC")
        for hour in range(24):
            ts = base + pd.Timedelta(hours=hour)
            rows.append((int(ts.value // 10**6), 100.0, 1_000_000.0))
    frame = pd.DataFrame(rows, columns=["timestamp", "close", "quote_vol"])
    frame.to_parquet(path / f"{symbol}.parquet", index=False)


def test_mask_matches_pit_roster_without_preroll() -> None:
    close, qv = _census()
    mask = seat_day_mask(close, qv, breadth=2, preroll_days=0)
    expected = build_pit_roster(close, qv, tuple(close.columns), breadth=2)
    pd.testing.assert_frame_equal(mask, expected)


def test_preroll_extends_backwards_only() -> None:
    n = 150
    index = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC"), tz="UTC")
    close = pd.DataFrame({"AUSDT": 100.0, "BUSDT": 100.0}, index=index, dtype="float64")
    vols_a = [10_000_000.0] * n
    vols_b = [1_000_000.0] * 60 + [20_000_000.0] * (n - 60)
    qv = pd.DataFrame({"AUSDT": vols_a, "BUSDT": vols_b}, index=index, dtype="float64")
    seat = seat_day_mask(close, qv, breadth=1, preroll_days=0)
    first_b = seat.index[seat["BUSDT"].to_numpy().argmax()]
    pos = seat.index.get_loc(first_b)
    assert seat["BUSDT"].iloc[pos:].all()
    assert pos >= 31
    mask = seat_day_mask(close, qv, breadth=1, preroll_days=30)
    assert bool(mask["BUSDT"].iloc[pos - 30])
    assert not bool(mask["BUSDT"].iloc[pos - 31])
    assert bool(mask["BUSDT"].iloc[pos:].all())


def test_delisted_symbol_stays_in_census(tmp_path: Path) -> None:
    start = date(2021, 1, 1)
    full = [start + pd.Timedelta(days=i).to_pytimedelta() for i in range(200)]
    half = full[:150]
    _write_lake(tmp_path, "AUSDT", full)
    _write_lake(tmp_path, "BUSDT", half)
    (tmp_path / "junk.tmp.parquet").write_bytes(b"x")
    (tmp_path / "BTCUSDT_250627.parquet").write_bytes(b"x")
    close, qv = load_daily_census(tmp_path)
    assert list(close.columns) == ["AUSDT", "BUSDT"]
    assert len(close) == 200
    assert close["BUSDT"].iloc[150:].isna().all()
    mask = seat_day_mask(close, qv, breadth=2, preroll_days=0)
    assert "BUSDT" in mask.columns
    assert bool(mask["BUSDT"].iloc[:150].any())


def test_allowed_days_carry_forward_beyond_lake() -> None:
    index = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"), tz="UTC")
    mask = pd.DataFrame({"AUSDT": [False] * 9 + [True], "BUSDT": [False] * 10}, index=index, dtype=bool)
    out = allowed_days(mask, first=date(2024, 1, 1), last=date(2024, 1, 13))
    assert out["AUSDT"] == frozenset({date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12), date(2024, 1, 13)})
    assert "BUSDT" not in out


def test_allowed_days_omits_unseated_symbols() -> None:
    index = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"), tz="UTC")
    mask = pd.DataFrame({"AUSDT": [True] * 5, "BUSDT": [False] * 5}, index=index, dtype=bool)
    out = allowed_days(mask, first=date(2024, 1, 1), last=date(2024, 1, 5))
    assert set(out) == {"AUSDT"}


def test_invalid_parameters_rejected() -> None:
    close, qv = _census()
    with pytest.raises(ValueError, match="breadth"):
        seat_day_mask(close, qv, breadth=0, preroll_days=0)
    with pytest.raises(ValueError, match="preroll_days"):
        seat_day_mask(close, qv, breadth=1, preroll_days=-1)


def test_census_requires_lake(tmp_path: Path) -> None:
    with pytest.raises(DataIntegrityError):
        load_daily_census(tmp_path / "absent")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DataIntegrityError):
        load_daily_census(empty)


def test_census_skips_temp_and_non_perpetual(tmp_path: Path) -> None:
    start = date(2021, 1, 1)
    days = [start + pd.Timedelta(days=i).to_pytimedelta() for i in range(3)]
    _write_lake(tmp_path, "AUSDT", days)
    (tmp_path / ".hiddenUSDT.parquet").write_bytes(b"x")
    (tmp_path / "BUSDUSDT.tmp.parquet").write_bytes(b"x")
    (tmp_path / "BTCUSDT_250627.parquet").write_bytes(b"x")
    close, _ = load_daily_census(tmp_path)
    assert list(close.columns) == ["AUSDT"]


def test_census_handles_datetime_stamps_and_bad_frames(tmp_path: Path) -> None:
    start = date(2021, 1, 1)
    days = [start + pd.Timedelta(days=i).to_pytimedelta() for i in range(3)]
    _write_lake(tmp_path, "AUSDT", days)
    idx = pd.date_range("2021-01-01", periods=48, freq="h", tz="UTC")
    pd.DataFrame({"timestamp": idx, "close": 50.0, "quote_vol": 2_000_000.0}).to_parquet(
        tmp_path / "CUSDT.parquet", index=False
    )
    pd.DataFrame({"timestamp": [1, 2], "open": [1.0, 2.0]}).to_parquet(tmp_path / "DUSDT.parquet", index=False)
    pd.DataFrame({"timestamp": [None, None], "close": [1.0, 2.0]}).to_parquet(tmp_path / "EUSDT.parquet", index=False)
    close, qv = load_daily_census(tmp_path)
    assert list(close.columns) == ["AUSDT", "CUSDT", "DUSDT", "EUSDT"]
    assert close["CUSDT"].iloc[0] == 50.0
    assert close["DUSDT"].isna().all()


def test_census_all_bad_frames_raises(tmp_path: Path) -> None:
    pd.DataFrame({"timestamp": [None], "close": [1.0]}).to_parquet(tmp_path / "AUSDT.parquet", index=False)
    with pytest.raises(DataIntegrityError):
        load_daily_census(tmp_path)


def test_preroll_beyond_history_breaks() -> None:
    index = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"), tz="UTC")
    close = pd.DataFrame({"AUSDT": [100.0, 100.0, 100.0]}, index=index, dtype="float64")
    qv = pd.DataFrame({"AUSDT": [10_000_000.0] * 3}, index=index, dtype="float64")
    mask = seat_day_mask(close, qv, breadth=1, preroll_days=30)
    assert mask.shape == (3, 1)
    assert str(mask.dtypes.iloc[0]) == "bool"


def test_allowed_days_edge_cases() -> None:
    index = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"), tz="UTC")
    mask = pd.DataFrame({"AUSDT": [True, False, True]}, index=index, dtype=bool)
    assert allowed_days(mask, first=date(2024, 1, 5), last=date(2024, 1, 1)) == {}
    assert allowed_days(mask.iloc[:0], first=date(2024, 1, 1), last=date(2024, 1, 3)) == {}


def test_census_daily_aggregation_keeps_gaps_unknown(tmp_path: Path) -> None:
    timestamps = pd.to_datetime([
        "2024-01-03T23:00:00Z", "2024-01-01T23:00:00Z", "2024-01-01T00:00:00Z",
    ])
    pd.DataFrame({
        "timestamp": timestamps, "close": [30.0, 20.0, 10.0],
        "quote_vol": [float("nan"), 3.0, 2.0],
    }).to_parquet(tmp_path / "BTCUSDT.parquet", index=False)
    close, quote = load_daily_census(tmp_path)
    assert close.index.equals(pd.date_range("2024-01-01", periods=3, tz="UTC"))
    assert close.iloc[0, 0] == 20.0
    assert quote.iloc[0, 0] == 5.0
    assert pd.isna(close.iloc[1, 0])
    assert pd.isna(quote.iloc[1, 0])
    assert pd.isna(quote.iloc[2, 0])
