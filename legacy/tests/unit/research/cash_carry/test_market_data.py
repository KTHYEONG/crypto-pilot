from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.research.cash_carry.market_data as carry
from src.research.cash_carry.contracts import CarryMarketData
from src.research.cash_carry.market_data import validate_carry_market_data
from src.common.errors import DataIntegrityError

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ms(index: pd.DatetimeIndex) -> pd.Series:
    return (index - _EPOCH) // pd.Timedelta("1ms")


class TestCarryMarketDataValidation:
    def test_accepts_complete_two_bar_fixture_without_interpolation(
        self,
        make_carry_data,
    ) -> None:
        # SC-CARRY-DATA-01: complete event coverage is accepted as-is; no
        # missing funding/borrow observation is synthesized.
        data = make_carry_data(
            n_bars=2,
            funding={"2024-01-01 00:00": 0.001},
            borrow=[0.0001, 0.0001],
        )
        validate_carry_market_data(data)
        assert data.spot.index.equals(data.perp.index)
        assert len(data.funding) == 1
        assert data.borrow.notna().all()

    def test_rejects_funding_gap_over_eight_hours(self, make_carry_data) -> None:
        # SC-CARRY-DATA-02: a 12h funding gap is never accepted as zero funding.
        data = make_carry_data(
            n_bars=4,
            funding={
                "2024-01-01 00:00": 0.001,
                "2024-01-01 12:00": 0.001,
            },
            borrow=[0.0, 0.0, 0.0, 0.0],
        )
        with pytest.raises(DataIntegrityError, match="funding gap"):
            validate_carry_market_data(data)

    def test_accepts_funding_timestamp_jitter_within_tolerance(self, make_carry_data) -> None:
        data = make_carry_data(
            n_bars=4,
            funding={
                "2024-01-01 00:00:00": 0.001,
                "2024-01-01 08:00:00.026": 0.001,
            },
            borrow=[0.0, 0.0, 0.0, 0.0],
        )
        validate_carry_market_data(data)

    def test_rejects_funding_gap_beyond_jitter_tolerance(self, make_carry_data) -> None:
        data = make_carry_data(
            n_bars=4,
            funding={
                "2024-01-01 00:00:00": 0.001,
                "2024-01-01 08:00:00.051": 0.001,
            },
            borrow=[0.0, 0.0, 0.0, 0.0],
        )
        with pytest.raises(DataIntegrityError, match="funding gap"):
            validate_carry_market_data(data)

    def test_rejects_nan_borrow(self, make_carry_data) -> None:
        # SC-CARRY-DATA-02: NaN borrow is never replaced with a zero cost.
        data = make_carry_data(
            n_bars=2,
            funding={"2024-01-01 00:00": 0.001},
            borrow=[0.0, np.nan],
        )
        with pytest.raises(DataIntegrityError, match="borrow"):
            validate_carry_market_data(data)

    def test_rejects_missing_spot_bar(self) -> None:
        # SC-CARRY-DATA-02: a non-uniform spot grid is a missing interval.
        grid = pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 04:00", "2024-01-01 09:00"],
            tz="UTC",
        )
        spot = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0},
            index=grid,
        )
        perp = spot.copy()
        funding = pd.Series([0.001, 0.0, 0.0], index=grid)
        borrow = pd.Series([0.0, 0.0, 0.0], index=grid)
        data = CarryMarketData(symbol="BTCUSDT", spot=spot, perp=perp, funding=funding, borrow=borrow)
        with pytest.raises(DataIntegrityError, match="missing bars"):
            validate_carry_market_data(data)

    def test_rejects_missing_funding_events(self, make_carry_data) -> None:
        # An empty funding series is incomplete, never a zero-cost assumption.
        data = dataclasses.replace(
            make_carry_data(n_bars=2, borrow=[0.0, 0.0]),
            funding=pd.Series(dtype=float),
        )
        with pytest.raises(DataIntegrityError, match="funding"):
            validate_carry_market_data(data)

    def test_rejects_non_positive_price(self, make_carry_data) -> None:
        data = make_carry_data(
            n_bars=2,
            funding={"2024-01-01 00:00": 0.001},
            borrow=[0.0, 0.0],
            spot_close=[100.0, 0.0],
        )
        with pytest.raises(DataIntegrityError, match="strictly positive"):
            validate_carry_market_data(data)

    def test_rejects_misaligned_spot_perp_grid(self, make_carry_data) -> None:
        data = make_carry_data(
            n_bars=2,
            funding={"2024-01-01 00:00": 0.001},
            borrow=[0.0, 0.0],
        )
        perp_shifted = data.perp.copy()
        perp_shifted.index = perp_shifted.index + pd.Timedelta(hours=1)
        shifted = dataclasses.replace(data, perp=perp_shifted)
        with pytest.raises(DataIntegrityError, match="identical"):
            validate_carry_market_data(shifted)


def _events(ts, rates, accrual):
    return pd.DataFrame({
        "ts": pd.DatetimeIndex([pd.Timestamp(t, tz="UTC") for t in ts]),
        "borrow_rate": rates,
        "accrual_seconds": accrual,
    })


class TestBorrowEventConversion:
    def test_full_bar_event_yields_exact_per_bar_rate(self) -> None:
        # SC-CARRY-BORROW-01: a 4h event accruing 0.001 over exactly one 4h bar
        # maps to a 0.001 per-bar rate with no forward fill.
        grid = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
        events = _events(["2024-01-01 00:00", "2024-01-01 04:00"], [0.001, 0.001], [14400, 14400])
        series = carry._borrow_events_to_per_bar(events, grid, pd.Timedelta(hours=4))
        assert np.allclose(series.to_numpy(), [0.001, 0.001])
        assert series.index.equals(grid)

    def test_hourly_events_accrue_four_times_per_4h_bar(self) -> None:
        # SC-CARRY-BORROW-02: four 1h events at 0.001 each fully cover one 4h
        # bar, so the per-bar rate is 0.004.
        grid = pd.date_range("2024-01-01", periods=1, freq="4h", tz="UTC")
        events = _events(
            ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 03:00"],
            [0.001] * 4, [3600] * 4,
        )
        series = carry._borrow_events_to_per_bar(events, grid, pd.Timedelta(hours=4))
        assert np.isclose(series.iloc[0], 0.004)

    def test_eight_hour_event_splits_across_two_bars(self) -> None:
        # SC-CARRY-BORROW-03: an 8h event spanning two 4h bars accrues half per
        # bar (0.0005 each), matching the ledger's one-debit-per-held-bar.
        grid = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
        events = _events(["2024-01-01 00:00"], [0.001], [28800])
        series = carry._borrow_events_to_per_bar(events, grid, pd.Timedelta(hours=4))
        assert np.allclose(series.to_numpy(), [0.0005, 0.0005])

    def test_overlapping_events_rejected(self) -> None:
        grid = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
        events = _events(["2024-01-01 00:00", "2024-01-01 01:00"], [0.001, 0.001], [7200, 7200])
        with pytest.raises(DataIntegrityError, match="overlap"):
            carry._borrow_events_to_per_bar(events, grid, pd.Timedelta(hours=4))

    def test_uncovered_bar_rejected(self) -> None:
        grid = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
        events = _events(["2024-01-01 00:00"], [0.001], [14400])
        with pytest.raises(DataIntegrityError, match="coverage"):
            carry._borrow_events_to_per_bar(events, grid, pd.Timedelta(hours=4))

    def test_non_positive_duration_rejected(self) -> None:
        grid = pd.date_range("2024-01-01", periods=1, freq="4h", tz="UTC")
        events = _events(["2024-01-01 00:00"], [0.001], [0])
        with pytest.raises(DataIntegrityError, match="> 0"):
            carry._borrow_events_to_per_bar(events, grid, pd.Timedelta(hours=4))

    def test_non_finite_rate_rejected(self) -> None:
        grid = pd.date_range("2024-01-01", periods=1, freq="4h", tz="UTC")
        events = _events(["2024-01-01 00:00"], [np.nan], [14400])
        with pytest.raises(DataIntegrityError, match="finite"):
            carry._borrow_events_to_per_bar(events, grid, pd.Timedelta(hours=4))

    def test_duplicate_events_rejected(self) -> None:
        grid = pd.date_range("2024-01-01", periods=1, freq="4h", tz="UTC")
        events = _events(["2024-01-01 00:00", "2024-01-01 00:00"], [0.001, 0.001], [14400, 14400])
        with pytest.raises(DataIntegrityError, match="duplicates"):
            carry._borrow_events_to_per_bar(events, grid, pd.Timedelta(hours=4))

    def test_unsorted_events_rejected(self) -> None:
        grid = pd.date_range("2024-01-01", periods=1, freq="4h", tz="UTC")
        events = _events(["2024-01-01 01:00", "2024-01-01 00:00"], [0.001, 0.001], [3600, 3600])
        with pytest.raises(DataIntegrityError, match="monotonic"):
            carry._borrow_events_to_per_bar(events, grid, pd.Timedelta(hours=4))

    def test_empty_grid_rejected(self) -> None:
        events = _events(["2024-01-01 00:00"], [0.001], [14400])
        with pytest.raises(DataIntegrityError, match="non-empty bar grid"):
            carry._borrow_events_to_per_bar(events, pd.DatetimeIndex([], tz="UTC"), pd.Timedelta(hours=4))

    def test_naive_borrow_timestamps_rejected(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "borrow.parquet"
        pd.DataFrame({
            "timestamp": [1704067200000], "borrow_rate": [0.001], "accrual_seconds": [14400],
        }).to_parquet(path)
        original_to_datetime = carry.pd.to_datetime
        monkeypatch.setattr(carry.pd, "to_datetime", lambda values, **kwargs: original_to_datetime(values, utc=False))
        with pytest.raises(DataIntegrityError, match="tz-aware"):
            carry._load_borrow_events(path)

    def test_borrow_export_without_accrual_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "borrow.parquet"
        pd.DataFrame({"timestamp": [1704067200000], "borrow_rate": [0.001]}).to_parquet(path)

        with pytest.raises(DataIntegrityError, match="accrual_seconds"):
            carry._load_borrow_events(path)

    def test_borrow_export_without_rate_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "borrow.parquet"
        pd.DataFrame({"timestamp": [1704067200000], "accrual_seconds": [14400]}).to_parquet(path)

        with pytest.raises(DataIntegrityError, match="borrow_rate"):
            carry._load_borrow_events(path)

    def test_borrow_export_without_timestamp_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "borrow.parquet"
        pd.DataFrame({"borrow_rate": [0.001], "accrual_seconds": [14400]}).to_parquet(path)

        with pytest.raises(DataIntegrityError, match="timestamp"):
            carry._load_borrow_events(path)

    def test_borrow_datetime_without_rate_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "borrow.parquet"
        pd.DataFrame({"datetime": ["2024-01-01T00:00:00Z"], "accrual_seconds": [14400]}).to_parquet(path)

        with pytest.raises(DataIntegrityError, match="borrow_rate"):
            carry._load_borrow_events(path)

    def test_missing_borrow_file_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataIntegrityError, match="does not exist"):
            carry._load_borrow_events(tmp_path / "missing.parquet")


class TestLoadCarryMarketData:
    @pytest.fixture
    def carry_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        def _spot(symbol: str, timeframe: str) -> Path:
            return tmp_path / "spot" / "ohlcv" / timeframe / f"{symbol.replace('/', '_')}.parquet"

        def _perp(symbol: str, timeframe: str) -> Path:
            return tmp_path / "futures" / "ohlcv" / timeframe / f"{symbol.replace('/', '_')}.parquet"

        def _fund(symbol: str) -> Path:
            return tmp_path / "futures" / "funding" / f"{symbol.replace('/', '_')}.parquet"

        def _borrow(symbol: str) -> Path:
            return tmp_path / "spot" / "borrow" / f"{symbol.replace('/', '_')}.parquet"

        monkeypatch.setattr(carry, "spot_ohlcv_path", _spot)
        monkeypatch.setattr(carry, "ohlcv_path", _perp)
        monkeypatch.setattr(carry, "funding_path", _fund)
        monkeypatch.setattr(carry, "borrow_path", _borrow)

        hourly = pd.date_range("2024-01-01", "2024-01-02", freq="1h", inclusive="left", tz="UTC")
        n = len(hourly)
        price = 100.0 + np.arange(n, dtype=np.float64)
        frame = pd.DataFrame({
            "timestamp": _ms(hourly),
            "open": price, "high": price + 1.0, "low": price - 1.0,
            "close": price, "volume": 10.0,
        })
        for path in (_spot("BTCUSDT", "1h"), _perp("BTCUSDT", "1h")):
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path)

        fund_dir = tmp_path / "futures" / "funding"
        fund_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", "2024-01-02", freq="8h", inclusive="left", tz="UTC"),
            "funding_rate": 0.0002,
        }).to_parquet(_fund("BTCUSDT"))

        borrow_dir = tmp_path / "spot" / "borrow"
        borrow_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "timestamp": _ms(pd.date_range("2024-01-01", "2024-01-02", freq="4h", inclusive="left", tz="UTC")),
            "borrow_rate": 0.0001,
            "accrual_seconds": 14400.0,
        }).to_parquet(_borrow("BTCUSDT"))
        return tmp_path

    def test_loads_aligned_grid_with_converted_borrow(self, carry_files: Path) -> None:
        # SC-CARRY-LOAD-01: full end-to-end load resamples spot/perp to an
        # identical 4h grid, aligns funding, and converts borrow events.
        data = carry.load_carry_market_data("BTCUSDT", "2024-01-01", "2024-01-01 23:59:59")
        assert data.spot.index.equals(data.perp.index)
        period = data.spot.index[1] - data.spot.index[0]
        assert period == pd.Timedelta(hours=4)
        assert np.allclose(data.borrow.to_numpy(), 0.0001)
        assert len(data.funding) == 3

    def test_missing_spot_raises(self, carry_files: Path) -> None:
        (carry_files / "spot" / "ohlcv" / "1h" / "BTCUSDT.parquet").unlink()
        with pytest.raises(DataIntegrityError, match="spot data missing"):
            carry.load_carry_market_data("BTCUSDT", "2024-01-01", "2024-01-01 23:59:59")

    def test_missing_borrow_raises(self, carry_files: Path) -> None:
        (carry_files / "spot" / "borrow" / "BTCUSDT.parquet").unlink()
        with pytest.raises(DataIntegrityError, match="borrow data missing"):
            carry.load_carry_market_data("BTCUSDT", "2024-01-01", "2024-01-01 23:59:59")

    def test_single_resampled_spot_bar_is_rejected(self, carry_files: Path, monkeypatch) -> None:
        one_bar = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1.0]},
            index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
        )
        monkeypatch.setattr(carry, "load_ohlcv_1h_as_4h", lambda *args, **kwargs: one_bar)
        with pytest.raises(DataIntegrityError, match="fewer than 2 bars"):
            carry.load_carry_market_data("BTCUSDT", "2024-01-01", "2024-01-01 23:59:59")


def test_carry_market_data_rejects_missing_cost_input(make_carry_data) -> None:
    """RF-CARRY-01: a missing funding or borrow observation fails closed.

    Missing costs are never replaced with a zero-cost series: the validator
    blocks the run with ``DataIntegrityError``.
    """
    missing_borrow = make_carry_data(
        n_bars=2,
        funding={"2024-01-01 00:00": 0.001},
        borrow=[0.0, np.nan],
    )
    with pytest.raises(DataIntegrityError, match="borrow"):
        validate_carry_market_data(missing_borrow)

    missing_funding = dataclasses.replace(
        make_carry_data(n_bars=2, borrow=[0.0, 0.0]),
        funding=pd.Series(dtype=float),
    )
    with pytest.raises(DataIntegrityError, match="funding"):
        validate_carry_market_data(missing_funding)
