from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.services.futures_collection import DataCollector, MarkPriceCoverage

GRID_START = pd.Timestamp("2021-01-01", tz="UTC")
GRID_END = pd.Timestamp("2021-01-03", tz="UTC")
HOURS = pd.date_range(GRID_START, GRID_END, freq="1h", tz="UTC")


def _mark_frame(timestamps: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    ts = timestamps if timestamps is not None else HOURS
    return pd.DataFrame(
        {
            "timestamp": (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "datetime": ts,
        },
    )


class TestMarkPriceCoverage:
    """MHS-21-MARK-PRICE-COVERAGE-FAIL-CLOSED: unusable intervals never pass as usable."""

    def _collector(self) -> DataCollector:
        return DataCollector()

    def test_full_coverage_is_primary_usable(self) -> None:
        coverage = self._collector()._mark_price_coverage(
            "BTCUSDT", "1h", GRID_START, GRID_END, _mark_frame(),
        )
        assert coverage.primary_usable is True
        assert coverage.missing_intervals == ()
        assert coverage.observed_start == HOURS[0]
        assert coverage.observed_end == HOURS[-1]

    def test_empty_response_is_not_usable(self) -> None:
        coverage = self._collector()._mark_price_coverage(
            "BTCUSDT", "1h", GRID_START, GRID_END, pd.DataFrame(),
        )
        assert coverage.primary_usable is False
        assert coverage.missing_intervals != ()

    def test_interior_gap_is_not_usable(self) -> None:
        missing = HOURS[HOURS.hour == 12]
        frame = _mark_frame(HOURS[~HOURS.isin(missing)])
        coverage = self._collector()._mark_price_coverage(
            "BTCUSDT", "1h", GRID_START, GRID_END, frame,
        )
        assert coverage.primary_usable is False
        assert coverage.missing_intervals != ()

    def test_non_positive_mark_is_not_usable(self) -> None:
        frame = _mark_frame()
        frame.loc[frame.index[5], "close"] = 0.0
        coverage = self._collector()._mark_price_coverage(
            "BTCUSDT", "1h", GRID_START, GRID_END, frame,
        )
        assert coverage.primary_usable is False

    def test_never_substitutes_ohlcv_close_as_mark(self) -> None:
        coverage = self._collector()._mark_price_coverage(
            "BTCUSDT", "1h", GRID_START, GRID_END, pd.DataFrame(),
        )
        assert coverage.endpoint == "GET /fapi/v1/markPriceKlines"
        assert coverage.primary_usable is False

    def test_field_types_are_concrete(self) -> None:
        assert MarkPriceCoverage.__dataclass_fields__["primary_usable"].type is bool


class TestEnsureMarkPriceData:
    """MHS-21-MARK-PRICE-COVERAGE-FAIL-CLOSED: collection persists canonical mark candles."""

    def test_persists_parquet_and_manifest(self, tmp_path: Path, monkeypatch) -> None:
        collector = DataCollector()
        monkeypatch.setattr(
            "src.market_data.services.futures_collection._mark_price_path",
            lambda symbol, timeframe: tmp_path / timeframe / f"{symbol}.parquet",
        )
        monkeypatch.setattr(
            "src.market_data.services.futures_collection._mark_price_manifest_path",
            lambda symbol, timeframe: tmp_path / timeframe / f"{symbol}.coverage.json",
        )
        fetched = _mark_frame(HOURS[:10])
        collector.client.fetch_mark_price_klines = lambda symbol, timeframe, start, end: fetched
        coverage = collector.ensure_mark_price_data(
            "BTCUSDT", "1h", "2021-01-01", "2021-01-01T09:00:00Z",
        )
        assert coverage.primary_usable is True
        parquet = tmp_path / "1h" / "BTCUSDT.parquet"
        manifest = tmp_path / "1h" / "BTCUSDT.coverage.json"
        assert parquet.exists()
        assert manifest.exists()
        assert "primary_usable" in manifest.read_text()

    def test_empty_fetch_fails_closed_and_still_records_manifest(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        collector = DataCollector()
        monkeypatch.setattr(
            "src.market_data.services.futures_collection._mark_price_path",
            lambda symbol, timeframe: tmp_path / timeframe / f"{symbol}.parquet",
        )
        monkeypatch.setattr(
            "src.market_data.services.futures_collection._mark_price_manifest_path",
            lambda symbol, timeframe: tmp_path / timeframe / f"{symbol}.coverage.json",
        )
        collector.client.fetch_mark_price_klines = lambda symbol, timeframe, start, end: pd.DataFrame()
        coverage = collector.ensure_mark_price_data(
            "BTCUSDT", "1h", "2021-01-01", "2021-01-03",
        )
        assert coverage.primary_usable is False
        manifest = tmp_path / "1h" / "BTCUSDT.coverage.json"
        assert manifest.exists()


class _MarkResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return self.payload


class TestFetchMarkPriceKlinesBranches:
    """MHS-21-MARK-PRICE-COVERAGE-FAIL-CLOSED: pagination, empty, datetime inputs."""

    def test_accepts_datetime_inputs_and_market_id(self, monkeypatch) -> None:
        from datetime import UTC, datetime

        from src.market_data.binance.futures import BinanceClient

        client = BinanceClient()
        monkeypatch.setattr(
            client.exchange, "market",
            lambda symbol: {"id": "BTCUSDT"},
        )
        responses = iter([
            _MarkResponse(b'[[1609459200000,"100","101","99","100.5"]]'),
        ])
        monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: next(responses))
        frame = client.fetch_mark_price_klines(
            "BTCUSDT",
            "1h",
            datetime(2021, 1, 1, tzinfo=UTC),
            datetime(2021, 1, 2, tzinfo=UTC),
        )
        assert len(frame) == 1
        assert frame["close"].iloc[0] == 100.5

    def test_paginates_across_pages(self, monkeypatch) -> None:
        from src.market_data.binance.futures import BinanceClient

        client = BinanceClient()
        page1 = b'[[1609459200000,"100","101","99","100.5"]]'
        page2 = b'[[1609462800000,"100.5","102","100","101.0"]]'
        responses = iter([_MarkResponse(page1), _MarkResponse(page2)])
        monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: next(responses))
        # A wide window forces a second page.
        frame = client.fetch_mark_price_klines("BTCUSDT", "1m", "2021-01-01T00:00:00Z", "2021-01-31T00:00:00Z")
        assert len(frame) == 2

    def test_empty_response_returns_canonical_columns(self, monkeypatch) -> None:
        from src.market_data.binance.futures import BinanceClient

        client = BinanceClient()
        responses = iter([_MarkResponse(b"[]")])
        monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: next(responses))
        frame = client.fetch_mark_price_klines("BTCUSDT", "1h", "2021-01-01", "2021-01-02")
        assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "datetime"]
        assert frame.empty

    def test_network_error_returns_empty_frame(self, monkeypatch) -> None:
        from src.market_data.binance.futures import BinanceClient

        client = BinanceClient()

        def _boom(*args, **kwargs):
            raise OSError("boom")

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        frame = client.fetch_mark_price_klines("BTCUSDT", "1h", "2021-01-01", "2021-01-02")
        assert frame.empty


class TestFetchMarkPriceKlinesEdgeBranches:
    """MHS-21-MARK-PRICE-COVERAGE-FAIL-CLOSED: Z-less iso, missing end, malformed rows."""

    @staticmethod
    def _paged(*payloads: bytes) -> object:
        # Every page past the supplied ones returns an empty kline array so the
        # pagination loop terminates cleanly instead of spinning on a constant mock.
        responses = iter([_MarkResponse(p) for p in payloads] + [_MarkResponse(b"[]")])
        return lambda *args, **kwargs: next(responses)

    def test_iso_without_z_gets_z_suffix(self, monkeypatch) -> None:
        from src.market_data.binance.futures import BinanceClient

        client = BinanceClient()
        monkeypatch.setattr(
            "urllib.request.urlopen",
            self._paged(b'[[1609459200000,"100","101","99","100.5"]]'),
        )
        frame = client.fetch_mark_price_klines(
            "BTCUSDT", "1h",
            "2021-01-01T00:00:00", "2021-01-01T01:00:00",
        )
        assert len(frame) == 1

    def test_missing_end_date_uses_exchange_clock(self, monkeypatch) -> None:
        from src.market_data.binance.futures import BinanceClient

        client = BinanceClient()
        monkeypatch.setattr(client.exchange, "milliseconds", lambda: 4102444800000)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            self._paged(b'[[1609459200000,"100","101","99","100.5"]]'),
        )
        frame = client.fetch_mark_price_klines("BTCUSDT", "1h", "2021-01-01T00:00:00Z")
        assert len(frame) == 1

    def test_malformed_rows_are_skipped(self, monkeypatch) -> None:
        from src.market_data.binance.futures import BinanceClient

        client = BinanceClient()
        monkeypatch.setattr(
            "urllib.request.urlopen",
            self._paged(b'[[1609459200000,"100","101","99","100.5"],["not","a","row"]]'),
        )
        frame = client.fetch_mark_price_klines("BTCUSDT", "1h", "2021-01-01", "2021-01-02")
        assert len(frame) == 1

    def test_only_malformed_rows_return_empty(self, monkeypatch) -> None:
        from src.market_data.binance.futures import BinanceClient

        client = BinanceClient()
        monkeypatch.setattr(
            "urllib.request.urlopen",
            self._paged(b'[["not","a","row"]]'),
        )
        frame = client.fetch_mark_price_klines("BTCUSDT", "1h", "2021-01-01", "2021-01-02")
        assert frame.empty


class TestFetchMarkPriceKlinesPagination:
    """MHS-21-MARK-PRICE-COVERAGE-FAIL-CLOSED: single-page end break and multi-page."""

    def test_breaks_when_page_reaches_end(self, monkeypatch) -> None:
        from src.market_data.binance.futures import BinanceClient

        client = BinanceClient()
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *args, **kwargs: _MarkResponse(
                b'[[1609459200000,"100","101","99","100.5"]]',
            ),
        )
        # start before end; the single row sits exactly at the end timestamp.
        frame = client.fetch_mark_price_klines(
            "BTCUSDT", "1h", "2020-12-31T23:00:00Z", "2021-01-01T00:00:00Z",
        )
        assert len(frame) == 1


class TestMarkPriceCacheAndPaths:
    """MHS-21-MARK-PRICE-COVERAGE-FAIL-CLOSED: cache load branches and path helpers."""

    def test_mark_price_path_helpers(self) -> None:
        from src.market_data.services import futures_collection as fc

        path = fc._mark_price_path("BTC/USDT", "1h")
        assert str(path).endswith("markPriceKlines/1h/BTCUSDT.parquet")
        manifest = fc._mark_price_manifest_path("BTC/USDT", "1h")
        assert str(manifest).endswith("markPriceKlines/1h/BTCUSDT.coverage.json")

    def test_load_mark_price_cache_branches(self, tmp_path: Path) -> None:
        from src.market_data.services import futures_collection as fc

        missing = tmp_path / "missing.parquet"
        assert fc.DataCollector._load_mark_price_cache(missing).empty

        corrupt = tmp_path / "corrupt.parquet"
        corrupt.write_bytes(b"not a parquet")
        assert fc.DataCollector._load_mark_price_cache(corrupt).empty

        empty = tmp_path / "empty.parquet"
        pd.DataFrame().to_parquet(empty)
        assert fc.DataCollector._load_mark_price_cache(empty).empty

        no_dt = tmp_path / "no_dt.parquet"
        pd.DataFrame(
            {"timestamp": [1609459200000], "open": [100.0], "high": [101.0],
             "low": [99.0], "close": [100.0]},
        ).to_parquet(no_dt)
        loaded = fc.DataCollector._load_mark_price_cache(no_dt)
        assert len(loaded) == 1
        assert loaded["datetime"].iloc[0] == pd.Timestamp("2021-01-01 00:00", tz="UTC")

    def test_coverage_with_two_separate_gaps(self) -> None:
        from src.market_data.services.futures_collection import DataCollector

        hours = pd.date_range(GRID_START, GRID_END, freq="1h", tz="UTC")
        keep = hours[~hours.isin(hours[hours.hour == 12]) & ~hours.isin(hours[hours.hour == 18])]
        frame = _mark_frame(keep)
        coverage = DataCollector()._mark_price_coverage(
            "BTCUSDT", "1h", GRID_START, GRID_END, frame,
        )
        assert coverage.primary_usable is False
        assert len(coverage.missing_intervals) >= 2

    def test_ensure_mark_price_data_uses_complete_cache(self, tmp_path: Path, monkeypatch) -> None:
        from src.market_data.services import futures_collection as fc

        collector = fc.DataCollector()
        monkeypatch.setattr(
            "src.market_data.services.futures_collection._mark_price_path",
            lambda symbol, timeframe: tmp_path / timeframe / f"{symbol}.parquet",
        )
        monkeypatch.setattr(
            "src.market_data.services.futures_collection._mark_price_manifest_path",
            lambda symbol, timeframe: tmp_path / timeframe / f"{symbol}.coverage.json",
        )
        path = tmp_path / "1h" / "BTCUSDT.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        _mark_frame(HOURS[:30]).to_parquet(path)
        collector.client.fetch_mark_price_klines = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not fetch when cache is complete"),
        )
        coverage = collector.ensure_mark_price_data("BTCUSDT", "1h", "2021-01-01", "2021-01-01T12:00:00Z")
        assert coverage.primary_usable is True


class TestLoadMarkPricePanel:
    """MHS-MARK-01-CAUSAL-AVAILABILITY / MHS-MARK-02-CACHE-ONLY: the causal
    mark panel is read-only and never substitutes OHLCV closes."""

    @staticmethod
    def _write_cache(tmp_path: Path, symbol: str, timestamps: pd.DatetimeIndex, closes: list[float]) -> None:
        path = tmp_path / "1h" / f"{symbol}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = _mark_frame(timestamps)
        frame["close"] = closes
        frame.to_parquet(path, index=False)

    def _collector(self, tmp_path: Path, monkeypatch) -> DataCollector:
        monkeypatch.setattr(
            "src.market_data.services.futures_collection._mark_price_path",
            lambda symbol, timeframe: tmp_path / timeframe / f"{symbol}.parquet",
        )
        return DataCollector()

    def test_one_hour_availability_lag_and_no_cross_gap(self, tmp_path: Path, monkeypatch) -> None:
        collector = self._collector(tmp_path, monkeypatch)
        hourly = pd.DatetimeIndex(["2021-01-01 00:00", "2021-01-01 02:00"], tz="UTC")
        self._write_cache(tmp_path, "BTCUSDT", hourly, [100.0, 101.0])
        grid = pd.date_range("2021-01-01 00:00", "2021-01-01 03:00", freq="1min", tz="UTC")
        panel = collector.load_mark_price_panel(["BTCUSDT"], "1h", grid)
        assert list(panel.index) == list(grid)
        assert list(panel.columns) == ["BTCUSDT"]
        assert pd.isna(panel.loc["2021-01-01 00:59", "BTCUSDT"])
        assert panel.loc["2021-01-01 01:00", "BTCUSDT"] == 100.0
        assert panel.loc["2021-01-01 01:59", "BTCUSDT"] == 100.0
        assert pd.isna(panel.loc["2021-01-01 02:00", "BTCUSDT"])
        assert panel.loc["2021-01-01 03:00", "BTCUSDT"] == 101.0

    def test_full_hourly_cache_fills_each_minute(self, tmp_path: Path, monkeypatch) -> None:
        collector = self._collector(tmp_path, monkeypatch)
        hourly = pd.date_range("2021-01-01 00:00", "2021-01-01 03:00", freq="1h", tz="UTC")
        self._write_cache(tmp_path, "BTCUSDT", hourly, [100.0, 101.0, 102.0, 103.0])
        grid = pd.date_range("2021-01-01 00:00", "2021-01-01 03:59", freq="1min", tz="UTC")
        panel = collector.load_mark_price_panel(["BTCUSDT"], "1h", grid)
        assert panel.loc["2021-01-01 01:00", "BTCUSDT"] == 100.0
        assert panel.loc["2021-01-01 01:59", "BTCUSDT"] == 100.0
        assert panel.loc["2021-01-01 02:00", "BTCUSDT"] == 101.0
        assert panel.loc["2021-01-01 02:59", "BTCUSDT"] == 101.0

    def test_absent_cache_is_nan_not_ohlcv(self, tmp_path: Path, monkeypatch) -> None:
        collector = self._collector(tmp_path, monkeypatch)
        grid = pd.date_range("2021-01-01 00:00", "2021-01-01 02:00", freq="1min", tz="UTC")
        panel = collector.load_mark_price_panel(["BTCUSDT"], "1h", grid)
        assert panel["BTCUSDT"].isna().all()

    def test_never_calls_collector_client(self, tmp_path: Path, monkeypatch) -> None:
        collector = self._collector(tmp_path, monkeypatch)
        collector.client.fetch_mark_price_klines = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("load_mark_price_panel must not fetch"),
        )
        collector.ensure_mark_price_data = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("load_mark_price_panel must not collect"),
        )
        grid = pd.date_range("2021-01-01 00:00", "2021-01-01 02:00", freq="1min", tz="UTC")
        panel = collector.load_mark_price_panel(["BTCUSDT"], "1h", grid)
        assert panel["BTCUSDT"].isna().all()

    def test_rejects_invalid_inputs(self, tmp_path: Path, monkeypatch) -> None:
        collector = self._collector(tmp_path, monkeypatch)
        grid = pd.date_range("2021-01-01 00:00", periods=3, freq="1min", tz="UTC")
        with pytest.raises(ValueError, match="timeframe"):
            collector.load_mark_price_panel(["BTCUSDT"], "4h", grid)
        with pytest.raises(DataIntegrityError):
            collector.load_mark_price_panel([], "1h", grid)
        with pytest.raises(DataIntegrityError):
            collector.load_mark_price_panel(["A", "A"], "1h", grid)
        with pytest.raises(DataIntegrityError):
            collector.load_mark_price_panel(["BTCUSDT"], "1h", pd.date_range("2021-01-01", periods=3, freq="1min"))
        with pytest.raises(DataIntegrityError):
            collector.load_mark_price_panel(["BTCUSDT"], "1h", pd.DatetimeIndex([], tz="UTC"))
        with pytest.raises(DataIntegrityError):
            collector.load_mark_price_panel(["BTCUSDT"], "1h", grid[::-1])

    def test_multi_symbol_column_order(self, tmp_path: Path, monkeypatch) -> None:
        collector = self._collector(tmp_path, monkeypatch)
        hourly = pd.date_range("2021-01-01 00:00", "2021-01-01 02:00", freq="1h", tz="UTC")
        self._write_cache(tmp_path, "BTCUSDT", hourly, [100.0, 101.0, 102.0])
        grid = pd.date_range("2021-01-01 00:00", "2021-01-01 03:00", freq="1min", tz="UTC")
        panel = collector.load_mark_price_panel(["BTCUSDT", "ETHUSDT"], "1h", grid)
        assert list(panel.columns) == ["BTCUSDT", "ETHUSDT"]
        assert panel["ETHUSDT"].isna().all()
