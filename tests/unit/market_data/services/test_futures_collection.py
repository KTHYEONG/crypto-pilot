from __future__ import annotations

import logging

import pandas as pd
import pytest

import src.market_data.services.futures_collection as collector_module
from src.common.errors import DataIntegrityError
from src.market_data.services.futures_collection import (
    DataCollector,
    DataValidator,
    _METRICS_CANONICAL_COLUMNS,
    _normalize_funding_frame,
)

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ms(index: pd.DatetimeIndex) -> pd.Series:
    return (index - _EPOCH) // pd.Timedelta("1ms")


class TestDataValidator:
    def test_detects_gaps_and_inverted_high_low(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        frame = pd.DataFrame({
            "datetime": idx,
            "open": [100.0] * 3, "high": [101.0, 101.0, 90.0],
            "low": [99.0, 99.0, 95.0], "close": [100.0] * 3, "volume": [1.0] * 3,
        })
        issues = DataValidator.validate(frame, "BTCUSDT", "1h")
        assert any("High < Low" in issue for issue in issues)

        gapped_idx = pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 02:00"], tz="UTC",
        )
        gapped = pd.DataFrame({
            "datetime": gapped_idx,
            "open": [100.0] * 2, "high": [101.0] * 2, "low": [99.0] * 2,
            "close": [100.0] * 2, "volume": [1.0] * 2,
        })
        issues = DataValidator.validate(gapped, "BTCUSDT", "1h")
        assert any("time gaps" in issue for issue in issues)

        assert DataValidator.validate(frame.head(0), "BTCUSDT", "1h") == []


class TestNormalizeFundingFrame:
    def test_renames_calc_time_and_funding_rate(self) -> None:
        frame = pd.DataFrame({
            "calc_time": [1704067200000, 1704070800000],
            "fundingRate": [0.0001, 0.0002],
        })
        out = _normalize_funding_frame(frame)
        assert list(out.columns) == ["timestamp", "funding_rate", "datetime"]
        assert len(out) == 2
        assert out["datetime"].dt.tz is not None

    def test_returns_empty_for_unparseable_inputs(self) -> None:
        assert _normalize_funding_frame(pd.DataFrame()).empty
        assert _normalize_funding_frame(pd.DataFrame({"nope": [1]})).empty
        bad = pd.DataFrame({"timestamp": ["x", "y"], "funding_rate": ["a", "b"]})
        assert _normalize_funding_frame(bad).empty


def test_load_cache_raises_on_corrupt_file_without_deleting(tmp_path, monkeypatch) -> None:
    import pytest
    from src.common.errors import DataIntegrityError
    from src.market_data.services.futures_collection import DataCollector

    collector = DataCollector()
    missing = tmp_path / "missing" / "BTCUSDT.parquet"
    monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: missing)
    assert collector._load_cache("BTCUSDT", "1h").empty

    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"not a parquet")
    monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: corrupt)

    with pytest.raises(DataIntegrityError, match="ohlcv cache unreadable"):
        collector._load_cache("BTCUSDT", "1h")

    assert corrupt.read_bytes() == b"not a parquet"


class TestDataCollectorCache:
    def test_load_cache_strips_baggage_columns(self, tmp_path, monkeypatch) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        frame = pd.DataFrame({
            "timestamp": _ms(idx),
            "open": [100.0, 101.0], "high": [101.0, 102.0], "low": [99.0, 100.0],
            "close": [100.5, 101.5], "volume": [1.0, 2.0],
            "close_time": [0, 0], "ignore": [0, 0], "no_trades": [0, 0],
        })
        path = tmp_path / "BTCUSDT.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        collector = DataCollector()
        monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: path)
        out = collector._load_cache("BTCUSDT", "1h")
        assert "close_time" not in out.columns
        assert "ignore" not in out.columns

    def test_normalize_df_coerces_object_columns(self) -> None:
        collector = DataCollector()
        frame = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "open": ["100", "101"],
        })
        out = collector._normalize_df(frame)
        assert pd.api.types.is_numeric_dtype(out["open"])

    def test_load_cache_keeps_file_carrying_both_taker_forms(self, tmp_path, monkeypatch) -> None:
        # Given: an established 1h cache file carrying both taker column forms
        # (Vision rows unsuffixed, REST-appended rows *_volume)
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        path = tmp_path / "X.parquet"
        pd.DataFrame({
            "timestamp": _ms(idx),
            "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0],
            "volume": [1.0, 1.0], "quote_vol": [1.0, 1.0],
            "taker_buy_base": [0.4, float("nan")], "taker_buy_quote": [0.3, float("nan")],
            "taker_buy_base_volume": [float("nan"), 0.6], "taker_buy_quote_volume": [float("nan"), 0.7],
        }).to_parquet(path)
        monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: path)
        collector = DataCollector()
        # When: load and re-save (the daily refresh round trip)
        loaded = collector._load_cache("X", "1h")
        collector._save_cache("X", "1h", loaded)
        # Then: the cache is not discarded and the canonical tail is filled
        assert path.exists()
        persisted = pd.read_parquet(path)
        assert not persisted.columns.duplicated().any()
        assert list(persisted["taker_buy_base"]) == [0.4, 0.6]
        assert list(persisted["taker_buy_quote"]) == [0.3, 0.7]


class TestDataCollectorSaveCache:
    def test_save_cache_1h_matches_store_output(self, tmp_path, monkeypatch) -> None:
        # SC-STORE-03: DataCollector._save_cache delegates to the shared store,
        # so a canonical 1h frame written through the collector is row- and
        # column-equivalent with the canonical futures lake.
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": _ms(idx),
            "open": [100.0] * 4, "high": [101.0] * 4, "low": [99.0] * 4,
            "close": [100.5] * 4, "volume": [10.0] * 4,
            "quote_vol": [1000.0] * 4,
            "taker_buy_base": [5.0] * 4,
            "taker_buy_quote": [500.0] * 4,
            "taker_buy_base_volume": [5.0] * 4,
            "taker_buy_quote_volume": [500.0] * 4,
        })
        target = tmp_path / "futures" / "ohlcv" / "1h" / "BTCUSDT.parquet"
        monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: target)
        DataCollector()._save_cache("BTCUSDT", "1h", df)

        out = pd.read_parquet(target)
        assert list(out.columns) == list(df.columns)
        assert str(out["open"].dtype) == "float32"
        assert str(out["timestamp"].dtype) == "int64"
        assert len(out) == 4

    def test_ensure_ohlcv_data_fetches_and_persists_api_chunk(self, tmp_path, monkeypatch) -> None:
        target = tmp_path / "futures" / "ohlcv" / "1h" / "BTCUSDT.parquet"
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        chunk = pd.DataFrame({
            "timestamp": _ms(idx), "open": [100.0, 101.0], "high": [101.0, 102.0],
            "low": [99.0, 100.0], "close": [100.5, 101.5], "volume": [10.0, 11.0],
        })

        class EmptyVision:
            def fetch_klines_archive_monthly(self, *args, **kwargs):
                return pd.DataFrame()

        collector = DataCollector()
        collector.client.fetch_ohlcv_with_taker = lambda *args, **kwargs: chunk
        monkeypatch.setattr(collector_module, "BinanceVisionDownloader", EmptyVision)
        monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: target)
        collector.ensure_ohlcv_data("BTCUSDT", "1h", "2024-01-01", "2024-01-02")

        assert target.exists()
        assert len(pd.read_parquet(target)) == 2

    def test_ensure_funding_data_fetches_and_persists_api_chunk(self, tmp_path, monkeypatch) -> None:
        target = tmp_path / "futures" / "funding" / "BTCUSDT.parquet"
        funding = pd.DataFrame({
            "timestamp": [1704067200000], "funding_rate": [0.0001],
        })

        class EmptyVision:
            def fetch_funding_rate_monthly(self, *args, **kwargs):
                return pd.DataFrame()

        collector = DataCollector()
        collector.client.fetch_funding_rate_history = lambda *args, **kwargs: funding
        monkeypatch.setattr(collector_module, "BinanceVisionDownloader", EmptyVision)
        monkeypatch.setattr(collector_module, "funding_path", lambda symbol: target)
        target.parent.mkdir(parents=True, exist_ok=True)
        collector.ensure_funding_data("BTCUSDT", "2024-01-01", "2024-01-02")

        assert target.exists()
        assert len(pd.read_parquet(target)) == 1

    def test_save_cache_1m_layout(self, tmp_path, monkeypatch) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": _ms(idx),
            "open": [100.0] * 3, "high": [101.0] * 3, "low": [99.0] * 3,
            "close": [100.5] * 3, "volume": [10.0] * 3,
            "quote_volume": [1000.0] * 3,
        })
        target = tmp_path / "futures" / "ohlcv" / "1m" / "BTCUSDT.parquet"
        monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: target)
        DataCollector()._save_cache("BTCUSDT", "1m", df)

        out = pd.read_parquet(target)
        assert list(out.columns) == [
            "timestamp", "open", "high", "low", "close", "volume",
            "taker_buy_base_volume", "taker_buy_quote_volume", "quote_vol",
        ]
        assert str(out["open"].dtype) == "float32"


def _raw_metrics_frame() -> pd.DataFrame:
    """Vision-style raw daily metrics with one duplicated create_time."""
    return pd.DataFrame({
        "create_time": [
            "2024-01-01 00:00:00", "2024-01-01 00:00:00", "2024-01-02 00:00:00",
        ],
        "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        "sum_open_interest": [1.0, 2.0, 3.0],
        "sum_open_interest_value": [1000.0, 2000.0, 3000.0],
        "count_toptrader_long_short_ratio": [1, 1, 1],
        "sum_toptrader_long_short_ratio": [1.0, 1.0, 1.0],
        "count_long_short_ratio": [1.0, 1.0, 1.0],
        "sum_taker_long_short_vol_ratio": [1.0, 1.0, 1.0],
    })


class TestDataCollectorMetrics:
    def test_ensure_metrics_data_normalizes_duplicates_and_reports_gap(
        self, tmp_path, monkeypatch, caplog,
    ) -> None:
        # FD-02: canonical persistence normalizes duplicates deterministically
        # (keep=last) and surfaces the missing archive date in the coverage report.
        target = tmp_path / "futures" / "metrics" / "1d" / "BTCUSDT.parquet"
        monkeypatch.setattr(collector_module, "metrics_path", lambda symbol: target)
        monkeypatch.setattr(
            collector_module, "fetch_metrics_bulk",
            lambda symbol, start, end: _raw_metrics_frame(),
        )

        collector = DataCollector()
        with caplog.at_level(logging.WARNING, logger="DataCollector"):
            collector.ensure_metrics_data("BTCUSDT", "2024-01-01", "2024-01-03")

        assert target.exists()
        out = pd.read_parquet(target)
        assert set(out.columns) == set(_METRICS_CANONICAL_COLUMNS)
        assert len(out) == 2
        assert out["datetime"].dt.tz is not None
        assert out["datetime"].is_monotonic_increasing
        # the later duplicate wins deterministically
        jan1 = out[out["datetime"].dt.day == 1]
        assert len(jan1) == 1
        assert jan1["sum_open_interest_value"].iloc[0] == 2000.0
        assert (out["available_at"] - out["datetime"]).abs().max() <= pd.Timedelta(minutes=5)
        assert "2024-01-03" in caplog.text

    def test_ensure_metrics_data_raises_on_interior_coverage_gap(
        self, tmp_path, monkeypatch,
    ) -> None:
        # FD-02: a gap strictly inside the collected span is a coverage gap and
        # is never forward-filled; the collector fails closed.
        target = tmp_path / "futures" / "metrics" / "1d" / "BTCUSDT.parquet"
        monkeypatch.setattr(collector_module, "metrics_path", lambda symbol: target)
        gapped = pd.DataFrame({
            "create_time": ["2024-01-01 00:00:00", "2024-01-03 00:00:00"],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "sum_open_interest": [1.0, 3.0],
            "sum_open_interest_value": [1000.0, 3000.0],
            "count_toptrader_long_short_ratio": [1, 1],
            "sum_toptrader_long_short_ratio": [1.0, 1.0],
            "count_long_short_ratio": [1.0, 1.0],
            "sum_taker_long_short_vol_ratio": [1.0, 1.0],
        })
        monkeypatch.setattr(
            collector_module, "fetch_metrics_bulk",
            lambda symbol, start, end: gapped,
        )
        with pytest.raises(DataIntegrityError, match="coverage gap"):
            DataCollector().ensure_metrics_data("BTCUSDT", "2024-01-01", "2024-01-03")

    def test_ensure_metrics_data_merges_with_existing_cache(
        self, tmp_path, monkeypatch,
    ) -> None:
        # FD-02: a fetched archive merges with the canonical cache instead of
        # overwriting it.
        target = tmp_path / "futures" / "metrics" / "1d" / "BTCUSDT.parquet"
        monkeypatch.setattr(collector_module, "metrics_path", lambda symbol: target)
        cached = pd.DataFrame({
            "timestamp": [1704067200000],
            "datetime": [pd.Timestamp("2024-01-01", tz="UTC")],
            "available_at": [pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=5)],
            "symbol": ["BTCUSDT"],
            "sum_open_interest": [1.0],
            "sum_open_interest_value": [1000.0],
            "long_short_ratio": [1.0],
            "top_trader_long_short_ratio": [1.0],
            "sum_taker_long_short_vol_ratio": [1.0],
        })
        DataCollector()._save_metrics_cache("BTCUSDT", cached)
        jan2 = _raw_metrics_frame().iloc[[2]].reset_index(drop=True)
        monkeypatch.setattr(
            collector_module, "fetch_metrics_bulk",
            lambda symbol, start, end: jan2,
        )
        DataCollector().ensure_metrics_data("BTCUSDT", "2024-01-01", "2024-01-02")

        out = pd.read_parquet(target)
        assert set(out["datetime"].dt.day) == {1, 2}
        assert out["sum_open_interest_value"].tolist() == [1000.0, 3000.0]

_RAW_INDICATOR_KLINE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
]


def _raw_indicator_kline_frame() -> pd.DataFrame:
    """Vision-style raw premium/index klines with the always-zero synthetic fields."""
    return pd.DataFrame({
        "timestamp": [1704067200000, 1704067200000 + 4 * 3600 * 1000],
        "open": [100.0, 101.0],
        "high": [101.0, 102.0],
        "low": [99.0, 100.0],
        "close": [100.5, 101.5],
        "volume": [0.0, 0.0],
        "close_time": [1704067200000 + 4 * 3600 * 1000 - 1] * 2,
        "quote_volume": [0.0, 0.0],
        "count": [0, 0],
        "taker_buy_volume": [0.0, 0.0],
        "taker_buy_quote_volume": [0.0, 0.0],
        "ignore": [0, 0],
    })


class TestDataCollectorIndicatorKlines:
    def test_ensure_indicator_kline_data_fetches_and_persists_canonical_columns(
        self, tmp_path, monkeypatch,
    ) -> None:
        # XS-EXP-02 / SCENARIO_INDICATOR_KLINE_COLLECT_02: only the meaningful
        # canonical columns are persisted; the always-zero volume/count/taker-buy
        # fields are dropped, never fabricated.
        target = tmp_path / "futures" / "premiumIndexKlines" / "4h" / "BTCUSDT.parquet"
        monkeypatch.setattr(
            collector_module, "indicator_kline_path",
            lambda dataset, symbol, tf: target,
        )

        class FakeVision:
            def fetch_indicator_klines_monthly(self, dataset, symbol, interval, year, month):
                return _raw_indicator_kline_frame()

        monkeypatch.setattr(collector_module, "BinanceVisionDownloader", FakeVision)
        DataCollector().ensure_indicator_kline_data(
            "premiumIndexKlines", "BTCUSDT", "4h", "2024-01-01", "2024-01-02",
        )

        out = pd.read_parquet(target)
        assert list(out.columns) == [
            "timestamp", "datetime", "open", "high", "low", "close", "close_time",
        ]
        assert len(out) == 2
        assert out["datetime"].dt.tz is not None
        assert out["open"].tolist() == [100.0, 101.0]
        assert out["close"].tolist() == [100.5, 101.5]

    def test_ensure_indicator_kline_data_merges_with_existing_cache(
        self, tmp_path, monkeypatch,
    ) -> None:
        # XS-EXP-02: fetched months merge with the canonical cache instead of
        # overwriting it, deduplicated by timestamp (keep=last).
        target = tmp_path / "futures" / "premiumIndexKlines" / "4h" / "BTCUSDT.parquet"
        monkeypatch.setattr(
            collector_module, "indicator_kline_path",
            lambda dataset, symbol, tf: target,
        )
        cached = pd.DataFrame({
            "timestamp": [1704067200000],
            "datetime": [pd.Timestamp("2024-01-01", tz="UTC")],
            "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5],
            "close_time": [0],
        })
        collector = DataCollector()
        collector._save_indicator_kline_cache("premiumIndexKlines", "BTCUSDT", "4h", cached)
        fetched = _raw_indicator_kline_frame().iloc[[1]].reset_index(drop=True)

        class FakeVision:
            def fetch_indicator_klines_monthly(self, dataset, symbol, interval, year, month):
                return fetched

        monkeypatch.setattr(collector_module, "BinanceVisionDownloader", FakeVision)
        DataCollector().ensure_indicator_kline_data(
            "premiumIndexKlines", "BTCUSDT", "4h", "2024-01-01", "2024-01-02",
        )

        out = pd.read_parquet(target)
        assert set(out["datetime"].dt.day) == {1}
        assert len(out) == 2
        assert out["close"].tolist() == [100.5, 101.5]

    def test_ensure_indicator_kline_data_raises_value_error_on_unsupported_dataset(
        self, tmp_path, monkeypatch,
    ) -> None:
        # XS-EXP-02: the downloader's own allowed-set check fails closed.
        target = tmp_path / "futures" / "bogus" / "4h" / "BTCUSDT.parquet"
        monkeypatch.setattr(
            collector_module, "indicator_kline_path",
            lambda dataset, symbol, tf: target,
        )

        class BogusVision:
            def fetch_indicator_klines_monthly(self, *args, **kwargs):
                raise ValueError("unsupported indicator dataset: bogus")

        monkeypatch.setattr(collector_module, "BinanceVisionDownloader", BogusVision)
        with pytest.raises(ValueError, match="unsupported indicator dataset"):
            DataCollector().ensure_indicator_kline_data(
                "bogus", "BTCUSDT", "4h", "2024-01-01", "2024-01-02",
            )


class TestDataCollectorBookdepth:
    def test_ensure_bookdepth_data_fetches_and_persists_canonical(
        self, tmp_path, monkeypatch,
    ) -> None:
        # XS-EXP-03 / SCENARIO_BOOKDEPTH_COLLECT_03: book depth persists the
        # canonical timestamp/datetime/symbol/percentage/depth/notional schema.
        target = tmp_path / "futures" / "bookdepth" / "BTCUSDT.parquet"
        monkeypatch.setattr(collector_module, "bookdepth_path", lambda symbol: target)
        percentages = [0.0, 0.5, 1.0, 2.0, 5.0]
        depths = [100.0, 90.0, 80.0, 70.0, 60.0]
        notionals = [10000.0, 9000.0, 8000.0, 7000.0, 6000.0]

        class FakeVision:
            def fetch_bookdepth_daily(self, symbol, date, level="5"):
                day = pd.Timestamp(date)
                if day.tzinfo is None:
                    day = day.tz_localize("UTC")
                ts = int(_ms(pd.DatetimeIndex([day]))[0])
                return pd.DataFrame({
                    "timestamp": [ts] * 5,
                    "percentage": percentages,
                    "depth": depths,
                    "notional": notionals,
                })

        monkeypatch.setattr(collector_module, "BinanceVisionDownloader", FakeVision)
        DataCollector().ensure_bookdepth_data("BTCUSDT", "2024-01-01", "2024-01-02")

        out = pd.read_parquet(target)
        assert list(out.columns) == [
            "timestamp", "datetime", "symbol", "percentage", "depth", "notional",
        ]
        assert len(out) == 10  # 5 bands x 2 days
        assert out["symbol"].tolist() == ["BTCUSDT"] * 10
        assert out["datetime"].dt.tz is not None
        assert set(out["datetime"].dt.day) == {1, 2}
        assert out["depth"].tolist() == depths * 2

    def test_ensure_bookdepth_data_fetches_only_missing_days(
        self, tmp_path, monkeypatch,
    ) -> None:
        # XS-EXP-03: days already covered by the cache are not re-fetched.
        target = tmp_path / "futures" / "bookdepth" / "BTCUSDT.parquet"
        monkeypatch.setattr(collector_module, "bookdepth_path", lambda symbol: target)
        cached = pd.DataFrame({
            "timestamp": [1704067200000],
            "datetime": [pd.Timestamp("2024-01-01", tz="UTC")],
            "symbol": ["BTCUSDT"],
            "percentage": [0.0],
            "depth": [100.0],
            "notional": [10000.0],
        })
        DataCollector()._save_bookdepth_cache("BTCUSDT", cached)

        fetched: list[object] = []

        class FakeVision:
            def fetch_bookdepth_daily(self, symbol, date, level="5"):
                fetched.append(date)
                day = pd.Timestamp(date)
                if day.tzinfo is None:
                    day = day.tz_localize("UTC")
                ts = int(_ms(pd.DatetimeIndex([day]))[0])
                return pd.DataFrame({
                    "timestamp": [ts],
                    "percentage": [0.0],
                    "depth": [1.0],
                    "notional": [1.0],
                })

        monkeypatch.setattr(collector_module, "BinanceVisionDownloader", FakeVision)
        DataCollector().ensure_bookdepth_data("BTCUSDT", "2024-01-01", "2024-01-03")

        assert len(fetched) == 2  # 01-02, 01-03 only; 01-01 already cached
        out = pd.read_parquet(target)
        assert sorted(out["datetime"].dt.day.tolist()) == [1, 2, 3]

    def test_ensure_bookdepth_data_fails_closed_on_non_monotonic_band(
        self, tmp_path, monkeypatch,
    ) -> None:
        # XS-EXP-03: timestamps must be monotonic within each (symbol, percentage) band.
        target = tmp_path / "futures" / "bookdepth" / "BTCUSDT.parquet"
        monkeypatch.setattr(collector_module, "bookdepth_path", lambda symbol: target)

        class BadVision:
            def fetch_bookdepth_daily(self, symbol, date, level="5"):
                return pd.DataFrame({
                    "timestamp": [1704067200000, 1704067140000],
                    "percentage": [0.0, 0.0],
                    "depth": [100.0, 90.0],
                    "notional": [10000.0, 9000.0],
                })

        monkeypatch.setattr(collector_module, "BinanceVisionDownloader", BadVision)
        with pytest.raises(DataIntegrityError, match="not monotonic"):
            DataCollector().ensure_bookdepth_data("BTCUSDT", "2024-01-01", "2024-01-02")

    def test_ensure_bookdepth_data_fails_closed_on_duplicate_pairs(
        self, tmp_path, monkeypatch,
    ) -> None:
        # XS-EXP-03: duplicate (timestamp, percentage) pairs fail closed.
        target = tmp_path / "futures" / "bookdepth" / "BTCUSDT.parquet"
        monkeypatch.setattr(collector_module, "bookdepth_path", lambda symbol: target)

        class DupVision:
            def fetch_bookdepth_daily(self, symbol, date, level="5"):
                return pd.DataFrame({
                    "timestamp": [1704067200000, 1704067200000],
                    "percentage": [0.0, 0.0],
                    "depth": [100.0, 101.0],
                    "notional": [10000.0, 10100.0],
                })

        monkeypatch.setattr(collector_module, "BinanceVisionDownloader", DupVision)
        with pytest.raises(DataIntegrityError, match="duplicate"):
            DataCollector().ensure_bookdepth_data("BTCUSDT", "2024-01-01", "2024-01-02")


def test_load_cache_raises_on_frame_without_time_columns_without_deleting(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.market_data.services.futures_collection import DataCollector

    path = tmp_path / "NOTSUSDT.parquet"
    pd.DataFrame({"close": [1.0, 2.0]}).to_parquet(path, index=False)
    before = path.read_bytes()
    monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: path)

    with pytest.raises(DataIntegrityError, match="ohlcv cache invalid"):
        DataCollector()._load_cache("NOTSUSDT", "1h")

    assert path.read_bytes() == before


def test_ensure_ohlcv_data_never_persists_open_bar(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.market_data.services.futures_collection as collector_module
    from src.market_data.services.futures_collection import DataCollector

    epoch = pd.Timestamp("1970-01-01", tz="UTC")

    def _ms(ts: str) -> int:
        return int((pd.Timestamp(ts) - epoch) // pd.Timedelta("1ms"))

    def _rest_frame(stamps: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": [_ms(s) for s in stamps],
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
            "quote_vol": 1.0, "taker_buy_base_volume": 0.5, "taker_buy_quote_volume": 0.5,
        })

    cache = tmp_path / "ohlcv" / "1h" / "XUSDT.parquet"
    monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: cache)
    monkeypatch.setattr(collector_module, "_utc_now", lambda: pd.Timestamp("2026-09-14T01:03:00Z"))
    collector = DataCollector()
    collector.client.fetch_ohlcv_with_taker = lambda *a, **k: _rest_frame(
        ["2026-09-13T23:00Z", "2026-09-14T00:00Z", "2026-09-14T01:00Z"]
    )

    collector.ensure_ohlcv_data("XUSDT", "1h", "2026-09-13T23:00:00Z", "2026-09-14T01:03:00Z")

    persisted = pd.read_parquet(cache)
    assert persisted["timestamp"].tolist() == [_ms("2026-09-13T23:00Z"), _ms("2026-09-14T00:00Z")]


def test_ensure_ohlcv_data_refetches_when_latest_closed_bar_missing_within_8h(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.market_data.services.futures_collection as collector_module
    from src.market_data.services.futures_collection import DataCollector

    epoch = pd.Timestamp("1970-01-01", tz="UTC")

    def _ms(ts: str) -> int:
        return int((pd.Timestamp(ts) - epoch) // pd.Timedelta("1ms"))

    def _rest_frame(stamps: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": [_ms(s) for s in stamps],
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
            "quote_vol": 1.0, "taker_buy_base_volume": 0.5, "taker_buy_quote_volume": 0.5,
        })

    cache = tmp_path / "ohlcv" / "1h" / "XUSDT.parquet"
    monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: cache)
    monkeypatch.setattr(collector_module, "_utc_now", lambda: pd.Timestamp("2026-09-14T01:03:00Z"))
    collector = DataCollector()
    from src.market_data.storage.ohlcv import write_ohlcv

    write_ohlcv(cache, _rest_frame(["2026-09-13T20:00Z", "2026-09-13T21:00Z", "2026-09-13T22:00Z"]), timeframe="1h")
    fetch_calls: list[tuple] = []

    def _fetch(*args, **kwargs):
        fetch_calls.append(args)
        return _rest_frame(["2026-09-13T22:00Z", "2026-09-13T23:00Z", "2026-09-14T00:00Z", "2026-09-14T01:00Z"])

    collector.client.fetch_ohlcv_with_taker = _fetch

    collector.ensure_ohlcv_data("XUSDT", "1h", "2026-09-13T20:00:00Z", "2026-09-14T01:03:00Z")

    assert len(fetch_calls) == 1
    persisted = pd.read_parquet(cache)
    assert persisted["timestamp"].max() == _ms("2026-09-14T00:00Z")
    assert len(persisted) == 5


def test_ensure_ohlcv_data_early_returns_when_latest_closed_bar_present(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.market_data.services.futures_collection as collector_module
    from src.market_data.services.futures_collection import DataCollector

    epoch = pd.Timestamp("1970-01-01", tz="UTC")

    def _ms(ts: str) -> int:
        return int((pd.Timestamp(ts) - epoch) // pd.Timedelta("1ms"))

    def _rest_frame(stamps: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": [_ms(s) for s in stamps],
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
            "quote_vol": 1.0, "taker_buy_base_volume": 0.5, "taker_buy_quote_volume": 0.5,
        })

    cache = tmp_path / "ohlcv" / "1h" / "XUSDT.parquet"
    monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: cache)
    monkeypatch.setattr(collector_module, "_utc_now", lambda: pd.Timestamp("2026-09-14T01:03:00Z"))
    collector = DataCollector()
    from src.market_data.storage.ohlcv import write_ohlcv

    write_ohlcv(cache, _rest_frame(["2026-09-13T22:00Z", "2026-09-13T23:00Z", "2026-09-14T00:00Z"]), timeframe="1h")
    before = cache.read_bytes()

    def _fetch(*args, **kwargs):
        raise AssertionError("must not fetch")

    collector.client.fetch_ohlcv_with_taker = _fetch

    collector.ensure_ohlcv_data("XUSDT", "1h", "2026-09-13T22:00:00Z", "2026-09-14T01:03:00Z")

    assert cache.read_bytes() == before


def test_ensure_ohlcv_data_rejects_unknown_timeframe(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.market_data.services.futures_collection as collector_module
    from src.market_data.services.futures_collection import DataCollector

    epoch = pd.Timestamp("1970-01-01", tz="UTC")

    def _ms(ts: str) -> int:
        return int((pd.Timestamp(ts) - epoch) // pd.Timedelta("1ms"))

    def _rest_frame(stamps: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": [_ms(s) for s in stamps],
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
            "quote_vol": 1.0, "taker_buy_base_volume": 0.5, "taker_buy_quote_volume": 0.5,
        })

    cache = tmp_path / "ohlcv" / "1h" / "XUSDT.parquet"
    monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: cache)
    monkeypatch.setattr(collector_module, "_utc_now", lambda: pd.Timestamp("2026-09-14T01:03:00Z"))
    collector = DataCollector()
    import pytest

    with pytest.raises(ValueError, match="unsupported timeframe"):
        collector.ensure_ohlcv_data("XUSDT", "1M", "2026-09-13T22:00:00Z", "2026-09-14T01:03:00Z")



def test_infer_funding_interval_ms_from_series() -> None:
    from src.market_data.services.futures_collection import FUNDING_DEFAULT_INTERVAL_MS, infer_funding_interval_ms

    h = 3600000
    assert 8 * h == FUNDING_DEFAULT_INTERVAL_MS
    assert infer_funding_interval_ms([]) == 8 * h
    assert infer_funding_interval_ms([5 * h]) == 8 * h
    assert infer_funding_interval_ms([0, 4 * h + 3, 8 * h, 12 * h - 2]) == 4 * h
    assert infer_funding_interval_ms([0, 8 * h, 16 * h, 24 * h]) == 8 * h
    assert infer_funding_interval_ms([0, h, 2 * h, 3 * h, 4 * h]) == h
    # interval switch 8h -> 4h is followed once the recent spacings dominate the median
    series = [0, 8 * h, 16 * h, 20 * h, 24 * h, 28 * h, 32 * h, 36 * h]
    assert infer_funding_interval_ms(series) == 4 * h


def test_last_settled_funding_epoch_respects_publish_grace() -> None:
    import pandas as pd
    from src.market_data.services.futures_collection import FUNDING_SETTLEMENT_GRACE_MS, last_settled_funding_epoch_ms

    h = 3600000
    epoch = pd.Timestamp("1970-01-01", tz="UTC")

    def _ms(ts: str) -> int:
        return int((pd.Timestamp(ts) - epoch) // pd.Timedelta("1ms"))

    assert {"grace_ms": FUNDING_SETTLEMENT_GRACE_MS} == {"grace_ms": 5 * 60_000}
    assert last_settled_funding_epoch_ms(pd.Timestamp("2026-09-14T16:10:00Z"), 4 * h) == _ms("2026-09-14T16:00Z")
    assert last_settled_funding_epoch_ms(pd.Timestamp("2026-09-14T16:03:00Z"), 4 * h) == _ms("2026-09-14T12:00Z")
    assert last_settled_funding_epoch_ms(pd.Timestamp("2026-09-14T16:10:00Z"), 8 * h) == _ms("2026-09-14T16:00Z")
    assert last_settled_funding_epoch_ms(pd.Timestamp("2026-09-14T15:59:00Z"), 8 * h) == _ms("2026-09-14T08:00Z")


def test_funding_tail_is_fresh_detects_missed_settlement() -> None:
    import pandas as pd
    from src.market_data.services.futures_collection import FUNDING_TIME_TOLERANCE_MS, funding_tail_is_fresh

    h = 3600000
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    base = int((pd.Timestamp("2026-09-14T00:00Z") - epoch) // pd.Timedelta("1ms"))
    now = pd.Timestamp("2026-09-14T16:10:00Z")

    assert {"tolerance_ms": FUNDING_TIME_TOLERANCE_MS} == {"tolerance_ms": 60_000}
    assert funding_tail_is_fresh([], now) is False
    assert funding_tail_is_fresh([base, base + 4 * h, base + 8 * h], now) is False
    assert funding_tail_is_fresh([base + 8 * h, base + 12 * h, base + 16 * h], now) is True
    assert funding_tail_is_fresh([base + 8 * h, base + 12 * h, base + 16 * h - 30_000], now) is True


def test_ensure_funding_data_fetches_missed_4h_settlement_within_12h(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.market_data.services.futures_collection as collector_module
    from src.market_data.services.futures_collection import DataCollector

    epoch = pd.Timestamp("1970-01-01", tz="UTC")

    def _ms(ts: str) -> int:
        return int((pd.Timestamp(ts) - epoch) // pd.Timedelta("1ms"))

    class _NoVision:
        def fetch_funding_rate_monthly(self, *args, **kwargs):
            raise AssertionError("vision must not be used for a recent tail")

    target = tmp_path / "funding" / "XUSDT.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(collector_module, "funding_path", lambda symbol: target)
    monkeypatch.setattr(collector_module, "BinanceVisionDownloader", _NoVision)
    monkeypatch.setattr(collector_module, "_utc_now", lambda: pd.Timestamp("2026-09-14T16:10:00Z"))
    collector = DataCollector()
    cached = [_ms("2026-09-14T00:00Z"), _ms("2026-09-14T04:00Z"), _ms("2026-09-14T08:00Z")]
    pd.DataFrame({"timestamp": cached, "funding_rate": [0.0001] * 3}).to_parquet(target, index=False)
    calls: list[tuple] = []

    def _fetch(*args, **kwargs):
        calls.append(args)
        return pd.DataFrame({
            "timestamp": [_ms("2026-09-14T12:00Z"), _ms("2026-09-14T16:00Z")],
            "funding_rate": [0.0002, 0.0003],
        })

    collector.client.fetch_funding_rate_history = _fetch

    collector.ensure_funding_data("XUSDT", "2026-09-14T00:00:00Z", "2026-09-14T16:10:00Z")

    assert len(calls) == 1
    persisted = pd.read_parquet(target)
    assert persisted["timestamp"].tolist() == [*cached, _ms("2026-09-14T12:00Z"), _ms("2026-09-14T16:00Z")]


def test_ensure_funding_data_early_returns_when_last_settlement_present(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.market_data.services.futures_collection as collector_module
    from src.market_data.services.futures_collection import DataCollector

    epoch = pd.Timestamp("1970-01-01", tz="UTC")

    def _ms(ts: str) -> int:
        return int((pd.Timestamp(ts) - epoch) // pd.Timedelta("1ms"))

    class _NoVision:
        def fetch_funding_rate_monthly(self, *args, **kwargs):
            raise AssertionError("vision must not be used for a recent tail")

    target = tmp_path / "funding" / "XUSDT.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(collector_module, "funding_path", lambda symbol: target)
    monkeypatch.setattr(collector_module, "BinanceVisionDownloader", _NoVision)
    monkeypatch.setattr(collector_module, "_utc_now", lambda: pd.Timestamp("2026-09-14T16:10:00Z"))
    collector = DataCollector()
    cached = [_ms("2026-09-14T08:00Z"), _ms("2026-09-14T12:00Z"), _ms("2026-09-14T16:00Z")]
    pd.DataFrame({"timestamp": cached, "funding_rate": [0.0001] * 3}).to_parquet(target, index=False)
    before = target.read_bytes()

    def _fetch(*args, **kwargs):
        raise AssertionError("must not fetch")

    collector.client.fetch_funding_rate_history = _fetch

    collector.ensure_funding_data("XUSDT", "2026-09-14T08:00:00Z", "2026-09-14T16:10:00Z")

    assert target.read_bytes() == before


def test_ensure_funding_data_historical_end_uses_request_end_as_clock(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.market_data.services.futures_collection as collector_module
    from src.market_data.services.futures_collection import DataCollector

    epoch = pd.Timestamp("1970-01-01", tz="UTC")

    def _ms(ts: str) -> int:
        return int((pd.Timestamp(ts) - epoch) // pd.Timedelta("1ms"))

    class _NoVision:
        def fetch_funding_rate_monthly(self, *args, **kwargs):
            raise AssertionError("vision must not be used for a recent tail")

    target = tmp_path / "funding" / "XUSDT.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(collector_module, "funding_path", lambda symbol: target)
    monkeypatch.setattr(collector_module, "BinanceVisionDownloader", _NoVision)
    monkeypatch.setattr(collector_module, "_utc_now", lambda: pd.Timestamp("2026-09-14T16:10:00Z"))
    collector = DataCollector()
    cached = [_ms("2024-01-01T00:00Z"), _ms("2024-01-01T08:00Z"), _ms("2024-01-01T16:00Z")]
    pd.DataFrame({"timestamp": cached, "funding_rate": [0.0001] * 3}).to_parquet(target, index=False)

    def _fetch(*args, **kwargs):
        raise AssertionError("must not fetch")

    collector.client.fetch_funding_rate_history = _fetch

    collector.ensure_funding_data("XUSDT", "2024-01-01", "2024-01-01T20:00:00Z")

    assert pd.read_parquet(target)["timestamp"].tolist() == cached


def test_funding_gap_start_ms_detects_internal_gap_within_window() -> None:
    import pandas as pd
    from src.market_data.services.futures_collection import FUNDING_GAP_THRESHOLD_MS, funding_gap_start_ms

    h = 3_600_000
    base = int(pd.Timestamp("2026-09-01", tz="UTC").value // 10**6)
    four_h = [base + 4 * h * k for k in range(6)]
    after_hole = [base + 48 * h + 4 * h * k for k in range(3)]
    ts = after_hole + four_h + [four_h[0]]

    # Given/When/Then: 8h 표준 최대 간격 + 30분 초과만 공백으로 본다
    assert FUNDING_GAP_THRESHOLD_MS == 8 * h + 30 * 60_000  # noqa: SIM300 -- contract-mandated assert order
    # 창 안의 공백: 공백 직전 행을 돌려준다(정렬/중복 무관)
    assert funding_gap_start_ms(ts, base) == base + 20 * h
    # 창 시작에 걸친 공백: 공백 직전 행(창 이전)을 돌려주고 호출부가 clamp 한다
    assert funding_gap_start_ms(ts, base + 30 * h) == base + 20 * h
    # 창 시작 이전에 끝난 공백은 무시
    assert funding_gap_start_ms(ts, base + 48 * h) is None
    # 8h 간격 + ms 지터, 간격 변경(8h -> 1h)은 공백이 아니다
    eight_h = [base + 8 * h * k + (60_000 if k % 2 else 0) for k in range(5)]
    assert funding_gap_start_ms(eight_h, base) is None
    switch = [base, base + 8 * h, base + 16 * h, base + 17 * h, base + 18 * h]
    assert funding_gap_start_ms(switch, base) is None
    assert funding_gap_start_ms([], base) is None
    assert funding_gap_start_ms([base], base) is None

def test_ensure_funding_data_heals_internal_gap_despite_fresh_tail(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.market_data.services.futures_collection as collector_module
    from src.market_data.services.futures_collection import DataCollector, funding_gap_start_ms

    def _ms(ts: str) -> int:
        return int(pd.Timestamp(ts).value // 10**6)

    class _NoVision:
        def fetch_funding_rate_monthly(self, *args, **kwargs):
            raise AssertionError("vision must not be used for a recent window")

    target = tmp_path / "funding" / "XUSDT.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(collector_module, "funding_path", lambda symbol: target)
    monkeypatch.setattr(collector_module, "BinanceVisionDownloader", _NoVision)
    monkeypatch.setattr(collector_module, "_utc_now", lambda: pd.Timestamp("2026-09-14T16:10:00Z"))
    collector = DataCollector()
    # Given: 꼬리는 최신(09-14 16:00)이지만 09-11 00:00 ~ 09-13 00:00 사이 정산이 통째로 빠진 캐시
    before_hole = [_ms(t) for t in pd.date_range("2026-09-10", "2026-09-11", freq="4h", tz="UTC")]
    after_hole = [_ms(t) for t in pd.date_range("2026-09-13", "2026-09-14T16:00", freq="4h", tz="UTC")]
    cached = before_hole + after_hole
    pd.DataFrame({"timestamp": cached, "funding_rate": [0.0001] * len(cached)}).to_parquet(target, index=False)
    missing = [_ms(t) for t in pd.date_range("2026-09-11T04:00", "2026-09-12T20:00", freq="4h", tz="UTC")]
    calls: list[tuple] = []

    def _fetch(*args, **kwargs):
        calls.append(args)
        return pd.DataFrame({"timestamp": missing, "funding_rate": [0.0002] * len(missing)})

    collector.client.fetch_funding_rate_history = _fetch

    # When
    collector.ensure_funding_data("XUSDT", "2026-09-10T00:00:00Z", "2026-09-14T16:10:00Z")

    # Then: 공백 직전 행부터 재조회하고, 결과는 창 안에서 연속이다
    assert len(calls) == 1
    assert calls[0][1] == str(pd.Timestamp("2026-09-11T00:00:00Z"))
    persisted = pd.read_parquet(target)["timestamp"].tolist()
    assert persisted == sorted(cached + missing)
    assert funding_gap_start_ms(persisted, _ms("2026-09-10T00:00Z")) is None

def test_ensure_funding_data_ignores_gap_before_request_window(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.market_data.services.futures_collection as collector_module
    from src.market_data.services.futures_collection import DataCollector

    def _ms(ts: str) -> int:
        return int(pd.Timestamp(ts).value // 10**6)

    class _NoVision:
        def fetch_funding_rate_monthly(self, *args, **kwargs):
            raise AssertionError("vision must not be used for a recent window")

    target = tmp_path / "funding" / "XUSDT.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(collector_module, "funding_path", lambda symbol: target)
    monkeypatch.setattr(collector_module, "BinanceVisionDownloader", _NoVision)
    monkeypatch.setattr(collector_module, "_utc_now", lambda: pd.Timestamp("2026-09-14T16:10:00Z"))
    collector = DataCollector()
    # Given: 공백은 요청 창(09-13 00:00~) 이전에만 있고 창 안은 연속, 꼬리도 최신
    before_hole = [_ms(t) for t in pd.date_range("2026-09-10", "2026-09-11", freq="4h", tz="UTC")]
    in_window = [_ms(t) for t in pd.date_range("2026-09-13", "2026-09-14T16:00", freq="4h", tz="UTC")]
    cached = before_hole + in_window
    pd.DataFrame({"timestamp": cached, "funding_rate": [0.0001] * len(cached)}).to_parquet(target, index=False)
    before = target.read_bytes()

    def _fetch(*args, **kwargs):
        raise AssertionError("must not fetch when the requested window is contiguous and fresh")

    collector.client.fetch_funding_rate_history = _fetch

    # When
    collector.ensure_funding_data("XUSDT", "2026-09-13T00:00:00Z", "2026-09-14T16:10:00Z")

    # Then
    assert target.read_bytes() == before

