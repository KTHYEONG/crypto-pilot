from __future__ import annotations

import pandas as pd

import src.market_data.services.futures_collection as collector_module
from src.market_data.services.futures_collection import (
    DataCollector,
    DataValidator,
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


class TestDataCollectorCache:
    def test_load_cache_returns_empty_for_missing_and_corrupt(self, tmp_path, monkeypatch) -> None:
        collector = DataCollector()
        missing = tmp_path / "missing" / "BTCUSDT.parquet"
        monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: missing)
        assert collector._load_cache("BTCUSDT", "1h").empty

        corrupt = tmp_path / "corrupt.parquet"
        corrupt.parent.mkdir(parents=True, exist_ok=True)
        corrupt.write_bytes(b"not a parquet")
        monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: corrupt)
        assert collector._load_cache("BTCUSDT", "1h").empty

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
