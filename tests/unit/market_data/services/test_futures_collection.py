from __future__ import annotations

import logging

import pandas as pd

import src.market_data.services.futures_collection as collector_module
from src.market_data.binance.futures import BinanceKlinePermanentError
from src.market_data.services.futures_collection import (
    DataCollector,
    _fetch_months_parallel,
    _normalize_funding_frame,
    _vision_months_to_fetch,
)

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ms(index: pd.DatetimeIndex) -> pd.Series:
    return (index - _EPOCH) // pd.Timedelta("1ms")


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


def test_ensure_funding_data_refetches_vision_month_hidden_by_wide_span_gap(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.market_data.services.futures_collection as collector_module
    from src.market_data.services.futures_collection import DataCollector

    def _ms(ts: str) -> int:
        return int(pd.Timestamp(ts).value // 10**6)

    target = tmp_path / "funding" / "XUSDT.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(collector_module, "funding_path", lambda symbol: target)
    monkeypatch.setattr(collector_module, "_utc_now", lambda: pd.Timestamp("2026-09-14T00:00:00Z"))
    collector = DataCollector()
    # Given: 캐시 전체 범위(min/max)는 4월~8월을 덮지만, 5월 한 달이 통째로 비어 있다
    # (min/max span 검사만으로는 이 내부공백을 '이미 커버됨'으로 오판한다).
    april = [_ms(t) for t in pd.date_range("2026-04-01", "2026-04-30", freq="8h", tz="UTC")]
    august = [_ms(t) for t in pd.date_range("2026-08-01", "2026-08-31", freq="8h", tz="UTC")]
    cached = april + august
    pd.DataFrame({"timestamp": cached, "funding_rate": [0.0001] * len(cached)}).to_parquet(target, index=False)
    requested_months: list[tuple[int, int]] = []

    class _RecordingVision:
        def fetch_funding_rate_monthly(self, symbol, year, month):
            requested_months.append((year, month))
            if (year, month) == (2026, 5):
                may = [_ms(t) for t in pd.date_range("2026-05-01", "2026-05-31", freq="8h", tz="UTC")]
                return pd.DataFrame({"timestamp": may, "funding_rate": [0.0003] * len(may)})
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

    monkeypatch.setattr(collector_module, "BinanceVisionDownloader", _RecordingVision)

    # REST는 스팬 전체를 대상으로 한 번 더 보정 조회를 시도할 수 있다(기존 동작,
    # 이 테스트의 관심사는 vision_tasks가 5월을 놓치지 않는지이다); 빈 응답이면 무해하다.
    collector.client.fetch_funding_rate_history = lambda *a, **k: pd.DataFrame(columns=["timestamp", "funding_rate"])

    # When
    collector.ensure_funding_data("XUSDT", "2026-04-01T00:00:00Z", "2026-08-31T00:00:00Z")

    # Then: 5월이 vision_tasks에 포함되고, 영구 누락 없이 채워진다
    assert (2026, 5) in requested_months
    persisted = pd.read_parquet(target)
    may_start_ms = _ms("2026-05-01T00:00:00Z")
    may_end_ms = _ms("2026-06-01T00:00:00Z")
    may_rows = persisted[(persisted["timestamp"] >= may_start_ms) & (persisted["timestamp"] < may_end_ms)]
    assert len(may_rows) > 0



def test_ohlcv_gap_start_ms_detects_only_in_window_internal_gaps() -> None:
    import numpy as np
    from src.market_data.services.futures_collection import ohlcv_gap_start_ms
    h = 3_600_000
    continuous = np.arange(0, 10 * h, h, dtype='int64')
    assert ohlcv_gap_start_ms(continuous, 0, h) is None
    holed = np.concatenate([np.arange(0, 5 * h, h), np.arange(8 * h, 12 * h, h)]).astype('int64')
    # gap (4h -> 8h) right edge lies after the window start -> left edge returned
    assert ohlcv_gap_start_ms(holed, 0, h) == 4 * h
    assert ohlcv_gap_start_ms(holed, 7 * h, h) == 4 * h
    # a window starting at/after the right edge ignores the earlier gap
    assert ohlcv_gap_start_ms(holed, 8 * h, h) is None
    # empty and single-row inputs have no gap
    assert ohlcv_gap_start_ms(np.array([], dtype='int64'), 0, h) is None
    assert ohlcv_gap_start_ms(np.array([5 * h], dtype='int64'), 0, h) is None


def test_ensure_ohlcv_data_refetches_vision_month_hidden_by_internal_gap(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.market_data.services.futures_collection as collector_module
    from src.market_data.services.futures_collection import DataCollector
    from src.market_data.storage.ohlcv import write_ohlcv

    def _ms(ts) -> int:
        return int(pd.Timestamp(ts).value // 10**6)

    def _frame(stamps) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": [_ms(s) for s in stamps],
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
            "quote_vol": 1.0, "taker_buy_base_volume": 0.5, "taker_buy_quote_volume": 0.5,
        })

    cache = tmp_path / "ohlcv" / "1h" / "XUSDT.parquet"
    monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: cache)
    monkeypatch.setattr(collector_module, "_utc_now", lambda: pd.Timestamp("2026-09-14T00:00:00Z"))
    collector = DataCollector()
    # Given: min/max span covers April..June but all of May is missing
    april = pd.date_range("2026-04-01", "2026-04-30 23:00", freq="1h", tz="UTC")
    june = pd.date_range("2026-06-01", "2026-06-30 23:00", freq="1h", tz="UTC")
    write_ohlcv(cache, _frame(list(april) + list(june)), timeframe="1h")
    requested: list[tuple[int, int]] = []
    may = pd.date_range("2026-05-01", "2026-05-31 23:00", freq="1h", tz="UTC")

    class _RecordingVision:
        def fetch_klines_archive_monthly(self, symbol, timeframe, year, month):
            requested.append((year, month))
            if (year, month) != (2026, 5):
                return pd.DataFrame()
            n = len(may)
            return pd.DataFrame({
                "timestamp": [_ms(t) for t in may], "open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n,
                "close": [1.0] * n, "volume": [1.0] * n, "close_time": [0] * n, "quote_vol": [1.0] * n,
                "no_trades": [1] * n, "taker_buy_base": [0.5] * n, "taker_buy_quote": [0.5] * n, "ignore": [0] * n,
            })

    monkeypatch.setattr(collector_module, "BinanceVisionDownloader", _RecordingVision)
    collector.client.fetch_ohlcv_with_taker = lambda *a, **k: pd.DataFrame()
    # When
    collector.ensure_ohlcv_data("XUSDT", "1h", "2026-04-01T00:00:00Z", "2026-06-30T23:00:00Z")
    # Then: May is requested from Vision and persisted
    assert (2026, 5) in requested
    persisted = pd.read_parquet(cache)
    may_rows = persisted[(persisted["timestamp"] >= _ms("2026-05-01T00:00Z")) & (persisted["timestamp"] < _ms("2026-06-01T00:00Z"))]
    assert len(may_rows) == len(may)


class TestVisionMonthsToFetch:
    _NOW = pd.Timestamp("2026-09-14T00:00:00Z")

    def test_empty_cache_plans_every_month_before_cutoff(self) -> None:
        def _probe(_: int) -> int | None:
            raise AssertionError("probe must not be called for empty cache")

        out = _vision_months_to_fetch(
            pd.Timestamp("2026-04-15T00:00:00Z"),
            pd.Timestamp("2026-09-01T00:00:00Z"),
            self._NOW,
            None,
            _probe,
        )
        assert out == [(2026, 4), (2026, 5), (2026, 6), (2026, 7)]

    def test_contiguous_covered_months_are_skipped(self) -> None:
        out = _vision_months_to_fetch(
            pd.Timestamp("2026-04-01T00:00:00Z"),
            pd.Timestamp("2026-08-01T00:00:00Z"),
            self._NOW,
            (pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
            lambda _: None,
        )
        assert out == []

    def test_interior_gap_replans_touched_months(self) -> None:
        import numpy as np

        from src.market_data.services.futures_collection import ohlcv_gap_start_ms

        april = pd.date_range("2026-04-01", "2026-04-30 23:00", freq="1h", tz="UTC")
        june = pd.date_range("2026-06-01", "2026-07-01", freq="1h", tz="UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        idx = pd.DatetimeIndex(list(april) + list(june))
        ts = np.unique(((idx - epoch) // pd.Timedelta("1ms")).to_numpy(dtype="int64"))
        out = _vision_months_to_fetch(
            pd.Timestamp("2026-04-01T00:00:00Z"),
            pd.Timestamp("2026-07-01T00:00:00Z"),
            self._NOW,
            (pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-07-01T00:00:00Z")),
            lambda s: ohlcv_gap_start_ms(ts, s, 3_600_000),
        )
        assert out == [(2026, 4), (2026, 5)]


class TestFetchMonthsParallel:
    def test_no_months_means_no_fetch(self) -> None:
        def _fetch(_y: int, _m: int) -> pd.DataFrame:
            raise AssertionError("must not be called")

        assert _fetch_months_parallel([], _fetch, lambda *_: None) == []

    def test_failing_month_is_isolated_and_reported(self) -> None:
        frames = {
            (2026, 4): pd.DataFrame({"timestamp": [1]}),
            (2026, 6): pd.DataFrame({"timestamp": [3]}),
        }

        def _fetch(year: int, month: int) -> pd.DataFrame:
            if (year, month) == (2026, 5):
                raise RuntimeError("boom")
            if (year, month) == (2026, 7):
                return pd.DataFrame()
            return frames[(year, month)]

        errors: list[tuple[int, int, Exception]] = []
        out = _fetch_months_parallel(
            [(2026, 4), (2026, 5), (2026, 6), (2026, 7)],
            _fetch,
            lambda y, m, exc: errors.append((y, m, exc)),
        )
        assert [f["timestamp"].iloc[0] for f in out] == [1, 3]
        assert len(errors) == 1
        assert errors[0][:2] == (2026, 5)
        assert isinstance(errors[0][2], RuntimeError)

    def test_base_exception_propagates(self) -> None:
        import pytest

        def _fetch(_y: int, _m: int) -> pd.DataFrame:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            _fetch_months_parallel([(2026, 4)], _fetch, lambda *_: None)


class TestVisionMonthsToFetchBoundaries:
    _NOW = pd.Timestamp("2026-09-14T00:00:00Z")

    def test_request_end_bounds_plan_before_cutoff(self) -> None:
        out = _vision_months_to_fetch(
            pd.Timestamp("2026-04-15T00:00:00Z"), pd.Timestamp("2026-06-01T00:00:00Z"),
            self._NOW, None, lambda _: None,
        )
        assert out == [(2026, 4), (2026, 5)]

    def test_window_younger_than_cutoff_plans_nothing(self) -> None:
        out = _vision_months_to_fetch(
            pd.Timestamp("2026-08-01T00:00:00Z"), pd.Timestamp("2026-09-14T00:00:00Z"),
            self._NOW, None, lambda _: None,
        )
        assert out == []

    def test_span_edges_replan_boundary_months(self) -> None:
        out = _vision_months_to_fetch(
            pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-07-01T00:00:00Z"),
            self._NOW,
            (pd.Timestamp("2026-04-02T00:00:00Z"), pd.Timestamp("2026-06-30T22:00:00Z")),
            lambda _: None,
        )
        assert out == [(2026, 4), (2026, 6)]

    def test_gap_opening_before_month_replans_it(self) -> None:
        start = pd.Timestamp("2026-05-01T00:00:00Z")
        out = _vision_months_to_fetch(
            start, pd.Timestamp("2026-06-01T00:00:00Z"), self._NOW,
            (pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-07-01T00:00:00Z")),
            lambda s: s - 86_400_000,
        )
        assert out == [(2026, 5)]

    def test_plan_is_deterministic_and_ascending(self) -> None:
        args = (
            pd.Timestamp("2026-01-10T00:00:00Z"), pd.Timestamp("2026-09-01T00:00:00Z"),
            self._NOW, None, lambda _: None,
        )
        first = _vision_months_to_fetch(*args)
        assert first == _vision_months_to_fetch(*args)
        assert first == sorted(set(first))


class TestFetchMonthsParallelConcurrency:
    def test_results_follow_plan_order_not_completion_order(self) -> None:
        import threading

        later_done = threading.Event()
        later_count: list[int] = []
        lock = threading.Lock()

        def _fetch(_y: int, month: int) -> pd.DataFrame:
            if month == 4:
                assert later_done.wait(timeout=5)
            else:
                with lock:
                    later_count.append(month)
                    if len(later_count) == 2:
                        later_done.set()
            return pd.DataFrame({"m": [month]})

        out = _fetch_months_parallel([(2026, 4), (2026, 5), (2026, 6)], _fetch, lambda *_: None)
        assert [int(f["m"].iloc[0]) for f in out] == [4, 5, 6]

    def test_concurrency_never_exceeds_worker_bound(self) -> None:
        import threading
        import time

        lock = threading.Lock()
        state = {"now": 0, "peak": 0}

        def _fetch(_y: int, m: int) -> pd.DataFrame:
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.02)
            with lock:
                state["now"] -= 1
            return pd.DataFrame({"m": [m]})

        n = collector_module._VISION_FETCH_MAX_WORKERS + 2
        out = _fetch_months_parallel([(2026, m) for m in range(1, n + 1)], _fetch, lambda *_: None)
        assert len(out) == n
        assert state["peak"] <= collector_module._VISION_FETCH_MAX_WORKERS


def test_vision_downloader_not_constructed_for_recent_only_windows(tmp_path, monkeypatch) -> None:
    class ForbiddenVision:
        def __init__(self) -> None:
            raise AssertionError("Vision downloader must not be constructed")

    now = pd.Timestamp("2026-09-14T00:00:00Z")
    monkeypatch.setattr(collector_module, "_utc_now", lambda: now)
    monkeypatch.setattr(collector_module, "BinanceVisionDownloader", ForbiddenVision)
    ohlcv_target = tmp_path / "futures" / "ohlcv" / "1h" / "BTCUSDT.parquet"
    funding_target = tmp_path / "futures" / "funding" / "BTCUSDT.parquet"
    funding_target.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: ohlcv_target)
    monkeypatch.setattr(collector_module, "funding_path", lambda symbol: funding_target)
    idx = pd.date_range("2026-09-01", periods=2, freq="1h", tz="UTC")
    chunk = pd.DataFrame({
        "timestamp": _ms(idx), "open": [1.0, 1.0], "high": [1.0, 1.0],
        "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1.0, 1.0],
    })
    collector = DataCollector()
    fetched = {"ohlcv": 0, "funding": 0}

    def _ohlcv(*_a, **_k):
        fetched["ohlcv"] += 1
        return chunk

    def _funding(*_a, **_k):
        fetched["funding"] += 1
        return pd.DataFrame({"timestamp": [1788220800000], "funding_rate": [0.0001]})

    collector.client.fetch_ohlcv_with_taker = _ohlcv
    collector.client.fetch_funding_rate_history = _funding
    collector.ensure_ohlcv_data("BTCUSDT", "1h", "2026-09-01", "2026-09-02")
    collector.ensure_funding_data("BTCUSDT", "2026-09-01", "2026-09-02")
    assert fetched == {"ohlcv": 1, "funding": 1}


_WIRING_NOW = pd.Timestamp("2026-09-14T00:10:00Z")


def _wiring_ms(ts: str) -> int:
    return int(pd.Timestamp(ts).value // 10**6)


def _wiring_klines(stamps: list[str], close: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [_wiring_ms(s) for s in stamps],
        "open": 1.0, "high": 2.0, "low": 1.0, "close": close, "volume": 1.0,
        "quote_vol": 1.0, "taker_buy_base_volume": 0.5, "taker_buy_quote_volume": 0.5,
    })


def _wiring_collector(tmp_path, monkeypatch, vision: type) -> tuple[DataCollector, object, object]:
    ohlcv_target = tmp_path / "futures" / "ohlcv" / "1h" / "XUSDT.parquet"
    funding_target = tmp_path / "futures" / "funding" / "XUSDT.parquet"
    funding_target.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(collector_module, "_utc_now", lambda: _WIRING_NOW)
    monkeypatch.setattr(collector_module, "BinanceVisionDownloader", vision)
    monkeypatch.setattr(DataCollector, "_cache_path", lambda self, symbol, tf: ohlcv_target)
    monkeypatch.setattr(collector_module, "funding_path", lambda symbol: funding_target)
    return DataCollector(), ohlcv_target, funding_target


class _FailingVision:
    def fetch_klines_archive_monthly(self, symbol, timeframe, year, month):
        raise RuntimeError(f"archive {year}-{month:02d} unavailable")

    def fetch_funding_rate_monthly(self, symbol, year, month):
        raise RuntimeError(f"archive {year}-{month:02d} unavailable")


def test_ohlcv_rest_tail_overrides_cached_duplicate_bar(tmp_path, monkeypatch) -> None:
    from src.market_data.storage.ohlcv import write_ohlcv

    collector, cache, _ = _wiring_collector(tmp_path, monkeypatch, _FailingVision)
    # Given: cached bar T=22:00 close=1.0; REST tail re-serves T (closed) with close=2.0
    write_ohlcv(cache, _wiring_klines(["2026-09-13T20:00Z", "2026-09-13T21:00Z", "2026-09-13T22:00Z"]), timeframe="1h")
    collector.client.fetch_ohlcv_with_taker = lambda *a, **k: _wiring_klines(
        ["2026-09-13T22:00Z", "2026-09-13T23:00Z"], close=2.0
    )
    # When
    collector.ensure_ohlcv_data("XUSDT", "1h", "2026-09-13T20:00:00Z", "2026-09-14T00:10:00Z")
    # Then: REST wins (keep="last") and T is not duplicated (write_ohlcv does not dedupe 1h)
    persisted = pd.read_parquet(cache)
    t_ms = _wiring_ms("2026-09-13T22:00Z")
    assert persisted["timestamp"].is_unique
    assert persisted.loc[persisted["timestamp"] == t_ms, "close"].tolist() == [2.0]
    assert persisted["timestamp"].max() == _wiring_ms("2026-09-13T23:00Z")


def test_funding_cache_wins_over_refetched_duplicate_settlement(tmp_path, monkeypatch) -> None:
    collector, _, target = _wiring_collector(tmp_path, monkeypatch, _FailingVision)
    # Given: cache ends at T=16:00 (0.0001); the 00:00 settlement is missing, so the tail is stale
    cached = ["2026-09-13T00:00Z", "2026-09-13T08:00Z", "2026-09-13T16:00Z"]
    pd.DataFrame({"timestamp": [_wiring_ms(s) for s in cached], "funding_rate": [0.0001] * 3}).to_parquet(target, index=False)
    rest_calls: list[tuple] = []

    def _rest(*args, **kwargs) -> pd.DataFrame:
        rest_calls.append(args)
        return pd.DataFrame({
            "timestamp": [_wiring_ms("2026-09-13T16:00Z"), _wiring_ms("2026-09-14T00:00Z")],
            "funding_rate": [0.0009, 0.0002],
        })

    collector.client.fetch_funding_rate_history = _rest
    # When
    collector.ensure_funding_data("XUSDT", "2026-09-13T00:00:00Z", "2026-09-14T00:10:00Z")
    # Then: cached rate at T survives (keep="first") and the newer settlement is appended
    assert len(rest_calls) == 1
    persisted = pd.read_parquet(target).set_index("timestamp")["funding_rate"]
    assert persisted.index.is_unique
    assert persisted.loc[_wiring_ms("2026-09-13T16:00Z")] == 0.0001
    assert persisted.loc[_wiring_ms("2026-09-14T00:00Z")] == 0.0002
    assert len(persisted) == 4


def test_ohlcv_vision_month_failure_is_logged_and_run_completes(tmp_path, monkeypatch, caplog) -> None:
    collector, cache, _ = _wiring_collector(tmp_path, monkeypatch, _FailingVision)
    # Given: empty cache; July 2026 is before the archive cutoff (2026-07-31), so it is planned
    collector.client.fetch_ohlcv_with_taker = lambda *a, **k: _wiring_klines(["2026-07-31T00:00Z", "2026-07-31T01:00Z"])
    caplog.set_level(logging.WARNING, logger="DataCollector")
    # When
    collector.ensure_ohlcv_data("XUSDT", "1h", "2026-07-31T00:00:00Z", "2026-07-31T02:00:00Z")
    # Then
    warnings = [r for r in caplog.records if r.name == "DataCollector" and r.levelno == logging.WARNING]
    assert any("Error fetching vision data for" in r.getMessage() for r in warnings)
    assert pd.read_parquet(cache)["timestamp"].tolist() == [_wiring_ms("2026-07-31T00:00Z"), _wiring_ms("2026-07-31T01:00Z")]


def test_funding_vision_month_failure_is_logged_and_run_completes(tmp_path, monkeypatch, caplog) -> None:
    collector, _, target = _wiring_collector(tmp_path, monkeypatch, _FailingVision)
    rest = [_wiring_ms("2026-07-31T00:00Z"), _wiring_ms("2026-07-31T08:00Z")]
    collector.client.fetch_funding_rate_history = lambda *a, **k: pd.DataFrame(
        {"timestamp": rest, "funding_rate": [0.0001, 0.0002]}
    )
    caplog.set_level(logging.WARNING, logger="DataCollector")
    # When
    collector.ensure_funding_data("XUSDT", "2026-07-31T00:00:00Z", "2026-07-31T16:00:00Z")
    # Then
    warnings = [r for r in caplog.records if r.name == "DataCollector" and r.levelno == logging.WARNING]
    assert any("Error fetching vision funding data for" in r.getMessage() for r in warnings)
    assert pd.read_parquet(target)["timestamp"].tolist() == rest


def test_ohlcv_permanent_kline_error_is_logged_not_raised(tmp_path, monkeypatch, caplog) -> None:
    july = ["2026-07-30T22:00Z", "2026-07-30T23:00Z"]

    class _JulyVision:
        def fetch_klines_archive_monthly(self, symbol, timeframe, year, month):
            assert (year, month) == (2026, 7)
            n = len(july)
            return pd.DataFrame({
                "timestamp": [_wiring_ms(s) for s in july], "open": [1.0] * n, "high": [1.0] * n,
                "low": [1.0] * n, "close": [1.0] * n, "volume": [1.0] * n, "close_time": [0] * n,
                "quote_vol": [1.0] * n, "no_trades": [1] * n, "taker_buy_base": [0.5] * n,
                "taker_buy_quote": [0.5] * n, "ignore": [0] * n,
            })

    collector, cache, _ = _wiring_collector(tmp_path, monkeypatch, _JulyVision)
    rest_calls: list[tuple] = []

    def _rest(*args, **kwargs) -> pd.DataFrame:
        rest_calls.append(args)
        raise BinanceKlinePermanentError(
            symbol="XUSDT", timeframe="1h", http_code=400, start_time_ms=0, end_time_ms=0, url="u",
        )

    collector.client.fetch_ohlcv_with_taker = _rest
    caplog.set_level(logging.WARNING, logger="DataCollector")
    # When
    collector.ensure_ohlcv_data("XUSDT", "1h", "2026-07-30T22:00:00Z", "2026-08-01T00:00:00Z")
    # Then
    assert len(rest_calls) == 1
    warnings = [r for r in caplog.records if r.name == "DataCollector" and r.levelno == logging.WARNING]
    assert any("Permanent OHLCV API failure" in r.getMessage() for r in warnings)
    assert pd.read_parquet(cache)["timestamp"].tolist() == [_wiring_ms(s) for s in july]
