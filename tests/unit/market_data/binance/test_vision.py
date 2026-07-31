from __future__ import annotations

import io
import zipfile
from datetime import datetime

import pandas as pd
import pytest

from src.market_data.binance.vision import (
    BinanceVisionDownloader,
    fetch_metrics_bulk,
)


def _zip_of_csv(csv_body: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("data.csv", csv_body)
    return buffer.getvalue()


@pytest.fixture
def downloader(monkeypatch) -> BinanceVisionDownloader:
    d = BinanceVisionDownloader()
    monkeypatch.setattr(d, "min_request_interval_seconds", 0.0)
    monkeypatch.setattr(d, "_next_request_monotonic", 0.0)
    return d


def test_env_parsers_apply_bounds(monkeypatch) -> None:
    monkeypatch.setenv("BINANCE_VISION_MAX_CONCURRENCY", "0")
    d = BinanceVisionDownloader()
    assert d.max_concurrency == 1
    monkeypatch.setenv("BINANCE_VISION_MAX_WEIGHT_PER_MIN", "not-a-number")
    assert d.max_weight_per_min == BinanceVisionDownloader.DEFAULT_MAX_WEIGHT_PER_MIN
    monkeypatch.setenv("BINANCE_VISION_MAX_RETRIES", "3")
    assert BinanceVisionDownloader().max_retries == 3


def test_retryability_classification() -> None:
    import urllib.error

    d = BinanceVisionDownloader()
    assert d._is_retryable_http_error(urllib.error.HTTPError("u", 429, "x", {}, None))
    assert d._is_retryable_http_error(urllib.error.HTTPError("u", 500, "x", {}, None))
    assert not d._is_retryable_http_error(urllib.error.HTTPError("u", 404, "x", {}, None))


def test_parse_retry_after_seconds() -> None:
    d = BinanceVisionDownloader()
    assert d._parse_retry_after_seconds("5") == 5.0
    assert d._parse_retry_after_seconds("garbage") is None
    assert d._parse_retry_after_seconds(None) is None


def test_verify_checksum() -> None:
    import hashlib

    d = BinanceVisionDownloader()
    payload = b"hello vision"
    digest = hashlib.sha256(payload).hexdigest()
    assert d.verify_checksum(payload, digest)
    assert not d.verify_checksum(b"tampered", digest)
    with pytest.raises(ValueError, match="algorithm"):
        d.verify_checksum(payload, digest, algorithm="md5-extra")


def test_fetch_zip_csv_parses_header_and_rows(downloader, monkeypatch) -> None:
    body = _zip_of_csv("calc_time,symbol\n2024-01-01 00:00:00,BTCUSDT\n")
    monkeypatch.setattr(downloader, "_read_url_bytes", lambda url, timeout=None: body)
    frame = downloader._fetch_zip_csv("https://data.binance.vision/x.zip")
    assert len(frame) == 1


def test_fetch_zip_by_path_returns_empty_on_404(downloader, monkeypatch) -> None:
    import urllib.error

    def _raise_404(url, timeout=None):
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)

    monkeypatch.setattr(downloader, "_read_url_bytes", _raise_404)
    assert downloader._fetch_zip_by_path("monthly", "klines", "BTCUSDT", "1h", "x.zip").empty


def test_monthly_archive_url_builders(downloader, monkeypatch) -> None:
    called: list[str] = []

    def _fetch(*parts: str) -> pd.DataFrame:
        called.append("/".join(parts))
        return pd.DataFrame({"open": [1.0]})

    monkeypatch.setattr(downloader, "_fetch_zip_by_path", _fetch)
    assert len(downloader.fetch_klines_archive_monthly("BTCUSDT", "1h", 2024, 1)) == 1
    assert len(downloader.fetch_klines_archive("BTCUSDT", "1h", 2024, 1)) == 1
    assert len(downloader.fetch_funding_rate_monthly("BTCUSDT", 2024, 1)) == 1
    assert len(downloader.fetch_funding_monthly("BTCUSDT", 2024, 1)) == 1
    assert len(downloader.fetch_indicator_klines_monthly("markPriceKlines", "BTCUSDT", "1h", 2024, 1)) == 1
    assert len(downloader.fetch_bookdepth_daily("BTCUSDT", datetime(2024, 1, 1))) == 1
    assert len(downloader.fetch_premiumindex_daily("BTCUSDT", datetime(2024, 1, 1))) == 1
    with pytest.raises(ValueError, match="unsupported indicator"):
        downloader.fetch_indicator_klines_monthly("bad", "BTCUSDT", "1h", 2024, 1)


def test_s3_listing_parses_symbols(downloader, monkeypatch) -> None:
    ns = "http://s3.amazonaws.com/doc/2006-03-01/"
    body = (
        f'<ListBucketResult xmlns="{ns}">'.encode()
        + b'<CommonPrefixes><Prefix>data/futures/um/daily/klines/ETHUSDT/</Prefix></CommonPrefixes>'
        + b'</ListBucketResult>'
    )
    monkeypatch.setattr(downloader, "_read_url_bytes", lambda url, timeout=None: body)
    assert downloader.list_all_symbols() == ["ETHUSDT"]


def test_normalize_metrics_frame_string_and_numeric_timestamps() -> None:
    d = BinanceVisionDownloader()
    string_frame = pd.DataFrame({
        "create_time": ["2024-01-01 00:00:00", "2024-01-01 00:05:00"],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "sum_open_interest": ["100", "101"],
        "sum_open_interest_value": ["1e6", "1e6"],
        "count_toptrader_long_short_ratio": ["1", "1"],
        "sum_toptrader_long_short_ratio": ["0.5", "0.6"],
        "count_long_short_ratio": ["1", "1"],
        "sum_taker_long_short_vol_ratio": ["1.2", "1.3"],
    })
    out = d._normalize_metrics_frame("BTCUSDT", string_frame)
    assert list(out.columns) == [
        "timestamp", "datetime", "available_at", "symbol",
        "sum_open_interest", "sum_open_interest_value",
        "long_short_ratio", "top_trader_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    assert len(out) == 2
    assert pd.api.types.is_numeric_dtype(out["timestamp"])
    assert out["available_at"].dt.tz is not None

    numeric_frame = pd.DataFrame({
        "create_time": [1704067200000, 1704067500000],
        "sum_open_interest": [100.0, 101.0],
    })
    numeric_out = d._normalize_metrics_frame("BTCUSDT", numeric_frame)
    assert len(numeric_out) == 2
    assert numeric_out.iloc[0]["symbol"] == "BTCUSDT"


def test_normalize_metrics_frame_missing_timestamp_is_empty() -> None:
    d = BinanceVisionDownloader()
    frame = pd.DataFrame({"symbol": ["BTCUSDT"]})
    assert d._normalize_metrics_frame("BTCUSDT", frame).empty
    assert d._normalize_metrics_frame("BTCUSDT", pd.DataFrame()).empty


def test_fetch_daily_metrics_returns_empty_on_missing_column(monkeypatch) -> None:
    d = BinanceVisionDownloader()
    frame = pd.DataFrame({"bad": [1]})
    monkeypatch.setattr(d, "_fetch_zip_csv", lambda url: frame)
    out = d.fetch_daily_metrics("BTCUSDT", datetime(2024, 1, 1))
    assert out.empty
    assert list(out.columns) == [
        "timestamp", "datetime", "available_at", "symbol",
        "sum_open_interest", "sum_open_interest_value",
        "long_short_ratio", "top_trader_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]


def test_fetch_metrics_bulk_uses_cache_and_bounds(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame({
        "timestamp": [1704067200000],
        "datetime": [pd.Timestamp("2024-01-01", tz="UTC")],
        "available_at": [pd.Timestamp("2024-01-01 00:05", tz="UTC")],
        "symbol": ["BTCUSDT"],
        "sum_open_interest": [100.0],
        "sum_open_interest_value": [1e6],
        "long_short_ratio": [0.5],
        "top_trader_long_short_ratio": [0.5],
        "sum_taker_long_short_vol_ratio": [1.0],
    })
    monkeypatch.setattr(
        BinanceVisionDownloader, "fetch_metrics_daily",
        lambda self, symbol, dt: frame,
    )
    before_start = fetch_metrics_bulk("BTCUSDT", "2019-01-01", "2019-01-02", cache_dir=str(tmp_path))
    assert before_start.empty

    combined = fetch_metrics_bulk("BTCUSDT", "2024-01-01", "2024-01-02", cache_dir=str(tmp_path))
    assert len(combined) >= 1
    cached = fetch_metrics_bulk("BTCUSDT", "2024-01-01", "2024-01-02", cache_dir=str(tmp_path))
    assert len(cached) >= 1


def test_read_url_bytes_retries_then_succeeds(downloader, monkeypatch) -> None:
    import urllib.error

    attempts = 0

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    def fake_urlopen(url, timeout=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.HTTPError(url, 429, "rate", {}, None)
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(downloader, "_wait_for_turn", lambda: None)
    monkeypatch.setattr("time.sleep", lambda *a: None)
    monkeypatch.setattr(downloader, "max_retries", 2)
    monkeypatch.setattr(downloader, "_compute_backoff_seconds", lambda attempt, http_error=None: 0.0)

    assert downloader._read_url_bytes("https://data.binance.vision/x") == b"ok"
    assert attempts == 2
