from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from src.market_data.binance.vision import BinanceVisionDownloader


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
    assert len(downloader.fetch_funding_rate_monthly("BTCUSDT", 2024, 1)) == 1
    assert called == [
        "monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip",
        "monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip",
    ]


def test_s3_listing_parses_symbols(downloader, monkeypatch) -> None:
    ns = "http://s3.amazonaws.com/doc/2006-03-01/"
    body = (
        f'<ListBucketResult xmlns="{ns}">'.encode()
        + b'<CommonPrefixes><Prefix>data/futures/um/daily/klines/ETHUSDT/</Prefix></CommonPrefixes>'
        + b'</ListBucketResult>'
    )
    monkeypatch.setattr(downloader, "_read_url_bytes", lambda url, timeout=None: body)
    assert downloader.list_all_symbols() == ["ETHUSDT"]


def _vision_page(prefixes: list[str], truncated: bool, next_marker: str | None = None) -> bytes:
    ns = "http://s3.amazonaws.com/doc/2006-03-01/"
    parts = [f'<ListBucketResult xmlns="{ns}">']
    parts.extend(f"<CommonPrefixes><Prefix>{prefix}</Prefix></CommonPrefixes>" for prefix in prefixes)
    parts.append(f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>")
    if next_marker is not None:
        parts.append(f"<NextMarker>{next_marker}</NextMarker>")
    parts.append("</ListBucketResult>")
    return "".join(parts).encode()


def test_s3_listing_follows_next_marker(downloader, monkeypatch) -> None:
    prefix = "data/futures/um/daily/klines/"
    first = _vision_page([f"{prefix}AAAUSDT/", f"{prefix}BBBUSDT/"], True, "MARKER-1")
    second = _vision_page([f"{prefix}ZZZUSDT/"], False)
    urls: list[str] = []

    def _fake(url, timeout=None):
        urls.append(url)
        return first if len(urls) == 1 else second

    monkeypatch.setattr(downloader, "_read_url_bytes", _fake)
    assert downloader.list_all_symbols() == ["AAAUSDT", "BBBUSDT", "ZZZUSDT"]
    assert "marker=MARKER-1" in urls[1]


def test_s3_listing_falls_back_to_last_prefix(downloader, monkeypatch) -> None:
    prefix = "data/futures/um/daily/klines/"
    first = _vision_page([f"{prefix}AAAUSDT/", f"{prefix}MMMUSDT/"], True)
    second = _vision_page([f"{prefix}ZZZUSDT/"], False)
    urls: list[str] = []

    def _fake(url, timeout=None):
        urls.append(url)
        return first if len(urls) == 1 else second

    monkeypatch.setattr(downloader, "_read_url_bytes", _fake)
    assert downloader.list_all_symbols() == ["AAAUSDT", "MMMUSDT", "ZZZUSDT"]
    assert "marker=" in urls[1]


def test_s3_listing_second_page_failure_returns_empty(downloader, monkeypatch) -> None:
    prefix = "data/futures/um/daily/klines/"
    first = _vision_page([f"{prefix}AAAUSDT/"], True, "MARKER-1")
    calls = 0

    def _fake(url, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        raise RuntimeError("boom")

    monkeypatch.setattr(downloader, "_read_url_bytes", _fake)
    assert downloader.list_all_symbols() == []


def test_s3_listing_skips_foreign_and_empty_prefixes(downloader, monkeypatch) -> None:
    prefix = "data/futures/um/daily/klines/"
    body = _vision_page(
        [f"{prefix}AAAUSDT/", "data/futures/um/monthly/klines/BBBUSDT/", prefix], False
    )
    monkeypatch.setattr(downloader, "_read_url_bytes", lambda url, timeout=None: body)
    assert downloader.list_all_symbols() == ["AAAUSDT"]


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
