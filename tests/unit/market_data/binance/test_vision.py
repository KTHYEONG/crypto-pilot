from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

import src.market_data.binance.vision as vision_module
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
        + b"<CommonPrefixes><Prefix>data/futures/um/daily/klines/ETHUSDT/</Prefix></CommonPrefixes>"
        + b"</ListBucketResult>"
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
    body = _vision_page([f"{prefix}AAAUSDT/", "data/futures/um/monthly/klines/BBBUSDT/", prefix], False)
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


_LISTING_NS = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'


def test_list_objects_parses_etag_size(downloader, monkeypatch) -> None:
    body = (
        f"<ListBucketResult {_LISTING_NS}><IsTruncated>false</IsTruncated>"
        "<Contents><Key>data/a.zip</Key><ETag>&quot;0123&quot;</ETag><Size>42</Size></Contents>"
        "</ListBucketResult>"
    ).encode()
    monkeypatch.setattr(downloader, "_read_url_bytes", lambda url, timeout=None: body)

    objects = downloader.list_objects(prefix="data/")

    assert objects is not None
    assert [(o.key, o.etag, o.size) for o in objects] == [("data/a.zip", "0123", 42)]


def test_list_objects_fails_closed_on_missing_field(downloader, monkeypatch) -> None:
    body = (
        f"<ListBucketResult {_LISTING_NS}><IsTruncated>false</IsTruncated>"
        "<Contents><Key>data/a.zip</Key><Size>42</Size></Contents>"
        "</ListBucketResult>"
    ).encode()
    monkeypatch.setattr(downloader, "_read_url_bytes", lambda url, timeout=None: body)

    assert downloader.list_objects(prefix="data/") is None


class _FailingStream:
    def __init__(self) -> None:
        self._chunks = [b"abc", None]

    def read(self, _size: int) -> bytes:
        chunk = self._chunks.pop(0)
        if chunk is None:
            raise OSError("connection reset")
        return chunk

    def __enter__(self) -> _FailingStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_download_object_atomic(downloader, monkeypatch, tmp_path) -> None:
    import urllib.request

    monkeypatch.setattr(downloader, "max_retries", 0)
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: _FailingStream())
    dest = tmp_path / "obj.zip"

    with pytest.raises(OSError, match="connection reset"):
        downloader.download_object(key="data/futures/um/x y.zip", dest=dest)

    assert not dest.exists()
    assert not dest.with_name(dest.name + ".partial").exists()


def test_download_object_404_false(downloader, monkeypatch, tmp_path) -> None:
    import urllib.error
    import urllib.request

    def _missing(url, timeout=None):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _missing)
    dest = tmp_path / "obj.zip"

    assert downloader.download_object(key="data/missing.zip", dest=dest) is False
    assert not dest.exists()


class _Body:
    def __init__(self, chunks: list[bytes | Exception]) -> None:
        self._chunks = chunks

    def read(self, _size: int) -> bytes:
        item = self._chunks.pop(0) if self._chunks else b""
        if isinstance(item, Exception):
            raise item
        return item

    def __enter__(self) -> _Body:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_download_object_streams_full_body(downloader, monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: _Body([b"ab", b"cd"]))
    dest = tmp_path / "nested" / "obj.zip"

    assert downloader.download_object(key="data/futures/um/x y/z.zip", dest=dest) is True
    assert dest.read_bytes() == b"abcd"
    assert not dest.with_name(dest.name + ".partial").exists()


def test_download_object_retries_retryable_http_error(downloader, monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import urllib.error
    import urllib.request

    monkeypatch.setattr(downloader, "max_retries", 2)
    monkeypatch.setattr(vision_module.time, "sleep", lambda seconds: None)
    responses = [urllib.error.HTTPError("u", 503, "busy", {}, None), _Body([b"ok"])]

    def _urlopen(url, timeout=None):  # type: ignore[no-untyped-def]
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    dest = tmp_path / "obj.zip"

    assert downloader.download_object(key="data/a.zip", dest=dest) is True
    assert dest.read_bytes() == b"ok"


def test_download_object_non_retryable_error_propagates(downloader, monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import urllib.error
    import urllib.request

    def _forbidden(url, timeout=None):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    with pytest.raises(urllib.error.HTTPError):
        downloader.download_object(key="data/a.zip", dest=tmp_path / "a.zip")


def test_list_object_keys_returns_sorted_keys(downloader, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    body = (
        f"<ListBucketResult {_LISTING_NS}><IsTruncated>false</IsTruncated>"
        "<Contents><Key>b.zip</Key></Contents><Contents><Key>a.zip</Key></Contents>"
        "</ListBucketResult>"
    ).encode()
    monkeypatch.setattr(downloader, "_read_url_bytes", lambda url, timeout=None: body)

    assert downloader.list_object_keys(prefix="") == ["a.zip", "b.zip"]


def test_list_object_keys_failure_returns_none(downloader, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(downloader, "_read_url_bytes", lambda url, timeout=None: b"<not-xml")
    assert downloader.list_object_keys(prefix="") is None


def test_list_objects_rejects_negative_size(downloader, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    body = (
        f"<ListBucketResult {_LISTING_NS}><IsTruncated>false</IsTruncated>"
        "<Contents><Key>a</Key><ETag>&quot;1&quot;</ETag><Size>-1</Size></Contents>"
        "</ListBucketResult>"
    ).encode()
    monkeypatch.setattr(downloader, "_read_url_bytes", lambda url, timeout=None: body)
    assert downloader.list_objects(prefix="") is None


def test_list_objects_follows_pages_by_marker_and_last_key(downloader, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pages = [
        (
            f"<ListBucketResult {_LISTING_NS}><IsTruncated>true</IsTruncated>"
            "<Contents><Key>a.zip</Key><ETag>&quot;1&quot;</ETag><Size>1</Size></Contents>"
            "</ListBucketResult>"
        ).encode(),
        (
            f"<ListBucketResult {_LISTING_NS}><IsTruncated>true</IsTruncated>"
            "<NextMarker>c.zip</NextMarker>"
            "<Contents><Key>b.zip</Key><ETag>&quot;2&quot;</ETag><Size>2</Size></Contents>"
            "</ListBucketResult>"
        ).encode(),
        (
            f"<ListBucketResult {_LISTING_NS}><IsTruncated>false</IsTruncated>"
            "<Contents><Key>c.zip</Key><ETag>&quot;3&quot;</ETag><Size>3</Size></Contents>"
            "</ListBucketResult>"
        ).encode(),
    ]
    urls: list[str] = []

    def _read(url, timeout=None):  # type: ignore[no-untyped-def]
        urls.append(url)
        return pages.pop(0)

    monkeypatch.setattr(downloader, "_read_url_bytes", _read)

    objects = downloader.list_objects(prefix="")

    assert objects is not None
    assert [o.key for o in objects] == ["a.zip", "b.zip", "c.zip"]
    assert "marker=a.zip" in urls[1]
    assert "marker=c.zip" in urls[2]


def test_list_objects_stalled_marker_fails_closed(downloader, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    page = (
        f"<ListBucketResult {_LISTING_NS}><IsTruncated>true</IsTruncated>"
        "<Contents><Key>a.zip</Key><ETag>&quot;1&quot;</ETag><Size>1</Size></Contents>"
        "</ListBucketResult>"
    ).encode()
    monkeypatch.setattr(downloader, "_read_url_bytes", lambda url, timeout=None: page)
    assert downloader.list_objects(prefix="") is None


def test_download_object_retries_network_error(downloader, monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import urllib.error
    import urllib.request

    monkeypatch.setattr(downloader, "max_retries", 1)
    monkeypatch.setattr(vision_module.time, "sleep", lambda seconds: None)
    responses: list[object] = [urllib.error.URLError("reset"), _Body([b"ok"])]

    def _urlopen(url, timeout=None):  # type: ignore[no-untyped-def]
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    dest = tmp_path / "obj.zip"

    assert downloader.download_object(key="data/a.zip", dest=dest) is True
    assert dest.read_bytes() == b"ok"
