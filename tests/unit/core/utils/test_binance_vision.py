from __future__ import annotations

from http.client import HTTPMessage
from typing import Literal
from urllib.error import HTTPError

import pytest

from src.core.exchange.binance_vision import BinanceVisionDownloader


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        _ = (exc_type, exc, tb)
        return False

    def read(self) -> bytes:
        return self._payload


def _http_error(code: int, retry_after: str | None = None) -> HTTPError:
    headers = HTTPMessage()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(
        url="https://data.binance.vision/fake",
        code=code,
        msg="error",
        hdrs=headers,
        fp=None,
    )


def test_default_rate_limit_is_conservative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BINANCE_VISION_MAX_WEIGHT_PER_MIN", raising=False)
    monkeypatch.delenv("BINANCE_VISION_MIN_REQUEST_INTERVAL_SECONDS", raising=False)
    downloader = BinanceVisionDownloader()
    assert downloader.max_weight_per_min == 600
    assert downloader.min_request_interval_seconds >= 0.1


def test_read_url_bytes_retries_for_429_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = BinanceVisionDownloader()
    sleep_calls: list[float] = []
    call_count = {"n": 0}

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    def fake_urlopen(url: str, timeout: int) -> _FakeResponse:
        _ = (url, timeout)
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _http_error(429, retry_after="2")
        return _FakeResponse(b"ok")

    monkeypatch.setattr("src.core.exchange.binance_vision.time.sleep", fake_sleep)
    monkeypatch.setattr("src.core.exchange.binance_vision.urllib.request.urlopen", fake_urlopen)

    data = downloader._read_url_bytes("https://data.binance.vision/fake")
    assert data == b"ok"
    assert call_count["n"] == 2
    assert any(delay >= 2.0 for delay in sleep_calls)


def test_read_url_bytes_does_not_retry_for_404(monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = BinanceVisionDownloader()
    call_count = {"n": 0}

    def fake_urlopen(url: str, timeout: int) -> _FakeResponse:
        _ = (url, timeout)
        call_count["n"] += 1
        raise _http_error(404)

    monkeypatch.setattr("src.core.exchange.binance_vision.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(HTTPError):
        downloader._read_url_bytes("https://data.binance.vision/fake")
    assert call_count["n"] == 1
