from __future__ import annotations

from http.client import HTTPMessage
from typing import Literal
from urllib.error import HTTPError

import pandas as pd
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


def test_fetch_daily_metrics_normalizes_headerless_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 1: 실제 Binance Vision 포맷(datetime 문자열 create_time) — 기존 버그 검증."""
    downloader = BinanceVisionDownloader()

    raw = pd.DataFrame(
        [
            [
                "2026-03-15 00:05:00",
                "BTCUSDT",
                "123.4",
                "456.7",
                "0",
                "1.8",
                "0.9",
                "1.2",
            ]
        ]
    )

    monkeypatch.setattr(
        downloader,
        "_fetch_zip_csv",
        lambda _url: raw,
    )

    out = downloader.fetch_daily_metrics("BTCUSDT", pd.Timestamp("2024-04-01"))

    assert list(out.columns) == [
        "timestamp",
        "datetime",
        "available_at",
        "symbol",
        "sum_open_interest",
        "sum_open_interest_value",
        "long_short_ratio",
        "top_trader_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    assert out.shape[0] == 1
    assert out.loc[0, "datetime"] == pd.Timestamp("2026-03-15 00:05:00", tz="UTC")
    assert out.loc[0, "timestamp"] == 1773533100000
    assert out.loc[0, "available_at"] == out.loc[0, "datetime"] + pd.Timedelta(minutes=5)
    assert out.loc[0, "symbol"] == "BTCUSDT"
    assert out.loc[0, "long_short_ratio"] == pytest.approx(0.9)


def test_fetch_daily_metrics_normalizes_numeric_epoch_backward_compat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 2: numeric epoch-ms 하위호환 경로 (dtype 분기 검증)."""
    downloader = BinanceVisionDownloader()

    raw = pd.DataFrame(
        [
            [
                1711929600000,
                "BTCUSDT",
                "123.4",
                "456.7",
                "0",
                "1.8",
                "0.9",
                "1.2",
            ]
        ]
    )

    monkeypatch.setattr(
        downloader,
        "_fetch_zip_csv",
        lambda _url: raw,
    )

    out = downloader.fetch_daily_metrics("BTCUSDT", pd.Timestamp("2024-04-01"))

    assert out.shape[0] == 1
    assert out.loc[0, "datetime"] == pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    assert out.loc[0, "timestamp"] == 1711929600000


def test_fetch_daily_metrics_drops_unparseable_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 3: 부분 파싱 실패 — 유효 1행만 남고 garbage 행은 drop."""
    downloader = BinanceVisionDownloader()

    raw = pd.DataFrame(
        [
            ["2026-03-15 00:05:00", "BTCUSDT", "123.4", "456.7", "0", "1.8", "0.9", "1.2"],
            ["not-a-date", "BTCUSDT", "200", "300", "0", "1.0", "0.5", "0.8"],
        ]
    )

    monkeypatch.setattr(
        downloader,
        "_fetch_zip_csv",
        lambda _url: raw,
    )

    out = downloader.fetch_daily_metrics("BTCUSDT", pd.Timestamp("2024-04-01"))

    assert out.shape[0] == 1


def test_fetch_daily_metrics_returns_empty_canonical_frame_on_unparseable_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 4: 전체 파싱 실패 — 빈 canonical frame 반환."""
    downloader = BinanceVisionDownloader()

    raw = pd.DataFrame(
        [
            ["N/A", "BTCUSDT", "123.4", "456.7", "0", "1.8", "0.9", "1.2"],
        ]
    )

    monkeypatch.setattr(
        downloader,
        "_fetch_zip_csv",
        lambda _url: raw,
    )

    out = downloader.fetch_daily_metrics("BTCUSDT", pd.Timestamp("2024-04-01"))

    assert out.empty is True
    assert list(out.columns) == [
        "timestamp",
        "datetime",
        "available_at",
        "symbol",
        "sum_open_interest",
        "sum_open_interest_value",
        "long_short_ratio",
        "top_trader_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]


def test_fetch_metrics_daily_delegates_to_normalized_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = BinanceVisionDownloader()
    expected = pd.DataFrame({"timestamp": [1]})

    monkeypatch_value = {"called": 0}

    def _fake_fetch(symbol: str, date: pd.Timestamp) -> pd.DataFrame:
        monkeypatch_value["called"] += 1
        assert symbol == "ETHUSDT"
        return expected

    monkeypatch.setattr(downloader, "fetch_daily_metrics", _fake_fetch)

    out = downloader.fetch_metrics_daily("ETHUSDT", pd.Timestamp("2024-04-01"))

    assert monkeypatch_value["called"] == 1
    assert out is expected


def test_fetch_daily_metrics_returns_empty_canonical_frame_on_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = BinanceVisionDownloader()

    def _raise(_url: str) -> pd.DataFrame:
        raise _http_error(404)

    monkeypatch.setattr(downloader, "_fetch_zip_csv", _raise)
    out = downloader.fetch_daily_metrics("BTCUSDT", pd.Timestamp("2024-04-01"))

    assert out.empty
    assert "available_at" in out.columns


def test_normalize_metrics_frame_empty_input() -> None:
    """L392: 빈 frame 입력 시 빈 canonical frame 반환."""
    downloader = BinanceVisionDownloader()
    empty = pd.DataFrame()
    out = downloader._normalize_metrics_frame("BTCUSDT", empty)
    assert out.empty
    assert list(out.columns) == [
        "timestamp", "datetime", "available_at", "symbol",
        "sum_open_interest", "sum_open_interest_value",
        "long_short_ratio", "top_trader_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]


def test_normalize_metrics_frame_no_timestamp_column() -> None:
    """L427: timestamp 컬럼이 없으면 빈 canonical frame 반환."""
    downloader = BinanceVisionDownloader()
    df = pd.DataFrame({"foo": [1], "symbol": ["BTCUSDT"]})
    out = downloader._normalize_metrics_frame("BTCUSDT", df)
    assert out.empty
    assert list(out.columns) == [
        "timestamp", "datetime", "available_at", "symbol",
        "sum_open_interest", "sum_open_interest_value",
        "long_short_ratio", "top_trader_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]


def test_normalize_metrics_frame_symbol_not_in_columns() -> None:
    """L422: symbol 컬럼이 없으면 인자 symbol로 채움."""
    downloader = BinanceVisionDownloader()
    df = pd.DataFrame({"create_time": ["2026-03-15 00:05:00"]})
    out = downloader._normalize_metrics_frame("BTCUSDT", df)
    assert out.shape[0] == 1
    assert out.loc[0, "symbol"] == "BTCUSDT"


def test_normalize_metrics_frame_missing_optional_numeric_col() -> None:
    """L456: optional numeric 컬럼이 없으면 NaN으로 채움."""
    downloader = BinanceVisionDownloader()
    df = pd.DataFrame({"create_time": ["2026-03-15 00:05:00"], "symbol": ["BTCUSDT"]})
    out = downloader._normalize_metrics_frame("BTCUSDT", df)
    assert out.shape[0] == 1
    assert pd.isna(out.loc[0, "long_short_ratio"])
