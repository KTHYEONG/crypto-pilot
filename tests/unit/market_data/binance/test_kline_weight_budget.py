"""Kline requests must stay inside the shared per-IP weight budget (429 → IP block otherwise)."""

from __future__ import annotations

import json
import urllib.parse

import pytest

from src.market_data.binance import futures as fut


def test_kline_request_weight_follows_documented_tiers() -> None:
    assert [fut.kline_request_weight(n) for n in (1, 99, 100, 499, 500, 1000, 1500)] == [1, 1, 2, 2, 5, 5, 10]


def test_kline_page_limit_sizes_a_daily_tail_to_weight_one() -> None:
    hour = 3_600_000
    limit = fut.kline_page_limit("1h", 0, 25 * hour)
    assert limit == 27
    assert fut.kline_request_weight(limit) == 1
    assert fut.kline_page_limit("1h", 0, 5000 * hour) == fut.KLINE_MAX_LIMIT
    assert fut.kline_page_limit("7h", 0, hour) == fut.KLINE_MAX_LIMIT


def test_token_bucket_paces_by_weight_and_rejects_oversized_request() -> None:
    clock = [0.0]
    sleeps: list[float] = []

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    bucket = fut.TokenBucket(10.0, 20, clock=lambda: clock[0], sleep=_sleep)
    assert bucket.acquire(20) == 0.0
    assert bucket.acquire(5) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="tokens must be in"):
        bucket.acquire(21)


def test_daily_tail_fetch_requests_small_limit_and_charges_its_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    charged: list[float] = []

    class _Recorder:
        def acquire(self, tokens: float = 1.0) -> float:
            charged.append(tokens)
            return 0.0

    monkeypatch.setattr(fut, "KLINE_WEIGHT_LIMITER", _Recorder())
    monkeypatch.setattr(fut.time, "sleep", lambda _s: None)
    urls: list[str] = []
    start_ms = 1_790_380_800_000  # 2026-09-26T00:00:00Z
    rows = [[start_ms + i * 3_600_000, "1", "1", "1", "1", "1", 0, "1", 1, "0.5", "0.5", "0"] for i in range(25)]

    class _Resp:
        def __init__(self) -> None:
            self.headers = {"x-mbx-used-weight-1m": "10"}

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def read(self) -> bytes:
            return json.dumps(rows).encode()

    def _urlopen(req: object, timeout: float = 0) -> _Resp:
        urls.append(req.full_url)  # type: ignore[attr-defined]
        return _Resp()

    monkeypatch.setattr(fut.urllib.request, "urlopen", _urlopen)
    client = fut.BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": "BTCUSDT"})

    frame = client.fetch_ohlcv_with_taker("BTCUSDT", "1h", "2026-09-26T00:00:00Z", "2026-09-27T00:00:00Z")

    assert len(frame) == 25
    limits = [int(urllib.parse.parse_qs(urllib.parse.urlparse(u).query)["limit"][0]) for u in urls]
    assert limits
    assert all(limit < 100 for limit in limits)
    assert charged == [1] * len(urls)
