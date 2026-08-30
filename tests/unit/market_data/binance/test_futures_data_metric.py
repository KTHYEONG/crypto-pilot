"""Contract coverage for BinanceClient.fetch_futures_data_metric.

Verifies the /futures/data/{endpoint} URL construction (unsigned) and the
endpoint whitelist.
"""

from __future__ import annotations

import pytest

from src.market_data.binance.futures import BinanceClient


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return self._body


def test_fetch_futures_data_metric_builds_url_and_rejects_unknown_endpoint(monkeypatch) -> None:
    client = BinanceClient()
    captured: dict = {}

    def _fake_urlopen(req, *args, **kwargs):
        captured["url"] = req.get_full_url()
        return _Response(
            b'[{"symbol":"BTCUSDT","sumOpenInterest":"1.0","sumOpenInterestValue":"2.0","timestamp":1788000000000}]'
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    out = client.fetch_futures_data_metric("openInterestHist", "BTCUSDT", period="5m", limit=100)
    url = captured["url"]
    assert "futures/data/openInterestHist" in url
    assert "symbol=BTCUSDT" in url
    assert "period=5m" in url
    assert "limit=100" in url
    assert "signature=" not in url
    assert isinstance(out, list)
    assert out[0]["sumOpenInterest"] == "1.0"

    with pytest.raises(ValueError, match="endpoint"):
        client.fetch_futures_data_metric("badEndpoint", "BTCUSDT")
