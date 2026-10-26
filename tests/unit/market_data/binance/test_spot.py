from __future__ import annotations

from urllib.error import HTTPError

import pytest

from src.market_data.binance.spot import BinanceSpotClient


@pytest.mark.parametrize("failure", [OSError("offline"), HTTPError("https://example.com", 503, "unavailable", {}, None)])
def test_exhausted_transport_retries_never_return_empty_history(monkeypatch, failure) -> None:
    attempts = []

    def unavailable(*args, **kwargs):
        attempts.append(1)
        raise failure

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(OSError, match="retries exhausted"):
        BinanceSpotClient().fetch_spot_ohlcv("BTCUSDT", "1h", "2023-01-01", "2023-01-02")
    assert len(attempts) == 5
