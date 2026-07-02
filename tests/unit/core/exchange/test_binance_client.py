from __future__ import annotations

import json
import logging
from http.client import HTTPMessage
from typing import Literal
from urllib.error import HTTPError

import pandas as pd
import pytest

from src.core.exchange.binance_client import BinanceClient, BinanceKlinePermanentError


class _FakeExchange:
    def parse8601(self, s: str) -> int:
        if s.startswith("2026-01-01"):
            return 1_000
        return 2_000

    def milliseconds(self) -> int:
        return 2_000

    def market(self, symbol: str) -> dict[str, str]:
        return {"id": symbol.replace("/", "")}


class _FakeResp:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")
        self.headers = {"x-mbx-used-weight-1m": "0"}

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        _ = (exc_type, exc, tb)
        return False

    def read(self) -> bytes:
        return self._payload


def _build_client() -> BinanceClient:
    client = BinanceClient.__new__(BinanceClient)
    client.exchange = _FakeExchange()
    client.logger = logging.getLogger("test-binance-client")
    return client


def test_fetch_ohlcv_with_taker_when_http_400_raises_permanent_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client()

    def _raise_http_400(*_args: object, **_kwargs: object) -> _FakeResp:
        raise HTTPError(
            url="https://example.com",
            code=400,
            msg="bad request",
            hdrs=HTTPMessage(),
            fp=None,
        )

    monkeypatch.setattr("src.core.exchange.binance_client.urllib.request.urlopen", _raise_http_400)

    with pytest.raises(BinanceKlinePermanentError) as exc_info:
        client.fetch_ohlcv_with_taker("GAIBUSDT", "1h", "2026-01-01", "2026-01-02")
    assert exc_info.value.http_code == 400
    assert exc_info.value.symbol == "GAIBUSDT"


def test_fetch_ohlcv_with_taker_when_http_429_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client()
    calls = {"n": 0}

    def _flaky(*_args: object, **_kwargs: object) -> _FakeResp:
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError(
                url="https://example.com",
                code=429,
                msg="rate",
                hdrs=HTTPMessage(),
                fp=None,
            )
        payload = json.dumps(
            [
                # [open_time, open, high, low, close, volume, close_time,
                #  quote_asset_volume, n_trades, taker_buy_base, taker_buy_quote, ignore]
                [1500, "1", "2", "0.5", "1.2", "10", 0, "123.45", 0, "4", "5", "0"],
                [2000, "1.1", "2.1", "0.6", "1.3", "11", 0, "234.56", 0, "4.1", "5.1", "0"],
            ]
        )
        return _FakeResp(payload)

    monkeypatch.setattr(
        "src.core.exchange.binance_client.time.sleep",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("src.core.exchange.binance_client.urllib.request.urlopen", _flaky)

    df = client.fetch_ohlcv_with_taker("BTCUSDT", "1h", "2026-01-01", "2026-01-02")
    assert calls["n"] >= 2
    assert not df.empty
    assert "timestamp" in df.columns


def test_fetch_ohlcv_with_taker_extracts_quote_asset_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: row[7] (quote_asset_volume) must populate the quote_vol column.

    Prior bug: quote_asset_volume (Binance klines row index 7) was dropped entirely,
    leaving quote_vol missing -> NaN after merge with Vision-archived history, which
    universe eligibility gates then misread as zero trading volume.
    """
    client = _build_client()

    def _fake_resp(*_args: object, **_kwargs: object) -> _FakeResp:
        payload = json.dumps(
            [
                [1500, "1", "2", "0.5", "1.2", "10", 0, "999.5", 0, "4", "5", "0"],
                # second row's open_time (2000) >= end_timestamp terminates the fetch loop
                [2000, "1.1", "2.1", "0.6", "1.3", "11", 0, "888.5", 0, "4.1", "5.1", "0"],
            ]
        )
        return _FakeResp(payload)

    monkeypatch.setattr("src.core.exchange.binance_client.urllib.request.urlopen", _fake_resp)

    df = client.fetch_ohlcv_with_taker("BTCUSDT", "1h", "2026-01-01", "2026-01-02")
    assert "quote_vol" in df.columns
    assert not df["quote_vol"].isna().any()
    assert df["quote_vol"].iloc[0] == pytest.approx(999.5)
    assert df["quote_vol"].iloc[1] == pytest.approx(888.5)


def test_fetch_ohlcv_with_taker_defaults_quote_vol_to_zero_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge case: null/empty quote_asset_volume in the API payload must not crash."""
    client = _build_client()

    def _fake_resp(*_args: object, **_kwargs: object) -> _FakeResp:
        payload = json.dumps(
            [
                [1500, "1", "2", "0.5", "1.2", "10", 0, None, 0, "4", "5", "0"],
                # second row's open_time (2000) >= end_timestamp terminates the fetch loop
                [2000, "1.1", "2.1", "0.6", "1.3", "11", 0, "1.0", 0, "4.1", "5.1", "0"],
            ]
        )
        return _FakeResp(payload)

    monkeypatch.setattr("src.core.exchange.binance_client.urllib.request.urlopen", _fake_resp)

    df = client.fetch_ohlcv_with_taker("BTCUSDT", "1h", "2026-01-01", "2026-01-02")
    assert df["quote_vol"].iloc[0] == pytest.approx(0.0)


def test_fetch_open_interest_history_normalizes_amount_and_value() -> None:
    client = _build_client()
    client.exchange.fetch_open_interest_history = lambda *_args, **_kwargs: [
        {
            "timestamp": 1711929600000,
            "openInterestAmount": "123.45",
            "openInterestValue": "678.9",
        }
    ]

    out = client.fetch_open_interest_history("BTCUSDT", "4h", 1_000)

    assert out.loc[0, "sum_open_interest"] == pytest.approx(123.45)
    assert out.loc[0, "sum_open_interest_value"] == pytest.approx(678.9)
    assert out.loc[0, "available_at"] == out.loc[0, "datetime"] + pd.Timedelta(minutes=5)


def test_fetch_long_short_ratio_history_maps_global_ratio() -> None:
    client = _build_client()
    client.exchange.fetch_long_short_ratio_history = lambda *_args, **_kwargs: [
        {"timestamp": 1711929600000, "longShortRatio": "1.25"}
    ]

    out = client.fetch_long_short_ratio_history("BTCUSDT", "4h", 1_000)

    assert out.loc[0, "long_short_ratio"] == pytest.approx(1.25)
    assert "sum_open_interest" in out.columns


def test_fetch_global_long_short_ratio_history_delegates() -> None:
    client = _build_client()
    expected = pd.DataFrame({"timestamp": [1], "long_short_ratio": [1.1]})
    calls = {"n": 0}

    def _fake_fetch(*_args: object, **_kwargs: object) -> pd.DataFrame:
        calls["n"] += 1
        return expected

    client.fetch_long_short_ratio_history = _fake_fetch  # type: ignore[method-assign]

    out = client.fetch_global_long_short_ratio_history("BTCUSDT")

    assert calls["n"] == 1
    assert out is expected
