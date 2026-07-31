from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import pytest

from src.market_data.binance.futures import BinanceClient, BinanceKlinePermanentError
from src.market_data.binance.margin import BinanceMarginClient
from src.market_data.binance.spot import BinanceSpotClient


def test_spot_client_uses_spot_endpoint_and_utc_parser() -> None:
    assert BinanceSpotClient.BASE_URL == "https://api.binance.com/api/v3/klines"
    assert BinanceSpotClient._parse_iso("2024-01-01", end_of_day=False) == int(
        datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000
    )
    assert BinanceSpotClient._parse_iso("2024-01-01T00:00:00", end_of_day=False) == int(
        datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000
    )
    assert BinanceSpotClient._parse_iso(datetime(2024, 1, 1, tzinfo=UTC), end_of_day=False) == int(
        datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000
    )
    assert BinanceSpotClient._parse_iso("2024-01-01 00:00:00+00:00", end_of_day=False) == int(
        datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000
    )


def test_margin_client_signs_and_normalizes_interest_history(monkeypatch) -> None:
    credential = "test-credential"
    client = BinanceMarginClient(api_key=credential, secret=credential)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return (
                b'[{"asset":"USDT","timestamp":3600000,"dailyInterestRate":"0.024","vipLevel":0},'
                b'{"asset":"USDT","timestamp":0,"dailyInterestRate":"0.012","vipLevel":0}]'
            )

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    frame = client.fetch_margin_interest_rate_history("USDT", "1970-01-01", "1970-01-02")

    assert list(frame.columns) == ["timestamp", "dailyInterestRate", "asset", "vipLevel"]
    assert frame["timestamp"].tolist() == [0, 3600000]
    assert frame["dailyInterestRate"].tolist() == [0.012, 0.024]


def test_margin_client_deduplicates_identical_timestamps(monkeypatch) -> None:
    credential = "test-credential"
    client = BinanceMarginClient(api_key=credential, secret=credential)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return (
                b'[{"asset":"USDT","timestamp":0,"dailyInterestRate":"0.012","vipLevel":0},'
                b'{"asset":"USDT","timestamp":0,"dailyInterestRate":"0.012","vipLevel":0},'
                b'{"asset":"USDT","timestamp":3600000,"dailyInterestRate":"0.024","vipLevel":0}]'
            )

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    frame = client.fetch_margin_interest_rate_history("USDT", "1970-01-01", "1970-01-02")

    assert frame["timestamp"].tolist() == [0, 3600000]


def test_margin_client_rejects_conflicting_duplicate_timestamps(monkeypatch) -> None:
    credential = "test-credential"
    client = BinanceMarginClient(api_key=credential, secret=credential)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return (
                b'[{"asset":"USDT","timestamp":0,"dailyInterestRate":"0.012","vipLevel":0},'
                b'{"asset":"USDT","timestamp":0,"dailyInterestRate":"0.013","vipLevel":0}]'
            )

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    with pytest.raises(RuntimeError, match="conflicting duplicate"):
        client.fetch_margin_interest_rate_history("USDT", "1970-01-01", "1970-01-02")


def test_margin_client_rejects_unexpected_asset(monkeypatch) -> None:
    credential = "test-credential"
    client = BinanceMarginClient(api_key=credential, secret=credential)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return b'[{"asset":"BTC","timestamp":0,"dailyInterestRate":"0.012","vipLevel":0}]'

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    with pytest.raises(RuntimeError, match="unexpected asset"):
        client.fetch_margin_interest_rate_history("USDT", "1970-01-01", "1970-01-02")


def test_margin_client_rejects_invalid_numeric_payload(monkeypatch) -> None:
    credential = "test-credential"
    client = BinanceMarginClient(api_key=credential, secret=credential)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return b'[{"asset":"USDT","timestamp":0,"dailyInterestRate":"bad","vipLevel":0}]'

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    with pytest.raises(RuntimeError, match="invalid numeric"):
        client.fetch_margin_interest_rate_history("USDT", "1970-01-01", "1970-01-02")


def test_margin_client_rejects_missing_payload_fields(monkeypatch) -> None:
    credential = "test-credential"
    client = BinanceMarginClient(api_key=credential, secret=credential)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return b'[{"asset":"USDT","timestamp":0}]'

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    with pytest.raises(RuntimeError, match="missing required fields"):
        client.fetch_margin_interest_rate_history("USDT", "1970-01-01", "1970-01-02")


def test_margin_client_returns_empty_frame_for_empty_payload(monkeypatch) -> None:
    credential = "test-credential"
    client = BinanceMarginClient(api_key=credential, secret=credential)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return b"[]"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    frame = client.fetch_margin_interest_rate_history("USDT", "1970-01-01", "1970-01-02")

    assert frame.empty
    assert list(frame.columns) == ["timestamp", "dailyInterestRate", "asset", "vipLevel"]


def test_margin_client_rejects_invalid_asset_and_range() -> None:
    credential = "test-credential"
    client = BinanceMarginClient(api_key=credential, secret=credential)

    with pytest.raises(ValueError, match="asset"):
        client.fetch_margin_interest_rate_history("", "1970-01-01", "1970-01-02")
    with pytest.raises(ValueError, match="invalid interest-rate range"):
        client.fetch_margin_interest_rate_history("USDT", "1970-01-02", "1970-01-01")


def test_margin_client_rejects_non_object_payload_rows(monkeypatch) -> None:
    credential = "test-credential"
    client = BinanceMarginClient(api_key=credential, secret=credential)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return b"[1]"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    with pytest.raises(RuntimeError, match="non-object"):
        client.fetch_margin_interest_rate_history("USDT", "1970-01-01", "1970-01-02")


def test_margin_client_rejects_non_list_payload(monkeypatch) -> None:
    credential = "test-credential"
    client = BinanceMarginClient(api_key=credential, secret=credential)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return b"{}"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    with pytest.raises(RuntimeError, match="must be a list"):
        client.fetch_margin_interest_rate_history("USDT", "1970-01-01", "1970-01-02")


def test_margin_client_maps_http_errors(monkeypatch) -> None:
    import io
    import urllib.error

    credential = "test-credential"
    client = BinanceMarginClient(api_key=credential, secret=credential)
    error = urllib.error.HTTPError(
        "https://api.binance.com", 401, "unauthorized", {}, io.BytesIO(b'{"code":-2015}'),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(RuntimeError, match="HTTP 401"):
        client.fetch_margin_interest_rate_history("USDT", "1970-01-01", "1970-01-02")


def test_margin_client_rejects_invalid_url_scheme() -> None:
    credential = "test-credential"
    client = BinanceMarginClient(api_key=credential, secret=credential)
    client.BASE_URL = "ftp://invalid"

    with pytest.raises(ValueError, match="Invalid URL scheme"):
        client.fetch_margin_interest_rate_history("USDT", "1970-01-01", "1970-01-02")


def test_margin_client_requires_credentials() -> None:
    client = BinanceMarginClient(api_key="", secret="")

    with pytest.raises(RuntimeError, match="credentials"):
        client.fetch_margin_interest_rate_history("USDT", "1970-01-01", "1970-01-02")


def test_spot_client_returns_empty_canonical_frame_for_empty_response(monkeypatch) -> None:
    client = BinanceSpotClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "milliseconds", lambda: 1_704_153_600_000)

    class EmptyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return b"[]"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: EmptyResponse())
    frame = client.fetch_spot_ohlcv("BTC/USDT", "1h", "2024-01-01", "2024-01-01")

    assert list(frame.columns) == [
        "timestamp", "open", "high", "low", "close", "volume",
        "quote_vol", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    assert frame.empty


def test_spot_client_normalizes_nonempty_kline_response(monkeypatch) -> None:
    client = BinanceSpotClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "milliseconds", lambda: 1_704_067_200_000)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return (
                b'[["malformed"],[1704067200000,"100","101","99","100.5","10",'
                b'1704067259999,"1000","5","500","0"]]'
            )

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    frame = client.fetch_spot_ohlcv("BTC/USDT", "1h", "2023-12-31T23:00:00Z", "2024-01-01T00:00:00Z")

    assert len(frame) == 1
    assert frame.loc[0, "close"] == 100.5


def test_spot_client_maps_permanent_http_error(monkeypatch) -> None:
    import urllib.error

    client = BinanceSpotClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    error = urllib.error.HTTPError("https://api.binance.com", 400, "bad request", {}, None)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(error))

    with pytest.raises(BinanceKlinePermanentError) as raised:
        client.fetch_spot_ohlcv("BTC/USDT", "1h", "2024-01-01", "2024-01-01")
    assert raised.value.http_code == 400


def test_spot_client_paces_between_partial_pages(monkeypatch) -> None:
    client = BinanceSpotClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    calls = 0

    class Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return self.payload

    row = b'[[1704067200000,"100","101","99","100.5","10",1704067259999,"1000","5","500","0"]]'

    def fake_urlopen(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Response(row if calls == 1 else b"[]")

    sleeps: list[float] = []
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.market_data.binance.spot.time.sleep", sleeps.append)
    frame = client.fetch_spot_ohlcv("BTC/USDT", "1h", "2023-12-31T23:00:00Z", "2024-01-01T01:00:00Z")

    assert len(frame) == 1
    assert sleeps == [0.1]


def test_spot_client_stops_after_transient_failures(monkeypatch) -> None:
    client = BinanceSpotClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    sleeps: list[float] = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr("src.market_data.binance.spot.time.sleep", sleeps.append)

    result = client.fetch_spot_ohlcv("BTC/USDT", "1h", "2024-01-01", "2024-01-01")

    assert result.empty
    assert sleeps == [1, 2, 3, 4, 5]


def test_spot_client_retries_server_http_failures(monkeypatch) -> None:
    import urllib.error

    client = BinanceSpotClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    error = urllib.error.HTTPError("https://api.binance.com", 301, "redirect", {}, None)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    sleeps: list[float] = []
    monkeypatch.setattr("src.market_data.binance.spot.time.sleep", sleeps.append)

    result = client.fetch_spot_ohlcv("BTC/USDT", "1h", "2024-01-01", "2024-01-01")

    assert result.empty
    assert sleeps == [1, 2, 3, 4, 5]


def test_spot_client_backs_off_on_5xx(monkeypatch) -> None:
    import urllib.error

    client = BinanceSpotClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    error = urllib.error.HTTPError("https://api.binance.com", 500, "server", {}, None)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    sleeps: list[float] = []
    monkeypatch.setattr("src.market_data.binance.spot.time.sleep", sleeps.append)

    result = client.fetch_spot_ohlcv("BTC/USDT", "1h", "2024-01-01", "2024-01-01")

    assert result.empty
    assert sleeps == [2, 4, 6, 8, 10]


def test_spot_client_rejects_invalid_url_scheme(monkeypatch) -> None:
    client = BinanceSpotClient()
    monkeypatch.setattr(client, "BASE_URL", "ftp://invalid")

    with pytest.raises(ValueError, match="Invalid URL scheme"):
        client.fetch_spot_ohlcv("BTC/USDT", "1h", "2024-01-01", "2024-01-01")


def test_spot_client_uses_exchange_clock_when_end_is_omitted(monkeypatch) -> None:
    client = BinanceSpotClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "milliseconds", lambda: 1704067200000)

    class EmptyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return b"[]"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: EmptyResponse())
    result = client.fetch_spot_ohlcv("BTC/USDT", "1h", "2024-01-01")

    assert result.empty


def test_futures_client_parses_ohlcv_and_funding_responses(monkeypatch) -> None:
    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "parse8601", lambda value: 0 if "00:00:00" in value else 1000)

    class Response:
        headers: ClassVar[dict[str, str]] = {"x-mbx-used-weight-1m": "0"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def read(self) -> bytes:
            return self.payload

    kline = b'[[1000,"100","101","99","100.5","10",1001,"1000","5","500","0"]]'
    funding = b'[{"fundingTime":1000,"fundingRate":"0.0001"}]'
    responses = iter([Response(kline), Response(funding)])
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: next(responses))

    ohlcv = client.fetch_ohlcv_with_taker("BTC/USDT", "1h", "2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z")
    rates = client.fetch_funding_rate_history("BTC/USDT", "2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z")

    assert len(ohlcv) == 1
    assert rates.to_dict("records") == [{"timestamp": 1000, "funding_rate": 0.0001}]
