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


def test_fetch_mark_price_klines_normalizes_ohlc(monkeypatch) -> None:
    """MHS-21-MARK-PRICE-COVERAGE-FAIL-CLOSED: mark klines normalize like OHLCV."""

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return (
                b'[[1609459200000,"100","101","99","100.5"],'
                b'[1609462800000,"100.5","102","100","101.0"]]'
            )

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    client = BinanceClient()
    frame = client.fetch_mark_price_klines(
        "BTC/USDT", "1h", "2021-01-01T00:00:00Z", "2021-01-01T01:00:00Z",
    )

    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "datetime"]
    assert frame["timestamp"].tolist() == [1609459200000, 1609462800000]
    assert frame["close"].tolist() == [100.5, 101.0]
    assert frame["datetime"].is_monotonic_increasing


def test_fetch_funding_rate_history_raises_on_http_error(monkeypatch) -> None:
    import urllib.error
    import pytest
    from src.market_data.binance.futures import BinanceClient, BinanceFundingFetchError

    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "parse8601", lambda value: 0 if "00:00:00" in value else 1000)

    def _raise(*args, **kwargs):
        raise urllib.error.HTTPError("https://fapi.binance.com/fapi/v1/fundingRate", 400, "Bad Request", None, None)

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    with pytest.raises(BinanceFundingFetchError) as exc_info:
        client.fetch_funding_rate_history("BTC/USDT", "2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z")

    assert exc_info.value.http_code == 400
    assert exc_info.value.symbol == "BTC/USDT"
    assert "fundingRate" in exc_info.value.url
    assert isinstance(exc_info.value.__cause__, urllib.error.HTTPError)



def test_fetch_funding_rate_history_raises_on_transport_error(monkeypatch) -> None:
    import urllib.error
    import pytest
    from src.market_data.binance.futures import BinanceClient, BinanceFundingFetchError

    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "parse8601", lambda value: 0 if "00:00:00" in value else 1000)

    def _raise(*args, **kwargs):
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    with pytest.raises(BinanceFundingFetchError) as exc_info:
        client.fetch_funding_rate_history("BTC/USDT", "2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z")

    assert exc_info.value.http_code is None


def test_fetch_funding_rate_history_raises_on_malformed_body(monkeypatch) -> None:
    import pytest
    from src.market_data.binance.futures import BinanceClient, BinanceFundingFetchError

    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "parse8601", lambda value: 0 if "00:00:00" in value else 1000)


    class Response:
        headers: ClassVar[dict[str, str]] = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return b"<html>blocked</html>"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    with pytest.raises(BinanceFundingFetchError) as exc_info:
        client.fetch_funding_rate_history("BTC/USDT", "2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z")

    assert exc_info.value.http_code is None




@pytest.mark.parametrize("code", [403, 418, 429])
def test_fetch_ohlcv_with_taker_raises_ip_blocked_without_retry(monkeypatch, code) -> None:
    import urllib.error
    import pytest
    from src.market_data.binance.futures import BinanceClient

    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "parse8601", lambda value: 0 if "00:00:00" in value else 3_600_000)
    sleeps: list[float] = []
    monkeypatch.setattr("src.market_data.binance.futures.time.sleep", sleeps.append)
    from src.market_data.binance.futures import BinanceIpBlockedError

    calls: list[int] = []

    def _raise(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://fapi.binance.com/fapi/v1/klines", code, "blocked", None, None)

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    with pytest.raises(BinanceIpBlockedError) as exc_info:
        client.fetch_ohlcv_with_taker("BTC/USDT", "1h", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z")

    assert exc_info.value.http_code == code
    assert "klines" in exc_info.value.url
    assert calls == [1]
    assert sleeps == []


def test_fetch_ohlcv_with_taker_raises_permanent_on_400(monkeypatch) -> None:
    import urllib.error
    import pytest
    from src.market_data.binance.futures import BinanceClient

    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "parse8601", lambda value: 0 if "00:00:00" in value else 3_600_000)
    sleeps: list[float] = []
    monkeypatch.setattr("src.market_data.binance.futures.time.sleep", sleeps.append)
    from src.market_data.binance.futures import BinanceKlinePermanentError

    def _raise(*args, **kwargs):
        raise urllib.error.HTTPError("https://fapi.binance.com/fapi/v1/klines", 400, "Invalid symbol", None, None)

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    with pytest.raises(BinanceKlinePermanentError) as exc_info:
        client.fetch_ohlcv_with_taker("BTC/USDT", "1h", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z")

    assert exc_info.value.http_code == 400
    assert sleeps == []


def test_fetch_ohlcv_with_taker_raises_transient_after_5xx_exhaustion(monkeypatch) -> None:
    import urllib.error
    import pytest
    from src.market_data.binance.futures import BinanceClient

    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "parse8601", lambda value: 0 if "00:00:00" in value else 3_600_000)
    sleeps: list[float] = []
    monkeypatch.setattr("src.market_data.binance.futures.time.sleep", sleeps.append)
    from src.market_data.binance.futures import BinanceKlineTransientError

    def _raise(*args, **kwargs):
        raise urllib.error.HTTPError("https://fapi.binance.com/fapi/v1/klines", 503, "unavailable", None, None)

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    with pytest.raises(BinanceKlineTransientError) as exc_info:
        client.fetch_ohlcv_with_taker("BTC/USDT", "1h", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z")

    assert exc_info.value.http_code == 503
    assert exc_info.value.symbol == "BTC/USDT"
    assert exc_info.value.timeframe == "1h"
    assert sleeps == [2, 4, 6, 8]


def test_fetch_ohlcv_with_taker_raises_transient_after_transport_exhaustion(monkeypatch) -> None:
    import urllib.error
    import pytest
    from src.market_data.binance.futures import BinanceClient

    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "parse8601", lambda value: 0 if "00:00:00" in value else 3_600_000)
    sleeps: list[float] = []
    monkeypatch.setattr("src.market_data.binance.futures.time.sleep", sleeps.append)
    from src.market_data.binance.futures import BinanceKlineTransientError

    def _raise(*args, **kwargs):
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    with pytest.raises(BinanceKlineTransientError) as exc_info:
        client.fetch_ohlcv_with_taker("BTC/USDT", "1h", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z")

    assert exc_info.value.http_code is None
    assert sleeps == [1, 2, 3, 4]




@pytest.mark.parametrize("code", [403, 418, 429])
def test_fetch_funding_rate_history_raises_ip_blocked_on_waf_codes(monkeypatch, code) -> None:
    import urllib.error
    import pytest
    from src.market_data.binance.futures import BinanceClient

    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "parse8601", lambda value: 0 if "00:00:00" in value else 3_600_000)
    sleeps: list[float] = []
    monkeypatch.setattr("src.market_data.binance.futures.time.sleep", sleeps.append)
    from src.market_data.binance.futures import BinanceIpBlockedError

    def _raise(*args, **kwargs):
        raise urllib.error.HTTPError("https://fapi.binance.com/fapi/v1/fundingRate", code, "blocked", None, None)

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    with pytest.raises(BinanceIpBlockedError) as exc_info:
        client.fetch_funding_rate_history("BTC/USDT", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z")

    assert exc_info.value.http_code == code


def test_binance_error_types_survive_contextmanager_passthrough() -> None:
    import contextlib
    import pytest
    from src.market_data.binance.futures import (
        BinanceFundingFetchError,
        BinanceIpBlockedError,
        BinanceKlinePermanentError,
        BinanceKlineTransientError,
    )

    @contextlib.contextmanager
    def _passthrough():
        yield

    errors = [
        BinanceKlinePermanentError(symbol="X", timeframe="1h", http_code=400, start_time_ms=0, end_time_ms=1, url="u"),
        BinanceFundingFetchError(symbol="X", http_code=400, url="u"),
        BinanceIpBlockedError(http_code=418, url="u"),
        BinanceKlineTransientError(symbol="X", timeframe="1h", http_code=503, url="u"),
    ]
    for error in errors:
        with pytest.raises(type(error)) as exc_info, _passthrough():
            raise error
        assert exc_info.value is error
        error.add_note("context")
        assert error.__notes__ == ["context"]



def test_token_bucket_allows_burst_then_paces_at_rate() -> None:
    import pytest
    from src.market_data.binance.futures import TokenBucket

    clock = [100.0]
    sleeps: list[float] = []

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    bucket = TokenBucket(2.0, 3, clock=lambda: clock[0], sleep=_sleep)

    waits = [bucket.acquire() for _ in range(5)]

    assert waits[:3] == [0.0, 0.0, 0.0]
    assert waits[3] == pytest.approx(0.5)
    assert waits[4] == pytest.approx(0.5)
    assert sleeps == [pytest.approx(0.5), pytest.approx(0.5)]
    clock[0] += 10.0
    assert [bucket.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]


def test_token_bucket_rejects_invalid_parameters() -> None:
    import pytest
    from src.market_data.binance.futures import TokenBucket

    with pytest.raises(ValueError, match="rate_per_s"):
        TokenBucket(0.0, 1)
    with pytest.raises(ValueError, match="burst"):
        TokenBucket(1.0, 0)


def test_token_bucket_serializes_concurrent_acquires() -> None:
    import threading
    import time
    from src.market_data.binance.futures import TokenBucket

    bucket = TokenBucket(50.0, 1)
    start = time.monotonic()

    def _worker() -> None:
        for _ in range(5):
            bucket.acquire()

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - start

    assert elapsed >= (20 - 1) / 50.0 * 0.95


def test_funding_rate_limiter_stays_within_binance_budget() -> None:
    import pytest
    from src.market_data.binance import futures

    budget = (futures.FUNDING_RATE_LIMIT_PER_WINDOW, futures.FUNDING_RATE_WINDOW_S, futures.FUNDING_BURST)
    assert budget == (500, 300.0, 5)
    assert pytest.approx(500 / 300 * 0.8) == futures.FUNDING_REQUESTS_PER_SECOND
    assert isinstance(futures.FUNDING_RATE_LIMITER, futures.TokenBucket)
    assert futures.FUNDING_RATE_LIMITER.rate_per_s == pytest.approx(futures.FUNDING_REQUESTS_PER_SECOND)


def test_fetch_funding_rate_history_acquires_limiter_per_page_without_fixed_sleep(monkeypatch) -> None:
    import src.market_data.binance.futures as futures_module
    from src.market_data.binance.futures import BinanceClient

    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "parse8601", lambda value: 0 if "00:00:00" in value else 3_600_000)
    sleeps: list[float] = []
    monkeypatch.setattr("src.market_data.binance.futures.time.sleep", sleeps.append)

    events: list[str] = []

    class _Spy:
        def acquire(self) -> float:
            events.append("acquire")
            return 0.0

    monkeypatch.setattr(futures_module, "FUNDING_RATE_LIMITER", _Spy())

    class Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload
            self.headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return self.payload

    pages = iter([
        Response(b'[{"fundingTime":1000,"fundingRate":"0.0001"}]'),
        Response(b"[]"),
    ])

    def _urlopen(*args, **kwargs):
        events.append("request")
        return next(pages)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    rates = client.fetch_funding_rate_history("BTC/USDT", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z")

    assert rates.to_dict("records") == [{"timestamp": 1000, "funding_rate": 0.0001}]
    assert events == ["acquire", "request"]
    assert sleeps == []


def test_fetch_funding_rate_history_uses_waf_safe_request_limit(monkeypatch) -> None:
    import urllib.parse
    import src.market_data.binance.futures as futures_module
    from src.market_data.binance.futures import BinanceClient, FUNDING_RATE_REQUEST_LIMIT

    # Given
    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "parse8601", lambda value: 0 if "00:00:00" in value else 3_600_000)

    class _Spy:
        def acquire(self) -> float:
            return 0.0

    monkeypatch.setattr(futures_module, "FUNDING_RATE_LIMITER", _Spy())

    class Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload
            self.headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return self.payload

    urls: list[str] = []

    def _urlopen(req, *args, **kwargs):
        urls.append(req.full_url)
        return Response(b"[]")

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    # When
    client.fetch_funding_rate_history("BTC/USDT", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z")

    # Then
    assert FUNDING_RATE_REQUEST_LIMIT == 100
    assert len(urls) == 1
    query = urllib.parse.parse_qs(urllib.parse.urlparse(urls[0]).query)
    assert query["limit"] == [str(FUNDING_RATE_REQUEST_LIMIT)]


def _funding_stub_client(monkeypatch, pages: list[bytes]):
    """Stub BinanceClient with queued fundingRate payloads; returns (client, calls)."""
    import src.market_data.binance.futures as futures_module
    from src.market_data.binance.futures import BinanceClient

    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(
        client.exchange,
        "parse8601",
        lambda value: 0 if value.startswith("2024-01-01") else 10_000_000,
    )

    class _Spy:
        def acquire(self) -> float:
            return 0.0

    monkeypatch.setattr(futures_module, "FUNDING_RATE_LIMITER", _Spy())

    class Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return self.payload

    queue = iter([Response(p) for p in pages])
    calls: list[str] = []

    def _urlopen(req, *args, **kwargs):
        calls.append(req.full_url)
        return next(queue)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    return client, calls


def test_fetch_funding_rate_history_short_page_issues_one_request(monkeypatch) -> None:
    import json

    rows = [{"fundingTime": t, "fundingRate": "0.0001"} for t in (100, 200, 300, 400, 500, 600)]
    client, calls = _funding_stub_client(monkeypatch, [json.dumps(rows).encode()])

    rates = client.fetch_funding_rate_history("BTC/USDT", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")

    assert len(calls) == 1
    assert len(rates) == 6


def test_fetch_funding_rate_history_full_page_continues(monkeypatch) -> None:
    import json

    first = [{"fundingTime": t, "fundingRate": "0.0001"} for t in range(1, 101)]
    second = [{"fundingTime": t, "fundingRate": "0.0002"} for t in range(101, 201)]
    third = [{"fundingTime": t, "fundingRate": "0.0003"} for t in range(201, 208)]
    client, calls = _funding_stub_client(
        monkeypatch,
        [json.dumps(first).encode(), json.dumps(second).encode(), json.dumps(third).encode()],
    )

    rates = client.fetch_funding_rate_history("BTC/USDT", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")

    assert len(calls) == 3
    assert len(rates) == 207
    assert rates["timestamp"].is_unique
    assert rates["timestamp"].is_monotonic_increasing


def test_fetch_funding_rate_history_empty_range_still_one_request(monkeypatch) -> None:
    client, calls = _funding_stub_client(monkeypatch, [b"[]"])

    rates = client.fetch_funding_rate_history("BTC/USDT", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")

    assert len(calls) == 1
    assert list(rates.columns) == ["timestamp", "funding_rate"]
    assert rates.empty


def test_fetch_funding_rate_history_rows_unchanged_from_legacy(monkeypatch) -> None:
    import pandas as pd

    payload = (
        b'[{"fundingTime":300,"fundingRate":"0.0003"},'
        b'{"fundingTime":100,"fundingRate":"0.0001"},'
        b'{"fundingTime":200,"fundingRate":"0.0002"},'
        b'{"fundingTime":100,"fundingRate":"0.0001"}]'
    )
    client, calls = _funding_stub_client(monkeypatch, [payload])

    rates = client.fetch_funding_rate_history("BTC/USDT", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")

    expected = pd.DataFrame(
        [(100, 0.0001), (200, 0.0002), (300, 0.0003)],
        columns=["timestamp", "funding_rate"],
    )
    pd.testing.assert_frame_equal(rates, expected)
    assert len(calls) == 1


def test_fetch_funding_rate_history_paginates_beyond_request_limit(monkeypatch) -> None:
    import json
    import urllib.parse
    import src.market_data.binance.futures as futures_module
    from src.market_data.binance.futures import BinanceClient, FUNDING_RATE_REQUEST_LIMIT

    # Given
    client = BinanceClient()
    monkeypatch.setattr(client.exchange, "market", lambda symbol: {"id": symbol.replace("/", "")})
    monkeypatch.setattr(client.exchange, "parse8601", lambda value: 0 if value.startswith("2024-01-01") else 10_000_000)

    class _Spy:
        def acquire(self) -> float:
            return 0.0

    monkeypatch.setattr(futures_module, "FUNDING_RATE_LIMITER", _Spy())

    class Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload
            self.headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return self.payload

    first = [{"fundingTime": t, "fundingRate": "0.0001"} for t in range(1, FUNDING_RATE_REQUEST_LIMIT + 1)]
    second = [{"fundingTime": 200, "fundingRate": "0.0002"}]
    pages = iter([json.dumps(first).encode(), json.dumps(second).encode()])
    starts: list[int] = []

    def _urlopen(req, *args, **kwargs):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)
        starts.append(int(query["startTime"][0]))
        return Response(next(pages))

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    # When
    rates = client.fetch_funding_rate_history("BTC/USDT", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")

    # Then
    assert starts == [0, 101]
    assert len(rates) == FUNDING_RATE_REQUEST_LIMIT + 1
    assert rates["timestamp"].tolist()[-1] == 200
