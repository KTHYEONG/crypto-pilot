"""Binance USDⓈ-M REST client with the SHADOW/PAPER mutation chokepoint.

I-SHADOW-CHOKE: SHADOW/PAPER 모드의 변이 요청은 전송 계층 최상단에서 억제되며
네트워크에 절대 도달하지 않는다. 오류 처리 정책은 오직 레지스트리 결정만 따른다.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import http.client
import json
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from pydantic import SecretStr

from src.common.errors import DataIntegrityError
from src.live.audit import AuditLog
from src.live.errors import (
    ErrorAction,
    LiveTradingError,
    OrderObsolete,
    ShadowModeViolation,
    VenueError,
    payload_digest,
    resolve_error_action,
)
from src.live.settings import ExecutionMode

#: SHADOW 모드에서 예외적으로 허용되는 변이 경로. 검증 기간 전체에 비워둔다.
SHADOW_ALLOWED_MUTATIONS: frozenset[str] = frozenset()

_RETRY_BACKOFF_SECONDS = 1.0
_RETRY_BACKOFF_LONG_SECONDS = 10.0
_MAX_ATTEMPTS = 3
_HTTP_TIMEOUT_SECONDS = 30.0

_USED_WEIGHT_1M_HEADER = "X-MBX-USED-WEIGHT-1M"
_ORDER_COUNT_1M_HEADER = "X-MBX-ORDER-COUNT-1M"
_ORDER_COUNT_10S_HEADER = "X-MBX-ORDER-COUNT-10S"


@dataclass(frozen=True, slots=True)
class ShadowResponse:
    """전송되지 않은 변이 요청의 자리표시자 응답."""

    method: str
    path: str
    payload_digest: str
    status: str = "suppressed"

    @classmethod
    def suppressed(cls, method: str, path: str, payload_digest_str: str) -> ShadowResponse:
        return cls(method=method, path=path, payload_digest=payload_digest_str)


@dataclass(frozen=True, slots=True)
class PaperResponse:
    """PAPER 모드에서 억제된 변이 요청의 자리표시자 응답.

    SHADOW 억제와 동일한 출처 필드(method/path/payload_digest)를 운반하지만
    별개 타입이라 호출부가 로컬 체결 시뮬레이터로 라우팅할 수 있다. 이 응답이
    반환되었다는 것은 요청이 네트워크에 절대 도달하지 않았음을 보증한다.
    """

    method: str
    path: str
    payload_digest: str
    status: str = "suppressed"

    @classmethod
    def suppressed(cls, method: str, path: str, payload_digest_str: str) -> PaperResponse:
        return cls(method=method, path=path, payload_digest=payload_digest_str)


@dataclass(slots=True)
class RateLimitState:
    """거래소 보고값 그대로 보관하는 레이트리밋 상태(자체 추정 금지)."""

    used_weight_1m: int | None = None
    order_count_1m: int | None = None
    order_count_10s: int | None = None


@dataclass(slots=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True, slots=True)
class RateLimits:
    """exchangeInfo.rateLimits 에서 파싱한 거래소 공식 한도(추정 금지)."""

    request_weight_1m: int
    orders_1m: int
    orders_10s: int


def parse_rate_limits(exchange_info: Mapping[str, Any]) -> RateLimits:
    """REQUEST_WEIGHT/ORDERS 한도를 읽는다. 누락 시 DataIntegrityError(추정 금지)."""
    rate_limits = exchange_info.get("rateLimits")
    if not isinstance(rate_limits, list):
        raise DataIntegrityError("exchangeInfo missing rateLimits")
    weight_1m: int | None = None
    orders_1m: int | None = None
    orders_10s: int | None = None
    for entry in rate_limits:
        if not isinstance(entry, dict):
            continue
        filter_type = entry.get("filterType")
        interval = entry.get("interval")
        limit = entry.get("limit")
        if filter_type == "REQUEST_WEIGHT" and interval == "1m" and limit is not None:
            weight_1m = int(limit)
        elif filter_type == "ORDERS" and interval == "1m" and limit is not None:
            orders_1m = int(limit)
        elif filter_type == "ORDERS" and interval == "10s" and limit is not None:
            orders_10s = int(limit)
    if weight_1m is None or orders_1m is None or orders_10s is None:
        raise DataIntegrityError(
            "exchangeInfo rateLimits missing REQUEST_WEIGHT/ORDERS limits"
        )
    return RateLimits(
        request_weight_1m=weight_1m, orders_1m=orders_1m, orders_10s=orders_10s
    )


class KeepAliveTransport:
    """호스트당 1개의 HTTPSConnection 을 재사용해 TLS 핸드셰이크를 제거한다.

    UrllibTransport 와 동일한 call 시그니처를 유지하므로 기존 스텁 테스트는 영향받지
    않는다. 연결 사망 시 1회 재연결 후 재시도하고, 그래도 실패하면 예외를 전파한다.
    """

    def __init__(self, base_url: str) -> None:
        parsed = urllib.parse.urlsplit(base_url if "://" in base_url else f"https://{base_url}")
        self._host = parsed.hostname or parsed.path
        self._port = parsed.port or (443 if parsed.scheme != "http" else 80)
        self._scheme = parsed.scheme
        self._connection: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        if self._scheme == "http":
            connection: http.client.HTTPConnection = http.client.HTTPConnection(
                self._host, self._port, timeout=_HTTP_TIMEOUT_SECONDS
            )
        else:
            connection = http.client.HTTPSConnection(
                self._host, self._port, timeout=_HTTP_TIMEOUT_SECONDS
            )
        self._connection = connection
        return connection

    def _drop(self) -> None:
        if self._connection is not None:
            with contextlib.suppress(OSError):
                self._connection.close()
            self._connection = None

    def call(self, method: str, url: str, headers: Mapping[str, str]) -> HttpResponse:
        parts = urllib.parse.urlsplit(url)
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        last_error: Exception | None = None
        for attempt in range(2):
            connection = self._connection if attempt == 0 and self._connection else self._connect()
            try:
                connection.request(method.upper(), path, body=None, headers=dict(headers))
                response = connection.getresponse()
                body = response.read()
                return HttpResponse(
                    status_code=response.status,
                    headers=dict(response.getheaders()),
                    body=body,
                )
            except (http.client.HTTPException, OSError, TimeoutError) as exc:
                last_error = exc
                self._drop()
        raise last_error if last_error is not None else OSError("transport failed")

    def close(self) -> None:
        self._drop()


class UrllibTransport:
    """기본 전송 계층. 단위 테스트에서는 항상 스텁으로 대체된다."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def call(self, method: str, url: str, headers: Mapping[str, str]) -> HttpResponse:
        request = urllib.request.Request(  # noqa: S310
            url, method=method.upper(), headers=dict(headers)
        )
        try:
            with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310
                return HttpResponse(
                    status_code=response.status,
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(
                status_code=exc.code,
                headers=dict(exc.headers.items()) if exc.headers else {},
                body=exc.read(),
            )


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


class BinanceFuturesRestClient:
    """서명/레이트리밋/정책 재시도/SHADOW 초크포인트를 담당하는 REST 클라이언트."""

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr | None,
        api_secret: SecretStr | None,
        mode: ExecutionMode,
        audit: AuditLog,
        *,
        recv_window_ms: int = 5000,
        session: Any | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._api_secret = api_secret
        self._mode = mode
        self._audit = audit
        self._recv_window_ms = recv_window_ms
        self._transport = session if session is not None else KeepAliveTransport(base_url)
        self.rate_state = RateLimitState()
        self._time_offset_ms = 0

    # ------------------------------------------------------------------ helpers

    def sync_server_time(self) -> None:
        payload = self.request("GET", "/fapi/v1/time")
        server_ms = int(payload["serverTime"])
        self._time_offset_ms = server_ms - int(time.time() * 1000)

    def _signed_query(self, params: Mapping[str, Any]) -> str:
        if self._api_secret is None:
            raise LiveTradingError("signed request requires API secret")
        signed_params = dict(params)
        signed_params["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
        signed_params["recvWindow"] = self._recv_window_ms
        query = urllib.parse.urlencode(signed_params)
        signature = hmac.new(
            self._api_secret.get_secret_value().encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        return f"{query}&signature={signature}"

    def _update_rate_state(self, headers: Mapping[str, str]) -> None:
        weight = _header_value(headers, _USED_WEIGHT_1M_HEADER)
        count_1m = _header_value(headers, _ORDER_COUNT_1M_HEADER)
        count_10s = _header_value(headers, _ORDER_COUNT_10S_HEADER)
        if weight is not None:
            self.rate_state.used_weight_1m = int(weight)
        if count_1m is not None:
            self.rate_state.order_count_1m = int(count_1m)
        if count_10s is not None:
            self.rate_state.order_count_10s = int(count_10s)

    @staticmethod
    def _error_code(body: Any) -> int | None:
        if isinstance(body, dict) and "code" in body:
            try:
                return int(body["code"])
            except (TypeError, ValueError):
                return None
        return None

    # ------------------------------------------------------------------ request

    def request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        signed: bool = False,
    ) -> Any:
        method = method.upper()
        base_params = dict(params or {})

        if self._mode in (ExecutionMode.SHADOW, ExecutionMode.PAPER) and method != "GET":
            if path not in SHADOW_ALLOWED_MUTATIONS:
                digest = payload_digest(repr(base_params))
                self._audit.record_suppressed(method, path, base_params)
                if self._mode is ExecutionMode.PAPER:
                    return PaperResponse.suppressed(method, path, digest)
                return ShadowResponse.suppressed(method, path, digest)
            raise ShadowModeViolation(
                f"mutation to {path} is not allowed by the shadow whitelist"
            )

        resynced_clock = False
        last_error: VenueError | None = None
        for _attempt in range(1, _MAX_ATTEMPTS + 1):
            query = self._signed_query(base_params) if signed else (
                urllib.parse.urlencode(base_params) if base_params else ""
            )
            url = f"{self._base_url}{path}" + (f"?{query}" if query else "")
            headers: dict[str, str] = {}
            if self._api_key is not None:
                headers["X-MBX-APIKEY"] = self._api_key.get_secret_value()

            response = self._transport.call(method, url, headers)
            self._update_rate_state(response.headers)

            body: Any = None
            if response.body:
                try:
                    body = json.loads(response.body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body = None

            if response.status_code == 418:
                raise LiveTradingError(f"IP ban received at {path}; halting")
            if response.status_code == 429:
                retry_after = _header_value(response.headers, "Retry-After")
                time.sleep(float(retry_after) if retry_after else _RETRY_BACKOFF_SECONDS)
                last_error = VenueError(
                    "rate limited", code=None, http_status=429, path=path,
                    payload_digest=payload_digest(response.body.decode("utf-8", errors="replace")),
                )
                continue

            error_code = self._error_code(body)
            if response.status_code < 400 and error_code is None:
                return body

            action = resolve_error_action(error_code)
            if action is ErrorAction.FAIL_CLOSED or action is ErrorAction.RESYNC_THEN_DECIDE:
                raise VenueError(
                    f"venue rejected request at {path}",
                    code=error_code,
                    http_status=response.status_code,
                    path=path,
                    payload_digest=payload_digest(
                        response.body.decode("utf-8", errors="replace") if response.body else ""
                    ),
                )
            if action is ErrorAction.BENIGN:
                return body
            if action is ErrorAction.BENIGN_ABORT:
                raise OrderObsolete(
                    f"venue reports intent is obsolete at {path} (code={error_code})"
                )
            if action is ErrorAction.BENIGN_REPRICE:
                raise VenueError(
                    "post-only rejection (reprice signal)",
                    code=error_code,
                    http_status=response.status_code,
                    path=path,
                    payload_digest=payload_digest(
                        response.body.decode("utf-8", errors="replace") if response.body else ""
                    ),
                )
            if action is ErrorAction.RESYNC_CLOCK:
                if resynced_clock:
                    break
                resynced_clock = True
                self.sync_server_time()
                continue
            time.sleep(
                _RETRY_BACKOFF_LONG_SECONDS
                if action is ErrorAction.RETRY_BACKOFF_LONG
                else _RETRY_BACKOFF_SECONDS
            )

        raise last_error or VenueError(
            f"request to {path} exhausted retries",
            code=None,
            http_status=0,
            path=path,
            payload_digest=payload_digest(None),
        )

    # ------------------------------------------------------- endpoint shortcuts

    def exchange_info(self) -> Any:
        return self.request("GET", "/fapi/v1/exchangeInfo")

    def book_ticker(self, symbol: str) -> dict[str, Any]:
        payload = self.request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        return cast(dict[str, Any], payload)

    def book_tickers(self) -> dict[str, dict[str, Any]]:
        """무심볼 GET /fapi/v1/ticker/bookTicker(weight 5)로 전 종목을 1회에 수집한다."""
        payload = self.request("GET", "/fapi/v1/ticker/bookTicker")
        if not isinstance(payload, list):
            raise DataIntegrityError("book_tickers endpoint returned an unexpected schema")
        indexed: dict[str, dict[str, Any]] = {}
        for entry in payload:
            if isinstance(entry, dict) and "symbol" in entry:
                indexed[str(entry["symbol"])] = entry
        return indexed

    def new_order(self, params: Mapping[str, Any]) -> Any:
        return self.request("POST", "/fapi/v1/order", params, signed=True)

    def cancel_order(self, symbol: str, orig_client_order_id: str) -> Any:
        return self.request(
            "DELETE", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": orig_client_order_id},
            signed=True,
        )

    def query_order(self, symbol: str, orig_client_order_id: str) -> Any:
        return self.request(
            "GET", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": orig_client_order_id},
            signed=True,
        )

    def open_orders(self) -> list[dict[str, Any]]:
        """무심볼 GET /fapi/v1/openOrders: 고아 주문 정리를 위한 전량 조회."""
        payload = self.request("GET", "/fapi/v1/openOrders", signed=True)
        if not isinstance(payload, list):
            raise DataIntegrityError("openOrders endpoint returned an unexpected schema")
        return cast(list[dict[str, Any]], payload)

    def account(self) -> Any:
        return self.request("GET", "/fapi/v2/account", signed=True)

    def position_risk(self) -> Any:
        return self.request("GET", "/fapi/v2/positionRisk", signed=True)
