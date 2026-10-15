"""Fault-injecting transport for drill runs."""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, Protocol
from urllib.parse import urlsplit

from src.live.rest import HttpResponse
from src.live.settings import TESTNET_FAPI_URL
from src.live.testnet_drill.guard import DrillRefused, _host


class _Transport(Protocol):
    """Structural transport contract shared by production and stub transports."""

    def call(self, method: str, url: str, headers: Mapping[str, str]) -> HttpResponse: ...


class FaultAction(str, Enum):  # noqa: UP042 - contract pins the (str, Enum) base
    """Closed set of transport fault actions."""

    PASS = "pass"  # noqa: S105 - enum member name, not a credential
    DROP_REQUEST = "drop_request"
    DELIVER_DROP_RESPONSE = "deliver_drop_response"
    DELAY_DELIVER = "delay_deliver"
    SYNTHETIC_STATUS = "synthetic_status"
    CORRUPT_BODY = "corrupt_body"
    TRUNCATE_BODY = "truncate_body"
    KILL_BEFORE_DELIVERY = "kill_before_delivery"
    KILL_AFTER_DELIVERY = "kill_after_delivery"
    KILL_ON_RESPONSE = "kill_on_response"


_RESERVED_QUERY_PARAMS = frozenset({"signature", "timestamp", "recvWindow"})


@dataclass(frozen=True, slots=True)
class FaultRule:
    """Match one request class and apply one action."""

    method: str
    path: str
    params: tuple[tuple[str, str], ...] = ()
    occurrence: int | None = None
    action: FaultAction = FaultAction.PASS
    times: int = 1
    status: int = 200
    headers: tuple[tuple[str, str], ...] = ()
    body: bytes = b"{}"
    delay_s: float = 0.0
    truncate_to: int = 0
    exit_code: int = 137

    def __post_init__(self) -> None:
        if not self.method or not self.path:
            raise ValueError("FaultRule requires a method and a path")
        for key, _ in self.params:
            if key in _RESERVED_QUERY_PARAMS:
                raise ValueError(f"FaultRule predicate must not use {key!r}")
        if self.occurrence is not None and self.occurrence < 1:
            raise ValueError("FaultRule occurrence is 1-based")
        if self.times < 1:
            raise ValueError("FaultRule times must be at least 1")
        if self.delay_s < 0:
            raise ValueError("FaultRule delay_s must be non-negative")
        if self.truncate_to < 0:
            raise ValueError("FaultRule truncate_to must be non-negative")


@dataclass(frozen=True, slots=True)
class WireRecord:
    """One observed request/response pair with delivery ground truth."""

    seq: int
    method: str
    path: str
    url: str
    request_headers: tuple[tuple[str, str], ...]
    status_code: int | None
    response_headers: tuple[tuple[str, str], ...]
    response_body: bytes
    delivered: bool | None
    error: str | None = None


def _filter_headers(headers: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    kept: list[tuple[str, str]] = []
    for key, value in headers.items():
        lowered = key.lower()
        if lowered == "x-mbx-apikey":
            continue
        if lowered.startswith("x-mbx-") or lowered in ("retry-after", "content-type"):
            kept.append((key, value))
    return tuple(kept)


def redact(text: str) -> str:
    """Strip credential fields from query strings, JSON and header-like text."""
    return re.sub(
        r'''(?i)((?:["']?)(?:signature|x-mbx-apikey|api[_-]?key|api[_-]?secret|secret)(?:["']?)\s*[:=]\s*["']?)[^&\s,"'}]+''',
        r"\1<stripped>", text,
    )


def _strip_url(url: str) -> tuple[str, str, dict[str, list[str]]]:
    parts = urlsplit(url)
    kept = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in {"signature", "apikey", "api_key", "secret", "apisecret", "api_secret"}]
    clean = parts._replace(query=urllib.parse.urlencode(kept)).geturl()
    multi: dict[str, list[str]] = {}
    for key, value in kept:
        multi.setdefault(key, []).append(value)
    return parts.path, clean, multi


class FaultInjectingTransport:
    """Transport wrapper recording every exchange and applying FaultRules below signing."""

    def __init__(
        self,
        inner: _Transport,
        rules: Sequence[FaultRule],
        *,
        base_url: str,
        sleep: Callable[[float], None],
        kill: Callable[[int], NoReturn],
        on_record: Callable[[WireRecord], None] | None = None,
    ) -> None:
        if _host(base_url) != _host(TESTNET_FAPI_URL):
            raise DrillRefused("drill refused: base_url")
        self._inner = inner
        self._rules = list(rules)
        self._remaining = [rule.times for rule in self._rules]
        self._sleep = sleep
        self._kill = kill
        self._on_record = on_record
        self._records: list[WireRecord] = []
        self._history: list[tuple[str, str, dict[str, list[str]]]] = []

    @property
    def records(self) -> tuple[WireRecord, ...]:
        """Append-only wire records with monotonic sequence numbers."""
        return tuple(self._records)

    def close(self) -> None:
        """Close the inner transport when it supports it."""
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()

    def _rule_matches_key(self, rule: FaultRule, method: str, path: str, query: Mapping[str, list[str]]) -> bool:
        if rule.method.upper() != method or rule.path != path:
            return False
        return all(value in query.get(key, []) for key, value in rule.params)

    def _select_rule(self, method: str, path: str, query: Mapping[str, list[str]]) -> int | None:
        for index, rule in enumerate(self._rules):
            if self._remaining[index] <= 0:
                continue
            if not self._rule_matches_key(rule, method, path, query):
                continue
            if rule.occurrence is not None:
                seen = sum(1 for past in self._history if self._rule_matches_key(rule, *past))
                if seen < rule.occurrence:
                    continue
            self._remaining[index] -= 1
            return index
        return None

    def _append(
        self,
        method: str,
        path: str,
        url: str,
        headers: Mapping[str, str],
        *,
        status_code: int | None,
        response_headers: Mapping[str, str] | tuple[tuple[str, str], ...],
        response_body: bytes,
        delivered: bool | None,
        error: str | None = None,
    ) -> None:
        stored_headers = response_headers if isinstance(response_headers, tuple) else _filter_headers(response_headers)
        self._records.append(
            WireRecord(
                seq=len(self._records) + 1,
                method=method,
                path=path,
                url=url,
                request_headers=_filter_headers(headers),
                status_code=status_code,
                response_headers=tuple((k, redact(v)) for k, v in stored_headers),
                response_body=_redact_body(response_body, headers),
                delivered=delivered,
                error=error,
            )
        )
        if self._on_record is not None:
            self._on_record(self._records[-1])

    def call(self, method: str, url: str, headers: Mapping[str, str]) -> HttpResponse:
        if _host(url) != _host(TESTNET_FAPI_URL):
            raise DrillRefused("drill refused: request_host")
        method = method.upper()
        path, clean_url, query = _strip_url(url)
        self._history.append((method, path, query))
        selected = self._select_rule(method, path, query)
        if selected is None:
            return self._forward(method, path, clean_url, url, headers)
        rule = self._rules[selected]
        if rule.action is FaultAction.DROP_REQUEST:
            self._append(method, path, clean_url, headers, status_code=None, response_headers=(), response_body=b"", delivered=False)
            raise TimeoutError(f"drill fault {rule.action.value} at {path}")
        if rule.action is FaultAction.SYNTHETIC_STATUS:
            stored = _filter_headers(dict(rule.headers))
            self._append(method, path, clean_url, headers, status_code=rule.status, response_headers=stored, response_body=rule.body, delivered=False)
            return HttpResponse(status_code=rule.status, headers=dict(rule.headers), body=rule.body)
        if rule.action is FaultAction.DELAY_DELIVER:
            self._sleep(rule.delay_s)
            return self._forward(method, path, clean_url, url, headers)
        if rule.action is FaultAction.KILL_BEFORE_DELIVERY:
            self._append(method, path, clean_url, headers, status_code=None, response_headers=(), response_body=b"", delivered=False)
            self._kill(rule.exit_code)
            raise AssertionError("unreachable: kill must not return")
        response = self._forward(method, path, clean_url, url, headers, record=False)
        if rule.action is FaultAction.DELIVER_DROP_RESPONSE:
            self._append(method, path, clean_url, headers, status_code=response.status_code, response_headers=response.headers, response_body=response.body, delivered=True)
            raise TimeoutError(f"drill fault {rule.action.value} at {path}")
        if rule.action in (FaultAction.KILL_AFTER_DELIVERY, FaultAction.KILL_ON_RESPONSE):
            self._append(method, path, clean_url, headers, status_code=response.status_code, response_headers=response.headers, response_body=response.body, delivered=True)
            self._kill(rule.exit_code)
            raise AssertionError("unreachable: kill must not return")
        if rule.action is FaultAction.CORRUPT_BODY:
            mutated = HttpResponse(status_code=response.status_code, headers=dict(response.headers), body=rule.body)
        elif rule.action is FaultAction.TRUNCATE_BODY:
            mutated = HttpResponse(status_code=response.status_code, headers=dict(response.headers), body=response.body[: rule.truncate_to])
        else:
            mutated = response
        self._append(method, path, clean_url, headers, status_code=response.status_code, response_headers=response.headers, response_body=response.body, delivered=True)
        return mutated

    def _forward(
        self,
        method: str,
        path: str,
        clean_url: str,
        original_url: str,
        headers: Mapping[str, str],
        *,
        record: bool = True,
    ) -> HttpResponse:
        try:
            response = self._inner.call(method, original_url, headers)
        except Exception as exc:
            self._append(method, path, clean_url, headers, status_code=None, response_headers=(), response_body=b"", delivered=None, error=type(exc).__name__)
            raise
        if record:
            self._append(method, path, clean_url, headers, status_code=response.status_code, response_headers=response.headers, response_body=response.body, delivered=True)
        return response


def _redact_body(body: bytes, headers: Mapping[str, str]) -> bytes:
    for key, value in headers.items():
        if key.lower() == "x-mbx-apikey" and value:
            body = body.replace(value.encode(), b"<stripped>")
    return re.sub(
        rb'''(?i)((?:["']?)(?:signature|x-mbx-apikey|api[_-]?key|api[_-]?secret|secret)(?:["']?)\s*[:=]\s*["']?)[^&\s,"'}]+''',
        rb"\1<stripped>", body,
    )
