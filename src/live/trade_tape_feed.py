"""Binance aggTrade WebSocket capture for PAPER tape-through fills (observability-only)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.live.depth_capture import _default_connect

_logger = logging.getLogger(__name__)

_RECONNECT_BACKOFF_MAX_S: float = 30.0
_STOP_JOIN_TIMEOUT_S: float = 10.0


@dataclass(frozen=True, slots=True)
class TapeReadiness:
    ready: bool
    seen: int
    required: int
    missing: tuple[str, ...]
    waited_s: float


@dataclass(frozen=True, slots=True)
class TapeSummary:
    trades_received: int
    symbols_seen: int
    reconnects: int
    dropped_oldest: int


def _tape_stream_url(stream_url: str, symbols: Sequence[str]) -> str:
    streams = "/".join(f"{str(symbol).lower()}@aggTrade" for symbol in symbols)
    return f"{stream_url}?streams={streams}"


def _parse_agg_trade(message: Any) -> tuple[str, int, float, bool] | None:
    if not isinstance(message, dict):
        return None
    data = message.get("data")
    if isinstance(data, dict):
        message = data
    symbol = message.get("s")
    if symbol is None:
        return None
    try:
        transact_ms = int(message.get("T"))
    except (TypeError, ValueError):
        return None
    try:
        price = float(message.get("p"))
    except (TypeError, ValueError):
        return None
    buyer_maker = message.get("m")
    if not isinstance(buyer_maker, bool):
        return None
    if not (price > 0.0):
        return None
    return (str(symbol), transact_ms, price, buyer_maker)


class LiveTradeTape:
    """In-memory aggTrade feed for the traded symbols during one execution window."""

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        stream_url: str,
        max_trades_per_symbol: int,
        stale_after_s: float,
        connect: Callable[[str], Any] | None = None,
    ) -> None:
        self._symbols = [str(s) for s in symbols]
        self._symbol_set = frozenset(self._symbols)
        self._stream_url = str(stream_url)
        self._max_trades = int(max_trades_per_symbol)
        if self._max_trades < 1:
            raise ValueError(f"max_trades_per_symbol must be >= 1, got {max_trades_per_symbol!r}")
        self._stale_after_s = float(stale_after_s)
        self._connect = connect if connect is not None else _default_connect
        self._lock = threading.Lock()
        self._trades: dict[str, deque[tuple[int, float, bool]]] = {
            symbol: deque() for symbol in self._symbols
        }
        self._last_seen_s: dict[str, float] = {}
        self._last_any_s: float = 0.0
        self._connected = False
        self._trades_received = 0
        self._dropped_oldest = 0
        self._malformed = 0
        self._reconnects = 0
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._started = False
        self._summary: TapeSummary | None = None

    def _observe(self, symbol: str, transact_ms: int, price: float, buyer_maker: bool) -> None:
        now_s = time.time()
        with self._lock:
            buf = self._trades[symbol]
            buf.append((transact_ms, price, buyer_maker))
            if len(buf) > self._max_trades:
                buf.popleft()
                self._dropped_oldest += 1
            self._trades_received += 1
            self._last_seen_s[symbol] = now_s
            self._last_any_s = now_s
            if not self._ready_event.is_set() and all(s in self._last_seen_s for s in self._symbols):
                self._ready_event.set()

    def _snapshot_seen(self) -> set[str]:
        with self._lock:
            return set(self._last_seen_s)

    def start(
        self,
        *,
        ready_timeout_s: float = 0.0,
        shutdown: Any | None = None,
    ) -> TapeReadiness:
        start_mono = time.monotonic()
        try:
            timeout = float(ready_timeout_s)
            if not timeout >= 0:
                raise ValueError("ready_timeout_s must be non-negative")
        except Exception as exc:
            _logger.warning("[EXEC] stage=paper_tape status=READY_WAIT_FAILED error=%s", exc)
            return TapeReadiness(ready=False, seen=0, required=len(self._symbols),
                                 missing=tuple(self._symbols), waited_s=0.0)
        try:
            if not self._symbols:
                return TapeReadiness(ready=True, seen=0, required=0, missing=(), waited_s=0.0)
            if self._started:
                seen = self._snapshot_seen()
                missing = tuple(s for s in self._symbols if s not in seen)
                return TapeReadiness(ready=not missing, seen=len(seen), required=len(self._symbols),
                                     missing=missing, waited_s=0.0)
            self._started = True
            self._ready_event.clear()
            thread = threading.Thread(target=self._thread_main, name="paper-trade-tape", daemon=True)
            thread.start()
            self._thread = thread
            if timeout == 0.0:
                seen = self._snapshot_seen()
                missing = tuple(s for s in self._symbols if s not in seen)
                return TapeReadiness(ready=False, seen=len(seen), required=len(self._symbols),
                                     missing=missing, waited_s=0.0)
            deadline = start_mono + timeout
            while True:
                if self._ready_event.is_set():
                    break
                if shutdown is not None and bool(getattr(shutdown, "requested", False)):
                    break
                thread = self._thread
                if thread is not None and not thread.is_alive():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._ready_event.wait(timeout=min(0.25, remaining))
            waited = max(0.0, time.monotonic() - start_mono)
            seen = self._snapshot_seen()
            missing = tuple(s for s in self._symbols if s not in seen)
            return TapeReadiness(ready=not missing, seen=len(seen), required=len(self._symbols),
                                 missing=missing, waited_s=waited)
        except Exception as exc:
            _logger.warning("[EXEC] stage=paper_tape status=READY_WAIT_FAILED error=%s", exc)
            seen = self._snapshot_seen()
            missing = tuple(s for s in self._symbols if s not in seen)
            return TapeReadiness(ready=False, seen=len(seen), required=len(self._symbols),
                                 missing=missing, waited_s=max(0.0, time.monotonic() - start_mono))

    def traded_through(
        self, symbol: str, side: str, limit: Decimal, since_ms: int
    ) -> int | None:
        try:
            name = str(symbol)
            threshold = float(limit)
            floor = int(since_ms)
            if side not in ("BUY", "SELL"):
                return None
            want_buyer_maker = side == "BUY"
            with self._lock:
                buf = self._trades.get(name)
                if not buf:
                    return None
                return min(
                    (
                        t for t, p, bm in buf
                        if t >= floor and bm is want_buyer_maker
                        and (p < threshold if want_buyer_maker else p > threshold)
                    ),
                    default=None,
                )
        except Exception:
            return None

    def is_live(self, symbol: str, now_ms: int) -> bool:
        try:
            if not self._connected:
                return False
            stale_ms = self._stale_after_s * 1000.0
            now_s = float(now_ms) / 1000.0
            with self._lock:
                last_any = self._last_any_s
                last_sym = self._last_seen_s.get(str(symbol), 0.0)
            if last_any > 0.0 and (now_s - last_any) * 1000.0 <= stale_ms:
                return True
            return last_sym > 0.0 and (now_s - last_sym) * 1000.0 <= stale_ms
        except Exception:
            return False

    def stop(self) -> TapeSummary:
        if self._summary is not None:
            return self._summary
        try:
            if not self._started:
                summary = TapeSummary(trades_received=0, symbols_seen=0, reconnects=0, dropped_oldest=0)
                self._summary = summary
                return summary
            self._stop_requested = True
            thread = self._thread
            if thread is not None:
                thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
                if thread.is_alive():
                    _logger.error("[EXEC] stage=paper_tape status=THREAD_STUCK")
            with self._lock:
                seen = len(self._last_seen_s)
                summary = TapeSummary(
                    trades_received=self._trades_received,
                    symbols_seen=seen,
                    reconnects=self._reconnects,
                    dropped_oldest=self._dropped_oldest,
                )
            self._summary = summary
            _logger.info(
                "[EXEC] stage=paper_tape trades=%d symbols_seen=%d reconnects=%d dropped=%d",
                summary.trades_received, summary.symbols_seen,
                summary.reconnects, summary.dropped_oldest,
            )
            return summary
        except Exception as exc:
            _logger.error("[EXEC] stage=paper_tape status=STOP_FAILED error=%s", exc)
            with self._lock:
                summary = TapeSummary(
                    trades_received=self._trades_received,
                    symbols_seen=len(self._last_seen_s),
                    reconnects=self._reconnects,
                    dropped_oldest=self._dropped_oldest,
                )
            self._summary = summary
            return summary

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run_session())
        except Exception as exc:
            _logger.warning("[EXEC] stage=paper_tape status=SESSION_FAILED error=%s", exc)

    async def _run_session(self) -> None:
        url = _tape_stream_url(self._stream_url, self._symbols)
        backoff = 1.0
        connected_once = False
        while not self._stop_requested:
            try:
                async with self._connect(url) as conn:
                    if connected_once:
                        with self._lock:
                            self._reconnects += 1
                    connected_once = True
                    self._connected = True
                    backoff = 1.0
                    while not self._stop_requested:
                        try:
                            message = await asyncio.wait_for(conn.recv(), timeout=1.0)
                        except TimeoutError:
                            message = None
                        if message is None:
                            continue
                        if isinstance(message, (bytes, bytearray)):
                            try:
                                message = json.loads(message.decode("utf-8"))
                            except Exception:
                                with self._lock:
                                    self._malformed += 1
                                continue
                        if isinstance(message, str):
                            try:
                                message = json.loads(message)
                            except Exception:
                                with self._lock:
                                    self._malformed += 1
                                continue
                        parsed = _parse_agg_trade(message)
                        if parsed is None:
                            with self._lock:
                                self._malformed += 1
                            continue
                        sym, transact_ms, price, buyer_maker = parsed
                        sym = sym.upper()
                        if sym not in self._symbol_set:
                            continue
                        self._observe(sym, transact_ms, price, buyer_maker)
            except Exception as exc:
                self._connected = False
                _logger.warning("[EXEC] stage=paper_tape status=DISCONNECTED error=%s", exc)
                wait = min(backoff, _RECONNECT_BACKOFF_MAX_S)
                elapsed = 0.0
                while elapsed < wait and not self._stop_requested:
                    await asyncio.sleep(0.1)
                    elapsed += 0.1
                backoff = min(_RECONNECT_BACKOFF_MAX_S, backoff * 2.0)
            finally:
                with contextlib.suppress(Exception):
                    if self._stop_requested:
                        self._connected = False
        self._connected = False
