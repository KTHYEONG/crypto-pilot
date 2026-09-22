"""Binance partial-depth WebSocket capture over the execution window (observability-only)."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.market_data.streams.coverage import CoverageTracker

_logger = logging.getLogger(__name__)

EXEC_DEPTH_DATASET: str = "exec_depth"

_DEPTH_LEVELS: frozenset[int] = frozenset({5, 10, 20})
_DEPTH_UPDATE_MS: frozenset[int] = frozenset({100, 250, 500})
_RECONNECT_BACKOFF_MAX_S: float = 30.0
_STOP_JOIN_TIMEOUT_S: float = 10.0


@dataclass(frozen=True, slots=True)
class DepthCaptureSummary:
    """Outcome of one execution-window capture session (for audit/alerting)."""

    rows: int
    symbols_requested: int
    symbols_seen: int
    reconnects: int
    parts: int


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _depth_stream_url(stream_url: str, symbols: Sequence[str], levels: int, update_ms: int) -> str:
    streams = "/".join(f"{str(symbol).lower()}@depth{levels}@{update_ms}ms" for symbol in symbols)
    return f"{stream_url}?streams={streams}"


def _to_opt_int(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_depth_message(
    message: Any,
    *,
    decision_time: pd.Timestamp,
    run_id: str,
    mode: str,
    received_at: pd.Timestamp,
    levels: int,
) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return None
    data = message.get("data")
    if not isinstance(data, dict):
        data = message
    symbol = data.get("s")
    bids = data.get("bids", data.get("b"))
    asks = data.get("asks", data.get("a"))
    if symbol is None or not isinstance(bids, list) or not isinstance(asks, list):
        return None
    row: dict[str, Any] = {
        "decision_time": decision_time,
        "run_id": str(run_id),
        "mode": str(mode),
        "received_at": received_at,
        "event_time_ms": _to_opt_int(data.get("E")),
        "transact_time_ms": _to_opt_int(data.get("T")),
        "symbol": str(symbol),
        "update_id": _to_opt_int(data.get("u", data.get("lastUpdateId"))),
    }
    for index in range(levels):
        for side, key in (("bid", bids), ("ask", asks)):
            level = key[index] if index < len(key) else None
            price: float | None = None
            qty: float | None = None
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                try:
                    price = float(level[0])
                except (TypeError, ValueError):
                    price = None
                try:
                    qty = float(level[1])
                except (TypeError, ValueError):
                    qty = None
            row[f"{side}_px_{index}"] = price
            row[f"{side}_qty_{index}"] = qty
    return row


def _depth_columns(levels: int) -> list[str]:
    columns = [
        "decision_time", "run_id", "mode", "received_at",
        "event_time_ms", "transact_time_ms", "symbol", "update_id",
    ]
    for index in range(levels):
        columns.extend((f"bid_px_{index}", f"bid_qty_{index}", f"ask_px_{index}", f"ask_qty_{index}"))
    return columns


def _rows_to_frame(rows: Sequence[dict[str, Any]], levels: int) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=_depth_columns(levels))
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True)
    frame["received_at"] = pd.to_datetime(frame["received_at"], utc=True)
    for column in ("run_id", "mode", "symbol"):
        frame[column] = frame[column].astype("string")
    for column in ("event_time_ms", "transact_time_ms", "update_id"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for index in range(levels):
        frame[f"bid_px_{index}"] = pd.to_numeric(frame[f"bid_px_{index}"], errors="coerce").astype("float64")
        frame[f"ask_px_{index}"] = pd.to_numeric(frame[f"ask_px_{index}"], errors="coerce").astype("float64")
        frame[f"bid_qty_{index}"] = pd.to_numeric(frame[f"bid_qty_{index}"], errors="coerce").astype("float32")
        frame[f"ask_qty_{index}"] = pd.to_numeric(frame[f"ask_qty_{index}"], errors="coerce").astype("float32")
    return frame


def _as_utc(stamp: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(stamp)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class _AiohttpDepthConnection:
    def __init__(self, ws: Any) -> None:
        self._ws = ws

    async def recv(self) -> Any:
        return await self._ws.receive_json()


@asynccontextmanager
async def _default_connect(url: str) -> AsyncIterator[Any]:
    import aiohttp

    session = aiohttp.ClientSession()
    try:
        async with session.ws_connect(url, heartbeat=30.0) as ws:
            yield _AiohttpDepthConnection(ws)
    finally:
        await session.close()


class ExecutionDepthRecorder:
    """Records Binance partial-depth (top ``levels``) WebSocket updates for the symbols being traded.

    Runs its own asyncio loop in a daemon thread from just before order submission until a post-trade
    window has elapsed, so queue position and touch dynamics during the resting window can be replayed
    against later-downloaded trades. Capture is observability-only: no exception ever reaches the
    trading cycle and trading never waits on it.

    Output: immutable part files ``<root>/exec_depth/<decision YYYYMMDD>/part_<start HHMMSS>_<seq>.parquet``
    with columns ``decision_time``, ``run_id``, ``mode``, ``received_at``, ``event_time_ms``,
    ``transact_time_ms``, ``symbol``, ``update_id`` and ``bid_px_i``/``bid_qty_i``/``ask_px_i``/
    ``ask_qty_i`` for ``i in range(levels)`` (prices float64, quantities float32).
    """

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        decision_time: pd.Timestamp,
        run_id: str,
        mode: str,
        root: Path,
        stream_url: str,
        levels: int,
        update_ms: int,
        flush_interval_s: float,
        max_session_s: float,
        connect: Callable[[str], Any] | None = None,
        now_fn: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        if levels not in _DEPTH_LEVELS:
            raise ValueError(f"levels must be one of {sorted(_DEPTH_LEVELS)}, got {levels!r}")
        if update_ms not in _DEPTH_UPDATE_MS:
            raise ValueError(f"update_ms must be one of {sorted(_DEPTH_UPDATE_MS)}, got {update_ms!r}")
        self._symbols = [str(symbol) for symbol in symbols]
        stamp = pd.Timestamp(decision_time)
        self._decision_time = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        self._run_id = str(run_id)
        self._mode = str(mode)
        self._root = Path(root)
        self._stream_url = str(stream_url)
        self._levels = int(levels)
        self._update_ms = int(update_ms)
        self._flush_interval_s = float(flush_interval_s)
        self._max_session_s = float(max_session_s)
        self._connect = connect if connect is not None else _default_connect
        self._now = now_fn if now_fn is not None else _utc_now
        self._empty = len(self._symbols) == 0
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._started = False
        self._start_failed = False
        self._summary: DepthCaptureSummary | None = None
        self._rows = 0
        self._symbols_seen: set[str] = set()
        self._reconnects = 0
        self._parts = 0

    def start(self) -> None:
        """Start the capture thread; returns immediately (never blocks order submission)."""
        if self._empty or self._started:
            return
        self._started = True
        try:
            thread = threading.Thread(target=self._thread_main, name="exec-depth-capture", daemon=True)
            thread.start()
            self._thread = thread
        except Exception as exc:
            _logger.warning("[EXEC] stage=exec_depth_capture status=START_FAILED error=%s", exc)
            self._start_failed = True

    def stop(self, *, post_window_s: float, shutdown: Any | None = None) -> DepthCaptureSummary:
        """Keep capturing for ``post_window_s`` seconds (0 = stop now), then stop, flush and join.

        Returns early when ``shutdown.requested`` becomes true. Idempotent; safe if ``start`` failed.
        """
        if self._empty or self._start_failed or not self._started:
            return DepthCaptureSummary(rows=0, symbols_requested=len(self._symbols), symbols_seen=0, reconnects=0, parts=0)
        if self._summary is not None:
            return self._summary
        try:
            remaining = max(0.0, float(post_window_s))
            while remaining > 0:
                if shutdown is not None and bool(getattr(shutdown, "requested", False)):
                    break
                step = min(0.25, remaining)
                time.sleep(step)
                remaining -= step
            self._stop_requested = True
            thread = self._thread
            if thread is not None:
                thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
                if thread.is_alive():
                    _logger.error("[EXEC] stage=exec_depth_capture status=THREAD_STUCK")
            summary = DepthCaptureSummary(
                rows=self._rows,
                symbols_requested=len(self._symbols),
                symbols_seen=len(self._symbols_seen),
                reconnects=self._reconnects,
                parts=self._parts,
            )
            self._summary = summary
            _logger.info(
                "[EXEC] stage=exec_depth_capture decision_time=%s rows=%d symbols_seen=%d/%d reconnects=%d parts=%d",
                self._decision_time.isoformat(), summary.rows, summary.symbols_seen,
                summary.symbols_requested, summary.reconnects, summary.parts,
            )
            return summary
        except Exception as exc:
            _logger.error("[EXEC] stage=exec_depth_capture status=STOP_FAILED error=%s", exc)
            return DepthCaptureSummary(rows=self._rows, symbols_requested=len(self._symbols), symbols_seen=len(self._symbols_seen), reconnects=self._reconnects, parts=self._parts)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run_session())
        except Exception as exc:
            _logger.warning("[EXEC] stage=exec_depth_capture status=SESSION_FAILED error=%s", exc)

    def _part_path(self, first_received: pd.Timestamp, seq: int) -> Path:
        day: str = self._decision_time.strftime("%Y%m%d")
        stamp: str = pd.Timestamp(first_received).tz_convert("UTC").strftime("%H%M%S")
        directory = self._root / EXEC_DEPTH_DATASET / day
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"part_{stamp}_{seq}.parquet"
        while target.exists():
            seq += 1
            target = directory / f"part_{stamp}_{seq}.parquet"
        return target

    def _flush_rows(self, rows: list[dict[str, Any]], coverage: CoverageTracker) -> None:
        if not rows:
            return
        frame = _rows_to_frame(rows, self._levels)
        first = pd.Timestamp(frame["received_at"].iloc[0]).tz_convert("UTC")
        target = self._part_path(first, self._parts)
        tmp = target.with_name(f".{target.name}.tmp")
        frame.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, target)
        self._parts += 1
        try:
            coverage.flush()
        except Exception as exc:
            _logger.warning("[EXEC] stage=exec_depth_capture status=COVERAGE_FLUSH_FAILED error=%s", exc)

    async def _run_session(self) -> None:
        url = _depth_stream_url(self._stream_url, self._symbols, self._levels, self._update_ms)
        coverage = CoverageTracker("exec_depth", self._root)
        buffer: list[dict[str, Any]] = []
        backoff = 1.0
        connected_once = False
        started_at = time.monotonic()
        last_flush = time.monotonic()
        while not self._stop_requested and time.monotonic() - started_at < self._max_session_s:
            try:
                async with self._connect(url) as conn:
                    if connected_once:
                        self._reconnects += 1
                    connected_once = True
                    backoff = 1.0
                    while not self._stop_requested and time.monotonic() - started_at < self._max_session_s:
                        try:
                            message = await asyncio.wait_for(conn.recv(), timeout=1.0)
                        except TimeoutError:
                            message = None
                        if message is not None:
                            row = _parse_depth_message(
                                message, decision_time=self._decision_time, run_id=self._run_id,
                                mode=self._mode, received_at=self._now(), levels=self._levels,
                            )
                            if row is None:
                                continue
                            buffer.append(row)
                            self._rows += 1
                            self._symbols_seen.add(str(row["symbol"]))
                            coverage.mark_ok(_as_utc(pd.Timestamp(row["received_at"])))
                        if buffer and time.monotonic() - last_flush >= self._flush_interval_s:
                            self._flush_rows(buffer, coverage)
                            buffer = []
                            last_flush = time.monotonic()
            except Exception as exc:
                _logger.warning("[EXEC] stage=exec_depth_capture status=DISCONNECTED error=%s", exc)
                coverage.mark_error(_as_utc(self._now()))
                wait = min(backoff, _RECONNECT_BACKOFF_MAX_S)
                elapsed = 0.0
                while elapsed < wait and not self._stop_requested and time.monotonic() - started_at < self._max_session_s:
                    await asyncio.sleep(0.1)
                    elapsed += 0.1
                backoff = min(_RECONNECT_BACKOFF_MAX_S, backoff * 2.0)
        self._flush_rows(buffer, coverage)
