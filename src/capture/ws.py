"""All-market forceOrder stream capture as raw text frames with connection markers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .config import CaptureConfig
from .journal import RECORD_VERSION, SegmentWriter

logger = logging.getLogger(__name__)


@dataclass
class WsStatus:
    """Connection and coverage counters of the forceOrder stream."""

    connected_at_ns: int | None = None
    first_frame_at_ns: int | None = None
    last_frame_at_ns: int | None = None
    reconnects: int = 0
    pending_dropped: int = 0


class WsConnection(Protocol):
    """Minimal stream interface consumed by ``ForceOrderCapture``.

    ``read`` returns one of ``("text", str)``, ``("binary", bytes)``,
    ``("control", None)`` (ping/pong or other non-data frames),
    ``("close", reason)`` or ``("timeout", None)`` when nothing arrived within the wait. ``ping`` sends a client ping; ``close`` releases the socket.
    """

    async def read(self, wait_s: float) -> tuple[str, Any]: ...
    async def ping(self) -> None: ...
    async def close(self) -> None: ...


class ForceOrderCapture:
    """Journals the all-market forceOrder stream as raw text frames with connection markers.

    Each connection writes ``ws_open``; every text frame writes a ``frame`` record with the
    exact received text; the connection ends with ``ws_close`` carrying the reason. The
    normalizer attests coverage only from frames inside an open/close pair, so a connection that
    only exchanges control frames (a wrong endpoint) attests nothing.
    """

    def __init__(
        self,
        *,
        config: CaptureConfig,
        writer: SegmentWriter,
        status: WsStatus,
        connect: Callable[[str], Awaitable[WsConnection]],
        clock_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        shutdown: Callable[[], bool],
    ) -> None:
        """Bind the stream capture to its writer, status and connector."""
        self._config = config
        self._writer = writer
        self._status = status
        self._connect = connect
        self._clock_ns = clock_ns
        self._sleep = sleep
        self._shutdown = shutdown

    def _write_open(self, url: str) -> None:
        self._writer.add(
            {
                "v": RECORD_VERSION,
                "stream": self._writer.stream,
                "slot": self._writer.slot,
                "kind": "ws_open",
                "recv_ns": self._clock_ns(),
                "url": url,
            }
        )

    def _write_close(self, reason: str) -> None:
        self._writer.add(
            {
                "v": RECORD_VERSION,
                "stream": self._writer.stream,
                "slot": self._writer.slot,
                "kind": "ws_close",
                "recv_ns": self._clock_ns(),
                "reason": reason,
            }
        )
        self._flush_bounded()

    def _flush_bounded(self) -> None:
        try:
            self._writer.flush()
        except OSError as exc:
            overflow = self._writer.pending() - self._config.ws_max_pending_frames
            if overflow > 0:
                dropped = self._writer.discard_oldest(overflow)
                self._status.pending_dropped += dropped
                logger.error(
                    "[DATA] stage=capture stream=%s status=BUFFER_OVERFLOW dropped=%d",
                    self._writer.stream,
                    dropped,
                )
            else:
                logger.warning(
                    "[DATA] stage=capture stream=%s status=FLUSH_FAILED error=%s",
                    self._writer.stream,
                    f"{type(exc).__name__}: {exc}",
                )

    async def _serve(self, conn: WsConnection, url: str) -> str | None:
        """Serve one connection; return the reconnect reason, or None on shutdown."""
        config = self._config
        open_ns = self._clock_ns()
        self._status.connected_at_ns = open_ns
        self._write_open(url)
        last_any_ns = open_ns
        frame_ref_ns = open_ns
        last_ping_ns = open_ns
        last_flush_ns = open_ns
        while True:
            now_ns = self._clock_ns()
            if (now_ns - last_ping_ns) / 1_000_000_000 >= config.ws_ping_interval_s:
                try:
                    await conn.ping()
                except Exception as exc:  # noqa: BLE001 - ping failure ends this connection
                    return f"error:{type(exc).__name__}"
                last_ping_ns = now_ns
            try:
                kind, payload = await conn.read(config.ws_receive_timeout_s)
            except Exception as exc:  # noqa: BLE001 - read failure ends this connection
                return f"error:{type(exc).__name__}"
            now_ns = self._clock_ns()
            if kind == "text":
                text = str(payload)
                last_any_ns = now_ns
                frame_ref_ns = now_ns
                self._status.last_frame_at_ns = now_ns
                if self._status.first_frame_at_ns is None:
                    self._status.first_frame_at_ns = now_ns
                self._writer.add(
                    {
                        "v": RECORD_VERSION,
                        "stream": self._writer.stream,
                        "slot": self._writer.slot,
                        "kind": "frame",
                        "recv_ns": now_ns,
                        "frame": text,
                    }
                )
                if (
                    self._writer.pending() >= config.ws_flush_max_frames
                    or (now_ns - last_flush_ns) / 1_000_000_000 >= config.ws_flush_interval_s
                ):
                    self._flush_bounded()
                    last_flush_ns = now_ns
            elif kind in ("binary", "control"):
                last_any_ns = now_ns
            elif kind == "close":
                return str(payload) if payload else "server_close"
            if (now_ns - last_any_ns) / 1_000_000_000 >= config.ws_liveness_timeout_s:
                return "liveness_timeout"
            if (now_ns - frame_ref_ns) / 1_000_000_000 >= config.ws_event_stall_timeout_s:
                return "event_stall"
            if self._shutdown():
                return None

    async def run(self) -> None:
        """Hold the forceOrder stream across reconnects until shutdown."""
        backoff_s = 1.0
        connected_once = False
        url = self._config.force_order_url
        while not self._shutdown():
            try:
                conn = await self._connect(url)
            except Exception as exc:  # noqa: BLE001 - connect failure retries with backoff
                logger.warning(
                    "[DATA] stage=capture stream=%s status=CONNECT_FAILED error=%s",
                    self._writer.stream,
                    type(exc).__name__,
                )
                await self._sleep(backoff_s)
                backoff_s = min(self._config.restart_backoff_max_s, backoff_s * 2.0)
                continue
            if connected_once:
                self._status.reconnects += 1
            connected_once = True
            backoff_s = 1.0
            reason = await self._serve(conn, url)
            with contextlib.suppress(Exception):  # noqa: BLE001 - close is best effort
                await conn.close()
            if reason is None:
                self._write_close("shutdown")
                return
            self._write_close(reason)
            logger.warning(
                "[DATA] stage=capture stream=%s status=RECONNECT reason=%s",
                self._writer.stream,
                reason,
            )
            if self._shutdown():
                return
            await self._sleep(backoff_s)
            backoff_s = min(self._config.restart_backoff_max_s, backoff_s * 2.0)
