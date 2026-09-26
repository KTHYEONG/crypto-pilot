"""Raw-first capture process entry point: one slot, one asyncio loop, supervised tasks."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import aiohttp

from .config import CaptureConfig
from .deadman import DeadmanPinger
from .heartbeat import build_capture_heartbeat, write_capture_heartbeat
from .journal import SegmentWriter
from .logsetup import configure_capture_logging
from .rest import Fetch, GridSampler, RateGate, ReferenceCapture, RestStatus
from .ws import ForceOrderCapture, WsConnection, WsStatus

logger = logging.getLogger(__name__)

CAPTURE_APP_ROOT = Path(__file__).resolve().parents[2]
DEADMAN_ENV_VAR = "LIVE_RECORDER_DEADMAN_PING_URL"
_FINGERPRINT_PATH = Path("/app/.capture_fingerprint")


class _FlushState:
    """Consecutive flush-failure count plus the last successful flush time, shared by writers."""

    __slots__ = ("failures", "last_flush_at_ns")

    def __init__(self) -> None:
        """Start with no failures and no flush yet."""
        self.failures = 0
        self.last_flush_at_ns: int | None = None


class _TrackedWriter(SegmentWriter):
    """Segment writer that mirrors flush outcomes into a shared ``_FlushState``."""

    def __init__(
        self,
        capture_root: Path,
        stream: str,
        slot: str,
        *,
        state: _FlushState,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        """Bind the writer to its journal tree and shared flush state."""
        super().__init__(capture_root, stream, slot)
        self._state = state
        self._clock_ns = clock_ns

    def flush(self) -> int:
        """Flush and record the outcome; failures increment, any success resets."""
        try:
            count = super().flush()
        except OSError:
            self._state.failures += 1
            raise
        self._state.failures = 0
        self._state.last_flush_at_ns = self._clock_ns()
        return count


class _AiohttpWsAdapter:
    """Adapt an aiohttp websocket to the minimal ``WsConnection`` protocol."""

    def __init__(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Wrap a connected aiohttp websocket."""
        self._ws = ws

    async def read(self, wait_s: float) -> tuple[str, Any]:
        """Return one frame as a ``(kind, payload)`` pair, or ``("timeout", None)``."""
        try:
            message = await self._ws.receive(timeout=wait_s)
        except TimeoutError:
            return ("timeout", None)
        if message.type == aiohttp.WSMsgType.TEXT:
            return ("text", message.data)
        if message.type == aiohttp.WSMsgType.BINARY:
            return ("binary", message.data)
        if message.type == aiohttp.WSMsgType.PING:
            with contextlib.suppress(Exception):
                await self._ws.pong(message.data)
            return ("control", None)
        if message.type == aiohttp.WSMsgType.PONG:
            return ("control", None)
        if message.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.ERROR,
        ):
            return ("close", "server_close")
        return ("control", None)

    async def ping(self) -> None:
        """Send a client ping."""
        await self._ws.ping()

    async def close(self) -> None:
        """Release the socket."""
        await self._ws.close()


def _make_fetch(session: aiohttp.ClientSession) -> Fetch:
    """Return a fetcher yielding ``(status, body, headers)`` over the shared session."""

    async def fetch(url: str) -> tuple[int, str, Mapping[str, str]]:
        async with session.get(url) as response:
            body = await response.text()
            return response.status, body, dict(response.headers)

    return fetch


def _make_connect(session: aiohttp.ClientSession) -> Callable[[str], Awaitable[WsConnection]]:
    """Return a connector opening the forceOrder stream with recorder-parity options."""

    async def connect(url: str) -> WsConnection:
        ws = await session.ws_connect(url, autoping=False, heartbeat=None)
        return _AiohttpWsAdapter(ws)

    return connect


def _read_fingerprint() -> str | None:
    """Return the image fingerprint written by the Dockerfile, or None when absent."""
    try:
        value = _FINGERPRINT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


async def _supervised(
    name: str,
    run: Callable[[], Awaitable[None]],
    *,
    shutdown: Callable[[], bool],
    sleep: Callable[[float], Awaitable[None]],
    backoff_max_s: float,
) -> None:
    """Run ``run`` until shutdown, restarting it with capped exponential backoff on failure."""
    backoff_s = 1.0
    while not shutdown():
        try:
            await run()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one failing component never stops the others
            logger.exception("[SYS] stage=capture task=%s status=RESTART", name)
        else:
            if shutdown():
                return
            logger.error("[SYS] stage=capture task=%s status=UNEXPECTED_EXIT", name)
        if shutdown():
            return
        await sleep(min(backoff_s, backoff_max_s))
        backoff_s = min(backoff_max_s, backoff_s * 2.0)


def _shutdown_aware_sleep(
    base_sleep: Callable[[float], Awaitable[None]],
    shutdown: Callable[[], bool],
) -> Callable[[float], Awaitable[None]]:
    """Wrap ``base_sleep`` so long waits observe shutdown within about half a second."""

    async def sleep(delay: float) -> None:
        remaining = delay
        while remaining > 0:
            if shutdown():
                return
            step = min(0.5, remaining)
            await base_sleep(step)
            remaining -= step

    return sleep


def _deadman_failing(
    *,
    config: CaptureConfig,
    rest: Mapping[str, RestStatus],
    ws_status: WsStatus,
    state: _FlushState,
    now_ns: int,
) -> bool:
    """Return True when a REST stream fails, a flush fails, records were dropped, or WS is silent."""
    if state.failures > 0 or ws_status.pending_dropped > 0:
        return True
    for entry in rest.values():
        # 버퍼 초과로 버린 원본은 복구할 수 없으므로 재시작 전까지 /fail을 유지한다.
        if entry.dropped_records > 0 or entry.consecutive_failures >= config.deadman_fail_consecutive_failures:
            return True
    return (
        ws_status.connected_at_ns is not None
        and ws_status.last_frame_at_ns is not None
        and (now_ns - ws_status.last_frame_at_ns) / 1_000_000_000 > config.ws_event_stall_timeout_s
    )


async def _heartbeat_loop(
    *,
    slot: str,
    capture_root: Path,
    config: CaptureConfig,
    rest: Mapping[str, RestStatus],
    ws_status: WsStatus,
    state: _FlushState,
    started_at_ns: int,
    fingerprint: str | None,
    pinger: DeadmanPinger,
    clock_ns: Callable[[], int],
    sleep: Callable[[float], Awaitable[None]],
    shutdown: Callable[[], bool],
) -> None:
    """Rewrite the capture heartbeat every ``heartbeat_interval_s`` until shutdown."""
    pid = os.getpid()
    while not shutdown():
        now_ns = clock_ns()
        failing = _deadman_failing(
            config=config, rest=rest, ws_status=ws_status, state=state, now_ns=now_ns
        )
        with contextlib.suppress(Exception):
            await asyncio.to_thread(pinger.maybe_ping, now_ns=now_ns, failing=failing)
        payload = build_capture_heartbeat(
            slot=slot,
            pid=pid,
            fingerprint=fingerprint,
            started_at_ns=started_at_ns,
            stopped_at_ns=None,
            rest=rest,
            ws=ws_status,
            last_flush_at_ns=state.last_flush_at_ns,
            flush_failures=state.failures,
            now_ns=now_ns,
        )
        try:
            write_capture_heartbeat(capture_root, slot, payload)
        except OSError as exc:
            logger.warning("[SYS] stage=capture component=heartbeat status=WRITE_FAILED error=%s", type(exc).__name__)
        await sleep(config.heartbeat_interval_s)


async def _amain(
    *,
    slot: str,
    capture_root: Path,
    config: CaptureConfig,
    fetch: Fetch,
    connect: Callable[[str], Awaitable[WsConnection]],
    clock_ns: Callable[[], int] = time.time_ns,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    shutdown: Callable[[], bool],
    deadman_url: str | None = None,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    """Run every capture task until ``shutdown`` is set, then flush and write ``stopped_at``."""
    gate = RateGate(config.rate_limit_cooldown_s)
    state = _FlushState()
    writers = {
        stream: _TrackedWriter(capture_root, stream, slot, state=state, clock_ns=clock_ns)
        for stream in ("book_ticker", "premium_index", "force_order")
    }
    rest: dict[str, RestStatus] = {"book_ticker": RestStatus(), "premium_index": RestStatus()}
    ws_status = WsStatus()
    started_at_ns = clock_ns()
    task_sleep = _shutdown_aware_sleep(sleep, shutdown)
    samplers = {
        "book_ticker": GridSampler(
            stream="book_ticker",
            url=config.book_ticker_url,
            interval_s=config.book_ticker_interval_s,
            config=config,
            writer=writers["book_ticker"],
            fetch=fetch,
            gate=gate,
            status=rest["book_ticker"],
            clock_ns=clock_ns,
            sleep=task_sleep,
            shutdown=shutdown,
        ),
        "premium_index": GridSampler(
            stream="premium_index",
            url=config.premium_index_url,
            interval_s=config.premium_index_interval_s,
            config=config,
            writer=writers["premium_index"],
            fetch=fetch,
            gate=gate,
            status=rest["premium_index"],
            clock_ns=clock_ns,
            sleep=task_sleep,
            shutdown=shutdown,
        ),
    }
    reference = ReferenceCapture(
        config=config,
        capture_root=capture_root,
        fetch=fetch,
        gate=gate,
        clock_ns=clock_ns,
        sleep=task_sleep,
        shutdown=shutdown,
    )
    force_order = ForceOrderCapture(
        config=config,
        writer=writers["force_order"],
        status=ws_status,
        connect=connect,
        clock_ns=clock_ns,
        sleep=task_sleep,
        shutdown=shutdown,
    )
    pinger = DeadmanPinger(
        deadman_url,
        interval_s=config.deadman_ping_interval_s,
        timeout_s=config.deadman_ping_timeout_s,
    )
    tasks = [
        asyncio.ensure_future(
            _supervised(
                name,
                runner,
                shutdown=shutdown,
                sleep=task_sleep,
                backoff_max_s=config.restart_backoff_max_s,
            )
        )
        for name, runner in (
            ("book_ticker", samplers["book_ticker"].run),
            ("premium_index", samplers["premium_index"].run),
            ("reference", reference.run),
            ("force_order", force_order.run),
            (
                "heartbeat",
                lambda: _heartbeat_loop(
                    slot=slot,
                    capture_root=capture_root,
                    config=config,
                    rest=rest,
                    ws_status=ws_status,
                    state=state,
                    started_at_ns=started_at_ns,
                    fingerprint=fingerprint,
                    pinger=pinger,
                    clock_ns=clock_ns,
                    sleep=task_sleep,
                    shutdown=shutdown,
                ),
            ),
        )
    ]
    while not shutdown():
        await sleep(0.5)
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=config.stop_flush_budget_s)
    except TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    for writer in writers.values():
        with contextlib.suppress(OSError):
            writer.flush()
    stopped_at_ns = clock_ns()
    payload = build_capture_heartbeat(
        slot=slot,
        pid=os.getpid(),
        fingerprint=fingerprint,
        started_at_ns=started_at_ns,
        stopped_at_ns=stopped_at_ns,
        rest=rest,
        ws=ws_status,
        last_flush_at_ns=state.last_flush_at_ns,
        flush_failures=state.failures,
        now_ns=stopped_at_ns,
    )
    with contextlib.suppress(OSError):
        write_capture_heartbeat(capture_root, slot, payload)
    logger.info("[SYS] stage=capture slot=%s status=STOPPED", slot)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the capture process for one slot until SIGTERM/SIGINT, then flush and exit 0.

    Starts the bookTicker and premiumIndex samplers, reference capture, the forceOrder
    capture, the heartbeat loop (every ``heartbeat_interval_s``) and the dead-man pinger in
    one asyncio loop with one shared ``aiohttp.ClientSession`` and ``RateGate``. Each task runs
    under a supervisor that restarts it with capped exponential backoff after an unexpected
    exception, so one failing component never stops the others.

    Args:
        argv: ``--slot {blue,green}`` (required) and optional ``--capture-root`` (tests).

    Returns:
        0 after a graceful shutdown; 2 on invalid arguments.
    """
    parser = argparse.ArgumentParser(description="Raw-first live capture process.")
    parser.add_argument("--slot", required=True, choices=("blue", "green"))
    parser.add_argument("--capture-root", type=Path, default=None)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    slot: str = args.slot
    capture_root: Path = args.capture_root if args.capture_root is not None else CAPTURE_APP_ROOT / "data" / "live_capture"
    config = CaptureConfig()
    configure_capture_logging(CAPTURE_APP_ROOT / "logs" / "capture", slot)
    logger.info("[SYS] stage=capture slot=%s status=STARTING", slot)
    started_at_ns = time.time_ns()
    with contextlib.suppress(OSError):
        write_capture_heartbeat(
            capture_root,
            slot,
            build_capture_heartbeat(
                slot=slot,
                pid=os.getpid(),
                fingerprint=_read_fingerprint(),
                started_at_ns=started_at_ns,
                stopped_at_ns=None,
                rest={"book_ticker": RestStatus(), "premium_index": RestStatus()},
                ws=WsStatus(),
                last_flush_at_ns=None,
                flush_failures=0,
                now_ns=started_at_ns,
            ),
        )

    async def _runner() -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, shutdown_event.set)
            except (NotImplementedError, RuntimeError, ValueError):  # pragma: no cover - non-Unix loop
                continue
        timeout = aiohttp.ClientTimeout(total=config.http_timeout_s)
        session = aiohttp.ClientSession(timeout=timeout)
        try:
            return await _amain(
                slot=slot,
                capture_root=capture_root,
                config=config,
                fetch=_make_fetch(session),
                connect=_make_connect(session),
                shutdown=shutdown_event.is_set,
                deadman_url=os.environ.get(DEADMAN_ENV_VAR) or None,
                fingerprint=_read_fingerprint(),
            )
        finally:
            await session.close()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_runner())
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
