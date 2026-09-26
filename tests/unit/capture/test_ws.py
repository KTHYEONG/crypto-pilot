"""Invariant guards for the forceOrder stream capture."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.capture.config import CaptureConfig
from src.capture.journal import SegmentWriter, hot_segment_path, iter_complete_records
from src.capture.ws import ForceOrderCapture, WsStatus


def _ns(year: int, month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> int:
    moment = datetime(year, month, day, hour, minute, second, tzinfo=UTC)
    return int(moment.timestamp() * 1_000_000_000)


class FakeClock:
    """Manually advanced nanosecond clock with an event-loop-friendly sleep."""

    def __init__(self, start_ns: int) -> None:
        """Start the clock at ``start_ns``."""
        self.now_ns = start_ns

    def __call__(self) -> int:
        """Return the current fake time."""
        return self.now_ns

    async def sleep(self, delay: float) -> None:
        """Advance the clock by ``delay`` seconds and yield to the loop."""
        self.now_ns += int(delay * 1_000_000_000)
        await asyncio.sleep(0)


class FakeConn:
    """Scripted connection: read advances the clock by the wait, then replays the script."""

    def __init__(self, clock: FakeClock, script: list[tuple[str, Any]], on_read: Any = None) -> None:
        """Queue ``script`` frames for replay."""
        self._clock = clock
        self._script = deque(script)
        self._on_read = on_read
        self.pings = 0
        self.closed = False

    async def read(self, wait_s: float) -> tuple[str, Any]:
        """Advance the clock by the wait and return the next scripted message."""
        self._clock.now_ns += int(wait_s * 1_000_000_000)
        await asyncio.sleep(0)
        if self._on_read is not None:
            self._on_read()
        if self._script:
            return self._script.popleft()
        return ("timeout", None)

    async def ping(self) -> None:
        """Count the client ping."""
        self.pings += 1

    async def close(self) -> None:
        """Mark the socket closed."""
        self.closed = True


def _capture(
    tmp_path: Path,
    clock: FakeClock,
    conn: FakeConn,
    stop: Any,
    connects: list[str] | None = None,
    **overrides: Any,
) -> tuple[ForceOrderCapture, WsStatus]:
    values: dict[str, Any] = {"force_order_url": "wss://example.invalid/stream"}
    values.update(overrides)
    config = CaptureConfig(**values)
    status = WsStatus()

    async def connect(url: str) -> FakeConn:
        if connects is not None:
            connects.append(url)
        return conn

    capture = ForceOrderCapture(
        config=config,
        writer=SegmentWriter(tmp_path, "force_order", "blue"),
        status=status,
        connect=connect,  # type: ignore[arg-type]
        clock_ns=clock,
        sleep=clock.sleep,
        shutdown=stop,
    )
    return capture, status


def _force_order_records(tmp_path: Path, recv_ns: int) -> list[dict[str, Any]]:
    dest = hot_segment_path(tmp_path, "force_order", "blue", recv_ns)
    return [record for record, _ in iter_complete_records(dest, 0)]


def test_frames_journaled_verbatim_with_markers(tmp_path: Path) -> None:
    """Spec 01: ws_open, exact frame texts, then ws_close with the server reason."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 0))
    frames = ['{"e":"forceOrder","o":{"s":"BTCUSDT"}}'] * 3
    script: list[tuple[str, Any]] = [("text", item) for item in frames] + [("close", "server_close")]
    stopped = False
    connects: list[str] = []
    conn = FakeConn(clock, script)
    capture, _ = _capture(tmp_path, clock, conn, lambda: stopped, connects)

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(capture.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            if len(connects) >= 2:
                stopped = True
                break
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(scenario())
    records = _force_order_records(tmp_path, _ns(2026, 9, 26, 10, 0, 0))
    kinds = [item["kind"] for item in records]
    assert kinds[:5] == ["ws_open", "frame", "frame", "frame", "ws_close"]
    assert [item["frame"] for item in records if item["kind"] == "frame"][:3] == frames
    assert records[4]["reason"] == "server_close"


def test_pong_only_connection_reconnects(tmp_path: Path) -> None:
    """Spec 01: control frames alone attest nothing and trigger an event-stall reconnect."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 0))
    stopped = False
    connects: list[str] = []
    conn = FakeConn(clock, [("control", None)] * 200)
    capture, status = _capture(
        tmp_path, clock, conn, lambda: stopped, connects, ws_event_stall_timeout_s=60.0
    )

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(capture.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            if status.reconnects >= 1:
                stopped = True
                break
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(scenario())
    records = _force_order_records(tmp_path, _ns(2026, 9, 26, 10, 2, 0))
    closes = [item for item in records if item["kind"] == "ws_close"]
    assert closes
    assert closes[0]["reason"] == "event_stall"
    assert all(item["kind"] != "frame" for item in records)
    assert status.reconnects == 1


def test_liveness_timeout_reconnects(tmp_path: Path) -> None:
    """Spec 01: total silence beyond 15 s ends the connection and reconnects once."""

    class SilentConn(FakeConn):
        async def read(self, wait_s: float) -> tuple[str, Any]:
            await asyncio.sleep(0)
            return ("timeout", None)

    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 0))
    stopped = False

    async def sleep_and_advance(delay: float) -> None:
        clock.now_ns += int(delay * 1_000_000_000)
        await asyncio.sleep(0)

    conn = SilentConn(clock, [])
    config = CaptureConfig(force_order_url="wss://example.invalid/stream")
    status = WsStatus()

    async def connect(url: str) -> SilentConn:
        return conn

    capture = ForceOrderCapture(
        config=config,
        writer=SegmentWriter(tmp_path, "force_order", "blue"),
        status=status,
        connect=connect,  # type: ignore[arg-type]
        clock_ns=clock,
        sleep=sleep_and_advance,
        shutdown=lambda: stopped,
    )

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(capture.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            clock.now_ns += 1_000_000_000
            if status.reconnects >= 1:
                stopped = True
                break
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(scenario())
    records = _force_order_records(tmp_path, _ns(2026, 9, 26, 10, 0, 30))
    closes = [item for item in records if item["kind"] == "ws_close"]
    assert closes
    assert closes[0]["reason"] == "liveness_timeout"
    assert status.reconnects == 1


def test_flush_on_size_and_interval(tmp_path: Path) -> None:
    """Spec 01: size triggers a flush at 500 frames; silence plus a frame triggers interval flush."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 0))
    script: list[tuple[str, Any]] = [("text", f"frame-{index}") for index in range(501)]
    script.append(("close", "server_close"))
    stopped = False
    connects: list[str] = []
    conn = FakeConn(clock, script)
    capture, _ = _capture(tmp_path, clock, conn, lambda: stopped, connects)

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(capture.run())
        for _ in range(40000):
            await asyncio.sleep(0)
            if len(connects) >= 2:
                stopped = True
                break
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(scenario())
    dest = hot_segment_path(tmp_path, "force_order", "blue", _ns(2026, 9, 26, 10, 0, 0))
    members = 0
    seen = 0
    offset = 0
    for _, end in iter_complete_records(dest, 0):
        seen += 1
        if end != offset:
            members += 1
            offset = end
    assert seen >= 503
    assert members >= 2


def test_interval_flush_without_size_pressure(tmp_path: Path) -> None:
    """Spec 01: one frame, five silent seconds, then a second frame flushes twice."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 0))
    script: list[tuple[str, Any]] = [("text", "first")]
    script += [("timeout", None)] * 6
    script += [("text", "second"), ("close", "server_close")]
    stopped = False
    connects: list[str] = []
    conn = FakeConn(clock, script)
    capture, _ = _capture(tmp_path, clock, conn, lambda: stopped, connects)

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(capture.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            if len(connects) >= 2:
                stopped = True
                break
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(scenario())
    dest = hot_segment_path(tmp_path, "force_order", "blue", _ns(2026, 9, 26, 10, 0, 0))
    offsets: list[int] = []
    for _, end in iter_complete_records(dest, 0):
        if not offsets or offsets[-1] != end:
            offsets.append(end)
    assert len(offsets) >= 2


def test_bounded_pending_on_flush_failure(tmp_path: Path, monkeypatch: Any) -> None:
    """Spec 01: failed flushes bound memory; overflow drops the oldest and is counted."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 0))
    script: list[tuple[str, Any]] = [("text", f"f-{index}") for index in range(15)]
    script.append(("close", "server_close"))
    stopped = False
    conn = FakeConn(clock, script)

    def failing_flush(self: SegmentWriter) -> int:
        raise OSError("disk gone")

    monkeypatch.setattr(SegmentWriter, "flush", failing_flush)
    capture, status = _capture(
        tmp_path,
        clock,
        conn,
        lambda: stopped,
        None,
        ws_flush_max_frames=5,
        ws_max_pending_frames=10,
    )

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(capture.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            if status.pending_dropped > 0 and not conn._script:
                stopped = True
                break
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(scenario())
    assert capture._writer.pending() <= 10
    assert status.pending_dropped > 0
    assert capture._writer.pending() + status.pending_dropped >= 17


def test_shutdown_writes_close_marker(tmp_path: Path) -> None:
    """Spec 01: shutdown mid-stream writes ws_close shutdown and flushes it."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 0))
    reads = 0
    nonlocal_stopped = [False]

    def on_read() -> None:
        nonlocal reads
        reads += 1
        if reads >= 3:
            nonlocal_stopped[0] = True

    conn = FakeConn(clock, [("text", "live")] * 100, on_read=on_read)
    capture, _ = _capture(tmp_path, clock, conn, lambda: nonlocal_stopped[0], None)

    async def scenario() -> None:
        await asyncio.wait_for(capture.run(), timeout=30)

    asyncio.run(scenario())
    records = _force_order_records(tmp_path, _ns(2026, 9, 26, 10, 0, 5))
    assert records[-1]["kind"] == "ws_close"
    assert records[-1]["reason"] == "shutdown"
    assert sum(1 for item in records if item["kind"] == "frame") >= 3


def test_first_frame_gate(tmp_path: Path) -> None:
    """Spec 01: first_frame_at stays None without frames, then pins the first frame time."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 0))
    stopped = False
    conn = FakeConn(clock, [])
    capture, status = _capture(tmp_path, clock, conn, lambda: stopped, None)

    async def idle() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(capture.run())
        for _ in range(10):
            await asyncio.sleep(0)
        stopped = True
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(idle())
    assert status.first_frame_at_ns is None

    clock2 = FakeClock(_ns(2026, 9, 26, 11, 0, 0))
    stopped2 = False
    conn2 = FakeConn(clock2, [("text", "hello")])
    capture2, status2 = _capture(tmp_path, clock2, conn2, lambda: stopped2, None)

    async def framed() -> None:
        nonlocal stopped2
        task = asyncio.ensure_future(capture2.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            if status2.first_frame_at_ns is not None:
                stopped2 = True
                break
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(framed())
    records = _force_order_records(tmp_path, _ns(2026, 9, 26, 11, 0, 0))
    frames = [item for item in records if item["kind"] == "frame"]
    assert status2.first_frame_at_ns == frames[0]["recv_ns"]


def test_ping_failure_ends_connection(tmp_path: Path) -> None:
    """A failing client ping closes the connection with an error reason."""

    attempts: list[int] = []

    class BadPingConn(FakeConn):
        async def ping(self) -> None:
            attempts.append(1)
            raise RuntimeError("ping gone")

    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 0))
    stopped = False
    conn = BadPingConn(clock, [])
    capture, _ = _capture(tmp_path, clock, conn, lambda: stopped, None)

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(capture.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            if attempts:
                stopped = True
                break
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(scenario())
    records = _force_order_records(tmp_path, _ns(2026, 9, 26, 10, 0, 10))
    closes = [item for item in records if item["kind"] == "ws_close"]
    assert closes
    assert closes[0]["reason"] == "error:RuntimeError"


def test_read_failure_ends_connection(tmp_path: Path) -> None:
    """A failing read closes the connection with an error reason."""

    class BadReadConn(FakeConn):
        async def read(self, wait_s: float) -> tuple[str, Any]:
            await asyncio.sleep(0)
            raise ConnectionError("read gone")

    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 0))
    stopped = False
    conn = BadReadConn(clock, [])
    capture, _ = _capture(tmp_path, clock, conn, lambda: stopped, None)

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(capture.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            if _force_order_records(tmp_path, _ns(2026, 9, 26, 10, 0, 0)):
                stopped = True
                break
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(scenario())
    records = _force_order_records(tmp_path, _ns(2026, 9, 26, 10, 0, 0))
    closes = [item for item in records if item["kind"] == "ws_close"]
    assert closes
    assert closes[0]["reason"] == "error:ConnectionError"


def test_connect_failure_retries_with_backoff(tmp_path: Path) -> None:
    """A refused first dial backs off once, then the stream flows."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 0))
    stopped = False
    attempts = 0
    conn = FakeConn(clock, [("text", "late-hello")])

    async def flaky_connect(url: str) -> FakeConn:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("refused")
        return conn

    status = WsStatus()
    capture = ForceOrderCapture(
        config=CaptureConfig(force_order_url="wss://example.invalid/stream"),
        writer=SegmentWriter(tmp_path, "force_order", "blue"),
        status=status,
        connect=flaky_connect,  # type: ignore[arg-type]
        clock_ns=clock,
        sleep=clock.sleep,
        shutdown=lambda: stopped,
    )

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(capture.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            if status.first_frame_at_ns is not None:
                stopped = True
                break
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(scenario())
    assert attempts >= 2
    records = _force_order_records(tmp_path, _ns(2026, 9, 26, 10, 0, 5))
    assert any(item["kind"] == "frame" for item in records)


def test_shutdown_right_after_close_skips_reconnect(tmp_path: Path) -> None:
    """Shutdown observed between close and redial returns without another connection."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 0))
    dials: list[str] = []
    conn = FakeConn(clock, [("close", "server_close")])
    capture_holder: dict[str, ForceOrderCapture] = {}

    def is_shutdown() -> bool:
        return len(dials) >= 1 and capture_holder["capture"]._writer.pending() == 0

    capture, status = _capture(tmp_path, clock, conn, is_shutdown, dials)
    capture_holder["capture"] = capture

    async def scenario() -> None:
        task = asyncio.ensure_future(capture.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            records = _force_order_records(tmp_path, _ns(2026, 9, 26, 10, 0, 0))
            if any(item["kind"] == "ws_close" for item in records):
                break
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(scenario())
    assert len(dials) == 1
    assert status.reconnects == 0
