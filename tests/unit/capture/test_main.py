"""Invariant guards for capture process orchestration, isolation and handover overlap."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.capture.config import CaptureConfig
from src.capture.journal import hot_segment_path, iter_complete_records
from src.capture.main import _amain, _supervised, main

_HALF_SECOND_NS = 500_000_000


def _ns(hour: int, minute: int = 0, second: int = 0) -> int:
    moment = datetime(2026, 9, 26, hour, minute, second, tzinfo=UTC)
    return int(moment.timestamp() * 1_000_000_000)


class FakeClock:
    """Scenario-driven nanosecond clock; tasks observe it, only the scenario advances it."""

    def __init__(self, start_ns: int) -> None:
        """Start the clock at ``start_ns``."""
        self.now_ns = start_ns

    def __call__(self) -> int:
        """Return the current fake time."""
        return self.now_ns

    def tick(self) -> None:
        """Advance the clock by half a second (one scheduling round)."""
        self.now_ns += _HALF_SECOND_NS


async def _yield_sleep(delay: float) -> None:
    """Pure cooperative sleep: yield without moving the fake clock."""
    _ = delay
    await asyncio.sleep(0)


class FakeConn:
    """Replay scripted text frames, then report silence."""

    def __init__(self, script: list[str]) -> None:
        """Queue ``script`` frame texts for replay."""
        self._script = deque(script)
        self.reads = 0

    async def read(self, wait_s: float) -> tuple[str, Any]:
        """Return the next frame or a timeout."""
        _ = wait_s
        self.reads += 1
        await asyncio.sleep(0)
        if self._script:
            return ("text", self._script.popleft())
        return ("timeout", None)

    async def ping(self) -> None:
        """Accept the client ping."""
        await asyncio.sleep(0)

    async def close(self) -> None:
        """Accept the release."""
        await asyncio.sleep(0)


def _all_records(root: Path, stream: str, slot: str, hour: int) -> list[dict[str, Any]]:
    dest = hot_segment_path(root, stream, slot, _ns(hour))
    if not dest.exists():
        return []
    return [record for record, _ in iter_complete_records(dest, 0)]


async def _tick_until(clock: FakeClock, cond: Callable[[], bool], limit: int = 20000) -> None:
    for _ in range(limit):
        await asyncio.sleep(0)
        clock.tick()
        if cond():
            return
    raise AssertionError("stop condition was not met")


def test_graceful_shutdown_flushes_everything(tmp_path: Path) -> None:
    """Spec 01: SIGTERM-equivalent shutdown exits 0 with rest, frames, markers and stopped_at."""
    clock = FakeClock(_ns(10, 0, 1))
    book_calls = 0
    stopped = False
    conn = FakeConn(["frame-a", "frame-b", "frame-c"])

    async def fetch(url: str) -> tuple[int, str, dict[str, str]]:
        nonlocal book_calls
        if "bookTicker" in url:
            book_calls += 1
        return (200, '{"ok":true}', {})

    async def connect(url: str) -> FakeConn:
        return conn

    async def scenario() -> dict[str, Any]:
        nonlocal stopped
        task = asyncio.ensure_future(
            _amain(
                slot="blue",
                capture_root=tmp_path,
                config=CaptureConfig(),
                fetch=fetch,  # type: ignore[arg-type]
                connect=connect,  # type: ignore[arg-type]
                clock_ns=clock,
                sleep=_yield_sleep,
                shutdown=lambda: stopped,
                fingerprint="test",
            )
        )
        await _tick_until(clock, lambda: book_calls >= 2 and conn.reads >= 6)
        stopped = True
        return await asyncio.wait_for(task, timeout=10)

    payload = asyncio.run(scenario())
    assert payload["stopped_at"] is not None
    assert payload["ready"] is True
    rests = [item for item in _all_records(tmp_path, "book_ticker", "blue", 10) if item["kind"] == "rest"]
    assert len(rests) >= 2
    frames = _all_records(tmp_path, "force_order", "blue", 10)
    kinds = [item["kind"] for item in frames]
    assert kinds[0] == "ws_open"
    assert kinds[-1] == "ws_close"
    assert frames[-1]["reason"] == "shutdown"
    assert sum(1 for item in frames if item["kind"] == "frame") >= 3
    heartbeat = json.loads((tmp_path / "raw" / "capture_blue.json").read_text(encoding="utf-8"))
    assert heartbeat["stopped_at"] is not None


def test_component_failure_isolated(tmp_path: Path) -> None:
    """Spec 01: a perpetually failing premium endpoint cannot stop bookTicker or WS."""
    clock = FakeClock(_ns(10, 0, 1))
    book_calls = 0
    stopped = False
    conn = FakeConn(["live"])

    async def fetch(url: str) -> tuple[int, str, dict[str, str]]:
        nonlocal book_calls
        if "premiumIndex" in url:
            raise RuntimeError("unexpected")
        if "bookTicker" in url:
            book_calls += 1
        return (200, '{"ok":true}', {})

    async def connect(url: str) -> FakeConn:
        return conn

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(
            _amain(
                slot="blue",
                capture_root=tmp_path,
                config=CaptureConfig(),
                fetch=fetch,  # type: ignore[arg-type]
                connect=connect,  # type: ignore[arg-type]
                clock_ns=clock,
                sleep=_yield_sleep,
                shutdown=lambda: stopped,
                fingerprint="test",
            )
        )
        await _tick_until(clock, lambda: book_calls >= 2 and clock.now_ns >= _ns(10, 6))
        stopped = True
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(scenario())
    rests = [item for item in _all_records(tmp_path, "book_ticker", "blue", 10) if item["kind"] == "rest"]
    assert len(rests) >= 2
    premium = _all_records(tmp_path, "premium_index", "blue", 10)
    assert any(item["kind"] == "rest_error" for item in premium)


def test_supervisor_restarts_failed_task() -> None:
    """A task raising once is restarted with backoff; a clean exit under shutdown returns."""
    clock = FakeClock(_ns(10, 0, 0))
    attempts = 0
    stopped = False

    async def flaky() -> None:
        nonlocal attempts, stopped
        attempts += 1
        if attempts == 1:
            raise RuntimeError("once")
        stopped = True

    async def scenario() -> None:
        await _supervised("flaky", flaky, shutdown=lambda: stopped, sleep=_yield_sleep, backoff_max_s=60.0)

    asyncio.run(scenario())
    assert attempts == 2


def test_invalid_slot_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    """Spec 01: an unknown slot exits 2 and creates no files."""
    monkeypatch.chdir(tmp_path)
    assert main(["--slot", "purple"]) == 2
    assert list(tmp_path.rglob("*")) == []


def test_two_slots_coexist_without_collision(tmp_path: Path) -> None:
    """Spec 01: blue and green share one root with disjoint hot files and one reference set."""
    for slot in ("blue", "green"):
        clock = FakeClock(_ns(10, 0, 1))
        stopped = False
        conn = FakeConn(["frame"])

        async def fetch(url: str) -> tuple[int, str, dict[str, str]]:
            return (200, '{"ok":true}', {})

        async def connect(url: str, _conn: FakeConn = conn) -> FakeConn:
            return _conn

        async def scenario(_slot: str = slot, _clock: FakeClock = clock) -> None:
            nonlocal stopped
            task = asyncio.ensure_future(
                _amain(
                    slot=_slot,
                    capture_root=tmp_path,
                    config=CaptureConfig(),
                    fetch=fetch,  # type: ignore[arg-type]
                    connect=connect,  # type: ignore[arg-type]
                    clock_ns=_clock,
                    sleep=_yield_sleep,
                    shutdown=lambda: stopped,
                    fingerprint="test",
                )
            )
            await _tick_until(
                _clock,
                lambda: (tmp_path / "raw" / f"capture_{_slot}.json").exists() and _clock.now_ns >= _ns(10, 3),
            )
            stopped = True
            await asyncio.wait_for(task, timeout=10)

        asyncio.run(scenario())
    blue_files = [path.name for path in (tmp_path / "raw" / "hot").rglob("*.blue.jsonl.gz")]
    green_files = [path.name for path in (tmp_path / "raw" / "hot").rglob("*.green.jsonl.gz")]
    assert blue_files
    assert green_files
    assert all(".blue." in name for name in blue_files)
    assert all(".green." in name for name in green_files)
    references = list((tmp_path / "reference").rglob("*.json.gz"))
    assert len(references) == 3
    assert (tmp_path / "raw" / "capture_blue.json").exists()
    assert (tmp_path / "raw" / "capture_green.json").exists()


def test_no_residue_after_run(tmp_path: Path) -> None:
    """Spec 01: only hot segments, slot heartbeats and reference files remain; zero partials."""
    clock = FakeClock(_ns(10, 0, 1))
    stopped = False
    conn = FakeConn(["frame"])

    async def fetch(url: str) -> tuple[int, str, dict[str, str]]:
        return (200, '{"ok":true}', {})

    async def connect(url: str) -> FakeConn:
        return conn

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(
            _amain(
                slot="blue",
                capture_root=tmp_path,
                config=CaptureConfig(),
                fetch=fetch,  # type: ignore[arg-type]
                connect=connect,  # type: ignore[arg-type]
                clock_ns=clock,
                sleep=_yield_sleep,
                shutdown=lambda: stopped,
                fingerprint="test",
            )
        )
        await _tick_until(clock, lambda: clock.now_ns >= _ns(10, 4))
        stopped = True
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(scenario())
    assert list(tmp_path.rglob("*.partial")) == []
    for path in tmp_path.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(tmp_path).as_posix()
        assert relative.startswith("raw/hot/") or relative.startswith("reference/") or relative in (
            "raw/capture_blue.json",
        ), relative


def test_tracked_writer_counts_failures(tmp_path: Path) -> None:
    """A failed flush increments the shared state; any success resets it."""
    from src.capture.main import _FlushState, _TrackedWriter

    state = _FlushState()
    writer = _TrackedWriter(tmp_path, "book_ticker", "blue", state=state, clock_ns=FakeClock(_ns(10))  )
    blocker = tmp_path / "raw"
    blocker.mkdir(parents=True)
    (blocker / "hot").write_text("not-a-directory")
    writer.add({"v": 1, "stream": "book_ticker", "slot": "blue", "kind": "rest",
                "recv_ns": _ns(10), "grid": "g", "status": 200, "body": "b"})
    try:
        writer.flush()
    except OSError:
        pass
    else:
        raise AssertionError("expected OSError")
    assert state.failures == 1
    (blocker / "hot").unlink()
    assert writer.flush() == 1
    assert state.failures == 0
    assert state.last_flush_at_ns is not None


def test_aiohttp_adapter_branches() -> None:
    """Every websocket message kind maps to the minimal protocol."""
    import aiohttp

    from src.capture.main import _AiohttpWsAdapter

    received: list[tuple[str, Any]] = []
    ponged: list[Any] = []
    pinged = False
    closed = False

    class _Message:
        def __init__(self, kind: Any, data: Any = None) -> None:
            self.type = kind
            self.data = data

    class _Socket:
        def __init__(self, script: list[Any]) -> None:
            self._script = list(script)

        async def receive(self, wait_s: float = 0.0, **kwargs: Any) -> Any:
            _ = (wait_s, kwargs)
            item = self._script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        async def pong(self, data: Any) -> None:
            ponged.append(data)

        async def ping(self) -> None:
            nonlocal pinged
            pinged = True

        async def close(self) -> None:
            nonlocal closed
            closed = True

    async def scenario() -> None:
        socket = _Socket([
            TimeoutError(),
            _Message(aiohttp.WSMsgType.TEXT, "t"),
            _Message(aiohttp.WSMsgType.BINARY, b"b"),
            _Message(aiohttp.WSMsgType.PING, b"p"),
            _Message(aiohttp.WSMsgType.PONG),
            _Message(aiohttp.WSMsgType.CLOSE),
            _Message(aiohttp.WSMsgType.CLOSED),
            _Message(aiohttp.WSMsgType.CLOSING),
            _Message(aiohttp.WSMsgType.ERROR),
            _Message(aiohttp.WSMsgType.TEXT, "u", ),
        ])
        adapter = _AiohttpWsAdapter(socket)  # type: ignore[arg-type]
        received.extend([await adapter.read(1.0) for _ in range(10)])
        socket._script.append(_Message(999, "odd"))
        received.append(await adapter.read(1.0))
        await adapter.ping()
        await adapter.close()

    asyncio.run(scenario())
    assert received[0] == ("timeout", None)
    assert received[1] == ("text", "t")
    assert received[2] == ("binary", b"b")
    assert received[3] == ("control", None)
    assert received[4] == ("control", None)
    assert received[5] == ("close", "server_close")
    assert received[6] == ("close", "server_close")
    assert received[7] == ("close", "server_close")
    assert received[8] == ("close", "server_close")
    assert received[9] == ("text", "u")
    assert received[10] == ("control", None)
    assert ponged == [b"p"]
    assert pinged
    assert closed


def test_fetch_and_connect_factories() -> None:
    """The shared session yields status/body/headers and recorder-parity sockets."""
    from src.capture.main import _AiohttpWsAdapter, _make_connect, _make_fetch

    gotten: list[str] = []
    connected: list[tuple[str, dict[str, Any]]] = []

    class _Response:
        status = 201

        def __init__(self) -> None:
            self.headers = {"Retry-After": "3"}

        async def text(self) -> str:
            return "hello"

        async def __aenter__(self) -> _Response:
            return self

        async def __aexit__(self, *args: object) -> bool:
            return False

    class _Session:
        def get(self, url: str) -> _Response:
            gotten.append(url)
            return _Response()

        async def ws_connect(self, url: str, **kwargs: Any) -> object:
            connected.append((url, kwargs))
            return object()

    async def scenario() -> None:
        fetch = _make_fetch(_Session())  # type: ignore[arg-type]
        assert await fetch("https://example.invalid/x") == (201, "hello", {"Retry-After": "3"})
        connect = _make_connect(_Session())  # type: ignore[arg-type]
        adapter = await connect("wss://example.invalid/s")
        assert isinstance(adapter, _AiohttpWsAdapter)

    asyncio.run(scenario())
    assert gotten == ["https://example.invalid/x"]
    assert connected[0][0] == "wss://example.invalid/s"
    assert connected[0][1] == {"autoping": False, "heartbeat": None}


def test_read_fingerprint_branches(tmp_path: Path, monkeypatch: Any) -> None:
    """Missing, blank and valued fingerprint files map to None/None/value."""
    import src.capture.main as _main

    monkeypatch.setattr(_main, "_FINGERPRINT_PATH", tmp_path / "missing")
    assert _main._read_fingerprint() is None
    blank = tmp_path / "blank"
    blank.write_text("  \n")
    monkeypatch.setattr(_main, "_FINGERPRINT_PATH", blank)
    assert _main._read_fingerprint() is None
    valued = tmp_path / "valued"
    valued.write_text("fp-123\n")
    monkeypatch.setattr(_main, "_FINGERPRINT_PATH", valued)
    assert _main._read_fingerprint() == "fp-123"


def test_supervisor_cancel_and_unexpected_exit() -> None:
    """Cancellation propagates; a silent return without shutdown restarts the task."""
    runs = 0
    stopped = False

    async def silent() -> None:
        nonlocal runs, stopped
        runs += 1
        if runs >= 3:
            stopped = True

    async def scenario() -> None:
        await _supervised("silent", silent, shutdown=lambda: stopped, sleep=_yield_sleep, backoff_max_s=60.0)

    asyncio.run(scenario())
    assert runs >= 2

    async def expect_cancel() -> None:
        task = asyncio.ensure_future(
            _supervised("slow", _sleep_forever, shutdown=lambda: False, sleep=_yield_sleep, backoff_max_s=1.0)
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(expect_cancel())


async def _sleep_forever() -> None:
    """Block until cancelled (supervisor cancellation path)."""
    await asyncio.sleep(3600)


def test_deadman_failing_branches() -> None:
    """Flush failures, rest failures and WS silence each warrant /fail."""
    from src.capture.main import _FlushState, _deadman_failing
    from src.capture.rest import RestStatus
    from src.capture.ws import WsStatus

    config = CaptureConfig()

    def failing(config: CaptureConfig, rest: Any, ws: Any, failures: int, now_ns: int) -> bool:
        state = _FlushState()
        state.failures = failures
        return _deadman_failing(config=config, rest=rest, ws_status=ws, state=state, now_ns=now_ns)

    now_ns = _ns(10, 30)
    healthy_rest = {"book_ticker": RestStatus(), "premium_index": RestStatus()}
    assert failing(config, healthy_rest, WsStatus(), 1, now_ns) is True
    bad_rest = {"book_ticker": RestStatus(consecutive_failures=3), "premium_index": RestStatus()}
    assert failing(config, bad_rest, WsStatus(), 0, now_ns) is True
    silent_ws = WsStatus(connected_at_ns=_ns(10, 0), last_frame_at_ns=_ns(10, 0))
    assert failing(config, healthy_rest, silent_ws, 0, now_ns) is True
    fresh_ws = WsStatus(connected_at_ns=now_ns, last_frame_at_ns=now_ns)
    assert failing(config, healthy_rest, fresh_ws, 0, now_ns) is False
    assert failing(config, healthy_rest, WsStatus(), 0, now_ns) is False
    dropped_rest = {"book_ticker": RestStatus(dropped_records=1)}
    assert failing(config, dropped_rest, WsStatus(), 0, now_ns) is True
    assert failing(config, healthy_rest, WsStatus(pending_dropped=1), 0, now_ns) is True


def test_heartbeat_loop_tolerates_write_failures(tmp_path: Path, monkeypatch: Any) -> None:
    """An unwritable heartbeat file warns but never stops the loop."""
    import src.capture.main as _main
    from src.capture.rest import RestStatus
    from src.capture.ws import WsStatus

    clock = FakeClock(_ns(10, 0, 0))
    stopped = False

    def failing_write(root: Path, slot: str, payload: Any) -> Path:
        raise OSError("read-only")

    monkeypatch.setattr(_main, "write_capture_heartbeat", failing_write)
    pinger = _main.DeadmanPinger(None, interval_s=300.0, timeout_s=10.0)

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(
            _main._heartbeat_loop(
                slot="blue",
                capture_root=tmp_path,
                config=CaptureConfig(),
                rest={"book_ticker": RestStatus(), "premium_index": RestStatus()},
                ws_status=WsStatus(),
                state=_main._FlushState(),
                started_at_ns=clock(),
                fingerprint=None,
                pinger=pinger,
                clock_ns=clock,
                sleep=_yield_sleep,
                shutdown=lambda: stopped,
            )
        )
        await _tick_until(clock, lambda: clock.now_ns >= _ns(10, 1))
        stopped = True
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(scenario())


def test_amain_cancel_on_shutdown_timeout(tmp_path: Path) -> None:
    """Tasks ignoring shutdown are cancelled within the flush budget; files still finalize."""
    clock = FakeClock(_ns(10, 0, 1))
    stopped = False

    async def fetch(url: str) -> tuple[int, str, dict[str, str]]:
        return (200, '{"ok":true}', {})

    class BlockingConn:
        async def read(self, wait_s: float) -> tuple[str, Any]:
            await asyncio.sleep(3600)

        async def ping(self) -> None:
            await asyncio.sleep(0)

        async def close(self) -> None:
            await asyncio.sleep(0)

    async def connect(url: str) -> BlockingConn:
        return BlockingConn()

    async def scenario() -> dict[str, Any]:
        nonlocal stopped
        task = asyncio.ensure_future(
            _amain(
                slot="blue",
                capture_root=tmp_path,
                config=CaptureConfig(stop_flush_budget_s=0.5),
                fetch=fetch,  # type: ignore[arg-type]
                connect=connect,  # type: ignore[arg-type]
                clock_ns=clock,
                sleep=_yield_sleep,
                shutdown=lambda: stopped,
                fingerprint="test",
            )
        )
        await _tick_until(clock, lambda: clock.now_ns >= _ns(10, 1))
        stopped = True
        return await asyncio.wait_for(task, timeout=30)

    payload = asyncio.run(scenario())
    assert payload["stopped_at"] is not None
    assert (tmp_path / "raw" / "capture_blue.json").exists()


def test_main_happy_path(tmp_path: Path, monkeypatch: Any) -> None:
    """A valid slot writes the starting heartbeat, runs, and exits 0."""
    import src.capture.main as _main

    seen: dict[str, Any] = {}

    async def fake_amain(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"slot": kwargs["slot"]}

    monkeypatch.setattr(_main, "configure_capture_logging", lambda *a, **k: None)
    monkeypatch.setattr(_main, "_amain", fake_amain)
    assert main(["--slot", "blue", "--capture-root", str(tmp_path)]) == 0
    assert seen["slot"] == "blue"
    assert seen["capture_root"] == tmp_path
    heartbeat = json.loads((tmp_path / "raw" / "capture_blue.json").read_text(encoding="utf-8"))
    assert heartbeat["ready"] is False
    assert heartbeat["stopped_at"] is None


def test_supervisor_returns_when_shutdown_during_restart() -> None:
    """An exception observed while shutdown is already set returns without sleeping."""
    stopped = False

    async def boom() -> None:
        nonlocal stopped
        stopped = True
        raise RuntimeError("late")

    async def scenario() -> None:
        await _supervised("boom", boom, shutdown=lambda: stopped, sleep=_yield_sleep, backoff_max_s=60.0)

    asyncio.run(scenario())
    assert stopped
