"""Invariant guards for the execution-window depth recorder (no network)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.live.depth_capture import (
    EXEC_DEPTH_DATASET,
    DepthCaptureSummary,
    ExecutionDepthRecorder,
)

_DECISION = pd.Timestamp("2026-09-22T00:25:00Z")


def _depth_message(
    symbol: str,
    bids: list[tuple[str, str]],
    asks: list[tuple[str, str]],
    *,
    event_ms: int | None = 1000,
    transact_ms: int | None = 1001,
    update_id: int = 5,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "e": "depthUpdate",
        "s": symbol,
        "U": 1,
        "u": update_id,
        "b": [[p, q] for p, q in bids],
        "a": [[p, q] for p, q in asks],
    }
    if event_ms is not None:
        data["E"] = event_ms
    if transact_ms is not None:
        data["T"] = transact_ms
    return {"stream": f"{symbol.lower()}@depth5@500ms", "data": data}


def _full_book(symbol: str, base: float = 100.0) -> dict[str, Any]:
    bids = [(f"{base - i * 0.1:.1f}", "1.0") for i in range(5)]
    asks = [(f"{base + 0.2 + i * 0.1:.1f}", "2.0") for i in range(5)]
    return _depth_message(symbol, bids, asks)


class _FakeConnection:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)

    async def recv(self) -> Any:
        if not self._script:
            await asyncio.sleep(3600.0)
            raise AssertionError("unreachable")
        action = self._script.pop(0)
        if action == "drop":
            raise ConnectionError("ws dropped")
        return action


class _FakeContext:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConnection:
        return self._conn

    async def __aexit__(self, *args: Any) -> bool:
        return False


def _connect_factory(scripts: list[list[Any]], calls: list[str]):
    def _connect(url: str) -> _FakeContext:
        calls.append(url)
        script = scripts[min(len(calls) - 1, len(scripts) - 1)]
        return _FakeContext(_FakeConnection(list(script)))

    return _connect


def _ticking_now(start: pd.Timestamp, step_s: float = 1.0):
    state = {"n": 0}

    def _now() -> pd.Timestamp:
        out = start + pd.Timedelta(seconds=state["n"] * step_s)
        state["n"] += 1
        return out

    return _now


def _make_recorder(
    tmp_path: Path,
    symbols: list[str],
    scripts: list[list[Any]],
    calls: list[str],
    *,
    now: Any | None = None,
    flush_interval_s: float = 3600.0,
    max_session_s: float = 60.0,
    **overrides: Any,
) -> ExecutionDepthRecorder:
    return ExecutionDepthRecorder(
        symbols,
        decision_time=_DECISION,
        run_id="20260922",
        mode="paper",
        root=tmp_path,
        stream_url="wss://test/stream",
        levels=5,
        update_ms=500,
        flush_interval_s=flush_interval_s,
        max_session_s=max_session_s,
        connect=_connect_factory(scripts, calls),
        now_fn=now if now is not None else _ticking_now(_DECISION),
        **overrides,
    )


def _wait_for(predicate: Any, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for capture"
        time.sleep(0.05)


def _read_parts(tmp_path: Path) -> pd.DataFrame:
    files = sorted((tmp_path / EXEC_DEPTH_DATASET / "20260922").glob("part_*.parquet"))
    assert files
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def test_messages_become_wide_rows(tmp_path: Path) -> None:
    """Depth messages become wide top-of-book rows with NaN for missing levels."""
    calls: list[str] = []
    short = _depth_message("ETHUSDT", [(f"{50 - i * 0.1:.1f}", "1") for i in range(3)], [("50.5", "1")])
    rec = _make_recorder(tmp_path, ["BTCUSDT", "ETHUSDT"], [[_full_book("BTCUSDT"), _full_book("BTCUSDT"), short]], calls)
    rec.start()
    _wait_for(lambda: rec._rows >= 3)
    summary = rec.stop(post_window_s=0.0)
    assert summary.rows == 3
    assert summary.symbols_seen == 2
    assert isinstance(summary, DepthCaptureSummary)
    frame = _read_parts(tmp_path)
    assert len(frame) == 3
    eth = frame[frame["symbol"] == "ETHUSDT"].iloc[0]
    assert pd.isna(eth["bid_px_3"])
    assert pd.isna(eth["bid_px_4"])
    assert str(frame["bid_qty_0"].dtype) == "float32"
    assert str(frame["bid_px_0"].dtype) == "float64"
    assert "depth5@500ms" in calls[0]


def test_parts_immutable_across_sessions(tmp_path: Path) -> None:
    """A retried decision adds parts without touching existing files."""
    frozen = pd.Timestamp("2026-09-22T00:25:01Z")
    first_calls: list[str] = []
    first = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], first_calls, now=lambda: frozen)
    first.start()
    _wait_for(lambda: first._rows >= 1)
    first.stop(post_window_s=0.0)
    before = {p.name: p.read_bytes() for p in sorted((tmp_path / EXEC_DEPTH_DATASET / "20260922").glob("part_*.parquet"))}
    assert len(before) == 1
    second_calls: list[str] = []
    second = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], second_calls, now=lambda: frozen)
    second.start()
    _wait_for(lambda: second._rows >= 1)
    second.stop(post_window_s=0.0)
    after = {p.name: p.read_bytes() for p in sorted((tmp_path / EXEC_DEPTH_DATASET / "20260922").glob("part_*.parquet"))}
    assert len(after) == 2
    for name, blob in before.items():
        assert after[name] == blob


def test_reconnect_counted_and_gap_attested(tmp_path: Path) -> None:
    """A dropped connection reconnects and the outage stays a visible gap."""
    from src.market_data.streams.coverage import load_coverage

    calls: list[str] = []
    drop_script: list[Any] = [_full_book("BTCUSDT"), _full_book("BTCUSDT"), "drop"]
    live_script: list[Any] = [_full_book("BTCUSDT"), _full_book("BTCUSDT")]
    rec = _make_recorder(tmp_path, ["BTCUSDT"], [drop_script, live_script], calls)
    rec.start()
    _wait_for(lambda: rec._rows >= 4)
    summary = rec.stop(post_window_s=0.0)
    assert summary.reconnects == 1
    assert summary.rows == 4
    out = load_coverage(
        tmp_path, "exec_depth",
        start=pd.Timestamp("2026-09-22T00:00:00Z"), end=pd.Timestamp("2026-09-22T02:00:00Z"),
    )
    assert len(out) == 2


def test_hard_cap_ends_forgotten_session(tmp_path: Path) -> None:
    """A session ends by itself at max_session_s even without stop."""
    calls: list[str] = []
    rec = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], calls, max_session_s=0.3)
    rec.start()
    deadline = time.monotonic() + 5.0
    while rec._thread is not None and rec._thread.is_alive():
        assert time.monotonic() < deadline, "session thread did not exit at the hard cap"
        time.sleep(0.05)
    frame = _read_parts(tmp_path)
    assert len(frame) == 1


def test_shutdown_shortens_post_window(tmp_path: Path) -> None:
    """A requested shutdown ends the post window promptly with flushed rows."""
    calls: list[str] = []
    rec = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], calls)
    rec.start()
    _wait_for(lambda: rec._rows >= 1)

    class _Shutdown:
        requested = True

    started = time.monotonic()
    summary = rec.stop(post_window_s=3600.0, shutdown=_Shutdown())
    assert time.monotonic() - started < 10.0
    assert summary.rows == 1
    assert len(_read_parts(tmp_path)) == 1


def test_connect_failure_never_raises(tmp_path: Path) -> None:
    """A dead venue yields an empty summary, never an exception."""
    def _boom(url: str) -> Any:
        raise RuntimeError("connect down")

    rec = ExecutionDepthRecorder(
        ["BTCUSDT"], decision_time=_DECISION, run_id="20260922", mode="paper", root=tmp_path,
        stream_url="wss://test/stream", levels=5, update_ms=500,
        flush_interval_s=3600.0, max_session_s=60.0, connect=_boom,
        now_fn=_ticking_now(_DECISION),
    )
    rec.start()
    time.sleep(0.5)
    summary = rec.stop(post_window_s=0.0)
    assert summary.rows == 0
    assert summary.parts == 0


def test_invalid_depth_variant_rejected(tmp_path: Path) -> None:
    """Only Binance partial-depth variants are accepted."""
    with pytest.raises(ValueError, match="levels"):
        ExecutionDepthRecorder(
            ["BTCUSDT"], decision_time=_DECISION, run_id="x", mode="paper", root=tmp_path,
            stream_url="wss://test", levels=7, update_ms=500,
            flush_interval_s=1.0, max_session_s=1.0,
        )
    with pytest.raises(ValueError, match="update_ms"):
        ExecutionDepthRecorder(
            ["BTCUSDT"], decision_time=_DECISION, run_id="x", mode="paper", root=tmp_path,
            stream_url="wss://test", levels=5, update_ms=700,
            flush_interval_s=1.0, max_session_s=1.0,
        )


def test_empty_symbols_is_noop(tmp_path: Path) -> None:
    """Empty symbol lists start and stop without work."""
    rec = ExecutionDepthRecorder(
        [], decision_time=_DECISION, run_id="x", mode="paper", root=tmp_path,
        stream_url="wss://test", levels=5, update_ms=500,
        flush_interval_s=1.0, max_session_s=1.0,
    )
    rec.start()
    summary = rec.stop(post_window_s=0.0)
    assert summary.rows == 0
    assert summary.symbols_requested == 0


def test_stop_without_start_is_safe(tmp_path: Path) -> None:
    """Stopping a never-started recorder is a zero summary."""
    rec = ExecutionDepthRecorder(
        ["BTCUSDT"], decision_time=_DECISION, run_id="x", mode="paper", root=tmp_path,
        stream_url="wss://test", levels=5, update_ms=500,
        flush_interval_s=1.0, max_session_s=1.0,
    )
    summary = rec.stop(post_window_s=0.0)
    assert summary.rows == 0


def test_coverage_flush_failure_keeps_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing coverage flush never loses captured rows."""
    from src.market_data.streams.coverage import CoverageTracker

    calls: list[str] = []
    monkeypatch.setattr(CoverageTracker, "flush", lambda self: (_ for _ in ()).throw(OSError("disk full")))
    rec = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], calls)
    rec.start()
    _wait_for(lambda: rec._rows >= 1)
    summary = rec.stop(post_window_s=0.0)
    assert summary.rows == 1
    assert len(_read_parts(tmp_path)) == 1


def test_start_failure_is_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A thread that cannot start degrades to a zero summary."""
    import threading

    class _BoomThread:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("no threads")

    monkeypatch.setattr(threading, "Thread", _BoomThread)
    rec = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], [])
    rec.start()
    summary = rec.stop(post_window_s=0.0)
    assert summary.rows == 0


def test_stuck_thread_abandoned_with_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A thread ignoring stop is abandoned but still reports flushed rows."""
    import threading

    calls: list[str] = []
    real_join = threading.Thread.join
    monkeypatch.setattr(threading.Thread, "join", lambda self, timeout=None: (_ for _ in ()).throw(RuntimeError("stuck")))
    rec = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], calls)
    rec.start()
    _wait_for(lambda: rec._rows >= 1)
    summary = rec.stop(post_window_s=0.0)
    assert summary.rows == 1
    real_join(rec._thread, timeout=5.0)


def test_depth_helper_edge_cases() -> None:
    """Parse helpers reject malformed input without raising."""
    from src.live.depth_capture import (
        _depth_stream_url,
        _parse_depth_message,
        _to_opt_int,
        _utc_now,
    )

    assert _utc_now().tzinfo is not None
    assert _depth_stream_url("wss://x", ["BTCUSDT"], 5, 500) == "wss://x?streams=btcusdt@depth5@500ms"
    assert _to_opt_int(None) is None
    assert _to_opt_int("") is None
    assert _to_opt_int("  ") is None
    assert _to_opt_int("abc") is None
    assert _to_opt_int(123) == 123
    assert _to_opt_int("456") == 456
    cap = pd.Timestamp("2026-09-22T00:25:00Z")
    assert _parse_depth_message(None, decision_time=cap, run_id="r", mode="m", received_at=cap, levels=5) is None
    assert _parse_depth_message({"data": 5}, decision_time=cap, run_id="r", mode="m", received_at=cap, levels=5) is None
    assert _parse_depth_message({"s": "X"}, decision_time=cap, run_id="r", mode="m", received_at=cap, levels=5) is None
    row = _parse_depth_message(
        {"s": "BTCUSDT", "E": "", "T": "1001", "u": 7, "b": [["bad", "1"], "oops"], "a": [["50.5", "bad"]]},
        decision_time=cap, run_id="r", mode="m", received_at=cap, levels=2,
    )
    assert row is not None
    assert row["event_time_ms"] is None
    assert row["transact_time_ms"] == 1001
    assert row["bid_px_0"] is None
    assert row["bid_px_1"] is None
    assert row["ask_px_0"] == 50.5
    assert row["ask_qty_0"] is None


def test_default_connect_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default aiohttp connector delivers messages and closes the session."""
    import asyncio

    import aiohttp
    from src.live.depth_capture import _default_connect

    canned = {"data": {"s": "BTCUSDT"}}
    closed = {"n": 0}

    class _FakeWS:
        async def receive_json(self) -> object:
            return canned

    class _FakeWSContext:
        def __init__(self, ws: _FakeWS) -> None:
            self._ws = ws

        async def __aenter__(self) -> _FakeWS:
            return self._ws

        async def __aexit__(self, *args: object) -> bool:
            return False

    class _FakeSession:
        def __init__(self, **kwargs: object) -> None:
            return None

        def ws_connect(self, url: str, **kwargs: object) -> _FakeWSContext:
            assert url == "wss://test/stream"
            return _FakeWSContext(_FakeWS())

        async def close(self) -> None:
            closed["n"] += 1

    monkeypatch.setattr(aiohttp, "ClientSession", _FakeSession)

    async def _run() -> None:
        async with _default_connect("wss://test/stream") as conn:
            assert await conn.recv() == canned

    asyncio.run(_run())
    assert closed["n"] == 1


def test_periodic_flush_and_malformed_message(tmp_path: Path) -> None:
    """Buffers flush mid-session; malformed dicts are skipped."""
    calls: list[str] = []
    script: list[Any] = [_full_book("BTCUSDT"), {"junk": True}, _full_book("BTCUSDT")]
    rec = _make_recorder(tmp_path, ["BTCUSDT"], [script], calls, flush_interval_s=0.1)
    rec.start()
    _wait_for(lambda: rec._rows >= 2)
    _wait_for(lambda: list((tmp_path / EXEC_DEPTH_DATASET / "20260922").glob("part_*.parquet")) != [])
    summary = rec.stop(post_window_s=0.0)
    assert summary.rows == 2


def test_stop_waits_post_window_and_is_idempotent(tmp_path: Path) -> None:
    """A positive post window is honored; repeat stops reuse the summary."""
    calls: list[str] = []
    rec = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], calls)
    rec.start()
    _wait_for(lambda: rec._rows >= 1)
    started = time.monotonic()
    first = rec.stop(post_window_s=0.3)
    assert time.monotonic() - started >= 0.3
    assert rec.stop(post_window_s=0.0) == first


def test_stuck_thread_abandoned_and_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A thread ignoring join is abandoned with an error and kept summary."""
    import logging
    import threading

    calls: list[str] = []
    real_join = threading.Thread.join
    monkeypatch.setattr(threading.Thread, "join", lambda self, timeout=None: None)
    rec = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], calls)
    rec.start()
    _wait_for(lambda: rec._rows >= 1)
    with caplog.at_level(logging.ERROR, logger="src.live.depth_capture"):
        summary = rec.stop(post_window_s=0.0)
    assert summary.rows == 1
    assert any("THREAD_STUCK" in record.message for record in caplog.records)
    assert rec._thread is not None
    real_join(rec._thread, timeout=5.0)


def test_session_failure_logged_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A session that blows up is logged inside the thread."""
    import logging

    from src.live.depth_capture import ExecutionDepthRecorder

    async def _boom(self: ExecutionDepthRecorder) -> None:
        raise RuntimeError("loop broken")

    monkeypatch.setattr(ExecutionDepthRecorder, "_run_session", _boom)
    rec = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], [])
    with caplog.at_level(logging.WARNING, logger="src.live.depth_capture"):
        rec.start()
        assert rec._thread is not None
        rec._thread.join(timeout=5.0)
    assert any("SESSION_FAILED" in record.message for record in caplog.records)
    assert rec.stop(post_window_s=0.0).rows == 0
