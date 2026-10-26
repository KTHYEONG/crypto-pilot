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
    select_depth_symbols,
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
        if isinstance(action, float):
            await asyncio.sleep(action)
            return await self.recv()
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
    assert summary.rows_received == 3
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
    now_fixed = pd.Timestamp("2026-09-22T00:25:01Z")
    first_calls: list[str] = []
    first = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], first_calls, now=lambda: now_fixed)
    first.start()
    _wait_for(lambda: first._rows >= 1)
    first.stop(post_window_s=0.0)
    before = {p.name: p.read_bytes() for p in sorted((tmp_path / EXEC_DEPTH_DATASET / "20260922").glob("part_*.parquet"))}
    assert len(before) == 1
    second_calls: list[str] = []
    second = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], second_calls, now=lambda: now_fixed)
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
    assert summary.rows_received == 4
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
    assert summary.rows_received == 1
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
    assert summary.rows_received == 0
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
    assert summary.rows_received == 0
    assert summary.symbols_requested == 0


def test_stop_without_start_is_safe(tmp_path: Path) -> None:
    """Stopping a never-started recorder is a zero summary."""
    rec = ExecutionDepthRecorder(
        ["BTCUSDT"], decision_time=_DECISION, run_id="x", mode="paper", root=tmp_path,
        stream_url="wss://test", levels=5, update_ms=500,
        flush_interval_s=1.0, max_session_s=1.0,
    )
    summary = rec.stop(post_window_s=0.0)
    assert summary.rows_received == 0


def test_coverage_flush_failure_keeps_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing coverage flush never loses captured rows."""
    from src.market_data.streams.coverage import CoverageTracker

    calls: list[str] = []
    monkeypatch.setattr(CoverageTracker, "flush", lambda self: (_ for _ in ()).throw(OSError("disk full")))
    rec = _make_recorder(tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], calls)
    rec.start()
    _wait_for(lambda: rec._rows >= 1)
    summary = rec.stop(post_window_s=0.0)
    assert summary.rows_received == 1
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
    assert summary.rows_received == 0


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
    assert summary.rows_received == 1
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
    assert summary.rows_received == 2


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
    assert summary.rows_received == 1
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
    assert rec.stop(post_window_s=0.0).rows_received == 0


def test_silent_symbol_listed_as_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A requested symbol with no message is named, warned, and manifested with nulls."""
    import json as _json
    import logging

    calls: list[str] = []
    rec = _make_recorder(
        tmp_path,
        ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        [[_full_book("BTCUSDT"), _full_book("ETHUSDT")]],
        calls,
    )
    with caplog.at_level(logging.WARNING, logger="src.live.depth_capture"):
        rec.start()
        _wait_for(lambda: rec._rows >= 2)
        summary = rec.stop(post_window_s=0.0)
    assert summary.symbols_missing == ("SOLUSDT",)
    assert any("SYMBOLS_MISSING" in record.message for record in caplog.records)
    manifests = list((tmp_path / EXEC_DEPTH_DATASET / "20260922").glob("manifest_*.json"))
    assert len(manifests) == 1
    manifest = _json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["symbols_missing"] == ["SOLUSDT"]
    missing = manifest["symbols"]["SOLUSDT"]
    assert missing["first_received_at"] is None
    assert missing["last_received_at"] is None
    assert missing["rows_persisted"] == 0
    assert manifest["symbols"]["BTCUSDT"]["rows_persisted"] == 1


def test_persisted_rows_exclude_failed_final_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed final flush keeps rows out of the persisted count without raising."""
    import logging

    calls: list[str] = []
    rec = _make_recorder(
        tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT"), _full_book("BTCUSDT")]], calls
    )

    def _boom(target: Any, source: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("src.common.durable_io.os.replace", _boom)
    with caplog.at_level(logging.ERROR, logger="src.live.depth_capture"):
        rec.start()
        _wait_for(lambda: rec._rows >= 2)
        summary = rec.stop(post_window_s=0.0)
    assert summary.rows_received == 2
    assert summary.rows_persisted == 0
    assert summary.rows_persisted < summary.rows_received
    assert any("ROWS_LOST" in record.message for record in caplog.records)


def test_manifest_rows_reconcile_with_part_files(tmp_path: Path) -> None:
    """Per-symbol manifest counts sum to the rows in part files and the summary."""
    import json as _json

    calls: list[str] = []
    rec = _make_recorder(
        tmp_path,
        ["BTCUSDT", "ETHUSDT"],
        [[_full_book("BTCUSDT"), _full_book("ETHUSDT")]],
        calls,
        flush_interval_s=0.0,
    )
    rec.start()
    _wait_for(lambda: rec._parts >= 1)
    time.sleep(0.3)
    summary = rec.stop(post_window_s=0.0)
    manifests = list((tmp_path / EXEC_DEPTH_DATASET / "20260922").glob("manifest_*.json"))
    assert len(manifests) == 1
    manifest = _json.loads(manifests[0].read_text(encoding="utf-8"))
    parts = _read_parts(tmp_path)
    assert sum(item["rows_persisted"] for item in manifest["symbols"].values()) == len(parts)
    assert summary.rows_persisted == len(parts)
    assert summary.rows_received == len(parts)


def test_manifest_failure_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A manifest write failure is a warning; the summary still returns."""
    import logging

    calls: list[str] = []
    rec = _make_recorder(
        tmp_path, ["BTCUSDT"], [[_full_book("BTCUSDT")]], calls
    )

    def _boom(self: Any, stamp: str) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(ExecutionDepthRecorder, "_manifest_path", _boom)
    with caplog.at_level(logging.WARNING, logger="src.live.depth_capture"):
        rec.start()
        _wait_for(lambda: rec._rows >= 1)
        summary = rec.stop(post_window_s=0.0)
    assert summary.rows_received == 1
    assert any("MANIFEST_FAILED" in record.message for record in caplog.records)


def test_flush_empty_rows_is_noop(tmp_path: Path) -> None:
    """Flushing nothing persists nothing and reports success."""
    from src.market_data.streams.coverage import CoverageTracker

    calls: list[str] = []
    rec = _make_recorder(tmp_path, ["BTCUSDT"], [[]], calls)
    assert rec._flush_rows([], CoverageTracker("exec_depth", tmp_path)) is True
    assert rec._parts == 0


def test_traded_first_then_candidates() -> None:
    """Candidates append after traded symbols with overlap skipped."""
    assert select_depth_symbols(["A", "B"], ["C", "A", "D"], max_symbols=10) == ["A", "B", "C", "D"]


def test_candidates_never_displace_traded() -> None:
    """Traded symbols keep their slots when the cap is already full."""
    traded = ["A", "B", "C"]
    assert select_depth_symbols(traded, ["D", "E"], max_symbols=3) == traded


def test_truncates_candidates_at_cap() -> None:
    """The combined list is cut at the cap with the traded symbol first."""
    out = select_depth_symbols(["A"], [f"C{i:03d}" for i in range(100)], max_symbols=5)
    assert len(out) == 5
    assert out[0] == "A"


def test_duplicates_removed() -> None:
    """Repeated symbols in either input appear once."""
    assert select_depth_symbols(["A", "A", "B"], ["B", "C", "C"], max_symbols=10) == ["A", "B", "C"]


def test_ready_gate_waits_for_all_symbols(tmp_path: Path) -> None:
    """Both symbols stream; the gate returns ready before the timeout."""
    calls: list[str] = []
    rec = _make_recorder(
        tmp_path, ["AAAUSDT", "BBBUSDT"],
        [[_full_book("AAAUSDT"), 0.05, _full_book("BBBUSDT")]], calls,
    )
    readiness = rec.start(ready_timeout_s=2.0)
    assert readiness.reason == "ready"
    assert readiness.ready is True
    assert readiness.missing == ()
    assert readiness.seen == readiness.required == 2
    assert 0.05 <= readiness.waited_s < 2.0
    summary = rec.stop(post_window_s=0.0)
    assert summary.rows_received == 2


def test_ready_gate_timeout_names_silent(tmp_path: Path) -> None:
    """Only one symbol streams; the gate times out naming the silent one."""
    calls: list[str] = []
    rec = _make_recorder(
        tmp_path, ["AAAUSDT", "BBBUSDT"],
        [[_full_book("AAAUSDT")]], calls,
    )
    readiness = rec.start(ready_timeout_s=0.5)
    assert readiness.reason == "timeout"
    assert readiness.ready is False
    assert readiness.missing == ("BBBUSDT",)
    assert readiness.waited_s >= 0.5
    rec.stop(post_window_s=0.0)


def test_ready_gate_subset_and_unknown(tmp_path: Path) -> None:
    """Candidate-only symbols never block; unknown required symbols are ignored."""
    calls: list[str] = []
    rec = _make_recorder(
        tmp_path, ["AAAUSDT", "CANDUSDT"],
        [[_full_book("AAAUSDT")]], calls,
    )
    readiness = rec.start(ready_timeout_s=1.0, required=["AAAUSDT"])
    assert readiness.reason == "ready"
    rec.stop(post_window_s=0.0)
    calls2: list[str] = []
    rec2 = _make_recorder(tmp_path, ["AAAUSDT"], [[_full_book("AAAUSDT")]], calls2)
    readiness2 = rec2.start(ready_timeout_s=1.0, required=["ZZZUSDT"])
    assert readiness2.reason == "ready"
    assert readiness2.required == 0
    assert readiness2.missing == ()
    rec2.stop(post_window_s=0.0)


def test_ready_gate_zero_timeout_legacy(tmp_path: Path) -> None:
    """Default start stays non-blocking and still captures."""
    calls: list[str] = []
    rec = _make_recorder(tmp_path, ["AAAUSDT"], [[_full_book("AAAUSDT")]], calls)
    readiness = rec.start()
    assert readiness.reason == "not_waited"
    assert readiness.ready is False
    _wait_for(lambda: rec._rows >= 1)
    rec.stop(post_window_s=0.0)


def test_ready_gate_connect_failure(tmp_path: Path) -> None:
    """A dead venue ends the wait without raising."""
    def _boom(url: str) -> Any:
        raise RuntimeError("connect down")

    rec = ExecutionDepthRecorder(
        ["AAAUSDT"], decision_time=_DECISION, run_id="20260922", mode="paper", root=tmp_path,
        stream_url="wss://test/stream", levels=5, update_ms=500,
        flush_interval_s=3600.0, max_session_s=60.0, connect=_boom,
        now_fn=_ticking_now(_DECISION),
    )
    readiness = rec.start(ready_timeout_s=1.0)
    assert readiness.reason in {"timeout", "session_ended"}
    assert readiness.ready is False
    assert rec.stop(post_window_s=0.0).rows_received == 0


def test_ready_gate_shutdown(tmp_path: Path) -> None:
    """A requested shutdown interrupts the wait quickly."""
    import threading as _threading

    calls: list[str] = []
    rec = _make_recorder(tmp_path, ["AAAUSDT"], [[]], calls)

    class _Shutdown:
        requested = False

    shutdown = _Shutdown()

    def _flip() -> None:
        time.sleep(0.2)
        shutdown.requested = True

    flipper = _threading.Thread(target=_flip, daemon=True)
    flipper.start()
    readiness = rec.start(ready_timeout_s=5.0, shutdown=shutdown)
    assert readiness.reason == "shutdown"
    assert readiness.waited_s < 1.0
    flipper.join(timeout=5.0)
    rec.stop(post_window_s=0.0)


def test_ready_gate_empty(tmp_path: Path) -> None:
    """No requested symbols returns immediately without a thread."""
    rec = _make_recorder(tmp_path, [], [[]], [])
    readiness = rec.start(ready_timeout_s=5.0)
    assert readiness.reason == "empty"
    assert readiness.waited_s == 0.0


def test_ready_gate_second_start(tmp_path: Path) -> None:
    """A second start never spawns a second thread and never waits."""
    import threading as _threading

    calls: list[str] = []
    rec = _make_recorder(tmp_path, ["AAAUSDT"], [[_full_book("AAAUSDT")]], calls)
    first = rec.start(ready_timeout_s=1.0)
    assert first.reason == "ready"
    before = len([t for t in _threading.enumerate() if t.name == "exec-depth-capture"])
    second = rec.start(ready_timeout_s=1.0)
    after = len([t for t in _threading.enumerate() if t.name == "exec-depth-capture"])
    assert second.waited_s == 0.0
    assert after == before
    rec.stop(post_window_s=0.0)


def test_ready_gate_manifest_reconciles(tmp_path: Path) -> None:
    """A ready-gated session persists rows that reconcile with the manifest."""
    import json as _json

    calls: list[str] = []
    rec = _make_recorder(
        tmp_path, ["BTCUSDT", "ETHUSDT"],
        [[_full_book("BTCUSDT"), _full_book("ETHUSDT")]], calls,
        flush_interval_s=0.0,
    )
    readiness = rec.start(ready_timeout_s=2.0)
    assert readiness.reason == "ready"
    _wait_for(lambda: rec._parts >= 1)
    time.sleep(0.1)
    summary = rec.stop(post_window_s=0.0)
    manifests = list((tmp_path / EXEC_DEPTH_DATASET / "20260922").glob("manifest_*.json"))
    assert len(manifests) == 1
    manifest = _json.loads(manifests[0].read_text(encoding="utf-8"))
    parts = _read_parts(tmp_path)
    assert sum(item["rows_persisted"] for item in manifest["symbols"].values()) == len(parts)
    assert summary.rows_persisted == len(parts)


def test_ready_gate_second_start_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Second-start precedence covers start_failed, not_waited, shutdown, ended and timeout."""
    import threading as _threading

    # start_failed persists on second start.
    boom_calls: list[str] = []
    boom_rec = _make_recorder(tmp_path, ["AAAUSDT"], [[_full_book("AAAUSDT")]], boom_calls)
    with monkeypatch.context() as patch:
        patch.setattr(_threading, "Thread", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no threads")))
        first_failed = boom_rec.start(ready_timeout_s=1.0)
        assert first_failed.reason == "start_failed"
        assert boom_rec.start(ready_timeout_s=1.0).reason == "start_failed"

    # not_waited on second start with zero timeout.
    silent = _make_recorder(tmp_path, ["AAAUSDT", "BBBUSDT"], [[_full_book("AAAUSDT")]], [])
    assert silent.start(ready_timeout_s=0.5).reason == "timeout"
    assert silent.start(ready_timeout_s=0.0).reason == "not_waited"
    silent.stop(post_window_s=0.0)

    # shutdown wins on second start when still missing.
    pending = _make_recorder(tmp_path, ["AAAUSDT", "BBBUSDT"], [[_full_book("AAAUSDT")]], [])
    assert pending.start(ready_timeout_s=0.5).reason == "timeout"

    class _Shutdown:
        requested = True

    assert pending.start(ready_timeout_s=1.0, shutdown=_Shutdown()).reason == "shutdown"
    # timeout when still missing, alive and no shutdown.
    assert pending.start(ready_timeout_s=1.0).reason == "timeout"
    pending.stop(post_window_s=0.0)

    # session_ended when the capture thread already exited.
    short = _make_recorder(
        tmp_path, ["AAAUSDT", "BBBUSDT"], [[_full_book("AAAUSDT")]], [],
        max_session_s=0.2,
    )
    assert short.start(ready_timeout_s=0.5).reason == "timeout"
    deadline = time.monotonic() + 5.0
    while short._thread is not None and short._thread.is_alive():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert short.start(ready_timeout_s=1.0).reason == "session_ended"
    short.stop(post_window_s=0.0)

    # Unusable required type never raises; the wait reports timeout.
    bad = _make_recorder(tmp_path, ["AAAUSDT"], [[_full_book("AAAUSDT")]], [])
    assert bad.start(ready_timeout_s=1.0).reason == "ready"
    assert bad.start(ready_timeout_s=1.0, required=123).reason == "timeout"  # type: ignore[arg-type]
    bad.stop(post_window_s=0.0)


def test_ready_gate_wait_failure_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exceptions while waiting degrade to timeout instead of raising."""
    calls: list[str] = []
    rec = _make_recorder(tmp_path, ["AAAUSDT", "BBBUSDT"], [[_full_book("AAAUSDT")]], calls)

    class _BadShutdown:
        @property
        def requested(self) -> bool:
            _wait_for(lambda: rec._rows == 1)
            raise RuntimeError("shutdown broken")

    readiness = rec.start(ready_timeout_s=0.5, shutdown=_BadShutdown())
    assert readiness.reason == "timeout"
    assert readiness.ready is False
    assert readiness.missing == ("BBBUSDT",)
    assert readiness.seen == 1
    assert readiness.waited_s < 0.5
    rec.stop(post_window_s=0.0)

    rec2 = _make_recorder(tmp_path, ["AAAUSDT"], [[]], [])
    def _bad_wait(timeout: float | None = None) -> bool:
        raise RuntimeError("wait broken")

    monkeypatch.setattr(rec2._ready_event, "wait", _bad_wait)
    readiness2 = rec2.start(ready_timeout_s=1.0)
    assert readiness2.reason == "timeout"
    rec2.stop(post_window_s=0.0)


def test_ready_gate_covers_dead_thread_and_foreign_symbol(tmp_path: Path) -> None:
    """Non-required rows never trip the event; a dead thread ends the wait."""
    rec = _make_recorder(tmp_path, ["AAAUSDT", "BBBUSDT"], [[_full_book("BBBUSDT")]], [])
    outcome = rec.start(ready_timeout_s=0.1, required=["AAAUSDT"])
    assert outcome.reason == "timeout"
    assert outcome.missing == ("AAAUSDT",)
    rec.stop(post_window_s=0.0)

    def _broken_clock() -> pd.Timestamp:
        raise RuntimeError("session startup failed")

    dead = _make_recorder(tmp_path, ["AAAUSDT"], [[]], [], now=_broken_clock)
    outcome = dead.start(ready_timeout_s=1.0)
    assert outcome.reason == "session_ended"
    dead.stop(post_window_s=0.0)

    bad_timeout = _make_recorder(tmp_path, ["AAAUSDT"], [[_full_book("AAAUSDT")]], [])
    readiness = bad_timeout.start(ready_timeout_s="bad")  # type: ignore[arg-type]
    assert readiness.reason == "timeout"
    bad_timeout.stop(post_window_s=0.0)


def test_ready_gate_required_order_and_zero_precedence(tmp_path: Path) -> None:
    rec = _make_recorder(tmp_path, ["AAAUSDT", "BBBUSDT", "CUSDT"], [[_full_book("AAAUSDT")]], [])
    try:
        outcome = rec.start(ready_timeout_s=0.1, required=["CUSDT", "BBBUSDT", "CUSDT", "UNKNOWN"])
        assert outcome.reason == "timeout"
        assert outcome.missing == ("CUSDT", "BBBUSDT")
        assert outcome.required == 2
        assert outcome.seen == 0
        assert rec.start(required=[]).reason == "not_waited"
        assert rec.start(ready_timeout_s=1.0, required=[]).reason == "ready"
    finally:
        rec.stop(post_window_s=0.0)


@pytest.mark.parametrize("timeout", [float("inf"), float("nan"), -1.0])
def test_ready_gate_invalid_timeout_is_bounded(tmp_path: Path, timeout: float) -> None:
    rec = _make_recorder(tmp_path, ["AAAUSDT"], [[]], [])
    outcome = rec.start(ready_timeout_s=timeout)
    assert outcome.reason == "timeout"
    assert 0 <= outcome.waited_s < 1.0
    assert outcome.missing == ("AAAUSDT",)


def test_ready_gate_monotonic_failure_is_fail_soft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _make_recorder(tmp_path, ["AAAUSDT"], [[]], [])

    def broken() -> float:
        raise RuntimeError("clock unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(time, "monotonic", broken)
        outcome = rec.start(ready_timeout_s=1.0)
    assert outcome.reason == "timeout"
    assert outcome.waited_s == 0.0
    assert outcome.missing == ("AAAUSDT",)
