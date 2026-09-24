"""Invariant guards for the always-on live-only market recorder (no network)."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.market_data.streams.recorder import MarketRecorderConfig, run_market_recorder
from src.market_data.streams.snapshots import REFERENCE_URLS, load_snapshot_dataset

_BOOK_ROW = {
    "symbol": "BTCUSDT", "bidPrice": "60000", "bidQty": "1",
    "askPrice": "60001", "askQty": "1", "time": 1758531600000,
}
_PREMIUM_ROW = {
    "symbol": "BTCUSDT", "markPrice": "60000", "indexPrice": "60000",
    "estimatedSettlePrice": "60000", "lastFundingRate": "0.0001",
    "interestRate": "0.0", "nextFundingTime": 1758531600000, "time": 1758531590000,
}
_REF_BYTES = b'{"symbols": []}'


class _Flag:
    requested = False


class _Clock:
    def __init__(self, start: pd.Timestamp) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> pd.Timestamp:
        return self.t

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(float(delay))
        self.t += pd.Timedelta(seconds=float(delay))
        await asyncio.sleep(0)


def _default_config(**overrides: Any) -> MarketRecorderConfig:
    params: dict[str, Any] = {
        "book_ticker_interval_s": 60,
        "premium_index_interval_s": 300,
        "flush_interval_s": 300.0,
        "reference_capture_after_utc": "00:05",
        "liquidation_flush_interval_s": 60.0,
    }
    params.update(overrides)
    return MarketRecorderConfig(**params)


async def _quiet_liquidations(flag: _Flag, **kwargs: Any) -> None:
    while not flag.requested:  # noqa: ASYNC110 - test stub parks until the flag flips
        await asyncio.sleep(0)


def _run(coro: Any, timeout: float = 30.0) -> None:
    async def _guard() -> None:
        await asyncio.wait_for(coro, timeout=timeout)

    asyncio.run(_guard())


def _fetch_router(
    flag: _Flag,
    clock: _Clock,
    *,
    book_calls: list[int],
    book_fail_first: int = 0,
    premium_fail: bool = False,
    stop_after_book: int | None = None,
    ref_bytes_fn: Any | None = None,
):
    async def _fetch(url: str) -> bytes:
        if "bookTicker" in url:
            book_calls.append(1)
            if len(book_calls) <= book_fail_first:
                raise RuntimeError("book fetch down")
            if stop_after_book is not None and len(book_calls) >= stop_after_book:
                flag.requested = True
            return json.dumps([dict(_BOOK_ROW)]).encode()
        if "premiumIndex" in url:
            if premium_fail:
                raise RuntimeError("premium down")
            return json.dumps([dict(_PREMIUM_ROW)]).encode()
        if ref_bytes_fn is not None:
            return ref_bytes_fn(url)
        return _REF_BYTES

    return _fetch


def test_grid_samples_stamped_on_grid(tmp_path: Path) -> None:
    """Samples carry the grid instant they were issued for."""
    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []
    fetch = _fetch_router(flag, clock, book_calls=book_calls, stop_after_book=2)
    cap, liq = tmp_path / "cap", tmp_path / "liq"
    _run(
        run_market_recorder(
            _default_config(), capture_root=cap, liquidations_dir=liq, shutdown=flag,
            fetch=fetch, liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
            now_fn=clock.now, sleep=clock.sleep,
        )
    )
    out = load_snapshot_dataset(
        cap, "book_ticker",
        start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T10:03:00Z"),
    )
    assert sorted(out["captured_at"].unique()) == [
        pd.Timestamp("2026-09-22T10:01:00Z"), pd.Timestamp("2026-09-22T10:02:00Z")
    ]


def test_failed_grid_point_skipped_not_shifted(tmp_path: Path) -> None:
    """A failed grid point is skipped; the next sample keeps its own stamp."""
    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []
    fetch = _fetch_router(flag, clock, book_calls=book_calls, book_fail_first=1, stop_after_book=2)
    cap, liq = tmp_path / "cap", tmp_path / "liq"
    _run(
        run_market_recorder(
            _default_config(), capture_root=cap, liquidations_dir=liq, shutdown=flag,
            fetch=fetch, liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
            now_fn=clock.now, sleep=clock.sleep,
        )
    )
    out = load_snapshot_dataset(
        cap, "book_ticker",
        start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T10:03:00Z"),
    )
    assert sorted(out["captured_at"].unique()) == [pd.Timestamp("2026-09-22T10:02:00Z")]
    heartbeat = json.loads((cap / "recorder_heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["book_ticker"]["consecutive_failures"] == 0


def test_hour_rollover_flushes_closed_hour(tmp_path: Path) -> None:
    """The closed hour is persisted as soon as the next hour is sampled."""
    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:58:20Z"))
    book_calls: list[int] = []
    cap = tmp_path / "cap"
    seen: dict[str, bool] = {}

    async def _fetch(url: str) -> bytes:
        if "bookTicker" in url:
            book_calls.append(1)
            if len(book_calls) == 3:
                seen["ten"] = (cap / "book_ticker" / "20260922" / "10.parquet").exists()
                flag.requested = True
            return json.dumps([dict(_BOOK_ROW)]).encode()
        if "premiumIndex" in url:
            return json.dumps([dict(_PREMIUM_ROW)]).encode()
        return _REF_BYTES

    _run(
        run_market_recorder(
            _default_config(flush_interval_s=3600.0), capture_root=cap,
            liquidations_dir=tmp_path / "liq", shutdown=flag,
            fetch=_fetch, liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
            now_fn=clock.now, sleep=clock.sleep,
        )
    )
    assert seen.get("ten") is True
    ten = pd.read_parquet(cap / "book_ticker" / "20260922" / "10.parquet")
    assert (ten["captured_at"] == pd.Timestamp("2026-09-22T10:59:00Z")).all()


def test_failing_task_does_not_stop_others(tmp_path: Path) -> None:
    """A permanently failing source never stops the healthy samplers."""
    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []
    fetch = _fetch_router(
        flag, clock, book_calls=book_calls, premium_fail=True, stop_after_book=4
    )
    cap = tmp_path / "cap"
    _run(
        run_market_recorder(
            _default_config(), capture_root=cap, liquidations_dir=tmp_path / "liq",
            shutdown=flag, fetch=fetch,
            liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
            now_fn=clock.now, sleep=clock.sleep,
        )
    )
    out = load_snapshot_dataset(
        cap, "book_ticker",
        start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T10:06:00Z"),
    )
    assert len(out["captured_at"].unique()) >= 3


def test_crashing_task_restarted_with_capped_backoff(tmp_path: Path) -> None:
    """A crashing task is retried with backoff bounded by the configured max."""
    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []
    fetch = _fetch_router(flag, clock, book_calls=book_calls)
    calls: list[int] = []

    async def _flaky_runner(**kwargs: Any) -> None:
        calls.append(1)
        if len(calls) >= 3:
            flag.requested = True
        raise RuntimeError("boom")

    cap = tmp_path / "cap"
    _run(
        run_market_recorder(
            _default_config(restart_backoff_max_s=5.0), capture_root=cap,
            liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=fetch,
            liquidation_runner=_flaky_runner, now_fn=clock.now, sleep=clock.sleep,
        )
    )
    assert len(calls) == 3
    assert clock.sleeps
    assert max(clock.sleeps) <= 5.0


def test_reference_captured_once_per_day_after_cutoff(tmp_path: Path) -> None:
    """Reference payloads are captured once per day, only after the cutoff."""
    flag = _Flag()
    t002 = pd.Timestamp("2026-09-22T00:02:00Z")
    t006 = pd.Timestamp("2026-09-22T00:06:00Z")
    t1300 = pd.Timestamp("2026-09-22T13:00:00Z")
    state = {"n": 0}
    ref_fetches: list[tuple[int, str]] = []

    def _now() -> pd.Timestamp:
        n = state["n"]
        state["n"] += 1
        if n < 400:
            return t002
        if n < 900:
            return t006
        return t1300

    async def _sleep(delay: float) -> None:
        await asyncio.sleep(0)
        if state["n"] > 4000:
            flag.requested = True

    async def _fetch(url: str) -> bytes:
        if "bookTicker" in url:
            return json.dumps([dict(_BOOK_ROW)]).encode()
        if "premiumIndex" in url:
            return json.dumps([dict(_PREMIUM_ROW)]).encode()
        ref_fetches.append((state["n"], url))
        if state["n"] >= 900:
            return b'{"symbols": [{"symbol": "LATE"}]}'
        return _REF_BYTES

    cap = tmp_path / "cap"
    _run(
        run_market_recorder(
            _default_config(), capture_root=cap, liquidations_dir=tmp_path / "liq",
            shutdown=flag, fetch=_fetch,
            liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
            now_fn=_now, sleep=_sleep,
        )
    )
    assert ref_fetches
    assert all(n >= 400 for n, _ in ref_fetches)
    per_url: dict[str, int] = {}
    for _, url in ref_fetches:
        per_url[url] = per_url.get(url, 0) + 1
    assert sorted(per_url.values()) == [1] * len(REFERENCE_URLS)
    for name in REFERENCE_URLS:
        raw = gzip.decompress((cap / "reference" / name / "20260922.json.gz").read_bytes())
        assert raw == _REF_BYTES


def test_reference_failure_retried_same_day(tmp_path: Path) -> None:
    """A failed reference payload is retried until the day is captured."""
    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T06:00:00Z"))
    attempts = {"n": 0}

    async def _fetch(url: str) -> bytes:
        if "bookTicker" in url:
            return json.dumps([dict(_BOOK_ROW)]).encode()
        if "premiumIndex" in url:
            return json.dumps([dict(_PREMIUM_ROW)]).encode()
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("ref down")
        if (tmp_path / "cap" / "reference" / "exchange_info" / "20260922.json.gz").exists():
            flag.requested = True
        return _REF_BYTES

    _run(
        run_market_recorder(
            _default_config(), capture_root=tmp_path / "cap",
            liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=_fetch,
            liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
            now_fn=clock.now, sleep=clock.sleep,
        )
    )
    assert attempts["n"] > len(REFERENCE_URLS)
    assert (tmp_path / "cap" / "reference" / "exchange_info" / "20260922.json.gz").exists()


def test_shutdown_flushes_buffers_and_heartbeat(tmp_path: Path) -> None:
    """Shutdown persists mid-hour buffers and updates the heartbeat."""
    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:30:20Z"))
    book_calls: list[int] = []
    fetch = _fetch_router(flag, clock, book_calls=book_calls, stop_after_book=1)
    cap = tmp_path / "cap"
    _run(
        run_market_recorder(
            _default_config(), capture_root=cap, liquidations_dir=tmp_path / "liq",
            shutdown=flag, fetch=fetch,
            liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
            now_fn=clock.now, sleep=clock.sleep,
        )
    )
    out = load_snapshot_dataset(
        cap, "book_ticker",
        start=pd.Timestamp("2026-09-22T10:30:00Z"), end=pd.Timestamp("2026-09-22T10:32:00Z"),
    )
    assert len(out) == 1
    heartbeat = json.loads((cap / "recorder_heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["book_ticker"]["last_success_at"] == "2026-09-22T10:31:00+00:00"
    assert heartbeat["book_ticker"]["rows_last_flush"] == 1


def test_invalid_config_rejected() -> None:
    """Grid, flush, cutoff, and backoff values are validated."""
    with pytest.raises(ValueError, match="divisor"):
        MarketRecorderConfig(book_ticker_interval_s=7)
    with pytest.raises(ValueError, match="divisor"):
        MarketRecorderConfig(premium_index_interval_s=7)
    with pytest.raises(ValueError, match="positive"):
        MarketRecorderConfig(flush_interval_s=0)
    with pytest.raises(ValueError, match="HH:MM"):
        MarketRecorderConfig(reference_capture_after_utc="nope")
    with pytest.raises(ValueError, match=">="):
        MarketRecorderConfig(restart_backoff_max_s=0.5)


def test_buffer_overflow_drops_oldest_and_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unflushable buffers stay bounded and the drop is logged."""
    import src.market_data.streams.recorder as recorder_mod

    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []

    async def _fetch(url: str) -> bytes:
        if "bookTicker" in url:
            book_calls.append(1)
            if len(book_calls) >= 9:
                flag.requested = True
            return json.dumps([dict(_BOOK_ROW)]).encode()
        if "premiumIndex" in url:
            return json.dumps([dict(_PREMIUM_ROW)]).encode()
        return _REF_BYTES

    def _boom(frame: Any, root: Any, dataset: str) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(recorder_mod, "write_hourly_partition", _boom)
    with caplog.at_level(logging.ERROR, logger="src.market_data.streams.recorder"):
        _run(
            run_market_recorder(
                MarketRecorderConfig(
                    book_ticker_interval_s=60, premium_index_interval_s=3600,
                    flush_interval_s=120.0, liquidation_flush_interval_s=60.0,
                ),
                capture_root=tmp_path / "cap", liquidations_dir=tmp_path / "liq",
                shutdown=flag, fetch=_fetch,
                liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
                now_fn=clock.now, sleep=clock.sleep,
            )
        )
    assert any("BUFFER_OVERFLOW" in record.message for record in caplog.records)


def test_default_session_fetch_and_close(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The default aiohttp session fetches and is closed on shutdown."""
    import aiohttp

    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []
    closed = {"n": 0}

    class _FakeResponse:
        def __init__(self, url: str) -> None:
            self._url = url

        async def __aenter__(self) -> _FakeResponse:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        def raise_for_status(self) -> None:
            return None

        async def read(self) -> bytes:
            if "bookTicker" in self._url:
                book_calls.append(1)
                if len(book_calls) >= 1:
                    flag.requested = True
                return json.dumps([dict(_BOOK_ROW)]).encode()
            if "premiumIndex" in self._url:
                return json.dumps([dict(_PREMIUM_ROW)]).encode()
            return _REF_BYTES

    class _FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            return None

        def get(self, url: str) -> _FakeResponse:
            return _FakeResponse(url)

        async def close(self) -> None:
            closed["n"] += 1

    monkeypatch.setattr(aiohttp, "ClientSession", _FakeSession)
    _run(
        run_market_recorder(
            _default_config(), capture_root=tmp_path / "cap",
            liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=None,
            liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
            now_fn=clock.now, sleep=clock.sleep,
        )
    )
    assert closed["n"] == 1
    assert book_calls


def test_liquidation_tracker_flush_failure_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Coverage flush failures inside the recorder never stop capture."""
    from src.market_data.streams.coverage import CoverageTracker

    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []
    fetch = _fetch_router(flag, clock, book_calls=book_calls, stop_after_book=1)

    async def _quick_liq(**kwargs: Any) -> None:
        return None

    def _boom(self: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(CoverageTracker, "flush", _boom)
    _run(
        run_market_recorder(
            _default_config(), capture_root=tmp_path / "cap",
            liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=fetch,
            liquidation_runner=_quick_liq, now_fn=clock.now, sleep=clock.sleep,
        )
    )
    assert book_calls


def test_bound_frames_helper() -> None:
    """Oldest rows are dropped beyond the bound."""
    import src.market_data.streams.recorder as recorder_mod

    frames = [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3]})]
    kept, dropped = recorder_mod._bound_frames(frames, 10)
    assert dropped == 0
    kept, dropped = recorder_mod._bound_frames(frames, 2)
    assert dropped == 1
    assert kept[0]["a"].tolist() == [2, 3]


def test_cutoff_helper_rejects_bad_spec() -> None:
    """Malformed cutoff specs fail closed."""
    import src.market_data.streams.recorder as recorder_mod

    with pytest.raises(ValueError, match="HH:MM"):
        recorder_mod._cutoff_for_day(pd.Timestamp("2026-09-22T10:00:00Z"), "nope")


def test_utc_now_returns_utc_wall_clock() -> None:
    """The default wall clock is tz-aware UTC."""
    import src.market_data.streams.recorder as recorder_mod

    ts = recorder_mod._utc_now()
    assert ts.tzinfo is not None
    assert (pd.Timestamp.now(tz="UTC") - ts).total_seconds() < 60.0


def test_liquidation_runner_receives_timing_config(tmp_path: Path) -> None:
    """The recorder forwards its liquidation timing config to the stream runner."""
    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []
    fetch = _fetch_router(flag, clock, book_calls=book_calls, stop_after_book=1)
    seen: dict[str, Any] = {}

    async def _capturing(**kwargs: Any) -> None:
        seen.update(kwargs)
        await _quiet_liquidations(flag, **kwargs)

    _run(
        run_market_recorder(
            _default_config(
                liquidation_receive_timeout_s=0.5,
                liquidation_liveness_timeout_s=20.0,
                liquidation_ping_interval_s=2.0,
                liquidation_event_stall_timeout_s=300.0,
            ),
            capture_root=tmp_path / "cap", liquidations_dir=tmp_path / "liq",
            shutdown=flag, fetch=fetch, liquidation_runner=_capturing,
            now_fn=clock.now, sleep=clock.sleep,
        )
    )
    assert seen["receive_timeout_s"] == 0.5
    assert seen["liveness_timeout_s"] == 20.0
    assert seen["ping_interval_s"] == 2.0
    assert seen["event_stall_timeout_s"] == 300.0


def test_liquidation_timing_config_rejected() -> None:
    """Non-positive receive/ping intervals and liveness <= ping interval fail closed."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MarketRecorderConfig(liquidation_receive_timeout_s=0.0)
    with pytest.raises(ValidationError):
        MarketRecorderConfig(liquidation_ping_interval_s=-1.0)
    with pytest.raises(ValidationError):
        MarketRecorderConfig(liquidation_liveness_timeout_s=5.0, liquidation_ping_interval_s=5.0)
    with pytest.raises(ValidationError):
        MarketRecorderConfig(liquidation_liveness_timeout_s=1.0)
    with pytest.raises(ValidationError):
        MarketRecorderConfig(liquidation_event_stall_timeout_s=15.0)
