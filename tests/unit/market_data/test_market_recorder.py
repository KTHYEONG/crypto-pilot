"""Invariant guards for the always-on live-only market recorder (no network)."""

from __future__ import annotations

import asyncio
import gzip
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.market_data.streams.recorder import MarketRecorderConfig, RateLimitedError, run_market_recorder
from src.market_data.streams.snapshots import (
    REFERENCE_URLS,
    load_snapshot_dataset,
    parse_book_ticker_payload,
    parse_premium_index_payload,
)

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


class _ManualClock:
    """Isolated scripted clock for driving one sampler with exact timing."""

    def __init__(self, start: pd.Timestamp) -> None:
        self.t = start

    def now(self) -> pd.Timestamp:
        return self.t

    async def sleep(self, delay: float) -> None:
        self.t += pd.Timedelta(seconds=float(delay))
        await asyncio.sleep(0)


class _BonusSleep:
    """Exact sleep with a one-time bonus on the first call (to stage a late wake)."""

    def __init__(self, clock: _ManualClock, bonus_s: float) -> None:
        self._clock = clock
        self._bonus = float(bonus_s)

    async def __call__(self, delay: float) -> None:
        extra, self._bonus = self._bonus, 0.0
        self._clock.t += pd.Timedelta(seconds=float(delay) + extra)
        await asyncio.sleep(0)


def _stop_after_elapsed(flag: _Flag, clock: Any, start: pd.Timestamp, seconds: float) -> Any:
    """Sleep wrapper requesting shutdown once the fake clock has advanced past a budget.

    Recorder-level sampler timing drifts under the shared fake clock, so tests that are
    not about sampling terminate on elapsed fake time instead of fetch counts.
    """

    async def _sleep(delay: float) -> None:
        if clock.now() - start >= pd.Timedelta(seconds=seconds):
            flag.requested = True
        await clock.sleep(delay)

    return _sleep


def _drive_sampler(
    *,
    dataset: str = "book_ticker",
    interval_s: int = 60,
    config: Any | None = None,
    root: Path,
    fetch: Any,
    clock: _ManualClock,
    flag: _Flag,
    sleep: Any | None = None,
    gate: Any | None = None,
    parse_fn: Any | None = None,
) -> Any:
    """Build a single grid sampler wired to scripted time (deterministic, no cross-task drift)."""
    import src.market_data.streams.recorder as recorder_mod

    Path(root).mkdir(parents=True, exist_ok=True)
    return recorder_mod._GridSampler(
        dataset=dataset,
        url="http://test.local/bookTicker",
        interval_s=interval_s,
        parse_fn=parse_fn or parse_book_ticker_payload,
        config=config or _default_config(),
        capture_root=root,
        heartbeat=recorder_mod._Heartbeat(started_at=clock.t),
        fetch=fetch,
        now_fn=clock.now,
        sleep=sleep or clock.sleep,
        shutdown=flag,
        rate_gate=gate or recorder_mod._RateLimitGate(cooldown_s=60.0),
    )


def _book_fetch(
    flag: _Flag,
    clock: _ManualClock,
    calls: list[pd.Timestamp],
    *,
    fail_until: pd.Timestamp | None = None,
    fail_first: int = 0,
    stop_after: int | None = None,
    rate_limited: Any | None = None,
) -> Any:
    """Scripted book-ticker fetch recording attempt starts in fake time."""
    state = {"n": 0}

    async def _fetch(url: str) -> bytes:
        state["n"] += 1
        calls.append(clock.now())
        if rate_limited is not None:
            raise rate_limited
        if state["n"] <= fail_first or (fail_until is not None and clock.now() < fail_until):
            raise RuntimeError("fetch down")
        if stop_after is not None and state["n"] >= stop_after:
            flag.requested = True
        return json.dumps([dict(_BOOK_ROW)]).encode()

    return _fetch


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
    """Samples carry the grid instant they were issued for, with receipt times attached."""
    flag = _Flag()
    clock = _ManualClock(pd.Timestamp("2026-09-22T10:00:20Z"))
    starts: list[pd.Timestamp] = []
    fetch = _book_fetch(flag, clock, starts, stop_after=2)
    cap = tmp_path / "cap"

    async def _scenario() -> None:
        await _drive_sampler(root=cap, fetch=fetch, clock=clock, flag=flag).run()

    _run(_scenario())
    out = load_snapshot_dataset(
        cap, "book_ticker",
        start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T10:03:00Z"),
    )
    assert sorted(out["captured_at"].unique()) == [
        pd.Timestamp("2026-09-22T10:01:00Z"), pd.Timestamp("2026-09-22T10:02:00Z")
    ]
    captured_ms = out["captured_at"].astype("int64") // 1_000_000
    assert (out["fetched_at_ms"].astype("int64") >= captured_ms).all()
    assert starts == [pd.Timestamp("2026-09-22T10:01:00Z"), pd.Timestamp("2026-09-22T10:02:00Z")]


def test_persistent_failure_is_a_gap_not_a_shift(tmp_path: Path) -> None:
    """Attempts exhausted inside the slot leave a gap; no attempt starts past the lag bound."""
    flag = _Flag()
    clock = _ManualClock(pd.Timestamp("2026-09-22T10:00:20Z"))
    starts: list[pd.Timestamp] = []
    fetch = _book_fetch(flag, clock, starts, fail_first=6, stop_after=7)
    cap = tmp_path / "cap"

    async def _scenario() -> None:
        await _drive_sampler(root=cap, fetch=fetch, clock=clock, flag=flag).run()

    _run(_scenario())
    out = load_snapshot_dataset(
        cap, "book_ticker",
        start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T10:03:00Z"),
    )
    assert sorted(out["captured_at"].unique()) == [pd.Timestamp("2026-09-22T10:02:00Z")]
    slot_starts = [s for s in starts if s < pd.Timestamp("2026-09-22T10:02:00Z")]
    assert len(slot_starts) == 6
    assert max(slot_starts) <= pd.Timestamp("2026-09-22T10:01:05Z")
    heartbeat = json.loads((cap / "recorder_heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["book_ticker"]["consecutive_failures"] == 0


def test_late_wake_samples_only_current_grid_point(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A wake past several grids samples only the current point and counts the skips."""
    flag = _Flag()
    clock = _ManualClock(pd.Timestamp("2026-09-22T10:00:20Z"))
    starts: list[pd.Timestamp] = []
    fetch = _book_fetch(flag, clock, starts, stop_after=1)
    cap = tmp_path / "cap"

    async def _scenario() -> None:
        sampler = _drive_sampler(
            root=cap, fetch=fetch, clock=clock, flag=flag,
            sleep=_BonusSleep(clock, bonus_s=180.0),
        )
        await sampler.run()

    with caplog.at_level(logging.WARNING, logger="src.market_data.streams.recorder"):
        _run(_scenario())
    out = load_snapshot_dataset(
        cap, "book_ticker",
        start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T10:05:00Z"),
    )
    assert sorted(out["captured_at"].unique()) == [pd.Timestamp("2026-09-22T10:04:00Z")]
    assert starts == [pd.Timestamp("2026-09-22T10:04:00Z")]
    heartbeat = json.loads((cap / "recorder_heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["book_ticker"]["skipped_grid_points"] == 3
    skipped = [r for r in caplog.records if "GRID_SKIPPED" in r.message]
    assert len(skipped) == 1
    assert "count=3" in skipped[0].message
    assert "reason=late" in skipped[0].message


def test_start_lag_beyond_bound_skips_point(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A wake past the start-lag bound skips the point and continues at the next grid."""
    flag = _Flag()
    clock = _ManualClock(pd.Timestamp("2026-09-22T10:00:20Z"))
    starts: list[pd.Timestamp] = []
    fetch = _book_fetch(flag, clock, starts, stop_after=1)
    cap = tmp_path / "cap"

    async def _scenario() -> None:
        sampler = _drive_sampler(
            root=cap, fetch=fetch, clock=clock, flag=flag,
            sleep=_BonusSleep(clock, bonus_s=7.0),
            config=_default_config(grid_max_start_lag_s=5.0),
        )
        await sampler.run()

    with caplog.at_level(logging.WARNING, logger="src.market_data.streams.recorder"):
        _run(_scenario())
    out = load_snapshot_dataset(
        cap, "book_ticker",
        start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T10:03:00Z"),
    )
    assert sorted(out["captured_at"].unique()) == [pd.Timestamp("2026-09-22T10:02:00Z")]
    assert starts == [pd.Timestamp("2026-09-22T10:02:00Z")]
    heartbeat = json.loads((cap / "recorder_heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["book_ticker"]["skipped_grid_points"] == 1
    skipped = [r for r in caplog.records if "GRID_SKIPPED" in r.message]
    assert len(skipped) == 1
    assert "count=1" in skipped[0].message
    assert "reason=lag" in skipped[0].message


def test_transient_failure_retried_in_same_slot(tmp_path: Path) -> None:
    """One failed attempt is retried inside the slot; the slot still succeeds cleanly."""
    flag = _Flag()
    clock = _ManualClock(pd.Timestamp("2026-09-22T10:00:20Z"))
    starts: list[pd.Timestamp] = []
    fetch = _book_fetch(flag, clock, starts, fail_first=1, stop_after=2)
    cap = tmp_path / "cap"

    async def _scenario() -> None:
        await _drive_sampler(root=cap, fetch=fetch, clock=clock, flag=flag).run()

    _run(_scenario())
    out = load_snapshot_dataset(
        cap, "book_ticker",
        start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T10:03:00Z"),
    )
    assert sorted(out["captured_at"].unique()) == [pd.Timestamp("2026-09-22T10:01:00Z")]
    assert starts == [
        pd.Timestamp("2026-09-22T10:01:00Z"),
        pd.Timestamp("2026-09-22T10:01:01Z"),
    ]
    row = out.iloc[0]
    assert row["fetched_at_ms"] == int(pd.Timestamp("2026-09-22T10:01:01Z").value // 1_000_000)
    heartbeat = json.loads((cap / "recorder_heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["book_ticker"]["consecutive_failures"] == 0


def test_shutdown_during_retry_abandons_slot_quietly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Shutdown while waiting to retry ends the slot without failure accounting."""
    flag = _Flag()
    clock = _ManualClock(pd.Timestamp("2026-09-22T10:00:20Z"))
    calls = {"n": 0}

    async def _fetch(url: str) -> bytes:
        calls["n"] += 1
        raise RuntimeError("fetch down")

    async def _sleep(delay: float) -> None:
        if calls["n"] >= 1:
            flag.requested = True
        clock.t += pd.Timedelta(seconds=float(delay))
        await asyncio.sleep(0)

    cap = tmp_path / "cap"

    async def _scenario() -> None:
        await _drive_sampler(root=cap, fetch=_fetch, clock=clock, flag=flag, sleep=_sleep).run()

    with caplog.at_level(logging.WARNING, logger="src.market_data.streams.recorder"):
        _run(_scenario())
    assert calls["n"] == 1
    out = load_snapshot_dataset(
        cap, "book_ticker",
        start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T10:03:00Z"),
    )
    assert out.empty
    assert not any("status=FAILED" in r.message for r in caplog.records)
    heartbeat = json.loads((cap / "recorder_heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["book_ticker"]["consecutive_failures"] == 0


def test_rate_limit_gate_block_cooldown_and_expiry() -> None:
    """The embargo spans max(Retry-After, cooldown), never shortens, and expires."""
    import src.market_data.streams.recorder as recorder_mod

    t0 = pd.Timestamp("2026-09-22T10:01:00Z")
    gate = recorder_mod._RateLimitGate(cooldown_s=60.0)
    assert gate.blocked_until(t0) is None
    until = gate.block(t0, 120.0)
    assert until == t0 + pd.Timedelta(seconds=120)
    assert gate.blocked_until(t0) == until
    gate.block(t0 + pd.Timedelta(seconds=1), 5.0)
    assert gate.blocked_until(t0 + pd.Timedelta(seconds=1)) == until
    assert gate.blocked_until(until) is None
    fresh = recorder_mod._RateLimitGate(cooldown_s=60.0)
    assert fresh.block(t0, None) == t0 + pd.Timedelta(seconds=60)


def test_rate_limit_cooldown_floor_without_retry_after(tmp_path: Path) -> None:
    """Without Retry-After the next fetch waits out the configured cooldown."""
    import src.market_data.streams.recorder as recorder_mod

    flag = _Flag()
    clock = _ManualClock(pd.Timestamp("2026-09-22T10:00:20Z"))
    starts: list[pd.Timestamp] = []
    state = {"n": 0}

    async def _fetch(url: str) -> bytes:
        state["n"] += 1
        starts.append(clock.now())
        if state["n"] == 1:
            raise recorder_mod.RateLimitedError(418, None)
        flag.requested = True
        return json.dumps([dict(_BOOK_ROW)]).encode()

    cap = tmp_path / "cap"

    async def _scenario() -> None:
        await _drive_sampler(root=cap, fetch=_fetch, clock=clock, flag=flag).run()

    _run(_scenario())
    assert starts == [
        pd.Timestamp("2026-09-22T10:01:00Z"),
        pd.Timestamp("2026-09-22T10:02:00Z"),
    ]


def test_rate_limit_embargo_shared_by_both_samplers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A 429 on one sampler blocks the other through the process-wide gate."""
    import src.market_data.streams.recorder as recorder_mod

    clock = _ManualClock(pd.Timestamp("2026-09-22T10:00:20Z"))
    gate = recorder_mod._RateLimitGate(cooldown_s=60.0)
    book_starts: list[pd.Timestamp] = []
    premium_starts: list[pd.Timestamp] = []

    async def _book_fetch_429(url: str) -> bytes:
        book_starts.append(clock.now())
        if len(book_starts) >= 2:
            book_flag.requested = True
        raise recorder_mod.RateLimitedError(429, 120.0)

    async def _premium_fetch_ok(url: str) -> bytes:
        premium_starts.append(clock.now())
        if premium_starts:
            premium_flag.requested = True
        return json.dumps([dict(_PREMIUM_ROW)]).encode()

    book_flag = _Flag()
    cap = tmp_path / "cap"

    async def _book_phase() -> None:
        await _drive_sampler(
            root=cap, fetch=_book_fetch_429, clock=clock, flag=book_flag, gate=gate,
            config=_default_config(flush_interval_s=3600.0),
        ).run()

    with caplog.at_level(logging.WARNING, logger="src.market_data.streams.recorder"):
        _run(_book_phase())
    assert book_starts == [
        pd.Timestamp("2026-09-22T10:01:00Z"),
        pd.Timestamp("2026-09-22T10:03:00Z"),
    ]
    assert any("status=RATE_LIMITED" in r.message for r in caplog.records)
    assert any("http_status=429" in r.message for r in caplog.records)
    heartbeat = json.loads((cap / "recorder_heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["book_ticker"]["skipped_grid_points"] == 1

    premium_flag = _Flag()

    async def _premium_phase() -> None:
        await _drive_sampler(
            dataset="premium_index", root=cap, fetch=_premium_fetch_ok, clock=clock,
            flag=premium_flag, gate=gate, config=_default_config(flush_interval_s=3600.0),
            parse_fn=parse_premium_index_payload,
        ).run()

    _run(_premium_phase())
    assert premium_starts == [pd.Timestamp("2026-09-22T10:05:00Z")]
    heartbeat = json.loads((cap / "recorder_heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["premium_index"]["skipped_grid_points"] == 1


def test_rate_limit_embargo_spacing_between_fetches(tmp_path: Path) -> None:
    """Every fetch after a 429 waits out the Retry-After embargo, never the grid."""
    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    starts: list[pd.Timestamp] = []

    async def _fetch(url: str) -> bytes:
        if "bookTicker" in url:
            starts.append(clock.now())
            if clock.now() - pd.Timestamp("2026-09-22T10:00:20Z") >= pd.Timedelta(minutes=6):
                flag.requested = True
            raise RateLimitedError(429, 120.0)
        if "premiumIndex" in url:
            return json.dumps([dict(_PREMIUM_ROW)]).encode()
        return _REF_BYTES

    cap = tmp_path / "cap"
    _run(
        run_market_recorder(
            _default_config(), capture_root=cap,
            liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=_fetch,
            liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
            now_fn=clock.now, sleep=clock.sleep,
        )
    )
    assert len(starts) >= 2
    for first, second in itertools.pairwise(starts):
        assert (second - first).total_seconds() >= 120.0
    heartbeat = json.loads((cap / "recorder_heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["book_ticker"]["skipped_grid_points"] >= 1


def test_default_fetch_maps_429_with_retry_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The default session fetch surfaces 429 + Retry-After as RateLimitedError."""
    import aiohttp

    import src.market_data.streams.recorder as recorder_mod

    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    blocked: list[tuple[Any, float | None]] = []
    real_block = recorder_mod._RateLimitGate.block
    def _recording_block(
        self: Any, now: pd.Timestamp, retry_after_s: float | None
    ) -> pd.Timestamp:
        blocked.append((now, retry_after_s))
        return real_block(self, now, retry_after_s)

    monkeypatch.setattr(recorder_mod._RateLimitGate, "block", _recording_block)

    book_hits = {"n": 0}

    class _StatusResponse:
        def __init__(self, url: str) -> None:
            self._url = url
            self.status = 429 if "bookTicker" in url else 200
            if "bookTicker" in url:
                book_hits["n"] += 1
                retry = "7" if book_hits["n"] == 1 else "soon"
                self.headers = {"Retry-After": retry}
            else:
                self.headers = {}

        async def __aenter__(self) -> _StatusResponse:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        def raise_for_status(self) -> None:
            return None

        async def read(self) -> bytes:
            if "premiumIndex" in self._url:
                return json.dumps([dict(_PREMIUM_ROW)]).encode()
            return _REF_BYTES

    class _StatusSession:
        def __init__(self, **kwargs: Any) -> None:
            return None

        def get(self, url: str) -> _StatusResponse:
            return _StatusResponse(url)

        async def close(self) -> None:
            return None

    monkeypatch.setattr(aiohttp, "ClientSession", _StatusSession)
    start = pd.Timestamp("2026-09-22T10:00:20Z")
    with caplog.at_level(logging.WARNING, logger="src.market_data.streams.recorder"):
        _run(
            run_market_recorder(
                _default_config(), capture_root=tmp_path / "cap",
                liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=None,
                liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
                now_fn=clock.now, sleep=_stop_after_elapsed(flag, clock, start, 600.0),
            )
        )
    assert blocked
    assert blocked[0][1] == 7.0
    assert any(retry is None for _, retry in blocked[1:])
    assert any("http_status=429" in r.message for r in caplog.records)


def test_grid_lag_and_retry_config_validated() -> None:
    """Lag/retry/cooldown cadences must nest inside the sampler grids."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="grid_retry_delay_s"):
        MarketRecorderConfig(grid_retry_delay_s=5.0, grid_max_start_lag_s=5.0)
    with pytest.raises(ValidationError, match="grid_retry_delay_s"):
        MarketRecorderConfig(grid_retry_delay_s=0)
    with pytest.raises(ValidationError, match="grid_max_start_lag_s"):
        MarketRecorderConfig(grid_max_start_lag_s=60.0)
    with pytest.raises(ValidationError, match="rate_limit_cooldown_s"):
        MarketRecorderConfig(rate_limit_cooldown_s=0)


def test_hour_rollover_flushes_closed_hour(tmp_path: Path) -> None:
    """The closed hour is persisted as soon as the next hour is sampled."""
    flag = _Flag()
    clock = _ManualClock(pd.Timestamp("2026-09-22T10:58:20Z"))
    calls = {"n": 0}
    cap = tmp_path / "cap"
    seen: dict[str, bool] = {}

    async def _fetch(url: str) -> bytes:
        calls["n"] += 1
        if calls["n"] == 3:
            seen["ten"] = (cap / "book_ticker" / "20260922" / "10.parquet").exists()
            flag.requested = True
        return json.dumps([dict(_BOOK_ROW)]).encode()

    async def _scenario() -> None:
        await _drive_sampler(
            root=cap, fetch=_fetch, clock=clock, flag=flag,
            config=_default_config(flush_interval_s=3600.0),
        ).run()

    _run(_scenario())
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
        start=pd.Timestamp("2026-09-22T10:00:00Z"), end=pd.Timestamp("2026-09-22T10:20:00Z"),
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
    clock = _ManualClock(pd.Timestamp("2026-09-22T10:30:20Z"))
    calls = {"n": 0}
    cap = tmp_path / "cap"

    async def _fetch(url: str) -> bytes:
        calls["n"] += 1
        if calls["n"] >= 1:
            flag.requested = True
        return json.dumps([dict(_BOOK_ROW)]).encode()

    async def _scenario() -> None:
        await _drive_sampler(root=cap, fetch=_fetch, clock=clock, flag=flag).run()

    _run(_scenario())
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
            self.status = 200
            self.headers: dict[str, str] = {}

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
    cap = tmp_path / "cap"
    start = pd.Timestamp("2026-09-22T10:00:20Z")
    _run(
        run_market_recorder(
            _default_config(), capture_root=cap,
            liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=None,
            liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
            now_fn=clock.now, sleep=_stop_after_elapsed(flag, clock, start, 120.0),
        )
    )
    assert closed["n"] == 1
    assert (cap / "reference" / "exchange_info" / "20260922.json.gz").exists()


def test_liquidation_tracker_flush_failure_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Coverage flush failures inside the recorder never stop capture."""
    from src.market_data.streams.coverage import CoverageTracker

    flag = _Flag()
    start = pd.Timestamp("2026-09-22T10:00:20Z")
    clock = _Clock(start)
    book_calls: list[int] = []
    fetch = _fetch_router(flag, clock, book_calls=book_calls)

    async def _quick_liq(**kwargs: Any) -> None:
        return None

    def _boom(self: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(CoverageTracker, "flush", _boom)
    cap = tmp_path / "cap"
    _run(
        run_market_recorder(
            _default_config(), capture_root=cap,
            liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=fetch,
            liquidation_runner=_quick_liq, now_fn=clock.now,
            sleep=_stop_after_elapsed(flag, clock, start, 120.0),
        )
    )
    assert (cap / "reference" / "exchange_info" / "20260922.json.gz").exists()


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
    from src.market_data.streams.liquidations import LiquidationHealth

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
    assert isinstance(seen["health"], LiquidationHealth)


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


def test_unexpected_exit_restarts_task(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A normal return without shutdown is an error and restarts the task."""
    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []
    fetch = _fetch_router(flag, clock, book_calls=book_calls)
    calls = {"n": 0}

    async def _quitting_runner(**kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] >= 3:
            flag.requested = True

    with caplog.at_level(logging.ERROR, logger="src.market_data.streams.recorder"):
        _run(
            run_market_recorder(
                _default_config(), capture_root=tmp_path / "cap",
                liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=fetch,
                liquidation_runner=_quitting_runner, now_fn=clock.now, sleep=clock.sleep,
            )
        )
    assert calls["n"] == 3
    assert any("status=UNEXPECTED_EXIT" in record.message for record in caplog.records)


def test_return_after_shutdown_is_final(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A normal return once shutdown is requested ends the task without restart or error."""
    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []
    fetch = _fetch_router(flag, clock, book_calls=book_calls, stop_after_book=1)
    seen = {"n": 0}

    async def _counting(**kwargs: Any) -> None:
        seen["n"] += 1
        await _quiet_liquidations(flag, **kwargs)

    with caplog.at_level(logging.ERROR, logger="src.market_data.streams.recorder"):
        _run(
            run_market_recorder(
                _default_config(), capture_root=tmp_path / "cap",
                liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=fetch,
                liquidation_runner=_counting, now_fn=clock.now, sleep=clock.sleep,
            )
        )
    assert seen["n"] == 1
    assert not any("UNEXPECTED_EXIT" in record.message for record in caplog.records)


def test_backoff_resets_after_long_healthy_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run lasting at least the backoff cap resets the next restart delay to 1 s."""
    import src.market_data.streams.recorder as recorder_mod

    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []
    fetch = _fetch_router(flag, clock, book_calls=book_calls)
    calls = {"n": 0}
    starts: list[pd.Timestamp] = []

    async def _runner(**kwargs: Any) -> None:
        calls["n"] += 1
        starts.append(clock.now())
        if calls["n"] <= 2:
            raise RuntimeError("boom")
        if calls["n"] == 3:
            clock.t += pd.Timedelta(minutes=10)
            raise RuntimeError("boom")
        flag.requested = True

    delays: list[float] = []
    real_sleep_capped = recorder_mod._sleep_capped

    async def _recording_sleep_capped(
        sleep_fn: Any, delay: float, shutdown: Any, step: float = 1.0
    ) -> None:
        delays.append(float(delay))
        await real_sleep_capped(sleep_fn, delay, shutdown, step)

    monkeypatch.setattr(recorder_mod, "_sleep_capped", _recording_sleep_capped)
    _run(
        run_market_recorder(
            _default_config(restart_backoff_max_s=5.0), capture_root=tmp_path / "cap",
            liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=fetch,
            liquidation_runner=_runner, now_fn=clock.now, sleep=clock.sleep,
        )
    )
    assert calls["n"] == 4
    supervisor_delays = [d for d in delays if d <= 5.0 and d in (1.0, 2.0, 4.0)]
    assert supervisor_delays[:3] == [1.0, 2.0, 1.0]


def test_same_health_object_across_restarts(tmp_path: Path) -> None:
    """Restarts reuse the single process-wide health record instead of resetting it."""
    from src.market_data.streams.liquidations import LiquidationHealth

    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []
    fetch = _fetch_router(flag, clock, book_calls=book_calls)
    seen: list[Any] = []
    calls = {"n": 0}

    async def _runner(**kwargs: Any) -> None:
        calls["n"] += 1
        seen.append(kwargs["health"])
        if calls["n"] == 1:
            raise RuntimeError("boom")
        flag.requested = True

    _run(
        run_market_recorder(
            _default_config(), capture_root=tmp_path / "cap",
            liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=fetch,
            liquidation_runner=_runner, now_fn=clock.now, sleep=clock.sleep,
        )
    )
    assert calls["n"] == 2
    assert isinstance(seen[0], LiquidationHealth)
    assert seen[0] is seen[1]


def test_heartbeat_publishes_liquidation_entry(tmp_path: Path) -> None:
    """The heartbeat file carries started_at and the runner's liquidation facts."""
    flag = _Flag()
    clock = _Clock(pd.Timestamp("2026-09-22T10:00:20Z"))
    book_calls: list[int] = []
    fetch = _fetch_router(flag, clock, book_calls=book_calls, stop_after_book=3)
    event_at = pd.Timestamp("2026-09-22T10:01:00Z")

    async def _runner(**kwargs: Any) -> None:
        health = kwargs["health"]
        health.last_event_at = event_at
        health.consecutive_failed_connections = 3
        await _quiet_liquidations(flag, **kwargs)

    cap = tmp_path / "cap"
    _run(
        run_market_recorder(
            _default_config(heartbeat_interval_s=30.0), capture_root=cap,
            liquidations_dir=tmp_path / "liq", shutdown=flag, fetch=fetch,
            liquidation_runner=_runner, now_fn=clock.now, sleep=clock.sleep,
        )
    )
    heartbeat = json.loads((cap / "recorder_heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["started_at"] == "2026-09-22T10:00:20+00:00"
    assert heartbeat["liquidations"]["last_event_at"] == event_at.isoformat()
    assert heartbeat["liquidations"]["consecutive_failed_connections"] == 3
    assert heartbeat["liquidations"]["last_disconnect_reason"] is None


def test_heartbeat_ts_refreshes_without_sampler_flushes(tmp_path: Path) -> None:
    """The dedicated heartbeat task bounds ts staleness even when samplers never flush."""
    flag = _Flag()
    start = pd.Timestamp("2026-09-22T10:00:20Z")
    clock = _Clock(start)
    book_calls: list[int] = []

    async def _fetch(url: str) -> bytes:
        if "bookTicker" in url:
            book_calls.append(1)
            if clock.t - start >= pd.Timedelta(minutes=5):
                flag.requested = True
            return json.dumps([dict(_BOOK_ROW)]).encode()
        if "premiumIndex" in url:
            return json.dumps([dict(_PREMIUM_ROW)]).encode()
        return _REF_BYTES

    cap = tmp_path / "cap"
    _run(
        run_market_recorder(
            _default_config(flush_interval_s=3600.0, heartbeat_interval_s=60.0),
            capture_root=cap, liquidations_dir=tmp_path / "liq", shutdown=flag,
            fetch=_fetch, liquidation_runner=lambda **k: _quiet_liquidations(flag, **k),
            now_fn=clock.now, sleep=clock.sleep,
        )
    )
    assert clock.t - start >= pd.Timedelta(minutes=5)
    heartbeat = json.loads((cap / "recorder_heartbeat.json").read_text(encoding="utf-8"))
    ts = pd.Timestamp(heartbeat["ts"])
    assert (clock.t - ts).total_seconds() <= 60.0


def test_heartbeat_interval_rejected() -> None:
    """A non-positive heartbeat cadence fails closed naming the field."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="heartbeat_interval_s"):
        MarketRecorderConfig(heartbeat_interval_s=0)
    with pytest.raises(ValidationError, match="heartbeat_interval_s"):
        MarketRecorderConfig(heartbeat_interval_s=-5.0)
