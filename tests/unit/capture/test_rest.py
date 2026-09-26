"""Invariant guards for grid REST sampling and daily reference capture."""

from __future__ import annotations

import asyncio
import contextlib
import gzip
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from src.capture.config import CaptureConfig
from src.capture.journal import SegmentWriter, hot_segment_path, iter_complete_records
from src.capture.rest import (
    FetchResult,
    GridSampler,
    RateGate,
    ReferenceCapture,
    RestStatus,
    floor_grid,
)

BOOK = "book_ticker"


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


def _config(**overrides: Any) -> CaptureConfig:
    values: dict[str, Any] = {
        "book_ticker_interval_s": 60,
        "premium_index_interval_s": 300,
    }
    values.update(overrides)
    return CaptureConfig(**values)


def _sampler(
    tmp_path: Path,
    clock: FakeClock,
    fetch: Callable[[str], Awaitable[FetchResult]],
    stop: Callable[[], bool],
    stream: str = BOOK,
    url: str = "https://example.invalid/book",
    interval: int = 60,
    gate: RateGate | None = None,
    status: RestStatus | None = None,
    config: CaptureConfig | None = None,
) -> tuple[GridSampler, RestStatus]:
    resolved = status if status is not None else RestStatus()
    sampler = GridSampler(
        stream=stream,
        url=url,
        interval_s=interval,
        config=config if config is not None else _config(),
        writer=SegmentWriter(tmp_path, stream, "blue"),
        fetch=fetch,
        gate=gate if gate is not None else RateGate(60.0),
        status=resolved,
        clock_ns=clock,
        sleep=clock.sleep,
        shutdown=stop,
    )
    return sampler, resolved


def _rest_records(tmp_path: Path, stream: str, recv_ns: int) -> list[dict[str, Any]]:
    dest = hot_segment_path(tmp_path, stream, "blue", recv_ns)
    return [record for record, _ in iter_complete_records(dest, 0)]


def test_floor_grid_alignment() -> None:
    """Grid labels sit on epoch-aligned boundaries at or before the receive time."""
    assert floor_grid(_ns(2026, 9, 26, 10, 1, 1), 60) == _ns(2026, 9, 26, 10, 1, 0)
    assert floor_grid(_ns(2026, 9, 26, 10, 1, 0), 60) == _ns(2026, 9, 26, 10, 1, 0)


def test_one_outcome_per_grid_point(tmp_path: Path) -> None:
    """Spec 01: five grid points yield five rest records with consecutive 60 s labels."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 1))
    calls = 0
    stopped = False

    async def fetch(url: str) -> FetchResult:
        nonlocal calls, stopped
        calls += 1
        if calls >= 5:
            stopped = True
        return (200, '{"rows":[]}', {})

    sampler, _ = _sampler(tmp_path, clock, fetch, lambda: stopped)

    async def scenario() -> None:
        await sampler.run()

    asyncio.run(scenario())
    records = _rest_records(tmp_path, BOOK, _ns(2026, 9, 26, 10, 2, 0))
    grids = sorted(record["grid"] for record in records if record["kind"] == "rest")
    assert len(grids) == 5
    moments = [datetime.fromisoformat(item.replace("Z", "+00:00")) for item in grids]
    for first, second in pairwise(moments):
        assert (second - first).total_seconds() == 60.0
    assert all(record["body"] == '{"rows":[]}' for record in records if record["kind"] == "rest")


def test_error_status_keeps_body(tmp_path: Path) -> None:
    """Spec 01: an HTTP 503 with a body is a rest record; failures count up."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 1))
    stopped = False

    async def fetch(url: str) -> FetchResult:
        nonlocal stopped
        stopped = True
        return (503, "<html>busy</html>", {})

    sampler, status = _sampler(tmp_path, clock, fetch, lambda: stopped)

    async def scenario() -> None:
        await sampler.run()

    asyncio.run(scenario())
    records = _rest_records(tmp_path, BOOK, _ns(2026, 9, 26, 10, 1, 0))
    assert records[0]["kind"] == "rest"
    assert records[0]["status"] == 503
    assert records[0]["body"] == "<html>busy</html>"
    assert status.consecutive_failures == 1


def test_transport_failure_recorded(tmp_path: Path) -> None:
    """Spec 01: two timeouts inside the lag window yield one rest_error; the next grid proceeds."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 1))
    calls = 0
    stopped = False

    async def fetch(url: str) -> FetchResult:
        nonlocal calls, stopped
        calls += 1
        if calls <= 2:
            raise TimeoutError("boom")
        if calls >= 3:
            stopped = True
        return (200, "ok", {})

    sampler, _ = _sampler(tmp_path, clock, fetch, lambda: stopped)

    async def scenario() -> None:
        await sampler.run()

    asyncio.run(scenario())
    records = _rest_records(tmp_path, BOOK, _ns(2026, 9, 26, 10, 2, 0))
    errors = [item for item in records if item["kind"] == "rest_error"]
    oks = [item for item in records if item["kind"] == "rest"]
    assert len(errors) == 1
    assert errors[0]["status"] is None
    assert len(oks) == 1


def test_missed_grids_recorded_as_skipped(tmp_path: Path) -> None:
    """Spec 01: a three-interval pause writes two skipped errors and never samples late."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 1))
    calls = 0
    stopped = False

    async def fetch(url: str) -> FetchResult:
        nonlocal calls, stopped
        calls += 1
        if calls == 1:
            clock.now_ns += 180_000_000_000
        if calls >= 2:
            stopped = True
        return (200, "ok", {})

    sampler, _ = _sampler(tmp_path, clock, fetch, lambda: stopped)

    async def scenario() -> None:
        await sampler.run()

    asyncio.run(scenario())
    records = _rest_records(tmp_path, BOOK, _ns(2026, 9, 26, 10, 4, 0))
    errors = [item for item in records if item["kind"] == "rest_error"]
    oks = [item for item in records if item["kind"] == "rest"]
    assert len(errors) == 2
    assert all(str(item["error"]).startswith("skipped:") for item in errors)
    assert len(oks) == 2
    assert errors[0]["grid"] != oks[0]["grid"]


def test_rate_limit_embargo_shared(tmp_path: Path) -> None:
    """Spec 01: a 429 with Retry-After blocks the sibling sampler through the shared gate."""
    gate = RateGate(60.0)
    moment = _ns(2026, 9, 26, 10, 0, 1)
    clock = FakeClock(moment)
    book_stopped = False

    async def book_fetch(url: str) -> FetchResult:
        nonlocal book_stopped
        book_stopped = True
        return (429, "limited", {"Retry-After": "120"})

    book, _ = _sampler(tmp_path, clock, book_fetch, lambda: book_stopped, gate=gate)

    async def establish() -> None:
        await book.run()

    asyncio.run(establish())
    embargo_end = gate.blocked_until(moment / 1_000_000_000)
    assert embargo_end is not None

    premium_clock = FakeClock(moment + 1_000_000_000)
    call_times: list[int] = []
    premium_stopped = False

    async def premium_fetch(url: str) -> FetchResult:
        nonlocal premium_stopped
        call_times.append(premium_clock())
        if len(call_times) >= 1:
            premium_stopped = True
        return (200, "ok", {})

    premium, _ = _sampler(
        tmp_path,
        premium_clock,
        premium_fetch,
        lambda: premium_stopped,
        stream="premium_index",
        gate=gate,
    )

    async def run_premium() -> None:
        await premium.run()

    asyncio.run(run_premium())
    assert call_times
    assert min(call_times) / 1_000_000_000 >= embargo_end


def test_zero_price_payload_not_judged(tmp_path: Path) -> None:
    """Regression guard for the 09-25 incident: a zero bid price is stored verbatim."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 1))
    body = '[{"symbol":"BTCUSDT","bidPrice":"0.0","askPrice":"1.0"}]'
    stopped = False

    async def fetch(url: str) -> FetchResult:
        nonlocal stopped
        stopped = True
        return (200, body, {})

    sampler, _ = _sampler(tmp_path, clock, fetch, lambda: stopped)

    async def scenario() -> None:
        await sampler.run()

    asyncio.run(scenario())
    records = _rest_records(tmp_path, BOOK, _ns(2026, 9, 26, 10, 1, 0))
    assert records[0]["kind"] == "rest"
    assert records[0]["status"] == 200
    assert records[0]["body"] == body


def test_first_ok_timestamp(tmp_path: Path) -> None:
    """Spec 01: first_ok_at marks the first 2xx; the failure count resets."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 1))
    calls = 0
    stopped = False

    async def fetch(url: str) -> FetchResult:
        nonlocal calls, stopped
        calls += 1
        if calls == 1:
            return (500, "bad", {})
        stopped = True
        return (200, "ok", {})

    sampler, status = _sampler(tmp_path, clock, fetch, lambda: stopped)

    async def scenario() -> None:
        await sampler.run()

    asyncio.run(scenario())
    records = _rest_records(tmp_path, BOOK, _ns(2026, 9, 26, 10, 2, 0))
    oks = [item for item in records if item["kind"] == "rest" and item["status"] == 200]
    assert len(oks) == 1
    assert status.first_ok_at_ns == oks[0]["recv_ns"]
    assert status.consecutive_failures == 0


def _reference_setup(
    tmp_path: Path,
    clock: FakeClock,
    bodies: dict[str, str],
    fail_once: set[str] | None = None,
    stop: Callable[[], bool] | None = None,
) -> tuple[ReferenceCapture, dict[str, int]]:
    calls: dict[str, int] = {}
    wanted = fail_once or set()

    async def fetch(url: str) -> FetchResult:
        name = next(key for key, value in _config().reference_urls if value == url)
        calls[name] = calls.get(name, 0) + 1
        if name in wanted and calls[name] == 1:
            raise TimeoutError("flaky")
        return (200, bodies[name], {})

    capture = ReferenceCapture(
        config=_config(),
        capture_root=tmp_path,
        fetch=fetch,
        gate=RateGate(60.0),
        clock_ns=clock,
        sleep=clock.sleep,
        shutdown=stop if stop is not None else lambda: False,
    )
    return capture, calls


def test_reference_once_per_day_per_endpoint(tmp_path: Path) -> None:
    """Spec 01: endpoints capture independently after the cutoff; retries heal failures."""
    clock = FakeClock(_ns(2026, 9, 26, 0, 1, 0))
    names = [name for name, _ in _config().reference_urls]
    bodies = {name: f'{{"endpoint":{name!r}}}'.replace("'", '"') for name in names}
    stopped = False
    capture, _ = _reference_setup(tmp_path, clock, bodies, fail_once={"funding_info"}, stop=lambda: stopped)

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(capture.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            if all((tmp_path / "reference" / name / "20260926.json.gz").exists() for name in names):
                stopped = True
                break
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(scenario())
    for name in names:
        dest = tmp_path / "reference" / name / "20260926.json.gz"
        assert dest.exists()
        assert gzip.decompress(dest.read_bytes()).decode("utf-8") == bodies[name]


def test_reference_second_run_writes_nothing_new(tmp_path: Path) -> None:
    """Spec 01: a day already on disk is never rewritten."""
    clock = FakeClock(_ns(2026, 9, 26, 0, 6, 0))
    names = [name for name, _ in _config().reference_urls]
    bodies = {name: f"body-{name}" for name in names}
    first_done = False
    first, _ = _reference_setup(tmp_path, clock, bodies, stop=lambda: first_done)

    async def run_first() -> None:
        task = asyncio.ensure_future(first.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            if all((tmp_path / "reference" / name / "20260926.json.gz").exists() for name in names):
                break
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run_first())
    calls: dict[str, int] = {}

    async def counting_fetch(url: str) -> FetchResult:
        calls[url] = calls.get(url, 0) + 1
        return (200, "new", {})

    stopped = False
    second = ReferenceCapture(
        config=_config(),
        capture_root=tmp_path,
        fetch=counting_fetch,
        gate=RateGate(60.0),
        clock_ns=clock,
        sleep=clock.sleep,
        shutdown=lambda: stopped,
    )

    async def run_second() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(second.run())
        for _ in range(200):
            await asyncio.sleep(0)
        stopped = True
        await asyncio.wait_for(task, timeout=30)

    asyncio.run(run_second())
    assert calls == {}
    for name in names:
        assert gzip.decompress((tmp_path / "reference" / name / "20260926.json.gz").read_bytes()) == bodies[name].encode()


def test_two_slots_one_reference_file(tmp_path: Path) -> None:
    """Spec 01: two overlapping slots produce exactly one reference file."""
    clock = FakeClock(_ns(2026, 9, 26, 0, 6, 0))
    stopped = False

    async def fetch(url: str) -> FetchResult:
        return (200, "shared-body", {})

    first = ReferenceCapture(
        config=_config(),
        capture_root=tmp_path,
        fetch=fetch,
        gate=RateGate(60.0),
        clock_ns=clock,
        sleep=clock.sleep,
        shutdown=lambda: stopped,
    )
    second = ReferenceCapture(
        config=_config(),
        capture_root=tmp_path,
        fetch=fetch,
        gate=RateGate(60.0),
        clock_ns=clock,
        sleep=clock.sleep,
        shutdown=lambda: stopped,
    )

    async def scenario() -> None:
        nonlocal stopped
        first_task = asyncio.ensure_future(first.run())
        second_task = asyncio.ensure_future(second.run())
        for _ in range(20000):
            await asyncio.sleep(0)
            if (tmp_path / "reference" / "exchange_info" / "20260926.json.gz").exists():
                stopped = True
                break
        await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=30)

    asyncio.run(scenario())
    dest = tmp_path / "reference" / "exchange_info" / "20260926.json.gz"
    assert gzip.decompress(dest.read_bytes()).decode("utf-8") == "shared-body"


def test_parse_retry_after_variants() -> None:
    """Retry-After parses numerics, ignores junk and defaults to None."""
    from src.capture.rest import _parse_retry_after

    assert _parse_retry_after({"Retry-After": "120"}) == 120.0
    assert _parse_retry_after({"retry-after": "soon"}) is None
    assert _parse_retry_after({}) is None


def test_flush_failure_is_logged_not_raised(tmp_path: Path) -> None:
    """Spec 01: a failing journal flush never breaks sampling; the buffer is kept."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 1))
    stopped = False

    async def fetch(url: str) -> FetchResult:
        nonlocal stopped
        stopped = True
        return (200, "ok", {})

    sampler, _ = _sampler(tmp_path, clock, fetch, lambda: stopped)

    def failing_flush() -> int:
        raise OSError("disk gone")

    sampler._writer.flush = failing_flush  # type: ignore[method-assign]

    async def scenario() -> None:
        await sampler.run()

    asyncio.run(scenario())
    assert sampler._writer.pending() == 1


def test_sample_slot_shutdown_during_retry(tmp_path: Path) -> None:
    """A shutdown while waiting out the retry delay records one rest_error."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 1, 0))
    grid = _ns(2026, 9, 26, 10, 1, 0)

    async def fetch(url: str) -> FetchResult:
        raise TimeoutError("boom")

    sampler, _ = _sampler(tmp_path, clock, fetch, lambda: True)

    async def scenario() -> None:
        await sampler._sample_slot(grid)

    asyncio.run(scenario())
    records = _rest_records(tmp_path, BOOK, _ns(2026, 9, 26, 10, 1, 0))
    assert len(records) == 1
    assert records[0]["kind"] == "rest_error"
    assert records[0]["status"] is None


def test_sample_slot_retry_rate_limited(tmp_path: Path) -> None:
    """A 429 on the retry attempt is recorded with its status and extends the embargo."""
    from src.capture.rest import RateLimited

    clock = FakeClock(_ns(2026, 9, 26, 10, 1, 0))
    grid = _ns(2026, 9, 26, 10, 1, 0)
    calls = 0
    gate = RateGate(60.0)

    async def fetch(url: str) -> FetchResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("boom")
        raise RateLimited(429, 30.0)

    sampler, _ = _sampler(tmp_path, clock, fetch, lambda: False, gate=gate)

    async def scenario() -> None:
        await sampler._sample_slot(grid)

    asyncio.run(scenario())
    records = _rest_records(tmp_path, BOOK, _ns(2026, 9, 26, 10, 1, 0))
    assert records[0]["kind"] == "rest_error"
    assert records[0]["status"] == 429
    assert gate.blocked_until(clock() / 1_000_000_000) is not None


def test_lag_skip_in_run(tmp_path: Path) -> None:
    """A grid missed beyond the start lag is recorded once and never sampled late."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 1))
    calls = 0
    stopped = False
    first = True

    async def overshoot(delay: float) -> None:
        nonlocal first
        clock.now_ns += int((70.0 if first else delay) * 1_000_000_000)
        first = False
        await asyncio.sleep(0)

    async def fetch(url: str) -> FetchResult:
        nonlocal calls, stopped
        calls += 1
        stopped = True
        return (200, "ok", {})

    sampler = GridSampler(
        stream=BOOK,
        url="https://example.invalid/book",
        interval_s=60,
        config=_config(),
        writer=SegmentWriter(tmp_path, BOOK, "blue"),
        fetch=fetch,
        gate=RateGate(60.0),
        status=RestStatus(),
        clock_ns=clock,
        sleep=overshoot,
        shutdown=lambda: stopped,
    )

    async def scenario() -> None:
        await sampler.run()

    asyncio.run(scenario())
    records = _rest_records(tmp_path, BOOK, _ns(2026, 9, 26, 10, 2, 0))
    errors = [item for item in records if item["kind"] == "rest_error"]
    assert calls == 1
    assert len(errors) == 1
    assert str(errors[0]["error"]).startswith("skipped:")


def _reference_direct(
    tmp_path: Path,
    fetch: Callable[[str], Awaitable[FetchResult]],
) -> ReferenceCapture:
    clock = FakeClock(_ns(2026, 9, 26, 0, 6, 0))
    return ReferenceCapture(
        config=_config(),
        capture_root=tmp_path,
        fetch=fetch,
        gate=RateGate(60.0),
        clock_ns=clock,
        sleep=clock.sleep,
        shutdown=lambda: True,
    )


def test_reference_attempt_outcomes(tmp_path: Path) -> None:
    """Each reference failure mode returns False without writing; success writes once."""
    from src.capture.rest import RateLimited

    async def check(outcome: FetchResult | BaseException, written: bool) -> None:
        async def fetch(url: str) -> FetchResult:
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        capture = _reference_direct(tmp_path, fetch)
        assert await capture._attempt_endpoint("exchange_info", "https://example.invalid/x", "20260926") is written

    async def scenario() -> None:
        await check(RateLimited(429, 10.0), False)
        await check((429, "limited", {}), False)
        await check((500, "bad", {}), False)
        await check((200, "", {}), False)
        await check((200, "ok-body", {}), True)

    asyncio.run(scenario())
    assert gzip.decompress((tmp_path / "reference" / "exchange_info" / "20260926.json.gz").read_bytes()) == b"ok-body"


def test_reference_write_failure_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unwritable reference file is a retryable failure, not a crash."""
    import src.capture.rest as _rest

    async def fetch(url: str) -> FetchResult:
        return (200, "ok", {})

    monkeypatch.setattr(_rest, "write_bytes_once", lambda dest, payload: (_ for _ in ()).throw(OSError("no")))
    capture = _reference_direct(tmp_path, fetch)

    async def scenario() -> bool:
        return await capture._attempt_endpoint("exchange_info", "https://example.invalid/x", "20260926")

    assert asyncio.run(scenario()) is False


def test_reference_honours_shared_embargo(tmp_path: Path) -> None:
    """An embargoed gate suspends every reference attempt until it lifts."""
    clock = FakeClock(_ns(2026, 9, 26, 0, 6, 0))
    gate = RateGate(60.0)
    gate.block(clock() / 1_000_000_000, 600.0)
    calls = 0
    stopped = False

    async def fetch(url: str) -> FetchResult:
        nonlocal calls
        calls += 1
        return (200, "ok", {})

    capture = ReferenceCapture(
        config=_config(),
        capture_root=tmp_path,
        fetch=fetch,
        gate=gate,
        clock_ns=clock,
        sleep=clock.sleep,
        shutdown=lambda: stopped,
    )

    async def scenario() -> None:
        nonlocal stopped
        task = asyncio.ensure_future(capture.run())
        for _ in range(10):
            await asyncio.sleep(0)
        stopped = True
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(scenario())
    assert calls == 0


def test_rest_buffer_cap_drops_oldest_and_counts(tmp_path: Path) -> None:
    """Persistent flush failure keeps at most ``rest_max_pending_records``; drops are counted, not silent."""
    clock = FakeClock(_ns(2026, 9, 26, 10, 0, 1))
    sampler, status = _sampler(tmp_path, clock, _never_called, lambda: True, config=_config(rest_max_pending_records=2))

    def failing_flush() -> int:
        raise OSError("disk full")

    sampler._writer.flush = failing_flush  # type: ignore[method-assign]
    for i in range(5):
        sampler._write_rest(_ns(2026, 9, 26, 10, i), 200, "{}")
    assert sampler._writer.pending() == 2
    assert status.dropped_records == 3


async def _never_called(url: str) -> FetchResult:
    raise AssertionError("fetch must not be called")
