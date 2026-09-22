"""Always-on recorder of live-only Binance market sources with no archival backfill."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from src.market_data.streams.coverage import CoverageTracker
from src.market_data.streams.liquidations import run_liquidation_stream
from src.market_data.streams.snapshots import (
    BOOK_TICKER_DATASET,
    BOOK_TICKER_URL,
    PREMIUM_INDEX_DATASET,
    PREMIUM_INDEX_URL,
    REFERENCE_URLS,
    next_grid_time,
    parse_book_ticker_payload,
    parse_premium_index_payload,
    write_hourly_partition,
    write_reference_snapshot,
)

_logger = logging.getLogger(__name__)

HEARTBEAT_NAME: str = "recorder_heartbeat.json"

_CUTOFF_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


class MarketRecorderConfig(BaseModel):
    """Cadences and limits of the always-on live-only market recorder (frozen, validated)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    book_ticker_interval_s: int = 60
    premium_index_interval_s: int = 300
    flush_interval_s: float = 300.0
    reference_capture_after_utc: str = "00:05"
    http_timeout_s: float = 10.0
    restart_backoff_max_s: float = 60.0
    liquidation_flush_interval_s: float = 60.0

    @field_validator("book_ticker_interval_s", "premium_index_interval_s")
    @classmethod
    def _check_grid(cls, value: int) -> int:
        if value <= 0 or 86400 % value != 0:
            raise ValueError("grid interval must be a positive divisor of 86400")
        return value

    @field_validator("flush_interval_s")
    @classmethod
    def _check_flush(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("flush_interval_s must be positive")
        return value

    @field_validator("reference_capture_after_utc")
    @classmethod
    def _check_cutoff(cls, value: str) -> str:
        if not _CUTOFF_RE.match(value):
            raise ValueError("reference_capture_after_utc must be HH:MM")
        return value

    @field_validator("restart_backoff_max_s")
    @classmethod
    def _check_backoff(cls, value: float) -> float:
        if value < 1:
            raise ValueError("restart_backoff_max_s must be >= 1")
        return value


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _cutoff_for_day(now: pd.Timestamp, spec: str) -> pd.Timestamp:
    match = _CUTOFF_RE.match(spec)
    if match is None:
        raise ValueError("reference_capture_after_utc must be HH:MM")
    day = now.tz_convert("UTC").normalize()
    return day + pd.Timedelta(hours=int(match.group(1)), minutes=int(match.group(2)))


async def _sleep_capped(
    sleep: Callable[[float], Awaitable[None]], delay: float, shutdown: Any, step: float = 1.0
) -> None:
    remaining = delay
    while remaining > 0:
        if bool(getattr(shutdown, "requested", False)):
            return
        chunk = min(step, remaining)
        await sleep(chunk)
        remaining -= chunk


def _bound_frames(frames: list[pd.DataFrame], limit: int) -> tuple[list[pd.DataFrame], int]:
    total = sum(len(frame) for frame in frames)
    if total <= limit or limit <= 0:
        return frames, 0
    merged = pd.concat(frames, ignore_index=True).iloc[-limit:]
    return [merged], total - limit


def _ensure_capture_dirs(capture_root: Path, liquidations_dir: Path) -> Path:
    root = Path(capture_root)
    root.mkdir(parents=True, exist_ok=True)
    Path(liquidations_dir).mkdir(parents=True, exist_ok=True)
    return root


def _write_heartbeat_file(root: Path, payload: dict[str, Any]) -> None:
    target = Path(root) / HEARTBEAT_NAME
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)


class _Heartbeat:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {
            BOOK_TICKER_DATASET: {"last_success_at": None, "rows_last_flush": 0, "consecutive_failures": 0},
            PREMIUM_INDEX_DATASET: {"last_success_at": None, "rows_last_flush": 0, "consecutive_failures": 0},
            "reference": {"last_success_at": None, "rows_last_flush": 0, "consecutive_failures": 0},
        }
        self._lock = asyncio.Lock()

    async def update(
        self,
        root: Path,
        dataset: str,
        *,
        ts: pd.Timestamp,
        last_success_at: pd.Timestamp | None,
        rows_last_flush: int,
        consecutive_failures: int,
    ) -> None:
        async with self._lock:
            entry = self._entries.setdefault(
                dataset, {"last_success_at": None, "rows_last_flush": 0, "consecutive_failures": 0}
            )
            entry["last_success_at"] = last_success_at.isoformat() if last_success_at is not None else None
            entry["rows_last_flush"] = rows_last_flush
            entry["consecutive_failures"] = consecutive_failures
            payload = {"ts": pd.Timestamp(ts).isoformat(), **self._entries}
            _write_heartbeat_file(root, payload)


class _GridSampler:
    def __init__(
        self,
        *,
        dataset: str,
        url: str,
        interval_s: int,
        parse_fn: Callable[..., pd.DataFrame],
        config: MarketRecorderConfig,
        capture_root: Path,
        heartbeat: _Heartbeat,
        fetch: Callable[[str], Awaitable[bytes]],
        now_fn: Callable[[], pd.Timestamp],
        sleep: Callable[[float], Awaitable[None]],
        shutdown: Any,
    ) -> None:
        self._dataset = dataset
        self._url = url
        self._interval_s = interval_s
        self._parse_fn = parse_fn
        self._config = config
        self._root = capture_root
        self._heartbeat = heartbeat
        self._fetch = fetch
        self._now = now_fn
        self._sleep = sleep
        self._shutdown = shutdown
        self._buffer: list[pd.DataFrame] = []
        self._last_sample_rows: int | None = None
        self._last_success: pd.Timestamp | None = None
        self._failures = 0
        self._rows_last_flush = 0
        self._prev_grid: pd.Timestamp | None = None

    def _is_shutdown(self) -> bool:
        return bool(getattr(self._shutdown, "requested", False))

    async def _flush(self) -> None:
        if not self._buffer:
            await self._heartbeat.update(
                self._root, self._dataset, ts=self._now(),
                last_success_at=self._last_success, rows_last_flush=0,
                consecutive_failures=self._failures,
            )
            return
        frame = pd.concat(self._buffer, ignore_index=True)
        try:
            files = write_hourly_partition(frame, self._root, self._dataset)
        except Exception as exc:
            _logger.warning(
                "[DATA] stage=market_recorder dataset=%s status=FLUSH_FAILED error=%s",
                self._dataset, exc, exc_info=True,
            )
            expected = max(1, int(self._config.flush_interval_s // self._interval_s) + 1)
            limit = 2 * expected * (self._last_sample_rows or 800)
            total = sum(len(item) for item in self._buffer)
            if total > limit:
                self._buffer, dropped = _bound_frames(self._buffer, limit)
                _logger.error(
                    "[DATA] stage=market_recorder dataset=%s status=BUFFER_OVERFLOW dropped=%d",
                    self._dataset, dropped,
                )
            await self._heartbeat.update(
                self._root, self._dataset, ts=self._now(),
                last_success_at=self._last_success, rows_last_flush=0,
                consecutive_failures=self._failures,
            )
            return
        rows = len(frame)
        self._buffer = []
        self._rows_last_flush = rows
        _logger.info(
            "[DATA] stage=market_recorder dataset=%s rows=%d files=%d",
            self._dataset, rows, len(files),
        )
        await self._heartbeat.update(
            self._root, self._dataset, ts=self._now(),
            last_success_at=self._last_success, rows_last_flush=rows,
            consecutive_failures=self._failures,
        )

    async def run(self) -> None:
        target = next_grid_time(self._now(), self._interval_s)
        last_flush_wall = self._now()
        while not self._is_shutdown():
            now = self._now()
            if now < target:
                await _sleep_capped(self._sleep, (target - now).total_seconds(), self._shutdown)
                continue
            grid_at = target
            target = next_grid_time(grid_at, self._interval_s)
            try:
                raw = await self._fetch(self._url)
                payload = json.loads(raw.decode("utf-8"))
                frame = self._parse_fn(payload, captured_at=grid_at)
            except Exception as exc:
                self._failures += 1
                _logger.warning(
                    "[DATA] stage=market_recorder dataset=%s status=FAILED grid=%s error=%s",
                    self._dataset, grid_at.isoformat(), exc, exc_info=True,
                )
            else:
                self._buffer.append(frame)
                self._last_sample_rows = len(frame)
                self._last_success = grid_at
                self._failures = 0
                if self._prev_grid is not None and grid_at.floor("h") != self._prev_grid.floor("h"):
                    await self._flush()
                    last_flush_wall = self._now()
                self._prev_grid = grid_at
            if (self._now() - last_flush_wall).total_seconds() >= self._config.flush_interval_s:
                await self._flush()
                last_flush_wall = self._now()
        await self._flush()


async def run_market_recorder(
    config: MarketRecorderConfig,
    *,
    capture_root: Path,
    liquidations_dir: Path,
    shutdown: Any,
    fetch: Callable[[str], Awaitable[bytes]] | None = None,
    liquidation_runner: Callable[..., Awaitable[None]] | None = None,
    now_fn: Callable[[], pd.Timestamp] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Run all live-only capture tasks concurrently until ``shutdown.requested``.

    Tasks: liquidation stream (with coverage), book-ticker grid sampler, premium-index grid sampler,
    daily reference capture. Each task is supervised independently: an exception is logged with its
    traceback and the task restarts after capped exponential backoff, so one failing source never stops
    the others or the process. Capture is observability-only and has no effect on trading.

    Args:
        config: validated cadences.
        capture_root: ``LIVE_CAPTURE_DIR`` in production.
        liquidations_dir: existing liquidation partition directory.
        shutdown: object exposing ``requested``; checked at least once per second by every task.
        fetch: GET returning raw response bytes (default: shared aiohttp session with ``http_timeout_s``).
        liquidation_runner: defaults to ``run_liquidation_stream``.
    """
    root = _ensure_capture_dirs(capture_root, liquidations_dir)
    _now = now_fn if now_fn is not None else _utc_now
    runner = liquidation_runner if liquidation_runner is not None else run_liquidation_stream
    heartbeat = _Heartbeat()
    session: Any | None = None
    if fetch is None:
        import aiohttp

        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.http_timeout_s))

        async def _session_fetch(url: str) -> bytes:
            assert session is not None
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.read()

        fetch_fn: Callable[[str], Awaitable[bytes]] = _session_fetch
    else:
        fetch_fn = fetch

    def _is_shutdown() -> bool:
        return bool(getattr(shutdown, "requested", False))

    async def _supervised(name: str, coro_fn: Callable[[], Awaitable[None]]) -> None:
        backoff = 1.0
        while not _is_shutdown():
            try:
                await coro_fn()
                return
            except Exception:
                _logger.exception("[DATA] stage=market_recorder dataset=%s status=RESTART", name)
                await _sleep_capped(sleep, min(backoff, config.restart_backoff_max_s), shutdown)
                backoff = min(config.restart_backoff_max_s, backoff * 2.0)

    book = _GridSampler(
        dataset=BOOK_TICKER_DATASET, url=BOOK_TICKER_URL,
        interval_s=config.book_ticker_interval_s, parse_fn=parse_book_ticker_payload,
        config=config, capture_root=root, heartbeat=heartbeat,
        fetch=fetch_fn, now_fn=_now, sleep=sleep, shutdown=shutdown,
    )
    premium = _GridSampler(
        dataset=PREMIUM_INDEX_DATASET, url=PREMIUM_INDEX_URL,
        interval_s=config.premium_index_interval_s, parse_fn=parse_premium_index_payload,
        config=config, capture_root=root, heartbeat=heartbeat,
        fetch=fetch_fn, now_fn=_now, sleep=sleep, shutdown=shutdown,
    )

    async def _liquidations() -> None:
        tracker = CoverageTracker("liquidations", root)
        await runner(
            symbols=None, directory=Path(liquidations_dir),
            flush_interval_s=config.liquidation_flush_interval_s,
            shutdown=shutdown, coverage=tracker,
        )
        try:
            tracker.flush()
        except Exception as exc:
            _logger.warning("[DATA] stage=market_recorder dataset=liquidations status=FLUSH_FAILED error=%s", exc)

    async def _reference() -> None:
        done: set[str] = set()
        failures = 0
        while not _is_shutdown():
            now = _now()
            day = now.tz_convert("UTC").strftime("%Y%m%d")
            cutoff = _cutoff_for_day(now, config.reference_capture_after_utc)
            if now >= cutoff and day not in done:
                try:
                    for name, url in REFERENCE_URLS.items():
                        raw = await fetch_fn(url)
                        write_reference_snapshot(raw, root, name, captured_at=now)
                    done.add(day)
                    failures = 0
                    await heartbeat.update(
                        root, "reference", ts=now, last_success_at=now,
                        rows_last_flush=len(REFERENCE_URLS), consecutive_failures=0,
                    )
                    _logger.info(
                        "[DATA] stage=market_recorder dataset=reference rows=%d files=%d",
                        len(REFERENCE_URLS), len(REFERENCE_URLS),
                    )
                except Exception as exc:
                    failures += 1
                    _logger.warning(
                        "[DATA] stage=market_recorder dataset=reference status=FAILED error=%s",
                        exc, exc_info=True,
                    )
                    await heartbeat.update(
                        root, "reference", ts=_now(), last_success_at=None,
                        rows_last_flush=0, consecutive_failures=failures,
                    )
                    await _sleep_capped(sleep, 3600.0, shutdown)
                    continue
            await _sleep_capped(sleep, 60.0, shutdown)

    try:
        await asyncio.gather(
            _supervised(BOOK_TICKER_DATASET, book.run),
            _supervised(PREMIUM_INDEX_DATASET, premium.run),
            _supervised("reference", _reference),
            _supervised("liquidations", _liquidations),
        )
    finally:
        if session is not None:
            await session.close()
