"""Always-on recorder of live-only Binance market sources with no archival backfill."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationInfo, field_validator, model_validator

from src.market_data.streams.coverage import CoverageTracker
from src.market_data.streams.liquidations import LiquidationHealth, run_liquidation_stream
from src.market_data.streams.snapshots import (
    BOOK_TICKER_DATASET,
    BOOK_TICKER_URL,
    PREMIUM_INDEX_DATASET,
    PREMIUM_INDEX_URL,
    REFERENCE_URLS,
    SnapshotParse,
    next_grid_time,
    parse_book_ticker_payload,
    parse_premium_index_payload,
    reference_snapshot_path,
    write_hourly_partition,
    write_reference_snapshot,
)

_logger = logging.getLogger(__name__)

HEARTBEAT_NAME: str = "recorder_heartbeat.json"
HEARTBEAT_SCHEMA_VERSION: int = 2

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
    liquidation_receive_timeout_s: float = 1.0
    liquidation_liveness_timeout_s: float = 15.0
    liquidation_ping_interval_s: float = 5.0
    liquidation_event_stall_timeout_s: float = 600.0
    heartbeat_interval_s: float = 60.0  # 워치독 heartbeat staleness 임계값(daemon recorder_heartbeat_stale_s, 기본 600 s)보다 충분히 작게 유지할 것
    grid_max_start_lag_s: float = 5.0
    grid_retry_delay_s: float = 1.0
    rate_limit_cooldown_s: float = 60.0
    snapshot_max_rejected_fraction: float = 0.05
    liquidation_max_pending_events: int = 100_000
    reference_retry_interval_s: float = 600.0
    grid_health_window_s: float = 3600.0
    deadman_ping_url: SecretStr | None = None
    deadman_ping_interval_s: float = 300.0
    deadman_ping_timeout_s: float = 10.0

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

    @field_validator("liquidation_receive_timeout_s", "liquidation_ping_interval_s")
    @classmethod
    def _check_liquidation_timing(cls, value: float, info: ValidationInfo) -> float:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @field_validator("liquidation_event_stall_timeout_s")
    @classmethod
    def _check_liquidation_event_stall(cls, value: float, info: ValidationInfo) -> float:
        liveness = info.data.get("liquidation_liveness_timeout_s", 15.0)
        if value <= liveness:
            raise ValueError("liquidation_event_stall_timeout_s must be greater than liquidation_liveness_timeout_s")
        return value

    @field_validator("liquidation_liveness_timeout_s")
    @classmethod
    def _check_liquidation_liveness(cls, value: float, info: ValidationInfo) -> float:
        ping = info.data.get("liquidation_ping_interval_s", 5.0)
        if value <= ping:
            raise ValueError("liquidation_liveness_timeout_s must be greater than liquidation_ping_interval_s")
        return value

    @field_validator("heartbeat_interval_s")
    @classmethod
    def _check_heartbeat(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("heartbeat_interval_s must be positive")
        return value

    @field_validator("snapshot_max_rejected_fraction")
    @classmethod
    def _check_rejected_fraction(cls, value: float) -> float:
        if not 0 < value < 1:
            raise ValueError("snapshot_max_rejected_fraction must be in (0, 1)")
        return value

    @field_validator("liquidation_max_pending_events")
    @classmethod
    def _check_liquidation_pending(cls, value: int) -> int:
        if value < 1:
            raise ValueError("liquidation_max_pending_events must be >= 1")
        return value

    @field_validator("reference_retry_interval_s", "grid_health_window_s")
    @classmethod
    def _check_reference_retry(cls, value: float, info: ValidationInfo) -> float:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @field_validator("deadman_ping_interval_s", "deadman_ping_timeout_s")
    @classmethod
    def _check_deadman_seconds(cls, value: float, info: ValidationInfo) -> float:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @model_validator(mode="after")
    def _check_grid_cadences(self) -> MarketRecorderConfig:
        if not self.grid_retry_delay_s > 0:
            raise ValueError("grid_retry_delay_s must be positive")
        if not self.grid_retry_delay_s < self.grid_max_start_lag_s:
            raise ValueError("grid_retry_delay_s must be less than grid_max_start_lag_s")
        if not self.grid_max_start_lag_s < min(self.book_ticker_interval_s, self.premium_index_interval_s):
            raise ValueError("grid_max_start_lag_s must be less than the sampler grid intervals")
        if not self.rate_limit_cooldown_s >= 1:
            raise ValueError("rate_limit_cooldown_s must be >= 1")
        if not self.deadman_ping_timeout_s < self.deadman_ping_interval_s:
            raise ValueError("deadman_ping_timeout_s must be < deadman_ping_interval_s")
        if not self.grid_health_window_s >= max(self.book_ticker_interval_s, self.premium_index_interval_s):
            raise ValueError("grid_health_window_s must be >= the sampler grid intervals")
        return self


class RateLimitedError(RuntimeError):
    """The venue answered HTTP 418 (IP ban) or 429 (weight limit exceeded).

    Attributes:
        status: HTTP status code (418 or 429).
        retry_after_s: ``Retry-After`` header in seconds when present and numeric, else ``None``.
    """

    def __init__(self, status: int, retry_after_s: float | None) -> None:
        super().__init__(f"rate limited: HTTP {status}")
        self.status = status
        self.retry_after_s = retry_after_s


class _RateLimitGate:
    """Process-wide fetch embargo shared by all grid samplers (Binance limits are per IP).

    ``block`` extends the embargo to ``now + max(retry_after_s or 0, cooldown_s)`` and never shortens
    an existing one; ``blocked_until`` returns the embargo end or ``None``.
    """

    def __init__(self, *, cooldown_s: float) -> None:
        self._cooldown_s = float(cooldown_s)
        self._blocked_until: pd.Timestamp | None = None

    def block(self, now: pd.Timestamp, retry_after_s: float | None) -> pd.Timestamp:
        end = pd.Timestamp(now) + pd.Timedelta(seconds=max(retry_after_s or 0.0, self._cooldown_s))
        if self._blocked_until is None or end > self._blocked_until:
            self._blocked_until = end
        return self._blocked_until

    def blocked_until(self, now: pd.Timestamp) -> pd.Timestamp | None:
        if self._blocked_until is None:
            return None
        if pd.Timestamp(now) >= self._blocked_until:
            return None
        return self._blocked_until


def _floor_grid_time(now: pd.Timestamp, interval_s: int) -> pd.Timestamp:
    """Latest epoch-aligned ``interval_s`` grid point at or before ``now`` (UTC)."""
    utc = pd.Timestamp(now).tz_convert("UTC")
    step_ns = int(interval_s) * 1_000_000_000
    floored = (int(utc.value) // step_ns) * step_ns
    return pd.Timestamp(floored, unit="ns", tz="UTC")


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


@dataclass(frozen=True, slots=True)
class SamplerHeartbeatEntry:
    """Point-in-time health of one grid sampler as published in the recorder heartbeat.

    Fetch success and persistence are reported separately because a sampler can keep fetching while
    every write fails (full disk, merge failure); the watchdog must see the data that reached disk,
    not the loop that produced it.

    Attributes:
        last_success_at: Grid instant of the latest accepted sample (fetch + parse), or None.
        consecutive_failures: Grid points in a row whose fetch/parse failed or was rate limited.
        skipped_grid_points: Grid points skipped (late, lag, embargo) since process start.
        rows_last_flush: Rows written by the latest successful flush.
        last_persisted_at: Wall-clock instant of the latest successful partition write, or None.
        consecutive_flush_failures: Flushes in a row that raised.
        pending_rows: Rows buffered in memory and not yet persisted.
        dropped_rows_total: Rows discarded by the buffer bound since process start.
        rejected_rows_last_sample: Rows excluded by the row contract in the latest accepted sample.
        rejected_rows_total: Rows excluded since process start.
        rejected_fraction_last_sample: Rejected rows / (accepted + rejected rows) of the latest accepted sample; 0.0 when none.
        consecutive_rejecting_points: Accepted grid points in a row whose sample rejected at least one row; reset to 0 by a sample with no rejected row.
        window_expected_points: Grid points inside the trailing health window since sampler start.
        window_captured_points: Of those, points whose sample was accepted.
    """

    last_success_at: pd.Timestamp | None
    consecutive_failures: int
    skipped_grid_points: int
    rows_last_flush: int
    last_persisted_at: pd.Timestamp | None
    consecutive_flush_failures: int
    pending_rows: int
    dropped_rows_total: int
    rejected_rows_last_sample: int
    rejected_rows_total: int
    rejected_fraction_last_sample: float
    consecutive_rejecting_points: int
    window_expected_points: int
    window_captured_points: int


@dataclass(frozen=True, slots=True)
class ReferenceEndpointStatus:
    """Capture state of one daily reference endpoint for the current UTC day."""

    captured: bool
    consecutive_failures: int
    last_attempt_at: pd.Timestamp | None
    last_error: str | None


@dataclass(frozen=True, slots=True)
class ReferenceHeartbeatEntry:
    """Daily reference capture state as published in the recorder heartbeat.

    Attributes:
        day: UTC day (``YYYYMMDD``) the endpoint statuses refer to.
        cutoff_utc: ``reference_capture_after_utc``, published so the watchdog can compute the
            deadline without sharing the recorder configuration.
        endpoints: Endpoint name (every key of ``REFERENCE_URLS``) to its status.
        last_success_at: Instant the day's set became complete, or None.
        previous_day: The preceding UTC day this process observed, or None after a restart.
        previous_day_complete: Whether ``previous_day`` ended with every endpoint captured.
    """

    day: str
    cutoff_utc: str
    endpoints: Mapping[str, ReferenceEndpointStatus]
    last_success_at: pd.Timestamp | None
    previous_day: str | None
    previous_day_complete: bool | None


def _sampler_entry_to_dict(entry: SamplerHeartbeatEntry) -> dict[str, Any]:
    return {
        "last_success_at": entry.last_success_at.isoformat() if entry.last_success_at is not None else None,
        "consecutive_failures": int(entry.consecutive_failures),
        "skipped_grid_points": int(entry.skipped_grid_points),
        "rows_last_flush": int(entry.rows_last_flush),
        "last_persisted_at": entry.last_persisted_at.isoformat() if entry.last_persisted_at is not None else None,
        "consecutive_flush_failures": int(entry.consecutive_flush_failures),
        "pending_rows": int(entry.pending_rows),
        "dropped_rows_total": int(entry.dropped_rows_total),
        "rejected_rows_last_sample": int(entry.rejected_rows_last_sample),
        "rejected_rows_total": int(entry.rejected_rows_total),
        "rejected_fraction_last_sample": float(entry.rejected_fraction_last_sample),
        "consecutive_rejecting_points": int(entry.consecutive_rejecting_points),
        "window_expected_points": int(entry.window_expected_points),
        "window_captured_points": int(entry.window_captured_points),
    }


def _reference_entry_to_dict(entry: ReferenceHeartbeatEntry) -> dict[str, Any]:
    return {
        "day": entry.day,
        "cutoff_utc": entry.cutoff_utc,
        "endpoints": {
            name: {
                "captured": bool(status.captured),
                "consecutive_failures": int(status.consecutive_failures),
                "last_attempt_at": status.last_attempt_at.isoformat() if status.last_attempt_at is not None else None,
                "last_error": status.last_error,
            }
            for name, status in entry.endpoints.items()
        },
        "last_success_at": entry.last_success_at.isoformat() if entry.last_success_at is not None else None,
        "previous_day": entry.previous_day,
        "previous_day_complete": entry.previous_day_complete,
    }


def _fresh_sampler_entry() -> dict[str, Any]:
    return _sampler_entry_to_dict(
        SamplerHeartbeatEntry(
            last_success_at=None,
            consecutive_failures=0,
            skipped_grid_points=0,
            rows_last_flush=0,
            last_persisted_at=None,
            consecutive_flush_failures=0,
            pending_rows=0,
            dropped_rows_total=0,
            rejected_rows_last_sample=0,
            rejected_rows_total=0,
            rejected_fraction_last_sample=0.0,
            consecutive_rejecting_points=0,
            window_expected_points=0,
            window_captured_points=0,
        )
    )


class _Heartbeat:
    def __init__(self, *, started_at: pd.Timestamp) -> None:
        self._started_at = pd.Timestamp(started_at)
        self._entries: dict[str, dict[str, Any]] = {
            BOOK_TICKER_DATASET: _fresh_sampler_entry(),
            PREMIUM_INDEX_DATASET: _fresh_sampler_entry(),
            "reference": _reference_entry_to_dict(
                ReferenceHeartbeatEntry(
                    day="",
                    cutoff_utc="",
                    endpoints={},
                    last_success_at=None,
                    previous_day=None,
                    previous_day_complete=None,
                )
            ),
            "liquidations": {
                "last_event_at": None,
                "last_connected_at": None,
                "consecutive_failed_connections": 0,
                "last_disconnect_reason": None,
            },
        }
        self._lock = asyncio.Lock()

    def _payload(self, ts: pd.Timestamp) -> dict[str, Any]:
        return {
            "schema_version": HEARTBEAT_SCHEMA_VERSION,
            "ts": pd.Timestamp(ts).isoformat(),
            "started_at": self._started_at.isoformat(),
            **self._entries,
        }

    async def update_sampler(
        self, root: Path, dataset: str, *, ts: pd.Timestamp, entry: SamplerHeartbeatEntry
    ) -> None:
        """Publish one sampler's persistence-aware entry, rewriting the whole file."""
        async with self._lock:
            self._entries[dataset] = _sampler_entry_to_dict(entry)
            _write_heartbeat_file(root, self._payload(ts))

    async def update_reference(
        self, root: Path, *, ts: pd.Timestamp, entry: ReferenceHeartbeatEntry
    ) -> None:
        """Publish the daily reference capture state, rewriting the whole file."""
        async with self._lock:
            self._entries["reference"] = _reference_entry_to_dict(entry)
            _write_heartbeat_file(root, self._payload(ts))

    async def update_liquidations(
        self, root: Path, *, ts: pd.Timestamp, entry: Mapping[str, Any]
    ) -> None:
        """Replace the ``liquidations`` entry and atomically rewrite the heartbeat file."""
        async with self._lock:
            self._entries["liquidations"] = dict(entry)
            _write_heartbeat_file(root, self._payload(ts))

    def is_failing(self) -> bool:
        """Return True when any dataset shows consecutive fetch or flush failures.

        A flush failure means fetched rows are not reaching disk, which is data loss even while
        fetching looks healthy, so it maps to the dead-man ``/fail`` endpoint as well.
        """
        for dataset, entry in self._entries.items():
            keys = ("consecutive_flush_failures",) if dataset == "liquidations" else (
                "consecutive_failures", "consecutive_flush_failures",
            )
            for key in keys:
                try:
                    if int(entry.get(key, 0)) >= 1:
                        return True
                except (TypeError, ValueError):
                    continue
        return False


class _GridSampler:
    def __init__(
        self,
        *,
        dataset: str,
        url: str,
        interval_s: int,
        parse_fn: Callable[..., SnapshotParse],
        config: MarketRecorderConfig,
        capture_root: Path,
        heartbeat: _Heartbeat,
        fetch: Callable[[str], Awaitable[bytes]],
        now_fn: Callable[[], pd.Timestamp],
        sleep: Callable[[float], Awaitable[None]],
        shutdown: Any,
        rate_gate: _RateLimitGate,
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
        self._rate_gate = rate_gate
        self._buffer: list[pd.DataFrame] = []
        self._last_sample_rows: int | None = None
        self._last_success: pd.Timestamp | None = None
        self._failures = 0
        self._rows_last_flush = 0
        self._prev_grid: pd.Timestamp | None = None
        self._skipped = 0
        self._rejected_rows_last_sample: int = 0
        self._rejected_rows_total: int = 0
        self._last_persisted: pd.Timestamp | None = None
        self._flush_failures = 0
        self._dropped_total = 0
        self._rejected_fraction_last = 0.0
        self._consecutive_rejecting = 0
        self._first_target: pd.Timestamp | None = None
        bound = max(8, int(config.grid_health_window_s // interval_s) + 8)
        self._outcomes: deque[tuple[pd.Timestamp, bool]] = deque(maxlen=bound)

    def _is_shutdown(self) -> bool:
        return bool(getattr(self._shutdown, "requested", False))

    def _prune_outcomes(self, now: pd.Timestamp) -> None:
        cutoff = pd.Timestamp(now) - pd.Timedelta(seconds=self._config.grid_health_window_s)
        outcomes = self._outcomes
        while outcomes and outcomes[0][0] <= cutoff:
            outcomes.popleft()

    def _heartbeat_entry(self, now: pd.Timestamp) -> SamplerHeartbeatEntry:
        moment = pd.Timestamp(now)
        self._prune_outcomes(moment)
        first = self._first_target
        if first is None:
            expected = 0
            captured = 0
        else:
            expected = sum(1 for ts, _ in self._outcomes if ts >= first)
            captured = sum(1 for ts, ok in self._outcomes if ts >= first and ok)
        return SamplerHeartbeatEntry(
            last_success_at=self._last_success,
            consecutive_failures=self._failures,
            skipped_grid_points=self._skipped,
            rows_last_flush=self._rows_last_flush,
            last_persisted_at=self._last_persisted,
            consecutive_flush_failures=self._flush_failures,
            pending_rows=sum(len(item) for item in self._buffer),
            dropped_rows_total=self._dropped_total,
            rejected_rows_last_sample=self._rejected_rows_last_sample,
            rejected_rows_total=self._rejected_rows_total,
            rejected_fraction_last_sample=self._rejected_fraction_last,
            consecutive_rejecting_points=self._consecutive_rejecting,
            window_expected_points=expected,
            window_captured_points=captured,
        )

    async def _publish(self) -> None:
        now = self._now()
        await self._heartbeat.update_sampler(
            self._root, self._dataset, ts=now, entry=self._heartbeat_entry(now)
        )

    async def _flush(self) -> None:
        if not self._buffer:
            await self._publish()
            return
        frame = pd.concat(self._buffer, ignore_index=True)
        try:
            files = write_hourly_partition(frame, self._root, self._dataset)
        except Exception as exc:
            self._flush_failures += 1
            _logger.warning(
                "[DATA] stage=market_recorder dataset=%s status=FLUSH_FAILED error=%s",
                self._dataset, exc, exc_info=True,
            )
            expected = max(1, int(self._config.flush_interval_s // self._interval_s) + 1)
            limit = 2 * expected * (self._last_sample_rows or 800)
            total = sum(len(item) for item in self._buffer)
            if total > limit:
                self._buffer, dropped = _bound_frames(self._buffer, limit)
                self._dropped_total += dropped
                _logger.error(
                    "[DATA] stage=market_recorder dataset=%s status=BUFFER_OVERFLOW dropped=%d",
                    self._dataset, dropped,
                )
            await self._publish()
            return
        rows = len(frame)
        self._buffer = []
        self._rows_last_flush = rows
        self._last_persisted = self._now()
        self._flush_failures = 0
        _logger.info(
            "[DATA] stage=market_recorder dataset=%s rows=%d files=%d",
            self._dataset, rows, len(files),
        )
        await self._publish()

    def _register_skips(self, skipped: list[tuple[pd.Timestamp, str]]) -> None:
        self._skipped += len(skipped)
        for ts, _reason in skipped:
            self._outcomes.append((ts, False))
        first = skipped[0][0]
        last = skipped[-1][0]
        reason = skipped[-1][1]
        _logger.warning(
            "[DATA] stage=market_recorder dataset=%s status=GRID_SKIPPED first=%s last=%s count=%d reason=%s",
            self._dataset, first.isoformat(), last.isoformat(), len(skipped), reason,
        )

    async def _sample_slot(self, g: pd.Timestamp) -> bool:
        """Fetch and store one grid point, retrying transient failures within the lag bound."""
        lag_s = self._config.grid_max_start_lag_s
        retry_delay_s = self._config.grid_retry_delay_s
        while True:
            try:
                raw = await self._fetch(self._url)
                fetched_at = self._now()
                payload = json.loads(raw.decode("utf-8"))
                parsed = self._parse_fn(
                    payload, captured_at=g, fetched_at=fetched_at,
                    max_rejected_fraction=self._config.snapshot_max_rejected_fraction,
                )
                frame = parsed.frame
            except RateLimitedError as exc:
                until = self._rate_gate.block(self._now(), exc.retry_after_s)
                self._failures += 1
                self._outcomes.append((g, False))
                _logger.warning(
                    "[DATA] stage=market_recorder dataset=%s status=RATE_LIMITED http_status=%s blocked_until=%s",
                    self._dataset, exc.status, until.isoformat(),
                )
                await self._publish()
                return False
            except Exception as exc:
                await _sleep_capped(self._sleep, retry_delay_s, self._shutdown)
                if self._is_shutdown():
                    self._outcomes.append((g, False))
                    await self._publish()
                    return False
                if (self._now() - g).total_seconds() > lag_s:
                    self._failures += 1
                    self._outcomes.append((g, False))
                    _logger.warning(
                        "[DATA] stage=market_recorder dataset=%s status=FAILED grid=%s error=%s",
                        self._dataset, g.isoformat(), exc, exc_info=exc,
                    )
                    await self._publish()
                    return False
                continue
            self._buffer.append(frame)
            self._last_sample_rows = len(frame)
            self._last_success = g
            self._failures = 0
            self._rejected_rows_last_sample = parsed.rejected_rows
            self._rejected_rows_total += parsed.rejected_rows
            total_rows = len(frame) + parsed.rejected_rows
            self._rejected_fraction_last = (
                parsed.rejected_rows / total_rows if total_rows > 0 else 0.0
            )
            if parsed.rejected_rows > 0:
                self._consecutive_rejecting += 1
                reasons = ",".join(
                    f"{key}:{parsed.rejected_reasons[key]}" for key in sorted(parsed.rejected_reasons)
                )
                symbols = ",".join(parsed.rejected_symbols)
                _logger.warning(
                    "[DATA] stage=market_recorder dataset=%s status=ROWS_REJECTED grid=%s rejected=%d total=%d reasons=%s symbols=%s",
                    self._dataset, g.isoformat(), parsed.rejected_rows, parsed.total_rows,
                    reasons, symbols,
                )
            else:
                self._consecutive_rejecting = 0
            self._outcomes.append((g, True))
            await self._publish()
            return True

    async def run(self) -> None:
        """Sample ``url`` on the epoch-aligned ``interval_s`` grid until shutdown, then flush.

        Each wake samples the latest grid point not after the current time; grid points passed over
        (late wake, fetch longer than one interval, rate-limit embargo, or a start later than
        ``grid_max_start_lag_s``) are skipped explicitly — logged and counted in the heartbeat — and
        never back-filled with later data. Rows are filed under the grid instant and carry the actual
        receipt time in ``fetched_at_ms``. A failed fetch is retried after ``grid_retry_delay_s`` while
        the retry still starts within ``grid_max_start_lag_s`` of its grid point.
        """
        interval = pd.Timedelta(seconds=self._interval_s)
        lag = pd.Timedelta(seconds=self._config.grid_max_start_lag_s)
        target = next_grid_time(self._now(), self._interval_s)
        self._first_target = target
        last_flush_wall = self._now()
        while not self._is_shutdown():
            now = self._now()
            if now < target:
                await _sleep_capped(self._sleep, (target - now).total_seconds(), self._shutdown)
                continue
            now = self._now()
            g = _floor_grid_time(now, self._interval_s)
            skipped: list[tuple[pd.Timestamp, str]] = []
            point = target
            while point < g:
                skipped.append((point, "late"))
                point += interval
            if self._rate_gate.blocked_until(now) is not None:
                skipped.append((g, "rate_limited"))
                self._register_skips(skipped)
                await self._publish()
                target = next_grid_time(g, self._interval_s)
            elif now - g > lag:
                skipped.append((g, "lag"))
                self._register_skips(skipped)
                await self._publish()
                target = next_grid_time(g, self._interval_s)
            else:
                if skipped:
                    self._register_skips(skipped)
                    await self._publish()
                if await self._sample_slot(g):
                    if self._prev_grid is not None and g.floor("h") != self._prev_grid.floor("h"):
                        await self._flush()
                        last_flush_wall = self._now()
                    self._prev_grid = g
                target = next_grid_time(g, self._interval_s)
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

    Tasks: liquidation stream (with coverage and a process-wide ``LiquidationHealth``), book-ticker
    grid sampler, premium-index grid sampler, daily reference capture, and a heartbeat writer. Each
    task is supervised independently: a task that raises *or returns while shutdown is not requested*
    is logged and restarted after capped exponential backoff, so one failing source never stops the
    others or the process and no source can end silently. The heartbeat file is rewritten at least
    every ``heartbeat_interval_s`` so an external watchdog can detect a stalled process, a silent
    liquidation stream, or a stale sampler. Capture is observability-only and has no effect on trading.

    Args:
        config: validated cadences.
        capture_root: ``LIVE_CAPTURE_DIR`` in production.
        liquidations_dir: existing liquidation partition directory.
        shutdown: object exposing ``requested``; checked at least once per second by every task.
        fetch: GET returning raw response bytes (default: shared aiohttp session with ``http_timeout_s``).
        liquidation_runner: defaults to ``run_liquidation_stream``; always called with ``health=``.
    """
    root = _ensure_capture_dirs(capture_root, liquidations_dir)
    _now = now_fn if now_fn is not None else _utc_now
    runner = liquidation_runner if liquidation_runner is not None else run_liquidation_stream
    heartbeat = _Heartbeat(started_at=_now())
    health = LiquidationHealth()
    session: Any | None = None
    if fetch is None:
        import aiohttp

        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.http_timeout_s))

        async def _session_fetch(url: str) -> bytes:
            assert session is not None
            async with session.get(url) as response:
                if response.status in (418, 429):
                    raw_retry = response.headers.get("Retry-After") if response.headers else None
                    try:
                        retry_after_s = float(raw_retry) if raw_retry is not None else None
                    except (TypeError, ValueError):
                        retry_after_s = None
                    raise RateLimitedError(response.status, retry_after_s)
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
            run_start = _now()
            try:
                await coro_fn()
            except Exception:
                _logger.exception("[DATA] stage=market_recorder dataset=%s status=RESTART", name)
            else:
                if _is_shutdown():
                    return
                _logger.error(
                    "[DATA] stage=market_recorder dataset=%s status=UNEXPECTED_EXIT", name
                )
            if _is_shutdown():
                return
            elapsed_s = (_now() - run_start).total_seconds()
            if elapsed_s >= config.restart_backoff_max_s:
                backoff = 1.0
            await _sleep_capped(sleep, min(backoff, config.restart_backoff_max_s), shutdown)
            backoff = min(config.restart_backoff_max_s, backoff * 2.0)

    rate_gate = _RateLimitGate(cooldown_s=config.rate_limit_cooldown_s)
    from src.live.deadman import DeadmanPinger

    pinger = DeadmanPinger(
        url=config.deadman_ping_url,
        interval_s=float(config.deadman_ping_interval_s),
        timeout_s=float(config.deadman_ping_timeout_s),
    )
    book = _GridSampler(
        dataset=BOOK_TICKER_DATASET, url=BOOK_TICKER_URL,
        interval_s=config.book_ticker_interval_s, parse_fn=parse_book_ticker_payload,
        config=config, capture_root=root, heartbeat=heartbeat,
        fetch=fetch_fn, now_fn=_now, sleep=sleep, shutdown=shutdown,
        rate_gate=rate_gate,
    )
    premium = _GridSampler(
        dataset=PREMIUM_INDEX_DATASET, url=PREMIUM_INDEX_URL,
        interval_s=config.premium_index_interval_s, parse_fn=parse_premium_index_payload,
        config=config, capture_root=root, heartbeat=heartbeat,
        fetch=fetch_fn, now_fn=_now, sleep=sleep, shutdown=shutdown,
        rate_gate=rate_gate,
    )

    async def _liquidations() -> None:
        tracker = CoverageTracker("liquidations", root)
        await runner(
            symbols=None, directory=Path(liquidations_dir),
            flush_interval_s=config.liquidation_flush_interval_s,
            shutdown=shutdown, coverage=tracker, health=health,
            receive_timeout_s=config.liquidation_receive_timeout_s,
            liveness_timeout_s=config.liquidation_liveness_timeout_s,
            ping_interval_s=config.liquidation_ping_interval_s,
            event_stall_timeout_s=config.liquidation_event_stall_timeout_s,
            max_pending_events=config.liquidation_max_pending_events,
        )
        try:
            tracker.flush()
        except Exception as exc:
            _logger.warning("[DATA] stage=market_recorder dataset=liquidations status=FLUSH_FAILED error=%s", exc)

    async def _heartbeat_loop() -> None:
        while not _is_shutdown():
            await heartbeat.update_liquidations(root, ts=_now(), entry=health.as_heartbeat_entry())
            if pinger.enabled:
                await asyncio.to_thread(pinger.maybe_ping, now=_now(), failing=heartbeat.is_failing())
            await _sleep_capped(sleep, config.heartbeat_interval_s, shutdown)
        await heartbeat.update_liquidations(root, ts=_now(), entry=health.as_heartbeat_entry())
        if pinger.enabled:
            await asyncio.to_thread(pinger.maybe_ping, now=_now(), failing=heartbeat.is_failing())

    async def _reference() -> None:
        current_day: str | None = None
        endpoints: dict[str, ReferenceEndpointStatus] = {}
        last_success_at: pd.Timestamp | None = None
        previous_day: str | None = None
        previous_day_complete: bool | None = None

        def _entry() -> ReferenceHeartbeatEntry:
            return ReferenceHeartbeatEntry(
                day=current_day or "",
                cutoff_utc=config.reference_capture_after_utc,
                endpoints=dict(endpoints),
                last_success_at=last_success_at,
                previous_day=previous_day,
                previous_day_complete=previous_day_complete,
            )

        async def _publish(now: pd.Timestamp) -> None:
            await heartbeat.update_reference(root, ts=now, entry=_entry())

        while not _is_shutdown():
            now = _now()
            today = now.tz_convert("UTC").strftime("%Y%m%d")
            if today != current_day:
                if current_day is not None:
                    previous_day = current_day
                    previous_day_complete = all(status.captured for status in endpoints.values())
                current_day = today
                last_success_at = None
                endpoints = {}
                for name in REFERENCE_URLS:
                    if reference_snapshot_path(root, name, today).exists():
                        endpoints[name] = ReferenceEndpointStatus(
                            captured=True, consecutive_failures=0,
                            last_attempt_at=None, last_error=None,
                        )
                    else:
                        endpoints[name] = ReferenceEndpointStatus(
                            captured=False, consecutive_failures=0,
                            last_attempt_at=None, last_error=None,
                        )
                await _publish(now)
            cutoff = _cutoff_for_day(now, config.reference_capture_after_utc)
            if now >= cutoff and current_day is not None:
                changed = False
                for name, url in REFERENCE_URLS.items():
                    status = endpoints[name]
                    if status.captured:
                        continue
                    if reference_snapshot_path(root, name, current_day).exists():
                        endpoints[name] = ReferenceEndpointStatus(
                            captured=True, consecutive_failures=0,
                            last_attempt_at=status.last_attempt_at, last_error=None,
                        )
                        changed = True
                        continue
                    if (
                        status.last_attempt_at is not None
                        and (now - status.last_attempt_at).total_seconds()
                        < config.reference_retry_interval_s
                    ):
                        continue
                    attempt_at = now
                    try:
                        raw = await fetch_fn(url)
                        write_reference_snapshot(raw, root, name, captured_at=now)
                    except Exception as exc:
                        err = f"{type(exc).__name__}: {exc}"[:200]
                        endpoints[name] = ReferenceEndpointStatus(
                            captured=False,
                            consecutive_failures=status.consecutive_failures + 1,
                            last_attempt_at=attempt_at,
                            last_error=err,
                        )
                        _logger.warning(
                            "[DATA] stage=market_recorder dataset=reference endpoint=%s status=FAILED error=%s",
                            name, exc, exc_info=True,
                        )
                        changed = True
                        continue
                    endpoints[name] = ReferenceEndpointStatus(
                        captured=True, consecutive_failures=0,
                        last_attempt_at=attempt_at, last_error=None,
                    )
                    changed = True
                if all(status.captured for status in endpoints.values()) and last_success_at is None:
                    last_success_at = now
                    _logger.info(
                        "[DATA] stage=market_recorder dataset=reference rows=%d files=%d",
                        len(REFERENCE_URLS), len(REFERENCE_URLS),
                    )
                    changed = True
                if changed:
                    await _publish(now)
            await _sleep_capped(sleep, 60.0, shutdown)

    try:
        await asyncio.gather(
            _supervised(BOOK_TICKER_DATASET, book.run),
            _supervised(PREMIUM_INDEX_DATASET, premium.run),
            _supervised("reference", _reference),
            _supervised("liquidations", _liquidations),
            _supervised("heartbeat", _heartbeat_loop),
        )
    finally:
        if session is not None:
            await session.close()
