"""Grid REST sampling and daily reference capture for raw-first capture."""

from __future__ import annotations

import asyncio
import gzip
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import CaptureConfig
from .journal import RECORD_VERSION, SegmentWriter, write_bytes_once

logger = logging.getLogger(__name__)

FetchResult = tuple[int, str, Mapping[str, str]]
Fetch = Callable[[str], Awaitable[FetchResult]]


class RateLimited(Exception):  # noqa: N818 - spec-mandated raw-first journal name
    """HTTP 418/429; carries ``retry_after_s`` parsed from ``Retry-After`` when numeric."""

    def __init__(self, status: int, retry_after_s: float | None = None) -> None:
        """Record the HTTP status and the parsed retry delay, if any."""
        super().__init__(f"rate limited with status {status}")
        self.status = status
        self.retry_after_s = retry_after_s


class RateGate:
    """Shared embargo: ``block(now, retry_after_s)`` sets ``until = now + max(retry_after_s or 0, cooldown_s)``, never shortening."""

    def __init__(self, cooldown_s: float) -> None:
        """Bind the gate to its minimum embargo duration."""
        self._cooldown_s = cooldown_s
        self._until: float | None = None

    def block(self, now: float, retry_after_s: float | None) -> float:
        """Extend the embargo from ``now``; return the new embargo end. Never shortens."""
        extra = retry_after_s if retry_after_s is not None and retry_after_s > 0 else 0.0
        until = now + max(extra, self._cooldown_s)
        if self._until is None or until > self._until:
            self._until = until
        return self._until

    def blocked_until(self, now: float) -> float | None:
        """Return the embargo end when ``now`` is still embargoed, else None."""
        if self._until is not None and now < self._until:
            return self._until
        return None


def floor_grid(recv_ns: int, interval_s: int) -> int:
    """Return the grid label (epoch ns) at or before ``recv_ns`` on the UTC-epoch-aligned grid."""
    step_ns = interval_s * 1_000_000_000
    return recv_ns // step_ns * step_ns


def grid_iso_label(grid_ns: int) -> str:
    """Return the ISO UTC label (``Z`` suffix) of a grid point in epoch ns."""
    return datetime.fromtimestamp(grid_ns / 1_000_000_000, tz=UTC).isoformat().replace("+00:00", "Z")


def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                delay = float(value)
            except (TypeError, ValueError):
                return None
            return delay if delay > 0 else 0.0
    return None


def _short_error(exc: BaseException, limit: int = 200) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:limit] if len(text) > limit else text


@dataclass
class RestStatus:
    """First/last success times and consecutive failure count of one REST stream."""

    first_ok_at_ns: int | None = None
    last_ok_at_ns: int | None = None
    consecutive_failures: int = 0
    dropped_records: int = 0


class GridSampler:
    """Samples one all-symbol REST endpoint on a fixed UTC grid and journals raw responses.

    For each grid point it records exactly one outcome. The outcome is a ``rest`` record (any
    HTTP status with a body, including 4xx/5xx, so the normalizer sees what the venue said) or a
    ``rest_error`` record (transport failure, timeout, embargo skip, or lag skip). The response
    body is never parsed here: validation lives in the normalizer, so a validation defect can
    never destroy a sample.
    """

    def __init__(
        self,
        *,
        stream: str,
        url: str,
        interval_s: int,
        config: CaptureConfig,
        writer: SegmentWriter,
        fetch: Fetch,
        gate: RateGate,
        status: RestStatus,
        clock_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        shutdown: Callable[[], bool],
    ) -> None:
        """Bind the sampler to its endpoint, journal writer and shared rate gate."""
        self._stream = stream
        self._url = url
        self._interval_s = interval_s
        self._config = config
        self._writer = writer
        self._fetch = fetch
        self._gate = gate
        self._status = status
        self._clock_ns = clock_ns
        self._sleep = sleep
        self._shutdown = shutdown

    def _flush_best_effort(self) -> None:
        try:
            self._writer.flush()
        except OSError as exc:
            logger.warning(
                "[DATA] stage=capture stream=%s status=FLUSH_FAILED error=%s",
                self._stream,
                _short_error(exc),
            )
            overflow = self._writer.pending() - self._config.rest_max_pending_records
            if overflow > 0:
                dropped = self._writer.discard_oldest(overflow)
                self._status.dropped_records += dropped
                logger.error(
                    "[DATA] stage=capture stream=%s status=BUFFER_OVERFLOW dropped=%d total_dropped=%d",
                    self._stream,
                    dropped,
                    self._status.dropped_records,
                )

    def _write_skip(self, grid_ns: int, cause: str) -> None:
        self._writer.add(
            {
                "v": RECORD_VERSION,
                "stream": self._stream,
                "slot": self._writer.slot,
                "kind": "rest_error",
                "recv_ns": self._clock_ns(),
                "grid": grid_iso_label(grid_ns),
                "status": None,
                "error": f"skipped:{cause}",
            }
        )

    def _write_rest_error(self, grid_ns: int, status_code: int | None, error: str) -> None:
        self._writer.add(
            {
                "v": RECORD_VERSION,
                "stream": self._stream,
                "slot": self._writer.slot,
                "kind": "rest_error",
                "recv_ns": self._clock_ns(),
                "grid": grid_iso_label(grid_ns),
                "status": status_code,
                "error": error,
            }
        )
        self._status.consecutive_failures += 1
        self._flush_best_effort()

    def _write_rest(self, grid_ns: int, status_code: int, body: str) -> None:
        recv_ns = self._clock_ns()
        self._writer.add(
            {
                "v": RECORD_VERSION,
                "stream": self._stream,
                "slot": self._writer.slot,
                "kind": "rest",
                "recv_ns": recv_ns,
                "grid": grid_iso_label(grid_ns),
                "status": status_code,
                "body": body,
            }
        )
        if 200 <= status_code < 300:
            if self._status.first_ok_at_ns is None:
                self._status.first_ok_at_ns = recv_ns
            self._status.last_ok_at_ns = recv_ns
            self._status.consecutive_failures = 0
        else:
            self._status.consecutive_failures += 1
        self._flush_best_effort()

    async def _attempt(self) -> FetchResult:
        result = await self._fetch(self._url)
        status_code, _body, headers = result
        if status_code in (418, 429):
            raise RateLimited(status_code, _parse_retry_after(headers))
        return result

    async def _sample_slot(self, grid_ns: int) -> None:
        lag_ns = int(self._config.grid_max_start_lag_s * 1_000_000_000)
        retry_delay_s = self._config.grid_retry_delay_s
        try:
            status_code, body, _headers = await self._attempt()
        except RateLimited as exc:
            self._gate.block(self._clock_ns() / 1_000_000_000, exc.retry_after_s)
            self._write_rest_error(grid_ns, exc.status, f"rate_limited:{exc.status}")
            logger.warning(
                "[DATA] stage=capture stream=%s status=RATE_LIMITED http_status=%s",
                self._stream,
                exc.status,
            )
            return
        except Exception as exc:  # noqa: BLE001 - transport failure becomes a rest_error record
            await self._sleep(retry_delay_s)
            if self._shutdown() or self._clock_ns() - grid_ns > lag_ns:
                self._write_rest_error(grid_ns, None, _short_error(exc))
                return
            try:
                status_code, body, _headers = await self._attempt()
            except RateLimited as limited:
                self._gate.block(self._clock_ns() / 1_000_000_000, limited.retry_after_s)
                self._write_rest_error(grid_ns, limited.status, f"rate_limited:{limited.status}")
                return
            except Exception as retry_exc:  # noqa: BLE001 - retry failure becomes a rest_error record
                self._write_rest_error(grid_ns, None, _short_error(retry_exc))
                return
        self._write_rest(grid_ns, status_code, body)

    async def run(self) -> None:
        """Sample the endpoint on its grid until shutdown; never sample a past grid late."""
        interval_ns = self._interval_s * 1_000_000_000
        lag_ns = int(self._config.grid_max_start_lag_s * 1_000_000_000)
        target = (self._clock_ns() // interval_ns + 1) * interval_ns
        while not self._shutdown():
            now_ns = self._clock_ns()
            if now_ns < target:
                await self._sleep((target - now_ns) / 1_000_000_000)
                continue
            now_ns = self._clock_ns()
            grid_ns = floor_grid(now_ns, self._interval_s)
            skipped: list[int] = []
            point = target
            while point < grid_ns:
                skipped.append(point)
                point += interval_ns
            if self._gate.blocked_until(now_ns / 1_000_000_000) is not None:
                skipped.append(grid_ns)
                for label in skipped:
                    self._write_skip(label, "rate_limited")
                self._flush_best_effort()
                target = (grid_ns // interval_ns + 1) * interval_ns
                continue
            if now_ns - grid_ns > lag_ns:
                skipped.append(grid_ns)
                for label in skipped:
                    self._write_skip(label, "lag")
                self._flush_best_effort()
                target = (grid_ns // interval_ns + 1) * interval_ns
                continue
            for label in skipped:
                self._write_skip(label, "late")
            if skipped:
                self._flush_best_effort()
            await self._sample_slot(grid_ns)
            target = (grid_ns // interval_ns + 1) * interval_ns


def _cutoff_reached(now_ns: int, cutoff: str) -> bool:
    hour, minute = int(cutoff[:2]), int(cutoff[3:5])
    moment = datetime.fromtimestamp(now_ns / 1_000_000_000, tz=UTC)
    return (moment.hour, moment.minute) >= (hour, minute)


def _utc_day(now_ns: int) -> str:
    return datetime.fromtimestamp(now_ns / 1_000_000_000, tz=UTC).strftime("%Y%m%d")


class ReferenceCapture:
    """Captures each reference endpoint once per UTC day after the cutoff, byte-exact.

    Each endpoint succeeds or fails independently. Output is
    ``<capture_root>/reference/<name>/<YYYYMMDD>.json.gz`` (gzip of the exact response bytes)
    via ``journal.write_bytes_once``, so two overlapping slots produce one file and a day
    already on disk is never rewritten.
    """

    def __init__(
        self,
        *,
        config: CaptureConfig,
        capture_root: Path,
        fetch: Fetch,
        gate: RateGate,
        clock_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        shutdown: Callable[[], bool],
    ) -> None:
        """Bind the daily reference capture to its root, fetcher and shared rate gate."""
        self._config = config
        self._capture_root = capture_root
        self._fetch = fetch
        self._gate = gate
        self._clock_ns = clock_ns
        self._sleep = sleep
        self._shutdown = shutdown

    def _dest(self, name: str, day: str) -> Path:
        return self._capture_root / "reference" / name / f"{day}.json.gz"

    async def _attempt_endpoint(self, name: str, url: str, day: str) -> bool:
        try:
            status_code, body, headers = await self._fetch(url)
        except RateLimited as exc:
            self._gate.block(self._clock_ns() / 1_000_000_000, exc.retry_after_s)
            return False
        except Exception as exc:  # noqa: BLE001 - endpoint failure retried later
            logger.warning(
                "[DATA] stage=capture stream=reference endpoint=%s status=FAILED error=%s",
                name,
                _short_error(exc),
            )
            return False
        if status_code in (418, 429):
            self._gate.block(self._clock_ns() / 1_000_000_000, _parse_retry_after(headers))
            return False
        if not 200 <= status_code < 300 or not body:
            return False
        recv_day = _utc_day(self._clock_ns())
        dest = self._dest(name, recv_day)
        try:
            write_bytes_once(dest, gzip.compress(body.encode("utf-8"), compresslevel=6))
        except OSError as exc:
            logger.warning(
                "[DATA] stage=capture stream=reference endpoint=%s status=WRITE_FAILED error=%s",
                name,
                _short_error(exc),
            )
            return False
        logger.info("[DATA] stage=capture stream=reference endpoint=%s day=%s status=OK", name, recv_day)
        return True

    async def run(self) -> None:
        """Capture every reference endpoint once per day until shutdown."""
        done_day = ""
        done: set[str] = set()
        next_retry_s: dict[str, float] = {}
        poll_s = 5.0
        while not self._shutdown():
            now_ns = self._clock_ns()
            now_s = now_ns / 1_000_000_000
            day = _utc_day(now_ns)
            if day != done_day:
                done_day = day
                done = set()
                next_retry_s = {}
            if not _cutoff_reached(now_ns, self._config.reference_capture_after_utc):
                await self._sleep(poll_s)
                continue
            if self._gate.blocked_until(now_s) is not None:
                await self._sleep(poll_s)
                continue
            for name, url in self._config.reference_urls:
                if name in done or self._shutdown():
                    continue
                if self._dest(name, day).exists():
                    done.add(name)
                    continue
                if next_retry_s.get(name, 0.0) > now_s:
                    continue
                if await self._attempt_endpoint(name, url, day):
                    done.add(name)
                else:
                    next_retry_s[name] = now_s + self._config.reference_retry_interval_s
            await self._sleep(poll_s)


__all__ = [
    "Fetch",
    "FetchResult",
    "GridSampler",
    "RateGate",
    "RateLimited",
    "ReferenceCapture",
    "RestStatus",
    "floor_grid",
    "grid_iso_label",
]
