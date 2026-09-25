"""Background watchdog over the market recorder heartbeat, living in the live daemon process.

The recorder holds no alert credentials (and cannot report its own death), so this thread runs
alongside the trading daemon, evaluates the heartbeat file every interval, and alerts through the
injected alert callable on episode start and on recovery. Every internal failure is logged and
swallowed: the watchdog must never disturb the trading loop.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.common.paths import LIVE_CAPTURE_DIR
from src.live.settings import LiveSettings
from src.market_data.streams.recorder import HEARTBEAT_NAME
from src.market_data.streams.recorder_health import (
    RecorderFinding,
    RecorderWatchThresholds,
    evaluate_recorder_heartbeat,
    read_recorder_heartbeat,
)

_logger = logging.getLogger(__name__)

RECORDER_WATCH_JOIN_TIMEOUT_S: float = 10.0


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class RecorderWatchdog:
    """Periodically evaluates the recorder heartbeat and alerts once per unhealthy episode.

    Runs in a daemon thread of the live daemon process, which owns the alert credentials, so the
    recorder needs no secrets and its death is still reported. An episode starts when a finding key
    appears that has not been alerted in the current episode; one ``recorder_unhealthy`` alert lists
    every current finding. Keys that clear are forgotten, so a recurrence alerts again. When all
    findings clear after an alerted episode a single ``recorder_recovered`` alert closes it.
    Undelivered alerts are retried on the next check. Every exception inside a check is logged and
    swallowed: the watchdog must never disturb the trading loop.
    """

    def __init__(
        self,
        *,
        heartbeat_path: Path,
        thresholds: RecorderWatchThresholds,
        interval_s: float,
        alert: Callable[[str, str], bool],
        now_fn: Callable[[], pd.Timestamp] = _utc_now,
    ) -> None:
        """
        Args:
            heartbeat_path: ``LIVE_CAPTURE_DIR / HEARTBEAT_NAME`` in production.
            thresholds: Evaluation thresholds.
            interval_s: Seconds between checks; the first check runs one interval after ``start``.
            alert: ``(event, detail) -> delivered``; must not raise for delivery failures.
            now_fn: UTC wall clock (real clock in production; the heartbeat is written by another
                process, so freshness must be measured in real elapsed time).
        """
        self._heartbeat_path = Path(heartbeat_path)
        self._thresholds = thresholds
        self._interval_s = float(interval_s)
        self._alert = alert
        self._now_fn = now_fn
        self._watch_started_at = now_fn()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._last_keys: frozenset[str] = frozenset()
        self._alerted: set[str] = set()
        self._episode_open = False

    def check_once(self) -> tuple[RecorderFinding, ...]:
        """Evaluate the heartbeat now, send any due alert, and return the current findings."""
        with self._state_lock:
            try:
                payload = read_recorder_heartbeat(self._heartbeat_path)
                findings = evaluate_recorder_heartbeat(
                    payload,
                    now=self._now_fn(),
                    watch_started_at=self._watch_started_at,
                    thresholds=self._thresholds,
                )
            except Exception as exc:  # noqa: BLE001
                _logger.error("[DATA] stage=recorder_watch status=CHECK_FAILED error=%s", exc, exc_info=True)
                return ()
            keys = frozenset(finding.key for finding in findings)
            if keys != self._last_keys:
                self._last_keys = keys
                if keys:
                    _logger.warning(
                        "[DATA] stage=recorder_watch status=UNHEALTHY findings=%s", ",".join(sorted(keys))
                    )
            self._alerted &= set(keys)
            new_keys = set(keys) - self._alerted
            if new_keys:
                detail = "; ".join(f"{finding.key} {finding.detail}" for finding in findings)
                try:
                    delivered = self._alert("recorder_unhealthy", detail)
                except Exception as exc:  # noqa: BLE001
                    _logger.error("[DATA] stage=recorder_watch status=ALERT_FAILED error=%s", exc, exc_info=True)
                    return findings
                if delivered:
                    self._alerted = set(keys)
                    self._episode_open = True
                return findings
            if not keys and self._episode_open:
                try:
                    delivered = self._alert("recorder_recovered", "all recorder checks healthy")
                except Exception as exc:  # noqa: BLE001
                    _logger.error("[DATA] stage=recorder_watch status=ALERT_FAILED error=%s", exc, exc_info=True)
                    return findings
                if delivered:
                    self._episode_open = False
                    _logger.info("[DATA] stage=recorder_watch status=RECOVERED")
            return findings

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self.check_once()

    def start(self) -> None:
        """Start the background daemon thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="recorder-watch", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = RECORDER_WATCH_JOIN_TIMEOUT_S) -> None:
        """Signal the thread to stop and join it for at most ``timeout_s`` seconds."""
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout_s)


def build_recorder_watchdog(
    settings: LiveSettings, *, alert: Callable[[str, str], bool]
) -> RecorderWatchdog | None:
    """Construct the production watchdog from ``LiveSettings``, or ``None`` when disabled.

    Returns:
        A watchdog reading ``LIVE_CAPTURE_DIR / HEARTBEAT_NAME`` with thresholds and interval taken
        from the ``recorder_*`` settings; ``None`` when ``recorder_watch_enabled`` is false.
    """
    if not settings.recorder_watch_enabled:
        return None
    thresholds = RecorderWatchThresholds(
        heartbeat_stale_s=settings.recorder_heartbeat_stale_s,
        liquidation_silence_s=settings.recorder_liquidation_silence_s,
        liquidation_max_failed_connections=settings.recorder_liquidation_max_failed_connections,
        sampler_stale_s=settings.recorder_sampler_stale_s,
    )
    return RecorderWatchdog(
        heartbeat_path=LIVE_CAPTURE_DIR / HEARTBEAT_NAME,
        thresholds=thresholds,
        interval_s=settings.recorder_watch_interval_s,
        alert=alert,
    )
