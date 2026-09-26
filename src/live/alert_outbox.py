"""Durable alert outbox shared by the daemon and the host liveness checker."""

from __future__ import annotations

import fcntl
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from src.common.paths import DATA_DIR

logger = logging.getLogger("LiveAlertOutbox")

AlertChannel = Literal["webhook", "email"]

NOTICE_SEVERITY = "NOTICE"


def default_dedupe_key(event: str, decision_time: pd.Timestamp | None) -> str:
    """Return the default dedupe key for an alert."""
    if decision_time is None:
        return f"{event}:none"
    try:
        return f"{event}:{pd.Timestamp(decision_time).isoformat()}"
    except Exception:
        return f"{event}:{decision_time}"


def resolve_outbox_path(settings: Any) -> Path:
    """Return the outbox file path for ``settings``."""
    raw = getattr(settings, "alert_outbox_path", None)
    if raw:
        return Path(raw)
    return DATA_DIR / "state" / "alert_outbox.json"


def _as_utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _parse_ts(raw: Any) -> pd.Timestamp | None:
    if raw is None:
        return None
    try:
        ts = pd.Timestamp(raw)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class AlertRecord:
    dedupe_key: str
    event: str
    severity: str
    detail: str
    decision_time: pd.Timestamp | None
    created_at: pd.Timestamp
    pending_channels: frozenset[AlertChannel]
    delivered_channels: frozenset[AlertChannel]
    attempts: int
    next_attempt_at: pd.Timestamp
    completed_at: pd.Timestamp | None
    expired: bool


@dataclass(frozen=True, slots=True)
class DrainReport:
    attempted: int
    completed: int
    expired: int
    pending: int


def _record_to_json(record: AlertRecord) -> dict[str, Any]:
    return {
        "dedupe_key": record.dedupe_key,
        "event": record.event,
        "severity": record.severity,
        "detail": record.detail,
        "decision_time": record.decision_time.isoformat() if record.decision_time is not None else None,
        "created_at": record.created_at.isoformat(),
        "pending_channels": sorted(record.pending_channels),
        "delivered_channels": sorted(record.delivered_channels),
        "attempts": record.attempts,
        "next_attempt_at": record.next_attempt_at.isoformat(),
        "completed_at": record.completed_at.isoformat() if record.completed_at is not None else None,
        "expired": record.expired,
    }


def _record_from_json(raw: dict[str, Any]) -> AlertRecord | None:
    try:
        created_at = _parse_ts(raw.get("created_at"))
        next_attempt_at = _parse_ts(raw.get("next_attempt_at"))
        if created_at is None or next_attempt_at is None:
            return None
        return AlertRecord(
            dedupe_key=str(raw["dedupe_key"]),
            event=str(raw["event"]),
            severity=str(raw.get("severity", "CRITICAL")),
            detail=str(raw.get("detail", "")),
            decision_time=_parse_ts(raw.get("decision_time")),
            created_at=created_at,
            pending_channels=cast("frozenset[AlertChannel]", frozenset(str(c) for c in raw.get("pending_channels", []))),
            delivered_channels=cast("frozenset[AlertChannel]", frozenset(str(c) for c in raw.get("delivered_channels", []))),
            attempts=int(raw.get("attempts", 0)),
            next_attempt_at=next_attempt_at,
            completed_at=_parse_ts(raw.get("completed_at")),
            expired=bool(raw.get("expired", False)),
        )
    except Exception:
        return None


class AlertOutbox:
    """Durable, lock-serialized alert queue shared by the daemon and the host liveness checker.

    State is one JSON document written atomically (temp file, fsync, os.replace, directory fsync)
    under an exclusive ``fcntl.flock`` on a sidecar ``<path>.lock``. The lock makes the daemon
    process, its pulse thread and the one-shot liveness container safe writers of the same file
    over a bind mount. Delivery is at-least-once per channel: a crash between a successful send and
    the state write re-sends that channel once; duplicates are preferred over loss.
    """

    def __init__(
        self,
        path: Path,
        *,
        retry_backoff_s: float,
        retry_backoff_max_s: float,
        max_age_s: float,
        retention_s: float,
        max_records: int,
    ) -> None:
        self._path = Path(path)
        self._retry_backoff_s = float(retry_backoff_s)
        self._retry_backoff_max_s = float(retry_backoff_max_s)
        self._max_age_s = float(max_age_s)
        self._retention_s = float(retention_s)
        self._max_records = int(max_records)

    @classmethod
    def from_settings(cls, path: Path, settings: Any) -> AlertOutbox:
        """Build an outbox from live settings."""
        return cls(
            path,
            retry_backoff_s=float(getattr(settings, "alert_retry_backoff_s", 60.0)),
            retry_backoff_max_s=float(getattr(settings, "alert_retry_backoff_max_s", 1800.0)),
            max_age_s=float(getattr(settings, "alert_outbox_max_age_s", 259200.0)),
            retention_s=float(getattr(settings, "alert_outbox_retention_s", 604800.0)),
            max_records=int(getattr(settings, "alert_outbox_max_records", 1000)),
        )

    def _lock_path(self) -> Path:
        return self._path.with_name(self._path.name + ".lock")

    def _load_locked(self) -> list[AlertRecord]:
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except OSError as exc:
            logger.error("[SYS] stage=alert_outbox status=READ_FAILED error=%s", exc)
            return []
        try:
            raw = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._quarantine_locked(exc)
            return []
        if not isinstance(raw, dict) or not isinstance(raw.get("records"), list):
            self._quarantine_locked(ValueError("unexpected outbox schema"))
            return []
        records: list[AlertRecord] = []
        for item in raw["records"]:
            if not isinstance(item, dict):
                continue
            record = _record_from_json(item)
            if record is not None:
                records.append(record)
        return records

    def _quarantine_locked(self, exc: BaseException) -> None:
        try:
            stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
            target = self._path.with_name(f"{self._path.name}.corrupt-{stamp}")
            if self._path.exists():
                os.replace(self._path, target)
            logger.error("[SYS] stage=alert_outbox status=CORRUPT_QUARANTINED path=%s error=%s", target, exc)
        except Exception:  # noqa: BLE001
            logger.exception("[SYS] stage=alert_outbox status=QUARANTINE_FAILED")

    def _write_locked(self, records: list[AlertRecord]) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps({"records": [_record_to_json(r) for r in records]}, sort_keys=True)
            tmp_path = self._path.with_name(self._path.name + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self._path)
            try:
                dir_fd = os.open(str(self._path.parent), os.O_RDONLY)
            except OSError:
                return True
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("[SYS] stage=alert_outbox status=WRITE_FAILED error=%s", exc)
            return False

    def _prune(self, records: list[AlertRecord], now: pd.Timestamp) -> list[AlertRecord]:
        cutoff = now - pd.Timedelta(seconds=self._retention_s)
        kept: list[AlertRecord] = []
        for record in records:
            if (record.completed_at is not None or record.expired) and record.created_at < cutoff:
                continue
            kept.append(record)
        return kept

    def _backoff_delay_s(self, attempts: int) -> float:
        return min(self._retry_backoff_s * (2.0 ** max(0, attempts - 1)), self._retry_backoff_max_s)

    def enqueue(
        self,
        *,
        event: str,
        detail: str,
        decision_time: pd.Timestamp | None,
        dedupe_key: str,
        channels: frozenset[AlertChannel],
        now: pd.Timestamp,
    ) -> bool:
        """Durably accept an alert.

        Returns:
            True when the alert is (or already was) in the outbox under ``dedupe_key``, i.e. the
            caller may treat it as handed off. False only when it could not be persisted.

        Raises:
            Never; persistence failures are logged at ERROR and reported as False.
        """
        try:
            now_utc = _as_utc(pd.Timestamp(now))
        except Exception:
            logger.error("[SYS] stage=alert_outbox status=ENQUEUE_REJECTED event=%s", event)
            return False
        try:
            from src.live.alerting import event_severity

            severity = event_severity(event)
        except Exception:
            severity = "CRITICAL"
        lock_path = self._lock_path()
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    records = self._load_locked()
                    for record in records:
                        if record.dedupe_key == dedupe_key:
                            return True
                    if not channels:
                        expired = AlertRecord(
                            dedupe_key=dedupe_key,
                            event=event,
                            severity=severity,
                            detail=detail,
                            decision_time=pd.Timestamp(decision_time) if decision_time is not None else None,
                            created_at=now_utc,
                            pending_channels=frozenset(),
                            delivered_channels=frozenset(),
                            attempts=0,
                            next_attempt_at=now_utc,
                            completed_at=now_utc,
                            expired=True,
                        )
                        records.append(expired)
                        records = self._prune(records, now_utc)
                        ok = self._write_locked(records)
                        logger.error("[SYS] stage=alert_outbox status=NO_CHANNEL event=%s dedupe_key=%s", event, dedupe_key)
                        return ok
                    records = self._prune(records, now_utc)
                    if len(records) >= self._max_records:
                        completed = [r for r in records if r.completed_at is not None or r.expired]
                        completed.sort(key=lambda r: r.created_at)
                        while completed and len(records) >= self._max_records:
                            oldest = completed.pop(0)
                            records = [r for r in records if r.dedupe_key != oldest.dedupe_key]
                    if len(records) >= self._max_records:
                        if severity == NOTICE_SEVERITY:
                            logger.error("[SYS] stage=alert_outbox status=OVERFLOW_REJECTED event=%s dedupe_key=%s", event, dedupe_key)
                            return False
                        overflow_key = f"alert_outbox_overflow:{now_utc.strftime('%Y-%m-%d')}"
                        if not any(r.dedupe_key == overflow_key for r in records):
                            overflow = AlertRecord(
                                dedupe_key=overflow_key,
                                event="alert_outbox_overflow",
                                severity="CRITICAL",
                                detail=f"outbox at capacity records={len(records)}",
                                decision_time=None,
                                created_at=now_utc,
                                pending_channels=frozenset(channels),
                                delivered_channels=frozenset(),
                                attempts=0,
                                next_attempt_at=now_utc,
                                completed_at=None,
                                expired=False,
                            )
                            records.append(overflow)
                            logger.error("[SYS] stage=alert_outbox status=OVERFLOW event=%s dedupe_key=%s", event, dedupe_key)
                    record = AlertRecord(
                        dedupe_key=dedupe_key,
                        event=event,
                        severity=severity,
                        detail=detail,
                        decision_time=pd.Timestamp(decision_time) if decision_time is not None else None,
                        created_at=now_utc,
                        pending_channels=frozenset(channels),
                        delivered_channels=frozenset(),
                        attempts=0,
                        next_attempt_at=now_utc,
                        completed_at=None,
                        expired=False,
                    )
                    records.append(record)
                    return self._write_locked(records)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except Exception as exc:  # noqa: BLE001
            logger.error("[SYS] stage=alert_outbox status=ENQUEUE_FAILED event=%s error=%s", event, exc)
            return False

    def drain(
        self,
        deliver: Callable[[AlertRecord, AlertChannel], bool],
        *,
        now: pd.Timestamp,
        blocking: bool,
    ) -> DrainReport:
        """Attempt every due pending channel once, then persist the outcome.

        Args:
            deliver: Sends one record on one channel; True on confirmed delivery. Must not raise.
            blocking: False makes the call return an empty report immediately when another
                process holds the lock (used by the heartbeat pulse thread).
        """
        try:
            now_utc = _as_utc(pd.Timestamp(now))
        except Exception:
            return DrainReport(attempted=0, completed=0, expired=0, pending=0)
        lock_path = self._lock_path()
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "a+", encoding="utf-8") as lock_handle:
                flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if not blocking else 0)
                try:
                    fcntl.flock(lock_handle.fileno(), flags)
                except BlockingIOError:
                    return DrainReport(attempted=0, completed=0, expired=0, pending=0)
                try:
                    records = self._load_locked()
                    due = [r for r in records if not r.expired and r.completed_at is None and r.pending_channels and r.next_attempt_at <= now_utc]
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001
            logger.error("[SYS] stage=alert_outbox status=DRAIN_LOAD_FAILED")
            return DrainReport(attempted=0, completed=0, expired=0, pending=0)
        if not due:
            return self._finalize(now_utc)
        outcomes: dict[str, frozenset[AlertChannel]] = {}
        attempted = 0
        for record in due:
            delivered: set[AlertChannel] = set()
            for channel in sorted(record.pending_channels):
                attempted += 1
                try:
                    ok = bool(deliver(record, channel))
                except Exception:  # noqa: BLE001
                    ok = False
                if ok:
                    delivered.add(channel)
            outcomes[record.dedupe_key] = frozenset(delivered)
        try:
            with open(lock_path, "a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    records = self._load_locked()
                    by_key = {r.dedupe_key: r for r in records}
                    for key, newly in outcomes.items():
                        current = by_key.get(key)
                        if current is None or current.expired or current.completed_at is not None:
                            continue
                        merged_delivered = set(current.delivered_channels) | set(newly)
                        remaining = set(current.pending_channels) - set(newly)
                        attempts = current.attempts + 1
                        if not remaining:
                            updated = AlertRecord(
                                dedupe_key=current.dedupe_key,
                                event=current.event,
                                severity=current.severity,
                                detail=current.detail,
                                decision_time=current.decision_time,
                                created_at=current.created_at,
                                pending_channels=frozenset(),
                                delivered_channels=frozenset(merged_delivered),
                                attempts=attempts,
                                next_attempt_at=current.next_attempt_at,
                                completed_at=now_utc,
                                expired=False,
                            )
                        else:
                            delay = self._backoff_delay_s(attempts)
                            updated = AlertRecord(
                                dedupe_key=current.dedupe_key,
                                event=current.event,
                                severity=current.severity,
                                detail=current.detail,
                                decision_time=current.decision_time,
                                created_at=current.created_at,
                                pending_channels=frozenset(remaining),
                                delivered_channels=frozenset(merged_delivered),
                                attempts=attempts,
                                next_attempt_at=now_utc + pd.Timedelta(seconds=delay),
                                completed_at=None,
                                expired=False,
                            )
                        by_key[key] = updated
                    merged = list(by_key.values())
                    self._write_locked(self._prune(merged, now_utc))
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001
            logger.error("[SYS] stage=alert_outbox status=DRAIN_WRITE_FAILED")
        report = self._finalize(now_utc)
        return DrainReport(attempted=attempted, completed=report.completed, expired=report.expired, pending=report.pending)

    def _finalize(self, now: pd.Timestamp) -> DrainReport:
        try:
            lock_path = self._lock_path()
            with open(lock_path, "a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    records = self._load_locked()
                    expired_now = 0
                    changed = False
                    updated: list[AlertRecord] = []
                    for record in records:
                        if (
                            not record.expired
                            and record.completed_at is None
                            and record.pending_channels
                            and (now - record.created_at).total_seconds() > self._max_age_s
                        ):
                            logger.error(
                                "[SYS] stage=alert_outbox status=EXPIRED event=%s dedupe_key=%s undelivered=%s detail=%s",
                                record.event,
                                record.dedupe_key,
                                sorted(record.pending_channels),
                                record.detail,
                            )
                            updated.append(
                                AlertRecord(
                                    dedupe_key=record.dedupe_key,
                                    event=record.event,
                                    severity=record.severity,
                                    detail=record.detail,
                                    decision_time=record.decision_time,
                                    created_at=record.created_at,
                                    pending_channels=frozenset(),
                                    delivered_channels=record.delivered_channels,
                                    attempts=record.attempts,
                                    next_attempt_at=record.next_attempt_at,
                                    completed_at=now,
                                    expired=True,
                                )
                            )
                            expired_now += 1
                            changed = True
                        else:
                            updated.append(record)
                    if changed:
                        self._write_locked(self._prune(updated, now))
                        records = self._prune(updated, now)
                    completed = sum(1 for r in records if r.completed_at is not None and not r.expired)
                    expired_total = sum(1 for r in records if r.expired)
                    pending = sum(1 for r in records if not r.expired and r.completed_at is None)
                    return DrainReport(attempted=0, completed=completed, expired=expired_now or expired_total, pending=pending)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001
            logger.error("[SYS] stage=alert_outbox status=FINALIZE_FAILED")
            return DrainReport(attempted=0, completed=0, expired=0, pending=0)
