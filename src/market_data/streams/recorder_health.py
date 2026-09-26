"""Pure evaluation of the normalizer heartbeat v3 for the daemon watchdog.

The normalizer (another process/container) publishes ``recorder_heartbeat.json``; this module
turns one decoded payload into health findings without any I/O or clock reads, so the logic
is unit-testable and the watchdog thread stays thin. Capture health is judged from the slot
heartbeats embedded by the normalizer, so a dead normalizer surfaces as ``heartbeat_stale``
while a dead capture with a live normalizer surfaces as ``capture_missing``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

RecorderCheck = Literal[
    "heartbeat_missing", "heartbeat_stale", "heartbeat_schema",
    "capture_missing", "capture_not_ready", "capture_dual_active", "capture_rest_failing",
    "capture_ws_silent", "capture_flush_failing",
    "sampler_stale", "sampler_degraded", "sampler_rejecting",
    "liquidation_unpersisted",
    "normalizer_lagging", "normalizer_failing",
    "compaction_failed", "compaction_overdue",
    "prune_blocked",
    "reference_missing",
]

_SAMPLER_DATASETS: tuple[str, str] = ("book_ticker", "premium_index")


@dataclass(frozen=True, slots=True)
class RecorderWatchThresholds:
    """Alert thresholds for one evaluation of the heartbeat (all durations in seconds)."""

    heartbeat_stale_s: float
    liquidation_silence_s: float
    sampler_stale_s: float
    sampler_max_consecutive_failures: int
    min_capture_ratio: float
    capture_ratio_min_points: int
    persist_stale_s: float
    max_consecutive_flush_failures: int
    reference_grace_s: float
    rejected_fraction_alert: float
    rejected_max_consecutive_points: int
    capture_stale_s: float
    capture_ready_grace_s: float
    capture_dual_active_max_s: float
    normalizer_max_lag_s: float
    normalizer_max_consecutive_failures: int
    compaction_max_delay_s: float


@dataclass(frozen=True, slots=True)
class RecorderFinding:
    """One failed health check.

    Attributes:
        check: Stable check id.
        subject: ``"recorder"``, ``"capture"``, a slot (``"blue"``/``"green"``), a dataset
            (``"book_ticker"``/``"premium_index"``), ``"force_order"``, ``"normalizer"``,
            ``"compaction"``, ``"retention"`` or ``"reference"``.
        detail: Space-separated ``key=value`` facts (ages in whole seconds, ISO timestamps).
    """

    check: RecorderCheck
    subject: str
    detail: str

    @property
    def key(self) -> str:
        """``"<check>:<subject>"`` — identity used for alert de-duplication."""
        return f"{self.check}:{self.subject}"


def read_recorder_heartbeat(path: Path) -> Mapping[str, Any] | None:
    """Load the recorder heartbeat JSON object.

    Returns:
        The decoded object, or ``None`` when the file is missing, unreadable, not valid JSON, or not a
        JSON object (the writer replaces it atomically, so a partial read is not expected).
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _require_aware(value: pd.Timestamp, label: str) -> pd.Timestamp:
    out = pd.Timestamp(value)
    if out.tzinfo is None:
        raise ValueError(f"{label} must be tz-aware UTC")
    return out.tz_convert("UTC")


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, pd.Timestamp):
        out = value
    elif isinstance(value, str):
        try:
            out = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
    elif isinstance(value, datetime):
        out = pd.Timestamp(value)
    else:
        return None
    if pd.isna(out) or out.tzinfo is None:
        return None
    return out.tz_convert("UTC")


def _parse_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _parse_ratio(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _parse_cutoff(value: Any) -> tuple[int, int] | None:
    if isinstance(value, bool) or not isinstance(value, str):
        return None
    match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", value)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _fmt_threshold(value: float) -> str:
    return f"{value:g}"


def _fresh_slots(
    capture: Any, *, now_utc: pd.Timestamp, stale_s: float
) -> dict[str, Mapping[str, Any]]:
    """Slot heartbeats with ``ts`` newer than the staleness threshold."""
    fresh: dict[str, Mapping[str, Any]] = {}
    if not isinstance(capture, Mapping):
        return fresh
    for slot, entry in capture.items():
        if not isinstance(entry, Mapping):
            continue
        ts = _parse_ts(entry.get("ts"))
        if ts is not None and (now_utc - ts).total_seconds() <= stale_s:
            fresh[str(slot)] = entry
    return fresh


def _slot_ready(entry: Mapping[str, Any]) -> bool:
    """Spec 01 READY: first ok and first frame at or after start, no flush failures."""
    started_at = _parse_ts(entry.get("started_at"))
    rest = entry.get("rest")
    ws = entry.get("ws")
    if started_at is None or not isinstance(rest, Mapping) or not isinstance(ws, Mapping):
        return False
    book = rest.get("book_ticker")
    if not isinstance(book, Mapping):
        return False
    first_ok = _parse_ts(book.get("first_ok_at"))
    first_frame = _parse_ts(ws.get("first_frame_at"))
    failures = _parse_count(entry.get("flush_failures"))
    return (
        first_ok is not None
        and first_ok >= started_at
        and first_frame is not None
        and first_frame >= started_at
        and failures == 0
    )


def _capture_findings(
    capture: Any, *, now_utc: pd.Timestamp, thresholds: RecorderWatchThresholds
) -> list[RecorderFinding]:
    findings: list[RecorderFinding] = []
    fresh = _fresh_slots(capture, now_utc=now_utc, stale_s=thresholds.capture_stale_s)
    if not fresh:
        return [RecorderFinding("capture_missing", "capture", "fresh_slots=0")]
    for slot, entry in sorted(fresh.items()):
        started_at = _parse_ts(entry.get("started_at"))
        age_s = (now_utc - started_at).total_seconds() if started_at is not None else None
        if not _slot_ready(entry) and (age_s is None or age_s > thresholds.capture_ready_grace_s):
            detail = f"started_at={started_at.isoformat() if started_at is not None else 'none'}"
            findings.append(RecorderFinding("capture_not_ready", slot, detail))
        failures = _parse_count(entry.get("flush_failures"))
        if failures is None:
            findings.append(RecorderFinding("capture_flush_failing", slot, "entry=malformed"))
        elif failures > 0:
            findings.append(
                RecorderFinding("capture_flush_failing", slot, f"flush_failures={failures}")
            )
    if len(fresh) >= 2:
        starts = {slot: _parse_ts(entry.get("started_at")) for slot, entry in fresh.items()}
        if any(value is None for value in starts.values()):
            findings.append(RecorderFinding("capture_dual_active", "capture", "started_at=missing"))
        else:
            latest = max(value for value in starts.values() if value is not None)
            assert latest is not None
            if (now_utc - latest).total_seconds() > thresholds.capture_dual_active_max_s:
                findings.append(
                    RecorderFinding(
                        "capture_dual_active", "capture",
                        f"slots={','.join(sorted(fresh))} newest_started_at={latest.isoformat()}",
                    )
                )
    for stream in _SAMPLER_DATASETS:
        failing: list[str] = []
        healthy = False
        for slot, entry in sorted(fresh.items()):
            rest = entry.get("rest")
            status = rest.get(stream) if isinstance(rest, Mapping) else None
            count = _parse_count(status.get("consecutive_failures")) if isinstance(status, Mapping) else None
            if count is None or count >= thresholds.sampler_max_consecutive_failures:
                failing.append(slot)
            else:
                healthy = True
        if failing and not healthy:
            findings.append(
                RecorderFinding(
                    "capture_rest_failing", stream,
                    f"slots={','.join(failing)} threshold={thresholds.sampler_max_consecutive_failures}",
                )
            )
    silent = True
    for _slot, entry in sorted(fresh.items()):
        ws = entry.get("ws")
        last_frame = _parse_ts(ws.get("last_frame_at")) if isinstance(ws, Mapping) else None
        if last_frame is not None:
            if (now_utc - last_frame).total_seconds() <= thresholds.liquidation_silence_s:
                silent = False
                break
            continue
        started_at = _parse_ts(entry.get("started_at"))
        if started_at is not None and (now_utc - started_at).total_seconds() <= thresholds.capture_ready_grace_s:
            silent = False
            break
    if silent:
        findings.append(RecorderFinding("capture_ws_silent", "force_order", "no fresh frame"))
    return findings


def _stream_findings(
    streams: Any, *, now_utc: pd.Timestamp, thresholds: RecorderWatchThresholds
) -> list[RecorderFinding]:
    findings: list[RecorderFinding] = []
    for dataset in _SAMPLER_DATASETS:
        entry = streams.get(dataset) if isinstance(streams, Mapping) else None
        if not isinstance(entry, Mapping):
            findings.append(RecorderFinding("sampler_stale", dataset, "entry=missing"))
            findings.append(RecorderFinding("sampler_degraded", dataset, "entry=missing"))
            findings.append(RecorderFinding("sampler_rejecting", dataset, "entry=missing"))
            continue
        persisted_at = _parse_ts(entry.get("last_persisted_at"))
        if persisted_at is None:
            findings.append(RecorderFinding("sampler_stale", dataset, "entry=malformed"))
        elif (now_utc - persisted_at).total_seconds() > thresholds.sampler_stale_s:
            findings.append(
                RecorderFinding(
                    "sampler_stale", dataset,
                    f"age_s={int((now_utc - persisted_at).total_seconds())} "
                    f"threshold_s={_fmt_threshold(thresholds.sampler_stale_s)}",
                )
            )
        expected = _parse_count(entry.get("window_expected_points"))
        captured = _parse_count(entry.get("window_captured_points"))
        if expected is None or captured is None or captured < 0 or captured > expected:
            findings.append(RecorderFinding("sampler_degraded", dataset, "entry=malformed"))
        elif expected >= thresholds.capture_ratio_min_points:
            ratio = captured / expected
            if ratio < thresholds.min_capture_ratio:
                findings.append(
                    RecorderFinding(
                        "sampler_degraded", dataset,
                        f"captured={captured} expected={expected} ratio={ratio:.3f}",
                    )
                )
        fraction = _parse_ratio(entry.get("rejected_fraction_last_sample"))
        points = _parse_count(entry.get("consecutive_rejecting_points"))
        if fraction is None or points is None:
            findings.append(RecorderFinding("sampler_rejecting", dataset, "entry=malformed"))
        elif fraction >= thresholds.rejected_fraction_alert or points >= thresholds.rejected_max_consecutive_points:
            findings.append(
                RecorderFinding(
                    "sampler_rejecting", dataset,
                    f"fraction={fraction:.4f} consecutive_points={points}",
                )
            )
    force = streams.get("force_order") if isinstance(streams, Mapping) else None
    if not isinstance(force, Mapping):
        findings.append(RecorderFinding("liquidation_unpersisted", "force_order", "entry=missing"))
    else:
        frame_at = _parse_ts(force.get("last_frame_recv_at"))
        persisted_at = _parse_ts(force.get("last_persisted_at"))
        if frame_at is None or persisted_at is None:
            findings.append(RecorderFinding("liquidation_unpersisted", "force_order", "entry=malformed"))
        elif (frame_at - persisted_at).total_seconds() > thresholds.persist_stale_s:
            findings.append(
                RecorderFinding(
                    "liquidation_unpersisted", "force_order",
                    f"lag_s={int((frame_at - persisted_at).total_seconds())} "
                    f"threshold_s={_fmt_threshold(thresholds.persist_stale_s)}",
                )
            )
    return findings


def _normalizer_findings(entry: Any, *, thresholds: RecorderWatchThresholds) -> list[RecorderFinding]:
    findings: list[RecorderFinding] = []
    if not isinstance(entry, Mapping):
        return [
            RecorderFinding("normalizer_lagging", "normalizer", "entry=missing"),
            RecorderFinding("normalizer_failing", "normalizer", "entry=missing"),
        ]
    lag = _parse_ratio(entry.get("lag_s"))
    if lag is None:
        findings.append(RecorderFinding("normalizer_lagging", "normalizer", "entry=malformed"))
    elif lag > thresholds.normalizer_max_lag_s:
        findings.append(
            RecorderFinding(
                "normalizer_lagging", "normalizer",
                f"lag_s={lag:g} threshold_s={_fmt_threshold(thresholds.normalizer_max_lag_s)}",
            )
        )
    failures = _parse_count(entry.get("consecutive_failures"))
    if failures is None:
        findings.append(RecorderFinding("normalizer_failing", "normalizer", "entry=malformed"))
    elif failures >= thresholds.normalizer_max_consecutive_failures:
        findings.append(
            RecorderFinding(
                "normalizer_failing", "normalizer",
                f"consecutive_failures={failures} threshold={thresholds.normalizer_max_consecutive_failures}",
            )
        )
    return findings


def _compaction_findings(
    entry: Any, *, now_utc: pd.Timestamp, thresholds: RecorderWatchThresholds
) -> list[RecorderFinding]:
    findings: list[RecorderFinding] = []
    if not isinstance(entry, Mapping):
        return [RecorderFinding("compaction_failed", "compaction", "entry=missing")]
    if entry.get("last_result") == "error":
        findings.append(
            RecorderFinding(
                "compaction_failed", "compaction", f"last_error={entry.get('last_error') or 'none'}"
            )
        )
    pending = entry.get("archived_days_pending")
    if not isinstance(pending, list):
        return [*findings, RecorderFinding("compaction_overdue", "compaction", "entry=malformed")]
    for item in pending:
        day = item.get("day") if isinstance(item, Mapping) else None
        try:
            day_end = pd.Timestamp(f"{str(day)[:4]}-{str(day)[4:6]}-{str(day)[6:8]}T00:00:00Z") + pd.Timedelta(days=1)
        except (TypeError, ValueError):
            findings.append(RecorderFinding("compaction_overdue", "compaction", "entry=malformed"))
            continue
        if (now_utc - day_end).total_seconds() > thresholds.compaction_max_delay_s:
            findings.append(
                RecorderFinding("compaction_overdue", "compaction", f"day={day} pending_beyond_deadline")
            )
    return findings


def _retention_findings(entry: Any) -> list[RecorderFinding]:
    if not isinstance(entry, Mapping):
        return [RecorderFinding("prune_blocked", "retention", "entry=missing")]
    blocked = entry.get("prune_blocked")
    if blocked is True:
        return [
            RecorderFinding(
                "prune_blocked", "retention", f"blocked_reason={entry.get('blocked_reason') or 'none'}"
            )
        ]
    if blocked is not True and blocked is not False:
        return [RecorderFinding("prune_blocked", "retention", "entry=malformed")]
    return []


def _reference_findings(
    payload: Mapping[str, Any],
    *,
    now_utc: pd.Timestamp,
    started_at: pd.Timestamp | None,
    thresholds: RecorderWatchThresholds,
) -> list[RecorderFinding]:
    entry = payload.get("reference")
    today = now_utc.strftime("%Y%m%d")
    yesterday = (now_utc - pd.Timedelta(days=1)).strftime("%Y%m%d")
    if not isinstance(entry, Mapping):
        return [RecorderFinding("reference_missing", "reference", "entry=malformed")]
    cutoff = _parse_cutoff(entry.get("cutoff_utc"))
    if cutoff is None:
        if started_at is not None and (now_utc - started_at).total_seconds() <= thresholds.reference_grace_s:
            return []
        return [RecorderFinding("reference_missing", "reference", "entry=malformed")]
    deadline = (
        now_utc.normalize()
        + pd.Timedelta(hours=cutoff[0], minutes=cutoff[1])
        + pd.Timedelta(seconds=thresholds.reference_grace_s)
    )
    day = entry.get("day")
    endpoints = entry.get("endpoints")
    if not isinstance(day, str) or not isinstance(endpoints, Mapping):
        if now_utc >= deadline:
            return [RecorderFinding("reference_missing", "reference", "entry=malformed")]
        return []

    def _missing_today() -> list[str]:
        missing: list[str] = []
        for name, status in endpoints.items():
            if not isinstance(status, Mapping) or status.get("captured") is not True:
                missing.append(str(name))
        return missing

    if now_utc >= deadline:
        if day != today:
            return [
                RecorderFinding(
                    "reference_missing", "reference", f"day={day} today={today} entry=stale"
                )
            ]
        missing = _missing_today()
        if missing:
            return [RecorderFinding("reference_missing", "reference", f"missing={','.join(missing)}")]
    previous_day = entry.get("previous_day")
    previous_complete = entry.get("previous_day_complete")
    if previous_day == yesterday and previous_complete is False:
        return [
            RecorderFinding(
                "reference_missing", "reference", f"previous_day={previous_day} complete=false"
            )
        ]
    if day == yesterday:
        missing = _missing_today()
        if missing:
            return [RecorderFinding("reference_missing", "reference", f"missing={','.join(missing)}")]
    return []


def evaluate_recorder_heartbeat(
    payload: Mapping[str, Any] | None,
    *,
    now: pd.Timestamp,
    watch_started_at: pd.Timestamp,
    thresholds: RecorderWatchThresholds,
) -> tuple[RecorderFinding, ...]:
    """Classify one heartbeat v3 snapshot (normalizer + embedded capture slots) into failed checks.

    Fail-closed as before: a missing or malformed entry fails its check. A schema version other than
    3 yields only ``heartbeat_schema``, and a stale heartbeat yields only ``heartbeat_stale``, because
    every other field describes the past. Capture health is judged from the slot heartbeats embedded
    by the normalizer, so a dead normalizer surfaces as ``heartbeat_stale`` while a dead capture with
    a live normalizer surfaces as ``capture_missing``.

    Args:
        payload: Decoded heartbeat or ``None`` when unavailable.
        now: Evaluation instant (tz-aware UTC).
        watch_started_at: When the watchdog started; bounds how long a missing heartbeat is tolerated.
        thresholds: Alert thresholds.

    Returns:
        Findings ordered by (check, subject); empty when healthy.

    Raises:
        ValueError: ``now`` or ``watch_started_at`` is tz-naive.
    """
    now_utc = _require_aware(now, "now")
    started_utc = _require_aware(watch_started_at, "watch_started_at")
    stale_s = thresholds.heartbeat_stale_s

    if payload is None or not isinstance(payload, Mapping):
        watch_age_s = (now_utc - started_utc).total_seconds()
        if watch_age_s <= stale_s:
            return ()
        return (
            RecorderFinding(
                "heartbeat_missing",
                "recorder",
                f"age_s={int(watch_age_s)} threshold_s={_fmt_threshold(stale_s)}",
            ),
        )

    ts = _parse_ts(payload.get("ts"))
    if ts is None:
        return (RecorderFinding("heartbeat_stale", "recorder", "ts=missing"),)
    stale_age_s = (now_utc - ts).total_seconds()
    if stale_age_s > stale_s:
        return (
            RecorderFinding(
                "heartbeat_stale",
                "recorder",
                f"age_s={int(stale_age_s)} threshold_s={_fmt_threshold(stale_s)} ts={ts.isoformat()}",
            ),
        )

    schema = payload.get("schema_version")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema != 3:
        return (
            RecorderFinding(
                "heartbeat_schema",
                "recorder",
                f"schema_version={schema if schema is not None else 'none'}",
            ),
        )

    started_at = _parse_ts(payload.get("started_at"))
    findings: list[RecorderFinding] = []
    findings.extend(_capture_findings(payload.get("capture"), now_utc=now_utc, thresholds=thresholds))
    findings.extend(_stream_findings(payload.get("streams"), now_utc=now_utc, thresholds=thresholds))
    findings.extend(_normalizer_findings(payload.get("normalizer"), thresholds=thresholds))
    findings.extend(_compaction_findings(payload.get("compaction"), now_utc=now_utc, thresholds=thresholds))
    findings.extend(_retention_findings(payload.get("retention")))
    findings.extend(
        _reference_findings(payload, now_utc=now_utc, started_at=started_at, thresholds=thresholds)
    )
    findings.sort(key=lambda finding: (finding.check, finding.subject))
    return tuple(findings)
