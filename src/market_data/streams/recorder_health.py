"""Pure evaluation of the market recorder heartbeat for the daemon watchdog.

The recorder (another process/container) publishes ``recorder_heartbeat.json``; this module
turns one decoded payload into health findings without any I/O or clock reads, so the logic
is unit-testable and the watchdog thread stays thin.
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
    "liquidation_silent", "liquidation_failing", "liquidation_unpersisted",
    "sampler_stale", "sampler_failing", "sampler_degraded", "sampler_unpersisted",
    "sampler_rows_rejected", "reference_missing",
]

_SAMPLER_DATASETS: tuple[str, str] = ("book_ticker", "premium_index")


@dataclass(frozen=True, slots=True)
class RecorderWatchThresholds:
    """Alert thresholds for one evaluation of the recorder heartbeat (all durations in seconds)."""

    heartbeat_stale_s: float
    liquidation_silence_s: float
    liquidation_max_failed_connections: int
    sampler_stale_s: float
    sampler_max_consecutive_failures: int
    min_capture_ratio: float
    capture_ratio_min_points: int
    persist_stale_s: float
    max_consecutive_flush_failures: int
    reference_grace_s: float
    rejected_fraction_alert: float
    rejected_max_consecutive_points: int


@dataclass(frozen=True, slots=True)
class RecorderFinding:
    """One failed health check.

    Attributes:
        check: Stable check id.
        subject: ``"recorder"``, ``"liquidations"``, ``"book_ticker"`` or ``"premium_index"``.
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


def _sampler_stale_findings(
    payload: Mapping[str, Any],
    *,
    now_utc: pd.Timestamp,
    started_at: pd.Timestamp | None,
    thresholds: RecorderWatchThresholds,
) -> list[RecorderFinding]:
    findings: list[RecorderFinding] = []
    for dataset in _SAMPLER_DATASETS:
        entry = payload.get(dataset)
        if not isinstance(entry, Mapping):
            findings.append(RecorderFinding("sampler_stale", dataset, "entry=missing"))
            continue
        success_at = _parse_ts(entry.get("last_success_at"))
        ref = success_at if success_at is not None else started_at
        if ref is None:
            findings.append(RecorderFinding("sampler_stale", dataset, "entry=malformed"))
            continue
        sampler_stale_s = thresholds.sampler_stale_s
        age_s = (now_utc - ref).total_seconds()
        if age_s > sampler_stale_s:
            if success_at is not None:
                detail = (
                    f"age_s={int(age_s)} threshold_s={_fmt_threshold(sampler_stale_s)} "
                    f"last_success_at={success_at.isoformat()}"
                )
            else:
                detail = (
                    f"age_s={int(age_s)} threshold_s={_fmt_threshold(sampler_stale_s)} "
                    f"last_success_at=none started_at={started_at.isoformat() if started_at is not None else 'none'}"
                )
            findings.append(RecorderFinding("sampler_stale", dataset, detail))
    return findings


def _persistence_findings(
    entry: Mapping[str, Any],
    *,
    check: RecorderCheck,
    subject: str,
    pending_key: str,
    now_utc: pd.Timestamp,
    started_at: pd.Timestamp | None,
    thresholds: RecorderWatchThresholds,
) -> list[RecorderFinding]:
    flush_failures = _parse_count(entry.get("consecutive_flush_failures"))
    pending = _parse_count(entry.get(pending_key))
    persisted_at = _parse_ts(entry.get("last_persisted_at"))
    if flush_failures is None or pending is None:
        return [RecorderFinding(check, subject, "entry=malformed")]
    if entry.get("last_persisted_at") is not None and persisted_at is None:
        return [RecorderFinding(check, subject, "entry=malformed")]
    if flush_failures >= thresholds.max_consecutive_flush_failures:
        return [
            RecorderFinding(
                check,
                subject,
                f"consecutive_flush_failures={flush_failures} "
                f"threshold={thresholds.max_consecutive_flush_failures}",
            )
        ]
    if pending > 0:
        ref = persisted_at if persisted_at is not None else started_at
        if ref is None:
            return [RecorderFinding(check, subject, "entry=malformed")]
        age_s = (now_utc - ref).total_seconds()
        if age_s > thresholds.persist_stale_s:
            return [
                RecorderFinding(
                    check,
                    subject,
                    f"pending_rows={pending} age_s={int(age_s)} "
                    f"threshold_s={_fmt_threshold(thresholds.persist_stale_s)}",
                )
            ]
    return []


def _liquidation_v1_findings(
    payload: Mapping[str, Any],
    *,
    now_utc: pd.Timestamp,
    started_at: pd.Timestamp | None,
    thresholds: RecorderWatchThresholds,
    include_persistence: bool,
) -> list[RecorderFinding]:
    findings: list[RecorderFinding] = []
    liquidations = payload.get("liquidations")
    if not isinstance(liquidations, Mapping):
        findings.append(RecorderFinding("liquidation_silent", "liquidations", "entry=missing"))
        return findings
    event_at = _parse_ts(liquidations.get("last_event_at"))
    ref = event_at if event_at is not None else started_at
    if ref is None:
        findings.append(RecorderFinding("liquidation_silent", "liquidations", "entry=malformed"))
    else:
        silence_s = thresholds.liquidation_silence_s
        age_s = (now_utc - ref).total_seconds()
        if age_s > silence_s:
            if event_at is not None:
                detail = (
                    f"age_s={int(age_s)} threshold_s={_fmt_threshold(silence_s)} "
                    f"last_event_at={event_at.isoformat()}"
                )
            else:
                detail = (
                    f"age_s={int(age_s)} threshold_s={_fmt_threshold(silence_s)} "
                    f"last_event_at=none started_at={started_at.isoformat() if started_at is not None else 'none'}"
                )
            findings.append(RecorderFinding("liquidation_silent", "liquidations", detail))
    count = _parse_count(liquidations.get("consecutive_failed_connections"))
    if count is None:
        findings.append(RecorderFinding("liquidation_failing", "liquidations", "entry=malformed"))
    elif count >= thresholds.liquidation_max_failed_connections:
        findings.append(
            RecorderFinding(
                "liquidation_failing",
                "liquidations",
                f"consecutive_failed_connections={count} "
                f"threshold={thresholds.liquidation_max_failed_connections}",
            )
        )
    if include_persistence:
        findings.extend(
            _persistence_findings(
                liquidations,
                check="liquidation_unpersisted",
                subject="liquidations",
                pending_key="pending_events",
                now_utc=now_utc,
                started_at=started_at,
                thresholds=thresholds,
            )
        )
    return findings


def _sampler_v2_findings(
    payload: Mapping[str, Any],
    *,
    now_utc: pd.Timestamp,
    started_at: pd.Timestamp | None,
    thresholds: RecorderWatchThresholds,
) -> list[RecorderFinding]:
    findings: list[RecorderFinding] = []
    for dataset in _SAMPLER_DATASETS:
        entry = payload.get(dataset)
        if not isinstance(entry, Mapping):
            continue
        failures = _parse_count(entry.get("consecutive_failures"))
        if failures is None:
            findings.append(RecorderFinding("sampler_failing", dataset, "entry=malformed"))
        elif failures >= thresholds.sampler_max_consecutive_failures:
            findings.append(
                RecorderFinding(
                    "sampler_failing",
                    dataset,
                    f"consecutive_failures={failures} "
                    f"threshold={thresholds.sampler_max_consecutive_failures}",
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
                        "sampler_degraded",
                        dataset,
                        f"captured={captured} expected={expected} ratio={ratio:.3f}",
                    )
                )
        findings.extend(
            _persistence_findings(
                entry,
                check="sampler_unpersisted",
                subject=dataset,
                pending_key="pending_rows",
                now_utc=now_utc,
                started_at=started_at,
                thresholds=thresholds,
            )
        )
        rejected_last = _parse_count(entry.get("rejected_rows_last_sample"))
        rejected_total = _parse_count(entry.get("rejected_rows_total"))
        rejected_points = _parse_count(entry.get("consecutive_rejecting_points"))
        fraction = _parse_ratio(entry.get("rejected_fraction_last_sample"))
        if (
            rejected_last is None
            or rejected_total is None
            or rejected_points is None
            or fraction is None
        ):
            findings.append(RecorderFinding("sampler_rows_rejected", dataset, "entry=malformed"))
        elif (
            fraction >= thresholds.rejected_fraction_alert
            or rejected_points >= thresholds.rejected_max_consecutive_points
        ):
            findings.append(
                RecorderFinding(
                    "sampler_rows_rejected",
                    dataset,
                    f"rejected_last={rejected_last} fraction={fraction:.4f} "
                    f"consecutive_points={rejected_points} rejected_total={rejected_total}",
                )
            )
    return findings


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
        # 기동 직후 첫 publish 전에는 cutoff가 비어 있다: 기한을 알 수 없으므로 grace 동안은 판정하지 않는다.
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

    def _detail(missing: list[str]) -> str:
        errors = []
        for name in missing:
            status = endpoints.get(name)
            err = status.get("last_error") if isinstance(status, Mapping) else None
            errors.append(f"{name}:{err if err is not None else 'none'}")
        return f"missing={','.join(missing)} errors={';'.join(errors)}"

    if now_utc >= deadline:
        if day != today:
            return [
                RecorderFinding(
                    "reference_missing", "reference", f"day={day} today={today} entry=stale"
                )
            ]
        missing = _missing_today()
        if missing:
            return [RecorderFinding("reference_missing", "reference", _detail(missing))]
        return []
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
            return [RecorderFinding("reference_missing", "reference", _detail(missing))]
    return []


def evaluate_recorder_heartbeat(
    payload: Mapping[str, Any] | None,
    *,
    now: pd.Timestamp,
    watch_started_at: pd.Timestamp,
    thresholds: RecorderWatchThresholds,
) -> tuple[RecorderFinding, ...]:
    """Classify one recorder heartbeat snapshot into failed health checks.

    Monitoring is fail-closed: a missing or malformed entry counts as failing its check, because the
    silent failure mode being guarded against (a live socket that delivers nothing) looks healthy to
    every liveness probe except "time since the last real event". When the heartbeat itself is stale
    only that finding is returned, since every other field describes the past.

    Persistence is evaluated separately from fetch success, because a sampler that keeps fetching while
    every write fails looks healthy to a freshness check. Short outages are caught by consecutive failures,
    and intermittent ones by the trailing capture ratio.

    Args:
        payload: Decoded heartbeat or ``None`` when unavailable.
        now: Evaluation instant (tz-aware UTC).
        watch_started_at: When the watchdog started; bounds how long a missing heartbeat is tolerated
            (the recorder may still be starting after a deploy).
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

    started_at = _parse_ts(payload.get("started_at"))
    findings: list[RecorderFinding] = []

    schema = payload.get("schema_version")
    is_v2 = isinstance(schema, int) and not isinstance(schema, bool) and schema >= 2
    if not is_v2:
        findings.extend(
            _liquidation_v1_findings(
                payload, now_utc=now_utc, started_at=started_at, thresholds=thresholds,
                include_persistence=False,
            )
        )
        findings.extend(
            _sampler_stale_findings(
                payload, now_utc=now_utc, started_at=started_at, thresholds=thresholds
            )
        )
        watch_age_s = (now_utc - started_utc).total_seconds()
        if watch_age_s > stale_s:
            raw = schema if schema is not None else "none"
            findings.append(
                RecorderFinding(
                    "heartbeat_schema",
                    "recorder",
                    f"schema_version={raw} threshold_s={_fmt_threshold(stale_s)}",
                )
            )
        findings.sort(key=lambda finding: (finding.check, finding.subject))
        return tuple(findings)

    findings.extend(
        _liquidation_v1_findings(
            payload, now_utc=now_utc, started_at=started_at, thresholds=thresholds,
            include_persistence=True,
        )
    )
    findings.extend(
        _sampler_stale_findings(
            payload, now_utc=now_utc, started_at=started_at, thresholds=thresholds
        )
    )
    findings.extend(
        _sampler_v2_findings(
            payload, now_utc=now_utc, started_at=started_at, thresholds=thresholds
        )
    )
    findings.extend(_reference_findings(payload, now_utc=now_utc, started_at=started_at, thresholds=thresholds))

    findings.sort(key=lambda finding: (finding.check, finding.subject))
    return tuple(findings)
