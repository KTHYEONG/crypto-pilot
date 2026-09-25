"""Pure evaluation of the market recorder heartbeat for the daemon watchdog.

The recorder (another process/container) publishes ``recorder_heartbeat.json``; this module
turns one decoded payload into health findings without any I/O or clock reads, so the logic
is unit-testable and the watchdog thread stays thin.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

RecorderCheck = Literal[
    "heartbeat_missing", "heartbeat_stale", "liquidation_silent", "liquidation_failing", "sampler_stale"
]

_SAMPLER_DATASETS: tuple[str, str] = ("book_ticker", "premium_index")


@dataclass(frozen=True, slots=True)
class RecorderWatchThresholds:
    """Alert thresholds for one evaluation of the recorder heartbeat (all durations in seconds)."""

    heartbeat_stale_s: float
    liquidation_silence_s: float
    liquidation_max_failed_connections: int
    sampler_stale_s: float


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


def _fmt_threshold(value: float) -> str:
    return f"{value:g}"


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

    liquidations = payload.get("liquidations")
    if not isinstance(liquidations, Mapping):
        findings.append(RecorderFinding("liquidation_silent", "liquidations", "entry=missing"))
    else:
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

    findings.sort(key=lambda finding: (finding.check, finding.subject))
    return tuple(findings)
