"""Host-side daemon liveness evaluation and episode alerting."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.paths import DATA_DIR

logger = logging.getLogger("LiveLiveness")


@dataclass(frozen=True, slots=True)
class ContainerObservation:
    running: bool
    restarting: bool
    oom_killed: bool
    restart_count: int
    started_at: pd.Timestamp | None


@dataclass(frozen=True, slots=True)
class LivenessFinding:
    key: str
    event: str
    detail: str


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


def evaluate_daemon_liveness(
    heartbeat: Mapping[str, Any] | None,
    container: ContainerObservation,
    *,
    now: pd.Timestamp,
    heartbeat_stale_s: float,
) -> tuple[LivenessFinding, ...]:
    """Pure evaluation of daemon liveness from the heartbeat file and the host's container view.

    ``heartbeat_stale`` proves the process stopped writing; ``stage_overrun`` (now > expected_by)
    proves the loop stopped progressing even while a pulse keeps ``ts`` fresh (a hung socket in
    execute). ``container_down`` covers stopped, restarting and OOM-killed states that the heartbeat
    cannot express. Never raises; a malformed heartbeat is itself a finding.
    """
    try:
        now_utc = pd.Timestamp(now)
        now_utc = now_utc.tz_localize("UTC") if now_utc.tzinfo is None else now_utc.tz_convert("UTC")
    except Exception:
        return (LivenessFinding(key="heartbeat_missing", event="daemon_unresponsive", detail="now_unparseable"),)
    try:
        if container.oom_killed or (not container.running) or container.restarting:
            return (
                LivenessFinding(
                    key="container_down",
                    event="daemon_container_down",
                    detail=f"running={container.running} restarting={container.restarting} oom_killed={container.oom_killed} restart_count={container.restart_count}",
                ),
            )
        if heartbeat is None or not isinstance(heartbeat, Mapping):
            return (LivenessFinding(key="heartbeat_missing", event="daemon_unresponsive", detail="heartbeat_missing"),)
        ts = _parse_ts(heartbeat.get("ts"))
        if ts is None:
            return (LivenessFinding(key="heartbeat_missing", event="daemon_unresponsive", detail="heartbeat_ts_unparseable"),)
        age_s = (now_utc - ts).total_seconds()
        if age_s > float(heartbeat_stale_s):
            return (
                LivenessFinding(
                    key="heartbeat_stale",
                    event="daemon_unresponsive",
                    detail=f"heartbeat_age_s={age_s:.0f} stale_after_s={heartbeat_stale_s}",
                ),
            )
        expected_by = _parse_ts(heartbeat.get("expected_by"))
        if expected_by is not None and now_utc > expected_by:
            stage = heartbeat.get("stage", "?")
            return (
                LivenessFinding(
                    key="stage_overrun",
                    event="daemon_stage_overrun",
                    detail=f"stage={stage} expected_by={heartbeat.get('expected_by')} now={now_utc.isoformat()}",
                ),
            )
        return ()
    except Exception as exc:  # noqa: BLE001
        return (LivenessFinding(key="heartbeat_missing", event="daemon_unresponsive", detail=f"evaluation_failed:{type(exc).__name__}"),)


def _default_state_path() -> Path:
    return DATA_DIR / "state" / "live_liveness_state.json"


def _read_episode_state(state_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"open_keys": [], "episode_started_at": None}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"open_keys": [], "episode_started_at": None}
    if not isinstance(raw, dict):
        return {"open_keys": [], "episode_started_at": None}
    open_keys = raw.get("open_keys")
    if not isinstance(open_keys, list):
        open_keys = []
    return {"open_keys": [str(k) for k in open_keys], "episode_started_at": raw.get("episode_started_at")}


def _write_episode_state(state_path: Path, payload: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, state_path)


def _read_heartbeat_file(settings: Any) -> Mapping[str, Any] | None:
    """Read the daemon heartbeat without importing trading modules.

    The checker must never depend on runner or executor code paths; the
    heartbeat is plain JSON, so it is parsed directly here.
    """
    try:
        raw_path = getattr(settings, "heartbeat_path", None)
        hb_path = Path(raw_path) if raw_path else DATA_DIR / "state" / "live_daemon_heartbeat.json"
        raw = json.loads(hb_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def run_liveness_check(
    settings: Any,
    container: ContainerObservation,
    *,
    now: pd.Timestamp,
    state_path: Path,
) -> int:
    """Evaluate, alert once per episode through ``dispatch_alert``, drain the outbox, persist episode state.

    Returns:
        Process exit code: 0 when evaluation ran and every CRITICAL alert of this run was accepted
        and has at least one delivered channel or no findings exist; 2 when a CRITICAL finding
        exists but no channel delivered it (so the systemd OnFailure fallback fires); 1 on internal
        error.
    """
    from src.live.alert_outbox import default_dedupe_key
    from src.live.alerting import dispatch_alert, drain_alerts

    try:
        now_utc = pd.Timestamp(now)
        now_utc = now_utc.tz_localize("UTC") if now_utc.tzinfo is None else now_utc.tz_convert("UTC")
    except Exception:
        return 1
    try:
        heartbeat = _read_heartbeat_file(settings)
        findings = evaluate_daemon_liveness(
            heartbeat,
            container,
            now=now_utc,
            heartbeat_stale_s=float(getattr(settings, "liveness_heartbeat_stale_s", 900.0)),
        )
        try:
            drain_alerts(settings, now=now_utc, blocking=True)
        except Exception:  # noqa: BLE001
            logger.exception("[SYS] stage=liveness status=DRAIN_FAILED")
        state = _read_episode_state(Path(state_path))
        open_keys: list[str] = list(state.get("open_keys", []))
        episode_started_at = state.get("episode_started_at")
        current_keys = [f.key for f in findings]
        critical_failed = False
        if findings:
            if not open_keys:
                episode_started_at = now_utc.isoformat()
            for finding in findings:
                if finding.key not in open_keys:
                    key = f"{finding.event}:{episode_started_at}:{finding.key}"
                    accepted = dispatch_alert(
                        settings,
                        event=finding.event,
                        detail=finding.detail,
                        decision_time=None,
                        dedupe_key=key,
                        now=now_utc,
                    )
                    try:
                        delivered = _any_channel_delivered(settings, finding.event, key)
                    except Exception:  # noqa: BLE001
                        delivered = False
                    if (not accepted or not delivered) and finding.event != "daemon_liveness_recovered":
                        critical_failed = True
            open_keys = current_keys
        else:
            if open_keys:
                dispatch_alert(
                    settings,
                    event="daemon_liveness_recovered",
                    detail="all liveness findings cleared",
                    decision_time=None,
                    dedupe_key=default_dedupe_key("daemon_liveness_recovered", None) + f":{episode_started_at}",
                    now=now_utc,
                )
            open_keys = []
            episode_started_at = None
        try:
            _write_episode_state(Path(state_path), {"open_keys": open_keys, "episode_started_at": episode_started_at})
        except Exception:  # noqa: BLE001
            logger.exception("[SYS] stage=liveness status=STATE_WRITE_FAILED")
            return 1
        return 2 if critical_failed else 0
    except Exception:  # noqa: BLE001
        logger.exception("[SYS] stage=liveness status=CHECK_FAILED")
        return 1


def _any_channel_delivered(settings: Any, event: str, dedupe_key: str) -> bool:
    """Check the outbox for at least one delivered channel of ``dedupe_key``."""
    from src.live.alert_outbox import resolve_outbox_path

    path = resolve_outbox_path(settings)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(raw, dict) or not isinstance(raw.get("records"), list):
        return False
    for item in raw["records"]:
        if isinstance(item, dict) and item.get("dedupe_key") == dedupe_key:
            delivered = item.get("delivered_channels", [])
            pending = item.get("pending_channels", [])
            return bool(delivered) or (not pending and not item.get("expired", False))
    return False
