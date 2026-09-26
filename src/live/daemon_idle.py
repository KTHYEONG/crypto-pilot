"""Shared daemon-idle gate for operator-invoked ledger mutations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.daemon_stages import BUSY_STAGES
from src.common.errors import DataIntegrityError

BUSY_STALE_S: float = 2700.0


def assert_daemon_idle(heartbeat_path: Path, now: pd.Timestamp) -> None:
    """Reject an operator mutation while a cycle holds the ledger; stale busy is a crash remnant."""
    if not heartbeat_path.exists():
        return
    try:
        raw = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"heartbeat unreadable path={heartbeat_path}") from exc
    if not isinstance(raw, dict):
        raise DataIntegrityError(f"heartbeat unreadable path={heartbeat_path}")
    stage = raw.get("stage")
    ts_raw = raw.get("ts")
    if stage in BUSY_STAGES and ts_raw is not None:
        age_s = (now - pd.Timestamp(ts_raw)).total_seconds()
        if age_s <= BUSY_STALE_S:
            raise DataIntegrityError(f"daemon busy stage={stage}; retry when idle")
