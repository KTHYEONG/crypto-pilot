"""Opt-in webhook alerting for live daemon."""

from __future__ import annotations

import json
import logging
import urllib.request

import pandas as pd

logger = logging.getLogger("LiveAlerting")

ALERT_WEBHOOK_TIMEOUT_S: int = 10


def post_alert(
    webhook_url: str | None,
    *,
    event: str,
    detail: str,
    decision_time: pd.Timestamp | None,
    now: pd.Timestamp,
) -> bool:
    if not webhook_url:
        return False
    try:
        dt_iso = pd.Timestamp(decision_time).isoformat() if decision_time is not None else None
        ts_iso = pd.Timestamp(now).isoformat()
        payload = {
            "source": "mhs-live",
            "event": str(event),
            "detail": str(detail),
            "decision_time": dt_iso,
            "ts": ts_iso,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=ALERT_WEBHOOK_TIMEOUT_S) as resp:  # noqa: S310
            status = getattr(resp, "status", None)
            if status is None:
                # fallback for older Python where getcode() is used
                try:
                    status = resp.getcode()
                except Exception:
                    status = 200
            return 200 <= int(status) < 300  # noqa: SIM103
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SYS] alert webhook failed event=%s error=%s", event, exc)
        return False
