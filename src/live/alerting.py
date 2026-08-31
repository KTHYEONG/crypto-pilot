"""Opt-in webhook alerting for live daemon."""

from __future__ import annotations

import json
import logging
import smtplib
import urllib.request
from email.message import EmailMessage

import pandas as pd

logger = logging.getLogger("LiveAlerting")

ALERT_WEBHOOK_TIMEOUT_S: int = 10

SMTP_HOST: str = "smtp.gmail.com"
SMTP_PORT: int = 587
ALERT_EMAIL_TIMEOUT_S: int = 10


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


def send_email_alert(
    *,
    gmail_user: str | None,
    gmail_app_password: str | None,
    email_to: str | None,
    event: str,
    detail: str,
    decision_time: pd.Timestamp | None,
    now: pd.Timestamp,
) -> bool:
    if not gmail_user or not gmail_app_password:
        return False
    recipient = email_to or gmail_user
    try:
        dt_iso = pd.Timestamp(decision_time).isoformat() if decision_time is not None else None
        ts_iso = pd.Timestamp(now).isoformat()
        msg = EmailMessage()
        msg["From"] = gmail_user
        msg["To"] = recipient
        msg["Subject"] = f"[mhs-live] {event}"
        body_lines = [
            "source=mhs-live",
            f"event={event}",
            f"detail={detail}",
            f"decision_time={dt_iso}",
            f"ts={ts_iso}",
        ]
        msg.set_content("\n".join(body_lines))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=ALERT_EMAIL_TIMEOUT_S) as smtp:
            smtp.starttls()
            smtp.login(gmail_user, gmail_app_password)
            smtp.send_message(msg)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SYS] alert email failed event=%s error=%s", event, exc)
        return False
