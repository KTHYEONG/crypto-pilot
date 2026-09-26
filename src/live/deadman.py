"""Rate-limited pinger for an external dead-man's-switch check."""

from __future__ import annotations

import logging
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger("LiveDeadman")


def _default_transport(url: str, timeout_s: float) -> int:
    req = urllib.request.Request(url, method="GET")  # noqa: S310
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:  # noqa: S310
        status = getattr(resp, "status", None)
        if status is None:
            try:
                status = resp.getcode()
            except Exception:
                status = 200
        return int(status)


@dataclass(slots=True)
class DeadmanPinger:
    """Rate-limited, never-raising pinger for an external dead-man's-switch check.

    The external vendor raises the alarm when pings stop, which is the only signal that survives
    the death of the host, its network or the pinging process itself. A ping therefore means
    "this process is alive and its loop is advancing"; ``/fail`` means "alive but not healthy" so
    the vendor alerts immediately instead of after the grace period.

    Args:
        url: Check URL (secret); ``None`` disables every call.
        interval_s: Minimum seconds between two pings.
        timeout_s: Per-request HTTP timeout.
        transport: Injectable ``(url, timeout_s) -> int`` HTTP status callable (real client in
            production, fake in tests).
    """

    url: Any = None
    interval_s: float = 300.0
    timeout_s: float = 10.0
    transport: Callable[[str, float], int] = field(default=_default_transport)
    _last_attempt_at: pd.Timestamp | None = field(default=None, init=False, repr=False)
    _last_failing: bool | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _secret_value(self) -> str | None:
        if self.url is None:
            return None
        get_secret = getattr(self.url, "get_secret_value", None)
        if callable(get_secret):
            try:
                value = get_secret()
            except Exception:
                return None
            return str(value) if value else None
        value = str(self.url)
        return value or None

    @property
    def enabled(self) -> bool:
        """Return True when a check URL is configured."""
        return self._secret_value() is not None

    def maybe_ping(self, *, now: pd.Timestamp, failing: bool) -> bool:
        """Ping if due. Returns True when a request was sent and got a 2xx status."""
        secret = self._secret_value()
        if not secret:
            return False
        try:
            now_utc = pd.Timestamp(now)
            now_utc = now_utc.tz_localize("UTC") if now_utc.tzinfo is None else now_utc.tz_convert("UTC")
        except Exception:
            return False
        with self._lock:
            state_changed = self._last_failing is not None and bool(failing) != bool(self._last_failing)
            if self._last_attempt_at is not None and not state_changed:
                try:
                    elapsed = (now_utc - pd.Timestamp(self._last_attempt_at)).total_seconds()
                except Exception:
                    return False
                if elapsed < float(self.interval_s):
                    return False
            self._last_attempt_at = now_utc
            self._last_failing = bool(failing)
        target = f"{secret}/fail" if failing else secret
        try:
            status = int(self.transport(target, float(self.timeout_s)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SYS] stage=deadman status=PING_FAILED http=None error=%s", type(exc).__name__)
            return False
        if 200 <= status < 300:
            return True
        logger.warning("[SYS] stage=deadman status=PING_FAILED http=%s error=status", status)
        return False
