"""Stdlib dead-man ping to an external check URL."""

from __future__ import annotations

import logging
import urllib.request
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _default_transport(url: str, timeout_s: float) -> int:
    request = urllib.request.Request(url, method="GET")  # noqa: S310 - operator-configured check URL
    with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:  # noqa: S310
        status = getattr(response, "status", None)
        if status is None:
            try:
                status = response.getcode()
            except Exception:  # noqa: BLE001 - fall back to 200 when the code is unreadable
                status = 200
        return int(status)


class DeadmanPinger:
    """Stdlib dead-man ping to an external check URL; a no-op when the URL is unset.

    Success pings the base URL; ``failing=True`` pings ``<url>/fail``. The URL is a secret: it
    is never logged or written to the heartbeat. Failures are logged at WARNING without the URL
    and never raised, because the ping must not affect capture.
    """

    def __init__(
        self,
        url: str | None,
        *,
        interval_s: float,
        timeout_s: float,
        transport: Callable[[str, float], int] | None = None,
    ) -> None:
        """Bind the pinger to its check URL, cadence and timeout."""
        self._url = url
        self._interval_s = interval_s
        self._timeout_s = timeout_s
        self._transport = transport if transport is not None else _default_transport
        self._last_ping_ns: int | None = None
        self._last_failing = False

    def maybe_ping(self, *, now_ns: int, failing: bool) -> bool:
        """Ping when due; return True when a ping was attempted."""
        url = self._url
        if not url:
            return False
        interval_ns = int(self._interval_s * 1_000_000_000)
        due = (
            self._last_ping_ns is None
            or (failing and not self._last_failing)
            or now_ns - self._last_ping_ns >= interval_ns
        )
        if not due:
            self._last_failing = failing
            return False
        target = f"{url.rstrip('/')}/fail" if failing else url
        try:
            self._transport(target, self._timeout_s)
        except Exception as exc:  # noqa: BLE001 - ping must never affect capture
            text = str(exc)
            detail = type(exc).__name__ if (url in text or target in text) else text[:200]
            logger.warning("[SYS] stage=capture component=deadman status=PING_FAILED error=%s", detail)
        self._last_ping_ns = now_ns
        self._last_failing = failing
        return True
