"""Cadences, limits and endpoints of the raw-first capture process."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CUTOFF_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Cadences, limits and endpoints of the raw-first capture process.

    Values mirror the retired recorder so capture timing is unchanged across the migration. The
    class is stdlib-only because the capture package must not depend on pydantic or project code.
    ``__post_init__`` validates cross-field constraints and raises ``ValueError`` on violation.
    """

    book_ticker_url: str = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
    premium_index_url: str = "https://fapi.binance.com/fapi/v1/premiumIndex"
    reference_urls: tuple[tuple[str, str], ...] = (
        ("exchange_info", "https://fapi.binance.com/fapi/v1/exchangeInfo"),
        ("funding_info", "https://fapi.binance.com/fapi/v1/fundingInfo"),
        ("asset_index", "https://fapi.binance.com/fapi/v1/assetIndex"),
    )
    force_order_url: str = "wss://fstream.binance.com/market/ws/!forceOrder@arr"
    book_ticker_interval_s: int = 60
    premium_index_interval_s: int = 300
    grid_max_start_lag_s: float = 5.0
    grid_retry_delay_s: float = 1.0
    http_timeout_s: float = 10.0
    rate_limit_cooldown_s: float = 60.0
    reference_capture_after_utc: str = "00:05"
    reference_retry_interval_s: float = 600.0
    ws_ping_interval_s: float = 5.0
    ws_liveness_timeout_s: float = 15.0
    ws_event_stall_timeout_s: float = 600.0
    ws_receive_timeout_s: float = 1.0
    ws_flush_interval_s: float = 5.0
    ws_flush_max_frames: int = 500
    ws_max_pending_frames: int = 100_000
    # 영속 flush 실패 시 REST 버퍼 상한. bookTicker 약 120KB/샘플 기준 240개(약 4시간치)는
    # 두 스트림을 합쳐도 약 80MB로 256MB 컨테이너 한도 안에 머문다.
    rest_max_pending_records: int = 240
    restart_backoff_max_s: float = 60.0
    heartbeat_interval_s: float = 10.0
    deadman_ping_interval_s: float = 300.0
    deadman_ping_timeout_s: float = 10.0
    deadman_fail_consecutive_failures: int = 3
    stop_flush_budget_s: float = 15.0

    def __post_init__(self) -> None:
        """Validate cross-field constraints, raising ``ValueError`` on violation."""
        for name in ("book_ticker_interval_s", "premium_index_interval_s"):
            value = getattr(self, name)
            if value <= 0 or 86400 % value != 0:
                raise ValueError(f"{name} must be a positive divisor of 86400")
        if not self.grid_max_start_lag_s < min(self.book_ticker_interval_s, self.premium_index_interval_s):
            raise ValueError("grid_max_start_lag_s must be less than both grid intervals")
        if not self.grid_retry_delay_s < self.grid_max_start_lag_s:
            raise ValueError("grid_retry_delay_s must be less than grid_max_start_lag_s")
        if not self.ws_liveness_timeout_s > self.ws_ping_interval_s:
            raise ValueError("ws_liveness_timeout_s must exceed ws_ping_interval_s")
        if not self.deadman_ping_timeout_s < self.deadman_ping_interval_s:
            raise ValueError("deadman_ping_timeout_s must be less than deadman_ping_interval_s")
        if not self.stop_flush_budget_s < 20.0:
            raise ValueError("stop_flush_budget_s must be less than the compose stop grace period (20 s)")
        if not _CUTOFF_RE.match(self.reference_capture_after_utc):
            raise ValueError("reference_capture_after_utc must be HH:MM UTC")
        for name in (
            "http_timeout_s",
            "rate_limit_cooldown_s",
            "reference_retry_interval_s",
            "ws_ping_interval_s",
            "ws_liveness_timeout_s",
            "ws_event_stall_timeout_s",
            "ws_receive_timeout_s",
            "ws_flush_interval_s",
            "restart_backoff_max_s",
            "heartbeat_interval_s",
            "deadman_ping_interval_s",
            "deadman_ping_timeout_s",
            "stop_flush_budget_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.ws_flush_max_frames <= 0:
            raise ValueError("ws_flush_max_frames must be positive")
        if self.ws_max_pending_frames <= 0:
            raise ValueError("ws_max_pending_frames must be positive")
        if self.rest_max_pending_records <= 0:
            raise ValueError("rest_max_pending_records must be positive")
        if self.deadman_fail_consecutive_failures <= 0:
            raise ValueError("deadman_fail_consecutive_failures must be positive")
        if not self.reference_urls:
            raise ValueError("reference_urls must not be empty")
