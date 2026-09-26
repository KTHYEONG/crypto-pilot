"""Invariant guards for capture configuration defaults and validation."""

from __future__ import annotations

import pytest

from src.capture.config import CaptureConfig


def test_defaults_mirror_recorder() -> None:
    """Grid cadences, endpoints and limits match the retired recorder."""
    config = CaptureConfig()
    assert config.book_ticker_url == "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
    assert config.premium_index_url == "https://fapi.binance.com/fapi/v1/premiumIndex"
    assert config.force_order_url == "wss://fstream.binance.com/market/ws/!forceOrder@arr"
    assert config.book_ticker_interval_s == 60
    assert config.premium_index_interval_s == 300
    assert config.grid_max_start_lag_s == 5.0
    assert config.grid_retry_delay_s == 1.0
    assert config.ws_liveness_timeout_s == 15.0
    assert config.ws_event_stall_timeout_s == 600.0
    assert config.stop_flush_budget_s == 15.0
    assert [name for name, _ in config.reference_urls] == ["exchange_info", "funding_info", "asset_index"]


def test_grid_interval_must_divide_day() -> None:
    """Grid intervals must be positive divisors of 86400."""
    with pytest.raises(ValueError, match="divisor"):
        CaptureConfig(book_ticker_interval_s=7)
    with pytest.raises(ValueError, match="divisor"):
        CaptureConfig(premium_index_interval_s=0)


def test_cross_field_constraints() -> None:
    """Lag, retry, liveness, deadman and flush-budget relations are enforced."""
    with pytest.raises(ValueError, match="grid_max_start_lag_s"):
        CaptureConfig(grid_max_start_lag_s=60.0)
    with pytest.raises(ValueError, match="grid_retry_delay_s"):
        CaptureConfig(grid_retry_delay_s=5.0)
    with pytest.raises(ValueError, match="ws_liveness_timeout_s"):
        CaptureConfig(ws_liveness_timeout_s=5.0, ws_ping_interval_s=5.0)
    with pytest.raises(ValueError, match="deadman_ping_timeout_s"):
        CaptureConfig(deadman_ping_timeout_s=300.0)
    with pytest.raises(ValueError, match="stop_flush_budget_s"):
        CaptureConfig(stop_flush_budget_s=25.0)
    with pytest.raises(ValueError, match="HH:MM"):
        CaptureConfig(reference_capture_after_utc="25:00")


def test_positive_limits_and_reference_set() -> None:
    """Every positive limit and the reference set reject empty or non-positive values."""
    with pytest.raises(ValueError, match="http_timeout_s"):
        CaptureConfig(http_timeout_s=0.0)
    with pytest.raises(ValueError, match="ws_flush_max_frames"):
        CaptureConfig(ws_flush_max_frames=0)
    with pytest.raises(ValueError, match="ws_max_pending_frames"):
        CaptureConfig(ws_max_pending_frames=0)
    with pytest.raises(ValueError, match="deadman_fail_consecutive_failures"):
        CaptureConfig(deadman_fail_consecutive_failures=0)
    with pytest.raises(ValueError, match="reference_urls"):
        CaptureConfig(reference_urls=())
