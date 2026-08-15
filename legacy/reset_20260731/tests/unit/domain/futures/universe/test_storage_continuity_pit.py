"""Tests for Universe Ledger PIT Correctness (Part 2)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.universe.models import LedgerRow
from src.domain.futures.universe.storage import (
    compute_rolling_zero_volume_bars,
    detect_continuity_metric_regression,
)


def _make_klines_4h(n_days: int, start_date: date, zero_vol_start_day: int | None = None) -> pd.DataFrame:
    """Create synthetic 4h klines DataFrame.

    Args:
        n_days: Number of days of data.
        start_date: Start date for the data.
        zero_vol_start_day: If set, days >= this will have zero volume.
    """
    bars_per_day = 6
    n_bars = n_days * bars_per_day
    idx = pd.date_range(start=start_date, periods=n_bars, freq="4h")

    quote_vol = np.full(n_bars, 100.0)
    if zero_vol_start_day is not None:
        zero_start_bar = zero_vol_start_day * bars_per_day
        quote_vol[zero_start_bar:] = 0.0

    df = pd.DataFrame(
        {
            "datetime": idx,
            "quote_vol": quote_vol,
            "volume": quote_vol,
        }
    )
    return df


# ── Scenario 1: Happy Path (PIT correctness) ──


def test_compute_rolling_zero_volume_bars_reflects_point_in_time_window() -> None:
    """day_30 should have ~0 zero-vol bars, day_90 should have ~180 zero-vol bars."""
    klines = _make_klines_4h(n_days=120, start_date=date(2026, 1, 1), zero_vol_start_day=60)

    day_30 = date(2026, 1, 31)
    day_90 = date(2026, 3, 31)

    result = compute_rolling_zero_volume_bars(
        klines_tf=klines,
        dates=[day_30, day_90],
        window_days=60,
    )

    assert result[day_30] == pytest.approx(0, abs=10)
    assert result[day_90] >= 150
    assert result[day_30] != result[day_90]


# ── Scenario 2: Edge Cases ──


def test_compute_rolling_zero_volume_bars_all_zero_when_no_zero_volume_bars() -> None:
    """All-normal klines should yield 0 for all dates."""
    klines = _make_klines_4h(n_days=120, start_date=date(2026, 1, 1))

    dates = [date(2026, 1, 31), date(2026, 2, 28), date(2026, 3, 31)]
    result = compute_rolling_zero_volume_bars(klines_tf=klines, dates=dates, window_days=60)

    for d in dates:
        assert result[d] == 0


def test_compute_rolling_zero_volume_bars_returns_zero_for_dates_before_onboard() -> None:
    """Dates before klines history should return 0."""
    klines = _make_klines_4h(n_days=60, start_date=date(2026, 2, 1), zero_vol_start_day=0)

    early_date = date(2026, 1, 1)
    result = compute_rolling_zero_volume_bars(klines_tf=klines, dates=[early_date], window_days=60)

    assert result[early_date] == 0


def test_compute_rolling_zero_volume_bars_handles_empty_klines_frame() -> None:
    """Empty klines frame should return 0 for all dates."""
    klines = pd.DataFrame(columns=["datetime", "quote_vol", "volume"])

    dates = [date(2026, 1, 31), date(2026, 2, 28)]
    result = compute_rolling_zero_volume_bars(klines_tf=klines, dates=dates, window_days=60)

    for d in dates:
        assert result[d] == 0


def test_compute_rolling_zero_volume_bars_falls_back_to_volume_when_quote_vol_is_nan() -> None:
    """Regression test: row-level NaN in quote_vol (e.g. live-API-sourced recent bars

    that never populated quote_vol) must not be misread as zero volume when the
    volume column has real, nonzero data for those same rows.
    """
    n_days = 30
    bars_per_day = 6
    n_bars = n_days * bars_per_day
    idx = pd.date_range(start=date(2026, 6, 1), periods=n_bars, freq="4h")

    # quote_vol entirely NaN (simulates fetch_ohlcv_with_taker bug window),
    # volume fully populated with real nonzero trading data.
    klines = pd.DataFrame(
        {
            "datetime": idx,
            "quote_vol": np.full(n_bars, np.nan),
            "volume": np.full(n_bars, 500.0),
        }
    )

    as_of = date(2026, 6, 30)
    result = compute_rolling_zero_volume_bars(klines_tf=klines, dates=[as_of], window_days=60)

    assert result[as_of] == 0


# ── Scenario 3: Regression Guardrail ──


def _make_ledger_row(symbol: str, day: date, n_zero_vol: int) -> LedgerRow:
    """Create a minimal LedgerRow for testing."""
    return LedgerRow(
        symbol=symbol,
        date=day.isoformat(),
        knowledge_date=(day + timedelta(days=1)).isoformat(),
        is_listed=True,
        is_trading=True,
        status="TRADING",
        first_kline_date="2025-01-01",
        adv_usdt_median=1000.0,
        adv_usdt_mean=1000.0,
        has_kline=True,
        has_funding=True,
        n_bar_gaps=0,
        max_gap_bars=0,
        frozen_bars=0,
        last_60d_coverage=1.0,
        n_zero_volume_bars_60d=n_zero_vol,
        has_nan=False,
        has_inf=False,
        has_timestamp_issues=False,
        funding_rate_8h=0.0001,
        listing_age_days=365,
        vol_30d=0.5,
        risk_event_override=None,
        updated_at_utc="2026-07-01T00:00:00",
        is_coverage=True,
        n_is_bars=100,
        expected_is_bars=100,
        tf="4h",
        amihud_30d=0.001,
        mark_price=50000.0,
        funding_zscore=0.0,
    )


def test_detect_continuity_metric_regression_flags_anomalous_jump() -> None:
    """Jump from 1 to 180 (180x) should trigger warning."""
    prev_frame = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "n_zero_volume_bars_60d": [1],
        }
    )

    new_rows = [_make_ledger_row("BTCUSDT", date(2026, 3, 1), 180)]

    warnings = detect_continuity_metric_regression(
        symbol="BTCUSDT",
        new_rows=new_rows,
        previous_ledger_frame=prev_frame,
        jump_multiplier=20.0,
    )

    assert len(warnings) == 1
    assert "BTCUSDT" in warnings[0]
    assert "180" in warnings[0]


def test_detect_continuity_metric_regression_skips_first_ever_ledger_write() -> None:
    """No previous ledger should return empty warnings."""
    new_rows = [_make_ledger_row("BTCUSDT", date(2026, 3, 1), 180)]

    warnings = detect_continuity_metric_regression(
        symbol="BTCUSDT",
        new_rows=new_rows,
        previous_ledger_frame=None,
        jump_multiplier=20.0,
    )

    assert warnings == []


def test_detect_continuity_metric_regression_no_warning_for_gradual_change() -> None:
    """Gradual increase (1 -> 3) should not trigger warning."""
    prev_frame = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "n_zero_volume_bars_60d": [1],
        }
    )

    new_rows = [_make_ledger_row("BTCUSDT", date(2026, 3, 1), 3)]

    warnings = detect_continuity_metric_regression(
        symbol="BTCUSDT",
        new_rows=new_rows,
        previous_ledger_frame=prev_frame,
        jump_multiplier=20.0,
    )

    assert warnings == []
