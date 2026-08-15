"""Unit tests for build_pit_market_observations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.universe.observations import build_pit_market_observations

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_klines(
    symbol: str,
    timestamps: list[pd.Timestamp],
    close_prices: list[float] | None = None,
    quote_volumes: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal klines DataFrame for testing.

    Args:
        symbol: Instrument symbol string.
        timestamps: UTC-aware timestamps for each bar.
        close_prices: Close price per bar; defaults to 100.0.
        quote_volumes: Quote volume per bar; defaults to 1_000_000.0.

    Returns:
        klines DataFrame matching the expected schema.
    """
    n = len(timestamps)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": close_prices if close_prices is not None else [100.0] * n,
            "volume": 10.0,
            "quote_volume": quote_volumes if quote_volumes is not None else [1_000_000.0] * n,
        }
    )


def _make_funding(
    symbol: str,
    times: list[pd.Timestamp],
    rates: list[float],
) -> pd.DataFrame:
    """Build a minimal funding DataFrame for testing.

    Args:
        symbol: Instrument symbol string.
        times: UTC-aware funding_time timestamps.
        rates: Funding rate per event.

    Returns:
        funding DataFrame matching the expected schema.
    """
    return pd.DataFrame(
        {
            "symbol": symbol,
            "funding_time": times,
            "funding_rate": rates,
        }
    )


def _four_hour_range(start: str, periods: int) -> list[pd.Timestamp]:
    """Generate a list of UTC 4h-spaced timestamps.

    Args:
        start: ISO-format start timestamp string (tz-aware or naive UTC).
        periods: Number of 4h bars.

    Returns:
        List of UTC-aware Timestamps.
    """
    return list(pd.date_range(start, periods=periods, freq="4h", tz="UTC"))


_LAG_2H = pd.Timedelta("2h")
_MIN_OBS = 20
_LOOKBACK = 30


# ---------------------------------------------------------------------------
# Scenario 3: Information availability (available_at = observed_at + lag)
# ---------------------------------------------------------------------------


class TestScenario3InformationAvailability:
    """Verify that available_at == observed_at + availability_lag for every row."""

    def test_available_at_equals_observed_at_plus_lag_for_all_rows(self) -> None:
        """Scenario 3: available_at must equal observed_at + lag for every row."""
        # Arrange
        periods = _MIN_OBS * 6 + 10  # enough for vol30 to emit values
        timestamps = _four_hour_range("2024-01-01", periods)
        klines = _make_klines("BTCUSDT", timestamps)
        funding = _make_funding(
            "BTCUSDT",
            [pd.Timestamp("2024-01-15 00:00:00", tz="UTC")],
            [0.0001],
        )

        # Act
        obs = build_pit_market_observations(
            klines,
            funding,
            availability_lag=_LAG_2H,
            min_observations=_MIN_OBS,
            lookback_days=_LOOKBACK,
        )

        # Assert
        assert not obs.empty, "observations must not be empty"
        delta = obs["available_at"] - obs["observed_at"]
        assert (delta == _LAG_2H).all(), (
            f"All rows must satisfy available_at == observed_at + {_LAG_2H}; found deltas: {delta.unique()}"
        )

    def test_available_at_equals_observed_at_when_zero_lag(self) -> None:
        """available_at == observed_at when availability_lag is zero."""
        # Arrange
        timestamps = _four_hour_range("2024-01-01", _MIN_OBS * 6 + 5)
        klines = _make_klines("ETHUSDT", timestamps)
        funding = pd.DataFrame(columns=["symbol", "funding_time", "funding_rate"])

        # Act
        obs = build_pit_market_observations(
            klines,
            funding,
            availability_lag=pd.Timedelta(0),
            min_observations=_MIN_OBS,
        )

        # Assert
        delta = obs["available_at"] - obs["observed_at"]
        assert (delta == pd.Timedelta(0)).all()


# ---------------------------------------------------------------------------
# Scenario 7: Missing data and recovery
# ---------------------------------------------------------------------------


class TestScenario7MissingDataAndRecovery:
    """Verify NaN during gap, recovery after sufficient bars, no 0-fill."""

    def test_nan_metrics_during_insufficient_lookback_period(self) -> None:
        """ADV30 must be NaN for the first min_observations-1 valid days."""
        # Arrange
        # 10 days of 4h bars = 60 bars; fewer than min_observations=20 days
        timestamps = _four_hour_range("2024-01-01", 10 * 6)
        klines = _make_klines("SOLUSDT", timestamps)
        funding = pd.DataFrame(columns=["symbol", "funding_time", "funding_rate"])

        # Act
        obs = build_pit_market_observations(
            klines,
            funding,
            availability_lag=_LAG_2H,
            min_observations=_MIN_OBS,
            lookback_days=_LOOKBACK,
        )

        # Assert: all ADV30 rows should be NaN because < 20 valid days
        adv_rows = obs[obs["metric"] == "adv30"]
        assert adv_rows["value"].isna().all(), "ADV30 must be NaN when fewer than min_observations valid days"

    def test_metrics_recover_after_sufficient_bars_accumulate(self) -> None:
        """ADV30 becomes non-NaN once min_observations valid days accumulate."""
        # Arrange: 35 full trading days of 4h bars
        periods = 35 * _BARS_PER_DAY
        timestamps = _four_hour_range("2024-01-01", periods)
        klines = _make_klines("LTCUSDT", timestamps)
        funding = pd.DataFrame(columns=["symbol", "funding_time", "funding_rate"])

        # Act
        obs = build_pit_market_observations(
            klines,
            funding,
            availability_lag=_LAG_2H,
            min_observations=_MIN_OBS,
            lookback_days=_LOOKBACK,
        )

        # Assert: at least one ADV30 row is non-NaN (recovery)
        adv_rows = obs[obs["metric"] == "adv30"]
        assert adv_rows["value"].notna().any(), "ADV30 must recover to non-NaN once sufficient days accumulate"

    def test_gap_not_filled_with_zero_volume(self) -> None:
        """Quote volumes must never be silently replaced with 0 in observations."""
        # Arrange: 40 days normal bars, then 5-day gap (missing timestamps), then 10 more
        _BARS_PER_DAY = 6
        pre_gap = _four_hour_range("2024-01-01", 40 * _BARS_PER_DAY)
        # Skip 5 days intentionally — no bars created for that period
        post_gap = _four_hour_range("2024-02-15", 10 * _BARS_PER_DAY)
        timestamps = pre_gap + post_gap

        # Assign zero volume ONLY to post_gap bars to detect 0-fill contamination
        quote_volumes = [1_000_000.0] * (40 * _BARS_PER_DAY) + [500_000.0] * (10 * _BARS_PER_DAY)
        klines = _make_klines("BNBUSDT", timestamps, quote_volumes=quote_volumes)
        funding = pd.DataFrame(columns=["symbol", "funding_time", "funding_rate"])

        # Act
        obs = build_pit_market_observations(
            klines,
            funding,
            availability_lag=_LAG_2H,
            min_observations=_MIN_OBS,
            lookback_days=_LOOKBACK,
        )

        # Assert: ADV30 rows exist; gap period has no synthesised zero-volume bars
        adv_rows = obs[obs["metric"] == "adv30"]
        assert not adv_rows.empty

        # The gap dates should not appear as observed_at in the output
        gap_start = pd.Timestamp("2024-02-10 00:00:00", tz="UTC")
        gap_end = pd.Timestamp("2024-02-14 23:59:59", tz="UTC")
        gap_obs = adv_rows[(adv_rows["observed_at"] > gap_start) & (adv_rows["observed_at"] < gap_end)]
        assert gap_obs.empty, "No observations should exist for the gap period"


# ---------------------------------------------------------------------------
# Additional targeted tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case and contract-level tests for observations builder."""

    def test_empty_klines_returns_empty_frame(self) -> None:
        """Empty klines with empty funding must return an empty DataFrame."""
        # Arrange
        klines = pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
        funding = pd.DataFrame(columns=["symbol", "funding_time", "funding_rate"])

        # Act
        obs = build_pit_market_observations(
            klines,
            funding,
            availability_lag=_LAG_2H,
        )

        # Assert
        assert obs.empty
        assert list(obs.columns) == [
            "instrument_id",
            "metric",
            "observed_at",
            "available_at",
            "value",
            "source",
            "confidence",
        ]

    def test_duplicate_timestamp_raises_value_error(self) -> None:
        """Duplicate timestamps for a single symbol must raise ValueError."""
        # Arrange
        ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        klines = _make_klines("BTCUSDT", [ts, ts])  # duplicate
        funding = pd.DataFrame(columns=["symbol", "funding_time", "funding_rate"])

        # Act / Assert
        with pytest.raises(ValueError, match="market timestamps must be unique and monotonic"):
            build_pit_market_observations(klines, funding, availability_lag=_LAG_2H)

    def test_adv30_nan_before_min_observations(self) -> None:
        """ADV30 must be NaN when fewer than min_observations valid days exist."""
        # Arrange: only 5 days of data; min_observations=20
        periods = 5 * 6
        timestamps = _four_hour_range("2024-03-01", periods)
        klines = _make_klines("XRPUSDT", timestamps)
        funding = pd.DataFrame(columns=["symbol", "funding_time", "funding_rate"])

        # Act
        obs = build_pit_market_observations(
            klines,
            funding,
            availability_lag=_LAG_2H,
            min_observations=20,
        )

        # Assert
        adv_rows = obs[obs["metric"] == "adv30"]
        assert adv_rows["value"].isna().all(), "ADV30 must be NaN with fewer than min_observations valid days"

    def test_vol30_annualized_correctly(self) -> None:
        """vol30 = std(log_returns) * sqrt(6*365) for a known input."""
        # Arrange: 130 4h bars of constant price → log_returns = 0 → std = 0
        _BARS_PER_DAY = 6
        periods = 130  # > 20*6=120 min_periods
        timestamps = _four_hour_range("2024-01-01", periods)
        # Slightly varying prices to get a known vol
        prices = [100.0 + 0.01 * i for i in range(periods)]
        klines = _make_klines("DOGEUSDT", timestamps, close_prices=prices)
        funding = pd.DataFrame(columns=["symbol", "funding_time", "funding_rate"])

        # Act
        obs = build_pit_market_observations(
            klines,
            funding,
            availability_lag=_LAG_2H,
            min_observations=20,
        )

        # Assert: vol30 rows present; values are float, annualised by sqrt(6*365)
        vol_rows = obs[obs["metric"] == "vol30"]
        assert not vol_rows.empty
        non_nan = vol_rows["value"].dropna()
        assert not non_nan.empty, "Some vol30 values should be non-NaN with 130 bars"

        # Independently compute expected vol for the last window
        log_rets = np.diff(np.log(prices))  # length periods-1
        window = 20 * _BARS_PER_DAY
        window_rets = log_rets[-window:]
        expected_vol = float(np.std(window_rets, ddof=1) * math.sqrt(6 * 365))
        actual_last = float(non_nan.iloc[-1])
        assert actual_last == pytest.approx(expected_vol, rel=1e-4), (
            f"Last vol30={actual_last:.6f} != expected={expected_vol:.6f}"
        )

    def test_funding_observations_included_separately(self) -> None:
        """Funding events must appear as separate 'funding_rate' metric rows."""
        # Arrange
        periods = 30 * 6
        timestamps = _four_hour_range("2024-01-01", periods)
        klines = _make_klines("AAVEUSDT", timestamps)
        funding_times = [
            pd.Timestamp("2024-01-10 00:00:00", tz="UTC"),
            pd.Timestamp("2024-01-10 08:00:00", tz="UTC"),
            pd.Timestamp("2024-01-10 16:00:00", tz="UTC"),
        ]
        rates = [0.0001, -0.0002, 0.00015]
        funding = _make_funding("AAVEUSDT", funding_times, rates)

        # Act
        obs = build_pit_market_observations(
            klines,
            funding,
            availability_lag=_LAG_2H,
            min_observations=_MIN_OBS,
        )

        # Assert
        fund_rows = obs[obs["metric"] == "funding_rate"]
        assert len(fund_rows) == 3, "Must have exactly 3 funding_rate rows"

        # Verify source and confidence
        assert (fund_rows["source"] == "funding").all()
        assert (fund_rows["confidence"] == "observed").all()

        # Verify available_at = funding_time + lag
        expected_avail = pd.Series(funding_times) + _LAG_2H
        actual_avail = fund_rows["available_at"].reset_index(drop=True)
        pd.testing.assert_series_equal(
            actual_avail,
            expected_avail,
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Helper constant (module-level to match _BARS_PER_DAY used in assertions)
# ---------------------------------------------------------------------------
_BARS_PER_DAY = 6
