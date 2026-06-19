"""Tests for data_observable (Phase 3-2) in opt_data_utils."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.optimization.opt_data_utils import data_observable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_symbol_map(
    tf: str = "4h",
    n_bars: int = 100,
    start: str = "2023-01-01",
) -> dict[str, pd.DataFrame]:
    """Build minimal symbol_map with valid datetime-indexed OHLCV frame."""
    datetimes = pd.date_range(start, periods=n_bars, freq=tf, tz="UTC")
    df = pd.DataFrame(
        {
            "datetime": datetimes,
            "open": np.ones(n_bars, dtype=np.float64),
            "high": np.ones(n_bars, dtype=np.float64),
            "low": np.ones(n_bars, dtype=np.float64),
            "close": np.ones(n_bars, dtype=np.float64),
            "volume": np.ones(n_bars, dtype=np.float64),
        }
    )
    return {tf: df}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDataObservable:
    """Phase 3-2: data_observable lifecycle-aware check."""

    def test_data_observable_returns_pass_when_frame_valid(self) -> None:
        """A fully valid frame must return pass=True with correct metadata."""
        # Arrange
        symbol_map = _make_symbol_map(tf="4h", n_bars=100, start="2023-01-01")

        # Act
        result = data_observable(symbol="BTC", tf="4h", symbol_map=symbol_map)

        # Assert
        assert result["pass"] is True
        assert result["reason"] == "data_observable"
        assert result["n_bars"] == 100
        assert result["is_historical_stage5"] is False
        assert "first_dt" in result
        assert "last_dt" in result
        assert "effective_start" in result

    def test_data_observable_returns_fail_when_frame_missing(self) -> None:
        """symbol_map without the requested tf must return pass=False."""
        # Arrange — tf "1d" absent from map keyed on "4h"
        symbol_map = _make_symbol_map(tf="4h", n_bars=50)

        # Act
        result = data_observable(symbol="ETH", tf="1d", symbol_map=symbol_map)

        # Assert
        assert result["pass"] is False
        assert result["reason"] == "missing_tf_frame"

    def test_data_observable_does_not_require_full_oos_coverage(self) -> None:
        """Symbol with data only up to the mid-point of a hypothetical OOS must still pass.

        This is the critical invariant distinguishing data_observable from
        evaluate_symbol_data_sufficiency — delisted/partial-coverage symbols are
        allowed as long as the frame itself is non-empty and valid.
        """
        # Arrange — 30 bars ending well before a hypothetical OOS end date of 2024-01-01
        symbol_map = _make_symbol_map(tf="4h", n_bars=30, start="2023-01-01")

        # Act
        result = data_observable(symbol="DELISTED", tf="4h", symbol_map=symbol_map)

        # Assert — partial coverage must NOT cause a fail
        assert result["pass"] is True, (
            "data_observable must pass even when data ends before OOS end"
        )
        assert result["n_bars"] == 30

    def test_data_observable_onboard_date_adjusts_effective_start(self) -> None:
        """effective_start must be max(first_dt, onboard_date) when onboard_date is provided."""
        # Arrange
        symbol_map = _make_symbol_map(tf="4h", n_bars=100, start="2023-01-01")
        onboard_date = "2023-06-01"

        # Act
        result = data_observable(
            symbol="NEW", tf="4h", symbol_map=symbol_map, onboard_date=onboard_date
        )

        # Assert
        assert result["pass"] is True
        effective_start = pd.Timestamp(result["effective_start"])
        onboard_ts = pd.Timestamp(onboard_date, tz="UTC")
        assert effective_start >= onboard_ts, (
            "effective_start must be at or after onboard_date"
        )

    def test_data_observable_empty_frame_returns_fail(self) -> None:
        """Empty DataFrame must return pass=False with reason empty_datetime."""
        # Arrange
        empty_df = pd.DataFrame({"datetime": pd.Series([], dtype="datetime64[ns, UTC]")})
        symbol_map = {"4h": empty_df}

        # Act
        result = data_observable(symbol="EMPTY", tf="4h", symbol_map=symbol_map)

        # Assert
        assert result["pass"] is False
        assert result["reason"] in ("missing_tf_frame", "empty_datetime")

    def test_data_observable_onboard_date_before_first_bar_uses_first_bar(self) -> None:
        """If onboard_date is before first_dt, effective_start must equal first_dt."""
        # Arrange
        symbol_map = _make_symbol_map(tf="4h", n_bars=50, start="2023-06-01")
        # onboard before data starts
        onboard_date = "2020-01-01"

        # Act
        result = data_observable(
            symbol="OLD", tf="4h", symbol_map=symbol_map, onboard_date=onboard_date
        )

        # Assert
        first_dt = pd.Timestamp(result["first_dt"])
        effective_start = pd.Timestamp(result["effective_start"])
        assert effective_start == first_dt, (
            "effective_start must equal first_dt when onboard_date precedes data"
        )
