from __future__ import annotations

import pandas as pd
import pytest

from src.mhs.horizons import efficiency_ratio, horizon_log_return, realized_vol


class TestHorizonLogReturn:
    def test_shifted_lookback_return(self) -> None:
        log_price = pd.DataFrame({"A": [0.0, 1.0, 2.0, 3.0, 4.0]})
        result = horizon_log_return(log_price, 2)
        assert pd.isna(result.iloc[0, 0])
        assert pd.isna(result.iloc[1, 0])
        assert result.iloc[2, 0] == 2.0
        assert result.iloc[4, 0] == 2.0

    def test_fails_closed_on_zero_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizon_bars"):
            horizon_log_return(pd.DataFrame({"A": [1.0]}), 0)


class TestRealizedVol:
    def test_horizon_scaled_std(self) -> None:
        log_price = pd.DataFrame({"A": [0.0, 1.0, 2.0, 3.0, 4.0]})
        result = realized_vol(log_price, 3)
        assert result.iloc[4, 0] == pytest.approx(0.0)
        assert pd.isna(result.iloc[2, 0])

    def test_fails_closed_on_single_bar_window(self) -> None:
        with pytest.raises(ValueError, match="horizon_bars"):
            realized_vol(pd.DataFrame({"A": [1.0]}), 1)


class TestEfficiencyRatio:
    """MHS-04-EFFICIENCY-RATIO-BOUNDS: 1.0 monotone, 0.0 round trip, NaN flat."""

    def test_bounds(self) -> None:
        log_price = pd.DataFrame(
            {"A": [0.0, 1.0, 2.0, 3.0, 4.0], "B": [0.0, 1.0, 0.0, 1.0, 0.0]},
        )
        result = efficiency_ratio(log_price, 4)
        assert result.iloc[4]["A"] == pytest.approx(1.0)
        assert result.iloc[4]["B"] == pytest.approx(0.0)
        assert pd.isna(result.iloc[3]["A"])

    def test_flat_path_is_nan_not_zero(self) -> None:
        flat = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0]})
        result = efficiency_ratio(flat, 4)
        assert pd.isna(result.iloc[4, 0])

    def test_fails_closed_on_zero_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizon_bars"):
            efficiency_ratio(pd.DataFrame({"A": [1.0]}), 0)
