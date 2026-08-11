from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mhs.horizons import (
    efficiency_ratio,
    horizon_log_return,
    realized_vol,
    vol_normalized_horizon_signal,
)


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


class TestVolNormalizedHorizonSignal:
    """MHS momentum vol-normalization: raw horizon return scaled by its own
    realized vol, with the safe-divide contract (NaN, never 0.0 or inf).

    SCENARIO_VOL_NORMALIZED_HORIZON_SIGNAL_BASIC
    SCENARIO_VOL_NORMALIZED_HORIZON_SIGNAL_ZERO_VOL
    SCENARIO_VOL_NORMALIZED_HORIZON_SIGNAL_WARMUP_NAN
    """

    def test_equals_raw_over_vol_elementwise(self) -> None:
        log_price = pd.DataFrame(
            {"A": [0.0, 0.5, -0.2, 0.4, 0.8, 0.7], "B": [0.0, 0.01, 0.04, 0.06, 0.07, 0.10]},
        )
        result = vol_normalized_horizon_signal(log_price, 2)
        expected = horizon_log_return(log_price, 2) / realized_vol(log_price, 2)
        np.testing.assert_allclose(
            result.to_numpy(dtype="float64"),
            expected.to_numpy(dtype="float64"),
            atol=1e-12,
            equal_nan=True,
        )
        assert result.index.equals(horizon_log_return(log_price, 2).index)
        assert list(result.columns) == list(horizon_log_return(log_price, 2).columns)
        assert result.to_numpy(dtype="float64").dtype == np.float64

    def test_zero_vol_is_nan_not_inf(self) -> None:
        flat = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})
        with np.errstate(divide="raise", invalid="raise"):
            result = vol_normalized_horizon_signal(flat, 2)
        assert pd.isna(result.iloc[5, 0])
        assert np.isinf(result.to_numpy(dtype="float64")).sum() == 0

    def test_leading_warmup_rows_are_nan(self) -> None:
        log_price = pd.DataFrame({"A": [0.0, 0.5, -0.2, 0.4, 0.8, 0.7]})
        result = vol_normalized_horizon_signal(log_price, 2)
        assert pd.isna(result.iloc[0, 0])
        assert pd.isna(result.iloc[1, 0])
        assert pd.notna(result.iloc[2, 0])

    def test_fails_closed_via_existing_validation(self) -> None:
        with pytest.raises(ValueError, match="horizon_bars"):
            vol_normalized_horizon_signal(pd.DataFrame({"A": [1.0, 2.0]}), 1)
        with pytest.raises(ValueError, match="horizon_bars"):
            vol_normalized_horizon_signal(pd.DataFrame({"A": [1.0]}), 0)
