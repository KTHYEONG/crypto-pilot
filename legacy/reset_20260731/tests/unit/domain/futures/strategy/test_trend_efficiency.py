from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.config import RegimeConfig
from src.domain.futures.strategy.market_regime import compute_trend_efficiency_1d
from src.domain.futures.strategy.tiered_workflow.risk_deployment import trend_efficiency_gross_mult


class TestComputeTrendEfficiency:
    def test_efficiency_ratio_trend_vs_chop(self) -> None:
        close_mono = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        close_chop = np.array([1, 2, 1, 2, 1], dtype=np.float64)
        window = 4

        er_mono = compute_trend_efficiency_1d(close_mono, window)
        er_chop = compute_trend_efficiency_1d(close_chop, window)

        assert np.isnan(er_mono[:window]).all()
        assert er_mono[4] == pytest.approx(1.0, abs=1e-6)
        assert np.isnan(er_chop[:window]).all()
        assert er_chop[4] == pytest.approx(0.0, abs=0.1)

    def test_efficiency_ratio_flat_price_zero(self) -> None:
        close_flat = np.array([3, 3, 3, 3, 3], dtype=np.float64)
        window = 4
        er = compute_trend_efficiency_1d(close_flat, window)
        assert np.isnan(er[:window]).all()
        assert er[4] == pytest.approx(0.0, abs=1e-6)

    def test_trend_efficiency_gross_mult_bounds(self) -> None:
        target = 0.35
        floor_mult = 0.30
        inputs = [0.0, 0.175, 0.35, 0.50]
        expected = [0.30, 0.65, 1.0, 1.0]
        for er_val, exp in zip(inputs, expected, strict=True):
            result = trend_efficiency_gross_mult(er_val, target=target, floor_mult=floor_mult)
            assert result == pytest.approx(exp, abs=1e-6)


class TestRegimeConfigEfficiencyValidation:
    def test_regime_config_efficiency_validation(self) -> None:
        with pytest.raises(ValueError, match="trend_efficiency_window"):
            RegimeConfig(trend_efficiency_window=1)
        with pytest.raises(ValueError, match="trend_efficiency_target"):
            RegimeConfig(trend_efficiency_target=1.5)
        with pytest.raises(ValueError, match="trend_efficiency_floor_mult"):
            RegimeConfig(trend_efficiency_floor_mult=0.0)
