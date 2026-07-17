from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.forward_returns import (
    compute_causal_forward_returns_bps,
)


class TestComputeCausalForwardReturnsBps:
    def test_matches_hand_computed_log_return(self) -> None:
        close = np.array([100.0 + i * 0.5 for i in range(30)], dtype=np.float64).reshape(-1, 1)
        side = np.ones((30, 1), dtype=np.int8)
        fwd_ret = compute_causal_forward_returns_bps(
            close=close,
            side=side,
            causal_lag_bars=1,
            holding_bars=18,
        )
        expected = np.log(close[24, 0] / close[6, 0]) * 10000.0
        assert fwd_ret[5, 0] == pytest.approx(expected, rel=1e-9)
        assert np.isnan(fwd_ret[11, 0])

    def test_causal_lag_zero_regression_guard(self) -> None:
        close = np.array([100.0 + i * 0.5 for i in range(20)], dtype=np.float64).reshape(-1, 1)
        side = np.ones((20, 1), dtype=np.int8)
        fwd = compute_causal_forward_returns_bps(
            close=close,
            side=side,
            causal_lag_bars=0,
            holding_bars=3,
        )
        idx_end = 20 - 0 - 3
        expected_vals = np.log(close[3:20, :] / close[0:17, :]) * 10000.0
        expected = np.full((20, 1), np.nan)
        expected[:idx_end] = expected_vals
        assert np.allclose(fwd[:idx_end], expected[:idx_end], rtol=1e-9)
        assert np.all(np.isnan(fwd[idx_end:]))
        assert fwd[5, 0] == pytest.approx(expected[5, 0], rel=1e-9)

    def test_exact_boundary_returns_nan(self) -> None:
        close = np.ones((10, 1), dtype=np.float64)
        side = np.ones((10, 1), dtype=np.int8)
        fwd = compute_causal_forward_returns_bps(
            close=close,
            side=side,
            causal_lag_bars=2,
            holding_bars=8,
        )
        assert np.all(np.isnan(fwd))

    def test_rejects_negative_lag(self) -> None:
        close = np.full((10, 1), 100.0)
        side = np.ones((10, 1), dtype=np.int8)
        with pytest.raises(ValueError, match="causal_lag_bars"):
            compute_causal_forward_returns_bps(
                close=close,
                side=side,
                causal_lag_bars=-1,
                holding_bars=3,
            )

    def test_rejects_zero_holding(self) -> None:
        close = np.full((10, 1), 100.0)
        side = np.ones((10, 1), dtype=np.int8)
        with pytest.raises(ValueError, match="holding_bars"):
            compute_causal_forward_returns_bps(
                close=close,
                side=side,
                causal_lag_bars=1,
                holding_bars=0,
            )
