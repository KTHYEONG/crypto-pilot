"""Tests for forecast/risk.py — build_risk_forecast."""
from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.forecast.risk import (
    _rolling_beta_strict_causal,
    _rolling_residual_variance,
    build_risk_forecast,
)

_T, _N = 60, 4
_BTC_IDX = 0


def _close(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumprod(1 + rng.normal(0, 0.02, (_T, _N)), axis=0) * 1000.0


_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


class TestBuildRiskForecastOutputContract:
    def test_output_shapes(self) -> None:
        rf = build_risk_forecast(_close(), _SYMBOLS, "4h", {}, btc_index=_BTC_IDX, lookback=20)

        assert rf.covariance_3d.shape == (_T, _N, _N)
        assert rf.beta_2d is not None and rf.beta_2d.shape == (_T, _N)
        assert rf.residual_var_2d is not None and rf.residual_var_2d.shape == (_T, _N)
        assert rf.forecast_vol_2d.shape == (_T, _N)

    def test_covariance_is_finite(self) -> None:
        rf = build_risk_forecast(_close(), _SYMBOLS, "4h", {}, btc_index=_BTC_IDX, lookback=20)
        assert np.all(np.isfinite(rf.covariance_3d))

    def test_residual_var_nonnegative(self) -> None:
        rf = build_risk_forecast(_close(), _SYMBOLS, "4h", {}, btc_index=_BTC_IDX, lookback=20)
        assert rf.residual_var_2d is not None
        assert np.all(rf.residual_var_2d >= 0.0)

    def test_forecast_vol_nonnegative(self) -> None:
        rf = build_risk_forecast(_close(), _SYMBOLS, "4h", {}, btc_index=_BTC_IDX, lookback=20)
        assert np.all(rf.forecast_vol_2d >= 0.0)

    def test_beta_source_trailing_btc_when_found(self) -> None:
        rf = build_risk_forecast(_close(), _SYMBOLS, "4h", {}, btc_index=_BTC_IDX, lookback=20)
        assert rf.beta_source == "trailing_btc"

    def test_beta_source_unavailable_when_no_btc(self) -> None:
        symbols_no_btc = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
        rf = build_risk_forecast(
            _close(), symbols_no_btc, "4h", {}, btc_index=None, lookback=20
        )
        assert rf.beta_source == "unavailable"
        assert rf.beta_2d is not None
        # beta should be all zeros when unavailable
        assert np.all(rf.beta_2d == 0.0)


class TestRollingBetaStrictCausal:
    def test_causal_no_future_data(self) -> None:
        # bar t의 beta 산출에 t 이후 데이터가 사용되지 않음을 검증
        # 미래 데이터를 변조해도 과거 beta가 바뀌지 않아야 한다
        rng = np.random.default_rng(0)
        sym_ret = rng.normal(0, 0.01, (_T, _N))
        btc_ret = rng.normal(0, 0.01, _T)

        beta_orig, _ = _rolling_beta_strict_causal(sym_ret, btc_ret, lookback=20)

        # 미래 바(t=50~) 데이터를 변조
        sym_ret_mod = sym_ret.copy()
        sym_ret_mod[50:, :] = 999.0
        beta_mod, _ = _rolling_beta_strict_causal(sym_ret_mod, btc_ret, lookback=20)

        # t=49 이전 beta는 동일해야 한다 (현재 바 제외, [start:i] 윈도우)
        np.testing.assert_array_equal(beta_orig[:50], beta_mod[:50])

    def test_beta_finite(self) -> None:
        rng = np.random.default_rng(1)
        sym_ret = rng.normal(0, 0.01, (_T, _N))
        btc_ret = rng.normal(0, 0.01, _T)
        beta, _ = _rolling_beta_strict_causal(sym_ret, btc_ret, lookback=15)
        assert np.all(np.isfinite(beta))

    def test_beta_zero_at_start(self) -> None:
        # window 부족 시 beta=0 기본값
        rng = np.random.default_rng(2)
        sym_ret = rng.normal(0, 0.01, (_T, _N))
        btc_ret = rng.normal(0, 0.01, _T)
        beta, _ = _rolling_beta_strict_causal(sym_ret, btc_ret, lookback=15)
        assert float(beta[0, 0]) == pytest.approx(0.0)


class TestRollingResidualVariance:
    def test_residual_var_nonneg_always(self) -> None:
        rng = np.random.default_rng(3)
        sym_ret = rng.normal(0, 0.01, (_T, _N))
        btc_ret = rng.normal(0, 0.01, _T)
        beta = np.ones((_T, _N)) * 0.8
        rv = _rolling_residual_variance(sym_ret, btc_ret, beta, lookback=15)
        assert np.all(rv >= 0.0)

    def test_residual_var_zero_at_start(self) -> None:
        rng = np.random.default_rng(4)
        sym_ret = rng.normal(0, 0.01, (_T, _N))
        btc_ret = rng.normal(0, 0.01, _T)
        beta = np.zeros((_T, _N))
        rv = _rolling_residual_variance(sym_ret, btc_ret, beta, lookback=15)
        assert float(rv[0, 0]) == pytest.approx(0.0)

    def test_residual_var_grows_with_noise(self) -> None:
        # 잔차 노이즈가 클수록 residual var이 커야 한다
        rng = np.random.default_rng(5)
        btc_ret = rng.normal(0, 0.01, _T)
        beta = np.zeros((_T, _N))

        low_noise = rng.normal(0, 0.001, (_T, _N))
        high_noise = rng.normal(0, 0.05, (_T, _N))

        rv_low = _rolling_residual_variance(low_noise, btc_ret, beta, lookback=20)
        rv_high = _rolling_residual_variance(high_noise, btc_ret, beta, lookback=20)

        # 후반부(window 충분) 평균 비교
        assert float(rv_high[40:].mean()) > float(rv_low[40:].mean())


class TestCovariancePSD:
    def test_cov_diagonal_positive(self) -> None:
        rf = build_risk_forecast(_close(), _SYMBOLS, "4h", {}, btc_index=_BTC_IDX, lookback=20)
        # 대각 원소는 분산이므로 >= 0
        for t in range(1, _T):
            diag = np.diag(rf.covariance_3d[t])
            assert np.all(diag >= -1e-9), f"bar {t}: negative diagonal {diag.min()}"
