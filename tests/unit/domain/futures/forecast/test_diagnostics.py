"""Tests for forecast/diagnostics.py — LabelDiagnostics + diag functions."""
from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.forecast.contracts import (
    AlphaArtifactHash,
    AlphaForecast,
    CostForecast,
    RiskForecast,
)
from src.domain.futures.forecast.diagnostics import (
    LabelDiagnostics,
    alpha_diagnostics,
    cost_diagnostics,
    risk_diagnostics,
)

_SHAPE = (20, 4)
_DUMMY_HASH = AlphaArtifactHash("", "", "", "", "", "test", 0)


def _make_alpha(long_val: float = 0.005, short_val: float = 0.003) -> AlphaForecast:
    al = np.full(_SHAPE, long_val, dtype=np.float32)
    as_ = np.full(_SHAPE, short_val, dtype=np.float32)
    return AlphaForecast(
        datetimes=np.array([]), symbols=(), alpha_long_2d=al, alpha_short_2d=as_,
        q10_long_2d=None, q50_long_2d=None, q90_long_2d=None,
        q10_short_2d=None, q50_short_2d=None, q90_short_2d=None,
        confidence_long_2d=np.full(_SHAPE, 0.8), confidence_short_2d=np.full(_SHAPE, 0.7),
        eligible_mask=np.ones(_SHAPE, dtype=bool), source="test", artifact_hash=_DUMMY_HASH,
    )


def _make_cost(bps: float = 14.0) -> CostForecast:
    bps_2d = np.full(_SHAPE, bps)
    return CostForecast(
        execution_cost_bps_2d=bps_2d, execution_cost_fraction_2d=bps_2d / 10000.0,
        uncertainty_bps_2d=np.full(_SHAPE, bps * 0.1), capacity_notional_2d=None, source="test",
    )


def _make_risk() -> RiskForecast:
    t, n = _SHAPE
    cov = np.eye(n)[np.newaxis, :, :].repeat(t, axis=0) * 1e-4
    beta = np.full(_SHAPE, 0.8)
    res_var = np.full(_SHAPE, 1e-4)
    vol = np.full(_SHAPE, 0.02)
    return RiskForecast(
        covariance_3d=cov, beta_2d=beta, residual_var_2d=res_var,
        forecast_vol_2d=vol, beta_source="trailing_btc", source="test",
    )


class TestLabelDiagnostics:
    def test_stores_three_arrays(self) -> None:
        arr = np.ones(_SHAPE)
        ld = LabelDiagnostics(
            cost_clearance_target=arr,
            cost_clearance_target_long=arr * 0.5,
            cost_clearance_target_short=arr * 0.3,
        )
        assert ld.cost_clearance_target.shape == _SHAPE
        assert ld.cost_clearance_target_long.shape == _SHAPE
        assert ld.cost_clearance_target_short.shape == _SHAPE

    def test_is_frozen(self) -> None:
        ld = LabelDiagnostics(np.zeros(_SHAPE), np.zeros(_SHAPE), np.zeros(_SHAPE))
        with pytest.raises((AttributeError, TypeError)):
            ld.cost_clearance_target = np.ones(_SHAPE)  # type: ignore[misc]


class TestAlphaDiagnostics:
    def test_returns_dict_with_nz_ratios(self) -> None:
        af = _make_alpha(long_val=0.005, short_val=0.003)
        diag = alpha_diagnostics(af)
        assert "alpha_long_nz_ratio" in diag
        assert "alpha_short_nz_ratio" in diag
        assert 0.0 <= diag["alpha_long_nz_ratio"] <= 1.0

    def test_all_nonzero_when_uniform(self) -> None:
        af = _make_alpha(long_val=0.005)
        diag = alpha_diagnostics(af)
        assert diag["alpha_long_nz_ratio"] == pytest.approx(1.0)

    def test_confidence_mean_reported(self) -> None:
        af = _make_alpha()
        diag = alpha_diagnostics(af)
        assert "confidence_long_mean" in diag
        assert diag["confidence_long_mean"] == pytest.approx(0.8, rel=1e-5)
        assert "confidence_short_mean" in diag
        assert diag["confidence_short_mean"] == pytest.approx(0.7, rel=1e-5)

    def test_percentiles_monotone(self) -> None:
        rng = np.random.default_rng(42)
        al = np.abs(rng.normal(0, 0.01, _SHAPE)).astype(np.float32)
        af = AlphaForecast(
            datetimes=np.array([]), symbols=(), alpha_long_2d=al,
            alpha_short_2d=al * 0.8, q10_long_2d=None, q50_long_2d=None,
            q90_long_2d=None, q10_short_2d=None, q50_short_2d=None, q90_short_2d=None,
            confidence_long_2d=None, confidence_short_2d=None,
            eligible_mask=np.ones(_SHAPE, dtype=bool), source="test", artifact_hash=_DUMMY_HASH,
        )
        diag = alpha_diagnostics(af)
        assert diag["alpha_long_p50_bps"] <= diag["alpha_long_p95_bps"]
        assert diag["alpha_long_p95_bps"] <= diag["alpha_long_p99_bps"]

    def test_zero_alpha_all_zero_nz(self) -> None:
        af = _make_alpha(long_val=0.0, short_val=0.0)
        diag = alpha_diagnostics(af)
        assert diag["alpha_long_nz_ratio"] == pytest.approx(0.0)
        assert diag["alpha_short_nz_ratio"] == pytest.approx(0.0)


class TestCostDiagnostics:
    def test_returns_expected_keys(self) -> None:
        cf = _make_cost(bps=20.0)
        diag = cost_diagnostics(cf)
        assert "cost_mean_bps" in diag
        assert "cost_p95_bps" in diag
        assert "uncertainty_mean_bps" in diag

    def test_mean_bps_matches_input(self) -> None:
        cf = _make_cost(bps=20.0)
        diag = cost_diagnostics(cf)
        assert diag["cost_mean_bps"] == pytest.approx(20.0, rel=1e-5)
        assert diag["cost_p95_bps"] == pytest.approx(20.0, rel=1e-5)


class TestRiskDiagnostics:
    def test_returns_expected_keys(self) -> None:
        rf = _make_risk()
        diag = risk_diagnostics(rf)
        assert "residual_var_nan_ratio" in diag
        assert "beta_nonzero_ratio" in diag
        assert "cov_psd_fail_count" in diag

    def test_no_nan_in_valid_risk(self) -> None:
        rf = _make_risk()
        diag = risk_diagnostics(rf)
        assert diag["residual_var_nan_ratio"] == pytest.approx(0.0)

    def test_beta_nonzero_when_all_nonzero(self) -> None:
        rf = _make_risk()
        diag = risk_diagnostics(rf)
        assert diag["beta_nonzero_ratio"] == pytest.approx(1.0)

    def test_psd_pass_on_identity_cov(self) -> None:
        rf = _make_risk()
        diag = risk_diagnostics(rf)
        assert diag["cov_psd_fail_count"] == pytest.approx(0.0)
