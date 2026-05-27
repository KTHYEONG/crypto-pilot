"""Tests for forecast/cost.py — build_cost_forecast."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.settings import round_trip_cost_bps
from src.domain.futures.forecast.cost import CostModelConfig, build_cost_forecast

_T, _N = 30, 4
_SHAPE = (_T, _N)


def _close() -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.cumprod(1 + rng.normal(0, 0.01, _SHAPE), axis=0) * 100.0


def _volume() -> np.ndarray:
    return np.ones(_SHAPE) * 1_000_000.0


class TestBuildCostForecastFallback:
    def test_fallback_when_no_universe_cost(self) -> None:
        # universe_cost_bps_2d=None → fallback_global
        cf = build_cost_forecast(
            _close(), None, None, _volume(), None, None, None,
            CostModelConfig(), shape=_SHAPE,
        )
        expected_bps = round_trip_cost_bps()
        np.testing.assert_allclose(cf.execution_cost_bps_2d, expected_bps, rtol=1e-9)
        assert cf.source == "fallback_global"

    def test_fallback_on_wrong_shape(self) -> None:
        wrong = np.full((_T + 5, _N), 20.0)
        cf = build_cost_forecast(
            _close(), None, None, _volume(), None, None, wrong,
            CostModelConfig(), shape=_SHAPE,
        )
        assert cf.source == "fallback_global"

    def test_fallback_on_all_nan(self) -> None:
        # valid shape이지만 전부 NaN → 셀별로 fallback_bps 채움, source는 universe_static
        nan_cost = np.full(_SHAPE, np.nan)
        cf = build_cost_forecast(
            _close(), None, None, _volume(), None, None, nan_cost,
            CostModelConfig(), shape=_SHAPE,
        )
        fallback = round_trip_cost_bps()
        # 모든 셀이 fallback 값으로 채워져야 한다
        np.testing.assert_allclose(cf.execution_cost_bps_2d, fallback, rtol=1e-6)
        assert cf.source == "universe_static"  # shape 매칭으로 universe_static 경로 진입


class TestBuildCostForecastUniverseStatic:
    def test_universe_static_used_when_valid(self) -> None:
        per_sym = np.full(_SHAPE, 25.0)
        cf = build_cost_forecast(
            _close(), None, None, _volume(), None, None, per_sym,
            CostModelConfig(), shape=_SHAPE,
        )
        np.testing.assert_allclose(cf.execution_cost_bps_2d, 25.0, rtol=1e-9)
        assert cf.source == "universe_static"

    def test_floor_applied_on_zero_cost_rows(self) -> None:
        # 일부 셀이 0 (invalid) → fallback_bps 적용
        per_sym = np.full(_SHAPE, 20.0)
        per_sym[5, :] = 0.0  # invalid row
        cf = build_cost_forecast(
            _close(), None, None, _volume(), None, None, per_sym,
            CostModelConfig(), shape=_SHAPE,
        )
        fallback = round_trip_cost_bps()
        assert float(cf.execution_cost_bps_2d[5, 0]) == pytest.approx(fallback, rel=1e-6)

    def test_phase1_matches_existing_static_cost(self) -> None:
        # Phase 1에서 universe cost가 있으면 그대로 사용 (동일 수치)
        static_bps = 18.0
        per_sym = np.full(_SHAPE, static_bps)
        cf = build_cost_forecast(
            _close(), None, None, _volume(), None, None, per_sym,
            CostModelConfig(), shape=_SHAPE,
        )
        np.testing.assert_allclose(cf.execution_cost_bps_2d, static_bps, rtol=1e-9)


class TestBuildCostForecastOutputContract:
    def test_fraction_equals_bps_over_10000(self) -> None:
        per_sym = np.full(_SHAPE, 20.0)
        cf = build_cost_forecast(
            _close(), None, None, _volume(), None, None, per_sym,
            CostModelConfig(), shape=_SHAPE,
        )
        np.testing.assert_allclose(
            cf.execution_cost_fraction_2d, cf.execution_cost_bps_2d / 10000.0, rtol=1e-9
        )

    def test_uncertainty_nonnegative(self) -> None:
        cf = build_cost_forecast(
            _close(), None, None, _volume(), None, None, None,
            CostModelConfig(), shape=_SHAPE,
        )
        assert np.all(cf.uncertainty_bps_2d >= 0.0)

    def test_output_shape_matches(self) -> None:
        cf = build_cost_forecast(
            _close(), None, None, _volume(), None, None, None,
            CostModelConfig(), shape=_SHAPE,
        )
        assert cf.execution_cost_bps_2d.shape == _SHAPE
        assert cf.execution_cost_fraction_2d.shape == _SHAPE
        assert cf.uncertainty_bps_2d.shape == _SHAPE

    def test_all_finite(self) -> None:
        cf = build_cost_forecast(
            _close(), None, None, _volume(), None, None, None,
            CostModelConfig(), shape=_SHAPE,
        )
        assert np.all(np.isfinite(cf.execution_cost_bps_2d))
        assert np.all(np.isfinite(cf.execution_cost_fraction_2d))


class TestBuildCostForecastDynamic:
    def test_dynamic_components_raise_cost_above_floor(self) -> None:
        per_sym = np.full(_SHAPE, 15.0, dtype=np.float64)
        high = _close() * 1.002
        low = _close() * 0.998
        funding = np.full(_SHAPE, 0.0001, dtype=np.float64)
        cfg = CostModelConfig(
            vol_buffer_coef=0.2,
            latency_buffer_bps=0.5,
            impact_coef=0.5,
            funding_event_buffer_bps=0.2,
            estimated_order_notional=50_000.0,
            enable_dynamic_components=True,
        )
        cf = build_cost_forecast(
            _close(), high, low, _volume(), funding, None, per_sym, cfg, shape=_SHAPE
        )
        assert cf.source == "parametric_dynamic"
        assert np.all(cf.execution_cost_bps_2d >= 15.0)
        # floor semantics: dynamic candidate may be below floor on some cells,
        # but never below floor overall.
        assert np.all(cf.execution_cost_bps_2d >= per_sym)

    def test_dynamic_capacity_present_when_order_notional_positive(self) -> None:
        cfg = CostModelConfig(
            estimated_order_notional=10_000.0,
            enable_dynamic_components=True,
        )
        cf = build_cost_forecast(
            _close(),
            _close() * 1.001,
            _close() * 0.999,
            _volume(),
            np.zeros(_SHAPE),
            None,
            np.full(_SHAPE, 10.0),
            cfg,
            shape=_SHAPE,
        )
        assert cf.capacity_notional_2d is not None

    def test_dynamic_floor_is_max_of_floor_and_parametric(self) -> None:
        floor = np.full(_SHAPE, 20.0, dtype=np.float64)
        cfg = CostModelConfig(
            taker_fee_bps=4.0,
            vol_buffer_coef=0.0,
            latency_buffer_bps=0.0,
            impact_coef=0.0,
            funding_event_buffer_bps=0.0,
            estimated_order_notional=0.0,
            enable_dynamic_components=True,
        )
        cf = build_cost_forecast(
            _close(),
            None,
            None,
            _volume(),
            None,
            None,
            floor,
            cfg,
            shape=_SHAPE,
        )
        # parametric_total ~= taker_fee(4bps) < floor(20bps) 이므로 floor 유지
        np.testing.assert_allclose(cf.execution_cost_bps_2d, floor, rtol=1e-9)
