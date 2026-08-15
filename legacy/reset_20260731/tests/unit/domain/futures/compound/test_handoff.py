from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.config import CalibrationConfig, HandoffConfig
from src.domain.futures.compound.contracts import (
    CausalityError,
    HandoffAdmissionEvidence,
    HandoffResult,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.bar_engine import align_costs_to_decision_grid
from src.domain.futures.compound.handoff import (
    apply_causal_holding_kernel,
    build_prequential_handoff,
)

_HOUR_NS = 3_600_000_000_000


def _planted_panel(
    t: int = 200, n: int = 5, k: int = 3,
) -> RawSignalPanel:
    rng = np.random.default_rng(42)
    z = rng.standard_normal((t, n, k)).astype(np.float32).clip(-3, 3)
    desc = (
        SignalDescriptor("trend:fast", "trend", "fast", 24, "4h", target_horizon_hours=4),
        SignalDescriptor("trend:slow", "trend", "slow", 216, "4h", target_horizon_hours=216),
        SignalDescriptor("momentum:fast", "momentum", "fast", 24, "4h", target_horizon_hours=4),
    )
    return RawSignalPanel(
        decision_timestamps_ns=np.arange(t, dtype=np.int64) * 4 * _HOUR_NS,
        symbols=tuple(f"S{i}" for i in range(n)),
        descriptors=desc,
        z_3d=z,
        valid_3d=np.ones((t, n, k), dtype=np.bool_),
        sigma_2d=np.full((t, n), 0.02, dtype=np.float32),
    )


def _planted_bars(t: int = 200, n: int = 5) -> TimeframeBarCube:
    return TimeframeBarCube(
        timeframe="4h",
        timestamps_ns=np.arange(t, dtype=np.int64) * 4 * _HOUR_NS,
        symbols=tuple(f"S{i}" for i in range(n)),
        open_2d=np.full((t, n), 100.0, dtype=np.float32),
        high_2d=np.full((t, n), 101.0, dtype=np.float32),
        low_2d=np.full((t, n), 99.0, dtype=np.float32),
        close_2d=np.column_stack([np.linspace(100, 110 + i, t) for i in range(n)]).astype(np.float32),
        quote_volume_2d=np.full((t, n), 1e6, dtype=np.float32),
        complete_2d=np.ones((t, n), dtype=np.bool_),
    )


class TestApplyCausalHoldingKernel:
    def test_basic_smoothing(self):
        T, N, K = 6, 2, 1
        forecast = np.array([[[2.0], [4.0]], [[6.0], [8.0]], [[10.0], [12.0]],
                             [[14.0], [16.0]], [[18.0], [20.0]], [[22.0], [24.0]]], dtype=np.float32)
        hb = np.array([2], dtype=np.int16)
        result = apply_causal_holding_kernel(forecast, hb)
        expected_t0 = forecast[0] / 2
        expected_t1 = (forecast[0] + forecast[1]) / 2
        expected_t2 = (forecast[1] + forecast[2]) / 2
        expected_t3 = (forecast[2] + forecast[3]) / 2
        np.testing.assert_allclose(result[0], expected_t0, rtol=1e-6)
        np.testing.assert_allclose(result[1], expected_t1, rtol=1e-6)
        np.testing.assert_allclose(result[2], expected_t2, rtol=1e-6)
        np.testing.assert_allclose(result[3], expected_t3, rtol=1e-6)

    def test_horizon_one_passthrough(self):
        forecast = np.arange(12, dtype=np.float32).reshape(4, 3, 1)
        hb = np.array([1], dtype=np.int16)
        result = apply_causal_holding_kernel(forecast, hb)
        np.testing.assert_array_equal(result, forecast)

    def test_no_future_leakage(self):
        T, N, K = 5, 1, 1
        forecast = np.zeros((T, N, K), dtype=np.float32)
        forecast[3, 0, 0] = 10.0
        hb = np.array([4], dtype=np.int16)
        result = apply_causal_holding_kernel(forecast, hb)
        # at t=3, window = [0,3], sum = 10.0, a = 10/4 = 2.5
        assert result[3, 0, 0] == 10.0 / 4
        # at t=4, window = [1,4], sum = 10.0, a = 10/4 = 2.5
        assert result[4, 0, 0] == 10.0 / 4
        # t=0 window = [0,0], sum = 0
        assert result[0, 0, 0] == 0.0


class TestAlignCostsToDecisionGrid:
    def test_aligns_last_1h_cost_to_4h_decision(self):
        market_ts = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64) * _HOUR_NS
        decision_ts = np.array([3, 7], dtype=np.int64) * _HOUR_NS
        cost = np.array([[1, 2], [3, 4], [5, 6], [7, 8],
                         [9, 10], [11, 12], [13, 14], [15, 16]], dtype=np.float32)
        result = align_costs_to_decision_grid(market_ts, decision_ts, cost)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[0], cost[3])
        np.testing.assert_array_equal(result[1], cost[7])

    def test_empty_market_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            align_costs_to_decision_grid(
                np.array([], dtype=np.int64),
                np.array([1], dtype=np.int64) * _HOUR_NS,
                np.zeros((0, 2), dtype=np.float32),
            )


class TestBuildPrequentialHandoff:
    def test_no_folds_returns_cash_only(self):
        panel = _planted_panel(t=50, n=3, k=2)
        bars = _planted_bars(t=50, n=3)
        cost = np.full((50, 3), 8.0, dtype=np.float32)
        folds = ()
        result = build_prequential_handoff(
            panel, {}, folds, bars, cost, HandoffConfig(),
        )
        assert isinstance(result, HandoffResult)
        assert not result.evidence.admitted
        assert "no_folds" in result.evidence.reasons
        assert np.all(result.forecast.mu_2d == 0.0)

    def test_shape_mismatch_raises(self):
        panel = _planted_panel(t=50, n=3, k=2)
        bars = _planted_bars(t=50, n=10)
        cost = np.full((50, 3), 8.0, dtype=np.float32)
        cfg = CalibrationConfig(n_folds=3, purge_bars=2, embargo_bars=2)
        from src.domain.futures.compound.calibration import build_folds_4h
        folds = build_folds_4h(50, cfg)
        with pytest.raises((ValueError, CausalityError)):
            build_prequential_handoff(
                panel, {}, folds, bars, cost, HandoffConfig(),
            )

    def test_no_candidates_returns_cash_only(self):
        panel = _planted_panel(t=50, n=3, k=2)
        bars = _planted_bars(t=50, n=3)
        cost = np.full((50, 3), 8.0, dtype=np.float32)
        config = HandoffConfig()
        cfg = CalibrationConfig(n_folds=3, purge_bars=2, embargo_bars=2)
        from src.domain.futures.compound.calibration import build_calibration_target, build_folds_4h
        folds = build_folds_4h(50, cfg)
        targets = {}
        for h in (4, 216):
            targets[h] = build_calibration_target(
                type("MB", (), {
                    "decision_timestamps_ns": bars.timestamps_ns,
                    "cubes": {"4h": bars},
                })(),
                panel.sigma_2d, horizon_bars=h // 4,
            )
        result = build_prequential_handoff(
            panel, targets, folds, bars, cost, config,
        )
        assert isinstance(result, HandoffResult)

    def test_admitted_when_all_conditions_met(self):
        t = 120
        panel = _planted_panel(t=t, n=2, k=3)
        bars = _planted_bars(t=t, n=2)
        cost = np.full((t, 2), 8.0, dtype=np.float32)
        from src.domain.futures.compound.calibration import build_calibration_target, build_folds_4h
        cfg = CalibrationConfig(n_folds=3, purge_bars=2, embargo_bars=2)
        folds = build_folds_4h(t, cfg)
        targets = {}
        for h in (4, 216):
            targets[h] = build_calibration_target(
                type("MB", (), {
                    "decision_timestamps_ns": bars.timestamps_ns,
                    "cubes": {"4h": bars},
                })(),
                panel.sigma_2d, horizon_bars=h // 4,
            )
        result = build_prequential_handoff(
            panel, targets, folds, bars, cost, HandoffConfig(n_bootstrap=50),
        )
        assert isinstance(result, HandoffResult)
        assert isinstance(result.forecast, type(build_prequential_handoff(panel,{},(),bars,cost,HandoffConfig()).forecast))


class TestHandoffAdmissionEvidence:
    def test_default_construction(self):
        ev = HandoffAdmissionEvidence(
            annualized_log_growth=0.1,
            growth_lcb90=0.05,
            growth_2x_cost=0.05,
            max_drawdown=0.15,
            annual_volatility=0.18,
            positive_outer_folds=4,
            effective_breadth=2.0,
            active_signal_ids=("sig_a", "sig_b"),
            admitted=True,
            reasons=(),
        )
        assert ev.admitted
        assert ev.growth_lcb90 == 0.05

    def test_rejected_construction(self):
        ev = HandoffAdmissionEvidence(
            annualized_log_growth=-0.1,
            growth_lcb90=-0.05,
            growth_2x_cost=-0.05,
            max_drawdown=0.25,
            annual_volatility=0.22,
            positive_outer_folds=2,
            effective_breadth=0.0,
            active_signal_ids=(),
            admitted=False,
            reasons=("max_drawdown_exceeded",),
        )
        assert not ev.admitted


class TestHandoffResult:
    def test_result_roundtrip(self):
        from src.domain.futures.compound.contracts import CalibratedForecastPanel
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(10, dtype=np.int64),
            symbols=("A",),
            mu_2d=np.zeros((10, 1), dtype=np.float32),
            se_2d=np.full((10, 1), np.nan, dtype=np.float32),
            family_mu_3d=np.zeros((10, 1, 1), dtype=np.float32),
            family_ids=(),
            admitted_signal_ids=(),
            fold_manifest_hash="",
        )
        evidence = HandoffAdmissionEvidence(
            annualized_log_growth=0.0, growth_lcb90=0.0, growth_2x_cost=0.0,
            max_drawdown=0.0, annual_volatility=0.0, positive_outer_folds=0,
            effective_breadth=0.0, active_signal_ids=(),
            admitted=False, reasons=("test",),
        )
        result = HandoffResult(forecast=forecast, evidence=evidence)
        assert result.forecast.mu_2d.shape == (10, 1)
        assert not result.evidence.admitted


def test_align_costs_to_decision_grid_importable():
    from src.domain.futures.compound.bar_engine import align_costs_to_decision_grid
    assert align_costs_to_decision_grid is not None


def test_apply_causal_holding_kernel_importable():
    assert apply_causal_holding_kernel is not None


def test_build_prequential_handoff_importable():
    assert build_prequential_handoff is not None
