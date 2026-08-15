from __future__ import annotations

import numpy as np


from src.domain.futures.compound.config import RiskOverlayConfig
from src.domain.futures.compound.contracts import PortfolioDecision
from src.domain.futures.compound.risk_overlay import apply_fractional_kelly_scaling, apply_risk_overlay


class TestApplyFractionalKellyScaling:
    def test_quarter_kelly_scales_precisely(self) -> None:
        w = np.array([0.2, 0.15, -0.10], dtype=np.float64)
        port_var = 0.0001
        result = apply_fractional_kelly_scaling(w, port_var)
        ann_vol = np.sqrt(port_var) * np.sqrt(2190.0)
        expected_scale = 0.25 * (0.25 / max(ann_vol, 0.25))
        expected = w * expected_scale
        np.testing.assert_array_almost_equal(result, expected)

    def test_high_vol_brings_vol_below_target(self) -> None:
        w = np.array([0.5, 0.3], dtype=np.float64)
        port_var = 0.001
        result = apply_fractional_kelly_scaling(w, port_var)
        result_ann_vol = np.sqrt(np.sum(result ** 2) * port_var) * np.sqrt(2190.0)
        assert result_ann_vol <= 0.26

    def test_cvar_regime_reduces_scale_further(self) -> None:
        w = np.array([0.2, 0.15], dtype=np.float64)
        port_var = 0.0001
        normal = apply_fractional_kelly_scaling(w, port_var)
        cvar = apply_fractional_kelly_scaling(w, port_var, cvar_regime_active=True)
        assert np.all(cvar < normal)
        np.testing.assert_array_almost_equal(cvar, normal / 2.0)


class TestApplyRiskOverlay:
    def test_no_drawdown_returns_full_scale(self) -> None:
        decision = PortfolioDecision(
            decision_idx=0, decision_time_ns=0,
            target_weights_1d=np.array([0.1, 0.2], dtype=np.float64),
            gross_exposure=0.3, net_exposure=0.3, forecast_ann_vol=0.15,
            risk_scale=1.0, binding_constraints=(),
        )
        equity = np.array([1.0, 1.01, 1.02], dtype=np.float64)
        config = RiskOverlayConfig()
        result = apply_risk_overlay(decision=decision, equity_1d=equity, cooldown_remaining=0, config=config)
        assert result.risk_scale > 0

    def test_at_twenty_percent_drawdown_enters_cooldown(self) -> None:
        decision = PortfolioDecision(
            decision_idx=0, decision_time_ns=0,
            target_weights_1d=np.array([0.1, 0.2], dtype=np.float64),
            gross_exposure=0.3, net_exposure=0.3, forecast_ann_vol=0.15,
            risk_scale=1.0, binding_constraints=(),
        )
        equity = np.array([1.0, 1.5, 1.2], dtype=np.float64)
        config = RiskOverlayConfig()
        result = apply_risk_overlay(decision=decision, equity_1d=equity, cooldown_remaining=0, config=config)
        assert result.hard_block_reason == "hard_drawdown"
        assert np.all(result.target_weights_1d == 0)

    def test_cooldown_reduces_each_call(self) -> None:
        decision = PortfolioDecision(
            decision_idx=0, decision_time_ns=0,
            target_weights_1d=np.array([0.1], dtype=np.float64),
            gross_exposure=0.1, net_exposure=0.1, forecast_ann_vol=0.15,
            risk_scale=1.0, binding_constraints=(),
        )
        equity = np.array([1.0], dtype=np.float64)
        config = RiskOverlayConfig()
        result = apply_risk_overlay(decision=decision, equity_1d=equity, cooldown_remaining=5, config=config)
        assert result.cooldown_remaining == 4
        assert result.hard_block_reason == "cooldown"


def test_risk_overlay_at_twenty_percent_drawdown_enters_cooldown() -> None:
    decision = PortfolioDecision(
        decision_idx=0, decision_time_ns=0,
        target_weights_1d=np.array([0.1, 0.2], dtype=np.float64),
        gross_exposure=0.3, net_exposure=0.3, forecast_ann_vol=0.15,
        risk_scale=1.0, binding_constraints=(),
    )
    equity = np.array([1.0, 1.5, 1.2], dtype=np.float64)
    config = RiskOverlayConfig()
    result = apply_risk_overlay(decision=decision, equity_1d=equity, cooldown_remaining=0, config=config)
    assert result.hard_block_reason == "hard_drawdown"
    assert np.all(result.target_weights_1d == 0)
