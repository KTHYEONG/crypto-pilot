from __future__ import annotations

import numpy as np

from src.domain.futures.compound.config import RiskOverlayConfig
from src.domain.futures.compound.contracts import PortfolioDecision
from src.domain.futures.compound.risk_overlay import apply_risk_overlay


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
