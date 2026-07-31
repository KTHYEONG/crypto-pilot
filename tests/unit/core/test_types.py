from __future__ import annotations

import pytest

from src.core.types import CostModel, StrategySpec


def test_strategy_spec_contract() -> None:
    spec = StrategySpec()
    assert spec.symbol == "BTCUSDT"
    assert spec.entry_period == 20
    assert spec.timeframe == "4h"


def test_cost_model_round_trip_bps() -> None:
    cm = CostModel()
    assert abs(cm.round_trip_bps() - 16.0) < 1e-6


class TestStrategySpec:
    def test_defaults_are_frozen(self) -> None:
        s = StrategySpec()
        assert (s.entry_period, s.exit_period, s.ema_period, s.atr_period,
                s.stop_atr_mult, s.risk_per_trade, s.max_leverage) == (20, 10, 200, 14, 2.0, 0.005, 2.0)
        with pytest.raises(AttributeError):
            s.entry_period = 30  # type: ignore[misc]

    def test_validation_raises(self) -> None:
        with pytest.raises(ValueError, match="entry_period"):
            StrategySpec(entry_period=0)
        with pytest.raises(ValueError, match="stop_atr_mult"):
            StrategySpec(stop_atr_mult=0)
        with pytest.raises(ValueError, match="risk_per_trade"):
            StrategySpec(risk_per_trade=0)
        with pytest.raises(ValueError, match="risk_per_trade"):
            StrategySpec(risk_per_trade=1.5)
        with pytest.raises(ValueError, match="max_leverage"):
            StrategySpec(max_leverage=0)


class TestCostModel:
    def test_round_trip_and_adverse_slippage(self) -> None:
        c = CostModel()
        assert c.round_trip_bps() == 16.0
        assert c.buy_fill(100.0) == 100.03
        assert c.sell_fill(100.0) == 99.97

    def test_validation_raises(self) -> None:
        with pytest.raises(ValueError, match="fee_rate"):
            CostModel(fee_rate=-0.01)
        with pytest.raises(ValueError, match="slippage_rate"):
            CostModel(slippage_rate=-0.01)

    def test_buy_sell_fill_raises_on_non_positive(self) -> None:
        c = CostModel()
        with pytest.raises(ValueError, match="price must be > 0"):
            c.buy_fill(0)
        with pytest.raises(ValueError, match="price must be > 0"):
            c.sell_fill(-1.0)
