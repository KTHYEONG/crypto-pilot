from __future__ import annotations

import pytest

from src.research.baseline.backtest import calculate_position_size


class TestPositionSizing:
    def test_exact_and_clamped(self) -> None:
        assert calculate_position_size(
            equity=10_000.0, risk_fraction=0.01,
            entry_price=100.0, stop_price=95.0, max_leverage=100.0,
        ) == 20.0
        assert calculate_position_size(
            equity=10_000.0, risk_fraction=0.005,
            entry_price=100.0, stop_price=99.99, max_leverage=2.0,
        ) == 200.0

    def test_raises_on_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="equity"):
            calculate_position_size(equity=0, risk_fraction=0.01, entry_price=100.0, stop_price=95.0, max_leverage=2.0)
        with pytest.raises(ValueError, match="risk_fraction"):
            calculate_position_size(equity=10000, risk_fraction=0, entry_price=100.0, stop_price=95.0, max_leverage=2.0)
        with pytest.raises(ValueError, match="risk_fraction"):
            calculate_position_size(equity=10000, risk_fraction=1.5, entry_price=100.0, stop_price=95.0, max_leverage=2.0)
        with pytest.raises(ValueError, match="stop_distance"):
            calculate_position_size(equity=10000, risk_fraction=0.01, entry_price=100.0, stop_price=100.0, max_leverage=2.0)
