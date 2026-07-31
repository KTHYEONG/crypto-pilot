from __future__ import annotations

import pandas as pd
import pytest

from src.core.types import CashCarrySpec, CarryCostModel
from src.engine.cash_carry_backtest import run_cash_carry_backtest
from src.strategy.cash_carry import generate_cash_carry_target


def _run(data, zero_costs: bool = True):
    costs = CarryCostModel() if not zero_costs else CarryCostModel(
        spot_fee_rate=0.0, perp_fee_rate=0.0, slippage_rate=0.0,
    )
    return run_cash_carry_backtest(data, CashCarrySpec(symbol="BTCUSDT"), costs)


class TestCashCarryTarget:
    def test_cash_carry_target_waits_until_next_bar_after_settled_funding(
        self,
        make_carry_data,
    ) -> None:
        # SC-CARRY-SIGNAL-01: positive funding settled at bar 1 only makes the
        # target executable at the next bar (bar 2), never for the same event.
        data = make_carry_data(
            n_bars=4,
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 04:00": 0.001,
                "2024-01-01 08:00": 0.0,
                "2024-01-01 12:00": 0.0,
            },
            borrow=[0.0, 0.0, 0.0, 0.0],
        )
        grid = data.spot.index
        assert generate_cash_carry_target(data, grid[0], is_open=False) == "HOLD"
        assert generate_cash_carry_target(data, grid[1], is_open=False) == "OPEN"
        result = _run(data)
        assert len(result.trades) == 1
        assert result.trades.iloc[0]["entry_bar"] == 2

    def test_cash_carry_target_closes_on_nonpositive_net_carry(
        self,
        make_carry_data,
    ) -> None:
        # SC-CARRY-SIGNAL-02: an open pair with settled net carry <= 0 is flat
        # at the next executable bar.
        data = make_carry_data(
            n_bars=4,
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 04:00": 0.001,
                "2024-01-01 08:00": -0.001,
                "2024-01-01 12:00": 0.0,
            },
            borrow=[0.0, 0.0, 0.0, 0.0],
        )
        grid = data.spot.index
        assert generate_cash_carry_target(data, grid[1], is_open=False) == "OPEN"
        assert generate_cash_carry_target(data, grid[2], is_open=True) == "CLOSE"
        result = _run(data)
        assert len(result.trades) == 1
        assert result.trades.iloc[0]["reason"] == "carry_close"
        assert result.trades.iloc[0]["exit_bar"] == 3

    def test_cash_carry_target_closes_when_borrow_consumes_positive_funding(
        self,
        make_carry_data,
    ) -> None:
        # Positive funding alone is not enough: net carry is funding minus the
        # quote-cash borrow cost.
        data = make_carry_data(
            n_bars=4,
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 04:00": 0.001,
                "2024-01-01 08:00": 0.001,
                "2024-01-01 12:00": 0.0,
            },
            borrow=[0.0, 0.0, 0.002, 0.0],
        )
        grid = data.spot.index
        assert generate_cash_carry_target(data, grid[2], is_open=True) == "CLOSE"

    def test_cash_carry_target_preserves_state_on_bar_without_funding_event(
        self,
        make_carry_data,
    ) -> None:
        # A bar with no fresh settlement never flips the state on its own.
        data = make_carry_data(
            n_bars=5,
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 08:00": 0.001,
                "2024-01-01 16:00": 0.0,
            },
            borrow=[0.0, 0.0, 0.0, 0.0, 0.0],
        )
        grid = data.spot.index
        assert generate_cash_carry_target(data, grid[1], is_open=False) == "HOLD"
        assert generate_cash_carry_target(data, grid[1], is_open=True) == "HOLD"
        assert generate_cash_carry_target(data, grid[2], is_open=False) == "OPEN"

    def test_raises_for_unaligned_decision_time(self, make_carry_data) -> None:
        data = make_carry_data(n_bars=2, funding={"2024-01-01 00:00": 0.001}, borrow=[0.0, 0.0])
        with pytest.raises(ValueError, match="aligned"):
            generate_cash_carry_target(
                data, pd.Timestamp("2024-01-01 01:00", tz="UTC"), is_open=False,
            )
        with pytest.raises(ValueError, match="Timestamp"):
            generate_cash_carry_target(data, "not-a-timestamp", is_open=False)
