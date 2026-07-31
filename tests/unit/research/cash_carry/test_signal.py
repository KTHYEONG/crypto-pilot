from __future__ import annotations

import pandas as pd
import pytest

from src.research.cash_carry.contracts import (
    CarryCostModel,
    CarryHysteresisConfig,
    CashCarrySpec,
)
from src.research.cash_carry.backtest import run_cash_carry_backtest
from src.research.cash_carry.signal import generate_cash_carry_target

_ZERO_COSTS = CarryCostModel(spot_fee_rate=0.0, perp_fee_rate=0.0, slippage_rate=0.0)
# Short-fixture band: lookback=2 lets a two-settlement history decide OPEN;
# min_hold=1/confirm=1 keep the legacy single-reading close semantics intact.
_SHORT_HYST = CarryHysteresisConfig(
    lookback_settlements=2, min_hold_settlements=1, confirm_settlements=1,
)


def _run(data, costs: CarryCostModel = _ZERO_COSTS):
    return run_cash_carry_backtest(
        data, CashCarrySpec(symbol="BTCUSDT"), costs, hysteresis=_SHORT_HYST,
    )


class TestCarryHysteresisConfig:
    def test_hysteresis_config_defaults_are_structural_ratios(self) -> None:
        # SC-CARRY-01: defaults are structurally derived ratios (min_hold ==
        # lookback so the same window that sets the breakeven rate is the
        # window over which it is amortized; confirm == ceil(lookback/3)).
        h = CarryHysteresisConfig()
        assert h.lookback_settlements == 21
        assert h.min_hold_settlements == h.lookback_settlements == 21
        assert h.confirm_settlements == 7

    def test_hysteresis_config_rejects_non_positive_windows(self) -> None:
        for field in ("lookback_settlements", "min_hold_settlements", "confirm_settlements"):
            with pytest.raises(ValueError, match=field):
                CarryHysteresisConfig(**{field: 0})


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
        assert generate_cash_carry_target(data, grid[0], None, _ZERO_COSTS, _SHORT_HYST) == "HOLD"
        assert generate_cash_carry_target(data, grid[1], None, _ZERO_COSTS, _SHORT_HYST) == "OPEN"
        result = _run(data)
        assert len(result.trades) == 1
        assert result.trades.iloc[0]["entry_bar"] == 2

    def test_cash_carry_target_closes_on_nonpositive_net_carry(
        self,
        make_carry_data,
    ) -> None:
        # SC-CARRY-SIGNAL-02: an open pair whose latest settled net carry is
        # non-positive is flat at the next executable bar once the (single)
        # confirm reading is negative.
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
        assert generate_cash_carry_target(data, grid[1], None, _ZERO_COSTS, _SHORT_HYST) == "OPEN"
        assert generate_cash_carry_target(data, grid[2], 1, _ZERO_COSTS, _SHORT_HYST) == "CLOSE"
        result = _run(data)
        assert len(result.trades) == 1
        assert result.trades.iloc[0]["reason"] == "carry_close"
        assert result.trades.iloc[0]["exit_bar"] == 3

    def test_cash_carry_target_closes_when_borrow_consumes_positive_funding(
        self,
        make_carry_data,
    ) -> None:
        # Positive funding alone is not enough: net carry is funding minus the
        # quote-cash borrow cost, and a negative reading closes the pair.
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
        assert generate_cash_carry_target(data, grid[2], 2, _ZERO_COSTS, _SHORT_HYST) == "CLOSE"

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
        assert generate_cash_carry_target(data, grid[1], None, _ZERO_COSTS, _SHORT_HYST) == "HOLD"
        assert generate_cash_carry_target(data, grid[1], 3, _ZERO_COSTS, _SHORT_HYST) == "HOLD"
        assert generate_cash_carry_target(data, grid[2], None, _ZERO_COSTS, _SHORT_HYST) == "OPEN"

    def test_raises_for_unaligned_decision_time(self, make_carry_data) -> None:
        data = make_carry_data(n_bars=2, funding={"2024-01-01 00:00": 0.001}, borrow=[0.0, 0.0])
        with pytest.raises(ValueError, match="aligned"):
            generate_cash_carry_target(
                data, pd.Timestamp("2024-01-01 01:00", tz="UTC"), None, _ZERO_COSTS, _SHORT_HYST,
            )
        with pytest.raises(ValueError, match="Timestamp"):
            generate_cash_carry_target(data, "not-a-timestamp", None, _ZERO_COSTS, _SHORT_HYST)

    def test_accumulates_borrow_since_prior_funding_settlement(
        self,
        make_carry_data,
    ) -> None:
        # SC-CARRY-SIGNAL-02: a settlement reading weighs the borrow accrued
        # over the whole elapsed interval since the prior settlement, not a
        # single arbitrary bar: an 8h funding gap means two bars of borrow are
        # weighed against one funding settlement.
        hc = CarryHysteresisConfig(
            lookback_settlements=3, min_hold_settlements=1, confirm_settlements=1,
        )
        data = make_carry_data(
            n_bars=6,
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 04:00": 0.001,
                "2024-01-01 12:00": 0.001,
                "2024-01-01 16:00": 0.0,
                "2024-01-01 20:00": 0.0,
            },
            borrow=[0.0, 0.0, 0.0007, 0.0004, 0.0, 0.0],
        )
        grid = data.spot.index
        # At bar 3 the 12:00 settlement is weighed against bar2 + bar3 borrow
        # (0.0011 > 0.001 funding) -> reading negative -> close.
        assert generate_cash_carry_target(data, grid[3], 3, _ZERO_COSTS, hc) == "CLOSE"

    def test_borrow_from_first_valid_event_boundary_without_prior_funding(
        self,
        make_carry_data,
    ) -> None:
        # The first funding settlement accrues borrow from the window start,
        # so a zero net reading neither opens nor holds a positive target.
        hc = CarryHysteresisConfig(
            lookback_settlements=3, min_hold_settlements=1, confirm_settlements=1,
        )
        data = make_carry_data(
            n_bars=3,
            funding={"2024-01-01 04:00": 0.001},
            borrow=[0.001, 0.0, 0.0],
        )
        grid = data.spot.index
        assert generate_cash_carry_target(data, grid[1], None, _ZERO_COSTS, hc) == "HOLD"
        assert generate_cash_carry_target(data, grid[1], 2, _ZERO_COSTS, hc) == "CLOSE"


class TestHysteresisBand:
    def test_hysteresis_open_requires_cleared_breakeven(self, make_carry_data) -> None:
        # SC-CARRY-03: flat position with a trailing mean below the cost-derived
        # breakeven stays HOLD even on a fresh settlement.
        hc = CarryHysteresisConfig(
            lookback_settlements=4, min_hold_settlements=4, confirm_settlements=2,
        )
        data = make_carry_data(
            n_bars=6,
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 04:00": 0.0005,
                "2024-01-01 08:00": 0.0005,
                "2024-01-01 12:00": 0.0005,
            },
            borrow=[0.0] * 6,
        )
        grid = data.spot.index
        # mean 0.000375 < breakeven 0.0042/4 = 0.00105 -> no OPEN.
        assert generate_cash_carry_target(data, grid[3], None, CarryCostModel(), hc) == "HOLD"

    def test_hysteresis_open_when_breakeven_cleared(self, make_carry_data) -> None:
        # A synthetic funding/borrow fixture whose trailing mean clears the
        # breakeven on a fresh settlement triggers OPEN.
        hc = CarryHysteresisConfig(
            lookback_settlements=4, min_hold_settlements=4, confirm_settlements=2,
        )
        data = make_carry_data(
            n_bars=6,
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 04:00": 0.002,
                "2024-01-01 08:00": 0.002,
                "2024-01-01 12:00": 0.002,
            },
            borrow=[0.0] * 6,
        )
        grid = data.spot.index
        # mean 0.0015 > breakeven 0.00105 -> OPEN.
        assert generate_cash_carry_target(data, grid[3], None, CarryCostModel(), hc) == "OPEN"

    def test_hysteresis_holds_through_min_hold_window(self, make_carry_data) -> None:
        # An open position below min_hold stays HOLD on negative readings;
        # only once the min-hold window has elapsed does the close rule apply.
        hc = CarryHysteresisConfig(
            lookback_settlements=4, min_hold_settlements=3, confirm_settlements=1,
        )
        data = make_carry_data(
            n_bars=6,
            funding={
                "2024-01-01 00:00": 0.001,
                "2024-01-01 04:00": -0.001,
                "2024-01-01 08:00": -0.001,
                "2024-01-01 12:00": -0.001,
            },
            borrow=[0.0] * 6,
        )
        grid = data.spot.index
        assert generate_cash_carry_target(data, grid[1], 1, _ZERO_COSTS, hc) == "HOLD"
        assert generate_cash_carry_target(data, grid[2], 2, _ZERO_COSTS, hc) == "HOLD"
        assert generate_cash_carry_target(data, grid[3], 3, _ZERO_COSTS, hc) == "CLOSE"

    def test_hysteresis_closes_after_confirm_streak(self, make_carry_data) -> None:
        # Past min_hold, confirm consecutive negative readings close the pair.
        hc = CarryHysteresisConfig(
            lookback_settlements=4, min_hold_settlements=1, confirm_settlements=2,
        )
        data = make_carry_data(
            n_bars=6,
            funding={
                "2024-01-01 00:00": 0.001,
                "2024-01-01 04:00": -0.001,
                "2024-01-01 08:00": -0.001,
                "2024-01-01 12:00": -0.001,
            },
            borrow=[0.0] * 6,
        )
        grid = data.spot.index
        assert generate_cash_carry_target(data, grid[2], 2, _ZERO_COSTS, hc) == "CLOSE"

    def test_hysteresis_single_negative_reading_does_not_close(self, make_carry_data) -> None:
        # Past min_hold, only one of the confirm readings being negative holds
        # the pair instead of closing it.
        hc = CarryHysteresisConfig(
            lookback_settlements=4, min_hold_settlements=1, confirm_settlements=2,
        )
        data = make_carry_data(
            n_bars=6,
            funding={
                "2024-01-01 00:00": 0.001,
                "2024-01-01 04:00": -0.001,
                "2024-01-01 08:00": 0.001,
                "2024-01-01 12:00": -0.001,
            },
            borrow=[0.0] * 6,
        )
        grid = data.spot.index
        assert generate_cash_carry_target(data, grid[3], 3, _ZERO_COSTS, hc) == "HOLD"
