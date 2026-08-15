from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.research.cash_carry.contracts import CarryCostModel, CashCarrySpec
from src.research.cash_carry.contracts import CarryHysteresisConfig, CarryMarketData
from src.research.cash_carry.backtest import run_cash_carry_backtest
from src.research.cash_carry.market_data import load_carry_market_data
from src.research.evaluation.promotion import compose_promotion_verdict
from src.research.evaluation.reliability import (
    ReliabilityGateConfig,
    compute_equity_reliability_gate,
    compute_fold_distribution,
)

SPEC = CashCarrySpec(symbol="BTCUSDT")
ZERO_COSTS = CarryCostModel(spot_fee_rate=0.0, perp_fee_rate=0.0, slippage_rate=0.0)
# Short-fixture band for the ledger mechanics tests: a two-settlement history
# can decide OPEN and a single negative reading closes, so the 6-bar fixtures
# keep exercising the ledger rather than the hysteresis gate itself.
HYST = CarryHysteresisConfig(
    lookback_settlements=2, min_hold_settlements=1, confirm_settlements=1,
)


def _run(
    data,
    *,
    costs: CarryCostModel = ZERO_COSTS,
    delay: int = 0,
    hysteresis: CarryHysteresisConfig = HYST,
):
    return run_cash_carry_backtest(
        data, SPEC, costs, signal_delay_bars=delay, hysteresis=hysteresis,
    )


class TestCashCarryLedger:
    def test_cash_carry_ledger_is_delta_neutral_before_costs_and_funding(
        self,
        make_carry_data,
    ) -> None:
        # SC-CARRY-LEDGER-01: spot +1 / perp short -1 at equal quantity means
        # equal price moves cancel before financing and cost; the only P&L of a
        # fully delta-hedged hold is the accrued short-side funding.
        ramp = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
        data = make_carry_data(
            n_bars=6,
            spot_open=ramp, spot_close=ramp,
            perp_open=ramp, perp_close=ramp,
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 04:00": 0.001,
                "2024-01-01 08:00": 0.001,
                "2024-01-01 12:00": 0.001,
                "2024-01-01 16:00": 0.0,
                "2024-01-01 20:00": 0.0,
            },
            borrow=[0.0] * 6,
        )
        result = _run(data)
        equity = result.equity
        assert np.isfinite(equity.to_numpy()).all()
        assert (equity.to_numpy() > 0).all()
        # entry at bar 2 leaves equity unchanged (no costs, no funding charged
        # on the entry bar); the held bar 4 has zero funding/borrow so the equal
        # +2 price move must leave equity flat.
        assert np.isclose(equity.iloc[2], 10_000.0)
        assert np.isclose(equity.iloc[4], equity.iloc[3])
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["reason"] == "carry_close"
        # qty = equity / (fill_spot * (1 + initial_margin)) with zero costs;
        # the only funding credited is the 12:00 settlement at bar 3.
        qty = 10_000.0 / (104.0 * (1 + SPEC.initial_margin_rate))
        expected_funding = qty * 106.0 * 0.001
        assert np.isclose(trade["funding_pnl"], expected_funding)
        assert np.isclose(trade["pnl"], trade["funding_pnl"])

    def test_cash_carry_ledger_charges_both_legs_and_credits_short(
        self,
        make_carry_data,
    ) -> None:
        # SC-CARRY-LEDGER-02: positive funding credits the short leg and both
        # legs incur fees and slippage, so the costed ledger trails the
        # zero-cost ledger by the full round-trip cost. Funding is set above
        # the costed breakeven (0.0042/2) so both ledgers actually open.
        ramp = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0]
        data = make_carry_data(
            n_bars=7,
            spot_open=ramp, spot_close=ramp,
            perp_open=ramp, perp_close=ramp,
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 04:00": 0.003,
                "2024-01-01 08:00": 0.003,
                "2024-01-01 12:00": 0.003,
                "2024-01-01 16:00": 0.003,
                "2024-01-01 20:00": 0.0,
                "2024-01-02 00:00": 0.0,
            },
            borrow=[0.0] * 7,
        )
        zero = _run(data)
        costed = _run(data, costs=CarryCostModel())
        assert len(costed.trades) == 1
        trade = costed.trades.iloc[0]
        assert trade["funding_pnl"] > 0.0
        assert trade["pnl"] < trade["funding_pnl"]
        assert costed.equity.iloc[-1] < zero.equity.iloc[-1]

    def test_cash_carry_ledger_force_closes_on_maintenance_violation(
        self,
        make_carry_data,
    ) -> None:
        # SC-CARRY-LEDGER-03: an adverse basis that exhausts the maintenance
        # buffer force-closes at the adverse executable mark and records
        # margin_liquidation while equity stays finite. The window ends on the
        # liquidation bar so the re-entry signal has no bar to execute on.
        data = make_carry_data(
            n_bars=4,
            spot_open=[100.0] * 4, spot_close=[100.0] * 4,
            perp_open=[100.0] * 4,
            perp_close=[100.0, 100.0, 100.0, 115.0],
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 04:00": 0.001,
                "2024-01-01 08:00": 0.001,
                "2024-01-01 12:00": 0.0,
            },
            borrow=[0.0] * 4,
        )
        result = _run(data)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["reason"] == "margin_liquidation"
        assert trade["pnl"] < 0.0
        equity = result.equity
        assert np.isfinite(equity.to_numpy()).all()
        assert (equity.to_numpy() > 0).all()

    def test_signal_delay_pushes_execution_one_bar_later(self, make_carry_data) -> None:
        data = make_carry_data(
            n_bars=6,
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 04:00": 0.001,
                "2024-01-01 08:00": 0.0,
                "2024-01-01 12:00": 0.0,
                "2024-01-01 16:00": 0.0,
                "2024-01-01 20:00": 0.0,
            },
            borrow=[0.0] * 6,
        )
        base = _run(data)
        delayed = _run(data, delay=1)
        assert base.trades.iloc[0]["entry_bar"] == 2
        assert delayed.trades.iloc[0]["entry_bar"] == 3


class TestHysteresisWiring:
    def test_backtest_tracks_settlements_since_open(self, make_carry_data) -> None:
        # The backtest counts only fresh-settlement bars toward min_hold: the
        # position enters at bar 1, survives negative readings at bars 2 and 4,
        # and closes only once three settlements have elapsed at bar 6.
        hc = CarryHysteresisConfig(
            lookback_settlements=1, min_hold_settlements=3, confirm_settlements=1,
        )
        data = make_carry_data(
            n_bars=8,
            funding={
                "2024-01-01 00:00": 0.001,
                "2024-01-01 08:00": -0.001,
                "2024-01-01 16:00": -0.001,
                "2024-01-02 00:00": -0.001,
            },
            borrow=[0.0] * 8,
        )
        result = _run(data, hysteresis=hc)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["entry_bar"] == 1
        assert trade["exit_bar"] == 7
        assert trade["reason"] == "carry_close"

    def test_backtest_default_hysteresis_matches_frozen_config(self) -> None:
        # Calling run_cash_carry_backtest without hysteresis must wire the
        # frozen CarryHysteresisConfig() defaults (regression-lock of the
        # default-arg wiring), while an explicitly different band must diverge.
        data = _two_year_carry_data()
        result = run_cash_carry_backtest(data, SPEC, ZERO_COSTS)
        explicit = run_cash_carry_backtest(
            data, SPEC, ZERO_COSTS, hysteresis=CarryHysteresisConfig(),
        )
        permissive = run_cash_carry_backtest(
            data, SPEC, ZERO_COSTS,
            hysteresis=CarryHysteresisConfig(
                lookback_settlements=2, min_hold_settlements=1, confirm_settlements=1,
            ),
        )
        assert result.trades.equals(explicit.trades)
        assert result.equity.equals(explicit.equity)
        assert not result.trades.equals(permissive.trades)

    def test_margin_liquidation_threshold_widened(self, make_carry_data) -> None:
        # SC-CARRY-02: with the widened margin defaults a 6% adverse perp move
        # (previously a forced close at 4.76%) no longer liquidates, while a 14%
        # adverse move still does. The window ends on the liquidation bar so the
        # (still profitable) signal has no bar left on which to re-enter.
        n = 34
        grid = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        funding = {ts.strftime("%Y-%m-%d %H:%M"): 0.001 for ts in grid}
        perp_close = [100.0] * n
        for i in range(25, 33):
            perp_close[i] = 106.0
        perp_close[33] = 114.0
        data = make_carry_data(
            n_bars=n,
            funding=funding,
            perp_close=perp_close,
            borrow=[0.0] * n,
        )
        result = run_cash_carry_backtest(data, SPEC, ZERO_COSTS)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["reason"] == "margin_liquidation"
        assert trade["exit_bar"] == 33

    @pytest.mark.slow
    def test_btcusdt_hysteresis_yields_near_zero_trades_2023_2025(self) -> None:
        # Regression lock for the documented near-zero-edge finding (§2.3):
        # the cost-derived band must not silently regress into the 426-trade
        # churn artifact, and no trade may be a pure round-trip cost-drag on a
        # near-zero funding capture.
        data = load_carry_market_data("BTCUSDT", "2023-01-01", "2025-12-31 20:00:00+00:00")
        result = run_cash_carry_backtest(data, SPEC, CarryCostModel())
        assert len(result.trades) < 30
        for _, trade in result.trades.iterrows():
            if abs(trade["funding_pnl"]) <= 1e-9:
                assert trade["pnl"] >= -0.004 * trade["equity_before_entry"]


def _two_year_carry_data() -> CarryMarketData:
    grid = pd.date_range("2024-01-01", "2025-12-31 23:59", freq="4h", tz="UTC")
    n = len(grid)
    ramp = 100.0 + 0.002 * np.arange(n, dtype=np.float64)
    frame = pd.DataFrame(
        {"open": ramp, "high": ramp + 1.0, "low": ramp - 1.0, "close": ramp, "volume": 100.0},
        index=grid,
    )
    rates = np.where((np.arange(n) // 150) % 2 == 0, 0.0005, -0.0005).astype(np.float64)
    return CarryMarketData(
        symbol="BTCUSDT",
        spot=frame,
        perp=frame.copy(),
        funding=pd.Series(rates, index=grid, dtype=np.float64),
        borrow=pd.Series(0.0, index=grid, dtype=np.float64),
    )


class TestCarryGateComposition:
    def test_cash_carry_uses_unchanged_canonical_gate_and_fold(self) -> None:
        # SC-CARRY-GATE-01: the carry ledger plugs verbatim into the canonical
        # equity gate, fold distribution, and promotion composition.
        data = _two_year_carry_data()
        result = _run(data)
        assert len(result.trades) > 0
        gate = compute_equity_reliability_gate(result.equity, len(result.trades))
        folds = compute_fold_distribution(result)
        stress = compute_equity_reliability_gate(
            result.equity,
            len(result.trades),
            dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
        )
        promotion = compose_promotion_verdict(gate, folds, stress, None)
        assert gate.verdict in {"PASS", "FAIL", "PENDING"}
        assert isinstance(folds.max_period_contribution, float)
        assert promotion.observation_verdict == gate.verdict
        assert promotion.fold_gate_pass == folds.gate_pass
        assert promotion.stress_verdict == stress.verdict
        assert promotion.status in {"REJECTED", "OBSERVATION_PASS"}
