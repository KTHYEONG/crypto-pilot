from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.core.types import CashCarrySpec, CarryCostModel
from src.data.carry_data import CarryMarketData
from src.engine.cash_carry_backtest import run_cash_carry_backtest
from src.validation.candidate_promotion import compose_promotion_verdict
from src.validation.reliability_gate import (
    ReliabilityGateConfig,
    compute_equity_reliability_gate,
    compute_fold_distribution,
)

SPEC = CashCarrySpec(symbol="BTCUSDT")
ZERO_COSTS = CarryCostModel(spot_fee_rate=0.0, perp_fee_rate=0.0, slippage_rate=0.0)


def _run(data, *, costs: CarryCostModel = ZERO_COSTS, delay: int = 0):
    return run_cash_carry_backtest(data, SPEC, costs, signal_delay_bars=delay)


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
        expected_funding = (10_000.0 / 114.4) * 106.0 * 0.001
        assert np.isclose(trade["funding_pnl"], expected_funding)
        assert np.isclose(trade["pnl"], trade["funding_pnl"])

    def test_cash_carry_ledger_charges_both_legs_and_credits_short(
        self,
        make_carry_data,
    ) -> None:
        # SC-CARRY-LEDGER-02: positive funding credits the short leg and both
        # legs incur fees and slippage, so the costed ledger trails the
        # zero-cost ledger by the full round-trip cost.
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
        # margin_liquidation while equity stays finite.
        data = make_carry_data(
            n_bars=6,
            spot_open=[100.0] * 6, spot_close=[100.0] * 6,
            perp_open=[100.0] * 6,
            perp_close=[100.0, 100.0, 100.0, 115.0, 115.0, 115.0],
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 04:00": 0.001,
                "2024-01-01 08:00": 0.001,
                "2024-01-01 12:00": 0.0,
                "2024-01-01 16:00": 0.0,
                "2024-01-01 20:00": 0.0,
            },
            borrow=[0.0] * 6,
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
