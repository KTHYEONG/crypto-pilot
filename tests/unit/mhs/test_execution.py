from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mhs.execution import (
    ExecutionSpec,
    bar_funding_panel,
    mhs_ledger_pnl,
    passive_fill_shortfall_bps,
    simulated_inventory_ledger,
    strategy_aware_execution_replay,
)
from src.research.baseline.backtest import _align_funding_rates
from src.research.technical_experts.cross_sectional import XsCompositeSpec, run_xs_composite_ledger

SPEC = ExecutionSpec()


class TestPassiveFillShortfall:
    """MHS-07-PASSIVE-FILL-TRADE-THROUGH: touch fills only under the relaxed rule."""

    def test_buy_fills_when_low_trades_through(self) -> None:
        assert passive_fill_shortfall_bps(100.0, np.array([99.5, 100.2]), 101.0, 1, SPEC) == 2.0

    def test_buy_times_out_when_limit_not_traded_through(self) -> None:
        assert abs(
            passive_fill_shortfall_bps(100.0, np.array([100.5, 100.2]), 101.0, 1, SPEC) - 108.0
        ) < 1e-9

    def test_sell_fills_when_high_trades_through(self) -> None:
        assert passive_fill_shortfall_bps(100.0, np.array([100.5]), 99.0, -1, SPEC) == 2.0

    def test_sell_times_out_when_high_not_traded_through(self) -> None:
        assert abs(
            passive_fill_shortfall_bps(100.0, np.array([99.5]), 99.0, -1, SPEC) - 108.0
        ) < 1e-9

    def test_exact_touch_requires_relaxed_rule(self) -> None:
        touch = np.array([100.0])
        assert abs(passive_fill_shortfall_bps(100.0, touch, 101.0, 1, SPEC) - 108.0) < 1e-9
        assert passive_fill_shortfall_bps(
            100.0, touch, 101.0, 1, ExecutionSpec(require_trade_through=False),
        ) == 2.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"decision_price": 0.0, "adverse_path": np.array([1.0]), "timeout_price": 1.0, "side": 1},
            {"decision_price": 100.0, "adverse_path": np.array([]), "timeout_price": 101.0, "side": 1},
            {"decision_price": 100.0, "adverse_path": np.array([99.0]), "timeout_price": 101.0, "side": 0},
        ],
    )
    def test_fails_closed(self, kwargs: dict) -> None:
        with pytest.raises(ValueError, match=r"must|empty|side"):
            passive_fill_shortfall_bps(spec=SPEC, **kwargs)


class TestBarFundingPanel:
    """MHS-12-LEDGER-REUSES-PRODUCTION-FORMULA: funding aligns via _align_funding_rates."""

    def test_matches_align_funding_rates(self) -> None:
        grid = pd.date_range("2021-01-01", periods=8, freq="1h", tz="UTC")
        rates = pd.Series(
            [0.0001],
            index=[pd.Timestamp("2021-01-01 01:30", tz="UTC")],
        )
        panel = bar_funding_panel({"AAAUSDT": rates}, grid)
        assert list(panel.columns) == ["AAAUSDT"]
        assert panel.index.equals(grid)
        reference = pd.Series(
            _align_funding_rates(rates, grid), index=grid,
        )
        assert (panel["AAAUSDT"] - reference).abs().max() < 1e-12

    def test_excludes_symbol_with_unalignable_funding(self) -> None:
        grid = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
        bad = pd.Series(
            [0.0001],
            index=[pd.Timestamp("1999-01-01", tz="UTC")],
        )
        panel = bar_funding_panel({"AAAUSDT": bad}, grid)
        assert panel.empty


class TestMhsLedgerPnl:
    """MHS-12-LEDGER-REUSES-PRODUCTION-FORMULA: identical to run_xs_composite_ledger."""

    def test_reproduces_legacy_ledger_exactly(self) -> None:
        weights = pd.DataFrame({"A": [0.5, 0.5, -0.5], "B": [-0.5, -0.5, 0.5]})
        opens = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [50.0, 49.0, 48.0]})
        funding = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})
        net, turnover = mhs_ledger_pnl(weights, opens, funding, one_way_bps=8.0)
        equity_ref, turnover_ref = run_xs_composite_ledger(
            weights, opens, funding,
            XsCompositeSpec(
                halflife_bars=0, no_trade_band=0.0, execution_delay_bars=1,
                fee_rate=0.0004, slippage_rate=0.0004,
            ),
        )
        assert turnover.equals(turnover_ref)
        assert len(net) == len(equity_ref) - 1
        assert np.allclose(net.to_numpy(), equity_ref.pct_change().dropna().to_numpy())


class TestSimulatedInventoryLedger:
    """MHS-15-INVENTORY-DRIFT-NO-FREE-REBALANCE: fixed contracts drift; no free rebalance."""

    def test_fixed_contracts_return_10_then_909(self) -> None:
        marks = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 81.0]},
            index=pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC"),
        )
        fills = pd.DataFrame(
            [
                {"timestamp": marks.index[0], "symbol": "A", "quantity_delta": 0.005,
                 "fill_price": 100.0, "fee_bps": 0.0, "reason": "passive_fill"},
                {"timestamp": marks.index[0], "symbol": "B", "quantity_delta": -0.005,
                 "fill_price": 100.0, "fee_bps": 0.0, "reason": "passive_fill"},
            ],
        )
        result = simulated_inventory_ledger(
            fills, marks, pd.DataFrame(0.0, index=marks.index, columns=marks.columns),
            1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
        )
        assert abs(result.net_returns.iloc[0] - 0.10) < 1e-12
        assert abs(result.net_returns.iloc[1] - (0.10 / 1.10)) < 1e-12
        assert result.fill_turnover.iloc[0] == pytest.approx(1.0)
        assert (result.fill_source, result.mark_source) == ("OHLCV_STRICT_PROXY", "MARK_PRICE")

    def test_target_weight_ledger_is_a_separate_prescreen(self) -> None:
        weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5]})
        opens = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 81.0]})
        funding = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        net, _ = mhs_ledger_pnl(weights, opens, funding, 8.0)
        # Target-weight implicit rebalancing is a screening proxy, not inventory.
        assert len(net) == 2


class TestStrategyReplay:
    """MHS-14-CAUSAL-EXECUTION-AND-FORCED-EXIT: causal timing and forced exits."""

    def test_contract_assertion_scenario(self) -> None:
        idx = pd.date_range("2021-01-01 12:01", periods=31, freq="1min", tz="UTC")
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        px = pd.DataFrame({"A": [100.0] * 31}, index=idx)
        report = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        assert report.submit_times.iloc[0] > signal_at[0]
        assert report.fill_source == "OHLCV_STRICT_PROXY"
        assert report.unfilled_count == 1
        assert report.fallback_count == 1
        assert report.simulated_fills.iloc[0]["timestamp"] == report.fill_times.iloc[0]

    def test_no_order_at_or_before_signal_close(self) -> None:
        idx = pd.date_range("2021-01-01 12:01", periods=31, freq="1min", tz="UTC")
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        px = pd.DataFrame({"A": [100.0] * 31}, index=idx)
        report = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
            "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
        )
        assert report.submit_times.iloc[0] > signal_at[0]
        assert report.fill_times.iloc[0] > signal_at[0]

    def test_persistent_termination_creates_forced_exit_plus_stress_penalty(self) -> None:
        # A is held, then its minute data permanently ends mid-grid.
        grid = pd.date_range("2021-01-01 01:01", periods=120, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(grid)}, index=grid)
        px.loc["2021-01-01 02:00":, "A"] = np.nan
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 00:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 01:00", tz="UTC")])
        strict = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=grid, columns=["A"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        assert strict.termination_counts["UNKNOWN_TERMINATION"] == 1
        assert strict.forced_exit_count == 1
        assert strict.forced_exit_notional > 0
        assert "forced_exit" in strict.simulated_fills["reason"].tolist()

        stress = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=grid, columns=["A"]), 1.0,
            "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
        )
        strict_exit_fee = strict.simulated_fills[
            strict.simulated_fills["reason"] == "forced_exit"
        ]["fee_bps"].iloc[0]
        stress_exit_fee = stress.simulated_fills[
            stress.simulated_fills["reason"] == "forced_exit"
        ]["fee_bps"].iloc[0]
        assert stress_exit_fee > strict_exit_fee


class TestMissingDataTermination:
    """MHS-23-RELEVANT-MISSING-DATA-AND-TERMINATION: relevant gaps fail closed."""

    def test_flat_order_free_symbol_gap_does_not_count(self) -> None:
        grid = pd.date_range("2021-01-01 01:01", periods=60, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(grid), "B": [50.0] * len(grid)}, index=grid)
        px.loc["2021-01-01 01:20":, "B"] = np.nan
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 00:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 01:00", tz="UTC")])
        report = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=grid, columns=["A", "B"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        # B is flat and order-free: its gap must not be counted as relevant.
        assert report.termination_counts["MISSING_DATA"] == 0

    def test_relevant_gap_in_active_order_window_is_fail_closed(self) -> None:
        grid = pd.date_range("2021-01-01 01:01", periods=60, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(grid)}, index=grid)
        px.loc["2021-01-01 01:10":"2021-01-01 01:40", "A"] = np.nan
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 00:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 01:00", tz="UTC")])
        report = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=grid, columns=["A"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        assert report.termination_counts["MISSING_DATA"] == 1
        assert report.simulated_fills.empty

    def test_unverified_boundary_is_missing_data_not_delisting(self) -> None:
        grid = pd.date_range("2021-01-01 01:01", periods=60, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(grid)}, index=grid)
        px.loc["2021-01-01 01:30":, "A"] = np.nan
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 00:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 01:00", tz="UTC")])
        report = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=grid, columns=["A"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        assert set(report.termination_counts) == {"MISSING_DATA", "UNKNOWN_TERMINATION"}
        assert "DELIST" not in report.termination_counts
