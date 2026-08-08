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

    def test_zero_target_closes_existing_inventory(self) -> None:
        grid = pd.date_range("2021-01-01 12:01", periods=121, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(grid)}, index=grid)
        target = pd.DataFrame(
            {"A": [1.0, 0.0]},
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2021-01-01 11:00", tz="UTC"),
                    pd.Timestamp("2021-01-01 12:00", tz="UTC"),
                ]
            ),
        )
        signal_at = pd.DatetimeIndex(
            [
                pd.Timestamp("2021-01-01 12:00", tz="UTC"),
                pd.Timestamp("2021-01-01 13:00", tz="UTC"),
            ]
        )
        report = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=grid, columns=["A"]), 1.0,
            "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
        )
        fills = report.simulated_fills
        assert len(fills) == 2
        assert fills.iloc[0]["quantity_delta"] == pytest.approx(0.01)
        assert fills.iloc[1]["quantity_delta"] == pytest.approx(-0.01)
        assert report.simulated_units.iloc[-1]["A"] == pytest.approx(0.0)

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


class TestFlatMarkNanLedger:
    """MHS-27-FLAT-MARK-NAN-LEDGER: an unavailable mark is zero only when flat.

    Leading unavailable marks with zero units leave equity exactly at the
    initial equity; a held position at an unavailable mark remains
    primary-invalid instead of leaking ``0 * NaN`` into cash equity.
    """

    def test_leading_nan_marks_flat_stay_at_initial_equity(self) -> None:
        idx = pd.date_range("2021-01-01", periods=5, freq="1h", tz="UTC")
        marks = pd.DataFrame(
            {"A": [np.nan, np.nan, 100.0, 101.0, 102.0]}, index=idx,
        )
        result = simulated_inventory_ledger(
            pd.DataFrame(),
            marks,
            pd.DataFrame(0.0, index=idx, columns=["A"]),
            1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
        )
        assert np.isfinite(result.equity.to_numpy()).all()
        assert result.equity.iloc[0] == pytest.approx(1.0)
        assert result.equity.iloc[1] == pytest.approx(1.0)
        assert result.primary_valid is True
        assert result.net_returns.isna().sum() == 0

    def test_held_unavailable_mark_is_primary_invalid(self) -> None:
        idx = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
        marks = pd.DataFrame(
            {"A": [100.0, 100.0, np.nan, 100.0]}, index=idx,
        )
        fills = pd.DataFrame(
            [
                {"timestamp": idx[0], "symbol": "A", "quantity_delta": 0.01,
                 "fill_price": 100.0, "fee_bps": 2.0, "reason": "passive_fill"},
            ],
        )
        result = simulated_inventory_ledger(
            fills,
            marks,
            pd.DataFrame(0.0, index=idx, columns=["A"]),
            1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
        )
        assert result.primary_valid is False
        assert "MISSING_DATA" in result.invalid_reasons
        assert np.isfinite(result.equity.to_numpy()).all()
        assert (result.equity > 0).all()

    def test_flat_position_at_nan_never_raises_on_equity(self) -> None:
        idx = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
        marks = pd.DataFrame(
            {"A": [np.nan, np.nan, np.nan]}, index=idx,
        )
        result = simulated_inventory_ledger(
            pd.DataFrame(),
            marks,
            pd.DataFrame(0.0, index=idx, columns=["A"]),
            1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
        )
        assert (result.equity == 1.0).all()


class TestStreamedLedgerEquivalence:
    """MHS-30-STREAMED-LEDGER-EQUIVALENCE: the default streaming mode avoids the
    dense ledger unit matrix and matches the opt-in dense diagnostic mode."""

    def test_streaming_matches_dense_mode_within_1e12(self) -> None:
        idx = pd.date_range("2021-01-01", periods=40, freq="5min", tz="UTC")
        rng = np.random.default_rng(7)
        marks = pd.DataFrame(
            {
                sym: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, len(idx))))
                for sym in ("AAAUSDT", "BBBUSDT", "CCCUSDT")
            },
            index=idx,
        )
        marks.iloc[15:18, 1] = np.nan
        funding = pd.DataFrame(2.0e-5, index=idx, columns=list(marks.columns))
        fills = pd.DataFrame(
            [
                {"timestamp": idx[3], "symbol": "AAAUSDT", "quantity_delta": 0.02,
                 "fill_price": float(marks.loc[idx[3], "AAAUSDT"]), "fee_bps": 2.0,
                 "reason": "passive_fill"},
                {"timestamp": idx[3], "symbol": "BBBUSDT", "quantity_delta": -0.01,
                 "fill_price": float(marks.loc[idx[3], "BBBUSDT"]), "fee_bps": 2.0,
                 "reason": "passive_fill"},
                {"timestamp": idx[20], "symbol": "CCCUSDT", "quantity_delta": 0.03,
                 "fill_price": float(marks.loc[idx[20], "CCCUSDT"]), "fee_bps": 8.0,
                 "reason": "timeout_taker"},
            ],
        )
        streamed = simulated_inventory_ledger(
            fills, marks, funding, 1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
        )
        dense = simulated_inventory_ledger(
            fills, marks, funding, 1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
            retain_simulated_units=True,
        )
        assert streamed.simulated_units is None
        assert dense.simulated_units is not None
        assert list(dense.simulated_units.columns) == list(marks.columns)
        assert list(dense.simulated_units.index) == list(idx)
        for field in (
            "equity", "net_returns", "mark_to_market_pnl",
            "funding_charge", "fee_charge", "fill_turnover",
        ):
            np.testing.assert_allclose(
                getattr(streamed, field).to_numpy(),
                getattr(dense, field).to_numpy(),
                rtol=1e-12, atol=1e-12,
            )
        assert streamed.primary_valid is dense.primary_valid
        assert streamed.invalid_reasons == dense.invalid_reasons
        np.testing.assert_allclose(
            dense.simulated_units.loc[idx[3]].to_numpy(), [0.02, -0.01, 0.0], atol=1e-12,
        )
        np.testing.assert_allclose(
            dense.simulated_units.loc[idx[20]].to_numpy(), [0.02, -0.01, 0.03], atol=1e-12,
        )


class TestReplayEquivalencePerformance:
    """MHS-28-REPLAY-EQUIVALENCE-PERFORMANCE: optimized replay matches the
    frozen-fixture fills, equity, fees, funding, and turnover while reporting
    measured elapsed seconds on a small deterministic fixture."""

    def test_strict_timeout_matches_frozen_fixture(self) -> None:
        idx = pd.date_range("2021-01-01 12:01", periods=31, freq="1min", tz="UTC")
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        px = pd.DataFrame({"A": [100.0] * 31}, index=idx)
        report = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        assert report.elapsed_seconds >= 0.0
        fills = report.simulated_fills
        assert len(fills) == 1
        assert fills.iloc[0]["timestamp"] == pd.Timestamp("2021-01-01 12:31", tz="UTC")
        assert fills.iloc[0]["symbol"] == "A"
        assert fills.iloc[0]["quantity_delta"] == pytest.approx(0.01)
        assert fills.iloc[0]["fill_price"] == pytest.approx(100.0)
        assert fills.iloc[0]["fee_bps"] == pytest.approx(8.0)
        assert report.submit_times.iloc[0] == pd.Timestamp("2021-01-01 12:01", tz="UTC")
        equity = report.ledger.equity
        assert np.allclose(equity.iloc[:30].to_numpy(), 1.0)
        assert equity.iloc[30] == pytest.approx(0.9992)
        assert report.ledger.fee_charge.iloc[30] == pytest.approx(0.0008)
        assert (report.ledger.funding_charge == 0.0).all()
        assert report.ledger.fill_turnover.iloc[30] == pytest.approx(1.0)
        assert report.unfilled_count == 1
        assert report.fallback_count == 1

    def test_passive_fill_matches_frozen_fixture(self) -> None:
        idx = pd.date_range("2021-01-01 12:01", periods=31, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * 31}, index=idx)
        px.loc["2021-01-01 12:10", "A"] = 99.0
        lows = px.copy()
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        report = strategy_aware_execution_replay(
            target, signal_at, px, lows, px, px,
            pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        fills = report.simulated_fills
        assert len(fills) == 1
        assert fills.iloc[0]["timestamp"] == pd.Timestamp("2021-01-01 12:10", tz="UTC")
        assert fills.iloc[0]["fill_price"] == pytest.approx(100.0)
        assert fills.iloc[0]["fee_bps"] == pytest.approx(2.0)
        assert fills.iloc[0]["reason"] == "passive_fill"
        assert report.fill_count == 1
        assert report.unfilled_count == 0
