from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mhs.execution import (
    ExecutionReplayWindow,
    ExecutionSpec,
    bar_funding_panel,
    mhs_ledger_pnl,
    passive_fill_shortfall_bps,
    replay_execution_windows,
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


def _partition_windows(
    grid: pd.DatetimeIndex,
    weights: pd.DataFrame,
    signals: pd.DatetimeIndex,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    closes: pd.DataFrame,
    marks: pd.DataFrame,
    funding: pd.DataFrame,
    spec: ExecutionSpec,
    n_windows: int = 2,
) -> list[ExecutionReplayWindow]:
    """Split a full fixture into contiguous execution windows exactly like the
    application planner: grid start at the previous window's last decision,
    grid end at the final order's strict timeout bar (last window covers the
    full grid)."""
    full_ns = np.asarray(grid, dtype="datetime64[ns]").astype("int64")
    timeout = spec.passive_timeout_minutes * 60_000_000_000
    sig_ns = np.asarray(signals, dtype="datetime64[ns]").astype("int64")
    spos = np.searchsorted(full_ns, sig_ns, side="right")
    resolve = [None] * len(weights)
    for i in range(len(weights)):
        if spos[i] >= len(full_ns):
            continue
        tns = full_ns[spos[i]] + timeout
        tpos = int(np.searchsorted(full_ns, tns, side="left"))
        if tpos < len(full_ns) and full_ns[tpos] == tns:
            resolve[i] = pd.Timestamp(tns, unit="ns", tz="UTC")
    bounds = np.array_split(np.arange(len(weights)), n_windows)
    out: list[ExecutionReplayWindow] = []
    prev_last: pd.Timestamp | None = None
    for bi, idxs in enumerate(bounds):
        is_last = bi == len(bounds) - 1
        ws = weights.iloc[idxs]
        sg = signals[idxs]
        grid_start = grid[0] if prev_last is None else prev_last
        if is_last:
            grid_end = grid[-1]
        else:
            grid_end = max((resolve[i] for i in idxs if resolve[i] is not None), default=ws.index[-1] + pd.Timedelta(hours=2))
        wgrid = pd.date_range(grid_start, grid_end, freq="5min", tz="UTC")
        out.append(
            ExecutionReplayWindow(
                window_start=grid_start,
                window_end=grid_end,
                columns=tuple(weights.columns),
                symbols=tuple(weights.columns),
                minute_grid=wgrid,
                highs=highs.loc[wgrid],
                lows=lows.loc[wgrid],
                closes=closes.loc[wgrid],
                marks=marks.loc[wgrid],
                bar_funding=funding.loc[wgrid],
                target_weights=ws,
                signal_available_at=sg,
            )
        )
        prev_last = ws.index[-1]
    return out


def _assert_replay_equivalent(oracle, windowed) -> None:
    fill_o = oracle.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    fill_w = windowed.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    assert len(fill_o) == len(fill_w)
    for col in ("timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps", "reason"):
        assert fill_o[col].tolist() == fill_w[col].tolist()
    for field in ("equity", "net_returns", "mark_to_market_pnl", "funding_charge", "fee_charge", "fill_turnover"):
        np.testing.assert_allclose(
            getattr(oracle.ledger, field).to_numpy(),
            getattr(windowed.ledger, field).to_numpy(),
            rtol=1e-12, atol=1e-12,
        )
    assert oracle.ledger.primary_valid == windowed.ledger.primary_valid
    assert oracle.ledger.invalid_reasons == windowed.ledger.invalid_reasons
    assert dict(oracle.termination_counts) == dict(windowed.termination_counts)
    assert oracle.fill_count == windowed.fill_count
    assert oracle.unfilled_count == windowed.unfilled_count
    assert oracle.fallback_count == windowed.fallback_count
    assert list(oracle.simulated_units.columns) == list(windowed.simulated_units.columns)
    assert len(oracle.simulated_units) == len(windowed.simulated_units)


class TestWindowedReplayEquivalence:
    """MHS-30-STREAMED-LEDGER-EQUIVALENCE: windowed strict and stress replays
    match the single-panel replay in fills, termination counts, ledger series,
    and validity at 1e-12 tolerance."""

    def _workload(self, days: int = 40, n_symbols: int = 8) -> dict[str, object]:
        grid = pd.date_range("2021-01-01", periods=days * 24 * 12, freq="5min", tz="UTC")
        symbols = [f"SYM{i:03d}USDT" for i in range(n_symbols)]
        rng = np.random.default_rng(7)
        closes = pd.DataFrame(
            {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(grid)))) for s in symbols},
            index=grid,
        )
        marks = closes.copy()
        marks.iloc[100:105, 3] = np.nan
        decision_grid = pd.date_range("2021-01-01", periods=days * 4, freq="6h", tz="UTC")
        weights = pd.DataFrame(0.0, index=decision_grid, columns=symbols)
        rng_w = np.random.default_rng(8)
        for ts in decision_grid:
            active = rng_w.choice(symbols, size=4, replace=False)
            weights.loc[ts, active] = rng_w.uniform(0.01, 0.06, 4)
        return {
            "grid": grid,
            "symbols": symbols,
            "highs": closes * 1.001,
            "lows": closes * 0.999,
            "closes": closes,
            "marks": marks,
            "funding": pd.DataFrame(1.0e-5, index=grid, columns=symbols),
            "weights": weights,
            "signals": decision_grid + pd.Timedelta(hours=1),
        }

    @pytest.mark.parametrize("bound", ["OHLCV_STRICT_PROXY", "OHLCV_IMMEDIATE_TAKER"])
    def test_windowed_matches_single_panel(self, bound: str) -> None:
        wl = self._workload()
        grid = wl["grid"]
        weights = wl["weights"]
        signals = wl["signals"]
        oracle = strategy_aware_execution_replay(
            weights, signals, wl["highs"], wl["lows"], wl["closes"], wl["marks"],
            wl["funding"], 1.0, bound, ExecutionSpec(),
        )
        windows = _partition_windows(
            grid, weights, signals, wl["highs"], wl["lows"], wl["closes"],
            wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        windowed = replay_execution_windows(
            windows, 1.0, bound, ExecutionSpec(), retain_event_snapshots=True,
        )
        _assert_replay_equivalent(oracle, windowed)

    def test_three_windows_follow_strict_timeout_overlap(self) -> None:
        wl = self._workload(days=45)
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        assert len(windows) == 3
        for w in windows[1:]:
            assert w.minute_grid[0] == w.window_start
            assert w.window_start < w.target_weights.index[0]
        assert windows[-1].minute_grid[-1] == wl["grid"][-1]

    def test_data_gap_provenance_codes(self) -> None:
        wl = self._workload()
        windowed = replay_execution_windows(
            _partition_windows(
                wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
                wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=2,
            ),
            1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        codes = {g.code for g in windowed.data_gaps}
        # The synthetic mark gap at bars 100-105 is held through by a fill,
        # so the real held-mark/held-funding provenance must be attributed.
        assert codes >= {"MISSING_HELD_MARK", "MISSING_HELD_FUNDING"}
        for gap in windowed.data_gaps:
            assert gap.symbol
            assert gap.timestamp is not None
            assert gap.execution_bound == "OHLCV_STRICT_PROXY"


def _assert_full_equivalence(enabled, disabled) -> None:
    """MHS-MEM-01: fills, six ledger series, validity, gaps, counters, and
    terminal state are identical between snapshot-disabled and snapshot-enabled
    replay at rtol=atol=1e-12."""
    fill_e = enabled.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    fill_d = disabled.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    assert len(fill_e) == len(fill_d)
    for col in ("timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps", "reason"):
        assert fill_e[col].tolist() == fill_d[col].tolist()
    for field in ("equity", "net_returns", "mark_to_market_pnl", "funding_charge", "fee_charge", "fill_turnover"):
        np.testing.assert_allclose(
            getattr(enabled.ledger, field).to_numpy(),
            getattr(disabled.ledger, field).to_numpy(),
            rtol=1e-12, atol=1e-12,
        )
    assert enabled.ledger.primary_valid == disabled.ledger.primary_valid
    assert enabled.ledger.invalid_reasons == disabled.ledger.invalid_reasons
    assert enabled.ledger.data_gaps == disabled.ledger.data_gaps
    assert enabled.data_gaps == disabled.data_gaps
    assert dict(enabled.termination_counts) == dict(disabled.termination_counts)
    assert enabled.fill_count == disabled.fill_count
    assert enabled.unfilled_count == disabled.unfilled_count
    assert enabled.fallback_count == disabled.fallback_count
    assert enabled.forced_exit_count == disabled.forced_exit_count
    assert enabled.forced_exit_notional == disabled.forced_exit_notional
    assert enabled.submit_times.tolist() == disabled.submit_times.tolist()
    assert enabled.fill_times.tolist() == disabled.fill_times.tolist()
    assert enabled.all_intent_shortfall_bps == disabled.all_intent_shortfall_bps
    assert list(enabled.simulated_units.columns) == list(disabled.simulated_units.columns)
    assert list(enabled.simulated_notional_weights.columns) == list(
        disabled.simulated_notional_weights.columns
    )


class TestEventSnapshotOptIn:
    """MHS-MEM-01/02: dense event snapshots are opt-in and bounded-memory by default."""

    def _workload(self, days: int = 40, n_symbols: int = 8) -> dict[str, object]:
        grid = pd.date_range("2021-01-01", periods=days * 24 * 12, freq="5min", tz="UTC")
        symbols = [f"SYM{i:03d}USDT" for i in range(n_symbols)]
        rng = np.random.default_rng(7)
        closes = pd.DataFrame(
            {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(grid)))) for s in symbols},
            index=grid,
        )
        marks = closes.copy()
        marks.iloc[100:105, 3] = np.nan
        decision_grid = pd.date_range("2021-01-01", periods=days * 4, freq="6h", tz="UTC")
        weights = pd.DataFrame(0.0, index=decision_grid, columns=symbols)
        rng_w = np.random.default_rng(8)
        for ts in decision_grid:
            active = rng_w.choice(symbols, size=4, replace=False)
            weights.loc[ts, active] = rng_w.uniform(0.01, 0.06, 4)
        return {
            "grid": grid,
            "symbols": symbols,
            "highs": closes * 1.001,
            "lows": closes * 0.999,
            "closes": closes,
            "marks": marks,
            "funding": pd.DataFrame(1.0e-5, index=grid, columns=symbols),
            "weights": weights,
            "signals": decision_grid + pd.Timedelta(hours=1),
        }

    @pytest.mark.parametrize("bound", ["OHLCV_STRICT_PROXY", "OHLCV_IMMEDIATE_TAKER"])
    def test_snapshot_disabled_equals_enabled(self, bound: str) -> None:
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        enabled = replay_execution_windows(
            windows, 1.0, bound, ExecutionSpec(), retain_event_snapshots=True,
        )
        disabled = replay_execution_windows(
            windows, 1.0, bound, ExecutionSpec(),
        )
        assert disabled.event_snapshots_retained is False
        assert enabled.event_snapshots_retained is True
        assert len(disabled.simulated_units) == 0
        assert len(disabled.simulated_notional_weights) == 0
        assert len(enabled.simulated_units) > 0
        _assert_full_equivalence(enabled, disabled)

    def test_wide_fill_disabled_retains_zero_dense_rows(self) -> None:
        wl = self._workload(days=60, n_symbols=32)
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=4,
        )
        disabled = replay_execution_windows(
            windows, 1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        assert disabled.fill_count > 0
        assert len(disabled.simulated_fills) > 0
        assert disabled.event_snapshots_retained is False
        assert disabled.simulated_units.empty
        assert disabled.simulated_notional_weights.empty
        assert list(disabled.simulated_units.columns) == list(wl["symbols"])


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
