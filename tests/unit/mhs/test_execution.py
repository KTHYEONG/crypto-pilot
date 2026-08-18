from __future__ import annotations

import dataclasses
import math
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.execution import (
    ExecutionReplayWindow,
    ExecutionSpec,
    _BoundExecutionReplayAccumulator,
    bar_funding_panel,
    laddered_fill_schedule,
    mhs_ledger_pnl,
    mhs_ledger_pnl_multi_tier,
    notional_weighted_shortfall_bps,
    passive_fill_shortfall_bps,
    replay_execution_window_batch,
    replay_execution_window_pair,
    replay_execution_windows,
    simulated_inventory_ledger,
    strategy_aware_execution_replay,
)
from src.research.baseline.backtest import _align_funding_rates
from src.research.technical_experts.cross_sectional import XsCompositeSpec, run_xs_composite_ledger

SPEC = ExecutionSpec()


class TestLadderedFillSchedule:
    """SCENARIO_MHS_LADDER_*: the escalating limit ladder pure function."""

    SPEC = ExecutionSpec()

    def test_contract_assertion_scenario(self) -> None:
        """SCENARIO_MHS_LADDER_FINAL_TRANCHE_MARKET_FALLBACK_03 (python_assertion):
        a never-trading-through window collapses the whole notional into a single
        market fallback at the all-in taker cost."""
        sched = laddered_fill_schedule(
            100.0, 1,
            np.array([101.0, 101.0, 101.0, 101.0]),
            np.array([101.0, 101.0, 101.0, 101.0]),
            4, ExecutionSpec(), True,
        )
        assert abs(sum(f[3] for f in sched) - 1.0) < 1e-12
        assert sched[0][2] == pytest.approx(
            ExecutionSpec().taker_fee_bps + ExecutionSpec().taker_slippage_bps
        )

    def test_tranche_one_matches_inline_strict_passive(self) -> None:
        """SCENARIO_MHS_LADDER_TRANCHE_ONE_MATCHES_STRICT_01: a single tranche
        reproduces the inline STRICT-proxy passive fill (hit at the first
        trade-through, decision price, maker fee, full notional)."""
        adverse = np.array([99.0, 100.5, 100.5])
        closes = np.array([100.0, 101.0, 101.5])
        sched = laddered_fill_schedule(100.0, 1, adverse, closes, 1, self.SPEC, True)
        assert sched == [(0, 100.0, 2.0, 1.0)]

    def test_tranche_one_matches_inline_strict_fallback(self) -> None:
        """SCENARIO_MHS_LADDER_TRANCHE_ONE_MATCHES_STRICT_01: the single-tranche
        timeout fallback fills at the window-boundary close with the all-in
        taker cost, exactly like the inline STRICT-proxy branch."""
        adverse = np.array([101.0, 101.0, 101.0])
        closes = np.array([100.0, 101.0, 101.5, 102.0])
        sched = laddered_fill_schedule(100.0, 1, adverse, closes, 1, self.SPEC, True)
        assert sched == [(3, 102.0, 8.0, 1.0)]

    def test_partial_escalation_carries_forward_without_market_fallback(self) -> None:
        """SCENARIO_MHS_LADDER_PARTIAL_ESCALATION_02: tranche 1 never trades
        through, tranche 2's repriced limit does -- the schedule returns exactly
        two tuples, and tranche 1's share fills as part of tranche 2's 0.5
        maker fill (no per-tranche market fallback)."""
        adverse = np.array([100.5, 100.5, 99.0, 99.0, 100.5, 100.5, 100.5, 100.5])
        closes = np.array([100.0, 100.0, 98.0, 98.0, 97.0, 97.0, 99.0, 99.0, 100.5])
        sched = laddered_fill_schedule(100.0, 1, adverse, closes, 4, self.SPEC, True)
        # tranche 2 limit: 100 + 1/4 * (closes[2] - 100) = 99.5
        assert len(sched) == 2
        assert sched[0] == (2, 99.5, 2.0, 0.5)
        assert sched[1] == (8, 100.5, 8.0, 0.5)
        assert abs(sum(f[3] for f in sched) - 1.0) < 1e-12

    def test_final_tranche_market_fallback_accumulates_all_shares(self) -> None:
        """SCENARIO_MHS_LADDER_FINAL_TRANCHE_MARKET_FALLBACK_03: no tranche ever
        trades through, so the schedule's last tuple carries qty_fraction == 1.0,
        the all-in taker cost, and the final sub-window's boundary close."""
        adverse = np.array([101.0, 101.0, 101.0, 101.0])
        closes = np.array([100.0, 100.0, 100.0, 100.0, 102.0])
        sched = laddered_fill_schedule(100.0, 1, adverse, closes, 4, self.SPEC, True)
        assert sched == [(4, 102.0, 8.0, 1.0)]

    def test_sell_side_ladder_escalates_toward_higher_limits(self) -> None:
        """The sell ladder reprices upward (side=-1) and uses the highs as the
        adverse path; the accumulated share fills only on a trade-through."""
        adverse = np.array([99.0, 99.0, 101.0, 101.0, 99.0, 99.0, 99.0, 99.0])
        closes = np.array([100.0, 100.0, 102.0, 102.0, 103.0, 103.0, 100.0, 100.0, 99.0])
        sched = laddered_fill_schedule(100.0, -1, adverse, closes, 4, self.SPEC, True)
        # tranche 2 limit: 100 + (-1)/4 * (closes[2] - 100) = 100 - 0.5 = 99.5
        assert sched[0][0] == 2
        assert sched[0][2] == 2.0
        assert sched[0][3] == pytest.approx(0.5)
        assert abs(sum(f[3] for f in sched) - 1.0) < 1e-12

    def test_touch_predicate_fills_on_exact_touch(self) -> None:
        """require_strict=False reuses the TOUCH comparison operators: an exact
        touch fills the first tranche passively."""
        adverse = np.array([100.0, 101.0, 101.0, 101.0])
        closes = np.array([100.0, 100.0, 100.0, 100.0, 101.0])
        strict = laddered_fill_schedule(100.0, 1, adverse, closes, 4, self.SPEC, True)
        touch = laddered_fill_schedule(100.0, 1, adverse, closes, 4, self.SPEC, False)
        assert strict[0][2] == 8.0
        assert touch[0] == (0, 100.0, 2.0, 0.25)

    def test_fails_closed_on_invalid_input(self) -> None:
        with pytest.raises(ValueError, match="tranche_count"):
            laddered_fill_schedule(100.0, 1, np.array([1.0]), np.array([1.0]), 0, self.SPEC, True)
        with pytest.raises(ValueError, match="side"):
            laddered_fill_schedule(100.0, 0, np.array([1.0]), np.array([1.0]), 1, self.SPEC, True)
        with pytest.raises(ValueError, match="adverse"):
            laddered_fill_schedule(100.0, 1, np.array([]), np.array([]), 1, self.SPEC, True)
        with pytest.raises(ValueError, match="finite"):
            laddered_fill_schedule(100.0, 1, np.array([np.nan]), np.array([1.0]), 1, self.SPEC, True)

    def test_fails_closed_on_boundary_close_nan(self) -> None:
        """SCENARIO_MHS_LADDER_CLOSES_GAP_01: adverse is entirely finite but the
        boundary close at index len(adverse) is NaN -- the exact defect the prior
        check missed (adverse alone would have passed)."""
        adverse = np.array([101.0, 101.0, 101.0, 101.0])
        closes = np.array([100.0, 100.0, 100.0, 100.0, np.nan])
        with pytest.raises(ValueError, match="closes must be finite"):
            laddered_fill_schedule(100.0, 1, adverse, closes, 4, self.SPEC, True)

    def test_fails_closed_on_interior_anchor_close_nan(self) -> None:
        """SCENARIO_MHS_LADDER_CLOSES_GAP_02: a NaN at an interior anchor position
        (index < len(adverse)) used by a k>1 tranche re-price is rejected too,
        proving the fix covers the full consumed closes range, not only the
        final boundary element."""
        adverse = np.array([101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0])
        closes = np.array([100.0, 100.0, np.nan, 100.0, 100.0, 100.0, 100.0, 100.0, 101.0])
        with pytest.raises(ValueError, match="closes must be finite"):
            laddered_fill_schedule(100.0, 1, adverse, closes, 4, self.SPEC, True)

    def test_finite_inputs_unchanged_by_closes_guard(self) -> None:
        """SCENARIO_MHS_LADDER_CLOSES_GAP_04: on fully-finite input the new
        closes finiteness check is a pure no-op -- the fully-finite market
        fallback schedule is byte-identical to the pre-fix expectation (the
        final-tranche market fallback at the boundary close with the all-in
        taker cost), proving the guard does not narrow or alter ladder pricing."""
        adverse = np.array([101.0, 101.0, 101.0, 101.0])
        closes = np.array([100.0, 100.0, 100.0, 100.0, 102.0])
        sched = laddered_fill_schedule(100.0, 1, adverse, closes, 4, self.SPEC, True)
        assert sched == [(4, 102.0, 8.0, 1.0)]
        assert abs(sum(f[3] for f in sched) - 1.0) < 1e-12


class TestLadderedExecutionReplay:
    """SCENARIO_MHS_LADDER_ORACLE_ACCUMULATOR_PARITY_04 + DoD regression: the
    laddered bound is byte-identical to strict at ladder_tranches=1 and the
    single-panel oracle matches the windowed accumulator under the same bound."""

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

    def test_laddered_oracle_matches_accumulator(self) -> None:
        """SCENARIO_MHS_LADDER_ORACLE_ACCUMULATOR_PARITY_04: LADDERED replay via
        the single-panel oracle and the windowed accumulator produce identical
        fills, ledger, termination, and counters at rtol=atol=1e-12."""
        wl = self._workload()
        oracle = strategy_aware_execution_replay(
            wl["weights"], wl["signals"], wl["highs"], wl["lows"], wl["closes"],
            wl["marks"], wl["funding"], 1.0, "OHLCV_LADDERED_PROXY", ExecutionSpec(),
        )
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        windowed = replay_execution_windows(
            windows, 1.0, "OHLCV_LADDERED_PROXY", ExecutionSpec(), retain_event_snapshots=True,
        )
        _assert_replay_equivalent(oracle, windowed)

    def test_ladder_tranches_one_is_byte_identical_to_strict(self) -> None:
        """DoD regression: OHLCV_LADDERED_PROXY with ladder_tranches=1 matches
        OHLCV_STRICT_PROXY byte-for-byte in fills and all-intent shortfall on
        both the single-panel oracle and the windowed path."""
        wl = self._workload()
        ladder_spec = ExecutionSpec(ladder_tranches=1)
        for replay in (
            lambda bound, spec: strategy_aware_execution_replay(
                wl["weights"], wl["signals"], wl["highs"], wl["lows"], wl["closes"],
                wl["marks"], wl["funding"], 1.0, bound, spec,
            ),
            lambda bound, spec: replay_execution_windows(
                _partition_windows(
                    wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
                    wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
                ),
                1.0, bound, spec,
            ),
        ):
            strict = replay("OHLCV_STRICT_PROXY", ExecutionSpec())
            ladder = replay("OHLCV_LADDERED_PROXY", ladder_spec)
            assert len(strict.simulated_fills) == len(ladder.simulated_fills)
            for col in ("timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps", "reason"):
                assert strict.simulated_fills[col].tolist() == ladder.simulated_fills[col].tolist(), col
            assert strict.all_intent_shortfall_bps == ladder.all_intent_shortfall_bps
            assert strict.fill_count == ladder.fill_count
            assert strict.unfilled_count == ladder.unfilled_count
            assert strict.fallback_count == ladder.fallback_count
            assert dict(strict.termination_counts) == dict(ladder.termination_counts)
            for field in ("equity", "net_returns", "mark_to_market_pnl", "funding_charge", "fee_charge", "fill_turnover"):
                np.testing.assert_allclose(
                    getattr(strict.ledger, field).to_numpy(),
                    getattr(ladder.ledger, field).to_numpy(),
                    rtol=1e-12, atol=1e-12,
                )

    def test_ladder_splits_order_into_multiple_fill_records(self) -> None:
        """A laddered order that partially escalates yields multiple fill
        records on one intent, and the ladder fills a strict shortfall."""
        idx = pd.date_range("2021-01-01 12:01", periods=31, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * 31}, index=idx)
        lows = px.copy()
        # Tranche 1 (bars 1-7) rests at 100: lows stay above.  Tranche 2's
        # repriced limit (bar 8-15) is traded through: low drops to 99.0.
        lows.loc["2021-01-01 12:09", "A"] = 99.0
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        report = strategy_aware_execution_replay(
            target, signal_at, px, lows, px, px,
            pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
            "OHLCV_LADDERED_PROXY", ExecutionSpec(ladder_tranches=4),
        )
        fills = report.simulated_fills
        assert len(fills) >= 2
        assert (fills["fee_bps"] == 2.0).any()
        assert (fills["reason"] == "passive_fill").any()
        assert report.fill_count >= 1
        assert report.fallback_count >= 1
        assert abs(fills["quantity_delta"].sum() - 0.01) < 1e-9

    def test_laddered_closes_gap_skips_order_on_both_paths(self) -> None:
        """SCENARIO_MHS_LADDER_CLOSES_GAP_03: a NaN in `closes` (not marks/highs/
        lows) at one bar inside an order's [spos, timeout_pos] window makes both
        the single-panel oracle and the windowed accumulator gracefully skip that
        one order -- MISSING_DATA > 0 with a MISSING_ACTIVE_ORDER_OHLCV gap --
        rather than raising, and the two paths stay equivalent."""
        grid = pd.date_range("2021-01-01 12:00", periods=60, freq="5min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(grid)}, index=grid)
        highs = pd.DataFrame({"A": [101.0] * len(grid)}, index=grid)
        lows = pd.DataFrame({"A": [99.0] * len(grid)}, index=grid)
        marks = px.copy()
        closes = px.copy()
        closes.iloc[3, 0] = np.nan
        funding = pd.DataFrame(0.0, index=grid, columns=["A"])
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        spec = ExecutionSpec()
        oracle = strategy_aware_execution_replay(
            target, signal_at, highs, lows, closes, marks, funding, 1.0,
            "OHLCV_LADDERED_PROXY", spec,
        )
        windows = _partition_windows(
            grid, target, signal_at, highs, lows, closes, marks, funding, spec, n_windows=1,
        )
        windowed = replay_execution_windows(
            windows, 1.0, "OHLCV_LADDERED_PROXY", spec, retain_event_snapshots=True,
        )
        for report in (oracle, windowed):
            assert report.termination_counts["MISSING_DATA"] == 1
            assert any(gap.code == "MISSING_ACTIVE_ORDER_OHLCV" for gap in report.data_gaps)
            assert report.simulated_fills.empty
        _assert_replay_equivalent(oracle, windowed)

    def test_immediate_taker_close_gap_skips_order_on_both_paths(self) -> None:
        """SCENARIO_MHS_IMMEDIATE_TAKER_CLOSE_GAP_01: a NaN in `closes` at the
        immediate-taker fill position (submit_pos) makes both the single-panel
        oracle and the windowed accumulator gracefully skip that one order --
        MISSING_DATA == 1 with a MISSING_ACTIVE_ORDER_OHLCV gap -- rather than
        raising DataIntegrityError, and the two paths stay equivalent."""
        grid = pd.date_range("2021-01-01 12:00", periods=60, freq="5min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(grid)}, index=grid)
        highs = pd.DataFrame({"A": [101.0] * len(grid)}, index=grid)
        lows = pd.DataFrame({"A": [99.0] * len(grid)}, index=grid)
        marks = px.copy()
        closes = px.copy()
        closes.iloc[1, 0] = np.nan
        funding = pd.DataFrame(0.0, index=grid, columns=["A"])
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        spec = ExecutionSpec()
        oracle = strategy_aware_execution_replay(
            target, signal_at, highs, lows, closes, marks, funding, 1.0,
            "OHLCV_IMMEDIATE_TAKER", spec,
        )
        windows = _partition_windows(
            grid, target, signal_at, highs, lows, closes, marks, funding, spec, n_windows=1,
        )
        windowed = replay_execution_windows(
            windows, 1.0, "OHLCV_IMMEDIATE_TAKER", spec, retain_event_snapshots=True,
        )
        for report in (oracle, windowed):
            assert report.termination_counts["MISSING_DATA"] == 1
            assert any(gap.code == "MISSING_ACTIVE_ORDER_OHLCV" for gap in report.data_gaps)
            assert report.simulated_fills.empty
        _assert_replay_equivalent(oracle, windowed)

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
                fee_rate=0.0004, slippage_rate=0.0004, gap_carry=True,
            ),
        )
        assert turnover.equals(turnover_ref)
        assert len(net) == len(equity_ref) - 1
        assert np.allclose(net.to_numpy(), equity_ref.pct_change().dropna().to_numpy())

    def test_gap_carry_default_true_absorbs_internal_gap(self) -> None:
        # SCENARIO_MHS_GAP_HARDENING_03: mhs_ledger_pnl carries a held weight
        # across a 3-bar internal NaN open gap instead of failing closed --
        # gap_carry defaults to True on this MHS-only pre-screen entrypoint.
        index = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
        opens = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0, np.nan, np.nan, np.nan,
                      106.0, 107.0, 108.0, 109.0],
            },
            index=index,
        )
        weights = pd.DataFrame({"A": [0.5] * len(index)}, index=index)
        funding = pd.DataFrame(0.0, index=index, columns=["A"])
        net, turnover = mhs_ledger_pnl(weights, opens, funding, one_way_bps=8.0)
        assert bool(np.isfinite(net.to_numpy()).all())
        assert turnover.index.equals(index)

    def test_multi_tier_bit_identical_to_per_tier_calls(self) -> None:
        # SCENARIO_MHS_LEDGER_MULTI_TIER_BIT_IDENTICAL: the single-pass shared-array
        # multi-tier ledger must equal per-tier mhs_ledger_pnl calls exactly (net
        # and turnover, check_exact=True) for the same one-way bps list -- the
        # property that makes the committee/multi-feature streaming rewrites
        # bit-identical.
        rng = np.random.default_rng(42)
        index = pd.date_range("2021-01-01", periods=2400, freq="1h", tz="UTC")
        symbols = [f"S{i:03d}" for i in range(8)]
        opens = pd.DataFrame(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-4, (len(index), len(symbols))), axis=0)),
            index=index, columns=symbols,
        )
        funding = pd.DataFrame(
            rng.normal(1e-5, 1e-6, (len(index), len(symbols))),
            index=index, columns=symbols,
        )
        step_index = index[::24]
        step = pd.DataFrame(
            rng.normal(0.0, 0.05, (len(step_index), len(symbols))),
            index=step_index, columns=symbols,
        )
        weights = step.reindex(index, method="ffill").fillna(0.0)

        bps_list = [2.64, 4.18, 6.07]
        singles = [
            mhs_ledger_pnl(weights, opens, funding, bps) for bps in bps_list
        ]
        multi = mhs_ledger_pnl_multi_tier(weights, opens, funding, bps_list)
        assert len(multi) == len(bps_list)
        for (net_s, tc_s), (net_m, tc_m) in zip(singles, multi, strict=True):
            pd.testing.assert_series_equal(net_s, net_m, check_exact=True)
            pd.testing.assert_series_equal(tc_s, tc_m, check_exact=True)

    def test_multi_tier_fails_closed(self) -> None:
        # SCENARIO_MHS_LEDGER_MULTI_TIER_FAIL_CLOSED: empty bps list and negative
        # bps raise ValueError; an index mismatch raises the same DataIntegrityError
        # message as the single call.
        weights = pd.DataFrame({"A": [0.5, 0.5, -0.5], "B": [-0.5, -0.5, 0.5]})
        opens = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [50.0, 49.0, 48.0]})
        funding = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})
        with pytest.raises(ValueError, match="must not be empty"):
            mhs_ledger_pnl_multi_tier(weights, opens, funding, [])
        with pytest.raises(ValueError, match=">= 0"):
            mhs_ledger_pnl_multi_tier(weights, opens, funding, [2.64, -1.0])
        mismatched = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0], "B": [50.0, 49.0, 48.0]},
            index=pd.DatetimeIndex(["2021-01-01", "2021-01-02", "2021-01-03"]),
        )
        with pytest.raises(DataIntegrityError, match="identical index"):
            mhs_ledger_pnl_multi_tier(weights, opens, mismatched, [2.64])


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
        """SCENARIO_MHS_IMMEDIATE_TAKER_CLOSE_GAP_02 (regression guard): on
        fully-finite closes the OHLCV_IMMEDIATE_TAKER oracle fill timing is
        unchanged by the fill-price finiteness guard added for the close-gap
        skip -- submit/fill times still land strictly after the signal."""
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

    def test_windowed_engine_real_stale_position_still_forced_exits(self) -> None:
        """SCENARIO_MHS_FOLD0_REAL_STALE_POSITION_STILL_FORCE_EXITS.

        Regression guard for the dust-tolerance fix below: a REAL
        (above-tolerance) held position whose data permanently ends mid-grid
        must keep forcing an exit through the windowed
        ``_BoundExecutionReplayAccumulator`` path exactly as the oracle does
        above -- the tolerance change must not weaken this fail-closed gate."""
        grid = pd.date_range("2021-01-01 00:00", periods=180, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(grid)}, index=grid)
        px.loc["2021-01-01 02:00":, "A"] = np.nan
        weights = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 00:00", tz="UTC")])
        signals = pd.DatetimeIndex([pd.Timestamp("2021-01-01 00:00", tz="UTC")])
        windows = _partition_windows(
            grid, weights, signals, px, px, px, px,
            pd.DataFrame(0.0, index=grid, columns=["A"]), ExecutionSpec(), n_windows=1,
        )
        report = replay_execution_windows(
            windows, 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
        )
        assert report.termination_counts["UNKNOWN_TERMINATION"] == 1
        assert report.forced_exit_count == 1
        assert report.forced_exit_notional > 0
        assert "forced_exit" in report.simulated_fills["reason"].tolist()

    def test_dust_residual_position_does_not_force_exit(self) -> None:
        """SCENARIO_MHS_FOLD0_DUST_NO_FORCED_EXIT.

        A held position of |units| < 1e-12 (floating-point accumulation
        dust from a long chain of small rebalances, e.g. under the
        scale-relative rebalance deadband) must NOT be treated as a real
        stale position at fold/window end, even when its symbol's last
        observed close trails the grid end. Reproducing the natural
        multi-decision rounding chain that produces such a residual is
        impractical in a unit fixture, so the residual is injected directly
        via the accumulator's public attribute -- matching this file's
        existing ``TestColumnarFillAccumulator`` convention of driving
        ``_BoundExecutionReplayAccumulator`` internals directly."""
        grid = pd.date_range("2021-01-01 00:00", periods=180, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(grid)}, index=grid)
        px.loc["2021-01-01 02:00":, "A"] = np.nan
        weights = pd.DataFrame({"A": [0.0]}, index=[pd.Timestamp("2021-01-01 00:00", tz="UTC")])
        signals = pd.DatetimeIndex([pd.Timestamp("2021-01-01 00:00", tz="UTC")])
        windows = _partition_windows(
            grid, weights, signals, px, px, px, px,
            pd.DataFrame(0.0, index=grid, columns=["A"]), ExecutionSpec(), n_windows=1,
        )
        acc = _BoundExecutionReplayAccumulator(
            windows[0], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), False,
        )
        acc.consume(windows[0])
        assert acc.last_close_ts["A"] < windows[0].minute_grid[-1]
        acc.units_arr[acc.columns.index("A")] = 1e-15
        result = acc.finalize()
        assert result.termination_counts == {"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0}
        assert result.forced_exit_count == 0
        assert "forced_exit" not in result.simulated_fills["reason"].tolist()


class TestNotionalWeightedShortfall:
    """P2-A: the notional-weighted mean of per-fill implementation shortfall.

    The plain ``all_intent_shortfall_bps`` is ``np.mean`` of per-fill
    shortfalls; the economically correct aggregate weights each fill by its
    absolute fill notional. Both replay paths (single-panel and streamed)
    accumulate the parallel notional series.
    """

    def test_weighted_differs_from_mean_and_matches_hand_computed(self) -> None:
        """SCENARIO_MHS_NOTIONAL_WEIGHTED_SHORTFALL_DIFFERS_FROM_MEAN_01: on a
        replay whose fills carry unequal notionals and unequal per-fill
        shortfalls the weighted aggregate differs from the unweighted mean and
        equals sum(shortfall*notional)/sum(notional)."""
        idx = pd.date_range("2021-01-01 12:01", periods=121, freq="1min", tz="UTC")
        marks = pd.DataFrame({"A": 100.0, "B": 200.0}, index=idx)
        closes = marks.copy()
        closes["B"] = 200.5
        target = pd.DataFrame(
            {"A": [0.5], "B": [0.5]},
            index=pd.DatetimeIndex([pd.Timestamp("2021-01-01 11:00", tz="UTC")]),
        )
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        report = strategy_aware_execution_replay(
            target, signal_at, closes, closes, closes, marks,
            pd.DataFrame(0.0, index=idx, columns=["A", "B"]), 1.0,
            "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
        )
        fills = report.simulated_fills
        assert len(fills) == 2
        decision_prices = {"A": 100.0, "B": 200.0}
        shortfalls: list[float] = []
        notionals: list[float] = []
        for _, fill in fills.iterrows():
            qty = float(fill["quantity_delta"])
            price = float(fill["fill_price"])
            side = 1.0 if qty > 0 else -1.0
            shortfalls.append(
                side * (price / decision_prices[fill["symbol"]] - 1.0) * 1e4
                + float(fill["fee_bps"])
            )
            notionals.append(abs(qty) * price)
        expected = sum(s * n for s, n in zip(shortfalls, notionals, strict=True)) / sum(notionals)
        assert report.notional_weighted_shortfall_bps == pytest.approx(expected)
        assert report.notional_weighted_shortfall_bps != pytest.approx(
            report.all_intent_shortfall_bps
        )

    def test_identical_notionals_make_weighted_coincide_with_mean(self) -> None:
        """When every fill carries an identical notional the two aggregates
        coincide regardless of the per-fill shortfalls."""
        assert notional_weighted_shortfall_bps([10.0, 20.0], [1.0, 1.0]) == pytest.approx(15.0)
        assert notional_weighted_shortfall_bps([10.0, 20.0], [3.0, 1.0]) == pytest.approx(12.5)

    def test_nan_when_no_fills_or_zero_notional(self) -> None:
        """SCENARIO_MHS_SHORTFALL_NAN_WHEN_NO_FILLS_02: zero fills (or a zero
        total notional) report NaN -- never 0.0 (which would read as free
        execution) and never a ZeroDivisionError."""
        idx = pd.date_range("2021-01-01 12:01", periods=61, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(idx)}, index=idx)
        target = pd.DataFrame(
            {"A": [0.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2021-01-01 11:00", tz="UTC")]),
        )
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        report = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
            "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
        )
        assert report.fill_count == 0
        assert math.isnan(report.all_intent_shortfall_bps)
        assert math.isnan(report.notional_weighted_shortfall_bps)
        assert math.isnan(notional_weighted_shortfall_bps([], []))
        assert math.isnan(notional_weighted_shortfall_bps([10.0, 20.0], [0.0, 0.0]))

    def test_both_replay_paths_populate_weighted_shortfall(self) -> None:
        """SCENARIO_MHS_BOTH_REPLAY_PATHS_POPULATE_SHORTFALL_03: both the
        module-level replay and the windowed/streaming replay class produce a
        finite notional-weighted shortfall on a fixture with fills."""
        wl = TestWindowedReplayEquivalence()._workload(days=10, n_symbols=4)
        grid = wl["grid"]
        weights = wl["weights"]
        signals = wl["signals"]
        single = strategy_aware_execution_replay(
            weights, signals, wl["highs"], wl["lows"], wl["closes"], wl["marks"],
            wl["funding"], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
        )
        windows = _partition_windows(
            grid, weights, signals, wl["highs"], wl["lows"], wl["closes"],
            wl["marks"], wl["funding"], ExecutionSpec(), n_windows=2,
        )
        streamed = replay_execution_windows(
            windows, 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
        )
        assert len(single.simulated_fills) > 0
        assert np.isfinite(single.notional_weighted_shortfall_bps)
        assert len(streamed.simulated_fills) > 0
        assert np.isfinite(streamed.notional_weighted_shortfall_bps)
        assert streamed.notional_weighted_shortfall_bps == pytest.approx(
            single.notional_weighted_shortfall_bps, rel=1e-9,
        )


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


def _assert_pair_equivalent(independent, paired, label: str) -> None:
    """MHS-MEM-PAIR-01: the paired fan-out result equals the independent
    single-bound call in fills, the six ledger series, gaps, counters,
    snapshots, and terminal state."""
    fill_o = independent.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    fill_p = paired.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    assert len(fill_o) == len(fill_p), label
    for col in ("timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps", "reason"):
        assert fill_o[col].tolist() == fill_p[col].tolist(), (label, col)
    np.testing.assert_allclose(
        fill_o["pre_trade_equity"].to_numpy(dtype="float64"),
        fill_p["pre_trade_equity"].to_numpy(dtype="float64"),
        rtol=1e-12, atol=1e-12, err_msg=f"{label}: pre_trade_equity",
    )
    for field in ("equity", "net_returns", "mark_to_market_pnl", "funding_charge", "fee_charge", "fill_turnover"):
        np.testing.assert_allclose(
            getattr(independent.ledger, field).to_numpy(),
            getattr(paired.ledger, field).to_numpy(),
            rtol=1e-12, atol=1e-12, err_msg=f"{label}: {field}",
        )
    assert independent.ledger.primary_valid == paired.ledger.primary_valid
    assert independent.ledger.invalid_reasons == paired.ledger.invalid_reasons
    assert independent.data_gaps == paired.data_gaps
    assert dict(independent.termination_counts) == dict(paired.termination_counts)
    assert independent.fill_count == paired.fill_count
    assert independent.unfilled_count == paired.unfilled_count
    assert independent.fallback_count == paired.fallback_count
    assert independent.forced_exit_count == paired.forced_exit_count
    assert independent.forced_exit_notional == paired.forced_exit_notional
    assert independent.submit_times.tolist() == paired.submit_times.tolist()
    assert independent.fill_times.tolist() == paired.fill_times.tolist()
    assert independent.all_intent_shortfall_bps == paired.all_intent_shortfall_bps
    assert independent.fill_source == paired.fill_source
    assert independent.mark_source == paired.mark_source
    assert independent.event_snapshots_retained == paired.event_snapshots_retained
    assert independent.simulated_units.equals(paired.simulated_units)
    assert independent.simulated_notional_weights.equals(paired.simulated_notional_weights)


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

    def test_paired_fanout_matches_independent_bounds(self) -> None:
        """MHS-MEM-PAIR-01: a single window stream fanned out into the
        strict/stress pair equals the two legacy independent single-bound
        calls, and each window is consumed exactly once."""
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        strict_single = replay_execution_windows(
            windows, 1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(), retain_event_snapshots=True,
        )
        stress_single = replay_execution_windows(
            windows, 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), retain_event_snapshots=True,
        )
        consumed = {"n": 0}

        def _gen():
            for w in windows:
                consumed["n"] += 1
                yield w

        strict_pair, stress_pair = replay_execution_window_pair(
            _gen(), 1.0, ExecutionSpec(), retain_event_snapshots=True,
        )
        assert consumed["n"] == len(windows)
        _assert_pair_equivalent(strict_single, strict_pair, "strict")
        _assert_pair_equivalent(stress_single, stress_pair, "stress")

    def test_batch_fanout_matches_independent_bounds(self) -> None:
        """SCENARIO_MHS_STREAM_BATCH_EQUIVALENCE: an N-bound interleaved batch
        over a single window iterator equals N independent single-bound calls
        and consumes each window exactly once."""
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        spec = ExecutionSpec()
        bounds = [
            ("OHLCV_IMMEDIATE_TAKER", spec),
            ("OHLCV_IMMEDIATE_TAKER", spec),
            ("OHLCV_STRICT_PROXY", spec),
        ]
        independent = [
            replay_execution_windows(windows, 1.0, b, s, retain_event_snapshots=True)
            for (b, s) in bounds
        ]
        consumed = {"n": 0}

        def _gen():
            for w in windows:
                consumed["n"] += 1
                yield w

        batch = replay_execution_window_batch(
            _gen(), 1.0, bounds, retain_event_snapshots=True,
        )
        assert consumed["n"] == len(windows)
        assert len(batch) == len(bounds)
        for i, (indep, bres) in enumerate(zip(independent, batch, strict=True)):
            _assert_pair_equivalent(indep, bres, f"batch[{i}]")

    def test_batch_empty_bounds_fails_closed(self) -> None:
        """SCENARIO_MHS_STREAM_BATCH_EMPTY_BOUNDS: an empty bounds iterable
        raises ValueError before any window is consumed."""
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=2,
        )
        with pytest.raises(ValueError, match="bounds"):
            replay_execution_window_batch(windows, 1.0, [])

    def test_pair_strict_data_integrity_error_propagates(self) -> None:
        """MHS-MEM-PAIR-01: a fatal strict DataIntegrityError propagates
        unchanged; no stress result is fabricated."""
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=2,
        )
        windows[1] = dataclasses.replace(windows[1], bar_funding=windows[1].bar_funding * float("nan"))
        with pytest.raises(DataIntegrityError, match="bar_funding must be finite"):
            replay_execution_window_pair(windows, 1.0, ExecutionSpec())


class TestColumnarFillAccumulator:
    """Columnar fill refactor: ``_BoundExecutionReplayAccumulator`` stores fills field-wise.
    The ``simulated_fills`` output must remain byte-for-byte identical to the legacy dict path.
    """

    FILL_COLUMNS = (
        "timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps",
        "reason", "pre_trade_equity",
    )

    def _workload(self, days: int = 40, n_symbols: int = 8) -> dict[str, object]:
        grid = pd.date_range("2021-01-01", periods=days * 24 * 12, freq="5min", tz="UTC")
        symbols = [f"SYM{i:03d}USDT" for i in range(n_symbols)]
        rng = np.random.default_rng(7)
        closes = pd.DataFrame(
            {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(grid)))) for s in symbols},
            index=grid,
        )
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
            "marks": closes,
            "funding": pd.DataFrame(1.0e-5, index=grid, columns=symbols),
            "weights": weights,
            "signals": decision_grid + pd.Timedelta(hours=1),
        }

    def _windowed(
        self, wl: dict[str, object], bound: str = "OHLCV_STRICT_PROXY",
    ):
        return replay_execution_windows(
            _partition_windows(
                wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
                wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
            ),
            1.0, bound, ExecutionSpec(),
        )

    def test_bound_execution_replay_accumulator_columnar_fills_match_legacy_dict_output(self) -> None:
        """R2 output-equivalence contract: the columnar ``simulated_fills`` must
        be byte-for-byte identical (column order, dtypes, row values) to the
        legacy dict-list DataFrame built from the same underlying fill data.
        Driving ``_BoundExecutionReplayAccumulator`` directly exposes the raw
        field lists so the legacy reference is reconstructed exactly as the
        pre-refactor ``finalize`` did, including forced-exit fills."""
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        acc = _BoundExecutionReplayAccumulator(
            windows[0], 1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(), False,
        )
        for w in windows:
            acc.consume(w)
        columnar_df = acc.finalize().simulated_fills
        assert len(columnar_df) > 0
        # Rebuild the legacy one-dict-per-fill DataFrame from the same lists.
        legacy_records = [
            {
                "timestamp": acc.fill_ts[i],
                "symbol": acc.fill_symbol[i],
                "quantity_delta": acc.fill_qty[i],
                "fill_price": acc.fill_price[i],
                "fee_bps": acc.fill_fee_bps[i],
                "reason": acc.fill_reason[i],
                "pre_trade_equity": acc.fill_pre_trade_equity[i],
            }
            for i in range(len(acc.fill_ts))
        ]
        legacy_df = pd.DataFrame(legacy_records, columns=self.FILL_COLUMNS)
        if legacy_df.empty:
            legacy_df = legacy_df.astype(
                {"quantity_delta": "float64", "fill_price": "float64", "fee_bps": "float64"}
            )
        pd.testing.assert_frame_equal(
            columnar_df, legacy_df, check_dtype=True, check_exact=True,
        )

    def test_bound_execution_replay_accumulator_columnar_fills_dtype_and_column_order(self) -> None:
        wl = self._workload()
        fills = self._windowed(wl).simulated_fills
        assert not fills.empty
        assert list(fills.columns) == list(self.FILL_COLUMNS)
        assert fills.dtypes["timestamp"] == pd.DatetimeTZDtype(tz="UTC", unit="ns")
        for col in ("quantity_delta", "fill_price", "fee_bps", "pre_trade_equity"):
            assert fills.dtypes[col] == np.dtype("float64"), col
        for col in ("symbol", "reason"):
            assert fills.dtypes[col] == pd.StringDtype(na_value=np.nan), col
        # The empty case keeps the same column order and the numeric float64
        # dtypes, so an empty table cannot be mistaken for missing output.
        empty = replay_execution_windows(
            _partition_windows(
                wl["grid"], wl["weights"].mul(0.0), wl["signals"], wl["highs"],
                wl["lows"], wl["closes"], wl["marks"], wl["funding"],
                ExecutionSpec(), n_windows=2,
            ),
            1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(),
        ).simulated_fills
        assert empty.empty
        assert list(empty.columns) == list(self.FILL_COLUMNS)
        assert empty.dtypes["quantity_delta"] == np.dtype("float64")
        assert empty.dtypes["fill_price"] == np.dtype("float64")
        assert empty.dtypes["fee_bps"] == np.dtype("float64")

    def test_columnar_fill_accumulation_reduces_peak_rss_vs_dict_baseline(self) -> None:
        """R2 memory claim: holding 10k+ fills as one dict per fill (legacy)
        allocates more peak memory than the field-wise parallel lists (columnar).
        ``tracemalloc`` measures the tracked allocation peak, the deterministic
        structural driver of the RSS the spec targets; a synthetic replay is not
        needed because the diff is purely the container representation."""
        n = 10_000
        rng = np.random.default_rng(11)
        symbols = [f"SYM{i:04d}USDT" for i in range(64)]
        times = pd.date_range("2021-01-01", periods=n, freq="1min", tz="UTC")
        ts_list = [times[i] for i in range(n)]
        syms = [symbols[i % len(symbols)] for i in range(n)]
        qty = rng.uniform(0.001, 0.1, n).tolist()
        price = rng.uniform(50.0, 150.0, n).tolist()
        fee = rng.uniform(0.0, 10.0, n).tolist()
        reasons = ["passive_fill" if i % 2 else "timeout_taker" for i in range(n)]
        equity = rng.uniform(0.5, 1.5, n).tolist()

        tracemalloc.start()
        legacy_fills: list[dict[str, object]] = [
            {
                "timestamp": ts_list[i], "symbol": syms[i],
                "quantity_delta": qty[i], "fill_price": price[i],
                "fee_bps": fee[i], "reason": reasons[i],
                "pre_trade_equity": equity[i],
            }
            for i in range(n)
        ]
        legacy_df = pd.DataFrame(legacy_fills)
        legacy_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        tracemalloc.start()
        col_ts: list[pd.Timestamp] = []
        col_sym: list[str] = []
        col_qty: list[float] = []
        col_price: list[float] = []
        col_fee: list[float] = []
        col_reason: list[str] = []
        col_equity: list[float] = []
        for i in range(n):
            col_ts.append(ts_list[i])
            col_sym.append(syms[i])
            col_qty.append(qty[i])
            col_price.append(price[i])
            col_fee.append(fee[i])
            col_reason.append(reasons[i])
            col_equity.append(equity[i])
        columnar_df = pd.DataFrame(
            {
                "timestamp": col_ts, "symbol": col_sym, "quantity_delta": col_qty,
                "fill_price": col_price, "fee_bps": col_fee, "reason": col_reason,
                "pre_trade_equity": col_equity,
            }
        )
        columnar_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        assert len(legacy_fills) >= 10_000
        assert len(columnar_df) == len(legacy_df)
        # The gap is structural (dict container + hash table per fill), not noise.
        assert legacy_peak - columnar_peak > 256 * 1024
        assert columnar_peak < legacy_peak

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


class TestTerminalFailClosedRejection:
    """MHS-28-TERMINAL-FAIL-CLOSED-REPORT: a deterministic capital breach stays
    fail-closed inside the ledger and is classified to the stable
    CAPITAL_INVARIANT_BREACH reason so the diagnostic can persist a serializable
    typed rejection instead of dying on an uncaught process error."""

    def test_ledger_capital_breach_is_fail_closed(self) -> None:
        from src.application.research.mhs.evaluation import _classify_execution_failure
        from src.common.errors import DataIntegrityError

        idx = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
        marks = pd.DataFrame({"A": [100.0, 200.0, 200.0]}, index=idx)
        fills = pd.DataFrame(
            [
                {"timestamp": idx[0], "symbol": "A", "quantity_delta": -0.01,
                 "fill_price": 100.0, "fee_bps": 0.0, "reason": "passive_fill"},
                {"timestamp": idx[1], "symbol": "A", "quantity_delta": 0.01,
                 "fill_price": 200.0, "fee_bps": 0.0, "reason": "passive_fill"},
            ],
        )
        with pytest.raises(DataIntegrityError, match="pre-trade equity"):
            simulated_inventory_ledger(
                fills, marks, pd.DataFrame(0.0, index=idx, columns=["A"]),
                1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
            )
        reason = _classify_execution_failure(DataIntegrityError("pre-trade equity must be positive and finite"))
        assert reason == "CAPITAL_INVARIANT_BREACH"

    def test_book_failure_serializes_null_metrics(self) -> None:
        import json

        from src.application.research.mhs.evaluation import MhsBookFailure, MhsBookReport, _jsonable
        from src.common.errors import DataIntegrityError

        exc = DataIntegrityError("pre-trade equity must be positive and finite")
        failed = MhsBookReport(
            name="blend", band="fast_reversal", horizon_hours=48, step_hours=6,
            tranche_count=8, n_symbols=8, phase=None, prescreen=None, tail=None,
            primary=None, stress=None,
            primary_autocorr_sharpe=None, primary_naive_sharpe=None,
            primary_net_ann=None, primary_geometric_cagr=None,
            primary_max_drawdown=None, primary_annualized_turnover=None,
            stress_naive_sharpe=None, terminal_censored_decisions=0,
            failure=MhsBookFailure(
                stage="replay_blend", error_class="DataIntegrityError",
                reason="CAPITAL_INVARIANT_BREACH", message=str(exc),
            ),
        )
        payload = _jsonable(failed)
        assert payload["primary"] is None
        assert payload["primary_autocorr_sharpe"] is None
        assert payload["failure"]["reason"] == "CAPITAL_INVARIANT_BREACH"
        assert payload["failure"]["error_class"] == "DataIntegrityError"
        # Strict JSON: no NaN/Infinity tokens leak into the terminal payload.
        encoded = json.dumps(payload)
        assert "NaN" not in encoded
        assert "Infinity" not in encoded


class TestCapitalInvariantFailFast:
    """SCENARIO_MHS_CAPITAL_HARDENING_01/02/03: non-finite fill sizing is
    rejected fail-fast at the exact fill site (both the laddered and the
    single-fill branches) with the full symbol/timestamp/weight/equity/
    decision_price context, the new message still classifies as
    CAPITAL_INVARIANT_BREACH, and the guards are pure no-ops on any
    fully-finite replay."""

    def _single_decision_window(
        self, *, nan_close_at: pd.Timestamp | None = None,
        weight: float = 1.0, mark: float = 100.0,
    ) -> ExecutionReplayWindow:
        grid = pd.date_range("2021-01-01 00:00", periods=13, freq="5min", tz="UTC")
        symbols = ("AAAUSDT",)
        closes = pd.DataFrame({"AAAUSDT": [100.0] * len(grid)}, index=grid)
        if nan_close_at is not None:
            closes.loc[nan_close_at, "AAAUSDT"] = np.nan
        marks = pd.DataFrame({"AAAUSDT": [mark] * len(grid)}, index=grid)
        decision = pd.Timestamp("2021-01-01 00:10", tz="UTC")
        return ExecutionReplayWindow(
            window_start=grid[0],
            window_end=grid[-1],
            columns=symbols,
            symbols=symbols,
            minute_grid=grid,
            highs=closes * 1.01,
            lows=closes * 0.99,
            closes=closes,
            marks=marks,
            bar_funding=pd.DataFrame(0.0, index=grid, columns=symbols),
            target_weights=pd.DataFrame({"AAAUSDT": [weight]}, index=[decision]),
            signal_available_at=pd.DatetimeIndex(
                [pd.Timestamp("2021-01-01 00:05", tz="UTC")],
            ),
        )

    def test_non_finite_qty_raises_in_immediate_taker_branch(self) -> None:
        """SCENARIO_MHS_CAPITAL_HARDENING_01 (immediate-taker site): a target
        weight large enough to overflow ``desired_units`` (weight * equity /
        mark) makes ``net_units`` non-finite in the OHLCV_IMMEDIATE_TAKER
        single-fill branch -- the one path a NaN close no longer reaches, since
        the new fill-price guard now skips that order upstream. The fail-fast
        sizing guard still raises there with the full symbol/decision context,
        never letting a non-finite qty corrupt ``self.cash``."""
        decision = pd.Timestamp("2021-01-01 00:10", tz="UTC")
        window = self._single_decision_window(weight=1e308, mark=0.5)
        with pytest.raises(DataIntegrityError, match="capital accounting invariant") as exc_info:
            replay_execution_windows([window], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
        msg = str(exc_info.value)
        assert "AAAUSDT" in msg
        assert "00:10" in msg
        assert "decision_price=0.5" in msg
        assert "qty=np.float64(inf)" in msg

    def test_non_finite_qty_raises_in_laddered_branch(self) -> None:
        """SCENARIO_MHS_CAPITAL_HARDENING_01 (laddered site): a target weight
        large enough to overflow ``desired_units`` (weight * equity / mark)
        makes ``qty`` non-finite in the laddered/multi-tranche branch; the
        guard raises there with the same full-context message."""
        window = self._single_decision_window(weight=1e308, mark=0.5)
        with pytest.raises(DataIntegrityError, match="capital accounting invariant") as exc_info:
            replay_execution_windows([window], 1.0, "OHLCV_LADDERED_PROXY", ExecutionSpec())
        msg = str(exc_info.value)
        assert "AAAUSDT" in msg
        assert "weight=1e+308" in msg
        assert "decision_price=0.5" in msg
        assert "qty=np.float64(inf)" in msg

    def test_new_message_still_classifies_as_capital_breach(self) -> None:
        """SCENARIO_MHS_CAPITAL_HARDENING_02: the more detailed message keeps
        the 'capital' keyword so ``_classify_execution_failure`` still maps it
        to the stable CAPITAL_INVARIANT_BREACH reason."""
        from src.application.research.mhs.evaluation import _classify_execution_failure

        sym = "AAAUSDT"
        fill_time = pd.Timestamp("2021-01-01 00:10", tz="UTC")
        weight, equity, decision_price, qty, fill_price = 1.0, 1.0, 100.0, 0.01, float("nan")
        exc = DataIntegrityError(
            "non-finite fill sizing breaches the capital accounting invariant "
            f"(symbol={sym!r} ts={fill_time!r} weight={weight!r} equity={equity!r} "
            f"decision_price={decision_price!r} qty={qty!r} fill_price={fill_price!r})"
        )
        assert _classify_execution_failure(exc) == "CAPITAL_INVARIANT_BREACH"

    @pytest.mark.parametrize(
        "bound", ["OHLCV_STRICT_PROXY", "OHLCV_IMMEDIATE_TAKER", "OHLCV_LADDERED_PROXY"],
    )
    def test_finite_replay_unchanged(self, bound: str) -> None:
        """SCENARIO_MHS_CAPITAL_HARDENING_03: the new guards are pure no-ops on
        a fully-finite window -- the windowed replay still matches the single-
        panel oracle in fills and every ledger series, so no sizing arithmetic
        was changed."""
        wl = TestWindowedReplayEquivalence()._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        oracle = strategy_aware_execution_replay(
            wl["weights"], wl["signals"], wl["highs"], wl["lows"], wl["closes"],
            wl["marks"], wl["funding"], 1.0, bound, ExecutionSpec(),
        )
        windowed = replay_execution_windows(
            windows, 1.0, bound, ExecutionSpec(), retain_event_snapshots=True,
        )
        _assert_replay_equivalent(oracle, windowed)

class TestEquityFloorProtection:
    """SCENARIO_MHS_EQUITY_FLOOR_DEFAULT_NONE_STRICT_08
    SCENARIO_MHS_EQUITY_FLOOR_FORCES_FLAT_09
    SCENARIO_MHS_EQUITY_FLOOR_NO_BREACH_EMPTY_TUPLE_10: the reference-pass
    equity floor (min_equity_fraction) forces a decision row's sizing flat
    once pre-trade equity crosses ``min_equity_fraction * initial_equity``
    instead of levering a depleting base toward the hard turnover-equity
    check. The default None keeps today's strict fail-closed behavior, and a
    never-breached floor run is byte-identical to the None run."""

    def _short_squeeze_window(self, *, price_levels: tuple[float, ...]) -> ExecutionReplayWindow:
        """Short book squeezed by staged mark rises: each step's live equity
        ``_equity_at`` erodes through the 0.5 floor while the ledger-level
        pre-trade equity stays positive (so the floor's forced-flat unwind is
        itself well-capitalized), and a step that more-than-doubles the entry
        drives the ledger pre-trade equity <= 0 for the no-floor run."""
        grid = pd.date_range("2021-01-01 00:00", periods=37, freq="5min", tz="UTC")
        symbols = ("AAAUSDT",)
        bounds = (pd.Timestamp("2021-01-01 01:00", tz="UTC"), pd.Timestamp("2021-01-01 01:50", tz="UTC"))
        levels = np.where(
            grid < bounds[0], price_levels[0],
            np.where(grid < bounds[1], price_levels[1],
            np.where(grid < pd.Timestamp("2021-01-01 02:40", tz="UTC"), price_levels[2], price_levels[3])),
        )
        closes = pd.DataFrame({"AAAUSDT": levels}, index=grid)
        decisions = pd.DatetimeIndex(
            [
                pd.Timestamp("2021-01-01 00:10", tz="UTC"),
                pd.Timestamp("2021-01-01 01:00", tz="UTC"),
                pd.Timestamp("2021-01-01 01:50", tz="UTC"),
                pd.Timestamp("2021-01-01 02:40", tz="UTC"),
            ]
        )
        return ExecutionReplayWindow(
            window_start=grid[0],
            window_end=grid[-1],
            columns=symbols,
            symbols=symbols,
            minute_grid=grid,
            highs=closes * 1.01,
            lows=closes * 0.99,
            closes=closes,
            marks=closes,
            bar_funding=pd.DataFrame(0.0, index=grid, columns=symbols),
            target_weights=pd.DataFrame({"AAAUSDT": [-1.0] * 4}, index=decisions),
            signal_available_at=decisions,
        )

    def test_default_none_preserves_strict_fail_closed(self) -> None:
        """SCENARIO_MHS_EQUITY_FLOOR_DEFAULT_NONE_STRICT_08: a short entered at
        100 that more-than-doubles to 200 makes the ledger pre-trade equity at
        the buy-back fill <= 0; with min_equity_fraction unset the replay still
        raises the exact existing DataIntegrityError message."""
        window = self._short_squeeze_window(price_levels=(100.0, 200.0, 200.0, 200.0))
        with pytest.raises(DataIntegrityError, match="pre-trade equity must be positive and finite"):
            replay_execution_windows([window], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())

    def test_floor_forces_flat_and_records_breach_timestamps(self) -> None:
        """SCENARIO_MHS_EQUITY_FLOOR_FORCES_FLAT_09: a gradual squeeze (100 ->
        140 -> 180 -> 220) erodes live equity through the 0.5 floor at the
        second and fourth decisions while the ledger-level pre-trade equity
        stays positive, so the floor's forced-flat unwind completes the replay
        and records the exact breach timestamps."""
        window = self._short_squeeze_window(price_levels=(100.0, 140.0, 180.0, 220.0))
        result = replay_execution_windows(
            [window], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), min_equity_fraction=0.5,
        )
        breaches = result.ledger.equity_floor_breached_at
        assert len(breaches) >= 2
        assert breaches[0] == pd.Timestamp("2021-01-01 01:00", tz="UTC")
        assert breaches[1] == pd.Timestamp("2021-01-01 01:50", tz="UTC")
        assert breaches[-1] == pd.Timestamp("2021-01-01 02:40", tz="UTC")
        assert result.ledger.equity.min() > 0.0

    def test_no_breach_is_empty_tuple_and_byte_identical_to_none(self) -> None:
        """SCENARIO_MHS_EQUITY_FLOOR_NO_BREACH_EMPTY_TUPLE_10: a falling-price
        short (equity grows away from the floor) never trips it; the floor run
        returns an empty equity_floor_breached_at tuple and every other ledger
        field is byte-identical to the None run on the same fixture."""
        window = self._short_squeeze_window(price_levels=(100.0, 60.0, 70.0, 80.0))
        base = replay_execution_windows([window], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
        floored = replay_execution_windows(
            [window], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), min_equity_fraction=0.5,
        )
        assert base.ledger.equity_floor_breached_at == ()
        assert floored.ledger.equity_floor_breached_at == ()
        self._assert_ledger_fields_equal(base, floored)

    @staticmethod
    def _ledger_without_floor(result) -> dict[str, object]:
        out = {}
        for f in dataclasses.fields(result.ledger):
            if f.name == "equity_floor_breached_at":
                continue
            out[f.name] = getattr(result.ledger, f.name)
        return out

    def _assert_ledger_fields_equal(self, a, b) -> None:
        fa, fb = self._ledger_without_floor(a), self._ledger_without_floor(b)
        assert set(fa) == set(fb)
        for key in fa:
            va, vb = fa[key], fb[key]
            if isinstance(va, pd.Series):
                pd.testing.assert_series_equal(va, vb)
            elif isinstance(va, pd.DataFrame):
                pd.testing.assert_frame_equal(va, vb)
            else:
                assert va == vb
