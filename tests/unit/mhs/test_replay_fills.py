"""Execution replay engine contract tests (split by behavioral domain)."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from src.mhs.execution import (
    SPREAD_ESTIMATE_CEILING_BPS,
    ExecutionSpec,
    corwin_schultz_half_spread_bps,
    laddered_fill_schedule,
    notional_weighted_shortfall_bps,
    passive_fill_shortfall_bps,
    peg_chase_fill_schedule,
    replay_execution_windows,
    strategy_aware_execution_replay,
)

from tests.unit.mhs.test_replay_accumulator import (
    TestWindowedReplayEquivalence as _WindowedReplayWorkload,
)
from tests.unit.mhs.test_execution import (  # noqa: F401
    SPEC,
    _assert_replay_equivalent,
    _partition_windows,
)

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

class TestPegChaseFillSchedule:
    """SCENARIO_MHS_PEG_CHASE_*: the own-touch repricing peg-chase pure function."""

    def test_SCENARIO_MHS_PEG_CHASE_01_MAKER_FILL_AT_FIRST_TRADE_THROUGH(self) -> None:
        """SCENARIO_MHS_PEG_CHASE_01_MAKER_FILL_AT_FIRST_TRADE_THROUGH:
        Bar 0 does not fill (100.5 < 100.0 is False under the strict
        trade-through predicate); bar 1's re-pegged limit is traded through."""
        adverse = np.array([100.5, 99.5] + [100.5] * 8)
        closes = np.full(10, 100.0)
        sched = peg_chase_fill_schedule(100.0, 1, adverse, closes, ExecutionSpec())
        assert sched == (1, 100.0, ExecutionSpec().maker_fee_bps, "maker_fill")

    def test_SCENARIO_MHS_PEG_CHASE_02_BACKSTOP_TAKER_AFTER_PASSIVE_PHASE(self) -> None:
        """SCENARIO_MHS_PEG_CHASE_02_BACKSTOP_TAKER_AFTER_PASSIVE_PHASE:
        P=ceil(0.6*10)=6; no crossing during bars 0..5, then closes[6]=100.05
        stays within anchor*(1+10bps)=100.10 -> taker backstop with fee 8.0."""
        adverse = np.full(10, 100.5)
        closes = np.full(10, 100.0)
        closes[6] = 100.05
        sched = peg_chase_fill_schedule(100.0, 1, adverse, closes, ExecutionSpec())
        assert sched is not None
        assert sched == (6, 100.05, ExecutionSpec().taker_fee_bps + ExecutionSpec().taker_slippage_bps, "backstop_taker")
        assert sched[2] == 8.0

    def test_SCENARIO_MHS_FAIR_01_BACKSTOP_ALWAYS_COMPLETES(self) -> None:
        """SCENARIO_MHS_FAIR_01_BACKSTOP_ALWAYS_COMPLETES: the taker backstop
        crosses unconditionally -- every adverse value and close sits far
        outside the 10bps band that previously vetoed the backstop, yet the
        schedule now fills at the first post-passive bar's close."""
        adverse = np.full(10, 100.20)
        closes = np.full(10, 100.20)
        sched = peg_chase_fill_schedule(100.0, 1, adverse, closes, ExecutionSpec())
        assert sched is not None
        assert sched == (6, 100.20, ExecutionSpec().taker_fee_bps + ExecutionSpec().taker_slippage_bps, "backstop_taker")

    def test_SCENARIO_MHS_FAIR_02_RESIDUAL_ONLY_ON_DATA_GAP(self) -> None:
        """SCENARIO_MHS_FAIR_02_RESIDUAL_ONLY_ON_DATA_GAP: over 500 randomised
        finite-close windows the schedule never returns None; an all-NaN close
        window is the only residual (data gap)."""
        rng = np.random.default_rng(20260824)
        for _ in range(500):
            n = int(rng.integers(1, 21))
            drift = float(rng.uniform(-0.02, 0.02))
            path = 100.0 * np.exp(np.cumsum(rng.normal(drift / n, 0.004, n)))
            adverse = path * float(rng.uniform(1.0005, 1.005))
            for side in (-1, 1):
                assert peg_chase_fill_schedule(
                    100.0, side, adverse, path, ExecutionSpec(),
                ) is not None, (n, drift, side)
        all_nan = np.full(6, np.nan)
        assert peg_chase_fill_schedule(
            100.0, 1, np.full(6, 100.2), all_nan, ExecutionSpec(),
        ) is None

    def test_SCENARIO_MHS_FAIR_03_CORWIN_SCHULTZ_BOUNDED_AND_ORDERED(self) -> None:
        """SCENARIO_MHS_FAIR_03_CORWIN_SCHULTZ_BOUNDED_AND_ORDERED: zero-range
        bars price 0.0, wider bands price strictly higher inside the hard
        ceiling, a 2-valid-bar column is nan, and malformed inputs fail closed."""
        n = 12
        lows = np.full((n, 3), 100.0)
        highs = np.full((n, 3), 100.0)
        highs[:, 1] = 100.0 * (1.0 + 10.0 / 1e4)
        highs[:, 2] = 100.0 * (1.0 + 100.0 / 1e4)
        out = corwin_schultz_half_spread_bps(highs, lows)
        assert out.shape == (3,)
        assert out[0] == 0.0
        assert np.isfinite(out).all()
        assert 0.0 < out[1] < out[2] <= SPREAD_ESTIMATE_CEILING_BPS
        two_bars = np.array([[100.0, 101.0], [100.0, 101.0]])
        assert np.isnan(corwin_schultz_half_spread_bps(two_bars, two_bars * 0.999)[0])
        with pytest.raises(ValueError, match="shape"):
            corwin_schultz_half_spread_bps(highs, lows[:, :2])
        with pytest.raises(ValueError, match="rows"):
            corwin_schultz_half_spread_bps(np.array([[1.0]]), np.array([[1.0]]))

    def test_SCENARIO_MHS_PEG_CHASE_03_PRICE_LEAVING_BAND_STILL_FILLS(self) -> None:
        """SCENARIO_MHS_FAIR supersedes the pre-fix residual expectation: a
        price leaving the band no longer strands the intent -- the backstop
        completes it at the first post-passive finite close, so the full replay
        books one fallback fill and zero residuals."""
        adverse = np.full(10, 100.20)
        closes = np.full(10, 100.20)
        sched = peg_chase_fill_schedule(100.0, 1, adverse, closes, ExecutionSpec())
        assert sched is not None
        assert sched[3] == "backstop_taker"

        grid = pd.date_range("2021-01-01 12:00", periods=40, freq="5min", tz="UTC")
        px = pd.DataFrame({"A": [100.20] * len(grid)}, index=grid)
        highs = pd.DataFrame({"A": [100.25] * len(grid)}, index=grid)
        lows = pd.DataFrame({"A": [100.15] * len(grid)}, index=grid)
        marks = px.copy()
        marks.iloc[0, 0] = 100.0
        funding = pd.DataFrame(0.0, index=grid, columns=["A"])
        target = pd.DataFrame(
            {"A": [0.01]}, index=pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        )
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 13:00", tz="UTC")])
        report = replay_execution_windows(
            _partition_windows(grid, target, signal_at, highs, lows, px, marks, funding, ExecutionSpec(), n_windows=1),
            1.0, "OHLCV_PEG_CHASE_PROXY", ExecutionSpec(), retain_event_snapshots=True,
        )
        assert not report.simulated_fills.empty
        assert len(report.simulated_fills) == 1
        assert report.simulated_fills["reason"].iloc[0] == "timeout_taker"
        assert report.simulated_fills["fee_bps"].iloc[0] == (
            ExecutionSpec().taker_fee_bps + ExecutionSpec().taker_slippage_bps
        )
        assert report.fill_count + report.fallback_count + report.residual_count == 1
        assert report.residual_count == 0

    def test_SCENARIO_MHS_PEG_CHASE_04_FAVOURABLE_SIDE_IS_NEVER_CLAMPED(self) -> None:
        """SCENARIO_MHS_PEG_CHASE_04_FAVOURABLE_SIDE_IS_NEVER_CLAMPED:
        A 100bps favourable fall carries the peg below anchor*(1-10bps); a
        symmetric band clamp would have blocked this maker_fill."""
        closes = np.full(10, 99.85)
        closes[0] = 100.0
        adverse = np.full(10, 100.5)
        adverse[2] = 99.70
        sched = peg_chase_fill_schedule(100.0, 1, adverse, closes, ExecutionSpec())
        assert sched is not None
        assert sched[3] == "maker_fill"
        assert sched[1] < 99.90
        assert sched == (2, 99.85, ExecutionSpec().maker_fee_bps, "maker_fill")

    def test_sell_side_mirrors_and_follows_favourable_rise(self) -> None:
        """Sell side uses highs as the adverse path and its peg follows a
        favourable rise above the cap without clamping."""
        closes = np.full(10, 100.0)
        closes[1:] = 100.15
        adverse = np.full(10, 99.5)
        adverse[2] = 100.25
        sched = peg_chase_fill_schedule(100.0, -1, adverse, closes, ExecutionSpec())
        assert sched is not None
        assert sched[0] == 2
        assert sched[2] == ExecutionSpec().maker_fee_bps
        assert sched[3] == "maker_fill"
        assert sched[1] > 100.0 * (1 - ExecutionSpec().peg_chase_band_bps / 1e4)

    def test_fails_closed_on_invalid_input(self) -> None:
        spec = ExecutionSpec()
        with pytest.raises(ValueError, match="side"):
            peg_chase_fill_schedule(100.0, 0, np.array([99.0]), np.array([100.0]), spec)
        with pytest.raises(ValueError, match="anchor"):
            peg_chase_fill_schedule(0.0, 1, np.array([99.0]), np.array([100.0]), spec)
        with pytest.raises(ValueError, match="anchor"):
            peg_chase_fill_schedule(-1.0, -1, np.array([101.0]), np.array([100.0]), spec)
        with pytest.raises(ValueError, match="adverse"):
            peg_chase_fill_schedule(100.0, 1, np.array([]), np.array([]), spec)
        with pytest.raises(ValueError, match="finite"):
            peg_chase_fill_schedule(100.0, 1, np.array([np.nan]), np.array([100.0]), spec)
        with pytest.raises(ValueError, match="finite"):
            peg_chase_fill_schedule(100.0, 1, np.array([np.inf]), np.array([100.0]), spec)
        # A window with no finite close is a data gap: None, never a raise.
        assert peg_chase_fill_schedule(100.0, 1, np.array([99.0]), np.array([np.nan]), spec) is None
        assert peg_chase_fill_schedule(100.0, -1, np.array([101.0]), np.array([np.inf]), spec) is None

    def test_fails_closed_when_closes_shorter_than_adverse(self) -> None:
        with pytest.raises(ValueError, match="closes must be at least as long as adverse"):
            peg_chase_fill_schedule(100.0, 1, np.arange(4.0), np.arange(3.0), ExecutionSpec())


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
        wl = _WindowedReplayWorkload()._workload(days=10, n_symbols=4)
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


class TestPegChasePartialSchedule:
    """S7: tranched peg-chase with exact quantity conservation and completion."""

    def test_SCENARIO_MHS_EVID_04_PARTIAL_SCHEDULE_CONSERVES_AND_COMPLETES(self) -> None:
        """SCENARIO_MHS_EVID_04_PARTIAL_SCHEDULE_CONSERVES_AND_COMPLETES: over
        seeded random windows every tranched schedule is non-empty, conserves
        quantity (qty_fraction sums to 1.0 within 1e-12), and the tranches==1
        output is exactly the single-fill schedule's tuple."""
        from src.mhs.execution import peg_chase_partial_schedule

        rng = np.random.default_rng(20260824)
        for _case in range(100):
            n = int(rng.integers(2, 40))
            anchor = float(rng.uniform(50.0, 150.0))
            side = int(rng.choice([-1, 1]))
            closes = anchor * np.exp(np.cumsum(rng.normal(0.0, 0.004, n)))
            # adverse = lows(buy) / highs(sell): straddle around each bar close.
            adverse = closes * (
                1.0 - side * np.abs(rng.normal(0.0, 0.003, n))
            )
            for tranches in (1, 2, 4):
                spec = ExecutionSpec(peg_chase_tranches=tranches)
                schedule = peg_chase_partial_schedule(
                    anchor, side, adverse, closes, spec
                )
                assert schedule, "a completable window must never yield an empty schedule"
                assert abs(sum(entry[3] for entry in schedule) - 1.0) <= 1e-12
                for rel_pos, price, _fee, _fraction, reason in schedule:
                    assert 0 <= rel_pos < n
                    assert np.isfinite(price)
                    assert price > 0
                    assert reason in ("maker_fill", "backstop_taker")
                if tranches == 1:
                    single = peg_chase_fill_schedule(anchor, side, adverse, closes, spec)
                    assert single is not None
                    assert len(schedule) == 1
                    rel_pos, price, fee_bps, _fraction, reason = schedule[0]
                    assert (rel_pos, price, fee_bps, reason) == single

    def test_partial_schedule_splits_into_maker_tranches_with_carried_share(self) -> None:
        """A mid-window trade-through fills only its tranche share as maker;
        the remaining carried share crosses via the final backstop."""
        from types import SimpleNamespace

        from src.mhs.execution import peg_chase_partial_schedule

        n = 12
        anchor = 100.0
        closes = np.full(n, 100.20)
        adverse = np.full(n, 100.30)  # never trades through on its own
        adverse[5] = 99.50  # one deep trade-through inside the middle third
        spec = ExecutionSpec(peg_chase_tranches=3)
        schedule = peg_chase_partial_schedule(anchor, 1, adverse, closes, spec)
        assert [entry[4] for entry in schedule] == ["maker_fill", "backstop_taker"]
        # Tranche 0 failed -> its 1/3 share is carried into tranche 1, whose
        # own deep trade-through fills both shares as maker.
        maker_fraction = schedule[0][3]
        assert abs(maker_fraction - 2.0 / 3) <= 1e-12
        assert schedule[0][0] == 5
        backstop_fraction = schedule[1][3]
        assert abs(maker_fraction + backstop_fraction - 1.0) <= 1e-12
        assert schedule[1][0] == 11

        # Fail-closed: a malformed tranches value raises even pre-validation.
        bogus = SimpleNamespace(peg_chase_tranches=0)
        with pytest.raises(ValueError, match="peg_chase_tranches"):
            peg_chase_partial_schedule(100.0, 1, np.array([99.0]), np.array([100.0]), bogus)

    def test_partial_schedule_returns_empty_on_total_data_gap(self) -> None:
        from src.mhs.execution import peg_chase_partial_schedule

        all_nan = np.full(6, np.nan)
        assert (
            peg_chase_partial_schedule(
                100.0, 1, np.full(6, 100.2), all_nan, ExecutionSpec(peg_chase_tranches=2)
            )
            == []
        )
