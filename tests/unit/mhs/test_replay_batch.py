"""Execution replay engine contract tests (split by behavioral domain)."""

from __future__ import annotations
import dataclasses
import numpy as np
import pandas as pd
import pytest
from src.common.errors import DataIntegrityError
from src.mhs.execution import (
    ExecutionSpec,
    replay_execution_windows,
    strategy_aware_execution_replay,
)

from tests.unit.mhs.test_execution import (  # noqa: F401
    _assert_pair_equivalent,
    _partition_windows,
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

class TestIsolatedBoundReplay:
    """SCENARIO_MHS_ISOLATED_BOUND_RETIRED + SCENARIO_MHS_ISOLATED_EMPTY_SET_PROPAGATES:
    isolated reference-only bounds retire cleanly while capital bounds continue."""

    def _workload(self, days: int = 40, n_symbols: int = 8) -> dict[str, object]:
        grid = pd.date_range("2021-01-01", periods=days * 24 * 12, freq="5min", tz="UTC")
        symbols = [f"SYM{i:03d}USDT" for i in range(n_symbols)]
        rng = np.random.default_rng(7)
        closes = pd.DataFrame(
            {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(grid)))) for s in symbols},
            index=grid,
        )
        marks = closes.copy()
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

    def test_isolated_bound_retired_while_others_continue(self) -> None:
        """SCENARIO_MHS_ISOLATED_BOUND_RETIRED: a DataIntegrityError from
        isolated index 2 retires that bound (results[2] is None) while
        capital-carrying bounds (0, 1) complete independently and match
        single-bound replays at rtol=atol=1e-12."""
        from src.mhs.execution import (
            BatchReplayOutcome,
            replay_execution_window_batch_isolated,
        )
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        spec = ExecutionSpec()
        bounds = [
            ("OHLCV_IMMEDIATE_TAKER", spec),
            ("OHLCV_IMMEDIATE_TAKER", ExecutionSpec(maker_fee_bps=6.0)),
            ("OHLCV_STRICT_PROXY", spec),
        ]
        # Run independent replays for bounds 0,1 BEFORE monkeypatching
        independent = [
            replay_execution_windows(windows, 1.0, b, s, retain_event_snapshots=True)
            for (b, s) in bounds[:2]
        ]
        from src.mhs.execution import _BoundExecutionReplayAccumulator
        original_consume = _BoundExecutionReplayAccumulator.consume
        window_index = [0]
        second_start = windows[1].window_start if len(windows) > 1 else None

        def _failing_consume(self, w):
            window_index[0] += 1
            if self.execution_bound == "OHLCV_STRICT_PROXY" and w.window_start == second_start:
                raise DataIntegrityError("pre-trade equity must be positive and finite (ts=fail)")
            return original_consume(self, w)
        _BoundExecutionReplayAccumulator.consume = _failing_consume
        try:
            consumed = {"n": 0}
            def _gen():
                for w in windows:
                    consumed["n"] += 1
                    yield w
            outcome: BatchReplayOutcome = replay_execution_window_batch_isolated(
                _gen(), 1.0, bounds, retain_event_snapshots=True,
                isolated_bound_indices=frozenset({2}),
            )
        finally:
            _BoundExecutionReplayAccumulator.consume = original_consume
        assert consumed["n"] == len(windows)
        # Bounds 0 and 1 are not None and match independent replays
        for i in range(2):
            assert outcome.results[i] is not None
            _assert_pair_equivalent(independent[i], outcome.results[i], f"batch[{i}]")
        # Bound 2 (strict) was retired mid-stream
        assert outcome.results[2] is None
        assert len(outcome.isolated_failures) == 1
        f = outcome.isolated_failures[0]
        assert f.bound_index == 2
        assert f.execution_bound == "OHLCV_STRICT_PROXY"
        assert f.error_class == "DataIntegrityError"
        assert f.windows_consumed >= 1

    def test_out_of_range_isolated_index_raises_value_error(self) -> None:
        """SCENARIO_MHS_ISOLATED_EMPTY_SET_PROPAGATES: an out-of-range
        isolated index raises ValueError before any window is consumed."""
        from src.mhs.execution import replay_execution_window_batch_isolated
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=2,
        )
        bounds = [
            ("OHLCV_IMMEDIATE_TAKER", ExecutionSpec()),
            ("OHLCV_IMMEDIATE_TAKER", ExecutionSpec()),
            ("OHLCV_STRICT_PROXY", ExecutionSpec()),
        ]
        consumed = {"n": 0}
        def _gen():
            for w in windows:
                consumed["n"] += 1
                yield w
        with pytest.raises(ValueError, match="out of range"):
            replay_execution_window_batch_isolated(
                _gen(), 1.0, bounds, isolated_bound_indices=frozenset({7}),
            )
        assert consumed["n"] == 0

    def test_empty_isolated_set_propagates_error(self) -> None:
        """SCENARIO_MHS_ISOLATED_EMPTY_SET_PROPAGATES: with isolated_bound_indices=frozenset(),
        a failing stream propagates DataIntegrityError identical to the non-isolated wrapper."""
        from src.mhs.execution import replay_execution_window_batch_isolated
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=2,
        )
        windows[1] = dataclasses.replace(windows[1], bar_funding=windows[1].bar_funding * float("nan"))
        bounds = [
            ("OHLCV_IMMEDIATE_TAKER", ExecutionSpec()),
            ("OHLCV_IMMEDIATE_TAKER", ExecutionSpec()),
            ("OHLCV_STRICT_PROXY", ExecutionSpec()),
        ]
        with pytest.raises(DataIntegrityError, match="bar_funding must be finite"):
            replay_execution_window_batch_isolated(
                iter(windows), 1.0, bounds, isolated_bound_indices=frozenset(),
            )
