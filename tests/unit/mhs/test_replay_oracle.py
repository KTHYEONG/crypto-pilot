"""Execution replay engine contract tests (split by behavioral domain)."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.mhs.execution import (
    ExecutionSpec,
    _BoundExecutionReplayAccumulator,
    replay_execution_windows,
    strategy_aware_execution_replay,
)

from tests.unit.mhs.test_execution import (  # noqa: F401
    _partition_windows,
)

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
        above -- the tolerance change must not weaken this fail-closed gate.

        SCENARIO_MHS_FORCED_EXIT_COST_MODEL_FLAT_BACKWARD_COMPAT: under the
        production-default ``liquidity_cost_model="flat"`` the forced_exit
        fee_bps stays byte-identical -- the shared taker-cost wiring reduces
        exactly to the frozen slippage under flat."""
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
