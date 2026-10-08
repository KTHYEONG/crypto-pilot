"""Symbol-level no-trade exit deferral: bounded, disclosed, fail-closed."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.engine.execution import (
    ExecutionReplayWindow,
    ExecutionSpec,
    replay_execution_window_batch_isolated,
    replay_execution_windows,
    replay_execution_windows_coupled,
)
from src.core.venue_halts import VenueHaltInterval

COLS = ("AUSDT", "BUSDT")


def _halt(start: pd.Timestamp, bars: int) -> VenueHaltInterval:
    return VenueHaltInterval(
        halt_id=start.strftime("%Y-%m-%dT%H:%MZ"),
        start=start,
        end=start + pd.Timedelta(minutes=3 * bars),
        present_symbols=12,
        zero_symbols=12,
        evidence="test halt",
        verified_at=pd.Timestamp("2026-07-01T00:00:00Z"),
    )


def _window(
    grid: pd.DatetimeIndex,
    decisions: list[str],
    weights: dict[str, list[float]],
    halt_bars: set[str] | None = None,
    zero_bars: set[str] | None = None,
    nan_qv_at: set[str] | None = None,
    funding_unknown_at: set[str] | None = None,
    nan_close_at: set[str] | None = None,
    nan_mark_at: set[str] | None = None,
) -> ExecutionReplayWindow:
    px = pd.DataFrame({s: np.full(len(grid), 100.0) for s in COLS}, index=grid)
    for stamp in nan_close_at or ():
        px.loc[pd.Timestamp(stamp, tz="UTC"), "AUSDT"] = np.nan
    marks = px.copy()
    for stamp in nan_mark_at or ():
        marks.loc[pd.Timestamp(stamp, tz="UTC"), "AUSDT"] = np.nan
    qv = pd.DataFrame({s: np.full(len(grid), 1000.0) for s in COLS}, index=grid)
    for stamp in (halt_bars or set()) | (zero_bars or set()):
        qv.loc[pd.Timestamp(stamp, tz="UTC")] = 0.0
    for stamp in nan_qv_at or ():
        qv.loc[pd.Timestamp(stamp, tz="UTC"), "AUSDT"] = np.nan
    known = pd.DataFrame(True, index=grid, columns=list(COLS))
    for stamp in funding_unknown_at or ():
        known.loc[pd.Timestamp(stamp, tz="UTC"), "AUSDT"] = False
    fund = pd.DataFrame({s: np.zeros(len(grid)) for s in COLS}, index=grid)
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in decisions])
    w = pd.DataFrame(0.0, index=idx, columns=list(COLS))
    for s, vals in weights.items():
        w[s] = vals
    halts: tuple[VenueHaltInterval, ...] = ()
    if halt_bars:
        ordered = sorted(pd.Timestamp(s, tz="UTC") for s in halt_bars)
        run_start = ordered[0]
        prev = ordered[0]
        runs: list[tuple[pd.Timestamp, int]] = []
        for stamp in ordered[1:]:
            if stamp == prev + pd.Timedelta(minutes=3):
                prev = stamp
                continue
            runs.append((run_start, int((prev - run_start).total_seconds() // 180) + 1))
            run_start = stamp
            prev = stamp
        runs.append((run_start, int((prev - run_start).total_seconds() // 180) + 1))
        halts = tuple(_halt(s, n) for s, n in runs)
    return ExecutionReplayWindow(
        window_start=grid[0],
        window_end=grid[-1] + pd.Timedelta(minutes=3),
        columns=COLS,
        symbols=COLS,
        minute_grid=grid,
        highs=px,
        lows=px,
        closes=px,
        marks=marks,
        bar_funding=fund,
        target_weights=w,
        signal_available_at=idx,
        quote_volumes=qv,
        funding_known=known,
        bar_available_at=grid + pd.Timedelta(minutes=3),
        venue_halts=halts,
    )


def _grid(start: str = "2022-01-14 00:00", periods: int = 60) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="3min", tz="UTC")


def _exit_setup(periods: int = 60, exit_at: int = 20) -> tuple[pd.DatetimeIndex, list[str], dict[str, list[float]]]:
    grid = _grid(periods=periods)
    decisions = [str(grid[0]), str(grid[exit_at])]
    return grid, decisions, {"AUSDT": [0.5, 0.0], "BUSDT": [0.0, 0.0]}


def test_symbol_freeze_exit_defers_to_first_viable_bar() -> None:
    grid, decisions, weights = _exit_setup()
    w = _window(grid, decisions, weights, zero_bars={str(grid[21])})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("SYMBOL_NO_TRADE_DEFERRED_EXIT") == 1
    assert len(result.exit_block_disclosures) == 1
    block = result.exit_block_disclosures[0]
    assert block.cause == "SYMBOL_NO_TRADE"
    assert block.halt_id is None
    assert block.outcome == "deferred_fill"
    assert block.blocked_bar == grid[21]
    assert block.filled_bar == grid[22]
    taker = result.simulated_fills[
        (result.simulated_fills["reason"] == "timeout_taker") & (result.simulated_fills["quantity_delta"] < 0)
    ]
    assert len(taker) == 1
    assert float(taker.iloc[0]["fill_price"]) == 100.0
    assert result.data_gaps == ()
    assert result.ledger.primary_valid


def test_symbol_freeze_through_timeout_retries_then_exits() -> None:
    grid = _grid(periods=80)
    decisions = [str(grid[0]), str(grid[20]), str(grid[40]), str(grid[60])]
    weights = {"AUSDT": [0.5, 0.0, 0.0, 0.0], "BUSDT": [0.0, 0.0, 0.0, 0.0]}
    w = _window(grid, decisions, weights, zero_bars={str(grid[i]) for i in range(21, 61)})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("BLOCKED_EXIT_SYMBOL_NO_TRADE") == 2
    assert result.termination_counts.get("SYMBOL_NO_TRADE_DEFERRED_EXIT", 0) == 0
    assert len(result.exit_block_disclosures) == 2
    assert {b.outcome for b in result.exit_block_disclosures} == {"retry_next_decision"}
    assert all(b.cause == "SYMBOL_NO_TRADE" for b in result.exit_block_disclosures)
    assert result.ledger.primary_valid
    assert result.data_gaps == ()
    exits = result.simulated_fills[result.simulated_fills["quantity_delta"] < 0]
    assert len(exits) == 1


def test_entry_on_frozen_symbol_is_unchanged() -> None:
    grid = _grid(periods=40)
    w = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]}, zero_bars={str(grid[1])})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("NO_VOLUME_UNFILLED") == 1
    assert result.exit_block_disclosures == ()
    assert result.data_gaps == ()
    assert result.simulated_fills.empty


def test_nan_quote_volume_exit_stays_fatal() -> None:
    grid, decisions, weights = _exit_setup()
    w = _window(grid, decisions, weights, nan_qv_at={str(grid[21])})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert not result.ledger.primary_valid
    assert any(g.code == "ZERO_OR_UNKNOWN_VOLUME" for g in result.data_gaps)
    assert result.exit_block_disclosures == ()


def test_unknown_funding_on_frozen_exit_stays_fatal() -> None:
    grid, decisions, weights = _exit_setup()
    bar = {str(grid[21])}
    w = _window(grid, decisions, weights, zero_bars=bar, funding_unknown_at=bar)
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert not result.ledger.primary_valid
    assert any(g.code == "BLOCKED_EXIT_UNKNOWN_FUNDING" for g in result.data_gaps)
    assert result.exit_block_disclosures == ()


def test_invalid_mark_on_frozen_exit_stays_fatal() -> None:
    grid, decisions, weights = _exit_setup()
    w = _window(grid, decisions, weights, zero_bars={str(grid[21])}, nan_mark_at={str(grid[21])})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert not result.ledger.primary_valid
    assert any(g.code == "KNOWN_ZERO_VOLUME" for g in result.data_gaps)
    assert result.exit_block_disclosures == ()


def _hourly_freeze(n_exits: int, periods: int) -> tuple[pd.DatetimeIndex, list[str], dict[str, list[float]]]:
    grid = _grid(periods=periods)
    decisions = [str(grid[20 * k]) for k in range(n_exits + 1)]
    return grid, decisions, {"AUSDT": [0.5] + [0.0] * n_exits, "BUSDT": [0.0] * (n_exits + 1)}


def test_deferral_bound_is_inclusive_at_24h() -> None:
    grid, decisions, weights = _hourly_freeze(25, 540)
    w = _window(grid, decisions, weights, zero_bars={str(grid[i]) for i in range(20, 540)})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("BLOCKED_EXIT_SYMBOL_NO_TRADE") == 25
    assert result.termination_counts.get("EXIT_BLOCK_BOUND_EXCEEDED", 0) == 0
    assert len(result.exit_block_disclosures) == 25
    assert result.ledger.primary_valid
    assert result.data_gaps == ()


def test_deferral_bound_exceeded_fails_closed() -> None:
    grid, decisions, weights = _hourly_freeze(26, 560)
    w = _window(grid, decisions, weights, zero_bars={str(grid[i]) for i in range(20, 560)})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("BLOCKED_EXIT_SYMBOL_NO_TRADE") == 25
    assert result.termination_counts.get("EXIT_BLOCK_BOUND_EXCEEDED") == 1
    assert len(result.exit_block_disclosures) == 25
    assert not result.ledger.primary_valid
    assert any(g.code == "KNOWN_ZERO_VOLUME" for g in result.data_gaps)


def test_episode_resets_after_inventory_change() -> None:
    grid = _grid(periods=640)
    decisions = [str(grid[20 * k]) for k in range(31)]
    weights = {"AUSDT": [0.5] + [0.0] * 11 + [0.25] + [0.0] * 18, "BUSDT": [0.0] * 31}
    zero = {str(grid[i]) for i in range(20, 640)} - {str(grid[241])}
    w = _window(grid, decisions, weights, zero_bars=zero)
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("EXIT_BLOCK_BOUND_EXCEEDED", 0) == 0
    assert len(result.exit_block_disclosures) == 29
    assert result.ledger.primary_valid
    assert result.data_gaps == ()


def test_episode_resets_after_exit_and_reentry_at_same_inventory() -> None:
    grid = _grid(periods=660)
    decisions = [str(grid[i]) for i in (0, 20, 40, 60, 620)]
    spec = ExecutionSpec()
    initial_units = 50.0
    round_trip_fees = 2 * initial_units * 100.0 * (spec.taker_fee_bps + spec.taker_slippage_bps) / 1e4
    reentry_weight = initial_units * 100.0 / (10000.0 - round_trip_fees)
    weights = {"AUSDT": [0.5, 0.0, 0.0, reentry_weight, 0.0], "BUSDT": [0.0] * 5}
    zero = {str(grid[i]) for i in range(21, 32)} | {str(grid[i]) for i in range(621, 660)}
    result = replay_execution_windows((_window(grid, decisions, weights, zero_bars=zero),),
                                      10000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    first, second = result.exit_block_disclosures
    assert first.quantity == second.quantity == -initial_units
    assert second.decision_time - first.decision_time > pd.Timedelta(hours=24)
    assert result.termination_counts.get("EXIT_BLOCK_BOUND_EXCEEDED", 0) == 0
    assert result.ledger.primary_valid
    assert result.data_gaps == ()


def test_halt_wins_cause_precedence_over_zero_volume() -> None:
    grid, decisions, weights = _exit_setup()
    w = _window(grid, decisions, weights, halt_bars={str(grid[i]) for i in (21, 22, 23)})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("VENUE_HALT_DEFERRED_EXIT") == 1
    block = result.exit_block_disclosures[0]
    assert block.cause == "VENUE_HALT"
    assert block.halt_id == "2022-01-14T01:03Z"
    assert block.filled_bar == grid[24]


def test_zero_volume_cannot_hide_invalid_submit_timing() -> None:
    grid, decisions, weights = _exit_setup()
    window = _window(grid, decisions, weights, zero_bars={str(grid[21])})
    availability = list(window.bar_available_at)
    availability[21] = grid[20]
    window = dataclasses.replace(window, bar_available_at=pd.DatetimeIndex(availability))
    result = replay_execution_windows((window,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert not result.ledger.primary_valid
    assert any(g.code == "CAUSAL_TIMING_VIOLATION" for g in result.data_gaps)
    assert result.exit_block_disclosures == ()
    assert len(result.simulated_fills) == 1


def test_no_lookahead_past_order_deadline() -> None:
    grid, decisions, weights = _exit_setup()
    frozen = {str(grid[i]) for i in range(21, 32)}
    viable = _window(grid, decisions, weights, zero_bars=frozen)
    zeroed = _window(grid, decisions, weights, zero_bars=frozen | {str(grid[32])})
    first = replay_execution_windows((viable,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    second = replay_execution_windows((zeroed,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert first.exit_block_disclosures == second.exit_block_disclosures
    pd.testing.assert_frame_equal(first.simulated_fills, second.simulated_fills, check_exact=True)
    pd.testing.assert_series_equal(first.ledger.equity, second.ledger.equity, check_exact=True)
    assert first.exit_block_disclosures[0].outcome == "retry_next_decision"


def test_window_without_blocked_exit_has_no_disclosures() -> None:
    grid, decisions, weights = _exit_setup()
    w = _window(grid, decisions, weights)
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.ledger.primary_valid
    assert result.exit_block_disclosures == ()
    assert result.termination_counts.get("BLOCKED_EXIT_SYMBOL_NO_TRADE", 0) == 0
    assert result.termination_counts.get("SYMBOL_NO_TRADE_DEFERRED_EXIT", 0) == 0
    assert result.termination_counts.get("EXIT_BLOCK_BOUND_EXCEEDED", 0) == 0


def test_batch_parity_for_symbol_freeze_exit() -> None:
    grid, decisions, weights = _exit_setup()
    w = _window(grid, decisions, weights, zero_bars={str(grid[21])})
    spec = ExecutionSpec()
    single = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    outcome = replay_execution_window_batch_isolated(
        (w,), 10000.0, [("OHLCV_IMMEDIATE_TAKER", spec)],
    )
    batched = outcome.results[0]
    assert batched is not None
    assert batched.exit_block_disclosures == single.exit_block_disclosures
    assert batched.simulated_fills.reset_index(drop=True).equals(single.simulated_fills.reset_index(drop=True))
    assert dict(batched.termination_counts) == dict(single.termination_counts)
    reference, coupled_outcome = replay_execution_windows_coupled(
        (w,), 10000.0, ("OHLCV_IMMEDIATE_TAKER", spec), [("OHLCV_IMMEDIATE_TAKER", spec)],
        lambda _daily: pd.Series(1.0, index=w.target_weights.index),
    )
    assert coupled_outcome.isolated_failures == ()
    for result in (reference, coupled_outcome.results[0]):
        assert result is not None
        assert result.exit_block_disclosures == batched.exit_block_disclosures
        assert dict(result.termination_counts) == dict(batched.termination_counts)
        pd.testing.assert_frame_equal(result.simulated_fills, batched.simulated_fills, check_exact=True)
        pd.testing.assert_series_equal(result.ledger.equity, batched.ledger.equity, check_exact=True)
