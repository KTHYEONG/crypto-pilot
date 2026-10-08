"""Part 3 venue-halt replay invariants: deferred exits inside the order timeout."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
from src.mhs.execution.contracts import InstrumentSettlementEvent, VenueHaltExitBlock
from src.mhs.evaluation.windows import _load_window_from_ipc, _spill_window_to_ipc
from src.mhs.venue_halts import VenueHaltInterval

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
    funding_unknown_at: set[str] | None = None,
    events: tuple[InstrumentSettlementEvent, ...] = (),
    high_spike: dict[str, float] | None = None,
    nan_close_at: set[str] | None = None,
) -> ExecutionReplayWindow:
    px = pd.DataFrame({s: np.full(len(grid), 100.0) for s in COLS}, index=grid)
    hi = px.copy()
    for stamp, value in (high_spike or {}).items():
        hi.loc[pd.Timestamp(stamp, tz="UTC"), "AUSDT"] = float(value)
    for stamp in nan_close_at or ():
        px.loc[pd.Timestamp(stamp, tz="UTC"), "AUSDT"] = np.nan
    qv = pd.DataFrame({s: np.full(len(grid), 1000.0) for s in COLS}, index=grid)
    for stamp in (halt_bars or set()) | (zero_bars or set()):
        qv.loc[pd.Timestamp(stamp, tz="UTC")] = 0.0
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
        highs=hi,
        lows=px,
        closes=px,
        marks=px,
        bar_funding=fund,
        target_weights=w,
        signal_available_at=idx,
        quote_volumes=qv,
        funding_known=known,
        bar_available_at=grid + pd.Timedelta(minutes=3),
        settlement_events=events,
        venue_halts=halts,
    )


def _grid(start: str = "2022-01-14 00:00", periods: int = 60) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="3min", tz="UTC")


def _exit_setup(periods: int = 60, exit_at: int = 20) -> tuple[pd.DatetimeIndex, list[str], dict[str, list[float]]]:
    grid = _grid(periods=periods)
    decisions = [str(grid[0]), str(grid[exit_at])]
    return grid, decisions, {"AUSDT": [0.5, 0.0], "BUSDT": [0.0, 0.0]}


def test_halted_exit_deferred_inside_timeout() -> None:
    grid, decisions, weights = _exit_setup()
    halt = {str(grid[i]) for i in (21, 22, 23)}
    w = _window(grid, decisions, weights, halt_bars=halt)
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("VENUE_HALT_DEFERRED_EXIT") == 1
    assert len(result.venue_halt_exit_blocks) == 1
    block = result.venue_halt_exit_blocks[0]
    assert block.outcome == "deferred_fill"
    assert block.blocked_bar == grid[21]
    assert block.filled_bar == grid[24]
    assert block.halt_id == "2022-01-14T01:03Z"
    taker = result.simulated_fills[
        (result.simulated_fills["reason"] == "timeout_taker") & (result.simulated_fills["quantity_delta"] < 0)
    ]
    assert len(taker) == 1
    assert float(taker.iloc[0]["fill_price"]) == 100.0
    assert result.data_gaps == ()
    assert result.ledger.primary_valid
    assert float(result.simulated_units.iloc[-1].get("AUSDT", 0.0) if len(result.simulated_units) else 0.0) == 0.0


def test_halt_through_deadline_retries_next_decision() -> None:
    grid = _grid(periods=80)
    decisions = [str(grid[0]), str(grid[20]), str(grid[40])]
    weights = {"AUSDT": [0.5, 0.0, 0.0], "BUSDT": [0.0, 0.0, 0.0]}
    halt = {str(grid[i]) for i in range(21, 36)}
    w = _window(grid, decisions, weights, halt_bars=halt)
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("BLOCKED_EXIT_VENUE_HALT") == 1
    assert result.termination_counts.get("VENUE_HALT_DEFERRED_EXIT", 0) == 0
    assert len(result.venue_halt_exit_blocks) == 1
    block = result.venue_halt_exit_blocks[0]
    assert block.outcome == "retry_next_decision"
    assert block.filled_bar is None
    assert result.ledger.primary_valid
    exits = result.simulated_fills[
        (result.simulated_fills["symbol"] == "AUSDT")
        & (result.simulated_fills["quantity_delta"] < 0)
        & (result.simulated_fills["reason"] != "delist_settlement")
    ]
    assert len(exits) == 1


def test_unknown_funding_during_halt_still_invalidates() -> None:
    grid, decisions, weights = _exit_setup()
    halt = {str(grid[i]) for i in (21, 22, 23)}
    w = _window(grid, decisions, weights, halt_bars=halt, funding_unknown_at=halt)
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert not result.ledger.primary_valid
    assert any(g.code == "BLOCKED_EXIT_UNKNOWN_FUNDING" for g in result.data_gaps)
    assert result.venue_halt_exit_blocks == ()
    assert result.termination_counts.get("VENUE_HALT_DEFERRED_EXIT", 0) == 0


def test_zero_volume_outside_halt_still_invalidates_exits() -> None:
    grid, decisions, weights = _exit_setup()
    w = _window(grid, decisions, weights, zero_bars={str(grid[21])})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert not result.ledger.primary_valid
    assert any(g.code == "KNOWN_ZERO_VOLUME" for g in result.data_gaps)


def test_entries_during_halt_unchanged() -> None:
    grid = _grid(periods=40)
    halt = {str(grid[i]) for i in (1, 2, 3)}
    w = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]}, halt_bars=halt)
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("NO_VOLUME_UNFILLED") == 1
    assert result.venue_halt_exit_blocks == ()
    assert result.simulated_fills.empty


@pytest.mark.parametrize("bound", ["OHLCV_IMMEDIATE_TAKER", "OHLCV_TOUCH_PROXY", "OHLCV_LADDERED_PROXY", "OHLCV_PEG_CHASE_PROXY"])
def test_halt_blocks_even_positive_volume_and_defers_once(bound) -> None:
    grid, decisions, weights = _exit_setup()
    w = _window(grid, decisions, weights, halt_bars={str(grid[i]) for i in (21, 22, 23)},
                high_spike={str(grid[21]): 101.0})
    volumes = w.quote_volumes.copy()
    volumes.loc[grid[21]:grid[23]] = 1000.0
    w = dataclasses.replace(w, quote_volumes=volumes)
    result = replay_execution_windows((w,), 10000.0, bound, ExecutionSpec(), retain_event_snapshots=True)
    assert result.venue_halt_exit_blocks[0].filled_bar == grid[24]
    assert result.termination_counts["VENUE_HALT_DEFERRED_EXIT"] == 1
    exits = result.simulated_fills.loc[lambda f: f.quantity_delta < 0]
    assert len(exits) == 1
    assert result.simulated_units.iloc[-1]["AUSDT"] == pytest.approx(0.0, abs=1e-12)


def test_deferral_respects_last_trade_cutoff() -> None:
    grid, decisions, weights = _exit_setup()
    halt = {str(grid[i]) for i in (21, 22, 23)}
    last_trade = grid[24]
    delivery = grid[25]
    event = InstrumentSettlementEvent(
        event_id=f"AUSDT:{int(delivery.value // 1_000_000)}",
        symbol="AUSDT",
        effective_at=delivery,
        available_at=delivery,
        settlement_price=100.0,
        fee_bps=5.0,
        source_digest="sha256:test",
        announced_at=last_trade,
        last_trade_at=last_trade,
        price_source="venue",  # type: ignore[arg-type]
    )
    w = _window(grid, decisions, weights, halt_bars=halt, events=(event,))
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("VENUE_HALT_DEFERRED_EXIT", 0) == 0
    assert len(result.venue_halt_exit_blocks) == 1
    assert result.venue_halt_exit_blocks[0].outcome == "retry_next_decision"


def test_laddered_halt_defers_remaining_quantity() -> None:
    grid, decisions, weights = _exit_setup()
    halt = {str(grid[i]) for i in (21, 22, 23)}
    w = _window(
        grid,
        decisions,
        weights,
        halt_bars=halt,
        high_spike={str(grid[21]): 101.0},
    )
    result = replay_execution_windows((w,), 10000.0, "OHLCV_LADDERED_PROXY", ExecutionSpec())
    assert result.termination_counts.get("VENUE_HALT_DEFERRED_EXIT") == 1
    assert len(result.venue_halt_exit_blocks) == 1
    assert result.venue_halt_exit_blocks[0].outcome == "deferred_fill"
    assert result.venue_halt_exit_blocks[0].filled_bar == grid[24]
    assert result.data_gaps == ()


def test_spill_round_trip_keeps_halts(tmp_path) -> None:
    grid, decisions, weights = _exit_setup(periods=30)
    w = _window(grid, decisions, weights, halt_bars={str(grid[5]), str(grid[6])})
    assert len(w.venue_halts) == 1
    path = str(tmp_path / "window_00000.arrow")
    _spill_window_to_ipc(w, path)
    loaded = _load_window_from_ipc(path)
    assert loaded.venue_halts == w.venue_halts


def test_halt_disclosure_record_contract() -> None:
    block = VenueHaltExitBlock(
        symbol="AUSDT",
        decision_time=pd.Timestamp("2022-01-14 01:00", tz="UTC"),
        blocked_bar=pd.Timestamp("2022-01-14 01:03", tz="UTC"),
        halt_id="2022-01-14T01:03Z",
        outcome="deferred_fill",
        filled_bar=pd.Timestamp("2022-01-14 01:12", tz="UTC"),
        quantity=-5.0,
    )
    assert block.outcome == "deferred_fill"
    retried = dataclasses.replace(block, outcome="retry_next_decision", filled_bar=None)
    assert retried.filled_bar is None


def test_deferral_scan_skips_inviable_bars() -> None:
    grid, decisions, weights = _exit_setup(periods=80)
    halt = {str(grid[i]) for i in (21, 22, 23)}
    w = _window(
        grid, decisions, weights, halt_bars=halt,
        zero_bars={str(grid[24])},
        funding_unknown_at={str(grid[25])},
        nan_close_at={str(grid[26])},
    )
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("VENUE_HALT_DEFERRED_EXIT") == 1
    assert len(result.venue_halt_exit_blocks) == 1
    assert result.venue_halt_exit_blocks[0].filled_bar == grid[27]
    assert not any(g.code == "BLOCKED_EXIT_VENUE_HALT" for g in result.data_gaps)


def test_halt_disclosure_rejects_malformed_records() -> None:
    good = {
        "symbol": "AUSDT",
        "decision_time": pd.Timestamp("2022-01-14 01:00", tz="UTC"),
        "blocked_bar": pd.Timestamp("2022-01-14 01:03", tz="UTC"),
        "halt_id": "2022-01-14T01:03Z",
        "outcome": "deferred_fill",
        "filled_bar": pd.Timestamp("2022-01-14 01:12", tz="UTC"),
        "quantity": -5.0,
    }
    bad = [
        {**good, "symbol": ""},
        {**good, "outcome": "filled"},
        {**good, "decision_time": "not-a-timestamp"},
        {**good, "blocked_bar": pd.NaT},
        {**good, "filled_bar": "not-a-timestamp"},
        {**good, "outcome": "deferred_fill", "filled_bar": None},
        {**good, "outcome": "retry_next_decision"},
        {**good, "quantity": float("nan")},
    ]
    for kwargs in bad:
        with pytest.raises(DataIntegrityError):
            VenueHaltExitBlock(**kwargs)  # type: ignore[arg-type]


def test_deferred_exit_under_corwin_schultz_costs() -> None:
    import dataclasses as _dc

    grid, decisions, weights = _exit_setup()
    halt = {str(grid[i]) for i in (21, 22, 23)}
    w = _window(grid, decisions, weights, halt_bars=halt)
    spec = _dc.replace(ExecutionSpec(), liquidity_cost_model="corwin_schultz")
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    assert result.termination_counts.get("VENUE_HALT_DEFERRED_EXIT") == 1
    assert result.venue_halt_exit_blocks[0].filled_bar == grid[24]
    assert result.data_gaps == ()


def test_nan_hold_skips_without_known_forced_exit() -> None:
    grid = _grid(periods=60)
    ev = _event_unknown_announcement(grid)
    decisions = [str(grid[0]), str(grid[20])]
    w = _window(
        grid, decisions,
        {"AUSDT": [0.5, float("nan")], "BUSDT": [0.0, float("nan")]},
        events=(ev,),
    )
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("DELISTING_FORCED_EXIT", 0) in (None, 0)
    assert len(result.simulated_fills) == 1


def test_nan_hold_skips_outside_lead_and_eventless() -> None:
    grid = _grid(periods=60)
    far = grid[0] + pd.Timedelta(days=100)
    ev = InstrumentSettlementEvent(
        event_id=f"AUSDT:{int(far.value // 1_000_000)}",
        symbol="AUSDT",
        effective_at=far,
        available_at=far,
        settlement_price=100.0,
        fee_bps=5.0,
        source_digest="sha256:test",
        announced_at=grid[10],
        last_trade_at=far,
        price_source="venue",  # type: ignore[arg-type]
    )
    decisions = [str(grid[0]), str(grid[20])]
    w = _window(
        grid, decisions,
        {"AUSDT": [0.5, float("nan")], "BUSDT": [0.5, float("nan")]},
        events=(ev,),
    )
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("DELISTING_FORCED_EXIT", 0) in (None, 0)
    assert len(result.simulated_fills) == 2


def _event_unknown_announcement(grid: pd.DatetimeIndex) -> InstrumentSettlementEvent:
    far = grid[0] + pd.Timedelta(days=100)
    return InstrumentSettlementEvent(
        event_id=f"AUSDT:{int(far.value // 1_000_000)}",
        symbol="AUSDT",
        effective_at=far,
        available_at=far,
        settlement_price=100.0,
        fee_bps=5.0,
        source_digest="sha256:test",
        announced_at=grid[50],
        last_trade_at=far,
        price_source="venue",  # type: ignore[arg-type]
    )
