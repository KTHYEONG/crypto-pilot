"""Part 2 replay invariants: causal settlement booking in the accumulator."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows
from src.mhs.execution.batch import replay_execution_window_batch
from src.mhs.execution.contracts import InstrumentSettlementEvent


def _event(
    symbol: str = "AUSDT",
    delivery: str = "2022-01-14 01:03",
    last_trade: str = "2022-01-14 01:00",
    price: float = 2.0,
    source: str = "venue",
    fee: float = 5.0,
    announced: str | None = None,
) -> InstrumentSettlementEvent:
    dv = pd.Timestamp(delivery, tz="UTC")
    lt = pd.Timestamp(last_trade, tz="UTC")
    return InstrumentSettlementEvent(
        event_id=f"{symbol}:{int(dv.value // 1_000_000)}",
        symbol=symbol,
        effective_at=dv,
        available_at=dv,
        settlement_price=price,
        fee_bps=fee,
        source_digest="sha256:test",
        announced_at=pd.Timestamp(announced, tz="UTC") if announced is not None else lt,
        last_trade_at=lt,
        price_source=source,  # type: ignore[arg-type]
    )


def _window(
    grid: pd.DatetimeIndex,
    decisions: list[str],
    weights: dict[str, list[float]],
    events: tuple[InstrumentSettlementEvent, ...] = (),
    closes: dict[str, float] | None = None,
    nan_close_at: set[str] | None = None,
    qv_nan_from: str | None = None,
    funding_rate: float = 0.0,
    funding_unknown_symbols: tuple[str, ...] = (),
) -> ExecutionReplayWindow:
    cols = ("AUSDT", "BUSDT")
    px = pd.DataFrame({s: np.full(len(grid), 100.0) for s in cols}, index=grid)
    if closes:
        for s, v in closes.items():
            px[s] = float(v)
    if nan_close_at:
        for stamp in nan_close_at:
            px.loc[pd.Timestamp(stamp, tz="UTC")] = np.nan
    qv = pd.DataFrame({s: np.full(len(grid), 1000.0) for s in cols}, index=grid)
    if qv_nan_from is not None:
        cut = pd.Timestamp(qv_nan_from, tz="UTC")
        qv.loc[grid >= cut] = np.nan
    known = pd.DataFrame(True, index=grid, columns=list(cols))
    for s in funding_unknown_symbols:
        known[s] = False
    fund = pd.DataFrame({s: np.full(len(grid), funding_rate) for s in cols}, index=grid)
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in decisions])
    w = pd.DataFrame(0.0, index=idx, columns=list(cols))
    for s, vals in weights.items():
        w[s] = vals
    return ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1] + pd.Timedelta(minutes=3),
        columns=cols, symbols=cols, minute_grid=grid,
        highs=px, lows=px, closes=px, marks=px, bar_funding=fund,
        target_weights=w, signal_available_at=idx, quote_volumes=qv,
        funding_known=known, bar_available_at=grid + pd.Timedelta(minutes=3),
        settlement_events=events,
    )


def _grid(start: str = "2022-01-14 00:00", periods: int = 60) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="3min", tz="UTC")


def test_held_long_settled_at_evidenced_price() -> None:
    grid = _grid(periods=40)
    ev = _event(price=2.0)
    w = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]}, events=(ev,),
                closes={"AUSDT": 100.0, "BUSDT": 100.0})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    fills = result.simulated_fills
    settle = fills[fills["reason"] == "delist_settlement"]
    assert len(settle) == 1
    assert float(settle.iloc[0]["quantity_delta"]) == pytest.approx(-float(fills.iloc[0]["quantity_delta"]), rel=1e-9)
    assert float(settle.iloc[0]["fill_price"]) == pytest.approx(2.0)
    assert result.terminal_positions[-1].status == "settled"
    assert result.ledger.primary_valid
    assert result.settlement_events == (ev,)


def test_decision_after_delivery_sizes_on_settled_cash() -> None:
    grid = _grid(periods=60)
    ev = _event(delivery=str(grid[30]), last_trade=str(grid[29]))
    w = _window(grid, [str(grid[0]), str(grid[40])], {"AUSDT": [0.5, 0.0], "BUSDT": [0.0, 0.5]},
                events=(ev,), closes={"AUSDT": 100.0, "BUSDT": 100.0})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    fills = result.simulated_fills
    settle_ts = fills[fills["reason"] == "delist_settlement"].iloc[0]["timestamp"]
    later = fills[fills["timestamp"] > settle_ts]
    assert len(later) >= 1
    assert later.iloc[0]["pre_trade_equity"] == pytest.approx(result.ledger.equity.loc[grid[40]])
    assert result.termination_counts.get("DELIST_SETTLEMENT") == 1


def test_no_funding_after_delivery() -> None:
    grid = _grid(periods=50)
    ev = _event(delivery=str(grid[20]), last_trade=str(grid[19]))
    w = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]}, events=(ev,),
                closes={"AUSDT": 100.0, "BUSDT": 100.0}, funding_rate=0.001)
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    funding = result.ledger.funding_charge
    after = funding.loc[funding.index > pd.Timestamp(str(grid[20]), tz="UTC")]
    assert float(after.abs().sum()) == pytest.approx(0.0, abs=1e-9)


def test_order_reaching_last_trade_is_cancelled() -> None:
    grid = _grid(periods=60)
    ev = _event(delivery=str(grid[40]), last_trade=str(grid[30]))
    w = _window(grid, [str(grid[0]), str(grid[20])], {"AUSDT": [0.5, 0.0], "BUSDT": [0.0, 0.0]},
                events=(ev,), closes={"AUSDT": 100.0, "BUSDT": 100.0})
    spec = dataclasses.replace(ExecutionSpec(), passive_timeout_minutes=30)
    result = replay_execution_windows((w,), 10000.0, "OHLCV_STRICT_PROXY", spec)
    assert result.termination_counts.get("CANCELLED_AT_DELIVERY", 0) >= 1
    assert result.termination_counts.get("DELIST_SETTLEMENT") == 1


def test_order_submitted_after_last_trade_cancelled_before_data_checks() -> None:
    grid = _grid(periods=60)
    ev = _event(delivery=str(grid[30]), last_trade=str(grid[20]))
    w = _window(grid, [str(grid[0]), str(grid[40])], {"AUSDT": [0.5, 0.5], "BUSDT": [0.0, 0.0]},
                events=(ev,), closes={"AUSDT": 100.0, "BUSDT": 100.0},
                qv_nan_from=str(grid[21]))
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert not any(g.code == "MISSING_ACTIVE_ORDER_OHLCV" for g in result.data_gaps)
    assert result.termination_counts.get("CANCELLED_AT_DELIVERY", 0) >= 1


def test_passive_fill_before_last_trade_still_books() -> None:
    grid = _grid(periods=60)
    ev = _event(delivery=str(grid[50]), last_trade=str(grid[40]))
    w = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]},
                events=(ev,), closes={"AUSDT": 100.0, "BUSDT": 100.0})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_TOUCH_PROXY", ExecutionSpec())
    assert "passive_fill" in list(result.simulated_fills["reason"])


def test_noop_event_leaves_no_trace() -> None:
    grid = _grid(periods=40)
    ev = _event(symbol="BUSDT", price=3.0)
    w = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]}, events=(ev,),
                closes={"AUSDT": 100.0, "BUSDT": 100.0})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("DELIST_SETTLEMENT_NOOP") == 1
    assert result.settlement_events == ()
    assert not any(t.symbol == "BUSDT" and t.status == "settled" for t in result.terminal_positions)


def test_idempotent_across_overlapping_windows() -> None:
    import dataclasses as _dc2

    grid = _grid(periods=40)
    grid_b = pd.date_range("2022-01-14 01:00", periods=40, freq="3min", tz="UTC")
    ev = _event(price=2.0)
    w1 = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]}, events=(ev,),
                 closes={"AUSDT": 100.0, "BUSDT": 100.0})
    w2base = _window(grid_b, [], {"AUSDT": [], "BUSDT": []}, events=(ev,),
                 closes={"AUSDT": 100.0, "BUSDT": 100.0})
    w2 = _dc2.replace(w2base, target_weights=w2base.target_weights.iloc[0:0], signal_available_at=w2base.signal_available_at[0:0])
    result = replay_execution_windows((w1, w2), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("DELIST_SETTLEMENT") == 1


@pytest.mark.parametrize("advance_decision", [False, True])
def test_late_admission_fails_closed(advance_decision) -> None:
    grid1 = _grid("2022-01-14 00:00", periods=40)
    grid2 = _grid("2022-01-14 02:00", periods=40)
    ev = _event(delivery=str(grid1[10]), last_trade=str(grid1[9]))
    w1 = _window(grid1, [str(grid1[0]), str(grid1[30])], {"AUSDT": [0.5, 0.5], "BUSDT": [0.0, 0.0]},
                 closes={"AUSDT": 100.0, "BUSDT": 100.0})
    if not advance_decision:
        w1 = dataclasses.replace(w1, target_weights=w1.target_weights.iloc[:1], signal_available_at=w1.signal_available_at[:1])
    w2 = _window(grid2, [str(grid2[5])], {"AUSDT": [0.5], "BUSDT": [0.0]}, events=(ev,),
                 closes={"AUSDT": 100.0, "BUSDT": 100.0})
    with pytest.raises(DataIntegrityError, match="admitted after its due time"):
        replay_execution_windows((w1, w2), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())


def test_missing_delivery_bar_close_does_not_invalidate() -> None:
    grid = _grid(periods=40)
    ev = _event(price=2.0, delivery=str(grid[25]), last_trade=str(grid[24]))
    w = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]}, events=(ev,),
                closes={"AUSDT": 100.0, "BUSDT": 100.0}, nan_close_at={str(grid[25])})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert not any(g.code == "MISSING_HELD_MARK" and g.symbol == "AUSDT" for g in result.data_gaps)
    assert result.ledger.primary_valid


def test_unsettled_after_last_trade_is_uncertified() -> None:
    grid = _grid(periods=30)
    ev = _event(delivery=str(grid[-1] + pd.Timedelta(minutes=60)), last_trade=str(grid[10]))
    w = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]}, events=(ev,),
                closes={"AUSDT": 100.0, "BUSDT": 100.0})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert any(g.code == "UNSETTLED_DELIVERY" for g in result.data_gaps)
    assert any(t.status == "unresolved" for t in result.terminal_positions if abs(t.quantity) > 0)
    assert not result.ledger.primary_valid


def test_every_held_unknown_funding_symbol_recorded() -> None:
    import dataclasses as _dc

    grid = _grid(periods=30)
    w = _window(grid, [str(grid[0])], {"AUSDT": [0.3], "BUSDT": [0.3]},
                closes={"AUSDT": 100.0, "BUSDT": 100.0},
                funding_rate=0.001)
    known = w.funding_known.copy()
    known.iloc[10:] = False
    w = _dc.replace(w, funding_known=known)
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    symbols = {g.symbol for g in result.data_gaps if g.code == "MISSING_HELD_FUNDING"}
    assert {"AUSDT", "BUSDT"} <= symbols
    incomplete = {t.symbol for t in result.terminal_positions if not t.funding_complete}
    assert {"AUSDT", "BUSDT"} <= incomplete


def test_stress_haircut_only_on_proxies() -> None:
    grid = _grid(periods=40)
    proxy = _event(symbol="AUSDT", price=100.0, source="twap30_proxy")
    curated = _event(symbol="BUSDT", price=100.0, source="curated")
    w = _window(grid, [str(grid[0])], {"AUSDT": [0.2], "BUSDT": [0.2]}, events=(proxy, curated),
                closes={"AUSDT": 100.0, "BUSDT": 100.0})
    base = ExecutionSpec()
    stress = dataclasses.replace(ExecutionSpec(), settlement_price_haircut_bps=450.0)
    results = replay_execution_window_batch(
        (w,), 10000.0,
        [( "OHLCV_IMMEDIATE_TAKER", base), ("OHLCV_IMMEDIATE_TAKER", stress)],
    )
    base_fills = results[0].simulated_fills
    stress_fills = results[1].simulated_fills
    base_a = float(base_fills[base_fills["symbol"] == "AUSDT"].iloc[-1]["fill_price"])
    stress_a = float(stress_fills[stress_fills["symbol"] == "AUSDT"].iloc[-1]["fill_price"])
    base_b = float(base_fills[base_fills["symbol"] == "BUSDT"].iloc[-1]["fill_price"])
    stress_b = float(stress_fills[stress_fills["symbol"] == "BUSDT"].iloc[-1]["fill_price"])
    assert base_a == pytest.approx(100.0)
    assert stress_a == pytest.approx(95.5)
    assert base_b == pytest.approx(100.0)
    assert stress_b == pytest.approx(100.0)


def test_all_bounds_settle_on_same_bar() -> None:
    grid = _grid(periods=40)
    ev = _event(price=2.0)
    w = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]}, events=(ev,),
                closes={"AUSDT": 100.0, "BUSDT": 100.0})
    bounds = [
        ("OHLCV_IMMEDIATE_TAKER", ExecutionSpec()),
        ("OHLCV_STRICT_PROXY", ExecutionSpec()),
        ("OHLCV_TOUCH_PROXY", ExecutionSpec()),
        ("OHLCV_LADDERED_PROXY", ExecutionSpec()),
        ("OHLCV_PEG_CHASE_PROXY", ExecutionSpec()),
    ]
    results = replay_execution_window_batch((w,), 10000.0, bounds)  # type: ignore[arg-type]
    bars = set()
    for r in results:
        assert r is not None
        settle = r.simulated_fills[r.simulated_fills["reason"] == "delist_settlement"]
        assert len(settle) == 1
        bars.add(str(settle.iloc[0]["timestamp"]))
    assert len(bars) == 1


def test_empty_registry_is_bit_identical() -> None:
    grid = _grid(periods=20)
    w = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]},
                closes={"AUSDT": 100.0, "BUSDT": 100.0})
    r1 = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    r2 = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    pd.testing.assert_series_equal(r1.ledger.equity, r2.ledger.equity)


def test_oversubmit_after_last_grid_bar_is_cancelled() -> None:
    grid = _grid(periods=40)
    ev = _event(delivery=str(grid[20]), last_trade=str(grid[19]))
    w = _window(grid, [str(grid[-1])], {"AUSDT": [0.5], "BUSDT": [0.0]}, events=(ev,),
                closes={"AUSDT": 100.0, "BUSDT": 100.0})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("CANCELLED_AT_DELIVERY", 0) >= 1


def test_laddered_submit_after_last_trade_is_cancelled() -> None:
    grid = _grid(periods=60)
    ev = _event(delivery=str(grid[30]), last_trade=str(grid[20]))
    w = _window(grid, [str(grid[0]), str(grid[40])], {"AUSDT": [0.5, 0.5], "BUSDT": [0.0, 0.0]},
                events=(ev,), closes={"AUSDT": 100.0, "BUSDT": 100.0})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_LADDERED_PROXY", ExecutionSpec())
    assert result.termination_counts.get("CANCELLED_AT_DELIVERY", 0) >= 1


def test_laddered_timeout_past_last_trade_has_no_taker_fallback() -> None:
    grid = _grid(periods=60)
    ev = _event(delivery=str(grid[41]), last_trade=str(grid[40]))
    w = _window(grid, [str(grid[30])], {"AUSDT": [0.5], "BUSDT": [0.0]},
                events=(ev,), closes={"AUSDT": 100.0, "BUSDT": 100.0})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_LADDERED_PROXY", ExecutionSpec())
    fills = result.simulated_fills
    stamps = pd.to_datetime(fills["timestamp"], utc=True)
    assert not any(
        (fills["reason"] == "timeout_taker") & (stamps >= pd.Timestamp(str(grid[40]), tz="UTC"))
    )
    assert result.termination_counts.get("CANCELLED_AT_DELIVERY", 0) >= 1


def test_peg_submit_after_last_trade_is_cancelled() -> None:
    grid = _grid(periods=60)
    ev = _event(delivery=str(grid[30]), last_trade=str(grid[20]))
    w = _window(grid, [str(grid[0]), str(grid[40])], {"AUSDT": [0.5, 0.5], "BUSDT": [0.0, 0.0]},
                events=(ev,), closes={"AUSDT": 100.0, "BUSDT": 100.0})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_PEG_CHASE_PROXY", ExecutionSpec())
    assert result.termination_counts.get("CANCELLED_AT_DELIVERY", 0) >= 1


def test_peg_timeout_past_last_trade_has_no_taker_fallback() -> None:
    grid = _grid(periods=60)
    ev = _event(delivery=str(grid[41]), last_trade=str(grid[40]))
    w = _window(grid, [str(grid[36])], {"AUSDT": [0.5], "BUSDT": [0.0]},
                events=(ev,), closes={"AUSDT": 100.0, "BUSDT": 100.0})
    result = replay_execution_windows((w,), 10000.0, "OHLCV_PEG_CHASE_PROXY", ExecutionSpec())
    fills = result.simulated_fills
    stamps = pd.to_datetime(fills["timestamp"], utc=True)
    assert not any(
        (fills["reason"] == "timeout_taker") & (stamps >= pd.Timestamp(str(grid[40]), tz="UTC"))
    )
    assert result.termination_counts.get("CANCELLED_AT_DELIVERY", 0) >= 1


def test_interleaved_funding_reconciles_decision_cash_and_delivery_valuation() -> None:
    grid = _grid(periods=60)
    events = tuple(_event(symbol=symbol, delivery=str(grid[20]), last_trade=str(grid[19]), price=80.0, source="twap30_proxy") for symbol in ("AUSDT", "BUSDT"))
    window = _window(grid, [str(grid[0]), str(grid[40])], {"AUSDT": [0.2, 0.0], "BUSDT": [0.2, 0.0]}, events=events, funding_rate=0.001)
    spec = ExecutionSpec(settlement_price_haircut_bps=450.0)
    live: list = []
    result = replay_execution_windows((window,), 10000.0, "OHLCV_IMMEDIATE_TAKER", spec, live_accumulators=live)
    accumulator = live[0][0]
    assert accumulator.cash == pytest.approx(result.ledger.equity.iloc[-1], abs=1e-9)
    assert result.ledger.funding_charge.loc[grid[20]] == pytest.approx(40.0 * 76.4 * 0.001)
    assert result.ledger.funding_charge.loc[grid[21]:].sum() == 0.0


@pytest.mark.parametrize("bound", ["OHLCV_STRICT_PROXY", "OHLCV_LADDERED_PROXY"])
def test_timeout_equal_last_trade_cancels_before_dead_close(bound) -> None:
    grid = _grid(periods=60)
    event = _event(delivery=str(grid[35]), last_trade=str(grid[11]))
    window = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]}, events=(event,))
    closes = window.closes.copy()
    closes.loc[grid[11]:, "AUSDT"] = np.nan
    window = dataclasses.replace(window, closes=closes)
    result = replay_execution_windows((window,), 10000.0, bound, ExecutionSpec())
    assert result.simulated_fills.empty
    assert result.data_gaps == ()
    assert result.termination_counts["CANCELLED_AT_DELIVERY"] == 1


def test_zero_rate_unknown_funding_marks_every_held_terminal_incomplete() -> None:
    grid = _grid(periods=30)
    window = _window(grid, [str(grid[0])], {"AUSDT": [0.2], "BUSDT": [0.2]})
    known = window.funding_known.copy()
    known.iloc[10:] = False
    result = replay_execution_windows((dataclasses.replace(window, funding_known=known),), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert {position.symbol for position in result.terminal_positions if not position.funding_complete} == {"AUSDT", "BUSDT"}


@pytest.mark.parametrize("bound", ["OHLCV_LADDERED_PROXY", "OHLCV_PEG_CHASE_PROXY"])
def test_future_lifecycle_changes_preserve_earlier_tranche_clock(bound) -> None:
    grid = _grid(periods=30)
    events = [_event(delivery=str(grid[due]), last_trade=str(grid[last]), price=price) for due, last, price in [(15, 5, 80.0), (18, 7, 90.0)]]
    window = _window(grid, [str(grid[0])], {"AUSDT": [0.5], "BUSDT": [0.0]})
    closes = window.closes.copy()
    closes.loc[grid[1]:, "AUSDT"] = 110.0
    window = dataclasses.replace(window, closes=closes)
    spec = ExecutionSpec(peg_chase_tranches=3)
    results = [replay_execution_windows((dataclasses.replace(window, settlement_events=(event,)),), 10000.0, bound, spec) for event in events]
    cutoff = grid[4]
    pd.testing.assert_frame_equal(
        results[0].simulated_fills.loc[lambda frame: frame.timestamp <= cutoff].reset_index(drop=True),
        results[1].simulated_fills.loc[lambda frame: frame.timestamp <= cutoff].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(results[0].ledger.equity.loc[:cutoff], results[1].ledger.equity.loc[:cutoff])


def test_conflicting_overlap_event_fails_closed() -> None:
    grid = _grid(periods=40)
    event = _event()
    window = _window(grid, [str(grid[0])], {"AUSDT": [0.5]}, events=(event, dataclasses.replace(event, settlement_price=3.0)))
    with pytest.raises(DataIntegrityError, match="conflicting settlement"):
        replay_execution_windows((window,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())


def _far_event(announced: str, base: pd.Timestamp) -> InstrumentSettlementEvent:
    far = base + pd.Timedelta(days=100)
    return _event(
        delivery=str(far), last_trade=str(far), announced=announced,
    )


def test_post_announcement_reentry_blocked_in_replay() -> None:
    grid = _grid(periods=60)
    ev = _far_event(str(grid[15]), grid[0])
    w = _window(
        grid, [str(grid[0]), str(grid[10]), str(grid[20])],
        {"AUSDT": [0.5, 0.0, 0.5], "BUSDT": [0.0, 0.0, 0.0]},
        events=(ev,), closes={"AUSDT": 100.0, "BUSDT": 100.0},
    )
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("DELISTING_ENTRY_BLOCKED", 0) >= 1
    exit_ts = result.simulated_fills[
        (result.simulated_fills["symbol"] == "AUSDT")
        & (result.simulated_fills["quantity_delta"] < 0)
    ].iloc[0]["timestamp"]
    later_entries = result.simulated_fills[
        (result.simulated_fills["symbol"] == "AUSDT")
        & (result.simulated_fills["quantity_delta"] > 0)
        & (result.simulated_fills["timestamp"] > exit_ts)
    ]
    assert later_entries.empty


def test_forced_exit_before_delivery() -> None:
    grid = _grid(periods=1600)
    announced = str(grid[100])
    delivery = grid[1550]
    ev = _event(delivery=str(delivery), last_trade=str(delivery), announced=announced)
    decisions = [str(grid[0]), str(grid[500]), str(grid[1000])]
    w = _window(
        grid, decisions,
        {"AUSDT": [0.5, 0.5, 0.5], "BUSDT": [0.0, 0.0, 0.0]},
        events=(ev,), closes={"AUSDT": 100.0, "BUSDT": 100.0},
    )
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("DELISTING_FORCED_EXIT", 0) >= 1
    exits = result.simulated_fills[
        (result.simulated_fills["symbol"] == "AUSDT")
        & (result.simulated_fills["quantity_delta"] < 0)
        & (result.simulated_fills["reason"] != "delist_settlement")
    ]
    assert len(exits) == 1
    assert pd.Timestamp(exits.iloc[0]["timestamp"]) < pd.Timestamp(str(delivery))
    assert result.termination_counts.get("DELIST_SETTLEMENT", 0) in (None, 0)


def test_nan_hold_overridden_by_forced_exit() -> None:
    grid = _grid(periods=1600)
    announced = str(grid[100])
    delivery = grid[1550]
    ev = _event(delivery=str(delivery), last_trade=str(delivery), announced=announced)
    decisions = [str(grid[0]), str(grid[500])]
    w = _window(
        grid, decisions,
        {"AUSDT": [0.5, float("nan")], "BUSDT": [0.0, 0.0]},
        events=(ev,), closes={"AUSDT": 100.0, "BUSDT": 100.0},
    )
    result = replay_execution_windows((w,), 10000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert result.termination_counts.get("DELISTING_FORCED_EXIT", 0) >= 1
    exits = result.simulated_fills[
        (result.simulated_fills["symbol"] == "AUSDT")
        & (result.simulated_fills["quantity_delta"] < 0)
    ]
    assert len(exits) == 1


def _i5_pair() -> tuple:
    grid = _grid(periods=60)
    delivery = grid[0] + pd.Timedelta(days=100)
    decisions = [str(grid[0]), str(grid[20]), str(grid[45])]
    weights = {"AUSDT": [0.5, 0.5, 0.8], "BUSDT": [0.0, 0.0, 0.0]}
    closes = {"AUSDT": 100.0, "BUSDT": 100.0}
    ev_a = _event(delivery=str(delivery), last_trade=str(delivery), announced=str(grid[30]))
    ev_b = _event(delivery=str(delivery), last_trade=str(delivery), announced=str(grid[50]))
    w_a = _window(grid, decisions, weights, events=(ev_a,), closes=closes)
    w_b = _window(grid, decisions, weights, events=(ev_b,), closes=closes)
    return grid, w_a, w_b


def test_announcement_perturbation_after_T_invisible() -> None:
    grid, w_a, w_b = _i5_pair()
    bound: str = "OHLCV_IMMEDIATE_TAKER"
    r_a = replay_execution_windows((w_a,), 10000.0, bound, ExecutionSpec(), retain_event_snapshots=True)  # type: ignore[arg-type]
    r_b = replay_execution_windows((w_b,), 10000.0, bound, ExecutionSpec(), retain_event_snapshots=True)  # type: ignore[arg-type]
    cutoff = grid[25]
    early_a = r_a.simulated_fills[pd.to_datetime(r_a.simulated_fills["timestamp"], utc=True) <= cutoff]
    early_b = r_b.simulated_fills[pd.to_datetime(r_b.simulated_fills["timestamp"], utc=True) <= cutoff]
    pd.testing.assert_frame_equal(
        early_a.reset_index(drop=True), early_b.reset_index(drop=True),
    )
    pd.testing.assert_series_equal(r_a.ledger.equity.loc[:cutoff], r_b.ledger.equity.loc[:cutoff])
    pd.testing.assert_frame_equal(r_a.simulated_units.loc[:cutoff], r_b.simulated_units.loc[:cutoff])
    pd.testing.assert_series_equal(r_a.ledger.fee_charge.loc[:cutoff], r_b.ledger.fee_charge.loc[:cutoff])
    pd.testing.assert_series_equal(r_a.ledger.funding_charge.loc[:cutoff], r_b.ledger.funding_charge.loc[:cutoff])
    assert [g for g in r_a.data_gaps if g.timestamp <= cutoff] == [g for g in r_b.data_gaps if g.timestamp <= cutoff]


def test_announcement_difference_starts_at_first_informed_decision() -> None:
    grid, w_a, w_b = _i5_pair()
    bound: str = "OHLCV_IMMEDIATE_TAKER"
    r_a = replay_execution_windows((w_a,), 10000.0, bound, ExecutionSpec())  # type: ignore[arg-type]
    r_b = replay_execution_windows((w_b,), 10000.0, bound, ExecutionSpec())  # type: ignore[arg-type]
    assert r_a.termination_counts.get("DELISTING_ENTRY_BLOCKED", 0) >= 1
    assert r_b.termination_counts.get("DELISTING_ENTRY_BLOCKED", 0) in (None, 0)
    assert len(r_a.simulated_fills) + 1 == len(r_b.simulated_fills)
    first_new = r_b.simulated_fills[~r_b.simulated_fills["timestamp"].isin(r_a.simulated_fills["timestamp"])]
    assert first_new["timestamp"].tolist() == [grid[47]]
