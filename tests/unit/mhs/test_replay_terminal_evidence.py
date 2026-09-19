"""Terminal evidence, evidenced settlement, and strict ledger certification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, InstrumentSettlementEvent
from src.mhs.execution.contracts import (
    FundingKnowledgeObservation,
    TerminalPositionEvidence,
    align_funding_with_knowledge,
)
from src.mhs.execution.integrity import replay_ledger_certified
from src.mhs.execution import replay_execution_windows

SPEC = ExecutionSpec()
STEP = pd.Timedelta(minutes=3)
SYMBOLS = ("AUSDT", "BUSDT")


def _grid(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="3min", tz="UTC")


def _window(
    grid: pd.DatetimeIndex,
    weights: pd.DataFrame,
    signals: pd.DatetimeIndex,
    *,
    marks: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    funding_known: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    settlement_events: tuple[InstrumentSettlementEvent, ...] = (),
    bar_available_at: pd.DatetimeIndex | str | None = "auto",
) -> ExecutionReplayWindow:
    symbols = tuple(weights.columns)
    closes = pd.DataFrame(100.0, index=grid, columns=list(symbols))
    panel = pd.DataFrame(100.0, index=grid, columns=list(symbols)) if marks is None else marks
    funding_panel = pd.DataFrame(0.0, index=grid, columns=list(symbols)) if funding is None else funding
    known_panel = (
        pd.DataFrame(True, index=grid, columns=list(symbols)) if funding_known is None else funding_known
    ).astype(bool)
    volume_panel = pd.DataFrame(1000.0, index=grid, columns=list(symbols)) if volumes is None else volumes
    if bar_available_at == "auto":
        availability: pd.DatetimeIndex | None = grid + STEP
    elif bar_available_at is None:
        availability = None
    else:
        availability = bar_available_at  # type: ignore[assignment]
    return ExecutionReplayWindow(
        window_start=grid[0],
        window_end=grid[-1] + STEP,
        columns=tuple(weights.columns),
        symbols=symbols,
        minute_grid=grid,
        highs=closes * 1.001,
        lows=closes * 0.999,
        closes=closes,
        marks=panel,
        bar_funding=funding_panel,
        target_weights=weights,
        signal_available_at=signals,
        quote_volumes=volume_panel,
        funding_known=known_panel,
        bar_available_at=availability,
        settlement_events=settlement_events,
    )


def _open_weights(grid: pd.DatetimeIndex, **active: float) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    weights = pd.DataFrame(0.0, index=grid[:1], columns=list(SYMBOLS))
    for symbol, weight in active.items():
        weights.loc[grid[0], symbol] = weight
    return weights, pd.DatetimeIndex([grid[1]])


def _run(windows: list[ExecutionReplayWindow], equity: float = 10000.0):
    return replay_execution_windows(windows, equity, "OHLCV_STRICT_PROXY", SPEC)


def _event(
    event_id: str,
    symbol: str,
    when: pd.Timestamp,
    price: float,
    fee_bps: float = 5.0,
    available: pd.Timestamp | None = None,
) -> InstrumentSettlementEvent:
    return InstrumentSettlementEvent(
        event_id=event_id,
        symbol=symbol,
        effective_at=when,
        available_at=when if available is None else available,
        settlement_price=price,
        fee_bps=fee_bps,
        source_digest="exchange-settlement-v1",
    )


def test_settlement_without_source_identity_is_rejected() -> None:
    """Typed settlement source: an event without price/fee/source identity is rejected."""
    grid = _grid(40)
    stamp = grid[10]
    with pytest.raises(DataIntegrityError):
        _event("bad-price", "AUSDT", stamp, float("nan"))
    with pytest.raises(DataIntegrityError):
        InstrumentSettlementEvent(
            event_id="bad-source",
            symbol="AUSDT",
            effective_at=stamp,
            available_at=stamp,
            settlement_price=100.0,
            fee_bps=0.0,
            source_digest="",
        )
    weights, signals = _open_weights(grid, AUSDT=0.05)
    window = _window(grid, weights, signals, settlement_events=("not-an-event",))  # type: ignore[list-item]
    with pytest.raises(DataIntegrityError):
        _run([window])
    unknown = _event("unknown-symbol", "ZZZUSDT", stamp, 100.0)
    with pytest.raises(DataIntegrityError):
        _run([_window(grid, weights, signals, settlement_events=(unknown,))])


def test_settlement_before_publication_books_nothing() -> None:
    """Late settlement publication: no fill or cash revision appears before publication."""
    grid = _grid(40)
    weights, signals = _open_weights(grid, AUSDT=0.05)
    event = _event("late-1", "AUSDT", grid[5], 110.0, available=grid[30])
    prefix = _run([_window(grid[:20], weights, signals, settlement_events=(event,))])
    assert not (prefix.simulated_fills["reason"] == "delist_settlement").any()
    assert abs(float(prefix.terminal_positions[0].quantity)) > 0.0
    baseline = _run([_window(grid[:20], weights, signals)])
    np.testing.assert_allclose(
        prefix.ledger.equity.to_numpy(), baseline.ledger.equity.to_numpy(), rtol=1e-12, atol=1e-12,
    )
    assert len(prefix.simulated_fills) == len(baseline.simulated_fills)
    full = _run([_window(grid, weights, signals, settlement_events=(event,))])
    assert (full.simulated_fills["reason"] == "delist_settlement").sum() == 1
    assert all(abs(float(q)) < 1e-12 for q in [p.quantity for p in full.terminal_positions if p.symbol == "AUSDT" and p.status == "settled"])


def test_repeated_settlement_event_books_once() -> None:
    """Overlap event identity: one event in two overlapping chunks closes once."""
    grid = _grid(40)
    weights, signals = _open_weights(grid, AUSDT=0.05)
    event = _event("overlap-1", "AUSDT", grid[10], 110.0)
    first = _window(grid[:21], weights, signals, settlement_events=(event,))
    empty_weights = pd.DataFrame(0.0, index=grid[0:0], columns=list(SYMBOLS))
    second = _window(
        grid[20:], empty_weights, pd.DatetimeIndex([]), settlement_events=(event,),
    )
    result = _run([first, second])
    fills = result.simulated_fills[result.simulated_fills["reason"] == "delist_settlement"]
    assert len(fills) == 1
    assert float(fills["fill_price"].iloc[0]) == pytest.approx(110.0)
    assert not any(p.status == "open_marked" and p.symbol == "AUSDT" for p in result.terminal_positions)


def test_open_cutoff_keeps_units_without_exit_fill() -> None:
    """Normal open cutoff: equity reconciles, units stay open, no exit fee appears."""
    grid = _grid(40)
    weights, signals = _open_weights(grid, AUSDT=0.05, BUSDT=-0.05)
    result = _run([_window(grid, weights, signals)])
    assert not (result.simulated_fills["reason"] == "delist_settlement").any()
    assert len(result.simulated_fills) == 2
    assert float(result.ledger.fee_charge.sum()) == pytest.approx(0.2)
    assert float(result.ledger.funding_charge.sum()) == pytest.approx(0.0)
    cash = 10000.0 - 500.0 - 0.1 + 500.0 - 0.1
    assert float(result.ledger.equity.iloc[-1]) == pytest.approx(cash)
    by_symbol = {p.symbol: p for p in result.terminal_positions}
    assert set(by_symbol) == {"AUSDT", "BUSDT"}
    assert by_symbol["AUSDT"].status == "open_marked"
    assert by_symbol["BUSDT"].status == "open_marked"
    assert by_symbol["AUSDT"].quantity == pytest.approx(5.0)
    assert by_symbol["BUSDT"].quantity == pytest.approx(-5.0)
    assert result.ledger.primary_valid is True
    assert replay_ledger_certified(result) is True
    assert result.ledger_available_at is not None
    assert len(result.ledger_available_at) == len(result.ledger.equity.index)
    assert bool((result.ledger_available_at >= result.ledger.equity.index).all())
    assert result.ledger_available_at[0] == grid[0] + STEP


def test_stale_cutoff_mark_is_unresolved() -> None:
    """Stale terminal mark: valuation is unresolved and accounting is invalid."""
    grid = _grid(40)
    weights, signals = _open_weights(grid, AUSDT=0.05)
    marks = pd.DataFrame(100.0, index=grid, columns=list(SYMBOLS))
    marks.loc[grid[-6]:, "AUSDT"] = np.nan
    result = _run([_window(grid, weights, signals, marks=marks)])
    by_symbol = {p.symbol: p for p in result.terminal_positions}
    assert by_symbol["AUSDT"].status == "unresolved"
    assert result.ledger.primary_valid is False
    assert any(g.code == "MISSING_HELD_MARK" for g in result.ledger.data_gaps)
    assert replay_ledger_certified(result) is False


def test_unknown_held_funding_stays_invalid() -> None:
    """Terminal unknown funding: missing financing is invalid, never terminal-certified."""
    grid = _grid(40)
    weights, signals = _open_weights(grid, AUSDT=0.05)
    funding = pd.DataFrame(1.0e-4, index=grid, columns=list(SYMBOLS))
    known = pd.DataFrame(True, index=grid, columns=list(SYMBOLS))
    known.loc[grid[4]:, "AUSDT"] = False
    result = _run([_window(grid, weights, signals, funding=funding, funding_known=known)])
    assert any(g.code == "MISSING_HELD_FUNDING" and g.symbol == "AUSDT" for g in result.ledger.data_gaps)
    assert result.ledger.primary_valid is False
    by_symbol = {p.symbol: p for p in result.terminal_positions}
    assert by_symbol["AUSDT"].funding_complete is False
    assert replay_ledger_certified(result) is False


def test_idle_zero_volume_never_settles() -> None:
    """Idle does not settle: prolonged zero volume books no free settlement."""
    grid = _grid(521)
    weights = pd.DataFrame(0.0, index=pd.DatetimeIndex([grid[0], grid[500]]), columns=list(SYMBOLS))
    weights.loc[grid[0], "AUSDT"] = 0.05
    weights.loc[grid[500], "AUSDT"] = 0.05
    signals = pd.DatetimeIndex([grid[1], grid[501]])
    volumes = pd.DataFrame(1000.0, index=grid, columns=list(SYMBOLS))
    volumes.loc[grid[40]:] = 0.0
    result = _run([_window(grid, weights, signals, volumes=volumes)])
    assert not (result.simulated_fills["reason"] == "delist_settlement").any()
    by_symbol = {p.symbol: p for p in result.terminal_positions}
    assert by_symbol["AUSDT"].status == "open_marked"
    assert by_symbol["AUSDT"].quantity == pytest.approx(5.0)


def test_settlement_cash_flows_match_hand_calculation() -> None:
    """Hand-computed settlement: exact cash, units, price PnL and fee reconciliation."""
    grid = _grid(40)
    weights, signals = _open_weights(grid, AUSDT=0.10, BUSDT=-0.10)
    events = (_event("hand-A", "AUSDT", grid[10], 110.0), _event("hand-B", "BUSDT", grid[10], 90.0))
    result = _run([_window(grid, weights, signals, settlement_events=events)])
    assert float(result.ledger.equity.iloc[-1]) == pytest.approx(10198.6)
    assert float(result.ledger.fee_charge.sum()) == pytest.approx(1.4)
    assert float(result.ledger.funding_charge.sum()) == pytest.approx(0.0)
    fills = result.simulated_fills[result.simulated_fills["reason"] == "delist_settlement"]
    assert len(fills) == 2
    assert float(fills[fills["symbol"] == "AUSDT"]["fill_price"].iloc[0]) == pytest.approx(110.0)
    assert float(fills[fills["symbol"] == "BUSDT"]["fill_price"].iloc[0]) == pytest.approx(90.0)
    settled = {p.symbol: p for p in result.terminal_positions if p.status == "settled"}
    assert set(settled) == {"AUSDT", "BUSDT"}
    assert all(p.quantity == pytest.approx(0.0) for p in settled.values())
    assert result.ledger.primary_valid is True
    assert replay_ledger_certified(result) is True


def test_physical_split_preserves_terminal_state() -> None:
    """Chunk-independent terminal state: physical boundaries never finalize finance."""
    grid = _grid(60)
    weights, signals = _open_weights(grid, AUSDT=0.05, BUSDT=-0.05)
    single = _run([_window(grid, weights, signals)])
    first = _window(grid[:30], weights, signals)
    empty_weights = pd.DataFrame(0.0, index=grid[0:0], columns=list(SYMBOLS))
    second = _window(grid[29:], empty_weights, pd.DatetimeIndex([]))
    split = _run([first, second])
    assert [(p.symbol, p.status) for p in single.terminal_positions] == [
        (p.symbol, p.status) for p in split.terminal_positions
    ]
    np.testing.assert_allclose(
        [p.quantity for p in single.terminal_positions],
        [p.quantity for p in split.terminal_positions],
        rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        single.ledger.equity.to_numpy(), split.ledger.equity.to_numpy(), rtol=1e-12, atol=1e-12,
    )
    assert float(single.ledger.fee_charge.sum()) == pytest.approx(float(split.ledger.fee_charge.sum()))
    assert float(single.ledger.funding_charge.sum()) == pytest.approx(float(split.ledger.funding_charge.sum()))
    assert single.ledger.primary_valid == split.ledger.primary_valid
    assert single.ledger_available_at is not None
    assert split.ledger_available_at is not None
    assert single.ledger_available_at.equals(split.ledger_available_at)


def test_settlement_defers_until_symbol_and_due_bar_are_reachable() -> None:
    """Deferred settlement: an event for an absent roster symbol waits without booking."""
    import dataclasses

    grid = _grid(40)
    weights = pd.DataFrame(0.0, index=grid[:1], columns=list(SYMBOLS))
    weights.loc[grid[0], "AUSDT"] = 0.05
    signals = pd.DatetimeIndex([grid[1]])
    event = _event("deferred-B", "BUSDT", grid[5], 90.0)
    first = _window(grid[:20], weights, signals, settlement_events=(event,))
    only_a = ("AUSDT",)
    first = dataclasses.replace(
        first,
        symbols=only_a,
        highs=first.highs.loc[:, list(only_a)],
        lows=first.lows.loc[:, list(only_a)],
        closes=first.closes.loc[:, list(only_a)],
        marks=first.marks.loc[:, list(only_a)] if first.marks is not None else None,
        bar_funding=first.bar_funding.loc[:, list(only_a)],
        target_weights=first.target_weights.loc[:, list(only_a)],
        quote_volumes=first.quote_volumes.loc[:, list(only_a)] if first.quote_volumes is not None else None,
        funding_known=first.funding_known.loc[:, list(only_a)] if first.funding_known is not None else None,
    )
    empty_weights = pd.DataFrame(0.0, index=grid[0:0], columns=list(SYMBOLS))
    second = _window(grid[19:], empty_weights, pd.DatetimeIndex([]), settlement_events=(event,))
    result = _run([first, second])
    assert not (result.simulated_fills["reason"] == "delist_settlement").any()
    settled = {p.symbol: p for p in result.terminal_positions if p.status == "settled"}
    assert set(settled) == {"BUSDT"}
    assert settled["BUSDT"].quantity == pytest.approx(0.0)
    by_symbol = {p.symbol: p for p in result.terminal_positions if p.status == "open_marked"}
    assert by_symbol["AUSDT"].quantity == pytest.approx(5.0)


def test_settled_without_funding_proof_never_certifies() -> None:
    """Observed settlement without historical financing proof never certifies."""
    from types import SimpleNamespace

    import pandas as pd

    stamp = pd.Timestamp("2024-01-01", tz="UTC")
    settled = TerminalPositionEvidence(
        symbol="AUSDT", quantity=0.0, cutoff=stamp, status="settled",
        mark=100.0, mark_available_at=stamp, funding_complete=False,
        reason_codes=("SETTLEMENT_EVENT",),
    )
    replay = SimpleNamespace(
        ledger=SimpleNamespace(primary_valid=True, invalid_reasons=(), data_gaps=()),
        simulated_fills=pd.DataFrame(),
        terminal_positions=(settled,),
    )
    assert replay_ledger_certified(replay) is False


def test_legacy_window_without_availability_yields_no_certified_timing() -> None:
    """Absent legacy timing cannot pass forward label certification."""
    grid = _grid(40)
    weights, signals = _open_weights(grid, AUSDT=0.05)
    result = _run([_window(grid, weights, signals, bar_available_at=None)])
    assert result.ledger_available_at is None
    assert {p.status for p in result.terminal_positions} == {"open_marked"}


def test_recorded_knowledge_gates_on_publication() -> None:
    """Recorded knowledge is admitted only when available at consumption time."""
    grid = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")
    series = pd.Series([0.0001, 0.0002], index=pd.DatetimeIndex([grid[0], grid[6]]))
    observations = (
        FundingKnowledgeObservation(
            symbol="AUSDT", event_time=grid[2], available_at=grid[4],
            status="no_settlement", source_digest="funding-source-v1",
        ),
        FundingKnowledgeObservation(
            symbol="AUSDT", event_time=grid[6], available_at=grid[10],
            status="settled", source_digest="funding-source-v1",
        ),
    )
    out = align_funding_with_knowledge(
        {"AUSDT": series}, grid, symbols=["AUSDT"], knowledge_observations=observations,
    )
    assert out.knowledge_source == "recorded"
    assert out.limitations == ()
    known = out.known["AUSDT"].to_numpy(dtype=bool)
    assert known[:4].tolist() == [False] * 4
    assert known[4:].tolist() == [True] * 8


def test_future_events_cannot_certify_earlier_rows() -> None:
    """Prefix-only knowledge: later funding events never rewrite earlier rows."""
    grid = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")
    prefix = grid[:6]
    truncated = pd.Series([0.0001], index=pd.DatetimeIndex([grid[0]]))
    extended = pd.Series([0.0001, 0.0005], index=pd.DatetimeIndex([grid[0], grid[-1]]))
    base = align_funding_with_knowledge({"AUSDT": truncated}, prefix, symbols=["AUSDT"])
    extended_out = align_funding_with_knowledge({"AUSDT": extended}, grid, symbols=["AUSDT"])
    np.testing.assert_array_equal(
        base.known["AUSDT"].to_numpy(dtype=bool),
        extended_out.known["AUSDT"].to_numpy(dtype=bool)[:6],
    )
    np.testing.assert_allclose(
        base.rates["AUSDT"].to_numpy(dtype="float64"),
        extended_out.rates["AUSDT"].to_numpy(dtype="float64")[:6],
    )
    assert base.limitations == extended_out.limitations
    observations = (
        FundingKnowledgeObservation(
            symbol="AUSDT", event_time=grid[2], available_at=grid[8],
            status="no_settlement", source_digest="funding-source-v1",
        ),
    )
    recorded_prefix = align_funding_with_knowledge(
        {"AUSDT": extended}, prefix, symbols=["AUSDT"], knowledge_observations=observations,
    )
    recorded_full = align_funding_with_knowledge(
        {"AUSDT": extended}, grid, symbols=["AUSDT"], knowledge_observations=observations,
    )
    np.testing.assert_array_equal(
        recorded_prefix.known["AUSDT"].to_numpy(dtype=bool),
        recorded_full.known["AUSDT"].to_numpy(dtype=bool)[:6],
    )
    assert recorded_prefix.known["AUSDT"].to_numpy(dtype=bool).tolist() == [False] * 6


def test_knowledge_source_marking_and_validation() -> None:
    """Unobserved knowledge stays a visible proxy; bad observations fail closed."""
    grid = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    series = pd.Series([0.0001], index=pd.DatetimeIndex([grid[0]]))
    proxy = align_funding_with_knowledge({"AUSDT": series}, grid, symbols=["AUSDT"])
    assert proxy.knowledge_source == "archive_recency_proxy"
    assert proxy.limitations != ()
    with pytest.raises(DataIntegrityError):
        FundingKnowledgeObservation(
            symbol="AUSDT", event_time=grid[2], available_at=grid[1],
            status="no_settlement", source_digest="funding-source-v1",
        )
    with pytest.raises(DataIntegrityError):
        align_funding_with_knowledge(
            {"AUSDT": series}, grid, symbols=["AUSDT"],
            knowledge_observations=("not-an-observation",),  # type: ignore[list-item]
        )


def test_contract_validation_rejects_bad_evidence() -> None:
    """Evidence contracts reject empty identities, bad prices, and naive times."""
    from zoneinfo import ZoneInfo

    grid = _grid(4)
    stamp = grid[0]
    naive = pd.Timestamp("2024-01-01")
    seoul = pd.Timestamp("2024-01-01", tz=ZoneInfo("Asia/Seoul"))
    with pytest.raises(DataIntegrityError):
        FundingKnowledgeObservation(
            symbol="", event_time=stamp, available_at=stamp,
            status="settled", source_digest="digest",
        )
    with pytest.raises(DataIntegrityError):
        FundingKnowledgeObservation(
            symbol="AUSDT", event_time=stamp, available_at=stamp,
            status="invented", source_digest="digest",  # type: ignore[arg-type]
        )
    with pytest.raises(DataIntegrityError):
        FundingKnowledgeObservation(
            symbol="AUSDT", event_time=stamp, available_at=stamp,
            status="settled", source_digest="",
        )
    with pytest.raises(DataIntegrityError):
        FundingKnowledgeObservation(
            symbol="AUSDT", event_time=naive, available_at=stamp,
            status="settled", source_digest="digest",
        )
    with pytest.raises(DataIntegrityError):
        FundingKnowledgeObservation(
            symbol="AUSDT", event_time="2024-01-01", available_at=stamp,  # type: ignore[arg-type]
            status="settled", source_digest="digest",
        )
    with pytest.raises(DataIntegrityError):
        FundingKnowledgeObservation(
            symbol="AUSDT", event_time=seoul, available_at=seoul + pd.Timedelta(hours=1),
            status="settled", source_digest="digest",
        )
    with pytest.raises(DataIntegrityError):
        _event("", "AUSDT", stamp, 100.0)
    with pytest.raises(DataIntegrityError):
        _event("symbol", "", stamp, 100.0)
    with pytest.raises(DataIntegrityError):
        InstrumentSettlementEvent(
            event_id="digest", symbol="AUSDT", effective_at=stamp, available_at=stamp,
            settlement_price=100.0, fee_bps=0.0, source_digest="",
        )
    with pytest.raises(DataIntegrityError):
        InstrumentSettlementEvent(
            event_id="time", symbol="AUSDT", effective_at="2024-01-01",  # type: ignore[arg-type]
            available_at=stamp, settlement_price=100.0, fee_bps=0.0, source_digest="digest",
        )
    with pytest.raises(DataIntegrityError):
        InstrumentSettlementEvent(
            event_id="time", symbol="AUSDT", effective_at=naive,
            available_at=stamp, settlement_price=100.0, fee_bps=0.0, source_digest="digest",
        )
    with pytest.raises(DataIntegrityError):
        InstrumentSettlementEvent(
            event_id="time", symbol="AUSDT", effective_at=seoul,
            available_at=seoul + pd.Timedelta(hours=1),
            settlement_price=100.0, fee_bps=0.0, source_digest="digest",
        )
    with pytest.raises(DataIntegrityError):
        _event("price", "AUSDT", stamp, 0.0)
    with pytest.raises(DataIntegrityError):
        _event("fee", "AUSDT", stamp, 100.0, fee_bps=float("nan"))
    with pytest.raises(DataIntegrityError):
        TerminalPositionEvidence(
            symbol="", quantity=1.0, cutoff=stamp, status="open_marked",
            mark=100.0, mark_available_at=stamp, funding_complete=True, reason_codes=(),
        )
    with pytest.raises(DataIntegrityError):
        TerminalPositionEvidence(
            symbol="AUSDT", quantity=float("nan"), cutoff=stamp, status="open_marked",
            mark=100.0, mark_available_at=stamp, funding_complete=True, reason_codes=(),
        )
    with pytest.raises(DataIntegrityError):
        TerminalPositionEvidence(
            symbol="AUSDT", quantity=1.0, cutoff=pd.NaT, status="open_marked",  # type: ignore[arg-type]
            mark=100.0, mark_available_at=stamp, funding_complete=True, reason_codes=(),
        )
    with pytest.raises(DataIntegrityError):
        TerminalPositionEvidence(
            symbol="AUSDT", quantity=1.0, cutoff=naive, status="open_marked",
            mark=100.0, mark_available_at=stamp, funding_complete=True, reason_codes=(),
        )
    with pytest.raises(DataIntegrityError):
        TerminalPositionEvidence(
            symbol="AUSDT", quantity=1.0, cutoff=stamp, status="exited",  # type: ignore[arg-type]
            mark=100.0, mark_available_at=stamp, funding_complete=True, reason_codes=(),
        )
    with pytest.raises(DataIntegrityError):
        TerminalPositionEvidence(
            symbol="AUSDT", quantity=1.0, cutoff=stamp, status="open_marked",
            mark=-5.0, mark_available_at=stamp, funding_complete=True, reason_codes=(),
        )
    with pytest.raises(DataIntegrityError):
        TerminalPositionEvidence(
            symbol="AUSDT", quantity=1.0, cutoff=stamp, status="open_marked",
            mark=100.0, mark_available_at=pd.NaT, funding_complete=True, reason_codes=(),  # type: ignore[arg-type]
        )
    with pytest.raises(DataIntegrityError):
        TerminalPositionEvidence(
            symbol="AUSDT", quantity=1.0, cutoff=stamp, status="open_marked",
            mark=100.0, mark_available_at=seoul, funding_complete=True, reason_codes=(),
        )
