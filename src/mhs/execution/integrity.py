"""Execution-ledger certification owned by the execution core."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.mhs.execution.contracts import ExecutionDataGap

_RECOVERABLE_HELD_GAP_CODES = frozenset({"MISSING_HELD_FUNDING", "MISSING_HELD_MARK"})


def _funding_gap_terminal_symbols(
    data_gaps: Sequence[ExecutionDataGap],
    simulated_fills: pd.DataFrame,
) -> frozenset[str]:
    """Post-hoc classification of terminal held-position gaps.

    This is a finalize-time classification only and is never fed back into any
    trading decision (INV-PIT-RESUME-CAUSAL). A later fill for the same symbol
    proves the position resumed normal trading, so that symbol's gap is NOT
    terminal-equivalent. A ``delist_settlement`` fill (the causal idle-holdings
    settlement, ``_settle_idle_holdings``) is excluded from that "later fill"
    evidence: it is itself the terminal disclosure closing out a position that
    could never resume normal trading, not proof that trading recovered.
    """
    missing_last: dict[str, pd.Timestamp] = {}
    for g in data_gaps:
        if g.code in _RECOVERABLE_HELD_GAP_CODES:
            prev = missing_last.get(g.symbol)
            if prev is None or g.timestamp > prev:
                missing_last[g.symbol] = g.timestamp
    if not missing_last:
        return frozenset()
    if "reason" in simulated_fills.columns:
        resumable_fills = simulated_fills[simulated_fills["reason"] != "delist_settlement"]
    else:
        resumable_fills = simulated_fills
    fill_symbols = resumable_fills["symbol"] if "symbol" in resumable_fills.columns else pd.Series(dtype="object")
    fill_ts = pd.to_datetime(resumable_fills["timestamp"], utc=True) if "timestamp" in resumable_fills.columns else pd.Series(dtype="datetime64[ns, UTC]")
    terminal: set[str] = set()
    for sym, last_ts in missing_last.items():
        if not bool(((fill_symbols == sym) & (fill_ts > last_ts)).any()):
            terminal.add(sym)
    return frozenset(terminal)


def ledger_terminal_only(
    data_gaps: Sequence[ExecutionDataGap],
    simulated_fills: pd.DataFrame,
) -> bool:
    """Certify a ledger whose gaps are all disclosed terminal inventory.

    Generalizes the existing UNKNOWN_TERMINATION-only exception to also accept
    a MISSING_HELD_FUNDING or MISSING_HELD_MARK episode that never recovers
    before the replay's own grid end -- symmetric with the pre-existing 'held
    to backtest end is disclosed evidence, not a crash' precedent; no
    fabricated settlement, no change to funding/mark accounting
    (INV-NO-FABRICATED-SETTLEMENT).
    """
    if not data_gaps:
        return False
    terminal_funding_symbols = _funding_gap_terminal_symbols(data_gaps, simulated_fills)
    return all(
        g.code == "UNKNOWN_TERMINATION"
        or (g.code in _RECOVERABLE_HELD_GAP_CODES and g.symbol in terminal_funding_symbols)
        for g in data_gaps
    )


def replay_ledger_certified(replay: Any) -> bool:
    """Single certification point for a replay's execution ledger.

    Returns True when the ledger was already marked ``primary_valid``;
    otherwise delegates unchanged to :func:`ledger_terminal_only`. Fails
    closed to False when the replay carries no ledger, no data gaps, or no
    fills. This is the only place the terminal-inventory exception is
    expressed; all consumers must call it instead of re-implementing the
    ``primary_valid or ledger_terminal_only(...)`` pair inline.
    """
    ledger = getattr(replay, "ledger", None)
    if getattr(ledger, "primary_valid", None) is True:
        return True
    gaps = getattr(ledger, "data_gaps", None)
    fills = getattr(replay, "simulated_fills", None)
    if gaps is None or fills is None:
        return False
    return bool(ledger_terminal_only(gaps, fills))


__all__ = ["_funding_gap_terminal_symbols", "ledger_terminal_only", "replay_ledger_certified"]
