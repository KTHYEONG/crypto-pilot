"""Execution-ledger certification owned by the execution core."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.mhs.execution.contracts import ExecutionDataGap, StrategyExecutionReplayResult

_RECOVERABLE_HELD_GAP_CODES = frozenset({"MISSING_HELD_FUNDING", "MISSING_HELD_MARK"})


def _funding_gap_terminal_symbols(
    data_gaps: Sequence[ExecutionDataGap],
    simulated_fills: pd.DataFrame,
) -> frozenset[str]:
    """Post-hoc classification of terminal held-position gaps.

    Historical classification utility only; it must never certify a ledger.
    Unknown financing or valuation is not rescued by a terminal-only label, a
    metadata span, or a later recovery event. This helper is retained for
    diagnostics and legacy-read parity, never as economic proof.
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
    """Classify whether gaps are all disclosed terminal inventory.

    Historical classification utility only; it must never certify a ledger.
    A terminal-only label cannot rescue unknown financing or valuation, and no
    certification path may call this helper as a validity exception.
    """
    if not data_gaps:
        return False
    terminal_funding_symbols = _funding_gap_terminal_symbols(data_gaps, simulated_fills)
    return all(
        g.code == "UNKNOWN_TERMINATION"
        or (g.code in _RECOVERABLE_HELD_GAP_CODES and g.symbol in terminal_funding_symbols)
        for g in data_gaps
    )


def replay_ledger_certified(replay: StrategyExecutionReplayResult | None) -> bool:
    """Certify execution accounting only when all required observed financial state is valid.

    Args:
        replay: Engine-native result carrying ledger, gaps and terminal evidence.
    Returns:
        False for missing evidence, reconciliation failure, any recorded data gap
        (held valuation/financing gaps as well as unpriced or unfilled intents),
        or unresolved settlement; priced open inventory alone is not a failure.
    """
    if replay is None:
        return False
    ledger = getattr(replay, "ledger", None)
    if ledger is None:
        return False
    if getattr(ledger, "primary_valid", None) is not True:
        return False
    if tuple(getattr(ledger, "invalid_reasons", ()) or ()) != ():
        return False
    gaps: Any = getattr(ledger, "data_gaps", None)
    if gaps is None or len(list(gaps)) != 0:
        return False
    positions: Any = getattr(replay, "terminal_positions", None)
    if positions is None:
        return False
    for position in list(positions):
        if getattr(position, "status", None) == "unresolved":
            return False
        if getattr(position, "funding_complete", True) is not True:
            return False
    return True


__all__ = ["_funding_gap_terminal_symbols", "ledger_terminal_only", "replay_ledger_certified"]
