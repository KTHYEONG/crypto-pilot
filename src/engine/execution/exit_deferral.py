"""Cause-tagged exit deferral: eligibility, retry search, and episode age bound."""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from src.common.errors import DataIntegrityError


class ExitBlockCause(StrEnum):
    """Deferrable exit-block cause tag."""

    VENUE_HALT = "VENUE_HALT"
    SYMBOL_NO_TRADE = "SYMBOL_NO_TRADE"


def classify_exit_block_cause(
    *,
    halted: bool,
    quote_volume: float,
    mark_valid: bool,
    funding_known: bool,
) -> ExitBlockCause | None:
    """Cause of a blocked exit that qualifies for deferral, else None. VENUE_HALT when the bar lies in a registry halt; SYMBOL_NO_TRADE when the bar's quote volume is exactly 0.0 outside any halt. Both require a finite strictly positive mark and known funding: an exit that cannot be valued or financed is missing data, not a trading pause, and must stay fail-closed. NaN, negative or non-finite quote volume never qualifies."""
    if not bool(mark_valid) or not bool(funding_known):
        return None
    qv = float(quote_volume)
    if not np.isfinite(qv) or qv < 0.0:
        return None
    if bool(halted):
        return ExitBlockCause.VENUE_HALT
    if qv == 0.0:
        return ExitBlockCause.SYMBOL_NO_TRADE
    return None


def first_viable_retry_bar(
    *,
    after_pos: int,
    deadline_pos: int,
    halted: np.ndarray,
    quote_volume: np.ndarray,
    funding_known: np.ndarray,
    close: np.ndarray,
    grid_ns: np.ndarray,
    last_trade_ns: int,
) -> int:
    """Index of the first bar in ``(after_pos, deadline_pos]`` that is not halted, has quote volume strictly greater than zero, known funding, a finite close and a label before ``last_trade_ns``; -1 when none. Reads only bars at or before ``deadline_pos`` (the order's own timeout), so the decision never consumes data past the deadline."""
    start = int(after_pos) + 1
    n = min(len(halted), len(quote_volume), len(funding_known), len(close), len(grid_ns))
    stop = min(int(deadline_pos), n - 1)
    limit = int(last_trade_ns)
    for b in range(start, stop + 1):
        if b < 0:
            continue
        if bool(halted[b]):
            continue
        if not float(quote_volume[b]) > 0.0:
            continue
        if not bool(funding_known[b]):
            continue
        if not np.isfinite(float(close[b])):
            continue
        if int(grid_ns[b]) >= limit:
            continue
        return b
    return -1


class ExitEpisodeClock:
    """Bounded age for one deferred-exit episode per symbol inventory."""

    def __init__(self, max_age_ns: int) -> None:
        if int(max_age_ns) < 0:
            raise ValueError(f"max_age_ns must be >= 0, got {max_age_ns}")
        self._max_age_ns = int(max_age_ns)
        self._episodes: dict[int, tuple[float, int]] = {}
        self._last_ns: int | None = None

    def inventory_changed(self, gcol: int) -> None:
        """End an episode on a booked inventory change, including exit/re-entry round trips."""
        self._episodes.pop(int(gcol), None)

    def admit(self, gcol: int, inventory_units: float, decision_ns: int) -> bool:
        """Whether a blocked exit may still be deferred. An episode is identified by ``(gcol, inventory_units)`` and starts at the first blocked decision; any change of the symbol's inventory ends it. Returns True while ``decision_ns - episode_start <= max_age_ns``; the age of a new episode is zero. Decisions must be admitted in non-decreasing time order."""
        now = int(decision_ns)
        if self._last_ns is not None and now < self._last_ns:
            raise DataIntegrityError("exit deferral decisions must be admitted in non-decreasing time order")
        self._last_ns = now
        key = int(gcol)
        units = float(inventory_units)
        rec = self._episodes.get(key)
        if rec is None or float(rec[0]) != units:
            self._episodes[key] = (units, now)
            return True
        start = int(rec[1])
        return (now - start) <= self._max_age_ns
