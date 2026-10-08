"""Daily portfolio evidence shared by lab certification and the engine backtest."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from src.common.errors import DataIntegrityError
from src.engine.execution.contracts import StrategyExecutionReplayResult


@dataclass(frozen=True, slots=True)
class DailyPortfolioEvidence:
    """Bind portfolio returns to the two actual equity observations and latest input
    publication they require, rather than assume that a calendar label is completion."""

    returns: pd.Series
    label_start: pd.DatetimeIndex
    label_end: pd.DatetimeIndex
    available_at: pd.DatetimeIndex

    def __post_init__(self) -> None:
        n = len(self.returns)
        if len(self.label_start) != n or len(self.label_end) != n or len(self.available_at) != n:
            raise DataIntegrityError("daily evidence arrays must share one length")


def inventory_daily_evidence(replay: StrategyExecutionReplayResult) -> DailyPortfolioEvidence:
    """Construct complete daily inventory return intervals from actual equity/timing rows.

    Args:
        replay: Engine-owned equity series and aligned input-availability evidence.
    Returns:
        Observed return values and full economic/publication dependencies. Incomplete
        boundary intervals remain outside formal inference, not zero-filled.
    Raises:
        DataIntegrityError: Equity, chronology or required timing evidence is missing.
    """
    ledger = getattr(replay, "ledger", None)
    if ledger is None:
        raise DataIntegrityError("replay has no ledger")
    equity = getattr(ledger, "equity", None)
    if equity is None or len(equity) == 0:
        raise DataIntegrityError("ledger equity is missing")
    if not isinstance(equity.index, pd.DatetimeIndex):
        raise DataIntegrityError("ledger equity must have a DatetimeIndex")
    if equity.index.hasnans:
        raise DataIntegrityError("ledger equity index must not contain NaT")
    if equity.index.tz is None:
        raise DataIntegrityError("ledger equity index must be timezone-aware")
    if equity.index.has_duplicates or not equity.index.is_monotonic_increasing:
        raise DataIntegrityError("ledger equity index must be unique and increasing")
    values = equity.to_numpy(dtype="float64")
    if not bool(pd.notna(values).all()):
        raise DataIntegrityError("ledger equity must not be missing")
    avail = getattr(replay, "ledger_available_at", None)
    if avail is None:
        raise DataIntegrityError("ledger timing evidence is missing")
    if not isinstance(avail, pd.DatetimeIndex):
        raise DataIntegrityError("ledger timing evidence must be a DatetimeIndex")
    if len(avail) != len(equity):
        raise DataIntegrityError("ledger timing evidence must align with equity rows")
    if avail.hasnans:
        raise DataIntegrityError("ledger timing evidence must not contain NaT")
    if avail.tz is None:
        raise DataIntegrityError("ledger timing evidence must be timezone-aware")
    avail_utc = avail.tz_convert("UTC")
    equity_utc = equity.tz_convert("UTC")
    for event_ts, avail_ts in zip(equity_utc.index, avail_utc, strict=True):
        if avail_ts < event_ts:
            raise DataIntegrityError("ledger availability must be no earlier than its observation")
    days = equity_utc.index.normalize()
    unique_days = pd.DatetimeIndex(sorted(set(days)), tz="UTC")
    if len(unique_days) < 2:
        raise DataIntegrityError("ledger equity must span at least two calendar days")
    day_end: dict[pd.Timestamp, pd.Timestamp] = {}
    day_avail: dict[pd.Timestamp, pd.Timestamp] = {}
    day_level: dict[pd.Timestamp, float] = {}
    for day in unique_days:
        mask = days == day
        positions = equity_utc.index[mask]
        day_end[day] = positions.max()
        day_avail[day] = avail_utc[mask].max()
        day_level[day] = float(equity_utc.loc[mask].iloc[-1])
    ordered = list(unique_days)
    returns: list[float] = []
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    avails: list[pd.Timestamp] = []
    for prev, cur in itertools.pairwise(ordered):
        prev_level = day_level[prev]
        cur_level = day_level[cur]
        if prev_level <= 0.0:
            raise DataIntegrityError("ledger equity levels must be positive")
        returns.append(float(cur_level / prev_level - 1.0))
        starts.append(day_end[prev])
        ends.append(day_end[cur])
        later = day_avail[prev] if day_avail[prev] > day_avail[cur] else day_avail[cur]
        avails.append(later)
    index = pd.DatetimeIndex(ordered[1:], tz="UTC")
    return DailyPortfolioEvidence(
        returns=pd.Series(returns, index=index, dtype="float64"),
        label_start=pd.DatetimeIndex(starts, tz="UTC"),
        label_end=pd.DatetimeIndex(ends, tz="UTC"),
        available_at=pd.DatetimeIndex(avails, tz="UTC"),
    )
