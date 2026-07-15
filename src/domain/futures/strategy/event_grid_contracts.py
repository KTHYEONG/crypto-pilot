from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class MissingNativeTfEventsError(RuntimeError):
    """Raised when a requested L1 timeframe has no native event frame key."""


class EventGridContractError(ValueError):
    """Raised when event timestamps and native entry indices disagree."""


@dataclass(frozen=True, slots=True)
class NativeEventGridAudit:
    timeframe: str
    input_count: int
    eligible_count: int
    terminal_maturity_count: int
    mismatch_count: int
    required_horizon_bars: int
    status: Literal["passed", "terminal_excluded"]
    first_terminal_event_id: int | None
    last_terminal_event_id: int | None
    first_mismatch_event_id: int | None


@dataclass(frozen=True, slots=True)
class NativeEventGridResult:
    eligible_events: pd.DataFrame
    audit: NativeEventGridAudit


def normalize_native_l1_events(
    *,
    events: pd.DataFrame,
    native_datetimes: NDArray[np.datetime64],
    timeframe: str,
    required_horizon_bars: int,
) -> NativeEventGridResult:
    """Validate native identity and remove only terminally immature events."""
    signal_ns = pd.to_datetime(events["datetime"], utc=True).to_numpy(dtype="datetime64[ns]")
    n_grid = len(native_datetimes)
    expected = np.searchsorted(native_datetimes, signal_ns, side="right")
    stored = events["entry_idx"].to_numpy(dtype=np.int64)

    identity_match = stored == expected
    mismatch = ~identity_match
    terminal = identity_match & ((stored + required_horizon_bars) >= n_grid)
    eligible = identity_match & ~terminal

    input_count = len(events)
    terminal_count = int(terminal.sum())
    mismatch_count = int(mismatch.sum())
    eligible_count = int(eligible.sum())

    first_terminal_id: int | None = None
    last_terminal_id: int | None = None
    first_mismatch_id: int | None = None

    if mismatch_count > 0:
        mismatch_indices = np.flatnonzero(mismatch)
        first_mismatch_id = int(events["event_id"].to_numpy(dtype=np.int64)[mismatch_indices[0]])
        raise EventGridContractError(
            f"timeframe={timeframe} event_id={first_mismatch_id} entry_idx mismatch "
            f"(total mismatches={mismatch_count})"
        )

    if terminal_count > 0:
        term_indices = np.flatnonzero(terminal)
        event_ids = events["event_id"].to_numpy(dtype=np.int64)
        first_terminal_id = int(event_ids[term_indices[0]])
        last_terminal_id = int(event_ids[term_indices[-1]])

    eligible_events = events.iloc[np.flatnonzero(eligible)] if eligible_count < input_count else events

    status: Literal["passed", "terminal_excluded"] = "terminal_excluded" if terminal_count > 0 else "passed"

    audit = NativeEventGridAudit(
        timeframe=timeframe,
        input_count=input_count,
        eligible_count=eligible_count,
        terminal_maturity_count=terminal_count,
        mismatch_count=mismatch_count,
        required_horizon_bars=required_horizon_bars,
        status=status,
        first_terminal_event_id=first_terminal_id,
        last_terminal_event_id=last_terminal_id,
        first_mismatch_event_id=first_mismatch_id,
    )
    return NativeEventGridResult(eligible_events=eligible_events, audit=audit)
