from __future__ import annotations

from dataclasses import dataclass

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
    valid_count: int
    terminal_maturity_count: int
    mismatch_count: int
    first_mismatch_event_id: int | None


def validate_native_event_grid(
    *,
    events: pd.DataFrame,
    native_datetimes: NDArray[np.datetime64],
    timeframe: str,
) -> tuple[pd.DataFrame, NativeEventGridAudit]:
    """[ADR_20260715_L0_L1_NATIVE_CONTRACT] Validate native event identity and maturity."""
    signal_ns = pd.to_datetime(events["datetime"], utc=True).to_numpy(dtype="datetime64[ns]")
    expected = np.searchsorted(native_datetimes, signal_ns, side="right")
    terminal = expected == len(native_datetimes)
    mismatch = (~terminal) & (events["entry_idx"].to_numpy(dtype=np.int64) != expected)

    input_count = len(events)
    terminal_count = int(terminal.sum())
    mismatch_count = int(mismatch.sum())
    valid_count = input_count - terminal_count - mismatch_count

    first_mismatch_id: int | None = None
    if mismatch_count > 0:
        mismatch_indices = np.flatnonzero(mismatch)
        first_mismatch_id = int(events["event_id"].to_numpy(dtype=np.int64)[mismatch_indices[0]])
        raise EventGridContractError(
            f"timeframe={timeframe} event_id={first_mismatch_id} entry_idx mismatch "
            f"(total mismatches={mismatch_count})"
        )

    result = events.loc[~terminal]
    audit = NativeEventGridAudit(
        timeframe=timeframe,
        input_count=input_count,
        valid_count=valid_count,
        terminal_maturity_count=terminal_count,
        mismatch_count=mismatch_count,
        first_mismatch_event_id=first_mismatch_id,
    )
    return result, audit
