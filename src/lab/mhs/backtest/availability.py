"""Point-in-time observation availability for process inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError

PublicationPlanes = Mapping[str, pd.DataFrame]


@dataclass(frozen=True, slots=True)
class ObservationAvailability:
    """Describe when observations may enter a decision, independently of file ingestion.

    Event time identifies the market observation; availability identifies its
    earliest usable publication. Archive completion is a named approximation,
    never evidence of measured exchange or live delivery latency.
    """

    event_time: pd.DatetimeIndex
    completed_at: pd.DatetimeIndex
    available_at: pd.DataFrame
    provenance: Literal["recorded", "archive_completion_proxy"]


@dataclass(frozen=True, slots=True)
class ObservationAsOf:
    """Preserve observation values and knowledge separately so absent data cannot
    become a normal price or a free financing assumption.
    """

    values: pd.DataFrame
    known: pd.DataFrame
    availability: ObservationAvailability


def _require_event_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataIntegrityError("event labels must be a UTC DatetimeIndex")
    if index.tz is None or str(index.tz) != "UTC":
        raise DataIntegrityError("event labels must be timezone-aware UTC")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise DataIntegrityError(
            "event labels must be unique and ordered;"
            " conflicting versions need recorded version availability"
        )


def _require_unique_symbols(columns: Sequence[str]) -> None:
    names = list(columns)
    if len(set(names)) != len(names):
        raise DataIntegrityError("symbol columns must be unique canonical identifiers")
    for name in names:
        if not isinstance(name, str) or not name:
            raise DataIntegrityError("symbol columns must be non-empty strings")


def _publication_ns(frame: pd.DataFrame, symbol: str) -> np.ndarray:
    stamped = pd.DatetimeIndex(pd.to_datetime(frame[symbol], utc=True, errors="coerce"))
    return np.asarray(stamped.as_unit("ns").asi8, dtype="int64")


def _require_availability(values: pd.DataFrame, availability: ObservationAvailability) -> None:
    if availability.provenance not in ("recorded", "archive_completion_proxy"):
        raise DataIntegrityError("provenance must be a named publication approximation")
    if not availability.event_time.equals(values.index):
        raise DataIntegrityError("availability event labels must align with observation rows")
    _require_event_index(availability.completed_at)
    if len(availability.completed_at) != len(values):
        raise DataIntegrityError("completion labels must align with event rows")
    if not availability.available_at.index.equals(values.index) or list(
        availability.available_at.columns
    ) != list(values.columns):
        raise DataIntegrityError(
            "publication timestamps must align with event rows and symbol columns"
        )
    completed_ns = np.asarray(availability.completed_at.as_unit("ns").asi8, dtype="int64")
    for symbol in list(values.columns):
        published_ns = _publication_ns(availability.available_at, symbol)
        observed = published_ns != np.iinfo(np.int64).min
        if bool((published_ns[observed] < completed_ns[observed]).any()):
            raise DataIntegrityError("availability cannot precede event completion")


def select_available_observations(
    values: pd.DataFrame,
    availability: ObservationAvailability,
    decision_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Select the most recent published observation at each decision time.

    Args:
        values: UTC event-indexed numeric observations in canonical symbol order.
        availability: Aligned event labels and their publication timestamps.
        decision_times: Strictly increasing UTC times at which inputs are consumed.
    Returns:
        Decision-indexed observations; symbols without an available value remain missing.
    Raises:
        DataIntegrityError: Alignment, publication ordering or duplicate versions are ambiguous.
    """
    _require_event_index(values.index)
    _require_unique_symbols(list(values.columns))
    _require_event_index(decision_times)
    _require_availability(values, availability)
    data = values.to_numpy(dtype="float64")
    decision_ns = np.asarray(decision_times.as_unit("ns").asi8, dtype="int64")
    out = np.full((len(decision_times), len(values.columns)), np.nan, dtype="float64")
    missing_sentinel = np.iinfo(np.int64).max
    for pos, symbol in enumerate(list(values.columns)):
        published_ns = _publication_ns(availability.available_at, symbol)
        safe_ns = np.where(published_ns == np.iinfo(np.int64).min, missing_sentinel, published_ns)
        order = np.argsort(safe_ns, kind="stable")
        ranked_ns = safe_ns[order]
        slot = np.searchsorted(ranked_ns, decision_ns, side="right") - 1
        prefix_best = np.maximum.accumulate(np.arange(len(values))[order])
        chosen = np.full(len(decision_times), -1, dtype="int64")
        admitted = slot >= 0
        chosen[admitted] = prefix_best[slot[admitted]]
        rows = np.flatnonzero(chosen >= 0)
        out[rows, pos] = data[chosen[rows], pos]
    return pd.DataFrame(out, index=decision_times, columns=list(values.columns))


def observed_history_mask(values: pd.DataFrame, *, min_history_bars: int) -> pd.DataFrame:
    """Admit history using only valid observations accumulated through the current row.

    Args:
        values: Published observations, with unknown values represented as missing.
        min_history_bars: Registered positive history requirement.
    Returns:
        Aligned boolean history eligibility without whole-period survivor filtering.
    Raises:
        ValueError: The history requirement is not a positive integer.
        DataIntegrityError: Labels or symbol columns are ambiguous.
    """
    if isinstance(min_history_bars, bool) or not isinstance(min_history_bars, int) or min_history_bars < 1:
        raise ValueError("min_history_bars must be a positive integer")
    _require_event_index(values.index)
    _require_unique_symbols(list(values.columns))
    accumulated = np.cumsum(values.notna().to_numpy(dtype=bool), axis=0)
    return pd.DataFrame(
        accumulated >= int(min_history_bars), index=values.index, columns=list(values.columns)
    )
