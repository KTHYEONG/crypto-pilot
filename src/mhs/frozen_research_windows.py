"""Validated one-pass window stream for the frozen MHS research replay."""

from __future__ import annotations

import datetime as _datetime
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.execution import ExecutionReplayWindow
from src.mhs.frozen_research_candidate import FrozenMhsCandidate


def validated_frozen_research_windows(
    candidate: FrozenMhsCandidate,
    windows: Iterable[ExecutionReplayWindow],
) -> Iterator[ExecutionReplayWindow]:
    """Yield exact 3m execution windows for one frozen entry-target plan.

    Validation proves that every target row is replayed once with market data
    available after its signal release.  It permits the execution engine's
    documented OHLCV-close marking fallback while retaining the candidate's
    wider canonical universe and each window's smaller active/held roster.

    Args:
        candidate: Complete canonical target plan with entry and release clocks.
        windows: Chronological bounded 3m windows supplied by the execution core.
    Yields:
        The original windows after fail-closed provenance validation.
    Raises:
        DataIntegrityError: Coverage, timing, roster, funding, price, or
            terminal-position evidence is incomplete or inconsistent.
    """
    expected_columns = tuple(candidate.target_weights.columns)
    expected_labels = list(candidate.target_weights.index)
    if not expected_labels:
        raise DataIntegrityError("candidate must carry at least one target row")
    avail_by_label = dict(zip(expected_labels, candidate.signal_available_at, strict=True))
    expected_set = set(expected_labels)
    canon_pos = {sym: i for i, sym in enumerate(expected_columns)}
    seen: set[pd.Timestamp] = set()
    prev_max: pd.Timestamp | None = None
    last_grid_end: pd.Timestamp | None = None
    for window in windows:
        if not _window_clocks_valid(window):
            raise DataIntegrityError("window clocks, 3-minute grid, or bar availability are invalid")
        if tuple(window.columns) != expected_columns:
            raise DataIntegrityError("window columns must match the candidate canonical column order")
        local_cols = list(window.target_weights.columns)
        local_syms = list(window.symbols)
        if not local_cols or not local_syms:
            raise DataIntegrityError("window local roster must be non-empty")
        if any(sym not in canon_pos for sym in local_cols) or any(sym not in canon_pos for sym in local_syms):
            raise DataIntegrityError("window local roster must be a subset of the candidate canonical order")
        if [canon_pos[s] for s in local_cols] != sorted(canon_pos[s] for s in local_cols):
            raise DataIntegrityError("window local roster must preserve canonical ordering")
        if [canon_pos[s] for s in local_syms] != sorted(canon_pos[s] for s in local_syms):
            raise DataIntegrityError("window local roster must preserve canonical ordering")
        if set(local_cols) != set(local_syms):
            raise DataIntegrityError("window symbols and target columns must cover the same local roster")
        _check_required_frames(window)
        labels = list(window.target_weights.index)
        if len(labels) != len(window.signal_available_at):
            raise DataIntegrityError("window targets and release timestamps must share one length")
        if labels != sorted(labels) or (prev_max is not None and labels and labels[0] <= prev_max):
            raise DataIntegrityError("window decisions must be unseen and in chronological order")
        if len(set(labels)) != len(labels):
            raise DataIntegrityError("window decisions must not duplicate a candidate row")
        for pos, label in enumerate(labels):
            if label not in avail_by_label or label in seen:
                raise DataIntegrityError("window decision is unknown to or duplicated from the candidate path")
            full = candidate.target_weights.loc[label].to_numpy(dtype="float64")
            local = window.target_weights.loc[label].to_numpy(dtype="float64")
            projected = np.array([full[canon_pos[s]] for s in local_cols], dtype="float64")
            if not bool(np.array_equal(local, projected)):
                raise DataIntegrityError("window target values must equal the frozen candidate row")
            omitted_zero = all(
                float(full[canon_pos[s]]) == 0.0 for s in expected_columns if s not in canon_pos or s not in set(local_cols)
            )
            if not omitted_zero:
                raise DataIntegrityError("window omits a canonical symbol with nonzero target")
            release = window.signal_available_at[pos]
            if release != avail_by_label[label] or not (release < label):
                raise DataIntegrityError("window release must equal the candidate availability and precede entry")
            seen.add(label)
        if labels:
            prev_max = labels[-1]
        last_grid_end = window.minute_grid[-1]
        yield window
    if last_grid_end is None or last_grid_end <= expected_labels[-1]:
        raise DataIntegrityError("final decision cannot be resolved within the execution horizon")
    if seen != expected_set:
        raise DataIntegrityError("stream omitted or truncated candidate decisions")


def _is_utc(ts: pd.Timestamp) -> bool:
    """Check timezone-aware UTC without consulting the zone database."""
    return ts.tzinfo is not None and ts.utcoffset() == _datetime.timedelta(0)


def _window_clocks_valid(window: ExecutionReplayWindow) -> bool:
    """Check UTC clocks, 3-minute grid continuity, and bar availability."""
    if not _is_utc(window.window_start) or not _is_utc(window.window_end):
        return False
    if not (window.window_start < window.window_end):
        return False
    grid = window.minute_grid
    if not isinstance(grid, pd.DatetimeIndex) or grid.hasnans or grid.tz is None:
        return False
    if not grid.equals(grid.tz_convert("UTC")):
        return False
    if len(grid) < 2 or grid.has_duplicates or not grid.is_monotonic_increasing:
        return False
    if bool(((grid[1:] - grid[:-1]) != pd.Timedelta(minutes=3)).any()):
        return False
    if grid[0] < window.window_start or grid[-1] > window.window_end:
        return False
    avail = window.bar_available_at
    if avail is None or not isinstance(avail, pd.DatetimeIndex) or len(avail) != len(grid):
        return False
    if avail.hasnans or avail.tz is None or not avail.equals(avail.tz_convert("UTC")):
        return False
    return not bool((avail < grid).any())


def _check_required_frames(window: ExecutionReplayWindow) -> None:
    """Require explicit aligned source frames covering every local symbol."""
    if (
        window.quote_volumes is None
        or window.bar_funding is None
        or window.funding_known is None
        or window.bar_available_at is None
    ):
        raise DataIntegrityError("window must carry explicit marks, volumes, funding, knowledge, and availability")
    grid = window.minute_grid
    local = list(window.symbols)
    frames = (window.highs, window.lows, window.closes, window.quote_volumes, window.bar_funding, window.funding_known)
    for frame in frames:
        if not frame.index.equals(grid) or any(sym not in frame.columns for sym in local):
            raise DataIntegrityError("window source frames must align to the grid and cover active symbols")
    prices = np.concatenate(
        [window.highs[local].to_numpy(dtype="float64"), window.lows[local].to_numpy(dtype="float64"),
         window.closes[local].to_numpy(dtype="float64"), window.bar_funding[local].to_numpy(dtype="float64")]
    )
    if not bool(np.isfinite(prices).all()):
        raise DataIntegrityError("window trade prices and funding rates must be finite")
    if window.marks is None:
        effective = window.closes[local].to_numpy(dtype="float64")
        if not bool(np.isfinite(effective).all()) or bool((effective <= 0.0).any()):
            raise DataIntegrityError("window effective close fallback marks must be finite positive prices")
    else:
        if not window.marks.index.equals(grid) or any(sym not in window.marks.columns for sym in local):
            raise DataIntegrityError("window source frames must align to the grid and cover active symbols")
        marks = window.marks[local].to_numpy(dtype="float64")
        if bool(np.isinf(marks).any()) or bool(((marks <= 0.0) & np.isfinite(marks)).any()):
            raise DataIntegrityError("window marks must be absent or finite positive prices")
    volumes = window.quote_volumes[local].to_numpy(dtype="float64")
    if not bool(np.isfinite(volumes).all()) or bool((volumes < 0.0).any()):
        raise DataIntegrityError("window quote volumes must be finite nonnegative observations")
