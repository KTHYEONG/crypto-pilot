"""PIT uniform-grid panel loading for the MHS pipeline.

The uniform grid built by ``build_uniform_grid`` is the single decision clock:
every panel is reindexed onto it so phase offsets are integer row offsets.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.mhs.contracts import MHS_FILL_MARK_MAX_LOG_DIVERGENCE
from src.research.universe.pit_universe import symbol_partition


def build_uniform_grid(start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DatetimeIndex:
    """Return a tz-aware UTC grid inclusive of both endpoints.

    This grid is the single decision clock: every panel is reindexed onto it so
    phase offsets are integer row offsets.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be tz-aware")
    if start >= end:
        raise ValueError(f"start must be < end, got start={start} end={end}")
    return pd.date_range(start, end, freq=interval, tz="UTC")


def partition_symbols(
    symbols: Sequence[str], partition: Literal["dev", "holdout", "all"],
) -> list[str]:
    """Order-preserving delegate to ``pit_universe.symbol_partition``.

    The holdout partition must stay unread for all of Phase 1; routing every
    symbol list through this helper enforces that in one place.
    """
    if partition == "all":
        return list(symbols)
    if partition not in ("dev", "holdout"):
        raise ValueError(f"unknown partition '{partition}'")
    return [s for s in symbols if symbol_partition(s) == partition]


def load_base_panel(
    root: str,
    interval: str,
    columns: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    partition: Literal["dev", "holdout", "all"] = "dev",
    min_bars: int = 2000,
) -> dict[str, pd.DataFrame]:
    """Read ``<root>/<interval>/<SYMBOL>.parquet`` into wide per-column panels.

    Returns one wide DataFrame per requested column, all sharing
    ``build_uniform_grid(start, end, interval)`` as index and identical sorted
    column order. No survivorship filter: symbols that delisted inside the
    window are kept with NaN outside their life.
    """
    grid = build_uniform_grid(start, end, interval)
    paths = sorted(glob.glob(os.path.join(root, interval, "*.parquet")))
    names = [os.path.basename(p).removesuffix(".parquet") for p in paths]
    keep = set(partition_symbols(names, partition))

    start_ms = int(start.value // 1_000_000)
    end_ms = int(end.value // 1_000_000)

    # Discover survivors before allocating the wide panel.  The prior
    # ``dict[Series] -> DataFrame -> reindex`` construction held as many as
    # three full copies of every requested field while assembling long MHS
    # folds.  A full 2021--2025 dev panel can contain hundreds of symbols, so
    # that transient amplification terminates the process before the
    # fail-closed replay/report path can run.
    survivors: list[tuple[str, str]] = []
    for path, sym in zip(paths, names, strict=True):
        if sym not in keep:
            continue
        table = pq.read_table(
            path,
            columns=["timestamp"],
            filters=[[("timestamp", ">=", start_ms), ("timestamp", "<=", end_ms)]],
        )
        idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
        idx = idx[(idx >= start) & (idx <= end)]
        if len(idx.drop_duplicates(keep="last")) < min_bars:
            continue
        survivors.append((path, sym))

    if not survivors:
        raise ValueError("no symbol survived the panel filters")

    values = {
        column: np.full((len(grid), len(survivors)), np.nan, dtype="float64")
        for column in columns
    }
    for column_index, (path, _) in enumerate(survivors):
        table = pq.read_table(
            path,
            columns=["timestamp", *columns],
            filters=[[("timestamp", ">=", start_ms), ("timestamp", "<=", end_ms)]],
        )
        idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
        in_window = (idx >= start) & (idx <= end)
        window_sources = np.flatnonzero(in_window)
        positions = grid.get_indexer(idx[in_window])
        valid_positions = positions >= 0
        source_positions = window_sources[valid_positions]
        target_positions = positions[valid_positions]
        if not len(target_positions):
            continue

        # Stable sorting makes the final source row win for duplicate
        # timestamps, exactly matching ``duplicated(keep='last')``.
        order = np.argsort(target_positions, kind="stable")
        ordered_targets = target_positions[order]
        keep_last = np.empty(len(order), dtype=bool)
        keep_last[:-1] = ordered_targets[:-1] != ordered_targets[1:]
        keep_last[-1] = True
        selected_sources = source_positions[order[keep_last]]
        selected_targets = ordered_targets[keep_last]
        for column in columns:
            field = table.column(column).to_numpy().astype("float64", copy=False)
            values[column][selected_targets, column_index] = field[selected_sources]

    symbols = [sym for _, sym in survivors]
    return {
        column: pd.DataFrame(values[column], index=grid, columns=symbols, copy=False)
        for column in columns
    }


def liquid_half_eligibility(
    quote_volume: pd.DataFrame,
    lookback_bars: int,
    min_history_bars: int,
) -> pd.DataFrame:
    """Boolean PIT liquidity eligibility using a trailing cross-sectional median.

    At timestamp ``t`` each symbol's trailing mean quote volume uses only bars
    at or before ``t``; a symbol is eligible exactly when that mean is at least
    the valid-symbol cross-sectional median at ``t`` and it has observed
    ``min_history_bars`` bars. Missing history is False, never zero-filled.
    """
    if lookback_bars < 1 or min_history_bars < 1 or min_history_bars > lookback_bars:
        raise ValueError(
            "lookback_bars and min_history_bars must satisfy 1 <= min_history_bars <= lookback_bars"
        )
    trailing_mean = quote_volume.rolling(
        lookback_bars, min_periods=min_history_bars
    ).mean()
    median = trailing_mean.median(axis=1)
    eligible = trailing_mean.ge(median, axis=0)
    return eligible.fillna(False)


def fill_mark_parity_mask(
    fill_close: pd.DataFrame,
    mark_close: pd.DataFrame,
    max_log_divergence: float = MHS_FILL_MARK_MAX_LOG_DIVERGENCE,
) -> pd.DataFrame:
    """Boolean mask: True where the bar is tradeable at the modelled fill price.

    False only where both prices are finite and strictly positive AND
    ``abs(log(fill/mark)) > max_log_divergence``.  NaN, zero, or negative
    prices on either axis yield True (fail-open per I2/I6).
    """
    if max_log_divergence <= 0:
        raise ValueError(f"max_log_divergence must be > 0, got {max_log_divergence}")

    if not fill_close.index.equals(mark_close.index):
        raise ValueError("fill_close and mark_close must have identical index")
    if not fill_close.columns.equals(mark_close.columns):
        raise ValueError("fill_close and mark_close must have identical columns")

    fill_vals = fill_close.to_numpy(dtype="float64", copy=False)
    mark_vals = mark_close.to_numpy(dtype="float64", copy=False)

    both_positive = (fill_vals > 0) & (mark_vals > 0)
    both_finite = np.isfinite(fill_vals) & np.isfinite(mark_vals)
    comparable = both_positive & both_finite

    log_div = np.full_like(fill_vals, np.nan)
    log_div[comparable] = np.abs(
        np.log(fill_vals[comparable]) - np.log(mark_vals[comparable])
    )

    over_band = comparable & (log_div > max_log_divergence)

    mask = np.ones(fill_vals.shape, dtype=bool)
    mask[over_band] = False

    return pd.DataFrame(mask, index=fill_close.index, columns=fill_close.columns)
