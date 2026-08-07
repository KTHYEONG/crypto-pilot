"""PIT uniform-grid panel loading for the MHS pipeline.

The uniform grid built by ``build_uniform_grid`` is the single decision clock:
every panel is reindexed onto it so phase offsets are integer row offsets.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq

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

    frames: dict[str, dict[str, pd.Series]] = {c: {} for c in columns}
    for path, sym in zip(paths, names, strict=True):
        if sym not in keep:
            continue
        table = pq.read_table(path, columns=["timestamp", *columns])
        idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
        sub = pd.DataFrame(
            {c: table.column(c).to_numpy().astype("float64") for c in columns},
            index=idx,
        )
        sub = sub[(sub.index >= start) & (sub.index <= end)]
        sub = sub[~sub.index.duplicated(keep="last")].sort_index()
        if len(sub) < min_bars:
            continue
        for c in columns:
            frames[c][sym] = sub[c]

    if not frames[columns[0]]:
        raise ValueError("no symbol survived the panel filters")
    return {c: pd.DataFrame(frames[c]).reindex(grid).sort_index(axis=1) for c in columns}


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
