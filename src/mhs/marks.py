"""Load observed funding events and derive the point-in-time trade-volume execution roster without an external mark-price history.

MHS historical and paper consumers price completed trade OHLCV and observed
funding only: no mark-price cache, panel or replay-valuation helper remains in
this module. The optional fill/mark parity gate (explicitly user-enabled) is
the sole remaining mark reader; all canonical preparation and replay paths use
the funding series, the PIT roster mask and the minute-frame loaders below.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.common.paths import funding_path
from src.market_data.services import futures_collection as _futures_collection
from src.market_data.storage.loaders import load_funding_rates
from src.mhs.params import EXECUTION_ROSTER_EXIT_MULTIPLIER
from src.mhs.types import FILL_MARK_MAX_LOG_DIVERGENCE

_logger = logging.getLogger("MhsHorizonDiagnostic")


def _load_funding_series(
    symbols: list[str],
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Load per-symbol funding series plus the symbols silently dropped on load.

    Returns ``(series, dropped)``: ``series`` maps symbol -> funding rates as
    before, and ``dropped`` maps each symbol whose parquet raised on load (or
    produced no rates) to the failure reason -- the drop is no longer
    observable only via a warning log line, so a corrupted funding file can
    never change the universe composition invisibly. Exception swallowing
    itself is kept: one corrupt file must not kill the whole diagnostic.
    """
    series: dict[str, pd.Series] = {}
    dropped: dict[str, str] = {}
    for sym in symbols:
        path = funding_path(sym)
        if not path.exists():
            dropped[sym] = "missing"
            continue
        try:
            rates = load_funding_rates(str(path))
        except Exception as exc:  # noqa: BLE001
            dropped[sym] = f"load_error: {exc}"
            _logger.warning("[DATA] funding load failed symbol=%s error=%s", sym, exc)
            continue
        if len(rates):
            series[sym] = rates
        else:
            dropped[sym] = "empty"
    return series, dropped


def _pit_execution_mask(
    quote_volume: pd.DataFrame,
    eligible: pd.DataFrame,
    universe_size: int,
) -> pd.DataFrame:
    """Select the PIT top-volume execution roster with entry/exit hysteresis.

    ``universe_size`` is the ENTRY rank threshold only: a symbol enters by
    reaching the top ``universe_size`` trailing-volume rank, and once a member
    it is kept until its rank falls outside
    ``universe_size * EXECUTION_ROSTER_EXIT_MULTIPLIER`` (a Schmitt-trigger
    band). Because hysteresis retains members that have slipped past the entry
    threshold, the realized number of holdings is approximately
    ``universe_size * (1 + hysteresis effect)``, NOT ``universe_size`` (measured
    ~41.9 vs a declared 30) -- the true mean per-row True count is exposed as
    when the signal itself has not changed.
    """
    exit_size = universe_size * EXECUTION_ROSTER_EXIT_MULTIPLIER
    trailing = quote_volume.rolling(720, min_periods=720).mean()
    ranked = trailing.where(eligible).rank(axis=1, ascending=False, method="first")
    enter = ranked.le(universe_size).fillna(False).to_numpy()
    keep = ranked.le(exit_size).fillna(False).to_numpy()
    held = np.zeros(enter.shape[1], dtype=bool)
    out = np.zeros_like(enter, dtype=bool)
    for i in range(len(enter)):
        held = enter[i] | (held & keep[i])
        out[i] = held
    return pd.DataFrame(out, index=quote_volume.index, columns=quote_volume.columns)


def _contemporaneous_mark_close_panel(
    symbols: list[str],
    grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Contemporaneous mark-price close panel (no +1h shift, no ffill).

    An absent mark stays NaN so the parity mask fails open per I2.  Deliberately
    does NOT apply any ``+1h`` availability shift — the gate detects a stalled
    price feed, not the replay's valuation lag. Mark paths resolve dynamically
    at call time so test monkeypatches keep working.
    """
    panel = pd.DataFrame(index=grid, columns=list(symbols), dtype="float64")
    for sym in symbols:
        try:
            path = _futures_collection._mark_price_path(sym, "1h")
            if not path.exists():
                continue
            available = set(pq.ParquetFile(path).schema_arrow.names)
            if "close" not in available or ("datetime" not in available and "timestamp" not in available):
                continue
            columns = (
                ["datetime", "close"] if "datetime" in available else ["timestamp", "close"]
            )
            frame = pq.read_table(path, columns=columns).to_pandas()
        except (KeyError, ValueError, OSError):
            # A malformed/incomplete mark cache is a data-integrity condition
            # owned elsewhere; this parity gate stays fail-open per I2 rather
            # than pre-empting it with an unrelated crash.
            continue
        if frame.empty:
            continue
        if "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        else:
            frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        valid = (
            frame["datetime"].notna()
            & frame["close"].notna()
            & (frame["close"] > 0)
        )
        closes = (
            frame.loc[valid, ["datetime", "close"]]
            .drop_duplicates(subset=["datetime"], keep="last")
            .sort_values("datetime")
        )
        if closes.empty:
            continue
        available = pd.Series(
            closes["close"].to_numpy(dtype="float64"),
            index=closes["datetime"],
        )
        aligned = available.reindex(grid)
        panel[sym] = aligned.to_numpy(dtype="float64")
    return panel


def _fill_mark_parity_eligibility(
    close: pd.DataFrame,
    eligible: pd.DataFrame,
    enabled: bool,
    *,
    mark_close: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Single shared entry point for BOTH the top-level and the fold path (I4).

    Returns ``(eligible, None)`` unchanged when ``enabled`` is False.
    Otherwise returns ``(eligible & fill_mark_parity_mask(...), census)``.
    """
    if not enabled:
        return eligible, None
    if mark_close is None:
        mark_close = _contemporaneous_mark_close_panel(
            list(close.columns), close.index,
        )
    from src.mhs.panel import fill_mark_parity_mask

    parity = fill_mark_parity_mask(close, mark_close)
    removed = eligible & ~parity
    cells_over_band = int(removed.to_numpy().sum())
    eligible_cells_removed = int((removed & eligible).to_numpy().sum())
    per_symbol = removed.sum(axis=0)
    top_symbols = per_symbol[per_symbol > 0].sort_values(ascending=False)
    truncated = len(top_symbols) > 5
    symbols_dict: dict[str, int] = {}
    for sym in top_symbols.index[:5]:
        symbols_dict[str(sym)] = int(top_symbols[sym])
    if truncated:
        symbols_dict["truncated"] = len(top_symbols) - 5
    census: dict[str, Any] = {
        "band": FILL_MARK_MAX_LOG_DIVERGENCE,
        "cells_over_band": cells_over_band,
        "eligible_cells_removed": eligible_cells_removed,
        "symbols": symbols_dict,
    }
    return eligible & parity, census


def clear_mhs_market_data_caches() -> None:
    """Invalidate MHS market-data caches for run isolation (INV-CACHE-RUN-ISOLATION).

    The retired mark-price frame/series caches no longer exist; the retained
    funding, roster and minute-frame loaders are stateless and read the lake
    directly, so repeated runs in one process always observe refreshed files.
    Kept as the single invalidation entry point for pipeline orchestration,
    diagnostics and test isolation.
    """


def _load_symbol_minute_frame(
    path: str,
    sym: str,
    start_ms: int,
    end_ms: int,
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
) -> tuple[str, pd.DataFrame | None]:
    """Load one symbol's minute slice; ``None`` when missing or empty.

    ``quote_vol`` travels with the OHLCV slice when the file carries it; a
    file without a quote-volume column yields a frame without one and the
    window layer treats its bars as unknown volume (untradable).
    """
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = ["timestamp", "high", "low", "close"] + (["quote_vol"] if "quote_vol" in available else [])
    table = pq.read_table(
        path,
        columns=columns,
        filters=[
            [
                ("timestamp", ">=", start_ms),
                ("timestamp", "<=", end_ms),
            ]
        ],
    )
    idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
    frame = pd.DataFrame(
        {
            c: table.column(c).to_numpy().astype("float64")
            for c in table.column_names
            if c != "timestamp"
        },
        index=idx,
    )
    frame = frame[(frame.index >= grid_start) & (frame.index <= grid_end)]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return sym, (None if frame.empty else frame)


def _load_window_minute_frames(
    root: str,
    symbols: list[str],
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
    timeframe: Literal["3m"],
) -> dict[str, pd.DataFrame]:
    """Load one execution window's minute OHLCV slices directly from Parquet.

    The window generator's minute-frame source: each symbol's frame is read
    with a ``[grid_start, grid_end]`` timestamp filter (row-group pruning +
    kernel page cache make repeated window reads cheap), then post-processed
    identically (ms->datetime UTC, ``drop_duplicates(keep="last")``,
    ``sort_index``). For a given window the returned frames equal the
    full-period-frame ``.loc`` slice byte-for-byte. Missing Parquet files are
    skipped.
    """
    frames: dict[str, pd.DataFrame] = {}
    start_ms = int(grid_start.value // 1_000_000)
    end_ms = int(grid_end.value // 1_000_000)
    jobs = [
        (os.path.join(root, timeframe, f"{sym}.parquet"), sym)
        for sym in symbols
        if os.path.exists(os.path.join(root, timeframe, f"{sym}.parquet"))
    ]
    if not jobs:
        return frames
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_load_symbol_minute_frame, path, sym, start_ms, end_ms, grid_start, grid_end)
            for path, sym in jobs
        ]
        for fut in futures:
            sym, frame = fut.result()
            if frame is not None:
                frames[sym] = frame
    return frames


def _build_window_frames(
    symbol_frames: dict[str, pd.DataFrame],
    roster: list[str],
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
    minute_grid: pd.DatetimeIndex,
    timeframe: Literal["3m"],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Slice per-symbol full-period frames onto a window minute grid.

    Identical output to the pre-window-keyed path (same slicing, same reindex,
    same column order) but reads each symbol's frame from the in-memory
    per-window Parquet slices. Returns ``None`` when no roster symbol has
    usable data.
    """
    if not symbol_frames:
        return None
    if grid_start >= grid_end:
        return None
    sliced: dict[str, pd.DataFrame] = {}
    for s in sorted(roster):
        full = symbol_frames.get(s)
        if full is None or full.empty:
            continue
        frame = full.loc[(full.index >= grid_start) & (full.index <= grid_end)]
        if not frame.empty:
            sliced[s] = frame
    if not sliced:
        return None
    highs = pd.DataFrame({s: f["high"] for s, f in sliced.items()}).reindex(minute_grid)
    lows = pd.DataFrame({s: f["low"] for s, f in sliced.items()}).reindex(minute_grid)
    closes = pd.DataFrame({s: f["close"] for s, f in sliced.items()}).reindex(minute_grid)
    return highs, lows, closes


def _align_minute_frames(
    frames: dict[str, pd.DataFrame], timeframe: Literal["3m"],
    start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if not frames:
        return None
    if start >= end:
        return None
    # The requested evaluation grid is the replay grid. A late listing is kept
    # as NaN on that grid and never trims the global start, so the replay
    # horizon is never shortened by the union of first-observed timestamps.
    grid = pd.date_range(
        start, end,
        freq="3min",
        tz="UTC",
    )
    highs = pd.DataFrame({s: f["high"] for s, f in frames.items()}).reindex(grid)
    lows = pd.DataFrame({s: f["low"] for s, f in frames.items()}).reindex(grid)
    closes = pd.DataFrame({s: f["close"] for s, f in frames.items()}).reindex(grid)
    return highs, lows, closes
