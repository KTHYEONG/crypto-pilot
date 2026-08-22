"""Mark-price / minute-frame loading and parity (I4 seam: pit_execution_mask, funding_path)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import src.market_data.services.futures_collection as _futures_collection
from src.common.config import funding_path
from src.common.errors import DataIntegrityError
from src.market_data.services.futures_collection import DataCollector
from src.market_data.storage.loaders import load_funding_rates
from src.mhs.params import EXECUTION_ROSTER_EXIT_MULTIPLIER
from src.mhs.types import FILL_MARK_MAX_LOG_DIVERGENCE

_logger = logging.getLogger("MhsHorizonDiagnostic")

_DATA_COLLECTOR: DataCollector | None = None


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
            _logger.warning("[MHS] funding load failed symbol=%s error=%s", sym, exc)
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


def _data_collector() -> DataCollector:
    """Lazily-instantiated shared mark-price collector.

    ``_iter_mhs_execution_windows`` previously constructed one ``DataCollector``
    per 31-day window; a module-level singleton pays the collector's
    construction cost once per diagnostic run instead of once per window
    (spec O5).  ``load_mark_price_panel`` resolves ``_mark_price_path``
    dynamically at call time, so test monkeypatches keep working.
    """
    global _DATA_COLLECTOR
    if _DATA_COLLECTOR is None:
        _DATA_COLLECTOR = DataCollector()
    return _DATA_COLLECTOR


@lru_cache(maxsize=512)
def _get_symbol_mark_frame(symbol: str, timeframe: str) -> pd.DataFrame:
    """Full-period mark-price frame for one symbol, cached for the process.

    The ``markPriceKlines`` parquet is read once per ``(symbol, timeframe)`` per
    process and sliced per window instead of being re-read for every window. The
    frame is produced through ``DataCollector._load_mark_price_cache`` so its
    preprocessing (ms->datetime, numeric coercion,
    ``drop_duplicates(keep="last")``, ``sort_values``) is byte-identical to the
    DataCollector panel path. ``_mark_price_path`` is resolved dynamically at
    call time so test monkeypatches keep working; the returned frame is
    read-only.
    """
    return DataCollector._load_mark_price_cache(
        _futures_collection._mark_price_path(symbol, timeframe)
    )


def _compact_mark_series(symbol: str, timeframe: str) -> tuple[np.ndarray, np.ndarray]:
    """Validated ``(availability_ns int64, close float64)`` arrays, built once.

    Applies the per-symbol prologue (``datetime.notna() & close.notna() &
    close > 0``, ``drop_duplicates(subset=['datetime'], keep='last')``,
    ``sort_values('datetime')`` and the ``+1h`` availability shift) a single
    time per symbol instead of once per window. A malformed or empty mark
    cache yields empty arrays; the missing-mark fail-closed path downstream is
    unchanged. Retention drops from six cached columns to the two ever read.
    The resolved source path is part of the cache key so redirected data roots
    (tests, synthetic markets) can never poison one another.
    """
    path = str(_futures_collection._mark_price_path(symbol, timeframe))
    return _compact_mark_series_for_path(symbol, timeframe, path)


@lru_cache(maxsize=512)
def _compact_mark_series_for_path(
    symbol: str,
    timeframe: str,
    path_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    del path_key  # identity-only: separates same-symbol data from different roots
    cache = _get_symbol_mark_frame(symbol, timeframe)
    if cache.empty or "close" not in cache.columns:
        return (
            np.empty(0, dtype="int64"),
            np.empty(0, dtype="float64"),
        )
    valid = (
        cache["datetime"].notna()
        & cache["close"].notna()
        & (cache["close"] > 0)
    )
    closes = (
        cache.loc[valid, ["datetime", "close"]]
        .drop_duplicates(subset=["datetime"], keep="last")
        .sort_values("datetime")
    )
    if closes.empty:
        return (
            np.empty(0, dtype="int64"),
            np.empty(0, dtype="float64"),
        )
    availability_ns = (closes["datetime"] + pd.Timedelta(hours=1)).to_numpy(
        dtype="datetime64[ns]",
    ).astype("int64")
    return (
        np.ascontiguousarray(availability_ns),
        np.ascontiguousarray(closes["close"].to_numpy(dtype="float64")),
    )


def _prewarm_mark_frames(symbols: list[str], timeframe: str = "1h") -> None:
    """Populate the parent-side compact mark cache before forking workers.

    Fork children inherit the warmed compact arrays copy-on-write, so the three
    books and the anchored folds share one set of validated
    ``(availability_ns, close)`` arrays instead of each process re-reading its
    own copy. Missing mark parquet files stay skipped (the window path applies
    the same existence semantics for non-roster symbols).
    """
    for sym in symbols:
        if os.path.exists(_futures_collection._mark_price_path(sym, timeframe)):
            _compact_mark_series(sym, timeframe)


def _contemporaneous_mark_close_panel(
    symbols: list[str],
    grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Contemporaneous mark-price close panel (no +1h shift, no ffill).

    An absent mark stays NaN so the parity mask fails open per I2.  Deliberately
    does NOT apply ``_cached_mark_panel``'s ``+1h`` availability shift — the gate
    detects a stalled price feed, not the replay's valuation lag.
    """
    panel = pd.DataFrame(index=grid, columns=list(symbols), dtype="float64")
    for sym in symbols:
        try:
            cache = _get_symbol_mark_frame(sym, "1h")
        except (KeyError, ValueError):
            # A malformed/incomplete mark cache (e.g. missing the
            # open/high/low columns DataCollector._load_mark_price_cache
            # unconditionally coerces) is a data-integrity condition the
            # existing mark gates (_assert_cache_required_marks,
            # apply_dynamic_mark_gap_exclusion) already own; this parity
            # gate stays fail-open per I2 rather than pre-empting them with
            # an unrelated crash.
            continue
        if cache.empty or "close" not in cache.columns:
            continue
        valid = (
            cache["datetime"].notna()
            & cache["close"].notna()
            & (cache["close"] > 0)
        )
        closes = (
            cache.loc[valid, ["datetime", "close"]]
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


def _cached_mark_panel(
    roster: list[str],
    timeframe: str,
    minute_grid: pd.DatetimeIndex,
    max_stale_hours: int,
) -> pd.DataFrame:
    """Build the causal mark-price panel from per-symbol cached mark series.

    Element-for-element equivalent to
    ``DataCollector.load_mark_price_panel`` (same validation, same ``+1h``
    availability shift, same ``ffill`` limit, same NaN for absent/non-finite/
    non-positive marks) but sources each symbol from the process-level
    ``_compact_mark_series`` cache instead of re-running the frame prologue per
    window. The returned frame has exactly ``minute_grid`` as its index and
    exactly ``roster`` as its column order.
    """
    if timeframe != "1h":
        raise ValueError(f"unsupported timeframe '{timeframe}'")
    if max_stale_hours < 0:
        raise ValueError("max_stale_hours must be non-negative")
    if not isinstance(minute_grid, pd.DatetimeIndex) or minute_grid.empty:
        raise DataIntegrityError("grid must be a non-empty DatetimeIndex")
    if minute_grid.tz is None:
        raise DataIntegrityError("grid must be tz-aware UTC")
    if not minute_grid.is_monotonic_increasing or minute_grid.has_duplicates:
        raise DataIntegrityError("grid must be monotonically increasing with no duplicates")
    if not roster:
        raise DataIntegrityError("roster must be non-empty")
    if len(set(roster)) != len(roster):
        raise DataIntegrityError("roster must be unique")

    panel = pd.DataFrame(index=minute_grid, columns=list(roster), dtype="float64")
    if len(minute_grid) > 1:
        step = minute_grid[1] - minute_grid[0]
        step_minutes = step / pd.Timedelta(minutes=1)
        if step_minutes <= 0 or 60 % step_minutes != 0:
            raise DataIntegrityError(
                "grid frequency must be a positive divisor of one hour"
            )
        if max_stale_hours == 0:
            ffill_limit = int(60 // step_minutes - 1)
        else:
            ffill_limit = int(max_stale_hours * 60 // step_minutes - 1)
    else:
        ffill_limit = 0
    grid_ns = np.asarray(minute_grid, dtype="datetime64[ns]").astype("int64")
    n_grid = len(grid_ns)
    grid_pos = np.arange(n_grid, dtype=np.intp)
    for sym in roster:
        avail_ns, close_arr = _compact_mark_series(sym, timeframe)
        if avail_ns.size == 0:
            continue
        # reindex() places a source value only where its timestamp EXACTLY
        # equals a grid stamp; method='ffill', limit=L then carries it into at
        # most L consecutive missing grid rows. Reproduced exactly: fill
        # origins are exact matches only, carried while i - last_match <= L.
        hit = np.searchsorted(avail_ns, grid_ns, side="left")
        hit_clipped = np.minimum(hit, len(avail_ns) - 1)
        matched = (hit < len(avail_ns)) & (avail_ns[hit_clipped] == grid_ns)
        last_match = np.maximum.accumulate(np.where(matched, grid_pos, -1))
        carry_idx = np.maximum(last_match, 0)
        values = np.where(matched, close_arr[hit_clipped], np.nan)
        filled = values[carry_idx]
        ok = (
            matched
            if ffill_limit == 0
            else (last_match >= 0) & ((grid_pos - last_match) <= ffill_limit)
        )
        panel[sym] = np.where(ok, filled, np.nan)
    return panel


def _load_window_minute_frames(
    root: str,
    symbols: list[str],
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
    timeframe: Literal["1m", "3m", "5m"],
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
    for sym in symbols:
        path = os.path.join(root, timeframe, f"{sym}.parquet")
        if not os.path.exists(path):
            continue
        table = pq.read_table(
            path,
            columns=["timestamp", "high", "low", "close"],
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
                for c in ("high", "low", "close")
            },
            index=idx,
        )
        frame = frame[(frame.index >= grid_start) & (frame.index <= grid_end)]
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        if not frame.empty:
            frames[sym] = frame
    return frames


def _build_window_frames(
    symbol_frames: dict[str, pd.DataFrame],
    roster: list[str],
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
    minute_grid: pd.DatetimeIndex,
    timeframe: Literal["1m", "3m", "5m"],
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
    frames: dict[str, pd.DataFrame], timeframe: Literal["1m", "3m", "5m"],
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
        freq={"1m": "1min", "3m": "3min", "5m": "5min"}[timeframe],
        tz="UTC",
    )
    highs = pd.DataFrame({s: f["high"] for s, f in frames.items()}).reindex(grid)
    lows = pd.DataFrame({s: f["low"] for s, f in frames.items()}).reindex(grid)
    closes = pd.DataFrame({s: f["close"] for s, f in frames.items()}).reindex(grid)
    return highs, lows, closes
