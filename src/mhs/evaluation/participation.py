from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.mhs.execution import StrategyExecutionReplayResult


def _load_symbol_quote_volume(
    root: str,
    symbol: str,
    timeframe: Literal["1m", "3m", "5m"],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    """Load one symbol's ``quote_vol`` over ``[start, end]`` on demand.

    Reads only the ``timestamp``/``quote_vol`` columns so participation metrics
    never retain a wide quote-volume panel alongside the minute OHLCV frames.
    Returns ``None`` when the symbol has no data (the same absent-data behavior
    as a symbol missing from the historical wide panel).
    """
    path = os.path.join(root, timeframe, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None
    start_ms = int(start.value // 1_000_000)
    end_ms = int(end.value // 1_000_000)
    table = pq.read_table(
        path,
        columns=["timestamp", "quote_vol"],
        filters=[[("timestamp", ">=", start_ms), ("timestamp", "<=", end_ms)]],
    )
    idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
    series = pd.Series(
        table.column("quote_vol").to_numpy().astype("float64"), index=idx,
    )
    series = series[(series.index >= start) & (series.index <= end)]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        return None
    return series


def _participation_warnings(
    replay: StrategyExecutionReplayResult,
    root: str,
    timeframe: Literal["1m", "3m", "5m"],
    symbols: list[str],
    minute_grid: pd.DatetimeIndex,
) -> dict[str, float]:
    if replay.simulated_fills.empty:
        return {}
    fills = replay.simulated_fills
    notional = float((fills["quantity_delta"].abs() * fills["fill_price"]).sum())
    fills_by_symbol: dict[str, pd.DataFrame] = {}
    for _sym, group in fills.groupby("symbol"):
        fills_by_symbol[str(_sym)] = group
    daily_volume = 0.0
    window_totals: dict[str, float] = {"1m": 0.0, "30m": 0.0}
    window_minutes = (("1m", 1), ("30m", 30))
    for sym in symbols:
        series = _load_symbol_quote_volume(
            root, sym, timeframe, minute_grid[0], minute_grid[-1],
        )
        if series is None:
            continue
        daily_volume += float(series.sum())
        group = fills_by_symbol.get(sym)
        if group is None:
            continue
        # Locate each fill's ``[t, t+window]`` inclusive span with two
        # ``searchsorted`` lookups (instead of an ``iterrows``/``.loc`` slice)
        # and sum the small bounded window with ``np.add.reduce``, which is
        # bit-identical to pandas ``Series.sum()`` over the same labels.
        vol = series.to_numpy(dtype="float64")
        idx = series.index.to_numpy(dtype="datetime64[ns]")
        t_arr = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        start_pos = np.searchsorted(idx, t_arr, side="left")
        in_idx = (start_pos < len(idx)) & (idx[start_pos] == t_arr)
        valid_pos = start_pos[in_idx]
        for window_label, minutes in window_minutes:
            ends = t_arr[in_idx] + np.timedelta64(minutes, "m")
            end_pos = np.searchsorted(idx, ends, side="right") - 1
            # Accumulate directly into the running total in fill order -- the
            # baseline's flat ``window_totals += window_sum`` chain -- so the
            # float addition sequence is bit-identical to the iterrows baseline.
            for p, e in zip(valid_pos, end_pos, strict=True):
                window_totals[window_label] += float(np.add.reduce(vol[p : e + 1]))
    warnings: dict[str, float] = {}
    for window_label, _minutes in window_minutes:
        total_volume = window_totals[window_label]
        warnings[f"fill_notional_to_{window_label}_quote_volume"] = (
            notional / total_volume if total_volume > 0 else float("nan")
        )
    warnings["daily_trade_notional_to_daily_quote_volume"] = (
        notional / daily_volume if daily_volume > 0 else float("nan")
    )
    return warnings










