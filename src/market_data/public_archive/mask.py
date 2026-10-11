from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.catalog import is_perpetual_usdt
from src.strategy.universe import build_pit_roster

_QUOTE_CANDIDATES: tuple[str, ...] = ("quote_volume", "quote_vol")


def _coerce_ts(raw: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(raw):
        return pd.to_datetime(pd.to_numeric(raw, errors="coerce"), unit="ms", utc=True)
    return pd.to_datetime(raw, utc=True, errors="coerce")


def load_daily_census(ohlcv_1h_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build gap-free daily close and quote-volume census from the 1h lake."""
    root = Path(ohlcv_1h_dir)
    if not root.is_dir():
        raise DataIntegrityError(f"missing ohlcv 1h lake: {root}")
    paths = sorted(root.glob("*.parquet"))
    symbols: list[str] = []
    for path in paths:
        name = path.name
        if name.startswith(".") or name.endswith(".tmp.parquet"):
            continue
        symbol = path.stem
        if not is_perpetual_usdt(symbol):
            continue
        symbols.append(symbol)
    symbols = sorted(set(symbols))
    if not symbols:
        raise DataIntegrityError(f"empty ohlcv 1h lake: {root}")
    per_close: dict[str, pd.Series] = {}
    per_qv: dict[str, pd.Series] = {}
    min_day: date | None = None
    max_day: date | None = None
    for symbol in symbols:
        frame = pd.read_parquet(root / f"{symbol}.parquet")
        if frame.empty or "close" not in frame.columns or "timestamp" not in frame.columns:
            continue
        ts = _coerce_ts(frame["timestamp"])
        close = pd.to_numeric(frame["close"], errors="coerce")
        qv_col = next((c for c in _QUOTE_CANDIDATES if c in frame.columns), None)
        qv = pd.to_numeric(frame[qv_col], errors="coerce") if qv_col is not None else pd.Series(np.nan, index=frame.index)
        valid = ts.notna()
        ts = ts[valid]
        close = close[valid]
        qv = qv[valid]
        if ts.empty:
            continue
        order = np.argsort(ts.to_numpy())
        ts = ts.iloc[order].reset_index(drop=True)
        close = close.iloc[order].reset_index(drop=True)
        qv = qv.iloc[order].reset_index(drop=True)
        days = ts.dt.normalize()
        close_last = close.groupby(days).last()
        qv_sum = qv.groupby(days).sum(min_count=1)
        per_close[symbol] = close_last
        per_qv[symbol] = qv_sum
        lo, hi = close_last.index.min(), close_last.index.max()
        if min_day is None or lo.date() < min_day:
            min_day = lo.date()
        if max_day is None or hi.date() > max_day:
            max_day = hi.date()
    if min_day is None or max_day is None:
        raise DataIntegrityError(f"empty ohlcv 1h lake: {root}")
    index = pd.DatetimeIndex(pd.date_range(min_day, max_day, freq="D", tz="UTC"), tz="UTC")
    close_cols: dict[str, np.ndarray] = {}
    qv_cols: dict[str, np.ndarray] = {}
    for symbol in symbols:
        series_c = per_close.get(symbol)
        series_q = per_qv.get(symbol)
        aligned_c = series_c.reindex(index) if series_c is not None else pd.Series(np.nan, index=index)
        aligned_q = series_q.reindex(index) if series_q is not None else pd.Series(np.nan, index=index)
        close_cols[symbol] = aligned_c.astype("float64").to_numpy()
        qv_cols[symbol] = aligned_q.astype("float64").to_numpy()
    close_frame = pd.DataFrame(close_cols, index=index, dtype="float64")
    qv_frame = pd.DataFrame(qv_cols, index=index, dtype="float64")
    return close_frame, qv_frame


def seat_day_mask(
    daily_close: pd.DataFrame,
    daily_quote_volume: pd.DataFrame,
    *,
    breadth: int,
    preroll_days: int,
) -> pd.DataFrame:
    """Expand the PIT roster backwards so feature lookback days are collected."""
    if isinstance(breadth, bool) or not isinstance(breadth, int) or breadth < 1:
        raise ValueError(f"breadth must be a positive integer, got {breadth!r}")
    if isinstance(preroll_days, bool) or not isinstance(preroll_days, int) or preroll_days < 0:
        raise ValueError(f"preroll_days must be a non-negative integer, got {preroll_days!r}")
    seat = build_pit_roster(daily_close, daily_quote_volume, tuple(daily_close.columns), breadth=breadth)
    if preroll_days == 0:
        return seat
    mat = seat.to_numpy(dtype=bool)
    out = mat.copy()
    n = mat.shape[0]
    for k in range(1, preroll_days + 1):
        if k >= n:
            break
        out[:-k] |= mat[k:]
    return pd.DataFrame(out, index=seat.index, columns=seat.columns, dtype=bool)


def allowed_days(mask: pd.DataFrame, *, first: date, last: date) -> dict[str, frozenset[date]]:
    """Restrict the seat mask to a closed date range with lake-lag carry-forward."""
    if first > last:
        return {}
    if mask.empty:
        return {}
    idx = mask.index
    row_of: dict[date, int] = {ts.date(): i for i, ts in enumerate(idx)}
    last_day = idx[-1].date()
    last_row = mask.iloc[-1]
    mat = mask.to_numpy(dtype=bool)
    result: dict[str, frozenset[date]] = {}
    days = (last - first).days
    for j, symbol in enumerate(mask.columns):
        seated_last = bool(last_row.iloc[j])
        kept: set[date] = set()
        for offset in range(days + 1):
            day = first + timedelta(days=offset)
            pos = row_of.get(day)
            if pos is not None:
                if bool(mat[pos, j]):
                    kept.add(day)
            elif day > last_day and seated_last:
                kept.add(day)
        if kept:
            result[str(symbol)] = frozenset(kept)
    return result
