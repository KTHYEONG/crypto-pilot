"""Frozen point-in-time liquidity roster for MHS research."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError

_LIQUIDITY_FLOOR_USD = 1_000_000.0


def build_frozen_pit_roster(
    daily_close: pd.DataFrame,
    daily_quote_volume: pd.DataFrame,
    census_symbols: tuple[str, ...],
    *,
    breadth: int,
) -> pd.DataFrame:
    """Return a point-in-time daily liquidity roster with no future constituent knowledge.

    The roster is the investable universe for a frozen research strategy.  It
    admits only symbols whose prior completed observations satisfy the
    registered liquidity and trading-history requirements, so later listings,
    delistings, and volume changes cannot alter an earlier decision.

    Args:
        daily_close: Complete UTC daily close census in canonical symbol order.
        daily_quote_volume: Matching completed daily USD quote turnover census.
        census_symbols: Complete historical exchange symbol order.
        breadth: Positive maximum number of liquidity-ranked members.
    Returns:
        Boolean decision-day roster in the supplied canonical column order.
    Raises:
        DataIntegrityError: Inputs are not a complete aligned UTC PIT census.
        ValueError: ``breadth`` is not a positive integer.
    """
    if isinstance(breadth, bool) or not isinstance(breadth, int) or breadth <= 0:
        raise ValueError(f"breadth must be a positive integer, got {breadth!r}")
    census = list(census_symbols)
    if len(census) == 0 or any(not isinstance(s, str) or not s for s in census) or len(set(census)) != len(census):
        raise DataIntegrityError("census_symbols must be a non-empty tuple of unique non-empty symbols")
    if (
        list(daily_close.columns) != census
        or list(daily_quote_volume.columns) != census
        or not daily_close.index.equals(daily_quote_volume.index)
    ):
        raise DataIntegrityError("daily inputs must share an identical index and census-ordered columns")
    idx = daily_close.index
    if not isinstance(idx, pd.DatetimeIndex) or not _is_utc_midnight_daily(idx):
        raise DataIntegrityError("daily index must be unique increasing UTC-midnight days without gaps")
    close_vals = daily_close.to_numpy(dtype="float64", na_value=np.nan)
    qv_vals = daily_quote_volume.to_numpy(dtype="float64", na_value=np.nan)
    close_bad = ~np.isnan(close_vals) & ~(np.isfinite(close_vals) & (close_vals > 0.0))
    qv_bad = ~np.isnan(qv_vals) & ~(np.isfinite(qv_vals) & (qv_vals >= 0.0))
    if bool(close_bad.any()) or bool(qv_bad.any()):
        raise DataIntegrityError("observed closes must be positive finite and turnover nonnegative finite")
    traded = daily_close.notna() & daily_quote_volume.notna() & (daily_quote_volume > 0.0)
    traded_count = traded.astype(np.int64).rolling(90, min_periods=1).sum()
    median_turnover = daily_quote_volume.rolling(30, min_periods=30).median()
    eligible_src = traded & (traded_count >= 85) & (median_turnover >= _LIQUIDITY_FLOOR_USD)
    median_src = median_turnover.where(eligible_src)
    med_mat = median_src.to_numpy(dtype="float64", na_value=np.nan)
    elig_mat = eligible_src.to_numpy(dtype=bool)
    out = np.zeros((len(idx), len(census)), dtype=bool)
    for i in range(1, len(idx)):
        elig = elig_mat[i - 1]
        if not bool(elig.any()):
            continue
        meds = med_mat[i - 1]
        order = np.argsort(-np.where(elig, meds, -np.inf), kind="stable")
        for j in order[: int(breadth)]:
            if not bool(elig[int(j)]):
                break
            out[i, int(j)] = True
    return pd.DataFrame(out, index=idx, columns=census, dtype=bool)


def _is_utc_midnight_daily(idx: pd.DatetimeIndex) -> bool:
    """Check unique increasing UTC-midnight daily continuity without gaps."""
    if idx.tz is None:
        return False
    utc = idx.tz_convert("UTC")
    if not idx.equals(utc):
        return False
    if bool((idx.normalize() != idx).any()):
        return False
    if bool(idx.duplicated().any()) or not bool(idx.is_monotonic_increasing):
        return False
    return not bool(((idx[1:] - idx[:-1]) != pd.Timedelta(days=1)).any())
