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
    blocked_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a point-in-time daily liquidity roster with no future constituent knowledge.

    A symbol loses its seat only on the decision days whose execution window overlaps an
    evidenced source gap. Outside those days it competes normally, because the exchange
    recorded it normally and discarding its whole history would rewrite the investable
    universe that actually existed.

    Args:
        daily_close: Complete historical daily close census for PIT membership.
        daily_quote_volume: Matching completed daily USD quote turnover census.
        census_symbols: Complete historical exchange symbol order.
        breadth: Positive maximum number of liquidity-ranked members.
        blocked_decisions: Boolean frame sharing the daily index and census columns; True
            withdraws that symbol's seat for that decision day only.
    Returns:
        Boolean decision-day roster in the supplied canonical column order.
    Raises:
        DataIntegrityError: Inputs disagree on index, columns, or observed value domain.
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
    blocked_mat = _blocked_matrix(blocked_decisions, idx, census)
    out = np.zeros((len(idx), len(census)), dtype=bool)
    for i in range(1, len(idx)):
        elig = elig_mat[i - 1].copy()
        if blocked_mat is not None:
            elig = elig & ~blocked_mat[i]
        if not bool(elig.any()):
            continue
        meds = med_mat[i - 1]
        order = np.argsort(-np.where(elig, meds, -np.inf), kind="stable")
        ranked = [int(j) for j in order if bool(elig[int(j)])][: int(breadth)]
        out[i, ranked] = True
    return pd.DataFrame(out, index=idx, columns=census, dtype=bool)


def _blocked_matrix(
    blocked_decisions: pd.DataFrame | None, idx: pd.DatetimeIndex, census: list[str]
) -> np.ndarray | None:
    """Validate the decision-day block frame and return its boolean matrix."""
    if blocked_decisions is None:
        return None
    if not isinstance(blocked_decisions, pd.DataFrame):
        raise DataIntegrityError("blocked_decisions must be a boolean frame or None")
    if not blocked_decisions.index.equals(idx) or list(blocked_decisions.columns) != census:
        raise DataIntegrityError("blocked_decisions must share the daily index and census column order")
    if not bool((blocked_decisions.dtypes == "bool").all()):
        raise DataIntegrityError("blocked_decisions must contain only boolean values")
    return np.asarray(blocked_decisions.to_numpy(dtype=bool), dtype=bool)


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
