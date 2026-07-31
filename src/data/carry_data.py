from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.logging_setup import setup_logger
from src.data.loader import DataIntegrityError

_logger = setup_logger("CarryData")

_OHLCV_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class CarryMarketData:
    """Aligned cash-and-carry research inputs for one symbol.

    ``spot`` and ``perp`` are identical-grid tz-aware OHLCV frames of the same
    asset; ``funding`` carries the actual settlement timestamps of the perpetual
    short leg (variable sub-eight-hour cadence is allowed, never a fixed
    three-events-per-day assumption); ``borrow`` holds the per-bar finite
    quote-cash financing rate. The grid is the spot index.
    """

    symbol: str
    spot: pd.DataFrame
    perp: pd.DataFrame
    funding: pd.Series
    borrow: pd.Series


def _validate_ohlcv_grid(df: pd.DataFrame, name: str) -> pd.DatetimeIndex:
    if not isinstance(df, pd.DataFrame):
        raise DataIntegrityError(f"{name} must be a DataFrame, got {type(df).__name__}")
    missing = set(_OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise DataIntegrityError(f"{name} missing columns: {sorted(missing)}")
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise DataIntegrityError(f"{name} index must be a DatetimeIndex")
    if index.tz is None:
        raise DataIntegrityError(f"{name} index must be tz-aware UTC")
    if not index.is_monotonic_increasing:
        raise DataIntegrityError(f"{name} index must be monotonic increasing")
    if index.has_duplicates:
        raise DataIntegrityError(f"{name} index must not contain duplicates")
    if len(index) < 2:
        raise DataIntegrityError(f"{name} must contain at least 2 bars")
    diffs = index.to_series().diff().dropna()
    period = diffs.iloc[0]
    if period <= pd.Timedelta(0):
        raise DataIntegrityError(f"{name} grid must be strictly increasing")
    gaps = diffs[diffs != period]
    if not gaps.empty:
        raise DataIntegrityError(
            f"{name} missing bars detected at {gaps.index[0]} (expected {period} grid)"
        )
    for col in _OHLCV_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise DataIntegrityError(f"{name} {col} must be finite")
        if (values <= 0).any():
            raise DataIntegrityError(f"{name} {col} must be strictly positive")
    return index


def validate_carry_market_data(
    data: CarryMarketData,
    max_funding_gap: pd.Timedelta = pd.Timedelta(hours=8),  # noqa: B008
) -> None:
    """Fail-closed integrity gate for cash-and-carry research inputs.

    Raises ``DataIntegrityError`` for non-UTC, duplicate, non-monotonic,
    non-finite, non-positive, incomplete, or funding-gap inputs. A missing
    funding or borrow observation is never filled with zero: it always blocks
    the backtest and the promotion evaluation. ``data`` is never mutated.
    """
    if not isinstance(data, CarryMarketData):
        raise TypeError(f"data must be a CarryMarketData, got {type(data).__name__}")
    if not data.symbol:
        raise DataIntegrityError("symbol must not be empty")
    if max_funding_gap <= pd.Timedelta(0):
        raise ValueError(f"max_funding_gap must be > 0, got {max_funding_gap}")

    grid = _validate_ohlcv_grid(data.spot, "spot")
    _validate_ohlcv_grid(data.perp, "perp")
    if not data.spot.index.equals(data.perp.index):
        raise DataIntegrityError("spot and perpetual grids must be identical after alignment")
    period = grid[1] - grid[0]
    window_end = grid[-1] + period

    if len(data.funding) == 0:
        raise DataIntegrityError("funding must contain at least one settled event")
    funding = pd.to_numeric(data.funding, errors="coerce").astype("float64")
    fts = pd.DatetimeIndex(data.funding.index)
    if fts.tz is None:
        raise DataIntegrityError("funding index must be tz-aware UTC")
    if fts.tz != grid.tz:
        raise DataIntegrityError("funding index timezone must match the bar grid")
    if fts.has_duplicates:
        raise DataIntegrityError("funding index must not contain duplicates")
    if not fts.is_monotonic_increasing:
        raise DataIntegrityError("funding index must be monotonic increasing")
    if not np.isfinite(funding.to_numpy(dtype=np.float64)).all():
        raise DataIntegrityError("funding rates must be finite")
    if not ((fts >= grid[0]) & (fts < window_end)).all():
        raise DataIntegrityError("funding events outside the bar window")
    event_gaps = fts.to_series().diff().dropna()
    if len(event_gaps) > 0 and event_gaps.max() > max_funding_gap:
        raise DataIntegrityError(
            f"funding gap > {max_funding_gap} detected between events"
        )
    if fts[0] > grid[0] + max_funding_gap:
        raise DataIntegrityError("funding coverage missing at window start")
    if fts[-1] < window_end - max_funding_gap:
        raise DataIntegrityError("funding coverage missing at window end")

    if len(data.borrow) == 0:
        raise DataIntegrityError("borrow must contain a finite rate for every bar")
    if not data.borrow.index.equals(grid):
        raise DataIntegrityError("borrow index must match the bar grid exactly")
    borrow = pd.to_numeric(data.borrow, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(borrow).all():
        raise DataIntegrityError("borrow rates must be finite")

    _logger.info(
        "carry data symbol=%s bars=%d funding_events=%d grid=%s",
        data.symbol, len(grid), len(funding), grid[0].isoformat(),
        extra={"tag": "DATA"},
    )


def _check_contract() -> None:
    """Executable assertions locking the frozen carry data surface."""
    assert validate_carry_market_data.__name__ == "validate_carry_market_data"
    assert {f.name for f in CarryMarketData.__dataclass_fields__.values()} == {
        "symbol", "spot", "perp", "funding", "borrow",
    }
    assert validate_carry_market_data.__defaults__ == (pd.Timedelta(hours=8),)


_check_contract()
