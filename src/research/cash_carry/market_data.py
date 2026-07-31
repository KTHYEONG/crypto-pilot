from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import borrow_path, funding_path, ohlcv_path, spot_ohlcv_path
from src.common.errors import DataIntegrityError
from src.common.logging import setup_logger
from src.market_data.storage.loaders import load_funding_rates, load_ohlcv_1h_as_4h
from src.research.cash_carry.contracts import CarryMarketData

_logger = setup_logger("CarryData")

_OHLCV_COLUMNS = ("open", "high", "low", "close")
_FUNDING_GAP_JITTER_TOLERANCE = pd.Timedelta(milliseconds=50)


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
    allowed_funding_gap = max_funding_gap + _FUNDING_GAP_JITTER_TOLERANCE
    if len(event_gaps) > 0 and event_gaps.max() > allowed_funding_gap:
        raise DataIntegrityError(
            f"funding gap > {allowed_funding_gap} detected between events"
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


def _load_borrow_events(path: str | Path) -> pd.DataFrame:
    """Load canonical quote-borrow events (``timestamp``, ``borrow_rate``, ``accrual_seconds``).

    ``borrow_rate`` is a decimal cost accruing over exactly ``accrual_seconds``
    beginning at ``timestamp``. Ambiguous-unit exports (no ``accrual_seconds``)
    are rejected rather than inferred.
    """
    p = Path(path)
    if not p.exists():
        raise DataIntegrityError(f"borrow path does not exist: {path}")
    df = pd.read_parquet(p)
    if "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True)
    else:
        raise DataIntegrityError("borrow parquet must contain a 'timestamp' or 'datetime' column")
    if "borrow_rate" not in df.columns:
        raise DataIntegrityError("borrow parquet must contain a 'borrow_rate' column")
    if "accrual_seconds" not in df.columns:
        raise DataIntegrityError(
            "borrow parquet must contain an 'accrual_seconds' column; the source cadence "
            "cannot be inferred from an ambiguous export"
        )
    if ts.dt.tz is None or ts.dt.tz.utcoffset(None) is None:
        raise DataIntegrityError("borrow timestamps must be tz-aware UTC")
    rates = pd.to_numeric(df["borrow_rate"], errors="coerce").astype("float64")
    accrual = pd.to_numeric(df["accrual_seconds"], errors="coerce").astype("float64")
    events = pd.DataFrame({"ts": ts, "borrow_rate": rates, "accrual_seconds": accrual})
    events = events[events["ts"].notna()]
    events = events.sort_values("ts").reset_index(drop=True)
    return events


def _borrow_events_to_per_bar(
    events: pd.DataFrame,
    grid: pd.DatetimeIndex,
    bar_period: pd.Timedelta,
) -> pd.Series:
    """Convert borrow events into a finite per-4h-bar rate on the exact grid.

    Each bar receives the overlap-weighted share of every event covering it, so
    an event accruing over ``accrual_seconds`` never forward-fills beyond its
    declared interval. Rejects duplicate, overlapping, non-positive-duration,
    uncovered, non-finite, and ambiguous rows fail-closed.
    """
    if len(grid) == 0:
        raise DataIntegrityError("borrow conversion requires a non-empty bar grid")
    ts = events["ts"]
    rates = events["borrow_rate"].to_numpy(dtype=np.float64)
    accrual = events["accrual_seconds"].to_numpy(dtype=np.float64)
    if not np.isfinite(rates).all() or not np.isfinite(accrual).all():
        raise DataIntegrityError("borrow events must be finite")
    if (accrual <= 0).any():
        raise DataIntegrityError("borrow accrual_seconds must be > 0")
    if ts.duplicated().any():
        raise DataIntegrityError("borrow events must not contain duplicates")
    if not ts.is_monotonic_increasing:
        raise DataIntegrityError("borrow events must be monotonic in time")
    ends = ts + pd.to_timedelta(accrual, unit="s")
    if len(ts) > 1 and (ts.iloc[1:].to_numpy() < ends.iloc[:-1].to_numpy()).any():
        raise DataIntegrityError("borrow events must not overlap")

    bar_ns = int(bar_period / pd.Timedelta("1ns"))
    bar_starts = grid.to_numpy(dtype="datetime64[ns]")
    bar_ends = bar_starts + bar_ns
    ev_starts = ts.to_numpy(dtype="datetime64[ns]")
    ev_ends = ends.to_numpy(dtype="datetime64[ns]")
    ev_ns = accrual * 1_000_000_000.0

    per_bar = np.zeros(len(grid), dtype=np.float64)
    for i in range(len(grid)):
        overlap_ns = np.minimum(bar_ends[i], ev_ends) - np.maximum(bar_starts[i], ev_starts)
        overlap_ns = np.maximum(overlap_ns, 0).astype(np.float64)
        covered = float(overlap_ns.sum())
        if not np.isclose(covered, float(bar_ns)):
            raise DataIntegrityError(
                f"borrow coverage incomplete at {pd.Timestamp(bar_starts[i])}: "
                f"covered {covered} of {bar_ns} ns"
            )
        per_bar[i] = float(np.sum(rates * overlap_ns / ev_ns))
    return pd.Series(per_bar, index=grid, name="borrow_rate")


def _load_borrow_rates(path: str | Path, grid: pd.DatetimeIndex, bar_period: pd.Timedelta) -> pd.Series:
    """Load canonical borrow events and convert them to per-bar rates on ``grid``."""
    events = _load_borrow_events(path)
    return _borrow_events_to_per_bar(events, grid, bar_period)


def load_carry_market_data(
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
) -> CarryMarketData:
    """Load and fail-closed validate the four cash-and-carry inputs for one symbol.

    Loads exact spot/perp 1h coverage, resamples to an identical 4h grid,
    aligns the actual funding settlement events (gap at most eight hours), and
    converts quote-borrow events to per-4h-bar rates. Every missing input or
    uncovered borrow bar raises ``DataIntegrityError``; nothing is zero-filled.
    """
    spot_p = spot_ohlcv_path(symbol, "1h")
    perp_p = ohlcv_path(symbol, "1h")
    fund_p = funding_path(symbol)
    borrow_p = borrow_path(symbol)
    for path, name in [(spot_p, "spot"), (perp_p, "perp"), (fund_p, "funding"), (borrow_p, "borrow")]:
        if not path.exists():
            raise DataIntegrityError(f"{name} data missing for {symbol}: {path}")

    spot = load_ohlcv_1h_as_4h(spot_p, start=start, end=end)
    perp = load_ohlcv_1h_as_4h(perp_p, start=start, end=end)
    if len(spot) < 2:
        raise DataIntegrityError(f"spot data has fewer than 2 bars for {symbol}")
    period = spot.index[1] - spot.index[0]
    window_end = spot.index[-1] + period
    funding = load_funding_rates(str(fund_p))
    funding = funding[(funding.index >= spot.index[0]) & (funding.index < window_end)]
    borrow = _load_borrow_rates(str(borrow_p), spot.index, period)

    market_data = CarryMarketData(symbol=symbol, spot=spot, perp=perp, funding=funding, borrow=borrow)
    validate_carry_market_data(market_data)
    return market_data


def _check_contract() -> None:
    """Executable assertions locking the frozen carry data surface."""
    assert validate_carry_market_data.__name__ == "validate_carry_market_data"
    assert {f.name for f in CarryMarketData.__dataclass_fields__.values()} == {
        "symbol", "spot", "perp", "funding", "borrow",
    }
    assert validate_carry_market_data.__defaults__ == (pd.Timedelta(hours=8),)
    assert load_carry_market_data.__name__ == "load_carry_market_data"


_check_contract()
