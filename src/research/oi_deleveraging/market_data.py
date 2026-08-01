from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import funding_path, metrics_path, ohlcv_path
from src.common.errors import DataIntegrityError
from src.common.logging import setup_logger
from src.market_data.storage.loaders import load_funding_rates, load_ohlcv_1h_as_4h
from src.research.oi_deleveraging.contracts import OIDeleveragingMarketData

_logger = setup_logger("OIDeleveragingData")

_OHLCV_COLUMNS = ("open", "high", "low", "close")
_METRICS_REQUIRED_COLUMNS = (
    "timestamp",
    "datetime",
    "available_at",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "long_short_ratio",
    "top_trader_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
_METRICS_JOINED_COLUMNS = (
    "feature_datetime",
    "feature_available_at",
    "feature_sum_open_interest",
    "feature_sum_open_interest_value",
    "feature_long_short_ratio",
    "feature_top_trader_long_short_ratio",
    "feature_sum_taker_long_short_vol_ratio",
    "feature_oi_value_change",
)
_JOINED_REQUIRED_COLUMNS = (
    "decision_time",
    "mark_return_24h",
    "feature_datetime",
    "feature_available_at",
    "feature_sum_open_interest",
    "feature_sum_open_interest_value",
    "feature_long_short_ratio",
    "feature_top_trader_long_short_ratio",
    "feature_sum_taker_long_short_vol_ratio",
    "feature_oi_value_change",
)


def _validate_4h_grid(df: pd.DataFrame, name: str) -> pd.DatetimeIndex:
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


def _load_metrics_frame(symbol: str) -> pd.DataFrame:
    """Load and fail-closed validate the canonical daily metrics frame."""
    path = metrics_path(symbol)
    if not path.exists():
        raise DataIntegrityError(f"metrics data missing for {symbol}: {path}")
    df = pd.read_parquet(path)
    missing = set(_METRICS_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DataIntegrityError(
            f"metrics frame for {symbol} missing canonical columns: {sorted(missing)}"
        )
    df = df.loc[:, list(_METRICS_REQUIRED_COLUMNS)].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    if df["datetime"].dt.tz is None:
        raise DataIntegrityError(f"metrics datetimes for {symbol} must be tz-aware UTC")
    df = df.dropna(subset=["datetime", "available_at", "timestamp"])
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    if not df["datetime"].is_monotonic_increasing:
        raise DataIntegrityError(f"metrics timestamps for {symbol} are not monotonic")
    df = df.sort_values("timestamp").reset_index(drop=True)
    lag = (df["available_at"] - df["datetime"]).abs()
    if lag.gt(pd.Timedelta(minutes=5)).any():
        raise DataIntegrityError(
            f"metrics available_at for {symbol} must equal datetime + 5 minutes"
        )
    return df


def load_metrics_asof(
    symbol: str,
    bars: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    """Causally join canonical daily metrics onto the 4h bar grid.

    Each 4h decision timestamp (the bar's close) selects the latest daily metric
    whose ``available_at <= decision_time`` via ``merge_asof(direction='backward')``.
    A metric row released after the decision is never visible, and a missing
    metric leaves a no-signal interval rather than an imputed feature. The
    output carries ``decision_time`` plus ``feature_datetime`` and
    ``feature_available_at`` for audit, and raises ``DataIntegrityError`` for a
    future feature timestamp, non-monotonic feature timestamps, or a missing
    required column.
    """
    grid = _validate_4h_grid(bars, "bars")
    period = grid[1] - grid[0]
    last_decision = grid[-1] + period

    metrics = _load_metrics_frame(symbol)
    metrics = metrics[metrics["available_at"] <= last_decision].copy()
    if metrics.empty:
        daily_join = pd.DataFrame(columns=list(_METRICS_JOINED_COLUMNS))
    else:
        daily = (
            metrics.sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
        )
        daily["prev_oi_value"] = daily["sum_open_interest_value"].shift(1)
        daily["feature_oi_value_change"] = (
            daily["sum_open_interest_value"] - daily["prev_oi_value"]
        )
        daily_join = daily.rename(columns={
            "datetime": "feature_datetime",
            "available_at": "feature_available_at",
            "sum_open_interest": "feature_sum_open_interest",
            "sum_open_interest_value": "feature_sum_open_interest_value",
            "long_short_ratio": "feature_long_short_ratio",
            "top_trader_long_short_ratio": "feature_top_trader_long_short_ratio",
            "sum_taker_long_short_vol_ratio": "feature_sum_taker_long_short_vol_ratio",
        })
        daily_join = daily_join[list(_METRICS_JOINED_COLUMNS)].sort_values(
            "feature_available_at"
        )
        daily_join["feature_available_at"] = pd.to_datetime(
            daily_join["feature_available_at"], utc=True
        ).astype("datetime64[ns, UTC]")

    bars_sorted = bars.sort_index()
    out = bars_sorted.copy()
    out["decision_time"] = pd.DatetimeIndex(out.index + period).astype("datetime64[ns, UTC]")
    out["mark_return_24h"] = (
        bars_sorted["close"] / bars_sorted["close"].shift(6) - 1.0
    ).to_numpy()
    out = out.reset_index(drop=True)

    if daily_join.empty:
        joined = out.copy()
        for col in _METRICS_JOINED_COLUMNS:
            joined[col] = np.nan
    else:
        joined = pd.merge_asof(
            out,
            daily_join,
            left_on="decision_time",
            right_on="feature_available_at",
            direction="backward",
        )

    mask = joined["feature_available_at"].notna()
    if (joined.loc[mask, "feature_available_at"] > joined.loc[mask, "decision_time"]).any():
        raise DataIntegrityError(
            f"future feature timestamp leaked into a decision for {symbol}"
        )
    fa = joined.loc[mask, "feature_available_at"]
    if not fa.is_monotonic_increasing:
        raise DataIntegrityError(f"feature timestamps for {symbol} must be monotonic")

    _logger.info(
        "metrics asof symbol=%s bars=%d feature_rows=%d window=%s..%s",
        symbol, len(grid), int(mask.sum()),
        grid[0].isoformat(), grid[-1].isoformat(),
    )
    return joined.reset_index(drop=True)


def validate_oi_deleveraging_market_data(data: OIDeleveragingMarketData) -> None:
    """Fail-closed integrity gate for the OI-deleveraging research inputs.

    Raises ``DataIntegrityError`` for non-UTC, duplicate, non-monotonic,
    non-finite, non-positive, incomplete, or causally invalid inputs. A missing
    funding observation or a future feature timestamp never proceeds; nothing is
    imputed. ``data`` is never mutated.
    """
    if not isinstance(data, OIDeleveragingMarketData):
        raise TypeError(
            f"data must be an OIDeleveragingMarketData, got {type(data).__name__}"
        )
    if not data.symbol:
        raise DataIntegrityError("symbol must not be empty")

    grid = _validate_4h_grid(data.bars, "bars")
    period = grid[1] - grid[0]
    window_end = grid[-1] + period

    if not isinstance(data.joined, pd.DataFrame) or len(data.joined) != len(grid):
        raise DataIntegrityError("joined metrics frame must have one row per bar")
    missing = set(_JOINED_REQUIRED_COLUMNS) - set(data.joined.columns)
    if missing:
        raise DataIntegrityError(f"joined frame missing columns: {sorted(missing)}")
    decision_col = pd.to_datetime(data.joined["decision_time"], utc=True, errors="coerce")
    if decision_col.hasnans:
        raise DataIntegrityError("decision_time must be a valid UTC timestamp")
    if not pd.DatetimeIndex(decision_col).equals(pd.DatetimeIndex(grid) + period):
        raise DataIntegrityError("decision_time must match the bar-close grid")
    mask = data.joined["feature_available_at"].notna()
    if (data.joined.loc[mask, "feature_available_at"] > data.joined.loc[mask, "decision_time"]).any():
        raise DataIntegrityError("future feature timestamp in joined metrics frame")

    if len(data.funding) == 0:
        raise DataIntegrityError("funding must contain at least one settled event")
    funding = pd.to_numeric(data.funding, errors="coerce").astype("float64")
    fts = pd.DatetimeIndex(data.funding.index)
    if fts.tz is None or fts.tz != grid.tz:
        raise DataIntegrityError("funding index must be tz-aware UTC matching the bar grid")
    if fts.has_duplicates:
        raise DataIntegrityError("funding index must not contain duplicates")
    if not fts.is_monotonic_increasing:
        raise DataIntegrityError("funding index must be monotonic increasing")
    if not np.isfinite(funding.to_numpy(dtype=np.float64)).all():
        raise DataIntegrityError("funding rates must be finite")
    if not ((fts >= grid[0]) & (fts < window_end)).all():
        raise DataIntegrityError("funding events outside the bar window")


def load_oi_deleveraging_market_data(
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
) -> OIDeleveragingMarketData:
    """Load and fail-closed validate the causal OI-deleveraging inputs for one symbol.

    Loads the exact 4h futures grid, aligns the published funding settlement
    events, and as-of joins the canonical daily metrics so every decision sees
    only metrics released before its timestamp. Any missing input or future
    feature raises ``DataIntegrityError``; nothing is zero-filled or imputed.
    """
    perp_p = ohlcv_path(symbol, "1h")
    fund_p = funding_path(symbol)
    for path, name in [(perp_p, "perp_ohlcv"), (fund_p, "funding")]:
        if not path.exists():
            raise DataIntegrityError(f"{name} data missing for {symbol}: {path}")

    bars = load_ohlcv_1h_as_4h(perp_p, start=start, end=end)
    if len(bars) < 2:
        raise DataIntegrityError(f"bars data has fewer than 2 bars for {symbol}")
    period = bars.index[1] - bars.index[0]
    window_end = bars.index[-1] + period
    funding = load_funding_rates(str(fund_p))
    funding = funding[(funding.index >= bars.index[0]) & (funding.index < window_end)]
    joined = load_metrics_asof(symbol, bars, start, end)

    market_data = OIDeleveragingMarketData(
        symbol=symbol, bars=bars, joined=joined, funding=funding,
    )
    validate_oi_deleveraging_market_data(market_data)
    return market_data


def _check_contract() -> None:
    """Executable assertions locking the frozen OI data surface."""
    from inspect import signature

    assert load_metrics_asof.__name__ == "load_metrics_asof"
    assert list(signature(load_metrics_asof).parameters) == ["symbol", "bars", "start", "end"]
    assert {f.name for f in OIDeleveragingMarketData.__dataclass_fields__.values()} == {
        "symbol", "bars", "joined", "funding",
    }


_check_contract()
