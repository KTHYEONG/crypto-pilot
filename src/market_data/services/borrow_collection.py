from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import borrow_path
from src.common.errors import DataIntegrityError
from src.market_data.binance.margin import BinanceMarginClient
from src.market_data.services.rate_common import (
    BORROW_CANONICAL_COLUMNS as _BORROW_CANONICAL_COLUMNS,
)
from src.market_data.services.rate_common import (
    INTEREST_HISTORY_BOUNDARY as _INTEREST_HISTORY_BOUNDARY,
)
from src.market_data.services.rate_common import (
    RATE_PERIOD_SECONDS as _RATE_PERIOD_SECONDS,
)
from src.market_data.services.rate_common import (
    SECONDS_PER_DAY as _SECONDS_PER_DAY,
)
from src.market_data.storage.manifest import (
    _file_sha256,
    _manifest_record,
    _update_manifest_record,
)

_logger = logging.getLogger("SpotDataCollector")


def _parse_rate_period(rate_period: str) -> int:
    """Resolve a documented borrow-rate cadence to accrual seconds."""
    key = str(rate_period).strip().lower()
    if key in _RATE_PERIOD_SECONDS:
        return _RATE_PERIOD_SECONDS[key]
    try:
        seconds = int(key)
    except ValueError as exc:
        raise DataIntegrityError(
            f"ambiguous borrow rate period: {rate_period!r} "
            "(use an explicit accrual_seconds column or a documented unit "
            "annual/daily/hourly, 1y/1d/1h, or integer seconds)"
        ) from exc
    if seconds <= 0:
        raise DataIntegrityError(f"borrow rate period must be > 0 seconds, got {seconds}")
    return seconds


def _normalize_borrow_events(
    df: pd.DataFrame,
    *,
    source_id: str,
    rate_period: str,
) -> pd.DataFrame:
    """Normalize an operator-supplied borrow export to canonical columns.

    Produces ``timestamp`` (int64 ms), ``borrow_rate`` and ``accrual_seconds``.
    Rejects duplicate, overlapping, non-positive-duration, uncovered (gapped),
    non-finite, or ambiguous-unit rows. Never infers units or invents history.
    """
    if df is None or df.empty:
        raise DataIntegrityError(f"borrow export is empty: source_id={source_id}")
    events = df.copy()
    events = events.loc[:, ~events.columns.duplicated(keep="first")]

    if "timestamp" in events.columns:
        ts = pd.to_datetime(pd.to_numeric(events["timestamp"], errors="coerce"), unit="ms", utc=True)
    elif "datetime" in events.columns:
        ts = pd.to_datetime(events["datetime"], utc=True, errors="coerce")
    else:
        raise DataIntegrityError("borrow export must contain a 'timestamp' or 'datetime' column")
    if "borrow_rate" not in events.columns:
        raise DataIntegrityError("borrow export must contain a 'borrow_rate' column")

    rates = pd.to_numeric(events["borrow_rate"], errors="coerce")
    if "accrual_seconds" in events.columns:
        accrual = pd.to_numeric(events["accrual_seconds"], errors="coerce")
    else:
        accrual_seconds = _parse_rate_period(rate_period)
        accrual = pd.Series(accrual_seconds, index=events.index, dtype="float64")

    valid = ts.notna() & rates.notna() & accrual.notna()
    if not valid.all():
        raise DataIntegrityError(
            "borrow export contains non-finite rows (ts/borrow_rate/accrual_seconds)"
        )
    if (accrual <= 0).any():
        raise DataIntegrityError("borrow accrual_seconds must be > 0 for every row")
    if not np.isfinite(rates.astype("float64")).all():
        raise DataIntegrityError("borrow_rate must be finite")

    out = pd.DataFrame({
        "timestamp": ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).astype("int64"),
        "borrow_rate": rates.astype("float64"),
        "accrual_seconds": accrual.astype("float64"),
    })
    if out["timestamp"].duplicated().any():
        raise DataIntegrityError("duplicate borrow events detected")
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["ts"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    out["end_ts"] = out["ts"] + pd.to_timedelta(out["accrual_seconds"], unit="s")

    prev_end = out["end_ts"].shift(1)
    if (out["ts"] < prev_end).fillna(False).any():
        raise DataIntegrityError("borrow events must not overlap")
    if (out["ts"] > prev_end).fillna(False).any():
        raise DataIntegrityError("borrow coverage gap detected between events")

    return out[list(_BORROW_CANONICAL_COLUMNS)]


def _daily_rates_to_borrow_events(
    rates: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Convert Binance daily rate-change events to exact accrued intervals.

    A Binance ``dailyInterestRate`` remains effective from its timestamp until
    the next observed source timestamp.  The requested range must be bracketed
    by source events; terminal rates are never extrapolated.
    """
    required = {"timestamp", "dailyInterestRate"}
    if not required.issubset(rates.columns):
        raise DataIntegrityError(f"Binance interest history missing columns: {sorted(required - set(rates.columns))}")
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if start_ts >= end_ts:
        raise ValueError(f"invalid borrow range start={start} end={end}")
    events = rates.loc[:, ["timestamp", "dailyInterestRate"]].copy()
    events["timestamp"] = pd.to_datetime(
        pd.to_numeric(events["timestamp"], errors="coerce"), unit="ms", utc=True,
    )
    events["dailyInterestRate"] = pd.to_numeric(events["dailyInterestRate"], errors="coerce")
    if events.isna().any().any() or not np.isfinite(events["dailyInterestRate"]).all():
        raise DataIntegrityError("Binance interest history contains non-finite timestamps or rates")
    if (events["dailyInterestRate"] < 0).any():
        raise DataIntegrityError("Binance interest history contains a negative daily rate")
    events = events.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if events.empty or events["timestamp"].iloc[0] > start_ts:
        raise DataIntegrityError("Binance interest history lacks coverage at borrow range start")
    if events["timestamp"].iloc[-1] <= end_ts:
        raise DataIntegrityError("Binance interest history lacks a boundary event after borrow range end")

    next_timestamp = events["timestamp"].iloc[::-1].shift(1).iloc[::-1]
    interval_seconds = (next_timestamp - events["timestamp"]).dt.total_seconds()
    valid = next_timestamp.notna() & (interval_seconds > 0)
    intervals = events.loc[valid].copy()
    intervals["accrual_seconds"] = interval_seconds.loc[valid].astype("float64")
    intervals["borrow_rate"] = (
        intervals["dailyInterestRate"].astype("float64")
        * intervals["accrual_seconds"] / _SECONDS_PER_DAY
    )
    overlaps_request = (
        (intervals["timestamp"] < end_ts)
        & (next_timestamp.loc[valid] > start_ts)
    )
    canonical = pd.DataFrame({
        "timestamp": (
            (intervals.loc[overlaps_request, "timestamp"] - pd.Timestamp("1970-01-01", tz="UTC"))
            // pd.Timedelta("1ms")
        ).astype("int64"),
        "borrow_rate": intervals.loc[overlaps_request, "borrow_rate"].astype("float64"),
        "accrual_seconds": intervals.loc[overlaps_request, "accrual_seconds"].astype("float64"),
    })
    if canonical.empty:
        raise DataIntegrityError("Binance interest history has no intervals covering borrow range")
    return _normalize_borrow_events(
        canonical, source_id="binance_margin", rate_period="daily",
    )


def _parse_borrow_collection_bound(value: str, *, end_of_day: bool) -> pd.Timestamp:
    text = value.strip()
    if "T" not in text and " " not in text and end_of_day:
        text = f"{text} 23:59:59.999999999"
    return pd.to_datetime(text, utc=True)


def collect_binance_quote_borrow_history(
    symbol: str,
    asset: str,
    start: str,
    end: str,
    *,
    client: BinanceMarginClient | None = None,
) -> None:
    """Collect Binance Margin quote-borrow history into the canonical spot lake."""
    start_ts = _parse_borrow_collection_bound(start, end_of_day=False)
    end_ts = _parse_borrow_collection_bound(end, end_of_day=True)
    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts >= end_ts:
        raise ValueError(f"invalid borrow range start={start} end={end}")
    margin_client = client or BinanceMarginClient()
    raw = margin_client.fetch_margin_interest_rate_history(
        asset, start_ts - _INTEREST_HISTORY_BOUNDARY, end_ts + _INTEREST_HISTORY_BOUNDARY,
    )
    events = _daily_rates_to_borrow_events(raw, start_ts, end_ts)
    path = borrow_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp.parquet")
    events.to_parquet(temp_path, index=False, compression="zstd")
    temp_path.replace(path)

    ts = pd.to_datetime(events["timestamp"], unit="ms", utc=True)
    record = _manifest_record(
        venue="binance",
        instrument=f"spot:{symbol}:borrow:{asset.upper()}",
        source_locator="sapi/v1/margin/interestRateHistory",
        retrieved_at=pd.Timestamp.now(tz="UTC").isoformat(),
        requested_range=f"{start_ts.isoformat()}..{end_ts.isoformat()}",
        row_count=len(events),
        min_ts=str(ts.min()),
        max_ts=str(ts.max()),
        sha256=_file_sha256(path),
        conversion={
            "source_units": "dailyInterestRate",
            "conversion": "daily_rate_times_interval_seconds_over_86400",
            "asset": asset.upper(),
        },
    )
    _update_manifest_record("borrow", symbol, record)
    _logger.info(
        "[DATA] borrow symbol=%s asset=%s status=PASS rows=%d start=%s end=%s",
        symbol, asset.upper(), len(events), start_ts.isoformat(), end_ts.isoformat(),
        extra={"tag": "DATA"},
    )


def import_quote_borrow_history(
    symbol: str,
    source_path: str | Path,
    source_id: str,
    rate_period: str,
) -> None:
    """Import a versioned historical quote-borrow export into the spot lake.

    Persists ``data/spot/borrow/<SYMBOL>.parquet`` (canonical borrow columns)
    plus a spot manifest record carrying the source identifier and conversion
    metadata. Historical borrow is never invented from a current rate and never
    replaced by zero.
    """
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"borrow export not found: {src}")
    if not source_id.strip():
        raise DataIntegrityError("source_id must not be empty")
    events = _normalize_borrow_events(
        pd.read_parquet(src), source_id=source_id, rate_period=rate_period,
    )

    path = borrow_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp.parquet")
    events.to_parquet(temp_path, index=False, compression="zstd")
    temp_path.replace(path)

    ts = pd.to_datetime(events["timestamp"], unit="ms", utc=True)
    record = _manifest_record(
        venue="operator",
        instrument=f"spot:{symbol}:borrow",
        source_locator=source_id,
        retrieved_at=pd.Timestamp.now(tz="UTC").isoformat(),
        requested_range=f"{ts.min().isoformat()}..{ts.max().isoformat()}",
        row_count=len(events),
        min_ts=str(ts.min()),
        max_ts=str(ts.max()),
        sha256=_file_sha256(path),
        conversion={"source_units": str(rate_period)},
    )
    _update_manifest_record("borrow", symbol, record)
    _logger.info(
        "[DATA] borrow symbol=%s status=PASS rows=%d start=%s end=%s source_id=%s",
        symbol, len(events), ts.min().isoformat(), ts.max().isoformat(), source_id,
        extra={"tag": "DATA"},
    )
