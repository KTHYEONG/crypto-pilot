from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.core.config import SPOT_DATA_DIR, borrow_path, spot_ohlcv_path
from src.data.binance import BinanceMarginClient, BinanceSpotClient
from src.data.loader import DataIntegrityError
from src.data.ohlcv_store import merge_ohlcv_frames, normalize_frame, write_ohlcv

_logger = logging.getLogger("SpotDataCollector")

MANIFEST_PATH = SPOT_DATA_DIR / "manifest.json"
MANIFEST_SCHEMA_VERSION = 1

_BORROW_CANONICAL_COLUMNS: tuple[str, ...] = ("timestamp", "borrow_rate", "accrual_seconds")

_RATE_PERIOD_SECONDS: dict[str, int] = {
    "annual": 365 * 86400,
    "1y": 365 * 86400,
    "365d": 365 * 86400,
    "daily": 86400,
    "1d": 86400,
    "hourly": 3600,
    "1h": 3600,
    "3600s": 3600,
}

_SECONDS_PER_DAY = 86400.0
_INTEREST_HISTORY_BOUNDARY = pd.Timedelta(days=31)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict[str, object]:
    if not MANIFEST_PATH.exists():
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "datasets": {}}
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = cast(dict[str, object], json.load(handle))
    manifest.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    manifest.setdefault("datasets", {})
    return manifest


def _save_manifest(manifest: dict[str, object]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MANIFEST_PATH.with_suffix(".tmp.json")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(MANIFEST_PATH)


def load_spot_manifest() -> dict[str, object]:
    """Return the canonical spot manifest (empty datasets when absent)."""
    return _load_manifest()


def _update_manifest_record(dataset: str, symbol: str, record: dict[str, object]) -> None:
    """Replace only the matching manifest record after the new file is durable."""
    manifest = _load_manifest()
    datasets = manifest["datasets"]
    assert isinstance(datasets, dict)
    datasets.setdefault(dataset, {})
    dataset_records = datasets[dataset]
    assert isinstance(dataset_records, dict)
    dataset_records[symbol] = record
    _save_manifest(manifest)


def _manifest_record(
    *,
    venue: str,
    instrument: str,
    source_locator: str,
    retrieved_at: str,
    requested_range: str,
    row_count: int,
    min_ts: str,
    max_ts: str,
    sha256: str,
    conversion: dict[str, object] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "venue": venue,
        "instrument": instrument,
        "source_locator": source_locator,
        "retrieved_at": retrieved_at,
        "requested_range": requested_range,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "row_count": row_count,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "sha256": sha256,
    }
    if conversion is not None:
        record["conversion"] = conversion
    return record


def _prior_quality_metadata(dataset: str, symbol: str) -> dict[str, object]:
    """Carry synthetic-bar provenance across ordinary incremental refreshes."""
    manifest = _load_manifest()
    datasets = manifest.get("datasets", {})
    if not isinstance(datasets, dict):
        return {}
    previous = datasets.get(dataset, {})
    if not isinstance(previous, dict) or not isinstance(previous.get(symbol), dict):
        return {}
    old = previous[symbol]
    quality: dict[str, object] = {}
    imputations = old.get("imputations")
    if isinstance(imputations, list):
        quality["imputations"] = imputations
    elif isinstance(old.get("imputation"), dict):
        quality["imputations"] = [old["imputation"]]
    if isinstance(old.get("data_quality"), dict):
        quality["data_quality"] = old["data_quality"]
    return quality


def _read_spot_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return normalize_frame(pd.read_parquet(path))
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Failed to read spot cache %s: %s", path, exc)
        return pd.DataFrame()


def _missing_bar_timestamps(
    frame: pd.DataFrame,
    req_start: pd.Timestamp,
    req_end: pd.Timestamp,
    period: pd.Timedelta,
) -> pd.DatetimeIndex:
    timestamps = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    expected_start = req_start.ceil(period)
    expected_end = (req_end - pd.Timedelta(milliseconds=1)).floor(period)
    if expected_end < expected_start:
        return pd.DatetimeIndex([], tz="UTC")
    expected = pd.date_range(expected_start, expected_end, freq=period, tz="UTC")
    observed = pd.DatetimeIndex(
        timestamps[(timestamps >= expected_start) & (timestamps <= expected_end)]
    )
    return expected.difference(observed)


def repair_spot_ohlcv_gap(
    symbol: str,
    timeframe: str,
    timestamp: str,
    *,
    method: str = "bridge_unknown_volume",
) -> None:
    """Repair exactly one isolated gap using a marked, unknown-volume bridge.

    Price fields are bridged from the adjacent candles.  Flow fields use
    log-scale interpolation (and adjacent VWAP/taker ratios), so a missing
    candle is not silently treated as a zero-volume observation.  Existing
    legacy zero-volume imputations may be replaced in-place.
    """
    if method != "bridge_unknown_volume":
        raise ValueError(f"unsupported gap repair method: {method}")
    period = pd.Timedelta(timeframe)
    target = pd.Timestamp(timestamp, tz="UTC")
    path = spot_ohlcv_path(symbol, timeframe)
    frame = _read_spot_cache(path)
    if frame.empty:
        raise DataIntegrityError(f"spot cache is empty: {path}")
    frame_ts = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    manifest = _load_manifest()
    datasets = manifest["datasets"]
    assert isinstance(datasets, dict)
    records = datasets.get(f"ohlcv/{timeframe}", {})
    assert isinstance(records, dict)
    record = dict(records.get(symbol, {}))
    legacy_imputation = record.pop("imputation", None)
    imputations = list(record.get("imputations", []))
    if isinstance(legacy_imputation, dict):
        imputations.append(legacy_imputation)
    existing_positions = frame_ts == target
    if existing_positions.any():
        if not any(
            isinstance(item, dict) and item.get("timestamp") == target.isoformat()
            for item in imputations
        ):
            raise DataIntegrityError(f"spot timestamp already exists: {target.isoformat()}")
        frame = frame.loc[~existing_positions].copy()
        frame_ts = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    left = frame_ts[frame_ts < target]
    right = frame_ts[frame_ts > target]
    if left.empty or right.empty:
        raise DataIntegrityError("gap repair requires observed bars on both sides")
    left_ts = left.max()
    right_ts = right.min()
    if target - left_ts != period or right_ts - target != period:
        raise DataIntegrityError(
            f"gap repair requires adjacent bars at {target - period} and {target + period}"
        )
    left_row = frame.loc[frame_ts == left_ts].iloc[0]
    right_row = frame.loc[frame_ts == right_ts].iloc[0]
    left_close = float(left_row["close"])
    right_open = float(right_row["open"])

    def _number(row: pd.Series, name: str, default: float = 0.0) -> float:
        value = pd.to_numeric(row.get(name, default), errors="coerce")
        return default if pd.isna(value) else max(float(value), 0.0)

    left_volume = _number(left_row, "volume")
    right_volume = _number(right_row, "volume")
    volume = float(np.expm1((np.log1p(left_volume) + np.log1p(right_volume)) / 2.0))

    def _vwap(row: pd.Series, volume_value: float) -> float:
        quote = _number(row, "quote_vol")
        if volume_value > 0 and quote > 0:
            return quote / volume_value
        prices = [float(row[name]) for name in ("open", "high", "low", "close")]
        return float(np.mean(prices))

    quote = volume * float(np.sqrt(_vwap(left_row, left_volume) * _vwap(right_row, right_volume)))

    def _ratio(row: pd.Series, numerator: str, denominator: float) -> float | None:
        if denominator <= 0:
            return None
        value = _number(row, numerator) / denominator
        return float(np.clip(value, 0.0, 1.0))

    base_ratios = [r for r in (_ratio(left_row, "taker_buy_base_volume", left_volume),
                               _ratio(right_row, "taker_buy_base_volume", right_volume))
                   if r is not None]
    quote_ratios = [r for r in (_ratio(left_row, "taker_buy_quote_volume", _number(left_row, "quote_vol")),
                                _ratio(right_row, "taker_buy_quote_volume", _number(right_row, "quote_vol")))
                    if r is not None]
    base_ratio = float(np.mean(base_ratios)) if base_ratios else (0.5 if volume > 0 else 0.0)
    quote_ratio = float(np.mean(quote_ratios)) if quote_ratios else (0.5 if quote > 0 else 0.0)
    synthetic = pd.DataFrame([{
        "timestamp": int(target.timestamp() * 1000),
        "open": left_close,
        "high": max(left_close, right_open),
        "low": min(left_close, right_open),
        "close": right_open,
        "volume": volume,
        "taker_buy_base_volume": volume * base_ratio,
        "taker_buy_quote_volume": quote * quote_ratio,
        "quote_vol": quote,
    }])
    combined = merge_ohlcv_frames([frame, synthetic])
    write_ohlcv(path, combined, timeframe=timeframe)

    stored_ts = pd.to_datetime(combined["timestamp"], unit="ms", utc=True)
    updated_imputations = [
        item for item in imputations
        if not (isinstance(item, dict) and item.get("timestamp") == target.isoformat())
    ] + [{
        "method": method,
        "timestamp": target.isoformat(),
        "left_timestamp": left_ts.isoformat(),
        "right_timestamp": right_ts.isoformat(),
        "imputed_fields": ["ohlcv", "volume", "quote_vol", "taker_flow"],
        "volume_model": "log1p_geometric_bridge_v1",
        "quality": "unknown",
    }]
    record.update({
        "row_count": len(combined),
        "min_ts": str(stored_ts.min()),
        "max_ts": str(stored_ts.max()),
        "sha256": _file_sha256(path),
        "imputations": updated_imputations,
        "data_quality": {
            "status": "IMPUTED",
            "imputed_bar_count": len(updated_imputations),
            "volume_quality": "unknown",
        },
    })
    _update_manifest_record(f"ohlcv/{timeframe}", symbol, record)
    _logger.info(
        "[DATA] spot_ohlcv symbol=%s timeframe=%s status=REPAIRED timestamp=%s method=%s",
        symbol, timeframe, target.isoformat(), method, extra={"tag": "DATA"},
    )


def ensure_spot_ohlcv(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    *,
    client: BinanceSpotClient | None = None,
) -> None:
    """Incrementally collect spot klines and merge them into the canonical lake.

    Fetches only uncovered ranges via the Binance spot endpoint, merges through
    ``ohlcv_store``, writes only ``data/spot/ohlcv/<timeframe>/<SYMBOL>.parquet``,
    and refreshes the spot manifest. Emits coverage diagnostics.
    """
    req_start = pd.to_datetime(start, utc=True)
    req_end = pd.to_datetime(end, utc=True)
    if pd.isna(req_start) or pd.isna(req_end) or req_start >= req_end:
        raise ValueError(f"invalid range start={start} end={end}")

    path = spot_ohlcv_path(symbol, timeframe)
    cache_df = _read_spot_cache(path)
    new_parts: list[pd.DataFrame] = []
    try:
        period = pd.Timedelta(timeframe)
    except ValueError as exc:
        raise ValueError(f"unsupported spot timeframe for gap repair: {timeframe}") from exc

    fetch_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if cache_df.empty:
        fetch_ranges.append((req_start, req_end))
    else:
        cache_ts = pd.to_datetime(cache_df["datetime"], utc=True).sort_values()
        cache_min = cache_ts.iloc[0]
        cache_max = cache_ts.iloc[-1]
        if cache_min > req_start:
            fetch_ranges.append((req_start, cache_min))
        if cache_max < req_end:
            fetch_ranges.append((cache_max, req_end))
        deltas = cache_ts.diff().to_numpy()
        for pos in range(1, len(cache_ts)):
            if deltas[pos] <= period:
                continue
            current = cache_ts.iloc[pos]
            previous = cache_ts.iloc[pos - 1]
            gap_start = previous + period
            if gap_start < current and gap_start < req_end and current > req_start:
                fetch_ranges.append((max(gap_start, req_start), min(current, req_end)))

    if fetch_ranges:
        spot_client = client or BinanceSpotClient()
        for fetch_start, fetch_end in fetch_ranges:
            if fetch_start >= fetch_end:
                continue
            chunk = spot_client.fetch_spot_ohlcv(
                symbol, timeframe, str(fetch_start), str(fetch_end),
            )
            if not chunk.empty:
                new_parts.append(chunk)

    combined = merge_ohlcv_frames([cache_df, *new_parts])
    if combined.empty:
        _logger.info(
            "[DATA] spot_ohlcv symbol=%s timeframe=%s status=PENDING rows=0",
            symbol, timeframe, extra={"tag": "DATA"},
        )
        return
    write_ohlcv(path, combined, timeframe=timeframe)

    ts = pd.to_datetime(combined["timestamp"], unit="ms", utc=True)
    record = _manifest_record(
        venue="binance",
        instrument=f"spot:{symbol}:{timeframe}",
        source_locator=f"api.binance.com/api/v3/klines/{symbol}/{timeframe}",
        retrieved_at=pd.Timestamp.now(tz="UTC").isoformat(),
        requested_range=f"{req_start.isoformat()}..{req_end.isoformat()}",
        row_count=len(combined),
        min_ts=str(ts.min()),
        max_ts=str(ts.max()),
        sha256=_file_sha256(path),
    )
    record.update(_prior_quality_metadata(f"ohlcv/{timeframe}", symbol))
    _update_manifest_record(f"ohlcv/{timeframe}", symbol, record)
    missing = _missing_bar_timestamps(combined, req_start, req_end, period)
    if len(missing) > 0:
        _logger.info(
            "[DATA] spot_ohlcv symbol=%s timeframe=%s status=PENDING rows=%d "
            "reason=coverage_gap missing=%d first_missing=%s",
            symbol, timeframe, len(combined), len(missing), missing[0].isoformat(),
            extra={"tag": "DATA"},
        )
        return
    _logger.info(
        "[DATA] spot_ohlcv symbol=%s timeframe=%s status=PASS rows=%d start=%s end=%s",
        symbol, timeframe, len(combined), ts.min().isoformat(), ts.max().isoformat(),
        extra={"tag": "DATA"},
    )


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


class SpotDataCollector:
    """Spot OHLCV acquisition and historical quote-borrow import for one symbol."""

    def __init__(self, api_key: str | None = None, secret: str | None = None) -> None:
        self.client = BinanceSpotClient(api_key, secret)

    def ensure_spot_ohlcv(self, symbol: str, timeframe: str, start: str, end: str) -> None:
        ensure_spot_ohlcv(symbol, timeframe, start, end, client=self.client)

    def import_quote_borrow_history(
        self, symbol: str, source_path: str | Path, source_id: str, rate_period: str,
    ) -> None:
        import_quote_borrow_history(symbol, source_path, source_id, rate_period)

    def collect_binance_quote_borrow_history(
        self, symbol: str, asset: str, start: str, end: str,
    ) -> None:
        collect_binance_quote_borrow_history(
            symbol, asset, start, end, client=BinanceMarginClient(),
        )
