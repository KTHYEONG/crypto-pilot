from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import spot_ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.binance.margin import BinanceMarginClient
from src.market_data.binance.spot import BinanceSpotClient
from src.market_data.services.borrow_collection import (
    collect_binance_quote_borrow_history,
    import_quote_borrow_history,
)
from src.market_data.storage.manifest import (
    _file_sha256,
    _load_manifest,
    _manifest_record,
    _prior_quality_metadata,
    _update_manifest_record,
)
from src.market_data.storage.ohlcv import merge_ohlcv_frames, normalize_frame, write_ohlcv

_logger = logging.getLogger("SpotDataCollector")

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

_BORROW_CANONICAL_COLUMNS: tuple[str, ...] = ("timestamp", "borrow_rate", "accrual_seconds")


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
    the shared ohlcv store, writes only
    ``data/spot/ohlcv/<timeframe>/<SYMBOL>.parquet``, and refreshes the spot
    manifest. Emits coverage diagnostics.
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
