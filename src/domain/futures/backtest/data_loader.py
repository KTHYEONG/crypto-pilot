"""Data Loading and Preparation for Futures.
Combines Data Collection (API/Vision), Metadata management, and Merging of
Funding/Metrics into OHLCV.
"""

from __future__ import annotations

import concurrent.futures
import fcntl
import json
import logging
import os
import threading
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.exchange.binance_client import BinanceClient, BinanceKlinePermanentError
from src.core.exchange.binance_vision import BinanceVisionDownloader
from src.core.settings import FUTURES_DATA_DIR
from src.core.utils.utils import setup_logger

_logger = logging.getLogger("DataCollector")

_METRICS_CANONICAL_COLUMNS: tuple[str, ...] = (
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
_METRICS_NUMERIC_COLUMNS: tuple[str, ...] = (
    "sum_open_interest",
    "sum_open_interest_value",
    "long_short_ratio",
    "top_trader_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
_METRICS_RELEASE_LAG = pd.Timedelta(minutes=5)
_METRICS_MERGE_TOLERANCE = pd.Timedelta(hours=6)


def _empty_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_METRICS_CANONICAL_COLUMNS))


def _normalize_metrics_frame(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """Return the canonical, unique, UTC metrics schema."""
    if frame is None or frame.empty:
        return _empty_metrics_frame()

    df = frame.copy()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    rename_map = {
        "create_time": "timestamp",
        "open_interest": "sum_open_interest",
        "oi": "sum_open_interest",
        "open_interest_value": "sum_open_interest_value",
        "global_long_short_ratio": "long_short_ratio",
        "sum_toptrader_long_short_ratio": "top_trader_long_short_ratio",
        "count_long_short_ratio": "long_short_ratio",
    }
    df = df.rename(columns=rename_map)
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    else:
        df["symbol"] = df["symbol"].fillna(symbol).astype(str)

    if "timestamp" not in df.columns and "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["timestamp"] = dt.astype("int64") // 10**6
    if "timestamp" not in df.columns:
        return _empty_metrics_frame()

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return _empty_metrics_frame()
    df["timestamp"] = df["timestamp"].astype("int64")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return _empty_metrics_frame()
    if "available_at" in df.columns:
        df["available_at"] = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
    else:
        df["available_at"] = df["datetime"] + _METRICS_RELEASE_LAG
    df["available_at"] = df["available_at"].fillna(df["datetime"] + _METRICS_RELEASE_LAG)
    for col in _METRICS_NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.loc[:, list(_METRICS_CANONICAL_COLUMNS)]
        .sort_values(["timestamp", "available_at"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _coalesce_metrics_frames(
    frames: Iterable[pd.DataFrame],
    *,
    symbol: str,
) -> pd.DataFrame:
    """Coalesce complementary OI/LSR rows by timestamp without dropping fields."""
    normalized = [
        _normalize_metrics_frame(frame, symbol=symbol)
        for frame in frames
        if frame is not None and not frame.empty
    ]
    if not normalized:
        return _empty_metrics_frame()

    combined = pd.concat(normalized, ignore_index=True, sort=False)
    if combined.empty:
        return _empty_metrics_frame()
    combined = combined.sort_values(["timestamp", "available_at"]).reset_index(drop=True)
    grouped = (
        combined.groupby("timestamp", as_index=False)
        .agg(
            {
                "datetime": "last",
                "available_at": "last",
                "symbol": "last",
                **dict.fromkeys(_METRICS_NUMERIC_COLUMNS, "last"),
            }
        )
        .reset_index(drop=True)
    )
    return _normalize_metrics_frame(grouped, symbol=symbol)


def _normalize_funding_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize funding dataframe to a stable 3-column schema."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "datetime"])

    df = frame.copy()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    if "calc_time" in df.columns:
        df = df.rename(columns={"calc_time": "timestamp"})
    if "fundingRate" in df.columns:
        df = df.rename(columns={"fundingRate": "funding_rate"})
    if "timestamp" not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    if "funding_rate" not in df.columns and len(df.columns) > 2:
        df = df.rename(columns={df.columns[2]: "funding_rate"})
    if "timestamp" not in df.columns or "funding_rate" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "datetime"])

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df = df.dropna(subset=["timestamp", "funding_rate"])
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "datetime"])

    df["timestamp"] = df["timestamp"].astype("int64")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "datetime"])

    return (
        df[["timestamp", "funding_rate", "datetime"]]
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def summarize_dataframe_integrity(
    df: pd.DataFrame,
    *,
    timeframe: str | None = None,
    datetime_col: str = "datetime",
) -> dict[str, float]:
    """Compute lightweight integrity metrics for a dataframe snapshot."""
    if df is None or df.empty:
        return {
            "nan_pct": 0.0,
            "inf_count": 0.0,
            "zero_ratio": 0.0,
            "duplicate_dt": 0.0,
            "gap_count": 0.0,
            "nonpositive_price_count": 0.0,
            "rows": 0.0,
            "cols": 0.0,
        }

    rows = len(df)
    cols = len(df.columns)
    total_cells = max(rows * cols, 1)
    nan_pct = float(df.isna().sum().sum() / total_cells)
    num_df = df.select_dtypes(include=[np.number])
    inf_count = (
        float(np.isinf(num_df.to_numpy(dtype=np.float64, copy=False)).sum())
        if not num_df.empty
        else 0.0
    )
    zero_ratio = (
        float((num_df.to_numpy(dtype=np.float64, copy=False) == 0.0).sum() / max(num_df.size, 1))
        if not num_df.empty
        else 0.0
    )

    duplicate_dt = 0.0
    gap_count = 0.0
    if datetime_col in df.columns:
        dt = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")
        duplicate_dt = float(dt.duplicated().sum())
        tf_to_delta = {
            "1m": pd.Timedelta(minutes=1),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "1d": pd.Timedelta(days=1),
        }
        expected = tf_to_delta.get(str(timeframe or "").lower())
        if expected is not None:
            ddt = dt.sort_values().diff().dropna()
            gap_count = float((ddt != expected).sum())

    nonpositive_price_count = 0.0
    for pcol in ("open", "high", "low", "close"):
        if pcol in df.columns:
            v = pd.to_numeric(df[pcol], errors="coerce")
            nonpositive_price_count += float((v <= 0.0).sum())

    return {
        "nan_pct": nan_pct,
        "inf_count": inf_count,
        "zero_ratio": zero_ratio,
        "duplicate_dt": duplicate_dt,
        "gap_count": gap_count,
        "nonpositive_price_count": nonpositive_price_count,
        "rows": float(rows),
        "cols": float(cols),
    }


def summarize_ohlcv_collection_integrity(
    df: pd.DataFrame,
    *,
    timeframe: str,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
    datetime_col: str = "datetime",
) -> dict[str, float]:
    """Compute OHLCV-collection specific integrity metrics."""
    if df is None or df.empty:
        return {
            "nan_pct": 0.0,
            "inf_count": 0.0,
            "duplicate_dt": 0.0,
            "gap_count": 0.0,
            "missing_bar_ratio": 1.0,
            "datetime_nat_count": 0.0,
            "non_monotonic_dt_count": 0.0,
            "nonpositive_price_count": 0.0,
            "high_lt_low_count": 0.0,
            "open_outside_hl_count": 0.0,
            "close_outside_hl_count": 0.0,
            "negative_volume_count": 0.0,
            "coverage_start_miss": 0.0,
            "coverage_end_miss": 0.0,
            "rows": 0.0,
            "cols": 0.0,
        }

    rows = len(df)
    cols = len(df.columns)
    total_cells = max(rows * cols, 1)
    nan_pct = float(df.isna().sum().sum() / total_cells)
    num_df = df.select_dtypes(include=[np.number])
    inf_count = (
        float(np.isinf(num_df.to_numpy(dtype=np.float64, copy=False)).sum())
        if not num_df.empty
        else 0.0
    )

    tf_to_delta = {
        "1m": pd.Timedelta(minutes=1),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
    }
    expected_delta = tf_to_delta.get(str(timeframe).lower())
    gap_count = 0.0
    duplicate_dt = 0.0
    datetime_nat_count = 0.0
    non_monotonic_dt_count = 0.0
    missing_bar_ratio = 0.0
    coverage_start_miss = 0.0
    coverage_end_miss = 0.0

    if datetime_col in df.columns:
        dt_raw = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")
        datetime_nat_count = float(dt_raw.isna().sum())
        dt = dt_raw.dropna()
        if not dt.empty:
            duplicate_dt = float(dt.duplicated().sum())
            dt_sorted = dt.sort_values()
            diffs = dt_sorted.diff().dropna()
            if expected_delta is not None:
                gap_count = float((diffs != expected_delta).sum())
                if expected_start is not None and expected_end is not None and expected_end >= expected_start:
                    expected_bars = int((expected_end - expected_start) / expected_delta) + 1
                    expected_bars = max(expected_bars, 1)
                    actual_bars = int(dt.nunique())
                    missing_bar_ratio = float(
                        max(expected_bars - actual_bars, 0) / expected_bars
                    )
            if len(dt_raw) > 1:
                raw_diffs = dt_raw.diff()
                non_monotonic_dt_count = float((raw_diffs.dropna() < pd.Timedelta(0)).sum())
            first_dt = dt.min()
            last_dt = dt.max()
            if expected_start is not None:
                coverage_start_miss = float(first_dt > expected_start)
            if expected_end is not None:
                coverage_end_miss = float(last_dt < expected_end)

    nonpositive_price_count = 0.0
    high_lt_low_count = 0.0
    open_outside_hl_count = 0.0
    close_outside_hl_count = 0.0
    negative_volume_count = 0.0

    open_v = pd.to_numeric(df["open"], errors="coerce") if "open" in df.columns else None
    high_v = pd.to_numeric(df["high"], errors="coerce") if "high" in df.columns else None
    low_v = pd.to_numeric(df["low"], errors="coerce") if "low" in df.columns else None
    close_v = pd.to_numeric(df["close"], errors="coerce") if "close" in df.columns else None
    vol_v = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else None

    for series in (open_v, high_v, low_v, close_v):
        if series is not None:
            nonpositive_price_count += float((series <= 0.0).sum())

    if high_v is not None and low_v is not None:
        high_lt_low_count = float((high_v < low_v).sum())
        if open_v is not None:
            open_outside_hl_count = float(((open_v > high_v) | (open_v < low_v)).sum())
        if close_v is not None:
            close_outside_hl_count = float(((close_v > high_v) | (close_v < low_v)).sum())
    if vol_v is not None:
        negative_volume_count = float((vol_v < 0.0).sum())

    return {
        "nan_pct": nan_pct,
        "inf_count": inf_count,
        "duplicate_dt": duplicate_dt,
        "gap_count": gap_count,
        "missing_bar_ratio": missing_bar_ratio,
        "datetime_nat_count": datetime_nat_count,
        "non_monotonic_dt_count": non_monotonic_dt_count,
        "nonpositive_price_count": nonpositive_price_count,
        "high_lt_low_count": high_lt_low_count,
        "open_outside_hl_count": open_outside_hl_count,
        "close_outside_hl_count": close_outside_hl_count,
        "negative_volume_count": negative_volume_count,
        "coverage_start_miss": coverage_start_miss,
        "coverage_end_miss": coverage_end_miss,
        "rows": float(rows),
        "cols": float(cols),
    }


# --- Data Validator (from data_collector.py) ---


class DataValidator:
    """Validator for data integrity of OHLCV DataFrames."""

    @staticmethod
    def validate(df: pd.DataFrame, symbol: str, timeframe: str) -> list[str]:
        """데이터 무결성 검증."""
        issues = []
        if df.isnull().any().any():
            missing_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"Missing values in columns: {missing_cols}")

        df.set_index("datetime", inplace=True, drop=False)
        df.sort_index(inplace=True)

        expected_diff = {
            "1m": pd.Timedelta(minutes=1),
            "1h": pd.Timedelta(hours=1),
            "1d": pd.Timedelta(days=1),
            "4h": pd.Timedelta(hours=4),
        }.get(timeframe)

        if expected_diff:
            time_diff = df.index.to_series().diff().dropna()
            gaps = time_diff[time_diff != expected_diff]
            if not gaps.empty:
                issues.append(f"Found {len(gaps)} time gaps. First gap at {gaps.index[0]}")

        if not df.empty and (df["high"] < df["low"]).any():
            issues.append("High < Low detected in some rows")

        return issues


# --- Data Collector (from data_collector.py) ---


class DataCollector:
    """Collector for futures market data from Binance API and Vision."""

    _meta_lock = threading.Lock()
    _collect_1m_semaphore = threading.Semaphore(3)

    def __init__(self, api_key: str | None = None, secret: str | None = None) -> None:
        self.client = BinanceClient(api_key, secret)
        self.logger = setup_logger("DataCollector")
        self._metadata_cache: dict[str, Any] | None = None

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        return symbol.replace("/", "_")

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = self._safe_symbol(symbol)
        return FUTURES_DATA_DIR / f"{safe_symbol}_{timeframe}.parquet"

    def list_cached_parquet_symbols(self, timeframe: str) -> list[str]:
        suf = f"_{timeframe}.parquet"
        out: list[str] = []
        for p in FUTURES_DATA_DIR.glob(f"*{suf}"):
            stem = p.name[: -len(suf)]
            if "_" not in stem:
                continue
            base, quote = stem.rsplit("_", 1)
            out.append(f"{base}/{quote}")
        return sorted(set(out))

    def _meta_path(self) -> Path:
        return FUTURES_DATA_DIR / "parquet_cache_meta.json"

    def _meta_key(self, symbol: str, timeframe: str) -> str:
        return f"{self._safe_symbol(symbol)}::{timeframe}"

    def _is_range_blocked_by_permanent_failure(
        self,
        *,
        symbol: str,
        timeframe: str,
        requested_start: pd.Timestamp,
        requested_end: pd.Timestamp,
    ) -> bool:
        entry = self._load_meta().get(self._meta_key(symbol, timeframe), {})
        failure = entry.get("last_permanent_failure")
        if not isinstance(failure, dict):
            return False
        fail_start = failure.get("requested_start")
        fail_end = failure.get("requested_end")
        if not fail_start or not fail_end:
            return False
        try:
            fail_start_dt = pd.to_datetime(str(fail_start), utc=True)
            fail_end_dt = pd.to_datetime(str(fail_end), utc=True)
            return bool(fail_start_dt <= requested_start and fail_end_dt >= requested_end)
        except Exception:
            return False

    def _record_permanent_fetch_failure(
        self,
        *,
        symbol: str,
        timeframe: str,
        error: BinanceKlinePermanentError,
        requested_start: pd.Timestamp,
        requested_end: pd.Timestamp,
    ) -> None:
        self._save_meta(
            {
                self._meta_key(symbol, timeframe): {
                    "last_permanent_failure": {
                        "http_code": int(error.http_code),
                        "start_time_ms": int(error.start_time_ms),
                        "end_time_ms": int(error.end_time_ms),
                        "requested_start": str(requested_start),
                        "requested_end": str(requested_end),
                        "recorded_at": str(pd.Timestamp.now(tz="UTC")),
                        "url": str(error.url),
                    }
                }
            }
        )

    def _load_meta(self) -> dict[str, Any]:
        if self._metadata_cache is not None:
            return self._metadata_cache
        path = self._meta_path()
        if not path.exists():
            self._metadata_cache = {}
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._metadata_cache = data if isinstance(data, dict) else {}
            return self._metadata_cache
        except Exception:
            self._metadata_cache = {}
            return {}

    def _save_meta(self, meta_updates: dict[str, Any]) -> None:
        path = self._meta_path()
        lock_path = path.with_suffix(".lock")
        with self._meta_lock:
            try:
                with open(lock_path, "w") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    disk_meta = {}
                    if path.exists():
                        try:
                            with open(path, encoding="utf-8") as f:
                                data = json.load(f)
                            if isinstance(data, dict):
                                disk_meta = data
                        except Exception as exc:
                            self.logger.debug("Failed to read disk meta: %s", exc)
                    for mk, updates in meta_updates.items():
                        if mk not in disk_meta:
                            disk_meta[mk] = {}
                        if isinstance(updates, dict) and isinstance(disk_meta[mk], dict):
                            disk_meta[mk].update(updates)
                        else:
                            disk_meta[mk] = updates
                    tmp = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}.json")
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(disk_meta, f, ensure_ascii=False, indent=2)
                    os.replace(tmp, path)
                    self._metadata_cache = disk_meta
            except Exception as e:
                self.logger.error(f"Failed to save metadata: {e}")

    def _load_cache(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self._cache_path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
            if df.empty or ("timestamp" not in df.columns and "datetime" not in df.columns):
                path.unlink()
                return pd.DataFrame()
            # OPT-5: Drop Binance baggage columns (never used downstream)
            _baggage = [c for c in ("close_time", "no_trades", "ignore") if c in df.columns]
            if _baggage:
                df = df.drop(columns=_baggage)
            return self._normalize_df(df)
        except Exception as exc:
            try:
                path.unlink()
            except Exception as unlink_exc:
                self.logger.debug("Failed to unlink cache %s: %s", path, unlink_exc)
            self.logger.debug("Failed to load cache %s: %s", path, exc)
            return pd.DataFrame()

    def _save_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = self._normalize_df(df)
        path = self._cache_path(symbol, timeframe)
        temp_path = path.with_suffix(".tmp.parquet")
        df.to_parquet(temp_path, index=False)
        temp_path.replace(path)

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        elif "datetime" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            elif getattr(df["datetime"].dtype, "tz", None) is None:
                df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
            else:
                df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert("UTC")

        # OPT-6: Skip string→numeric loop when all non-datetime cols are already numeric
        # (cache path — parquet write already normalized types; fast-path early exit)
        non_dt = [c for c in df.columns if c != "datetime"]
        if non_dt and all(pd.api.types.is_numeric_dtype(df[c]) for c in non_dt):
            return df

        # Automatically normalize any object/string columns (except datetime) to numeric
        # to prevent PyArrow's ArrowInvalid type-mix errors (e.g. for close_time, ignore, etc.)
        for col in df.columns:
            if col == "datetime":
                continue
            if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0:
                    df[col] = converted

        return df

    def collect_and_save(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        fetch_network: bool = True,
    ) -> pd.DataFrame:
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)
        cache_df = self._load_cache(symbol, timeframe)

        if not fetch_network:
            if cache_df.empty:
                return pd.DataFrame()
            mask = (cache_df["datetime"] >= req_start) & (cache_df["datetime"] <= req_end)
            return cache_df.loc[mask]  # OPT-7: boolean indexing always returns a copy

        meta = self._load_meta().get(self._meta_key(symbol, timeframe), {})
        ea = meta.get("earliest_available")
        if ea:
            ea_dt = pd.to_datetime(ea, utc=True)
            if req_start < ea_dt:
                req_start = ea_dt

        fetch_tasks = []
        if cache_df.empty:
            fetch_tasks.append((req_start, req_end))
        else:
            c_start, c_end = cache_df["datetime"].min(), cache_df["datetime"].max()
            if req_start < c_start:
                fetch_tasks.append((req_start, c_start))
            if req_end > c_end:
                fetch_tasks.append((c_end, req_end))

        new_dfs = []
        for f_start, f_end in fetch_tasks:
            chunk = self.client.fetch_ohlcv_with_taker(symbol, timeframe, str(f_start), str(f_end))
            if not chunk.empty:
                new_dfs.append(self._normalize_df(chunk))

        if new_dfs:
            combined = pd.concat([cache_df, *new_dfs]).drop_duplicates(subset=["timestamp"])
            combined.sort_values("timestamp", inplace=True)
            self._save_cache(symbol, timeframe, combined)
            self._save_meta(
                {
                    self._meta_key(symbol, timeframe): {
                        "earliest_available": str(combined["datetime"].min()),
                        "latest_available": str(combined["datetime"].max()),
                    }
                }
            )
            cache_df = combined

        mask = (cache_df["datetime"] >= req_start) & (cache_df["datetime"] <= req_end)
        return cache_df.loc[mask].copy()

    def ensure_ohlcv_data(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> None:
        """Generic OHLCV data collection: Vision for bulk, API for recent."""
        req_start, req_end = (
            pd.to_datetime(start_date, utc=True),
            pd.to_datetime(end_date, utc=True),
        )

        # 1. 메타데이터 사전 검사 (디스크 I/O 최적화)
        meta = self._load_meta().get(self._meta_key(symbol, timeframe), {})
        ea = meta.get("earliest_available")
        la = meta.get("latest_available")
        if ea and la:
            try:
                ea_dt = pd.to_datetime(ea, utc=True)
                la_dt = pd.to_datetime(la, utc=True)
                if ea_dt <= req_start and la_dt >= req_end - pd.Timedelta(hours=8):
                    return
            except Exception as exc:
                self.logger.debug(
                    "Failed to parse metadata for %s %s: %s", symbol, timeframe, exc
                )

        cache_df = self._load_cache(symbol, timeframe)
        if (
            not cache_df.empty
            and cache_df["datetime"].min() <= req_start
            and cache_df["datetime"].max() >= req_end - pd.Timedelta(hours=8)
        ):
            return

        # API cutoff: data from the last 32 days might not be in Vision monthly archives
        api_cutoff = pd.Timestamp.now(tz="UTC").replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ) - pd.Timedelta(days=32)

        new_parts = []

        # 1. Vision monthly archives for full months in the past
        vision_symbol = symbol.replace("/", "")
        current_month_start = req_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        vision_tasks = []
        while current_month_start < min(req_end, api_cutoff):
            # Check if this month is already in cache_df
            month_end = (current_month_start + pd.offsets.MonthEnd(1)).replace(
                hour=23, minute=59, second=59
            )
            if (
                cache_df.empty
                or cache_df["datetime"].min() > current_month_start
                or cache_df["datetime"].max() < month_end
            ):
                vision_tasks.append((current_month_start.year, current_month_start.month))
            current_month_start += pd.offsets.MonthBegin(1)

        if vision_tasks:
            vision = BinanceVisionDownloader()

            def _fetch_month(year: int, month: int) -> pd.DataFrame:
                v_df = vision.fetch_klines_archive_monthly(vision_symbol, timeframe, year, month)
                if not v_df.empty:
                    v_df.columns = [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_vol",
                        "no_trades",
                        "taker_buy_base",
                        "taker_buy_quote",
                        "ignore",
                    ][: v_df.shape[1]]
                    for col in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "quote_vol",
                        "taker_buy_base",
                        "taker_buy_quote",
                    ]:
                        if col in v_df.columns:
                            v_df[col] = pd.to_numeric(v_df[col], errors="coerce")
                    return self._normalize_df(v_df)
                return pd.DataFrame()

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_task = {
                    executor.submit(_fetch_month, y, m): (y, m) for y, m in vision_tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        res_df = future.result()
                        if not res_df.empty:
                            new_parts.append(res_df)
                    except Exception as e:
                        self.logger.warning(f"Error fetching vision data for {symbol}: {e}")

        # 2. API for recent data or gaps
        
        # [Fix] 과거 데이터 갭 백필 (req_start ~ 캐시 최저점)
        if not cache_df.empty:
            cache_min_dt = cache_df["datetime"].min()
            
            # 상장일 프로필 조회하여 시작점 보정
            effective_req_start = req_start
            try:
                from src.domain.futures.universe.storage import _load_symbol_sync_profiles
                profiles = _load_symbol_sync_profiles()
                if symbol in profiles and profiles[symbol].onboard_date is not None:
                    onboard_dt = pd.to_datetime(profiles[symbol].onboard_date, utc=True)
                    if effective_req_start < onboard_dt:
                        effective_req_start = onboard_dt
            except Exception as exc:
                self.logger.debug(
                    "Failed to resolve onboard_date for %s: %s", symbol, exc
                )

            # metadata에서 이전에 성공적으로 조회해 본 최소 시점(earliest_searched) 확인
            earliest_searched = meta.get("earliest_searched")
            already_searched = False
            if earliest_searched:
                try:
                    es_dt = pd.to_datetime(earliest_searched, utc=True)
                    if effective_req_start >= es_dt:
                        already_searched = True
                except Exception as exc:
                    self.logger.debug("Parsing earliest_searched failed: %s", exc)

            # 상장일 보정 후 시작 시점보다 캐시 최저점이 뒤에 있고,
            # 그 시간차가 24시간 이상이고, 아직 조회하지 않은 범위인 경우에만 과거 갭 백필 진행
            time_gap = cache_min_dt - effective_req_start
            should_backfill = (
                not already_searched
                and effective_req_start < cache_min_dt
                and time_gap > pd.Timedelta(hours=24)
            )
            if should_backfill:
                gap_start = effective_req_start
                gap_end = cache_min_dt
                if not self._is_range_blocked_by_permanent_failure(
                    symbol=symbol,
                    timeframe=timeframe,
                    requested_start=gap_start,
                    requested_end=gap_end,
                ):
                    try:
                        gap_chunk = self.client.fetch_ohlcv_with_taker(
                            symbol, timeframe, str(gap_start), str(gap_end)
                        )
                        if not gap_chunk.empty:
                            new_parts.append(self._normalize_df(gap_chunk))
                        # 성공적으로 조회했으므로 (데이터가 비어있더라도)
                        # earliest_searched를 업데이트하여 무한 루프 방지
                        self._save_meta(
                            {
                                self._meta_key(symbol, timeframe): {
                                    "earliest_searched": str(gap_start),
                                }
                            }
                        )
                    except BinanceKlinePermanentError as exc:
                        self._record_permanent_fetch_failure(
                            symbol=symbol,
                            timeframe=timeframe,
                            error=exc,
                            requested_start=gap_start,
                            requested_end=gap_end,
                        )
                        self.logger.warning(
                            "Permanent OHLCV API failure for past gap %s %s (%d). range=%s..%s",
                            symbol,
                            timeframe,
                            exc.http_code,
                            gap_start,
                            gap_end,
                        )

        latest_cached_dt = cache_df["datetime"].max() if not cache_df.empty else None
        for part in new_parts:
            if part.empty or "datetime" not in part.columns:
                continue
            part_max_dt = pd.to_datetime(part["datetime"], utc=True).max()
            if pd.isna(part_max_dt):
                continue
            if latest_cached_dt is None or part_max_dt > latest_cached_dt:
                latest_cached_dt = part_max_dt

        remaining_start = (
            max(req_start, latest_cached_dt) if latest_cached_dt is not None else req_start
        )
        if remaining_start < req_end and not self._is_range_blocked_by_permanent_failure(
            symbol=symbol,
            timeframe=timeframe,
            requested_start=remaining_start,
            requested_end=req_end,
        ):
            try:
                chunk = self.client.fetch_ohlcv_with_taker(
                    symbol, timeframe, str(remaining_start), str(req_end)
                )
                if not chunk.empty:
                    new_parts.append(self._normalize_df(chunk))
            except BinanceKlinePermanentError as exc:
                self._record_permanent_fetch_failure(
                    symbol=symbol,
                    timeframe=timeframe,
                    error=exc,
                    requested_start=remaining_start,
                    requested_end=req_end,
                )
                self.logger.warning(
                    "Permanent OHLCV API failure for %s %s (%d). range=%s..%s",
                    symbol,
                    timeframe,
                    exc.http_code,
                    remaining_start,
                    req_end,
                )

        if new_parts:
            if not cache_df.empty and "timestamp" in cache_df.columns:
                cache_df["timestamp"] = pd.to_numeric(cache_df["timestamp"], errors="coerce")
            combined = (
                pd.concat([cache_df, *new_parts])
                .drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
            )
            self._save_cache(symbol, timeframe, combined)
            self._save_meta(
                {
                    self._meta_key(symbol, timeframe): {
                        "earliest_available": str(combined["datetime"].min()),
                        "latest_available": str(combined["datetime"].max()),
                        "earliest_searched": str(req_start),
                    }
                }
            )

    def collect_1m_ohlcv(
        self, symbol: str, start_date: str, end_date: str, fetch_network: bool = True
    ) -> pd.DataFrame:
        timeframe = "1m"
        req_start, req_end = (
            pd.to_datetime(start_date, utc=True),
            pd.to_datetime(end_date, utc=True),
        )
        cache_df = self._load_cache(symbol, timeframe)

        if not fetch_network:
            if cache_df.empty:
                return pd.DataFrame()
            mask = (cache_df["datetime"] >= req_start) & (cache_df["datetime"] <= req_end)
            return cache_df.loc[mask].copy()

        cache_covers_range = (
            not cache_df.empty
            and cache_df["datetime"].min() <= req_start
            and cache_df["datetime"].max() >= req_end
        )

        if not cache_covers_range:
            with self._collect_1m_semaphore:
                self.ensure_1m_data(symbol, start_date, end_date)
            cache_df = self._load_cache(symbol, timeframe)

        mask = (cache_df["datetime"] >= req_start) & (cache_df["datetime"] <= req_end)
        return cache_df.loc[mask].copy()

    def ensure_1m_data(self, symbol: str, start_date: str, end_date: str) -> None:
        """Optimized 1m data collection: Vision for bulk, API for recent."""
        timeframe = "1m"
        req_start, req_end = (
            pd.to_datetime(start_date, utc=True),
            pd.to_datetime(end_date, utc=True),
        )

        # 1. 메타데이터 사전 검사 (디스크 I/O 최적화)
        meta = self._load_meta().get(self._meta_key(symbol, timeframe), {})
        ea = meta.get("earliest_available")
        la = meta.get("latest_available")
        if ea and la:
            try:
                ea_dt = pd.to_datetime(ea, utc=True)
                la_dt = pd.to_datetime(la, utc=True)
                if ea_dt <= req_start and la_dt >= req_end - pd.Timedelta(hours=8):
                    return
            except Exception as exc:
                self.logger.debug(
                    "Failed to parse metadata for %s %s: %s", symbol, timeframe, exc
                )

        cache_df = self._load_cache(symbol, timeframe)
        if (
            not cache_df.empty
            and cache_df["datetime"].min() <= req_start
            and cache_df["datetime"].max() >= req_end - pd.Timedelta(hours=8)
        ):
            return

        # API cutoff: data from the last 35 days might not be in Vision monthly archives
        api_cutoff = pd.Timestamp.now(tz="UTC").replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ) - pd.Timedelta(days=32)

        new_parts = []

        # 1. Vision monthly archives for full months in the past
        vision_symbol = symbol.replace("/", "")
        current_month_start = req_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        vision_tasks = []
        while current_month_start < min(req_end, api_cutoff):
            # Check if this month is already in cache_df
            month_end = (current_month_start + pd.offsets.MonthEnd(1)).replace(
                hour=23, minute=59, second=59
            )
            if (
                cache_df.empty
                or cache_df["datetime"].min() > current_month_start
                or cache_df["datetime"].max() < month_end
            ):
                vision_tasks.append((current_month_start.year, current_month_start.month))
            current_month_start += pd.offsets.MonthBegin(1)

        if vision_tasks:
            vision = BinanceVisionDownloader()

            def _fetch_month(year: int, month: int) -> pd.DataFrame:
                v_df = vision.fetch_klines_archive_monthly(vision_symbol, "1m", year, month)
                if not v_df.empty:
                    # Klines columns include timestamp, OHLCV, and taker volume fields.
                    v_df.columns = [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_asset_volume",
                        "number_of_trades",
                        "taker_buy_base_volume",
                        "taker_buy_quote_volume",
                        "ignore",
                    ]
                    # Convert to numeric
                    for col in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "taker_buy_base_volume",
                        "taker_buy_quote_volume",
                    ]:
                        v_df[col] = pd.to_numeric(v_df[col], errors="coerce")

                    v_df = v_df[
                        [
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "taker_buy_base_volume",
                            "taker_buy_quote_volume",
                        ]
                    ]
                    return self._normalize_df(v_df)
                return pd.DataFrame()

            # 무리하지 않게 4개의 스레드로 제한하여 I/O 대기와 파싱(CPU)을 교차 병렬화
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_task = {
                    executor.submit(_fetch_month, y, m): (y, m) for y, m in vision_tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        res_df = future.result()
                        if not res_df.empty:
                            new_parts.append(res_df)
                    except Exception as e:
                        self.logger.warning(f"Error fetching vision data for {symbol}: {e}")

        # 2. API for recent data or gaps
        
        # [Fix] 과거 데이터 갭 백필 (req_start ~ 캐시 최저점)
        if not cache_df.empty:
            cache_min_dt = cache_df["datetime"].min()
            
            # 상장일 프로필 조회하여 시작점 보정
            effective_req_start = req_start
            try:
                from src.domain.futures.universe.storage import _load_symbol_sync_profiles
                profiles = _load_symbol_sync_profiles()
                if symbol in profiles and profiles[symbol].onboard_date is not None:
                    onboard_dt = pd.to_datetime(profiles[symbol].onboard_date, utc=True)
                    if effective_req_start < onboard_dt:
                        effective_req_start = onboard_dt
            except Exception as exc:
                self.logger.debug(
                    "Failed to resolve onboard_date for %s: %s", symbol, exc
                )

            # metadata에서 이전에 성공적으로 조회해 본 최소 시점(earliest_searched) 확인
            earliest_searched = meta.get("earliest_searched")
            already_searched = False
            if earliest_searched:
                try:
                    es_dt = pd.to_datetime(earliest_searched, utc=True)
                    if effective_req_start >= es_dt:
                        already_searched = True
                except Exception as exc:
                    self.logger.debug("Parsing earliest_searched failed: %s", exc)

            # 상장일 보정 후 시작 시점보다 캐시 최저점이 뒤에 있고,
            # 그 시간차가 24시간 이상이고, 아직 조회하지 않은 범위인 경우에만 과거 갭 백필 진행
            time_gap = cache_min_dt - effective_req_start
            should_backfill = (
                not already_searched
                and effective_req_start < cache_min_dt
                and time_gap > pd.Timedelta(hours=24)
            )
            if should_backfill:
                gap_start = effective_req_start
                gap_end = cache_min_dt
                if not self._is_range_blocked_by_permanent_failure(
                    symbol=symbol,
                    timeframe=timeframe,
                    requested_start=gap_start,
                    requested_end=gap_end,
                ):
                    try:
                        gap_chunk = self.client.fetch_ohlcv_with_taker(
                            symbol, timeframe, str(gap_start), str(gap_end)
                        )
                        if not gap_chunk.empty:
                            new_parts.append(self._normalize_df(gap_chunk))
                        # 성공적으로 조회했으므로 (데이터가 비어있더라도)
                        # earliest_searched를 업데이트하여 무한 루프 방지
                        self._save_meta(
                            {
                                self._meta_key(symbol, timeframe): {
                                    "earliest_searched": str(gap_start),
                                }
                            }
                        )
                    except BinanceKlinePermanentError as exc:
                        self._record_permanent_fetch_failure(
                            symbol=symbol,
                            timeframe=timeframe,
                            error=exc,
                            requested_start=gap_start,
                            requested_end=gap_end,
                        )
                        self.logger.warning(
                            "Permanent OHLCV API failure for 1m past gap %s (%d). range=%s..%s",
                            symbol,
                            exc.http_code,
                            gap_start,
                            gap_end,
                        )

        latest_cached_dt = cache_df["datetime"].max() if not cache_df.empty else None
        for part in new_parts:
            if part.empty or "datetime" not in part.columns:
                continue
            part_max_dt = pd.to_datetime(part["datetime"], utc=True).max()
            if pd.isna(part_max_dt):
                continue
            if latest_cached_dt is None or part_max_dt > latest_cached_dt:
                latest_cached_dt = part_max_dt

        remaining_start = (
            max(req_start, latest_cached_dt) if latest_cached_dt is not None else req_start
        )
        if remaining_start < req_end and not self._is_range_blocked_by_permanent_failure(
            symbol=symbol,
            timeframe=timeframe,
            requested_start=remaining_start,
            requested_end=req_end,
        ):
            try:
                chunk = self.client.fetch_ohlcv_with_taker(
                    symbol, timeframe, str(remaining_start), str(req_end)
                )
                if not chunk.empty:
                    new_parts.append(self._normalize_df(chunk))
            except BinanceKlinePermanentError as exc:
                self._record_permanent_fetch_failure(
                    symbol=symbol,
                    timeframe=timeframe,
                    error=exc,
                    requested_start=remaining_start,
                    requested_end=req_end,
                )
                self.logger.warning(
                    "Permanent OHLCV API failure for 1m %s (%d). range=%s..%s",
                    symbol,
                    exc.http_code,
                    remaining_start,
                    req_end,
                )

        if new_parts:
            if not cache_df.empty and "timestamp" in cache_df.columns:
                cache_df["timestamp"] = pd.to_numeric(cache_df["timestamp"], errors="coerce")
            combined = (
                pd.concat([cache_df, *new_parts])
                .drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
            )
            self._save_cache(symbol, timeframe, combined)
            self._save_meta(
                {
                    self._meta_key(symbol, timeframe): {
                        "earliest_available": str(combined["datetime"].min()),
                        "latest_available": str(combined["datetime"].max()),
                        "earliest_searched": str(req_start),
                    }
                }
            )

    def ensure_metrics_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Ensure metrics data (OI, LSR) is cached and up to date.

        Args:
            symbol: Trading pair symbol.
            start_date: Start date string.
            end_date: End date string.

        Returns:
            DataFrame containing the requested metrics.

        """
        safe_symbol = self._safe_symbol(symbol)
        path = FUTURES_DATA_DIR / f"{safe_symbol}_metrics.parquet"
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)
        cache_df = _empty_metrics_frame()
        if path.exists():
            try:
                cache_df = _normalize_metrics_frame(pd.read_parquet(path), symbol=symbol)
            except Exception as exc:
                self.logger.warning(
                    "metrics cache read failed; fallback to rebuild symbol=%s error=%s",
                    symbol,
                    type(exc).__name__,
                )
                cache_df = _empty_metrics_frame()

        api_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=28)
        vision_floor = pd.Timestamp("2020-09-01", tz="UTC")
        new_parts: list[pd.DataFrame] = []
        vision = BinanceVisionDownloader()

        if cache_df.empty or cache_df["datetime"].min() > req_start:
            vision_end = min(
                req_end,
                api_cutoff,
                cache_df["datetime"].min() if not cache_df.empty else req_end,
            )
            vision_start = max(req_start, vision_floor)
            if vision_start < vision_end:
                vision_df = vision.fetch_range_metrics(symbol.replace("/", ""), vision_start, vision_end)
                if not vision_df.empty:
                    new_parts.append(vision_df)

        if req_end >= api_cutoff:
            recent_start = max(
                req_start,
                api_cutoff,
                cache_df["datetime"].max() if not cache_df.empty else api_cutoff,
            )
            if recent_start < req_end:
                since = int(recent_start.timestamp() * 1000)
                until = int(req_end.timestamp() * 1000)
                oi = self.client.fetch_open_interest_history(symbol, "4h", since, until=until)
                lsr = self.client.fetch_long_short_ratio_history(symbol, "4h", since, until=until)
                merged_recent = _coalesce_metrics_frames([oi, lsr], symbol=symbol)
                if not merged_recent.empty:
                    new_parts.append(merged_recent)

        combined = _coalesce_metrics_frames([cache_df, *new_parts], symbol=symbol)
        if not combined.empty:
            temp_path = path.with_suffix(".tmp.parquet")
            combined.to_parquet(temp_path, index=False)
            temp_path.replace(path)
            coverage = {
                f"{col}_coverage": float(combined[col].notna().mean())
                for col in _METRICS_NUMERIC_COLUMNS
            }
            self._save_meta(
                {
                    self._meta_key(symbol, "metrics"): {
                        "earliest_available": str(combined["available_at"].min()),
                        "latest_available": str(combined["available_at"].max()),
                        "earliest_timestamp": str(combined["datetime"].min()),
                        "latest_timestamp": str(combined["datetime"].max()),
                        **coverage,
                    }
                }
            )
            cache_df = combined
        else:
            cache_df = _empty_metrics_frame()

        if cache_df.empty:
            return cache_df
        mask = (cache_df["datetime"] >= req_start) & (cache_df["datetime"] <= req_end)
        return cache_df.loc[mask].copy()

    def ensure_funding_data(self, symbol: str, start_date: str, end_date: str) -> None:
        """Ensure funding rate history is cached and up to date.

        Uses Vision for bulk history and API for recent data.
        """
        safe_symbol = self._safe_symbol(symbol)
        path = FUTURES_DATA_DIR / f"{safe_symbol}_funding.parquet"
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)

        # 1. 메타데이터 사전 검사 (디스크 I/O 최적화)
        meta = self._load_meta().get(self._meta_key(symbol, "funding"), {})
        ea = meta.get("earliest_available")
        la = meta.get("latest_available")
        if ea and la:
            try:
                ea_dt = pd.to_datetime(ea, utc=True)
                la_dt = pd.to_datetime(la, utc=True)
                if (
                    ea_dt <= req_start + pd.Timedelta(days=1)
                    and la_dt >= req_end - pd.Timedelta(hours=12)
                ):
                    return
            except Exception as exc:
                self.logger.debug(
                    "Failed to parse metadata for %s funding: %s", symbol, exc
                )

        cache_df = pd.DataFrame()
        if path.exists():
            try:
                cache_df = _normalize_funding_frame(pd.read_parquet(path))
            except Exception as e:
                self.logger.warning(
                    "funding cache read failed; fallback to rebuild symbol=%s error=%s",
                    symbol,
                    type(e).__name__,
                )
                cache_df = pd.DataFrame(columns=["timestamp", "funding_rate", "datetime"])

        if (
            not cache_df.empty
            and (cache_df["datetime"].min() <= req_start + pd.Timedelta(days=1))
            and cache_df["datetime"].max() >= req_end - pd.Timedelta(hours=12)
        ):
            # 메타데이터에 등록되어 있지 않은 경우 채워넣어 다음 호출 시 Parquet read 방지
            if not ea or not la:
                self._save_meta(
                    {
                        self._meta_key(symbol, "funding"): {
                            "earliest_available": str(cache_df["datetime"].min()),
                            "latest_available": str(cache_df["datetime"].max()),
                        }
                    }
                )
            return

        api_cutoff = pd.Timestamp.now(tz="UTC").replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ) - pd.Timedelta(days=32)
        new_parts = []
        vision_symbol = symbol.replace("/", "")
        current_month_start = req_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        vision_tasks = []
        while current_month_start < min(req_end, api_cutoff):
            month_end = (current_month_start + pd.offsets.MonthEnd(1)).replace(
                hour=23, minute=59, second=59
            )
            if (
                cache_df.empty
                or cache_df["datetime"].min() > current_month_start
                or cache_df["datetime"].max() < month_end
            ):
                vision_tasks.append((current_month_start.year, current_month_start.month))
            current_month_start += pd.offsets.MonthBegin(1)

        if vision_tasks:
            vision = BinanceVisionDownloader()

            def _fetch_month_funding(year: int, month: int) -> pd.DataFrame:
                v_df = vision.fetch_funding_rate_monthly(vision_symbol, year, month)
                if not v_df.empty:
                    return _normalize_funding_frame(v_df)
                return pd.DataFrame()

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_task = {
                    executor.submit(_fetch_month_funding, y, m): (y, m) for y, m in vision_tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        res_df = future.result()
                        if not res_df.empty:
                            new_parts.append(res_df)
                    except Exception as e:
                        self.logger.warning(f"Error fetching vision funding data for {symbol}: {e}")

        latest_cached_dt = cache_df["datetime"].max() if not cache_df.empty else None
        for part in new_parts:
            if part.empty or "datetime" not in part.columns:
                continue
            part_max_dt = part["datetime"].max()
            if pd.isna(part_max_dt):
                continue
            if latest_cached_dt is None or part_max_dt > latest_cached_dt:
                latest_cached_dt = part_max_dt

        remaining_start = (
            max(req_start, latest_cached_dt) if latest_cached_dt is not None else req_start
        )
        if remaining_start < req_end:
            new_funding = self.client.fetch_funding_rate_history(
                symbol, str(remaining_start), str(req_end)
            )
            if not new_funding.empty:
                new_parts.append(_normalize_funding_frame(new_funding))

        if new_parts:
            clean_parts = [_normalize_funding_frame(part) for part in new_parts if not part.empty]
            clean_parts = [part for part in clean_parts if not part.empty]
            if not clean_parts and cache_df.empty:
                return
            combined = (
                pd.concat([cache_df, *clean_parts], ignore_index=True)
                .drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
            )
            _normalize_funding_frame(combined).to_parquet(path, index=False)
            self._save_meta(
                {
                    self._meta_key(symbol, "funding"): {
                        "earliest_available": str(combined["datetime"].min()),
                        "latest_available": str(combined["datetime"].max()),
                    }
                }
            )


# --- Merging Utilities (from funding_utils.py and metrics_utils.py) ---


def merge_funding_into_ohlcv(symbol: str, df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Merge funding rate information into OHLCV."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    out = df.copy()
    for col in ["funding_rate", "funding_event_count", "funding_rate_sum"]:
        if col not in out.columns:
            out[col] = 0.0

    path = Path(data_dir) / f"{symbol.replace('/', '_')}_funding.parquet"
    if not path.exists():
        return out

    fr_df = pd.read_parquet(path)
    if fr_df.empty:
        return out

    # Simple asof merge
    out["timestamp"] = pd.to_datetime(out["datetime"]).astype("int64") // 10**6
    fr_df["timestamp"] = pd.to_datetime(fr_df["timestamp"], unit="ms").astype("int64") // 10**6

    # Merge only the funding columns from fr_df and keep the real values.
    out = out.drop(
        columns=["funding_rate", "funding_event_count", "funding_rate_sum"],
        errors="ignore",
    )

    exclude_fr = ["datetime", "symbol"]
    cols_fr = [c for c in fr_df.columns if c not in exclude_fr]

    out = pd.merge_asof(
        out.sort_values("timestamp"),
        fr_df[cols_fr].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return out


def merge_metrics_into_ohlcv(
    symbol: str,
    df: pd.DataFrame,
    data_dir: Path,
    *,
    tolerance: pd.Timedelta = _METRICS_MERGE_TOLERANCE,
) -> pd.DataFrame:
    """Merge metrics (OI, LSR) into OHLCV."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    path = Path(data_dir) / f"{symbol.replace('/', '_')}_metrics.parquet"
    if not path.exists():
        return df.copy()

    metrics_df = _normalize_metrics_frame(pd.read_parquet(path), symbol=symbol)
    if metrics_df.empty:
        return df.copy()

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["symbol"] = symbol

    metrics_prepared = metrics_df.copy()
    metrics_prepared = metrics_prepared.sort_values("available_at").reset_index(drop=True)
    metrics_prepared["symbol"] = symbol
    metrics_prepared = metrics_prepared.drop_duplicates(subset=["available_at", "symbol"], keep="last")
    metrics_prepared = metrics_prepared.rename(columns={"available_at": "metrics_available_at"})

    merged = pd.merge_asof(
        out,
        metrics_prepared,
        left_on="timestamp",
        right_on="metrics_available_at",
        by="symbol",
        direction="backward",
        tolerance=tolerance,
        allow_exact_matches=True,
    )
    return merged.drop(columns=["symbol"], errors="ignore")


def fetch_premiumindex_bulk(
    symbol: str,
    start_date: date,
    end_date: date,
    interval: str = "1m",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """premiumIndexKlines를 날짜 범위로 일괄 수집.

    이미 캐시된 날짜는 skip.
    반환: columns=[open_time, open, high, low, close, ...], index=UTC datetime
    """
    downloader = BinanceVisionDownloader()
    dfs: list[pd.DataFrame] = []
    curr = start_date
    safe_symbol = symbol.replace("/", "").replace("_", "")

    while curr <= end_date:
        date_str = curr.strftime("%Y-%m-%d")
        cache_file = None
        if cache_dir is not None:
            cache_file = Path(cache_dir) / safe_symbol / f"{date_str}.parquet"

        if cache_file is not None and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                dfs.append(df)
            except Exception as e:
                _logger.warning("Failed to read cache file %s: %s", cache_file, e)
                cache_file = None

        if cache_file is None or not cache_file.exists():
            dt = datetime.combine(curr, datetime.min.time())
            try:
                df = downloader.fetch_premiumindex_daily(safe_symbol, dt)
                if not df.empty:
                    if df.shape[1] >= 6:
                        cols = [
                            "open_time",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "close_time",
                            "quote_asset_volume",
                            "number_of_trades",
                            "taker_buy_base_asset_volume",
                            "taker_buy_quote_asset_volume",
                            "ignore",
                        ][: df.shape[1]]
                        df.columns = cols
                    if cache_dir is not None and cache_file is not None:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        df.to_parquet(cache_file, index=False)
                    dfs.append(df)
            except Exception as e:
                _logger.warning(
                    "Failed to fetch premium index for %s on %s: %s",
                    symbol,
                    date_str,
                    e,
                )

        curr += timedelta(days=1)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if "open_time" in combined.columns:
        combined = combined.drop_duplicates(subset=["open_time"]).sort_values("open_time")
        combined.index = pd.to_datetime(combined["open_time"], unit="ms", utc=True)
    return combined


def build_mark_price_1m_array(
    symbols: list[str],
    start_ts: int,
    end_ts: int,
    cache_dir: Path,
) -> np.ndarray:
    """각 심볼의 premiumIndex close를 로드하여 [B_1m, N] 배열 구성.

    결측 구간: forward-fill (직전 값). 완전 결측 심볼: NaN column 허용.
    B_1m = (end_ts - start_ts) / (60 * 1000) 기준 정렬.
    """
    start_dt = pd.to_datetime(start_ts, unit="ms", utc=True)
    end_dt = pd.to_datetime(end_ts, unit="ms", utc=True)

    expected_len = (end_ts - start_ts) // 60000
    if expected_len <= 0:
        return np.empty((0, len(symbols)))

    target_index = pd.date_range(start=start_dt, periods=expected_len, freq="1min", tz="UTC")

    out_df = pd.DataFrame(index=target_index)

    for symbol in symbols:
        try:
            df = fetch_premiumindex_bulk(
                symbol=symbol,
                start_date=start_dt.date(),
                end_date=end_dt.date(),
                cache_dir=cache_dir,
            )
            if not df.empty and "close" in df.columns:
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df = df[~df.index.duplicated(keep="first")]
                s_series = df["close"].reindex(target_index, method="ffill")
                s_series = s_series.ffill().bfill()
                out_df[symbol] = s_series
            else:
                out_df[symbol] = np.nan
        except Exception as e:
            _logger.warning("Error building mark price for %s: %s", symbol, e)
            out_df[symbol] = np.nan

    return np.asarray(out_df.to_numpy(dtype=np.float64), dtype=np.float64)
