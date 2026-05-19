"""
Data Loading and Preparation for Futures.
Combines Data Collection (API/Vision), Metadata management, and Merging of 
Funding/Metrics into OHLCV.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import FUTURES_DATA_DIR
from src.core.exchange.binance_client import BinanceClient
from src.core.utils.binance_vision import BinanceVisionDownloader
from src.core.utils.utils import setup_logger

_logger = logging.getLogger("DataCollector")


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

    rows = int(len(df))
    cols = int(len(df.columns))
    total_cells = max(rows * cols, 1)
    nan_pct = float(df.isna().sum().sum() / total_cells)
    num_df = df.select_dtypes(include=[np.number])
    inf_count = float(np.isinf(num_df.to_numpy(dtype=np.float64, copy=False)).sum()) if not num_df.empty else 0.0
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
            if "_" not in stem: continue
            base, quote = stem.rsplit("_", 1)
            out.append(f"{base}/{quote}")
        return sorted(set(out))

    def _meta_path(self) -> Path:
        return FUTURES_DATA_DIR / "parquet_cache_meta.json"

    def _meta_key(self, symbol: str, timeframe: str) -> str:
        return f"{self._safe_symbol(symbol)}::{timeframe}"

    def _load_meta(self) -> dict[str, Any]:
        path = self._meta_path()
        if not path.exists(): return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception: return {}

    def _save_meta(self, meta_updates: dict[str, Any]) -> None:
        path = self._meta_path()
        lock_path = path.with_suffix(".lock")
        with self._meta_lock:
            try:
                with open(lock_path, "w") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    current_meta = self._load_meta()
                    for mk, updates in meta_updates.items():
                        if mk not in current_meta: current_meta[mk] = {}
                        if isinstance(updates, dict) and isinstance(current_meta[mk], dict):
                            current_meta[mk].update(updates)
                        else: current_meta[mk] = updates
                    tmp = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}.json")
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(current_meta, f, ensure_ascii=False, indent=2)
                    os.replace(tmp, path)
            except Exception as e: self.logger.error(f"Failed to save metadata: {e}")

    def _load_cache(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self._cache_path(symbol, timeframe)
        if not path.exists(): return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
            if df.empty or ("timestamp" not in df.columns and "datetime" not in df.columns):
                path.unlink()
                return pd.DataFrame()
            return self._normalize_df(df)
        except Exception:
            try: path.unlink()
            except Exception: pass
            return pd.DataFrame()

    def _save_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df.empty: return
        path = self._cache_path(symbol, timeframe)
        temp_path = path.with_suffix(".tmp.parquet")
        df.to_parquet(temp_path, index=False)
        temp_path.replace(path)

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        if "timestamp" in df.columns:
            # Ensure timestamp is numeric to prevent pandas from parsing it as a date string
            # and to avoid TypeError during sort_values if there's a mix of int and str
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df

    def collect_and_save(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)
        cache_df = self._load_cache(symbol, timeframe)
        meta = self._load_meta().get(self._meta_key(symbol, timeframe), {})
        ea = meta.get("earliest_available")
        if ea:
            ea_dt = pd.to_datetime(ea, utc=True)
            if req_start < ea_dt: req_start = ea_dt

        fetch_tasks = []
        if cache_df.empty: fetch_tasks.append((req_start, req_end))
        else:
            c_start, c_end = cache_df["datetime"].min(), cache_df["datetime"].max()
            if req_start < c_start: fetch_tasks.append((req_start, c_start))
            if req_end > c_end: fetch_tasks.append((c_end, req_end))
        
        new_dfs = []
        for f_start, f_end in fetch_tasks:
            chunk = self.client.fetch_ohlcv_with_taker(symbol, timeframe, str(f_start), str(f_end))
            if not chunk.empty: new_dfs.append(self._normalize_df(chunk))

        if new_dfs:
            combined = pd.concat([cache_df, *new_dfs]).drop_duplicates(subset=["timestamp"])
            combined.sort_values("timestamp", inplace=True)
            self._save_cache(symbol, timeframe, combined)
            self._save_meta({self._meta_key(symbol, timeframe): {"earliest_available": str(combined["datetime"].min())}})
            cache_df = combined

        mask = (cache_df["datetime"] >= req_start) & (cache_df["datetime"] <= req_end)
        return cache_df.loc[mask].copy()

    def collect_1m_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        timeframe = "1m"
        req_start, req_end = pd.to_datetime(start_date, utc=True), pd.to_datetime(end_date, utc=True)
        cache_df = self._load_cache(symbol, timeframe)
        new_dfs = []
        fetch_tasks = []
        if cache_df.empty: fetch_tasks.append((req_start, req_end))
        else:
            c_s, c_e = cache_df["datetime"].min(), cache_df["datetime"].max()
            if req_start < c_s: fetch_tasks.append((req_start, c_s))
            if req_end > c_e: fetch_tasks.append((c_e, req_end))

        if fetch_tasks:
            with self._collect_1m_semaphore:
                for f_s, f_e in fetch_tasks:
                    chunk = self.client.fetch_ohlcv_with_taker(symbol, timeframe, str(f_s), str(f_e))
                    if not chunk.empty: new_dfs.append(self._normalize_df(chunk))
        
        if new_dfs:
            combined = pd.concat([cache_df, *new_dfs]).drop_duplicates(subset=["timestamp"])
            combined.sort_values("timestamp", inplace=True)
            self._save_cache(symbol, timeframe, combined)
            cache_df = combined

        mask = (cache_df["datetime"] >= req_start) & (cache_df["datetime"] <= req_end)
        return cache_df.loc[mask].copy()

    def ensure_1m_data(self, symbol: str, start_date: str, end_date: str) -> None:
        """Optimized 1m data collection: Vision for bulk, API for recent."""
        timeframe = "1m"
        req_start, req_end = pd.to_datetime(start_date, utc=True), pd.to_datetime(end_date, utc=True)
        
        cache_df = self._load_cache(symbol, timeframe)
        if not cache_df.empty and cache_df["datetime"].min() <= req_start and cache_df["datetime"].max() >= req_end:
            return

        # API cutoff: data from the last 35 days might not be in Vision monthly archives
        api_cutoff = pd.Timestamp.now(tz="UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(days=32)
        
        new_parts = []
        
        # 1. Vision monthly archives for full months in the past
        vision_symbol = symbol.replace("/", "")
        current_month_start = req_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        vision_tasks = []
        while current_month_start < min(req_end, api_cutoff):
            # Check if this month is already in cache_df
            month_end = (current_month_start + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
            if cache_df.empty or cache_df["datetime"].min() > current_month_start or cache_df["datetime"].max() < month_end:
                vision_tasks.append((current_month_start.year, current_month_start.month))
            current_month_start += pd.offsets.MonthBegin(1)

        if vision_tasks:
            from src.core.utils.binance_vision import BinanceVisionDownloader
            import concurrent.futures
            
            vision = BinanceVisionDownloader()

            def _fetch_month(year: int, month: int) -> pd.DataFrame:
                v_df = vision.fetch_klines_archive_monthly(vision_symbol, "1m", year, month)
                if not v_df.empty:
                    # Klines columns: timestamp, open, high, low, close, volume, close_time, quote_asset_volume, trades, taker_buy_base, taker_buy_quote, ignore
                    v_df.columns = [
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_asset_volume", "number_of_trades",
                        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
                    ]
                    # Convert to numeric
                    for col in ["open", "high", "low", "close", "volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
                        v_df[col] = pd.to_numeric(v_df[col], errors="coerce")
                    
                    v_df = v_df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base_volume", "taker_buy_quote_volume"]]
                    return self._normalize_df(v_df)
                return pd.DataFrame()

            # 무리하지 않게 4개의 스레드로 제한하여 I/O 대기와 파싱(CPU)을 교차 병렬화
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_task = {executor.submit(_fetch_month, y, m): (y, m) for y, m in vision_tasks}
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        res_df = future.result()
                        if not res_df.empty:
                            new_parts.append(res_df)
                    except Exception as e:
                        self.logger.warning(f"Error fetching vision data for {symbol}: {e}")

        # 2. API for recent data or gaps
        remaining_start = max(req_start, cache_df["datetime"].max() if not cache_df.empty else req_start)
        if remaining_start < req_end:
            chunk = self.client.fetch_ohlcv_with_taker(symbol, timeframe, str(remaining_start), str(req_end))
            if not chunk.empty:
                new_parts.append(self._normalize_df(chunk))

        if new_parts:
            if not cache_df.empty and "timestamp" in cache_df.columns:
                cache_df["timestamp"] = pd.to_numeric(cache_df["timestamp"], errors="coerce")
            combined = pd.concat([cache_df, *new_parts]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            self._save_cache(symbol, timeframe, combined)

    def ensure_metrics_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        safe_symbol = self._safe_symbol(symbol)
        path = FUTURES_DATA_DIR / f"{safe_symbol}_metrics.parquet"
        req_start, req_end = pd.to_datetime(start_date, utc=True), pd.to_datetime(end_date, utc=True)
        
        cache_df = pd.DataFrame()
        if path.exists(): cache_df = pd.read_parquet(path)
        
        # simplified logic for Vision/API collection
        api_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=28)
        new_parts = []
        if cache_df.empty or cache_df["datetime"].min() > req_start:
            v_start = req_start
            v_end = min(req_end, api_cutoff, cache_df["datetime"].min() if not cache_df.empty else req_end)
            if v_start < v_end:
                v_df = BinanceVisionDownloader().fetch_range_metrics(symbol.replace("/", ""), v_start, v_end)
                if not v_df.empty:
                    v_df = self._normalize_df(v_df)
                    new_parts.append(v_df)
        
        if req_end >= api_cutoff:
            a_start = max(req_start, api_cutoff, cache_df["datetime"].max() if not cache_df.empty else api_cutoff)
            if a_start < req_end:
                since = int(a_start.timestamp() * 1000)
                oi = self.client.fetch_open_interest_history(symbol, "1h", since)
                if not oi.empty:
                    oi = self._normalize_df(oi)
                    new_parts.append(oi)

        if new_parts:
            combined = pd.concat([cache_df, *new_parts]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            combined = self._normalize_df(combined)
            combined.to_parquet(path, index=False)
            cache_df = combined
        
        mask = (cache_df["datetime"] >= req_start) & (cache_df["datetime"] <= req_end)
        return cache_df.loc[mask].copy()

    def ensure_funding_data(self, symbol: str, start_date: str, end_date: str) -> None:
        safe_symbol = self._safe_symbol(symbol)
        path = FUTURES_DATA_DIR / f"{safe_symbol}_funding.parquet"
        req_start, req_end = pd.to_datetime(start_date, utc=True), pd.to_datetime(end_date, utc=True)
        
        cache_df = pd.DataFrame()
        if path.exists(): cache_df = pd.read_parquet(path)
        
        if not cache_df.empty and cache_df["datetime"].min() <= req_start and cache_df["datetime"].max() >= req_end:
            return
        
        new_funding = self.client.fetch_funding_rate_history(symbol, str(req_start), str(req_end))
        if not new_funding.empty:
            new_funding["datetime"] = pd.to_datetime(new_funding["timestamp"], unit="ms", utc=True)
            combined = pd.concat([cache_df, new_funding]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            combined.to_parquet(path, index=False)

# --- Merging Utilities (from funding_utils.py and metrics_utils.py) ---

def merge_funding_into_ohlcv(symbol: str, df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Merge funding rate information into OHLCV."""
    if df is None or df.empty: return df.copy() if df is not None else pd.DataFrame()
    out = df.copy()
    for col in ["funding_rate", "funding_event_count", "funding_rate_sum"]:
        if col not in out.columns: out[col] = 0.0
    
    path = Path(data_dir) / f"{symbol.replace('/', '_')}_funding.parquet"
    if not path.exists(): return out
    
    fr_df = pd.read_parquet(path)
    if fr_df.empty: return out
    
    # Simple asof merge
    out["timestamp"] = pd.to_datetime(out["datetime"]).astype("int64") // 10**6
    fr_df["timestamp"] = pd.to_datetime(fr_df["timestamp"], unit="ms").astype("int64") // 10**6
    
    # Exclude columns already in out (except timestamp) to avoid _x/_y suffixes
    exclude = ["timestamp", "datetime", "symbol", "funding_rate", "funding_event_count", "funding_rate_sum"]
    cols = [c for c in fr_df.columns if c not in exclude]
    fr_cols_to_merge = ["timestamp"] + cols
    
    # We should merge only the new columns from fr_df, but fr_df IS providing funding_rate.
    # Wait, fr_df has "funding_rate", so we want to keep it from fr_df, and overwrite the dummy 0.0 in out.
    # So we should drop the dummy ones from out first!
    out = out.drop(columns=["funding_rate", "funding_event_count", "funding_rate_sum"], errors="ignore")
    
    exclude_fr = ["datetime", "symbol"]
    cols_fr = [c for c in fr_df.columns if c not in exclude_fr]
    
    out = pd.merge_asof(out.sort_values("timestamp"), fr_df[cols_fr].sort_values("timestamp"), on="timestamp", direction="backward")
    return out

def merge_metrics_into_ohlcv(symbol: str, df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Merge metrics (OI, LSR) into OHLCV."""
    if df is None or df.empty: return df.copy() if df is not None else pd.DataFrame()
    path = Path(data_dir) / f"{symbol.replace('/', '_')}_metrics.parquet"
    if not path.exists(): return df

    m_df = pd.read_parquet(path)
    if m_df.empty: return df

    m_df["timestamp"] = pd.to_datetime(m_df["datetime"]).astype("int64") // 10**6
    df["timestamp"] = pd.to_datetime(df["datetime"]).astype("int64") // 10**6    
    exclude = ["timestamp", "datetime", "create_time", "symbol"]
    cols = [c for c in m_df.columns if c not in exclude]
    return pd.merge_asof(df.sort_values("timestamp"), m_df[["timestamp"] + cols].sort_values("timestamp"), on="timestamp", direction="backward")
