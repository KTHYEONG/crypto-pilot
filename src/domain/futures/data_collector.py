from __future__ import annotations

import fcntl
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from typing import Any

from src.core.exchange.binance_client import BinanceClient
from src.core.utils.utils import setup_logger
from config.settings import FUTURES_DATA_DIR

# 프로젝트 루트 경로 추가 (모듈 import 문제 해결)
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DataValidator:
    @staticmethod
    def validate(df, symbol, timeframe):
        """데이터 무결성 검증"""
        issues = []

        # 1. 결측치 확인
        if df.isnull().any().any():
            missing_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"Missing values in columns: {missing_cols}")

        # 2. 시간 연속성 확인
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

        # 3. 논리적 오류 확인 (Low > High, Open/Close outside High/Low)
        invalid_high_low = df[df["high"] < df["low"]]
        if not invalid_high_low.empty:
            issues.append(f"High < Low detected in {len(invalid_high_low)} rows")

        return issues


class DataCollector:
    def __init__(self, api_key: str | None = None, secret: str | None = None) -> None:
        self.client = BinanceClient(api_key, secret)
        self.logger = setup_logger("DataCollector")

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        return symbol.replace("/", "_")

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = self._safe_symbol(symbol)
        return FUTURES_DATA_DIR / f"{safe_symbol}_{timeframe}.parquet"

    def _meta_path(self) -> Path:
        return FUTURES_DATA_DIR / "parquet_cache_meta.json"

    def _meta_key(self, symbol: str, timeframe: str) -> str:
        return f"{self._safe_symbol(symbol)}::{timeframe}"

    def _load_meta(self) -> dict[str, Any]:
        path = self._meta_path()
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_meta(self, meta_updates: dict[str, Any]):
        """
        Concurrency-aware metadata update with deep merge and file locking.
        Prevents losing 'earliest_available' when updating 'last_checked' 
        and avoids race conditions during file replacement.
        """
        path = self._meta_path()
        lock_path = path.with_suffix(".lock")

        try:
            # Use a separate lock file to coordinate access
            with open(lock_path, "w") as lock_file:
                # Acquire exclusive lock (blocking)
                fcntl.flock(lock_file, fcntl.LOCK_EX)

                current_meta = self._load_meta()

                for mk, updates in meta_updates.items():
                    if mk not in current_meta:
                        current_meta[mk] = {}
                    if isinstance(updates, dict) and isinstance(current_meta[mk], dict):
                        current_meta[mk].update(updates)
                    else:
                        current_meta[mk] = updates

                # Use process-unique temp file to avoid concurrent replace errors
                tmp = path.with_suffix(f".tmp.{os.getpid()}.json")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(current_meta, f, ensure_ascii=False, indent=2)

                # Atomic replace
                os.replace(tmp, path)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
        finally:
            # Cleanup temp file if it exists
            tmp_pattern = path.with_suffix(f".tmp.{os.getpid()}.json")
            if tmp_pattern.exists():
                try:
                    tmp_pattern.unlink()
                except Exception:
                    pass

    TAKE_COLUMNS = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "datetime",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    )

    def _load_cache(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self._cache_path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
        except ImportError as e:
            raise RuntimeError(
                "Parquet engine is not installed. Install 'pyarrow' or 'fastparquet'."
            ) from e
        return self._normalize_df(df)

    def _save_cache(self, symbol: str, timeframe: str, df: pd.DataFrame):
        path = self._cache_path(symbol, timeframe)
        temp_path = path.with_suffix(".tmp.parquet")
        try:
            df.to_parquet(temp_path, index=False)
        except ImportError as e:
            raise RuntimeError(
                "Parquet engine is not installed. Install 'pyarrow' or 'fastparquet'."
            ) from e
        temp_path.replace(path)

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Do NOT early-return on empty: dtype normalization must run even for empty DataFrames.
        # An empty parquet (e.g. delisted symbol) retains its tz-naive datetime64[ms] schema;
        # skipping normalization causes "Invalid comparison between dtype=datetime64[ms] and
        # Timestamp" when the empty df is concat-ed with fetched data and then filtered.
        if "datetime" not in df.columns:
            if "timestamp" in df.columns and not df.empty:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df

    def collect_and_save(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """데이터 수집, 로컬 캐시 결합 및 저장"""
        self.logger.info(f"Collecting {symbol} {timeframe} ({start_date} ~ {end_date})")

        # 1. 로컬 캐시 로드
        cache_df = self._load_cache(symbol, timeframe)
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)

        # 2. 메타데이터 로드 (상장일 및 기 체크 정보)
        meta_key = self._meta_key(symbol, timeframe)
        meta = self._load_meta().get(meta_key, {})
        earliest_available = meta.get("earliest_available")
        if earliest_available:
            earliest_available = pd.to_datetime(earliest_available, utc=True)
            if req_start < earliest_available:
                req_start = earliest_available

        # 3. 필요한 구간 판단
        fetch_needed = True
        if not cache_df.empty:
            c_start = cache_df["datetime"].min()
            c_end = cache_df["datetime"].max()

            # 요청 구간이 캐시 범위 내에 있으면 fetch 불필요
            if c_start <= req_start and c_end >= req_end:
                fetch_needed = False

        new_df = pd.DataFrame()
        if fetch_needed:
            # 바이낸스에서 데이터 가져오기
            new_df = self.client.fetch_ohlcv(symbol, timeframe, start_date, end_date)
            if not new_df.empty:
                new_df = self._normalize_df(new_df)
                # 실제 데이터의 시작점 기록 (상장일 추정)
                actual_start = new_df["datetime"].min()
                if not earliest_available or actual_start < earliest_available:
                    self._save_meta({meta_key: {"earliest_available": str(actual_start)}})

        # 4. 데이터 병합 및 중복 제거
        combined_df = pd.concat([cache_df, new_df]).drop_duplicates(subset=["timestamp"])
        combined_df.sort_values("timestamp", inplace=True)

        # 5. 캐시 저장
        if not new_df.empty or cache_df.empty:
            self._save_cache(symbol, timeframe, combined_df)
            self._save_meta({meta_key: {"last_checked": str(pd.Timestamp.now(tz="UTC"))}})

        # 6. 요청한 범위만 필터링하여 반환
        mask = (combined_df["datetime"] >= req_start) & (combined_df["datetime"] <= req_end)
        return combined_df.loc[mask].copy()

    def _fetch_ranges(self, symbol: str, timeframe: str, ranges: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[pd.DataFrame]:
        results = []
        for start, end in ranges:
            df = self.client.fetch_ohlcv(symbol, timeframe, str(start), str(end))
            if not df.empty:
                results.append(self._normalize_df(df))
        return results

    def collect_1m_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load or fetch 1m OHLCV (Binance USDT-M futures); cache `{safe_symbol}_1m.parquet`.
        Uses the same incremental merge pattern as collect_and_save.
        """
        timeframe = "1m"
        self.logger.info("Collecting %s %s (%s ~ %s)", symbol, timeframe, start_date, end_date)
        cache_df = self._load_cache(symbol, timeframe)
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)

        meta_key = self._meta_key(symbol, timeframe)
        meta = self._load_meta().get(meta_key, {})
        earliest_available = meta.get("earliest_available")
        if earliest_available:
            earliest_available = pd.to_datetime(earliest_available, utc=True)
            if req_start < earliest_available:
                req_start = earliest_available

        fetch_needed = True
        if not cache_df.empty:
            c_start = cache_df["datetime"].min()
            c_end = cache_df["datetime"].max()
            if c_start <= req_start and c_end >= req_end:
                fetch_needed = False

        new_df = pd.DataFrame()
        if fetch_needed:
            new_df = self.client.fetch_ohlcv_with_taker(symbol, timeframe, start_date, end_date)
            if not new_df.empty:
                new_df = self._normalize_df(new_df)
                actual_start = new_df["datetime"].min()
                ea = pd.to_datetime(earliest_available, utc=True) if earliest_available else None
                if not earliest_available or actual_start < ea:
                    self._save_meta({meta_key: {"earliest_available": str(actual_start)}})

        combined_df = pd.concat([cache_df, new_df]).drop_duplicates(subset=["timestamp"])
        combined_df.sort_values("timestamp", inplace=True)

        if not new_df.empty or cache_df.empty:
            self._save_cache(symbol, timeframe, combined_df)
            self._save_meta({meta_key: {"last_checked": str(pd.Timestamp.now(tz="UTC"))}})

        mask = (combined_df["datetime"] >= req_start) & (combined_df["datetime"] <= req_end)
        return combined_df.loc[mask].copy()

    def ensure_funding_data(self, symbol: str, start_date: str, end_date: str):
        """펀딩 데이터 수집 및 저장 (Binance 전용)"""
        from config.settings import FUTURES_DATA_DIR
        safe_symbol = self._safe_symbol(symbol)
        path = FUTURES_DATA_DIR / f"{safe_symbol}_funding.parquet"
        meta_key = f"{safe_symbol}::funding"

        cache_df = pd.DataFrame()
        if path.exists():
            cache_df = self._normalize_df(pd.read_parquet(path))

        last_checked = None
        meta = self._load_meta().get(meta_key, {})
        if "last_checked" in meta:
            last_checked = pd.to_datetime(meta["last_checked"], utc=True)

        req_end = pd.to_datetime(end_date, utc=True)

        if last_checked and req_end <= last_checked and not cache_df.empty:
            return

        self.logger.info(f"Fetching funding rate for {symbol} from {start_date} to {end_date}...")
        new_funding = self.client.fetch_funding_rate_history(symbol, start_date, end_date)

        if not new_funding.empty:
            new_funding["datetime"] = pd.to_datetime(new_funding["timestamp"], unit="ms", utc=True)
            combined = pd.concat([cache_df, new_funding]).drop_duplicates(subset=["timestamp"])
            combined.sort_values("timestamp", inplace=True)
            combined.to_parquet(path, index=False)
            self._save_meta({meta_key: {"last_checked": str(pd.Timestamp.now(tz="UTC"))}})
            self.logger.info(f"Updated funding parquet cache: {path}")
