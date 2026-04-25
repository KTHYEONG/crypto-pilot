from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

# 프로젝트 루트 경로 추가 (E402 방지 및 모듈 import 문제 해결)
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import (  # noqa: E402
    SPOT_BACKTEST_END_DATE,
    SPOT_BACKTEST_START_DATE,
    SPOT_DATA_DIR,
)
from src.core.exchange.upbit_client import UpbitClient, UpbitOhlcvFetchError  # noqa: E402
from src.core.utils.utils import setup_logger  # noqa: E402

logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for Upbit spot data integrity."""

    @staticmethod
    def validate(df: pd.DataFrame, symbol: str, timeframe: str) -> list[str]:
        """데이터 무결성 검증.

        1. 결측치 확인
        2. 시간 연속성 확인
        3. 논리적 오류 확인 (Low > High)
        """
        issues: list[str] = []
        if df.isnull().any().any():
            missing_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"Missing values in columns: {missing_cols}")

        df_sorted = df.sort_values("datetime")
        df_sorted.set_index("datetime", inplace=True, drop=False)

        expected_diff = {
            "1d": pd.Timedelta(days=1),
            "4h": pd.Timedelta(hours=4),
        }.get(timeframe)

        if expected_diff:
            time_diff = df_sorted.index.to_series().diff().dropna()
            gaps = time_diff[time_diff != expected_diff]
            if not gaps.empty:
                issues.append(f"Found {len(gaps)} time gaps. First gap at {gaps.index[0]}")

        invalid_high_low = df[df["high"] < df["low"]]
        if not invalid_high_low.empty:
            issues.append(f"High < Low detected in {len(invalid_high_low)} rows")

        return issues


class DataCollectorSpot:
    """Upbit Spot Data Collector.

    Mimics DataCollector but uses UpbitClient.
    """

    def __init__(self, api_key: str | None = None, secret: str | None = None) -> None:
        """Initialize DataCollectorSpot."""
        self.client = UpbitClient(api_key, secret)
        self.logger = setup_logger("DataCollectorSpot")

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        """Upbit symbols KRW-BTC -> KRW_BTC."""
        return symbol.replace("/", "_").replace("-", "_")

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get path to the local parquet cache."""
        safe_symbol = self._safe_symbol(symbol)
        return SPOT_DATA_DIR / f"spot_{safe_symbol}_{timeframe}.parquet"

    def _meta_path(self) -> Path:
        """Get path to the metadata JSON file."""
        return SPOT_DATA_DIR / "parquet_spot_cache_meta.json"

    def _meta_key(self, symbol: str, timeframe: str) -> str:
        """Get unique key for the symbol-timeframe pair."""
        return f"spot::{self._safe_symbol(symbol)}::{timeframe}"

    def _load_meta(self) -> dict[str, Any]:
        """Load metadata from JSON."""
        path = self._meta_path()
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_meta(self, meta: dict[str, Any]) -> None:
        """Save metadata to JSON."""
        path = self._meta_path()
        tmp = path.with_suffix(".tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _normalize_df(self, df: pd.DataFrame | None) -> pd.DataFrame:
        """Ensure standard OHLCV columns and datetime format."""
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"]
            )

        out = df.copy()
        if "datetime" not in out.columns and "timestamp" in out.columns:
            out["datetime"] = pd.to_datetime(out["timestamp"], unit="ms")
        elif "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"])

        out = (
            out.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return out

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        """Read parquet cache and normalize."""
        try:
            df = pd.read_parquet(path)
            return self._normalize_df(df)
        except Exception as e:
            self.logger.error(f"Error reading parquet {path}: {e}")
            return self._normalize_df(None)

    def _write_parquet(self, path: Path, df: pd.DataFrame) -> None:
        """Write DataFrame to parquet atomically."""
        temp_path = path.with_suffix(".tmp.parquet")
        try:
            df.to_parquet(temp_path, index=False)
            os.replace(temp_path, path)
        except Exception as e:
            self.logger.error(f"Error writing parquet {path}: {e}")

    def _slice_by_date(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Slice DataFrame by given date range."""
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        if df.empty:
            return df
        return df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].copy()

    def _timeframe_to_timedelta(self, timeframe: str) -> pd.Timedelta | None:
        """Convert timeframe string to pd.Timedelta."""
        m = re.match(r"^(\d+)([mhdw])$", str(timeframe).strip().lower())
        if not m:
            return None
        n = int(m.group(1))
        u = m.group(2)
        if u == "m":
            return pd.Timedelta(minutes=n)
        if u == "h":
            return pd.Timedelta(hours=n)
        if u == "d":
            return pd.Timedelta(days=n)
        if u == "w":
            return pd.Timedelta(weeks=n)
        return None

    def _fetch_ranges(
        self,
        symbol: str,
        timeframe: str,
        ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
    ) -> list[pd.DataFrame]:
        """Fetch multiple OHLCV ranges from the API."""
        fetched_frames: list[pd.DataFrame] = []
        for miss_start, miss_end in ranges:
            if miss_start > miss_end:
                continue
            s_ts = int(miss_start.timestamp() * 1000)
            e_ts = int(miss_end.timestamp() * 1000)
            s_str = miss_start.strftime("%Y-%m-%d")
            e_str = miss_end.strftime("%Y-%m-%d")
            self.logger.info(
                f"Fetching missing Spot range for {symbol} {timeframe}: {s_str} ~ {e_str}"
            )
            try:
                # [FIX] fetch_ohlcv -> fetch_ohlcv_all
                fetched = self.client.fetch_ohlcv_all(symbol, timeframe, s_ts, e_ts)
            except UpbitOhlcvFetchError as exc:
                partial_len = len(exc.partial_ohlcv) if exc.partial_ohlcv else 0
                self.logger.error(
                    "Upbit OHLCV fetch failed for %s %s %s~%s: partial_rows=%s since_ms=%s",
                    symbol,
                    timeframe,
                    s_str,
                    e_str,
                    partial_len,
                    exc.since_ms,
                )
                raise
            fetched_norm = self._normalize_df(fetched)
            if not fetched_norm.empty:
                validator = DataValidator()
                issues = validator.validate(fetched_norm.copy(), symbol, timeframe)
                if issues:
                    self.logger.warning(
                        f"Data Validation Issues for {symbol} {timeframe}: {issues}"
                    )
                fetched_frames.append(fetched_norm)
        return fetched_frames

    def ensure_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Ensure local data exists and fetch missing parts if necessary."""
        start = start_date or SPOT_BACKTEST_START_DATE
        end = end_date or SPOT_BACKTEST_END_DATE
        req_start = pd.Timestamp(start).normalize()
        req_end = pd.Timestamp(end).normalize()

        cache_path = self._cache_path(symbol, timeframe)

        cache_df = pd.DataFrame()
        if cache_path.exists():
            cache_df = self._read_parquet(cache_path)

        missing_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if cache_df.empty:
            missing_ranges.append((req_start, req_end))
        else:
            cache_start = cast("pd.Timestamp", cache_df["datetime"].min().normalize())
            cache_end = cast("pd.Timestamp", cache_df["datetime"].max().normalize())

            if req_start < cache_start:
                missing_ranges.append((req_start, cache_start - pd.Timedelta(days=1)))
            if req_end > cache_end:
                missing_ranges.append((cache_end + pd.Timedelta(days=1), req_end))

        fetched_frames = self._fetch_ranges(symbol, timeframe, missing_ranges)

        if fetched_frames:
            all_frames = [cache_df] if not cache_df.empty else []
            all_frames.extend(fetched_frames)
            merged = self._normalize_df(pd.concat(all_frames, ignore_index=True))
            self._write_parquet(cache_path, merged)
            cache_df = merged
        elif not cache_path.exists() and not cache_df.empty:
            self._write_parquet(cache_path, cache_df)

        return self._slice_by_date(cache_df, start, end)

    def collect_and_save(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Wrap ensure_data for consistency."""
        return self.ensure_data(symbol, timeframe, start_date, end_date)
