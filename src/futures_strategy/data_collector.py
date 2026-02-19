import pandas as pd
from .binance_client import BinanceClient
import sys
import os
import re
import json

# 프로젝트 루트 경로 추가 (모듈 import 문제 해결)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.settings import DATA_DIR, FUTURES_BACKTEST_START_DATE, FUTURES_BACKTEST_END_DATE
from src.common.utils import setup_logger

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
        df.set_index('datetime', inplace=True, drop=False)
        df.sort_index(inplace=True)
        
        expected_diff = {
            '1h': pd.Timedelta(hours=1),
            '1d': pd.Timedelta(days=1),
            '4h': pd.Timedelta(hours=4)
        }.get(timeframe)
        
        if expected_diff:
            time_diff = df.index.to_series().diff().dropna()
            gaps = time_diff[time_diff != expected_diff]
            if not gaps.empty:
                issues.append(f"Found {len(gaps)} time gaps. First gap at {gaps.index[0]}")

        # 3. 논리적 오류 확인 (Low > High, Open/Close outside High/Low)
        invalid_high_low = df[df['high'] < df['low']]
        if not invalid_high_low.empty:
            issues.append(f"High < Low detected in {len(invalid_high_low)} rows")
            
        return issues

class DataCollector:
    def __init__(self, api_key=None, secret=None):
        self.client = BinanceClient(api_key, secret)
        self.logger = setup_logger("DataCollector")

    @staticmethod
    def _safe_symbol(symbol):
        return symbol.replace('/', '_')

    def _cache_path(self, symbol, timeframe):
        safe_symbol = self._safe_symbol(symbol)
        return DATA_DIR / f"{safe_symbol}_{timeframe}.parquet"

    def _meta_path(self):
        return DATA_DIR / "parquet_cache_meta.json"

    def _meta_key(self, symbol, timeframe):
        return f"{self._safe_symbol(symbol)}::{timeframe}"

    def _load_meta(self):
        path = self._meta_path()
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_meta(self, meta):
        path = self._meta_path()
        tmp = path.with_suffix(".tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def _normalize_df(self, df):
        if df is None or df.empty:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'datetime'])

        out = df.copy()
        if 'datetime' not in out.columns and 'timestamp' in out.columns:
            out['datetime'] = pd.to_datetime(out['timestamp'], unit='ms')
        elif 'datetime' in out.columns:
            out['datetime'] = pd.to_datetime(out['datetime'])

        out = out.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        return out

    def _read_parquet(self, path):
        try:
            df = pd.read_parquet(path)
        except ImportError as e:
            raise RuntimeError(
                "Parquet engine is not installed. Please install 'pyarrow' (recommended) or 'fastparquet'."
            ) from e
        return self._normalize_df(df)

    def _write_parquet(self, path, df):
        temp_path = path.with_suffix(".tmp.parquet")
        try:
            df.to_parquet(temp_path, index=False)
        except ImportError as e:
            raise RuntimeError(
                "Parquet engine is not installed. Please install 'pyarrow' (recommended) or 'fastparquet'."
            ) from e
        temp_path.replace(path)

    def _load_legacy_csv_seed(self, symbol, timeframe):
        safe_symbol = self._safe_symbol(symbol)
        pattern = f"{safe_symbol}_{timeframe}_*.csv"
        candidates = sorted(DATA_DIR.glob(pattern))
        if not candidates:
            return pd.DataFrame()

        frames = []
        for p in candidates:
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()
        return self._normalize_df(pd.concat(frames, ignore_index=True))

    def _slice_by_date(self, df, start_date, end_date):
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        if df.empty:
            return df
        return df[(df['datetime'] >= start_ts) & (df['datetime'] <= end_ts)].copy()

    def _timeframe_to_timedelta(self, timeframe):
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

    def _find_internal_gap_ranges(self, df, timeframe, start_date, end_date):
        """
        Detect missing internal candles and return fetch ranges as date pairs.
        Note: API fetch granularity is date-level in this project, so ranges are day-based.
        """
        step = self._timeframe_to_timedelta(timeframe)
        if step is None or df.empty:
            return []

        sliced = self._slice_by_date(df, start_date, end_date)
        if sliced.empty:
            return []

        ts = sliced["timestamp"].drop_duplicates().sort_values().to_numpy(dtype="int64")
        if len(ts) < 2:
            return []

        expected_ms = int(step / pd.Timedelta(milliseconds=1))
        if expected_ms <= 0:
            return []

        gap_ranges = []
        for i in range(1, len(ts)):
            diff = int(ts[i] - ts[i - 1])
            if diff > expected_ms:
                gap_start_ms = ts[i - 1] + expected_ms
                gap_end_ms = ts[i] - expected_ms
                if gap_start_ms <= gap_end_ms:
                    ds = pd.to_datetime(gap_start_ms, unit="ms").normalize()
                    de = pd.to_datetime(gap_end_ms, unit="ms").normalize()
                    gap_ranges.append((ds, de))

        if not gap_ranges:
            return []

        # Merge overlapping/adjacent day ranges.
        gap_ranges.sort(key=lambda x: x[0])
        merged = [gap_ranges[0]]
        for cur_s, cur_e in gap_ranges[1:]:
            prev_s, prev_e = merged[-1]
            if cur_s <= (prev_e + pd.Timedelta(days=1)):
                merged[-1] = (prev_s, max(prev_e, cur_e))
            else:
                merged.append((cur_s, cur_e))
        return merged

    def _fetch_ranges(self, symbol, timeframe, ranges):
        fetched_frames = []
        for miss_start, miss_end in ranges:
            if miss_start > miss_end:
                continue
            s = pd.Timestamp(miss_start).strftime("%Y-%m-%d")
            e = pd.Timestamp(miss_end).strftime("%Y-%m-%d")
            self.logger.info(f"Fetching missing range for {symbol} {timeframe}: {s} ~ {e}")
            fetched = self.client.fetch_ohlcv(symbol, timeframe, s, e)
            fetched = self._normalize_df(fetched)
            if not fetched.empty:
                validator = DataValidator()
                issues = validator.validate(fetched.copy(), symbol, timeframe)
                if issues:
                    self.logger.warning("Data Validation Issues Found:")
                    for issue in issues:
                        self.logger.warning(f"- {issue}")
                fetched_frames.append(fetched)
        return fetched_frames

    def ensure_data(self, symbol, timeframe, start_date=None, end_date=None):
        """
        Parquet-based range cache with incremental fetch:
        - cache file: data/{symbol}_{timeframe}.parquet
        - only missing front/back date ranges are fetched from API
        """
        start = start_date or FUTURES_BACKTEST_START_DATE
        end = end_date or FUTURES_BACKTEST_END_DATE
        req_start = pd.Timestamp(start).normalize()
        req_end = pd.Timestamp(end).normalize()

        if req_start > req_end:
            raise ValueError(f"Invalid date range: start={start}, end={end}")

        cache_path = self._cache_path(symbol, timeframe)
        meta = self._load_meta()
        mk = self._meta_key(symbol, timeframe)
        earliest_known = None
        if mk in meta and isinstance(meta[mk], dict):
            v = meta[mk].get("earliest_available")
            if v:
                try:
                    earliest_known = pd.Timestamp(v).normalize()
                except Exception:
                    earliest_known = None

        if cache_path.exists():
            cache_df = self._read_parquet(cache_path)
        else:
            cache_df = self._load_legacy_csv_seed(symbol, timeframe)
            if not cache_df.empty:
                self.logger.info(f"Seeded parquet cache from legacy CSV files: {cache_path.name}")
                self._write_parquet(cache_path, cache_df)

        missing_ranges = []
        if cache_df.empty:
            missing_ranges.append((req_start, req_end))
        else:
            cache_start = cache_df['datetime'].min().normalize()
            cache_end = cache_df['datetime'].max().normalize()
            if req_start < cache_start:
                # If we already learned listing start (earliest_available), skip repeated pre-listing fetches.
                if earliest_known is not None and req_start < earliest_known:
                    pass
                else:
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
            self.logger.info(f"Updated parquet cache: {cache_path}")
        elif not cache_path.exists() and not cache_df.empty:
            self._write_parquet(cache_path, cache_df)

        # Internal gap backfill within the requested range.
        internal_gap_ranges = self._find_internal_gap_ranges(cache_df, timeframe, start, end)
        if internal_gap_ranges:
            self.logger.info(
                f"Detected {len(internal_gap_ranges)} internal gap range(s) for {symbol} {timeframe}. Backfilling."
            )
            gap_frames = self._fetch_ranges(symbol, timeframe, internal_gap_ranges)
            if gap_frames:
                all_frames = [cache_df]
                all_frames.extend(gap_frames)
                merged = self._normalize_df(pd.concat(all_frames, ignore_index=True))
                self._write_parquet(cache_path, merged)
                cache_df = merged
                self.logger.info(f"Backfilled internal gaps and updated cache: {cache_path}")

        # Persist earliest available date to avoid repeated pre-listing downloads.
        if not cache_df.empty:
            earliest = cache_df["datetime"].min().normalize().strftime("%Y-%m-%d")
            prev = None
            if mk in meta and isinstance(meta[mk], dict):
                prev = meta[mk].get("earliest_available")
            if prev != earliest:
                if mk not in meta or not isinstance(meta[mk], dict):
                    meta[mk] = {}
                meta[mk]["earliest_available"] = earliest
                self._save_meta(meta)

        return self._slice_by_date(cache_df, start, end)

    def collect_and_save(self, symbol, timeframe, start_date=None, end_date=None):
        """
        Backward-compatible API.
        Previously stored period-specific CSV files.
        Now stores/updates a single parquet cache and returns requested slice.
        """
        return self.ensure_data(symbol, timeframe, start_date, end_date)

if __name__ == "__main__":
    # 테스트 실행
    collector = DataCollector()
    collector.collect_and_save("BTC/USDT", "1d", "2022-01-01", "2024-12-31")
    collector.collect_and_save("BTC/USDT", "1h", "2022-01-01", "2024-12-31")
