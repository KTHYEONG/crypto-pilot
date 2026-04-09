import pandas as pd
from src.core.exchange.upbit_client import UpbitClient, UpbitOhlcvFetchError
import sys
import os
import re
import json
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import DATA_DIR, SPOT_DATA_DIR, SPOT_BACKTEST_START_DATE, SPOT_BACKTEST_END_DATE
from src.core.utils.utils import setup_logger

class DataValidator:
    @staticmethod
    def validate(df, symbol, timeframe):
        """데이터 무결성 검증"""
        issues = []
        if df.isnull().any().any():
            missing_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"Missing values in columns: {missing_cols}")
            
        df.set_index('datetime', inplace=True, drop=False)
        df.sort_index(inplace=True)
        
        expected_diff = {
            "1d": pd.Timedelta(days=1),
            "4h": pd.Timedelta(hours=4),
        }.get(timeframe)
        
        if expected_diff:
            time_diff = df.index.to_series().diff().dropna()
            gaps = time_diff[time_diff != expected_diff]
            if not gaps.empty:
                issues.append(f"Found {len(gaps)} time gaps. First gap at {gaps.index[0]}")

        invalid_high_low = df[df['high'] < df['low']]
        if not invalid_high_low.empty:
            issues.append(f"High < Low detected in {len(invalid_high_low)} rows")
            
        return issues

class DataCollectorSpot:
    """
    Upbit Spot Data Collector.
    Mimics DataCollector but uses UpbitClient.
    """
    def __init__(self, api_key=None, secret=None):
        self.client = UpbitClient(api_key, secret)
        self.logger = setup_logger("DataCollectorSpot")

    @staticmethod
    def _safe_symbol(symbol):
        # Upbit symbols KRW-BTC -> KRW_BTC
        return symbol.replace('/', '_').replace('-', '_')

    def _cache_path(self, symbol, timeframe):
        safe_symbol = self._safe_symbol(symbol)
        return SPOT_DATA_DIR / f"spot_{safe_symbol}_{timeframe}.parquet"

    def _meta_path(self):
        return SPOT_DATA_DIR / "parquet_spot_cache_meta.json"

    def _meta_key(self, symbol, timeframe):
        return f"spot::{self._safe_symbol(symbol)}::{timeframe}"

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
            return self._normalize_df(df)
        except Exception as e:
            self.logger.error(f"Error reading parquet {path}: {e}")
            return self._normalize_df(None)

    def _write_parquet(self, path, df):
        temp_path = path.with_suffix(".tmp.parquet")
        try:
            df.to_parquet(temp_path, index=False)
            temp_path.replace(path)
        except Exception as e:
            self.logger.error(f"Error writing parquet {path}: {e}")

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
        if u == "m": return pd.Timedelta(minutes=n)
        if u == "h": return pd.Timedelta(hours=n)
        if u == "d": return pd.Timedelta(days=n)
        if u == "w": return pd.Timedelta(weeks=n)
        return None

    def _fetch_ranges(self, symbol, timeframe, ranges):
        fetched_frames = []
        for miss_start, miss_end in ranges:
            if miss_start > miss_end:
                continue
            s = pd.Timestamp(miss_start).strftime("%Y-%m-%d")
            e = pd.Timestamp(miss_end).strftime("%Y-%m-%d")
            self.logger.info(f"Fetching missing Spot range for {symbol} {timeframe}: {s} ~ {e}")
            try:
                fetched = self.client.fetch_ohlcv(symbol, timeframe, s, e)
            except UpbitOhlcvFetchError as exc:
                self.logger.error(
                    "Upbit OHLCV fetch failed for %s %s %s~%s: partial_rows=%s since_ms=%s",
                    symbol,
                    timeframe,
                    s,
                    e,
                    len(exc.partial_ohlcv),
                    exc.since_ms,
                )
                raise
            fetched = self._normalize_df(fetched)
            if not fetched.empty:
                validator = DataValidator()
                issues = validator.validate(fetched.copy(), symbol, timeframe)
                if issues:
                    self.logger.warning(f"Data Validation Issues for {symbol} {timeframe}: {issues}")
                fetched_frames.append(fetched)
        return fetched_frames

    def ensure_data(self, symbol, timeframe, start_date=None, end_date=None):
        start = start_date or SPOT_BACKTEST_START_DATE
        end = end_date or SPOT_BACKTEST_END_DATE
        req_start = pd.Timestamp(start).normalize()
        req_end = pd.Timestamp(end).normalize()

        cache_path = self._cache_path(symbol, timeframe)
        meta = self._load_meta()
        mk = self._meta_key(symbol, timeframe)
        
        cache_df = pd.DataFrame()
        if cache_path.exists():
            cache_df = self._read_parquet(cache_path)

        missing_ranges = []
        if cache_df.empty:
            missing_ranges.append((req_start, req_end))
        else:
            cache_start = cache_df['datetime'].min().normalize()
            cache_end = cache_df['datetime'].max().normalize()
            
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

    def collect_and_save(self, symbol, timeframe, start_date=None, end_date=None):
        return self.ensure_data(symbol, timeframe, start_date, end_date)
