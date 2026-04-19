from __future__ import annotations

import fcntl
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import FUTURES_DATA_DIR
from src.core.exchange.binance_client import BinanceClient
from src.core.utils.binance_vision import BinanceVisionDownloader
from src.core.utils.utils import setup_logger

# 프로젝트 루트 경로 추가 (모듈 import 문제 해결)
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DataValidator:
    @staticmethod
    def validate(df: pd.DataFrame, symbol: str, timeframe: str) -> list[str]:
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
    _meta_lock = threading.Lock()
    # 스마트 스로틀링이 적용되었으므로 동시 수집 숫자를 3개로 확대하여 효율성 극대화
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
        """Symbols with an existing OHLCV parquet under FUTURES_DATA_DIR (delisted survivors)."""
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

    def _save_meta(self, meta_updates: dict[str, Any]) -> None:
        """
        Concurrency-aware metadata update with deep merge and file locking.
        Prevents losing 'earliest_available' when updating 'last_checked' 
        and avoids race conditions during file replacement.
        """
        path = self._meta_path()
        lock_path = path.with_suffix(".lock")

        with self._meta_lock:
            try:
                # Use a separate lock file to coordinate access across processes
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

                    # Use thread-unique temp file to avoid concurrent replace errors within process
                    tmp = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}.json")
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(current_meta, f, ensure_ascii=False, indent=2)

                    # Atomic replace
                    os.replace(tmp, path)
            except Exception as e:
                self.logger.error(f"Failed to save metadata: {e}")
            finally:
                # Cleanup temp file if it exists
                tmp_clean = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}.json")
                if tmp_clean.exists():
                    try:
                        tmp_clean.unlink()
                    except Exception:  # noqa: S110
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
            # [CRITICAL FIX] 데이터 손상 감지: 필수 컬럼이 없거나 비어있는 경우
            if df.empty or ("timestamp" not in df.columns and "datetime" not in df.columns):
                self.logger.warning(f"Corrupted cache detected for {symbol} {timeframe}. Deleting.")
                path.unlink()
                return pd.DataFrame()
        except ImportError as e:
            raise RuntimeError(
                "Parquet engine is not installed. Install 'pyarrow' or 'fastparquet'."
            ) from e
        except Exception as e:
            self.logger.warning(f"Failed to read cache {path}: {e}. Deleting.")
            try:
                path.unlink()
            except Exception:  # noqa: S110
                pass
            return pd.DataFrame()

        return self._normalize_df(df)

    def _save_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df.empty or ("timestamp" not in df.columns and "datetime" not in df.columns):
            self.logger.warning(
                f"Attempted to save empty or invalid data for {symbol} {timeframe}."
            )
            return

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
        # Avoid redundant work but ensure datetime exists and is UTC-aware.
        if df.empty:
            return df

        if "timestamp" in df.columns:
            # Re-generate from internal integer timestamp to fix any naive/bad resolution datetime
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        else:
            # 필수 컬럼 부재 시 빈 데이터프레임 반환하여 상위 로직에서 에러 감지 유도
            return pd.DataFrame()
        return df

    @staticmethod
    def _get_timeframe_delta(timeframe: str) -> pd.Timedelta:
        """타임프레임 문자열(1m, 1h, 1d 등)을 pd.Timedelta로 변환"""
        import re
        match = re.match(r"(\d+)([mhdDwW])", timeframe)
        if not match:
            # 매칭 실패 시 기본값 (1m)
            return pd.Timedelta(minutes=1)
        
        val, unit = int(match.group(1)), match.group(2).lower()
        mapping = {"m": "T", "h": "H", "d": "D", "w": "W"}
        return pd.Timedelta(f"{val}{mapping.get(unit, 'T')}")

    def _identify_middle_gaps(
        self, df: pd.DataFrame, timeframe: str, start_bound: pd.Timestamp, end_bound: pd.Timestamp
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """데이터프레임 내에서 중간에 누락된 구간(Holes)을 찾아 리스트로 반환"""
        if df.empty:
            return []
        
        expected_delta = self._get_timeframe_delta(timeframe)
        # 요청 범위 내의 데이터만 필터링하여 정렬
        mask = (df["datetime"] >= start_bound) & (df["datetime"] <= end_bound)
        sub_df = df.loc[mask].sort_values("datetime")
        
        if len(sub_df) < 2:
            return []

        # 시간 차이 계산
        diffs = sub_df["datetime"].diff()
        # 기대되는 시간 간격보다 1.5배 이상 큰 경우 구멍으로 판단 (네트워크 지연 등 고려)
        gap_mask = diffs > (expected_delta * 1.5)
        gap_indices = diffs[gap_mask].index
        
        gaps = []
        for idx in gap_indices:
            gap_end = sub_df.loc[idx, "datetime"]
            # 이전 행의 시간이 구멍의 시작점
            gap_start = sub_df.loc[sub_df.index.get_loc(idx) - 1, "datetime"]
            gaps.append((gap_start, gap_end))
            self.logger.warning(
                f"Found middle gap in {timeframe} data: {gap_start} ~ {gap_end}"
            )
        
        return gaps

    def collect_and_save(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """데이터 수집, 로컬 캐시 결합 및 저장 (Taker Volume 포함)"""
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)

        # 1. 로컬 캐시 로드
        cache_df = self._load_cache(symbol, timeframe)

        # 2. 메타데이터 로드
        meta_key = self._meta_key(symbol, timeframe)
        meta = self._load_meta().get(meta_key, {})
        earliest_available = meta.get("earliest_available")
        if earliest_available:
            ea_dt = pd.to_datetime(earliest_available, utc=True)
            if req_start < ea_dt:
                req_start = ea_dt

        # 3. 필요한 구간 판단 (Incremental + Gap Filling)
        fetch_tasks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if cache_df.empty:
            fetch_tasks.append((req_start, req_end))
        else:
            c_start = cache_df["datetime"].min()
            c_end = cache_df["datetime"].max()
            
            # 3a. 과거/미래 확장
            if req_start < c_start:
                fetch_tasks.append((req_start, c_start))
            if req_end > c_end:
                fetch_tasks.append((c_end, req_end))
            
            # 3b. 중간 구멍 탐지 (Self-Healing)
            middle_gaps = self._identify_middle_gaps(cache_df, timeframe, req_start, req_end)
            fetch_tasks.extend(middle_gaps)

        new_dfs: list[pd.DataFrame] = []
        for f_start, f_end in fetch_tasks:
            self.logger.info(
                f"Fetching {symbol} {timeframe} task (with taker): {f_start} ~ {f_end}"
            )
            # Use fetch_ohlcv_with_taker to include CVD/microstructure data for GP
            chunk = self.client.fetch_ohlcv_with_taker(symbol, timeframe, str(f_start), str(f_end))
            if not chunk.empty:
                new_dfs.append(self._normalize_df(chunk))

        # 4. 데이터 병합 및 중복 제거
        if new_dfs:
            combined_df = pd.concat([cache_df, *new_dfs]).drop_duplicates(subset=["timestamp"])
            combined_df.sort_values("timestamp", inplace=True)
            
            # 실제 데이터의 시작점 기록 (상장일 추정)
            actual_start = combined_df["datetime"].min()
            ea_dt_meta = (
                pd.to_datetime(earliest_available, utc=True)
                if earliest_available else None
            )
            if not earliest_available or (ea_dt_meta and actual_start < ea_dt_meta):
                self._save_meta({meta_key: {"earliest_available": str(actual_start)}})
            
            # 캐시 저장
            self._save_cache(symbol, timeframe, combined_df)
            self._save_meta({meta_key: {"last_checked": str(pd.Timestamp.now(tz="UTC"))}})
            cache_df = combined_df

        # 6. 요청한 범위만 필터링하여 반환
        mask = (cache_df["datetime"] >= req_start) & (cache_df["datetime"] <= req_end)
        return cache_df.loc[mask].copy()

    def _fetch_ranges(
        self, symbol: str, timeframe: str, ranges: list[tuple[pd.Timestamp, pd.Timestamp]]
    ) -> list[pd.DataFrame]:
        results = []
        for start, end in ranges:
            df = self.client.fetch_ohlcv(symbol, timeframe, str(start), str(end))
            if not df.empty:
                results.append(self._normalize_df(df))
        return results

    def collect_1m_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load or fetch 1m OHLCV (Binance USDT-M futures); cache `{safe_symbol}_1m.parquet`.
        누락된 구간만 증분 수집하여 병목 현상 해결.
        """
        timeframe = "1m"
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)

        # 1. 로컬 캐시 로드
        cache_df = self._load_cache(symbol, timeframe)

        # 2. 메타데이터 로드
        meta_key = self._meta_key(symbol, timeframe)
        meta = self._load_meta().get(meta_key, {})
        earliest_available = meta.get("earliest_available")
        if earliest_available:
            ea_dt = pd.to_datetime(earliest_available, utc=True)
            if req_start < ea_dt:
                req_start = ea_dt

        # 3. 필요한 구간 판단
        fetch_tasks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if cache_df.empty:
            fetch_tasks.append((req_start, req_end))
            self.logger.info("1m cache miss for %s — fetching (%s ~ %s)",
                             symbol, start_date, end_date)
        else:
            c_start = cache_df["datetime"].min()
            c_end = cache_df["datetime"].max()
            
            # 3a. 과거/미래 확장
            if req_start < c_start:
                fetch_tasks.append((req_start, c_start))
            if req_end > c_end:
                fetch_tasks.append((c_end, req_end))
            
            # 3b. 중간 구멍 탐지 (Self-Healing)
            middle_gaps = self._identify_middle_gaps(cache_df, timeframe, req_start, req_end)
            fetch_tasks.extend(middle_gaps)
            
            if fetch_tasks:
                for f_s, f_e in fetch_tasks:
                    self.logger.warning(
                        "1m data task for %s: missing [%s, %s]; will fetch via API.",
                        symbol, f_s, f_e
                    )
            else:
                self.logger.debug("1m cache hit for %s (%s ~ %s)", symbol, start_date, end_date)

        new_dfs: list[pd.DataFrame] = []
        if fetch_tasks:
            # 세마포어를 사용하여 최대 2개 심볼만 동시에 API 호출 수행
            with self._collect_1m_semaphore:
                for idx, (f_start, f_end) in enumerate(fetch_tasks):
                    total_days = (f_end - f_start).days + 1
                    self.logger.info(
                        " [%s] Fetching 1m gap %d/%d: %s ~ %s (approx. %d days)",
                        symbol, idx + 1, len(fetch_tasks), f_start.date(), f_end.date(), total_days
                    )
                    chunk = self.client.fetch_ohlcv_with_taker(
                        symbol, timeframe, str(f_start), str(f_end)
                    )
                    if not chunk.empty:
                        chunk = self._normalize_df(chunk)
                        new_dfs.append(chunk)
                        self.logger.info(" [%s] OK. Fetched %d rows.", symbol, len(chunk))

        # 4. 데이터 병합 및 중복 제거
        if new_dfs:
            combined_df = pd.concat([cache_df, *new_dfs]).drop_duplicates(subset=["timestamp"])
            combined_df.sort_values("timestamp", inplace=True)
            
            actual_start = combined_df["datetime"].min()
            ea_dt_meta_1m = (
                pd.to_datetime(earliest_available, utc=True)
                if earliest_available else None
            )
            if not earliest_available or (ea_dt_meta_1m and actual_start < ea_dt_meta_1m):
                self._save_meta({meta_key: {"earliest_available": str(actual_start)}})
            
            self._save_cache(symbol, timeframe, combined_df)
            self._save_meta({meta_key: {"last_checked": str(pd.Timestamp.now(tz="UTC"))}})
            cache_df = combined_df

        mask = (cache_df["datetime"] >= req_start) & (cache_df["datetime"] <= req_end)
        return cache_df.loc[mask].copy()

    def _normalize_metrics_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """metrics 데이터의 timestamp를 integer ms로 통일"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # 1. datetime 컬럼 확보
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        elif "timestamp" in df.columns:
            s_ts = pd.to_numeric(df["timestamp"], errors="coerce")
            if s_ts.isna().all():
                df["datetime"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            else:
                val = s_ts.dropna().iloc[0]
                if val < 1e11: # seconds
                    df["datetime"] = pd.to_datetime(s_ts, unit="s", utc=True)
                elif val < 1e14: # ms
                    df["datetime"] = pd.to_datetime(s_ts, unit="ms", utc=True)
                else: # ns
                    df["datetime"] = pd.to_datetime(s_ts, unit="ns", utc=True)
        elif "create_time" in df.columns:
            df["datetime"] = pd.to_datetime(df["create_time"], utc=True, errors="coerce")
            
        # 2. datetime -> ms (가장 확실한 방법)
        if "datetime" in df.columns:
            df = df.dropna(subset=["datetime"])
            # x.timestamp() returns seconds (float), * 1000 -> ms
            df["timestamp"] = (df["datetime"].apply(lambda x: x.timestamp()) * 1000).astype("int64")
            
        return df

    def ensure_metrics_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """미결제약정(OI) 및 롱숏비율(LSR) 데이터를 Vision + API 조합으로 수집하여 parquet 저장"""
        safe_symbol = self._safe_symbol(symbol)
        path = FUTURES_DATA_DIR / f"{safe_symbol}_metrics.parquet"
        
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)
        
        # 1. 로컬 캐시 로드
        cache_df = pd.DataFrame()
        if path.exists():
            try:
                cache_df = self._normalize_metrics_df(pd.read_parquet(path))
            except Exception as e:
                self.logger.warning(f"Failed to read metrics cache {path}: {e}")
            
        # 2. 메타데이터에서 실제 상장일 확인 (OHLCV 수집 시 기록됨)
        meta_key = self._meta_key(symbol, "1h")
        meta = self._load_meta().get(meta_key, {})
        earliest_available = meta.get("earliest_available")

        effective_start = req_start
        if earliest_available:
            ea_dt = pd.to_datetime(earliest_available, utc=True)
            if req_start < ea_dt:
                effective_start = ea_dt
                self.logger.info(
                    f"Adjusting {symbol} metrics start to earliest available: {ea_dt.date()}"
                )

        # 3. 데이터 부족 여부 확인 (30일 분기점 계산)
        api_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=28) # 안전마진 28일

        vision_needed = False
        api_needed = False

        if cache_df.empty:
            vision_needed = effective_start < api_cutoff
            api_needed = req_end >= api_cutoff
        else:
            c_min = cache_df["datetime"].min()
            c_max = cache_df["datetime"].max()

            # Ensure c_min/c_max are tz-aware for comparison
            if c_min.tzinfo is None:
                c_min = c_min.tz_localize("UTC")
            if c_max.tzinfo is None:
                c_max = c_max.tz_localize("UTC")

            # [Optimization] 만약 캐시의 시작점이 상장일(ea_dt)과 거의 같다면 
            # (예: 1시간 이내 오차) 더 이상의 Vision 다운로드는 불필요함.
            # HYPEUSDT처럼 실제 첫 데이터가 ea_dt(10:00)보다 약간 늦은(10:35) 경우 방지.
            if effective_start < (c_min - pd.Timedelta(hours=1)) and effective_start < api_cutoff:
                vision_needed = True

            if req_end > c_max:
                api_needed = True

        if not vision_needed and not api_needed:
            self.logger.info(
                f" [CACHE] {symbol} metrics already covered " f"({c_min.date()} ~ {c_max.date()})"
            )
            mask = (cache_df["datetime"] >= effective_start) & (cache_df["datetime"] <= req_end)
            return cache_df.loc[mask].copy()

        new_parts = []

        # 4. Vision 수집 (30일 이전)
        if vision_needed:
            v_start = effective_start
            if not cache_df.empty:
                v_end = min(c_min, api_cutoff)
            else:
                v_end = min(req_end, api_cutoff)

            if v_start < v_end:
                downloader = BinanceVisionDownloader()
                v_df = downloader.fetch_range_metrics(symbol.replace("/", ""), v_start, v_end)
                if not v_df.empty:
                    new_parts.append(self._normalize_metrics_df(v_df))

        # 4. API 수집 (최근 30일)
        if api_needed:
            a_start = max(req_start, api_cutoff)
            if not cache_df.empty:
                a_start = max(a_start, cache_df["datetime"].max())
            
            if a_start < req_end:
                since_ms = int(a_start.timestamp() * 1000)
                # [REFACTORED] Fetch 1h metrics for better resolution
                oi_df = self.client.fetch_open_interest_history(symbol, "1h", since_ms)
                lsr_df = self.client.fetch_long_short_ratio_history(symbol, "1h", since_ms)
                
                if not oi_df.empty and not lsr_df.empty:
                    merged_api = pd.merge(oi_df, lsr_df, on="timestamp", how="outer")
                    new_parts.append(self._normalize_metrics_df(merged_api))

        # 5. 병합 및 저장
        if new_parts:
            combined = pd.concat([cache_df, *new_parts]).drop_duplicates(subset=["timestamp"])
            combined["timestamp"] = combined["timestamp"].astype("int64")
            combined.sort_values("timestamp", inplace=True)
            
            # [Institutional Optimization] Select only core alpha metrics & downcast to float32
            # These columns are standard for GP Miner / HMM regime filtering
            core_cols = [
                "timestamp", "datetime",
                "sum_open_interest", "sum_open_interest_value",
                "long_short_ratio", "top_trader_long_short_ratio",
                "taker_buy_sell_vol_value"
            ]
            existing_cols = [c for c in core_cols if c in combined.columns]
            combined = combined[existing_cols].copy()
            
            # Downcast floating point numbers to float32 (reduces size by ~50% with negligible loss)
            float_cols = combined.select_dtypes(include=["float64"]).columns
            combined[float_cols] = combined[float_cols].astype("float32")
            
            combined.to_parquet(path, index=False)
            self.logger.info(
                f"Updated metrics parquet cache (optimized): {path} | Size reduced via float32"
            )
            cache_df = combined

        mask = (cache_df["datetime"] >= req_start) & (cache_df["datetime"] <= req_end)
        return cache_df.loc[mask].copy()

    def ensure_funding_data(self, symbol: str, start_date: str, end_date: str) -> None:
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
