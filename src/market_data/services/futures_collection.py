from __future__ import annotations

import concurrent.futures
import contextlib
import logging
from pathlib import Path

import pandas as pd

from src.common.config import funding_path, ohlcv_path
from src.market_data.binance.futures import BinanceClient, BinanceKlinePermanentError
from src.market_data.binance.vision import BinanceVisionDownloader
from src.market_data.storage.ohlcv import write_ohlcv

_logger = logging.getLogger("DataCollector")

_METRICS_CANONICAL_COLUMNS: tuple[str, ...] = (
    "timestamp", "datetime", "available_at", "symbol",
    "sum_open_interest", "sum_open_interest_value",
    "long_short_ratio", "top_trader_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)

_METRICS_NUMERIC_COLUMNS: tuple[str, ...] = (
    "sum_open_interest", "sum_open_interest_value",
    "long_short_ratio", "top_trader_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)

_METRICS_RELEASE_LAG = pd.Timedelta(minutes=5)
_METRICS_MERGE_TOLERANCE = pd.Timedelta(hours=6)

def _empty_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_METRICS_CANONICAL_COLUMNS))


def _normalize_funding_frame(frame: pd.DataFrame) -> pd.DataFrame:
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


class DataValidator:
    @staticmethod
    def validate(df: pd.DataFrame, symbol: str, timeframe: str) -> list[str]:
        issues = []
        if df.isnull().any().any():
            missing_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"Missing values in columns: {missing_cols}")
        df_idx = df.set_index("datetime", drop=False)
        df_idx.sort_index(inplace=True)
        expected_diff = {"1m": pd.Timedelta(minutes=1), "1h": pd.Timedelta(hours=1),
                         "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}.get(timeframe)
        if expected_diff:
            time_diff = df_idx.index.to_series().diff().dropna()
            gaps = time_diff[time_diff != expected_diff]
            if not gaps.empty:
                issues.append(f"Found {len(gaps)} time gaps. First gap at {gaps.index[0]}")
        if not df.empty and (df["high"] < df["low"]).any():
            issues.append("High < Low detected in some rows")
        return issues


class DataCollector:
    def __init__(self, api_key: str | None = None, secret: str | None = None) -> None:
        self.client = BinanceClient(api_key, secret)
        self.logger = _logger

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        return symbol.replace("/", "_")

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        return ohlcv_path(symbol, timeframe)

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
        non_dt = [c for c in df.columns if c != "datetime"]
        if non_dt and all(pd.api.types.is_numeric_dtype(df[c]) for c in non_dt):
            return df
        for col in df.columns:
            if col == "datetime":
                continue
            if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0:
                    df[col] = converted
        return df

    def _load_cache(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self._cache_path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
            if df.empty or ("timestamp" not in df.columns and "datetime" not in df.columns):
                path.unlink()
                return pd.DataFrame()
            _baggage = [c for c in ("close_time", "no_trades", "ignore") if c in df.columns]
            if _baggage:
                df = df.drop(columns=_baggage)
            return self._normalize_df(df)
        except Exception as exc:
            with contextlib.suppress(Exception):
                path.unlink()
            self.logger.debug("Failed to load cache %s: %s", path, exc)
            return pd.DataFrame()

    def _save_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        write_ohlcv(self._cache_path(symbol, timeframe), df, timeframe=timeframe)

    def ensure_ohlcv_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> None:
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)
        cache_df = self._load_cache(symbol, timeframe)
        if (
            not cache_df.empty
            and cache_df["datetime"].min() <= req_start
            and cache_df["datetime"].max() >= req_end - pd.Timedelta(hours=8)
        ):
            return
        api_cutoff = pd.Timestamp.now(tz="UTC").replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ) - pd.Timedelta(days=32)
        new_parts: list[pd.DataFrame] = []
        vision_symbol = symbol.replace("/", "")
        current_month_start = req_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        vision_tasks: list[tuple[int, int]] = []
        while current_month_start < min(req_end, api_cutoff):
            month_end = (current_month_start + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
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
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_vol", "no_trades", "taker_buy_base",
                        "taker_buy_quote", "ignore",
                    ][: v_df.shape[1]]
                    for col in ["open", "high", "low", "close", "volume", "quote_vol",
                                "taker_buy_base", "taker_buy_quote"]:
                        if col in v_df.columns:
                            v_df[col] = pd.to_numeric(v_df[col], errors="coerce")
                    return self._normalize_df(v_df)
                return pd.DataFrame()

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_task = {executor.submit(_fetch_month, y, m): (y, m) for y, m in vision_tasks}
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        res_df = future.result()
                        if not res_df.empty:
                            new_parts.append(res_df)
                    except Exception as e:
                        self.logger.warning("Error fetching vision data for %s: %s", symbol, e)
        latest_cached_dt = cache_df["datetime"].max() if not cache_df.empty else None
        for part in new_parts:
            if part.empty or "datetime" not in part.columns:
                continue
            part_max_dt = pd.to_datetime(part["datetime"], utc=True).max()
            if pd.isna(part_max_dt):
                continue
            if latest_cached_dt is None or part_max_dt > latest_cached_dt:
                latest_cached_dt = part_max_dt
        remaining_start = max(req_start, latest_cached_dt) if latest_cached_dt is not None else req_start
        if remaining_start < req_end:
            try:
                chunk = self.client.fetch_ohlcv_with_taker(symbol, timeframe, str(remaining_start), str(req_end))
                if not chunk.empty:
                    new_parts.append(self._normalize_df(chunk))
            except BinanceKlinePermanentError as exc:
                self.logger.warning(
                    "Permanent OHLCV API failure for %s %s (%d). range=%s..%s",
                    symbol, timeframe, exc.http_code, remaining_start, req_end,
                )
        if new_parts:
            if not cache_df.empty and "timestamp" in cache_df.columns:
                cache_df["timestamp"] = pd.to_numeric(cache_df["timestamp"], errors="coerce")
            combined = (
                pd.concat([cache_df, *new_parts])
                .drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
            )
            self._save_cache(symbol, timeframe, combined)

    def ensure_funding_data(self, symbol: str, start_date: str, end_date: str) -> None:
        path = funding_path(symbol)
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)
        cache_df = pd.DataFrame()
        if path.exists():
            try:
                cache_df = _normalize_funding_frame(pd.read_parquet(path))
            except Exception as e:
                self.logger.warning(
                    "funding cache read failed; fallback to rebuild symbol=%s error=%s",
                    symbol, type(e).__name__,
                )
                cache_df = pd.DataFrame(columns=["timestamp", "funding_rate", "datetime"])
        if (
            not cache_df.empty
            and cache_df["datetime"].min() <= req_start + pd.Timedelta(days=1)
            and cache_df["datetime"].max() >= req_end - pd.Timedelta(hours=12)
        ):
            return
        api_cutoff = pd.Timestamp.now(tz="UTC").replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ) - pd.Timedelta(days=32)
        new_parts: list[pd.DataFrame] = []
        vision_symbol = symbol.replace("/", "")
        current_month_start = req_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        vision_tasks: list[tuple[int, int]] = []
        while current_month_start < min(req_end, api_cutoff):
            month_end = (current_month_start + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
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
                future_to_task = {executor.submit(_fetch_month_funding, y, m): (y, m) for y, m in vision_tasks}
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        res_df = future.result()
                        if not res_df.empty:
                            new_parts.append(res_df)
                    except Exception as e:
                        self.logger.warning("Error fetching vision funding data for %s: %s", symbol, e)
        latest_cached_dt = cache_df["datetime"].max() if not cache_df.empty else None
        for part in new_parts:
            if part.empty or "datetime" not in part.columns:
                continue
            part_max_dt = part["datetime"].max()
            if pd.isna(part_max_dt):
                continue
            if latest_cached_dt is None or part_max_dt > latest_cached_dt:
                latest_cached_dt = part_max_dt
        remaining_start = max(req_start, latest_cached_dt) if latest_cached_dt is not None else req_start
        if remaining_start < req_end:
            new_funding = self.client.fetch_funding_rate_history(symbol, str(remaining_start), str(req_end))
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
            _normalize_funding_frame(combined).to_parquet(path, index=False, compression="zstd")
