import concurrent.futures
import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import (
    FUTURES_DATA_DIR,
    bookdepth_path,
    funding_path,
    indicator_kline_path,
    metrics_path,
    ohlcv_path,
)
from src.market_data.binance.futures import BinanceClient, BinanceKlinePermanentError
from src.market_data.binance.vision import BinanceVisionDownloader, fetch_metrics_bulk
from src.market_data.storage.ohlcv import write_ohlcv
from src.market_data.storage.schemas import METRICS_CANONICAL_COLUMNS as _METRICS_CANONICAL_COLUMNS

_logger = logging.getLogger("DataCollector")


def _mark_price_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "").replace("_", "")
    return FUTURES_DATA_DIR / "markPriceKlines" / timeframe / f"{safe}.parquet"


def _mark_price_manifest_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "").replace("_", "")
    return FUTURES_DATA_DIR / "markPriceKlines" / timeframe / f"{safe}.coverage.json"

_METRICS_NUMERIC_COLUMNS: tuple[str, ...] = (
    "sum_open_interest", "sum_open_interest_value",
    "long_short_ratio", "top_trader_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)

_METRICS_RELEASE_LAG = pd.Timedelta(minutes=5)
_METRICS_MERGE_TOLERANCE = pd.Timedelta(hours=6)

_INDICATOR_KLINE_CANONICAL_COLUMNS: tuple[str, ...] = (
    "timestamp", "datetime", "open", "high", "low", "close", "close_time",
)

_BOOKDEPTH_CANONICAL_COLUMNS: tuple[str, ...] = (
    "timestamp", "datetime", "symbol", "percentage", "depth", "notional",
)

def _empty_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_METRICS_CANONICAL_COLUMNS))


def _empty_bookdepth_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_BOOKDEPTH_CANONICAL_COLUMNS))


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


@dataclass(frozen=True, slots=True)
class MarkPriceCoverage:
    """Source-controlled mark-kline coverage manifest for one requested window.

    All timestamps are tz-aware UTC. ``primary_usable`` exactly means no missing
    interval overlaps the requested interval and every mark is finite and
    strictly positive; it is not a statement about historical execution quality
    or margin/liquidation completeness.
    """

    symbol: str
    timeframe: str
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    observed_start: pd.Timestamp | None
    observed_end: pd.Timestamp | None
    missing_intervals: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    endpoint: str
    collected_at: pd.Timestamp
    primary_usable: bool


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

    @staticmethod
    def _load_mark_price_cache(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "datetime"])
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            _logger.warning("mark price cache read failed path=%s error=%s", path, exc)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "datetime"])
        if df.empty or "timestamp" not in df.columns:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "datetime"])
        if "datetime" not in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return (
            df.dropna(subset=["datetime"])
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
        )

    def _mark_price_coverage(
        self,
        symbol: str,
        timeframe: str,
        req_start: pd.Timestamp,
        req_end: pd.Timestamp,
        frame: pd.DataFrame,
    ) -> MarkPriceCoverage:
        """Compute the coverage manifest; empty/gapped/non-positive marks fail closed."""
        period_map = {
            "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h",
            "12h": "12h", "1d": "1D", "3d": "3D", "1w": "1W", "1M": "1ME",
        }
        period = period_map.get(timeframe, timeframe)
        grid = pd.date_range(req_start, req_end, freq=period, tz="UTC")
        observed = (
            set(pd.DatetimeIndex(frame["datetime"]))
            if not frame.empty
            else set()
        )
        present = grid.isin(observed)
        missing = grid[~present]
        intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if len(missing):
            start = missing[0]
            prev = start
            step = missing[1] - missing[0] if len(missing) > 1 else pd.Timedelta(0)
            for t in missing[1:]:
                if step != pd.Timedelta(0) and t != prev + step:
                    intervals.append((start, prev))
                    start = t
                prev = t
            intervals.append((start, prev))
        all_finite = bool(frame["close"].notna().all()) if not frame.empty else False
        all_positive = bool((frame["close"] > 0).all()) if not frame.empty else False
        primary_usable = (not intervals) and all_finite and all_positive
        return MarkPriceCoverage(
            symbol=symbol,
            timeframe=timeframe,
            requested_start=req_start,
            requested_end=req_end,
            observed_start=frame["datetime"].min() if not frame.empty else None,
            observed_end=frame["datetime"].max() if not frame.empty else None,
            missing_intervals=tuple(intervals),
            endpoint="GET /fapi/v1/markPriceKlines",
            collected_at=pd.Timestamp.now(tz="UTC"),
            primary_usable=primary_usable,
        )

    def _save_mark_price_coverage(
        self, symbol: str, timeframe: str, coverage: MarkPriceCoverage,
    ) -> None:
        manifest = _mark_price_manifest_path(symbol, timeframe)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol": coverage.symbol,
            "timeframe": coverage.timeframe,
            "requested_start": coverage.requested_start.isoformat(),
            "requested_end": coverage.requested_end.isoformat(),
            "observed_start": (
                coverage.observed_start.isoformat() if coverage.observed_start is not None else None
            ),
            "observed_end": (
                coverage.observed_end.isoformat() if coverage.observed_end is not None else None
            ),
            "missing_intervals": [
                [a.isoformat(), b.isoformat()] for a, b in coverage.missing_intervals
            ],
            "endpoint": coverage.endpoint,
            "collected_at": coverage.collected_at.isoformat(),
            "primary_usable": coverage.primary_usable,
        }
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def ensure_mark_price_data(
        self, symbol: str, timeframe: str, start_date: str, end_date: str,
    ) -> MarkPriceCoverage:
        """Collect historical mark-price klines and persist a coverage manifest.

        Canonical mark candles are stored separately at
        ``data/futures/markPriceKlines/<timeframe>/<SYMBOL>.parquet``; an OHLCV
        close is never coerced into a mark price while a mark interval is
        requested. ``primary_usable`` is true only when every requested primary
        interval is observed and valid.
        """
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)
        path = _mark_price_path(symbol, timeframe)
        cache_df = self._load_mark_price_cache(path)
        observed = cache_df
        coverage = self._mark_price_coverage(symbol, timeframe, req_start, req_end, observed)
        intervals = coverage.missing_intervals
        if observed.empty and not intervals:
            intervals = ((req_start, req_end),)
        new_parts: list[pd.DataFrame] = []
        vision_parts: list[pd.DataFrame] = []
        for interval_start, interval_end in intervals:
            fetched = self.client.fetch_mark_price_klines(
                symbol, timeframe, str(interval_start), str(interval_end),
            )
            if fetched.empty:
                continue
            fetched = fetched.copy()
            fetched["datetime"] = pd.to_datetime(
                fetched["timestamp"], unit="ms", utc=True,
            )
            new_parts.append(fetched)
        # REST mark-price history is retention-limited.  Vision monthly files
        # are the primary historical source, with daily files repairing partial
        # or missing monthly days (without ever substituting OHLCV).
        if intervals:
            vision = BinanceVisionDownloader()
            def _normalize_vision_frame(frame: pd.DataFrame) -> pd.DataFrame:
                if frame is None or frame.empty:
                    return pd.DataFrame()
                renamed = frame.copy()
                if "timestamp" not in renamed.columns:
                    renamed.columns = [
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "count", "taker_buy_volume",
                        "taker_buy_quote_volume", "ignore",
                    ][: len(renamed.columns)]
                return self._normalize_indicator_kline_frame(renamed)

            month_keys: set[tuple[int, int]] = set()
            for gap_start, gap_end in intervals:
                month_keys.update((p.year, p.month) for p in pd.period_range(gap_start, gap_end, freq="M"))
            sorted_month_keys = sorted(month_keys)
            covered_months: set[tuple[int, int]] = set()
            for year, month in sorted_month_keys:
                try:
                    fetched = vision.fetch_indicator_klines_monthly(
                        "markPriceKlines", symbol.replace("/", ""), timeframe,
                        year, month,
                    )
                    normalized = _normalize_vision_frame(fetched)
                    if not normalized.empty:
                        vision_parts.append(normalized)
                        covered_months.add((year, month))
                except Exception as exc:
                    self.logger.warning("Vision monthly mark fetch failed for %s %s-%02d: %s", symbol, year, month, exc)
            if vision_parts:
                observed = pd.concat([observed, *vision_parts], ignore_index=True).drop_duplicates(
                    subset=["timestamp"], keep="last"
                ).sort_values("timestamp")
            remaining = self._mark_price_coverage(symbol, timeframe, req_start, req_end, observed).missing_intervals
            for gap_start, gap_end in remaining:
                day = gap_start.normalize()
                if day > gap_end.normalize():
                    continue
                while day <= gap_end.normalize():
                    if (day.year, day.month) not in covered_months:
                        day += pd.Timedelta(days=1)
                        continue
                    try:
                        fetched = vision.fetch_indicator_klines_daily(
                            "markPriceKlines", symbol.replace("/", ""), timeframe, day.to_pydatetime()
                        )
                        normalized = _normalize_vision_frame(fetched)
                        if not normalized.empty:
                            vision_parts.append(normalized)
                    except Exception as exc:
                        self.logger.warning("Vision daily mark fetch failed for %s %s: %s", symbol, day, exc)
                    day += pd.Timedelta(days=1)
            if vision_parts:
                observed = pd.concat([observed, *vision_parts], ignore_index=True).drop_duplicates(
                    subset=["timestamp"], keep="last"
                ).sort_values("timestamp")
        if new_parts or vision_parts:
            observed = (
                pd.concat([cache_df, *new_parts, *vision_parts], ignore_index=True)
                .drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            observed[["timestamp", "open", "high", "low", "close", "datetime"]].to_parquet(
                path, index=False, compression="zstd",
            )
        coverage = self._mark_price_coverage(symbol, timeframe, req_start, req_end, observed)
        self._save_mark_price_coverage(symbol, timeframe, coverage)
        return coverage

    def load_mark_price_panel(
        self, symbols: list[str], timeframe: str, grid: pd.DatetimeIndex,
        max_stale_hours: int = 0,
    ) -> pd.DataFrame:
        """Build a read-only causal mark-price panel for replay valuation.

        Reads only the existing ``markPriceKlines/<timeframe>/<symbol>.parquet``
        cache through ``_load_mark_price_cache``; it never calls a network API,
        collects missing data, or substitutes an OHLCV close. For the Phase-1
        ``1h`` cache each hourly close is observable only from ``open + 1h``
        through the following 59 minutes (``ffill limit=59``) and is never
        propagated across a missing hourly candle. The returned frame has
        exactly ``grid`` as its index and exactly ``symbols`` as its column
        order, preserving ``NaN`` for absent/non-finite/non-positive marks.
        """
        if timeframe != "1h":
            raise ValueError(f"unsupported timeframe '{timeframe}'")
        if max_stale_hours < 0:
            raise ValueError("max_stale_hours must be non-negative")
        if not isinstance(grid, pd.DatetimeIndex) or grid.empty:
            raise DataIntegrityError("grid must be a non-empty DatetimeIndex")
        if grid.tz is None:
            raise DataIntegrityError("grid must be tz-aware UTC")
        if not grid.is_monotonic_increasing or grid.has_duplicates:
            raise DataIntegrityError("grid must be monotonically increasing with no duplicates")
        if not symbols:
            raise DataIntegrityError("symbols must be non-empty")
        if len(set(symbols)) != len(symbols):
            raise DataIntegrityError("symbols must be unique")

        panel = pd.DataFrame(index=grid, columns=list(symbols), dtype="float64")
        if len(grid) > 1:
            step = grid[1] - grid[0]
            step_minutes = step / pd.Timedelta(minutes=1)
            if step_minutes <= 0 or 60 % step_minutes != 0:
                raise DataIntegrityError(
                    "grid frequency must be a positive divisor of one hour"
                )
            if max_stale_hours == 0:
                ffill_limit = int(60 // step_minutes - 1)
            else:
                ffill_limit = int(max_stale_hours * 60 // step_minutes - 1)
        else:
            ffill_limit = 0
        for sym in symbols:
            cache = self._load_mark_price_cache(_mark_price_path(sym, timeframe))
            if cache.empty or "close" not in cache.columns:
                continue
            valid = (
                cache["datetime"].notna()
                & cache["close"].notna()
                & (cache["close"] > 0)
            )
            closes = (
                cache.loc[valid, ["datetime", "close"]]
                .drop_duplicates(subset=["datetime"], keep="last")
                .sort_values("datetime")
            )
            if closes.empty:
                continue
            available = pd.Series(
                closes["close"].to_numpy(dtype="float64"),
                index=closes["datetime"] + pd.Timedelta(hours=1),
            )
            aligned = (
                available.reindex(grid)
                if ffill_limit == 0
                else available.reindex(grid, method="ffill", limit=ffill_limit)
            )
            panel[sym] = aligned.to_numpy(dtype="float64")
        return panel

    def _load_metrics_cache(self, symbol: str) -> pd.DataFrame:
        path = metrics_path(symbol)
        if not path.exists():
            return _empty_metrics_frame()
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            self.logger.debug("Failed to load metrics cache %s: %s", path, exc)
            return _empty_metrics_frame()
        if frame.empty or not set(_METRICS_CANONICAL_COLUMNS).issubset(frame.columns):
            return _empty_metrics_frame()
        frame = frame.loc[:, list(_METRICS_CANONICAL_COLUMNS)]
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
        return frame.dropna(subset=["datetime", "available_at"]).sort_values("timestamp")

    def _save_metrics_cache(self, symbol: str, frame: pd.DataFrame) -> None:
        path = metrics_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[:, list(_METRICS_CANONICAL_COLUMNS)].to_parquet(
            path, index=False, compression="zstd",
        )

    @staticmethod
    def _validate_metrics_frame(frame: pd.DataFrame, symbol: str) -> None:
        missing = set(_METRICS_CANONICAL_COLUMNS) - set(frame.columns)
        if missing:
            raise DataIntegrityError(
                f"metrics frame for {symbol} missing canonical columns: {sorted(missing)}"
            )
        if not frame["timestamp"].is_monotonic_increasing:
            raise DataIntegrityError(f"metrics timestamps for {symbol} are not monotonic")
        if frame["timestamp"].duplicated().any():
            raise DataIntegrityError(f"metrics timestamps for {symbol} contain duplicates")
        if frame["datetime"].dt.tz is None:
            raise DataIntegrityError(f"metrics datetimes for {symbol} must be tz-aware UTC")
        lag = (frame["available_at"] - frame["datetime"]).abs()
        if lag.gt(pd.Timedelta(minutes=5)).any():
            raise DataIntegrityError(
                f"metrics available_at for {symbol} must equal datetime + 5 minutes"
            )

    def _metrics_coverage_report(
        self,
        symbol: str,
        req_start: pd.Timestamp,
        req_end: pd.Timestamp,
        frame: pd.DataFrame,
    ) -> dict[str, list[str]]:
        """Compute requested dates absent from the collected metrics frame."""
        if frame.empty:
            covered: set[pd.Timestamp] = set()
        else:
            covered = set(pd.DatetimeIndex(frame["datetime"]).date)
        dates = pd.date_range(req_start.normalize(), req_end.normalize(), freq="1D")
        missing = [
            d.date().isoformat() for d in dates if d.date() not in covered
        ]
        report = {"missing_dates": missing}
        for day in missing:
            self.logger.warning(
                "metrics unavailable symbol=%s date=%s (reported, no forward-fill)",
                symbol, day,
            )
        return report

    @staticmethod
    def _merge_metrics_frames(
        cache: pd.DataFrame,
        incoming: pd.DataFrame,
        *,
        incoming_is_authoritative: bool,
    ) -> pd.DataFrame:
        if cache.empty and incoming.empty:
            return _empty_metrics_frame()
        order = [cache, incoming] if incoming_is_authoritative else [incoming, cache]
        parts = [f for f in order if not f.empty]
        if not parts:
            return _empty_metrics_frame()
        return (
            pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def ensure_metrics_live_tail(self, symbol: str, *, lookback_days: int = 7) -> None:
        oi_raw = self.client.fetch_futures_data_metric("openInterestHist", symbol, period="5m", limit=500)
        gl_raw = self.client.fetch_futures_data_metric(
            "globalLongShortAccountRatio", symbol, period="5m", limit=500
        )
        top_raw = self.client.fetch_futures_data_metric(
            "topLongShortPositionRatio", symbol, period="5m", limit=500
        )
        taker_raw = self.client.fetch_futures_data_metric("takerlongshortRatio", symbol, period="5m", limit=500)

        def _to_frame(raw: list[dict[str, Any]], endpoint: str) -> pd.DataFrame:
            if not raw:
                return pd.DataFrame(columns=["timestamp"])
            df = pd.DataFrame(raw)
            ts_col: str | None = None
            for cand in ("timestamp", "time", "create_time", "T"):
                if cand in df.columns:
                    ts_col = cand
                    break
            if ts_col is None:
                ts_col = str(df.columns[0])
            df["timestamp"] = pd.to_numeric(df[ts_col], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            if df.empty:
                return pd.DataFrame(columns=["timestamp"])
            df["timestamp"] = df["timestamp"].astype("int64")
            if endpoint == "openInterestHist":
                if "sumOpenInterest" in df.columns:
                    df["sum_open_interest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
                elif "sum_open_interest" in df.columns:
                    df["sum_open_interest"] = pd.to_numeric(df["sum_open_interest"], errors="coerce")
                else:
                    df["sum_open_interest"] = pd.NA
                if "sumOpenInterestValue" in df.columns:
                    df["sum_open_interest_value"] = pd.to_numeric(
                        df["sumOpenInterestValue"], errors="coerce"
                    )
                elif "sum_open_interest_value" in df.columns:
                    df["sum_open_interest_value"] = pd.to_numeric(
                        df["sum_open_interest_value"], errors="coerce"
                    )
                else:
                    df["sum_open_interest_value"] = pd.NA
                return df[["timestamp", "sum_open_interest", "sum_open_interest_value"]]
            if endpoint == "globalLongShortAccountRatio":
                if "longShortRatio" in df.columns:
                    df["long_short_ratio"] = pd.to_numeric(df["longShortRatio"], errors="coerce")
                elif "long_short_ratio" in df.columns:
                    df["long_short_ratio"] = pd.to_numeric(df["long_short_ratio"], errors="coerce")
                else:
                    df["long_short_ratio"] = pd.NA
                return df[["timestamp", "long_short_ratio"]]
            if endpoint == "topLongShortPositionRatio":
                if "longShortRatio" in df.columns:
                    df["top_trader_long_short_ratio"] = pd.to_numeric(
                        df["longShortRatio"], errors="coerce"
                    )
                elif "top_trader_long_short_ratio" in df.columns:
                    df["top_trader_long_short_ratio"] = pd.to_numeric(
                        df["top_trader_long_short_ratio"], errors="coerce"
                    )
                else:
                    df["top_trader_long_short_ratio"] = pd.NA
                return df[["timestamp", "top_trader_long_short_ratio"]]
            if endpoint == "takerlongshortRatio":
                if "buySellRatio" in df.columns:
                    df["sum_taker_long_short_vol_ratio"] = pd.to_numeric(
                        df["buySellRatio"], errors="coerce"
                    )
                elif "sum_taker_long_short_vol_ratio" in df.columns:
                    df["sum_taker_long_short_vol_ratio"] = pd.to_numeric(
                        df["sum_taker_long_short_vol_ratio"], errors="coerce"
                    )
                else:
                    df["sum_taker_long_short_vol_ratio"] = pd.NA
                return df[["timestamp", "sum_taker_long_short_vol_ratio"]]
            return pd.DataFrame(columns=["timestamp"])

        oi_df = _to_frame(oi_raw, "openInterestHist")
        gl_df = _to_frame(gl_raw, "globalLongShortAccountRatio")
        top_df = _to_frame(top_raw, "topLongShortPositionRatio")
        taker_df = _to_frame(taker_raw, "takerlongshortRatio")

        # outer join on timestamp
        tail: pd.DataFrame | None = None
        for part in (oi_df, gl_df, top_df, taker_df):
            tail = part if tail is None else pd.merge(tail, part, on="timestamp", how="outer")
        if tail is None or tail.empty:
            return
        tail = tail.sort_values("timestamp").reset_index(drop=True)
        # keep last lookback_days relative to max timestamp in tail
        if not tail.empty and lookback_days > 0:
            max_ts = int(tail["timestamp"].max())
            cutoff_ms = max_ts - int(lookback_days * 24 * 3600 * 1000)
            tail = tail[tail["timestamp"] >= cutoff_ms].reset_index(drop=True)
            if tail.empty:
                return
        tail["datetime"] = pd.to_datetime(tail["timestamp"], unit="ms", utc=True)
        tail["available_at"] = tail["datetime"] + pd.Timedelta(minutes=5)
        tail["symbol"] = symbol
        # ensure all canonical columns present
        for col in _METRICS_CANONICAL_COLUMNS:
            if col not in tail.columns:
                tail[col] = pd.NA
        tail = tail.loc[:, list(_METRICS_CANONICAL_COLUMNS)]
        # numeric coercion
        for col in _METRICS_NUMERIC_COLUMNS:
            tail[col] = pd.to_numeric(tail[col], errors="coerce")
        tail = tail.sort_values("timestamp").reset_index(drop=True)

        cache = self._load_metrics_cache(symbol)
        combined = self._merge_metrics_frames(cache, tail, incoming_is_authoritative=False)
        self._validate_metrics_frame(combined, symbol)
        # 라이브 tail은 방금 받은 5m 구간의 내부 연속성만 검증한다. 과거 일간 Vision
        # 캐시의 아카이브 발행 지연(D+1 이상)은 ensure_metrics_data 소관이며 게이트 대상이 아니다.
        if not tail.empty and not combined.empty:
            tail_min = tail["datetime"].min()
            tail_max = tail["datetime"].max()
            coverage = self._metrics_coverage_report(symbol, tail_min, tail_max, combined)
            interior_before_tail = [
                day
                for day in coverage["missing_dates"]
                if tail_min < pd.Timestamp(day, tz="UTC") < tail_max
            ]
            if interior_before_tail:
                raise DataIntegrityError(
                    f"requested coverage gap for {symbol}: missing interior dates "
                    f"{interior_before_tail}; never forward-filled"
                )
        self._save_metrics_cache(symbol, combined)

    def ensure_metrics_data(self, symbol: str, start_date: str, end_date: str) -> None:
        """Collect and persist canonical daily Vision metrics for one symbol.

        Fetches Vision daily archives via ``fetch_metrics_bulk``, normalizes
        them through ``_normalize_metrics_frame``, merges with the canonical
        cache (deduplicated by timestamp, keep=last, ascending), validates the
        canonical schema and monotonic timestamps, and persists. Missing archive
        dates are surfaced explicitly in the coverage report and are never
        forward-filled; an interior coverage gap raises ``DataIntegrityError``.
        """
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)
        cache_df = self._load_metrics_cache(symbol)
        if (
            not cache_df.empty
            and cache_df["datetime"].min() <= req_start.normalize()
            and cache_df["datetime"].max() >= req_end.normalize()
        ):
            return

        vision = BinanceVisionDownloader()
        raw = fetch_metrics_bulk(symbol, start_date, end_date)
        fetched = vision._normalize_metrics_frame(symbol, raw)  # noqa: SLF001
        if cache_df.empty and fetched.empty:
            return
        combined = self._merge_metrics_frames(cache_df, fetched, incoming_is_authoritative=True)
        self._validate_metrics_frame(combined, symbol)
        coverage = self._metrics_coverage_report(symbol, req_start, req_end, combined)
        self._save_metrics_cache(symbol, combined)
        interior_missing = [
            day for day in coverage["missing_dates"]
            if not combined.empty
            and pd.Timestamp(day, tz="UTC") > combined["datetime"].min()
            and pd.Timestamp(day, tz="UTC") < combined["datetime"].max()
        ]
        if interior_missing:
            raise DataIntegrityError(
                f"requested coverage gap for {symbol}: missing interior dates "
                f"{interior_missing}; never forward-filled"
            )

    def _load_indicator_kline_cache(self, dataset: str, symbol: str, timeframe: str) -> pd.DataFrame:
        path = indicator_kline_path(dataset, symbol, timeframe)
        if not path.exists():
            return pd.DataFrame()
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            self.logger.debug("Failed to load indicator kline cache %s: %s", path, exc)
            return pd.DataFrame()
        if frame.empty or not set(_INDICATOR_KLINE_CANONICAL_COLUMNS).issubset(frame.columns):
            return pd.DataFrame()
        frame = frame.loc[:, list(_INDICATOR_KLINE_CANONICAL_COLUMNS)]
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")
        return frame.dropna(subset=["datetime"]).sort_values("timestamp")

    def _save_indicator_kline_cache(
        self, dataset: str, symbol: str, timeframe: str, frame: pd.DataFrame,
    ) -> None:
        path = indicator_kline_path(dataset, symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[:, list(_INDICATOR_KLINE_CANONICAL_COLUMNS)].to_parquet(
            path, index=False, compression="zstd",
        )

    def _normalize_indicator_kline_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize a raw Vision indicator-kline frame to the canonical columns."""
        if frame is None or frame.empty or "timestamp" not in frame.columns:
            return pd.DataFrame(columns=list(_INDICATOR_KLINE_CANONICAL_COLUMNS))
        df = frame.copy()
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            return pd.DataFrame(columns=list(_INDICATOR_KLINE_CANONICAL_COLUMNS))
        df["timestamp"] = df["timestamp"].astype("int64")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"])
        if df.empty:
            return pd.DataFrame(columns=list(_INDICATOR_KLINE_CANONICAL_COLUMNS))
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # close_time is left as raw object dtype otherwise: Vision monthly
        # archives inconsistently include a header row across months, which
        # poisons pandas' per-column dtype inference to str for header-having
        # months but leaves headerless months as native int64 -- concatenating
        # both across months produces a mixed str/int object column that
        # pyarrow's parquet writer rejects (ArrowTypeError: expected bytes,
        # got int). Force it numeric like the other integer/float fields.
        df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["close_time"])
        if df.empty:
            return pd.DataFrame(columns=list(_INDICATOR_KLINE_CANONICAL_COLUMNS))
        return (
            df.loc[:, list(_INDICATOR_KLINE_CANONICAL_COLUMNS)]
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def ensure_indicator_kline_data(self, dataset: str, symbol: str, timeframe: str, start_date: str, end_date: str) -> None:
        """Collect and persist canonical monthly mark/index/premium klines.

        Mirrors ``ensure_ohlcv_data``'s monthly-archive-then-cache pattern: only
        months whose archive is missing from the cache are fetched via
        ``BinanceVisionDownloader.fetch_indicator_klines_monthly`` (whose own
        allowed-set check fails closed on unsupported datasets), merged
        (deduplicated by timestamp, keep=last), and persisted as only the
        meaningful canonical columns. The always-zero synthetic volume/count/
        taker-buy fields of the raw Vision schema are dropped, never persisted.
        """
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)
        cache_df = self._load_indicator_kline_cache(dataset, symbol, timeframe)
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
                v_df = vision.fetch_indicator_klines_monthly(
                    dataset, vision_symbol, timeframe, year, month
                )
                if v_df.empty:
                    return pd.DataFrame()
                v_df.columns = [
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "count", "taker_buy_volume",
                    "taker_buy_quote_volume", "ignore",
                ][: v_df.shape[1]]
                return self._normalize_indicator_kline_frame(v_df)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_task = {
                    executor.submit(_fetch_month, y, m): (y, m) for y, m in vision_tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        res_df = future.result()
                        if not res_df.empty:
                            new_parts.append(res_df)
                    except ValueError:
                        raise
                    except Exception as e:
                        self.logger.warning(
                            "Error fetching indicator kline data for %s: %s", symbol, e
                        )
        if new_parts:
            combined = (
                pd.concat([cache_df, *new_parts], ignore_index=True)
                .drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            self._save_indicator_kline_cache(dataset, symbol, timeframe, combined)

    def _load_bookdepth_cache(self, symbol: str) -> pd.DataFrame:
        path = bookdepth_path(symbol)
        if not path.exists():
            return _empty_bookdepth_frame()
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            self.logger.debug("Failed to load bookdepth cache %s: %s", path, exc)
            return _empty_bookdepth_frame()
        if frame.empty or not set(_BOOKDEPTH_CANONICAL_COLUMNS).issubset(frame.columns):
            return _empty_bookdepth_frame()
        frame = frame.loc[:, list(_BOOKDEPTH_CANONICAL_COLUMNS)]
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        return frame.dropna(subset=["datetime"]).sort_values("timestamp")

    def _save_bookdepth_cache(self, symbol: str, frame: pd.DataFrame) -> None:
        path = bookdepth_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[:, list(_BOOKDEPTH_CANONICAL_COLUMNS)].to_parquet(
            path, index=False, compression="zstd",
        )

    def _normalize_bookdepth_frame(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize a raw Vision book-depth frame to the canonical schema."""
        if frame is None or frame.empty:
            return _empty_bookdepth_frame()
        df = frame.copy()
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        else:
            df["symbol"] = df["symbol"].fillna(symbol).astype(str)
        if "timestamp" not in df.columns:
            return _empty_bookdepth_frame()
        # bookDepth's raw timestamp is a human-readable "YYYY-MM-DD HH:MM:SS"
        # string (verified against a live download), not an epoch-ms integer
        # like klines/metrics -- branch on dtype the same way
        # _normalize_metrics_frame already does for the analogous Vision
        # format inconsistency, instead of assuming ms-epoch uniformly.
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["datetime"] = pd.to_datetime(
                pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True, errors="coerce",
            )
        else:
            df["datetime"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"])
        if df.empty:
            return _empty_bookdepth_frame()
        _dt_naive = df["datetime"].dt.tz_localize(None)
        df["timestamp"] = _dt_naive.astype("datetime64[ns]").astype("int64") // 10**6
        for col in ("percentage", "depth", "notional"):
            if col not in df.columns:
                df[col] = float("nan")
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.loc[:, list(_BOOKDEPTH_CANONICAL_COLUMNS)]

    @staticmethod
    def _validate_bookdepth_frame(frame: pd.DataFrame, symbol: str) -> None:
        missing = set(_BOOKDEPTH_CANONICAL_COLUMNS) - set(frame.columns)
        if missing:
            raise DataIntegrityError(
                f"bookdepth frame for {symbol} missing canonical columns: {sorted(missing)}"
            )
        if frame["datetime"].dt.tz is None:
            raise DataIntegrityError(f"bookdepth datetimes for {symbol} must be tz-aware UTC")
        for _band, group in frame.groupby(["symbol", "percentage"], dropna=False):
            if not group["timestamp"].is_monotonic_increasing:
                raise DataIntegrityError(
                    f"bookdepth timestamps for {symbol} are not monotonic within band"
                )
        if frame.duplicated(subset=["timestamp", "percentage"]).any():
            raise DataIntegrityError(
                f"bookdepth frame for {symbol} contains duplicate (timestamp, percentage) pairs"
            )

    def _bookdepth_coverage_report(
        self,
        symbol: str,
        req_start: pd.Timestamp,
        req_end: pd.Timestamp,
        frame: pd.DataFrame,
    ) -> dict[str, list[str]]:
        """Compute requested dates absent from the collected book-depth frame."""
        if frame.empty:
            covered: set[pd.Timestamp] = set()
        else:
            covered = set(pd.DatetimeIndex(frame["datetime"]).date)
        dates = pd.date_range(req_start.normalize(), req_end.normalize(), freq="1D")
        missing = [d.date().isoformat() for d in dates if d.date() not in covered]
        report = {"missing_dates": missing}
        for day in missing:
            self.logger.warning(
                "bookdepth unavailable symbol=%s date=%s (reported, no forward-fill)",
                symbol, day,
            )
        return report

    def ensure_bookdepth_data(self, symbol: str, start_date: str, end_date: str) -> None:
        """Collect and persist canonical daily Vision book depth for one symbol.

        Mirrors ``ensure_metrics_data``'s daily-archive/cache/coverage-report
        pattern: only days whose archive is missing from the cache are fetched
        via ``fetch_bookdepth_daily``, merged, validated (monotonic per-band
        timestamps, tz-aware UTC datetimes, no duplicate (timestamp, percentage)
        pairs), and persisted. Missing archive dates are surfaced in the coverage
        report and are never forward-filled; an interior coverage gap raises
        ``DataIntegrityError``.
        """
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)
        cache_df = self._load_bookdepth_cache(symbol)
        if (
            not cache_df.empty
            and cache_df["datetime"].min() <= req_start.normalize()
            and cache_df["datetime"].max() >= req_end.normalize()
        ):
            return

        vision = BinanceVisionDownloader()
        parts: list[pd.DataFrame] = []
        if not cache_df.empty:
            parts.append(cache_df)
        covered_dates = (
            set(pd.DatetimeIndex(cache_df["datetime"]).date) if not cache_df.empty else set()
        )
        curr = req_start.normalize()
        while curr <= req_end.normalize():
            if curr.date() not in covered_dates:
                raw = vision.fetch_bookdepth_daily(symbol, curr.to_pydatetime())
                if not raw.empty:
                    raw = raw.copy()
                    raw.columns = ["timestamp", "percentage", "depth", "notional"][: raw.shape[1]]
                    normalized = self._normalize_bookdepth_frame(raw, symbol)
                    if not normalized.empty:
                        parts.append(normalized)
            curr += pd.Timedelta(days=1)
        if not parts:
            return
        combined = pd.concat(parts, ignore_index=True)
        self._validate_bookdepth_frame(combined, symbol)
        combined = (
            combined.sort_values(["timestamp", "percentage"])
            .drop_duplicates(subset=["timestamp", "percentage"], keep="last")
            .reset_index(drop=True)
        )
        coverage = self._bookdepth_coverage_report(symbol, req_start, req_end, combined)
        self._save_bookdepth_cache(symbol, combined)
        interior_missing = [
            day for day in coverage["missing_dates"]
            if not combined.empty
            and pd.Timestamp(day, tz="UTC") > combined["datetime"].min()
            and pd.Timestamp(day, tz="UTC") < combined["datetime"].max()
        ]
        if interior_missing:
            raise DataIntegrityError(
                f"requested coverage gap for {symbol}: missing interior dates "
                f"{interior_missing}; never forward-filled"
            )

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
