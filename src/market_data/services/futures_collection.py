import concurrent.futures
import itertools
import logging
import statistics
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import (
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

_TIMEFRAME_MS: dict[str, int] = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000}

FUNDING_DEFAULT_INTERVAL_MS: int = 8 * 3_600_000  # 8시간 기본 간격
FUNDING_SETTLEMENT_GRACE_MS: int = 5 * 60_000  # 정산 직후 게시 지연 유예
FUNDING_TIME_TOLERANCE_MS: int = 60_000  # fundingTime의 ms 단위 지터 허용
FUNDING_GAP_THRESHOLD_MS: int = FUNDING_DEFAULT_INTERVAL_MS + 30 * 60_000  # 표준 최대 정산 간격(8h)을 넘는 간격만 공백으로 본다 -- 간격 변경(8h->4h->1h)을 공백으로 오탐해 매 사이클 재조회하는 것을 막는다.
_FUNDING_INTERVAL_SAMPLE: int = 6


def infer_funding_interval_ms(timestamps_ms: Iterable[int]) -> int:
    ts = sorted({int(t) for t in timestamps_ms})
    if len(ts) < 2:
        return FUNDING_DEFAULT_INTERVAL_MS
    tail = ts[-(_FUNDING_INTERVAL_SAMPLE + 1):]
    diffs = [b - a for a, b in itertools.pairwise(tail)]
    median = statistics.median_low(diffs)
    return max(1, round(median / 3_600_000)) * 3_600_000


def last_settled_funding_epoch_ms(now: pd.Timestamp, interval_ms: int) -> int:
    as_of_ms = int(now.value // 1_000_000) - FUNDING_SETTLEMENT_GRACE_MS
    return (as_of_ms // interval_ms) * interval_ms


def funding_tail_is_fresh(timestamps_ms: Iterable[int], now: pd.Timestamp) -> bool:
    ts = [int(t) for t in timestamps_ms]
    if not ts:
        return False
    return max(ts) >= last_settled_funding_epoch_ms(now, infer_funding_interval_ms(ts)) - FUNDING_TIME_TOLERANCE_MS


def funding_gap_start_ms(timestamps_ms: Iterable[int], window_start_ms: int) -> int | None:
    """Return the last present settlement before the first internal funding gap.

    Scans sorted unique timestamps for a gap wider than
    ``FUNDING_GAP_THRESHOLD_MS`` whose right edge lies inside the window
    (``b > window_start_ms``). Returns the left edge ``a`` (which may precede
    ``window_start_ms``; callers clamp it) or ``None`` when no in-window gap
    exists. A leading edge (cache starting after ``window_start_ms``) is not
    a gap.
    """
    ts = sorted({int(t) for t in timestamps_ms})
    for a, b in itertools.pairwise(ts):
        if b > window_start_ms and b - a > FUNDING_GAP_THRESHOLD_MS:
            return a
    return None


def ohlcv_gap_start_ms(sorted_timestamps_ms: np.ndarray, window_start_ms: int, interval_ms: int) -> int | None:
    """Return the left edge of the first internal OHLCV gap in the window.

    Mirrors funding_gap_start_ms semantics: the left edge is returned, and a
    leading edge (cache starting after the window start) is not a gap.
    """
    ts = np.asarray(sorted_timestamps_ms, dtype="int64")
    if ts.size < 2:
        return None
    hits = np.flatnonzero((ts[1:] > int(window_start_ms)) & (np.diff(ts) > int(interval_ms)))
    return int(ts[hits[0]]) if hits.size else None


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _timeframe_ms(timeframe: str) -> int:
    if timeframe not in _TIMEFRAME_MS:
        raise ValueError(f"unsupported timeframe for bar closure: {timeframe}")
    return _TIMEFRAME_MS[timeframe]


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


class DataCollector:
    def __init__(self, api_key: str | None = None, secret: str | None = None) -> None:
        self.client = BinanceClient(api_key, secret)
        self.logger = _logger

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
        except Exception as exc:
            raise DataIntegrityError(f"ohlcv cache unreadable: {path}") from exc
        if df.empty or ("timestamp" not in df.columns and "datetime" not in df.columns):
            raise DataIntegrityError(f"ohlcv cache invalid: {path}")
        _baggage = [c for c in ("close_time", "no_trades", "ignore") if c in df.columns]
        if _baggage:
            df = df.drop(columns=_baggage)
        return self._normalize_df(df)

    def _save_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        write_ohlcv(self._cache_path(symbol, timeframe), df, timeframe=timeframe)

    def ensure_ohlcv_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> None:
        req_start = pd.to_datetime(start_date, utc=True)
        req_end = pd.to_datetime(end_date, utc=True)
        interval_ms = _timeframe_ms(timeframe)
        cache_df = self._load_cache(symbol, timeframe)
        req_end_ms = int(req_end.value // 1_000_000)
        latest_closed_open_ms = (req_end_ms // interval_ms) * interval_ms - interval_ms
        # min/max 스팬만 보면 내부 공백(2022-02-26~28, 04-01~02 등)을 영구히 놓친다 — 펀딩과 같은 규칙.
        cache_ts = np.unique(pd.to_numeric(cache_df["timestamp"], errors="coerce").dropna().to_numpy(dtype="int64")) if not cache_df.empty else np.empty(0, dtype="int64")
        req_start_ms = int(req_start.value // 1_000_000)
        span_gap_ms = ohlcv_gap_start_ms(cache_ts, req_start_ms, interval_ms)
        if (
            not cache_df.empty
            and cache_df["datetime"].min() <= req_start
            and cache_df["datetime"].max() >= pd.Timestamp(latest_closed_open_ms, unit="ms", tz="UTC")
            and (span_gap_ms is None or span_gap_ms >= latest_closed_open_ms)
        ):
            return
        now = _utc_now()
        api_cutoff = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ) - pd.Timedelta(days=32)
        new_parts: list[pd.DataFrame] = []
        vision_symbol = symbol.replace("/", "")
        current_month_start = req_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        vision_tasks: list[tuple[int, int]] = []
        while current_month_start < min(req_end, api_cutoff):
            month_end = (current_month_start + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
            month_start_ms = int(current_month_start.value // 1_000_000)
            month_end_ms = int(month_end.value // 1_000_000)
            month_gap_ms = ohlcv_gap_start_ms(cache_ts, month_start_ms, interval_ms)
            if (
                cache_df.empty
                or cache_df["datetime"].min() > current_month_start
                or cache_df["datetime"].max() < month_end
                or (month_gap_ms is not None and month_gap_ms < month_end_ms)
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
                    chunk = self._normalize_df(chunk)
                    now_ms = int(now.value // 1_000_000)
                    chunk = chunk[pd.to_numeric(chunk["timestamp"], errors="coerce") + interval_ms <= now_ms]
                    if not chunk.empty:
                        new_parts.append(chunk)
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
        now = _utc_now()
        as_of = min(req_end, now)
        req_start_ms = int(req_start.value // 1_000_000)
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
            and funding_tail_is_fresh(cache_df["timestamp"], as_of)
            and funding_gap_start_ms(cache_df["timestamp"], req_start_ms) is None
        ):
            return
        api_cutoff = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ) - pd.Timedelta(days=32)
        new_parts: list[pd.DataFrame] = []
        vision_symbol = symbol.replace("/", "")
        current_month_start = req_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        vision_tasks: list[tuple[int, int]] = []
        while current_month_start < min(req_end, api_cutoff):
            month_end = (current_month_start + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
            month_start_ms = int(current_month_start.value // 1_000_000)
            month_end_ms = int(month_end.value // 1_000_000)
            # min/max 스팬만 보면 이 달 앞뒤로 캐시가 있다는 이유로 달 통째 내부공백을
            # 놓친다: 이 달과 겹치는 첫 내부공백이 있는지 별도로 확인한다.
            gap_ms = funding_gap_start_ms(cache_df["timestamp"], month_start_ms) if not cache_df.empty else None
            if (
                cache_df.empty
                or cache_df["datetime"].min() > current_month_start
                or cache_df["datetime"].max() < month_end
                or (gap_ms is not None and gap_ms < month_end_ms)
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
        gap_ms = funding_gap_start_ms(cache_df["timestamp"], req_start_ms) if not cache_df.empty else None
        if gap_ms is not None:
            # 창 안 내부 공백은 꼬리가 최신이어도 공백 직전부터 재조회해 영구 누락을 막는다.
            remaining_start = max(req_start, pd.Timestamp(gap_ms, unit="ms", tz="UTC"))
        else:
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
