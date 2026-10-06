import concurrent.futures
import itertools
import logging
import statistics
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import funding_path, ohlcv_path
from src.market_data.binance.futures import BinanceClient, BinanceKlinePermanentError
from src.market_data.binance.vision import BinanceVisionDownloader
from src.market_data.storage.ohlcv import write_ohlcv

_logger = logging.getLogger("DataCollector")

_TIMEFRAME_MS: dict[str, int] = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000}

FUNDING_DEFAULT_INTERVAL_MS: int = 8 * 3_600_000  # 8시간 기본 간격
FUNDING_SETTLEMENT_GRACE_MS: int = 5 * 60_000  # 정산 직후 게시 지연 유예
FUNDING_TIME_TOLERANCE_MS: int = 60_000  # fundingTime의 ms 단위 지터 허용
FUNDING_GAP_THRESHOLD_MS: int = FUNDING_DEFAULT_INTERVAL_MS + 30 * 60_000  # 표준 최대 정산 간격(8h)을 넘는 간격만 공백으로 본다 -- 간격 변경(8h->4h->1h)을 공백으로 오탐해 매 사이클 재조회하는 것을 막는다.
_FUNDING_INTERVAL_SAMPLE: int = 6

# Previous full calendar month must be published on Vision before it is trusted; younger months come from REST.
_VISION_ARCHIVE_LAG: pd.Timedelta = pd.Timedelta(days=32)
# Cap concurrent monthly archive downloads per symbol, independent of the downloader's request throttle.
_VISION_FETCH_MAX_WORKERS: int = 4


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


def _vision_months_to_fetch(
    req_start: pd.Timestamp,
    req_end: pd.Timestamp,
    now: pd.Timestamp,
    cached_span: tuple[pd.Timestamp, pd.Timestamp] | None,
    interior_gap_start_ms: Callable[[int], int | None],
) -> list[tuple[int, int]]:
    """Plan which monthly Vision archives must be (re)downloaded for a request window.

    Vision only publishes complete calendar months, and recently closed months may
    not be archived yet, so planning stops at ``now``'s month start minus
    ``_VISION_ARCHIVE_LAG``; anything younger is left to the caller's REST tail.
    A month is re-planned whenever the cache does not provably cover it: the cache
    is empty, starts after the month start, ends before the month's last second,
    or its first interior gap at/after the month start opens before the month end.
    The gap rule exists because a min/max span check alone permanently misses a
    month that is entirely absent between two cached months.

    Args:
        req_start: Inclusive tz-aware UTC request start; floored to its month start.
        req_end: tz-aware UTC request end; months starting at/after it are not planned.
        now: tz-aware UTC wall clock used only to derive the archive cutoff. Callers
            pass the same clock reading they use for the rest of the refresh.
        cached_span: ``(min datetime, max datetime)`` of the existing cache, or
            ``None`` when the cache is empty.
        interior_gap_start_ms: Dataset-specific gap probe. Given a window start in
            epoch ms, returns the left edge (epoch ms) of the first interior gap whose
            right edge lies after that start, or ``None``. Must be side-effect free.

    Returns:
        ``(year, month)`` pairs in strictly ascending chronological order, without
        duplicates; empty when the planning range is empty.
    """
    cutoff = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - _VISION_ARCHIVE_LAG
    bound = min(req_end, cutoff)
    current = req_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months: list[tuple[int, int]] = []
    while current < bound:
        month_end = (current + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
        if cached_span is None:
            months.append((current.year, current.month))
        else:
            span_min, span_max = cached_span
            if span_min > current or span_max < month_end:
                months.append((current.year, current.month))
            else:
                month_start_ms = int(current.value // 1_000_000)
                month_end_ms = int(month_end.value // 1_000_000)
                gap = interior_gap_start_ms(month_start_ms)
                if gap is not None and gap < month_end_ms:
                    months.append((current.year, current.month))
        current += pd.offsets.MonthBegin(1)
    return months


def _fetch_months_parallel(
    months: Sequence[tuple[int, int]],
    fetch_month: Callable[[int, int], pd.DataFrame],
    on_error: Callable[[int, int, Exception], None],
) -> list[pd.DataFrame]:
    """Download planned monthly archives concurrently with per-month failure isolation.

    A single unavailable or corrupt monthly archive must not abort the whole
    backfill: the REST tail and the next refresh cycle re-plan any month still
    missing, so per-month failures are reported through ``on_error`` and the
    remaining months proceed. Concurrency is bounded by
    ``_VISION_FETCH_MAX_WORKERS`` to cap simultaneous archive downloads per symbol.

    Args:
        months: ``(year, month)`` pairs, typically from ``_vision_months_to_fetch``.
        fetch_month: Downloads and normalises one month; returns an empty frame when
            the archive has no rows. Called at most once per pair, from worker threads.
        on_error: Invoked on the calling thread once per month whose ``fetch_month``
            raised an ``Exception``, with ``(year, month, exc)``.

    Returns:
        Non-empty frames in the order of ``months`` (not completion order), so
        downstream concatenation is deterministic.
    """
    if not months:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=_VISION_FETCH_MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_month, year, month) for year, month in months]
        parts: list[pd.DataFrame] = []
        for (year, month), future in zip(months, futures, strict=True):
            try:
                frame = future.result()
            except Exception as exc:
                on_error(year, month, exc)
                continue
            if not frame.empty:
                parts.append(frame)
        return parts


def _timeframe_ms(timeframe: str) -> int:
    if timeframe not in _TIMEFRAME_MS:
        raise ValueError(f"unsupported timeframe for bar closure: {timeframe}")
    return _TIMEFRAME_MS[timeframe]


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
        vision_symbol = symbol.replace("/", "")
        vision_months = _vision_months_to_fetch(
            req_start,
            req_end,
            now,
            None if cache_df.empty else (cache_df["datetime"].min(), cache_df["datetime"].max()),
            lambda start_ms: ohlcv_gap_start_ms(cache_ts, start_ms, interval_ms),
        )
        new_parts: list[pd.DataFrame] = []
        if vision_months:
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

            new_parts = _fetch_months_parallel(
                vision_months,
                _fetch_month,
                lambda _y, _m, exc: self.logger.warning("Error fetching vision data for %s: %s", symbol, exc),
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
        vision_symbol = symbol.replace("/", "")
        vision_months = _vision_months_to_fetch(
            req_start,
            req_end,
            now,
            None if cache_df.empty else (cache_df["datetime"].min(), cache_df["datetime"].max()),
            lambda start_ms: funding_gap_start_ms(cache_df["timestamp"], start_ms),
        )
        new_parts: list[pd.DataFrame] = []
        if vision_months:
            vision = BinanceVisionDownloader()

            def _fetch_month_funding(year: int, month: int) -> pd.DataFrame:
                v_df = vision.fetch_funding_rate_monthly(vision_symbol, year, month)
                if not v_df.empty:
                    return _normalize_funding_frame(v_df)
                return pd.DataFrame()

            new_parts = _fetch_months_parallel(
                vision_months,
                _fetch_month_funding,
                lambda _y, _m, exc: self.logger.warning("Error fetching vision funding data for %s: %s", symbol, exc),
            )
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
