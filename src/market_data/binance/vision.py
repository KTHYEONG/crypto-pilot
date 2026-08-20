import hashlib
import io
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import cast
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from src.market_data.storage.schemas import METRICS_CANONICAL_COLUMNS as _METRICS_CANONICAL_COLUMNS


class BinanceVisionDownloader:
    """Utility for collecting historical statistical data from Binance Vision (data.binance.vision)."""

    BASE_URL = "https://data.binance.vision/data/futures/um"
    S3_LISTING_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    DEFAULT_TIMEOUT_SECONDS = 20
    # Vision archive requests are globally paced below the configured RPM ceiling.
    # Four in-flight requests hide archive latency without increasing the request rate.
    DEFAULT_MAX_CONCURRENCY = 4
    DEFAULT_MAX_WEIGHT_PER_MIN = 600
    DEFAULT_BACKOFF_BASE_SECONDS = 1.0
    DEFAULT_BACKOFF_MAX_SECONDS = 30.0
    DEFAULT_MAX_RETRIES = 4

    def __init__(self) -> None:
        """Initializes Binance Vision downloader."""
        self.logger = logging.getLogger("BinanceVision")
        self.max_concurrency = self._env_int(
            "BINANCE_VISION_MAX_CONCURRENCY",
            default=self.DEFAULT_MAX_CONCURRENCY,
            min_value=1,
        )
        self.max_weight_per_min = self._env_int(
            "BINANCE_VISION_MAX_WEIGHT_PER_MIN",
            default=self.DEFAULT_MAX_WEIGHT_PER_MIN,
            min_value=1,
        )
        self.backoff_base_seconds = self._env_float(
            "BINANCE_VISION_BACKOFF_BASE_SECONDS",
            default=self.DEFAULT_BACKOFF_BASE_SECONDS,
            min_value=0.1,
        )
        self.backoff_max_seconds = self._env_float(
            "BINANCE_VISION_BACKOFF_MAX_SECONDS",
            default=self.DEFAULT_BACKOFF_MAX_SECONDS,
            min_value=self.backoff_base_seconds,
        )
        self.max_retries = self._env_int(
            "BINANCE_VISION_MAX_RETRIES",
            default=self.DEFAULT_MAX_RETRIES,
            min_value=0,
        )
        default_interval_seconds = (60.0 / float(self.max_weight_per_min)) * 1.2
        self.min_request_interval_seconds = self._env_float(
            "BINANCE_VISION_MIN_REQUEST_INTERVAL_SECONDS",
            default=default_interval_seconds,
            min_value=0.01,
        )
        self._request_semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._request_lock = threading.Lock()
        self._next_request_monotonic = 0.0

    @staticmethod
    def _env_int(name: str, default: int, min_value: int) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return max(min_value, int(value))
        except ValueError:
            return default

    @staticmethod
    def _env_float(name: str, default: float, min_value: float) -> float:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return max(min_value, float(value))
        except ValueError:
            return default

    def _wait_for_turn(self) -> None:
        """Ensures global minimum request interval."""
        with self._request_lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_request_monotonic - now)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = time.monotonic()
            self._next_request_monotonic = now + self.min_request_interval_seconds

    @staticmethod
    def _is_retryable_http_error(error: urllib.error.HTTPError) -> bool:
        return error.code == 429 or 500 <= error.code <= 599

    def _parse_retry_after_seconds(self, raw: str | None) -> float | None:
        if not raw:
            return None
        try:
            return max(0.0, float(raw.strip()))
        except ValueError:
            pass
        try:
            retry_at = parsedate_to_datetime(raw.strip())
            now = datetime.now(tz=retry_at.tzinfo)
            return max(0.0, (retry_at - now).total_seconds())
        except Exception:
            return None

    def _compute_backoff_seconds(
        self,
        attempt: int,
        http_error: urllib.error.HTTPError | None = None,
    ) -> float:
        base: float = min(self.backoff_max_seconds, self.backoff_base_seconds * (2**attempt))
        retry_after_seconds: float | None = None
        if http_error is not None:
            retry_after_seconds = self._parse_retry_after_seconds(
                http_error.headers.get("Retry-After"),
            )
        if retry_after_seconds is not None:
            bounded = min(self.backoff_max_seconds, max(base, retry_after_seconds))
            return float(bounded)
        return base

    def _read_url_bytes(self, url: str, timeout: int | None = None) -> bytes:
        timeout_seconds = timeout or self.DEFAULT_TIMEOUT_SECONDS
        for attempt in range(self.max_retries + 1):
            try:
                with self._request_semaphore:
                    self._wait_for_turn()
                    with urllib.request.urlopen(  # noqa: S310
                        url,
                        timeout=timeout_seconds,
                    ) as response:
                        return cast(bytes, response.read())
            except urllib.error.HTTPError as http_error:
                if not self._is_retryable_http_error(http_error) or attempt >= self.max_retries:
                    raise
                backoff = self._compute_backoff_seconds(attempt, http_error=http_error)
                self.logger.warning(
                    "Retryable HTTP error %s for %s (attempt %s/%s), backoff %.2fs",
                    http_error.code,
                    url,
                    attempt + 1,
                    self.max_retries + 1,
                    backoff,
                )
                time.sleep(backoff)
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                if attempt >= self.max_retries:
                    raise
                backoff = self._compute_backoff_seconds(attempt)
                self.logger.warning(
                    "Network error fetching %s (attempt %s/%s): %s; backoff %.2fs",
                    url,
                    attempt + 1,
                    self.max_retries + 1,
                    err,
                    backoff,
                )
                time.sleep(backoff)
        raise RuntimeError("Unreachable retry loop for _read_url_bytes")

    def _fetch_zip_csv(self, url: str) -> pd.DataFrame:
        zip_data = self._read_url_bytes(url)
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
            if not csv_names:
                return pd.DataFrame()
            with zf.open(csv_names[0]) as handle:
                df = pd.read_csv(handle, header=None)
                if not df.empty:
                    # 일부 파일(metrics, fundingRate 등)에는 첫 줄에 헤더가 포함된 경우가 있음.
                    # 첫 줄의 첫 번째 컬럼이 문자열(예: 'calc_time', 'create_time', 'timestamp')인 경우 헤더로 간주하고 제거.
                    first_val = str(df.iloc[0, 0]).lower()
                    if first_val in ("calc_time", "create_time", "timestamp", "open_time"):
                        df = df.iloc[1:].reset_index(drop=True)
                return df

    def _vision_path_url(self, *parts: str) -> str:
        encoded = "/".join(urllib.parse.quote(p.strip("/")) for p in parts if p)
        return f"{self.BASE_URL}/{encoded}"

    def _fetch_zip_by_path(self, *parts: str) -> pd.DataFrame:
        url = self._vision_path_url(*parts)
        try:
            return self._fetch_zip_csv(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                self.logger.debug("Vision data not found (404): %s", url)
                return pd.DataFrame()
            self.logger.warning("HTTP error fetching Vision zip (%s): %s", url, e)
            return pd.DataFrame()
        except Exception as e:
            self.logger.warning("Unexpected error fetching Vision zip (%s): %s", url, e)
            return pd.DataFrame()

    def fetch_daily_metrics(self, symbol: str, date: datetime) -> pd.DataFrame:
        """Downloads metrics ZIP file for a specific date and returns it as DataFrame."""
        date_str = date.strftime("%Y-%m-%d")
        url = self._vision_path_url(
            "daily",
            "metrics",
            symbol,
            f"{symbol}-metrics-{date_str}.zip",
        )

        try:
            self.logger.info("Downloading Vision metrics: %s @ %s", symbol, date_str)
            df = self._fetch_zip_csv(url)
            if df.empty:
                return pd.DataFrame(columns=list(_METRICS_CANONICAL_COLUMNS))
            return self._normalize_metrics_frame(symbol=symbol, frame=df)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # 404 is expected for dates before listing or missing data
                self.logger.debug(f"Vision data not found for {symbol} on {date_str} (404)")
            else:
                msg = f"HTTP Error fetching Vision data for {symbol} on {date_str}: {e}"
                self.logger.warning(msg)
            return pd.DataFrame(columns=list(_METRICS_CANONICAL_COLUMNS))
        except Exception as e:
            # [Fix] 에러 로그 출력 시 인코딩 안전성 확보
            try:
                sym_log = symbol.encode("ascii", "ignore").decode("ascii") or "Unknown"
            except Exception:
                sym_log = "EncodingError"
            msg = f"Unexpected error fetching Vision data for {sym_log} on {date_str}: {e}"
            self.logger.warning(msg)
            return pd.DataFrame(columns=list(_METRICS_CANONICAL_COLUMNS))

    def fetch_range_metrics(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetches and merges metrics for the entire specified date range."""
        dates: list[datetime] = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        all_dfs: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {
                executor.submit(self.fetch_daily_metrics, symbol, day): day
                for day in dates
            }
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["timestamp"])

    def fetch_klines_archive_monthly(
        self,
        symbol: str,
        interval: str,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """Downloads monthly klines archive ZIP and returns it as DataFrame."""
        month_str = f"{month:02d}"
        filename = f"{symbol}-{interval}-{year}-{month_str}.zip"
        return self._fetch_zip_by_path("monthly", "klines", symbol, interval, filename)

    def fetch_indicator_klines_monthly(
        self,
        dataset: str,
        symbol: str,
        interval: str,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """Fetch a monthly mark/index/premium kline archive."""
        allowed = {"markPriceKlines", "indexPriceKlines", "premiumIndexKlines"}
        if dataset not in allowed:
            raise ValueError(f"unsupported indicator dataset: {dataset}")
        filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
        return self._fetch_zip_by_path("monthly", dataset, symbol, interval, filename)

    def fetch_indicator_klines_daily(
        self,
        dataset: str,
        symbol: str,
        interval: str,
        date: datetime,
    ) -> pd.DataFrame:
        """Fetch one daily indicator-kline archive.

        Daily archives are required because Binance Vision monthly archives can
        contain partial days or repaired historical gaps.
        """
        allowed = {"markPriceKlines", "indexPriceKlines", "premiumIndexKlines"}
        if dataset not in allowed:
            raise ValueError(f"unsupported indicator dataset: {dataset}")
        filename = f"{symbol}-{interval}-{date:%Y-%m-%d}.zip"
        return self._fetch_zip_by_path("daily", dataset, symbol, interval, filename)

    def fetch_klines_archive(
        self,
        symbol: str,
        interval: str,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """Alias for docs name: fetch monthly klines archive."""
        return self.fetch_klines_archive_monthly(
            symbol=symbol,
            interval=interval,
            year=year,
            month=month,
        )

    def fetch_funding_rate_monthly(self, symbol: str, year: int, month: int) -> pd.DataFrame:
        """Downloads monthly fundingRate archive ZIP and returns it as DataFrame."""
        month_str = f"{month:02d}"
        filename = f"{symbol}-fundingRate-{year}-{month_str}.zip"
        return self._fetch_zip_by_path("monthly", "fundingRate", symbol, filename)

    def fetch_funding_monthly(self, symbol: str, year: int, month: int) -> pd.DataFrame:
        """Alias for docs name: fetch monthly funding archive."""
        return self.fetch_funding_rate_monthly(symbol=symbol, year=year, month=month)

    def fetch_bookdepth_daily(self, symbol: str, date: datetime) -> pd.DataFrame:
        """Downloads daily bookDepth archive ZIP and returns it as DataFrame.

        Vision's actual archive filename carries no level component (verified
        against the live S3 listing: ``SYMBOL-bookDepth-YYYY-MM-DD.zip``); the
        5-level depth granularity is fixed by the dataset itself, not
        selectable per file.
        """
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-bookDepth-{date_str}.zip"
        return self._fetch_zip_by_path("daily", "bookDepth", symbol, filename)

    def fetch_premiumindex_daily(self, symbol: str, date: datetime) -> pd.DataFrame:
        """Downloads daily premiumIndexKlines archive ZIP and returns it as DataFrame."""
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-premiumIndexKlines-5m-{date_str}.zip"
        return self._fetch_zip_by_path("daily", "premiumIndexKlines", symbol, filename)

    def list_symbols_from_s3_xml_listing(
        self,
        *,
        dataset_prefix: str = "data/futures/um/daily/klines/",
        timeout: int | None = None,
    ) -> list[str]:
        """Parses S3 XML listing and returns symbol list within dataset directory."""
        query = urllib.parse.urlencode({"prefix": dataset_prefix, "delimiter": "/"})
        url = f"{self.S3_LISTING_URL}?{query}"
        try:
            body = self._read_url_bytes(url, timeout=timeout)
            root = ElementTree.fromstring(body)  # noqa: S314
            symbols: list[str] = []
            ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
            for node in root.findall(f".//{ns}CommonPrefixes/{ns}Prefix"):
                prefix = (node.text or "").strip()
                if not prefix.startswith(dataset_prefix):
                    continue
                remain = prefix[len(dataset_prefix) :].strip("/")
                if remain:
                    symbols.append(remain.split("/")[0])
            return sorted(set(symbols))
        except Exception as e:
            self.logger.warning("Failed to list symbols from Vision S3 XML listing: %s", e)
            return []

    def list_all_symbols(
        self,
        *,
        dataset_prefix: str = "data/futures/um/daily/klines/",
        timeout: int | None = None,
    ) -> list[str]:
        """Alias for docs name: list all symbols from Vision listing."""
        return self.list_symbols_from_s3_xml_listing(
            dataset_prefix=dataset_prefix,
            timeout=timeout,
        )

    def verify_checksum(
        self,
        payload: bytes,
        expected_hex_digest: str,
        algorithm: str = "sha256",
    ) -> bool:
        """Verifies if downloaded payload's checksum (hex) matches expected value."""
        algo = algorithm.lower()
        if algo not in {"sha256", "md5"}:
            raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
        digest = hashlib.new(algo, payload).hexdigest()
        expected = expected_hex_digest.strip().lower().split()[0]
        return digest == expected

    def fetch_metrics_daily(self, symbol: str, date: datetime) -> pd.DataFrame:
        """Downloads daily metrics archive ZIP and returns it as DataFrame.

        Args:
            symbol: Futures symbol (e.g. 'BTCUSDT').
            date: Target date (datetime object).

        Returns:
            DataFrame containing sum_open_interest, count_toptrader_long_short_ratio, etc.
            Empty DataFrame if data is unavailable.

        Notes:
            Binance Vision daily metrics path: SYMBOL-metrics-YYYY-MM-DD.zip

        """
        return self.fetch_daily_metrics(symbol, date)

    def _normalize_metrics_frame(self, symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Normalizes Vision metrics frame to canonical schema."""
        if frame is None or frame.empty:
            return pd.DataFrame(columns=list(_METRICS_CANONICAL_COLUMNS))

        df = frame.copy()
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if not any(isinstance(col, str) for col in df.columns):
            expected_cols = [
                "create_time",
                "symbol",
                "sum_open_interest",
                "sum_open_interest_value",
                "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio",
                "count_long_short_ratio",
                "sum_taker_long_short_vol_ratio",
            ]
            if len(df.columns) >= len(expected_cols):
                df = df.rename(
                    columns={
                        src_col: expected_cols[idx] for idx, src_col in enumerate(df.columns[: len(expected_cols)])
                    }
                )

        rename_map = {
            "create_time": "timestamp",
            "count_long_short_ratio": "long_short_ratio",
            "sum_toptrader_long_short_ratio": "top_trader_long_short_ratio",
        }
        df = df.rename(columns=rename_map)
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        else:
            df["symbol"] = df["symbol"].fillna(symbol).astype(str)

        if "timestamp" not in df.columns:
            return pd.DataFrame(columns=list(_METRICS_CANONICAL_COLUMNS))

        # dtype으로 분기: count 휴리스틱 금지 (함정 A — 정수 컬럼 to_datetime이 NaN 없이 garbage)
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            ts_numeric = pd.to_numeric(df["timestamp"], errors="coerce")
            df["datetime"] = pd.to_datetime(ts_numeric, unit="ms", utc=True, errors="coerce")
        else:
            # 실측 Binance Vision metrics 포맷: create_time = "YYYY-MM-DD HH:MM:SS" 문자열
            df["datetime"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        df = df.dropna(subset=["datetime"])
        if df.empty:
            return pd.DataFrame(columns=list(_METRICS_CANONICAL_COLUMNS))

        # int64 ms epoch 역산출 — 함정 B 대응: to_datetime 결과 해상도(ns/us/ms)가 입력에 따라
        # 달라지므로 하드코드 나눗셈 금지. tz 제거 후 datetime64[ns]로 해상도 명시 고정.
        _dt_naive = df["datetime"].dt.tz_localize(None)
        df["timestamp"] = _dt_naive.astype("datetime64[ns]").astype("int64") // 10**6

        df["available_at"] = df["datetime"] + pd.Timedelta(minutes=5)

        for col in (
            "sum_open_interest",
            "sum_open_interest_value",
            "long_short_ratio",
            "top_trader_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ):
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return (
            df.loc[:, list(_METRICS_CANONICAL_COLUMNS)]
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# 모듈-레벨 helper: fetch_metrics_bulk
# ---------------------------------------------------------------------------

_OI_ADV_METRICS_START = "2020-09-01"  # Binance Vision metrics 제공 시작일


def fetch_metrics_bulk(
    symbol: str,
    start_date: "str | datetime",
    end_date: "str | datetime",
    cache_dir: "str | None" = None,
) -> pd.DataFrame:
    """Fetches daily metrics in bulk for a date range.

    Returns empty DataFrame for dates before 2020-09-01 as Binance Vision lacks metrics prior to this.

    Args:
        symbol: Futures symbol (e.g. 'BTCUSDT').
        start_date: Start date string or datetime.
        end_date: End date string or datetime.
        cache_dir: Optional cache directory path.

    Returns:
        Concatenated daily metrics DataFrame.

    Time Complexity: O(n_days)
    Space Complexity: O(n_days * n_cols)

    """
    import pathlib
    from datetime import date as _date
    from datetime import datetime as _datetime
    from datetime import timedelta as _timedelta

    # 날짜 타입 정규화
    def _to_date(d: object) -> _date:
        if isinstance(d, _datetime):
            return d.date()
        if isinstance(d, _date):
            return d
        return _datetime.strptime(str(d), "%Y-%m-%d").date()

    _start = _to_date(start_date)
    _end = _to_date(end_date)
    _metrics_start = _datetime.strptime(_OI_ADV_METRICS_START, "%Y-%m-%d").date()

    # 2020-09-01 이전 구간: 데이터 없음 → 빈 DataFrame
    if _end < _metrics_start:
        return pd.DataFrame()

    # 유효 시작일 조정
    _eff_start = max(_start, _metrics_start)

    downloader = BinanceVisionDownloader()
    dfs: list[pd.DataFrame] = []
    curr = _eff_start
    safe_symbol = symbol.replace("/", "").replace("_", "")

    while curr <= _end:
        date_str = curr.strftime("%Y-%m-%d")
        cache_file = None
        if cache_dir is not None:
            cache_file = pathlib.Path(cache_dir) / safe_symbol / f"metrics-{date_str}.parquet"

        if cache_file is not None and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                dfs.append(df)
                curr += _timedelta(days=1)
                continue
            except Exception as _e:
                downloader.logger.warning("Cache read failed %s: %s", cache_file, _e)

        dt = _datetime.combine(curr, _datetime.min.time())
        try:
            df = downloader.fetch_metrics_daily(safe_symbol, dt)
            if not df.empty:
                if cache_dir is not None and cache_file is not None:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(cache_file, index=False)
                dfs.append(df)
        except Exception as _e:
            downloader.logger.warning("fetch_metrics_daily failed symbol=%s date=%s: %s", symbol, date_str, _e)

        curr += _timedelta(days=1)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    return combined
