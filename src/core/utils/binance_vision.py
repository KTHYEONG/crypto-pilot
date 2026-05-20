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
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import cast
from xml.etree import ElementTree

import pandas as pd


class BinanceVisionDownloader:
    """Binance Vision(data.binance.vision)에서 과거 통계 데이터를 수집하는 유틸리티."""

    BASE_URL = "https://data.binance.vision/data/futures/um"
    S3_LISTING_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    DEFAULT_TIMEOUT_SECONDS = 20
    DEFAULT_MAX_CONCURRENCY = 2
    DEFAULT_MAX_WEIGHT_PER_MIN = 600
    DEFAULT_BACKOFF_BASE_SECONDS = 1.0
    DEFAULT_BACKOFF_MAX_SECONDS = 30.0
    DEFAULT_MAX_RETRIES = 4

    def __init__(self) -> None:
        """Binance Vision 다운로더 초기화."""
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
        """전역 최소 요청 간격을 보장한다."""
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
        """특정 날짜의 metrics ZIP 파일을 다운로드하여 DataFrame으로 반환합니다."""
        date_str = date.strftime("%Y-%m-%d")
        safe_symbol = urllib.parse.quote(symbol)
        url = self._vision_path_url(
            "daily",
            "metrics",
            safe_symbol,
            f"{safe_symbol}-metrics-{date_str}.zip",
        )

        try:
            self.logger.info("Downloading Vision metrics: %s @ %s", symbol, date_str)
            df = self._fetch_zip_csv(url)

            # 컬럼명 정규화
            # Binance Vision metrics columns:
            # create_time, symbol, sum_open_interest, sum_open_interest_value,
            # count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
            # count_long_short_ratio, sum_taker_long_short_vol_ratio

            if "create_time" in df.columns:
                df["datetime"] = pd.to_datetime(df["create_time"], utc=True)
                df["timestamp"] = df["datetime"].astype("int64") // 10**6

            rename_map = {
                "sum_toptrader_long_short_ratio": "top_trader_long_short_ratio",
                "count_long_short_ratio": "long_short_ratio",
            }
            df.rename(columns=rename_map, inplace=True)

            # Numeric conversion for key columns
            numeric_cols = [
                "sum_open_interest", "top_trader_long_short_ratio", "long_short_ratio",
                "sum_open_interest_value", "sum_taker_long_short_vol_ratio"
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # 404 is expected for dates before listing or missing data
                self.logger.debug(f"Vision data not found for {symbol} on {date_str} (404)")
            else:
                msg = f"HTTP Error fetching Vision data for {symbol} on {date_str}: {e}"
                self.logger.warning(msg)
            return pd.DataFrame()
        except Exception as e:
            # [Fix] 에러 로그 출력 시 인코딩 안전성 확보
            try:
                sym_log = symbol.encode("ascii", "ignore").decode("ascii") or "Unknown"
            except Exception:
                sym_log = "EncodingError"
            msg = f"Unexpected error fetching Vision data for {sym_log} on {date_str}: {e}"
            self.logger.warning(msg)
            return pd.DataFrame()

    def fetch_range_metrics(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """지정된 기간 전체의 metrics 수집 및 병합합니다."""
        all_dfs = []
        current = start_date
        while current <= end_date:
            df = self.fetch_daily_metrics(symbol, current)
            if not df.empty:
                all_dfs.append(df)
            current += timedelta(days=1)

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
        """월간 klines archive ZIP을 내려받아 DataFrame으로 반환합니다."""
        month_str = f"{month:02d}"
        filename = f"{symbol}-{interval}-{year}-{month_str}.zip"
        return self._fetch_zip_by_path("monthly", "klines", symbol, interval, filename)

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
        """월간 fundingRate archive ZIP을 내려받아 DataFrame으로 반환합니다."""
        month_str = f"{month:02d}"
        filename = f"{symbol}-fundingRate-{year}-{month_str}.zip"
        return self._fetch_zip_by_path("monthly", "fundingRate", symbol, filename)

    def fetch_funding_monthly(self, symbol: str, year: int, month: int) -> pd.DataFrame:
        """Alias for docs name: fetch monthly funding archive."""
        return self.fetch_funding_rate_monthly(symbol=symbol, year=year, month=month)

    def fetch_bookdepth_daily(self, symbol: str, date: datetime, level: str = "5") -> pd.DataFrame:
        """일간 bookDepth archive ZIP을 내려받아 DataFrame으로 반환합니다."""
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-bookDepth-{level}-{date_str}.zip"
        return self._fetch_zip_by_path("daily", "bookDepth", symbol, filename)

    def fetch_premiumindex_daily(self, symbol: str, date: datetime) -> pd.DataFrame:
        """일간 premiumIndexKlines archive ZIP을 내려받아 DataFrame으로 반환합니다."""
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-premiumIndexKlines-5m-{date_str}.zip"
        return self._fetch_zip_by_path("daily", "premiumIndexKlines", symbol, filename)

    def list_symbols_from_s3_xml_listing(
        self,
        *,
        dataset_prefix: str = "data/futures/um/daily/klines/",
        timeout: int | None = None,
    ) -> list[str]:
        """S3 XML listing을 파싱하여 데이터셋 디렉토리 내 심볼 목록을 반환합니다."""
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
                remain = prefix[len(dataset_prefix):].strip("/")
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
        """다운로드 payload의 checksum(hex)이 예상값과 일치하는지 검증합니다."""
        algo = algorithm.lower()
        if algo not in {"sha256", "md5"}:
            raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
        digest = hashlib.new(algo, payload).hexdigest()
        expected = expected_hex_digest.strip().lower().split()[0]
        return digest == expected

    def fetch_metrics_daily(self, symbol: str, date: datetime) -> pd.DataFrame:
        """일간 metrics archive ZIP을 내려받아 DataFrame으로 반환합니다.

        Args:
            symbol: 선물 심볼 (e.g. 'BTCUSDT').
            date: 수집 날짜 (datetime 오브젝트).

        Returns:
            columns에 sum_open_interest, count_toptrader_long_short_ratio 등을 포함한 DataFrame.
            데이터가 없으면 빈 DataFrame.

        Notes:
            Binance Vision daily/metrics 경로: SYMBOL-metrics-YYYY-MM-DD.zip
        """
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-metrics-{date_str}.zip"
        return self._fetch_zip_by_path("daily", "metrics", symbol, filename)


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
    """일간 metrics를 날짜 범위로 일괄 수집한다.

    2020-09-01 이전 구간은 Binance Vision에 데이터가 없으므로 빈 DataFrame을 반환한다.

    Args:
        symbol: 선물 심볼 (e.g. 'BTCUSDT').
        start_date: 수집 시작일 (date 또는 'YYYY-MM-DD' str).
        end_date: 수집 종료일.
        cache_dir: 로컬 캐시 디렉토리 (None이면 캐싱 안 함).

    Returns:
        columns=[open_time, sum_open_interest, ...] 형태의 DataFrame.
        시작일이 2020-09-01 이전이면 빈 DataFrame.

    Time Complexity: O(n_days)
    Space Complexity: O(n_days * n_cols)
    """
    import pathlib
    from datetime import date as _date, datetime as _datetime, timedelta as _timedelta

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
            downloader.logger.warning(
                "fetch_metrics_daily failed symbol=%s date=%s: %s", symbol, date_str, _e
            )

        curr += _timedelta(days=1)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    return combined

