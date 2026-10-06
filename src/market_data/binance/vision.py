import io
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import cast
from xml.etree import ElementTree

import pandas as pd


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
                    # Some archives (e.g. fundingRate) carry a header row.
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

    def fetch_funding_rate_monthly(self, symbol: str, year: int, month: int) -> pd.DataFrame:
        """Downloads monthly fundingRate archive ZIP and returns it as DataFrame."""
        month_str = f"{month:02d}"
        filename = f"{symbol}-fundingRate-{year}-{month_str}.zip"
        return self._fetch_zip_by_path("monthly", "fundingRate", symbol, filename)

    def list_symbols_from_s3_xml_listing(
        self,
        *,
        dataset_prefix: str = "data/futures/um/daily/klines/",
        timeout: int | None = None,
    ) -> list[str]:
        """Parse every page of the S3 XML listing and return the symbols under ``dataset_prefix``.

        The bucket listing caps one response at 1000 common prefixes and the UM
        futures archive exceeds that, so pages are followed via ``marker`` until
        ``IsTruncated`` is false; an unpaginated read silently truncates the
        alphabetical tail of the universe. Any page failure returns ``[]`` (the
        historical contract) -- callers that must not mistake a failure for an
        empty universe fail closed on the empty result.
        """
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
        symbols: set[str] = set()
        marker: str | None = None
        try:
            while True:
                params: dict[str, str] = {"prefix": dataset_prefix, "delimiter": "/"}
                if marker is not None:
                    params["marker"] = marker
                query = urllib.parse.urlencode(params)
                url = f"{self.S3_LISTING_URL}?{query}"
                body = self._read_url_bytes(url, timeout=timeout)
                root = ElementTree.fromstring(body)  # noqa: S314
                prefixes = root.findall(f".//{ns}CommonPrefixes/{ns}Prefix")
                page_prefixes: list[str] = [(node.text or "").strip() for node in prefixes]
                for prefix in page_prefixes:
                    if not prefix.startswith(dataset_prefix):
                        continue
                    remain = prefix[len(dataset_prefix):].strip("/")
                    if remain:
                        symbols.add(remain.split("/")[0])
                truncated_node = root.find(f".//{ns}IsTruncated")
                is_truncated = (
                    truncated_node is not None
                    and (truncated_node.text or "").strip().lower() == "true"
                )
                if not is_truncated or not page_prefixes:
                    break
                next_marker_node = root.find(f".//{ns}NextMarker")
                next_marker = (
                    (next_marker_node.text or "").strip()
                    if next_marker_node is not None and next_marker_node.text
                    else ""
                )
                marker = next_marker if next_marker else page_prefixes[-1]
            return sorted(symbols)
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
