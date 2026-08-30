from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import ccxt
import pandas as pd

_logger = logging.getLogger("BinanceClient")


@dataclass(slots=True, frozen=True)
class BinanceKlinePermanentError(RuntimeError):
    symbol: str
    timeframe: str
    http_code: int
    start_time_ms: int
    end_time_ms: int
    url: str


class BinanceClient:
    def __init__(self, api_key: str | None = None, secret: str | None = None) -> None:
        self.exchange = ccxt.binanceusdm(
            {
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                    "recvWindow": 5000,
                    "adjustForTimeDifference": True,
                },
                "timeout": 30000,
            }
        )
        self.logger = _logger

    def fetch_ohlcv_with_taker(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | datetime,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        limit = 1000
        base_url = "https://fapi.binance.com/fapi/v1/klines"
        timeout_sec = 30

        if isinstance(start_date, datetime):
            start_iso = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            start_str = str(start_date).strip()
            if "T" in start_str or " " in start_str:
                start_iso = start_str.replace(" ", "T")
                if not start_iso.endswith("Z"):
                    start_iso += "Z"
            else:
                start_iso = f"{start_str}T00:00:00Z"
        since = self.exchange.parse8601(start_iso)

        if end_date is not None:
            if isinstance(end_date, datetime):
                end_iso = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                end_str = str(end_date).strip()
                if "T" in end_str or " " in end_str:
                    end_iso = end_str.replace(" ", "T")
                    if not end_iso.endswith("Z"):
                        end_iso += "Z"
                else:
                    end_iso = f"{end_str}T23:59:59Z"
            end_timestamp = self.exchange.parse8601(end_iso)
        else:
            end_timestamp = self.exchange.milliseconds()

        try:
            market = self.exchange.market(symbol)
            binance_symbol: str = str(market.get("id", symbol).replace("/", ""))
        except Exception:
            binance_symbol = str(symbol).replace("/", "")

        interval_map = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h",
            "12h": "12h", "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M",
        }
        interval = interval_map.get(str(timeframe).strip().lower(), str(timeframe).strip().lower())

        all_rows: list[list[float | int]] = []
        while since < end_timestamp:
            retry_count = 0
            used_weight = 0
            data = None
            while retry_count < 5:
                params = {
                    "symbol": binance_symbol,
                    "interval": interval,
                    "startTime": since,
                    "endTime": end_timestamp,
                    "limit": limit,
                }
                qs = urllib.parse.urlencode(params)
                url = f"{base_url}?{qs}"
                if not url.startswith(("http://", "https://")):
                    raise ValueError(f"Invalid URL scheme: {url}")
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/91.0.4472.124 Safari/537.36"
                    )
                }
                req = urllib.request.Request(url, method="GET", headers=headers)  # noqa: S310
                try:
                    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
                        raw = resp.read().decode("utf-8")
                        used_weight_str = resp.headers.get("x-mbx-used-weight-1m", "0")
                        used_weight = int(used_weight_str) if used_weight_str.isdigit() else 0
                    data = json.loads(raw)
                    retry_count = 0
                except urllib.error.HTTPError as e:
                    if e.code == 429 or 500 <= e.code <= 599:
                        retry_count += 1
                        wait_sec = 120 if e.code == 429 else 2 * retry_count
                        self.logger.warning("HTTP %d for %s. Wait %ds...", e.code, symbol, wait_sec)
                        time.sleep(wait_sec)
                        if retry_count >= 5:
                            self.logger.error("Failed chunks for %s. Skipping.", symbol)
                            break
                        continue
                    if 400 <= e.code < 500:
                        raise BinanceKlinePermanentError(
                            symbol=symbol, timeframe=timeframe, http_code=e.code,
                            start_time_ms=int(since), end_time_ms=int(end_timestamp), url=url,
                        ) from e
                    retry_count += 1
                    wait_sec = 120 if e.code == 429 else 2 * retry_count
                    self.logger.error("Error (%d/5) for %s: %s", retry_count, symbol, e)
                    time.sleep(wait_sec)
                    if retry_count >= 5:
                        break
                    continue
                except Exception as e:
                    retry_count += 1
                    wait_sec = retry_count
                    self.logger.error("Error (%d/5) for %s: %s", retry_count, symbol, e)
                    if retry_count >= 5:
                        break
                    time.sleep(wait_sec)
                    continue

                if not data:
                    since = end_timestamp
                    break

                for row in data:
                    if not isinstance(row, (list, tuple)) or len(row) < 11:
                        continue
                    ts = int(row[0])
                    all_rows.append([
                        ts,
                        float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                        float(row[5]),
                        float(row[7]) if row[7] not in (None, "") else 0.0,
                        float(row[9]) if row[9] not in (None, "") else 0.0,
                        float(row[10]) if row[10] not in (None, "") else 0.0,
                    ])
                break

            if not data:
                break
            last_ts = int(data[-1][0])
            since = last_ts + 1
            cur_d = datetime.fromtimestamp(last_ts / 1000).strftime("%Y-%m-%d")
            self.logger.info("Klines with taker up to %s (%d candles)", cur_d, len(all_rows))
            if last_ts >= end_timestamp:
                break

            if used_weight > 2000:
                time.sleep(10)
            elif used_weight > 1500:
                time.sleep(2.0)
            elif used_weight > 1000:
                time.sleep(0.5)
            else:
                time.sleep(0.1)

        if not all_rows:
            return pd.DataFrame(columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "quote_vol", "taker_buy_base_volume", "taker_buy_quote_volume", "datetime",
            ])

        df = pd.DataFrame(
            all_rows,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "quote_vol", "taker_buy_base_volume", "taker_buy_quote_volume",
            ],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"])
        df = df[df["timestamp"] <= end_timestamp].copy()
        return df

    def fetch_mark_price_klines(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | datetime,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Page USD-S margined ``GET /fapi/v1/markPriceKlines``.

        Normalizes timestamp/open/high/low/close/datetime identically to OHLCV
        and retains no volume field the endpoint does not provide. This fetches
        historical mark-price candles for portfolio MTM/risk simulation only --
        it cannot reconstruct historical margin brackets, liquidation/ADL,
        queue, spread path, or fills.
        """
        limit = 1000
        base_url = "https://fapi.binance.com/fapi/v1/markPriceKlines"
        timeout_sec = 30

        if isinstance(start_date, datetime):
            start_iso = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            start_str = str(start_date).strip()
            if "T" in start_str or " " in start_str:
                start_iso = start_str.replace(" ", "T")
                if not start_iso.endswith("Z"):
                    start_iso += "Z"
            else:
                start_iso = f"{start_str}T00:00:00Z"
        since = self.exchange.parse8601(start_iso)

        if end_date is not None:
            if isinstance(end_date, datetime):
                end_iso = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                end_str = str(end_date).strip()
                if "T" in end_str or " " in end_str:
                    end_iso = end_str.replace(" ", "T")
                    if not end_iso.endswith("Z"):
                        end_iso += "Z"
                else:
                    end_iso = f"{end_str}T23:59:59Z"
            end_timestamp = self.exchange.parse8601(end_iso)
        else:
            end_timestamp = self.exchange.milliseconds()

        try:
            market = self.exchange.market(symbol)
            binance_symbol: str = str(market.get("id", symbol).replace("/", ""))
        except Exception:
            binance_symbol = str(symbol).replace("/", "")

        interval_map = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h",
            "12h": "12h", "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M",
        }
        interval = interval_map.get(str(timeframe).strip().lower(), str(timeframe).strip().lower())

        all_rows: list[list[float | int]] = []
        while since < end_timestamp:
            params = {
                "symbol": binance_symbol,
                "interval": interval,
                "startTime": since,
                "endTime": end_timestamp,
                "limit": limit,
            }
            qs = urllib.parse.urlencode(params)
            url = f"{base_url}?{qs}"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }
            req = urllib.request.Request(url, method="GET", headers=headers)  # noqa: S310
            try:
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
            except Exception as e:  # noqa: BLE001
                self.logger.error("Error fetching mark price klines for %s: %s", symbol, e)
                break

            if not data:
                break

            for row in data:
                if not isinstance(row, (list, tuple)) or len(row) < 5:
                    continue
                ts = int(row[0])
                all_rows.append([
                    ts,
                    float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                ])
            if not all_rows:
                break
            last_ts = int(all_rows[-1][0])
            since = last_ts + 1
            if last_ts >= end_timestamp:
                break
            time.sleep(0.1)

        if not all_rows:
            return pd.DataFrame(columns=[
                "timestamp", "open", "high", "low", "close", "datetime",
            ])

        df = pd.DataFrame(
            all_rows,
            columns=["timestamp", "open", "high", "low", "close"],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"])
        df = df[df["timestamp"] <= end_timestamp].copy()
        return df

    def fetch_funding_rate_history(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        timeout_sec = 30
        limit = 500

        try:
            market = self.exchange.market(symbol)
            binance_symbol: str = str(market.get("id", symbol).replace("/", ""))
        except Exception:
            binance_symbol = str(symbol).replace("/", "")

        if isinstance(start_date, datetime):
            start_ts = int(start_date.timestamp() * 1000)
        else:
            start_str = str(start_date).strip()
            if "T" in start_str or " " in start_str:
                start_iso = start_str.replace(" ", "T")
                if not start_iso.endswith("Z"):
                    start_iso += "Z"
            else:
                start_iso = f"{start_str}T00:00:00Z"
            start_ts = self.exchange.parse8601(start_iso)

        if end_date is not None:
            if isinstance(end_date, datetime):
                end_ts = int(end_date.timestamp() * 1000)
            else:
                end_str = str(end_date).strip()
                if "T" in end_str or " " in end_str:
                    end_iso = end_str.replace(" ", "T")
                    if not end_iso.endswith("Z"):
                        end_iso += "Z"
                else:
                    end_iso = f"{end_str}T23:59:59Z"
                end_ts = self.exchange.parse8601(end_iso)
        else:
            end_ts = self.exchange.milliseconds()

        all_rows: list[tuple[int, float]] = []
        since = start_ts

        while since < end_ts:
            params = {
                "symbol": binance_symbol,
                "startTime": since,
                "endTime": end_ts,
                "limit": limit,
            }
            qs = urllib.parse.urlencode(params)
            url = f"{base_url}?{qs}"
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid URL scheme: {url}")
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }
            req = urllib.request.Request(url, method="GET", headers=headers)  # noqa: S310
            try:
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
            except Exception as e:
                self.logger.error("Error fetching funding rate for %s: %s", symbol, e)
                break

            if not data:
                break

            for item in data:
                if not isinstance(item, dict):
                    continue
                ft = item.get("fundingTime")
                fr = item.get("fundingRate")
                if ft is None or fr is None:
                    continue
                try:
                    ts = int(ft)
                    rate = float(fr)
                except (TypeError, ValueError):
                    continue
                all_rows.append((ts, rate))

            last_ts = int(data[-1]["fundingTime"])
            since = last_ts + 1
            if last_ts >= end_ts:
                break
            time.sleep(0.1)

        if not all_rows:
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

        df = pd.DataFrame(all_rows, columns=["timestamp", "funding_rate"])
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_futures_data_metric(
        self,
        endpoint: str,
        symbol: str,
        *,
        period: str = "5m",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        allowed = {
            "openInterestHist",
            "globalLongShortAccountRatio",
            "topLongShortPositionRatio",
            "takerlongshortRatio",
        }
        if endpoint not in allowed:
            raise ValueError(f"unsupported futures data endpoint: {endpoint}")
        try:
            market = self.exchange.market(symbol)
            binance_symbol: str = str(market.get("id", symbol).replace("/", ""))
        except Exception:
            binance_symbol = str(symbol).replace("/", "")
        base_url = f"https://fapi.binance.com/futures/data/{endpoint}"
        params = {"symbol": binance_symbol, "period": period, "limit": int(limit)}
        qs = urllib.parse.urlencode(params)
        url = f"{base_url}?{qs}"
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL scheme: {url}")
        timeout_sec = 30
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        req = urllib.request.Request(url, method="GET", headers=headers)  # noqa: S310
        data: Any = None
        retry_count = 0
        while retry_count < 5:
            try:
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 or 500 <= e.code <= 599:
                    retry_count += 1
                    wait_sec = 2 * retry_count if e.code != 429 else 60
                    self.logger.warning("HTTP %d for %s %s. Wait %ds...", e.code, endpoint, symbol, wait_sec)
                    time.sleep(wait_sec)
                    if retry_count >= 5:
                        return []
                    continue
                if 400 <= e.code < 500:
                    return []
                retry_count += 1
                time.sleep(2 * retry_count)
                if retry_count >= 5:
                    return []
                continue
            except Exception as e:  # noqa: BLE001
                retry_count += 1
                self.logger.warning("Error fetching %s for %s: %s (%d/5)", endpoint, symbol, e, retry_count)
                if retry_count >= 5:
                    return []
                time.sleep(retry_count)
                continue
        if data is None:
            return []
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            return [data]
        return []
