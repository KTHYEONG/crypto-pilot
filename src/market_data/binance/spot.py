from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

import ccxt
import pandas as pd

from src.market_data.binance.futures import BinanceKlinePermanentError

_SPOT_LOGGER = logging.getLogger("BinanceSpotClient")


class BinanceSpotClient:
    """Binance spot kline adapter (api.binance.com/api/v3/klines).

    Independent of the USD-M futures client: it must never read a futures file
    or hit the fapi endpoint. Returns the common nine-column kline schema
    (``timestamp``, OHLCV, ``quote_vol``, and the two taker-buy volumes).
    """

    BASE_URL = "https://api.binance.com/api/v3/klines"
    TIMEOUT_SECONDS = 30

    def __init__(self, api_key: str | None = None, secret: str | None = None) -> None:
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot", "recvWindow": 5000},
                "timeout": 30000,
            }
        )
        self.logger = _SPOT_LOGGER

    @staticmethod
    def _parse_iso(value: str | datetime, *, end_of_day: bool) -> int:
        text = str(value).strip()
        if "T" not in text and " " not in text:
            text = f"{text}T{'23:59:59' if end_of_day else '00:00:00'}"
        timestamp = pd.Timestamp(text)
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        return int(timestamp.timestamp() * 1000)

    def fetch_spot_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | datetime,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        limit = 1000
        since = self._parse_iso(start_date, end_of_day=False)
        if end_date is not None:
            end_timestamp = self._parse_iso(end_date, end_of_day=True)
        else:
            end_timestamp = self.exchange.milliseconds()

        try:
            market = self.exchange.market(symbol)
            binance_symbol: str = str(market.get("id", symbol).replace("/", ""))
        except Exception:
            binance_symbol = str(symbol).replace("/", "")

        interval = str(timeframe).strip().lower()

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
            url = f"{self.BASE_URL}?{qs}"
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
            data: list[Any] | None = None
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=self.TIMEOUT_SECONDS) as resp:  # noqa: S310
                        raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 or 500 <= e.code <= 599:
                        wait_sec = 120 if e.code == 429 else 2 * (attempt + 1)
                        self.logger.warning("HTTP %d for %s. Wait %ds...", e.code, symbol, wait_sec)
                        time.sleep(wait_sec)
                    elif 400 <= e.code < 500:
                        raise BinanceKlinePermanentError(
                            symbol=symbol, timeframe=timeframe, http_code=e.code,
                            start_time_ms=int(since), end_time_ms=int(end_timestamp), url=url,
                        ) from e
                    else:
                        self.logger.error("Error (%d/5) for %s: %s", attempt + 1, symbol, e)
                        time.sleep(attempt + 1)
                except Exception as e:
                    self.logger.error("Error (%d/5) for %s: %s", attempt + 1, symbol, e)
                    time.sleep(attempt + 1)
            else:
                self.logger.error("Failed chunks for %s. Skipping.", symbol)
                break

            if not data:
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
            last_ts = int(data[-1][0])
            since = last_ts + 1
            if last_ts >= end_timestamp:
                break
            time.sleep(0.1)

        if not all_rows:
            return pd.DataFrame(columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "quote_vol", "taker_buy_base_volume", "taker_buy_quote_volume",
            ])
        df = pd.DataFrame(
            all_rows,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "quote_vol", "taker_buy_base_volume", "taker_buy_quote_volume",
            ],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df[df["timestamp"] <= end_timestamp].copy()
        return df
