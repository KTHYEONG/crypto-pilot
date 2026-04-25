from __future__ import annotations

import json
import logging
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import ccxt
import pandas as pd

# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    pass

from config.settings import API_READ_TIMEOUT


class OrderRateLimiter:
    """바이낸스 주문 횟수 제한 방어 (토큰 버킷 알고리즘).

    10초당 최대 80 orders (안전 마진 20%)
    """

    def __init__(self, max_orders_per_10s: int = 80) -> None:
        """Initialize OrderRateLimiter."""
        self.max_orders = max_orders_per_10s
        self.order_timestamps: deque[float] = deque(maxlen=max_orders_per_10s)
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

    def can_place_order(self) -> bool:
        """주문 가능 여부 확인."""
        now = time.time()
        with self._lock:
            while self.order_timestamps and now - self.order_timestamps[0] > 10:
                self.order_timestamps.popleft()

            if len(self.order_timestamps) >= self.max_orders:
                oldest = self.order_timestamps[0]
                wait_time = 10 - (now - oldest)
                self.logger.warning(f"⏸️ Order rate limit: wait {wait_time:.1f}s")
                return False

            return True

    def record_order(self) -> None:
        """주문 기록."""
        with self._lock:
            self.order_timestamps.append(time.time())

    def wait_if_needed(self) -> None:
        """필요시 대기 (원자성 보장)."""
        while True:
            now = time.time()
            wait_time = 0.0

            with self._lock:
                while self.order_timestamps and now - self.order_timestamps[0] > 10:
                    self.order_timestamps.popleft()

                if len(self.order_timestamps) < self.max_orders:
                    self.order_timestamps.append(now)
                    return

                oldest = self.order_timestamps[0]
                wait_time = 10 - (now - oldest)

            self.logger.warning(f"⏸️ Order rate limit: wait {wait_time:.1f}s")
            time.sleep(max(0.1, min(wait_time, 1.0)))


class OrderBookCache:
    """호가창 캐싱 (TTL 기반).

    목적: 0.5초 내 중복 API 호출 방지 → 응답 속도 60% 개선
    """

    def __init__(self, ttl_seconds: float = 0.3) -> None:
        """Initialize OrderBookCache."""
        self.cache: dict[str, tuple[Any, float]] = {}  # {symbol: (data, timestamp)}
        self.ttl = ttl_seconds
        self.logger = logging.getLogger(__name__)

    def get(self, symbol: str) -> Any | None:
        """캐시된 호가창 조회 (스레드 안전성 강화)."""
        cached_item = self.cache.get(symbol)
        if cached_item is not None:
            data, timestamp = cached_item
            age = time.time() - timestamp

            if age < self.ttl:
                self.logger.debug(f"📦 Cache HIT: {symbol} (age: {age * 1000:.0f}ms)")
                return data

        return None

    def set(self, symbol: str, data: Any) -> None:
        """호가창 캐싱."""
        self.cache[symbol] = (data, time.time())

    def invalidate(self, symbol: str | None = None) -> None:
        """캐시 무효화."""
        if symbol:
            self.cache.pop(symbol, None)
        else:
            self.cache.clear()


class BinanceClient:
    """Binance API Client for futures trading."""

    def __init__(
        self,
        api_key: str | None = None,
        secret: str | None = None,
        shared_rate_limiter: OrderRateLimiter | None = None,
    ) -> None:
        """Initialize BinanceClient."""
        # 1. CCXT Exchange 인스턴스 생성
        # 타임아웃: 데이터 조회(READ) 기준으로 기본 설정 (20초)
        self.exchange = ccxt.binanceusdm(
            {
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                    "recvWindow": 5000,
                    "adjustForTimeDifference": True,  # 시간 동기화
                },
                "timeout": API_READ_TIMEOUT * 1000,  # 밀리초 단위 (20초)
            }
        )
        from src.core.utils.utils import setup_logger

        self.logger = setup_logger("BinanceClient")

        # Order Rate Limiter: 주입된 공유 Limiter 사용 또는 자체 생성
        self.rate_limiter = (
            shared_rate_limiter
            if shared_rate_limiter is not None
            else OrderRateLimiter(max_orders_per_10s=80)
        )

        # Order Book Cache (0.3초 TTL - API 호출 60% 감소)
        self.orderbook_cache = OrderBookCache(ttl_seconds=0.3)

    def _ensure_markets_loaded(self) -> None:
        """Load market metadata once for precision/limits helpers."""
        try:
            if not getattr(self.exchange, "markets", None):
                self.exchange.load_markets()
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load markets metadata: {e}")

    def get_symbol_constraints(self, symbol: str) -> dict[str, float]:
        """Return exchange precision/limit constraints for a symbol."""
        self._ensure_markets_loaded()
        constraints = {
            "min_amount": 0.0,
            "min_cost": 0.0,
            "tick_size": 0.0,
        }

        try:
            market = self.exchange.market(symbol)
        except Exception:
            return constraints

        limits = market.get("limits", {}) or {}
        amount_limits = limits.get("amount", {}) or {}
        cost_limits = limits.get("cost", {}) or {}
        constraints["min_amount"] = float(amount_limits.get("min") or 0.0)
        constraints["min_cost"] = float(cost_limits.get("min") or 0.0)

        try:
            filters = (market.get("info") or {}).get("filters", [])
            for f in filters:
                if f.get("filterType") == "PRICE_FILTER":
                    constraints["tick_size"] = float(f.get("tickSize") or 0.0)
                    break
        except Exception:
            self.logger.debug("Failed to fetch tick_size constraints from filters", exc_info=True)
            pass

        return constraints

    def get_price_tick_size(self, symbol: str, fallback: float = 0.01) -> float:
        """Return symbol tick size if available, else fallback."""
        constraints = self.get_symbol_constraints(symbol)
        tick = float(constraints.get("tick_size") or 0.0)
        return tick if tick > 0 else float(fallback)

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to exchange precision."""
        try:
            rounded = self.exchange.price_to_precision(symbol, price)
            return float(rounded)
        except Exception:
            tick = self.get_price_tick_size(symbol, fallback=0.01)
            return round(float(price) / tick) * tick

    def round_amount(self, symbol: str, amount: float) -> float:
        """Round quantity to exchange precision."""
        try:
            rounded = self.exchange.amount_to_precision(symbol, amount)
            return float(rounded)
        except Exception:
            return float(amount)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | datetime,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """지정된 기간의 OHLCV 데이터를 수집합니다.

        Binance API 제한(limit=1000)을 고려하여 반복 호출합니다.
        """
        self._ensure_markets_loaded()
        try:
            market = self.exchange.market(symbol)
            resolved_symbol = market["symbol"]
        except Exception:
            resolved_symbol = symbol

        limit = 1000
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

        if end_date:
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

        all_ohlcv: list[list[float]] = []

        self.logger.info(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}...")

        max_iterations = 500
        iteration_count = 0
        while since < end_timestamp:
            iteration_count += 1
            if iteration_count > max_iterations:
                self.logger.error(
                    "fetch_ohlcv loop exceeded %d iterations. Breaking.",
                    max_iterations,
                )
                break
            retry_count = 0
            current_limit = limit
            while retry_count < 3:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        resolved_symbol, timeframe, since, current_limit
                    )

                    if not ohlcv:
                        since = end_timestamp
                        break

                    all_ohlcv.extend(ohlcv)

                    last_timestamp = ohlcv[-1][0]
                    new_since = int(last_timestamp) + 1
                    if new_since <= since:
                        self.logger.warning(
                            "Timestamp not advancing (%d -> %d). Breaking.",
                            since,
                            new_since,
                        )
                        since = end_timestamp
                        break
                    since = new_since

                    current_date = datetime.fromtimestamp(last_timestamp / 1000).strftime(
                        "%Y-%m-%d"
                    )
                    self.logger.info(f"Measured up to {current_date} ({len(all_ohlcv)} candles)")

                    time.sleep(0.2)

                    if last_timestamp >= end_timestamp:
                        since = end_timestamp
                    break
                except Exception as e:
                    retry_count += 1
                    # 타임아웃 발생 시 데이터량을 줄여서 재시도 (안정성 확보)
                    if "timed out" in str(e).lower():
                        current_limit = max(200, current_limit // 2)

                    wait_sec = 2 * retry_count
                    self.logger.error(
                        "Error fetching data (%d/3) for %s (limit=%d): %s. Waiting %ds...",
                        retry_count, symbol, current_limit, e, wait_sec
                    )
                    time.sleep(wait_sec)
                    if retry_count >= 3:
                        raise RuntimeError(f"Data fetch failed persistently for {symbol}") from e

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        # 중복 제거 및 기간 필터링
        df = df.drop_duplicates(subset=["timestamp"])
        df = df[(df["timestamp"] <= end_timestamp)]

        return df

    def fetch_ohlcv_with_taker(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | datetime,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV plus taker buy base/quote volume from Binance /fapi/v1/klines."""
        limit = 1000
        base_url = "https://fapi.binance.com/fapi/v1/klines"
        timeout_sec = API_READ_TIMEOUT

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
            while retry_count < 5:
                params = {
                    "symbol": binance_symbol, "interval": interval,
                    "startTime": since, "endTime": end_timestamp, "limit": limit,
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
                    retry_count = 0 # Success
                except urllib.error.HTTPError as e:
                    retry_count += 1
                    if e.code == 429:
                        wait_sec = 120
                        self.logger.warning(f"⚠️ HTTP 429 for {symbol}. Wait {wait_sec}s...")
                        time.sleep(wait_sec)
                    else:
                        wait_sec = 2 * retry_count
                        self.logger.error(f"Error ({retry_count}/5) for {symbol}: {e}")
                        time.sleep(wait_sec)
                    if retry_count >= 5:
                        self.logger.error(f"Failed chunks for {symbol}. Skipping.")
                        break
                    continue
                except Exception as e:
                    retry_count += 1
                    wait_sec = 1 * retry_count
                    if "timed out" in str(e).lower() or "connection reset" in str(e).lower():
                        self.logger.warning(f"Retry ({retry_count}/5) for {symbol}")
                    else:
                        self.logger.error(f"Error ({retry_count}/5) for {symbol}: {e}")
                    
                    if retry_count >= 5:
                        self.logger.error(f"Persistent exception for {symbol}. Skipping.")
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
                        ts, float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                        float(row[5]),
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
            
            # Smart Throttling based on Binance 1m used weight (Max: 2400)
            if used_weight > 2000:
                self.logger.warning(
                    f"⚠️ High API weight ({used_weight}/2400) for {symbol}. Sleeping 10s..."
                )
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
                "taker_buy_base_volume", "taker_buy_quote_volume", "datetime"
            ])

        df = pd.DataFrame(all_rows, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "taker_buy_base_volume", "taker_buy_quote_volume",
        ])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"])
        df = df[df["timestamp"] <= end_timestamp].copy()
        return df

    def fetch_recent_ohlcv(self, symbol: str, timeframe: str, limit: int = 3) -> pd.DataFrame:
        """Fetch small recent OHLCV window for live signal timing."""
        try:
            rows = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not rows:
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"]
                )
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching recent OHLCV for {symbol} {timeframe}: {e}")
            raise

    def get_market_price(self, symbol: str) -> float | None:
        """현재 시장가 조회 (캐시 적용)."""
        cached_price = self.orderbook_cache.get(f"{symbol}_ticker")
        if cached_price is not None:
            return float(cached_price)

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker.get("last")
            if price is None:
                self.logger.warning("Ticker 'last' is None for %s", symbol)
                return None
            price_val = float(price)
            self.orderbook_cache.set(f"{symbol}_ticker", price_val)
            return price_val
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {e}")
            return None

    def fetch_server_time_ms(self) -> int | None:
        """Exchange server time in milliseconds."""
        try:
            server_ms = self.exchange.fetch_time()
            if server_ms is not None:
                return int(server_ms)
        except Exception as e:
            self.logger.warning(f"Failed to fetch exchange server time: {e}")
        return None

    def fetch_funding_rate_history(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch funding rate history from Binance GET /fapi/v1/fundingRate."""
        base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        timeout_sec = API_READ_TIMEOUT
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

        self.logger.info(
            "Fetching funding rate for %s from %s to %s...",
            symbol,
            datetime.fromtimestamp(since / 1000).strftime("%Y-%m-%d"),
            datetime.fromtimestamp(end_ts / 1000).strftime("%Y-%m-%d"),
        )

        while since < end_ts:
            params = {
                "symbol": binance_symbol, "startTime": since, "endTime": end_ts,
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
                self.logger.error(f"Error fetching funding rate for {symbol}: {e}")
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
        df = (
            df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        )
        return df

    def fetch_balance(self) -> tuple[float, float]:
        """USDT 선물 지갑 잔고 조회."""
        try:
            balance = self.exchange.fetch_balance()
            total_bal: float = float(balance["total"]["USDT"])
            free_bal: float = float(balance["free"]["USDT"])
            return total_bal, free_bal
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            raise RuntimeError("fetch_balance_failed") from e

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """레버리지 설정."""
        try:
            self.exchange.set_leverage(leverage, symbol)
            self.logger.info(f"✅ Leverage set to {leverage}x for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error setting leverage {leverage} | {e}")
            return False

    def set_position_mode(self, dual_side_position: bool = False) -> bool:
        """포지션 모드 설정."""
        try:
            self.exchange.set_position_mode(hedged=dual_side_position)
            mode_str = "Hedge" if dual_side_position else "One-Way"
            self.logger.info(f"Position mode set to {mode_str} Mode")
            return True
        except Exception as e:
            if "No need to change" in str(e) or "already" in str(e).lower():
                return True
            try:
                self.exchange.fapiPrivate_post_positionside_dual(
                    {"dualSidePosition": "true" if dual_side_position else "false"}
                )
                return True
            except Exception as e2:
                self.logger.error(f"⚠️ Failed to set position mode: {e2}")
                return False

    def set_asset_mode(self, is_multi_asset: bool = False) -> bool:
        """자산 모드 설정."""
        try:
            if not hasattr(self.exchange, "fapiPrivate_get_multiassetsmargin"):
                return True
            res = self.exchange.fapiPrivate_get_multiassetsmargin()
            current = str(res.get("multiAssetsMargin", "false")).lower() == "true"
            if current == is_multi_asset:
                return True
            self.exchange.fapiPrivate_post_multiassetsmargin(
                {"multiAssetsMargin": "true" if is_multi_asset else "false"}
            )
            mode_str = "Multi-Asset" if is_multi_asset else "Single-Asset"
            self.logger.info(f"Asset mode updated to {mode_str} Mode")
            return True
        except Exception as e:
            if "No need to change" in str(e):
                return True
            self.logger.warning(f"Asset mode setting skipped: {e}")
            return True

    def set_margin_type(self, symbol: str, margin_type: str = "CROSSED") -> bool:
        """마진 모드 설정."""
        try:
            mode = "CROSS" if margin_type == "CROSSED" else margin_type
            self.exchange.set_margin_mode(mode, symbol)
            self.logger.info(f"Margin type set to {margin_type} for {symbol}")
            return True
        except Exception as e:
            if "No need to change" in str(e) or "already exists" in str(e).lower():
                return True
            self.logger.error(f"⚠️ Failed to set margin type for {symbol}: {e}")
            return False

    def fetch_position(self, symbol: str) -> dict[str, Any]:
        """현재 포지션 조회."""
        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                p_symbol = str(pos["symbol"])
                contracts = float(pos["contracts"] or 0)
                if p_symbol == symbol or p_symbol.split(":")[0] == symbol:
                    if contracts == 0:
                        continue
                    entry_p = float(pos.get("entryPrice") or 0)
                    upnl = float(pos.get("unrealizedPnl") or 0)
                    lev = int(pos.get("leverage") or 1)
                    result = {
                        "amount": contracts * (1 if pos["side"] == "long" else -1),
                        "entryPrice": entry_p, "unrealizedPnL": upnl, "leverage": lev,
                    }
                    msg = (
                        f"[{symbol}] Pos: {result['amount']} contracts @ "
                        f"{result['entryPrice']:.2f} "
                        f"(PnL: {result['unrealizedPnL']:.2f}, Lev: {lev}x)"
                    )
                    self.logger.info(msg)
                    return result
        except Exception as e:
            self.logger.error(f"Error fetching position for {symbol}: {e}")
            raise RuntimeError(f"fetch_position_failed:{symbol}") from e
        return {"amount": 0.0, "entryPrice": 0.0, "unrealizedPnL": 0.0, "leverage": 1}

    def fetch_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """미체결 주문 조회 (Standard + ALGO 주문 통합)."""
        try:
            orders: list[dict[str, Any]] = list(self.exchange.fetch_open_orders(symbol))
            try:
                algo_res = self.exchange.fapiPrivateGetOpenAlgoOrders()
                market = self.exchange.market(symbol)
                raw_symbol = market["id"]
                for ao in algo_res:
                    if ao.get("symbol") == raw_symbol:
                        algo_order = {
                            "id": ao.get("algoId"), "symbol": symbol,
                            "type": ao.get("orderType", "").lower(),
                            "side": ao.get("side", "").lower(),
                            "amount": float(ao.get("quantity") or 0),
                            "price": float(ao.get("price") or 0),
                            "stopPrice": float(ao.get("triggerPrice") or 0),
                            "status": (
                                "open" if ao.get("algoStatus") == "NEW"
                                else ao.get("algoStatus").lower()
                            ),
                            "timestamp": int(ao.get("createTime", 0)),
                            "datetime": self.exchange.iso8601(ao.get("createTime", 0)),
                            "info": ao, "clientOrderId": ao.get("clientAlgoId"),
                        }
                        if not any(o["id"] == algo_order["id"] for o in orders):
                            orders.append(algo_order)
            except Exception as ae:
                self.logger.warning(f"Failed to fetch ALGO orders for {symbol}: {ae}")
            return orders
        except Exception as e:
            self.logger.error(f"Error fetching open orders for {symbol}: {e}")
            raise RuntimeError(f"fetch_open_orders_failed:{symbol}") from e

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """주문 취소."""
        try:
            self.rate_limiter.wait_if_needed()
            try:
                res = self.exchange.cancel_order(order_id, symbol)
                return bool(res)
            except Exception as e:
                if "Unknown order sent" in str(e) or "-2011" in str(e):
                    res_algo = self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": order_id})
                    return bool(res_algo)
                raise e
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        """모든 오픈 주문 취소."""
        try:
            self.rate_limiter.wait_if_needed()
            try:
                self.exchange.cancel_all_orders(symbol)
            except Exception as e:
                self.logger.warning(f"Standard cancel failed for {symbol}: {e}")
            open_orders = self.fetch_open_orders(symbol)
            for o in open_orders:
                if "algoId" in o.get("info", {}):
                    try:
                        self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": o["id"]})
                    except Exception as ae:
                        self.logger.warning(f"Failed to cancel ALGO order {o['id']}: {ae}")
            return True
        except Exception as e:
            self.logger.error(f"Error in cancel_all_orders: {e}")
            return False

    def fetch_open_interest_history(
        self,
        symbol: str,
        timeframe: str = "4h",
        since: int | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch historical Open Interest from Binance API (Recent 30 days limit)."""
        try:
            rows = self.exchange.fetch_open_interest_history(symbol, timeframe, since, limit)
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            # CCXT usually returns 'openInterestAmount' or similar. 
            # We map the most common Binance field names to our standard 'sum_open_interest'.
            possible_cols = ["openInterestAmount", "openInterest", "amount"]
            for col in possible_cols:
                if col in df.columns:
                    df["sum_open_interest"] = df[col].astype(float)
                    break
            
            if "sum_open_interest" not in df.columns:
                self.logger.warning(
                    f"No OI column found in API response for {symbol}. "
                    f"Cols: {df.columns.tolist()}"
                )
                return pd.DataFrame()

            return df[["timestamp", "sum_open_interest"]]
        except Exception as e:
            self.logger.warning(f"Failed to fetch Open Interest for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_long_short_ratio_history(
        self,
        symbol: str,
        timeframe: str = "4h",
        since: int | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch historical Long/Short Ratio from Binance API (Recent 30 days limit)."""
        try:
            rows = self.exchange.fetch_long_short_ratio_history(symbol, timeframe, since, limit)
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            # Map possible LSR columns to our standard 'count_toptrader_long_short_ratio'
            possible_cols = ["longShortRatio", "ratio", "value"]
            for col in possible_cols:
                if col in df.columns:
                    df["count_toptrader_long_short_ratio"] = df[col].astype(float)
                    break

            if "count_toptrader_long_short_ratio" not in df.columns:
                self.logger.warning(
                    f"No LSR column found in API response for {symbol}. "
                    f"Cols: {df.columns.tolist()}"
                )
                return pd.DataFrame()

            return df[["timestamp", "count_toptrader_long_short_ratio"]]
        except Exception as e:
            self.logger.warning(f"Failed to fetch Long/Short Ratio for {symbol}: {e}")
            return pd.DataFrame()

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """기본 주문 실행."""
        try:
            order: dict[str, Any] = self.exchange.create_order(
                symbol=symbol, type=order_type, side=side,
                amount=amount, price=price, params=(params or {}),
            )
            msg = f"⚡ {order_type} {side} {amount} {symbol} @ {price if price else 'Market'}"
            self.logger.info(msg)
            return order
        except Exception as e:
            self.logger.error(f"❌ Order Failed: {e}")
            return None

    def place_stop_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        client_order_id: str | None = None,
    ) -> dict[str, Any] | None:
        """서버 사이드 Stop Market 주문 (손절용)."""
        try:
            self.rate_limiter.wait_if_needed()
            params: dict[str, object] = {"stopPrice": float(stop_price), "reduceOnly": True}
            if client_order_id:
                params["clientOrderId"] = client_order_id
            order: dict[str, Any] = self.exchange.create_order(
                symbol=symbol, type="STOP_MARKET", side=side,
                amount=amount, params=params
            )
            msg = f"🛡️ Server SL Placed: {symbol} {side} {amount} @ Stop {stop_price}"
            self.logger.info(msg)
            return order
        except Exception as e:
            if "-4130" in str(e) or "-2021" in str(e):
                return None
            self.logger.error(f"❌ Failed Server SL for {symbol}: {e}")
            return None

    def place_take_profit_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        tp_price: float,
        client_order_id: str | None = None,
    ) -> dict[str, Any] | None:
        """서버 사이드 Take Profit Market 주문 (익절용)."""
        try:
            self.rate_limiter.wait_if_needed()
            params: dict[str, object] = {"stopPrice": float(tp_price), "reduceOnly": True}
            if client_order_id:
                params["clientOrderId"] = client_order_id
            order: dict[str, Any] = self.exchange.create_order(
                symbol=symbol, type="TAKE_PROFIT_MARKET", side=side,
                amount=amount, params=params
            )
            msg = f"🎯 Server TP Placed: {symbol} {side} {amount} @ TP {tp_price}"
            self.logger.info(msg)
            return order
        except Exception as e:
            if "-4130" in str(e):
                return {"id": "triggered_4130_tp", "status": "closed", "info": "-4130"}
            if "-2021" in str(e):
                return None
            self.logger.error(f"❌ Failed Server TP for {symbol}: {e}")
            return None

    def place_order_smart(
        self,
        symbol: str,
        side: str,
        amount: float,
        atr: float | None = None,
        current_price: float | None = None,
        reduce_only: bool = False,
        allow_market_fallback: bool = True,
        order_deadline_ms: int | None = None,
        post_only_wait_seconds: float = 1.2,
        post_only_requote_max: int = 2,
        client_order_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Production waterfall order executor."""
        from config.settings import SMART_ORDER_OFFSET
        tick_size = self.get_price_tick_size(symbol, fallback=(0.1 if "BTC" in symbol else 0.01))

        def round_to_tick(price: float, tick: float) -> float:
            return round(float(price) / tick) * tick if tick > 0 else float(price)

        def get_best_fresh() -> tuple[float, float]:
            cached_ob = self.orderbook_cache.get(f"{symbol}_ob")
            if cached_ob:
                return float(cached_ob[0]), float(cached_ob[1])
            try:
                orderbook = self.exchange.fetch_order_book(symbol, limit=1)
                best_bid = (
                    float(orderbook["bids"][0][0])
                    if orderbook["bids"]
                    else float(current_price or 0)
                )
                best_ask = (
                    float(orderbook["asks"][0][0])
                    if orderbook["asks"]
                    else float(current_price or 0)
                )
                self.orderbook_cache.set(f"{symbol}_ob", (best_bid, best_ask))
                return best_bid, best_ask
            except Exception:
                return float(current_price or 0), float(current_price or 0)

        if not current_price:
            current_price = self.get_market_price(symbol)
        if not current_price:
            return None

        requested_total = float(amount)
        remaining_amount = requested_total
        if remaining_amount <= 0:
            return None

        total_filled = 0.0
        last_order = None
        order_fill_tracker: dict[str, float] = {}

        def build_params(
            base_params: dict[str, Any] | None = None,
            *,
            order_tag: str | None = None,
        ) -> dict[str, Any]:
            p = dict(base_params or {})
            if reduce_only:
                p["reduceOnly"] = True
            if client_order_id:
                cid = client_order_id
                if order_tag:
                    cid = f"{client_order_id}_{order_tag}"
                p["clientOrderId"] = cid[:36]
            return p

        def deadline_reached() -> bool:
            if order_deadline_ms is None:
                return False
            try:
                ms = self.exchange.milliseconds()
                return bool(ms >= int(order_deadline_ms))
            except Exception:
                return False

        def register_fill(order_obj: dict[str, Any] | None, req_amt: float) -> float:
            nonlocal remaining_amount, total_filled, last_order
            if not order_obj:
                return 0.0
            last_order = order_obj
            f = float(order_obj.get("filled") or 0.0)
            if req_amt > 0:
                f = min(f, req_amt)
            oid = str(order_obj.get("id") or "")
            if oid:
                prev_f = float(order_fill_tracker.get(oid, 0.0))
                delta = max(0.0, f - prev_f)
                order_fill_tracker[oid] = max(prev_f, f)
            else:
                delta = max(0.0, f)
            if delta > 0:
                total_filled += delta
                remaining_amount = max(0.0, requested_total - total_filled)
            return delta

        def refresh_order(order_obj: dict[str, Any] | None) -> dict[str, Any] | None:
            if not isinstance(order_obj, dict):
                return order_obj
            oid = order_obj.get("id")
            if not oid:
                return order_obj
            try:
                refreshed: dict[str, Any] = self.exchange.fetch_order(oid, symbol)
                return refreshed
            except Exception:
                try:
                    for o in self.fetch_open_orders(symbol):
                        if str(o.get("id")) == str(oid):
                            return o
                except Exception:
                    ...
            return order_obj

        def wait_post_only(order_obj: dict[str, Any], req_amt: float) -> dict[str, Any]:
            wait_sec = max(0.0, float(post_only_wait_seconds or 0.0))
            if wait_sec <= 0:
                return order_obj
            wait_dl = time.time() + wait_sec
            latest = order_obj
            while time.time() < wait_dl and remaining_amount > 0:
                if deadline_reached():
                    break
                time.sleep(0.2)
                latest_ref = refresh_order(latest)
                if latest_ref:
                    latest = latest_ref
                register_fill(latest, req_amt)
                st = str(latest.get("status", "")).lower()
                if st in ("closed", "filled", "canceled", "cancelled") or remaining_amount <= 0:
                    break
            return latest

        def build_partial_res(status: str = "partial") -> dict[str, Any]:
            res = {
                "symbol": symbol, "side": side, "type": "multi_tier",
                "status": status, "requested": requested_total,
                "filled": total_filled, "remaining": max(0.0, requested_total - total_filled),
                "reduceOnly": bool(reduce_only),
            }
            if isinstance(last_order, dict):
                res["last_order_id"] = last_order.get("id")
                res["average"] = last_order.get("average")
            return res

        high_vol = False
        if atr and current_price:
            vol_pct = (atr / current_price) * 100
            if vol_pct > 1.0:
                high_vol = True
                self.logger.info(f"🔥 High Vol ({vol_pct:.2f}%) -> Aggressive")

        if not high_vol and remaining_amount > 0:
            max_req = max(0, int(post_only_requote_max or 0))
            for req_idx in range(max_req + 1):
                if remaining_amount <= 0 or deadline_reached():
                    break
                try:
                    self.rate_limiter.wait_if_needed()
                    b_bid, b_ask = get_best_fresh()
                    target_p = round_to_tick(
                        b_bid + tick_size if side == "buy" else b_ask - tick_size, tick_size
                    )
                    msg_t1 = (
                        f"📦 Tier 1: Post-Only {side} @ {target_p} "
                        f"({req_idx + 1}/{max_req + 1})"
                    )
                    self.logger.info(msg_t1)
                    req_amt = remaining_amount
                    f_before = total_filled
                    order_limit = self.exchange.create_order(
                        symbol=symbol, type="limit", side=side, amount=req_amt,
                        price=target_p,
                        params=build_params({"postOnly": True}, order_tag=f"T1{req_idx}")
                    )
                    register_fill(order_limit, req_amt)
                    order_limit = wait_post_only(order_limit, req_amt)
                    st = str(order_limit.get("status", "")).lower()
                    f_round = max(0.0, total_filled - f_before)
                    if st in ("closed", "filled") or remaining_amount <= 0:
                        self.logger.info(f"Tier 1 Filled ({f_round}/{req_amt})")
                        return cast("dict[str, Any] | None", order_limit)
                    if f_round > 0:
                        self.logger.warning(f"⚠️ T1 Partial: {f_round}/{req_amt}")
                    if st in ("open", "new"):
                        try:
                            self.exchange.cancel_order(str(order_limit["id"]), symbol)
                            if req_idx < max_req and not deadline_reached():
                                continue
                        except Exception as cancel_err:
                            msg = f"🗑️ T1 cancel failed: {cancel_err}."
                            self.logger.warning(f"{msg} Reconciling...")
                            try:
                                self.exchange.cancel_all_orders(symbol)
                                if self.fetch_open_orders(symbol):
                                    return build_partial_res(status="cancel_failed_open")
                            except Exception:
                                return build_partial_res(status="cancel_failed_open")
                        break
                except Exception as e:
                    if "timeout" in str(e).lower():
                        try:
                            if self.fetch_open_orders(symbol):
                                self.exchange.cancel_all_orders(symbol)
                                if self.fetch_open_orders(symbol):
                                    return build_partial_res(status="timeout_open_orders")
                        except Exception:
                            ...
                    break

        if remaining_amount > 0:
            if deadline_reached():
                if total_filled > 0:
                    return build_partial_res(status="deadline_partial")
                return None
            try:
                self.rate_limiter.wait_if_needed()
                b_bid, b_ask = get_best_fresh()
                off = 0.001 if high_vol else SMART_ORDER_OFFSET
                limit_p = round_to_tick(
                    b_ask * (1 + off) if side == "buy" else b_bid * (1 - off), tick_size
                )
                self.logger.info(f"📦 Tier 2: Aggressive IOC {side} @ {limit_p}")
                req_amt_t2 = remaining_amount
                order_t2 = self.exchange.create_order(
                    symbol=symbol, type="limit", side=side, amount=req_amt_t2,
                    price=limit_p, params=build_params({"timeInForce": "IOC"}, order_tag="T2")
                )
                filled = register_fill(order_t2, req_amt_t2)
                if filled > 0:
                    if remaining_amount <= 0 or (filled / req_amt_t2) >= 0.999:
                        self.logger.info(f"Tier 2 Filled ({filled}/{req_amt_t2})")
                        return cast("dict[str, Any] | None", order_t2)
            except Exception as e:
                self.logger.error(f"❌ Tier 2 Failed: {e}")
                if "timeout" in str(e).lower():
                    try:
                        self.exchange.cancel_all_orders(symbol)
                    except Exception:
                        ...

        if remaining_amount > 0:
            if deadline_reached():
                if total_filled > 0:
                    return build_partial_res(status="deadline_partial")
                return None
            try:
                if not allow_market_fallback:
                    self.logger.warning(f"⚠️ Market fallback disabled for {symbol}")
                    if total_filled > 0:
                        return build_partial_res(status="partial_no_market")
                    return None
                self.rate_limiter.wait_if_needed()
                if current_price and atr:
                    slip = min(0.01, (float(atr) * 0.2) / float(current_price))
                    h_limit_p = round_to_tick(
                        current_price * (1 + slip) if side == "buy" else current_price * (1 - slip),
                        tick_size
                    )
                    self.logger.warning(f"🚨 Tier 3: Capped Market {side} @ {h_limit_p}")
                    order_t3 = self.exchange.create_order(
                        symbol=symbol, type="limit", side=side, amount=remaining_amount,
                        price=h_limit_p,
                        params=build_params({"timeInForce": "IOC"}, order_tag="T3L")
                    )
                else:
                    self.logger.warning(f"🚨 Tier 3: Market Order {side}")
                    order_t3 = self.exchange.create_order(
                        symbol=symbol, type="market", side=side, amount=remaining_amount,
                        params=build_params(order_tag="T3M")
                    )
                register_fill(order_t3, remaining_amount)
                return cast("dict[str, Any] | None", order_t3)
            except Exception as e:
                self.logger.error(f"❌ All Tiers Failed: {e}")
                if total_filled > 0:
                    return build_partial_res(status="partial_failed")
                return None
        final_res = last_order if last_order else build_partial_res(status="filled")
        return cast("dict[str, Any] | None", final_res)
