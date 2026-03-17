
from __future__ import annotations

import ccxt
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
from collections import deque
import sys
from pathlib import Path
import urllib.parse
import urllib.request
import json
import threading

# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    pass

from config.settings import API_READ_TIMEOUT, API_ORDER_TIMEOUT, API_CHECK_TIMEOUT

class OrderRateLimiter:
    """
    바이낸스 주문 횟수 제한 방어 (토큰 버킷 알고리즘)
    10초당 최대 80 orders (안전 마진 20%)
    """
    def __init__(self, max_orders_per_10s=80):
        self.max_orders = max_orders_per_10s
        self.order_timestamps = deque(maxlen=max_orders_per_10s)
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def can_place_order(self) -> bool:
        """주문 가능 여부 확인"""
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
    
    def record_order(self):
        """주문 기록"""
        with self._lock:
            self.order_timestamps.append(time.time())
    
    def wait_if_needed(self):
        """필요시 대기 (원자성 보장)"""
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
    """
    호가창 캐싱 (TTL 기반)
    목적: 0.5초 내 중복 API 호출 방지 → 응답 속도 60% 개선
    """
    def __init__(self, ttl_seconds=0.3):
        self.cache = {}  # {symbol: (data, timestamp)}
        self.ttl = ttl_seconds
        self.logger = logging.getLogger(__name__)
    
    def get(self, symbol):
        """캐시된 호가창 조회 (스레드 안전성 강화)"""
        cached_item = self.cache.get(symbol)
        if cached_item is not None:
            data, timestamp = cached_item
            age = time.time() - timestamp

            if age < self.ttl:
                self.logger.debug(f"📦 Cache HIT: {symbol} (age: {age*1000:.0f}ms)")
                return data

        return None
    
    def set(self, symbol, data):
        """호가창 캐싱"""
        self.cache[symbol] = (data, time.time())
    
    def invalidate(self, symbol=None):
        """캐시 무효화"""
        if symbol:
            self.cache.pop(symbol, None)
        else:
            self.cache.clear()


class BinanceClient:
    def __init__(self, api_key=None, secret=None, shared_rate_limiter=None):
        # 1. CCXT Exchange 인스턴스 생성
        # 타임아웃: 데이터 조회(READ) 기준으로 기본 설정 (20초)
        self.exchange = ccxt.binanceusdm({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'recvWindow': 5000,
                'adjustForTimeDifference': True,  # 시간 동기화
            },
            'timeout': API_READ_TIMEOUT * 1000  # 밀리초 단위 (20초)
        })
        from src.common.utils import setup_logger
        self.logger = setup_logger("BinanceClient")
        
        # Order Rate Limiter: 주입된 공유 Limiter 사용 또는 자체 생성
        self.rate_limiter = (
            shared_rate_limiter
            if shared_rate_limiter is not None
            else OrderRateLimiter(max_orders_per_10s=80)
        )
        
        # Order Book Cache (0.3초 TTL - API 호출 60% 감소)
        self.orderbook_cache = OrderBookCache(ttl_seconds=0.3)

    def _ensure_markets_loaded(self):
        """Load market metadata once for precision/limits helpers."""
        try:
            if not getattr(self.exchange, 'markets', None):
                self.exchange.load_markets()
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load markets metadata: {e}")

    def get_symbol_constraints(self, symbol):
        """
        Return exchange precision/limit constraints for a symbol.
        """
        self._ensure_markets_loaded()
        constraints = {
            'min_amount': 0.0,
            'min_cost': 0.0,
            'tick_size': 0.0,
        }

        try:
            market = self.exchange.market(symbol)
        except Exception:
            return constraints

        limits = market.get('limits', {}) or {}
        amount_limits = limits.get('amount', {}) or {}
        cost_limits = limits.get('cost', {}) or {}
        constraints['min_amount'] = float(amount_limits.get('min') or 0.0)
        constraints['min_cost'] = float(cost_limits.get('min') or 0.0)

        try:
            filters = (market.get('info') or {}).get('filters', [])
            for f in filters:
                if f.get('filterType') == 'PRICE_FILTER':
                    constraints['tick_size'] = float(f.get('tickSize') or 0.0)
                    break
        except Exception:
            pass

        return constraints

    def get_price_tick_size(self, symbol, fallback=0.01):
        """Return symbol tick size if available, else fallback."""
        constraints = self.get_symbol_constraints(symbol)
        tick = float(constraints.get('tick_size') or 0.0)
        return tick if tick > 0 else float(fallback)

    def round_price(self, symbol, price):
        """Round price to exchange precision."""
        try:
            rounded = self.exchange.price_to_precision(symbol, price)
            return float(rounded)
        except Exception:
            tick = self.get_price_tick_size(symbol, fallback=0.01)
            return round(float(price) / tick) * tick

    def round_amount(self, symbol, amount):
        """Round quantity to exchange precision."""
        try:
            rounded = self.exchange.amount_to_precision(symbol, amount)
            return float(rounded)
        except Exception:
            return float(amount)

    def fetch_ohlcv(self, symbol, timeframe, start_date, end_date=None):
        """
        지정된 기간의 OHLCV 데이터를 수집합니다.
        Binance API 제한(limit=1000)을 고려하여 반복 호출합니다.
        """
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

        all_ohlcv = []
        
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
            while retry_count < 3:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)

                    if not ohlcv:
                        since = end_timestamp
                        break

                    all_ohlcv.extend(ohlcv)

                    last_timestamp = ohlcv[-1][0]
                    new_since = last_timestamp + 1
                    if new_since <= since:
                        self.logger.warning(
                            "Timestamp not advancing (%d -> %d). Breaking.",
                            since,
                            new_since,
                        )
                        since = end_timestamp
                        break
                    since = new_since

                    current_date = datetime.fromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d')
                    self.logger.info(f"Measured up to {current_date} ({len(all_ohlcv)} candles)")

                    time.sleep(0.1)

                    if last_timestamp >= end_timestamp:
                        since = end_timestamp
                    break
                except Exception as e:
                    retry_count += 1
                    self.logger.error(f"Error fetching data ({retry_count}/3): {e}")
                    time.sleep(5)
                    if retry_count >= 3:
                        raise RuntimeError(f"Data fetch failed persistently for {symbol}") from e

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 중복 제거 및 기간 필터링
        df = df.drop_duplicates(subset=['timestamp'])
        df = df[(df['timestamp'] <= end_timestamp)]
        
        return df

    def fetch_ohlcv_with_taker(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | datetime,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV plus taker buy base/quote volume from Binance /fapi/v1/klines.
        Returns DataFrame with columns: timestamp, open, high, low, close, volume,
        taker_buy_base_volume, taker_buy_quote_volume, datetime.
        """
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

        interval_map = {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
                       "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h",
                       "12h": "12h", "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M"}
        interval = interval_map.get(str(timeframe).strip().lower(), str(timeframe).strip().lower())

        all_rows: list[list[float | int]] = []
        while since < end_timestamp:
            retry_count = 0
            while retry_count < 3:
                params = {
                    "symbol": binance_symbol,
                    "interval": interval,
                    "startTime": since,
                    "endTime": end_timestamp,
                    "limit": limit,
                }
                qs = urllib.parse.urlencode(params)
                url = f"{base_url}?{qs}"
                req = urllib.request.Request(url, method="GET")
                try:
                    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                        raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                except Exception as e:
                    retry_count += 1
                    self.logger.error(
                        "Error fetching klines with taker (%d/3) for %s: %s",
                        retry_count,
                        symbol,
                        e,
                    )
                    time.sleep(5)
                    if retry_count >= 3:
                        raise RuntimeError(
                            f"Data fetch with taker failed persistently for {symbol}"
                        ) from e
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
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                        float(row[9]) if row[9] not in (None, "") else 0.0,
                        float(row[10]) if row[10] not in (None, "") else 0.0,
                    ])

                break

            if not data:
                break

            last_ts = int(data[-1][0])
            since = last_ts + 1
            current_date = datetime.fromtimestamp(last_ts / 1000).strftime("%Y-%m-%d")
            self.logger.info("Klines with taker up to %s (%d candles)", current_date, len(all_rows))
            if last_ts >= end_timestamp:
                break
            time.sleep(0.1)

        if not all_rows:
            return pd.DataFrame(
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "taker_buy_base_volume", "taker_buy_quote_volume", "datetime",
                ]
            )

        df = pd.DataFrame(
            all_rows,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "taker_buy_base_volume", "taker_buy_quote_volume",
            ],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"])
        df = df[df["timestamp"] <= end_timestamp].copy()
        return df

    def fetch_recent_ohlcv(self, symbol, timeframe, limit=3):
        """Fetch small recent OHLCV window for live signal timing."""
        try:
            rows = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not rows:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'datetime'])
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.logger.error(f"Error fetching recent OHLCV for {symbol} {timeframe}: {e}")
            raise

    def get_market_price(self, symbol):
        """현재 시장가 조회 (캐시 적용)"""
        cached_price = self.orderbook_cache.get(f"{symbol}_ticker")
        if cached_price is not None:
            return cached_price

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker.get("last")
            if price is None:
                self.logger.warning("Ticker 'last' is None for %s", symbol)
                return None
            price = float(price)
            self.orderbook_cache.set(f"{symbol}_ticker", price)
            return price
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {e}")
            return None
    

    def fetch_server_time_ms(self):
        """Exchange server time in milliseconds."""
        try:
            server_ms = self.exchange.fetch_time()
            if server_ms is not None:
                return int(server_ms)
        except Exception as e:
            self.logger.warning(f"Failed to fetch exchange server time: {e}")
        return None

    def fetch_funding_rate(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch funding rate history from Binance GET /fapi/v1/fundingRate.
        Returns DataFrame with columns: timestamp (ms), funding_rate.
        """
        base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        timeout_sec = API_READ_TIMEOUT
        limit = 1000

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
        retry_count = 0

        self.logger.info(
            "Fetching funding rate for %s from %s to %s...",
            symbol,
            datetime.fromtimestamp(since / 1000).strftime("%Y-%m-%d"),
            datetime.fromtimestamp(end_ts / 1000).strftime("%Y-%m-%d"),
        )

        while since < end_ts:
            params = {
                "symbol": binance_symbol,
                "startTime": since,
                "endTime": end_ts,
                "limit": limit,
            }
            qs = urllib.parse.urlencode(params)
            url = f"{base_url}?{qs}"
            req = urllib.request.Request(url, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                retry_count = 0
            except Exception as e:
                retry_count += 1
                self.logger.error(
                    "Error fetching funding rate (%d/3): %s", retry_count, e
                )
                time.sleep(5)
                if retry_count >= 3:
                    raise RuntimeError(
                        f"Funding rate fetch failed persistently for {symbol}"
                    ) from e
                continue

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

            if not data:
                break

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

    def fetch_balance(self):
        """USDT 선물 지갑 잔고 조회"""
        try:
            balance = self.exchange.fetch_balance()
            # future wallet: total -> total margin balance, free -> available balance
            return balance['total']['USDT'], balance['free']['USDT']
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            raise RuntimeError("fetch_balance_failed") from e

    def set_leverage(self, symbol, leverage):
        """레버리지 설정"""
        try:
            # use standard ccxt method for better compatibility
            self.exchange.set_leverage(leverage, symbol)
            self.logger.info(f"✅ Leverage set to {leverage}x for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error setting leverage (Value: {leverage}) | Detail: {str(e)}")
            return False

    def set_position_mode(self, dual_side_position=False):
        """
        포지션 모드 설정 (봇의 안정성을 위해 필수)
        dual_side_position=False -> One-Way Mode (단방향)
        dual_side_position=True -> Hedge Mode (양방향)
        """
        try:
            # CCXT unified method
            self.exchange.set_position_mode(hedged=dual_side_position)
            mode_str = "Hedge" if dual_side_position else "One-Way"
            self.logger.info(f"Position mode set to {mode_str} Mode")
            return True
        except Exception as e:
            # 이미 설정됨 등 무시 가능한 에러 처리
            if "No need to change" in str(e) or "already" in str(e).lower():
                return True
            # 일부 구버전 CCXT나 거래소 응답에 따라 직접 호출이 필요할 수 있음 (Fallback)
            try:
                self.exchange.fapiPrivate_post_positionside_dual({
                    'dualSidePosition': 'true' if dual_side_position else 'false'
                })
                return True
            except Exception as e2:
                self.logger.error(f"⚠️ Failed to set position mode: {e2}")
                return False


    def set_asset_mode(self, is_multi_asset=False):
        """
        자산 모드 설정 (Single-Asset vs Multi-Asset)
        is_multi_asset=False -> Single-Asset Mode (USDT만 담보)
        is_multi_asset=True -> Multi-Asset Mode (타 코인도 담보 인정)
        
        Note: 일부 CCXT 버전에서 미지원될 수 있음. 실패해도 봇 운영에 영향 없음.
        """
        try:
            # CCXT fapiPrivate 메서드 존재 확인
            if not hasattr(self.exchange, 'fapiPrivate_get_multiassetsmargin'):
                self.logger.info("Asset mode API not available in this CCXT version. Skipping.")
                return True
            
            # 1. Check current mode
            res = self.exchange.fapiPrivate_get_multiassetsmargin()
            current = str(res.get('multiAssetsMargin', 'false')).lower() == 'true'
            
            if current == is_multi_asset:
                return True
                
            # 2. Set mode  
            self.exchange.fapiPrivate_post_multiassetsmargin({
                'multiAssetsMargin': 'true' if is_multi_asset else 'false'
            })
            mode_str = "Multi-Asset" if is_multi_asset else "Single-Asset"
            self.logger.info(f"Asset mode updated to {mode_str} Mode")
            return True
        except AttributeError:
            # CCXT 버전 문제로 메서드 미존재
            self.logger.info("Asset mode API not available. Skipping (non-critical).")
            return True
        except Exception as e:
            if "No need to change" in str(e):
                return True
            # 실패해도 봇 운영에 지장 없으므로 WARNING 레벨로 변경
            self.logger.warning(f"Asset mode setting skipped: {e}")
            return True  # 봇 계속 진행



    def set_margin_type(self, symbol, margin_type='CROSSED'):
        """
        마진 모드 설정 (ISOLATED / CROSSED)
        봇 운용 시 청산 방지를 위해 CROSSED 권장
        """
        try:
            # CCXT unified method: ISOLATED, CROSS
            mode = 'CROSS' if margin_type == 'CROSSED' else margin_type
            self.exchange.set_margin_mode(mode, symbol)
            self.logger.info(f"Margin type set to {margin_type} for {symbol}")
            return True
        except Exception as e:
            # 이미 설정되어 있거나 포지션이 있으면 에러 발생 (무시 가능)
            if "No need to change" in str(e) or "already exists" in str(e).lower():
                return True
            self.logger.error(f"⚠️ Failed to set margin type for {symbol}: {e}")
            return False



    def fetch_position(self, symbol):
        """
        현재 포지션 조회
        Returns: {
            'amount': 0.0,
            'entryPrice': 0.0,
            'unrealizedPnL': 0.0,
            'leverage': 1
        }
        """
        try:
            positions = self.exchange.fetch_positions([symbol])
            
            # [DEBUG] 모든 반환된 포지션 로깅
            self.logger.debug(f"[{symbol}] API returned {len(positions)} position(s)")
            
            for pos in positions:
                # Symbol Matching: Handle 'ETH/USDT:USDT' vs 'ETH/USDT'
                # CCXT often returns 'BASE/QUOTE:SETTLE' for futures
                p_symbol = pos['symbol']
                contracts = float(pos['contracts'] or 0)
                
                # [DEBUG] 각 포지션 상세 로깅
                self.logger.debug(
                    f"  → Symbol: {p_symbol}, Contracts: {contracts}, "
                    f"Side: {pos.get('side')}, Entry: {pos.get('entryPrice')}"
                )
                
                if p_symbol == symbol or p_symbol.split(':')[0] == symbol:
                    # 0인 포지션 데이터는 건너뜀 (Binance는 모든 페어의 0포지션을 리턴하기도 함)
                    if contracts == 0:
                        continue

                    # [FIX] None-safe 처리 (API가 null을 반환하는 경우 대비)
                    entry_price = float(pos.get('entryPrice') or 0)
                    unrealized_pnl = float(pos.get('unrealizedPnl') or 0)
                    leverage = int(pos.get('leverage') or 1)  # None이면 1로 기본값 설정
                    
                    result = {
                        'amount': contracts * (1 if pos['side'] == 'long' else -1), 
                        'entryPrice': entry_price,
                        'unrealizedPnL': unrealized_pnl,
                        'leverage': leverage
                    }
                    self.logger.info(
                        f"[{symbol}] Position Found: {result['amount']} contracts "
                        f"@ {result['entryPrice']:.2f} (PnL: {result['unrealizedPnL']:.2f} USDT, Lev: {leverage}x)"
                    )
                    return result
        except Exception as e:
            self.logger.error(f"Error fetching position for {symbol}: {e}")
            raise RuntimeError(f"fetch_position_failed:{symbol}") from e

        # 포지션 없으면 기본 0 반환
        self.logger.debug(f"[{symbol}] No active position found")
        return {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0, 'leverage': 1}

    def fetch_open_orders(self, symbol):
        """미체결 주문 조회"""
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching open orders for {symbol}: {e}")
            raise RuntimeError(f"fetch_open_orders_failed:{symbol}") from e

    def place_order(self, symbol, side, amount, order_type='market', price=None, params=None):
        """
        기본 주문 실행 (내부 사용)
        side: 'buy' or 'sell'
        amount: 코인 수량 (USDT 아님)
        """
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=(params or {})
            )
            self.logger.info(f"⚡ Order Placed: {order_type} {side} {amount} {symbol} @ {price if price else 'Market'}")
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
        client_order_id: Optional[str] | None = None,
    ):
        """
        서버 사이드 Stop Market 주문 (손절용)
        side: 'buy' (숏 손절용) or 'sell' (롱 손절용)
        stop_price: 트리거 가격
        """
        try:
            self.rate_limiter.wait_if_needed()

            params: dict[str, object] = {
                "stopPrice": float(stop_price),
                "reduceOnly": True,
            }
            if client_order_id:
                params["clientOrderId"] = client_order_id

            order = self.exchange.create_order(
                symbol=symbol,
                type="STOP_MARKET",
                side=side,
                amount=amount,
                params=params,
            )
            self.logger.info(
                f"🛡️ Server SL Placed: {symbol} {side} {amount} @ Stop {stop_price}"
            )
            return order
        except Exception as e:
            error_msg = str(e)
            if "-4130" in error_msg:
                self.logger.warning(
                    f"⚠️ [-4130] SL would immediately trigger for {symbol}. Returning None."
                )
                return None
            if "-2021" in error_msg:
                return None
            self.logger.error(f"❌ Failed to place Server SL for {symbol}: {e}")
            return None

    def place_take_profit_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        tp_price: float,
        client_order_id: Optional[str] | None = None,
    ):
        """
        서버 사이드 Take Profit Market 주문 (익절용)
        side: 'buy' (숏 익절용) or 'sell' (롱 익절용)
        tp_price: 트리거 가격
        """
        try:
            self.rate_limiter.wait_if_needed()

            params: dict[str, object] = {
                "stopPrice": float(tp_price),
                "reduceOnly": True,
            }
            if client_order_id:
                params["clientOrderId"] = client_order_id

            order = self.exchange.create_order(
                symbol=symbol,
                type="TAKE_PROFIT_MARKET",
                side=side,
                amount=amount,
                params=params,
            )
            self.logger.info(
                f"🎯 Server TP Placed: {symbol} {side} {amount} @ TP {tp_price}"
            )
            return order
        except Exception as e:
            error_msg = str(e)
            if "-4130" in error_msg:
                self.logger.warning(f"⚠️ [-4130] TP would immediately trigger for {symbol}. Bypassing.")
                return {"id": "triggered_4130_tp", "status": "closed", "info": "-4130"}
            if "-2021" in error_msg:
                return None
            self.logger.error(f"❌ Failed to place Server TP for {symbol}: {e}")
            return None

    def place_order_smart(
        self,
        symbol,
        side,
        amount,
        atr=None,
        current_price=None,
        reduce_only=False,
        allow_market_fallback=True,
        order_deadline_ms=None,
        post_only_wait_seconds=1.2,
        post_only_requote_max=2,
    ):
        """
        Production waterfall order executor.
        - Tier 1: Post-only limit
        - Tier 2: IOC aggressive limit
        - Tier 3: Market fallback
        """
        from config.settings import SMART_ORDER_OFFSET

        tick_size = self.get_price_tick_size(symbol, fallback=(0.1 if 'BTC' in symbol else 0.01))

        def round_to_tick(price, tick):
            if tick <= 0:
                return float(price)
            return round(float(price) / tick) * tick

        def get_best_price_fresh():
            cached_ob = self.orderbook_cache.get(f"{symbol}_ob")
            if cached_ob:
                return cached_ob[0], cached_ob[1]

            try:
                orderbook = self.exchange.fetch_order_book(symbol, limit=1)
                best_bid = (
                    orderbook["bids"][0][0] if orderbook["bids"] else current_price
                )
                best_ask = (
                    orderbook["asks"][0][0] if orderbook["asks"] else current_price
                )
                self.orderbook_cache.set(f"{symbol}_ob", (best_bid, best_ask))
                return best_bid, best_ask
            except Exception:
                return current_price, current_price

        if not current_price:
            current_price = self.get_market_price(symbol)
        if not current_price:
            self.logger.error(f"❌ Unable to fetch market price for smart order: {symbol}")
            return None

        requested_total = float(amount)
        remaining_amount = requested_total
        if remaining_amount <= 0:
            self.logger.error(f"❌ Invalid order amount: {amount}")
            return None

        total_filled = 0.0
        last_order = None
        order_fill_tracker = {}

        def build_params(base_params=None):
            params = dict(base_params or {})
            if reduce_only:
                params['reduceOnly'] = True
            return params

        start_local_time_ms = int(time.time() * 1000)

        def deadline_reached():
            if order_deadline_ms is None:
                return False
            try:
                elapsed_ms = int(time.time() * 1000) - start_local_time_ms
                _ = elapsed_ms  # keep for potential future use
                current_synced_ms = self.exchange.milliseconds()
                return current_synced_ms >= int(order_deadline_ms)
            except Exception:
                return False

        def register_fill(order_obj, request_amount):
            nonlocal remaining_amount, total_filled, last_order
            if not order_obj:
                return 0.0
            last_order = order_obj
            filled = float(order_obj.get('filled') or 0.0)
            if request_amount > 0:
                filled = min(filled, request_amount)
            order_id = str(order_obj.get('id') or '')
            if order_id:
                prev_filled = float(order_fill_tracker.get(order_id, 0.0))
                delta = max(0.0, filled - prev_filled)
                order_fill_tracker[order_id] = max(prev_filled, filled)
            else:
                delta = max(0.0, filled)

            if delta > 0:
                total_filled += delta
                remaining_amount = max(0.0, requested_total - total_filled)
            return delta

        def refresh_order(order_obj):
            if not isinstance(order_obj, dict):
                return order_obj
            order_id = order_obj.get('id')
            if not order_id:
                return order_obj
            try:
                return self.exchange.fetch_order(order_id, symbol)
            except Exception:
                try:
                    open_orders = self.exchange.fetch_open_orders(symbol)
                    for o in open_orders:
                        if str(o.get('id')) == str(order_id):
                            return o
                except Exception:
                    pass
            return order_obj

        def wait_post_only_fill(order_obj, request_amount):
            wait_seconds = max(0.0, float(post_only_wait_seconds or 0.0))
            if wait_seconds <= 0:
                return order_obj
            poll_seconds = 0.2
            wait_deadline = time.time() + wait_seconds
            latest = order_obj
            while time.time() < wait_deadline and remaining_amount > 0:
                if deadline_reached():
                    break
                time.sleep(poll_seconds)
                latest = refresh_order(latest)
                register_fill(latest, request_amount)
                status = str(latest.get('status', '')).lower()
                if status in ('closed', 'filled', 'canceled', 'cancelled') or remaining_amount <= 0:
                    break
            return latest

        def build_partial_result(status='partial'):
            result = {
                'symbol': symbol,
                'side': side,
                'type': 'multi_tier',
                'status': status,
                'requested': requested_total,
                'filled': total_filled,
                'remaining': max(0.0, requested_total - total_filled),
                'reduceOnly': bool(reduce_only),
            }
            if isinstance(last_order, dict):
                result['last_order_id'] = last_order.get('id')
                result['average'] = last_order.get('average')
            return result

        high_volatility = False
        if atr and current_price:
            volatility_pct = (atr / current_price) * 100
            if volatility_pct > 1.0:
                high_volatility = True
                self.logger.info(f"🔥 High Volatility ({volatility_pct:.2f}%) -> Aggressive Mode")

        # Tier 1: Post-only limit
        if not high_volatility and remaining_amount > 0:
            max_requotes = max(0, int(post_only_requote_max or 0))
            for requote_idx in range(max_requotes + 1):
                if remaining_amount <= 0:
                    break
                if deadline_reached():
                    self.logger.warning("⏱️ Entry deadline reached during Tier 1. Stop escalating.")
                    break
                try:
                    self.rate_limiter.wait_if_needed()
                    best_bid, best_ask = get_best_price_fresh()

                    if side == 'buy':
                        target_price = round_to_tick(best_bid + tick_size, tick_size)
                    else:
                        target_price = round_to_tick(best_ask - tick_size, tick_size)

                    self.logger.info(
                        f"📦 Tier 1: Post-Only {side} @ {target_price} "
                        f"(requote {requote_idx + 1}/{max_requotes + 1})"
                    )

                    request_amount = remaining_amount
                    filled_before_round = total_filled
                    order = self.exchange.create_order(
                        symbol=symbol,
                        type='limit',
                        side=side,
                        amount=request_amount,
                        price=target_price,
                        params=build_params({'postOnly': True}),
                    )
                    register_fill(order, request_amount)
                    order = wait_post_only_fill(order, request_amount)
                    status = str(order.get('status', '')).lower()
                    filled_this_round = max(0.0, total_filled - filled_before_round)

                    if status in ('closed', 'filled') or remaining_amount <= 0:
                        self.logger.info(
                            f"Tier 1 Filled ({filled_this_round}/{request_amount}) @ "
                            f"{order.get('average', target_price)}"
                        )
                        return order

                    if filled_this_round > 0:
                        self.logger.warning(
                            f"⚠️ Tier 1 Partial Fill: {filled_this_round}/{request_amount}. "
                            f"Remaining {remaining_amount}"
                        )
                        if remaining_amount <= 0:
                            return order

                    if status in ('open', 'new'):
                        try:
                            self.exchange.cancel_order(order['id'], symbol)
                            if requote_idx < max_requotes and not deadline_reached():
                                self.logger.info("🔁 Tier 1 open order canceled. Requoting post-only.")
                                continue
                            self.logger.info("⚠️ Tier 1 open order canceled. Escalating to Tier 2.")
                        except Exception as cancel_err:
                            self.logger.warning(
                                f"🗑️ Tier 1 cancel failed: {cancel_err}. Forcing cancel_all reconciliation..."
                            )
                            try:
                                self.exchange.cancel_all_orders(symbol)
                                lingering = self.exchange.fetch_open_orders(symbol)
                                if lingering:
                                    self.logger.error(
                                        f"❌ Tier 1 cancel reconciliation failed: {len(lingering)} open orders remain. "
                                        "Abort escalation to prevent orphan orders."
                                    )
                                    return build_partial_result(status='cancel_failed_open')
                            except Exception as cleanup_err:
                                self.logger.error(
                                    f"❌ Tier 1 cancel reconciliation exception: {cleanup_err}. "
                                    "Abort escalation to prevent orphan orders."
                                )
                                return build_partial_result(status='cancel_failed_open')
                        break

                except Exception as e:
                    if 'timeout' in str(e).lower():
                        self.logger.warning("⚠️ Tier 1 Timeout -> Reconciling...")
                        try:
                            open_orders = self.exchange.fetch_open_orders(symbol)
                            if open_orders:
                                self.exchange.cancel_all_orders(symbol)
                                self.logger.info("🗑️ Zombie Order Canceled")
                                lingering = self.exchange.fetch_open_orders(symbol)
                                if lingering:
                                    self.logger.error(
                                        f"❌ Tier 1 timeout reconciliation incomplete: {len(lingering)} open orders remain."
                                    )
                                    return build_partial_result(status='timeout_open_orders')
                        except Exception:
                            pass
                    else:
                        self.logger.info(f"❌ Tier 1 Skipped ({e}) -> Tier 2")
                    break

        if remaining_amount > 0 and deadline_reached():
            self.logger.warning("⏱️ Entry deadline reached before Tier 2. Skip further escalation.")
            if total_filled > 0:
                return build_partial_result(status='deadline_partial')
            return None

        # Tier 2: IOC aggressive limit
        if remaining_amount > 0:
            try:
                self.rate_limiter.wait_if_needed()
                best_bid, best_ask = get_best_price_fresh()

                offset = 0.001 if high_volatility else SMART_ORDER_OFFSET
                if side == 'buy':
                    limit_price = round_to_tick(best_ask * (1 + offset), tick_size)
                else:
                    limit_price = round_to_tick(best_bid * (1 - offset), tick_size)

                self.logger.info(f"📦 Tier 2: IOC Limit {side} @ {limit_price}")

                request_amount = remaining_amount
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=side,
                    amount=request_amount,
                    price=limit_price,
                    params=build_params({'timeInForce': 'IOC'}),
                )
                filled = register_fill(order, request_amount)

                if filled > 0:
                    fill_ratio = filled / request_amount if request_amount > 0 else 0.0
                    if remaining_amount <= 0 or fill_ratio >= 0.999:
                        self.logger.info(f"Tier 2 Filled ({filled}/{request_amount}) @ {order.get('average', limit_price)}")
                        return order

                    self.logger.warning(
                        f"⚠️ Tier 2 Partial Fill: {filled}/{request_amount} "
                        f"({fill_ratio*100:.1f}%). Remaining {remaining_amount}"
                    )

            except Exception as e:
                self.logger.error(f"❌ Tier 2 Failed: {e}")
                if 'timeout' in str(e).lower():
                    try:
                        self.exchange.cancel_all_orders(symbol)
                    except Exception:
                        pass

        if remaining_amount > 0 and deadline_reached():
            self.logger.warning("⏱️ Entry deadline reached before Tier 3. Skip market fallback.")
            if total_filled > 0:
                return build_partial_result(status='deadline_partial')
            return None

        # Tier 3: market fallback for remaining qty
        try:
            if remaining_amount <= 0:
                return last_order if last_order else build_partial_result(status='filled')
            if not allow_market_fallback:
                self.logger.warning(
                    f"⚠️ Market fallback disabled for {symbol} {side}. Remaining unfilled: {remaining_amount}"
                )
                if total_filled > 0:
                    return build_partial_result(status='partial_no_market')
                return None
            self.rate_limiter.wait_if_needed()
            if current_price and atr:
                max_slip_ratio = min(
                    0.01, (float(atr) * 0.2) / float(current_price)
                )
                if side == "buy":
                    hard_limit_price = round_to_tick(
                        current_price * (1 + max_slip_ratio), tick_size
                    )
                else:
                    hard_limit_price = round_to_tick(
                        current_price * (1 - max_slip_ratio), tick_size
                    )
                self.logger.warning(
                    f"🚨 Tier 3: Capped Market Order (Limit IOC fallback) {side} @ {hard_limit_price}"
                )
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side,
                    amount=remaining_amount,
                    price=hard_limit_price,
                    params=build_params({"timeInForce": "IOC"}),
                )
            else:
                self.logger.warning(
                    f"🚨 Tier 3: Market Order {side} (remaining: {remaining_amount})"
                )
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=remaining_amount,
                    params=build_params(),
                )
            register_fill(order, remaining_amount)
            return order
        except Exception as e:
            self.logger.error(f"❌ All Tiers Failed: {e}")
            if total_filled > 0:
                self.logger.warning(
                    f"⚠️ Partial fill preserved despite final failure: "
                    f"{total_filled}/{requested_total} {symbol}"
                )
                return build_partial_result(status='partial_failed')
            return None


    def cancel_all_orders(self, symbol: str) -> None:
        """미체결 주문 취소 (예외를 상위로 전파)"""
        self.logger.info("🗑️ Canceling all open orders for %s", symbol)
        self.exchange.cancel_all_orders(symbol)


