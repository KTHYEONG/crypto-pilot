from __future__ import annotations

import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import ccxt
import pandas as pd

# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except IndexError:
    pass

from config.settings import API_READ_TIMEOUT, API_ORDER_TIMEOUT, API_CHECK_TIMEOUT


class UpbitOhlcvFetchError(RuntimeError):
    """Raised when OHLCV pagination fails after bounded retries; carries partial rows for recovery."""

    def __init__(
        self,
        message: str,
        *,
        partial_ohlcv: list[list[float]] | None = None,
        since_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_ohlcv: list[list[float]] = partial_ohlcv or []
        self.since_ms: int | None = since_ms


class UpbitClient:
    """
    Upbit Spot Client based on CCXT.
    Mimics BinanceClient structure for compatibility with the optimization pipeline.
    """
    def __init__(self, api_key=None, secret=None):
        self.exchange = ccxt.upbit({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'timeout': API_READ_TIMEOUT * 1000
        })
        from src.common.utils import setup_logger
        self.logger = setup_logger("UpbitClient")
        self._ensure_markets_loaded()

    def _ensure_markets_loaded(self):
        try:
            if not getattr(self.exchange, 'markets', None):
                self.exchange.load_markets()
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load markets metadata: {e}")

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert Upbit format (KRW-ETH) to CCXT format (ETH/KRW)."""
        if '-' in symbol:
            parts = symbol.split('-')
            if parts[0] == 'KRW':
                return f"{parts[1]}/{parts[0]}"
        return symbol

    def get_symbol_constraints(self, symbol):
        self._ensure_markets_loaded()
        constraints = {
            'min_amount': 0.0,
            'min_cost': 0.0,
            'tick_size': 0.0,
        }

        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            market = self.exchange.market(ccxt_symbol)
        except Exception:
            return constraints

        limits = market.get('limits', {}) or {}
        amount_limits = limits.get('amount', {}) or {}
        cost_limits = limits.get('cost', {}) or {}
        constraints['min_amount'] = float(amount_limits.get('min') or 0.0)
        constraints['min_cost'] = float(cost_limits.get('min') or 0.0)
        constraints['tick_size'] = market.get('precision', {}).get('price', 0.0)

        return constraints

    def get_price_tick_size(self, symbol, fallback=1.0):
        constraints = self.get_symbol_constraints(symbol)
        tick = float(constraints.get('tick_size') or 0.0)
        return tick if tick > 0 else float(fallback)

    def round_price(self, symbol, price):
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            rounded = self.exchange.price_to_precision(ccxt_symbol, price)
            return float(rounded)
        except Exception:
            tick = self.get_price_tick_size(symbol, fallback=1.0)
            return round(float(price) / tick) * tick

    def round_amount(self, symbol, amount):
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            rounded = self.exchange.amount_to_precision(ccxt_symbol, amount)
            return float(rounded)
        except Exception:
            return float(amount)

    def fetch_ohlcv(
        self,
        symbol,
        timeframe,
        start_date,
        end_date=None,
        *,
        max_consecutive_retries: int = 10,
    ):
        limit = 200
        ccxt_symbol = self._normalize_symbol(symbol)
        
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
        self.logger.info(f"Fetching {ccxt_symbol} {timeframe} from Upbit...")

        max_retries = max(1, int(max_consecutive_retries))
        consecutive_failures = 0

        while since < end_timestamp:
            try:
                ohlcv = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, since, limit)
                consecutive_failures = 0
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                last_timestamp = ohlcv[-1][0]
                if last_timestamp <= since:
                    break
                since = last_timestamp + 1
                if last_timestamp >= end_timestamp:
                    break
                time.sleep(0.2)
            except Exception as e:
                consecutive_failures += 1
                self.logger.error("Error fetching data from Upbit: %s", e)
                if consecutive_failures >= max_retries:
                    raise UpbitOhlcvFetchError(
                        f"upbit_fetch_ohlcv_max_retries_exceeded after {max_retries} attempts: {e}",
                        partial_ohlcv=all_ohlcv,
                        since_ms=since,
                    ) from e
                attempt = consecutive_failures - 1
                sleep_sec = min(float(2**attempt), 30.0) + random.uniform(0.0, 1.0)
                time.sleep(sleep_sec)

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        df = df[df['timestamp'] <= end_timestamp]
        return df

    def fetch_recent_ohlcv(self, symbol, timeframe, limit=3):
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            rows = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)
            if not rows:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'datetime'])
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.logger.error(f"Error fetching recent OHLCV from Upbit for {symbol}: {e}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'datetime'])

    def get_market_price(self, symbol):
        try:
            self._ensure_markets_loaded()
            ccxt_symbol = self._normalize_symbol(symbol)
            
            # Check if symbol exists in loaded markets
            if ccxt_symbol not in self.exchange.markets:
                self.logger.warning(f"⚠️ Symbol {ccxt_symbol} not found in Upbit markets.")
                return None
                
            ticker = self.exchange.fetch_ticker(ccxt_symbol)
            if ticker is None or not isinstance(ticker, dict):
                self.logger.warning(f"⚠️ fetch_ticker returned invalid data for {ccxt_symbol}")
                return None
            return ticker.get('last')
        except Exception as e:
            self.logger.error(f"Error fetching Upbit ticker for {symbol}: {e}")
            return None

    def fetch_balance_dict(self):
        """Returns the full CCXT balance dictionary."""
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            self.logger.error(f"Error fetching Upbit balance dict: {e}")
            return {}

    def fetch_server_time_ms(self):
        try:
            server_ms = self.exchange.fetch_time()
            if server_ms is not None:
                return int(server_ms)
        except Exception as e:
            self.logger.warning(f"Failed to fetch Upbit server time: {e}")
        return int(time.time() * 1000)

    def fetch_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            return balance['total'].get('KRW', 0.0), balance['free'].get('KRW', 0.0)
        except Exception as e:
            self.logger.error(f"Error fetching Upbit balance: {e}")
            raise RuntimeError("fetch_balance_failed") from e

    def fetch_open_orders(self, symbol):
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            return self.exchange.fetch_open_orders(ccxt_symbol)
        except Exception as e:
            self.logger.error(f"Error fetching Upbit open orders for {symbol}: {e}")
            raise RuntimeError(f"fetch_open_orders_failed:{symbol}") from e

    def place_order(self, symbol, side, amount, order_type='market', price=None, params=None):
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            order = self.exchange.create_order(
                symbol=ccxt_symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=(params or {})
            )
            self.logger.info(f"⚡ Upbit Order Placed: {order_type} {side} {amount} {ccxt_symbol}")
            return order
        except Exception as e:
            self.logger.error(f"❌ Upbit Order Failed for {symbol}: {e}")
            return None

    def cancel_all_orders(self, symbol):
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            open_orders = self.fetch_open_orders(ccxt_symbol)
            for order in open_orders:
                self.exchange.cancel_order(order['id'], ccxt_symbol)
            self.logger.info(f"🗑️ Canceled all open orders for {symbol} on Upbit")
        except Exception as e:
            self.logger.error(f"Error canceling Upbit orders for {symbol}: {e}")
