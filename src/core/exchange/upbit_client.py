from __future__ import annotations

import logging
import random
import time
from typing import Any, cast

import ccxt
import pandas as pd

from config.settings import UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY

logger = logging.getLogger(__name__)


class UpbitOhlcvFetchError(Exception):
    """Exception raised when fetching OHLCV data from Upbit fails."""

    def __init__(
        self,
        message: str,
        partial_ohlcv: list[list[float]] | None = None,
        since_ms: int | None = None,
    ) -> None:
        """Initialize UpbitOhlcvFetchError."""
        super().__init__(message)
        self.partial_ohlcv = partial_ohlcv
        self.since_ms = since_ms


class UpbitClient:
    """API client for Upbit exchange."""

    def __init__(self, api_key: str | None = None, secret_key: str | None = None) -> None:
        """Initialize UpbitClient."""
        self.api_key = api_key or UPBIT_ACCESS_KEY
        self.secret_key = secret_key or UPBIT_SECRET_KEY
        self.logger = logging.getLogger(self.__class__.__name__)

        self.exchange = ccxt.upbit(
            {
                "apiKey": self.api_key,
                "secret": self.secret_key,
                "enableRateLimit": True,
                "options": {"fetchOHLCVWarning": False},
            }
        )
        self._markets_loaded = False

    def _ensure_markets_loaded(self) -> None:
        """Load market metadata if not already loaded."""
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    def _normalize_symbol(self, symbol: str) -> str:
        """Standardizes symbol format for Upbit (e.g., BTC/KRW)."""
        s = symbol.upper().replace("-", "/")
        if "/" not in s:
            s = f"{s}/KRW"
        return s

    def fetch_ohlcv_all(
        self,
        symbol: str,
        timeframe: str,
        start_timestamp: int,
        end_timestamp: int | None = None,
        max_retries: int = 10,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a given range (with pagination)."""
        self._ensure_markets_loaded()
        ccxt_symbol = self._normalize_symbol(symbol)
        if end_timestamp is None:
            end_timestamp = int(time.time() * 1000)

        all_ohlcv: list[list[float]] = []
        since = start_timestamp
        consecutive_failures = 0

        while since < end_timestamp:
            try:
                ohlcv = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=since, limit=200)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                last_timestamp = int(ohlcv[-1][0])
                consecutive_failures = 0
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
                sleep_sec = min(float(2**attempt), 30.0) + random.uniform(0.0, 1.0)  # noqa: S311
                time.sleep(sleep_sec)

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df[df["timestamp"] <= end_timestamp]
        return df

    def fetch_recent_ohlcv(self, symbol: str, timeframe: str, limit: int = 3) -> pd.DataFrame:
        """Fetch small recent OHLCV window for live signal timing."""
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            rows = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)
            if not rows:
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"]
                )
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching recent OHLCV from Upbit for {symbol}: {e}")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"]
            )

    def get_market_price(self, symbol: str) -> float | None:
        """Fetch current market price from Upbit."""
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
            val = ticker.get("last")
            return float(val) if val is not None else None
        except Exception as e:
            self.logger.error(f"Error fetching Upbit ticker for {symbol}: {e}")
            return None

    def fetch_balance_dict(self) -> dict[str, Any]:
        """Return the full CCXT balance dictionary."""
        try:
            res = self.exchange.fetch_balance()
            return cast("dict[str, Any]", res)
        except Exception as e:
            self.logger.error(f"Error fetching Upbit balance dict: {e}")
            return {}

    def fetch_server_time_ms(self) -> int:
        """Fetch exchange server time or fallback to local time."""
        try:
            server_ms = self.exchange.fetch_time()
            if server_ms is not None:
                return int(server_ms)
        except Exception:
            ...  # ccxt Upbit does not support fetchTime(); local time used as fallback
        return int(time.time() * 1000)

    def fetch_balance(self) -> tuple[float, float]:
        """Fetch KRW balance from Upbit."""
        try:
            balance = self.exchange.fetch_balance()
            total_krw = float(balance["total"].get("KRW", 0.0))
            free_krw = float(balance["free"].get("KRW", 0.0))
            return total_krw, free_krw
        except Exception as e:
            self.logger.error(f"Error fetching Upbit balance: {e}")
            raise RuntimeError("fetch_balance_failed") from e

    def fetch_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch open orders for a given symbol."""
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            res = self.exchange.fetch_open_orders(ccxt_symbol)
            return cast("list[dict[str, Any]]", res)
        except Exception as e:
            self.logger.error(f"Error fetching Upbit open orders for {symbol}: {e}")
            return []

    def place_market_buy_order(self, symbol: str, krw_amount: float) -> dict[str, Any] | None:
        """Execute a market-buy on Upbit."""
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            order = self.exchange.create_market_buy_order(ccxt_symbol, krw_amount)
            self.logger.info(f"🟢 Upbit Buy Order Placed: {symbol} for {krw_amount:,.0f} KRW")
            return cast("dict[str, Any]", order)
        except Exception as e:
            self.logger.error(f"❌ Upbit Buy Order Failed for {symbol}: {e}")
            return None

    def place_market_sell_order(self, symbol: str, amount: float) -> dict[str, Any] | None:
        """Execute a market-sell on Upbit."""
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            order = self.exchange.create_market_sell_order(ccxt_symbol, amount)
            self.logger.info(f"🔴 Upbit Sell Order Placed: {symbol} ({amount:.8f} units)")
            return cast("dict[str, Any]", order)
        except Exception as e:
            self.logger.error(f"❌ Upbit Sell Order Failed for {symbol}: {e}")
            return None

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Execute a general order on Upbit."""
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            order_type = "market" if price is None else "limit"
            order = self.exchange.create_order(ccxt_symbol, order_type, side, amount, price, params)
            self.logger.info(
                f"✅ Upbit Order Placed: {side} {symbol} ({amount:.8f} @ {price or 'Market'})"
            )
            return cast("dict[str, Any]", order)
        except Exception as e:
            self.logger.error(f"❌ Upbit Order Failed for {symbol}: {e}")
            return None

    def cancel_all_orders(self, symbol: str) -> None:
        """Cancel all open orders for a given symbol on Upbit."""
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            open_orders = self.fetch_open_orders(ccxt_symbol)
            for order in open_orders:
                self.exchange.cancel_order(str(order["id"]), ccxt_symbol)
            self.logger.info(f"🗑️ Canceled all open orders for {symbol} on Upbit")
        except Exception as e:
            self.logger.error(f"Error canceling Upbit orders for {symbol}: {e}")
