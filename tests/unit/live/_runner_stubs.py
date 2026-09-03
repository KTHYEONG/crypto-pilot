# ruff: noqa
"""Shared stubs for live runner tests."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

DECISION_TIME = pd.Timestamp("2026-08-24 00:00Z")
NOW = DECISION_TIME + pd.Timedelta(hours=2)


class StubMarketClient:
    def exchange_info(self) -> dict[str, Any]:
        def symbol_entry(symbol: str) -> dict[str, Any]:
            return {
                "symbol": symbol,
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "quantityPrecision": 3,
                "pricePrecision": 2,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100000"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "1"},
                ],
            }

        return {
            "symbols": [symbol_entry("AAAUSDT"), symbol_entry("BUSDT")],
            "rateLimits": [
                {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
                {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
                {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300},
            ],
        }

    def book_ticker(self, symbol: str) -> dict[str, str]:
        return {"bidPrice": "100.00", "askPrice": "101.00"}


class StubOrderClient:
    def request(self, method: str, path: str, params=None, *, signed=False) -> Any:
        if path == "/fapi/v2/account":
            return {
                "totalWalletBalance": "2000",
                "availableBalance": "1900",
                "totalInitialMargin": "10",
                "totalUnrealizedProfit": "0",
                "dualSidePosition": "false",
                "multiAssetsMargin": "false",
            }
        if path == "/fapi/v2/positionRisk":
            return []
        raise AssertionError(f"unexpected path {path}")

    def sync_server_time(self) -> None:
        return None

    def open_orders(self) -> list[dict[str, Any]]:
        return []
