"""Hermetic stubs for testnet drill unit tests."""

from __future__ import annotations

import json
import urllib.parse
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from src.live.rest import HttpResponse
from src.live.settings import ExecutionMode, LiveSettings
from src.live.testnet_drill import guard as guard_module
from src.live.testnet_drill.config import DrillConfig

ORDER_KEY = "order-key-abc"
MAIN_KEY = "main-key-xyz"
ORDER_CRED = "order-secret-abc"


def patch_drill_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the drill scratch root at an isolated directory."""
    root = tmp_path / "scratch" / "testnet_drill"
    monkeypatch.setattr(guard_module, "DRILL_SCRATCH_ROOT", root)
    return root


def make_settings(**overrides: Any) -> LiveSettings:
    """Hermetic LIVE_TESTNET settings bypassing env validation."""
    base: dict[str, Any] = {
        "mode": ExecutionMode.LIVE_TESTNET,
        "order_base_url": "https://testnet.binancefuture.com",
        "market_data_base_url": "https://testnet.binancefuture.com",
        "api_key": SecretStr(MAIN_KEY),
        "api_secret": SecretStr("main-secret"),
        "order_api_key": SecretStr(ORDER_KEY),
        "order_api_secret": SecretStr(ORDER_CRED),
    }
    base.update(overrides)
    return LiveSettings.model_construct(**base)


def make_config(**overrides: Any) -> DrillConfig:
    """Valid drill config with overridable fields."""
    base: dict[str, Any] = {
        "symbols": ("BTCUSDT",),
        "probe_notional_usdt": Decimal("10"),
        "resting_distance_bps": 50,
        "max_probe_notional_usdt": Decimal("20"),
        "max_open_orders_probe": 5,
        "scenario_timeout_s": 30.0,
        "recv_window_ms": 5000,
        "funding_soak_min_funding_events": 1,
        "qty_tolerance": Decimal("0"),
        "price_rel_tolerance": Decimal("0.001"),
        "fee_abs_tolerance_usdt": Decimal("0.01"),
    }
    base.update(overrides)
    return DrillConfig(**base)


def _json(payload: Any) -> HttpResponse:
    return HttpResponse(status_code=200, headers={}, body=json.dumps(payload).encode())


def symbol_entry(symbol: str, status: str = "TRADING") -> dict[str, Any]:
    """Minimal exchangeInfo entry accepted by parse_exchange_filters."""
    return {
        "symbol": symbol,
        "status": status,
        "contractType": "PERPETUAL",
        "quoteAsset": "USDT",
        "quantityPrecision": 3,
        "pricePrecision": 2,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
            {"filterType": "MIN_NOTIONAL", "notional": "5"},
        ],
    }


class ScriptedTransport:
    """Route-based stub transport recording every call."""

    def __init__(self, routes: Any = ()) -> None:
        self._routes = list(routes)
        self.calls: list[tuple[str, str, dict[str, str]]] = []
        self.closed = False

    def call(self, method: str, url: str, headers: Any) -> HttpResponse:
        self.calls.append((method, url, dict(headers)))
        parts = urllib.parse.urlsplit(url)
        query = dict(urllib.parse.parse_qsl(parts.query))
        for route_method, route_path, handler in self._routes:
            if route_method == method.upper() and route_path == parts.path:
                payload = handler(method, url, query, dict(headers))
                if isinstance(payload, HttpResponse):
                    return payload
                return _json(payload)
        raise AssertionError(f"unstubbed {method} {parts.path}")

    def close(self) -> None:
        self.closed = True


class StatefulVenue:
    """Scripted testnet venue with mutable positions and open orders."""

    def __init__(
        self,
        symbols: Any,
        *,
        positions: Any = None,
        orders: Any = None,
        wallet: str = "1000.0",
        auto_flatten: bool = True,
        auto_cancel: bool = True,
    ) -> None:
        self.symbols = list(symbols)
        self.positions: dict[str, Decimal] = {s: Decimal(str(q)) for s, q in dict(positions or {}).items()}
        self.orders: list[dict[str, Any]] = [dict(o) for o in (orders or [])]
        self.wallet = wallet
        self.auto_flatten = auto_flatten
        self.auto_cancel = auto_cancel
        self.calls: list[tuple[str, str, dict[str, str]]] = []
        self.closed = False
        self.failing = False
        self.malformed_account = False
        self.income: list[dict[str, str]] = []

    def call(self, method: str, url: str, headers: Any) -> HttpResponse:
        self.calls.append((method, url, dict(headers)))
        if self.failing:
            raise OSError("venue unreachable")
        parts = urllib.parse.urlsplit(url)
        path = parts.path
        query = dict(urllib.parse.parse_qsl(parts.query))
        if method == "GET" and path == "/fapi/v1/time":
            return _json({"serverTime": 1700000000000})
        if method == "GET" and path == "/fapi/v1/exchangeInfo":
            return _json({"symbols": [symbol_entry(s) for s in self.symbols]})
        if method == "GET" and path == "/fapi/v1/ticker/bookTicker":
            return _json({"bidPrice": "99", "askPrice": "100"})
        if method == "GET" and path == "/fapi/v1/income":
            return _json(self.income)
        if method == "GET" and path == "/fapi/v2/account":
            if self.malformed_account:
                return _json([])
            return _json(
                {
                    "totalWalletBalance": self.wallet,
                    "positions": [{"symbol": s, "leverage": "20", "marginType": "cross"} for s in self.symbols],
                }
            )
        if method == "GET" and path == "/fapi/v2/positionRisk":
            rows = [{"symbol": s, "positionAmt": str(q), "leverage": "20"} for s, q in self.positions.items() if q != 0]
            rows += [{"symbol": s, "positionAmt": "0"} for s in self.symbols if s not in self.positions]
            return _json(rows)
        if method == "GET" and path == "/fapi/v1/openOrders":
            return _json(self.orders)
        if method == "DELETE" and path == "/fapi/v1/allOpenOrders":
            if self.auto_cancel:
                self.orders = [o for o in self.orders if o.get("symbol") != query.get("symbol")]
            return _json({})
        if method == "POST" and path == "/fapi/v1/order":
            symbol = query.get("symbol", "")
            if query.get("reduceOnly") == "true" and self.auto_flatten:
                self.positions[symbol] = Decimal(0)
            return _json({"orderId": 1, "status": "NEW", "symbol": symbol})
        if method == "DELETE" and path == "/fapi/v1/order":
            self.orders = [o for o in self.orders if not isinstance(o, dict) or o.get("clientOrderId") != query.get("origClientOrderId")]
            return _json({"status": "CANCELED"})
        raise AssertionError(f"unstubbed {method} {path}")

    def close(self) -> None:
        self.closed = True
