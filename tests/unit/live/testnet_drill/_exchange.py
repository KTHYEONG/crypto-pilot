"""Small two-account futures venue at the HTTP boundary for drill integration tests."""

from __future__ import annotations

import json
import time
from decimal import Decimal
from threading import RLock
from urllib.parse import parse_qsl, urlsplit

from src.live.rest import HttpResponse
from tests.unit.live.testnet_drill._stubs import symbol_entry


class Exchange:
    def __init__(self):
        self.lock = RLock()
        self.accounts = {}
        self.orders = []
        self.calls = []
        self.close_count = 0
        self.uid = 0
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.funding_interval_ms = None
        self.funding_rate = Decimal("0.0001")
        self.next_funding_ms = None
        self.funding_rows = []

    def account(self, key):
        return self.accounts.setdefault(key, {"positions": {}, "entry": {}, "wallet": Decimal("1000"),
            "trades": [], "income": [], "dualSidePosition": False, "multiAssetsMargin": False,
            "margin": "cross", "leverage": 20})

    def response(self, payload, status=200):
        used = str(min(len(self.calls), 50))
        return HttpResponse(status, {"X-MBX-USED-WEIGHT-1M": used}, json.dumps(payload, default=str).encode())

    def error(self, code):
        return self.response({"code": code, "msg": "scripted rejection"}, 400)

    def quote(self, symbol):
        bids = [Decimal("99.95")]
        asks = [Decimal("100.05")]
        for row in self.orders:
            if row["symbol"] == symbol and row["status"] in ("NEW", "PARTIALLY_FILLED"):
                (bids if row["side"] == "BUY" else asks).append(Decimal(row["price"]))
        return {"symbol": symbol, "bidPrice": str(max(bids)), "askPrice": str(min(asks)), "bidQty": "1", "askQty": "1"}

    def fill(self, row, amount, price, maker):
        account = self.account(row["key"])
        symbol = row["symbol"]
        position = account["positions"].get(symbol, Decimal(0))
        signed = amount if row["side"] == "BUY" else -amount
        entry = account["entry"].get(symbol, Decimal(0))
        pnl = Decimal(0)
        if position * signed < 0:
            pnl = min(abs(position), amount) * (price - entry) * (1 if position > 0 else -1)
        after = position + signed
        if position * signed > 0:
            entry = (entry * abs(position) + price * amount) / abs(after)
        elif position == 0 or position * after < 0:
            entry = price
        account["entry"][symbol] = entry if after else Decimal(0)
        account["positions"][symbol] = after
        commission = amount * price * Decimal("0.0005")
        account["wallet"] += pnl - commission
        executed = Decimal(row["executedQty"])
        row["avgPrice"] = str((Decimal(row["avgPrice"]) * executed + amount * price) / (executed + amount))
        row["executedQty"] = str(executed + amount)
        row["status"] = "FILLED" if executed + amount >= Decimal(row["origQty"]) else "PARTIALLY_FILLED"
        row["updateTime"] = int(time.time() * 1000)
        self.uid += 1
        account["trades"].append({"id": self.uid, "orderId": row["orderId"], "symbol": symbol, "time": row["updateTime"],
            "qty": str(amount), "price": str(price), "quoteQty": str(amount * price), "side": row["side"],
            "positionSide": "BOTH", "buyer": row["side"] == "BUY", "realizedPnl": str(pnl),
            "commission": str(commission), "commissionAsset": "USDT", "maker": maker})
        account["income"].append({"tranId": self.uid, "incomeType": "COMMISSION", "asset": "USDT", "income": str(-commission), "time": row["updateTime"], "symbol": symbol})
        if pnl:
            account["income"].append({"tranId": self.uid, "incomeType": "REALIZED_PNL", "asset": "USDT", "income": str(pnl), "time": row["updateTime"], "symbol": symbol})

    def settle_funding(self):
        """Venue funding: positive rate means longs pay shorts, booked as FUNDING_FEE income."""
        now_ms = int(time.time() * 1000)
        if self.funding_interval_ms is None:
            return
        if self.next_funding_ms is None:
            self.next_funding_ms = now_ms + self.funding_interval_ms
        while now_ms >= self.next_funding_ms:
            at = self.next_funding_ms
            for symbol in self.symbols:
                mark = Decimal("100")
                self.funding_rows.append({"symbol": symbol, "fundingTime": at, "fundingRate": str(self.funding_rate),
                                          "markPrice": str(mark)})
                for account in self.accounts.values():
                    position = account["positions"].get(symbol, Decimal(0))
                    if position:
                        amount = -position * mark * self.funding_rate
                        account["wallet"] += amount
                        self.uid += 1
                        account["income"].append({"tranId": self.uid, "incomeType": "FUNDING_FEE", "asset": "USDT",
                                                  "income": str(amount), "time": at, "symbol": symbol})
            self.next_funding_ms += self.funding_interval_ms

    def public(self, row):
        return {key: value for key, value in row.items() if key != "key"}

    def call(self, method, url, headers):
        with self.lock:
            path = urlsplit(url).path
            params = dict(parse_qsl(urlsplit(url).query))
            key = headers.get("X-MBX-APIKEY", "")
            self.calls.append((key, method, path, params))
            self.settle_funding()
            account = self.account(key)
            if path == "/fapi/v1/time":
                return self.response({"serverTime": int(time.time() * 1000)})
            if path == "/fapi/v1/exchangeInfo":
                return self.response({"symbols": [symbol_entry(symbol) for symbol in self.symbols], "rateLimits": [
                    {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
                    {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
                    {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300}]})
            if "timestamp" in params:
                delta = int(params["timestamp"]) - int(time.time() * 1000)
                if delta > 1000 or delta < -int(params["recvWindow"]):
                    return self.error(-1021)
            if path == "/fapi/v1/ticker/bookTicker":
                return self.response(self.quote(params["symbol"]) if "symbol" in params else [self.quote(s) for s in self.symbols])
            if path == "/fapi/v1/depth":
                quote = self.quote(params["symbol"])
                return self.response({"bids": [[quote["bidPrice"], "0.3"]], "asks": [[quote["askPrice"], "0.3"]]})
            if path == "/fapi/v2/positionRisk":
                return self.response([{"symbol": s, "positionAmt": str(account["positions"].get(s, 0)),
                    "marginType": account["margin"], "leverage": str(account["leverage"])} for s in self.symbols])
            if path == "/fapi/v2/account":
                return self.response({"totalWalletBalance": str(account["wallet"]), "availableBalance": str(account["wallet"]),
                    "totalInitialMargin": "0", "totalMaintMargin": "0", "totalUnrealizedProfit": "0",
                    "dualSidePosition": account["dualSidePosition"], "multiAssetsMargin": account["multiAssetsMargin"],
                    "positions": [{"symbol": s, "leverage": str(account["leverage"]), "marginType": account["margin"],
                                   "positionAmt": str(account["positions"].get(s, 0))} for s in self.symbols]})
            if path in ("/fapi/v1/positionSide/dual", "/fapi/v1/multiAssetsMargin"):
                name = "dualSidePosition" if path.endswith("dual") else "multiAssetsMargin"
                if method == "POST":
                    account[name] = params[name] == "true"
                return self.response({name: account[name]})
            if path == "/fapi/v1/leverageBracket":
                return self.response([{"symbol": s, "brackets": [{"bracket": 1, "initialLeverage": 20, "notionalCap": "100000", "notionalFloor": "0"}]} for s in self.symbols])
            if path == "/fapi/v1/leverage":
                if int(params["leverage"]) > 20:
                    return self.error(-4028)
                account["leverage"] = int(params["leverage"])
                return self.response({"leverage": account["leverage"]})
            if path == "/fapi/v1/marginType":
                if any(account["positions"].values()):
                    return self.error(-4048)
                account["margin"] = params["marginType"].lower()
                return self.response({"code": 0})
            if path == "/fapi/v1/userTrades":
                rows = [row for row in account["trades"] if row["symbol"] == params["symbol"]]
                rows = [row for row in rows if row["id"] >= int(params.get("fromId", "0")) and int(params.get("startTime", "0")) <= row["time"] <= int(params.get("endTime", "9999999999999"))]
                return self.response(rows[:1000])
            if path == "/fapi/v1/fundingRate":
                return self.response([r for r in self.funding_rows if r["symbol"] == params["symbol"]
                                      and r["fundingTime"] >= int(params.get("startTime", "0"))])
            if path == "/fapi/v1/income":
                return self.response([row for row in account["income"] if int(params.get("startTime", "0")) <= row["time"] <= int(params.get("endTime", "9999999999999"))])
            if path == "/fapi/v1/openOrders":
                return self.response([self.public(row) for row in self.orders if row["key"] == key and row["status"] in ("NEW", "PARTIALLY_FILLED")])
            if path == "/fapi/v1/allOpenOrders":
                for row in self.orders:
                    if row["key"] == key and row["symbol"] == params["symbol"] and row["status"] in ("NEW", "PARTIALLY_FILLED"):
                        row["status"] = "CANCELED"
                return self.response({"code": 0})
            if path != "/fapi/v1/order":
                raise AssertionError((method, path))
            if method != "POST":
                row = next((row for row in reversed(self.orders) if row["key"] == key and row["clientOrderId"] == params["origClientOrderId"]), None)
                if row is None:
                    return self.error(-2013)
                if method == "DELETE":
                    if row["status"] == "FILLED":
                        return self.error(-2011)
                    row["status"] = "CANCELED"
                return self.response(self.public(row))
            amount = Decimal(params["quantity"])
            if amount % Decimal("0.001"):
                return self.error(-1111)
            if amount > 1000:
                return self.error(-4005)
            if amount < Decimal("0.001"):
                return self.error(-4004)
            if any(row["key"] == key and row["clientOrderId"] == params["newClientOrderId"] and row["status"] in ("NEW", "PARTIALLY_FILLED") for row in self.orders):
                return self.error(-4116)
            position = account["positions"].get(params["symbol"], Decimal(0))
            if params.get("reduceOnly") == "true" and (not position or (position > 0) == (params["side"] == "BUY")):
                return self.error(-2022)
            quote = self.quote(params["symbol"])
            limit = Decimal(params.get("price", quote["askPrice"] if params["side"] == "BUY" else quote["bidPrice"]))
            if params.get("timeInForce") == "GTX" and ((params["side"] == "BUY" and limit >= Decimal(quote["askPrice"])) or (params["side"] == "SELL" and limit <= Decimal(quote["bidPrice"]))):
                return self.error(-5022)
            if amount * limit < 5 and params.get("reduceOnly") != "true":
                return self.error(-4164)
            self.uid += 1
            row = {"key": key, "symbol": params["symbol"], "side": params["side"], "clientOrderId": params["newClientOrderId"],
                "orderId": self.uid, "origQty": str(amount), "executedQty": "0", "avgPrice": "0", "price": str(limit), "status": "NEW", "updateTime": int(time.time() * 1000)}
            self.orders.append(row)
            active = params["type"] == "MARKET" or params.get("timeInForce") in ("IOC", "GTC")
            remaining = min(amount, abs(position)) if params.get("reduceOnly") == "true" else amount
            if active:
                for maker in list(self.orders):
                    if maker is row or maker["key"] == key or maker["symbol"] != row["symbol"] or maker["side"] == row["side"] or maker["status"] not in ("NEW", "PARTIALLY_FILLED"):
                        continue
                    maker_price = Decimal(maker["price"])
                    if params["type"] != "MARKET" and ((row["side"] == "BUY" and limit < maker_price) or (row["side"] == "SELL" and limit > maker_price)):
                        continue
                    taken = min(remaining, Decimal(maker["origQty"]) - Decimal(maker["executedQty"]))
                    if taken > 0:
                        self.fill(maker, taken, maker_price, True)
                        self.fill(row, taken, maker_price, False)
                        remaining -= taken
                public_price = Decimal("100.05") if row["side"] == "BUY" else Decimal("99.95")
                if remaining > 0 and (params["type"] == "MARKET" or (row["side"] == "BUY" and limit >= public_price) or (row["side"] == "SELL" and limit <= public_price)):
                    self.fill(row, remaining, public_price, False)
                if params.get("timeInForce") == "IOC" and row["status"] != "FILLED":
                    row["status"] = "EXPIRED"
            return self.response(self.public(row))

    def close(self):
        self.close_count += 1
