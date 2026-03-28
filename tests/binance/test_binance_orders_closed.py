import ccxt
import os
import json
from config.settings import BINANCE_API_KEY, BINANCE_SECRET
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'options': {'defaultType': 'swap'},
})
orders = exchange.fetch_closed_orders('AVAX/USDT:USDT', limit=50)
for o in orders:
    if o.get("type") not in ("market", "limit"):
        print(o['id'])
        print("ccxt type:", o.get("type"))
        print("info type:", o.get("info", {}).get("type"))
        print("info origType:", o.get("info", {}).get("origType"))
        print("stopPrice:", o.get("stopPrice"))
        print("----")
