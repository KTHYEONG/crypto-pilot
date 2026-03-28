import ccxt
import os
import json
from config.settings import BINANCE_API_KEY, BINANCE_SECRET
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'options': {'defaultType': 'swap'},
})
orders = exchange.fetch_open_orders('AVAX/USDT:USDT')
for o in orders:
    print(o['id'])
    print("ccxt type:", o.get("type"))
    print("info type:", o.get("info", {}).get("type"))
    print("info origType:", o.get("info", {}).get("origType"))
    print("stopPrice:", o.get("stopPrice"))
    print("info raw:", json.dumps(o.get("info", {})))
    print("----")
