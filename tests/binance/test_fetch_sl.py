import ccxt
import json
from config.settings import BINANCE_API_KEY, BINANCE_SECRET
exchange = ccxt.binanceusdm({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'options': {'defaultType': 'future'},
})
orders = exchange.fetch_open_orders('AVAX/USDT')
print(json.dumps(orders, indent=2))
