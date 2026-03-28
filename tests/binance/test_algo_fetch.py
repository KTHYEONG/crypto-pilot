import ccxt
import os
import json
from config.settings import BINANCE_API_KEY, BINANCE_SECRET

exchange = ccxt.binanceusdm({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'options': {'defaultType': 'future'},
})

symbol = 'AVAX/USDT'

try:
    ticker = exchange.fetch_ticker(symbol)
    current_price = ticker['last']
    stop_price = current_price * 1.5
    
    print(f"Placing SL order (STOP_MARKET) at {stop_price}...")
    order = exchange.create_order(
        symbol=symbol,
        type='STOP_MARKET',
        side='buy',
        amount=1.0,
        params={'stopPrice': exchange.price_to_precision(symbol, stop_price)}
    )
    
    print("Fetching ALGO open orders...")
    try:
        # Correct implicit method name
        res = exchange.fapiPrivateGetOpenAlgoOrders()
        # The response is in 'orders' field for this endpoint usually, or just a list
        print("ALGO RESPONSE TYPE:", type(res))
        if isinstance(res, dict):
            print("KEYS:", list(res.keys()))
            algo_orders = res.get('orders', [])
        else:
            algo_orders = res
            
        print(f"Found {len(algo_orders)} algo orders.")
        for o in algo_orders:
            if o.get('symbol') == 'AVAXUSDT':
                print(f"  MATCH: {o.get('algoId')} ({o.get('orderType')})")
    except Exception as e:
        print(f"  Algo fetch failed: {e}")
        
    # Cleanup
    try:
        exchange.fapiPrivateDeleteAlgoOrder({'algoId': order['info']['algoId']})
        print("Algo order cancelled.")
    except:
        pass

except Exception as e:
    print(f"Error: {e}")
