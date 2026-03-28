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
    
    # 1. Test normal fetch_open_orders
    print("Fetching normal open orders...")
    orders = exchange.fetch_open_orders(symbol)
    print(f"  Found {len(orders)} normal orders.")
    
    # 2. Test algo open orders
    print("Fetching algo open orders...")
    try:
        # Implicit method for GET /fapi/v1/algo/openOrders
        algo_orders = exchange.fapiPrivateGetAlgoOpenOrders()
        print(f"  Found {len(algo_orders)} algo orders (total).")
        for o in algo_orders:
            if o['symbol'] == 'AVAXUSDT':
                print(f"  ALGO ORDER: {o['algoId']}, type: {o['orderType']}, status: {o['algoStatus']}")
    except Exception as e:
        print(f"  Algo fetch failed: {e}")
        
    # Cleanup
    # Note: To cancel an algo order, we might need a different endpoint?
    # DELETE /fapi/v1/algo/order
    try:
        exchange.fapiPrivateDeleteAlgoOrder({'algoId': order['info']['algoId']})
        print("Algo order cancelled successfully.")
    except Exception as e:
        print(f"Algo cancel failed: {e}")

except Exception as e:
    print(f"Error: {e}")
