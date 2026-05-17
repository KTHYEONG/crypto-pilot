import json
import ccxt
import pytest
from config.settings import BINANCE_API_KEY, BINANCE_SECRET

@pytest.mark.integration
def test_binance_algo_orders_fetch():
    """Tests fetching algo open orders from Binance."""
    if not BINANCE_API_KEY or not BINANCE_SECRET:
        pytest.skip("Binance API keys not found")
        
    exchange = ccxt.binanceusdm({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_SECRET,
        "options": {"defaultType": "future"},
    })
    
    try:
        # Implicit method for GET /fapi/v1/algo/openOrders
        algo_orders = exchange.fapiPrivateGetAlgoOpenOrders()
        assert isinstance(algo_orders, list)
        print(f"Found {len(algo_orders)} algo orders.")
    except Exception as e:
        pytest.fail(f"Algo fetch failed: {e}")

@pytest.mark.integration
def test_binance_fetch_open_orders():
    """Tests fetching open orders for a specific symbol."""
    if not BINANCE_API_KEY or not BINANCE_SECRET:
        pytest.skip("Binance API keys not found")
        
    exchange = ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_SECRET,
        "options": {"defaultType": "swap"},
    })
    symbol = "AVAX/USDT:USDT"
    try:
        orders = exchange.fetch_open_orders(symbol)
        assert isinstance(orders, list)
    except Exception as e:
        pytest.fail(f"Fetch open orders failed: {e}")

@pytest.mark.integration
def test_binance_fetch_closed_orders():
    """Tests fetching closed orders for a specific symbol."""
    if not BINANCE_API_KEY or not BINANCE_SECRET:
        pytest.skip("Binance API keys not found")
        
    exchange = ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_SECRET,
        "options": {"defaultType": "swap"},
    })
    symbol = "AVAX/USDT:USDT"
    try:
        orders = exchange.fetch_closed_orders(symbol, limit=10)
        assert isinstance(orders, list)
    except Exception as e:
        pytest.fail(f"Fetch closed orders failed: {e}")

def test_is_stop_loss_logic():
    """Tests the logic for identifying stop loss orders."""
    def _is_stop_loss_order(o: dict) -> bool:
        ccxt_type = str(o.get("type", "")).upper()
        raw_type = str(o.get("info", {}).get("type", "")).upper()
        is_stop = "STOP" in ccxt_type or "STOP" in raw_type
        is_take_profit = "TAKE_PROFIT" in ccxt_type or "TAKE_PROFIT" in raw_type
        return is_stop and not is_take_profit

    o1 = {"type": "market", "info": {"type": "STOP_MARKET"}}
    o2 = {"type": "market", "info": {"origType": "STOP_MARKET"}}
    o3 = {"type": "limit", "info": {"type": "STOP_MARKET"}}
    o4 = {"type": "stop", "info": {"type": "STOP"}}
    o5 = {"type": "market", "info": {"type": "TAKE_PROFIT_MARKET"}}

    assert _is_stop_loss_order(o1) is True
    assert _is_stop_loss_order(o2) is False # Based on the original script's logic which used o.get('info', {}).get('type')
    assert _is_stop_loss_order(o3) is True
    assert _is_stop_loss_order(o4) is True
    assert _is_stop_loss_order(o5) is False
