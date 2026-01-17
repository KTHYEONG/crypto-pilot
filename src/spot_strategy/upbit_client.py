import ccxt
import pandas as pd
import time
import logging
from datetime import datetime

class UpbitClient:
    def __init__(self, access_key=None, secret_key=None):
        self.logger = logging.getLogger("UpbitClient")
        
        try:
            self.exchange = ccxt.upbit({
                'apiKey': access_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'options': {
                    'createMarketBuyOrderRequiresPrice': False, # Upbit specific: Allows simplify market buy
                }
            })
            # Load markets to ensure symbols are available
            self.exchange.load_markets()
        except Exception as e:
            self.logger.error(f"Failed to initialize CCXT Upbit client: {e}")
            self.exchange = None

    def get_market_price(self, symbol):
        """현재 시장가 조회"""
        if not self.exchange: return None
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=200):
        """
        OHLCV 데이터 조회 (CCXT 기반)
        """
        if not self.exchange: return None
        try:
            # Timeframe standardization not strictly needed as CCXT handles '1m', '3m', '1d', etc. well.
            # Upbit supports: 1m, 3m, 5m, 10m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
            
            # Special handling for 2h -> Fetch 1h and resample needs higher level logic
            # Here we just pass through to CCXT. Caller should handle custom TFs.
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
             
            if not ohlcv:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Upbit CCXT returns valid OHLCV.
            return df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'timestamp']]

        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    def fetch_balance(self):
        """KRW 잔고 조회"""
        if not self.exchange: return 0.0, 0.0
        try:
            balance = self.exchange.fetch_balance()
            # Upbit KRW
            total_krw = balance['total'].get('KRW', 0.0)
            free_krw = balance['free'].get('KRW', 0.0)
            return total_krw, free_krw
        except Exception as e:
            self.logger.error(f"Error fetching KRW balance: {e}")
            return 0.0, 0.0

    def fetch_position(self, symbol):
        """
        현물 보유량 조회
        Symbol: 'KRW-BTC' -> checks 'BTC' balance
        """
        if not self.exchange: return {'amount': 0.0}
        try:
            currency = symbol.split('-')[1] # KRW-BTC -> BTC
            balance = self.exchange.fetch_balance()
            
            amount = balance['total'].get(currency, 0.0)
            # CCXT for Upbit might not provide average price directly in fetch_balance info depending on version
            # We might need private API call if needed, but for now returns basic info
            
            # To get avg buy price, we often need to look into 'info' part of balance or specific API
            # Upbit 'balance' info: [{'currency':'BTC', 'balance':'...', 'avg_buy_price':'...'}]
            avg_price = 0.0
            if 'info' in balance:
                # balance['info'] is the raw list from Upbit API
                for wallet in balance['info']:
                    if wallet['currency'] == currency:
                        avg_price = float(wallet.get('avg_buy_price', 0.0))
                        break
            
            return {
                'amount': amount,
                'entryPrice': avg_price,
                'unrealizedPnL': 0.0
            }
        except Exception as e:
            self.logger.error(f"Error fetching position for {symbol}: {e}")
            return {'amount': 0.0}

    def place_order(self, symbol, side, amount=None, price=None, order_type='market'):
        """
        주문 실행 (CCXT)
        Market Buy: price arg is Cost in KRW (Upbit special)
        Market Sell: amount arg is Volume
        """
        if not self.exchange: return None
        try:
            # CCXT Upbit implementation details:
            # create_order(symbol, type, side, amount, price)
            
            if order_type == 'market':
                if side == 'buy':
                    # For Upbit Market Buy, 'cost' (total quote currency) is passed.
                    # CCXT maps the 4th arg 'amount' to cost if type is market and side is buy (check ccxt docs/code)
                    # BUT Upbit is tricky. safest is using params={'cost': price} or similar depending on CCXT version.
                    # Looking at recent CCXT: for upbit, create_order(symbol, 'market', 'buy', cost) uses cost as price/total.
                    
                    # user passed 'price' as KRW amount.
                    # We pass 'price' as the 4th argument 'amount' in CCXT signature for Upbit Market Buy usually,
                    # OR we use 'cost' param.
                    
                    # Safer approach for Upbit Market Buy in CCXT:
                    # exchange.create_order(symbol, 'market', 'buy', amount, price)
                    # if amount is provided, it tries to buy that amount? No market buy is by cost usually.
                    
                    # Let's rely on 'price' being the COST (KRW).
                    # We pass it as 'amount' argument because CCXT upbit treats 1st numeric arg as cost for market buys often,
                    # OR we pass explicitly via params.
                    return self.exchange.create_order(symbol, 'market', 'buy', price, params={'ord_type': 'price'}) 
                    
                else:
                    return self.exchange.create_order(symbol, 'market', 'sell', amount)
            
            elif order_type == 'limit':
                return self.exchange.create_order(symbol, 'limit', side, amount, price)

        except Exception as e:
            self.logger.error(f"❌ Order Failed: {e}")
            return None

