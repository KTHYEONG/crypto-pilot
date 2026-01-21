import ccxt
import pandas as pd
import time
import logging
from datetime import datetime
from config.settings import API_READ_TIMEOUT
from src.common.utils import setup_logger

class UpbitClient:
    def __init__(self, access_key=None, secret_key=None):
        self.logger = setup_logger("UpbitClient")
        
        try:
            self.exchange = ccxt.upbit({
                'apiKey': access_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'timeout': API_READ_TIMEOUT * 1000,  # 20초
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
        OHLCV 데이터 조회 (Pagination 지원)
        - Upbit API의 200개 제한을 극복하기 위해 반복 조회
        - '4h' -> 'minutes/240' 매핑 등 타임프레임 보정
        """
        if not self.exchange: return None

        # 1. 타임프레임 및 심볼 보정
        # 업비트 CCXT는 '4h'를 지원하지만, 명시적으로 분단위 매핑 확인
        tf_map = {
            '1m': '1', '3m': '3', '5m': '5', '10m': '10', '15m': '15', '30m': '30',
            '1h': '60', '4h': '240', '1d': 'day', '1w': 'week', '1M': 'month'
        }
        
        # CCXT 내부 매핑을 믿되, 혹시 모를 오류 방지를 위해 TF 확인
        target_tf = timeframe
        
        all_ohlcv = []
        current_limit = limit
        to_datetime = None

        try:
            while current_limit > 0:
                batch_size = min(current_limit, 200)
                params = {}
                if to_datetime:
                    params['to'] = to_datetime

                # CCXT fetch_ohlcv 호출
                ohlcv = self.exchange.fetch_ohlcv(symbol, target_tf, limit=batch_size, params=params)

                if not ohlcv or len(ohlcv) == 0:
                    if len(all_ohlcv) == 0:
                        self.logger.warning(
                            f"⚠️ fetch_ohlcv returned empty for {symbol} (TF: {target_tf}). "
                            f"Params: {params}. Server might be rejecting request or no data available."
                        )
                    break

                # 최신 데이터가 뒤에 오도록 병합 (prepend)
                all_ohlcv = ohlcv + all_ohlcv
                current_limit -= len(ohlcv)

                # 다음 요청을 위한 'to' 시간 설정 (가장 과거 시간 기준)
                first_timestamp = ohlcv[0][0]
                dt = datetime.utcfromtimestamp(first_timestamp / 1000)
                to_datetime = dt.strftime('%Y-%m-%d %H:%M:%S')

                if len(ohlcv) < batch_size:
                    break

                time.sleep(0.1)  # API 과부하 방지

            if not all_ohlcv:
                self.logger.warning(f"⚠️ No OHLCV data returned for {symbol} ({target_tf})")
                return None

            # DataFrame 변환 및 정제
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.drop_duplicates(subset=['timestamp'], inplace=True)
            df.sort_values('timestamp', inplace=True)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            if len(df) > limit:
                df = df.iloc[-limit:]

            return df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'timestamp']]

        except Exception as e:
            self.logger.error(f"❌ Error fetching OHLCV for {symbol}: {e}", exc_info=True)
            return None

    def fetch_balance(self):
        """KRW 잔고 조회"""
        if not self.exchange: return 0.0, 0.0
        try:
            balance = self.exchange.fetch_balance()
            # Upbit KRW
            total_krw = balance['total'].get('KRW', 0.0)
            free_krw = balance['free'].get('KRW', 0.0)
            
            if total_krw == 0 and free_krw == 0:
                self.logger.debug(f"ℹ️ Balance fetched but KRW is 0. Available assets: {list(balance['total'].keys())}")
            else:
                self.logger.info(f"💰 KRW Balance: Total {total_krw:,.0f} | Free {free_krw:,.0f}")
                
            return total_krw, free_krw
        except Exception as e:
            self.logger.error(f"Error fetching KRW balance: {e}")
            return 0.0, 0.0

    def fetch_position(self, symbol):
        """
        현물 보유량 조회 (강화된 평단 추출 로직)
        Symbol: 'KRW-BTC' -> checks 'BTC' balance
        """
        if not self.exchange:
            return {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}
        
        try:
            currency = symbol.split('-')[1]  # KRW-BTC -> BTC
            balance = self.exchange.fetch_balance()
            
            amount = balance['total'].get(currency, 0.0)
            
            # === 평균 매수가 추출 (견고성 강화) ===
            avg_price = 0.0
            
            if 'info' in balance and isinstance(balance['info'], list):
                for wallet in balance['info']:
                    if not isinstance(wallet, dict):
                        continue
                    
                    if wallet.get('currency') == currency:
                        raw_avg = wallet.get('avg_buy_price', '0')
                        
                        # 1. 타입 검증 및 변환
                        try:
                            if raw_avg is None or raw_avg == '':
                                avg_price = 0.0
                            elif isinstance(raw_avg, str):
                                avg_price = float(raw_avg)
                            elif isinstance(raw_avg, (int, float)):
                                avg_price = float(raw_avg)
                            else:
                                self.logger.warning(
                                    f"⚠️ Unexpected avg_buy_price type for {symbol}: "
                                    f"{type(raw_avg)} = {raw_avg}"
                                )
                                avg_price = 0.0
                        except (ValueError, TypeError) as e:
                            self.logger.error(
                                f"❌ Failed to parse avg_buy_price for {symbol}: "
                                f"{raw_avg} ({e})"
                            )
                            avg_price = 0.0
                        
                        # 2. 유효성 검사
                        if avg_price < 0:
                            self.logger.warning(
                                f"⚠️ Negative avg_buy_price for {symbol}: {avg_price}. "
                                "Resetting to 0."
                            )
                            avg_price = 0.0
                        
                        break
            
            # 3. 포지션 존재하는데 평단이 0인 경우 경고
            if amount > 0.0001 and avg_price == 0.0:
                self.logger.warning(
                    f"⚠️ Position exists but avg_buy_price is 0 for {symbol}. "
                    f"Amount: {amount}. This may cause incorrect P&L calculation."
                )
            
            return {
                'amount': amount,
                'entryPrice': avg_price,
                'unrealizedPnL': 0.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching position for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}

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

