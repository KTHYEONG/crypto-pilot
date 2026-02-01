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
        if not self.exchange:
            self.logger.warning(f"⚠️ Exchange client is not initialized. Cannot fetch price for {symbol}.")
            return None
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            if ticker is not None:
                return ticker['last']
        except Exception:
            pass
            
        # [Fallback] Try alt symbol format (BTC/KRW) via CCXT
        if '-' in symbol:
            try:
                alt_symbol = symbol.split('-')[1] + "/" + symbol.split('-')[0]
                ticker = self.exchange.fetch_ticker(alt_symbol)
                if ticker is not None:
                     return ticker['last']
            except Exception:
                pass
            
        # [Final Fallback] Direct REST API Call
        try:
            import requests
            url = f"https://api.upbit.com/v1/ticker?markets={symbol}"
            headers = {"accept": "application/json"}
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                price = float(data[0]['trade_price'])
                self.logger.info(f"✅ Recovered price via REST API for {symbol}: {price}")
                return price
        except Exception as e_rest:
             self.logger.error(f"❌ REST API Fallback failed for {symbol}: {e_rest}")
             
        return None

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, end=None):
        """
        OHLCV 데이터 조회 (Historical Data Download)
        - Upbit는 'to' 파라미터만 지원하므로, end 시점부터 역순으로 조회하여 since까지 도달해야 함.
        - since: 시작 타임스탬프 (ms)
        - end: 종료 타임스탬프 (ms) - 생략 시 현재시간
        - limit: 가져올 최대 개수 (생략 시 since~end 기간 전체)
        """
        if not self.exchange: return None

        # 1. 타임프레임 보정
        tf_map = {'1h': '60', '4h': '240', '1d': 'day'} # 참고용
        target_tf = timeframe
        
        all_ohlcv = []
        
        # 종료 시점 설정 (ms -> datetime string for API)
        # Upbit API 'to' parameter usually expects KST (UTC+9) in 'YYYY-MM-DD HH:MM:SS' format
        # If we send UTC string, it might be interpreted as KST (9 hours past).
        # To get the absolute LATEST data, we should align with KST or use explicit ISO with +09:00,
        # but simplify by shifting UTC to KST for the string representation.
        from datetime import timedelta
        
        if end:
            cursor_dt = datetime.utcfromtimestamp(end / 1000) + timedelta(hours=9)
        else:
            cursor_dt = datetime.utcnow() + timedelta(hours=9)
            
        cursor_str = cursor_dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # since가 없으면 기본적으로 최근 200개만 가져오도록 (Safety)
        if since is None and limit is None:
            limit = 200
            
        try:
            # Loop Safety & Progress
            loop_count = 0
            last_oldest_ts = float('inf')
            
            while True:
                loop_count += 1
                if loop_count > 10000: # Safety Break
                    self.logger.warning(f"⚠️ Limit loop count exceeded ({loop_count}) for {symbol}")
                    break
                    
                params = {'to': cursor_str}
                req_limit = 200
                
                # Fetch candles
                ohlcv = self.exchange.fetch_ohlcv(symbol, target_tf, limit=req_limit, params=params)
                
                if not ohlcv or len(ohlcv) == 0:
                    self.logger.debug(f"   ℹ️ Reached end of data (Empty) at {cursor_str}")
                    break
                    
                # Prepend to list
                all_ohlcv = ohlcv + all_ohlcv
                
                # Get oldest timestamp in this batch
                oldest_ts = ohlcv[0][0]
                first_date_str = pd.to_datetime(oldest_ts, unit='ms').strftime('%Y-%m-%d %H:%M')
                
                # Progress Log (every 5 requests or if 1d)
                if target_tf == 'day' or loop_count % 5 == 0:
                     self.logger.info(f"   ... fetched batch {loop_count}: oldest {first_date_str} (Total {len(all_ohlcv)} rows)")
                
                # [CRITICAL] Loop Protection: Ensure we are moving backwards
                if oldest_ts >= last_oldest_ts:
                     self.logger.warning(f"⚠️ Detected Infinite Loop: Timestamp failed to decrease. {oldest_ts} >= {last_oldest_ts}. Stopping.")
                     break
                last_oldest_ts = oldest_ts
                
                # Update Cursor for next batch
                # Subtract 1 second from oldest_ts
                next_cursor_dt = datetime.utcfromtimestamp((oldest_ts / 1000) - 1) + timedelta(hours=9)
                cursor_str = next_cursor_dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Break Conditions
                # 1. Reached 'since'
                if since is not None and oldest_ts <= since:
                    self.logger.debug(f"   ✅ Reached start date: {first_date_str}")
                    break
                    
                # 2. Reached 'limit' count
                if limit is not None and len(all_ohlcv) >= limit:
                    all_ohlcv = all_ohlcv[-limit:] 
                    break
                    
                # Rate Limit Safety
                time.sleep(0.12)
                
            if not all_ohlcv:
                self.logger.warning(f"⚠️ No OHLCV data returned for {symbol} ({target_tf})")
                return None
                
            # DataFrame 변환
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # [CRITICAL] Remove duplicates caused by overlapping fetches
            df.drop_duplicates(subset=['timestamp'], inplace=True)
            df.sort_values('timestamp', inplace=True)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Final slicing by time (precise trim)
            if since:
                df = df[df['timestamp'] >= since]
            if end:
                df = df[df['timestamp'] <= end]
                
            return df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'timestamp']]
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching OHLCV for {symbol}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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

    def fetch_open_orders(self, symbol):
        """미체결 주문 조회"""
        if not self.exchange: return []
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching open orders for {symbol}: {e}")
            return []

    def place_order(self, symbol, side, amount=None, price=None, order_type='market'):
        """기본 주문 실행 (시장가/지정가)"""
        if not self.exchange: return None
        try:
            if order_type == 'market':
                if side == 'buy':
                    # Upbit Market Buy: 'price' is the KRW cost
                    return self.exchange.create_order(symbol, 'market', 'buy', price, params={'ord_type': 'price'}) 
                else:
                    return self.exchange.create_order(symbol, 'market', 'sell', amount)
            elif order_type == 'limit':
                return self.exchange.create_order(symbol, 'limit', side, amount, price)
        except Exception as e:
            self.logger.error(f"❌ Order Failed: {e}")
            return None

    def place_order_smart(self, symbol, side, amount=None, price=None):
        """
        Upbit 전용 Smart Aggressive Limit 주문
        - 수수료가 동일한 점을 활용, 시장가처럼 즉시 체결되되 호가 공백(슬리피지)을 제한.
        - Buy: price(KRW 예산)를 받아 '현재가 + 0.1%' 가격으로 계산된 수량만큼 지정가 매수.
        - Sell: amount(코인 수량)를 받아 '현재가 - 0.1%' 가격으로 지정가 매도.
        
        [ISSUE #5 FIX] 슬리피지 1% → 0.1%로 축소 (Binance Futures 수준)
        """
        if not self.exchange: return None
        
        try:
            # 1. 현재가 조회 (Ticker)
            ticker = self.exchange.fetch_ticker(symbol)
            if ticker is None:
                return None
            cur_price = ticker['last']
            
            if side == 'buy':
                # 예산(KRW) 기반 매수
                cost = price
                # [ISSUE #5 FIX] 현재가보다 0.1% 높은 가격으로 지정가 설정 (Slippage Cap: 1% → 0.1%)
                limit_price = self.exchange.price_to_precision(symbol, cur_price * 1.001)
                # 예산 내에서 살 수 있는 수량 계산 (수수료 0.05% 고려)
                qty = self.exchange.amount_to_precision(symbol, (cost * 0.9995) / float(limit_price))
                
                self.logger.info(f"⚡ Smart Buy {symbol} | Budget: {cost:,.0f} KRW | Target Limit: {limit_price:,.0f} (+0.1%)")
                return self.exchange.create_order(symbol, 'limit', 'buy', qty, limit_price)
                
            else:
                # 수량(Coin) 기반 매도
                qty = amount
                # [ISSUE #5 FIX] 현재가보다 0.1% 낮은 가격으로 지정가 설정 (Slippage Cap: 1% → 0.1%)
                limit_price = self.exchange.price_to_precision(symbol, cur_price * 0.999)
                
                self.logger.info(f"⚡ Smart Sell {symbol} | Qty: {qty} | Target Limit: {limit_price:,.0f} (-0.1%)")
                return self.exchange.create_order(symbol, 'limit', 'sell', qty, limit_price)

        except Exception as e:
            self.logger.error(f"❌ Smart Order Failed for {symbol}: {e}")
            # [Fallback] 에러 발생 시 최후의 수단으로 일반 시장가 주문 시도
            self.logger.warning(f"⚠️ Falling back to Market Order for {symbol}")
            return self.place_order(symbol, side, amount=amount, price=price, order_type='market')


