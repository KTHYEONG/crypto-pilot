
import ccxt
import time
import pandas as pd
from datetime import datetime, timedelta
import logging

class BinanceClient:
    def __init__(self, api_key=None, secret=None):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,  # CCXT의 내장 Rate Limit 준수 기능 활성화
            'options': {
                'defaultType': 'future',  # 선물 거래 기준 (현물인 경우 'spot')
            }
        })
        self.logger = logging.getLogger(__name__)

    def fetch_ohlcv(self, symbol, timeframe, start_date, end_date=None):
        """
        지정된 기간의 OHLCV 데이터를 수집합니다.
        Binance API 제한(limit=1000)을 고려하여 반복 호출합니다.
        """
        limit = 1000
        since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
        
        if end_date:
            end_timestamp = self.exchange.parse8601(f"{end_date}T23:59:59Z")
        else:
            end_timestamp = self.exchange.milliseconds()

        all_ohlcv = []
        
        self.logger.info(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}...")

        while since < end_timestamp:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                # 마지막 데이터의 타임스탬프 업데이트
                last_timestamp = ohlcv[-1][0]
                since = last_timestamp + 1  # 다음 데이터부터 조회
                
                # 진행 상황 로깅
                current_date = datetime.fromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d')
                self.logger.info(f"Measured up to {current_date} ({len(all_ohlcv)} candles)")
                
                # API 호출 간격 조절 (안전을 위해 추가 대기)
                time.sleep(0.1)
                
                # 목표 시점 도달 확인
                if last_timestamp >= end_timestamp:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error fetching data: {e}")
                time.sleep(5) # 에러 발생 시 5초 대기 후 재시도
                continue

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 중복 제거 및 기간 필터링
        df = df.drop_duplicates(subset=['timestamp'])
        df = df[(df['timestamp'] <= end_timestamp)]
        
        return df

    def get_market_price(self, symbol):
        """현재 시장가 조회"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {e}")
            return None
    
    def fetch_balance(self):
        """USDT 선물 지갑 잔고 조회"""
        try:
            balance = self.exchange.fetch_balance()
            # future wallet: total -> total margin balance, free -> available balance
            return balance['total']['USDT'], balance['free']['USDT']
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return 0.0, 0.0

    def set_leverage(self, symbol, leverage):
        """레버리지 설정"""
        try:
            # use standard ccxt method for better compatibility
            self.exchange.set_leverage(leverage, symbol)
            self.logger.info(f"✅ Leverage set to {leverage}x for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error setting leverage (Value: {leverage}) | Detail: {str(e)}")
            return False

    def fetch_position(self, symbol):
        """
        현재 포지션 조회
        Returns: {
            'amount': 0.0, # 양수면 롱, 음수면 숏
            'entryPrice': 0.0,
            'unrealizedPnL': 0.0,
            'leverage': 1
        }
        """
        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                if pos['symbol'] == symbol:
                    return {
                        'amount': float(pos['contracts']) * (1 if pos['side'] == 'long' else -1), # contracts가 수량, side로 부호 결정
                        'entryPrice': float(pos['entryPrice'] or 0),
                        'unrealizedPnL': float(pos['unrealizedPnl'] or 0),
                        'leverage': int(pos['leverage'])
                    }
        except Exception as e:
            self.logger.error(f"Error fetching position for {symbol}: {e}")
        
        # 포지션 없으면 기본 0 반환
        return {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0, 'leverage': 1}

    def place_order(self, symbol, side, amount, order_type='market', price=None, params={}):
        """
        주문 실행
        side: 'buy' or 'sell'
        amount: 코인 수량 (USDT 아님)
        """
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            self.logger.info(f"⚡ Order Placed: {side} {amount} {symbol} @ {price if price else 'Market'}")
            return order
        except Exception as e:
            self.logger.error(f"❌ Order Failed: {e}")
            return None

    def cancel_all_orders(self, symbol):
        """미체결 주문 취소"""
        try:
            self.exchange.cancel_all_orders(symbol)
            self.logger.info(f"🗑️ Canceled all open orders for {symbol}")
        except Exception as e:
            self.logger.error(f"Error canceling orders: {e}")
