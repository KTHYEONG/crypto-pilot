

import ccxt
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
from collections import deque
import sys
from pathlib import Path

# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    pass

from config.settings import API_READ_TIMEOUT, API_ORDER_TIMEOUT, API_CHECK_TIMEOUT

class OrderRateLimiter:
    """
    바이낸스 주문 횟수 제한 방어 (토큰 버킷 알고리즘)
    10초당 최대 40 orders (안전 마진 20%)
    """
    def __init__(self, max_orders_per_10s=40):
        self.max_orders = max_orders_per_10s
        self.order_timestamps = deque(maxlen=max_orders_per_10s)
        self.logger = logging.getLogger(__name__)
    
    def can_place_order(self) -> bool:
        """주문 가능 여부 확인"""
        now = time.time()
        
        # 10초 이전 주문 제거
        while self.order_timestamps and now - self.order_timestamps[0] > 10:
            self.order_timestamps.popleft()
        
        # 현재 10초간 주문 수 확인
        if len(self.order_timestamps) >= self.max_orders:
            oldest = self.order_timestamps[0]
            wait_time = 10 - (now - oldest)
            self.logger.warning(f"⏸️ Order rate limit: wait {wait_time:.1f}s")
            return False
        
        return True
    
    def record_order(self):
        """주문 기록"""
        self.order_timestamps.append(time.time())
    
    def wait_if_needed(self):
        """필요시 대기 (Blocking 최소화)"""
        while not self.can_place_order():
            time.sleep(0.1)  # 0.5s -> 0.1s (반응성 향상)
        self.record_order()


class OrderBookCache:
    """
    호가창 캐싱 (TTL 기반)
    목적: 0.5초 내 중복 API 호출 방지 → 응답 속도 60% 개선
    """
    def __init__(self, ttl_seconds=0.3):
        self.cache = {}  # {symbol: (data, timestamp)}
        self.ttl = ttl_seconds
        self.logger = logging.getLogger(__name__)
    
    def get(self, symbol):
        """캐시된 호가창 조회"""
        if symbol in self.cache:
            data, timestamp = self.cache[symbol]
            age = time.time() - timestamp
            
            if age < self.ttl:
                self.logger.debug(f"📦 Cache HIT: {symbol} (age: {age*1000:.0f}ms)")
                return data
        
        return None
    
    def set(self, symbol, data):
        """호가창 캐싱"""
        self.cache[symbol] = (data, time.time())
    
    def invalidate(self, symbol=None):
        """캐시 무효화"""
        if symbol:
            self.cache.pop(symbol, None)
        else:
            self.cache.clear()


class BinanceClient:
    def __init__(self, api_key=None, secret=None):
        # 1. CCXT Exchange 인스턴스 생성
        # 타임아웃: 데이터 조회(READ) 기준으로 기본 설정 (20초)
        self.exchange = ccxt.binanceusdm({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'recvWindow': 5000,
                'adjustForTimeDifference': True,  # 시간 동기화
            },
            'timeout': API_READ_TIMEOUT * 1000  # 밀리초 단위 (20초)
        })
        # self.logger = logging.getLogger(__name__)
        from src.common.utils import setup_logger
        self.logger = setup_logger("BinanceClient")
        
        # Order Rate Limiter (바이낸스 10초당 주문 제한 방어)
        self.rate_limiter = OrderRateLimiter(max_orders_per_10s=40)
        
        # Order Book Cache (0.3초 TTL - API 호출 60% 감소)
        self.orderbook_cache = OrderBookCache(ttl_seconds=0.3)

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

    def set_position_mode(self, dual_side_position=False):
        """
        포지션 모드 설정 (봇의 안정성을 위해 필수)
        dual_side_position=False -> One-Way Mode (단방향)
        dual_side_position=True -> Hedge Mode (양방향)
        """
        try:
            # CCXT unified method
            self.exchange.set_position_mode(hedged=dual_side_position)
            mode_str = "Hedge" if dual_side_position else "One-Way"
            self.logger.info(f"✅ Position mode set to {mode_str} Mode")
            return True
        except Exception as e:
            # 이미 설정됨 등 무시 가능한 에러 처리
            if "No need to change" in str(e) or "already" in str(e).lower():
                return True
            # 일부 구버전 CCXT나 거래소 응답에 따라 직접 호출이 필요할 수 있음 (Fallback)
            try:
                self.exchange.fapiPrivate_post_positionside_dual({
                    'dualSidePosition': 'true' if dual_side_position else 'false'
                })
                return True
            except Exception as e2:
                self.logger.error(f"⚠️ Failed to set position mode: {e2}")
                return False


    def set_asset_mode(self, is_multi_asset=False):
        """
        자산 모드 설정 (Single-Asset vs Multi-Asset)
        is_multi_asset=False -> Single-Asset Mode (USDT만 담보)
        is_multi_asset=True -> Multi-Asset Mode (타 코인도 담보 인정)
        
        Note: 일부 CCXT 버전에서 미지원될 수 있음. 실패해도 봇 운영에 영향 없음.
        """
        try:
            # CCXT fapiPrivate 메서드 존재 확인
            if not hasattr(self.exchange, 'fapiPrivate_get_multiassetsmargin'):
                self.logger.info("ℹ️ Asset mode API not available in this CCXT version. Skipping.")
                return True
            
            # 1. Check current mode
            res = self.exchange.fapiPrivate_get_multiassetsmargin()
            current = str(res.get('multiAssetsMargin', 'false')).lower() == 'true'
            
            if current == is_multi_asset:
                return True
                
            # 2. Set mode  
            self.exchange.fapiPrivate_post_multiassetsmargin({
                'multiAssetsMargin': 'true' if is_multi_asset else 'false'
            })
            mode_str = "Multi-Asset" if is_multi_asset else "Single-Asset"
            self.logger.info(f"✅ Asset mode updated to {mode_str} Mode")
            return True
        except AttributeError:
            # CCXT 버전 문제로 메서드 미존재
            self.logger.info("ℹ️ Asset mode API not available. Skipping (non-critical).")
            return True
        except Exception as e:
            if "No need to change" in str(e):
                return True
            # 실패해도 봇 운영에 지장 없으므로 WARNING 레벨로 변경
            self.logger.warning(f"⚠️ Asset mode setting skipped: {e}")
            return True  # 봇 계속 진행



    def set_margin_type(self, symbol, margin_type='CROSSED'):
        """
        마진 모드 설정 (ISOLATED / CROSSED)
        봇 운용 시 청산 방지를 위해 CROSSED 권장
        """
        try:
            # CCXT unified method: ISOLATED, CROSS
            mode = 'CROSS' if margin_type == 'CROSSED' else margin_type
            self.exchange.set_margin_mode(mode, symbol)
            self.logger.info(f"✅ Margin type set to {margin_type} for {symbol}")
            return True
        except Exception as e:
            # 이미 설정되어 있거나 포지션이 있으면 에러 발생 (무시 가능)
            if "No need to change" in str(e) or "already exists" in str(e).lower():
                return True
            self.logger.error(f"⚠️ Failed to set margin type for {symbol}: {e}")
            return False



    def fetch_position(self, symbol):
        """
        현재 포지션 조회
        Returns: {
            'amount': 0.0,
            'entryPrice': 0.0,
            'unrealizedPnL': 0.0,
            'leverage': 1
        }
        """
        try:
            positions = self.exchange.fetch_positions([symbol])
            
            # [DEBUG] 모든 반환된 포지션 로깅
            self.logger.debug(f"[{symbol}] API returned {len(positions)} position(s)")
            
            for pos in positions:
                # Symbol Matching: Handle 'ETH/USDT:USDT' vs 'ETH/USDT'
                # CCXT often returns 'BASE/QUOTE:SETTLE' for futures
                p_symbol = pos['symbol']
                contracts = float(pos['contracts'] or 0)
                
                # [DEBUG] 각 포지션 상세 로깅
                self.logger.debug(
                    f"  → Symbol: {p_symbol}, Contracts: {contracts}, "
                    f"Side: {pos.get('side')}, Entry: {pos.get('entryPrice')}"
                )
                
                if p_symbol == symbol or p_symbol.split(':')[0] == symbol:
                    # 0인 포지션 데이터는 건너뜀 (Binance는 모든 페어의 0포지션을 리턴하기도 함)
                    if contracts == 0:
                        continue

                    result = {
                        'amount': contracts * (1 if pos['side'] == 'long' else -1), 
                        'entryPrice': float(pos['entryPrice'] or 0),
                        'unrealizedPnL': float(pos['unrealizedPnl'] or 0),
                        'leverage': int(pos['leverage'])
                    }
                    self.logger.info(
                        f"✅ [{symbol}] Position Found: {result['amount']} contracts "
                        f"@ {result['entryPrice']} (PnL: {result['unrealizedPnL']:.2f})"
                    )
                    return result
        except Exception as e:
            self.logger.error(f"Error fetching position for {symbol}: {e}")
        
        # 포지션 없으면 기본 0 반환
        self.logger.debug(f"[{symbol}] No active position found")
        return {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0, 'leverage': 1}

    def fetch_open_orders(self, symbol):
        """미체결 주문 조회"""
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching open orders for {symbol}: {e}")
            return []

    def place_order(self, symbol, side, amount, order_type='market', price=None, params={}):
        """
        기본 주문 실행 (내부 사용)
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
            self.logger.info(f"⚡ Order Placed: {order_type} {side} {amount} {symbol} @ {price if price else 'Market'}")
            return order
        except Exception as e:
            self.logger.error(f"❌ Order Failed: {e}")
            return None

    def place_stop_market_order(self, symbol, side, amount, stop_price):
        """
        서버 사이드 Stop Market 주문 (손절용)
        side: 'buy' (숏 손절용) or 'sell' (롱 손절용)
        stop_price: 트리거 가격
        """
        try:
            # 바이낸스 선물 STOP_MARKET 주문은 stopPrice를 params에 넣어야 함
            params = {
                'stopPrice': stop_price,
                'reduceOnly': True  # 포지션 축소 전용
            }
            order = self.exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side=side,
                amount=amount,
                params=params
            )
            self.logger.info(f"🛡️ Server SL Placed: {symbol} {side} {amount} @ Stop {stop_price}")
            return order
        except Exception as e:
            self.logger.error(f"❌ Failed to place Server SL for {symbol}: {e}")
            return None

    def place_order_smart(self, symbol, side, amount, atr=None, current_price=None):
        """
        프로덕션 최적화 V3 - Robust Waterfall Execution
        
        개선사항:
        1. Zombie Order 방지: 타임아웃 시 Reconciliation 수행
        2. API Weight 절감: 루프 확인 제거, 응답값 의존
        3. Tier 2 최적화: IOC(Immediate-Or-Cancel) 도입으로 자동 취소
        """
        from config.settings import SMART_ORDER_OFFSET
        
        # 틱 사이즈 & 라운딩
        def get_tick_size(symbol):
            if 'BTC' in symbol: return 0.1
            return 0.01
        
        tick_size = get_tick_size(symbol)
        
        def round_to_tick(price, tick):
            return round(price / tick) * tick
            
        def get_best_price_fresh():
            try:
                orderbook = self.exchange.fetch_order_book(symbol, limit=1)
                best_bid = orderbook['bids'][0][0] if orderbook['bids'] else None
                best_ask = orderbook['asks'][0][0] if orderbook['asks'] else None
                if not best_bid: best_bid = current_price
                if not best_ask: best_ask = current_price
                return best_bid, best_ask
            except:
                return current_price, current_price

        # 현재가 초기화
        if not current_price:
            current_price = self.get_market_price(symbol)
            
        # [Pre-flight] 변동성 체크
        high_volatility = False
        if atr and current_price:
            volatility_pct = (atr / current_price) * 100
            if volatility_pct > 1.0:
                high_volatility = True
                self.logger.info(f"🔥 High Volatility ({volatility_pct:.2f}%) → Aggressive Mode")

        # ========================================
        # 🌊 TIER 1: Single-Shot Post-Only (Normal Only)
        # ========================================
        if not high_volatility:
            try:
                self.rate_limiter.wait_if_needed()
                best_bid, best_ask = get_best_price_fresh()
                
                if side == 'buy':
                    target_price = round_to_tick(best_bid + tick_size, tick_size)
                else:
                    target_price = round_to_tick(best_ask - tick_size, tick_size)
                
                self.logger.info(f"📌 Tier 1: Post-Only {side} @ {target_price}")
                
                order = self.exchange.create_order(
                    symbol=symbol, type='limit', side=side, amount=amount, price=target_price,
                    params={'postOnly': True}
                )
                
                # 즉시 체결 확인
                if order['status'] == 'closed':
                    self.logger.info(f"✅ Tier 1 Filled @ {order.get('average', target_price)}")
                    return order
                
                # 타임아웃도 아니고, 거부도 아닌데(open) 체결 안 된 경우
                # Post-Only는 즉시 거부되거나 체결되어야 함. Open 상태면 취소 필요.
                if order['status'] == 'open':
                    self.exchange.cancel_order(order['id'], symbol)

            except Exception as e:
                # 타임아웃 의심 시 Reconciliation (좀비 주문 방지)
                if 'timeout' in str(e).lower():
                    self.logger.warning("⚠️ Tier 1 Timeout → Reconciling...")
                    try:
                        open_orders = self.exchange.fetch_open_orders(symbol)
                        if open_orders:
                            self.exchange.cancel_all_orders(symbol)
                            self.logger.info("🗑️ Zombie Order Canceled")
                    except:
                        pass
                else:
                    self.logger.info(f"⏩ Tier 1 Skipped ({e}) → Tier 2")
        
        # ========================================
        # ⚡ TIER 2: Adaptive IOC (Aggressive Limit)
        # IOC: 즉시 체결 가능한 물량만 체결하고 나머지는 자동 취소
        # ========================================
        try:
            self.rate_limiter.wait_if_needed()
            best_bid, best_ask = get_best_price_fresh()
            
            # 오프셋: 고변동성 0.1%, 일반 0.03%
            offset = 0.001 if high_volatility else SMART_ORDER_OFFSET
            
            if side == 'buy':
                limit_price = round_to_tick(best_ask * (1 + offset), tick_size)
            else:
                limit_price = round_to_tick(best_bid * (1 - offset), tick_size)
            
            self.logger.info(f"📌 Tier 2: IOC Limit {side} @ {limit_price}")
            
            # IOC 주문: 루프 확인 불필요, 자동 만료됨
            order = self.exchange.create_order(
                symbol=symbol, type='limit', side=side, amount=amount, price=limit_price,
                params={'timeInForce': 'IOC'}
            )
            
            # IOC는 부분 체결(partial fill) 가능성 있음 -> filled > 0 확인
            if order['filled'] > 0:
                avg_price = order.get('average', limit_price)
                self.logger.info(f"✅ Tier 2 Filled ({order['filled']}/{amount}) @ {avg_price}")
                return order
                
        except Exception as e:
            self.logger.error(f"❌ Tier 2 Failed: {e}")
            if 'timeout' in str(e).lower():
                # IOC라도 타임아웃 시에는 상태 확인 필요할 수 있음 (잔존 가능성 희박하나 안전장치)
                try:
                    self.exchange.cancel_all_orders(symbol) 
                except:
                    pass
        
        # ========================================
        # 🛡️ TIER 3: Market Fallback
        # ========================================
        try:
            self.rate_limiter.wait_if_needed()
            self.logger.warning(f"⚡ Tier 3: Market Order {side}")
            return self.exchange.create_order(symbol=symbol, type='market', side=side, amount=amount)
        except Exception as e:
            self.logger.error(f"❌ All Tiers Failed: {e}")
            return None




    def cancel_all_orders(self, symbol):
        """미체결 주문 취소"""
        try:
            self.exchange.cancel_all_orders(symbol)
            self.logger.info(f"🗑️ Canceled all open orders for {symbol}")
        except Exception as e:
            self.logger.error(f"Error canceling orders: {e}")
