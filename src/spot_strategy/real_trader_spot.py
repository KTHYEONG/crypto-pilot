"""
RealTrader Spot - 24시간 자동 현물(Upbit) 트레이딩 봇
===================================================
- Upbit 현물 시장 특화 (Long-Only, 1x Leverage, KRW 마켓)
- 숏(Short), 레버리지, 펀딩비 관련 로직 완벽 제거
- 자본 분배(Capital Allocation) 기반의 현물 최적화 포지션 사이징 적용
"""

import os
import sys
import time
import signal
import json
import sqlite3
import logging
import gc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import RotatingFileHandler
from functools import wraps
from typing import Optional, Dict, Any, Tuple

try:
    import ccxt
except ImportError:
    ccxt = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    import fcntl
except ImportError:
    fcntl = None

# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())

from config.settings import (
    UPBIT_ACCESS_KEY, 
    UPBIT_SECRET_KEY,
    LOG_DIR,
    SPOT_STRATEGY_DB,
    SPOT_HEARTBEAT_FILE,
    API_RETRY_ATTEMPTS,
    API_RETRY_WAIT_MIN,
    API_RETRY_WAIT_MAX,
    MIN_ORDER_VALUE_KRW,
    MIN_POSITION_VALUE_KRW,
    MAX_INVEST_CAP_KRW,
    SPOT_LOOP_INTERVAL_SECONDS,
    SPOT_SYMBOL_DELAY_SECONDS,
    ERROR_SLEEP_SECONDS,
    CANDLE_SYNC_OFFSET_SECONDS,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    SPOT_TARGET_SYMBOLS,
    SPOT_ALLOCATION_WEIGHTS,
    SPOT_STATE_FILE,
    SLIPPAGE_RATE,
)
from src.spot_strategy.upbit_client import UpbitClient
from src.spot_strategy.strategies_spot import UltimateSpotStrategy
from src.common.utils import setup_logger
from src.common.components import TradeHistoryDB, HealthCheckManager, calculate_candle_wait_time

logger = setup_logger("RealTraderSpot")

_CCXT_TRANSIENT_ERRORS: Tuple[type, ...] = ()
if ccxt is not None:
    _CCXT_TRANSIENT_ERRORS = tuple(
        err for err in (
            getattr(ccxt, "NetworkError", None),
            getattr(ccxt, "ExchangeNotAvailable", None),
            getattr(ccxt, "RequestTimeout", None),
            getattr(ccxt, "DDoSProtection", None),
            getattr(ccxt, "RateLimitExceeded", None),
        ) if isinstance(err, type)
    )

def _is_retryable_api_exception(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if _CCXT_TRANSIENT_ERRORS and isinstance(exc, _CCXT_TRANSIENT_ERRORS):
        return True
    return False

def network_api_retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_error: Optional[Exception] = None
        max_attempts = max(1, int(API_RETRY_ATTEMPTS))
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not _is_retryable_api_exception(e):
                    raise
                last_error = e
                if attempt >= (max_attempts - 1):
                    break
                wait_time = min(float(API_RETRY_WAIT_MIN) * (2 ** attempt), float(API_RETRY_WAIT_MAX))
                logger.warning(f"⚠️ API transient error in {func.__name__} (attempt {attempt+1}/{max_attempts}): {e}. Waiting {wait_time:.1f}s")
                time.sleep(max(0.0, wait_time))
        raise last_error if last_error is not None else RuntimeError("retry wrapper reached unexpected state")
    return wrapper

class StateManager:
    """Spot 전용 거래 상태 관리 (JSON 파일 기반)"""
    def __init__(self, state_file: Path):
        self.state_file = Path(state_file)
        self.lock_file = self.state_file.with_suffix(self.state_file.suffix + ".lock")
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.state_file.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_unlocked({})
        if not self.lock_file.exists():
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            self.lock_file.touch(exist_ok=True)

    def _acquire_lock(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_fp = open(self.lock_file, 'a+b')
        try:
            if msvcrt is not None:
                lock_fp.seek(0)
                msvcrt.locking(lock_fp.fileno(), msvcrt.LK_LOCK, 1)
            elif fcntl is not None:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            return lock_fp
        except BaseException:
            try: lock_fp.close()
            except Exception: pass
            raise

    def _release_lock(self, lock_fp):
        try:
            if msvcrt is not None:
                lock_fp.seek(0)
                msvcrt.locking(lock_fp.fileno(), msvcrt.LK_UNLCK, 1)
            elif fcntl is not None:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        finally:
            try: lock_fp.close()
            except Exception: pass

    def _load_unlocked(self) -> dict:
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"⚠️ State load error: {e}")
            return {}

    def _save_unlocked(self, state: dict):
        tmp_file = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.state_file)
        except Exception as e:
            logger.error(f"⚠️ State save error: {e}")
            try:
                if tmp_file.exists(): tmp_file.unlink()
            except Exception: pass

    def _load(self) -> dict:
        lock_fp = self._acquire_lock()
        try: return self._load_unlocked()
        finally: self._release_lock(lock_fp)

    def _save(self, state: dict):
        lock_fp = self._acquire_lock()
        try: self._save_unlocked(state)
        finally: self._release_lock(lock_fp)

    def get_symbol_state(self, symbol: str) -> dict:
        return self._load().get(symbol, {})

    def update_symbol_state(self, symbol: str, data: dict):
        lock_fp = self._acquire_lock()
        try:
            state = self._load_unlocked()
            if symbol not in state: state[symbol] = {}
            state[symbol].update(data)
            self._save_unlocked(state)
        finally: self._release_lock(lock_fp)

    def clear_symbol_state(self, symbol: str):
        lock_fp = self._acquire_lock()
        try:
            state = self._load_unlocked()
            if symbol in state:
                state[symbol] = {}
                self._save_unlocked(state)
        finally: self._release_lock(lock_fp)

class RealTraderSpot:
    """Production-grade 업비트 현물 트레이딩 봇"""
    
    def __init__(self):
        self.client = UpbitClient(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        self.strategies: Dict[str, UltimateSpotStrategy] = {}
        self.params_map: Dict[str, dict] = {}
        self.symbols: list = []
        self.symbol_allocation_weights: Dict[str, float] = {}
        
        self.trade_db = TradeHistoryDB(SPOT_STRATEGY_DB)
        self.health_manager = HealthCheckManager(SPOT_HEARTBEAT_FILE)
        self.state_manager = StateManager(SPOT_STATE_FILE)
        
        self._shutdown_requested = False
        
        self.last_calc_candle: Dict[str, str] = {}
        self.last_exit_calc_candle: Dict[str, str] = {}
        self._server_time_offset_ms: int = 0
        self._last_server_time_sync: datetime = datetime.min
        
        self._log_last_emit_ts: Dict[str, float] = {}
        self._log_last_message: Dict[str, str] = {}
        
        self._setup_signal_handlers()

    def _should_emit_log(self, key: str, interval_seconds: float = 0.0, message: Optional[str] = None, emit_on_change: bool = False) -> bool:
        now_ts = time.time()
        last_ts = float(self._log_last_emit_ts.get(key, 0.0) or 0.0)
        if emit_on_change and message is not None:
            last_message = self._log_last_message.get(key)
            if last_message != message:
                self._log_last_message[key] = message
                self._log_last_emit_ts[key] = now_ts
                return True
        if now_ts - last_ts >= max(0.0, float(interval_seconds)):
            if message is not None:
                self._log_last_message[key] = message
            self._log_last_emit_ts[key] = now_ts
            return True
        return False

    def _log_throttled(self, level: str, key: str, message: str, interval_seconds: float, emit_on_change: bool = False) -> None:
        if not self._should_emit_log(key=key, interval_seconds=interval_seconds, message=message, emit_on_change=emit_on_change): return
        getattr(logger, level, logger.info)(message)

    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}. Initiating graceful shutdown...")
            self._shutdown_requested = True
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def load_strategies_from_json(self):
        logger.info("📂 Loading Spot strategies from JSON files in results/...")
        
        preferred_symbols = list(SPOT_TARGET_SYMBOLS) if SPOT_TARGET_SYMBOLS else list(SPOT_ALLOCATION_WEIGHTS.keys())
        if not preferred_symbols:
            raise ValueError("No live spot symbols configured. Check SPOT_TARGET_SYMBOLS in config/settings.py")
        self.symbols = preferred_symbols.copy()
        
        weights = {}
        for s in self.symbols:
            weights[s] = float(SPOT_ALLOCATION_WEIGHTS.get(s, 1.0))
        total_weight = sum(weights.values())
        self.symbol_allocation_weights = {s: (w/total_weight if total_weight > 0 else 1.0/len(self.symbols)) for s, w in weights.items()}
        
        logger.info("📌 Live symbol allocation applied: %s", ", ".join(f"{s}={self.symbol_allocation_weights.get(s, 0.0):.2f}" for s in self.symbols))
        
        results_dir = os.path.join(project_root, "results")
        
        for symbol in self.symbols:
            # KRW-BTC -> KRWBTC
            clean_sym = symbol.replace("/", "").replace("-", "")
            json_path = os.path.join(results_dir, f"best_params_{clean_sym}_4h.json")
            
            if not os.path.exists(json_path):
                logger.error(f"❌ JSON file not found for {symbol}: {json_path}")
                continue # 현물은 없는 코인이 있을 수 있으므로 에러 대신 패스
                
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    symbol_params = json.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to parse JSON for {symbol}: {e}")
                continue
                
            symbol_params.setdefault('INDICATOR_TIMEFRAME', '1d')
            self.params_map[symbol] = symbol_params
            strategy_name = f"RealSpot_{clean_sym}"
            self.strategies[symbol] = UltimateSpotStrategy(strategy_name, symbol_params)
            logger.info(f"✅ Strategy initialized from JSON: {symbol} | Exec TF: {symbol_params.get('TIMEFRAME', '4h')}")

    @network_api_retry
    def _fetch_balance_safe(self) -> tuple:
        """Returns (total_krw, free_krw)"""
        return self.client.fetch_balance()
    
    @network_api_retry
    def _fetch_ohlcv_safe(self, symbol: str, timeframe: str, start_str: str):
        return self.client.fetch_ohlcv(symbol, timeframe, start_date=start_str)

    @network_api_retry
    def _fetch_recent_ohlcv_safe(self, symbol: str, timeframe: str, limit: int = 3):
        return self.client.fetch_recent_ohlcv(symbol, timeframe, limit=limit)
        
    @network_api_retry
    def _get_market_price_safe(self, symbol: str) -> float:
        return self.client.get_market_price(symbol)

    @network_api_retry
    def _fetch_server_time_ms_safe(self) -> int:
        return int(self.client.fetch_server_time_ms())

    def _sync_server_time_offset(self, force: bool = False, sync_interval_seconds: int = 60):
        now = datetime.utcnow()
        elapsed = (now - self._last_server_time_sync).total_seconds()
        if (not force) and elapsed < max(5, int(sync_interval_seconds)): return
        local_ms = int(now.timestamp() * 1000)
        try:
            server_ms = self._fetch_server_time_ms_safe()
            if server_ms > 0:
                self._server_time_offset_ms = int(server_ms - local_ms)
                self._last_server_time_sync = now
        except Exception as e:
            self._last_server_time_sync = now

    def _get_reference_now_ms(self) -> int:
        self._sync_server_time_offset(force=False)
        return int(datetime.utcnow().timestamp() * 1000) + int(self._server_time_offset_ms)

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        tf = str(timeframe or '').strip().lower()
        try:
            if tf.endswith('m'): return max(1, int(tf[:-1]))
            if tf.endswith('h'): return max(1, int(tf[:-1]) * 60)
            if tf.endswith('d'): return max(1, int(tf[:-1]) * 1440)
        except ValueError: return -1
        return -1

    def _get_candle_slot_id(self, timeframe: str) -> str:
        """현재 시간에 해당하는 캔들 슬롯 ID (캐싱용)"""
        tf_min = self._timeframe_to_minutes(timeframe)
        if tf_min <= 0: return "unknown"
        now_ms = self._get_reference_now_ms()
        interval_ms = tf_min * 60 * 1000
        slot_start_ms = now_ms - (now_ms % interval_ms)
        return f"{timeframe}_{slot_start_ms}"

    def _select_last_closed_candle(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.Series]:
        if df is None or df.empty: return None
        if 'timestamp' not in df.columns: return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        interval_min = self._timeframe_to_minutes(timeframe)
        if interval_min <= 0: return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        interval_ms = interval_min * 60 * 1000
        now_ms = self._get_reference_now_ms()
        timestamps = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0).astype(np.int64)
        closed_mask = (timestamps + interval_ms) <= now_ms
        closed_indices = df.index[closed_mask.to_numpy()]
        if len(closed_indices) > 0: return df.loc[closed_indices[-1]]
        return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]

    def _extract_candle_timestamp_ms(self, candle: pd.Series) -> int:
        if candle is None: return 0
        raw_ts = candle.get('timestamp', 0)
        try: return int(raw_ts) if not pd.isna(raw_ts) else 0
        except: return 0

    def _cache_indicators(self, symbol: str, data: dict):
        if not hasattr(self, '_ind_cache'): self._ind_cache = {}
        self._ind_cache[symbol] = data

    def _get_cached_indicators(self, symbol: str) -> dict:
        if not hasattr(self, '_ind_cache'): self._ind_cache = {}
        return self._ind_cache.get(symbol, {})

    def _get_current_position(self, symbol: str, current_price: float) -> dict:
        """Upbit 잔고를 조회하여 특정 코인의 포지션(보유량)을 반환"""
        try:
            base_coin = symbol.split('-')[1] if '-' in symbol else symbol.split('/')[0]
            balance = self.client.fetch_balance_dict()
            
            if not balance or 'free' not in balance or 'used' not in balance:
                return {'amount': 0.0, 'entryPrice': 0.0, 'value': 0.0}
                
            amount = float(balance['free'].get(base_coin, 0.0)) + float(balance['used'].get(base_coin, 0.0))
            value_krw = amount * current_price
            
            # 최소 포지션 가치(1만원) 미만이면 먼지로 간주
            if value_krw < MIN_POSITION_VALUE_KRW:
                return {'amount': 0.0, 'entryPrice': 0.0, 'value': 0.0}
                
            # 매수 평균가는 로컬 state에서 관리하는 것을 원칙으로 하나, 업비트 API가 지원하면 사용 가능
            # CCXT Upbit는 fetch_balance 시 매수평균가를 주지 않으므로 로컬 State 활용이 필수적임
            state = self.state_manager.get_symbol_state(symbol)
            entry_price = float(state.get('entry_price', current_price))
            
            return {
                'amount': amount,
                'entryPrice': entry_price,
                'value': value_krw
            }
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch Spot position: {e}")
            return {'amount': 0.0, 'entryPrice': 0.0, 'value': 0.0}

    def _calculate_spot_position_size(self, current_price: float, params: dict, free_krw: float) -> float:
        """
        현물 전용 자본 분배 로직. 
        가용 KRW 잔고 * RISK_PER_TRADE 비율만큼 매수 금액 산출.
        """
        risk_pct = float(params.get('RISK_PER_TRADE', 0.2))
        target_krw = free_krw * risk_pct
        
        # 최대 투자 금액 제한 (안전장치)
        target_krw = min(target_krw, MAX_INVEST_CAP_KRW)
        
        # 업비트 최소 주문 금액 (보수적으로 5100원 세팅)
        if target_krw < MIN_ORDER_VALUE_KRW:
            logger.warning(f"⚠️ Calculated order value ({target_krw:.0f} KRW) is below minimum ({MIN_ORDER_VALUE_KRW} KRW).")
            return 0.0
            
        qty = target_krw / current_price
        return qty

    @network_api_retry
    def _place_order_safe(self, symbol: str, side: str, qty: float):
        try:
            ccxt_symbol = symbol
            if '-' in ccxt_symbol:
                parts = ccxt_symbol.split('-')
                if parts[0] == 'KRW':
                    ccxt_symbol = f"{parts[1]}/{parts[0]}"
            
            # Upbit specific handling for market orders
            if side.lower() == 'buy':
                # Market BUY on Upbit requires 'cost' (KRW value), not amount
                # CCXT Upbit implementation usually handles 'createMarketBuyOrderWithCost'
                # but the safest standard way is passing cost in params
                current_price = self.client.get_market_price(symbol)
                cost = qty * current_price
                cost = max(cost, MIN_ORDER_VALUE_KRW)
                
                order = self.client.exchange.create_order(
                    symbol=ccxt_symbol,
                    type='market',
                    side='buy',
                    amount=qty, # Some ccxt versions require this even if ignored
                    price=None,
                    params={'cost': cost}
                )
            else:
                # Market SELL on Upbit requires 'amount' (Coin qty)
                order = self.client.exchange.create_order(
                    symbol=ccxt_symbol,
                    type='market',
                    side='sell',
                    amount=qty
                )
                
            self.logger.info(f"⚡ Upbit Order Placed: market {side} {qty} {ccxt_symbol}")
            return order
        except Exception as e:
            self.logger.error(f"❌ Upbit Order Failed for {symbol}: {e}")
            return None

    def initialize(self):
        logger.info("🤖 RealTrader Spot (Upbit) Bot Initializing...")
        self._sync_server_time_offset(force=True)
        
        self.load_strategies_from_json()
        gc.collect()
        
        try:
            total_krw, free_krw = self._fetch_balance_safe()
            logger.info(f"💰 Upbit Account Balance: {free_krw:,.0f} KRW (Total: {total_krw:,.0f} KRW)")
            
            if free_krw < MIN_ORDER_VALUE_KRW:
                logger.warning(f"⚠️ Warning: Low KRW balance (< {MIN_ORDER_VALUE_KRW} KRW)! Trading might fail.")
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")
            
        self.health_manager.update_heartbeat(status="initialized")
        logger.info("🚀 Spot Initialization Complete. Bot is Running...")

    def execute_logic(self, symbol: str):
        try:
            if symbol not in self.params_map or symbol not in self.strategies:
                return

            params = self.params_map[symbol]
            strategy = self.strategies[symbol]
            execution_tf = str(params.get('TIMEFRAME', '4h'))
            indicator_tf = str(params.get('INDICATOR_TIMEFRAME', '1d'))

            current_price = self._get_market_price_safe(symbol)
            if current_price is None: return

            pos = self._get_current_position(symbol, current_price)
            amount = pos['amount']
            in_position = amount > 0

            # State Cleanup if flat
            stale_state = self.state_manager.get_symbol_state(symbol)
            if not in_position and stale_state and stale_state.get('entry_time'):
                logger.info(f"🧹 [{symbol}] Clearing stale local state (no active spot holdings).")
                self.state_manager.clear_symbol_state(symbol)

            current_slot = self._get_candle_slot_id(indicator_tf)
            already_calculated = self.last_calc_candle.get(symbol) == current_slot
            
            cached = self._get_cached_indicators(symbol)
            required_keys = ('trend_direction', 'atr', 'entry_upper', 'strength_filter')
            need_calculation = not cached or any(key not in cached for key in required_keys) or not already_calculated

            if need_calculation:
                tf_min = self._timeframe_to_minutes(indicator_tf)
                limit = 300
                lookback_days = (limit * tf_min) / 1440
                start_dt = datetime.utcnow() - timedelta(days=lookback_days + 2)
                start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                
                df = self._fetch_ohlcv_safe(symbol, indicator_tf, start_str)
                if df is None or len(df) < 50:
                    self.last_calc_candle[symbol] = current_slot
                    return
                
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(np.float64)
                df = strategy.generate_signals(df)
                last_candle = self._select_last_closed_candle(df, indicator_tf)
                
                if last_candle is not None:
                    self._cache_indicators(symbol, {
                        'trend_direction': int(last_candle.get('trend_direction', 0)),
                        'atr': float(last_candle.get('atr', 0.0)),
                        'entry_upper': float(last_candle.get('entry_upper', 0.0)),
                        'strength_filter': int(last_candle.get('strength_filter', 0)),
                        'indicator_timeframe': indicator_tf,
                    })
                    self.last_calc_candle[symbol] = current_slot
            
            cached = self._get_cached_indicators(symbol)
            trend_dir = cached.get('trend_direction', 0)
            atr = cached.get('atr', 0.0)
            entry_upper = cached.get('entry_upper', 999999.0)
            strength_ok = bool(cached.get('strength_filter', False))

            # --- EXIT LOGIC ---
            if in_position:
                state = self.state_manager.get_symbol_state(symbol)
                entry_price = float(state.get('entry_price', pos['entryPrice']))
                if entry_price <= 0: entry_price = current_price
                
                highest = float(state.get('highest_price', current_price))
                if current_price > highest:
                    highest = current_price
                    self.state_manager.update_symbol_state(symbol, {'highest_price': highest})
                
                pos_atr = float(state.get('entry_atr', atr))
                long_trail_mult = float(params.get('LONG_TRAIL_MULT', 5.0))
                long_tp_mult = float(params.get('LONG_TP_MULT', 10.0))
                
                stop_price = float(state.get('active_stop_price', entry_price * 0.9))
                
                # Update Trailing Stop
                new_stop = highest - (pos_atr * long_trail_mult)
                if new_stop > stop_price:
                    stop_price = new_stop
                    self.state_manager.update_symbol_state(symbol, {'active_stop_price': stop_price})
                
                tp_price = entry_price + (pos_atr * long_tp_mult)
                
                exit_triggered = False
                reason = ""
                
                if current_price <= stop_price:
                    exit_triggered = True
                    reason = f"Trailing Stop ({stop_price:,.0f})"
                elif current_price >= tp_price:
                    exit_triggered = True
                    reason = f"Take Profit ({tp_price:,.0f})"

                self._log_throttled("info", f"{symbol}:pos", 
                    f"📊 [{symbol}] Holding {amount:.4f} | Cur: {current_price:,.0f} | "
                    f"Stop: {stop_price:,.0f} | TP: {tp_price:,.0f}", 120.0)

                if exit_triggered:
                    logger.warning(f"🚨 [{symbol}] Spot Exit Triggered: {reason}. Selling {amount}...")
                    order = self._place_order_safe(symbol, 'sell', amount)
                    if order:
                        self.trade_db.record_trade(symbol, 'LONG', 'EXIT', amount, current_price, reason, {})
                        self.state_manager.clear_symbol_state(symbol)

            # --- ENTRY LOGIC ---
            elif not in_position:
                self._log_throttled("info", f"{symbol}:scan", f"ℹ️ [{symbol}] Scanning for entry...", 180.0)
                
                if trend_dir == 1 and strength_ok and current_price > entry_upper:
                    _, free_krw = self._fetch_balance_safe()
                    qty = self._calculate_spot_position_size(current_price, params, free_krw)
                    
                    if qty > 0:
                        logger.info(f"🟢 [{symbol}] Bull Breakout Detected! Buying {qty:.4f} @ {current_price:,.0f}")
                        order = self._place_order_safe(symbol, 'buy', qty)
                        
                        if order:
                            self.state_manager.update_symbol_state(symbol, {
                                'entry_time': datetime.utcnow().isoformat(),
                                'entry_price': current_price,
                                'entry_atr': atr,
                                'side': 'LONG',
                                'highest_price': current_price,
                                'active_stop_price': current_price - (atr * float(params.get('LONG_ATR_MULT', 2.0)))
                            })
                            self.trade_db.record_trade(symbol, 'LONG', 'ENTRY', qty, current_price, "Bull Breakout", {})

        except Exception as e:
            logger.error(f"🚨 Error executing spot logic for {symbol}: {e}")
            self.health_manager.record_error(e)

    def run(self):
        try:
            self.initialize()
            logger.info(f"▶️ Starting Main Loop (Interval: {SPOT_LOOP_INTERVAL_SECONDS}s)")
            
            while not self._shutdown_requested:
                cycle_start = time.time()
                try:
                    for symbol in self.symbols:
                        if self._shutdown_requested: break
                        self.execute_logic(symbol)
                        time.sleep(SPOT_SYMBOL_DELAY_SECONDS)
                    
                    self.health_manager.update_heartbeat()
                    
                except Exception as loop_e:
                    logger.error(f"💥 Unexpected error in Main Loop cycle: {loop_e}")
                    time.sleep(ERROR_SLEEP_SECONDS)
                
                elapsed = time.time() - cycle_start
                sleep_time = max(0.5, SPOT_LOOP_INTERVAL_SECONDS - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.critical(f"💥 Critical Failure during Initialization/Run: {e}")
        finally:
            logger.info("🛑 Bot Stopped.")

if __name__ == "__main__":
    bot = RealTraderSpot()
    bot.run()
