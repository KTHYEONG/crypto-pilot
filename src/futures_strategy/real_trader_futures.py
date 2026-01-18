"""
RealTrader Futures - 24시간 자동 선물 트레이딩 봇 (Production Grade)
===================================================================
P0/P1 개선사항 적용:
- 거래 기록 DB 영속화
- API 재시도 데코레이터 (tenacity)
- Health Check 메커니즘
- Graceful Shutdown (SIGTERM)
- 중복 코드 제거 (유틸 함수)
- 매직 넘버 → settings 이동
- 캔들 마감 동기화
- Structured JSON 로깅
"""

import os
import sys
import time
import signal
import json
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import RotatingFileHandler
from functools import wraps
from typing import Optional, Dict, Any, Tuple

# tenacity for retry logic
try:
    from tenacity import (
        retry, 
        stop_after_attempt, 
        wait_exponential, 
        retry_if_exception_type,
        before_sleep_log
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())

from config.settings import (
    BINANCE_API_KEY, 
    BINANCE_SECRET,
    LOG_DIR,
    FUTURES_STRATEGY_DB,
    TRADE_HISTORY_DB,
    HEARTBEAT_FILE,
    API_RETRY_ATTEMPTS,
    API_RETRY_WAIT_MIN,
    API_RETRY_WAIT_MAX,
    MIN_BALANCE_USDT,
    MIN_BALANCE_FOR_TRADE,
    MIN_ORDER_VALUE_USDT,
    MAX_EXCHANGE_LEVERAGE,
    LOOP_INTERVAL_SECONDS,
    SYMBOL_DELAY_SECONDS,
    ERROR_SLEEP_SECONDS,
    CANDLE_SYNC_OFFSET_SECONDS,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    FUTURES_TARGET_SYMBOLS,
    OPTUNA_STUDY_NAMES,
    SYMBOL_ALLOCATION_WEIGHTS,
)
from src.futures_strategy.binance_client import BinanceClient
from src.futures_strategy.strategies_futures import UltimateStrategy

# Oracle Cloud 최적화 (선택적)
try:
    from src.futures_strategy.oracle_cloud_optimizer import OracleCloudOptimizer
    ORACLE_OPTIMIZER_AVAILABLE = True
except ImportError:
    ORACLE_OPTIMIZER_AVAILABLE = False

# ============================================================
# Structured JSON Logger
# ============================================================
class JSONFormatter(logging.Formatter):
    """JSON 형식 로그 포맷터 (외부 모니터링 연동용)"""
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, 'extra_data'):
            log_obj["data"] = record.extra_data
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logger(name: str, log_prefix: str = None) -> logging.Logger:
    """
    통합 로거 설정 (동적 로그 파일명)
    
    Args:
        name: 로거 이름 (예: "RealTraderFutures", "RealTraderSpot")
        log_prefix: 로그 파일 접두사 (None이면 name을 snake_case로 변환)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    # 로그 파일명 자동 생성 (예: RealTraderSpot -> real_trader_spot)
    if log_prefix is None:
        import re
        # CamelCase to snake_case
        log_prefix = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    
    # 파일 핸들러 (회전 로그 - Plain Text)
    log_file = LOG_DIR / f"{log_prefix}.log"
    file_handler = RotatingFileHandler(
        str(log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # JSON 로그 파일 (모니터링 도구 연동용)
    json_log_file = LOG_DIR / f"{log_prefix}.jsonl"
    json_handler = RotatingFileHandler(
        str(json_log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    json_handler.setFormatter(JSONFormatter())
    
    # 콘솔 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(json_handler)
    logger.addHandler(stream_handler)
    
    return logger


logger = setup_logger("RealTraderFutures")

# ============================================================
# Retry Decorator (API 재시도)
# ============================================================
def create_retry_decorator():
    """tenacity 기반 재시도 데코레이터 생성"""
    if TENACITY_AVAILABLE:
        return retry(
            stop=stop_after_attempt(API_RETRY_ATTEMPTS),
            wait=wait_exponential(
                multiplier=1, 
                min=API_RETRY_WAIT_MIN, 
                max=API_RETRY_WAIT_MAX
            ),
            retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
    else:
        # Fallback: 단순 재시도 데코레이터
        def fallback_retry(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_error = None
                for attempt in range(API_RETRY_ATTEMPTS):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        wait_time = min(
                            API_RETRY_WAIT_MIN * (2 ** attempt), 
                            API_RETRY_WAIT_MAX
                        )
                        logger.warning(f"⚠️ Retry {attempt+1}/{API_RETRY_ATTEMPTS}: {e}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                raise last_error
            return wrapper
        return fallback_retry


api_retry = create_retry_decorator()

# ============================================================
# Trade History DB Manager
# ============================================================
class TradeHistoryDB:
    """거래 기록 영속화 매니저"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """거래 기록 테이블 생성 (WAL 모드 활성화)"""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            # WAL 모드 활성화 (동시 읽기/쓰기 성능 향상)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")  # 성능 최적화
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    entry_price REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    reason TEXT,
                    params_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp 
                ON trades(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol 
                ON trades(symbol)
            """)
            conn.commit()
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        action: str,  # 'ENTRY' or 'EXIT'
        quantity: float,
        price: float,
        entry_price: float = None,
        pnl: float = None,
        pnl_pct: float = None,
        reason: str = None,
        params: dict = None
    ):
        """거래 기록 저장 (동시 접근 대응)"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                    conn.execute("""
                        INSERT INTO trades 
                        (timestamp, symbol, side, action, quantity, price, 
                         entry_price, pnl, pnl_pct, reason, params_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.utcnow().isoformat(),
                        symbol,
                        side,
                        action,
                        quantity,
                        price,
                        entry_price,
                        pnl,
                        pnl_pct,
                        reason,
                        json.dumps(params) if params else None
                    ))
                    conn.commit()
                logger.info(f"📝 Trade recorded: {action} {side} {quantity} {symbol} @ {price}")
                break  # Success
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"⚠️ DB locked, retrying in {wait_time}s... ({attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ Failed to record trade after {max_retries} attempts: {e}")
                else:
                    logger.error(f"❌ Failed to record trade: {e}")
                    break
            except Exception as e:
                logger.error(f"❌ Failed to record trade: {e}")
                break
    
    def get_recent_trades(self, symbol: str = None, limit: int = 100) -> list:
        """최근 거래 조회"""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                    (symbol, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(row) for row in rows]


# ============================================================
# Health Check Manager
# ============================================================
class HealthCheckManager:
    """봇 생존 확인 매니저"""
    
    def __init__(self, heartbeat_file: Path):
        self.heartbeat_file = heartbeat_file
        self.start_time = datetime.utcnow()
        self.loop_count = 0
        self.last_error = None
    
    def update_heartbeat(
        self, 
        status: str = "running", 
        positions: dict = None,
        extra: dict = None
    ):
        """하트비트 파일 업데이트"""
        self.loop_count += 1
        heartbeat_data = {
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "loop_count": self.loop_count,
            "last_error": str(self.last_error) if self.last_error else None,
            "positions": positions or {},
            "pid": os.getpid(),
        }
        if extra:
            heartbeat_data.update(extra)
        
        try:
            with open(self.heartbeat_file, 'w', encoding='utf-8') as f:
                json.dump(heartbeat_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Failed to update heartbeat: {e}")
    
    def record_error(self, error: Exception):
        """에러 기록"""
        self.last_error = error


# ============================================================
# Utility Functions (중복 코드 제거)
# ============================================================
def parse_balance(ret: Any) -> float:
    """
    BinanceClient.fetch_balance() 반환값 파싱
    다양한 반환 형식 처리 (dict, tuple)
    """
    usdt_free = 0.0
    
    # Case A: Dictionary (Standard)
    if isinstance(ret, dict):
        if 'USDT' in ret:
            val = ret['USDT']
            if isinstance(val, dict):
                usdt_free = val.get('free', 0.0)
            else:
                usdt_free = float(val)
        elif 'free' in ret and isinstance(ret['free'], dict):
            usdt_free = ret['free'].get('USDT', 0.0)
    
    # Case B: Tuple based (Custom implementation)
    elif isinstance(ret, tuple):
        if len(ret) >= 2:
            free_part = ret[1]
            if isinstance(free_part, dict):
                usdt_free = free_part.get('USDT', 0.0)
            elif isinstance(free_part, (int, float)):
                usdt_free = float(free_part)
    
    return float(usdt_free)


def calculate_candle_wait_time(timeframe: str) -> int:
    """
    다음 캔들 마감까지 대기 시간 계산 (초)
    정확한 봉 마감 시점에 로직 실행
    """
    now = datetime.utcnow()
    
    # 타임프레임별 분 단위 변환
    tf_minutes = 60  # default 1h
    if 'm' in timeframe:
        tf_minutes = int(timeframe.replace('m', ''))
    elif 'h' in timeframe:
        tf_minutes = int(timeframe.replace('h', '')) * 60
    elif 'd' in timeframe:
        tf_minutes = int(timeframe.replace('d', '')) * 1440
    
    # 현재 시간을 분 단위로 변환
    current_minutes = now.hour * 60 + now.minute
    
    # 다음 봉 마감 시점 계산
    next_candle_minutes = ((current_minutes // tf_minutes) + 1) * tf_minutes
    
    # 자정 넘어가는 경우 처리
    if next_candle_minutes >= 1440:
        next_candle_minutes = next_candle_minutes % 1440
        next_candle = (now + timedelta(days=1)).replace(
            hour=next_candle_minutes // 60,
            minute=next_candle_minutes % 60,
            second=CANDLE_SYNC_OFFSET_SECONDS,
            microsecond=0
        )
    else:
        next_candle = now.replace(
            hour=next_candle_minutes // 60,
            minute=next_candle_minutes % 60,
            second=CANDLE_SYNC_OFFSET_SECONDS,
            microsecond=0
        )
    
    wait_seconds = (next_candle - now).total_seconds()
    
    # 이미 지났으면 다음 주기로
    if wait_seconds < 0:
        wait_seconds += tf_minutes * 60
    
    return int(wait_seconds)


# ============================================================
# Main Trader Class
# ============================================================
class RealTraderFutures:
    """Production-grade 선물 트레이딩 봇"""
    
    def __init__(self, db_path: str = None, enable_oracle_optimization: bool = False):
        self.client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
        self.db_path = db_path or str(FUTURES_STRATEGY_DB)
        self.strategies: Dict[str, UltimateStrategy] = {}
        self.params_map: Dict[str, dict] = {}
        self.symbols: list = []
        
        # 신규 컴포넌트
        self.trade_db = TradeHistoryDB(TRADE_HISTORY_DB)
        self.health_manager = HealthCheckManager(HEARTBEAT_FILE)
        
        # Oracle Cloud 최적화 (옵션)
        self.oracle_optimizer = None
        if enable_oracle_optimization and ORACLE_OPTIMIZER_AVAILABLE:
            self.oracle_optimizer = OracleCloudOptimizer()
            logger.info("☁️ Oracle Cloud optimization enabled")
        
        # Shutdown 플래그
        self._shutdown_requested = False
        
        # Signal handlers 등록
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Graceful Shutdown 시그널 핸들러 등록"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}. Initiating graceful shutdown...")
            self._shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        # Windows에서는 SIGBREAK도 처리
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def load_strategies_from_db(self):
        """Optuna DB에서 최적화된 파라미터 로드"""
        logger.info(f"📂 Loading strategies from {self.db_path}...")
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"DB file not found: {self.db_path}")
        
        import optuna
        storage = f"sqlite:///{self.db_path}"
        
        study = None
        for s_name in OPTUNA_STUDY_NAMES:
            try:
                study = optuna.load_study(study_name=s_name, storage=storage)
                logger.info(f"✅ Loaded Study: '{s_name}' (Score: {study.best_value:.4f})")
                break
            except KeyError:
                continue
        
        if study is None:
            raise ValueError(f"No valid study found in DB. Tried: {OPTUNA_STUDY_NAMES}")
        
        best_params = study.best_params
        self.symbols = FUTURES_TARGET_SYMBOLS.copy()
        
        for symbol in self.symbols:
            self.params_map[symbol] = best_params.copy()
            strategy_name = f"Real_{symbol.replace('/', '_')}"
            self.strategies[symbol] = UltimateStrategy(strategy_name, best_params)
            logger.info(f"🔹 Strategy initialized: {symbol} | TF: {best_params.get('TIMEFRAME')}")
    
    @api_retry
    def _fetch_balance_safe(self) -> float:
        """안전한 잔고 조회 (재시도 적용)"""
        ret = self.client.fetch_balance()
        return parse_balance(ret)
    
    @api_retry
    def _fetch_ohlcv_safe(self, symbol: str, timeframe: str, start_str: str):
        """안전한 OHLCV 조회 (재시도 적용)"""
        return self.client.fetch_ohlcv(symbol, timeframe, start_date=start_str)
    
    @api_retry
    def _fetch_position_safe(self, symbol: str) -> dict:
        """안전한 포지션 조회 (재시도 적용)"""
        return self.client.fetch_position(symbol)
    
    @api_retry
    def _get_market_price_safe(self, symbol: str) -> float:
        """안전한 시장가 조회 (재시도 적용)"""
        return self.client.get_market_price(symbol)
    
    @api_retry
    def _place_order_safe(self, symbol: str, side: str, qty: float):
        """안전한 주문 실행 (재시도 적용)"""
        return self.client.place_order(symbol, side, qty)
    
    def initialize(self):
        """초기화: 전략 로드, 레버리지 설정, 잔고 확인"""
        logger.info("🤖 RealTrader Futures Bot Initializing...")
        
        # 1. 전략 로드
        self.load_strategies_from_db()
        
        # 2. 잔고 확인
        try:
            usdt_free = self._fetch_balance_safe()
            logger.info(f"💰 Account Balance: {usdt_free:.2f} USDT")
            
            if usdt_free < MIN_BALANCE_USDT:
                logger.warning(f"⚠️ Warning: Low balance (< {MIN_BALANCE_USDT} USDT)!")
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")
        
        # 3. 레버리지 설정
        for symbol in self.symbols:
            try:
                success = self.client.set_leverage(symbol, MAX_EXCHANGE_LEVERAGE)
                target_lev = self.params_map[symbol].get('LEVERAGE', 1)
                if success:
                    logger.info(
                        f"✅ Exchange Leverage: {MAX_EXCHANGE_LEVERAGE}x for {symbol} "
                        f"(Strategy Target: {target_lev}x)"
                    )
            except Exception as e:
                logger.error(f"⚠️ Error setting leverage for {symbol}: {e}")
        
        # 4. 초기 헬스체크
        self.health_manager.update_heartbeat(status="initialized")
        
        logger.info("🚀 Initialization Complete. Bot is Running...")
    
    def execute_logic(self, symbol: str):
        """핵심 매매 로직 실행"""
        try:
            params = self.params_map[symbol]
            strategy = self.strategies[symbol]
            
            # 1. 데이터 조회
            timeframe = params.get('TIMEFRAME', '1h')
            limit = 500
            
            tf_min = 60
            if 'm' in timeframe:
                tf_min = int(timeframe.replace('m', ''))
            elif 'h' in timeframe:
                tf_min = int(timeframe.replace('h', '')) * 60
            elif 'd' in timeframe:
                tf_min = int(timeframe.replace('d', '')) * 1440
            
            lookback_days = (limit * tf_min) / 1440
            start_dt = datetime.now() - timedelta(days=lookback_days + 2)
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            
            df = self._fetch_ohlcv_safe(symbol, timeframe, start_str)
            
            if df is None or len(df) < limit * 0.9:
                logger.warning(
                    f"⚠️ Insufficient data for {symbol}. "
                    f"Needed ~{limit}, Got {len(df) if df is not None else 0}"
                )
                return
            
            # 2. 지표 계산
            df = strategy.generate_signals(df)
            
            # 3. 신호 확인 (-2: 확정된 마지막 봉)
            last_candle = df.iloc[-2]
            current_price = self._get_market_price_safe(symbol)
            
            if current_price is None:
                logger.warning(f"⚠️ Failed to get price for {symbol}")
                return
            
            entry_upper = last_candle.get('entry_upper')
            entry_lower = last_candle.get('entry_lower')
            trend_dir = last_candle.get('trend_direction', 0)
            strength_ok = (last_candle.get('strength_filter', 1) == 1)
            atr = last_candle.get('atr', 0.0)
            sar = last_candle.get('parabolic_sar', 0.0)
            
            # NaN 체크 강화
            if pd.isna(atr):
                atr = 0.0
            if pd.isna(sar):
                sar = 0.0
            
            # 4. 현재 포지션
            pos = self._fetch_position_safe(symbol)
            amount = float(pos['amount'])
            in_position = abs(amount) > 0
            
            # --- EXIT LOGIC ---
            if in_position:
                self._check_exit(
                    symbol, amount, current_price, params, pos, 
                    trend_dir, atr, sar
                )
            
            # --- ENTRY LOGIC ---
            elif not in_position and strength_ok:
                if pd.isna(entry_upper) or pd.isna(entry_lower):
                    return
                
                # LONG 진입
                if trend_dir == 1 and current_price > entry_upper:
                    logger.info(
                        f"🟢 ENTRY LONG Signal {symbol} | "
                        f"Price {current_price} > Upper {entry_upper:.2f}"
                    )
                    qty = self._calculate_position_size(symbol, current_price, params, atr)
                    if qty > 0:
                        order = self._place_order_safe(symbol, 'buy', qty)
                        if order:
                            self.trade_db.record_trade(
                                symbol=symbol,
                                side='LONG',
                                action='ENTRY',
                                quantity=qty,
                                price=current_price,
                                reason=f"Price > Upper ({entry_upper:.2f})",
                                params={'timeframe': timeframe, 'atr': atr}
                            )
                
                # SHORT 진입
                elif trend_dir == -1 and current_price < entry_lower:
                    logger.info(
                        f"🔴 ENTRY SHORT Signal {symbol} | "
                        f"Price {current_price} < Lower {entry_lower:.2f}"
                    )
                    qty = self._calculate_position_size(symbol, current_price, params, atr)
                    if qty > 0:
                        order = self._place_order_safe(symbol, 'sell', qty)
                        if order:
                            self.trade_db.record_trade(
                                symbol=symbol,
                                side='SHORT',
                                action='ENTRY',
                                quantity=qty,
                                price=current_price,
                                reason=f"Price < Lower ({entry_lower:.2f})",
                                params={'timeframe': timeframe, 'atr': atr}
                            )
        
        except Exception as e:
            logger.error(f"🚨 Error executing logic for {symbol}: {e}")
            self.health_manager.record_error(e)
    
    def _check_exit(
        self, 
        symbol: str, 
        amount: float, 
        current_price: float, 
        params: dict, 
        pos: dict, 
        trend_dir: int, 
        atr: float, 
        sar: float
    ):
        """청산 로직"""
        try:
            exit_triggered = False
            reason = ""
            
            use_tp = params.get('USE_TAKE_PROFIT', False)
            tp_atr_mult = params.get(
                'TAKE_PROFIT_ATR_MULT_FUTURES', 
                params.get('TAKE_PROFIT_ATR_MULT', 3.0)
            )
            entry_price = float(pos.get('entryPrice', 0))
            
            if amount > 0:  # LONG
                # 1. Parabolic SAR Exit
                if params.get('EXIT_TYPE') == 'PARABOLIC_SAR':
                    if sar > 0 and current_price < sar:
                        exit_triggered = True
                        reason = "Parabolic SAR Cross"
                
                # 2. Trend Reversal
                if trend_dir == -1:
                    exit_triggered = True
                    reason = "Trend Reversal"
                
                # 3. Take Profit
                if use_tp and entry_price > 0 and atr > 0:
                    tp_price = entry_price + (atr * tp_atr_mult)
                    if current_price >= tp_price:
                        exit_triggered = True
                        reason = "Take Profit"
                
                if exit_triggered:
                    pnl = (current_price - entry_price) * abs(amount)
                    pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                    
                    logger.info(
                        f"🛑 EXIT LONG {symbol} | Price: {current_price} | "
                        f"PnL: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {reason}"
                    )
                    order = self._place_order_safe(symbol, 'sell', abs(amount))
                    if order:
                        self.trade_db.record_trade(
                            symbol=symbol,
                            side='LONG',
                            action='EXIT',
                            quantity=abs(amount),
                            price=current_price,
                            entry_price=entry_price,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            reason=reason
                        )
            
            elif amount < 0:  # SHORT
                # 1. Parabolic SAR Exit
                if params.get('EXIT_TYPE') == 'PARABOLIC_SAR':
                    if sar > 0 and current_price > sar:
                        exit_triggered = True
                        reason = "Parabolic SAR Cross"
                
                # 2. Trend Reversal
                if trend_dir == 1:
                    exit_triggered = True
                    reason = "Trend Reversal"
                
                # 3. Take Profit
                if use_tp and entry_price > 0 and atr > 0:
                    tp_price = entry_price - (atr * tp_atr_mult)
                    if current_price <= tp_price:
                        exit_triggered = True
                        reason = "Take Profit"
                
                if exit_triggered:
                    pnl = (entry_price - current_price) * abs(amount)
                    pnl_pct = ((entry_price / current_price) - 1) * 100 if current_price > 0 else 0
                    
                    logger.info(
                        f"🛑 EXIT SHORT {symbol} | Price: {current_price} | "
                        f"PnL: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {reason}"
                    )
                    order = self._place_order_safe(symbol, 'buy', abs(amount))
                    if order:
                        self.trade_db.record_trade(
                            symbol=symbol,
                            side='SHORT',
                            action='EXIT',
                            quantity=abs(amount),
                            price=current_price,
                            entry_price=entry_price,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            reason=reason
                        )
        
        except Exception as e:
            logger.error(f"⚠️ Error in _check_exit: {e}")
            self.health_manager.record_error(e)
    
    def _calculate_position_size(
        self, 
        symbol: str, 
        price: float, 
        params: dict, 
        atr: float = 0.0
    ) -> float:
        """포지션 사이즈 계산 (성과 기반 가중치 적용)"""
        try:
            # Total Balance(총 자산)와 Free Balance(가용 잔고) 모두 필요
            total_balance, usdt_free = self._fetch_balance_safe()
        except Exception:
            return 0
        
        if usdt_free < MIN_BALANCE_FOR_TRADE:
            logger.warning(f"⚠️ Insufficient capital for {symbol}: ${usdt_free:.2f}")
            return 0
        
        # 1. 성과 기반 가중치 적용 (BTC 75%, ETH 25%)
        # 설정된 가중치가 없으면 균등 배분 가정 (1 / 심볼 수)
        default_weight = 1.0 / len(self.symbols) if self.symbols else 0.5
        allocation_weight = SYMBOL_ALLOCATION_WEIGHTS.get(symbol, default_weight)
        
        # 2. 할당된 자본금 (Allocated Capital)
        # 이 심볼이 운용할 수 있는 이론적 총 자본금
        allocated_capital = total_balance * allocation_weight
        
        leverage = params.get('LEVERAGE', 1)
        risk_per_trade = params.get(
            'RISK_PER_TRADE_FUTURES', 
            params.get('RISK_PER_TRADE', 0.02)
        )
        
        # 3. Stop Loss Distance 계산
        stop_distance_pct = 0.05  # Default Fallback
        
        if atr > 0 and price > 0:
            atr_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
            stop_distance = atr * atr_mult
            stop_distance_pct = stop_distance / price
            stop_distance_pct = max(0.005, min(stop_distance_pct, 0.10))
        
        # 4. Sizing Calculation (Allocated Capital 기준 리스크)
        # 예: 총자산 1000불, BTC(75%) -> 750불 할당
        # Risk 2% -> 15불 리스크 감수
        risk_amt = allocated_capital * risk_per_trade
        notional_value = risk_amt / stop_distance_pct
        
        # 5. 최대 허용 Notional (가용 잔고 제약)
        # 실제로 주문 가능한 금액은 (가용 잔고 * 레버리지)를 넘을 수 없음
        max_tradeable_notional = usdt_free * leverage
        
        # 6. 최종 Notional (할당된 리스크와 실제 가용액 중 작은 것)
        final_notional = min(notional_value, max_tradeable_notional)
        
        # 최소 주문 금액 확인 체크 미리 하기
        if final_notional < MIN_ORDER_VALUE_USDT:
             return 0

        quantity = final_notional / price
        
        # Precision Adjustment
        if 'BTC' in symbol:
            quantity = float(int(quantity * 1000) / 1000)
        elif 'ETH' in symbol:
            quantity = float(int(quantity * 100) / 100)
        else:
            quantity = float(int(quantity * 10) / 10)
        
        if quantity * price < MIN_ORDER_VALUE_USDT:
            return 0
        
        logger.info(
            f"🧮 Sizing {symbol} (Weight {allocation_weight*100:.0f}%): "
            f"Total ${total_balance:.0f} | Alloc ${allocated_capital:.0f}"
        )
        logger.info(
            f"   -> Risk: ${risk_amt:.1f} ({risk_per_trade*100}% of Alloc) | "
            f"StopDist {stop_distance_pct*100:.1f}%"
        )
        logger.info(
            f"   -> Target Size: {notional_value:.1f} USDT | "
            f"Final: {final_notional:.1f} USDT ({quantity} coins)"
        )
        
        return quantity
    
    def _get_current_positions(self) -> dict:
        """현재 포지션 상태 조회 (헬스체크용)"""
        positions = {}
        for symbol in self.symbols:
            try:
                pos = self._fetch_position_safe(symbol)
                if abs(pos['amount']) > 0:
                    positions[symbol] = {
                        'amount': pos['amount'],
                        'entryPrice': pos['entryPrice'],
                        'unrealizedPnL': pos['unrealizedPnL']
                    }
            except Exception:
                pass
        return positions
    
    def run_forever(self):
        """메인 무한 루프 (Graceful Shutdown 지원)"""
        try:
            self.initialize()
        except Exception as e:
            logger.error(f"🚨 Initialization failed: {e}")
            self.health_manager.update_heartbeat(status="init_failed")
            raise
        
        logger.info("⏳ Waiting for next candle close...")
        
        while not self._shutdown_requested:
            try:
                # 각 심볼 처리
                for symbol in self.symbols:
                    if self._shutdown_requested:
                        break
                    self.execute_logic(symbol)
                    time.sleep(SYMBOL_DELAY_SECONDS)
                
                # 헬스체크 업데이트
                positions = self._get_current_positions()
                self.health_manager.update_heartbeat(
                    status="running",
                    positions=positions
                )
                
                # Oracle Cloud 최적화 실행
                if self.oracle_optimizer:
                    # 1. Idle 방지 (CPU 사용률 증가)
                    self.oracle_optimizer.prevent_idle_shutdown(duration_seconds=3)
                    
                    # 2. 시간 동기화 검증
                    if not self.oracle_optimizer.check_time_sync():
                        logger.error("⏰ Time drift detected! Consider resyncing NTP.")
                    
                    # 3. 리소스 모니터링 (10분마다)
                    if self.health_manager.loop_count % 20 == 0:
                        self.oracle_optimizer.log_resource_usage()
                    
                    # 4. DB 정리 (24시간마다, 90일 이상 오래된 거래 삭제)
                    if self.health_manager.loop_count % 2880 == 0:
                        self.oracle_optimizer.cleanup_db_old_records(
                            TRADE_HISTORY_DB, 
                            days_to_keep=90
                        )
                    
                    # 5. 명시적 GC (1시간마다)
                    if self.health_manager.loop_count % 120 == 0:
                        self.oracle_optimizer.force_gc()
                
                # 캔들 동기화 대기 (옵션)
                # 첫 번째 심볼의 타임프레임 기준
                if self.symbols and self.params_map:
                    tf = self.params_map[self.symbols[0]].get('TIMEFRAME', '1h')
                    wait_time = calculate_candle_wait_time(tf)
                    
                    # 최소 대기 시간 적용 (너무 짧으면 기본 간격 사용)
                    if wait_time < LOOP_INTERVAL_SECONDS:
                        wait_time = LOOP_INTERVAL_SECONDS
                    
                    # 최대 대기 시간 제한 (1시간)
                    wait_time = min(wait_time, 3600)
                    
                    logger.info(f"💤 Next execution in {wait_time}s...")
                    
                    # Shutdown 체크하면서 대기
                    for _ in range(int(wait_time)):
                        if self._shutdown_requested:
                            break
                        time.sleep(1)
                else:
                    time.sleep(LOOP_INTERVAL_SECONDS)
            
            except Exception as e:
                logger.error(f"🚨 Critical Error in Main Loop: {e}")
                self.health_manager.record_error(e)
                self.health_manager.update_heartbeat(status="error")
                time.sleep(ERROR_SLEEP_SECONDS)
        
        # Graceful Shutdown 처리
        self._shutdown()
    
    def _shutdown(self):
        """Graceful Shutdown 처리"""
        logger.info("🛑 Shutting down gracefully...")
        
        # 현재 포지션 상태 기록
        positions = self._get_current_positions()
        if positions:
            logger.warning(f"⚠️ Open positions at shutdown: {positions}")
        
        self.health_manager.update_heartbeat(
            status="stopped",
            positions=positions,
            extra={"shutdown_time": datetime.utcnow().isoformat()}
        )
        
        logger.info("✅ Shutdown complete.")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 RealTrader Futures - Production Grade Bot")
    logger.info("=" * 60)
    
    # Oracle Cloud 환경 변수로 활성화 결정 (기본값: True)
    import os
    enable_oracle_opt = os.getenv("ENABLE_ORACLE_OPTIMIZATION", "true").lower() == "true"
    
    bot = RealTraderFutures(enable_oracle_optimization=enable_oracle_opt)
    bot.run_forever()
