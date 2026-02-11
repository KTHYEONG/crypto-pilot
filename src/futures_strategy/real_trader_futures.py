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
import gc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import RotatingFileHandler
from functools import wraps
from typing import Optional, Dict, Any, Tuple

try:
    import msvcrt  # Windows file lock
except ImportError:  # pragma: no cover - non-Windows fallback
    msvcrt = None

try:
    import fcntl  # POSIX file lock
except ImportError:  # pragma: no cover - Windows
    fcntl = None



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
    FUTURES_STATE_FILE,
    SLIPPAGE_RATE,
)
from src.futures_strategy.binance_client import BinanceClient
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.common.utils import setup_logger, api_retry
from src.common.components import TradeHistoryDB, HealthCheckManager, calculate_candle_wait_time

# Oracle Cloud 최적화 (선택적)
try:
    from src.common.cloud_optimizer import CloudOptimizer
    CLOUD_OPTIMIZER_AVAILABLE = True
except ImportError:
    CLOUD_OPTIMIZER_AVAILABLE = False

# ============================================================
# Structured JSON Logger
# ============================================================



logger = setup_logger("RealTraderFutures")



# ============================================================
# Trade History DB Manager
# ============================================================



# ============================================================
# Health Check Manager
# ============================================================



# ============================================================
# Utility Functions (중복 코드 제거)
# ============================================================



# ============================================================
# State Manager (JSON 파일 기반 - Futures 진입 시간 추적)
# ============================================================
class StateManager:
    """거래 상태 관리 (진입 시간, 진입가 등 로컬 저장)"""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.lock_file = self.state_file.with_suffix(self.state_file.suffix + ".lock")
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """상태 파일이 없으면 생성"""
        if not self.state_file.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_unlocked({})
        if not self.lock_file.exists():
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            self.lock_file.touch(exist_ok=True)

    def _acquire_lock(self):
        """
        Cross-process advisory lock for state read-modify-write consistency.
        """
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_fp = open(self.lock_file, 'a+b')
        try:
            if msvcrt is not None:
                # Lock 1 byte from start of file (blocking)
                lock_fp.seek(0)
                msvcrt.locking(lock_fp.fileno(), msvcrt.LK_LOCK, 1)
            elif fcntl is not None:  # pragma: no cover
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            return lock_fp
        except Exception:
            try:
                lock_fp.close()
            except Exception:
                pass
            raise

    def _release_lock(self, lock_fp):
        try:
            if msvcrt is not None:
                lock_fp.seek(0)
                msvcrt.locking(lock_fp.fileno(), msvcrt.LK_UNLCK, 1)
            elif fcntl is not None:  # pragma: no cover
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                lock_fp.close()
            except Exception:
                pass

    def _load_unlocked(self) -> dict:
        """상태 로드 (호출자가 파일 락 보유해야 함)"""
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"⚠️ State load error: {e}")
            return {}

    def _save_unlocked(self, state: dict):
        """상태 저장 (호출자가 파일 락 보유해야 함)"""
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
                if tmp_file.exists():
                    tmp_file.unlink()
            except Exception:
                pass

    def _load(self) -> dict:
        lock_fp = self._acquire_lock()
        try:
            return self._load_unlocked()
        finally:
            self._release_lock(lock_fp)

    def _save(self, state: dict):
        lock_fp = self._acquire_lock()
        try:
            self._save_unlocked(state)
        finally:
            self._release_lock(lock_fp)

    def get_symbol_state(self, symbol: str) -> dict:
        """특정 심볼의 상태 조회"""
        state = self._load()
        return state.get(symbol, {})

    def update_symbol_state(self, symbol: str, data: dict):
        """특정 심볼의 상태 업데이트"""
        lock_fp = self._acquire_lock()
        try:
            state = self._load_unlocked()
            if symbol not in state:
                state[symbol] = {}
            state[symbol].update(data)
            self._save_unlocked(state)
        finally:
            self._release_lock(lock_fp)

    def clear_symbol_state(self, symbol: str):
        """특정 심볼의 상태 초기화"""
        lock_fp = self._acquire_lock()
        try:
            state = self._load_unlocked()
            if symbol in state:
                state[symbol] = {}
                self._save_unlocked(state)
        finally:
            self._release_lock(lock_fp)
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
        self.symbol_allocation_weights: Dict[str, float] = {}
        
        # 신규 컴포넌트
        self.trade_db = TradeHistoryDB(TRADE_HISTORY_DB)
        self.health_manager = HealthCheckManager(HEARTBEAT_FILE)
        self.state_manager = StateManager(FUTURES_STATE_FILE)
        
        # 클라우드 최적화 (옵션)
        self.cloud_optimizer = None
        if enable_oracle_optimization and CLOUD_OPTIMIZER_AVAILABLE:
            self.cloud_optimizer = CloudOptimizer()
            logger.info("☁️ Cloud optimization enabled")
        
        # Shutdown 플래그
        self._shutdown_requested = False
        
        # [Optimization] 중복 계산 방지용 캐시
        self.last_calc_candle: Dict[str, str] = {}
        self.last_exit_calc_candle: Dict[str, str] = {}
        self._server_time_offset_ms: int = 0
        self._last_server_time_sync: datetime = datetime.min
        
        # [Cloud Optimization] 시간 기반 작업 추적 (loop_count 대신 사용)
        self._last_resource_check = datetime.utcnow()
        self._last_db_cleanup = datetime.utcnow()
        self._last_gc = datetime.utcnow()
        
        # Signal handlers 등록
        self._setup_signal_handlers()
        
        # 전략 로드는 initialize()에서 수행

    
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
        
        # [Lazy Loading] optuna is heavy, load only when needed
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
        preferred_symbols = list(FUTURES_TARGET_SYMBOLS) if FUTURES_TARGET_SYMBOLS else list(SYMBOL_ALLOCATION_WEIGHTS.keys())
        if not preferred_symbols:
            raise ValueError("No live futures symbols configured. Check FUTURES_TARGET_SYMBOLS in config/settings.py")
        self.symbols = preferred_symbols.copy()

        self.symbol_allocation_weights = self._build_symbol_allocation_weights(self.symbols)
        logger.info(
            "📌 Live symbol allocation applied: %s",
            ", ".join(f"{s}={self.symbol_allocation_weights.get(s, 0.0):.2f}" for s in self.symbols),
        )
        
        for symbol in self.symbols:
            symbol_params = best_params.copy()
            symbol_params.setdefault('INDICATOR_TIMEFRAME', '1d')
            self.params_map[symbol] = symbol_params
            strategy_name = f"Real_{symbol.replace('/', '_')}"
            self.strategies[symbol] = UltimateStrategy(strategy_name, symbol_params)
            logger.info(
                f"🔹 Strategy initialized: {symbol} | Exec TF: {symbol_params.get('TIMEFRAME', '1h')} | Indicator TF: {symbol_params.get('INDICATOR_TIMEFRAME', '1d')}"
            )

    def _build_symbol_allocation_weights(self, symbols: list) -> Dict[str, float]:
        """Build normalized allocation map for active symbols."""
        if not symbols:
            return {}

        weights: Dict[str, float] = {}
        for symbol in symbols:
            if symbol in SYMBOL_ALLOCATION_WEIGHTS:
                weights[symbol] = float(SYMBOL_ALLOCATION_WEIGHTS[symbol])
            else:
                weights[symbol] = 0.0

        total_weight = sum(w for w in weights.values() if w > 0.0)
        if total_weight <= 0.0:
            equal_weight = 1.0 / len(symbols)
            return {symbol: equal_weight for symbol in symbols}

        return {
            symbol: (weight / total_weight if weight > 0.0 else 0.0)
            for symbol, weight in weights.items()
        }
    
    @api_retry
    def _fetch_balance_safe(self) -> tuple:
        """안전한 잔고 조회 (Total, Free 반환)"""
        # BinanceClient.fetch_balance() already returns (total, free)
        return self.client.fetch_balance()
    
    @api_retry
    def _fetch_ohlcv_safe(self, symbol: str, timeframe: str, start_str: str):
        """안전한 OHLCV 조회 (재시도 적용)"""
        return self.client.fetch_ohlcv(symbol, timeframe, start_date=start_str)

    @api_retry
    def _fetch_recent_ohlcv_safe(self, symbol: str, timeframe: str, limit: int = 3):
        """Lightweight OHLCV fetch for signal timing checks."""
        return self.client.fetch_recent_ohlcv(symbol, timeframe, limit=limit)
    @api_retry
    def _fetch_position_safe(self, symbol: str) -> dict:
        """안전한 포지션 조회 (재시도 적용)"""
        return self.client.fetch_position(symbol)
    
    @api_retry
    def _get_market_price_safe(self, symbol: str) -> float:
        """안전한 시장가 조회 (재시도 적용)"""
        return self.client.get_market_price(symbol)

    @api_retry
    def _fetch_server_time_ms_safe(self) -> int:
        """Exchange server time(ms) 조회."""
        return int(self.client.fetch_server_time_ms())

    def _sync_server_time_offset(self, force: bool = False, sync_interval_seconds: int = 60):
        """로컬 UTC와 거래소 서버시간 오프셋을 주기적으로 동기화."""
        now = datetime.utcnow()
        elapsed = (now - self._last_server_time_sync).total_seconds()
        if (not force) and elapsed < max(5, int(sync_interval_seconds)):
            return
        local_ms = int(now.timestamp() * 1000)
        try:
            server_ms = self._fetch_server_time_ms_safe()
            if server_ms > 0:
                self._server_time_offset_ms = int(server_ms - local_ms)
                self._last_server_time_sync = now
        except Exception as e:
            self._last_server_time_sync = now
            logger.debug(f"[time-sync] server time sync failed: {e}")

    def _get_reference_now_ms(self) -> int:
        """거래소 기준 보정 UTC ms."""
        self._sync_server_time_offset(force=False)
        return int(datetime.utcnow().timestamp() * 1000) + int(self._server_time_offset_ms)

    def _get_reference_now_utc(self) -> datetime:
        """거래소 기준 보정 UTC datetime."""
        return datetime.utcfromtimestamp(self._get_reference_now_ms() / 1000.0)

    def _default_entry_grace_seconds(self, execution_tf: str = '1h', params: Optional[dict] = None) -> float:
        """
        Timeframe-aware default grace for live entry.
        - Low TF: keep grace short to preserve next-bar-open parity.
        - High TF: allow enough runtime slack for multi-symbol scanning.
        """
        cfg = params or {}
        tf_min = self._timeframe_to_minutes(execution_tf)
        tf_seconds = float(max(60, (tf_min * 60) if tf_min > 0 else 3600))

        grace_ratio = float(cfg.get('ENTRY_EXECUTION_GRACE_RATIO', 0.20))
        grace_min = float(cfg.get('ENTRY_EXECUTION_GRACE_MIN_SECONDS', 5.0))
        grace_max = float(cfg.get('ENTRY_EXECUTION_GRACE_MAX_SECONDS', 30.0))
        if grace_min > grace_max:
            grace_min, grace_max = grace_max, grace_min

        symbol_count = max(1, len(self.symbols))
        loop_cycle_est = float(LOOP_INTERVAL_SECONDS) + (float(SYMBOL_DELAY_SECONDS) * float(symbol_count))
        base = max(tf_seconds * grace_ratio, loop_cycle_est * 1.1)
        return max(grace_min, min(grace_max, base))

    def _resolve_entry_market_fallback(
        self,
        params: dict,
        atr: float,
        current_price: float,
        entry_lag_sec: float,
    ) -> bool:
        """
        Entry execution policy:
        - ENTRY_ALLOW_MARKET_FALLBACK가 명시되면 우선 적용
        - 아니면 ENTRY_EXECUTION_MODE로 동적 결정
        """
        explicit_flag = params.get('ENTRY_ALLOW_MARKET_FALLBACK', None)
        if explicit_flag is not None:
            return bool(explicit_flag)

        mode = str(params.get('ENTRY_EXECUTION_MODE', 'balanced')).strip().lower()
        if mode in ('maker_strict', 'strict', 'maker'):
            return False
        if mode in ('always_taker', 'taker'):
            return True

        vol_pct = 0.0
        if current_price and current_price > 0 and atr and atr > 0:
            vol_pct = (float(atr) / float(current_price)) * 100.0

        lag_trigger = float(params.get('ENTRY_MARKET_FALLBACK_LAG_SECONDS', 8.0))
        vol_trigger = float(params.get('ENTRY_MARKET_FALLBACK_VOL_PCT', 1.0))
        lag_sec = max(0.0, float(entry_lag_sec))

        if mode == 'balanced':
            return (lag_sec >= (lag_trigger * 0.6)) or (vol_pct >= (vol_trigger * 0.75))

        # default: adaptive
        return (lag_sec >= lag_trigger) or (vol_pct >= vol_trigger)

    def _enforce_min_fill_ratio(
        self,
        symbol: str,
        expected_qty: float,
        confirmed_amount: float,
        side: str,
        params: dict,
        current_price: float,
        atr: float,
    ) -> bool:
        """
        Validate minimum fill ratio.
        If underfilled, force immediate flatten to avoid state/stat distortion.
        """
        expected = abs(float(expected_qty) if expected_qty is not None else 0.0)
        actual = abs(float(confirmed_amount) if confirmed_amount is not None else 0.0)
        if expected <= 0 or actual <= 0:
            return actual > 0

        min_fill_ratio = float(params.get('ENTRY_MIN_FILL_RATIO', 0.60))
        min_fill_ratio = max(0.0, min(1.0, min_fill_ratio))
        if min_fill_ratio <= 0.0:
            return True

        fill_ratio = actual / expected
        if fill_ratio >= min_fill_ratio:
            return True

        logger.warning(
            f"⚠️ [{symbol}] Underfilled entry rejected: fill_ratio={fill_ratio:.3f} "
            f"< min_fill_ratio={min_fill_ratio:.3f} (actual={actual:.8f}, expected={expected:.8f})"
        )
        close_side = 'sell' if side == 'LONG' else 'buy'
        close_qty = self.client.round_amount(symbol, actual)
        if close_qty <= 0:
            close_qty = actual

        self._place_order_safe(
            symbol=symbol,
            side=close_side,
            qty=close_qty,
            atr=atr,
            current_price=current_price,
            reduce_only=True,
            allow_market_fallback=True,
        )
        is_flat, remaining_amount = self._wait_until_position_flat(symbol, timeout_seconds=6.0, poll_seconds=0.35)
        if not is_flat:
            logger.error(
                f"❌[{symbol}] Underfilled position flatten failed. Remaining={remaining_amount:+.8f}"
            )
        try:
            self._cancel_all_orders_safe(symbol)
        except Exception as cancel_err:
            logger.warning(f"⚠️ [{symbol}] cancel_all after underfill reject failed: {cancel_err}")

        return False

    def _position_state_missing_core(self, state: Optional[dict]) -> bool:
        """Check whether local state has minimum fields required for deterministic exits."""
        if not state:
            return True

        side = str(state.get('side', '') or '').upper()
        has_side = side in ('LONG', 'SHORT')

        try:
            entry_price = float(state.get('entry_price', 0.0) or 0.0)
        except (TypeError, ValueError):
            entry_price = 0.0
        has_entry_price = bool(np.isfinite(entry_price) and entry_price > 0.0)

        try:
            entry_target_open_ts = int(state.get('entry_target_open_ts', 0) or 0)
        except (TypeError, ValueError):
            entry_target_open_ts = 0
        has_entry_anchor = bool(entry_target_open_ts > 0 or state.get('entry_time'))

        return not (has_side and has_entry_price and has_entry_anchor)

    def _bootstrap_state_for_open_position(
        self,
        symbol: str,
        amount: float,
        pos: dict,
        current_price: float,
        params: dict,
        atr: float,
        execution_tf: str,
    ) -> bool:
        """
        Rebuild minimal local state from exchange position after restart/state loss.
        This keeps exit logic deterministic instead of silently disabling time-based exits.
        """
        try:
            side = 'LONG' if amount > 0 else 'SHORT'
            entry_price = float(pos.get('entryPrice', 0.0) or current_price or 0.0)
            if not np.isfinite(entry_price) or entry_price <= 0:
                logger.error(f"❌ [{symbol}] Cannot bootstrap state: invalid entry price {entry_price}")
                return False

            entry_atr = float(atr) if np.isfinite(atr) and atr > 0 else 0.0
            sl_type = str(params.get('STOP_LOSS_TYPE', 'FIXED') or 'FIXED').upper()
            if sl_type == 'ATR' and entry_atr > 0:
                sl_mult = float(params.get('ATR_STOP_LOSS_MULT', 1.5))
                stop_price = (
                    entry_price - (entry_atr * sl_mult)
                    if amount > 0 else
                    entry_price + (entry_atr * sl_mult)
                )
            else:
                sl_pct = float(params.get('STOP_LOSS_PCT', 0.02))
                stop_price = (
                    entry_price * (1 - sl_pct)
                    if amount > 0 else
                    entry_price * (1 + sl_pct)
                )
            stop_price = float(self.client.round_price(symbol, stop_price))

            tp_price = 0.0
            use_tp_entry = bool(params.get('USE_TAKE_PROFIT', False))
            if use_tp_entry and entry_atr > 0:
                tp_atr_mult = float(
                    params.get('TAKE_PROFIT_ATR_MULT_FUTURES', params.get('TAKE_PROFIT_ATR_MULT', 3.0))
                )
                raw_tp = (
                    entry_price + (entry_atr * tp_atr_mult)
                    if amount > 0 else
                    entry_price - (entry_atr * tp_atr_mult)
                )
                tp_price = float(self.client.round_price(symbol, raw_tp))

            entry_target_open_ts = self._infer_entry_open_ts_from_exchange_trades(
                symbol=symbol,
                amount=amount,
                execution_tf=execution_tf,
            )
            if entry_target_open_ts <= 0:
                tf_min = self._timeframe_to_minutes(execution_tf)
                interval_ms = int(max(1, tf_min) * 60 * 1000) if tf_min > 0 else 60 * 60 * 1000
                now_ref_ms = int(self._get_reference_now_ms())
                entry_target_open_ts = int(now_ref_ms - (now_ref_ms % interval_ms))
                if entry_target_open_ts <= 0:
                    entry_target_open_ts = now_ref_ms

            self.state_manager.update_symbol_state(symbol, {
                'entry_time': datetime.utcfromtimestamp(entry_target_open_ts / 1000.0).isoformat(),
                'entry_fill_time': datetime.utcnow().isoformat(),
                'entry_price': float(entry_price),
                'entry_atr': float(entry_atr),
                'side': side,
                'sl_required': True,  # Force watchdog reconciliation once after bootstrap.
                'last_sl_order_time': None,
                'active_stop_price': float(stop_price),
                'tp_price': float(tp_price),
                'pos_atr': float(entry_atr),
                'highest_price': float(entry_price),
                'lowest_price': float(entry_price),
                'last_processed_candle_ts': 0,
                'entry_target_open_ts': int(entry_target_open_ts),
                'entry_exec_lag_sec': 0.0,
                'recovery_bootstrapped': True,
                'recovery_mode': True,
                'recovery_bootstrapped_at': datetime.utcnow().isoformat(),
            })
            logger.warning(
                f"⚠️ [{symbol}] Bootstrapped local state from live exchange position "
                f"(side={side}, entry={entry_price:.6f}, atr={entry_atr:.6f}, tf={execution_tf})"
            )
            return True
        except Exception as e:
            logger.error(f"❌ [{symbol}] Failed to bootstrap local state: {e}")
            return False

    def _infer_entry_open_ts_from_exchange_trades(
        self,
        symbol: str,
        amount: float,
        execution_tf: str,
        lookback_limit: int = 200,
    ) -> int:
        """
        Infer entry anchor timestamp from recent exchange fills.
        Returns timeframe-aligned open timestamp(ms) or 0 when inference fails.
        """
        try:
            side_hint = 'buy' if amount > 0 else 'sell'
            trades = self.client.exchange.fetch_my_trades(symbol, limit=max(20, int(lookback_limit)))
            if not trades:
                return 0

            selected_ts = 0
            for tr in reversed(trades):
                tr_side = str(tr.get('side', '') or '').lower()
                if tr_side != side_hint:
                    continue
                tr_ts = int(tr.get('timestamp') or 0)
                if tr_ts <= 0:
                    continue
                selected_ts = tr_ts
                break
            if selected_ts <= 0:
                return 0

            tf_min = self._timeframe_to_minutes(execution_tf)
            interval_ms = int(max(1, tf_min) * 60 * 1000) if tf_min > 0 else 60 * 60 * 1000
            aligned_ts = int(selected_ts - (selected_ts % interval_ms))
            return aligned_ts if aligned_ts > 0 else 0
        except Exception as e:
            logger.debug(f"[{symbol}] Entry timestamp inference from trades failed: {e}")
            return 0
    
    @api_retry
    def _place_order_safe(
        self,
        symbol: str,
        side: str,
        qty: float,
        atr: float = None,
        current_price: float = None,
        reduce_only: bool = False,
        allow_market_fallback: bool = True,
        order_deadline_ms: Optional[int] = None,
        post_only_wait_seconds: Optional[float] = None,
        post_only_requote_max: Optional[int] = None,
    ):
        """안전한 주문 실행 (재시도 적용 + 스마트 주문 + 변동성 기반 최적화)"""
        return self.client.place_order_smart(
            symbol,
            side,
            qty,
            atr=atr,
            current_price=current_price,
            reduce_only=reduce_only,
            allow_market_fallback=allow_market_fallback,
            order_deadline_ms=order_deadline_ms,
            post_only_wait_seconds=post_only_wait_seconds,
            post_only_requote_max=post_only_requote_max,
        )

    @api_retry
    def _place_stop_loss_safe(self, symbol: str, side: str, qty: float, stop_price: float):
        """안전한 서버 사이드 Stop Loss 실행 (재시도 적용)"""
        return self.client.place_stop_market_order(symbol, side, qty, stop_price)

    @api_retry
    def _cancel_all_orders_safe(self, symbol: str):
        """안전한 모든 주문 취소 (재시도 적용)"""
        return self.client.cancel_all_orders(symbol)

    def _resolve_timeframes(self, params: dict) -> Tuple[str, str]:
        """Execution TF와 Indicator TF를 명확히 분리."""
        execution_tf = str(params.get('TIMEFRAME', '1h'))
        if self._timeframe_to_minutes(execution_tf) <= 0:
            logger.warning(f"⚠️ Invalid execution timeframe '{execution_tf}'. Falling back to '1h'.")
            execution_tf = '1h'
        indicator_tf = str(params.get('INDICATOR_TIMEFRAME', '1d'))
        # Invalid indicator TF -> 1d로 동기화 (backtest parity)
        if self._timeframe_to_minutes(indicator_tf) <= 0:
            logger.warning(
                f"⚠️ Invalid indicator timeframe '{indicator_tf}'. Falling back to '1d' for backtest parity."
            )
            indicator_tf = '1d'
        return execution_tf, indicator_tf

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert ccxt timeframe string to minutes."""
        tf = str(timeframe or '').strip().lower()
        try:
            if tf.endswith('m'):
                return max(1, int(tf[:-1]))
            if tf.endswith('h'):
                return max(1, int(tf[:-1]) * 60)
            if tf.endswith('d'):
                return max(1, int(tf[:-1]) * 1440)
        except ValueError:
            return -1
        return -1

    def _select_last_closed_candle(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.Series]:
        """
        Select latest closed candle robustly.
        - Prefer timestamp-based closed-candle detection.
        - Fallback to [-2] if needed.
        """
        if df is None or df.empty:
            return None
        if 'timestamp' not in df.columns:
            return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]

        interval_min = self._timeframe_to_minutes(timeframe)
        if interval_min <= 0:
            return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        interval_ms = interval_min * 60 * 1000
        now_ms = self._get_reference_now_ms()
        timestamps = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0).astype(np.int64)
        closed_mask = (timestamps + interval_ms) <= now_ms
        closed_indices = df.index[closed_mask.to_numpy()]
        if len(closed_indices) > 0:
            return df.loc[closed_indices[-1]]
        return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]

    def _extract_candle_timestamp_ms(self, candle: pd.Series) -> int:
        """캔들 시각(ms) 추출. 없으면 0."""
        if candle is None:
            return 0
        raw_ts = candle.get('timestamp', 0)
        try:
            if pd.isna(raw_ts):
                return 0
            return int(raw_ts)
        except Exception:
            return 0

    def _cache_exit_indicators(self, symbol: str, indicators: dict):
        """캐시된 청산 전용 지표값 저장."""
        if not hasattr(self, '_exit_indicator_cache'):
            self._exit_indicator_cache = {}
        self._exit_indicator_cache[symbol] = indicators

    def _get_cached_exit_indicators(self, symbol: str) -> dict:
        """캐시된 청산 전용 지표값 조회."""
        if not hasattr(self, '_exit_indicator_cache'):
            self._exit_indicator_cache = {}
        return self._exit_indicator_cache.get(symbol, {})

    def _refresh_exit_indicators_if_needed(
        self,
        symbol: str,
        strategy: UltimateStrategy,
        params: dict,
        execution_tf: str,
    ) -> dict:
        """
        청산 반응성을 위해 execution TF 기준으로 RSI/SAR/Trend/ATR를 캔들 마감마다 갱신.
        """
        current_slot = self._get_candle_slot_id(execution_tf)
        cached = self._get_cached_exit_indicators(symbol)
        if cached and self.last_exit_calc_candle.get(symbol) == current_slot:
            return cached

        interval_min = self._timeframe_to_minutes(execution_tf)
        if interval_min <= 0:
            logger.warning(f"⚠️ [{symbol}] Invalid execution timeframe for exit indicator refresh: {execution_tf}")
            return cached

        lookback_bars = max(300, int(params.get('HURST_PERIOD', 200)) + 80)
        df = self._fetch_recent_ohlcv_safe(symbol, execution_tf, limit=lookback_bars)
        if df is None or len(df) < 80:
            logger.warning(
                f"⚠️ [{symbol}] Exit indicator refresh skipped: insufficient {execution_tf} candles "
                f"({len(df) if df is not None else 0})"
            )
            return cached

        float_cols = ['open', 'high', 'low', 'close', 'volume']
        df[float_cols] = df[float_cols].astype(np.float64)
        df = strategy.generate_signals(df)
        last_candle = self._select_last_closed_candle(df, execution_tf)
        if last_candle is None:
            return cached

        trend_dir = last_candle.get('trend_direction', 0)
        atr = last_candle.get('atr', 0.0)
        sar = last_candle.get('parabolic_sar', 0.0)
        rsi_value = last_candle.get('rsi', 50.0)
        if pd.isna(trend_dir):
            trend_dir = 0
        if pd.isna(atr):
            atr = 0.0
        if pd.isna(sar):
            sar = 0.0
        if pd.isna(rsi_value):
            rsi_value = 50.0

        refreshed = {
            'trend_direction': int(trend_dir),
            'atr': float(atr),
            'parabolic_sar': float(sar),
            'rsi': float(rsi_value),
            'candle_open': float(last_candle.get('open', np.nan)),
            'candle_high': float(last_candle.get('high', np.nan)),
            'candle_low': float(last_candle.get('low', np.nan)),
            'candle_close': float(last_candle.get('close', np.nan)),
            'candle_ts': int(last_candle.get('timestamp', 0) or 0),
            'timeframe': execution_tf,
            'cached_at': datetime.utcnow().isoformat(),
        }
        self._cache_exit_indicators(symbol, refreshed)
        self.last_exit_calc_candle[symbol] = current_slot
        return refreshed

    def _confirm_position(
        self,
        symbol: str,
        expected_side: Optional[str] = None,
        retries: int = 5,
        sleep_seconds: float = 0.3,
    ) -> dict:
        """
        Reconcile actual position from exchange after entry/exit order.
        expected_side: 'LONG' | 'SHORT' | None
        """
        last_pos = {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0, 'leverage': 1}
        attempts = max(1, int(retries))
        for _ in range(attempts):
            pos = self._fetch_position_safe(symbol)
            if pos:
                last_pos = pos
            amount = float(last_pos.get('amount', 0.0) or 0.0)
            if expected_side == 'LONG' and amount > 0:
                return last_pos
            if expected_side == 'SHORT' and amount < 0:
                return last_pos
            if expected_side is None and abs(amount) > 0:
                return last_pos
            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))
        return last_pos

    def _wait_until_position_flat(
        self,
        symbol: str,
        timeout_seconds: float = 6.0,
        poll_seconds: float = 0.3,
    ) -> Tuple[bool, float]:
        """Wait until exchange position becomes flat."""
        flat_epsilon = 1e-8
        try:
            constraints = self.client.get_symbol_constraints(symbol)
            min_amount = float(constraints.get('min_amount') or 0.0)
            if min_amount > 0:
                flat_epsilon = min_amount * 0.25
        except Exception:
            pass

        deadline = time.time() + max(0.5, float(timeout_seconds))
        last_amount = 0.0
        while time.time() <= deadline:
            pos = self._fetch_position_safe(symbol)
            last_amount = float(pos.get('amount', 0.0) or 0.0)
            if abs(last_amount) <= flat_epsilon:
                return True, last_amount
            time.sleep(max(0.05, float(poll_seconds)))
        return False, last_amount
    
    def initialize(self):
        """초기화: 전략 로드, 레버리지 설정, 잔고 확인"""
        logger.info("🤖 RealTrader Futures Bot Initializing...")
        self._sync_server_time_offset(force=True)
        
        # 1. 전략 로드
        self.load_strategies_from_db()
        gc.collect() # 전략 객체 생성 후 메모리 정리
        
        # 2. 잔고 확인
        try:
            total_balance, usdt_free = self._fetch_balance_safe()
            logger.info(f"💰 Account Balance: {usdt_free:.2f} USDT (Total: {total_balance:.2f})")
            
            if usdt_free < MIN_BALANCE_USDT:
                logger.warning(f"⚠️ Warning: Low balance (< {MIN_BALANCE_USDT} USDT)!")
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")
        
        # 3. 레버리지 설정
        for symbol in self.symbols:
            try:
                target_lev = self.params_map[symbol].get('LEVERAGE', 1)
                applied_lev = int(max(1, min(float(target_lev), float(MAX_EXCHANGE_LEVERAGE))))
                success = self.client.set_leverage(symbol, applied_lev)
                if success:
                    logger.info(
                        f"✅ Exchange Leverage: {applied_lev}x for {symbol} "
                        f"(Strategy Target: {target_lev}x, Max Allowed: {MAX_EXCHANGE_LEVERAGE}x)"
                    )
                
                # 마진 모드 설정 (Cross 모드 강제)
                self.client.set_margin_type(symbol, margin_type='CROSSED')

            except Exception as e:
                logger.error(f"⚠️ Error setting leverage/margin for {symbol}: {e}")
        
        # 4. 포지션 모드 설정 (One-Way Mode 강제)
        # 봇 로직은 단방향 스위칭 구조이므로 Hedge Mode가 아닌 One-Way Mode가 필수입니다.
        try:
            self.client.set_position_mode(dual_side_position=False)
        except Exception as e:
            logger.error(f"⚠️ Failed to set One-Way Mode: {e}")

        # 5. 자산 모드 설정 (Single-Asset Mode 강제)
        # 봇은 USDT 단일 담보만 고려하여 잔고 계산을 하므로 Multi-Asset Mode는 끕니다.
        try:
            self.client.set_asset_mode(is_multi_asset=False)
        except Exception as e:
            logger.error(f"⚠️ Failed to set Single-Asset Mode: {e}")

        # 6. 초기 헬스체크
        self.health_manager.update_heartbeat(status="initialized")
        
        logger.info("🚀 Initialization Complete. Bot is Running...")
    
    def execute_logic(self, symbol: str):
        """핵심 매매 로직 실행 (메모리 최적화)"""
        try:
            params = self.params_map[symbol]
            strategy = self.strategies[symbol]
            execution_tf, indicator_tf = self._resolve_timeframes(params)
            df = None  # Ensure initialization for cleanup in finally/error block

            # 현재 포지션 확인 (가벼운 API)
            pos = self._fetch_position_safe(symbol)
            amount = float(pos['amount'])
            in_position = abs(amount) > 0

            # 포지션이 없는데 상태가 남아 있으면 stale state 정리
            stale_state = self.state_manager.get_symbol_state(symbol)
            if not in_position and stale_state and (
                stale_state.get('entry_time')
                or stale_state.get('side')
                or stale_state.get('exit_pending')
                or stale_state.get('exit_error')
            ):
                logger.info(f"🧹 [{symbol}] Clearing stale local state (no open position on exchange).")
                self.state_manager.clear_symbol_state(symbol)
            
            # [Optimization] 중복 계산 방지 체크
            current_slot = self._get_candle_slot_id(indicator_tf)
            already_calculated = self.last_calc_candle.get(symbol) == current_slot
            
            # [P0 FIX] 캐시 미존재 시 무한 재계산 방지
            # 캐시가 없으면 무조건 계산 (초기화), 이후에는 indicator TF 슬롯 변경 시에만 계산
            cached = self._get_cached_indicators(symbol)
            required_indicator_keys = (
                'trend_direction',
                'atr',
                'parabolic_sar',
                'entry_upper',
                'entry_lower',
                'strength_filter',
                'volume_ratio',
                'rsi',
                'hurst',
                'natr',
            )
            need_calculation = False
            
            if not cached:
                need_calculation = True
                logger.info(f"🔍 [{symbol}] Initial indicator calculation on {indicator_tf} (no cache)")
            elif any(key not in cached for key in required_indicator_keys):
                need_calculation = True
                logger.warning(f"⚠️ [{symbol}] Indicator cache incomplete. Rebuilding indicators on {indicator_tf}.")
            elif not already_calculated:
                need_calculation = True
                logger.info(f"🔍 [{symbol}] Indicator refresh on {indicator_tf}")

            # --- Case 1: 지표 계산이 필요한 경우 -> 무거운 데이터 로드 ---
            if need_calculation:

                # 전체 캔들 데이터 조회 (지표 계산용)
                tf_min = self._timeframe_to_minutes(indicator_tf)
                if tf_min <= 0:
                    logger.error(f"❌ [{symbol}] Invalid indicator timeframe: {indicator_tf}")
                    return
                
                limit = 700
                lookback_days = (limit * tf_min) / 1440
                start_dt = datetime.utcnow() - timedelta(days=lookback_days + 2)
                start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                
                df = self._fetch_ohlcv_safe(symbol, indicator_tf, start_str)
                
                if df is None or len(df) < 200:
                    logger.warning(
                        f"⚠️ Insufficient data for {symbol}. "
                        f"Got {len(df) if df is not None else 0}, need min 200."
                    )
                    # [P0 FIX] 계산 실패해도 슬롯 기록하여 무한 재시도 방지
                    self.last_calc_candle[symbol] = current_slot
                    return
                
                # [Correction] Ensure float64 for TA-Lib compatibility
                float_cols = ['open', 'high', 'low', 'close', 'volume']
                df[float_cols] = df[float_cols].astype(np.float64)
                
                # 지표 계산
                df = strategy.generate_signals(df)
                last_candle = self._select_last_closed_candle(df, indicator_tf)
                if last_candle is None:
                    logger.warning(f"⚠️ [{symbol}] No closed candle available for indicator TF {indicator_tf}.")
                    self.last_calc_candle[symbol] = current_slot
                    return
                
                # 지표값 추출 및 캐싱
                entry_upper = last_candle.get('entry_upper')
                entry_lower = last_candle.get('entry_lower')
                trend_dir = last_candle.get('trend_direction', 0)
                strength_ok = (last_candle.get('strength_filter', 1) == 1)
                atr = last_candle.get('atr', 0.0)
                sar = last_candle.get('parabolic_sar', 0.0)
                vol_ratio = last_candle.get('volume_ratio', -10.0)
                rsi_value = last_candle.get('rsi', 50.0)
                hurst_value = last_candle.get('hurst', 0.5)
                natr_value = last_candle.get('natr', 1.0)
                
                if pd.isna(atr): atr = 0.0
                if pd.isna(sar): sar = 0.0
                if pd.isna(trend_dir): trend_dir = 0
                if pd.isna(vol_ratio): vol_ratio = -10.0
                if pd.isna(rsi_value): rsi_value = 50.0
                if pd.isna(hurst_value): hurst_value = 0.5
                if pd.isna(natr_value): natr_value = 1.0
                
                # [CACHE] 다음 indicator TF 갱신 시점까지 재사용할 지표값 저장
                self._cache_indicators(symbol, {
                    'trend_direction': int(trend_dir),
                    'atr': float(atr),
                    'parabolic_sar': float(sar),
                    'entry_upper': float(entry_upper) if entry_upper is not None else None,
                    'entry_lower': float(entry_lower) if entry_lower is not None else None,
                    'strength_filter': int(strength_ok),
                    'volume_ratio': float(vol_ratio),
                    'rsi': float(rsi_value),
                    'hurst': float(hurst_value),
                    'natr': float(natr_value),
                    'indicator_timeframe': indicator_tf,
                    'cached_at': datetime.utcnow().isoformat()
                })
                
                # 계산 완료 기록
                self.last_calc_candle[symbol] = current_slot
                logger.info(f"📊 Indicators calculated for {symbol} on {indicator_tf}")
                
            # --- Case 2: 이미 계산됨 -> 캐시된 indicator TF 지표 사용 ---
            else:
                # 캐시된 지표값 로드 (indicator TF 기준)
                cached = self._get_cached_indicators(symbol)
                trend_dir = cached.get('trend_direction', 0)
                atr = cached.get('atr', 0.0)
                sar = cached.get('parabolic_sar', 0.0)
                
                # 진입 조건은 체크 안 함 (시간이 안 맞으므로)
                entry_upper = cached.get('entry_upper')
                entry_lower = cached.get('entry_lower')
                strength_ok = bool(cached.get('strength_filter', False))
                vol_ratio = cached.get('volume_ratio', -10.0)
                rsi_value = cached.get('rsi', 50.0)
                hurst_value = cached.get('hurst', 0.5)
                natr_value = cached.get('natr', 1.0)

            atr = float(0.0 if pd.isna(atr) else atr)
            sar = float(0.0 if pd.isna(sar) else sar)
            trend_dir = int(0 if pd.isna(trend_dir) else trend_dir)
            vol_ratio = float(-10.0 if pd.isna(vol_ratio) else vol_ratio)
            rsi_value = float(50.0 if pd.isna(rsi_value) else rsi_value)
            hurst_value = float(0.5 if pd.isna(hurst_value) else hurst_value)
            natr_value = float(1.0 if pd.isna(natr_value) else natr_value)
            
            # 현재가 조회 (가벼운 API - 항상 필요)
            current_price = self._get_market_price_safe(symbol)
            if current_price is None:
                logger.warning(f"⚠️ Failed to get price for {symbol}")
                return

            if in_position:
                entry_px_for_log = float(pos.get('entryPrice', 0.0) or 0.0)
                logger.info(
                    f"📊 [{symbol}] Position: {amount:+.4f} @ Entry {entry_px_for_log:.2f} | "
                    f"Current: {current_price:.2f} | Checking exit conditions..."
                )

                # Backtest parity: 청산 지표(trend/atr/sar/rsi)는 indicator TF 캐시값 유지.
                # execution TF에서는 bar O/H/L/C 컨텍스트만 보강한다.
                exit_ind = self._refresh_exit_indicators_if_needed(symbol, strategy, params, execution_tf)

                # Restart/state-loss guard: if exchange has a position but local state is missing,
                # rebuild a minimal deterministic state so exit logic remains consistent.
                state_snapshot = self.state_manager.get_symbol_state(symbol)
                if self._position_state_missing_core(state_snapshot):
                    self._bootstrap_state_for_open_position(
                        symbol=symbol,
                        amount=amount,
                        pos=pos,
                        current_price=current_price,
                        params=params,
                        atr=atr,
                        execution_tf=execution_tf,
                    )

                # [Watchdog] Stop Loss 주문 누락 감지 및 복구
                try:
                    # 중복 주문 방지: 최근 2시간 이내 주문한 기록이 있으면 스킵
                    state = self.state_manager.get_symbol_state(symbol)
                    last_sl_time_str = state.get('last_sl_order_time')
                    sl_required = bool(state.get('sl_required', False))
                    
                    # 쿨다운 체크
                    should_check = True
                    if last_sl_time_str and not sl_required:
                        last_sl_time = datetime.fromisoformat(last_sl_time_str)
                        cooldown = 7200  # 2시간 쿨다운 (로그 도배 방지)
                        elapsed = (datetime.utcnow() - last_sl_time).total_seconds()
                        if elapsed < cooldown:
                            logger.debug(f"[{symbol}] SL order cooldown active ({elapsed:.0f}s/{cooldown}s). Skipping check.")
                            should_check = False
                    
                    # SL Watchdog 실행
                    if should_check:
                        # 1. SL 주문 감지
                        sl_orders = self._detect_stop_loss_orders(symbol)
                        
                        # 2. 중복 주문 정리
                        sl_orders = self._cleanup_duplicate_sl_orders(symbol, sl_orders)
                        
                        # 3. 주문 없으면 복구
                        if not sl_orders:
                            entry_price = float(pos.get('entryPrice', current_price))
                            restored = self._restore_stop_loss(symbol, amount, entry_price, current_price, params, atr)
                            self.state_manager.update_symbol_state(symbol, {'sl_required': (not bool(restored))})
                        else:
                            if sl_required:
                                self.state_manager.update_symbol_state(symbol, {'sl_required': False})

                except Exception as e:
                    # [FIX] 에러 메시지에 -4130(이미 존재함)이 포함된 경우 성공으로 처리하여 쿨다운 적용
                    # [ISSUE #2 FIX] 실패해도 쿨다운 기록 (무한 재시도 방지)
                    self.state_manager.update_symbol_state(symbol, {
                        'last_sl_order_time': datetime.utcnow().isoformat(),
                        'sl_required': True
                    })
                    
                    if "-4130" in str(e):
                        logger.info(f"ℹ️ Stop Loss already exists on server for {symbol}. Synchronizing state...")
                    else:
                        logger.error(f"⚠️ SL Watchdog Failed: {e}")
                
                # [EXIT CHECK] 포지션 보유 중이라면 청산 조건 검토
                self._check_exit(
                    symbol, amount, current_price, params, pos,
                    trend_dir, atr, sar, rsi_value, execution_tf,
                    exit_ind=exit_ind,
                )
            
            # --- ENTRY LOGIC ---
            # Backtest parity 우선: 포지션 없으면 항상 최신 마감캔들을 확인하고,
            # 실제 진입 허용 여부는 target_entry_open_ts 게이트에서 결정.
            elif not in_position:
                logger.info(f"ℹ️ [{symbol}] No position. Checking entry conditions...")
                state = self.state_manager.get_symbol_state(symbol)

                # [Safety] 미체결 주문 확인 (Pending Orders) - 펜딩된 진입 주문이 있으면 스킵
                open_orders = self.client.fetch_open_orders(symbol)
                entry_orders = [o for o in open_orders if o.get('type') != 'STOP_MARKET']
                if len(entry_orders) > 0:
                    logger.warning(f"⚠️ Open entry orders exist {len(entry_orders)} for {symbol}. Skipping.")
                    return

                logger.info(f"🔎 [{symbol}] Checking Entry Conditions...")
                # [P2 FIX] Entry Level 유효성 검증 강화 (None/Inf 체크 추가)
                if (pd.isna(entry_upper) or pd.isna(entry_lower) or 
                    entry_upper is None or entry_lower is None or
                    not np.isfinite(entry_upper) or not np.isfinite(entry_lower)):
                    logger.info(f"⏭️ [{symbol}] Skip: Invalid Entry Levels")
                    return
                
                # Indicators from cache
                use_vol_filter = params.get('USE_VOLUME_FILTER', False)
                signal_df = self._fetch_recent_ohlcv_safe(symbol, execution_tf, limit=4)
                if signal_df is None or len(signal_df) < 2:
                    logger.warning(f"⚠️ [{symbol}] Cannot read latest closed candle for {execution_tf}.")
                    return

                signal_candle = self._select_last_closed_candle(signal_df, execution_tf)
                if signal_candle is None:
                    logger.warning(f"⚠️ [{symbol}] No closed signal candle on {execution_tf}.")
                    return

                signal_price = float(signal_candle.get('close', np.nan))
                if not np.isfinite(signal_price):
                    logger.warning(f"⚠️ [{symbol}] Invalid signal close price: {signal_price}")
                    return

                signal_candle_ts = self._extract_candle_timestamp_ms(signal_candle)
                tf_min = self._timeframe_to_minutes(execution_tf)
                if signal_candle_ts <= 0 or tf_min <= 0:
                    logger.warning(f"⚠️ [{symbol}] Invalid signal timing metadata. ts={signal_candle_ts}, tf={execution_tf}")
                    return

                # Backtest parity: signal at closed candle i, execute near next candle open(i+1).
                # Live safety: allow limited retries within grace window.
                interval_ms = tf_min * 60 * 1000
                target_entry_open_ts = signal_candle_ts + interval_ms
                now_ref_ms = self._get_reference_now_ms()
                entry_grace_sec = float(
                    params.get(
                        'ENTRY_EXECUTION_GRACE_SECONDS',
                        self._default_entry_grace_seconds(execution_tf=execution_tf, params=params),
                    )
                )
                early_tolerance_sec = float(params.get('ENTRY_EXECUTION_EARLY_TOLERANCE_SECONDS', 1.0))
                early_bound_ms = target_entry_open_ts - int(early_tolerance_sec * 1000)
                late_bound_ms = target_entry_open_ts + int(entry_grace_sec * 1000)
                if now_ref_ms < early_bound_ms:
                    logger.debug(
                        f"[{symbol}] Entry too early for signal_ts={signal_candle_ts}. "
                        f"now={now_ref_ms}, target={target_entry_open_ts}"
                    )
                    return
                if now_ref_ms > late_bound_ms:
                    logger.info(
                        f"⏭️ [{symbol}] Stale entry skipped. "
                        f"lag={(now_ref_ms - target_entry_open_ts)/1000:.1f}s > grace {entry_grace_sec:.1f}s"
                    )
                    return

                entry_retry_max = max(1, int(params.get('ENTRY_SIGNAL_RETRY_MAX', 2)))
                last_attempt_signal_ts = int(state.get('last_entry_attempt_signal_candle_ts', 0) or 0)
                attempt_count_for_signal = int(state.get('entry_attempt_count_for_signal', 0) or 0)
                if last_attempt_signal_ts != signal_candle_ts:
                    attempt_count_for_signal = 0
                if attempt_count_for_signal >= entry_retry_max:
                    logger.debug(
                        f"[{symbol}] Signal attempt cap reached ({attempt_count_for_signal}/{entry_retry_max}). "
                        f"ts={signal_candle_ts}"
                    )
                    return
                next_attempt_count = attempt_count_for_signal + 1

                entry_post_only_wait_seconds = max(0.0, float(params.get('ENTRY_POST_ONLY_WAIT_SECONDS', 1.2)))
                entry_post_only_requote_max = max(0, int(params.get('ENTRY_POST_ONLY_REQUOTE_MAX', 2)))
                
                # [ISSUE #6 FIX] Volume Filter: Z-Score 기준으로 명확화
                # 백테스트와 동일하게 volume_ratio를 Z-score로 해석한다.
                # 기본 임계값은 엔진과 동일하게 1.0을 사용한다.
                vol_z_threshold = float(params.get('VOLUME_Z_THRESHOLD', params.get('VOLUME_THRESHOLD_MULT', 1.0)))
                
                # Condition Check
                is_uptrend = (trend_dir == 1)
                is_downtrend = (trend_dir == -1)
                long_breakout = (signal_price > entry_upper)
                short_breakout = (signal_price < entry_lower)
                vol_ok = (not use_vol_filter) or (vol_ratio >= vol_z_threshold)
                entry_lag_sec = max(0.0, float((now_ref_ms - target_entry_open_ts) / 1000.0))
                entry_allow_market_fallback = self._resolve_entry_market_fallback(
                    params=params,
                    atr=atr,
                    current_price=current_price,
                    entry_lag_sec=entry_lag_sec,
                )

                # Entry Execution
                if is_uptrend and long_breakout and strength_ok and vol_ok:
                    logger.info(
                        f"🟢 LONG {symbol} | SignalClose: {signal_price:.4f}, ExecPx: {current_price:.4f} | "
                        f"Cond: Trend(↑), Breakout(UP), Strength(OK), Vol(OK)"
                    )

                    qty = self._calculate_position_size(
                        symbol, current_price, params, atr,
                        hurst=hurst_value, natr=natr_value
                    )
                    if qty > 0:
                        self.state_manager.update_symbol_state(symbol, {
                            'last_entry_attempt_signal_candle_ts': int(signal_candle_ts),
                            'entry_attempt_count_for_signal': int(next_attempt_count),
                            'last_entry_attempt_at': datetime.utcnow().isoformat(),
                        })
                        order = self._place_order_safe(
                            symbol, 'buy', qty, atr=atr, current_price=current_price,
                            reduce_only=False,
                            allow_market_fallback=entry_allow_market_fallback,
                            order_deadline_ms=int(late_bound_ms),
                            post_only_wait_seconds=entry_post_only_wait_seconds,
                            post_only_requote_max=entry_post_only_requote_max,
                        )
                        confirmed_pos = self._confirm_position(symbol, expected_side='LONG', retries=10, sleep_seconds=0.5)
                        confirmed_amount = float(confirmed_pos.get('amount', 0.0) or 0.0)
                        if confirmed_amount <= 0:
                            self._cancel_all_orders_safe(symbol)
                            logger.error(
                                f"❌ [{symbol}] LONG entry not confirmed on exchange. "
                                f"Requested {qty}, position amount={confirmed_amount}"
                            )
                            return
                        if not self._enforce_min_fill_ratio(
                            symbol=symbol,
                            expected_qty=qty,
                            confirmed_amount=confirmed_amount,
                            side='LONG',
                            params=params,
                            current_price=current_price,
                            atr=atr,
                        ):
                            return
                        if not order:
                            logger.warning(
                                f"⚠️ [{symbol}] Entry response missing, but LONG position detected. "
                                "Proceeding with confirmed position data."
                            )

                        filled_qty = self.client.round_amount(symbol, abs(confirmed_amount))
                        if filled_qty <= 0:
                            logger.error(f"❌ [{symbol}] Invalid confirmed LONG quantity: {filled_qty}")
                            return
                        confirmed_entry_price = float(confirmed_pos.get('entryPrice', current_price) or current_price)

                        sl_type = params.get('STOP_LOSS_TYPE', 'FIXED')
                        stop_price = 0.0
                        if sl_type == 'ATR' and atr > 0:
                            sl_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
                            stop_price = confirmed_entry_price - (atr * sl_mult)
                        else:
                            sl_pct = params.get('STOP_LOSS_PCT', 0.02)
                            stop_price = confirmed_entry_price * (1 - sl_pct)
                        stop_price = self.client.round_price(symbol, stop_price)

                        use_tp_entry = bool(params.get('USE_TAKE_PROFIT', False))
                        tp_atr_mult_entry = float(
                            params.get('TAKE_PROFIT_ATR_MULT_FUTURES', params.get('TAKE_PROFIT_ATR_MULT', 3.0))
                        )
                        entry_atr = float(max(0.0, atr))
                        tp_price = 0.0
                        if use_tp_entry and entry_atr > 0:
                            tp_price = self.client.round_price(
                                symbol,
                                confirmed_entry_price + (entry_atr * tp_atr_mult_entry),
                            )

                        sl_result = self._place_stop_loss_safe(symbol, 'sell', filled_qty, stop_price)
                        sl_active = bool(sl_result)
                        if not sl_active:
                            logger.error(f"❌ [{symbol}] LONG entry SL placement failed. Trying immediate restore.")
                            sl_active = bool(
                                self._restore_stop_loss(
                                    symbol=symbol,
                                    amount=confirmed_amount,
                                    entry_price=confirmed_entry_price,
                                    current_price=current_price,
                                    params=params,
                                    atr=atr,
                                )
                            )

                        self.trade_db.record_trade(
                            symbol=symbol,
                            side='LONG',
                            action='ENTRY',
                            quantity=filled_qty,
                            price=confirmed_entry_price,
                            reason=f"Trend(↑) & Breakout(>{entry_upper:.2f})",
                            params={'timeframe': execution_tf, 'atr': atr, 'sl': stop_price, 'sl_active': sl_active}
                        )

                        self.state_manager.update_symbol_state(symbol, {
                            'entry_time': datetime.utcfromtimestamp(target_entry_open_ts / 1000.0).isoformat(),
                            'entry_fill_time': datetime.utcnow().isoformat(),
                            'entry_price': confirmed_entry_price,
                            'entry_atr': entry_atr,
                            'side': 'LONG',
                            'sl_required': (not sl_active),
                            'last_sl_order_time': datetime.utcnow().isoformat() if sl_active else None,
                            'active_stop_price': float(stop_price),
                            'tp_price': float(tp_price),
                            'pos_atr': entry_atr,
                            'highest_price': float(confirmed_entry_price),
                            'lowest_price': float(confirmed_entry_price),
                            'last_processed_candle_ts': 0,
                            'last_entry_signal_candle_ts': int(signal_candle_ts),
                            'last_entry_attempt_signal_candle_ts': int(signal_candle_ts),
                            'entry_attempt_count_for_signal': int(next_attempt_count),
                            'entry_target_open_ts': int(target_entry_open_ts),
                            'entry_exec_lag_sec': float(entry_lag_sec),
                            'entry_market_fallback_used': bool(entry_allow_market_fallback),
                            'recovery_mode': False,
                            'last_exit_fallback_eval_ms': 0,
                        })

                elif is_downtrend and short_breakout and strength_ok and vol_ok:
                    logger.info(
                        f"🔴 SHORT {symbol} | SignalClose: {signal_price:.4f}, ExecPx: {current_price:.4f} | "
                        f"Cond: Trend(↓), Breakout(DN), Strength(OK), Vol(OK)"
                    )

                    qty = self._calculate_position_size(
                        symbol, current_price, params, atr,
                        hurst=hurst_value, natr=natr_value
                    )
                    if qty > 0:
                        self.state_manager.update_symbol_state(symbol, {
                            'last_entry_attempt_signal_candle_ts': int(signal_candle_ts),
                            'entry_attempt_count_for_signal': int(next_attempt_count),
                            'last_entry_attempt_at': datetime.utcnow().isoformat(),
                        })
                        order = self._place_order_safe(
                            symbol, 'sell', qty, atr=atr, current_price=current_price,
                            reduce_only=False,
                            allow_market_fallback=entry_allow_market_fallback,
                            order_deadline_ms=int(late_bound_ms),
                            post_only_wait_seconds=entry_post_only_wait_seconds,
                            post_only_requote_max=entry_post_only_requote_max,
                        )
                        confirmed_pos = self._confirm_position(symbol, expected_side='SHORT', retries=10, sleep_seconds=0.5)
                        confirmed_amount = float(confirmed_pos.get('amount', 0.0) or 0.0)
                        if confirmed_amount >= 0:
                            self._cancel_all_orders_safe(symbol)
                            logger.error(
                                f"❌ [{symbol}] SHORT entry not confirmed on exchange. "
                                f"Requested {qty}, position amount={confirmed_amount}"
                            )
                            return
                        if not self._enforce_min_fill_ratio(
                            symbol=symbol,
                            expected_qty=qty,
                            confirmed_amount=confirmed_amount,
                            side='SHORT',
                            params=params,
                            current_price=current_price,
                            atr=atr,
                        ):
                            return
                        if not order:
                            logger.warning(
                                f"⚠️ [{symbol}] Entry response missing, but SHORT position detected. "
                                "Proceeding with confirmed position data."
                            )

                        filled_qty = self.client.round_amount(symbol, abs(confirmed_amount))
                        if filled_qty <= 0:
                            logger.error(f"❌ [{symbol}] Invalid confirmed SHORT quantity: {filled_qty}")
                            return
                        confirmed_entry_price = float(confirmed_pos.get('entryPrice', current_price) or current_price)

                        sl_type = params.get('STOP_LOSS_TYPE', 'FIXED')
                        stop_price = 0.0
                        if sl_type == 'ATR' and atr > 0:
                            sl_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
                            stop_price = confirmed_entry_price + (atr * sl_mult)
                        else:
                            sl_pct = params.get('STOP_LOSS_PCT', 0.02)
                            stop_price = confirmed_entry_price * (1 + sl_pct)

                        stop_price = self.client.round_price(symbol, stop_price)

                        use_tp_entry = bool(params.get('USE_TAKE_PROFIT', False))
                        tp_atr_mult_entry = float(
                            params.get('TAKE_PROFIT_ATR_MULT_FUTURES', params.get('TAKE_PROFIT_ATR_MULT', 3.0))
                        )
                        entry_atr = float(max(0.0, atr))
                        tp_price = 0.0
                        if use_tp_entry and entry_atr > 0:
                            tp_price = self.client.round_price(
                                symbol,
                                confirmed_entry_price - (entry_atr * tp_atr_mult_entry),
                            )

                        sl_result = self._place_stop_loss_safe(symbol, 'buy', filled_qty, stop_price)
                        sl_active = bool(sl_result)
                        if not sl_active:
                            logger.error(f"❌ [{symbol}] SHORT entry SL placement failed. Trying immediate restore.")
                            sl_active = bool(
                                self._restore_stop_loss(
                                    symbol=symbol,
                                    amount=confirmed_amount,
                                    entry_price=confirmed_entry_price,
                                    current_price=current_price,
                                    params=params,
                                    atr=atr,
                                )
                            )

                        self.trade_db.record_trade(
                            symbol=symbol,
                            side='SHORT',
                            action='ENTRY',
                            quantity=filled_qty,
                            price=confirmed_entry_price,
                            reason=f"Trend(↓) & Breakout(<{entry_lower:.2f})",
                            params={'timeframe': execution_tf, 'atr': atr, 'sl': stop_price, 'sl_active': sl_active}
                        )

                        self.state_manager.update_symbol_state(symbol, {
                            'entry_time': datetime.utcfromtimestamp(target_entry_open_ts / 1000.0).isoformat(),
                            'entry_fill_time': datetime.utcnow().isoformat(),
                            'entry_price': confirmed_entry_price,
                            'entry_atr': entry_atr,
                            'side': 'SHORT',
                            'sl_required': (not sl_active),
                            'last_sl_order_time': datetime.utcnow().isoformat() if sl_active else None,
                            'active_stop_price': float(stop_price),
                            'tp_price': float(tp_price),
                            'pos_atr': entry_atr,
                            'highest_price': float(confirmed_entry_price),
                            'lowest_price': float(confirmed_entry_price),
                            'last_processed_candle_ts': 0,
                            'last_entry_signal_candle_ts': int(signal_candle_ts),
                            'last_entry_attempt_signal_candle_ts': int(signal_candle_ts),
                            'entry_attempt_count_for_signal': int(next_attempt_count),
                            'entry_target_open_ts': int(target_entry_open_ts),
                            'entry_exec_lag_sec': float(entry_lag_sec),
                            'entry_market_fallback_used': bool(entry_allow_market_fallback),
                            'recovery_mode': False,
                            'last_exit_fallback_eval_ms': 0,
                        })
                
                else:
                    # Skip Reason Logging
                    reasons = []
                    side_label = ""
                    if trend_dir == 1:
                        side_label = "LONG"
                        if not long_breakout: reasons.append(f"Price(≤{entry_upper:.2f})")
                    elif trend_dir == -1:
                        side_label = "SHORT"
                        if not short_breakout: reasons.append(f"Price(≥{entry_lower:.2f})")
                    else:
                        reasons.append(f"Trend({'─' if trend_dir == 0 else '?'})")
                    
                    if not strength_ok: reasons.append("Weak")
                    if not vol_ok: reasons.append(f"VolZ({vol_ratio:.2f})")
                    
                    if reasons:
                        logger.info(f"⏭️ [{symbol}] Skip {side_label}: {', '.join(reasons)}")
            
            # [Optimization] 대규모 데이터프레임 제거 (진입 시점에만 생성됨)
            if df is not None:
                del df
                gc.collect()
        
        except Exception as e:
            logger.error(f"🚨 Error executing logic for {symbol}: {e}")
            self.health_manager.record_error(e)
    
    def _detect_stop_loss_orders(self, symbol: str) -> list:
        """
        Stop Loss 주문 감지 (모든 조건부 주문 포함)
        Returns: 감지된 SL 주문 리스트
        """
        open_orders = self.client.fetch_open_orders(symbol)
        # Stop Loss 전용 주문만 감지 (TP 주문과 분리)
        sl_orders = [
            o for o in open_orders 
            if (
                ('STOP' in o.get('type', '').upper())
                and ('TAKE_PROFIT' not in o.get('type', '').upper())
            )
        ]
        return sl_orders

    def _cancel_stop_orders_only(self, symbol: str):
        """
        STOP 계열 주문만 취소한다.
        (백테스트 trailing과 유사하게 보호 스탑을 재배치할 때 사용)
        """
        try:
            open_orders = self.client.fetch_open_orders(symbol)
            stop_orders = [
                o for o in open_orders
                if (
                    ('STOP' in o.get('type', '').upper())
                    and ('TAKE_PROFIT' not in o.get('type', '').upper())
                )
            ]
            for o in stop_orders:
                try:
                    self.client.exchange.cancel_order(o['id'], symbol)
                except Exception:
                    pass
            if stop_orders:
                logger.info(f"🗑️ [{symbol}] Canceled {len(stop_orders)} STOP orders for re-sync.")
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] Failed to cancel STOP orders: {e}")
    
    def _cleanup_duplicate_sl_orders(self, symbol: str, sl_orders: list) -> list:
        """
        중복 SL 주문 자동 정리 (최신 주문만 유지)
        Returns: 정리 후 남은 SL 주문 리스트
        """
        if len(sl_orders) > 1:
            logger.warning(f"⚠️ Multiple Stop Loss orders detected ({len(sl_orders)}) for {symbol}! Cleaning up...")
            # 가장 최신 주문만 남기고 나머지 취소
            sorted_orders = sorted(sl_orders, key=lambda x: x.get('timestamp', 0), reverse=True)
            for old_order in sorted_orders[1:]:  # 첫 번째(최신) 제외
                try:
                    self.client.exchange.cancel_order(old_order['id'], symbol)
                    logger.info(f"🗑️ Canceled duplicate SL order: {old_order['id']}")
                except:
                    pass
            # 정리 후 1개만 남았으므로 반환
            return [sorted_orders[0]]
        return sl_orders
    
    def _restore_stop_loss(
        self, 
        symbol: str, 
        amount: float, 
        entry_price: float, 
        current_price: float,
        params: dict, 
        atr: float
    ) -> bool:
        """
        Stop Loss 주문 복구 (Watchdog)
        Returns: 복구 성공 여부
        """
        logger.warning(f"🛡️ NO Stop Loss found for {symbol}! Restoring safety net...")
        
        sl_qty = self.client.round_amount(symbol, abs(amount))
        if sl_qty <= 0:
            sl_qty = abs(amount)
        sl_type = params.get('STOP_LOSS_TYPE', 'FIXED')
        stop_price = 0.0

        # Prefer state stop (for trailing parity). Fallback to base SL formula.
        state = self.state_manager.get_symbol_state(symbol)
        state_stop = state.get('active_stop_price')
        try:
            state_stop_val = float(state_stop) if state_stop is not None else 0.0
        except (TypeError, ValueError):
            state_stop_val = 0.0
        using_state_stop = bool(np.isfinite(state_stop_val) and state_stop_val > 0)
        if using_state_stop:
            stop_price = state_stop_val
            sl_side = 'sell' if amount > 0 else 'buy'
        else:
            # SL Price Calculation
            if amount > 0:  # LONG -> Sell SL
                sl_side = 'sell'
                if sl_type == 'ATR' and atr > 0:
                    sl_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
                    stop_price = entry_price - (atr * sl_mult)
                else:
                    sl_pct = params.get('STOP_LOSS_PCT', 0.02)
                    stop_price = entry_price * (1 - sl_pct)
            else:  # SHORT -> Buy SL
                sl_side = 'buy'
                if sl_type == 'ATR' and atr > 0:
                    sl_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
                    stop_price = entry_price + (atr * sl_mult)
                else:
                    sl_pct = params.get('STOP_LOSS_PCT', 0.02)
                    stop_price = entry_price * (1 + sl_pct)
        
        # 거래소 정밀도 보정 및 최소 틱 사이즈 보장
        tick_size = self.client.get_price_tick_size(symbol, fallback=(0.1 if 'BTC' in symbol else 0.01))
        stop_price = self.client.round_price(symbol, stop_price)
        
        # [FIX] 기본 SL 계산 경로에서만 진입가와 동일한 손절가 방지.
        # trailing 복구(state_stop)는 진입가를 넘어간 보호 스탑도 유지해야 한다.
        if not using_state_stop:
            if amount > 0:  # LONG
                if stop_price >= entry_price:
                    stop_price = entry_price - (tick_size * 2)
            else:  # SHORT
                if stop_price <= entry_price:
                    stop_price = entry_price + (tick_size * 2)
        stop_price = self.client.round_price(symbol, stop_price)
        
        # SL 주문 실행
        sl_result = self._place_stop_loss_safe(symbol, sl_side, sl_qty, stop_price)
        
        if sl_result:
            logger.info(f"✅ Restored Stop Loss for {symbol} @ {stop_price}")
            
            # 주문 성공 시 타임스탬프 기록 (쿨다운 시작)
            self.state_manager.update_symbol_state(symbol, {
                'last_sl_order_time': datetime.utcnow().isoformat()
            })
            
            # API 반영 대기
            time.sleep(0.5)
            return True
        
        return False

    def _check_exit(
        self,
        symbol: str,
        amount: float,
        current_price: float,
        params: dict,
        pos: dict,
        trend_dir: int,
        atr: float,
        sar: float,
        rsi_value: float,
        execution_tf: str,
        exit_ind: Optional[dict] = None,
    ):
        """Backtest-parity exit logic (bar OHLC sequential checks)."""
        try:
            state = self.state_manager.get_symbol_state(symbol)
            side_str = "LONG" if amount > 0 else "SHORT"
            order_side = 'sell' if amount > 0 else 'buy'

            entry_price = float(pos.get('entryPrice', 0.0) or state.get('entry_price', 0.0) or 0.0)
            if entry_price <= 0:
                return

            # Use closed execution candle context for backtest parity.
            candle_open = float(current_price)
            candle_high = float(current_price)
            candle_low = float(current_price)
            candle_close = float(current_price)
            candle_ts = 0
            if exit_ind:
                candle_open = float(exit_ind.get('candle_open', candle_open))
                candle_high = float(exit_ind.get('candle_high', candle_high))
                candle_low = float(exit_ind.get('candle_low', candle_low))
                candle_close = float(exit_ind.get('candle_close', candle_close))
                candle_ts = int(exit_ind.get('candle_ts', 0) or 0)

            last_processed_candle_ts = int(state.get('last_processed_candle_ts', 0) or 0)
            exit_pending = bool(state.get('exit_pending', False))
            fallback_without_candle = False

            if not np.isfinite(candle_open):
                candle_open = float(current_price)
            if not np.isfinite(candle_high):
                candle_high = float(current_price)
            if not np.isfinite(candle_low):
                candle_low = float(current_price)
            if not np.isfinite(candle_close):
                candle_close = float(current_price)

            if candle_ts <= 0:
                fallback_max_delay_sec = float(
                    params.get(
                        'EXIT_FALLBACK_MAX_DELAY_SEC',
                        max(20.0, float(LOOP_INTERVAL_SECONDS) * 2.0),
                    )
                )
                now_ref_ms = int(self._get_reference_now_ms())
                last_fallback_eval_ms = int(state.get('last_exit_fallback_eval_ms', 0) or 0)
                last_ref_ms = last_processed_candle_ts if last_processed_candle_ts > 0 else last_fallback_eval_ms
                elapsed_ms = (now_ref_ms - last_ref_ms) if last_ref_ms > 0 else (10**9)
                if elapsed_ms < int(max(1.0, fallback_max_delay_sec) * 1000):
                    logger.debug(
                        f"[{symbol}] Exit evaluation skipped: no closed execution candle context "
                        f"(tf={execution_tf}, candle_ts={candle_ts}, elapsed={elapsed_ms/1000.0:.1f}s)"
                    )
                    return
                fallback_without_candle = True
                self.state_manager.update_symbol_state(symbol, {
                    'last_exit_fallback_eval_ms': int(now_ref_ms),
                })
                logger.warning(
                    f"⚠️ [{symbol}] Exit fallback mode activated (no closed candle for {execution_tf}). "
                    "Evaluating conditional exits only (no trailing update)."
                )

            if candle_ts > 0 and candle_ts == last_processed_candle_ts and not exit_pending:
                return

            exit_type = str(params.get('EXIT_TYPE', 'ATR') or 'ATR').upper()
            use_sar_exit = (exit_type == 'PARABOLIC_SAR')
            use_tp = bool(params.get('USE_TAKE_PROFIT', False))
            tp_atr_mult = float(
                params.get('TAKE_PROFIT_ATR_MULT_FUTURES', params.get('TAKE_PROFIT_ATR_MULT', 3.0))
            )

            pos_atr = float(state.get('pos_atr', state.get('entry_atr', atr)) or 0.0)
            if not np.isfinite(pos_atr) or pos_atr < 0:
                pos_atr = 0.0
            if pos_atr <= 0 and np.isfinite(atr) and atr > 0:
                pos_atr = float(atr)

            highest_price = float(state.get('highest_price', entry_price) or entry_price)
            lowest_price = float(state.get('lowest_price', entry_price) or entry_price)
            if not np.isfinite(highest_price) or highest_price <= 0:
                highest_price = entry_price
            if not np.isfinite(lowest_price) or lowest_price <= 0:
                lowest_price = entry_price

            sl_type = str(params.get('STOP_LOSS_TYPE', 'FIXED') or 'FIXED').upper()
            active_stop = float(state.get('active_stop_price', 0.0) or 0.0)
            if not np.isfinite(active_stop) or active_stop <= 0:
                if sl_type == 'ATR' and pos_atr > 0:
                    sl_mult = float(params.get('ATR_STOP_LOSS_MULT', 1.5))
                    active_stop = entry_price - (pos_atr * sl_mult) if amount > 0 else entry_price + (pos_atr * sl_mult)
                else:
                    sl_pct = float(params.get('STOP_LOSS_PCT', 0.02))
                    active_stop = entry_price * (1 - sl_pct) if amount > 0 else entry_price * (1 + sl_pct)

            tick_size = self.client.get_price_tick_size(symbol, fallback=(0.1 if 'BTC' in symbol else 0.01))
            if amount > 0 and active_stop >= entry_price:
                active_stop = entry_price - (tick_size * 2)
            elif amount < 0 and active_stop <= entry_price:
                active_stop = entry_price + (tick_size * 2)
            active_stop = self.client.round_price(symbol, active_stop)

            tp_price_val = float(state.get('tp_price', 0.0) or 0.0)
            if use_tp:
                if (not np.isfinite(tp_price_val) or tp_price_val <= 0) and pos_atr > 0:
                    tp_price_val = entry_price + (pos_atr * tp_atr_mult) if amount > 0 else entry_price - (pos_atr * tp_atr_mult)
                if np.isfinite(tp_price_val) and tp_price_val > 0:
                    tp_price_val = self.client.round_price(symbol, tp_price_val)
                else:
                    tp_price_val = 0.0
            else:
                tp_price_val = 0.0

            # [SEQ-1] Stop Loss (SAR is merged as stop boundary, same as backtest)
            current_stop = float(active_stop)
            if use_sar_exit and np.isfinite(sar) and sar > 0:
                current_stop = max(current_stop, float(sar)) if amount > 0 else min(current_stop, float(sar))
                current_stop = self.client.round_price(symbol, current_stop)

            exit_triggered = False
            reason = ""
            exit_price_for_calc = float(current_price)
            slippage = float(SLIPPAGE_RATE)

            if not fallback_without_candle:
                if amount > 0:
                    if candle_low <= current_stop:
                        exit_triggered = True
                        reason = f"Stop Loss ({current_stop:.2f})"
                        exit_price_for_calc = current_stop * (1 - slippage)
                else:
                    if candle_high >= current_stop:
                        exit_triggered = True
                        reason = f"Stop Loss ({current_stop:.2f})"
                        exit_price_for_calc = current_stop * (1 + slippage)

                # [SEQ-2] Take Profit (bar high/low)
                if not exit_triggered and use_tp and tp_price_val > 0:
                    if amount > 0 and candle_high >= tp_price_val:
                        exit_triggered = True
                        reason = f"Take Profit ({tp_price_val:.2f})"
                        exit_price_for_calc = tp_price_val
                    elif amount < 0 and candle_low <= tp_price_val:
                        exit_triggered = True
                        reason = f"Take Profit ({tp_price_val:.2f})"
                        exit_price_for_calc = tp_price_val

            # [SEQ-3] Conditional market exits (time -> RSI -> trend)
            bars_held = 0.0
            if not exit_triggered:
                interval_min = self._timeframe_to_minutes(execution_tf)
                if interval_min > 0:
                    interval_ms = interval_min * 60 * 1000
                    # Backtest parity: count bars from intended entry bar open.
                    entry_open_ts = int(state.get('entry_target_open_ts', 0) or 0)
                    if entry_open_ts <= 0:
                        entry_fill_time_str = state.get('entry_fill_time')
                        if entry_fill_time_str:
                            try:
                                entry_fill_dt = datetime.fromisoformat(entry_fill_time_str)
                                entry_open_ts = int(entry_fill_dt.timestamp() * 1000)
                            except Exception:
                                entry_open_ts = 0
                    if entry_open_ts <= 0:
                        entry_time_str = state.get('entry_time')
                        if entry_time_str:
                            try:
                                entry_dt = datetime.fromisoformat(entry_time_str)
                                entry_open_ts = int(entry_dt.timestamp() * 1000)
                            except Exception:
                                entry_open_ts = 0

                    if entry_open_ts > 0:
                        recovery_mode = bool(state.get('recovery_mode', False))
                        if recovery_mode and interval_ms > 0:
                            entry_open_ts = int(entry_open_ts - (entry_open_ts % interval_ms))
                        ref_ms = candle_ts if candle_ts > 0 else self._get_reference_now_ms()
                        bars_held = max(
                            0.0,
                            (ref_ms - entry_open_ts) / float(interval_ms),
                        )

                max_holding_bars = float(params.get('MAX_HOLDING_BARS', 9999))
                if bars_held >= max_holding_bars:
                    if pos_atr > 0:
                        if amount > 0:
                            unreal_p = (candle_close - entry_price) / pos_atr
                        else:
                            unreal_p = (entry_price - candle_close) / pos_atr
                    else:
                        unreal_p = 0.0

                    time_exit_profit_threshold = float(params.get('TIME_EXIT_PROFIT_THRESHOLD', 0.5))
                    if unreal_p < time_exit_profit_threshold:
                        exit_triggered = True
                        reason = f"Time Cut (Held {bars_held:.1f} bars, UnrealATR {unreal_p:.2f})"
                        exit_price_for_calc = candle_open * (1 - slippage if amount > 0 else 1 + slippage)

            if not exit_triggered:
                rsi = float(50.0 if pd.isna(rsi_value) else rsi_value)
                rsi_exit_thresh = float(params.get('RSI_EXIT_THRESHOLD', 80.0))
                if amount > 0 and rsi > rsi_exit_thresh:
                    exit_triggered = True
                    reason = f"Panic Exit (RSI {rsi:.1f})"
                    exit_price_for_calc = candle_open * (1 - slippage)
                elif amount < 0 and rsi < (100.0 - rsi_exit_thresh):
                    exit_triggered = True
                    reason = f"Panic Exit (RSI {rsi:.1f})"
                    exit_price_for_calc = candle_open * (1 + slippage)

            if not exit_triggered:
                if amount > 0 and trend_dir == -1:
                    exit_triggered = True
                    reason = "Trend Reversal"
                    exit_price_for_calc = candle_open * (1 - slippage)
                elif amount < 0 and trend_dir == 1:
                    exit_triggered = True
                    reason = "Trend Reversal"
                    exit_price_for_calc = candle_open * (1 + slippage)

            # [SEQ-4] trailing update only when no exit
            trailing_updated = False
            prev_stop = float(active_stop)
            if not exit_triggered and (not fallback_without_candle):
                trailing_activation_atr = float(params.get('TRAILING_ACTIVATION_ATR', 0.0))
                atr_mult = float(params.get('ATR_MULTIPLIER', 3.0))

                if amount > 0:
                    if candle_high > highest_price:
                        highest_price = candle_high
                    if exit_type != 'PARABOLIC_SAR' and pos_atr > 0:
                        unreal_profit = (highest_price - entry_price) / pos_atr
                        if unreal_profit >= trailing_activation_atr:
                            new_stop = highest_price - (pos_atr * atr_mult)
                            if new_stop > active_stop:
                                active_stop = new_stop
                                trailing_updated = True
                else:
                    if candle_low < lowest_price:
                        lowest_price = candle_low
                    if exit_type != 'PARABOLIC_SAR' and pos_atr > 0:
                        unreal_profit = (entry_price - lowest_price) / pos_atr
                        if unreal_profit >= trailing_activation_atr:
                            new_stop = lowest_price + (pos_atr * atr_mult)
                            if new_stop < active_stop:
                                active_stop = new_stop
                                trailing_updated = True

                active_stop = self.client.round_price(symbol, active_stop)
                base_update = {
                    'pos_atr': float(pos_atr),
                    'active_stop_price': float(active_stop),
                    'tp_price': float(tp_price_val),
                    'highest_price': float(highest_price),
                    'lowest_price': float(lowest_price),
                }
                if candle_ts > 0:
                    base_update['last_processed_candle_ts'] = int(candle_ts)
                self.state_manager.update_symbol_state(symbol, base_update)

                if trailing_updated:
                    stop_qty = self.client.round_amount(symbol, abs(amount))
                    if stop_qty <= 0:
                        stop_qty = abs(amount)
                    stop_side = 'sell' if amount > 0 else 'buy'
                    new_stop_price = self.client.round_price(symbol, active_stop)

                    self._cancel_stop_orders_only(symbol)
                    sl_result = self._place_stop_loss_safe(symbol, stop_side, stop_qty, new_stop_price)
                    if sl_result:
                        self.state_manager.update_symbol_state(symbol, {
                            'active_stop_price': float(new_stop_price),
                            'sl_required': False,
                            'last_sl_order_time': datetime.utcnow().isoformat(),
                        })
                    else:
                        fallback_ok = False
                        if prev_stop > 0:
                            fallback_stop = self.client.round_price(symbol, prev_stop)
                            fallback_ok = bool(
                                self._place_stop_loss_safe(symbol, stop_side, stop_qty, fallback_stop)
                            )
                            if fallback_ok:
                                self.state_manager.update_symbol_state(symbol, {
                                    'active_stop_price': float(fallback_stop),
                                    'sl_required': False,
                                    'last_sl_order_time': datetime.utcnow().isoformat(),
                                })
                        if not fallback_ok:
                            self.state_manager.update_symbol_state(symbol, {
                                'sl_required': True,
                                'last_sl_order_time': datetime.utcnow().isoformat(),
                            })
                            logger.error(
                                f"[{symbol}] Trailing SL re-sync failed (new={new_stop_price:.4f}, prev={prev_stop:.4f})."
                            )
                return
            elif not exit_triggered and fallback_without_candle:
                return

            # Exit order path
            if not np.isfinite(exit_price_for_calc) or exit_price_for_calc <= 0:
                exit_price_for_calc = float(current_price)
            pnl = (
                (exit_price_for_calc - entry_price) * abs(amount)
                if amount > 0 else
                (entry_price - exit_price_for_calc) * abs(amount)
            )
            pnl_pct = (
                ((exit_price_for_calc / entry_price) - 1) * 100
                if amount > 0 else
                ((entry_price / exit_price_for_calc) - 1) * 100
            )

            logger.info(
                f"EXIT {side_str} {symbol} | TriggerPx: {exit_price_for_calc:.4f} | "
                f"Candle O/H/L/C: {candle_open:.4f}/{candle_high:.4f}/{candle_low:.4f}/{candle_close:.4f} | "
                f"PnL: {pnl_pct:+.2f}% (${pnl:.2f}) | Reason: {reason}"
            )

            pending_update = {
                'exit_pending': True,
                'exit_reason': reason,
                'exit_attempt_at': datetime.utcnow().isoformat(),
                'exit_requested_qty': float(abs(amount)),
                'exit_error': None,
            }
            if candle_ts > 0:
                pending_update['last_processed_candle_ts'] = int(candle_ts)
            self.state_manager.update_symbol_state(symbol, pending_update)

            order_result = self._place_order_safe(symbol, order_side, abs(amount), reduce_only=True)
            if not order_result:
                logger.warning(
                    f"[{symbol}] Exit order response missing/failed. "
                    "Will verify actual position before state transition."
                )
                self.state_manager.update_symbol_state(symbol, {
                    'exit_error': "order_submit_failed_or_unknown",
                    'exit_attempt_at': datetime.utcnow().isoformat(),
                })

            is_flat, remaining_amount = self._wait_until_position_flat(symbol, timeout_seconds=6.0, poll_seconds=0.35)
            if not is_flat:
                logger.error(
                    f"[{symbol}] Exit not confirmed. Remaining position: {remaining_amount:+.8f}. "
                    "Skip cancel/state clear to avoid orphaned risk."
                )
                self.state_manager.update_symbol_state(symbol, {
                    'exit_pending': True,
                    'exit_error': f"not_flat_after_exit_attempt:{remaining_amount:+.8f}",
                    'exit_remaining_amount': float(remaining_amount),
                    'exit_attempt_at': datetime.utcnow().isoformat(),
                })
                return

            try:
                self._cancel_all_orders_safe(symbol)
            except Exception as cancel_err:
                logger.warning(f"[{symbol}] Post-exit cancel_all_orders failed: {cancel_err}")
            self.trade_db.record_trade(
                symbol=symbol,
                side=side_str,
                action='EXIT',
                quantity=abs(amount),
                price=exit_price_for_calc,
                entry_price=entry_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                reason=reason,
            )
            self.state_manager.clear_symbol_state(symbol)

        except Exception as e:
            logger.error(f"Error in _check_exit for {symbol}: {e}")
            try:
                self.state_manager.update_symbol_state(symbol, {
                    'exit_pending': True,
                    'exit_error': f"exception:{e}",
                    'exit_attempt_at': datetime.utcnow().isoformat(),
                })
            except Exception:
                pass
            self.health_manager.record_error(e)

    def _calculate_position_size(
        self, 
        symbol: str, 
        price: float, 
        params: dict, 
        atr: float = 0.0,
        hurst: float = 0.5,
        natr: float = 0.0
    ) -> float:
        """
        포지션 사이즈 계산 (견고성 강화)
        
        개선사항:
        - 거래소 정밀도(precision) 자동 조회
        - 최소 주문 금액/수량 검증
        - Edge case 방어 (0 나누기, 음수 레버리지 등)
        """
        # === 0. Input Validation ===
        if price <= 0:
            logger.error(f"❌ Invalid price for {symbol}: {price}")
            return 0.0
        
        # === 1. 잔고 조회 ===
        try:
            total_balance, usdt_free = self._fetch_balance_safe()
        except Exception as e:
            logger.error(f"❌ Balance fetch failed for {symbol}: {e}")
            return 0.0
        
        if usdt_free < MIN_BALANCE_FOR_TRADE:
            logger.warning(f"⚠️ Insufficient capital for {symbol}: ${usdt_free:.2f}")
            return 0.0
        
        # === 2. 성과 기반 가중치 적용 ===
        default_weight = 1.0 / len(self.symbols) if self.symbols else 0.5
        allocation_weight = self.symbol_allocation_weights.get(symbol, default_weight)
        
        # [Engine alignment] Dynamic risk applies only when explicitly enabled.
        regime_mult = 1.0
        use_dynamic_risk = bool(params.get('USE_DYNAMIC_RISK', False))
        if use_dynamic_risk:
            strong_hurst = params.get('STRONG_REGIME_HURST', 0.60)
            strong_natr = params.get('STRONG_REGIME_NATR', 1.5)
            strong_mult = params.get('STRONG_REGIME_MULTIPLIER', 1.5)
            weak_hurst = params.get('WEAK_REGIME_HURST', 0.55)
            weak_mult = params.get('WEAK_REGIME_MULTIPLIER', 0.5)
            panic_natr = params.get('PANIC_REGIME_NATR', 4.0)
            panic_mult = params.get('PANIC_REGIME_MULTIPLIER', 0.25)

            if natr > panic_natr:
                regime_mult = panic_mult
                logger.info(f"😱 Panic Regime detected (NATR {natr:.2f}). Mult: {panic_mult}")
            elif hurst > strong_hurst and natr > strong_natr:
                regime_mult = strong_mult
                logger.info(f"💪 Strong Regime detected (Hurst {hurst:.2f}, NATR {natr:.2f}). Mult: {strong_mult}")
            elif hurst < weak_hurst:
                regime_mult = weak_mult
                logger.info(f"🧊 Weak Regime detected (Hurst {hurst:.2f}). Mult: {weak_mult}")
            
        allocation_weight *= regime_mult
        
        allocated_capital = total_balance * allocation_weight
        
        # === 3. 전략 파라미터 ===
        leverage = params.get('LEVERAGE', 1)
        if leverage <= 0:
            logger.warning(f"⚠️ Invalid leverage for {symbol}: {leverage}. Using 1x.")
            leverage = 1
        
        risk_per_trade = params.get(
            'RISK_PER_TRADE_FUTURES', 
            params.get('RISK_PER_TRADE', 0.02)
        )
        
        # === 4. Stop Loss Distance 계산 ===
        stop_distance_pct = 0.05  # Default fallback
        
        if atr > 0 and price > 0:
            atr_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
            stop_distance = atr * atr_mult
            stop_distance_pct = stop_distance / price
            # Clamp to reasonable range (0.5% ~ 10%)
            stop_distance_pct = max(0.005, min(stop_distance_pct, 0.10))
        
        # === 5. Sizing Calculation ===
        risk_amt = allocated_capital * risk_per_trade
        
        # Division by zero 방어
        if stop_distance_pct <= 0:
            logger.error(
                f"❌ Invalid stop_distance_pct for {symbol}: {stop_distance_pct}. "
                "Cannot calculate position size."
            )
            return 0.0
        
        notional_value = risk_amt / stop_distance_pct
        max_tradeable_notional = usdt_free * leverage
        final_notional = min(notional_value, max_tradeable_notional)
        
        constraints = self.client.get_symbol_constraints(symbol)
        exchange_min_cost = float(constraints.get('min_cost') or 0.0)
        effective_min_order_value = max(float(MIN_ORDER_VALUE_USDT), exchange_min_cost)

        # === 6. 최소 주문 금액 체크 ===
        if final_notional < effective_min_order_value:
            logger.warning(
                f"⚠️ Calculated size too small for {symbol}: ${final_notional:.2f} "
                f"< Min ${effective_min_order_value:.2f}. Increase weight or balance."
            )
            return 0.0
        
        # === 7. 수량 계산 (Quantity) ===
        raw_quantity = final_notional / price
        
        # === 8. 거래소 정밀도 적용 (Precision) ===
        quantity = self.client.round_amount(symbol, raw_quantity)
        
        # === 9. 최종 검증 ===
        # a. 수량이 0이 아닌지
        if quantity <= 0:
            logger.warning(
                f"⚠️ Calculated quantity is zero for {symbol}. "
                f"Raw: {raw_quantity:.6f}"
            )
            return 0.0

        min_amount = float(constraints.get('min_amount') or 0.0)
        if min_amount > 0 and quantity < min_amount:
            logger.warning(
                f"⚠️ Quantity below exchange minimum for {symbol}: {quantity} < {min_amount}"
            )
            return 0.0
        
        # b. 최소 주문 금액 재확인 (정밀도 적용 후)
        final_order_value = quantity * price
        if final_order_value < effective_min_order_value:
            logger.warning(
                f"⚠️ Final order value too small for {symbol}: "
                f"${final_order_value:.2f} (Qty: {quantity}). "
                f"Min ${effective_min_order_value:.2f} required."
            )
            return 0.0
        
        # === 10. 상세 로깅 ===
        logger.info(
            f"🧮 Sizing {symbol} (Weight {allocation_weight*100:.0f}%, Leverage {leverage}x): "
            f"Total ${total_balance:.0f} | Alloc ${allocated_capital:.0f}"
        )
        logger.info(
            f"   -> Risk: ${risk_amt:.1f} ({risk_per_trade*100:.1f}% of Alloc) | "
            f"StopDist {stop_distance_pct*100:.2f}%"
        )
        logger.info(
            f"   -> Target Size: ${notional_value:.1f} | "
            f"Final: ${final_notional:.1f} ({quantity} {symbol.split('/')[0]})"
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
    
    def _is_entry_time(self, timeframe: str) -> bool:
        """
        현재 시간이 타임프레임별 진입(봉 마감) 시점인지 확인
        예: 4h봉 -> 00, 04, 08, 12, 16, 20시의 0분~5분 사이인지
        """
        now = self._get_reference_now_utc()
        entry_window_min = 2
        tf = str(timeframe or '').strip().lower()

        if tf.endswith('m'):
            interval = self._timeframe_to_minutes(tf)
            if interval <= 0:
                return False
            return (now.minute % interval) <= entry_window_min
        
        elif tf.endswith('h'):
            interval_min = self._timeframe_to_minutes(tf)
            if interval_min <= 0:
                return False
            interval = max(1, interval_min // 60)
            return (now.hour % interval) == 0 and now.minute <= entry_window_min
        
        elif tf.endswith('d'):
            return now.hour == 0 and now.minute <= entry_window_min  # 00:00 UTC 기준
        
        return False
    
    def _cache_indicators(self, symbol: str, indicators: dict):
        """지표값을 메모리에 캐싱 (indicator TF 기준 재사용)"""
        if not hasattr(self, '_indicator_cache'):
            self._indicator_cache = {}
        self._indicator_cache[symbol] = indicators
    
    def _get_cached_indicators(self, symbol: str) -> dict:
        """캐시된 지표값 조회"""
        if not hasattr(self, '_indicator_cache'):
            self._indicator_cache = {}
        return self._indicator_cache.get(symbol, {})

    def _get_candle_slot_id(self, timeframe: str) -> str:
        """현재 타임프레임 기준의 고유 캔들 ID 생성 (중복 계산 방지용)"""
        now = self._get_reference_now_utc()
        tf = str(timeframe or '').strip().lower()
        if tf.endswith('m'):
            interval = self._timeframe_to_minutes(tf)
            if interval <= 0:
                return now.strftime("%Y%m%d%H%M")
            slot = (now.minute // interval) * interval
            return now.strftime(f"%Y%m%d%H{slot:02d}")
        elif tf.endswith('h'):
            interval_min = self._timeframe_to_minutes(tf)
            if interval_min <= 0:
                return now.strftime("%Y%m%d%H%M")
            interval = max(1, interval_min // 60)
            slot = (now.hour // interval) * interval
            return now.strftime(f"%Y%m%d{slot:02d}00")
        elif tf.endswith('d'):
            return now.strftime("%Y%m%d0000")
        return now.strftime("%Y%m%d%H%M")

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
                # [FIX] 루프 시작 시간 기록 (정확한 주기 유지)
                loop_start_time = time.time()
                
                # 각 심볼 처리
                for symbol in self.symbols:
                    if self._shutdown_requested:
                        break
                    
                    # 1. 실행 (execute_logic 내부에서 진입 시점 체크)
                    try:
                        self.execute_logic(symbol)
                    except Exception as e:
                        logger.error(f"⚠️ Error processing {symbol}: {e}")

                    time.sleep(SYMBOL_DELAY_SECONDS)
                
                # 헬스체크 업데이트
                positions = self._get_current_positions()
                self.health_manager.update_heartbeat(
                    status="running",
                    positions=positions
                )
                
                # 클라우드 최적화 실행 (시간 기반 체크)
                if self.cloud_optimizer:
                    now = datetime.utcnow()
                    
                    # 1. 시간 동기화 검증 (매 루프)
                    if not self.cloud_optimizer.check_time_sync_ntp():
                        logger.error("⏰ Time drift detected! Bot may fail to place orders on Binance.")
                    
                    # 2. 리소스 모니터링 (10분마다) & 메모리 보호
                    if (now - self._last_resource_check).total_seconds() >= 600:  # 10분 = 600초
                        usage = self.cloud_optimizer.log_resource_usage()
                        if usage.get('memory_percent', 0) > 85.0:
                            logger.warning(f"⚠️ High Memory ({usage.get('memory_percent')}%) detected. Forcing GC...")
                            self.cloud_optimizer.force_gc()
                        self._last_resource_check = now
                    
                    # 3. DB 정리 (24시간마다)
                    if (now - self._last_db_cleanup).total_seconds() >= 86400:  # 24시간 = 86400초
                        self.cloud_optimizer.cleanup_db_old_records(
                            TRADE_HISTORY_DB, 
                            days_to_keep=90
                        )
                        self._last_db_cleanup = now
                    
                    # 4. 명시적 GC (2시간마다)
                    if (now - self._last_gc).total_seconds() >= 7200:  # 2시간 = 7200초
                        self.cloud_optimizer.force_gc()
                        self._last_gc = now
                
                # [FIX] 심볼 처리 시간을 고려한 동적 대기 시간 계산
                # 목표: 전체 루프 주기를 정확히 LOOP_INTERVAL_SECONDS(10초)로 유지
                elapsed_processing = time.time() - loop_start_time
                target_interval = float(LOOP_INTERVAL_SECONDS)
                # 처리 시간이 길어지더라도 최소 0.5초는 대기하여 CPU 과부하 방지
                adjusted_wait = max(0.5, target_interval - elapsed_processing)
                
                logger.debug(
                    f"💤 Loop took {elapsed_processing:.2f}s. "
                    f"Sleeping {adjusted_wait:.2f}s (Target: {target_interval:.1f}s cycle)"
                )
                
                # Shutdown 체크하면서 대기
                start_wait = time.time()
                while time.time() - start_wait < adjusted_wait:
                    if self._shutdown_requested:
                        break
                    time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"🚨 Critical Error in Main Loop: {e}")
                self.health_manager.record_error(e)
                self.health_manager.update_heartbeat(status="error")
                time.sleep(10) # 에러 시 10초 대기 후 재시도
        
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

