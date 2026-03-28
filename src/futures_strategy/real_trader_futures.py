"""
RealTrader Futures - 24시간 자동 2D 마진 공유 포트폴리오 봇 (Production Grade)
===================================================================
P0/P1 개선사항 적용:
- 2D 마진 공유 3-Phase 아키텍처 (Exit -> Scan -> Rank & Allocate)
- Z-Score, Seasonality(Datetime) 데이터 파이프라인 완벽 동기화
- 수수료/슬리피지 5% 마진 버퍼 적용 (Insufficient Margin 에러 원천 차단)
- Golden 6 심볼(AVAX, DOGE, ETH, LINK, NEAR, SUI) 필터링 자동화
"""

import os
import sys
import time
import math
import signal
import json
import logging
import gc
import hashlib
import uuid
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    BINANCE_API_KEY,
    BINANCE_SECRET,
    LOG_DIR,
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
    FUTURES_STATE_FILE,
    SLIPPAGE_RATE,
    FUTURES_LIVE_SYMBOLS,
)
from src.futures_strategy.binance_client import BinanceClient, OrderRateLimiter
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.common.utils import setup_logger
from src.common.components import TradeHistoryDB, HealthCheckManager

# Oracle Cloud 최적화 (선택적)
try:
    from src.common.cloud_optimizer import CloudOptimizer

    CLOUD_OPTIMIZER_AVAILABLE = True
except ImportError:
    CLOUD_OPTIMIZER_AVAILABLE = False

logger = setup_logger("RealTraderFutures")

_CCXT_TRANSIENT_ERRORS: Tuple[type, ...] = ()
if ccxt is not None:
    _CCXT_TRANSIENT_ERRORS = tuple(
        err
        for err in (
            getattr(ccxt, "NetworkError", None),
            getattr(ccxt, "ExchangeNotAvailable", None),
            getattr(ccxt, "RequestTimeout", None),
            getattr(ccxt, "RateLimitExceeded", None),
        )
        if isinstance(err, type)
    )


def _is_retryable_api_exception(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if _CCXT_TRANSIENT_ERRORS and isinstance(exc, _CCXT_TRANSIENT_ERRORS):
        return True

    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        if isinstance(cause, (ConnectionError, TimeoutError)):
            return True
        if _CCXT_TRANSIENT_ERRORS and isinstance(cause, _CCXT_TRANSIENT_ERRORS):
            return True

    error_text = str(exc).lower()
    if ("recvwindow" in error_text) or ("outside of the recvwindow" in error_text):
        return True
    if ("timestamp for this request is outside" in error_text) or ("code\":-1021" in error_text):
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
                wait_time = min(
                    float(API_RETRY_WAIT_MIN) * (2**attempt), float(API_RETRY_WAIT_MAX)
                )
                logger.warning(
                    f"API error in {func.__name__} (attempt {attempt+1}/{max_attempts}): {e}. Waiting {wait_time:.1f}s"
                )
                time.sleep(max(0.0, wait_time))
        raise (
            last_error
            if last_error is not None
            else RuntimeError("retry wrapper reached unexpected state")
        )

    return wrapper


class StateManager:
    """거래 상태 관리 (진입 시간, 진입가 등 로컬 저장)"""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._memory_cache: Dict[str, Any] = {}
        self._cache_initialized: bool = False
        self._thread_lock = threading.Lock()
        self._dirty: bool = False
        self._last_flush: float = 0.0
        self._flush_interval: float = 1.0
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.state_file.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_unlocked({})

    def _load_unlocked(self) -> dict:
        if self._cache_initialized:
            return self._memory_cache
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._memory_cache = data
                self._cache_initialized = True
                return data
        except Exception as e:
            logger.error(f"State load error: {e}")
            return {}

    def _save_unlocked(self, state: dict):
        tmp_file = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(tmp_file, self.state_file)
            except OSError:
                if self.state_file.exists():
                    self.state_file.unlink()
                os.rename(tmp_file, self.state_file)
            self._memory_cache = state
            self._cache_initialized = True
        except Exception as e:
            logger.error(f"State save error: {e}")
            try:
                tmp_file.unlink()
            except Exception:
                pass

    def _maybe_flush(self) -> None:
        if not self._dirty:
            return
        now = time.time()
        if (now - self._last_flush) >= self._flush_interval:
            self._save_unlocked(self._memory_cache)
            self._dirty = False
            self._last_flush = now

    def get_symbol_state(self, symbol: str) -> dict:
        with self._thread_lock:
            if not self._cache_initialized:
                self._memory_cache = self._load_unlocked()
                self._cache_initialized = True
            return dict(self._memory_cache.get(symbol, {}))

    def update_symbol_state(self, symbol: str, data: dict):
        with self._thread_lock:
            if not self._cache_initialized:
                self._memory_cache = self._load_unlocked()
                self._cache_initialized = True
            if symbol not in self._memory_cache:
                self._memory_cache[symbol] = {}
            self._memory_cache[symbol].update(data)
            self._dirty = True
            self._maybe_flush()

    def clear_symbol_state(
        self, symbol: str, preserve_keys: Optional[list] = None
    ) -> None:
        with self._thread_lock:
            if not self._cache_initialized:
                self._memory_cache = self._load_unlocked()
                self._cache_initialized = True
            if symbol in self._memory_cache:
                preserved: Dict[str, Any] = {}
                if preserve_keys:
                    preserved = {
                        k: self._memory_cache[symbol].get(k)
                        for k in preserve_keys
                        if k in self._memory_cache[symbol]
                    }
                self._memory_cache[symbol] = preserved
                self._dirty = True
                self._maybe_flush()

    def flush_now(self) -> None:
        with self._thread_lock:
            if self._dirty:
                self._save_unlocked(self._memory_cache)
                self._dirty = False
                self._last_flush = time.time()


class RealTraderFutures:
    def __init__(self, enable_oracle_optimization: bool = False, dry_run: bool = False):
        self.client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
        self.clients: Dict[str, BinanceClient] = {}
        self.strategies: Dict[str, UltimateStrategy] = {}
        self.params_map: Dict[str, dict] = {}
        self.symbols: list = []

        self.dry_run: bool = bool(dry_run)

        self._indicator_cache: Dict[str, dict] = {}
        self._exit_indicator_cache: Dict[str, dict] = {}
        self._cache_lock = threading.Lock()

        self.trade_db = TradeHistoryDB(TRADE_HISTORY_DB)
        self.health_manager = HealthCheckManager(HEARTBEAT_FILE)
        self.state_manager = StateManager(FUTURES_STATE_FILE)

        self.cloud_optimizer = None
        if enable_oracle_optimization and CLOUD_OPTIMIZER_AVAILABLE:
            self.cloud_optimizer = CloudOptimizer()
            logger.info("☁️ Cloud optimization enabled")

        self._shutdown_requested = False
        self.last_calc_candle: Dict[str, str] = {}
        self.last_exit_calc_candle: Dict[str, str] = {}
        self._server_time_offset_ms: int = 0
        self._last_server_time_sync: datetime = datetime.min

        self._last_resource_check = datetime.utcnow()
        self._last_db_cleanup = datetime.utcnow()
        self._last_gc = datetime.utcnow()
        self._last_ntp_check = datetime.min
        self._log_last_emit_ts: Dict[str, float] = {}
        self._log_last_message: Dict[str, str] = {}
        self._log_throttle_lock = threading.Lock()

        self._db_write_lock = threading.Lock()
        self._time_sync_lock = threading.Lock()

        self._executor: ThreadPoolExecutor | None = None

        self._setup_signal_handlers()

    def _should_emit_log(
        self,
        key: str,
        interval_seconds: float = 0.0,
        message: Optional[str] = None,
        emit_on_change: bool = False,
    ) -> bool:
        with self._log_throttle_lock:
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

    def _log_throttled(
        self,
        level: str,
        key: str,
        message: str,
        interval_seconds: float,
        emit_on_change: bool = False,
    ) -> None:
        if not self._should_emit_log(
            key=key,
            interval_seconds=interval_seconds,
            message=message,
            emit_on_change=emit_on_change,
        ):
            return
        getattr(logger, level, logger.info)(message)

    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal_handler)

    def _resolve_exchange_leverage(self, leverage_value: Any) -> int:
        try:
            target_lev = float(leverage_value)
        except (TypeError, ValueError):
            target_lev = 1.0
        target_lev = max(1.0, min(target_lev, float(MAX_EXCHANGE_LEVERAGE)))
        return int(math.ceil(target_lev))

    def load_strategies_from_json(self):
        """results 폴더의 JSON 파일에서 최적화된 파라미터 로드 (Golden 6 코인 필터링)"""
        logger.info("Loading strategies from JSON files in results/...")
        results_dir = os.path.join(project_root, "results")

        # [FIX] 명확하게 multi 포트폴리오 파일을 지정하여 오인 로드 방지
        multi_path = os.path.join(results_dir, "best_params_multi_4h.json")

        if os.path.exists(multi_path):
            logger.info(
                "Detected multi-portfolio config: %s",
                os.path.basename(multi_path),
            )
            try:
                with open(multi_path, "r", encoding="utf-8") as f:
                    base_params = json.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to parse multi-portfolio JSON: {e}")
                raise

            # [FIX 3] 엄격한 OOS 검증을 통과한 심볼만 실거래 대상으로 승인 (settings.py 참조)
            self.symbols = list(FUTURES_LIVE_SYMBOLS)
            logger.info(
                "Live multi-portfolio symbols: %s", ", ".join(self.symbols)
            )

            for symbol in self.symbols:
                clean_sym = symbol.replace("/", "")
                symbol_params = dict(base_params)
                symbol_params.setdefault("TIMEFRAME", "4h")
                symbol_params.setdefault("INDICATOR_TIMEFRAME", "4h")
                self.params_map[symbol] = symbol_params
                strategy_name = f"Real_{clean_sym}"
                self.strategies[symbol] = UltimateStrategy(strategy_name, symbol_params)
                logger.info("Strategy initialized for: %s", symbol)
            return
        else:
            raise FileNotFoundError(
                "❌ Multi-portfolio JSON file not found in results/ directory."
            )

    @network_api_retry
    def _fetch_balance_safe(self, symbol: Optional[str] = None) -> tuple:
        client = (
            self._get_client_for_symbol(symbol) if symbol else self.client
        )
        return client.fetch_balance()

    @network_api_retry
    def _fetch_ohlcv_safe(self, symbol: str, timeframe: str, start_str: str):
        client = self._get_client_for_symbol(symbol)
        return client.fetch_ohlcv(symbol, timeframe, start_date=start_str)

    @network_api_retry
    def _fetch_ohlcv_with_taker_safe(
        self,
        symbol: str,
        timeframe: str,
        start_str: str,
    ) -> pd.DataFrame:
        """Taker volume 포함 OHLCV 안전 래퍼 (공통 예외/리트라이 파이프라인 사용)."""
        client = self._get_client_for_symbol(symbol)
        return client.fetch_ohlcv_with_taker(symbol, timeframe, start_date=start_str)

    @network_api_retry
    def _fetch_recent_ohlcv_safe(self, symbol: str, timeframe: str, limit: int = 3):
        client = self._get_client_for_symbol(symbol)
        return client.fetch_recent_ohlcv(symbol, timeframe, limit=limit)

    @network_api_retry
    def _fetch_position_safe(self, symbol: str) -> dict:
        client = self._get_client_for_symbol(symbol)
        return client.fetch_position(symbol)

    @network_api_retry
    def _get_market_price_safe(self, symbol: str) -> float:
        client = self._get_client_for_symbol(symbol)
        return client.get_market_price(symbol)

    @network_api_retry
    def _fetch_server_time_ms_safe(self) -> int:
        result = self.client.fetch_server_time_ms()
        if result is None:
            raise ConnectionError("Server time returned None")
        return int(result)

    def _sync_server_time_offset(
        self, force: bool = False, sync_interval_seconds: int = 60
    ):
        with self._time_sync_lock:
            now = datetime.utcnow()
            minutes_in_4h_cycle = (now.hour * 60 + now.minute) % 240
            near_boundary = minutes_in_4h_cycle >= 239 or minutes_in_4h_cycle <= 0
            effective_interval = 5 if near_boundary else max(5, int(sync_interval_seconds))
            elapsed = (now - self._last_server_time_sync).total_seconds()
            if (not force) and elapsed < effective_interval:
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
        self._sync_server_time_offset(force=False)
        return int(datetime.utcnow().timestamp() * 1000) + int(
            self._server_time_offset_ms
        )

    def _get_reference_now_utc(self) -> datetime:
        return datetime.utcfromtimestamp(self._get_reference_now_ms() / 1000.0)

    def _default_entry_grace_seconds(
        self, execution_tf: str = "1h", params: Optional[dict] = None
    ) -> float:
        cfg = params or {}
        tf_min = self._timeframe_to_minutes(execution_tf)
        tf_seconds = float(max(60, (tf_min * 60) if tf_min > 0 else 3600))
        grace_ratio = float(cfg.get("ENTRY_EXECUTION_GRACE_RATIO", 0.20))
        grace_min = float(cfg.get("ENTRY_EXECUTION_GRACE_MIN_SECONDS", 5.0))
        grace_max = float(cfg.get("ENTRY_EXECUTION_GRACE_MAX_SECONDS", 30.0))
        symbol_count = max(1, len(self.symbols))
        loop_cycle_est = float(LOOP_INTERVAL_SECONDS) + (
            float(SYMBOL_DELAY_SECONDS) * float(symbol_count)
        )
        base = max(tf_seconds * grace_ratio, loop_cycle_est * 1.1)
        return max(grace_min, min(grace_max, base))

    def _resolve_entry_market_fallback(
        self, params: dict, atr: float, current_price: float, entry_lag_sec: float
    ) -> bool:
        explicit_flag = params.get("ENTRY_ALLOW_MARKET_FALLBACK", None)
        if explicit_flag is not None:
            return bool(explicit_flag)
        mode = str(params.get("ENTRY_EXECUTION_MODE", "balanced")).strip().lower()
        if mode in ("maker_strict", "strict", "maker"):
            return False
        if mode in ("always_taker", "taker"):
            return True
        vol_pct = (
            ((float(atr) / float(current_price)) * 100.0)
            if current_price and atr
            else 0.0
        )
        lag_trigger = float(params.get("ENTRY_MARKET_FALLBACK_LAG_SECONDS", 8.0))
        vol_trigger = float(params.get("ENTRY_MARKET_FALLBACK_VOL_PCT", 1.0))
        lag_sec = max(0.0, float(entry_lag_sec))
        if mode == "balanced":
            return (lag_sec >= (lag_trigger * 0.6)) or (vol_pct >= (vol_trigger * 0.75))
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
        expected = abs(float(expected_qty) if expected_qty else 0.0)
        actual = abs(float(confirmed_amount) if confirmed_amount else 0.0)

        if expected <= 0 or actual <= 0:
            return actual > 0

        min_fill_ratio = float(params.get("ENTRY_MIN_FILL_RATIO", 0.60))
        min_fill_ratio = max(0.0, min(1.0, min_fill_ratio))

        if min_fill_ratio <= 0.0:
            return True

        fill_ratio = actual / expected
        if fill_ratio < min_fill_ratio:
            logger.warning(
                "⚠️ [%s] Underfilled entry (Ratio: %.2f < %.2f). "
                "Accepting partial fill (%.8f) to ensure SL/TP protection.",
                symbol,
                fill_ratio,
                min_fill_ratio,
                actual,
            )

        return True

    def _position_state_missing_core(self, state: Optional[dict]) -> bool:
        if not state:
            return True
        side = str(state.get("side", "") or "").upper()
        if side not in ("LONG", "SHORT"):
            return True
        try:
            entry_price = float(state.get("entry_price", 0.0) or 0.0)
        except Exception:
            entry_price = 0.0
        if not (np.isfinite(entry_price) and entry_price > 0.0):
            return True
        return False

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
        try:
            side = "LONG" if amount > 0 else "SHORT"
            entry_price = float(pos.get("entryPrice", 0.0) or current_price or 0.0)
            entry_atr = float(atr) if np.isfinite(atr) and atr > 0 else 0.0

            sl_type = str(params.get("STOP_LOSS_TYPE", "ATR") or "ATR").upper()
            if sl_type == "ATR" and entry_atr > 0:
                sl_mult = float(
                    params.get("LONG_ATR_MULT" if amount > 0 else "SHORT_ATR_MULT", 2.0)
                )
                stop_price = (
                    entry_price - (entry_atr * sl_mult)
                    if amount > 0
                    else entry_price + (entry_atr * sl_mult)
                )
            else:
                sl_pct = float(params.get("STOP_LOSS_PCT", 0.02))
                stop_price = (
                    entry_price * (1 - sl_pct)
                    if amount > 0
                    else entry_price * (1 + sl_pct)
                )
            client = self._get_client_for_symbol(symbol)
            stop_price = float(client.round_price(symbol, stop_price))

            tp_price = 0.0
            use_tp_entry = bool(params.get("USE_TAKE_PROFIT", False))
            if use_tp_entry and entry_atr > 0:
                tp_atr_mult = float(
                    params.get(
                        "TAKE_PROFIT_ATR_MULT_FUTURES",
                        params.get("TAKE_PROFIT_ATR_MULT", 3.0),
                    )
                )
                raw_tp = (
                    entry_price + (entry_atr * tp_atr_mult)
                    if amount > 0
                    else entry_price - (entry_atr * tp_atr_mult)
                )
                client = self._get_client_for_symbol(symbol)
                tp_price = float(client.round_price(symbol, raw_tp))

            now_ref_ms = int(self._get_reference_now_ms())
            self.state_manager.update_symbol_state(
                symbol,
                {
                    "entry_time": datetime.utcfromtimestamp(
                        now_ref_ms / 1000.0
                    ).isoformat(),
                    "entry_fill_time": datetime.utcnow().isoformat(),
                    "entry_price": float(entry_price),
                    "entry_atr": float(entry_atr),
                    "side": side,
                    "sl_required": True,
                    "last_sl_order_time": None,
                    "active_stop_price": float(stop_price),
                    "tp_price": float(tp_price),
                    "pos_atr": float(entry_atr),
                    "highest_price": float(entry_price),
                    "lowest_price": float(entry_price),
                    "recovery_bootstrapped": True,
                    "has_scaled_out": True,
                    "initial_amount": float(abs(amount)),
                },
            )
            logger.warning("[%s] Bootstrapped local state from live position.", symbol)
            return True
        except Exception as e:
            logger.error(f"❌ [{symbol}] Failed to bootstrap local state: {e}")
            return False

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
        order_type: str = "MARKET",
        client_order_id: Optional[str] | None = None,
    ):
        """
        주문 실행 래퍼.
        - dry_run=True  -> 실제 주문 대신 DRY-RUN 더미 응답 반환
        - dry_run=False -> BinanceClient.place_order_smart로 실제 주문 실행
        """
        client = self._get_client_for_symbol(symbol)

        if self.dry_run:
            logger.info(
                "[DRY-RUN] place_order_smart(%s, side=%s, qty=%.8f, reduce_only=%s)",
                symbol,
                side,
                qty,
                reduce_only,
            )
            return {
                "id": "dry-run-order",
                "symbol": symbol,
                "side": side,
                "amount": float(qty),
                "price": float(current_price) if current_price is not None else None,
                "reduce_only": bool(reduce_only),
                "status": "closed",
            }

        if order_type.upper() == "LIMIT":
            params_dict: dict[str, object] = {}
            if reduce_only:
                params_dict["reduceOnly"] = True
            cid = client_order_id or ("RT_LMT_" + uuid.uuid4().hex[:20])
            params_dict["clientOrderId"] = cid

            exchange_symbol = symbol
            if hasattr(client, "get_market_symbol"):
                exchange_symbol = client.get_market_symbol(symbol)
            else:
                if "/" in symbol and ":" not in symbol:
                    base, quote = symbol.split("/", 1)
                    if quote in ["USDT", "USDC"]:
                        exchange_symbol = f"{symbol}:{quote}"

            try:
                client.rate_limiter.wait_if_needed()
                return client.exchange.create_order(
                    symbol=exchange_symbol,
                    type="limit",
                    side=side.lower(),
                    amount=qty,
                    price=current_price,
                    params=params_dict,
                )
            except ccxt.RequestTimeout:
                logger.warning(
                    "[%s] LIMIT order timeout. Reconciling with exchange...",
                    symbol,
                )
                open_orders = client.exchange.fetch_open_orders(
                    exchange_symbol
                )
                for order in open_orders:
                    if order.get("clientOrderId") == cid:
                        logger.info(
                            "[%s] LIMIT order reached exchange despite timeout.",
                            symbol,
                        )
                        return order
                logger.error(
                    "[%s] LIMIT order lost in transit. No retry to avoid double entry.",
                    symbol,
                )
                return None
            except Exception as e:
                logger.error("❌ LIMIT order failed for %s: %s", symbol, e)
                raise

        try:
            return client.place_order_smart(
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
                client_order_id=client_order_id,
            )
        except ccxt.RequestTimeout:
            logger.error(
                "[%s] Market order timeout. No reconciliation (no clientOrderId). No retry.",
                symbol,
            )
            return None

    def _place_stop_loss_safe(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        client_order_id: Optional[str] | None = None,
    ):
        """
        서버 사이드 Stop Loss 주문 실행 래퍼.
        - dry_run=True  -> DRY-RUN 더미 SL 주문
        - dry_run=False -> BinanceClient.place_stop_market_order 호출
        """
        client = self._get_client_for_symbol(symbol)

        if self.dry_run:
            logger.info(
                "[DRY-RUN] place_stop_market_order(%s, side=%s, qty=%.8f, stop=%.8f)",
                symbol,
                side,
                qty,
                stop_price,
            )
            return {
                "id": "dry-run-sl",
                "symbol": symbol,
                "side": side,
                "amount": float(qty),
                "stopPrice": float(stop_price),
                "status": "open",
                "type": "STOP_MARKET",
            }

        if client_order_id is None:
            unique_string = f"SL_{symbol}_{side}_{qty}_{stop_price}_{uuid.uuid4().hex}"
            client_order_id = "RT_" + hashlib.md5(unique_string.encode()).hexdigest()[:20]

        try:
            return client.place_stop_market_order(
                symbol,
                side,
                qty,
                stop_price,
                client_order_id=client_order_id,
            )
        except ccxt.RequestTimeout as e:
            logger.warning(
                "[%s] SL Order Timeout. Reconciling with exchange...", symbol
            )
            open_orders = client.fetch_open_orders(symbol)
            for order in open_orders:
                if order.get("clientOrderId") == client_order_id:
                    logger.info(
                        "[%s] SL Order successfully reached exchange despite timeout.",
                        symbol,
                    )
                    return order
            logger.error(
                "[%s] SL Order lost in transit. Needs manual intervention or next loop retry.",
                symbol,
            )
            return None
        except Exception as e:
            logger.error("[%s] SL Order failed: %s", symbol, e)
            return None

    def _place_take_profit_safe(
        self,
        symbol: str,
        side: str,
        qty: float,
        tp_price: float,
        client_order_id: Optional[str] = None,
    ):
        """
        서버 사이드 Take Profit 주문 실행 래퍼.
        - dry_run=True  -> DRY-RUN 더미 TP 주문
        - dry_run=False -> BinanceClient.place_take_profit_market_order (있을 때만)
        """
        client = self._get_client_for_symbol(symbol)

        if self.dry_run:
            logger.info(
                "[DRY-RUN] place_take_profit_market_order(%s, side=%s, qty=%.8f, tp=%.8f)",
                symbol,
                side,
                qty,
                tp_price,
            )
            return {
                "id": "dry-run-tp",
                "symbol": symbol,
                "side": side,
                "amount": float(qty),
                "tpPrice": float(tp_price),
                "status": "open",
                "type": "TAKE_PROFIT_MARKET",
            }

        if client_order_id is None:
            unique_string = f"TP_{symbol}_{side}_{qty}_{tp_price}_{uuid.uuid4().hex}"
            client_order_id = "RT_" + hashlib.md5(
                unique_string.encode()
            ).hexdigest()[:20]

        if not hasattr(client, "place_take_profit_market_order"):
            return None
        try:
            return client.place_take_profit_market_order(
                symbol,
                side,
                qty,
                tp_price,
                client_order_id=client_order_id,
            )
        except ccxt.RequestTimeout as e:
            logger.warning(
                "[%s] TP Order Timeout. Reconciling with exchange...", symbol
            )
            open_orders = client.fetch_open_orders(symbol)
            for order in open_orders:
                if order.get("clientOrderId") == client_order_id:
                    logger.info(
                        "[%s] TP Order successfully reached exchange despite timeout.",
                        symbol,
                    )
                    return order
            logger.error(
                "[%s] TP Order lost in transit. Needs manual intervention or next loop retry.",
                symbol,
            )
            return None
        except Exception as e:
            logger.error("[%s] TP Order failed: %s", symbol, e)
            return None

    @network_api_retry
    def _cancel_all_orders_safe(self, symbol: str):
        """
        모든 오픈 주문 취소 래퍼.
        - dry_run=True  -> 실제 취소 대신 DRY-RUN 로그만 남김
        - dry_run=False -> BinanceClient.cancel_all_orders 호출
        """
        client = self._get_client_for_symbol(symbol)

        if self.dry_run:
            logger.info("[DRY-RUN] cancel_all_orders(%s)", symbol)
            return True

        self._sync_server_time_offset(force=True)
        return client.cancel_all_orders(symbol)

    def _resolve_timeframes(self, params: dict) -> Tuple[str, str]:
        execution_tf = str(params.get("TIMEFRAME", "1h"))
        indicator_tf = str(params.get("INDICATOR_TIMEFRAME", "4h"))
        return execution_tf, indicator_tf

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        tf_raw = str(timeframe or "").strip()
        tf = tf_raw.lower()
        try:
            if tf_raw.endswith("M"):
                return max(1, int(tf_raw[:-1]) * 43200)
            if tf.endswith("m"):
                return max(1, int(tf[:-1]))
            if tf.endswith("h"):
                return max(1, int(tf[:-1]) * 60)
            if tf.endswith("d"):
                return max(1, int(tf[:-1]) * 1440)
            if tf.endswith("w"):
                return max(1, int(tf[:-1]) * 10080)
        except ValueError:
            return -1
        return -1

    def _get_candle_slot_id(self, timeframe: str) -> str:
        """현재 시간 기준 캔들 슬롯 ID (지표 캐싱용 키)."""
        tf_min = self._timeframe_to_minutes(timeframe)
        if tf_min <= 0:
            return "unknown"
        now_ms = self._get_reference_now_ms()
        interval_ms = tf_min * 60 * 1000
        slot_start_ms = now_ms - (now_ms % interval_ms)
        return f"{timeframe}_{slot_start_ms}"

    def _select_last_closed_candle(
        self, df: pd.DataFrame, timeframe: str
    ) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        if "timestamp" not in df.columns:
            return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        interval_min = self._timeframe_to_minutes(timeframe)
        if interval_min <= 0:
            return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        interval_ms = interval_min * 60 * 1000
        now_ms = self._get_reference_now_ms()
        timestamps = (
            pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
        )
        closed_mask = (timestamps + interval_ms) <= now_ms
        closed_indices = df.index[closed_mask.to_numpy()]
        if len(closed_indices) > 0:
            return df.loc[closed_indices[-1]]
        return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]

    def _extract_candle_timestamp_ms(self, candle: pd.Series) -> int:
        if candle is None:
            return 0
        raw_ts = candle.get("timestamp", 0)
        try:
            return int(raw_ts) if not pd.isna(raw_ts) else 0
        except Exception:
            return 0

    def _cache_indicators(self, symbol: str, data: dict) -> None:
        """Indicator TF 기준 지표 캐싱 (엔트리/청산 공용)."""
        with self._cache_lock:
            self._indicator_cache[symbol] = data

    def _get_cached_indicators(self, symbol: str) -> dict:
        """캐시된 지표 조회 (없으면 빈 dict)."""
        with self._cache_lock:
            return dict(self._indicator_cache.get(symbol, {}))

    def _cache_exit_indicators(self, symbol: str, indicators: dict) -> None:
        with self._cache_lock:
            self._exit_indicator_cache[symbol] = indicators

    def _get_cached_exit_indicators(self, symbol: str) -> dict:
        with self._cache_lock:
            return dict(self._exit_indicator_cache.get(symbol, {}))

    def _refresh_exit_indicators_if_needed(
        self, symbol: str, strategy: UltimateStrategy, params: dict, execution_tf: str
    ) -> dict:
        current_slot = self._get_candle_slot_id(execution_tf)
        cached = self._get_cached_exit_indicators(symbol)
        with self._cache_lock:
            already_calculated = self.last_exit_calc_candle.get(symbol) == current_slot
        if cached and already_calculated:
            return cached

        lookback_bars = max(300, int(params.get("HURST_PERIOD", 200)) + 80)
        df: Optional[pd.DataFrame] = None
        signal_df: Optional[pd.DataFrame] = None
        try:
            df = self._fetch_recent_ohlcv_safe(
                symbol, execution_tf, limit=lookback_bars
                )
            if df is None or len(df) < 80:
                return cached

            if "datetime" not in df.columns and "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

            float_cols = ["open", "high", "low", "close", "volume"]
            df[float_cols] = df[float_cols].astype(np.float64)
            signal_df = strategy.generate_signals(df)
            last_candle = self._select_last_closed_candle(signal_df, execution_tf)
            del signal_df
            del df
            signal_df = None
            df = None
            if last_candle is None:
                return cached

            refreshed = {
                "trend_direction": int(
                    last_candle.get("trend_direction", 0)
                    if not pd.isna(last_candle.get("trend_direction"))
                    else 0
                ),
                "atr": float(
                    last_candle.get("atr", 0.0)
                    if not pd.isna(last_candle.get("atr"))
                    else 0.0
                ),
                "parabolic_sar": float(
                    last_candle.get("parabolic_sar", 0.0)
                    if not pd.isna(last_candle.get("parabolic_sar"))
                    else 0.0
                ),
                "rsi": float(
                    last_candle.get("rsi", 50.0)
                    if not pd.isna(last_candle.get("rsi"))
                    else 50.0
                ),
                "candle_open": float(last_candle.get("open", np.nan)),
                "candle_high": float(last_candle.get("high", np.nan)),
                "candle_low": float(last_candle.get("low", np.nan)),
                "candle_close": float(last_candle.get("close", np.nan)),
                "candle_ts": int(last_candle.get("timestamp", 0) or 0),
                "timeframe": execution_tf,
                "cached_at": datetime.utcnow().isoformat(),
            }
            self._cache_exit_indicators(symbol, refreshed)
            with self._cache_lock:
                self.last_exit_calc_candle[symbol] = current_slot
            return refreshed
        finally:
            if df is not None:
                del df
            if signal_df is not None:
                del signal_df

    def _confirm_position(
        self,
        symbol: str,
        expected_side: Optional[str] = None,
        retries: int = 6,
        sleep_seconds: float = 0.3,
    ) -> dict:
        last_pos = {
            "amount": 0.0,
            "entryPrice": 0.0,
            "unrealizedPnL": 0.0,
            "leverage": 1,
        }
        attempts = max(1, int(retries))
        for attempt in range(attempts):
            pos = self._fetch_position_safe(symbol)
            if pos:
                last_pos = pos
            amount = float(last_pos.get("amount", 0.0) or 0.0)
            if expected_side == "LONG" and amount > 0:
                return last_pos
            if expected_side == "SHORT" and amount < 0:
                return last_pos
            if expected_side is None and abs(amount) > 0:
                return last_pos
            if attempt < attempts - 1 and sleep_seconds > 0:
                time.sleep(float(sleep_seconds))
        return last_pos

    def _wait_until_position_flat(
        self, symbol: str, timeout_seconds: float = 6.0, poll_seconds: float = 0.3
    ) -> Tuple[bool, float]:
        flat_epsilon = 1e-8
        try:
            client = self._get_client_for_symbol(symbol)
            constraints = client.get_symbol_constraints(symbol)
            min_amount = float(constraints.get("min_amount") or 0.0)
            if min_amount > 0:
                flat_epsilon = min_amount * 0.25
        except Exception:
            pass
        deadline = time.time() + max(0.5, float(timeout_seconds))
        last_amount = 0.0
        while time.time() <= deadline:
            pos = self._fetch_position_safe(symbol)
            last_amount = float(pos.get("amount", 0.0) or 0.0)
            if abs(last_amount) <= flat_epsilon:
                return True, last_amount
            time.sleep(max(0.05, float(poll_seconds)))
        return False, last_amount

    def _get_client_for_symbol(self, symbol: str) -> BinanceClient:
        return self.clients.get(symbol, self.client)

    def initialize(self):
        logger.info("🤖 RealTrader Futures 2D Portfolio Bot Initializing...")
        self._sync_server_time_offset(force=True)
        self.load_strategies_from_json()
        gc.collect()

        try:
            total_balance, usdt_free = self._fetch_balance_safe()
            logger.info(
                "Account Balance: %.2f USDT (Total: %.2f)",
                usdt_free,
                total_balance,
            )
            if usdt_free < MIN_BALANCE_USDT:
                logger.warning(f"⚠️ Warning: Low balance (< {MIN_BALANCE_USDT} USDT)!")
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")

        global_rate_limiter = OrderRateLimiter(max_orders_per_10s=80)
        for symbol in list(self.symbols):
            self.clients[symbol] = BinanceClient(
                BINANCE_API_KEY,
                BINANCE_SECRET,
                shared_rate_limiter=global_rate_limiter,
            )
            client = self._get_client_for_symbol(symbol)
            target_lev = self.params_map[symbol].get("LEVERAGE", 1)
            applied_lev = self._resolve_exchange_leverage(target_lev)

            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    client.set_leverage(symbol, applied_lev)
                    client.set_margin_type(symbol, margin_type="CROSSED")
                    success = True
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "⚠️ Retry %d/%d: Failed to set leverage for %s. %s",
                            attempt + 1,
                            max_retries,
                            symbol,
                            e,
                        )
                        time.sleep(2.0)
                    else:
                        logger.error(
                            "❌ Critical: Could not set leverage for %s after %d attempts.",
                            symbol,
                            max_retries,
                        )
            if not success:
                if symbol in self.symbols:
                    self.symbols.remove(symbol)

        try:
            self.client.set_position_mode(dual_side_position=False)
        except Exception as e:
            logger.error(f"⚠️ Failed to set One-Way Mode: {e}")

        try:
            self.client.set_asset_mode(is_multi_asset=False)
        except Exception as e:
            logger.error(f"⚠️ Failed to set Single-Asset Mode: {e}")

        if self._executor is None and self.symbols:
            max_workers = min(3, len(self.symbols))
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

        if os.getenv("SKIP_NUMBA_WARMUP", "true").lower() != "true":
            try:
                from src.futures_strategy.engine_fast_futures import (
                    backtest_loop_numba,
                )

                logger.info(
                "⚙️ Pre-compiling Numba JIT functions to prevent runtime CPU spike..."
            )
                dummy_arr = np.array([1.0], dtype=np.float64)
                dummy_int_arr = np.array([1], dtype=np.int64)
                _ = backtest_loop_numba(
                dummy_arr,
                dummy_arr,
                dummy_arr,
                dummy_arr,
                dummy_arr,
                dummy_arr,
                dummy_int_arr,
                dummy_arr,
                dummy_arr,
                dummy_arr,
                100.0,
                1.0,
                0.0005,
                0.0005,
                0.01,
                dummy_int_arr,
                dummy_arr,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                0,
                0,
                False,
                1000.0,
            )
                logger.info("✅ Numba JIT warm-up complete.")
            except ImportError:
                pass
        else:
            logger.info("??Skipping Numba warm-up (SKIP_NUMBA_WARMUP=true)")

        self.health_manager.update_heartbeat(status="initialized")
        logger.info("🚀 Initialization Complete. Bot is Running in 2D Mode...")

    def _execute_entry(
        self,
        symbol: str,
        side: str,
        signal_price: float,
        current_price: float,
        params: dict,
        atr: float,
        hurst_value: float,
        natr_value: float,
        signal_candle_ts: int,
        next_attempt_count: int,
        late_bound_ms: int,
        entry_post_only_wait_seconds: float,
        entry_post_only_requote_max: int,
        entry_allow_market_fallback: bool,
        target_entry_open_ts: int,
        entry_lag_sec: float,
        execution_tf: str,
        entry_upper: float,
        entry_lower: float,
        margin_context: Optional[dict] = None,
    ) -> bool:
        now_ms = self._get_reference_now_ms()
        if now_ms > int(late_bound_ms):
            logger.warning(
                f"[{symbol}] Entry timeout exceeded (now: {now_ms} > bound: {late_bound_ms}). Aborting."
            )
            return True

        is_long = side == "LONG"
        order_side = "buy" if is_long else "sell"
        expected_side = "LONG" if is_long else "SHORT"
        close_sl_side = "sell" if is_long else "buy"
        side_emoji = "🟢" if is_long else "🔴"

        logger.info(
            f"{side_emoji} {expected_side} {symbol} | Signal: {signal_price:.4f}, ExecPx: {current_price:.4f} | Preparing 2D Margin Alloc..."
        )

        if (not np.isfinite(atr)) or atr <= 0:
            atr = 0.0

        # 내부적으로 최신 가용 마진(Free Margin)을 체크하여 사이즈 도출
        qty = self._calculate_position_size(
            symbol=symbol,
            side=expected_side,
            price=current_price,
            params=params,
            atr=atr,
            margin_context=margin_context,
        )
        if qty <= 0:
            return False

        used_margin = 0.0
        if margin_context is not None:
            leverage = self._resolve_exchange_leverage(
                params.get("LEVERAGE", 20)
            )
            fee_rate = float(
                params.get("TAKER_FEE_RATE", params.get("FEE_RATE", 0.0005))
            )
            margin_buffer_multiplier = (
                1.0 + (fee_rate * 2.0) + float(SLIPPAGE_RATE)
            )
            used_margin = (
                (qty * current_price) / leverage
            ) * margin_buffer_multiplier

        self.state_manager.update_symbol_state(
            symbol,
            {
                "last_entry_attempt_signal_candle_ts": int(signal_candle_ts),
                "entry_attempt_count_for_signal": int(next_attempt_count),
                "last_entry_attempt_at": datetime.utcnow().isoformat(),
            },
        )

        order = self._place_order_safe(
            symbol,
            order_side,
            qty,
            atr=atr,
            current_price=current_price,
            reduce_only=False,
            allow_market_fallback=entry_allow_market_fallback,
            order_deadline_ms=int(late_bound_ms),
            post_only_wait_seconds=entry_post_only_wait_seconds,
            post_only_requote_max=entry_post_only_requote_max,
            client_order_id=(
                "RT_EN_"
                + hashlib.md5(
                    f"{symbol}|{expected_side}|{int(signal_candle_ts)}".encode("utf-8")
                ).hexdigest()[:20]
            ),
        )
        confirmed_pos = self._confirm_position(
            symbol, expected_side=expected_side, retries=6, sleep_seconds=0.3
        )
        confirmed_amount = float(confirmed_pos.get("amount", 0.0) or 0.0)

        invalid_confirmation = (
            (confirmed_amount <= 0) if is_long else (confirmed_amount >= 0)
        )
        if invalid_confirmation:
            self._cancel_all_orders_safe(symbol)
            if margin_context is not None:
                margin_context["free_usdt"] = float(
                    margin_context.get("free_usdt", 0.0)
                ) + used_margin

            current_state = self.state_manager.get_symbol_state(symbol)
            current_attempts = int(
                (current_state or {}).get("entry_attempt_count_for_signal", 1)
            )
            self.state_manager.update_symbol_state(
                symbol,
                {
                    "entry_attempt_count_for_signal": max(0, current_attempts - 1),
                },
            )

            logger.error(
                f"❌ [{symbol}] {expected_side} entry not confirmed on exchange. Requested {qty}, Actual {confirmed_amount}. Attempt count refunded."
            )
            return True

        if not self._enforce_min_fill_ratio(
            symbol=symbol,
            expected_qty=qty,
            confirmed_amount=confirmed_amount,
            side=expected_side,
            params=params,
            current_price=current_price,
            atr=atr,
        ):
            if margin_context is not None:
                margin_context["free_usdt"] = float(
                    margin_context.get("free_usdt", 0.0)
                ) + used_margin
            current_state = self.state_manager.get_symbol_state(symbol)
            current_attempts = int(
                (current_state or {}).get("entry_attempt_count_for_signal", 1)
            )
            self.state_manager.update_symbol_state(
                symbol,
                {
                    "entry_attempt_count_for_signal": max(0, current_attempts - 1),
                },
            )
            logger.info(
                f"🔄 [{symbol}] Entry attempt refunded due to underfill abort."
            )
            return True

        client = self._get_client_for_symbol(symbol)
        filled_qty = client.round_amount(symbol, abs(confirmed_amount))
        confirmed_entry_price = float(
            confirmed_pos.get("entryPrice", current_price) or current_price
        )

        sl_type = params.get("STOP_LOSS_TYPE", "ATR")
        if sl_type == "ATR" and atr > 0:
            sl_mult = float(
                params.get("LONG_ATR_MULT" if is_long else "SHORT_ATR_MULT", 2.0)
            )
            stop_price = (
                confirmed_entry_price - (atr * sl_mult)
                if is_long
                else confirmed_entry_price + (atr * sl_mult)
            )
        else:
            sl_pct = params.get("STOP_LOSS_PCT", 0.02)
            stop_price = (
                confirmed_entry_price * (1 - sl_pct)
                if is_long
                else confirmed_entry_price * (1 + sl_pct)
            )
        client = self._get_client_for_symbol(symbol)
        tick_size = float(client.get_price_tick_size(symbol, fallback=0.0001))
        stop_price = max(tick_size, stop_price)
        client = self._get_client_for_symbol(symbol)
        stop_price = client.round_price(symbol, stop_price)

        entry_atr = float(max(0.0, atr))
        long_scale_atr_mult = float(
            params.get("LONG_SCALE_ATR_MULT", params.get("LONG_TP_MULT", 3.0))
        )
        short_scale_atr_mult = float(
            params.get("SHORT_SCALE_ATR_MULT", params.get("SHORT_TP_MULT", 3.0))
        )

        scale_price = 0.0
        client = self._get_client_for_symbol(symbol)
        if is_long and long_scale_atr_mult > 0 and entry_atr > 0:
            scale_price = confirmed_entry_price + (entry_atr * long_scale_atr_mult)
        elif (not is_long) and short_scale_atr_mult > 0 and entry_atr > 0:
            scale_price = confirmed_entry_price - (entry_atr * short_scale_atr_mult)

        scale_qty = 0.0
        if scale_price > 0:
            scale_price = client.round_price(symbol, scale_price)
            scale_qty = client.round_amount(symbol, filled_qty * 0.5)

        use_tp_entry = bool(params.get("USE_TAKE_PROFIT", False))
        tp_atr_mult_entry = float(
            params.get(
                "TAKE_PROFIT_ATR_MULT_FUTURES", params.get("TAKE_PROFIT_ATR_MULT", 3.0)
            )
        )
        tp_price = 0.0
        if use_tp_entry and entry_atr > 0:
            raw_tp = (
                confirmed_entry_price + (entry_atr * tp_atr_mult_entry)
                if is_long
                else confirmed_entry_price - (entry_atr * tp_atr_mult_entry)
            )
            client = self._get_client_for_symbol(symbol)
            tp_price = client.round_price(symbol, raw_tp)

        sl_client_id = "RT_SL_" + uuid.uuid4().hex[:20]
        sl_result = self._place_stop_loss_safe(
            symbol, close_sl_side, filled_qty, stop_price, client_order_id=sl_client_id
        )
        sl_active = bool(sl_result)
        if sl_active:
            sl_order_id = (
                str(sl_result.get("id", "") or "")
                if isinstance(sl_result, dict)
                else ""
            )
            if sl_order_id:
                self.state_manager.update_symbol_state(
                    symbol,
                    {
                        "sl_order_id": sl_order_id,
                    },
                )

        scale_order_id: Optional[str] = None
        if scale_qty > 0:
            scale_cid = "RT_LMT_" + uuid.uuid4().hex[:20]
            try:
                scale_res = self._place_order_safe(
                    symbol=symbol,
                    side=close_sl_side,
                    qty=scale_qty,
                    current_price=scale_price,
                    order_type="LIMIT",
                    reduce_only=True,
                    allow_market_fallback=False,
                    client_order_id=scale_cid,
                )
                if isinstance(scale_res, dict):
                    scale_order_id = str(scale_res.get("id", "") or "")
            except Exception as e:
                logger.error(f"❌ [{symbol}] Scale-out LIMIT order failed: {e}")
                scale_order_id = None

        actual_scale_qty = scale_qty if scale_order_id else 0.0
        tp_qty = (
            client.round_amount(symbol, filled_qty - actual_scale_qty)
            if actual_scale_qty > 0
            else filled_qty
        )

        tp_active = False
        if use_tp_entry and tp_price > 0 and tp_qty > 0:
            tp_client_id = "RT_TP_" + uuid.uuid4().hex[:20]
            tp_result = self._place_take_profit_safe(
                symbol,
                close_sl_side,
                tp_qty,
                tp_price,
                client_order_id=tp_client_id,
            )
            tp_active = bool(tp_result)

        if not sl_active:
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

        with self._db_write_lock:
            self.trade_db.record_trade(
                symbol=symbol,
                side=expected_side,
                action="ENTRY",
                quantity=filled_qty,
                price=confirmed_entry_price,
                reason=f"2D Rank Alloc ({'UP' if is_long else 'DN'})",
                params={
                    "timeframe": execution_tf,
                    "atr": atr,
                    "sl": stop_price,
                    "sl_active": sl_active,
                    "tp": tp_price if use_tp_entry else 0.0,
                    "tp_active": tp_active,
                },
            )

        now_utc_str = datetime.utcnow().isoformat()
        update_payload = {
            "entry_time": now_utc_str,
            "entry_fill_time": now_utc_str,
            "target_candle_open_time": datetime.utcfromtimestamp(
                target_entry_open_ts / 1000.0
            ).isoformat(),
            "entry_price": confirmed_entry_price,
            "entry_atr": entry_atr,
            "side": expected_side,
            "sl_required": (not sl_active),
            "last_sl_order_time": (
                datetime.utcnow().isoformat() if sl_active else None
            ),
            "active_stop_price": float(stop_price),
            "tp_price": float(tp_price),
            "pos_atr": entry_atr,
            "highest_price": float(confirmed_entry_price),
            "lowest_price": float(confirmed_entry_price),
            "last_entry_signal_candle_ts": int(signal_candle_ts),
            "successful_entry_candle_ts": int(signal_candle_ts),
            "has_scaled_out": False,
            "initial_amount": float(confirmed_amount),
        }
        if scale_order_id:
            update_payload["scale_order_id"] = scale_order_id
            update_payload["scale_target_price"] = float(scale_price)

        self.state_manager.update_symbol_state(symbol, update_payload)
        return False

    def _process_exits(self, symbol: str) -> None:
        """Phase 1: Exit 처리 및 가용 마진 확보"""
        try:
            params = self.params_map[symbol]
            strategy = self.strategies[symbol]
            execution_tf, _ = self._resolve_timeframes(params)

            pos = self._fetch_position_safe(symbol)
            amount = float(pos.get("amount", 0.0) or 0.0)
            in_position = abs(amount) > 0

            state_snapshot = self.state_manager.get_symbol_state(symbol) or {}
            current_price = self._get_market_price_safe(symbol)
            if not in_position:
                try:
                    self._cancel_all_orders_safe(symbol)
                except Exception as e:
                    logger.warning(
                        "Failed to clear orphan orders for %s: %s", symbol, e
                    )

                if state_snapshot and (
                    state_snapshot.get("entry_time")
                    or state_snapshot.get("exit_pending")
                    or state_snapshot.get("exit_error")
                ):
                    if state_snapshot.get("entry_time") and not state_snapshot.get(
                        "exit_pending"
                    ):
                        safe_price = float(current_price) if current_price is not None else 0.0
                        self._log_silent_exchange_exit(
                            symbol=symbol,
                            state=state_snapshot,
                            current_price=safe_price,
                            params=params,
                        )

                    logger.info(
                        "🧹 [%s] Clearing stale local state (no open position).", symbol
                    )
                    self.state_manager.clear_symbol_state(
                        symbol,
                        preserve_keys=[
                            "last_entry_attempt_signal_candle_ts",
                            "entry_attempt_count_for_signal",
                            "last_entry_attempt_at",
                        ],
                    )
                return

            if current_price is None:
                return

            scale_order_id = str(state_snapshot.get("scale_order_id", "") or "")
            if scale_order_id and not bool(state_snapshot.get("has_scaled_out", False)):
                try:
                    client = self._get_client_for_symbol(symbol)
                    order_info = client.exchange.fetch_order(scale_order_id, symbol)
                    order_status = str(order_info.get("status", "") or "").lower()
                    if order_status in ("canceled", "cancelled", "expired", "rejected"):
                        logger.warning(
                            "[%s] Scale-out order %s was %s. Clearing.",
                            symbol,
                            scale_order_id,
                            order_status,
                        )
                        self.state_manager.update_symbol_state(
                            symbol,
                            {
                                "scale_order_id": None,
                                "has_scaled_out": False,
                            },
                        )
                except Exception:
                    pass

            cached = self._get_cached_indicators(symbol)
            atr = float(cached.get("atr", 0.0) or 0.0)
            trend_dir = int(cached.get("trend_direction", 0) or 0)
            sar = float(cached.get("parabolic_sar", 0.0) or 0.0)
            rsi_value = float(cached.get("rsi", 50.0) or 50.0)

            exit_ind = self._refresh_exit_indicators_if_needed(
                symbol, strategy, params, execution_tf
            )

            if exit_ind:
                atr_value = exit_ind.get("atr")
                if atr_value is not None:
                    atr = float(atr_value)

                trend_dir_value = exit_ind.get("trend_direction")
                if trend_dir_value is not None:
                    trend_dir = int(trend_dir_value)

                sar_value = exit_ind.get("parabolic_sar")
                if sar_value is not None:
                    sar = float(sar_value)

                rsi_exit_value = exit_ind.get("rsi")
                if rsi_exit_value is not None:
                    rsi_value = float(rsi_exit_value)

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
                state_snapshot = self.state_manager.get_symbol_state(symbol) or {}

            if bool(state_snapshot.get("exit_pending", False)):
                recovered = self._recover_pending_exit(
                    symbol=symbol,
                    amount=amount,
                    current_price=current_price,
                    params=params,
                    pos=pos,
                    state=state_snapshot,
                )
                if recovered:
                    return
                return

            try:
                state = self.state_manager.get_symbol_state(symbol)
                last_sl_time_str = state.get("last_sl_order_time")
                sl_required = bool(state.get("sl_required", False))
                should_check = True

                if last_sl_time_str:
                    last_sl_time = datetime.fromisoformat(last_sl_time_str)
                    elapsed = (datetime.utcnow() - last_sl_time).total_seconds()

                    if not sl_required:
                        cooldown = max(
                            300.0,
                            min(
                                1800.0,
                                float(params.get("SL_WATCHDOG_COOLDOWN_SECONDS", 600)),
                            ),
                        )
                        if elapsed < cooldown:
                            should_check = False
                    else:
                        if elapsed < 60.0:
                            should_check = False

                if should_check:
                    sl_orders = self._detect_stop_loss_orders(symbol)
                    sl_orders = self._cleanup_duplicate_sl_orders(symbol, sl_orders)
                    if not sl_orders:
                        entry_price = float(
                            pos.get("entryPrice", current_price) or current_price
                        )
                        restored = self._restore_stop_loss(
                            symbol, amount, entry_price, current_price, params, atr
                        )
                        self.state_manager.update_symbol_state(
                            symbol, {"sl_required": (not bool(restored))}
                        )
                    else:
                        update: dict = {
                            "last_sl_order_time": datetime.utcnow().isoformat(),
                        }
                        if sl_required:
                            update["sl_required"] = False
                        self.state_manager.update_symbol_state(symbol, update)
            except Exception as e:
                logger.error(f"🚨 [{symbol}] SL Watchdog evaluation failed: {e}")
                self.state_manager.update_symbol_state(
                    symbol,
                    {
                        "last_sl_order_time": datetime.utcnow().isoformat(),
                        "sl_required": True,
                    },
                )

            self._check_exit(
                symbol,
                amount,
                current_price,
                params,
                pos,
                trend_dir,
                atr,
                sar,
                rsi_value,
                execution_tf,
                exit_ind=exit_ind,
            )
        except Exception as e:
            logger.error("🚨 Error in _process_exits for %s: %s", symbol, e)
            self.health_manager.record_error(e)

    def _log_silent_exchange_exit(
        self, symbol: str, state: dict, current_price: float, params: dict
    ) -> None:
        try:
            entry_price = float(state.get("entry_price", 0.0) or 0.0)
            initial_amount = float(state.get("initial_amount", 0.0) or 0.0)
            side_str = str(state.get("side", "") or "")

            if entry_price <= 0 or initial_amount <= 0 or not side_str:
                return

            has_scaled_out = bool(state.get("has_scaled_out", False))
            actual_exit_amount = (
                initial_amount * 0.5 if has_scaled_out else initial_amount
            )

            exit_price = float(current_price)
            active_stop = float(state.get("active_stop_price", 0.0) or 0.0)
            tp_price = float(state.get("tp_price", 0.0) or 0.0)
            if active_stop > 0 and tp_price > 0:
                is_long = side_str.upper() == "LONG"
                if (is_long and current_price < entry_price) or (
                    not is_long and current_price > entry_price
                ):
                    exit_price = active_stop
                else:
                    exit_price = tp_price
            reason = "Exchange SL/TP Executed (Estimated)"

            client = self._get_client_for_symbol(symbol)
            try:
                closed_orders = client.exchange.fetch_closed_orders(symbol, limit=5)
                sorted_orders = sorted(
                    closed_orders,
                    key=lambda o: float(o.get("timestamp", 0) or 0.0),
                    reverse=True,
                )
                expected_close_side = "SELL" if side_str.upper() == "LONG" else "BUY"
                for o in sorted_orders:
                    cid = str(o.get("clientOrderId", "") or "")
                    if not cid.startswith("RT_"):
                        continue
                    order_side = str(o.get("side", "") or "").upper()
                    filled_qty = float(o.get("filled", 0.0) or 0.0)
                    if order_side != expected_close_side or filled_qty <= 0.0:
                        continue
                    avg_price = o.get("average")
                    raw_price = o.get("price")
                    if avg_price is not None:
                        exit_price = float(avg_price)
                    elif raw_price is not None:
                        exit_price = float(raw_price)
                    o_type = str(o.get("type", "") or "").upper()
                    if "STOP" in o_type:
                        reason = f"Exchange SL Hit ({exit_price:.2f})"
                    elif "TAKE_PROFIT" in o_type:
                        reason = f"Exchange TP Hit ({exit_price:.2f})"
                    break
            except Exception:
                pass

            fee_rate = float(
                params.get("TAKER_FEE_RATE", params.get("FEE_RATE", 0.0005))
            )
            entry_value = entry_price * actual_exit_amount
            exit_value = exit_price * actual_exit_amount
            total_fee = (entry_value + exit_value) * fee_rate

            if side_str.upper() == "LONG":
                gross_pnl = (exit_price - entry_price) * actual_exit_amount
            else:
                gross_pnl = (entry_price - exit_price) * actual_exit_amount

            pnl = gross_pnl - total_fee
            pnl_pct = (pnl / entry_value) * 100.0 if entry_value > 0 else 0.0
            funding_estimate = 0.0
            entry_time_str = str(state.get("entry_time", "") or "")
            if entry_time_str:
                try:
                    entry_dt = datetime.fromisoformat(entry_time_str)
                    hold_hours = max(
                        0.0, (datetime.utcnow() - entry_dt).total_seconds() / 3600
                    )
                    funding_sessions = int(hold_hours / 8)
                    avg_funding_rate = 0.0001
                    funding_estimate = (
                        abs(actual_exit_amount * entry_price)
                        * avg_funding_rate
                        * funding_sessions
                    )
                except Exception:
                    pass
            pnl_with_funding = pnl - funding_estimate
            pnl_with_funding_pct = (
                (pnl_with_funding / entry_value) * 100.0 if entry_value > 0 else 0.0
            )

            with self._db_write_lock:
                self.trade_db.record_trade(
                    symbol=symbol,
                    side=side_str,
                    action="EXIT",
                    quantity=actual_exit_amount,
                    price=exit_price,
                    entry_price=entry_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    reason=reason,
                )
        except Exception as e:
            logger.error("Failed to log silent exit for %s: %s", symbol, e)

    def _scan_entries(self, symbol: str) -> Optional[dict]:
        """Phase 2: 진입 시그널 동시 스캔 (No Execution)"""
        df: Optional[pd.DataFrame] = None
        signal_df: Optional[pd.DataFrame] = None
        try:
            params = self.params_map[symbol]
            strategy = self.strategies[symbol]
            execution_tf, indicator_tf = self._resolve_timeframes(params)

            pos = self._fetch_position_safe(symbol)
            amount = float(pos.get("amount", 0.0) or 0.0)
            in_position = abs(amount) > 0
            if in_position:
                return None

            state = self.state_manager.get_symbol_state(symbol)

            current_slot = self._get_candle_slot_id(indicator_tf)
            with self._cache_lock:
                already_calculated = self.last_calc_candle.get(symbol) == current_slot
            cached = self._get_cached_indicators(symbol)
            required_indicator_keys = (
                "trend_direction",
                "atr",
                "entry_upper",
                "entry_lower",
                "strength_filter",
                "vol_zscore",
            )

            need_calculation = False
            if (
                not cached
                or any(key not in cached for key in required_indicator_keys)
                or not already_calculated
            ):
                need_calculation = True

            now_ref_ms = self._get_reference_now_ms()
            if cached:
                cached_ts = cached.get("candle_ts", 0)
                if cached_ts > now_ref_ms:
                    logger.warning(
                        "[%s] Look-ahead detected in indicator cache. Forcing recalculation.",
                        symbol,
                    )
                    need_calculation = True

            if need_calculation:
                tf_min = self._timeframe_to_minutes(indicator_tf)
                if tf_min <= 0:
                    return None

                macro_period_raw = params.get("MACRO_EMA_PERIOD", 200)
                hurst_period_raw = params.get("HURST_PERIOD", 200)
                try:
                    macro_period = int(macro_period_raw)
                except (TypeError, ValueError):
                    macro_period = 200
                try:
                    hurst_period = int(hurst_period_raw)
                except (TypeError, ValueError):
                    hurst_period = 200

                max_period = max(macro_period, hurst_period)
                limit = max(700, max_period * 3 + 50)
                lookback_days = (limit * tf_min) / 1440
                start_dt = datetime.utcnow() - timedelta(days=lookback_days + 2)
                start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

                cvd_required = "TAKER_RATIO_THRESHOLD" in params

                if cvd_required:
                    df = self._fetch_ohlcv_with_taker_safe(
                        symbol,
                        indicator_tf,
                        start_str,
                    )
                else:
                    df = self._fetch_ohlcv_safe(symbol, indicator_tf, start_str)

                if df is None or len(df) < 200:
                    with self._cache_lock:
                        self.last_calc_candle[symbol] = current_slot
                    return None

                if cvd_required and "taker_buy_base_volume" not in df.columns:
                    logger.error(
                        "🚨 [%s] Critical Data Missing: 'taker_buy_base_volume' not found. Aborting entry scan.",
                        symbol,
                    )
                    return None

                # [FIX 4] Datetime 파이프라인 동기화 (Seasonality Taker 보호)
                if "datetime" not in df.columns and "timestamp" in df.columns:
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

                float_cols = ["open", "high", "low", "close", "volume"]
                df[float_cols] = df[float_cols].astype(np.float64)
                df = strategy.generate_signals(df)
                last_candle = self._select_last_closed_candle(df, indicator_tf)
                del df
                df = None
                if last_candle is None:
                    return None

                candle_ts = self._extract_candle_timestamp_ms(last_candle)
                interval_ms = tf_min * 60 * 1000
                candle_close_boundary_ms = candle_ts + interval_ms
                elapsed_since_close_ms = now_ref_ms - candle_close_boundary_ms
                if 0 < elapsed_since_close_ms < 2000:
                    logger.debug(
                        "[%s] Candle just closed %dms ago. Waiting for settlement.",
                        symbol,
                        elapsed_since_close_ms,
                    )
                    return None

                entry_upper = last_candle.get("entry_upper")
                entry_lower = last_candle.get("entry_lower")
                trend_dir = last_candle.get("trend_direction", 0)
                strength_ok = last_candle.get("strength_filter", 1) == 1
                atr = last_candle.get("atr", 0.0)
                sar = last_candle.get("parabolic_sar", 0.0)

                # [FIX 1] Key 매핑 동기화
                vol_ratio = last_candle.get(
                    "vol_zscore", last_candle.get("volume_ratio", -10.0)
                )

                rsi_value = last_candle.get("rsi", 50.0)
                hurst_value = last_candle.get("hurst", 0.5)
                natr_value = last_candle.get("natr", 1.0)

                atr_safe = float(0.0 if pd.isna(atr) else atr)
                self._cache_indicators(
                    symbol,
                    {
                        "trend_direction": int(0 if pd.isna(trend_dir) else trend_dir),
                        "atr": float(atr_safe),
                        "parabolic_sar": float(0.0 if pd.isna(sar) else sar),
                        "entry_upper": (
                            float(entry_upper) if pd.notna(entry_upper) else None
                        ),
                        "entry_lower": (
                            float(entry_lower) if pd.notna(entry_lower) else None
                        ),
                        "strength_filter": int(strength_ok),
                        "vol_zscore": float(-10.0 if pd.isna(vol_ratio) else vol_ratio),
                        "rsi": float(50.0 if pd.isna(rsi_value) else rsi_value),
                        "hurst": float(0.5 if pd.isna(hurst_value) else hurst_value),
                        "natr": float(1.0 if pd.isna(natr_value) else natr_value),
                        "indicator_timeframe": indicator_tf,
                        "cached_at": datetime.utcnow().isoformat(),
                    },
                )
                with self._cache_lock:
                    self.last_calc_candle[symbol] = current_slot
            else:
                trend_dir = cached.get("trend_direction", 0)
                atr = cached.get("atr", 0.0)
                sar = cached.get("parabolic_sar", 0.0)
                entry_upper = cached.get("entry_upper")
                entry_lower = cached.get("entry_lower")
                strength_ok = int(cached.get("strength_filter", 0)) == 1
                vol_ratio = cached.get("vol_zscore", -10.0)
                rsi_value = cached.get("rsi", 50.0)
                hurst_value = cached.get("hurst", 0.5)
                natr_value = cached.get("natr", 1.0)

            current_price = self._get_market_price_safe(symbol)
            if current_price is None:
                return None

            if (not np.isfinite(atr)) or atr <= 0:
                atr = 0.0

            if (
                pd.isna(entry_upper)
                or pd.isna(entry_lower)
                or entry_upper is None
                or entry_lower is None
            ):
                return None

            signal_df = self._fetch_recent_ohlcv_safe(symbol, execution_tf, limit=4)
            if signal_df is None or len(signal_df) < 2:
                return None

            signal_candle = self._select_last_closed_candle(signal_df, execution_tf)
            if signal_candle is None:
                return None

            signal_price = float(signal_candle.get("close", np.nan))
            if not np.isfinite(signal_price):
                return None

            signal_candle_ts = self._extract_candle_timestamp_ms(signal_candle)
            tf_min = self._timeframe_to_minutes(execution_tf)
            if signal_candle_ts <= 0 or tf_min <= 0:
                return None

            interval_ms = tf_min * 60 * 1000
            target_entry_open_ts = signal_candle_ts + interval_ms
            now_ref_ms = self._get_reference_now_ms()

            # API 지연에 의한 과거 캔들(Stale Candle) 필터링
            if now_ref_ms >= (target_entry_open_ts + interval_ms):
                logger.debug(f"[{symbol}] API delay detected (stale candle). Waiting for update.")
                return None
            entry_grace_sec = float(
                params.get(
                    "ENTRY_EXECUTION_GRACE_SECONDS",
                    self._default_entry_grace_seconds(
                        execution_tf=execution_tf, params=params
                    ),
                )
            )
            early_bound_ms = target_entry_open_ts - int(
                float(params.get("ENTRY_EXECUTION_EARLY_TOLERANCE_SECONDS", 1.0)) * 1000
            )
            late_bound_ms = target_entry_open_ts + int(entry_grace_sec * 1000)

            use_intrabar_entry = bool(params.get("USE_INTRABAR_ENTRY", False))
            if not use_intrabar_entry:
                if now_ref_ms < early_bound_ms or now_ref_ms > late_bound_ms:
                    return None
            else:
                # Intra-bar 돌파 진입: 미래 캔들 조기 진입만 방지
                if now_ref_ms < early_bound_ms:
                    return None

            entry_retry_max = max(1, int(params.get("ENTRY_SIGNAL_RETRY_MAX", 2)))
            successful_entry_ts = int(
                state.get("successful_entry_candle_ts", 0) or 0
            )
            if successful_entry_ts > 0 and successful_entry_ts == signal_candle_ts:
                return None
            last_attempt_signal_ts = int(
                state.get("last_entry_attempt_signal_candle_ts", 0) or 0
            )
            attempt_count_for_signal = int(
                state.get("entry_attempt_count_for_signal", 0) or 0
            )
            if last_attempt_signal_ts != signal_candle_ts:
                attempt_count_for_signal = 0
            if attempt_count_for_signal >= entry_retry_max:
                return None
            next_attempt_count = attempt_count_for_signal + 1

            is_uptrend = trend_dir == 1
            is_downtrend = trend_dir == -1

            if use_intrabar_entry:
                long_signal = is_uptrend and strength_ok
                short_signal = is_downtrend and strength_ok
            else:
                long_signal = is_uptrend and strength_ok and (
                    signal_price > float(entry_upper)
                )
                short_signal = is_downtrend and strength_ok and (
                    signal_price < float(entry_lower)
                )
            entry_lag_sec = max(
                0.0, float((now_ref_ms - target_entry_open_ts) / 1000.0)
            )
            entry_allow_market_fallback = self._resolve_entry_market_fallback(
                params=params,
                atr=atr,
                current_price=current_price,
                entry_lag_sec=entry_lag_sec,
            )

            side: Optional[str] = None
            if long_signal:
                side = "LONG"
            elif short_signal:
                side = "SHORT"

            if side is None:
                return None

            candidate: dict = {
                "symbol": symbol,
                "side": side,
                "vol_ratio": float(vol_ratio),
                "params": params,
                "atr": float(atr),
                "hurst_value": float(hurst_value),
                "natr_value": float(natr_value),
                "signal_price": float(signal_price),
                "current_price": float(current_price),
                "signal_candle_ts": int(signal_candle_ts),
                "next_attempt_count": int(next_attempt_count),
                "late_bound_ms": int(late_bound_ms),
                "entry_post_only_wait_seconds": float(
                    params.get("ENTRY_POST_ONLY_WAIT_SECONDS", 1.2)
                ),
                "entry_post_only_requote_max": int(
                    params.get("ENTRY_POST_ONLY_REQUOTE_MAX", 2)
                ),
                "entry_allow_market_fallback": bool(entry_allow_market_fallback),
                "target_entry_open_ts": int(target_entry_open_ts),
                "entry_lag_sec": float(entry_lag_sec),
                "execution_tf": execution_tf,
                "entry_upper": float(entry_upper),
                "entry_lower": float(entry_lower),
            }
            return candidate
        except Exception as e:
            logger.error("🚨 Error in _scan_entries for %s: %s", symbol, e)
            self.health_manager.record_error(e)
            return None
        finally:
            if df is not None:
                del df
            if signal_df is not None:
                del signal_df

    def _execute_entry_order(self, cand: dict) -> None:
        """Phase 3: 랭킹 기반 실진입 주문"""
        symbol = str(cand.get("symbol"))
        should_abort = self._execute_entry(
            symbol=symbol,
            side=str(cand.get("side")),
            signal_price=float(cand.get("signal_price")),
            current_price=float(cand.get("current_price")),
            params=cand.get("params", {}),
            atr=float(cand.get("atr")),
            hurst_value=float(cand.get("hurst_value")),
            natr_value=float(cand.get("natr_value")),
            signal_candle_ts=int(cand.get("signal_candle_ts")),
            next_attempt_count=int(cand.get("next_attempt_count")),
            late_bound_ms=int(cand.get("late_bound_ms")),
            entry_post_only_wait_seconds=float(
                cand.get("entry_post_only_wait_seconds")
            ),
            entry_post_only_requote_max=int(cand.get("entry_post_only_requote_max")),
            entry_allow_market_fallback=bool(cand.get("entry_allow_market_fallback")),
            target_entry_open_ts=int(cand.get("target_entry_open_ts")),
            entry_lag_sec=float(cand.get("entry_lag_sec")),
            execution_tf=str(cand.get("execution_tf")),
            entry_upper=float(cand.get("entry_upper")),
            entry_lower=float(cand.get("entry_lower")),
            margin_context=cand.get("margin_context"),
        )
        if should_abort:
            logger.warning("Entry execution aborted for %s", symbol)

    def _calculate_position_size(
        self,
        symbol: str,
        side: str,
        price: float,
        params: dict,
        atr: float = 0.0,
        hurst: float = 0.5,
        natr: float = 0.0,
        margin_context: Optional[dict] = None,
    ) -> float:
        """2D 마진 공유: 고정 할당량 폐지, 총 자본 및 실시간 가용 마진 기준 동적 계산"""
        if price <= 0:
            return 0.0

        if margin_context is not None:
            usdt_free = float(margin_context.get("free_usdt", 0.0))
            total_balance = float(
                margin_context.get("total_equity_snapshot", usdt_free)
            )
        else:
            try:
                total_balance, usdt_free = self._fetch_balance_safe(symbol=symbol)
            except Exception:
                return 0.0

        if usdt_free < MIN_BALANCE_FOR_TRADE:
            return 0.0

        regime_mult = 1.0

        use_compounding = bool(params.get("USE_COMPOUNDING", True))
        max_capital_usage = float(params.get("MAX_CAPITAL_USAGE", 1_000_000.0))
        trading_equity = (
            min(total_balance, max_capital_usage)
            if use_compounding
            else max_capital_usage
        )

        leverage = self._resolve_exchange_leverage(params.get("LEVERAGE", 20))
        risk_per_trade = params.get("RISK_PER_TRADE", 0.02)
        risk_amt = (trading_equity * risk_per_trade) * regime_mult

        stop_distance_pct = 0.05
        sl_type = str(params.get("STOP_LOSS_TYPE", "ATR")).upper()
        if sl_type == "ATR" and atr > 0 and price > 0:
            atr_mult = float(
                params.get("LONG_ATR_MULT" if side == "LONG" else "SHORT_ATR_MULT", 2.0)
            )
            stop_distance_pct = (atr * atr_mult) / price
        else:
            stop_distance_pct = float(params.get("STOP_LOSS_PCT", 0.02))
        stop_distance_pct = max(0.005, min(stop_distance_pct, 0.10))

        target_notional = risk_amt / stop_distance_pct

        # [FIX 2] 수수료 및 슬리피지 방어 버퍼 5% 적용
        max_tradeable_notional = (usdt_free * 0.95) * leverage

        final_notional = min(target_notional, max_tradeable_notional)

        client = self._get_client_for_symbol(symbol)
        constraints = client.get_symbol_constraints(symbol)
        effective_min_order_value = max(
            float(MIN_ORDER_VALUE_USDT), float(constraints.get("min_cost") or 0.0)
        )
        if final_notional < effective_min_order_value:
            logger.warning(
                f"[{symbol}] Free Margin too low (${usdt_free:.2f}). Need ${effective_min_order_value/leverage:.2f} to trade."
            )
            return 0.0

        if margin_context is not None:
            fee_rate = float(
                params.get("TAKER_FEE_RATE", params.get("FEE_RATE", 0.0005))
            )
            max_expected_funding_rate = 0.001
            margin_buffer_multiplier = (
                1.0
                + (fee_rate * 2.0)
                + float(SLIPPAGE_RATE)
                + max_expected_funding_rate
            )
            used_margin = (final_notional / leverage) * margin_buffer_multiplier

            margin_context["free_usdt"] = max(
                0.0,
                float(margin_context.get("free_usdt", 0.0)) - used_margin,
            )

        client = self._get_client_for_symbol(symbol)
        quantity = client.round_amount(symbol, final_notional / price)
        return quantity if quantity > 0 else 0.0

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
        try:
            state = self.state_manager.get_symbol_state(symbol)
            side_str = "LONG" if amount > 0 else "SHORT"
            order_side = "sell" if amount > 0 else "buy"
            entry_price = float(
                pos.get("entryPrice", 0.0) or state.get("entry_price", 0.0) or 0.0
            )
            if entry_price <= 0:
                return

            # 기본값으로 캐싱된 마지막 완성봉 데이터 호출 (백테스팅 일치성)
            cached_exit_ind = self._get_cached_exit_indicators(symbol)
            candle_open = float(cached_exit_ind.get("candle_open", current_price))
            candle_high = float(cached_exit_ind.get("candle_high", current_price))
            candle_low = float(cached_exit_ind.get("candle_low", current_price))
            candle_close = float(cached_exit_ind.get("candle_close", current_price))
            candle_ts = int(cached_exit_ind.get("candle_ts", 0))

            if exit_ind:
                candle_ts = int(exit_ind.get("candle_ts", candle_ts))
                high_val = exit_ind.get("candle_high")
                if high_val is not None and np.isfinite(high_val):
                    candle_high = float(high_val)
                low_val = exit_ind.get("candle_low")
                if low_val is not None and np.isfinite(low_val):
                    candle_low = float(low_val)

            last_processed_candle_ts = int(
                state.get("last_processed_candle_ts", 0) or 0
            )
            exit_pending = bool(state.get("exit_pending", False))
            fallback_without_candle = False

            if candle_ts <= 0:
                fallback_max_delay_sec = float(
                    params.get(
                        "EXIT_FALLBACK_MAX_DELAY_SEC",
                        max(20.0, float(LOOP_INTERVAL_SECONDS) * 2.0),
                    )
                )
                now_ref_ms = int(self._get_reference_now_ms())
                last_fallback_eval_ms = int(
                    state.get("last_exit_fallback_eval_ms", 0) or 0
                )
                last_ref_ms = (
                    last_processed_candle_ts
                    if last_processed_candle_ts > 0
                    else last_fallback_eval_ms
                )
                elapsed_ms = (now_ref_ms - last_ref_ms) if last_ref_ms > 0 else (10**9)
                if elapsed_ms < int(max(1.0, fallback_max_delay_sec) * 1000):
                    return
                fallback_without_candle = True
                self.state_manager.update_symbol_state(
                    symbol, {"last_exit_fallback_eval_ms": int(now_ref_ms)}
                )

            if (
                candle_ts > 0
                and candle_ts == last_processed_candle_ts
                and not exit_pending
            ):
                return

            exit_type = str(params.get("EXIT_TYPE", "ATR") or "ATR").upper()
            use_sar_exit = exit_type == "PARABOLIC_SAR"
            use_tp = bool(params.get("USE_TAKE_PROFIT", False))
            tp_atr_mult = float(
                params.get(
                    "TAKE_PROFIT_ATR_MULT_FUTURES",
                    params.get("TAKE_PROFIT_ATR_MULT", 3.0),
                )
            )

            pos_atr = float(state.get("pos_atr", state.get("entry_atr", atr)) or 0.0)
            if pos_atr <= 0 and np.isfinite(atr) and atr > 0:
                pos_atr = float(atr)

            highest_price = float(
                state.get("highest_price", entry_price) or entry_price
            )
            lowest_price = float(state.get("lowest_price", entry_price) or entry_price)
            has_scaled_out = bool(state.get("has_scaled_out", False))

            sl_type = str(params.get("STOP_LOSS_TYPE", "ATR") or "ATR").upper()
            active_stop = float(state.get("active_stop_price", 0.0) or 0.0)
            client = self._get_client_for_symbol(symbol)
            if active_stop <= 0:
                if sl_type == "ATR" and pos_atr > 0:
                    sl_mult = float(
                        params.get(
                            "LONG_ATR_MULT" if amount > 0 else "SHORT_ATR_MULT", 2.0
                        )
                    )
                    active_stop = (
                        entry_price - (pos_atr * sl_mult)
                        if amount > 0
                        else entry_price + (pos_atr * sl_mult)
                    )
                else:
                    sl_pct = float(params.get("STOP_LOSS_PCT", 0.02))
                    active_stop = (
                        entry_price * (1 - sl_pct)
                        if amount > 0
                        else entry_price * (1 + sl_pct)
                    )

                tick_size = client.get_price_tick_size(
                    symbol, fallback=(0.1 if "BTC" in symbol else 0.01)
                )
                if amount > 0 and active_stop >= entry_price:
                    active_stop = entry_price - (tick_size * 2)
                elif amount < 0 and active_stop <= entry_price:
                    active_stop = entry_price + (tick_size * 2)

            active_stop = client.round_price(symbol, active_stop)

            tp_price_val = float(state.get("tp_price", 0.0) or 0.0)
            if use_tp and (tp_price_val <= 0 and pos_atr > 0):
                tp_price_val = (
                    entry_price + (pos_atr * tp_atr_mult)
                    if amount > 0
                    else entry_price - (pos_atr * tp_atr_mult)
                )
                tp_price_val = client.round_price(symbol, tp_price_val)

            current_stop = float(active_stop)
            if use_sar_exit and np.isfinite(sar) and sar > 0:
                if amount > 0 and sar < candle_open:
                    current_stop = max(current_stop, float(sar))
                elif amount < 0 and sar > candle_open:
                    current_stop = min(current_stop, float(sar))
                current_stop = client.round_price(symbol, current_stop)

            exit_triggered, reason, exit_price_for_calc = (
                False,
                "",
                float(current_price),
            )
            slippage = float(SLIPPAGE_RATE)

            scale_out_done = False
            scale_qty = 0.0
            initial_amount = float(state.get("initial_amount", amount) or amount)

            if (not has_scaled_out) and initial_amount != 0.0:
                if abs(amount) <= abs(initial_amount) * 0.5:
                    scale_out_done = True
                    has_scaled_out = True
                    scale_exit_price = float(
                        state.get("scale_target_price", current_price)
                    )
                    scale_qty_raw = max(
                        0.0, abs(initial_amount) - abs(amount)
                    )
                    scale_qty = client.round_amount(symbol, scale_qty_raw)
                    if scale_qty > 0:
                        scale_qty_float = float(scale_qty)
                        fee_rate = float(
                            params.get("MAKER_FEE_RATE", params.get("FEE_RATE", 0.0002))
                        )
                        entry_value = entry_price * scale_qty_float
                        exit_value = scale_exit_price * scale_qty_float
                        total_fee = (entry_value + exit_value) * fee_rate

                        if amount > 0:
                            gross_pnl = (
                                scale_exit_price - entry_price
                            ) * scale_qty_float
                        else:
                            gross_pnl = (
                                entry_price - scale_exit_price
                            ) * scale_qty_float

                        scale_pnl = gross_pnl - total_fee
                        scale_pnl_pct = (
                            (scale_pnl / entry_value) * 100.0 if entry_value > 0 else 0.0
                        )
                        with self._db_write_lock:
                            self.trade_db.record_trade(
                                symbol=symbol,
                                side=side_str,
                                action="EXIT_PARTIAL",
                                quantity=scale_qty,
                                price=scale_exit_price,
                                entry_price=entry_price,
                                pnl=scale_pnl,
                                pnl_pct=scale_pnl_pct,
                                reason=(
                                    "Scale-Out 50% (LONG)"
                                    if amount > 0
                                    else "Scale-Out 50% (SHORT)"
                                ),
                            )
                    self.state_manager.update_symbol_state(
                        symbol,
                        {
                            "has_scaled_out": True,
                            "highest_price": float(highest_price),
                            "lowest_price": float(lowest_price),
                            "pos_atr": float(pos_atr),
                        },
                    )
                    exit_price_for_calc = float(scale_exit_price)

            if scale_out_done:
                scale_order_id_state = self.state_manager.get_symbol_state(symbol)
                if scale_order_id_state:
                    scale_order_id = str(
                        scale_order_id_state.get("scale_order_id", "") or ""
                    )
                    if scale_order_id:
                        try:
                            client = self._get_client_for_symbol(symbol)
                            client.exchange.cancel_order(scale_order_id, symbol)
                        except Exception:
                            pass

                fee_rate = float(
                    params.get("TAKER_FEE_RATE", params.get("FEE_RATE", 0.0005))
                )
                breakeven_raw = (
                    entry_price
                    * (1.0 + fee_rate * 2.0)
                    if amount > 0
                    else entry_price
                    * (1.0 - fee_rate * 2.0)
                )
                client = self._get_client_for_symbol(symbol)
                breakeven_stop = client.round_price(symbol, breakeven_raw)

                self.state_manager.update_symbol_state(
                    symbol,
                    {"has_scaled_out": True},
                )

                try:
                    self._cancel_stop_orders_only(symbol)
                    remaining_qty_raw = abs(amount)
                    client = self._get_client_for_symbol(symbol)
                    remaining_qty = client.round_amount(symbol, remaining_qty_raw)

                    if remaining_qty > 0:
                        stop_side = "sell" if amount > 0 else "buy"
                        be_client_id = "RT_BE_" + uuid.uuid4().hex[:20]
                        sl_result = self._place_stop_loss_safe(
                            symbol,
                            stop_side,
                            remaining_qty,
                            breakeven_stop,
                            client_order_id=be_client_id,
                        )

                        if sl_result and isinstance(sl_result, dict):
                            sl_order_id = str(sl_result.get("id", "") or "")
                            be_update: dict = {
                                "active_stop_price": float(breakeven_stop),
                                "last_sl_order_time": datetime.utcnow().isoformat(),
                                "sl_order_id": sl_order_id if sl_order_id else None,
                            }
                            if candle_ts > 0:
                                be_update["last_processed_candle_ts"] = int(candle_ts)
                            self.state_manager.update_symbol_state(symbol, be_update)
                except Exception as e:
                    logger.error(
                        "Error while updating breakeven stop after scale-out for %s: %s",
                        symbol,
                        e,
                    )
                return

            if not fallback_without_candle:
                if amount > 0:
                    long_min_price = min(current_price, candle_low)
                    if long_min_price <= current_stop:
                        exit_triggered, reason, exit_price_for_calc = (
                            True,
                            f"Stop Loss ({current_stop:.2f})",
                            current_stop * (1 - slippage),
                        )
                else:
                    short_max_price = max(current_price, candle_high)
                    if short_max_price >= current_stop:
                        exit_triggered, reason, exit_price_for_calc = (
                            True,
                            f"Stop Loss ({current_stop:.2f})",
                            current_stop * (1 + slippage),
                        )

                if not exit_triggered and use_tp and tp_price_val > 0:
                    if amount > 0 and max(current_price, candle_high) >= tp_price_val:
                        exit_triggered, reason, exit_price_for_calc = (
                            True,
                            f"Take Profit ({tp_price_val:.2f})",
                            tp_price_val,
                        )
                    elif amount < 0 and min(current_price, candle_low) <= tp_price_val:
                        exit_triggered, reason, exit_price_for_calc = (
                            True,
                            f"Take Profit ({tp_price_val:.2f})",
                            tp_price_val,
                        )

            trailing_updated = False
            prev_stop = float(active_stop)
            if not exit_triggered and not fallback_without_candle:
                min_update_step = 0.0
                if pos_atr > 0:
                    atr_step_ratio = float(
                        params.get("TRAIL_UPDATE_ATR_RATIO", 0.2)
                    )
                    min_update_step = max(0.0, pos_atr * atr_step_ratio)

                if amount > 0:
                    atr_mult = float(params.get("LONG_TRAIL_MULT", 3.0))
                    if candle_high > highest_price:
                        highest_price = candle_high
                    if exit_type != "PARABOLIC_SAR" and pos_atr > 0:
                        new_stop = highest_price - (pos_atr * atr_mult)
                        if min_update_step > 0.0:
                            if new_stop > (active_stop + min_update_step):
                                active_stop = new_stop
                                trailing_updated = True
                        elif new_stop > active_stop:
                            active_stop = new_stop
                            trailing_updated = True
                else:
                    short_trail_mult = float(params.get("SHORT_TRAIL_MULT", 3.0))
                    if candle_low < lowest_price:
                        lowest_price = candle_low
                    if exit_type != "PARABOLIC_SAR" and pos_atr > 0:
                        new_stop = lowest_price + (pos_atr * short_trail_mult)
                        if active_stop <= 0:
                            active_stop = new_stop
                            trailing_updated = True
                        elif min_update_step > 0.0:
                            if new_stop < (active_stop - min_update_step):
                                active_stop = new_stop
                                trailing_updated = True
                        elif new_stop < active_stop:
                            active_stop = new_stop
                            trailing_updated = True

                client = self._get_client_for_symbol(symbol)
                active_stop = client.round_price(symbol, active_stop)
                base_update = {
                    "pos_atr": float(pos_atr),
                    "tp_price": float(tp_price_val),
                    "highest_price": float(highest_price),
                    "lowest_price": float(lowest_price),
                }
                if not trailing_updated:
                    base_update["active_stop_price"] = float(active_stop)
                if candle_ts > 0:
                    base_update["last_processed_candle_ts"] = int(candle_ts)
                self.state_manager.update_symbol_state(symbol, base_update)

                if trailing_updated:
                    actual_amount = float(amount)

                    if abs(actual_amount) > 0:
                        client = self._get_client_for_symbol(symbol)
                        stop_qty = client.round_amount(symbol, abs(actual_amount)) or abs(
                            actual_amount
                        )
                        stop_side = "sell" if actual_amount > 0 else "buy"
                        new_stop_price = client.round_price(symbol, active_stop)
                        self._cancel_stop_orders_only(symbol)
                        trail_client_id = "RT_TR_" + uuid.uuid4().hex[:20]
                        sl_result = self._place_stop_loss_safe(
                            symbol,
                            stop_side,
                            stop_qty,
                            new_stop_price,
                            client_order_id=trail_client_id,
                        )
                        if sl_result:
                            sl_order_id = str(sl_result.get("id", "") or "")
                            update_payload = {
                                "active_stop_price": float(new_stop_price),
                                "sl_required": False,
                                "last_sl_order_time": datetime.utcnow().isoformat(),
                            }
                            if sl_order_id:
                                update_payload["sl_order_id"] = sl_order_id
                            self.state_manager.update_symbol_state(symbol, update_payload)
                        else:
                            if prev_stop > 0:
                                client = self._get_client_for_symbol(symbol)
                                prev_price_rounded = client.round_price(symbol, prev_stop)
                                prev_sl_client_id = "RT_TR_FB_" + uuid.uuid4().hex[:20]
                                sl_result_prev = self._place_stop_loss_safe(
                                    symbol,
                                    stop_side,
                                    stop_qty,
                                    prev_price_rounded,
                                    client_order_id=prev_sl_client_id,
                                )
                                if sl_result_prev and isinstance(sl_result_prev, dict):
                                    sl_order_id_prev = str(
                                        sl_result_prev.get("id", "") or ""
                                    )
                                    update_payload_prev = {
                                        "active_stop_price": float(prev_price_rounded),
                                        "sl_required": False,
                                        "last_sl_order_time": datetime.utcnow().isoformat(),
                                    }
                                    if sl_order_id_prev:
                                        update_payload_prev["sl_order_id"] = (
                                            sl_order_id_prev
                                        )
                                    self.state_manager.update_symbol_state(
                                        symbol, update_payload_prev
                                    )
                            else:
                                self.state_manager.update_symbol_state(
                                    symbol,
                                    {
                                        "sl_required": True,
                                        "last_sl_order_time": datetime.utcnow().isoformat(),
                                    },
                                )
                return
            elif not exit_triggered and fallback_without_candle:
                return

            if exit_price_for_calc <= 0:
                exit_price_for_calc = float(current_price)

            pending_update = {
                "exit_pending": True,
                "exit_reason": reason,
                "exit_attempt_at": datetime.utcnow().isoformat(),
            }
            if candle_ts > 0:
                pending_update["last_processed_candle_ts"] = int(candle_ts)
            self.state_manager.update_symbol_state(symbol, pending_update)

            try:
                self._cancel_all_orders_safe(symbol)
            except Exception:
                pass

            time.sleep(0.1)

            final_pos = self._fetch_position_safe(symbol)
            actual_amount = float(final_pos.get("amount", 0.0) or 0.0)
            closed_qty = abs(actual_amount)

            if closed_qty > 0:
                if not self._place_order_safe(
                    symbol, order_side, closed_qty, reduce_only=True
                ):
                    self.state_manager.update_symbol_state(
                        symbol,
                        {"exit_error": "order_submit_failed_or_unknown"},
                    )

            is_flat, remaining_amount = self._wait_until_position_flat(
                symbol, timeout_seconds=6.0, poll_seconds=0.35
            )
            if not is_flat:
                self.state_manager.update_symbol_state(
                    symbol,
                    {
                        "exit_pending": True,
                        "exit_error": f"not_flat_after_exit_attempt:{remaining_amount:+.8f}",
                        "exit_remaining_amount": float(remaining_amount),
                        "exit_attempt_at": datetime.utcnow().isoformat(),
                    },
                )
                return

            try:
                self._cancel_all_orders_safe(symbol)
            except Exception:
                pass

            fee_rate = float(
                params.get("TAKER_FEE_RATE", params.get("FEE_RATE", 0.0005))
            )
            entry_value = entry_price * closed_qty
            exit_value = exit_price_for_calc * closed_qty
            total_fee = (entry_value + exit_value) * fee_rate

            if amount > 0:
                gross_pnl = (exit_price_for_calc - entry_price) * closed_qty
            else:
                gross_pnl = (entry_price - exit_price_for_calc) * closed_qty

            pnl = gross_pnl - total_fee
            # Note: 이 PnL은 Trading Fee만 차감되었으며, 누적 Funding Fee는 포함되지 않은 Gross PnL임.
            # 실제 지갑 잔고와의 오차는 Phase 3의 _fetch_balance_safe() 호출 시 자동 교정됨.
            pnl_pct = (pnl / entry_value) * 100.0 if entry_value > 0 else 0.0

            funding_estimate = 0.0
            entry_time_str = str(state.get("entry_time", "") or "")
            if entry_time_str:
                try:
                    entry_dt = datetime.fromisoformat(entry_time_str)
                    hold_hours = max(0.0, (datetime.utcnow() - entry_dt).total_seconds() / 3600.0)
                    funding_sessions = int(hold_hours / 8)
                    avg_funding_rate = 0.0001
                    funding_estimate = abs(closed_qty * entry_price) * avg_funding_rate * funding_sessions
                except Exception:
                    pass
            pnl_with_funding = pnl - funding_estimate
            pnl_with_funding_pct = (pnl_with_funding / entry_value) * 100.0 if entry_value > 0 else 0.0

            logger.info(
                f"EXIT {side_str} {symbol} | Gross PnL(Excl. Funding): {pnl_pct:+.2f}% | "
                f"Est Funding: {funding_estimate:.4f} | Net PnL: {pnl_with_funding_pct:+.2f}% | "
                f"Reason: {reason}"
            )

            with self._db_write_lock:
                self.trade_db.record_trade(
                    symbol=symbol,
                    side=side_str,
                    action="EXIT",
                    quantity=closed_qty,
                    price=exit_price_for_calc,
                    entry_price=entry_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    reason=reason,
                )
            self.state_manager.clear_symbol_state(
                symbol,
                preserve_keys=[
                    "last_entry_attempt_signal_candle_ts",
                    "entry_attempt_count_for_signal",
                    "last_entry_attempt_at",
                ],
            )
        except Exception as e:
            logger.error(f"Error in _check_exit for {symbol}: {e}")
            try:
                self.state_manager.update_symbol_state(
                    symbol,
                    {
                        "exit_pending": True,
                        "exit_error": f"exception:{e}",
                        "exit_attempt_at": datetime.utcnow().isoformat(),
                    },
                )
            except Exception:
                pass

    # --- (End of _check_exit expansion) ---

    # ( _get_current_positions, _detect_stop_loss_orders, _cancel_stop_orders_only, _cleanup_duplicate_sl_orders, _restore_stop_loss, _recover_pending_exit 메서드는 이전 코드와 100% 동일하게 유지)
    def _get_current_positions(self) -> dict:
        positions = {}
        for symbol in self.symbols:
            try:
                pos = self._fetch_position_safe(symbol)
                if abs(pos["amount"]) > 0:
                    positions[symbol] = {
                        "amount": pos["amount"],
                        "entryPrice": pos["entryPrice"],
                        "unrealizedPnL": pos["unrealizedPnL"],
                    }
            except Exception:
                pass
        return positions

    @network_api_retry
    def _detect_stop_loss_orders(self, symbol: str) -> list:
        client = self._get_client_for_symbol(symbol)
        open_orders = client.fetch_open_orders(symbol)
        return [
            o
            for o in open_orders
            if ("STOP" in o.get("type", "").upper())
            and ("TAKE_PROFIT" not in o.get("type", "").upper())
        ]

    @network_api_retry
    def _cancel_stop_orders_only(self, symbol: str):
        state = self.state_manager.get_symbol_state(symbol)
        sl_order_id = str(state.get("sl_order_id", "") or "") if state else ""
        if sl_order_id:
            try:
                client = self._get_client_for_symbol(symbol)
                client.exchange.cancel_order(sl_order_id, symbol)
            except Exception:
                pass
            self.state_manager.update_symbol_state(symbol, {"sl_order_id": None})

        client = self._get_client_for_symbol(symbol)
        open_orders = client.fetch_open_orders(symbol)
        stop_orders = [
            o
            for o in open_orders
            if ("STOP" in o.get("type", "").upper())
            and ("TAKE_PROFIT" not in o.get("type", "").upper())
        ]
        failed_cancels: list[str] = []
        last_exception: Optional[Exception] = None
        for o in stop_orders:
            try:
                client.exchange.cancel_order(o["id"], symbol)
            except Exception as e:
                logger.warning(
                    "Failed to cancel SL order %s for %s: %s", o.get("id"), symbol, e
                )
                failed_cancels.append(str(o.get("id")))
                last_exception = e
        if failed_cancels and last_exception is not None:
            raise last_exception

    def _cleanup_duplicate_sl_orders(self, symbol: str, sl_orders: list) -> list:
        if len(sl_orders) > 1:
            sorted_orders = sorted(
                sl_orders, key=lambda x: x.get("timestamp", 0), reverse=True
            )
            client = self._get_client_for_symbol(symbol)
            for old_order in sorted_orders[1:]:
                for retry in range(2):
                    try:
                        client.exchange.cancel_order(old_order["id"], symbol)
                        break
                    except Exception as e:
                        logger.warning(
                            "Failed to cancel dup SL %s (attempt %d): %s",
                            old_order.get("id"),
                            retry + 1,
                            e,
                        )
            return [sorted_orders[0]]
        return sl_orders

    def _restore_stop_loss(
        self,
        symbol: str,
        amount: float,
        entry_price: float,
        current_price: float,
        params: dict,
        atr: float,
    ) -> bool:
        client = self._get_client_for_symbol(symbol)
        sl_qty = client.round_amount(symbol, abs(amount)) or abs(amount)
        sl_type = params.get("STOP_LOSS_TYPE", "ATR")

        state = self.state_manager.get_symbol_state(symbol)
        state_stop = state.get("active_stop_price")
        try:
            state_stop_val = float(state_stop) if state_stop is not None else 0.0
        except Exception:
            state_stop_val = 0.0

        using_state_stop = bool(np.isfinite(state_stop_val) and state_stop_val > 0)
        if using_state_stop:
            stop_price = state_stop_val
            sl_side = "sell" if amount > 0 else "buy"
        else:
            if amount > 0:
                sl_side = "sell"
                if sl_type == "ATR" and atr > 0:
                    stop_price = entry_price - (
                        atr * float(params.get("LONG_ATR_MULT", 2.0))
                    )
                else:
                    stop_price = entry_price * (1 - params.get("STOP_LOSS_PCT", 0.02))
            else:
                sl_side = "buy"
                if sl_type == "ATR" and atr > 0:
                    stop_price = entry_price + (
                        atr * float(params.get("SHORT_ATR_MULT", 2.0))
                    )
                else:
                    stop_price = entry_price * (1 + params.get("STOP_LOSS_PCT", 0.02))

        tick_size = float(
            client.get_price_tick_size(
                symbol, fallback=(0.1 if "BTC" in symbol else 0.01)
            )
        )
        stop_price = max(tick_size, stop_price)
        stop_price = client.round_price(symbol, stop_price)

        if not using_state_stop:
            if amount > 0 and stop_price >= entry_price:
                stop_price = entry_price - (tick_size * 2)
            elif amount < 0 and stop_price <= entry_price:
                stop_price = entry_price + (tick_size * 2)
        stop_price = client.round_price(symbol, stop_price)

        existing = client.fetch_open_orders(symbol)
        existing_sl = [
            o for o in existing
            if ("STOP" in o.get("type", "").upper())
            and ("TAKE_PROFIT" not in o.get("type", "").upper())
        ]
        if existing_sl:
            logger.warning(
                "[%s] _restore_stop_loss: %d existing SL order(s) found on exchange — aborting restore.",
                symbol,
                len(existing_sl),
            )
            sl_order_id_existing = str(existing_sl[0].get("id", "") or "")
            self.state_manager.update_symbol_state(
                symbol,
                {
                    "last_sl_order_time": datetime.utcnow().isoformat(),
                    "sl_required": False,
                    **({"sl_order_id": sl_order_id_existing} if sl_order_id_existing else {}),
                },
            )
            return True

        sl_client_id = f"RT_SL_RS_{uuid.uuid4().hex[:17]}"
        sl_result = self._place_stop_loss_safe(
            symbol=symbol,
            side=sl_side,
            qty=sl_qty,
            stop_price=stop_price,
            client_order_id=sl_client_id,
        )
        if sl_result and isinstance(sl_result, dict):
            sl_order_id = str(sl_result.get("id", "") or "")
            update_payload = {
                "last_sl_order_time": datetime.utcnow().isoformat(),
            }
            if sl_order_id:
                update_payload["sl_order_id"] = sl_order_id
            self.state_manager.update_symbol_state(symbol, update_payload)
            time.sleep(0.5)
            return True
        return False

    def _recover_pending_exit(
        self,
        symbol: str,
        amount: float,
        current_price: float,
        params: dict,
        pos: dict,
        state: Optional[dict] = None,
    ) -> bool:
        state = state or self.state_manager.get_symbol_state(symbol)
        if not state or not bool(state.get("exit_pending", False)):
            return False

        current_pos = self._fetch_position_safe(symbol)
        actual_amount = float(current_pos.get("amount", 0.0) or 0.0)

        if abs(actual_amount) <= 0:
            self.state_manager.clear_symbol_state(
                symbol,
                preserve_keys=[
                    "last_entry_attempt_signal_candle_ts",
                    "entry_attempt_count_for_signal",
                    "last_entry_attempt_at",
                ],
            )
            return True

        now = datetime.utcnow()
        retry_cooldown = max(
            3.0,
            min(30.0, float(params.get("EXIT_RECOVERY_RETRY_COOLDOWN_SECONDS", 3.0))),
        )
        last_attempt_at = str(state.get("exit_attempt_at") or "")
        if last_attempt_at:
            try:
                if (
                    now - datetime.fromisoformat(last_attempt_at)
                ).total_seconds() < retry_cooldown:
                    return False
            except Exception:
                pass

        side_str, order_side = (
            ("LONG", "sell") if actual_amount > 0 else ("SHORT", "buy")
        )
        client = self._get_client_for_symbol(symbol)
        exit_qty = (
            client.round_amount(symbol, abs(actual_amount)) or abs(actual_amount)
        )

        recovery_attempts = int(state.get("exit_recovery_attempt_count", 0) or 0) + 1
        self.state_manager.update_symbol_state(
            symbol,
            {
                "exit_pending": True,
                "exit_attempt_at": now.isoformat(),
                "exit_requested_qty": float(abs(actual_amount)),
                "exit_recovery_attempt_count": int(recovery_attempts),
                "exit_error": None,
            },
        )

        try:
            self._cancel_all_orders_safe(symbol)
        except Exception:
            pass

        if not self._place_order_safe(
            symbol=symbol,
            side=order_side,
            qty=exit_qty,
            current_price=current_price,
            reduce_only=True,
            allow_market_fallback=True,
        ):
            self.state_manager.update_symbol_state(
                symbol,
                {
                    "exit_error": "recovery_order_submit_failed_or_unknown",
                    "exit_attempt_at": datetime.utcnow().isoformat(),
                },
            )

        timeout_seconds = max(
            4.0,
            min(20.0, float(params.get("EXIT_RECOVERY_TIMEOUT_SECONDS", 8.0))),
        )
        is_flat, remaining_amount = self._wait_until_position_flat(
            symbol, timeout_seconds=timeout_seconds, poll_seconds=0.5
        )
        if not is_flat:
            self.state_manager.update_symbol_state(
                symbol,
                {
                    "exit_pending": True,
                    "exit_error": f"not_flat_after_recovery_attempt:{remaining_amount:+.8f}",
                    "exit_remaining_amount": float(remaining_amount),
                    "exit_attempt_at": datetime.utcnow().isoformat(),
                },
            )
            return False

        try:
            self._cancel_all_orders_safe(symbol)
        except Exception:
            pass

        entry_price = float(
            state.get("entry_price", 0.0) or pos.get("entryPrice", 0.0) or 0.0
        )
        exit_price = float(current_price)
        pnl, pnl_pct = None, None
        if entry_price > 0 and exit_price > 0:
            fee_rate = float(
                params.get("TAKER_FEE_RATE", params.get("FEE_RATE", 0.0005))
            )
            entry_value = entry_price * abs(actual_amount)
            exit_value = exit_price * abs(actual_amount)
            total_fee = (entry_value + exit_value) * fee_rate

            if actual_amount > 0:
                gross_pnl = (exit_price - entry_price) * abs(actual_amount)
            else:
                gross_pnl = (entry_price - exit_price) * abs(actual_amount)

            pnl = gross_pnl - total_fee
            pnl_pct = (pnl / entry_value) * 100.0 if entry_value > 0 else 0.0

        with self._db_write_lock:
            self.trade_db.record_trade(
                symbol=symbol,
                side=side_str,
                action="EXIT",
                quantity=abs(actual_amount),
                price=exit_price,
                entry_price=(entry_price if entry_price > 0 else None),
                pnl=pnl,
                pnl_pct=pnl_pct,
                reason="Exit Recovery",
                params={"recovery_attempt_count": int(recovery_attempts)},
            )
        self.state_manager.clear_symbol_state(
            symbol,
            preserve_keys=[
                "last_entry_attempt_signal_candle_ts",
                "entry_attempt_count_for_signal",
                "last_entry_attempt_at",
            ],
        )
        return True

    def run_forever(self):
        try:
            self.initialize()
        except Exception as e:
            logger.error(f"🚨 Initialization failed: {e}")
            self.health_manager.update_heartbeat(status="init_failed")
            raise

        logger.info("⏳ Waiting for next candle close in 2D Portfolio Mode...")

        while not self._shutdown_requested:
            try:
                loop_start_time = time.time()

                # ==============================================================
                # [2D 마진 공유 아키텍처] 3-Phase Execution
                # ==============================================================

                # Phase 1: Exit First (가용 마진 확보 최우선, 스레드풀 병렬 처리)
                if self._executor is not None:
                    future_exits = {
                        self._executor.submit(self._process_exits, symbol): symbol
                        for symbol in self.symbols
                        if not self._shutdown_requested
                    }
                    for future in as_completed(future_exits):
                        symbol = future_exits[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(
                                "Error in Phase1 (exit) for %s: %s", symbol, e
                            )
                else:
                    for symbol in self.symbols:
                        if self._shutdown_requested:
                            break
                        try:
                            self._process_exits(symbol)
                        except Exception as e:
                            logger.error(
                                "Error in Phase1 fallback for %s: %s", symbol, e
                            )

                # Phase 2: Signal Scan (동시 타점 스캔, 스레드풀 적용)
                entry_candidates: list[dict] = []
                if self._executor is not None:
                    future_to_symbol = {
                        self._executor.submit(self._scan_entries, symbol): symbol
                        for symbol in self.symbols
                        if not self._shutdown_requested
                    }
                    for future in as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
                        try:
                            candidate = future.result()
                            if candidate is not None:
                                entry_candidates.append(candidate)
                        except Exception as e:
                            logger.error(
                                "Error in Phase2 (scan) for %s: %s", symbol, e
                            )
                else:
                    for symbol in self.symbols:
                        if self._shutdown_requested:
                            break
                        try:
                            candidate = self._scan_entries(symbol)
                            if candidate is not None:
                                entry_candidates.append(candidate)
                        except Exception as e:
                            logger.error(
                                "Error in Phase2 fallback for %s: %s", symbol, e
                            )

                # Phase 3: Rank & Margin Allocation (백테스트와 동일한 심볼 순서 유지)
                if entry_candidates:
                    symbol_priority = {sym: idx for idx, sym in enumerate(self.symbols)}
                    entry_candidates.sort(
                        key=lambda x: symbol_priority.get(x.get("symbol"), 999)
                    )

                    try:
                        current_total_balance, current_free_usdt = self._fetch_balance_safe()
                    except Exception:
                        current_total_balance, current_free_usdt = 0.0, 0.0

                    margin_context: Dict[str, Any] = {
                        "free_usdt": current_free_usdt,
                        "total_equity_snapshot": current_total_balance,
                    }

                    for cand in entry_candidates:
                        if self._shutdown_requested:
                            break
                        cand["margin_context"] = margin_context
                        pre_entry_free_usdt = float(
                            margin_context.get("free_usdt", 0.0)
                        )
                        try:
                            self._execute_entry_order(cand)
                        except Exception as e:
                            logger.error(
                                "Unhandled entry exception for %s: %s",
                                cand.get("symbol"),
                                e,
                            )
                            margin_context["free_usdt"] = pre_entry_free_usdt

                # ==============================================================

                # 헬스체크 및 리소스 관리
                positions = self._get_current_positions()
                self.health_manager.update_heartbeat(
                    status="running", positions=positions
                )

                now = datetime.utcnow()
                if self.cloud_optimizer:
                    if (now - self._last_ntp_check).total_seconds() >= 3600:
                        if not self.cloud_optimizer.check_time_sync_ntp():
                            logger.error(
                                "⏰ Time drift detected! Bot may fail to place orders on Binance."
                            )
                        self._last_ntp_check = now

                    if (now - self._last_resource_check).total_seconds() >= 600:
                        usage = self.cloud_optimizer.log_resource_usage()
                        if usage.get("memory_percent", 0) > 70.0:
                            logger.warning(
                                "High Memory (%.1f%%) detected. Forcing GC...",
                                float(usage.get("memory_percent", 0)),
                            )
                            self.cloud_optimizer.force_gc()
                        self._last_resource_check = now

                    if (now - self._last_gc).total_seconds() >= 1800:
                        self.cloud_optimizer.force_gc()
                        self._last_gc = now

                if (
                    getattr(self, "trade_db", None)
                    and (now - self._last_db_cleanup).total_seconds() >= 86400
                ):
                    try:
                        if hasattr(self.trade_db, "cleanup_old_records"):
                            self.trade_db.cleanup_old_records(days_to_keep=90)
                        elif self.cloud_optimizer:
                            self.cloud_optimizer.cleanup_db_old_records(
                                TRADE_HISTORY_DB, days_to_keep=90
                            )
                        self._last_db_cleanup = now
                    except Exception as e:
                        logger.warning(f"Failed to cleanup DB: {e}")

                elapsed_processing = time.time() - loop_start_time
                target_interval = float(LOOP_INTERVAL_SECONDS)
                adjusted_wait = max(0.5, target_interval - elapsed_processing)

                start_wait = time.time()
                while time.time() - start_wait < adjusted_wait:
                    if self._shutdown_requested:
                        break
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"🚨 Critical Error in Main Loop: {e}")
                self.health_manager.record_error(e)
                self.health_manager.update_heartbeat(status="error")
                time.sleep(10)

        self._shutdown()

    def _shutdown(self):
        logger.info("🛑 Shutting down gracefully...")

        if hasattr(self, "state_manager"):
            self.state_manager.flush_now()

        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                self._executor.shutdown(wait=True)
            except Exception as e:
                logger.warning(f"Failed to shutdown executor: {e}")
            finally:
                self._executor = None

        positions = self._get_current_positions()
        if positions:
            logger.warning(f"Open positions at shutdown: {positions}")

        self.health_manager.update_heartbeat(
            status="stopped",
            positions=positions,
            extra={"shutdown_time": datetime.utcnow().isoformat()},
        )
        logger.info("✅ Shutdown complete.")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 RealTrader Futures 2D Portfolio - Production Grade Bot")
    logger.info("=" * 60)

    import os

    enable_oracle_opt = (
        os.getenv("ENABLE_ORACLE_OPTIMIZATION", "true").lower() == "true"
    )
    bot = RealTraderFutures(enable_oracle_optimization=enable_oracle_opt)
    bot.run_forever()
