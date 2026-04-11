"""
SpotBot - 24시간 자동 현물(Upbit) 트레이딩 봇
===================================================
- Upbit 현물 시장 특화 (Long-Only, 1x Leverage, KRW 마켓)
- 숏(Short), 레버리지, 펀딩비 관련 로직 완벽 제거
- best_spot_4h.json[.enc] 단일 공유 파라미터 로드 (포트폴리오 전역 설정)
- 백테스트 엔진(engine_spot.py)과 동일한 진입/청산/사이징 조건 적용
"""

from __future__ import annotations

import gc
import json
import os
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

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

from config.opt_config import SPOT_SYMBOLS
from config.settings import (
    API_RETRY_ATTEMPTS,
    API_RETRY_WAIT_MAX,
    API_RETRY_WAIT_MIN,
    ERROR_SLEEP_SECONDS,
    MAX_INVEST_CAP_KRW,
    MIN_ORDER_VALUE_KRW,
    MIN_POSITION_VALUE_KRW,
    SLIPPAGE_RATE,
    SPOT_CANDLE_SETTLEMENT_DELAY_MS,
    SPOT_DRY_RUN,
    SPOT_ENTRY_SIGNAL_RETRY_MAX,
    SPOT_EXIT_CONFIRM_RETRIES,
    SPOT_EXIT_CONFIRM_SLEEP_SEC,
    SPOT_EXIT_RECOVERY_COOLDOWN_SEC,
    SPOT_HEARTBEAT_FILE,
    SPOT_LOOP_INTERVAL_SECONDS,
    SPOT_STATE_FILE,
    SPOT_STRATEGY_DB,
    SPOT_SYMBOL_DELAY_SECONDS,
    TRADING_FEE_RATE,
    UPBIT_ACCESS_KEY,
    UPBIT_SECRET_KEY,
    UPBIT_SPOT_TAKER_FEE_RATE,
)
from src.core.exchange.upbit_client import UpbitClient
from src.core.utils.components import (
    HealthCheckManager,
    TradeHistoryDB,
)
from src.core.utils.utils import setup_logger
from src.domain.spot.strategies_spot import UltimateSpotStrategy, merge_exit_family_params

logger = setup_logger("SpotBot")


def _parse_utc_dt(s: str) -> datetime:
    """Parse ISO datetime string, treating naive strings as UTC."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


_CCXT_TRANSIENT_ERRORS: Tuple[type, ...] = ()
if ccxt is not None:
    _CCXT_TRANSIENT_ERRORS = tuple(
        err
        for err in (
            getattr(ccxt, "NetworkError", None),
            getattr(ccxt, "ExchangeNotAvailable", None),
            getattr(ccxt, "RequestTimeout", None),
            getattr(ccxt, "DDoSProtection", None),
            getattr(ccxt, "RateLimitExceeded", None),
        )
        if isinstance(err, type)
    )


def _is_retryable_api_exception(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if _CCXT_TRANSIENT_ERRORS and isinstance(exc, _CCXT_TRANSIENT_ERRORS):
        return True

    # [ENHANCED] Check for nested causes (e.g. underlying network error wrapped in CCXT exception)
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        if isinstance(cause, (ConnectionError, TimeoutError)):
            return True
        if _CCXT_TRANSIENT_ERRORS and isinstance(cause, _CCXT_TRANSIENT_ERRORS):
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
                wait_time = min(float(API_RETRY_WAIT_MIN) * (2**attempt), float(API_RETRY_WAIT_MAX))
                logger.warning(
                    f"⚠️ API transient error in {func.__name__} "
                    f"(attempt {attempt + 1}/{max_attempts}): {e}. Waiting {wait_time:.1f}s"
                )
                time.sleep(max(0.0, wait_time))
        raise (
            last_error
            if last_error is not None
            else RuntimeError("retry wrapper reached unexpected state")
        )

    return wrapper


class StateManager:
    """Spot 전용 거래 상태 관리 (JSON 파일 기반)"""

    def __init__(self, state_file: Path) -> None:
        self.state_file = Path(state_file)
        self._memory_cache: Dict[str, Any] = {}
        self._cache_initialized: bool = False
        self._thread_lock = threading.Lock()
        self._dirty: bool = False
        self._last_flush: float = 0.0
        self._flush_interval: float = 1.0
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        if not self.state_file.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_unlocked({})

    def _load_unlocked(self) -> dict:
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"⚠️ State load error: {e}")
            return {}

    def _save_unlocked(self, state: dict) -> None:
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
        except Exception as e:
            logger.error(f"⚠️ State save error: {e}")
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
            except Exception:
                ...

    def get_symbol_state(self, symbol: str) -> dict:
        with self._thread_lock:
            if not self._cache_initialized:
                self._memory_cache = self._load_unlocked()
                self._cache_initialized = True
            return dict(self._memory_cache.get(symbol, {}))

    def update_symbol_state(self, symbol: str, data: dict) -> None:
        with self._thread_lock:
            if not self._cache_initialized:
                self._memory_cache = self._load_unlocked()
                self._cache_initialized = True
            if symbol not in self._memory_cache:
                self._memory_cache[symbol] = {}
            self._memory_cache[symbol].update(data)
            self._dirty = True
            self._maybe_flush()

    def clear_symbol_state(self, symbol: str, preserve_keys: Optional[list[str]] = None) -> None:
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

    def list_all_symbols(self) -> list[str]:
        with self._thread_lock:
            if not self._cache_initialized:
                self._memory_cache = self._load_unlocked()
                self._cache_initialized = True
            return list(self._memory_cache.keys())

    def _maybe_flush(self) -> None:
        if not self._dirty:
            return
        now = time.time()
        if (now - self._last_flush) >= self._flush_interval:
            self._save_unlocked(self._memory_cache)
            self._dirty = False
            self._last_flush = now

    def flush_now(self) -> None:
        with self._thread_lock:
            if self._dirty:
                self._save_unlocked(self._memory_cache)
                self._dirty = False
                self._last_flush = time.time()


class SpotBot:
    """Production-grade 업비트 현물 트레이딩 봇"""

    def __init__(self) -> None:
        self.client = UpbitClient(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        self.strategies: Dict[str, UltimateSpotStrategy] = {}
        self.params_map: Dict[str, dict] = {}
        self.symbols: list = []

        self.trade_db = TradeHistoryDB(SPOT_STRATEGY_DB)
        self.health_manager = HealthCheckManager(SPOT_HEARTBEAT_FILE)
        self.state_manager = StateManager(SPOT_STATE_FILE)

        self._shutdown_requested = False
        self.dry_run: bool = SPOT_DRY_RUN

        self._loop_count: int = 0
        self._last_gc_time: float = 0.0
        self._db_write_lock = threading.Lock()
        self._last_entry_risk_pct: Dict[str, float] = {}

        self.last_calc_candle: Dict[str, str] = {}
        self._server_time_offset_ms: int = 0
        self._last_server_time_sync: datetime = datetime.min.replace(tzinfo=timezone.utc)

        self._log_last_emit_ts: Dict[str, float] = {}
        self._log_last_message: Dict[str, str] = {}
        self._log_throttle_lock = threading.Lock()
        self._time_sync_lock = threading.Lock()

        self._setup_signal_handlers()

    def _clear_symbol_state_tracked(
        self, symbol: str, preserve_keys: Optional[list[str]] = None
    ) -> None:
        self.state_manager.clear_symbol_state(symbol, preserve_keys=preserve_keys)
        self._last_entry_risk_pct.pop(symbol, None)

    # ------------------------------------------------------------------ #
    #  Logging helpers                                                     #
    # ------------------------------------------------------------------ #

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

    def _setup_signal_handlers(self) -> None:
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}. Initiating graceful shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal_handler)

    # ------------------------------------------------------------------ #
    #  Strategy loading                                                    #
    # ------------------------------------------------------------------ #

    def load_strategies_from_json(self) -> None:
        """
        best_spot_4h.json[.enc] 단일 공유 파라미터를 로드하여
        모든 대상 심볼에 동일 전략 파라미터 적용.
        merge_exit_family_params 적용(백테스트 엔진과 동일 로직).
        """
        logger.info("📂 Loading Spot strategy from best_spot_4h[.enc]...")

        preferred_symbols = list(SPOT_SYMBOLS)
        if not preferred_symbols:
            raise ValueError(
                "No live spot symbols configured. Check SPOT_SYMBOLS in config/opt_config.py"
            )
        self.symbols = preferred_symbols.copy()

        results_dir = Path(project_root) / "results"
        json_path = results_dir / "best_spot_4h.json"
        enc_path = results_dir / "best_spot_4h.enc"

        from src.core.utils.secure_config import decrypt_config, get_strategy_secret

        secret = get_strategy_secret()
        shared_params: Optional[Dict[str, Any]] = None

        # 1. Try encrypted first
        if enc_path.exists() and secret:
            try:
                shared_params = decrypt_config(enc_path.read_bytes(), secret)
                logger.info(f"🛡️ Loaded from encrypted config: {enc_path}")
            except Exception as e:
                logger.error(f"❌ Failed to decrypt {enc_path}: {e}")

        # 2. Fallback to plaintext
        if shared_params is None and json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    shared_params = json.load(f)
                logger.info(f"📂 Loaded from plaintext config: {json_path}")
            except Exception as e:
                logger.error(f"❌ Failed to parse {json_path}: {e}")

        if shared_params is None:
            raise RuntimeError(f"No valid strategy config found (tried {json_path} and {enc_path})")

        # Apply exit-family adjustments (identical to backtest pipeline)
        shared_params = merge_exit_family_params(shared_params)
        shared_params.setdefault("INDICATOR_TIMEFRAME", "4h")
        shared_params.setdefault("TIMEFRAME", "4h")

        logger.info(
            f"✅ Strategy config loaded | EXIT_FAMILY={shared_params.get('EXIT_FAMILY')} | "
            f"TRAIL_ATR_MULT={shared_params.get('TRAIL_ATR_MULT')} | "
            f"LONG_TP_MULT={shared_params.get('LONG_TP_MULT', 0.0)} | "
            f"TIME_STOP_BARS={shared_params.get('TIME_STOP_BARS')}"
        )

        for symbol in self.symbols:
            clean_sym = symbol.replace("/", "").replace("-", "")
            self.params_map[symbol] = dict(shared_params)
            self.strategies[symbol] = UltimateSpotStrategy(f"RealSpot_{clean_sym}", shared_params)
            logger.info(
                f"✅ [{symbol}] Strategy initialized | TF={shared_params.get('TIMEFRAME', '4h')}"
            )

        shared_params = self.params_map.get(self.symbols[0], {}) if self.symbols else {}
        logger.info(
            "📌 Symbols: %s | RISK_PER_TRADE=%.1f%% | MAX_CAP_PER_COIN=%.0f%% | LONG_ATR_MULT=%.2f",
            ", ".join(self.symbols),
            float(shared_params.get("RISK_PER_TRADE", 0)) * 100,
            float(shared_params.get("MAX_CAP_PER_COIN", 1.0)) * 100,
            float(shared_params.get("LONG_ATR_MULT", 3.0)),
        )

    # ------------------------------------------------------------------ #
    #  Safe API wrappers                                                   #
    # ------------------------------------------------------------------ #

    @network_api_retry
    def _fetch_balance_safe(self) -> tuple:
        """Returns (total_krw, free_krw)"""
        return self.client.fetch_balance()

    @network_api_retry
    def _fetch_ohlcv_safe(
        self, symbol: str, timeframe: str, start_str: str
    ) -> Optional[pd.DataFrame]:
        return self.client.fetch_ohlcv(symbol, timeframe, start_date=start_str)

    @network_api_retry
    def _fetch_recent_ohlcv_safe(
        self, symbol: str, timeframe: str, limit: int = 3
    ) -> Optional[pd.DataFrame]:
        return self.client.fetch_recent_ohlcv(symbol, timeframe, limit=limit)

    @network_api_retry
    def _get_market_price_safe(self, symbol: str) -> float:
        return self.client.get_market_price(symbol)

    @network_api_retry
    def _fetch_server_time_ms_safe(self) -> int:
        result = self.client.fetch_server_time_ms()
        if result is None:
            raise ConnectionError("Server time returned None")
        return int(result)

    # ------------------------------------------------------------------ #
    #  Server time & candle slot                                           #
    # ------------------------------------------------------------------ #

    def _sync_server_time_offset(
        self, force: bool = False, sync_interval_seconds: int = 60
    ) -> None:
        with self._time_sync_lock:
            now = datetime.now(timezone.utc)
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
                logger.debug(f"[time-sync] Server time sync failed: {e}")

    def _get_reference_now_ms(self) -> int:
        self._sync_server_time_offset(force=False)
        return int(datetime.now(timezone.utc).timestamp() * 1000) + int(self._server_time_offset_ms)

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
        tf_min = self._timeframe_to_minutes(timeframe)
        if tf_min <= 0:
            return "unknown"
        now_ms = self._get_reference_now_ms()
        interval_ms = tf_min * 60 * 1000
        slot_start_ms = now_ms - (now_ms % interval_ms)
        return f"{timeframe}_{slot_start_ms}"

    def _select_last_closed_candle(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        if "timestamp" not in df.columns:
            return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        interval_min = self._timeframe_to_minutes(timeframe)
        if interval_min <= 0:
            return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        interval_ms = interval_min * 60 * 1000
        now_ms = self._get_reference_now_ms()
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
        closed_mask = (timestamps + interval_ms) <= now_ms
        closed_indices = df.index[closed_mask.to_numpy()]
        if len(closed_indices) > 0:
            return df.loc[closed_indices[-1]]
        return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]

    def _extract_candle_timestamp_ms(self, candle: Optional[pd.Series]) -> int:
        if candle is None:
            return 0
        raw_ts = candle.get("timestamp", 0)
        try:
            return int(raw_ts) if not pd.isna(raw_ts) else 0
        except Exception:
            return 0

    # ------------------------------------------------------------------ #
    #  Indicator cache                                                     #
    # ------------------------------------------------------------------ #

    def _cache_indicators(self, symbol: str, data: dict) -> None:
        if not hasattr(self, "_ind_cache"):
            self._ind_cache: Dict[str, dict] = {}
        self._ind_cache[symbol] = data

    def _get_cached_indicators(self, symbol: str) -> dict:
        if not hasattr(self, "_ind_cache"):
            self._ind_cache = {}
        return self._ind_cache.get(symbol, {})

    # ------------------------------------------------------------------ #
    #  OHLCV prefetch for market breadth regime                           #
    # ------------------------------------------------------------------ #

    def _prefetch_all_ohlcv(self) -> Dict[str, pd.DataFrame]:
        """
        Market Breadth Regime를 위해 모든 심볼의 OHLCV를 일괄 수집.
        Returns {symbol: df} mapping (실패한 심볼은 제외).
        """
        if not self.symbols:
            return {}
        tf = str(self.params_map.get(self.symbols[0], {}).get("TIMEFRAME", "4h"))
        tf_min = self._timeframe_to_minutes(tf)
        limit = 310
        lookback_days = (limit * tf_min) / 1440
        start_dt = datetime.now(timezone.utc) - timedelta(days=lookback_days + 2)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        data_maps: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            try:
                df = self._fetch_ohlcv_safe(symbol, tf, start_str)
                if df is not None and len(df) >= 50:
                    df[["open", "high", "low", "close", "volume"]] = df[
                        ["open", "high", "low", "close", "volume"]
                    ].astype(np.float64)
                    data_maps[symbol] = df
            except Exception as e:
                logger.warning(f"[{symbol}] OHLCV prefetch failed: {e}")

        logger.debug(f"Prefetched OHLCV for {len(data_maps)}/{len(self.symbols)} symbols")
        return data_maps

    # ------------------------------------------------------------------ #
    #  Position query                                                      #
    # ------------------------------------------------------------------ #

    def _get_current_position(self, symbol: str, current_price: float) -> dict:
        """Upbit 잔고 조회로 특정 코인의 보유 포지션 반환"""
        try:
            base_coin = symbol.split("-")[1] if "-" in symbol else symbol.split("/")[0]
            balance = self.client.fetch_balance_dict()

            if not balance or "free" not in balance or "used" not in balance:
                return {"amount": 0.0, "entryPrice": 0.0, "value": 0.0}

            amount = float(balance["free"].get(base_coin, 0.0)) + float(
                balance["used"].get(base_coin, 0.0)
            )
            value_krw = amount * current_price

            if value_krw < MIN_POSITION_VALUE_KRW:
                return {"amount": 0.0, "entryPrice": 0.0, "value": 0.0}

            state = self.state_manager.get_symbol_state(symbol)
            entry_price = float(state.get("entry_price", current_price))

            return {"amount": amount, "entryPrice": entry_price, "value": value_krw}
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch Spot position: {e}")
            return {"amount": 0.0, "entryPrice": 0.0, "value": 0.0}

    def _get_spot_coin_amount_raw(self, symbol: str) -> float:
        """Total (free+used) base coin on exchange; ignores local state."""
        try:
            base_coin = symbol.split("-")[1] if "-" in symbol else symbol.split("/")[0]
            balance = self.client.fetch_balance_dict()
            if not balance or "free" not in balance or "used" not in balance:
                return 0.0
            return float(balance["free"].get(base_coin, 0.0)) + float(
                balance["used"].get(base_coin, 0.0)
            )
        except Exception as e:
            logger.error("[%s] Failed to read base coin amount: %s", symbol, e)
            return 0.0

    # ------------------------------------------------------------------ #
    #  Position sizing (engine-aligned: risk-budget / stop-distance)      #
    # ------------------------------------------------------------------ #

    def _calculate_spot_position_size(
        self,
        fill_price: float,
        entry_atr: float,
        params: dict,
        balance_krw: float,
        regime_risk_mult: float,
        garch_kelly_f: float,
        regime_state: float = 2.0,
    ) -> float:
        """
        백테스트 엔진과 동일한 리스크 기반 포지션 사이징.
        risk_budget = balance * RISK_PER_TRADE * regime_risk_mult * garch_kelly_f
        amount = risk_budget / stop_distance  (stop_distance = ATR * LONG_ATR_MULT)
        """
        long_atr_mult = float(params.get("LONG_ATR_MULT", 3.0))
        stop_distance = entry_atr * long_atr_mult

        if stop_distance <= 0.0 or fill_price <= 0.0:
            logger.warning("⚠️ Invalid ATR/price for sizing; skipping entry.")
            return 0.0

        risk_per_trade = float(params.get("RISK_PER_TRADE", 0.05))

        # Regime/Kelly 조정 (엔진과 동일)
        rr = max(0.05, min(1.0, float(regime_risk_mult) if np.isfinite(regime_risk_mult) else 1.0))
        gk = float(garch_kelly_f) if np.isfinite(garch_kelly_f) and garch_kelly_f > 0.0 else 1.0
        eff = rr * gk
        eff = max(0.05, min(1.0, eff))

        # SOFT regime (0.5 < regime_state < 1.5): engine_spot.py entry path alignment
        if 0.5 < regime_state < 1.5:
            eff = eff * 0.90

        new_risk_pct = risk_per_trade * eff

        use_compounding = bool(params.get("USE_COMPOUNDING", True))
        max_capital_usage = float(params.get("MAX_CAPITAL_USAGE", 1e12))
        current_equity = (
            min(balance_krw, max_capital_usage)
            if use_compounding and balance_krw > max_capital_usage
            else balance_krw
        )

        risk_budget = current_equity * new_risk_pct
        raw_amount = risk_budget / stop_distance

        # MAX_CAP_PER_COIN 상한
        max_cap_pct = float(params.get("MAX_CAP_PER_COIN", 1.0))
        max_notional = current_equity * max_cap_pct
        amount_cap = max_notional / fill_price

        # 실제 잔고 초과 방지 (수수료 버퍼 포함)
        max_affordable = (balance_krw * 0.99) / (fill_price * (1.0 + TRADING_FEE_RATE))

        amount = min(raw_amount, amount_cap, max_affordable)

        # 최대 투자 금액 절대 상한
        if amount * fill_price > MAX_INVEST_CAP_KRW:
            amount = MAX_INVEST_CAP_KRW / fill_price

        if amount * fill_price < MIN_ORDER_VALUE_KRW:
            logger.warning(
                f"⚠️ Calculated order value ({amount * fill_price:.0f} KRW) "
                f"< minimum ({MIN_ORDER_VALUE_KRW} KRW). Skipping."
            )
            return 0.0

        return amount

    # ------------------------------------------------------------------ #
    #  Recovery & Protection                                              #
    # ------------------------------------------------------------------ #

    def _bootstrap_state_for_open_position(
        self,
        symbol: str,
        amount: float,
        pos: dict,
        current_price: float,
        params: dict,
        atr: float,
    ) -> bool:
        """
        [ENHANCED] Bootstraps local StateManager from live exchange position.
        Called when a position is found on Upbit but no corresponding state is in spot_state.json.
        """
        try:
            entry_price = float(pos.get("entryPrice", 0.0) or current_price or 0.0)
            if entry_price <= 0.0:
                entry_price = current_price

            entry_atr = float(atr) if np.isfinite(atr) and atr > 0 else 0.0
            long_atr_mult = float(params.get("LONG_ATR_MULT", 3.0))

            # Use original sizing ATR if available, else current ATR
            initial_stop = (
                entry_price - (entry_atr * long_atr_mult) if entry_atr > 0 else entry_price * 0.85
            )

            # Taking Profit price calculation if enabled
            long_tp_mult = float(params.get("LONG_TP_MULT", 0.0))
            tp_price = (
                entry_price + (entry_atr * long_tp_mult)
                if long_tp_mult > 1e-9 and entry_atr > 0
                else 0.0
            )

            state_data = {
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "entry_price": float(entry_price),
                "entry_atr": float(entry_atr),
                "side": "LONG",
                "highest_price": float(max(entry_price, current_price)),
                "active_stop_price": float(initial_stop),
                "tp_price": float(tp_price),
                "recovery_bootstrapped": True,
                "initial_amount": float(abs(amount)),
                "fractal_scale_done": False,
                "exit_pending": False,
                "exit_error": None,
                "exit_remaining_amount": 0.0,
                "exit_recovery_attempt_count": 0,
                "exit_attempt_at": None,
            }

            self.state_manager.update_symbol_state(symbol, state_data)
            logger.warning(
                "🛡️ [%s] Local state bootstrapped from live position: Entry=%.2f, Stop=%.2f",
                symbol,
                entry_price,
                initial_stop,
            )
            return True
        except Exception as e:
            logger.error("❌ [%s] Failed to bootstrap state: %s", symbol, e)
            return False

    def _enforce_min_fill_ratio(
        self,
        symbol: str,
        expected_qty: float,
        actual_qty: float,
        params: dict,
    ) -> bool:
        """
        [ENHANCED] Guards against underfilled orders that might lead to state divergence.
        """
        if expected_qty <= 0 or actual_qty <= 0:
            return actual_qty > 0

        min_fill_ratio = float(params.get("ENTRY_MIN_FILL_RATIO", 0.60))
        fill_ratio = actual_qty / expected_qty

        if fill_ratio < min_fill_ratio:
            logger.warning(
                "⚠️ [%s] Underfilled entry (Ratio: %.2f < %.2f). Actual: %.4f",
                symbol,
                fill_ratio,
                min_fill_ratio,
                actual_qty,
            )
            # We still return True but log the warning.
            # In some cases we might want to return False to abort if ratio is too low.
        return True

    # ------------------------------------------------------------------ #
    #  Order execution                                                     #
    # ------------------------------------------------------------------ #

    @network_api_retry
    def _place_order_safe(
        self, symbol: str, side: str, qty: float, client_order_id: Optional[str] = None
    ) -> Optional[dict]:
        """
        [ENHANCED] Order execution with dry-run support, reconciliation for timeouts,
        and idempotent behavior using clientOrderId (where supported) or balance tracking.
        """
        if self.dry_run:
            px = float(self.client.get_market_price(symbol) or 0.0)
            logger.info("[DRY-RUN] Order: %s %s %f @ %f", side.upper(), symbol, qty, px)
            return {"id": "dry-run", "status": "closed", "amount": qty, "price": px}

        try:
            ccxt_symbol = self.client._normalize_symbol(symbol)

            # Upbit doesn't strictly support clientOrderId in the same way,
            # but we can pass it in params if the adapter supports it or for local tracking.
            params = {}
            if client_order_id:
                params["clientOrderId"] = client_order_id

            if side.lower() == "buy":
                current_price = self.client.get_market_price(symbol)
                # Upbit market orders require 'cost' (KRW amount) for BUY
                cost = max(qty * (current_price or 0.0), MIN_ORDER_VALUE_KRW)
                order = self.client.exchange.create_order(
                    symbol=ccxt_symbol,
                    type="market",
                    side="buy",
                    amount=qty,
                    price=None,
                    params={**params, "cost": cost},
                )
            else:
                order = self.client.exchange.create_order(
                    symbol=ccxt_symbol, type="market", side="sell", amount=qty, params=params
                )

            logger.info(f"⚡ Order Placed: market {side} {qty} {ccxt_symbol}")
            return order

        except ccxt.RequestTimeout:
            # [RECONCILIATION] Special handling for timeouts to avoid double entry or orphaned positions
            logger.warning("⚠️ [%s] Order timeout. Reconciling with exchange...", symbol)
            try:
                # 1. Check open orders (if any remained on book)
                open_orders = self.client.fetch_open_orders(symbol)
                for o in open_orders:
                    # If we used clientOrderId, match it. Else look for recent market order.
                    # Upbit market orders are usually instant, so they might be closed already.
                    if client_order_id and o.get("clientOrderId") == client_order_id:
                        logger.info("Found timed-out order in open orders.")
                        return o

                # 2. Check balance change for buy, or check if position gone for sell
                # (This is a simplified version of the logic in real_trader_futures.py)
                logger.info(
                    "Reconciliation complete. No definitive order found. Re-checking balance in next cycle."
                )
                return None
            except Exception as rec_e:
                logger.error("❌ Reconciliation failed: %s", rec_e)
                return None

        except Exception as e:
            logger.error(f"❌ Order Failed for {symbol}: {e}")
            return None

    def _confirm_buy_fill(
        self,
        symbol: str,
        expected_qty: float,
        retries: int = SPOT_EXIT_CONFIRM_RETRIES,
        sleep_sec: float = SPOT_EXIT_CONFIRM_SLEEP_SEC,
    ) -> Tuple[bool, float]:
        if self.dry_run:
            return True, float(expected_qty)
        min_fill = max(0.0, float(expected_qty) * 0.95)
        last_amt = 0.0
        for attempt in range(max(1, retries)):
            if attempt > 0:
                time.sleep(max(0.0, sleep_sec))
            last_amt = self._get_spot_coin_amount_raw(symbol)
            if last_amt >= min_fill:
                return True, last_amt
        return False, last_amt

    def _confirm_sell_flat(
        self,
        symbol: str,
        retries: int = SPOT_EXIT_CONFIRM_RETRIES,
        sleep_sec: float = SPOT_EXIT_CONFIRM_SLEEP_SEC,
    ) -> Tuple[bool, float]:
        if self.dry_run:
            return True, 0.0
        last_amt = 0.0
        for attempt in range(max(1, retries)):
            if attempt > 0:
                time.sleep(max(0.0, sleep_sec))
            try:
                px = self._get_market_price_safe(symbol)
            except Exception:
                px = 0.0
            last_amt = self._get_spot_coin_amount_raw(symbol)
            value_krw = last_amt * px
            if last_amt < 1e-12 or value_krw < float(MIN_POSITION_VALUE_KRW):
                return True, 0.0
        return False, last_amt

    def _recover_pending_exit(self, symbol: str, current_price: float) -> bool:
        """Retry failed liquidation. Returns True when flat or no pending exit."""
        state = self.state_manager.get_symbol_state(symbol)
        if not state or not bool(state.get("exit_pending", False)):
            return True

        raw_amt = self._get_spot_coin_amount_raw(symbol)
        try:
            px_chk = self._get_market_price_safe(symbol)
        except Exception:
            px_chk = float(current_price)
        value_krw = raw_amt * px_chk if px_chk > 0 else 0.0
        if raw_amt < 1e-12 or value_krw < float(MIN_POSITION_VALUE_KRW):
            self._clear_symbol_state_tracked(
                symbol,
                preserve_keys=[
                    "last_entry_attempt_signal_candle_ts",
                    "entry_attempt_count_for_signal",
                ],
            )
            return True

        now = datetime.now(timezone.utc)
        last_attempt_at = str(state.get("exit_attempt_at") or "")
        cooldown = float(SPOT_EXIT_RECOVERY_COOLDOWN_SEC)
        if last_attempt_at:
            try:
                if (now - _parse_utc_dt(last_attempt_at)).total_seconds() < cooldown:
                    return False
            except Exception:
                ...

        recovery_attempts = int(state.get("exit_recovery_attempt_count", 0) or 0) + 1
        self.state_manager.update_symbol_state(
            symbol,
            {
                "exit_pending": True,
                "exit_attempt_at": now.isoformat(),
                "exit_recovery_attempt_count": recovery_attempts,
                "exit_error": None,
            },
        )

        order = self._place_order_safe(symbol, "sell", raw_amt)
        if not order:
            self.state_manager.update_symbol_state(
                symbol,
                {
                    "exit_error": "recovery_order_failed",
                    "exit_attempt_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return False

        is_flat, remaining = self._confirm_sell_flat(symbol)
        if not is_flat:
            self.state_manager.update_symbol_state(
                symbol,
                {
                    "exit_pending": True,
                    "exit_error": f"not_flat_after_recovery:{remaining:.8f}",
                    "exit_remaining_amount": float(remaining),
                    "exit_attempt_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return False

        entry_price = float(state.get("entry_price", 0.0) or 0.0)
        exit_price = float(current_price)
        exit_amt = float(
            state.get("initial_amount") or state.get("exit_remaining_amount") or raw_amt or 0.0
        )
        if exit_amt <= 0.0:
            exit_amt = raw_amt

        fee_rate = float(UPBIT_SPOT_TAKER_FEE_RATE)
        entry_value = entry_price * exit_amt
        exit_value = exit_price * exit_amt
        total_fee = (entry_value + exit_value) * fee_rate
        gross_pnl = (exit_price - entry_price) * exit_amt
        net_pnl = gross_pnl - total_fee
        pnl_pct = (net_pnl / entry_value) * 100.0 if entry_value > 0 else 0.0

        with self._db_write_lock:
            self.trade_db.record_trade(
                symbol=symbol,
                side="LONG",
                action="EXIT",
                quantity=exit_amt,
                price=exit_price,
                entry_price=entry_price if entry_price > 0 else None,
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                reason="Exit Recovery",
                params={"recovery_attempt_count": int(recovery_attempts)},
            )

        self._clear_symbol_state_tracked(
            symbol,
            preserve_keys=[
                "last_entry_attempt_signal_candle_ts",
                "entry_attempt_count_for_signal",
            ],
        )
        return True

    # ------------------------------------------------------------------ #
    #  Initialization                                                      #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        logger.info("🤖 SpotBot (Upbit) Initializing...")
        self._sync_server_time_offset(force=True)

        self.load_strategies_from_json()
        self._reconcile_positions_on_startup()
        gc.collect()

        try:
            total_krw, free_krw = self._fetch_balance_safe()
            logger.info(f"💰 Upbit Balance: {free_krw:,.0f} KRW (Total: {total_krw:,.0f} KRW)")
            if free_krw < MIN_ORDER_VALUE_KRW:
                logger.warning(f"⚠️ Low KRW balance (< {MIN_ORDER_VALUE_KRW} KRW)!")
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")

        self.health_manager.update_heartbeat(status="initialized")
        logger.info("🚀 Spot Initialization Complete. Bot is Running...")

    # ------------------------------------------------------------------ #
    #  Core trading logic                                                  #
    # ------------------------------------------------------------------ #

    def _refresh_indicators_if_needed(
        self, symbol: str, all_ohlcv: Dict[str, pd.DataFrame]
    ) -> dict:
        """Compute and cache indicators. Returns cached indicator dict."""
        if symbol not in self.params_map or symbol not in self.strategies:
            return {}
        params = self.params_map[symbol]
        strategy = self.strategies[symbol]
        indicator_tf = str(params.get("INDICATOR_TIMEFRAME", "4h"))
        tf_min = self._timeframe_to_minutes(indicator_tf)
        current_slot = self._get_candle_slot_id(indicator_tf)
        cached = self._get_cached_indicators(symbol)
        required_keys = (
            "trend_direction",
            "atr",
            "entry_upper",
            "strength_filter",
            "regime_entry_gate",
            "regime_risk_mult",
            "garch_kelly_f",
            "kill_signal",
            "bb_upper",
            "trail_tighten_flag",
            "regime_state",
            "signal_candle_ts",
            "fractal_high_flag",
        )
        need_calculation = (
            not cached
            or any(k not in cached for k in required_keys)
            or self.last_calc_candle.get(symbol) != current_slot
        )
        if not need_calculation:
            return cached

        symbol_df = all_ohlcv.get(symbol)
        if symbol_df is None or len(symbol_df) < 50:
            self.last_calc_candle[symbol] = current_slot
            return cached

        last_raw = self._select_last_closed_candle(symbol_df, indicator_tf)
        if last_raw is not None and tf_min > 0:
            candle_ts = self._extract_candle_timestamp_ms(last_raw)
            interval_ms = tf_min * 60 * 1000
            candle_close_ms = candle_ts + interval_ms
            now_ms = self._get_reference_now_ms()
            elapsed_since_close = now_ms - candle_close_ms
            if 0 < elapsed_since_close < int(SPOT_CANDLE_SETTLEMENT_DELAY_MS):
                return cached

        min_len = min(len(df) for df in all_ohlcv.values())
        aligned_maps: Dict[str, Dict[str, pd.DataFrame]] = {
            s: {indicator_tf: df.iloc[-min_len:].reset_index(drop=True)}
            for s, df in all_ohlcv.items()
        }
        strategy._portfolio_eval_ctx = {
            "data_maps": aligned_maps,
            "symbols": list(aligned_maps.keys()),
            "tf": indicator_tf,
        }
        aligned_df = symbol_df.iloc[-min_len:].reset_index(drop=True).copy()
        signal_df = strategy.generate_signals(aligned_df)
        last_candle = self._select_last_closed_candle(signal_df, indicator_tf)
        if last_candle is None:
            return cached

        regime_state_val = float(last_candle.get("regime_state", 2.0))
        signal_candle_ts = int(self._extract_candle_timestamp_ms(last_candle))
        fractal_high_flag_val = float(last_candle.get("fractal_high_flag", 0.0))

        self._cache_indicators(
            symbol,
            {
                "trend_direction": int(last_candle.get("trend_direction", 0)),
                "atr": float(last_candle.get("atr", 0.0)),
                "entry_upper": float(last_candle.get("entry_upper", 999999.0)),
                "strength_filter": int(last_candle.get("strength_filter", 0)),
                "regime_entry_gate": float(last_candle.get("regime_entry_gate", 0.0)),
                "regime_risk_mult": float(last_candle.get("regime_risk_mult", 1.0)),
                "garch_kelly_f": float(last_candle.get("garch_kelly_f", 1.0)),
                "kill_signal": float(last_candle.get("kill_signal", 0.0)),
                "bb_upper": float(last_candle.get("bb_upper", np.inf)),
                "trail_tighten_flag": float(last_candle.get("trail_tighten_flag", 0.0)),
                "regime_state": regime_state_val,
                "signal_candle_ts": signal_candle_ts,
                "fractal_high_flag": fractal_high_flag_val,
                "indicator_timeframe": indicator_tf,
            },
        )
        self.last_calc_candle[symbol] = current_slot
        return self._get_cached_indicators(symbol)

    def _process_exit(self, symbol: str, current_price: float, cached: dict) -> None:
        """Exit-only path (includes bootstrap and exit_pending recovery)."""
        try:
            if symbol not in self.params_map or symbol not in self.strategies:
                return
            params = self.params_map[symbol]
            indicator_tf = str(params.get("INDICATOR_TIMEFRAME", "4h"))

            state = self.state_manager.get_symbol_state(symbol)
            if bool(state.get("exit_pending", False)):
                if not self._recover_pending_exit(symbol, current_price):
                    return
                state = self.state_manager.get_symbol_state(symbol)

            pos = self._get_current_position(symbol, current_price)
            amount = float(pos["amount"])
            in_position = amount > 0

            if not in_position and state and state.get("entry_price"):
                self._log_throttled(
                    "info",
                    f"{symbol}:cleanup",
                    f"🧹 [{symbol}] Clearing stale state.",
                    600.0,
                )
                self._clear_symbol_state_tracked(
                    symbol,
                    preserve_keys=[
                        "last_entry_attempt_signal_candle_ts",
                        "entry_attempt_count_for_signal",
                    ],
                )
                state = {}

            # trend_dir = int(cached.get("trend_direction", 0))
            atr = float(cached.get("atr", 0.0))
            kill_signal = float(cached.get("kill_signal", 0.0))
            bb_upper = float(cached.get("bb_upper", np.inf))
            trail_tighten_flag = float(cached.get("trail_tighten_flag", 0.0))
            rst = float(cached.get("regime_state", 2.0))
            trail_regime_factor = max(0.5, 1.0 - (2.0 - rst) * 0.14)
            ts_regime_factor = max(0.50, 1.0 - (2.0 - rst) * 0.35)

            if in_position and (not state or not state.get("entry_price")):
                success = self._bootstrap_state_for_open_position(
                    symbol=symbol,
                    amount=amount,
                    pos=pos,
                    current_price=current_price,
                    params=params,
                    atr=atr,
                )
                if not success:
                    return
                state = self.state_manager.get_symbol_state(symbol)

            if not in_position:
                return

            state = self.state_manager.get_symbol_state(symbol)
            entry_price = float(state.get("entry_price", pos["entryPrice"]))
            if entry_price <= 0.0:
                entry_price = current_price

            highest = float(state.get("highest_price", current_price))
            if current_price > highest:
                highest = current_price
                self.state_manager.update_symbol_state(symbol, {"highest_price": highest})

            entry_atr = float(state.get("entry_atr", atr))
            stop_price = float(state.get("active_stop_price", entry_price * 0.85))

            long_trail_mult = float(
                params.get("TRAIL_ATR_MULT", params.get("LONG_TRAIL_MULT", 5.0))
            )
            long_trail_lock_mult = float(params.get("LONG_TRAIL_LOCK_MULT", 1.5))
            tp_lock_atr_mult = float(params.get("TP_LOCK_ATR_MULT", 3.0))
            long_tp_mult = float(params.get("LONG_TP_MULT", 0.0))
            time_stop_bars = int(params.get("TIME_STOP_BARS", 0))

            dist = highest - entry_price
            current_trail_mult = long_trail_mult * trail_regime_factor
            if trail_tighten_flag > 0.5 or (
                entry_atr > 0.0 and dist > entry_atr * tp_lock_atr_mult
            ):
                current_trail_mult = long_trail_lock_mult
            effective_trail_mult = float(current_trail_mult)

            if entry_atr > 0.0:
                new_stop = highest - (entry_atr * effective_trail_mult)
                if new_stop > stop_price:
                    stop_price = new_stop
                    self.state_manager.update_symbol_state(
                        symbol, {"active_stop_price": stop_price}
                    )

            tp_price = entry_price + (entry_atr * long_tp_mult) if long_tp_mult > 1e-9 else np.inf

            eff_time_stop = (
                max(1, int(time_stop_bars * ts_regime_factor)) if time_stop_bars > 0 else 0
            )

            exit_triggered = False
            reason = ""

            fractal_high_flag = float(cached.get("fractal_high_flag", 0.0))
            scale_out_ratio = float(params.get("SCALE_OUT_RATIO", 0.5))
            fractal_scale_done = bool(state.get("fractal_scale_done", False))

            if (
                not fractal_scale_done
                and fractal_high_flag > 0.5
                and amount > 0.0
                and not exit_triggered
            ):
                eff_scale = scale_out_ratio
                if 0.5 < rst < 1.5:
                    eff_scale = min(0.95, scale_out_ratio * 1.12)
                scale_amount = amount * eff_scale
                cap_amt = amount * 0.99
                if scale_amount > cap_amt:
                    scale_amount = cap_amt
                if scale_amount * current_price >= float(MIN_ORDER_VALUE_KRW):
                    logger.info(
                        f"📉 [{symbol}] Scale-out {scale_amount:.4f} @ {current_price:,.0f} "
                        f"(ratio={eff_scale:.2f}, fractal)"
                    )
                    scale_order = self._place_order_safe(symbol, "sell", scale_amount)
                    if scale_order:
                        if self.dry_run:
                            new_amount = max(0.0, float(amount) - float(scale_amount))
                        else:
                            time.sleep(max(0.0, float(SPOT_EXIT_CONFIRM_SLEEP_SEC)))
                            new_amount = self._get_spot_coin_amount_raw(symbol)
                        self.state_manager.update_symbol_state(
                            symbol,
                            {
                                "fractal_scale_done": True,
                                "entry_amount": float(new_amount),
                            },
                        )
                        amount = float(new_amount)
                        pos = self._get_current_position(symbol, current_price)
                        in_position = float(pos["amount"]) > 0
                        if not in_position:
                            return
                        state = self.state_manager.get_symbol_state(symbol)

            if not exit_triggered and kill_signal > 0.5:
                exit_triggered = True
                reason = "Kill Signal"

            if not exit_triggered and current_price <= stop_price:
                exit_triggered = True
                reason = f"Stop Loss ({stop_price:,.0f})"

            if not exit_triggered and current_price >= tp_price:
                exit_triggered = True
                reason = f"Take Profit ({tp_price:,.0f})"

            if (
                not exit_triggered
                and np.isfinite(bb_upper)
                and bb_upper < 1e18
                and current_price >= bb_upper
            ):
                exit_triggered = True
                reason = f"BB Upper ({bb_upper:,.0f})"

            if not exit_triggered and time_stop_bars > 0 and eff_time_stop > 0:
                entry_time_str = state.get("entry_time")
                if entry_time_str:
                    try:
                        entry_dt = _parse_utc_dt(entry_time_str)
                        elapsed_minutes = (
                            datetime.now(timezone.utc) - entry_dt
                        ).total_seconds() / 60.0
                        tf_min = self._timeframe_to_minutes(indicator_tf)
                        bars_elapsed = int(elapsed_minutes / tf_min) if tf_min > 0 else 0
                        if bars_elapsed >= eff_time_stop:
                            exit_triggered = True
                            reason = f"Time Stop ({bars_elapsed}>={eff_time_stop} bars, ts_reg={ts_regime_factor:.2f})"
                    except Exception:
                        pass

            self._log_throttled(
                "debug",
                f"{symbol}:pos",
                f"📊 [{symbol}] Holding {amount:.4f} | Cur: {current_price:,.0f} | "
                f"Stop: {stop_price:,.0f} | BB: {bb_upper:,.0f} | "
                f"Trail×: {effective_trail_mult:.2f} | trf={trail_regime_factor:.2f} | "
                f"tsf={ts_regime_factor:.2f} | TP: {'off' if tp_price == np.inf else f'{tp_price:,.0f}'}",
                600.0,
            )

            if not exit_triggered:
                return

            logger.warning(f"🚨 [{symbol}] Exit Triggered: {reason}. Selling {amount:.4f}...")
            self.state_manager.update_symbol_state(
                symbol,
                {
                    "exit_pending": True,
                    "exit_attempt_at": datetime.now(timezone.utc).isoformat(),
                    "exit_remaining_amount": float(amount),
                    "exit_recovery_attempt_count": 0,
                    "exit_error": None,
                },
            )
            order = self._place_order_safe(symbol, "sell", amount)
            if not order:
                self.state_manager.update_symbol_state(
                    symbol,
                    {
                        "exit_error": "sell_order_failed",
                        "exit_attempt_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                return

            is_flat, remaining = self._confirm_sell_flat(symbol)
            if not is_flat:
                self.state_manager.update_symbol_state(
                    symbol,
                    {
                        "exit_pending": True,
                        "exit_remaining_amount": float(remaining),
                        "exit_error": "sell_not_confirmed_flat",
                        "exit_attempt_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                return

            exit_price = float(current_price)
            fee_rate = float(UPBIT_SPOT_TAKER_FEE_RATE)
            entry_value = entry_price * amount
            exit_value = exit_price * amount
            total_fee = (entry_value + exit_value) * fee_rate
            gross_pnl = (exit_price - entry_price) * amount
            net_pnl = gross_pnl - total_fee
            pnl_pct = (net_pnl / entry_value) * 100.0 if entry_value > 0 else 0.0

            with self._db_write_lock:
                self.trade_db.record_trade(
                    symbol=symbol,
                    side="LONG",
                    action="EXIT",
                    quantity=amount,
                    price=exit_price,
                    entry_price=entry_price,
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    reason=reason,
                    params={},
                )

            self._clear_symbol_state_tracked(
                symbol,
                preserve_keys=[
                    "last_entry_attempt_signal_candle_ts",
                    "entry_attempt_count_for_signal",
                ],
            )

        except Exception as e:
            logger.error(f"🚨 Error in exit logic for {symbol}: {e}")
            self.health_manager.record_error(e)

    def _process_entry(
        self,
        symbol: str,
        current_price: float,
        cached: dict,
        available_krw: float,
    ) -> float:
        """Entry-only path. Returns KRW consumed (0.0 if no entry)."""
        try:
            if symbol not in self.params_map or symbol not in self.strategies:
                return 0.0
            params = self.params_map[symbol]
            indicator_tf = str(params.get("INDICATOR_TIMEFRAME", "4h"))
            tf_min = self._timeframe_to_minutes(indicator_tf)

            pos = self._get_current_position(symbol, current_price)
            if float(pos["amount"]) > 0:
                return 0.0

            trend_dir = int(cached.get("trend_direction", 0))
            entry_upper = float(cached.get("entry_upper", 999999.0))
            strength_ok = int(cached.get("strength_filter", 0)) == 1
            regime_gate = float(cached.get("regime_entry_gate", 0.0))
            regime_risk_mult = float(cached.get("regime_risk_mult", 1.0))
            garch_kelly_f = float(cached.get("garch_kelly_f", 1.0))
            atr = float(cached.get("atr", 0.0))
            regime_state_val = float(cached.get("regime_state", 2.0))

            now_ms = self._get_reference_now_ms()
            signal_candle_ts = int(cached.get("signal_candle_ts", 0) or 0)
            interval_ms = tf_min * 60 * 1000
            if (
                signal_candle_ts > 0
                and tf_min > 0
                and now_ms >= signal_candle_ts + (2 * interval_ms)
            ):
                return 0.0

            state = self.state_manager.get_symbol_state(symbol)
            last_signal_ts = int(state.get("last_entry_attempt_signal_candle_ts", 0) or 0)
            attempt_count = int(state.get("entry_attempt_count_for_signal", 0) or 0)
            if last_signal_ts != signal_candle_ts:
                attempt_count = 0
            if attempt_count >= int(SPOT_ENTRY_SIGNAL_RETRY_MAX):
                return 0.0

            if not strength_ok or trend_dir != 1:
                return 0.0

            pullback_next_open = entry_upper < 1.0
            breakout_ok = pullback_next_open or (current_price > entry_upper)
            if not breakout_ok:
                return 0.0

            if regime_gate < 0.5:
                self._log_throttled(
                    "debug",
                    f"{symbol}:regime",
                    f"🚫 [{symbol}] Regime gate OFF ({regime_gate:.2f}). Entry blocked.",
                    600.0,
                )
                return 0.0

            if atr <= 0.0:
                return 0.0

            delta_gate = float(params.get("DELTA_GATE", 0.08))
            rr = max(
                0.05,
                min(
                    1.0,
                    float(regime_risk_mult) if np.isfinite(regime_risk_mult) else 1.0,
                ),
            )
            gk = float(garch_kelly_f) if np.isfinite(garch_kelly_f) and garch_kelly_f > 0.0 else 1.0
            eff_preview = rr * gk
            eff_preview = max(0.05, min(1.0, eff_preview))
            if 0.5 < regime_state_val < 1.5:
                eff_preview *= 0.90
            new_risk_pct = float(params.get("RISK_PER_TRADE", 0.05)) * eff_preview
            last_risk_pct = self._last_entry_risk_pct.get(symbol, 0.0)
            if last_risk_pct > 1e-12 and abs(new_risk_pct - last_risk_pct) < delta_gate:
                return 0.0

            fill_price_est = current_price * (1.0 + SLIPPAGE_RATE)
            qty = self._calculate_spot_position_size(
                fill_price=fill_price_est,
                entry_atr=atr,
                params=params,
                balance_krw=available_krw,
                regime_risk_mult=regime_risk_mult,
                garch_kelly_f=garch_kelly_f,
                regime_state=regime_state_val,
            )
            if qty <= 0.0:
                return 0.0

            tag = "Pullback (signal bar open)" if pullback_next_open else "Breakout"
            logger.info(
                f"🟢 [{symbol}] {tag} | Regime={regime_gate:.2f} | Kellyx={garch_kelly_f:.2f} | "
                f"Buying {qty:.4f} @ {current_price:,.0f}"
            )

            cid = "RT_SPT_" + uuid.uuid4().hex[:12]
            next_attempt = attempt_count + 1
            self.state_manager.update_symbol_state(
                symbol,
                {
                    "last_entry_attempt_signal_candle_ts": signal_candle_ts,
                    "entry_attempt_count_for_signal": next_attempt,
                },
            )

            order = self._place_order_safe(symbol, "buy", qty, client_order_id=cid)
            if not order:
                return 0.0

            expected_qty = float(order.get("amount", qty))
            confirmed, actual_fill_qty = self._confirm_buy_fill(symbol, expected_qty)
            if not confirmed:
                logger.warning(
                    "⚠️ [%s] Buy fill not confirmed (expected ~%.8f).",
                    symbol,
                    expected_qty,
                )
                return 0.0

            self._enforce_min_fill_ratio(symbol, qty, actual_fill_qty, params)

            fill_price = float(
                order.get("price", current_price * (1.0 + SLIPPAGE_RATE))
                or current_price * (1.0 + SLIPPAGE_RATE)
            )
            long_atr_mult = float(params.get("LONG_ATR_MULT", 3.0))
            initial_stop = fill_price - (atr * long_atr_mult)
            self.state_manager.update_symbol_state(
                symbol,
                {
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "entry_price": fill_price,
                    "entry_atr": atr,
                    "side": "LONG",
                    "highest_price": fill_price,
                    "active_stop_price": initial_stop,
                    "order_cid": cid,
                    "initial_amount": float(actual_fill_qty),
                    "fractal_scale_done": False,
                    "exit_pending": False,
                    "exit_error": None,
                    "exit_remaining_amount": 0.0,
                    "exit_recovery_attempt_count": 0,
                    "exit_attempt_at": None,
                },
            )
            self._last_entry_risk_pct[symbol] = new_risk_pct
            with self._db_write_lock:
                self.trade_db.record_trade(
                    symbol=symbol,
                    side="LONG",
                    action="ENTRY",
                    quantity=actual_fill_qty,
                    price=fill_price,
                    entry_price=None,
                    pnl=None,
                    pnl_pct=None,
                    reason=tag,
                    params={"cid": cid},
                )

            krw_used = float(actual_fill_qty * fill_price * (1.0 + TRADING_FEE_RATE))
            return krw_used

        except Exception as e:
            logger.error(f"🚨 Error in entry logic for {symbol}: {e}")
            self.health_manager.record_error(e)
            return 0.0

    def _reconcile_positions_on_startup(self) -> None:
        """Validate exchange vs local state on startup."""
        logger.info("🔍 Reconciling exchange positions vs local state...")
        target_set = set(SPOT_SYMBOLS)
        for sym in self.state_manager.list_all_symbols():
            if sym not in target_set:
                logger.warning("⚠️ Dropping state for non-target symbol: %s", sym)
                self._clear_symbol_state_tracked(sym)

        all_ohlcv = self._prefetch_all_ohlcv()
        for symbol in self.symbols:
            try:
                price = self._get_market_price_safe(symbol)
            except Exception as e:
                logger.warning("[%s] Reconcile skipped (price): %s", symbol, e)
                continue
            try:
                pos = self._get_current_position(symbol, price)
                state = self.state_manager.get_symbol_state(symbol)
                raw_amt = self._get_spot_coin_amount_raw(symbol)
                amt = float(pos.get("amount", 0.0) or 0.0)

                if bool(state.get("exit_pending", False)):
                    val = raw_amt * price if price > 0 else 0.0
                    if raw_amt < 1e-12 or val < float(MIN_POSITION_VALUE_KRW):
                        self._clear_symbol_state_tracked(
                            symbol,
                            preserve_keys=[
                                "last_entry_attempt_signal_candle_ts",
                                "entry_attempt_count_for_signal",
                            ],
                        )
                        state = self.state_manager.get_symbol_state(symbol)

                if amt > 0 and not state.get("entry_price"):
                    cached = self._refresh_indicators_if_needed(symbol, all_ohlcv)
                    atr = float(cached.get("atr", 0.0))
                    p = self.params_map.get(symbol, {})
                    self._bootstrap_state_for_open_position(
                        symbol=symbol,
                        amount=amt,
                        pos=pos,
                        current_price=price,
                        params=p,
                        atr=atr,
                    )

                if (
                    amt <= 0
                    and bool(state.get("entry_price"))
                    and not bool(state.get("exit_pending", False))
                ):
                    logger.warning(
                        "🧹 [%s] Reconcile: clearing stale local state (no exchange position)",
                        symbol,
                    )
                    self._clear_symbol_state_tracked(
                        symbol,
                        preserve_keys=[
                            "last_entry_attempt_signal_candle_ts",
                            "entry_attempt_count_for_signal",
                        ],
                    )
            except Exception as ex:
                logger.error("[%s] Reconcile error: %s", symbol, ex)

    def _get_positions_snapshot(self) -> dict:
        """Lightweight positions summary for heartbeat."""
        out: Dict[str, dict] = {}
        for symbol in self.symbols:
            try:
                price = self._get_market_price_safe(symbol)
                st = self.state_manager.get_symbol_state(symbol)
                ep = float(st.get("entry_price", 0.0) or 0.0)
                sp = float(st.get("active_stop_price", 0.0) or 0.0)
                pos = self._get_current_position(symbol, price)
                out[symbol] = {
                    "amount": float(pos.get("amount", 0.0) or 0.0),
                    "entry_price": ep,
                    "active_stop_price": sp,
                }
            except Exception:
                out[symbol] = {}
        return out

    def _shutdown(self) -> None:
        """Flush state and record shutdown heartbeat."""
        try:
            self.state_manager.flush_now()
        except Exception as e:
            logger.warning("State flush on shutdown failed: %s", e)

        open_positions = {}
        for symbol in self.symbols:
            try:
                price = self._get_market_price_safe(symbol)
                pos = self._get_current_position(symbol, price)
                amount = float(pos.get("amount", 0.0) or 0.0)
                if amount > 0:
                    open_positions[symbol] = amount
                    logger.warning(
                        "⚠️ Open position at shutdown: %s amount=%s",
                        symbol,
                        amount,
                    )
            except Exception as e:
                logger.debug("Shutdown position check [%s]: %s", symbol, e)

        try:
            self.health_manager.update_heartbeat(
                status="stopped",
                positions=open_positions,
                extra={"shutdown_time": datetime.now(timezone.utc).isoformat()},
            )
        except Exception as e:
            logger.warning("Heartbeat on shutdown failed: %s", e)

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        try:
            self.initialize()
            logger.info(f"▶️ Starting Main Loop (Interval: {SPOT_LOOP_INTERVAL_SECONDS}s)")

            while not self._shutdown_requested:
                cycle_start = time.time()
                self._loop_count += 1
                try:
                    all_ohlcv = self._prefetch_all_ohlcv()

                    for symbol in self.symbols:
                        if self._shutdown_requested:
                            break
                        try:
                            self._refresh_indicators_if_needed(symbol, all_ohlcv)
                        except Exception as sym_e:
                            logger.error(f"🚨 [{symbol}] Indicator refresh error: {sym_e}")
                        time.sleep(SPOT_SYMBOL_DELAY_SECONDS)

                    for symbol in self.symbols:
                        if self._shutdown_requested:
                            break
                        try:
                            price = self._get_market_price_safe(symbol)
                            if price is None or not np.isfinite(float(price)) or float(price) <= 0:
                                continue
                            cached = self._get_cached_indicators(symbol)
                            self._process_exit(symbol, float(price), cached)
                        except Exception as sym_e:
                            logger.error(f"🚨 [{symbol}] Exit processing error: {sym_e}")
                            self.health_manager.record_error(sym_e)
                        time.sleep(SPOT_SYMBOL_DELAY_SECONDS)

                    _, available_krw = self._fetch_balance_safe()
                    for symbol in self.symbols:
                        if self._shutdown_requested:
                            break
                        try:
                            price = self._get_market_price_safe(symbol)
                            if price is None or not np.isfinite(float(price)) or float(price) <= 0:
                                continue
                            cached = self._get_cached_indicators(symbol)
                            used = self._process_entry(symbol, float(price), cached, available_krw)
                            available_krw -= used
                        except Exception as sym_e:
                            logger.error(f"🚨 [{symbol}] Entry processing error: {sym_e}")
                            self.health_manager.record_error(sym_e)
                        time.sleep(SPOT_SYMBOL_DELAY_SECONDS)

                    snap = self._get_positions_snapshot()
                    self.health_manager.update_heartbeat(status="running", positions=snap)

                    now_gc = time.time()
                    if now_gc - self._last_gc_time >= 1800.0:
                        gc.collect()
                        self._last_gc_time = now_gc

                except Exception as loop_e:
                    logger.error(f"💥 Unexpected error in Main Loop cycle: {loop_e}")
                    time.sleep(ERROR_SLEEP_SECONDS)

                elapsed = time.time() - cycle_start
                sleep_time = max(0.5, SPOT_LOOP_INTERVAL_SECONDS - elapsed)

                start_wait = time.time()
                while time.time() - start_wait < sleep_time:
                    if self._shutdown_requested:
                        break
                    time.sleep(0.5)

        except Exception as e:
            logger.critical(f"💥 Critical Failure during Initialization/Run: {e}")
            try:
                self.health_manager.update_heartbeat(status="error")
            except Exception:
                ...
        finally:
            self._shutdown()
            logger.info("🛑 Bot Stopped.")


if __name__ == "__main__":
    bot = SpotBot()
    bot.run()
