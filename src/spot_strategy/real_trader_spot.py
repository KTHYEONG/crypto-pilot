"""
RealTrader Spot - production-grade spot trading bot (Upbit).
Core improvements:
- trade history persistence
- API retry wrapper
- health-check heartbeat
- graceful shutdown
- duplicate-logic cleanup
- settings-based constants
- candle close sync
"""

import os
import sys
import time
import signal
import json
import logging
import gc
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

# tenacity for retry logic


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
    TRADE_HISTORY_DB,
    SPOT_HEARTBEAT_FILE,
    SPOT_STATE_FILE,
    API_RETRY_ATTEMPTS,
    API_RETRY_WAIT_MIN,
    API_RETRY_WAIT_MAX,
    MIN_POSITION_VALUE_KRW,
    MIN_ORDER_VALUE_KRW,
    MAX_INVEST_CAP_KRW,
    SPOT_LOOP_INTERVAL_SECONDS,
    SPOT_SYMBOL_DELAY_SECONDS,
    ERROR_SLEEP_SECONDS,
    CANDLE_SYNC_OFFSET_SECONDS,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    SPOT_TARGET_SYMBOLS,
    SPOT_OPTUNA_STUDY_NAME,
    SPOT_ALLOCATION_WEIGHTS,
    MAX_TOTAL_BALANCE_KRW,
)

# Shared utilities/components
from src.common.utils import setup_logger, api_retry
from src.common.components import (
    TradeHistoryDB, 
    HealthCheckManager, 
    calculate_candle_wait_time
)

# Upbit client
from src.spot_strategy.upbit_client import UpbitClient
from src.strategy.strategies import UltimateStrategy

# Optional cloud optimizer
try:
    from src.common.cloud_optimizer import CloudOptimizer
    CLOUD_OPTIMIZER_AVAILABLE = True
except ImportError:
    CLOUD_OPTIMIZER_AVAILABLE = False

logger = setup_logger("RealTraderSpot")
EXPECTED_SPOT_POLICY_VERSION = os.getenv("SPOT_POLICY_VERSION", "SPOT_SELECTION_POLICY_V1")
SPOT_ALLOWED_MODES = {"SCALP", "DAY", "SWING", "UNIFIED", "ALL"}
SPOT_ALLOW_FALLBACK = os.getenv("SPOT_ALLOW_FALLBACK", "false").lower() == "true"


# ============================================================
# State Manager (JSON file)
# ============================================================
class StateManager:
    """Trade state manager (entry price, ATR snapshot, etc.)."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create state file if absent."""
        if not self.state_file.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self._save({})

    def _load(self) -> dict:
        """Load state dictionary."""
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"State load error: {e}")
            return {}

    def _save(self, state: dict):
        """Persist state dictionary."""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"State save error: {e}")

    def get_symbol_state(self, symbol: str) -> dict:
        """Get state for a symbol."""
        state = self._load()
        return state.get(symbol, {})

    def update_symbol_state(self, symbol: str, data: dict):
        """Update state for a symbol."""
        state = self._load()
        if symbol not in state:
            state[symbol] = {}
        state[symbol].update(data)
        self._save(state)

    def clear_symbol_state(self, symbol: str):
        """Clear state for a symbol."""
        state = self._load()
        if symbol in state:
            state[symbol] = {}
            self._save(state)


# ============================================================
# Main Trader Class
# ============================================================
class RealTraderSpot:
    """Production-grade spot trading bot (Upbit)."""

    def __init__(
        self,
        db_path: str = None,
        enable_oracle_optimization: bool = False
    ):
        self.client = UpbitClient(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        self.db_path = db_path or str(SPOT_STRATEGY_DB)
        self.symbols = SPOT_TARGET_SYMBOLS.copy()

        # Core components
        self.trade_db = TradeHistoryDB(TRADE_HISTORY_DB)
        self.health_manager = HealthCheckManager(SPOT_HEARTBEAT_FILE)
        self.state_manager = StateManager(SPOT_STATE_FILE)

        # Cloud optimizer (optional)
        self.cloud_optimizer = None
        if enable_oracle_optimization and CLOUD_OPTIMIZER_AVAILABLE:
            self.cloud_optimizer = CloudOptimizer()
            logger.info("Cloud optimization enabled")

        # Shutdown flag
        self._shutdown_requested = False

        # Signal handlers
        self._setup_signal_handlers()

        # Strategy maps
        self.params_map: Dict[str, dict] = {}
        self.strategies: Dict[str, UltimateStrategy] = {}
        self.strategy_meta_map: Dict[str, dict] = {}
        # Cache for duplicate signal calculation prevention
        self.last_calc_candle: Dict[str, str] = {}
        
        # Cloud maintenance timestamps
        self._last_resource_check = datetime.utcnow()
        self._last_db_cleanup = datetime.utcnow()
        self._last_gc = datetime.utcnow()
        
        self.load_strategies_from_db()

    @staticmethod
    def _to_decimal(value: Any, default: str = "0") -> Decimal:
        try:
            if isinstance(value, Decimal):
                return value
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return Decimal(default)

    @staticmethod
    def _to_krw(value: Any) -> Decimal:
        dec = RealTraderSpot._to_decimal(value)
        return dec.quantize(Decimal("1"), rounding=ROUND_DOWN)

    def _validate_deployment_metadata(self) -> None:
        errors = []
        for symbol in self.symbols:
            meta = self.strategy_meta_map.get(symbol, {}) or {}
            policy_version = str(meta.get("policy_version", "")).strip()
            selected_mode = str(meta.get("selected_mode", "")).strip().upper()

            if policy_version == "DEFAULT_FALLBACK":
                if SPOT_ALLOW_FALLBACK:
                    logger.warning(f"[{symbol}] Fallback policy allowed by SPOT_ALLOW_FALLBACK=true")
                    continue
                errors.append(f"{symbol}: fallback policy active")
                continue

            if policy_version != EXPECTED_SPOT_POLICY_VERSION:
                errors.append(
                    f"{symbol}: policy_version='{policy_version}' != '{EXPECTED_SPOT_POLICY_VERSION}'"
                )
            if selected_mode not in SPOT_ALLOWED_MODES:
                errors.append(f"{symbol}: invalid selected_mode='{selected_mode}'")

        if errors:
            raise RuntimeError("Deployment metadata guard failed: " + " | ".join(errors))

    def _setup_signal_handlers(self):
        """Register graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}. Initiating graceful shutdown.")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

    def load_strategies_from_db(self):
        """Load optimized params from Optuna DB."""
        logger.info(f"Loading strategies from {self.db_path}...")

        if not os.path.exists(self.db_path):
            logger.warning(f"DB file not found: {self.db_path}, using defaults")
            self._use_default_params()
            return

        try:
            # [Lazy Loading]
            import optuna
            storage = f"sqlite:///{self.db_path}"
            study = optuna.load_study(study_name=SPOT_OPTUNA_STUDY_NAME, storage=storage)
            logger.info(f"Loaded study '{SPOT_OPTUNA_STUDY_NAME}' (score={study.best_value:.4f})")

            best_params = study.best_params
            try:
                best_trial = study.best_trial
                user_meta = dict(getattr(best_trial, "user_attrs", {}) or {})
            except Exception:
                user_meta = {}

            for symbol in self.symbols:
                self.params_map[symbol] = best_params.copy()
                strategy_name = f"RealSpot_{symbol.replace('KRW-', '')}"
                self.strategies[symbol] = UltimateStrategy(strategy_name, best_params)
                self.strategy_meta_map[symbol] = user_meta.copy()
                logger.info(f"Strategy initialized: {symbol} | TF: {best_params.get('TIMEFRAME')}")
                if user_meta:
                    logger.info(
                        f"   policy={user_meta.get('policy_version', 'n/a')} "
                        f"| selected_mode={user_meta.get('selected_mode', 'n/a')}"
                    )

        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
            logger.warning("Using default fallback parameters.")
            self._use_default_params()

    def _use_default_params(self):
        """Use fallback default params."""
        default_params = {
            'TIMEFRAME': '1h',
            'ENTRY_TYPE': 'BOLLINGER',
            'ATR_PERIOD': 14,
            'STRENGTH_FILTER_PERIOD': 14,
            'EXIT_TYPE': 'ATR',
            'STOP_LOSS_TYPE': 'FIXED',
            'STOP_LOSS_PCT': 0.02,
            'RISK_PER_TRADE': 0.02,
            'ATR_MULTIPLIER': 3.0,
            'ATR_STOP_LOSS_MULT': 1.0,
            'USE_TAKE_PROFIT': False,
            'TAKE_PROFIT_ATR_MULT': 2.0,
        }

        for symbol in self.symbols:
            self.params_map[symbol] = default_params.copy()
            strategy_name = f"RealSpot_{symbol.replace('KRW-', '')}"
            self.strategies[symbol] = UltimateStrategy(strategy_name, default_params)
            self.strategy_meta_map[symbol] = {"policy_version": "DEFAULT_FALLBACK"}
            logger.info(f"Default strategy for {symbol}")

    @api_retry
    def _fetch_ohlcv_safe(self, symbol: str, timeframe: str, limit: int):
        """Fetch OHLCV with retry."""
        return self.client.fetch_ohlcv(symbol, timeframe, limit=limit)

    @api_retry
    def _fetch_daily_ohlcv_safe(self, symbol: str, limit: int = 260):
        """Fetch daily OHLCV for MTF filter with retry."""
        return self.client.fetch_ohlcv(symbol, "1d", limit=limit)

    @api_retry
    def _fetch_position_safe(self, symbol: str) -> dict:
        """Fetch position with retry."""
        return self.client.fetch_position(symbol)

    @api_retry
    def _fetch_balance_safe(self) -> tuple:
        """Fetch balance with retry."""
        return self.client.fetch_balance()

    @api_retry
    def _get_market_price_safe(self, symbol: str) -> float:
        """Fetch market price with retry."""
        price = self.client.get_market_price(symbol)
        if price is None:
            raise ValueError(f"Failed to fetch market price for {symbol}")
        return price

    @api_retry
    def _place_order_safe(self, symbol: str, side: str, **kwargs):
        """Place order with retry and minimum-order handling."""
        return self.client.place_order_smart(symbol, side, **kwargs)

    def _build_mtf_confirmed_candle(
        self,
        strategy: UltimateStrategy,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
    ) -> Optional[dict]:
        """
        Build live MTF-confirmed candle aligned with backtest semantics:
        - generate hourly/daily signals
        - map shifted daily trend onto hourly bars by as-of backward join
        - final trend = (hourly trend == 1) and (daily trend == 1)
        - return confirmed bar fields from i-1 (iloc[-2])
        """
        if hourly_df is None or daily_df is None:
            return None
        if len(hourly_df) < 220 or len(daily_df) < 40:
            return None

        hourly_sig = strategy.generate_signals(hourly_df.copy())
        daily_sig = strategy.generate_signals(daily_df.copy())

        if "trend_direction" not in hourly_sig.columns or "trend_direction" not in daily_sig.columns:
            return None
        if len(hourly_sig) < 3:
            return None

        hourly_days = pd.to_datetime(hourly_sig["datetime"]).dt.normalize().values.astype("datetime64[ns]")
        daily_days = pd.to_datetime(daily_sig["datetime"]).dt.normalize().values.astype("datetime64[ns]")
        if len(daily_days) == 0:
            return None

        daily_trend_shifted = daily_sig["trend_direction"].shift(1).fillna(0).values
        pos = np.searchsorted(daily_days, hourly_days, side="right") - 1
        pos = np.clip(pos, 0, len(daily_days) - 1).astype(np.int32)
        mapped_daily_trend = daily_trend_shifted[pos]

        hourly_trend = np.nan_to_num(hourly_sig["trend_direction"].values, nan=0).astype(int)
        aligned_trend = np.where((hourly_trend == 1) & (mapped_daily_trend == 1), 1, 0)
        hourly_sig["trend_direction"] = aligned_trend
        hourly_sig["daily_trend_direction"] = mapped_daily_trend

        confirmed = hourly_sig.iloc[-2]
        return {
            "signal_time": pd.Timestamp(confirmed["datetime"]).isoformat(),
            "signal_close": float(confirmed.get("close", np.nan)),
            "entry_upper": float(confirmed.get("entry_upper", np.nan)),
            "entry_lower": float(confirmed.get("entry_lower", np.nan)),
            "trend_direction": int(confirmed.get("trend_direction", 0)) if not pd.isna(confirmed.get("trend_direction", 0)) else 0,
            "daily_trend_direction": int(confirmed.get("daily_trend_direction", 0)) if not pd.isna(confirmed.get("daily_trend_direction", 0)) else 0,
            "strength_filter": int(confirmed.get("strength_filter", 0)) if not pd.isna(confirmed.get("strength_filter", 0)) else 0,
            "volume_ratio": float(confirmed.get("volume_ratio", np.nan)),
            "atr": float(confirmed.get("atr", 0.0)) if not pd.isna(confirmed.get("atr", 0.0)) else 0.0,
            "parabolic_sar": float(confirmed.get("parabolic_sar", 0.0)) if not pd.isna(confirmed.get("parabolic_sar", 0.0)) else 0.0,
            "rsi": float(confirmed.get("rsi", 50.0)) if not pd.isna(confirmed.get("rsi", 50.0)) else 50.0,
            "hurst": float(confirmed.get("hurst", 0.5)) if not pd.isna(confirmed.get("hurst", 0.5)) else 0.5,
            "natr": float(confirmed.get("natr", 0.0)) if not pd.isna(confirmed.get("natr", 0.0)) else 0.0,
        }

    def initialize(self):
        """Initialize bot and verify deployment metadata."""
        logger.info("RealTrader Spot bot initializing...")

        try:
            self._validate_deployment_metadata()
            logger.info("Deployment metadata guard passed.")
            total_krw, free_krw = self._fetch_balance_safe()
            logger.info(f"Account balance: total={total_krw:,.0f} KRW | free={free_krw:,.0f} KRW")

            if free_krw < MIN_ORDER_VALUE_KRW:
                logger.warning(f"Low balance (< {MIN_ORDER_VALUE_KRW:,} KRW)")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

        # Exchange Client Status Check
        if not self.client.exchange:
            logger.error("Upbit client is not initialized. Check API keys and network.")


        # Initial heartbeat
        self.health_manager.update_heartbeat(status="initialized")

        logger.info("Initialization complete. Bot is running.")

    def execute_logic(self, symbol: str):
        """Run per-symbol logic (confirmed signal at i-1, execution near i open)."""
        try:
            params = self.params_map[symbol]
            strategy = self.strategies[symbol]
            timeframe = params.get("TIMEFRAME", "1h")
            hourly_df = None
            daily_df = None

            pos = self._fetch_position_safe(symbol)
            balance_coin = pos.get("amount", 0.0)

            state = self.state_manager.get_symbol_state(symbol)
            entry_price = state.get("entry_price", 0.0)

            current_slot = self._get_candle_slot_id(timeframe)
            already_calculated = self.last_calc_candle.get(symbol) == current_slot
            is_entry_time = self._is_entry_time(timeframe)

            cached = self._get_cached_indicators(symbol)
            need_calculation = (not cached) or (is_entry_time and not already_calculated)
            if need_calculation:
                logger.info(f"[{symbol}] Signal refresh for slot={current_slot}")

                hourly_df = self._fetch_ohlcv_safe(symbol, timeframe, limit=600)
                daily_df = self._fetch_daily_ohlcv_safe(symbol, limit=300)

                if hourly_df is None or len(hourly_df) < 220:
                    logger.warning(
                        f"Insufficient hourly data for {symbol}: "
                        f"{len(hourly_df) if hourly_df is not None else 0} < 220"
                    )
                    self.last_calc_candle[symbol] = current_slot
                    return
                if daily_df is None or len(daily_df) < 40:
                    logger.warning(
                        f"Insufficient daily data for {symbol}: "
                        f"{len(daily_df) if daily_df is not None else 0} < 40"
                    )
                    self.last_calc_candle[symbol] = current_slot
                    return

                float_cols = ["open", "high", "low", "close", "volume"]
                hourly_df[float_cols] = hourly_df[float_cols].astype(np.float64)
                daily_df[float_cols] = daily_df[float_cols].astype(np.float64)

                confirmed = self._build_mtf_confirmed_candle(strategy, hourly_df, daily_df)
                if not confirmed:
                    logger.warning(f"Failed to build MTF confirmed signal for {symbol}")
                    self.last_calc_candle[symbol] = current_slot
                    return

                self._cache_indicators(
                    symbol,
                    {
                        "trend_direction": int(confirmed.get("trend_direction", 0)),
                        "daily_trend_direction": int(confirmed.get("daily_trend_direction", 0)),
                        "atr": float(confirmed.get("atr", 0.0)),
                        "parabolic_sar": float(confirmed.get("parabolic_sar", 0.0)),
                        "entry_upper": float(confirmed.get("entry_upper")) if np.isfinite(confirmed.get("entry_upper", np.nan)) else None,
                        "entry_lower": float(confirmed.get("entry_lower")) if np.isfinite(confirmed.get("entry_lower", np.nan)) else None,
                        "strength_filter": int(confirmed.get("strength_filter", 0)),
                        "volume_ratio": float(confirmed.get("volume_ratio", np.nan)),
                        "rsi": float(confirmed.get("rsi", 50.0)),
                        "hurst": float(confirmed.get("hurst", 0.5)),
                        "natr": float(confirmed.get("natr", 0.0)),
                        "signal_close": float(confirmed.get("signal_close", np.nan)),
                        "signal_time": confirmed.get("signal_time"),
                        "signal_slot": current_slot,
                        "cached_at": datetime.utcnow().isoformat(),
                    },
                )
                self.last_calc_candle[symbol] = current_slot
                cached = self._get_cached_indicators(symbol)
                logger.info(
                    f"[{symbol}] Cached signal: trend={cached.get('trend_direction')} "
                    f"(daily={cached.get('daily_trend_direction')}), close={cached.get('signal_close')}"
                )

            trend_dir = cached.get("trend_direction", 0)
            atr = cached.get("atr", 0.0)
            sar = cached.get("parabolic_sar", 0.0)

            last_price = self._get_market_price_safe(symbol)
            if last_price is None:
                logger.warning(f"[{symbol}] Could not fetch market price. Skipping cycle.")
                return

            current_value_d = self._to_decimal(balance_coin) * self._to_decimal(last_price)
            in_position = current_value_d > self._to_decimal(MIN_POSITION_VALUE_KRW)

            if in_position:
                logger.info(f"[{symbol}] Position exists ({float(current_value_d):,.0f} KRW). Checking exit...")
                mock_candle = {
                    "atr": atr,
                    "parabolic_sar": sar,
                    "trend_direction": trend_dir,
                    "rsi": cached.get("rsi", 50.0),
                }

                if entry_price == 0:
                    exchange_avg_price = pos.get("entryPrice", 0.0)
                    if exchange_avg_price > 0:
                        logger.warning(f"[{symbol}] Recovered entry_price from exchange: {exchange_avg_price}")
                        entry_price = exchange_avg_price
                        state.update(
                            {
                                "entry_price": entry_price,
                                "highest_high": max(last_price, exchange_avg_price),
                                "invest_amount": balance_coin * exchange_avg_price,
                            }
                        )
                        self.state_manager.update_symbol_state(symbol, state)

                self._check_exit(symbol, balance_coin, last_price, params, mock_candle, state)

            elif not in_position and is_entry_time:
                logger.info(f"[{symbol}] Entry window open. Checking entry conditions...")

                state = self.state_manager.get_symbol_state(symbol)
                last_entry_str = state.get("entry_time")
                if last_entry_str:
                    last_entry_dt = datetime.fromisoformat(last_entry_str)
                    if (datetime.utcnow() - last_entry_dt).total_seconds() < 180:
                        logger.info(f"[{symbol}] Recent entry detected ({last_entry_str}). Skip duplicate entry.")
                        return

                open_orders = self.client.fetch_open_orders(symbol)
                if open_orders and len(open_orders) > 0:
                    logger.warning(f"[{symbol}] Open orders exist ({len(open_orders)}). Skip entry.")
                    return

                entry_upper = cached.get("entry_upper")
                entry_lower = cached.get("entry_lower")
                if (
                    pd.isna(entry_upper)
                    or pd.isna(entry_lower)
                    or entry_upper is None
                    or entry_lower is None
                    or not np.isfinite(entry_upper)
                    or not np.isfinite(entry_lower)
                ):
                    logger.info(f"[{symbol}] Skip entry: invalid entry levels")
                    return

                self._check_entry(
                    symbol,
                    last_price,
                    params,
                    {
                        "entry_upper": entry_upper,
                        "entry_lower": entry_lower,
                        "atr": atr,
                        "trend_direction": trend_dir,
                        "strength_filter": cached.get("strength_filter", 1),
                        "volume_ratio": cached.get("volume_ratio", 100.0),
                        "hurst": cached.get("hurst", 0.5),
                        "natr": cached.get("natr", 0.0),
                        "signal_close": cached.get("signal_close", np.nan),
                        "signal_time": cached.get("signal_time"),
                        "signal_slot": cached.get("signal_slot"),
                    },
                )

            if need_calculation and hourly_df is not None:
                del hourly_df
                if daily_df is not None:
                    del daily_df
                gc.collect()

        except Exception as e:
            logger.error(f"Error executing logic for {symbol}: {e}")
            self.health_manager.record_error(e)

    def _check_exit(
        self,
        symbol: str,
        balance_coin: float,
        last_price: float,
        params: dict,
        candle: dict,
        state: dict
    ):
        """Exit logic for long-only spot positions."""
        try:
            should_sell = False
            reason = ""

            entry_price = state.get("entry_price", 0.0)
            entry_price_d = self._to_decimal(entry_price)
            last_price_d = self._to_decimal(last_price)
            balance_coin_d = self._to_decimal(balance_coin)

            # Fallback to current candle ATR if state is missing
            entry_atr = state.get("entry_atr", 0.0)
            if entry_atr == 0:
                entry_atr = candle.get("atr", 0.0)
                if entry_atr > 0:
                    logger.warning(f"Recovered entry_atr from current candle for {symbol}: {entry_atr}")
            entry_atr_d = self._to_decimal(entry_atr)

            highest_high_d = self._to_decimal(state.get("highest_high", entry_price))

            trend_dir = candle.get("trend_direction", 0)
            current_high_d = max(self._to_decimal(candle.get("high", last_price)), last_price_d)

            # Update highest high
            if current_high_d > highest_high_d:
                highest_high_d = current_high_d
                state["highest_high"] = float(highest_high_d)
                self.state_manager.update_symbol_state(symbol, state)

            exit_type = params.get("EXIT_TYPE", "ATR")
            atr_mult_d = self._to_decimal(params.get("ATR_MULTIPLIER", 3.0))
            sl_type = params.get("STOP_LOSS_TYPE", "FIXED")
            sl_pct_d = self._to_decimal(params.get("STOP_LOSS_PCT", 0.02))
            atr_sl_mult_d = self._to_decimal(params.get("ATR_STOP_LOSS_MULT", 1.0))
            use_tp = params.get("USE_TAKE_PROFIT", False)
            tp_mult_d = self._to_decimal(params.get("TAKE_PROFIT_ATR_MULT", 2.0))

            # 1) Stop loss
            if sl_type == "ATR" and entry_atr_d > 0:
                stop_price_d = entry_price_d - (entry_atr_d * atr_sl_mult_d)
            else:
                stop_price_d = entry_price_d * (Decimal("1") - sl_pct_d)

            if last_price_d < stop_price_d:
                should_sell = True
                reason = f"Stop Loss ({float(stop_price_d):,.0f})"

            # 2) Main exit (ATR trailing / SAR)
            if not should_sell:
                if exit_type == "ATR" and entry_atr_d > 0:
                    trailing_stop_d = highest_high_d - (entry_atr_d * atr_mult_d)
                    if last_price_d < trailing_stop_d:
                        should_sell = True
                        reason = f"ATR Trailing Stop ({float(trailing_stop_d):,.0f})"

                elif exit_type == "PARABOLIC_SAR":
                    p_sar = candle.get("parabolic_sar", 0)
                    if p_sar <= 0:
                        logger.warning(f"[{symbol}] SAR enabled but value invalid ({p_sar:.2f}).")
                    elif last_price_d < self._to_decimal(p_sar):
                        should_sell = True
                        reason = f"Parabolic SAR Exit ({p_sar:,.0f})"

            # 3) Trend reversal
            if not should_sell and trend_dir == -1:
                should_sell = True
                reason = "Trend Reversal"

            # 4) Take profit
            if not should_sell and use_tp and entry_atr_d > 0:
                target_price_d = entry_price_d + (entry_atr_d * tp_mult_d)
                if last_price_d > target_price_d:
                    should_sell = True
                    reason = f"Take Profit ({float(target_price_d):,.0f})"
            
            # 5) Panic exit (RSI)
            rsi = candle.get("rsi", 0)
            rsi_exit_thresh = params.get("RSI_EXIT_THRESHOLD", 93)
            if not should_sell and rsi > rsi_exit_thresh:
                should_sell = True
                reason = f"Panic Exit (RSI {rsi:.1f})"

            # 6) Time cut (profit check)
            max_holding_bars = params.get("MAX_HOLDING_BARS", 9999)
            entry_time_str = state.get("entry_time")
            if not should_sell and entry_time_str:
                entry_dt = datetime.fromisoformat(entry_time_str)
                tf = params.get("TIMEFRAME", "4h")
                interval_min = 240
                if tf.endswith("h"):
                    interval_min = int(tf[:-1]) * 60
                elif tf.endswith("m"):
                    interval_min = int(tf[:-1])
                
                elapsed_min = (datetime.utcnow() - entry_dt).total_seconds() / 60
                bars_held = elapsed_min / interval_min
                
                if bars_held >= max_holding_bars:
                    pnl_pct_current = float(((last_price_d / entry_price_d) - Decimal("1")) * Decimal("100")) if entry_price_d > 0 else 0.0
                    profit_thresh = params.get("TIME_EXIT_PROFIT_THRESHOLD", 1.4)
                    if pnl_pct_current <= profit_thresh:
                        should_sell = True
                        reason = f"Time Cut (Held {bars_held:.1f} bars, PnL {pnl_pct_current:.2f}%)"

            if should_sell:
                pnl_d = (last_price_d - entry_price_d) * balance_coin_d
                pnl_pct = float(((last_price_d / entry_price_d) - Decimal("1")) * Decimal("100")) if entry_price_d > 0 else 0.0

                logger.info(
                    f"EXIT {symbol} | Price: {last_price:,.0f} | "
                    f"PnL: {pnl_pct:+.2f}% ({float(pnl_d):,.0f} KRW) | Reason: {reason}"
                )

                res = self._place_order_safe(symbol, "sell", amount=float(balance_coin_d))

                if res and "uuid" in res:
                    self.trade_db.record_trade(
                        symbol=symbol,
                        side="LONG",
                        action="EXIT",
                        quantity=float(balance_coin_d),
                        price=float(last_price_d),
                        entry_price=float(entry_price_d),
                        pnl=float(pnl_d),
                        pnl_pct=pnl_pct,
                        reason=reason,
                    )
                    self.state_manager.clear_symbol_state(symbol)
                else:
                    logger.error(f"Sell order failed: {res}")

        except Exception as e:
            logger.error(f"Error in _check_exit: {e}")
            self.health_manager.record_error(e)

    def _calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        params: dict,
        hurst: float = 0.5,
        natr: float = 0.0
    ) -> float:
        """Calculate KRW position size with Decimal precision."""
        try:
            total_krw, free_krw = self._fetch_balance_safe()
            total_krw_d = self._to_decimal(total_krw)
            free_krw_d = self._to_decimal(free_krw)
            min_position_d = self._to_decimal(MIN_POSITION_VALUE_KRW)
            min_order_d = self._to_decimal(MIN_ORDER_VALUE_KRW)
            max_cap_d = self._to_decimal(MAX_INVEST_CAP_KRW)
            current_invested_total_d = Decimal("0")
            this_symbol_invested_d = Decimal("0")
            
            state_map = self.state_manager._load()
            for s in self.symbols:
                st = state_map.get(s, {})
                invest_amt_d = self._to_decimal(st.get("invest_amount", 0))
                current_invested_total_d += invest_amt_d
                
                if s == symbol:
                    this_symbol_invested_d = invest_amt_d
            
            # Single-entry enforcement for spot long-only
            if this_symbol_invested_d > min_position_d:
                logger.info(
                    f"Skipping entry for {symbol}: position already exists "
                    f"({float(this_symbol_invested_d):,.0f} KRW)."
                )
                return 0

            estimated_total_equity_d = total_krw_d + current_invested_total_d
            
            default_weight_d = (Decimal("1") / Decimal(str(len(self.symbols)))) if self.symbols else Decimal("0.5")
            base_weight_d = self._to_decimal(SPOT_ALLOCATION_WEIGHTS.get(symbol, float(default_weight_d)))
            
            regime_mult_d = Decimal("1")
            
            strong_hurst = params.get('STRONG_REGIME_HURST', 0.56)
            panic_natr = params.get('PANIC_REGIME_NATR', 9.5)
            strong_mult_d = self._to_decimal(params.get('STRONG_REGIME_MULTIPLIER', 1.6))
            panic_mult_d = self._to_decimal(params.get('PANIC_REGIME_MULTIPLIER', 0.35))
            
            if hurst > strong_hurst:
                regime_mult_d = strong_mult_d
                logger.info(f"Strong regime detected for {symbol} (Hurst {hurst:.2f}). Mult: {float(strong_mult_d):.2f}")
                
            if natr > panic_natr:
                regime_mult_d = panic_mult_d
                logger.info(f"Panic regime detected for {symbol} (NATR {natr:.2f}). Mult: {float(panic_mult_d):.2f}")
            
            final_weight_d = base_weight_d * regime_mult_d
            
            target_amount_d = estimated_total_equity_d * final_weight_d
            
            buy_amount_d = target_amount_d - this_symbol_invested_d
            
            buy_amount_d = min(buy_amount_d, free_krw_d)
            buy_amount_d = min(buy_amount_d, max_cap_d)
            buy_amount_d = self._to_krw(buy_amount_d)
            
            if buy_amount_d < min_order_d:
                logger.warning(
                    f"Calculated buy_amount too small for {symbol}: {float(buy_amount_d):,.0f} KRW "
                    f"< Min {MIN_ORDER_VALUE_KRW:,.0f} KRW. "
                    f"Equity: {float(estimated_total_equity_d):,.0f} KRW, Weight: {float(final_weight_d*Decimal('100')):.0f}%"
                )
                return 0
                
            logger.info(
                f"Sizing {symbol} (Weight {float(final_weight_d*Decimal('100')):.0f}%): "
                f"Equity {float(estimated_total_equity_d):,.0f} KRW | "
                f"Target {float(target_amount_d):,.0f} | Buy {float(buy_amount_d):,.0f}"
            )
            
            return float(buy_amount_d)
            
        except Exception as e:
            logger.error(f"Sizing Error: {e}")
            return 0

    def _check_entry(
        self,
        symbol: str,
        last_price: float,
        params: dict,
        candle: dict
    ):
        """Entry logic based on confirmed candle signal."""
        try:
            entry_upper = candle.get("entry_upper", 0.0)
            if pd.isna(entry_upper):
                entry_upper = 0.0

            signal_close = candle.get("signal_close", np.nan)
            signal_slot = candle.get("signal_slot")
            signal_time = candle.get("signal_time")

            trend_dir = candle.get("trend_direction", 0)
            strength = candle.get("strength_filter", 0)
            vol_ratio = candle.get("volume_ratio", 1.0)
            atr = candle.get("atr", 0.0)

            use_vol = params.get("USE_VOLUME_FILTER", False)
            vol_z_threshold = params.get("VOLUME_Z_THRESHOLD", params.get("VOLUME_THRESHOLD_MULT", 0.0))

            is_uptrend = trend_dir == 1
            entry_upper_d = self._to_decimal(entry_upper)
            signal_close_d = self._to_decimal(signal_close)
            breakout = np.isfinite(signal_close) and np.isfinite(entry_upper) and (signal_close_d > entry_upper_d)
            strong_momentum = strength == 1
            vol_ok = (not use_vol) or (vol_ratio >= vol_z_threshold)
            min_order_d = self._to_decimal(MIN_ORDER_VALUE_KRW)

            # Prevent duplicate entry in same signal slot
            state = self.state_manager.get_symbol_state(symbol)
            if signal_slot and state.get("entry_slot") == signal_slot:
                logger.info(f"[{symbol}] Skip duplicate entry for slot={signal_slot}")
                return

            if is_uptrend and breakout and strong_momentum and vol_ok and (entry_upper > 0):
                invest_amount = self._calculate_position_size(
                    symbol,
                    last_price,
                    params,
                    hurst=candle.get("hurst", 0.5),
                    natr=candle.get("natr", 0.0),
                )
                invest_amount_d = self._to_decimal(invest_amount)
                last_price_d = self._to_decimal(last_price)

                if invest_amount_d > min_order_d:
                    logger.info(
                        f"ENTRY {symbol} | FillPrice~{last_price:,.0f} | SignalClose={signal_close:,.0f} "
                        f"| Cond: Trend(UP), Breakout(>{entry_upper:,.0f}), Strength(OK), Vol(OK) "
                        f"| Invest: {float(invest_amount_d):,.0f} KRW"
                    )

                    res = self._place_order_safe(symbol, "buy", price=float(invest_amount_d))
                    if res and "uuid" in res:
                        quantity_d = (invest_amount_d / last_price_d) if last_price_d > 0 else Decimal("0")
                        self.trade_db.record_trade(
                            symbol=symbol,
                            side="LONG",
                            action="ENTRY",
                            quantity=float(quantity_d),
                            price=float(last_price_d),
                            reason=(
                                f"SignalClose({signal_close:,.0f}) > EntryUpper({entry_upper:,.0f}) "
                                f"@slot={signal_slot}"
                            ),
                        )

                        self.state_manager.update_symbol_state(
                            symbol,
                            {
                                "entry_price": float(last_price_d),
                                "entry_time": datetime.utcnow().isoformat(),
                                "entry_atr": atr,
                                "highest_high": float(last_price_d),
                                "invest_amount": float(invest_amount_d),
                                "entry_slot": signal_slot,
                                "entry_signal_close": float(signal_close_d) if np.isfinite(signal_close) else None,
                                "entry_signal_time": signal_time,
                            },
                        )
                    else:
                        logger.error(f"Buy order failed: {res}")
                else:
                    logger.warning(
                        f"Order skipped for {symbol}: amount {float(invest_amount_d):,.0f} KRW "
                        f"< minimum {MIN_ORDER_VALUE_KRW:,.0f} KRW."
                    )
            else:
                reasons = []
                if pd.isna(entry_upper) or entry_upper <= 0:
                    reasons.append("WaitingData")
                else:
                    if not is_uptrend:
                        reasons.append("TrendNotUp")
                    if not breakout:
                        reasons.append(f"NoBreakout(close={signal_close}, upper={entry_upper:,.0f})")
                    if not strong_momentum:
                        reasons.append("WeakStrength")
                    if not vol_ok:
                        reasons.append(f"VolumeLow({vol_ratio:.2f})")
                if reasons:
                    logger.info(f"[{symbol}] Skip LONG: {', '.join(reasons)}")

        except Exception as e:
            logger.error(f"Error in _check_entry: {e}")
            self.health_manager.record_error(e)

    def _get_current_positions(self) -> dict:
        """Fetch current position summary for heartbeat."""
        positions = {}
        for symbol in self.symbols:
            try:
                state = self.state_manager.get_symbol_state(symbol)
                if state.get('entry_price', 0) > 0:
                    pos = self._fetch_position_safe(symbol)
                    positions[symbol] = {
                        'entry_price': state.get('entry_price'),
                        'amount': pos.get('amount', 0),
                        'unrealized_pnl': (
                            (pos.get('amount', 0) * pos.get('current_price', 0)) -
                            (state.get('invest_amount', 0))
                        ) if pos.get('amount', 0) > 0 else 0
                    }
            except Exception:
                pass
        return positions

    def _is_entry_time(self, timeframe: str) -> bool:
        """
        Check whether now is an entry window for timeframe.
        Example: 4h candle -> hour 00/04/08/12/16/20 and minute <= 2.
        """
        now = datetime.utcnow()
        minutes = now.minute

        # Keep strict sync with backtest entry timing
        if minutes > 2:
            return False

        if timeframe.endswith('m'):
            interval = int(timeframe[:-1])
            return (now.minute % interval) <= 2
        
        elif timeframe.endswith('h'):
            interval = int(timeframe[:-1])
            return (now.hour % interval) == 0 and minutes <= 2
        
        elif timeframe.endswith('d'):
            return now.hour == 0 and minutes <= 2
        
        return False

    def _get_candle_slot_id(self, timeframe: str) -> str:
        """Build unique candle slot ID for duplicate-calc prevention."""
        now = datetime.utcnow()
        if timeframe.endswith('m'):
            interval = int(timeframe[:-1])
            slot = (now.minute // interval) * interval
            return now.strftime(f"%Y%m%d%H{slot:02d}")
        elif timeframe.endswith('h'):
            interval = int(timeframe[:-1])
            slot = (now.hour // interval) * interval
            return now.strftime(f"%Y%m%d{slot:02d}00")
        elif timeframe.endswith('d'):
            return now.strftime("%Y%m%d0000")
        return now.strftime("%Y%m%d%H%M")

    def run_forever(self):
        """Main loop with graceful shutdown support."""
        try:
            self.initialize()
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.health_manager.update_heartbeat(status="init_failed")
            raise

        logger.info("Waiting for next candle close...")

        while not self._shutdown_requested:
            try:
                # Track loop start for stable cycle time
                loop_start_time = time.time()
                
                # Process each symbol
                for symbol in self.symbols:
                    if self._shutdown_requested:
                        break
                    
                    # 1) Load state and params
                    params = self.params_map[symbol]
                    timeframe = params.get('TIMEFRAME', '1h')
                    
                    # 2) Run symbol logic
                    try:
                        self.execute_logic(symbol)
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")

                    time.sleep(SPOT_SYMBOL_DELAY_SECONDS)

                # Heartbeat update
                positions = self._get_current_positions()
                self.health_manager.update_heartbeat(
                    status="running",
                    positions=positions
                )

                # Cloud maintenance
                if self.cloud_optimizer:
                    now = datetime.utcnow()
                    
                    # 1) time sync
                    if not self.cloud_optimizer.check_time_sync_ntp():
                        logger.error("Time drift detected. API errors may occur.")
                    
                    # 2) resource monitoring every 10m
                    if (now - self._last_resource_check).total_seconds() >= 600:
                        usage = self.cloud_optimizer.log_resource_usage()
                        if usage.get('memory_percent', 0) > 85.0:
                            logger.warning(f"High memory ({usage.get('memory_percent')}%). Forcing GC.")
                            self.cloud_optimizer.force_gc()
                        self._last_resource_check = now
                    
                    # 3) DB cleanup every 24h
                    if (now - self._last_db_cleanup).total_seconds() >= 86400:
                        self.cloud_optimizer.cleanup_db_old_records(
                            TRADE_HISTORY_DB, 
                            days_to_keep=90
                        )
                        self._last_db_cleanup = now
                    
                    # 4) explicit GC every 2h
                    if (now - self._last_gc).total_seconds() >= 7200:
                        self.cloud_optimizer.force_gc()
                        self._last_gc = now

                # Dynamic sleep to keep target loop interval
                elapsed_processing = time.time() - loop_start_time
                target_interval = float(SPOT_LOOP_INTERVAL_SECONDS)
                adjusted_wait = max(0.5, target_interval - elapsed_processing)
                
                logger.debug(
                    f"Loop took {elapsed_processing:.2f}s. "
                    f"Sleeping {adjusted_wait:.2f}s (Target: {target_interval:.1f}s cycle)"
                )
                
                # Sleep with shutdown checks
                start_wait = time.time()
                while time.time() - start_wait < adjusted_wait:
                    if self._shutdown_requested:
                        break
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Critical error in main loop: {e}")
                self.health_manager.record_error(e)
                self.health_manager.update_heartbeat(status="error")
                time.sleep(10)

        # Graceful shutdown
        self._shutdown()

    def _shutdown(self):
        """Handle graceful shutdown."""
        logger.info("Shutting down gracefully...")

        # Snapshot open positions on shutdown
        positions = self._get_current_positions()
        if positions:
            logger.warning(f"Open positions at shutdown: {positions}")

        self.health_manager.update_heartbeat(
            status="stopped",
            positions=positions,
            extra={"shutdown_time": datetime.utcnow().isoformat()}
        )

        logger.info("Shutdown complete.")

    def _cache_indicators(self, symbol: str, indicators: dict):
        """Cache indicators in memory."""
        if not hasattr(self, '_indicator_cache'):
            self._indicator_cache = {}
        self._indicator_cache[symbol] = indicators
    
    def _get_cached_indicators(self, symbol: str) -> dict:
        """Get cached indicators."""
        if not hasattr(self, '_indicator_cache'):
            self._indicator_cache = {}
        return self._indicator_cache.get(symbol, {})


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RealTrader Spot - Production Grade Bot (Upbit)")
    logger.info("=" * 60)

    # Controlled by environment variable (default: true)
    enable_oracle_opt = os.getenv("ENABLE_ORACLE_OPTIMIZATION", "true").lower() == "true"

    bot = RealTraderSpot(enable_oracle_optimization=enable_oracle_opt)
    bot.run_forever()


