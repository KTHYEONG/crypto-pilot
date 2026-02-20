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
from typing import Optional, Dict, Any, List, Tuple

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
    SPOT_OPTUNA_STUDY_NAME,
    MAX_TOTAL_BALANCE_KRW,
    SPOT_TARGET_SYMBOLS,
    SPOT_ALLOCATION_WEIGHTS,
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
_default_policy_versions = "SPOT_SELECTION_POLICY_V7_PARITY,SPOT_SELECTION_POLICY_V1,SPOT_SELECTION_POLICY_V11_SIMPLE"
SPOT_ALLOWED_POLICY_VERSIONS = {
    v.strip() for v in os.getenv("SPOT_POLICY_VERSIONS", _default_policy_versions).split(",") if v.strip()
}
SPOT_ALLOWED_MODES = {"UNIFIED", "ALL"}
SPOT_ALLOW_FALLBACK = os.getenv("SPOT_ALLOW_FALLBACK", "false").lower() == "true"
SPOT_FEE_RATE_D = Decimal("0.0005")
SPOT_SLIPPAGE_RATE_D = Decimal("0.0003")
SPOT_ORDER_FILL_TIMEOUT_SEC = float(os.getenv("SPOT_ORDER_FILL_TIMEOUT_SEC", "20"))
SPOT_ORDER_POLL_INTERVAL_SEC = float(os.getenv("SPOT_ORDER_POLL_INTERVAL_SEC", "1.0"))
SPOT_ENTRY_MIN_FILL_RATIO_DEFAULT = float(os.getenv("SPOT_ENTRY_MIN_FILL_RATIO", "0.60"))
SPOT_TIME_SYNC_INTERVAL_SEC = int(os.getenv("SPOT_TIME_SYNC_INTERVAL_SEC", "60"))

# Portfolio from config/settings (Option B: BTC 40% / DOGE 35% / SOL 25%)
SPOT_ALLOCATION_WEIGHTS_D: Dict[str, Decimal] = {
    k: Decimal(str(v)) for k, v in SPOT_ALLOCATION_WEIGHTS.items()
}


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
        self.symbols = list(SPOT_TARGET_SYMBOLS)
        self.symbol_weights = self._build_fixed_symbol_weights(self.symbols)

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
        self._server_time_offset_ms = 0
        self._last_server_time_sync = datetime.min
        
        self.load_strategies_from_db()
        logger.info(
            "Spot portfolio (settings): "
            + ", ".join(f"{s}={float(self.symbol_weights.get(s, Decimal('0'))*Decimal('100')):.0f}%" for s in self.symbols)
        )

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

    @staticmethod
    def _build_fixed_symbol_weights(symbols: List[str]) -> Dict[str, Decimal]:
        weights: Dict[str, Decimal] = {}
        total = Decimal("0")
        for symbol in symbols:
            w = RealTraderSpot._to_decimal(SPOT_ALLOCATION_WEIGHTS_D.get(symbol, Decimal("0")))
            if w < Decimal("0"):
                w = Decimal("0")
            weights[symbol] = w
            total += w

        if total <= Decimal("0"):
            equal = Decimal("1") / Decimal(str(max(1, len(symbols))))
            return {symbol: equal for symbol in symbols}

        return {symbol: (weights.get(symbol, Decimal("0")) / total) for symbol in symbols}

    @staticmethod
    def _resolve_trend_gate_mode(params: dict) -> str:
        mode = str((params or {}).get("TREND_GATE_MODE", "STRICT")).strip().upper()
        if mode not in {"STRICT", "SOFT", "OFF"}:
            mode = "STRICT"
        return mode

    @staticmethod
    def _compute_risk_off_state(
        params: dict,
        trend_direction: int,
        hurst: float,
        natr: float,
        prev_cooldown: int = 0,
    ) -> Tuple[bool, bool, int]:
        enable_risk_off_hard_gate = bool((params or {}).get("ENABLE_RISK_OFF_HARD_GATE", False))
        risk_off_cooldown_bars = max(0, int((params or {}).get("RISK_OFF_COOLDOWN_BARS", 2)))
        panic_natr = float((params or {}).get("PANIC_REGIME_NATR", 4.5))
        weak_hurst = float((params or {}).get("WEAK_REGIME_HURST", 0.45))

        risk_off = False
        if enable_risk_off_hard_gate:
            if int(trend_direction) != 1:
                risk_off = True
            elif float(natr) > panic_natr:
                risk_off = True
            elif float(hurst) < weak_hurst:
                risk_off = True

        cooldown_remaining = max(0, int(prev_cooldown))
        if risk_off:
            cooldown_remaining = max(cooldown_remaining, risk_off_cooldown_bars)
        elif cooldown_remaining > 0:
            cooldown_remaining -= 1

        risk_blocked = risk_off or (cooldown_remaining > 0)
        return risk_off, risk_blocked, cooldown_remaining

    def _get_effective_risk_weight(self, symbol: str, params: dict, hurst: float = 0.5, natr: float = 0.0) -> Decimal:
        default_weight_d = (Decimal("1") / Decimal(str(len(self.symbols)))) if self.symbols else Decimal("0.5")
        base_weight_d = self._to_decimal(self.symbol_weights.get(symbol, default_weight_d))
        if base_weight_d <= Decimal("0"):
            base_weight_d = default_weight_d
            
        # [ISSUE FIX] 라이브 봇-백테스트 간 Sizing 불일치 해결
        # Options의 할당량(base_weight)에 옵티마이저가 도출한 RISK_PER_TRADE_SPOT(리스크 웨이트) 적용
        risk_per_trade_d = self._to_decimal(params.get("RISK_PER_TRADE_SPOT", 0.99))
        if risk_per_trade_d <= Decimal("0") or risk_per_trade_d > Decimal("1.0"):
            risk_per_trade_d = Decimal("0.99")
            
        base_weight_d = base_weight_d * risk_per_trade_d
        
        if base_weight_d > Decimal("0.99"):
            base_weight_d = Decimal("0.99")

        regime_mult_d = Decimal("1")
        # Keep fixed portfolio weights as baseline, but apply regime multiplier when enabled.
        use_dynamic_risk = bool(params.get("USE_DYNAMIC_RISK", False))
        strong_hurst = params.get("HURST_TREND_THRESHOLD", params.get("STRONG_REGIME_HURST", 0.60))
        strong_natr = params.get("STRONG_REGIME_NATR", 1.0)
        weak_hurst = params.get("WEAK_REGIME_HURST", 0.45)
        panic_natr = params.get("PANIC_REGIME_NATR", 4.5)
        strong_mult_d = self._to_decimal(params.get("STRONG_REGIME_MULTIPLIER", 1.3))
        weak_mult_d = self._to_decimal(params.get("WEAK_REGIME_MULTIPLIER", 0.6))
        panic_mult_d = self._to_decimal(params.get("PANIC_REGIME_MULTIPLIER", 0.15))

        if use_dynamic_risk:
            if natr > panic_natr:
                regime_mult_d = panic_mult_d
                logger.info(f"Panic regime detected for {symbol} (NATR {natr:.2f}). Mult: {float(panic_mult_d):.2f}")
            elif hurst > strong_hurst and natr > strong_natr:
                regime_mult_d = strong_mult_d
                logger.info(
                    f"Strong regime detected for {symbol} (Hurst {hurst:.2f}, NATR {natr:.2f}). "
                    f"Mult: {float(strong_mult_d):.2f}"
                )
            elif hurst < weak_hurst:
                regime_mult_d = weak_mult_d
                logger.info(f"Weak regime detected for {symbol} (Hurst {hurst:.2f}). Mult: {float(weak_mult_d):.2f}")

        final_weight_d = base_weight_d * regime_mult_d
        if final_weight_d > Decimal("0.99"):
            final_weight_d = Decimal("0.99")
        if final_weight_d < Decimal("0"):
            final_weight_d = Decimal("0")
        return final_weight_d

    def _extract_order_fill_summary(
        self,
        order: Optional[dict],
        fallback_price: Decimal,
    ) -> Dict[str, Any]:
        if not order:
            return {
                "order_id": "",
                "status": "unknown",
                "filled_qty": Decimal("0"),
                "remaining_qty": Decimal("0"),
                "avg_price": fallback_price,
                "filled_cost": Decimal("0"),
            }

        order_id = str(order.get("id") or order.get("uuid") or "")
        status = str(order.get("status") or "unknown").lower()
        filled_qty = self._to_decimal(order.get("filled", 0.0))
        remaining_qty = self._to_decimal(order.get("remaining", 0.0))
        avg_price = self._to_decimal(order.get("average", order.get("price", 0.0)))
        if avg_price <= Decimal("0"):
            avg_price = fallback_price
        filled_cost = self._to_decimal(order.get("cost", 0.0))
        if filled_cost <= Decimal("0") and filled_qty > Decimal("0") and avg_price > Decimal("0"):
            filled_cost = filled_qty * avg_price

        return {
            "order_id": order_id,
            "status": status,
            "filled_qty": filled_qty,
            "remaining_qty": remaining_qty,
            "avg_price": avg_price,
            "filled_cost": filled_cost,
        }

    def _wait_for_order_fill(
        self,
        symbol: str,
        side: str,
        order: Optional[dict],
        fallback_price: Decimal,
        expected_qty: Optional[Decimal] = None,
        expected_cost: Optional[Decimal] = None,
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        summary = self._extract_order_fill_summary(order, fallback_price)
        order_id = summary["order_id"]
        if not order_id:
            return summary

        timeout = SPOT_ORDER_FILL_TIMEOUT_SEC if timeout_sec is None else max(1.0, float(timeout_sec))
        poll_interval = max(0.2, SPOT_ORDER_POLL_INTERVAL_SEC)
        deadline = time.time() + timeout
        last_status = summary["status"]

        while time.time() < deadline:
            ord_now = self._fetch_order_safe(symbol, order_id)
            if ord_now is None:
                time.sleep(poll_interval)
                continue
            ord_summary = self._extract_order_fill_summary(ord_now, fallback_price)
            summary = ord_summary
            status = ord_summary["status"]
            remaining = ord_summary["remaining_qty"]
            if status in {"closed", "canceled"}:
                break
            if remaining <= Decimal("0"):
                break
            if status != last_status:
                logger.info(f"[{symbol}] Order {order_id} status: {status}")
                last_status = status
            time.sleep(poll_interval)

        # Timeout handling: cancel remaining qty and re-sync.
        if summary["status"] not in {"closed", "canceled"} and summary["remaining_qty"] > Decimal("0"):
            logger.warning(
                f"[{symbol}] Order {order_id} timeout ({timeout:.0f}s). "
                f"Canceling remaining {float(summary['remaining_qty']):.8f}."
            )
            self._cancel_order_safe(symbol, order_id)
            ord_after_cancel = self._fetch_order_safe(symbol, order_id)
            summary = self._extract_order_fill_summary(ord_after_cancel, fallback_price)

        fill_ratio = Decimal("0")
        filled_qty = summary["filled_qty"]
        filled_cost = summary["filled_cost"]
        if expected_qty is not None and self._to_decimal(expected_qty) > Decimal("0"):
            fill_ratio = filled_qty / self._to_decimal(expected_qty)
        elif expected_cost is not None and self._to_decimal(expected_cost) > Decimal("0"):
            fill_ratio = filled_cost / self._to_decimal(expected_cost)
        else:
            total_qty = filled_qty + max(Decimal("0"), summary["remaining_qty"])
            if total_qty > Decimal("0"):
                fill_ratio = filled_qty / total_qty
        if fill_ratio < Decimal("0"):
            fill_ratio = Decimal("0")
        if fill_ratio > Decimal("1"):
            fill_ratio = Decimal("1")
        summary["fill_ratio"] = float(fill_ratio)
        summary["is_filled"] = bool(
            summary["status"] == "closed"
            or (summary["remaining_qty"] <= Decimal("0") and summary["filled_qty"] > Decimal("0"))
        )
        summary["side"] = side
        return summary

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

            if policy_version not in SPOT_ALLOWED_POLICY_VERSIONS:
                errors.append(
                    f"{symbol}: policy_version='{policy_version}' not in {sorted(SPOT_ALLOWED_POLICY_VERSIONS)}"
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
            'ENABLE_SCALE_OUT': False,
            'SCALE_OUT_TRIGGER_ATR': 1.2,
            'SCALE_OUT_RATIO': 0.5,
            'ENABLE_BREAKEVEN': True,
            'BREAKEVEN_BUFFER_PCT': 0.001,
            'ENABLE_PYRAMIDING': False,
            'PYRAMID_TRIGGER_ATR': 1.8,
            'PYRAMID_STEP_ATR': 1.0,
            'PYRAMID_RISK_RATIO': 0.30,
            'PYRAMID_MAX_ADDS': 1,
            'ENTRY_MIN_FILL_RATIO': SPOT_ENTRY_MIN_FILL_RATIO_DEFAULT,
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

    @api_retry
    def _place_market_order_safe(self, symbol: str, side: str, amount: Optional[float] = None, price: Optional[float] = None):
        """Place market order with retry."""
        return self.client.place_order(symbol, side, amount=amount, price=price, order_type="market")

    @api_retry
    def _fetch_order_safe(self, symbol: str, order_id: str):
        """Fetch order status with retry."""
        return self.client.fetch_order(order_id, symbol)

    @api_retry
    def _cancel_order_safe(self, symbol: str, order_id: str):
        """Cancel order with retry."""
        return self.client.cancel_order(order_id, symbol)

    @api_retry
    def _fetch_open_orders_safe(self, symbol: str) -> list:
        """Fetch open orders with retry."""
        return self.client.fetch_open_orders(symbol)

    @api_retry
    def _fetch_server_time_ms_safe(self) -> int:
        """Fetch exchange server time (ms) with retry."""
        server_ms = self.client.fetch_server_time_ms()
        if server_ms is None:
            raise ValueError("Failed to fetch Upbit server time")
        return int(server_ms)

    def _sync_server_time_offset(self, force: bool = False, sync_interval_seconds: int = SPOT_TIME_SYNC_INTERVAL_SEC) -> None:
        """Synchronize local-reference clock to exchange server time."""
        now = datetime.utcnow()
        elapsed = (now - self._last_server_time_sync).total_seconds()
        if (not force) and elapsed < max(5, int(sync_interval_seconds)):
            return
        try:
            server_ms = self._fetch_server_time_ms_safe()
            local_ms = int(time.time() * 1000)
            self._server_time_offset_ms = int(server_ms - local_ms)
            self._last_server_time_sync = now
        except Exception as e:
            self._last_server_time_sync = now
            logger.debug(f"[time-sync] Upbit server time sync failed: {e}")

    def _get_reference_now_ms(self) -> int:
        self._sync_server_time_offset(force=False)
        return int(time.time() * 1000) + int(self._server_time_offset_ms)

    def _get_reference_now_utc(self) -> datetime:
        return datetime.utcfromtimestamp(self._get_reference_now_ms() / 1000.0)

    def _cancel_open_orders_best_effort(self, symbol: str, reason: str = "") -> None:
        """Best-effort cancel for stale/open orders before critical actions."""
        try:
            open_orders = self._fetch_open_orders_safe(symbol)
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to fetch open orders before {reason}: {e}")
            return
        if not open_orders:
            return
        logger.warning(
            f"[{symbol}] Found {len(open_orders)} open order(s) before {reason}. "
            "Canceling for state consistency."
        )
        for order in open_orders:
            order_id = str(order.get("id") or order.get("uuid") or "")
            if not order_id:
                continue
            try:
                self._cancel_order_safe(symbol, order_id)
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to cancel open order {order_id}: {e}")

    @staticmethod
    def _state_has_meaningful_data(state: Optional[dict]) -> bool:
        if not state:
            return False
        if RealTraderSpot._to_decimal(state.get("entry_price", 0.0)) > Decimal("0"):
            return True
        if RealTraderSpot._to_decimal(state.get("invest_amount", 0.0)) > Decimal("0"):
            return True
        if RealTraderSpot._to_decimal(state.get("realized_pnl", 0.0)) != Decimal("0"):
            return True
        if state.get("entry_time"):
            return True
        return False

    def _position_state_missing_core(self, state: Optional[dict]) -> bool:
        """Check whether local state has minimum fields required for deterministic exits."""
        if not state:
            return True
        entry_price_d = self._to_decimal(state.get("entry_price", 0.0))
        has_entry_price = entry_price_d > Decimal("0")
        has_entry_time = bool(state.get("entry_time"))
        return not (has_entry_price and has_entry_time)

    def _bootstrap_state_for_open_position(
        self,
        symbol: str,
        amount: float,
        pos: dict,
        current_price: float,
        params: dict,
        atr: float,
        prev_state: Optional[dict] = None,
    ) -> bool:
        """
        Rebuild local state from exchange position when state is missing/corrupted.
        Keeps exit logic deterministic after restart/state loss.
        """
        try:
            amount_d = self._to_decimal(amount)
            current_price_d = self._to_decimal(current_price)
            if amount_d <= Decimal("0") or current_price_d <= Decimal("0"):
                return False

            entry_price_d = self._to_decimal(pos.get("entryPrice", 0.0))
            if entry_price_d <= Decimal("0"):
                entry_price_d = current_price_d

            atr_d = self._to_decimal(atr)
            if atr_d <= Decimal("0") and prev_state:
                atr_d = self._to_decimal(prev_state.get("entry_atr", 0.0))

            invest_amount_d = amount_d * entry_price_d
            now_iso = self._get_reference_now_utc().isoformat()
            new_state = {
                "entry_price": float(entry_price_d),
                "entry_time": now_iso,
                "entry_atr": float(max(Decimal("0"), atr_d)),
                "highest_high": float(max(entry_price_d, current_price_d)),
                "invest_amount": float(max(Decimal("0"), invest_amount_d)),
                "realized_pnl": float(self._to_decimal((prev_state or {}).get("realized_pnl", 0.0))),
                "scale_out_done": bool((prev_state or {}).get("scale_out_done", False)),
                "stop_price_override": float(self._to_decimal((prev_state or {}).get("stop_price_override", 0.0))),
                "pyramid_add_count": int((prev_state or {}).get("pyramid_add_count", 0)),
                "next_pyramid_trigger": float(self._to_decimal((prev_state or {}).get("next_pyramid_trigger", 0.0))),
                "risk_off_cooldown_remaining": int((prev_state or {}).get("risk_off_cooldown_remaining", 0)),
            }
            self.state_manager.update_symbol_state(symbol, new_state)
            logger.warning(
                f"[{symbol}] Bootstrapped local state from live exchange position "
                f"(entry={float(entry_price_d):,.0f}, qty={float(amount_d):.8f})."
            )
            return True
        except Exception as e:
            logger.error(f"[{symbol}] Failed to bootstrap spot state: {e}")
            return False

    def _build_mtf_confirmed_candle(
        self,
        strategy: UltimateStrategy,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        params: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Build live MTF-confirmed candle aligned with backtest semantics:
        - generate hourly/daily signals
        - map shifted daily trend onto hourly bars by as-of backward join
        - final trend applies TREND_GATE_MODE (STRICT/SOFT/OFF)
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
        trend_gate_mode = self._resolve_trend_gate_mode(params if params is not None else strategy.params)
        if trend_gate_mode == "OFF":
            aligned_trend = np.where(hourly_trend == 1, 1, 0)
        elif trend_gate_mode == "SOFT":
            aligned_trend = np.where((hourly_trend == 1) | (mapped_daily_trend == 1), 1, 0)
        else:
            aligned_trend = np.where((hourly_trend == 1) & (mapped_daily_trend == 1), 1, 0)
        hourly_sig["trend_direction"] = aligned_trend
        hourly_sig["daily_trend_direction"] = mapped_daily_trend

        confirmed = hourly_sig.iloc[-2]
        return {
            "signal_time": pd.Timestamp(confirmed["datetime"]).isoformat(),
            "signal_open": float(confirmed.get("open", np.nan)),
            "signal_high": float(confirmed.get("high", np.nan)),
            "signal_low": float(confirmed.get("low", np.nan)),
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
            self._sync_server_time_offset(force=True)
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

                confirmed = self._build_mtf_confirmed_candle(strategy, hourly_df, daily_df, params=params)
                if not confirmed:
                    logger.warning(f"Failed to build MTF confirmed signal for {symbol}")
                    self.last_calc_candle[symbol] = current_slot
                    return

                prev_cooldown = int(state.get("risk_off_cooldown_remaining", 0))
                risk_off, risk_blocked, risk_off_cooldown_remaining = self._compute_risk_off_state(
                    params=params,
                    trend_direction=int(confirmed.get("trend_direction", 0)),
                    hurst=float(confirmed.get("hurst", 0.5)),
                    natr=float(confirmed.get("natr", 0.0)),
                    prev_cooldown=prev_cooldown,
                )
                if prev_cooldown != int(risk_off_cooldown_remaining):
                    self.state_manager.update_symbol_state(
                        symbol,
                        {"risk_off_cooldown_remaining": int(risk_off_cooldown_remaining)},
                    )
                    state = self.state_manager.get_symbol_state(symbol)

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
                        "signal_open": float(confirmed.get("signal_open", np.nan)),
                        "signal_high": float(confirmed.get("signal_high", np.nan)),
                        "signal_low": float(confirmed.get("signal_low", np.nan)),
                        "signal_close": float(confirmed.get("signal_close", np.nan)),
                        "signal_time": confirmed.get("signal_time"),
                        "signal_slot": current_slot,
                        "risk_off": bool(risk_off),
                        "risk_blocked": bool(risk_blocked),
                        "risk_off_cooldown_remaining": int(risk_off_cooldown_remaining),
                        "cached_at": datetime.utcnow().isoformat(),
                    },
                )
                self.last_calc_candle[symbol] = current_slot
                cached = self._get_cached_indicators(symbol)
                logger.info(
                    f"[{symbol}] Cached signal: trend={cached.get('trend_direction')} "
                    f"(daily={cached.get('daily_trend_direction')}), close={cached.get('signal_close')} "
                    f"| risk_blocked={cached.get('risk_blocked')} cooldown={cached.get('risk_off_cooldown_remaining')}"
                )

            trend_dir = cached.get("trend_direction", 0)
            atr = cached.get("atr", 0.0)
            sar = cached.get("parabolic_sar", 0.0)
            risk_blocked = bool(cached.get("risk_blocked", False))

            last_price = self._get_market_price_safe(symbol)
            if last_price is None:
                logger.warning(f"[{symbol}] Could not fetch market price. Skipping cycle.")
                return

            current_value_d = self._to_decimal(balance_coin) * self._to_decimal(last_price)
            in_position = current_value_d > self._to_decimal(MIN_POSITION_VALUE_KRW)
            now_ref = self._get_reference_now_utc()

            if (not in_position) and self._state_has_meaningful_data(state):
                logger.info(f"[{symbol}] Clearing stale local state (no open position on exchange).")
                self.state_manager.clear_symbol_state(symbol)
                state = {}

            if in_position:
                logger.info(f"[{symbol}] Position exists ({float(current_value_d):,.0f} KRW). Checking exit...")
                if self._position_state_missing_core(state):
                    self._bootstrap_state_for_open_position(
                        symbol=symbol,
                        amount=balance_coin,
                        pos=pos,
                        current_price=last_price,
                        params=params,
                        atr=atr,
                        prev_state=state,
                    )
                    state = self.state_manager.get_symbol_state(symbol)

                mock_candle = {
                    "atr": atr,
                    "parabolic_sar": sar,
                    "trend_direction": trend_dir,
                    "rsi": cached.get("rsi", 50.0),
                    "high": last_price,
                    "low": last_price,
                    "strength_filter": cached.get("strength_filter", 0),
                    "volume_ratio": cached.get("volume_ratio", 1.0),
                    "hurst": cached.get("hurst", 0.5),
                    "natr": cached.get("natr", 0.0),
                    "entry_upper": cached.get("entry_upper"),
                    "signal_close": cached.get("signal_close", np.nan),
                    "signal_slot": cached.get("signal_slot"),
                    "risk_off": bool(cached.get("risk_off", False)),
                    "risk_blocked": risk_blocked,
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
                                "realized_pnl": float(state.get("realized_pnl", 0.0)),
                                "scale_out_done": bool(state.get("scale_out_done", False)),
                                "stop_price_override": float(state.get("stop_price_override", 0.0)),
                                "pyramid_add_count": int(state.get("pyramid_add_count", 0)),
                                "next_pyramid_trigger": float(state.get("next_pyramid_trigger", 0.0)),
                            }
                        )
                        self.state_manager.update_symbol_state(symbol, state)

                position_closed = self._check_exit(symbol, balance_coin, last_price, params, mock_candle, state)
                if (
                    (not position_closed)
                    and is_entry_time
                    and bool(params.get("ENABLE_PYRAMIDING", False))
                    and (not risk_blocked)
                ):
                    refreshed_state = self.state_manager.get_symbol_state(symbol)
                    last_scale_out_time = refreshed_state.get("last_scale_out_time")
                    if last_scale_out_time:
                        try:
                            if (now_ref - datetime.fromisoformat(last_scale_out_time)).total_seconds() < 120:
                                return
                        except Exception:
                            pass
                    refreshed_pos = self._fetch_position_safe(symbol)
                    self._check_pyramiding(
                        symbol,
                        refreshed_pos.get("amount", balance_coin),
                        last_price,
                        params,
                        mock_candle,
                        refreshed_state,
                    )

            elif not in_position and is_entry_time:
                logger.info(f"[{symbol}] Entry window open. Checking entry conditions...")

                state = self.state_manager.get_symbol_state(symbol)
                last_entry_str = state.get("entry_time")
                if last_entry_str:
                    last_entry_dt = datetime.fromisoformat(last_entry_str)
                    if (now_ref - last_entry_dt).total_seconds() < 180:
                        logger.info(f"[{symbol}] Recent entry detected ({last_entry_str}). Skip duplicate entry.")
                        return

                open_orders = self._fetch_open_orders_safe(symbol)
                if open_orders and len(open_orders) > 0:
                    logger.warning(f"[{symbol}] Open orders exist ({len(open_orders)}). Skip entry.")
                    return

                if risk_blocked:
                    logger.info(
                        f"[{symbol}] Skip entry: risk-off gate active "
                        f"(cooldown={int(cached.get('risk_off_cooldown_remaining', 0))})."
                    )
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
                        "risk_off": bool(cached.get("risk_off", False)),
                        "risk_blocked": risk_blocked,
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
    ) -> bool:
        """Exit/management logic for long-only spot positions."""
        try:
            should_sell = False
            reason = ""

            entry_price = state.get("entry_price", 0.0)
            entry_price_d = self._to_decimal(entry_price)
            last_price_d = self._to_decimal(last_price)
            balance_coin_d = self._to_decimal(balance_coin)
            if entry_price_d <= Decimal("0") or balance_coin_d <= Decimal("0"):
                return False

            entry_atr = state.get("entry_atr", 0.0)
            if entry_atr == 0:
                entry_atr = candle.get("atr", 0.0)
                if entry_atr > 0:
                    logger.warning(f"Recovered entry_atr from current candle for {symbol}: {entry_atr}")
            entry_atr_d = self._to_decimal(entry_atr)

            highest_high_d = self._to_decimal(state.get("highest_high", entry_price))
            # Live execution must use only currently observable price to avoid implicit lookahead.
            current_high_d = last_price_d
            current_low_d = last_price_d
            if current_high_d > highest_high_d:
                highest_high_d = current_high_d
                state["highest_high"] = float(highest_high_d)
                self.state_manager.update_symbol_state(symbol, state)

            exit_type = params.get("EXIT_TYPE", "ATR")
            atr_mult_d = self._to_decimal(params.get("ATR_MULTIPLIER", 3.0))
            sl_type = params.get("STOP_LOSS_TYPE", "FIXED")
            sl_pct_d = self._to_decimal(params.get("STOP_LOSS_PCT", 0.02))
            atr_sl_mult_d = self._to_decimal(params.get("ATR_STOP_LOSS_MULT", 1.0))
            use_tp = bool(params.get("USE_TAKE_PROFIT", False))
            tp_mult_d = self._to_decimal(params.get("TAKE_PROFIT_ATR_MULT", 2.0))
            trailing_activation_atr_d = self._to_decimal(params.get("TRAILING_ACTIVATION_ATR", 0.0))
            if trailing_activation_atr_d < Decimal("0"):
                trailing_activation_atr_d = Decimal("0")
            trend_dir = int(candle.get("trend_direction", 0))
            enable_risk_off_hard_gate = bool(params.get("ENABLE_RISK_OFF_HARD_GATE", False))
            risk_off_exit_on_trigger = bool(params.get("RISK_OFF_EXIT_ON_TRIGGER", False))
            risk_blocked = bool(candle.get("risk_blocked", False))

            # 1) Base stop loss with breakeven override (if enabled/armed)
            if sl_type == "ATR" and entry_atr_d > 0:
                stop_price_d = entry_price_d - (entry_atr_d * atr_sl_mult_d)
            else:
                stop_price_d = entry_price_d * (Decimal("1") - sl_pct_d)
            stop_override_d = self._to_decimal(state.get("stop_price_override", 0.0))
            if stop_override_d > stop_price_d:
                stop_price_d = stop_override_d

            if current_low_d <= stop_price_d:
                should_sell = True
                reason = f"Stop Loss ({float(stop_price_d):,.0f})"

            # 1.5) Scale-out + breakeven arm (run only when full exit is not triggered)
            enable_scale_out = bool(params.get("ENABLE_SCALE_OUT", False))
            enable_breakeven = bool(params.get("ENABLE_BREAKEVEN", True))
            scale_out_done = bool(state.get("scale_out_done", False))
            if (not should_sell) and enable_scale_out and (not scale_out_done) and entry_atr_d > 0:
                scale_trigger_atr_d = self._to_decimal(params.get("SCALE_OUT_TRIGGER_ATR", 1.2))
                if scale_trigger_atr_d < Decimal("0.1"):
                    scale_trigger_atr_d = Decimal("0.1")
                scale_out_ratio_d = self._to_decimal(params.get("SCALE_OUT_RATIO", 0.5))
                if scale_out_ratio_d < Decimal("0"):
                    scale_out_ratio_d = Decimal("0")
                if scale_out_ratio_d > Decimal("0.95"):
                    scale_out_ratio_d = Decimal("0.95")
                scale_trigger_price_d = entry_price_d + (entry_atr_d * scale_trigger_atr_d)

                if current_high_d >= scale_trigger_price_d:
                    scale_qty_d = balance_coin_d * scale_out_ratio_d
                    if scale_qty_d > balance_coin_d:
                        scale_qty_d = balance_coin_d
                    remain_qty_d = balance_coin_d - scale_qty_d
                    scale_notional_d = scale_qty_d * last_price_d
                    remain_notional_d = remain_qty_d * last_price_d
                    min_order_d = self._to_decimal(MIN_ORDER_VALUE_KRW)
                    min_pos_d = self._to_decimal(MIN_POSITION_VALUE_KRW)

                    if (
                        scale_qty_d > Decimal("0")
                        and scale_notional_d >= min_order_d
                        and remain_notional_d >= min_pos_d
                    ):
                        logger.info(
                            f"SCALE-OUT {symbol} | Trigger={float(scale_trigger_price_d):,.0f} "
                            f"| Qty={float(scale_qty_d):.8f} | Last={float(last_price_d):,.0f}"
                        )
                        self._cancel_open_orders_best_effort(symbol, reason="scale-out")
                        open_orders = self._fetch_open_orders_safe(symbol)
                        if open_orders and len(open_orders) > 0:
                            logger.warning(f"[{symbol}] Skip scale-out: open orders still exist ({len(open_orders)}).")
                            return False
                        res = self._place_order_safe(symbol, "sell", amount=float(scale_qty_d))
                        if not res:
                            logger.error(f"Scale-out sell order failed: {res}")
                            return False

                        fill = self._wait_for_order_fill(
                            symbol=symbol,
                            side="sell",
                            order=res,
                            fallback_price=last_price_d,
                            expected_qty=scale_qty_d,
                        )
                        filled_qty_d = self._to_decimal(fill.get("filled_qty", 0.0))
                        if filled_qty_d <= Decimal("0"):
                            logger.warning(f"[{symbol}] Scale-out not filled.")
                            return False
                        if filled_qty_d > balance_coin_d:
                            filled_qty_d = balance_coin_d

                        fill_price_d = self._to_decimal(fill.get("avg_price", float(last_price_d)))
                        filled_cost_d = self._to_decimal(fill.get("filled_cost", 0.0))
                        if filled_cost_d <= Decimal("0"):
                            filled_cost_d = filled_qty_d * fill_price_d

                        prev_invest_d = self._to_decimal(state.get("invest_amount", float(entry_price_d * balance_coin_d)))
                        sold_cost_d = Decimal("0")
                        if balance_coin_d > Decimal("0"):
                            sold_cost_d = prev_invest_d * (filled_qty_d / balance_coin_d)
                        sold_pnl_d = filled_cost_d - sold_cost_d
                        remain_invest_d = prev_invest_d - sold_cost_d
                        realized_pnl_d = self._to_decimal(state.get("realized_pnl", 0.0)) + sold_pnl_d

                        state["scale_out_done"] = True
                        state["invest_amount"] = float(max(Decimal("0"), remain_invest_d))
                        state["realized_pnl"] = float(realized_pnl_d)
                        state["last_scale_out_time"] = self._get_reference_now_utc().isoformat()

                        if enable_breakeven:
                            be_buffer_d = self._to_decimal(params.get("BREAKEVEN_BUFFER_PCT", 0.001))
                            if be_buffer_d < Decimal("0"):
                                be_buffer_d = Decimal("0")
                            be_stop_d = entry_price_d * (
                                Decimal("1") + (Decimal("2") * SPOT_FEE_RATE_D) + SPOT_SLIPPAGE_RATE_D + be_buffer_d
                            )
                            prev_stop_override_d = self._to_decimal(state.get("stop_price_override", 0.0))
                            if be_stop_d > prev_stop_override_d:
                                state["stop_price_override"] = float(be_stop_d)

                        self.state_manager.update_symbol_state(symbol, state)
                        self.trade_db.record_trade(
                            symbol=symbol,
                            side="LONG",
                            action="SCALE_OUT",
                            quantity=float(filled_qty_d),
                            price=float(fill_price_d),
                            entry_price=float(entry_price_d),
                            pnl=float(sold_pnl_d),
                            pnl_pct=float(((fill_price_d / entry_price_d) - Decimal("1")) * Decimal("100")) if entry_price_d > 0 else 0.0,
                            reason=f"ScaleOut Trigger ({float(scale_trigger_price_d):,.0f})",
                        )
                        return False

            # 2) Main exit (ATR trailing / SAR)
            if not should_sell:
                if exit_type == "ATR" and entry_atr_d > 0:
                    unrealized_profit_atr_d = (highest_high_d - entry_price_d) / entry_atr_d
                    if unrealized_profit_atr_d >= trailing_activation_atr_d:
                        trailing_stop_d = highest_high_d - (entry_atr_d * atr_mult_d)
                        if current_low_d <= trailing_stop_d:
                            should_sell = True
                            reason = f"ATR Trailing Stop ({float(trailing_stop_d):,.0f})"
                elif exit_type == "PARABOLIC_SAR":
                    p_sar = candle.get("parabolic_sar", 0)
                    if p_sar <= 0:
                        logger.warning(f"[{symbol}] SAR enabled but value invalid ({p_sar:.2f}).")
                    elif current_low_d <= self._to_decimal(p_sar):
                        should_sell = True
                        reason = f"Parabolic SAR Exit ({p_sar:,.0f})"

            # 3) Take profit
            if not should_sell and use_tp and entry_atr_d > 0:
                target_price_d = entry_price_d + (entry_atr_d * tp_mult_d)
                if current_high_d >= target_price_d:
                    should_sell = True
                    reason = f"Take Profit ({float(target_price_d):,.0f})"

            # 4) Hard risk-off exit
            if (
                (not should_sell)
                and enable_risk_off_hard_gate
                and risk_off_exit_on_trigger
                and risk_blocked
            ):
                should_sell = True
                reason = "Risk-Off Hard Exit"

            # 5) Panic exit (RSI)
            rsi = candle.get("rsi", 0)
            rsi_exit_thresh = params.get("RSI_EXIT_THRESHOLD", 93)
            if not should_sell and rsi > rsi_exit_thresh:
                should_sell = True
                reason = f"Panic Exit (RSI {rsi:.1f})"

            # 6) Trend reversal
            if not should_sell and trend_dir <= 0:
                should_sell = True
                reason = "Trend Reversal"

            # 7) Time cut (profit check)
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
                elapsed_min = (self._get_reference_now_utc() - entry_dt).total_seconds() / 60
                bars_held = elapsed_min / interval_min
                if bars_held >= max_holding_bars and entry_atr_d > 0:
                    unrealized_profit_atr_d = (last_price_d - entry_price_d) / entry_atr_d
                    profit_thresh_d = self._to_decimal(params.get("TIME_EXIT_PROFIT_THRESHOLD", 1.4))
                    if unrealized_profit_atr_d <= profit_thresh_d:
                        should_sell = True
                        reason = (
                            f"Time Cut (Held {bars_held:.1f} bars, "
                            f"Unrealized {float(unrealized_profit_atr_d):.2f} ATR)"
                        )

            if should_sell:
                invest_amount_d = self._to_decimal(state.get("invest_amount", float(entry_price_d * balance_coin_d)))
                realized_pnl_d = self._to_decimal(state.get("realized_pnl", 0.0))
                logger.info(
                    f"EXIT {symbol} | Price: {last_price:,.0f} | "
                    f"Reason: {reason}"
                )
                self._cancel_open_orders_best_effort(symbol, reason="full-exit")
                open_orders = self._fetch_open_orders_safe(symbol)
                if open_orders and len(open_orders) > 0:
                    logger.warning(f"[{symbol}] Skip full exit: open orders still exist ({len(open_orders)}).")
                    return False
                res = self._place_order_safe(symbol, "sell", amount=float(balance_coin_d))
                if not res:
                    logger.error(f"Sell order failed: {res}")
                    return False

                fill = self._wait_for_order_fill(
                    symbol=symbol,
                    side="sell",
                    order=res,
                    fallback_price=last_price_d,
                    expected_qty=balance_coin_d,
                )
                filled_qty_d = self._to_decimal(fill.get("filled_qty", 0.0))
                if filled_qty_d <= Decimal("0"):
                    logger.warning(f"[{symbol}] Exit order not filled.")
                    return False
                if filled_qty_d > balance_coin_d:
                    filled_qty_d = balance_coin_d

                fill_price_d = self._to_decimal(fill.get("avg_price", float(last_price_d)))
                filled_cost_d = self._to_decimal(fill.get("filled_cost", 0.0))
                if filled_cost_d <= Decimal("0"):
                    filled_cost_d = filled_qty_d * fill_price_d

                sold_cost_d = Decimal("0")
                if balance_coin_d > Decimal("0"):
                    sold_cost_d = invest_amount_d * (filled_qty_d / balance_coin_d)
                sold_cost_total_d = sold_cost_d
                this_sell_pnl_d = filled_cost_d - sold_cost_d
                new_realized_pnl_d = realized_pnl_d + this_sell_pnl_d
                remain_qty_d = balance_coin_d - filled_qty_d
                remain_notional_d = remain_qty_d * last_price_d

                closed = (
                    remain_qty_d <= Decimal("0")
                    or remain_notional_d < self._to_decimal(MIN_POSITION_VALUE_KRW)
                )

                if closed:
                    pnl_pct = float(((fill_price_d / entry_price_d) - Decimal("1")) * Decimal("100")) if entry_price_d > 0 else 0.0
                    self.trade_db.record_trade(
                        symbol=symbol,
                        side="LONG",
                        action="EXIT",
                        quantity=float(filled_qty_d),
                        price=float(fill_price_d),
                        entry_price=float(entry_price_d),
                        pnl=float(new_realized_pnl_d),
                        pnl_pct=pnl_pct,
                        reason=reason,
                    )
                    self.state_manager.clear_symbol_state(symbol)
                    return True

                # Try one market close for the remaining quantity to mimic backtest full-exit semantics.
                min_order_d = self._to_decimal(MIN_ORDER_VALUE_KRW)
                if remain_notional_d >= min_order_d:
                    logger.warning(
                        f"[{symbol}] Exit partial fill. Trying market close for remaining "
                        f"{float(remain_qty_d):.8f}."
                    )
                    mkt_res = self._place_market_order_safe(symbol, "sell", amount=float(remain_qty_d))
                    if mkt_res:
                        mkt_fill = self._wait_for_order_fill(
                            symbol=symbol,
                            side="sell",
                            order=mkt_res,
                            fallback_price=last_price_d,
                            expected_qty=remain_qty_d,
                            timeout_sec=max(8.0, SPOT_ORDER_FILL_TIMEOUT_SEC * 0.6),
                        )
                        mkt_filled_qty_d = self._to_decimal(mkt_fill.get("filled_qty", 0.0))
                        if mkt_filled_qty_d > remain_qty_d:
                            mkt_filled_qty_d = remain_qty_d
                        if mkt_filled_qty_d > Decimal("0"):
                            mkt_price_d = self._to_decimal(mkt_fill.get("avg_price", float(last_price_d)))
                            mkt_cost_d = self._to_decimal(mkt_fill.get("filled_cost", 0.0))
                            if mkt_cost_d <= Decimal("0"):
                                mkt_cost_d = mkt_filled_qty_d * mkt_price_d

                            remain_invest_before_d = invest_amount_d - sold_cost_d
                            if remain_invest_before_d < Decimal("0"):
                                remain_invest_before_d = Decimal("0")
                            mkt_sold_cost_d = Decimal("0")
                            if remain_qty_d > Decimal("0"):
                                mkt_sold_cost_d = remain_invest_before_d * (mkt_filled_qty_d / remain_qty_d)
                            sold_cost_total_d += mkt_sold_cost_d
                            mkt_pnl_d = mkt_cost_d - mkt_sold_cost_d
                            new_realized_pnl_d += mkt_pnl_d

                            total_qty_d = filled_qty_d + mkt_filled_qty_d
                            total_proceeds_d = filled_cost_d + mkt_cost_d
                            remain_qty_d = remain_qty_d - mkt_filled_qty_d
                            remain_notional_d = remain_qty_d * last_price_d
                            closed_after_mkt = (
                                remain_qty_d <= Decimal("0")
                                or remain_notional_d < self._to_decimal(MIN_POSITION_VALUE_KRW)
                            )
                            if closed_after_mkt and total_qty_d > Decimal("0"):
                                blended_exit_price_d = total_proceeds_d / total_qty_d
                                pnl_pct = float(((blended_exit_price_d / entry_price_d) - Decimal("1")) * Decimal("100")) if entry_price_d > 0 else 0.0
                                self.trade_db.record_trade(
                                    symbol=symbol,
                                    side="LONG",
                                    action="EXIT",
                                    quantity=float(total_qty_d),
                                    price=float(blended_exit_price_d),
                                    entry_price=float(entry_price_d),
                                    pnl=float(new_realized_pnl_d),
                                    pnl_pct=pnl_pct,
                                    reason=f"{reason} (limit+market)",
                                )
                                self.state_manager.clear_symbol_state(symbol)
                                return True

                # Partial exit fallback: keep state in sync with remaining position.
                remain_invest_d = invest_amount_d - sold_cost_total_d
                if remain_invest_d < Decimal("0"):
                    remain_invest_d = Decimal("0")
                state.update(
                    {
                        "invest_amount": float(remain_invest_d),
                        "realized_pnl": float(new_realized_pnl_d),
                    }
                )
                self.state_manager.update_symbol_state(symbol, state)
                self.trade_db.record_trade(
                    symbol=symbol,
                    side="LONG",
                    action="PARTIAL_EXIT",
                    quantity=float(filled_qty_d),
                    price=float(fill_price_d),
                    entry_price=float(entry_price_d),
                    pnl=float(this_sell_pnl_d),
                    pnl_pct=float(((fill_price_d / entry_price_d) - Decimal("1")) * Decimal("100")) if entry_price_d > 0 else 0.0,
                    reason=f"{reason} (partial fill)",
                )
                logger.warning(
                    f"[{symbol}] Exit partially filled. "
                    f"remain_qty={float(remain_qty_d):.8f}, remain_notional={float(remain_notional_d):,.0f}"
                )
            return False
        except Exception as e:
            logger.error(f"Error in _check_exit: {e}")
            self.health_manager.record_error(e)
            return False

    def _check_pyramiding(
        self,
        symbol: str,
        balance_coin: float,
        last_price: float,
        params: dict,
        candle: dict,
        state: dict,
    ) -> None:
        """Pyramiding add-on logic aligned with spot backtest constraints."""
        try:
            if not bool(params.get("ENABLE_PYRAMIDING", False)):
                return

            add_count = int(state.get("pyramid_add_count", 0))
            max_adds = int(params.get("PYRAMID_MAX_ADDS", 1))
            if max_adds < 0:
                max_adds = 0
            if add_count >= max_adds:
                return

            open_orders = self.client.fetch_open_orders(symbol)
            if open_orders and len(open_orders) > 0:
                logger.info(f"[{symbol}] Skip pyramiding: open orders exist ({len(open_orders)}).")
                return

            entry_price_d = self._to_decimal(state.get("entry_price", 0.0))
            entry_atr_d = self._to_decimal(state.get("entry_atr", 0.0))
            if entry_atr_d <= Decimal("0"):
                entry_atr_d = self._to_decimal(candle.get("atr", 0.0))
            if entry_price_d <= Decimal("0") or entry_atr_d <= Decimal("0"):
                return

            signal_close = candle.get("signal_close", np.nan)
            signal_slot = candle.get("signal_slot")
            if signal_slot and state.get("last_pyramid_slot") == signal_slot:
                return
            entry_upper = candle.get("entry_upper", np.nan)
            strength = int(candle.get("strength_filter", 0))
            trend_dir = int(candle.get("trend_direction", 0))
            vol_ratio = float(candle.get("volume_ratio", 1.0))
            rsi_value = candle.get("rsi", np.nan)
            natr_value = candle.get("natr", np.nan)
            risk_blocked = bool(candle.get("risk_blocked", False))

            use_vol = bool(params.get("USE_VOLUME_FILTER", False))
            vol_z_threshold = float(params.get("VOLUME_Z_THRESHOLD", params.get("VOLUME_THRESHOLD_MULT", 0.0)))
            rsi_entry_max_raw = params.get("RSI_ENTRY_MAX", 100.0)
            rsi_entry_max = 100.0 if rsi_entry_max_raw is None else float(rsi_entry_max_raw)
            natr_entry_min = float(params.get("NATR_ENTRY_MIN", 0.0))
            try:
                rsi_float = float(rsi_value) if rsi_value is not None else np.nan
            except (TypeError, ValueError):
                rsi_float = np.nan
            try:
                natr_float = float(natr_value) if natr_value is not None else np.nan
            except (TypeError, ValueError):
                natr_float = np.nan

            breakout = (
                np.isfinite(signal_close)
                and np.isfinite(entry_upper)
                and (self._to_decimal(signal_close) > self._to_decimal(entry_upper))
                and entry_upper > 0
            )
            vol_ok = (not use_vol) or (vol_ratio >= vol_z_threshold)
            rsi_ok = np.isfinite(rsi_float) and (rsi_float < rsi_entry_max)
            natr_ok = np.isfinite(natr_float) and (natr_float >= natr_entry_min)
            if risk_blocked:
                return
            if not (trend_dir == 1 and strength == 1 and breakout and vol_ok and rsi_ok and natr_ok):
                return

            trigger_atr_d = self._to_decimal(params.get("PYRAMID_TRIGGER_ATR", 1.8))
            step_atr_d = self._to_decimal(params.get("PYRAMID_STEP_ATR", 1.0))
            if trigger_atr_d < Decimal("0.1"):
                trigger_atr_d = Decimal("0.1")
            if step_atr_d < Decimal("0.1"):
                step_atr_d = Decimal("0.1")
            expected_trigger_d = entry_price_d + (
                entry_atr_d * (trigger_atr_d + (self._to_decimal(add_count) * step_atr_d))
            )
            next_trigger_d = self._to_decimal(state.get("next_pyramid_trigger", float(expected_trigger_d)))
            if next_trigger_d <= Decimal("0"):
                next_trigger_d = expected_trigger_d
            if self._to_decimal(signal_close) < next_trigger_d:
                return

            add_amount = self._calculate_pyramid_size(
                symbol=symbol,
                current_price=last_price,
                params=params,
                current_coin_balance=balance_coin,
                hurst=float(candle.get("hurst", 0.5)),
                natr=float(candle.get("natr", 0.0)),
            )
            add_amount_d = self._to_decimal(add_amount)
            min_order_d = self._to_decimal(MIN_ORDER_VALUE_KRW)
            if add_amount_d < min_order_d:
                return

            logger.info(
                f"PYRAMID {symbol} | Trigger={float(next_trigger_d):,.0f} "
                f"| SignalClose={signal_close:,.0f} | Add={float(add_amount_d):,.0f} KRW"
            )
            res = self._place_order_safe(symbol, "buy", price=float(add_amount_d))
            if not res:
                logger.error(f"Pyramiding buy order failed: {res}")
                return

            last_price_d = self._to_decimal(last_price)
            fill = self._wait_for_order_fill(
                symbol=symbol,
                side="buy",
                order=res,
                fallback_price=last_price_d,
                expected_cost=add_amount_d,
            )
            add_qty_d = self._to_decimal(fill.get("filled_qty", 0.0))
            fill_price_d = self._to_decimal(fill.get("avg_price", float(last_price_d)))
            fill_cost_d = self._to_decimal(fill.get("filled_cost", 0.0))
            if fill_cost_d <= Decimal("0") and add_qty_d > Decimal("0"):
                fill_cost_d = add_qty_d * fill_price_d
            if add_qty_d <= Decimal("0") or fill_cost_d < min_order_d:
                logger.warning(
                    f"[{symbol}] Pyramiding order not filled enough. "
                    f"filled_qty={float(add_qty_d):.8f}, filled_cost={float(fill_cost_d):,.0f}"
                )
                return
            if float(fill.get("fill_ratio", 1.0)) < 0.999:
                logger.warning(
                    f"[{symbol}] Pyramiding partially filled "
                    f"(ratio={fill.get('fill_ratio', 0.0):.2f})."
                )

            prev_qty_d = self._to_decimal(balance_coin)
            new_qty_d = prev_qty_d + add_qty_d
            prev_entry_d = self._to_decimal(state.get("entry_price", float(fill_price_d)))
            new_entry_d = (
                ((prev_entry_d * prev_qty_d) + (fill_price_d * add_qty_d)) / new_qty_d
                if new_qty_d > Decimal("0") else prev_entry_d
            )

            prev_invest_d = self._to_decimal(state.get("invest_amount", float(prev_entry_d * prev_qty_d)))
            new_invest_d = prev_invest_d + fill_cost_d
            new_add_count = add_count + 1
            new_next_trigger_d = new_entry_d + (
                entry_atr_d * (trigger_atr_d + (self._to_decimal(new_add_count) * step_atr_d))
            )

            state.update(
                {
                    "entry_price": float(new_entry_d),
                    "invest_amount": float(new_invest_d),
                    "highest_high": float(max(self._to_decimal(state.get("highest_high", 0.0)), last_price_d)),
                    "pyramid_add_count": int(new_add_count),
                    "next_pyramid_trigger": float(new_next_trigger_d),
                    "last_pyramid_slot": signal_slot,
                }
            )
            self.state_manager.update_symbol_state(symbol, state)
            self.trade_db.record_trade(
                symbol=symbol,
                side="LONG",
                action="PYRAMID",
                quantity=float(add_qty_d),
                price=float(fill_price_d),
                entry_price=float(new_entry_d),
                reason=f"Pyramid Trigger ({float(next_trigger_d):,.0f})",
            )
        except Exception as e:
            logger.error(f"Error in _check_pyramiding: {e}")
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
            
            final_weight_d = self._get_effective_risk_weight(
                symbol=symbol,
                params=params,
                hurst=hurst,
                natr=natr,
            )
            
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

    def _calculate_pyramid_size(
        self,
        symbol: str,
        current_price: float,
        params: dict,
        current_coin_balance: float,
        hurst: float,
        natr: float,
    ) -> float:
        """Calculate additional KRW amount for pyramiding."""
        try:
            total_krw, free_krw = self._fetch_balance_safe()
            total_krw_d = self._to_decimal(total_krw)
            free_krw_d = self._to_decimal(free_krw)
            min_order_d = self._to_decimal(MIN_ORDER_VALUE_KRW)
            hard_cap_d = self._to_decimal(MAX_INVEST_CAP_KRW)
            current_price_d = self._to_decimal(current_price)
            current_coin_d = self._to_decimal(current_coin_balance)

            state_map = self.state_manager._load()
            current_invested_total_d = Decimal("0")
            for s in self.symbols:
                st = state_map.get(s, {})
                current_invested_total_d += self._to_decimal(st.get("invest_amount", 0))

            estimated_total_equity_d = total_krw_d + current_invested_total_d
            effective_weight_d = self._get_effective_risk_weight(symbol, params, hurst=hurst, natr=natr)

            pyramid_ratio_d = self._to_decimal(params.get("PYRAMID_RISK_RATIO", 0.30))
            if pyramid_ratio_d < Decimal("0"):
                pyramid_ratio_d = Decimal("0")
            if pyramid_ratio_d > Decimal("0.95"):
                pyramid_ratio_d = Decimal("0.95")

            target_risk_d = effective_weight_d * pyramid_ratio_d
            if target_risk_d > Decimal("0.99"):
                target_risk_d = Decimal("0.99")

            add_amount_d = estimated_total_equity_d * target_risk_d

            max_capital_usage_d = self._to_decimal(params.get("MAX_CAPITAL_USAGE", float(MAX_INVEST_CAP_KRW)))
            if max_capital_usage_d <= Decimal("0"):
                max_capital_usage_d = hard_cap_d
            symbol_cap_d = min(max_capital_usage_d, hard_cap_d)
            current_symbol_notional_d = current_coin_d * current_price_d
            remaining_cap_d = symbol_cap_d - current_symbol_notional_d
            if remaining_cap_d < Decimal("0"):
                remaining_cap_d = Decimal("0")

            add_amount_d = min(add_amount_d, free_krw_d, remaining_cap_d)
            add_amount_d = self._to_krw(add_amount_d)
            if add_amount_d < min_order_d:
                return 0.0

            logger.info(
                f"Pyramiding size {symbol}: Add {float(add_amount_d):,.0f} KRW "
                f"(risk={float(target_risk_d*Decimal('100')):.2f}%, "
                f"remaining_cap={float(remaining_cap_d):,.0f})"
            )
            return float(add_amount_d)
        except Exception as e:
            logger.error(f"Pyramiding sizing error for {symbol}: {e}")
            return 0.0

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
            rsi_value = candle.get("rsi", np.nan)
            natr_value = candle.get("natr", np.nan)
            risk_blocked = bool(candle.get("risk_blocked", False))

            use_vol = params.get("USE_VOLUME_FILTER", False)
            vol_z_threshold = params.get("VOLUME_Z_THRESHOLD", params.get("VOLUME_THRESHOLD_MULT", 0.0))
            rsi_entry_max_raw = params.get("RSI_ENTRY_MAX", 100.0)
            rsi_entry_max = 100.0 if rsi_entry_max_raw is None else float(rsi_entry_max_raw)
            natr_entry_min = float(params.get("NATR_ENTRY_MIN", 0.0))
            try:
                rsi_float = float(rsi_value) if rsi_value is not None else np.nan
            except (TypeError, ValueError):
                rsi_float = np.nan
            try:
                natr_float = float(natr_value) if natr_value is not None else np.nan
            except (TypeError, ValueError):
                natr_float = np.nan

            is_uptrend = trend_dir == 1
            entry_upper_d = self._to_decimal(entry_upper)
            signal_close_d = self._to_decimal(signal_close)
            breakout = np.isfinite(signal_close) and np.isfinite(entry_upper) and (signal_close_d > entry_upper_d)
            strong_momentum = strength == 1
            vol_ok = (not use_vol) or (vol_ratio >= vol_z_threshold)
            rsi_ok = np.isfinite(rsi_float) and (rsi_float < rsi_entry_max)
            natr_ok = np.isfinite(natr_float) and (natr_float >= natr_entry_min)
            min_order_d = self._to_decimal(MIN_ORDER_VALUE_KRW)

            # Prevent duplicate entry in same signal slot
            state = self.state_manager.get_symbol_state(symbol)
            if signal_slot and state.get("entry_slot") == signal_slot:
                logger.info(f"[{symbol}] Skip duplicate entry for slot={signal_slot}")
                return

            if (
                is_uptrend
                and breakout
                and strong_momentum
                and vol_ok
                and rsi_ok
                and natr_ok
                and (not risk_blocked)
                and (entry_upper > 0)
            ):
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
                    if res:
                        fill = self._wait_for_order_fill(
                            symbol=symbol,
                            side="buy",
                            order=res,
                            fallback_price=last_price_d,
                            expected_cost=invest_amount_d,
                        )
                        quantity_d = self._to_decimal(fill.get("filled_qty", 0.0))
                        fill_price_d = self._to_decimal(fill.get("avg_price", float(last_price_d)))
                        fill_cost_d = self._to_decimal(fill.get("filled_cost", 0.0))
                        if fill_cost_d <= Decimal("0") and quantity_d > Decimal("0") and fill_price_d > Decimal("0"):
                            fill_cost_d = quantity_d * fill_price_d

                        if quantity_d <= Decimal("0") or fill_cost_d < min_order_d:
                            logger.warning(
                                f"[{symbol}] Entry order not filled enough. "
                                f"filled_qty={float(quantity_d):.8f}, filled_cost={float(fill_cost_d):,.0f}"
                            )
                            return

                        min_fill_ratio_d = self._to_decimal(
                            params.get("ENTRY_MIN_FILL_RATIO", SPOT_ENTRY_MIN_FILL_RATIO_DEFAULT),
                            default=str(SPOT_ENTRY_MIN_FILL_RATIO_DEFAULT),
                        )
                        if min_fill_ratio_d < Decimal("0"):
                            min_fill_ratio_d = Decimal("0")
                        if min_fill_ratio_d > Decimal("1"):
                            min_fill_ratio_d = Decimal("1")
                        fill_ratio_d = self._to_decimal(fill.get("fill_ratio", 1.0), default="1")
                        if fill_ratio_d < min_fill_ratio_d:
                            logger.warning(
                                f"[{symbol}] Underfilled entry rejected: "
                                f"fill_ratio={float(fill_ratio_d):.3f} < min={float(min_fill_ratio_d):.3f}. "
                                "Trying immediate flatten."
                            )
                            try:
                                flatten_res = self._place_market_order_safe(symbol, "sell", amount=float(quantity_d))
                                if flatten_res:
                                    self._wait_for_order_fill(
                                        symbol=symbol,
                                        side="sell",
                                        order=flatten_res,
                                        fallback_price=last_price_d,
                                        expected_qty=quantity_d,
                                        timeout_sec=max(8.0, SPOT_ORDER_FILL_TIMEOUT_SEC * 0.6),
                                    )
                            except Exception as flat_e:
                                logger.error(f"[{symbol}] Failed to flatten underfilled entry: {flat_e}")

                            refreshed_pos = self._fetch_position_safe(symbol)
                            remain_qty_d = self._to_decimal(refreshed_pos.get("amount", 0.0))
                            remain_notional_d = remain_qty_d * last_price_d
                            if remain_notional_d >= self._to_decimal(MIN_POSITION_VALUE_KRW):
                                self._bootstrap_state_for_open_position(
                                    symbol=symbol,
                                    amount=float(remain_qty_d),
                                    pos=refreshed_pos,
                                    current_price=float(last_price_d),
                                    params=params,
                                    atr=atr,
                                    prev_state=state,
                                )
                            return

                        if float(fill.get("fill_ratio", 1.0)) < 0.999:
                            logger.warning(
                                f"[{symbol}] Entry partially filled "
                                f"(ratio={fill.get('fill_ratio', 0.0):.2f})."
                            )

                        self.trade_db.record_trade(
                            symbol=symbol,
                            side="LONG",
                            action="ENTRY",
                            quantity=float(quantity_d),
                            price=float(fill_price_d),
                            reason=(
                                f"SignalClose({signal_close:,.0f}) > EntryUpper({entry_upper:,.0f}) "
                                f"@slot={signal_slot}"
                            ),
                        )

                        self.state_manager.update_symbol_state(
                            symbol,
                            {
                                "entry_price": float(fill_price_d),
                                "entry_time": self._get_reference_now_utc().isoformat(),
                                "entry_atr": atr,
                                "highest_high": float(fill_price_d),
                                "invest_amount": float(fill_cost_d),
                                "realized_pnl": 0.0,
                                "scale_out_done": False,
                                "stop_price_override": 0.0,
                                "pyramid_add_count": 0,
                                "next_pyramid_trigger": float(
                                    self._to_decimal(fill_price_d)
                                    + (self._to_decimal(atr) * self._to_decimal(params.get("PYRAMID_TRIGGER_ATR", 1.8)))
                                ) if self._to_decimal(atr) > Decimal("0") else 0.0,
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
                    if not rsi_ok:
                        reasons.append(f"RSIHigh({rsi_float:.2f})")
                    if not natr_ok:
                        reasons.append(f"NATRLow({natr_float:.2f})")
                    if risk_blocked:
                        reasons.append("RiskOffBlocked")
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
        now = self._get_reference_now_utc()
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
        now = self._get_reference_now_utc()
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


