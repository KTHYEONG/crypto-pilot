from __future__ import annotations

from typing import Optional, Sequence
from pathlib import Path
import json
import time

import pandas as pd
import numpy as np
import logging
from numba import njit


# Defaults for daily indicator columns when missing (futures-style: daily-only signals).
_SPOT_DAILY_INDICATOR_DEFAULTS: dict[str, float] = {
    "entry_upper": np.nan,
    "trend_direction": np.nan,
    "strength_filter": 0.0,
    "volume_ratio": 0.0,
    "atr": np.nan,
    "parabolic_sar": np.nan,
    "hurst": np.nan,
    "natr": np.nan,
    "rsi": np.nan,
}

DEBUG_LOG_PATH: Path = Path("debug-fb9f6d.log")


def _append_debug_log(
    message: str,
    data: dict[str, object],
    run_id: str,
    hypothesis_id: str,
    location: str,
) -> None:
    payload: dict[str, object] = {
        "sessionId": "fb9f6d",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        # Swallow logging errors silently to avoid impacting backtest
        return


class BacktestEngineFastSpot:
    """
    Numba-accelerated Backtest Engine for Spot (Long-Only)
    Reuses architecture from BacktestEngineFast (Futures): daily-only generate_signals,
    mapped to hourly via merge_index (no hourly indicator computation).
    """
    def __init__(
        self,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        strategy: object,
        backtest_func: object,
        initial_balance: float = 10_000_000,
        fee_rate: float = 0.0005,
        slippage_rate: float = 0.0003,
        merge_index_map: Optional[Sequence[int]] = None,
        precomputed_daily_df: Optional[pd.DataFrame] = None,
    ):
        # [MEMORY] Use shallow copy to prevent contaminating the global cached dataframe
        self.hourly_df = hourly_df.copy(deep=False)
        self.daily_df = daily_df.copy(deep=False)
        self.strategy = strategy
        self.backtest_func = backtest_func
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self._precomputed_daily_df = precomputed_daily_df

        # Injected by optimization script
        self.risk_per_trade = 0.99

        if merge_index_map is not None:
            self._merge_index_map = np.asarray(merge_index_map, dtype=np.int64)

        # [WARMUP] Extract from hourly df attrs
        self._warmup_bars = getattr(hourly_df, "attrs", {}).get("warmup_bars", 0)

        self.logger = logging.getLogger(__name__)

        # region agent log
        _append_debug_log(
            message="BacktestEngineFastSpot.__init__",
            data={"has_merge_index_map": merge_index_map is not None},
            run_id="pre-fix",
            hypothesis_id="H1",
            location="engine_fast_spot.py:62",
        )
        # endregion agent log

        self._prepare_data()

    def _resolve_trend_gate_mode(self):
        mode = str(self.strategy.params.get("TREND_GATE_MODE", "STRICT")).strip().upper()
        if mode not in {"STRICT", "SOFT", "OFF"}:
            mode = "STRICT"
        return mode

    @staticmethod
    def _trend_gate_mode_to_code(mode: str) -> int:
        # 0: STRICT, 1: SOFT, 2: OFF
        m = str(mode).strip().upper()
        if m == "SOFT":
            return 1
        if m == "OFF":
            return 2
        return 0

    @staticmethod
    def _shift_1bar(values: np.ndarray) -> np.ndarray:
        out = np.empty_like(values, dtype=np.float64)
        if len(out) == 0:
            return out
        out[0] = np.nan
        if len(out) > 1:
            out[1:] = values[:-1]
        return out

    def _extract_column(self, df: pd.DataFrame, col: str, default_value: float = np.nan) -> np.ndarray:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        return np.full(len(df), float(default_value), dtype=np.float64)
    
    def _prepare_data(self):
        """
        Prepare data with Daily Trend mapping.
        """
        # [OPTIMIZATION] Use pre-computed merge index if available
        if hasattr(self, '_merge_index_map'):
            self._prepare_data_with_index()
        else:
            self._prepare_data_with_merge() # Fallback for verify/live

    def _prepare_data_with_index(self) -> None:
        """
        [FAST PATH] Same as futures: use precomputed daily signals only; map to hourly via merge_index.
        Hourly df is raw OHLCV; no generate_signals on hourly.
        """
        # Use precomputed daily signals if provided (optimizer path), else compute from daily_df
        if self._precomputed_daily_df is not None:
            daily_src = self._precomputed_daily_df
        else:
            daily_src = self.strategy.generate_signals(self.daily_df.copy())

        # daily_indicators (n_daily, 3) for merge_idx lookup in numba
        daily_cols = ["trend_direction", "hurst", "natr"]
        self.daily_indicators = np.column_stack([
            self._shift_1bar(self._extract_column(daily_src, col, default_value=np.nan))
            for col in daily_cols
        ])
        self.merge_idx = np.asarray(self._merge_index_map, dtype=np.int64)
        self._trend_gate_mode_code = self._trend_gate_mode_to_code(self._resolve_trend_gate_mode())

        # Map daily indicator columns to hourly length (single shift like futures: shift(1) then map, no second shift)
        shifted_daily = daily_src.shift(1)
        midx = self._merge_index_map

        def _mapped(col: str) -> np.ndarray:
            arr = self._extract_column(shifted_daily, col, _SPOT_DAILY_INDICATOR_DEFAULTS[col])
            return arr[midx].astype(np.float64)

        self.entry_upper = _mapped("entry_upper")
        self.hourly_trend_dir = _mapped("trend_direction")
        self.strength_filter = _mapped("strength_filter")
        self.volume_ratio = _mapped("volume_ratio")
        self.atr = _mapped("atr")
        self.parabolic_sar = _mapped("parabolic_sar")
        self.hurst_h = _mapped("hurst")
        self.natr_h = _mapped("natr")
        self.rsi = _mapped("rsi")

        self._extract_arrays(from_daily_mapping=True)

    def _prepare_data_with_merge(self):
        """[FALLBACK] Slow merge for verification - assume indicators pre-computed"""
        if 'date_key' not in self.daily_df.columns:
             self.daily_df['date_key'] = pd.to_datetime(self.daily_df['datetime']).dt.strftime('%Y-%m-%d')
        if 'date_key' not in self.hourly_df.columns:
            self.hourly_df['date_key'] = pd.to_datetime(self.hourly_df['datetime']).dt.strftime('%Y-%m-%d')

        daily_cols = ["trend_direction", "hurst", "natr"]
        daily_data = []
        for col in daily_cols:
            vals = self._shift_1bar(self._extract_column(self.daily_df, col, default_value=np.nan))
            mapper = pd.Series(vals, index=self.daily_df["date_key"].values)
            daily_data.append(self.hourly_df["date_key"].map(mapper).to_numpy(dtype=np.float64))
        
        self.daily_indicators = np.column_stack(daily_data)
        self.merge_idx = np.arange(len(self.hourly_df), dtype=np.int64) 
        self._trend_gate_mode_code = self._trend_gate_mode_to_code(self._resolve_trend_gate_mode())
        
        self._extract_arrays()

    def _extract_arrays(self, from_daily_mapping: bool = False) -> None:
        """Extract numpy arrays: OHLC/datetime from hourly; signals from daily mapping or hourly."""
        self.close = pd.to_numeric(self.hourly_df["close"], errors="coerce").to_numpy(dtype=np.float64)
        self.high = pd.to_numeric(self.hourly_df["high"], errors="coerce").to_numpy(dtype=np.float64)
        self.low = pd.to_numeric(self.hourly_df["low"], errors="coerce").to_numpy(dtype=np.float64)
        self.open_prices = pd.to_numeric(self.hourly_df["open"], errors="coerce").to_numpy(dtype=np.float64)
        self.datetime_values = pd.to_datetime(self.hourly_df["datetime"]).values

        if not from_daily_mapping:
            # [FALLBACK] Indicators from hourly (verify/live when hourly has signals)
            self.entry_upper = self._shift_1bar(self._extract_column(self.hourly_df, "entry_upper", default_value=np.nan))
            self.hourly_trend_dir = self._shift_1bar(self._extract_column(self.hourly_df, "trend_direction", default_value=np.nan))
            self.strength_filter = self._shift_1bar(self._extract_column(self.hourly_df, "strength_filter", default_value=0.0))
            self.volume_ratio = self._shift_1bar(self._extract_column(self.hourly_df, "volume_ratio", default_value=0.0))
            self.atr = self._shift_1bar(self._extract_column(self.hourly_df, "atr", default_value=np.nan))
            self.parabolic_sar = self._shift_1bar(self._extract_column(self.hourly_df, "parabolic_sar", default_value=np.nan))
            self.hurst_h = self._shift_1bar(self._extract_column(self.hourly_df, "hurst", default_value=np.nan))
            self.natr_h = self._shift_1bar(self._extract_column(self.hourly_df, "natr", default_value=np.nan))
            self.rsi = self._shift_1bar(self._extract_column(self.hourly_df, "rsi", default_value=np.nan))

        # Release source DataFrames early to reduce trial memory pressure
        self.hourly_df = None
        self.daily_df = None
    
    def run(self, return_equity: bool = True) -> dict[str, object]:
        """
        Execute backtest using Numba-accelerated loop.
        When return_equity is False, 'equity_curve' is omitted from the returned dict.
        """
        # Extract strategy params
        exit_type = 1 if self.strategy.params.get("EXIT_TYPE") == "PARABOLIC_SAR" else 0
        stop_loss_type = 1 if self.strategy.params.get("STOP_LOSS_TYPE") == "ATR" else 0
        stop_loss_pct = self.strategy.params.get("STOP_LOSS_PCT", 0.03)
        atr_sl_mult = self.strategy.params.get("ATR_STOP_LOSS_MULT", 1.5)
        atr_mult = self.strategy.params.get("ATR_MULTIPLIER", 3.0)

        use_volume_filter = self.strategy.params.get("USE_VOLUME_FILTER", False)
        vol_threshold = self.strategy.params.get("VOLUME_THRESHOLD_MULT", 1.0)

        use_take_profit = self.strategy.params.get("USE_TAKE_PROFIT", False)
        tp_atr_mult = self.strategy.params.get("TAKE_PROFIT_ATR_MULT", 3.0)

        max_holding_bars = self.strategy.params.get("MAX_HOLDING_BARS", 999999)
        trailing_activation_atr = self.strategy.params.get("TRAILING_ACTIVATION_ATR", 0.0)
        time_exit_profit_threshold = self.strategy.params.get("TIME_EXIT_PROFIT_THRESHOLD", 1.4)

        warmup_bars = getattr(self, "_warmup_bars", 0)

        hurst_threshold = self.strategy.params.get(
            "HURST_TREND_THRESHOLD",
            self.strategy.params.get("STRONG_REGIME_HURST", 0.6),
        )
        strong_regime_natr = self.strategy.params.get("STRONG_REGIME_NATR", 1.0)
        natr_panic_threshold = self.strategy.params.get("PANIC_REGIME_NATR", 4.5)
        rsi_panic_threshold = self.strategy.params.get("RSI_EXIT_THRESHOLD", 94)
        use_dynamic_risk = self.strategy.params.get("USE_DYNAMIC_RISK", True)

        strong_regime_multiplier = self.strategy.params.get("STRONG_REGIME_MULTIPLIER", 1.3)
        panic_regime_multiplier = self.strategy.params.get("PANIC_REGIME_MULTIPLIER", 0.15)

        rsi_entry_max_raw = self.strategy.params.get("RSI_ENTRY_MAX", 100)
        rsi_entry_max = 100.0 if rsi_entry_max_raw is None else float(rsi_entry_max_raw)
        natr_entry_min = float(self.strategy.params.get("NATR_ENTRY_MIN", 0.0))

        enable_scale_out = bool(self.strategy.params.get("ENABLE_SCALE_OUT", False))
        scale_out_trigger_atr = float(self.strategy.params.get("SCALE_OUT_TRIGGER_ATR", 1.2))
        scale_out_ratio = float(self.strategy.params.get("SCALE_OUT_RATIO", 0.5))
        enable_breakeven = bool(self.strategy.params.get("ENABLE_BREAKEVEN", False))
        breakeven_buffer_pct = float(self.strategy.params.get("BREAKEVEN_BUFFER_PCT", 0.001))
        enable_pyramiding = bool(self.strategy.params.get("ENABLE_PYRAMIDING", False))
        pyramid_trigger_atr = float(self.strategy.params.get("PYRAMID_TRIGGER_ATR", 1.8))
        pyramid_step_atr = float(self.strategy.params.get("PYRAMID_STEP_ATR", 1.0))
        pyramid_risk_ratio = float(self.strategy.params.get("PYRAMID_RISK_RATIO", 0.30))
        pyramid_max_adds = int(self.strategy.params.get("PYRAMID_MAX_ADDS", 1))

        hurst_weak_threshold = self.strategy.params.get("WEAK_REGIME_HURST", 0.45)
        weak_regime_multiplier = self.strategy.params.get("WEAK_REGIME_MULTIPLIER", 0.6)
        enable_risk_off_hard_gate = bool(self.strategy.params.get("ENABLE_RISK_OFF_HARD_GATE", False))
        risk_off_exit_on_trigger = bool(self.strategy.params.get("RISK_OFF_EXIT_ON_TRIGGER", False))
        risk_off_cooldown_bars = int(self.strategy.params.get("RISK_OFF_COOLDOWN_BARS", 2))

        use_compounding = self.strategy.params.get("USE_COMPOUNDING", False)
        max_capital_usage = self.strategy.params.get("MAX_CAPITAL_USAGE", 100_000_000_000.0)

        # Run Numba loop with Daily Indicators + Merge Index
        trades, equity, final_bal = self.backtest_func(
            self.close,
            self.high,
            self.low,
            self.open_prices,
            self.entry_upper,
            self.hourly_trend_dir,
            self.strength_filter,
            self.volume_ratio,
            self.atr,
            self.parabolic_sar,
            self.hurst_h,
            self.natr_h,
            self.rsi,
            self.daily_indicators,
            self.merge_idx,
            self._trend_gate_mode_code,
            self.initial_balance,
            self.fee_rate,
            self.slippage_rate,
            exit_type,
            stop_loss_type,
            stop_loss_pct,
            atr_sl_mult,
            atr_mult,
            self.risk_per_trade,
            use_volume_filter,
            vol_threshold,
            use_take_profit,
            tp_atr_mult,
            max_holding_bars,
            trailing_activation_atr,
            time_exit_profit_threshold,
            use_dynamic_risk,
            hurst_threshold,
            strong_regime_natr,
            natr_panic_threshold,
            rsi_panic_threshold,
            strong_regime_multiplier,
            panic_regime_multiplier,
            hurst_weak_threshold,
            weak_regime_multiplier,
            enable_risk_off_hard_gate,
            risk_off_exit_on_trigger,
            risk_off_cooldown_bars,
            rsi_entry_max,
            natr_entry_min,
            enable_scale_out,
            scale_out_trigger_atr,
            scale_out_ratio,
            enable_breakeven,
            breakeven_buffer_pct,
            enable_pyramiding,
            pyramid_trigger_atr,
            pyramid_step_atr,
            pyramid_risk_ratio,
            pyramid_max_adds,
            warmup_bars,
            use_compounding,
            max_capital_usage,
        )

        # Calculate metrics
        total_return_pct = (final_bal - self.initial_balance) / self.initial_balance * 100

        # MDD calculation
        peak = np.maximum.accumulate(equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            mdd_series = np.where(peak > 0, (equity - peak) / peak * 100, 0.0)
            mdd_pct = np.min(mdd_series)
            if np.isnan(mdd_pct):
                mdd_pct = 0.0

        # Trade statistics
        num_trades = len(trades)
        if num_trades > 0:
            pnl_pcts = trades[:, 4]
            win_rate = (len(pnl_pcts[pnl_pcts > 0]) / num_trades * 100)
        else:
            win_rate = 0.0

        # Convert trades to DataFrame
        if num_trades > 0:
            entry_idx = np.clip(trades[:, 0].astype(np.int64), 0, len(self.datetime_values) - 1)
            exit_idx = np.clip(trades[:, 1].astype(np.int64), 0, len(self.datetime_values) - 1)

            trades_df = pd.DataFrame(
                {
                    "entry_idx": entry_idx,
                    "exit_idx": exit_idx,
                    "entry_time": pd.to_datetime(self.datetime_values[entry_idx]),
                    "exit_time": pd.to_datetime(self.datetime_values[exit_idx]),
                    "entry_price": trades[:, 2],
                    "exit_price": trades[:, 3],
                    "pnl_pct": trades[:, 4],
                    "pnl": trades[:, 5],
                    "duration_bars": trades[:, 6].astype(np.int64),
                },
            )
        else:
            trades_df = pd.DataFrame()

        out: dict[str, object] = {
            "total_return_pct": total_return_pct,
            "mdd_pct": mdd_pct,
            "total_trades": num_trades,
            "win_rate": win_rate,
            "final_balance": final_bal,
            "trades_df": trades_df,
        }
        if return_equity:
            out["equity_curve"] = equity

        return out


@njit(nogil=True, cache=True)
def backtest_loop_spot_numba(
    close, high, low, open_prices,
    entry_upper,
    hourly_trend_dir, strength_filter, volume_ratio, atr, parabolic_sar,
    hurst_h, natr_h, rsi,
    daily_indicators, merge_idx, trend_gate_mode,
    initial_balance, fee_rate, slippage_rate,
    exit_type,
    stop_loss_type, stop_loss_pct, atr_sl_mult,
    atr_mult, risk_per_trade,
    use_volume_filter, vol_threshold,
    use_take_profit, tp_atr_mult,
    max_holding_bars, trailing_activation_atr, time_exit_profit_threshold,
    use_dynamic_risk,
    hurst_threshold, strong_regime_natr, natr_panic_threshold, rsi_panic_threshold,
    strong_regime_multiplier, panic_regime_multiplier,
    hurst_weak_threshold, weak_regime_multiplier,
    enable_risk_off_hard_gate, risk_off_exit_on_trigger, risk_off_cooldown_bars,
    rsi_entry_max, natr_entry_min,
    enable_scale_out, scale_out_trigger_atr, scale_out_ratio,
    enable_breakeven, breakeven_buffer_pct,
    enable_pyramiding, pyramid_trigger_atr, pyramid_step_atr, pyramid_risk_ratio, pyramid_max_adds,
    warmup_bars,
    use_compounding, max_capital_usage
):
    """
    Numba Backtest Loop v16.0: Dual Timeframe Mapping & Logic Integration
    """
    n = len(close)
    balance = initial_balance
    coin = 0.0
    in_position = False
    pending_entry = False
    entry_price = 0.0
    entry_idx = 0
    highest = 0.0
    pos_atr = 0.0
    stop_price = 0.0
    tp_price = 0.0
    entry_cost = 0.0
    realized_revenue = 0.0
    scale_out_done = False
    pending_pyramid = False
    pending_pyramid_risk = 0.0
    next_pyramid_trigger = 0.0
    pyramid_add_count = 0
    risk_off_cooldown_remaining = 0
    
    max_trades = 30000
    trades = np.zeros((max_trades, 7))
    trade_count = 0
    
    equity_curve = np.zeros(n)
    exec_risk = risk_per_trade 

    for i in range(n):
        if i < warmup_bars:
            equity_curve[i] = balance
            continue
            
        c_open = open_prices[i]
        c_price = close[i]
        c_high = high[i]
        c_low = low[i]

        # --- 0. DUAL TIMEFRAME MAPPING & TREND GATE ---
        d_idx = merge_idx[i]
        # daily_indicators indices: 0: trend_direction, 1: hurst, 2: natr
        d_trend = daily_indicators[d_idx, 0]
        d_hurst = daily_indicators[d_idx, 1]
        d_natr = daily_indicators[d_idx, 2]

        # Resolve Trend Gate (STRICT: 0, SOFT: 1, OFF: 2)
        h_trend = hourly_trend_dir[i]
        final_trend = 0
        if trend_gate_mode == 2: # OFF
            final_trend = 1 if h_trend == 1 else 0
        elif trend_gate_mode == 1: # SOFT
            final_trend = 1 if (h_trend == 1 or d_trend == 1) else 0
        else: # STRICT
            final_trend = 1 if (h_trend == 1 and d_trend == 1) else 0

        risk_off = False
        if enable_risk_off_hard_gate:
            if final_trend != 1:
                risk_off = True
            elif d_natr > natr_panic_threshold: # Use Daily for regime
                risk_off = True
            elif d_hurst < hurst_weak_threshold:
                risk_off = True
        
        if risk_off:
            if risk_off_cooldown_bars > risk_off_cooldown_remaining:
                risk_off_cooldown_remaining = risk_off_cooldown_bars
        elif risk_off_cooldown_remaining > 0:
            risk_off_cooldown_remaining -= 1
        risk_blocked = risk_off or (risk_off_cooldown_remaining > 0)
        
        # --- 1. EXECUTION: BUY AT OPEN ---
        if pending_entry and not in_position:
            fill_price = c_open * (1 + slippage_rate)
            sig_atr = atr[i-1] if i > 0 else atr[i]
            
            if stop_loss_type == 1:
                stop_price = fill_price - (sig_atr * atr_sl_mult)
            else:
                stop_price = fill_price * (1 - stop_loss_pct)
            
            tp_price = fill_price + (sig_atr * tp_atr_mult) if use_take_profit else 0.0
            
            target_risk = exec_risk
            if target_risk > 0.99: target_risk = 0.99
            
            current_capital = balance if use_compounding else min(balance, initial_balance)
            cost = min(current_capital * target_risk, balance)
            
            current_exposure = coin * fill_price
            remaining_cap = max(0.0, max_capital_usage - current_exposure)
            cost = min(cost, remaining_cap)
            
            if cost > 0:
                coin = (cost * (1 - fee_rate)) / fill_price
                balance -= cost
                entry_cost = cost
                realized_revenue = 0.0
                in_position, pending_entry = True, False
                entry_price, entry_idx, highest, pos_atr = fill_price, i, fill_price, sig_atr
                scale_out_done, pyramid_add_count = False, 0
                next_pyramid_trigger = entry_price + (max(pos_atr, 1e-9) * pyramid_trigger_atr)
            else:
                pending_entry = False

        # --- 1b. EXECUTION: PYRAMID ADD ---
        if pending_pyramid and in_position:
            fill_price = c_open * (1 + slippage_rate)
            sig_atr = atr[i-1] if i > 0 else atr[i]
            target_risk = min(max(0.0, pending_pyramid_risk), 0.99)

            current_capital = balance if use_compounding else min(balance, initial_balance)
            cost = min(current_capital * target_risk, balance)
            current_exposure = coin * fill_price
            cost = min(cost, max(0.0, max_capital_usage - current_exposure))

            if cost > 0:
                add_qty = (cost * (1 - fee_rate)) / fill_price
                if add_qty > 0:
                    prev_coin = coin
                    coin = prev_coin + add_qty
                    balance -= cost
                    entry_cost += cost
                    entry_price = ((entry_price * prev_coin) + (fill_price * add_qty)) / max(coin, 1e-12)
                    pos_atr = ((pos_atr * prev_coin) + (sig_atr * add_qty)) / max(coin, 1e-12)
                    highest = max(highest, fill_price)
                    base_stop = entry_price - (pos_atr * atr_sl_mult) if stop_loss_type == 1 else entry_price * (1 - stop_loss_pct)
                    stop_price = max(stop_price, base_stop)
                    if use_take_profit:
                        tp_price = entry_price + (pos_atr * tp_atr_mult)
                    pyramid_add_count += 1
                    next_pyramid_trigger = entry_price + (max(pos_atr, 1e-9) * (pyramid_trigger_atr + (float(pyramid_add_count) * pyramid_step_atr)))
            pending_pyramid = False
        
        # --- 2. EXECUTION: EXIT CHECKS ---
        if in_position:
            exit_triggered = False
            exit_price = 0.0
            
            current_stop = stop_price
            if exit_type == 1 and parabolic_sar[i] > 0:
                current_stop = max(stop_price, parabolic_sar[i])

            if c_low <= current_stop:
                exit_price = (c_open if c_open < current_stop else current_stop) * (1 - slippage_rate)
                exit_triggered = True
            elif enable_scale_out and (not scale_out_done) and coin > 0 and pos_atr > 0:
                scale_out_price = entry_price + (pos_atr * scale_out_trigger_atr)
                if c_high >= scale_out_price:
                    scale_qty = coin * scale_out_ratio
                    remain_qty = coin - scale_qty
                    if scale_qty > 0 and remain_qty >= 1e-12:
                        scale_revenue = scale_qty * scale_out_price * (1 - fee_rate)
                        balance += scale_revenue
                        realized_revenue += scale_revenue
                        coin, scale_out_done = remain_qty, True
                        if enable_breakeven:
                            stop_price = max(stop_price, entry_price * (1.0 + (2.0 * fee_rate) + slippage_rate + breakeven_buffer_pct))
                        if use_take_profit and tp_price > 0 and tp_price <= scale_out_price:
                            tp_price = scale_out_price + (pos_atr * 0.25)

            if not exit_triggered and use_take_profit and tp_price > 0 and c_high >= tp_price:
                exit_price, exit_triggered = tp_price, True
            elif not exit_triggered:
                if enable_risk_off_hard_gate and risk_off_exit_on_trigger and risk_blocked:
                    exit_price, exit_triggered = c_price * (1 - slippage_rate), True
                elif rsi[i] > rsi_panic_threshold:
                    exit_price, exit_triggered = c_price * (1 - slippage_rate), True
                elif i > 0 and final_trend <= 0:
                    exit_price, exit_triggered = c_price * (1 - slippage_rate), True
                elif (i - entry_idx) >= max_holding_bars:
                    if (c_price - entry_price) / (pos_atr if pos_atr > 0 else 1e-9) <= time_exit_profit_threshold:
                        exit_price, exit_triggered = c_price * (1 - slippage_rate), True

            if exit_triggered:
                revenue = coin * exit_price * (1 - fee_rate)
                balance += revenue 
                realized_revenue += revenue
                pnl = realized_revenue - entry_cost
                if trade_count < max_trades:
                    trades[trade_count] = [float(entry_idx), float(i), entry_price, exit_price, (pnl / max(entry_cost, 1e-9)) * 100.0, pnl, float(i - entry_idx)]
                    trade_count += 1
                coin, in_position = 0.0, False
                entry_cost, realized_revenue, scale_out_done, pending_pyramid = 0.0, 0.0, False, False
                pyramid_add_count, next_pyramid_trigger = 0, 0.0
            else:
                highest = max(highest, c_high)
                if exit_type == 0:
                    if (highest - entry_price) / (pos_atr if pos_atr > 0 else 1e-9) >= trailing_activation_atr:
                        stop_price = max(stop_price, highest - (pos_atr * atr_mult))

                if enable_pyramiding and (not pending_pyramid) and pyramid_add_count < pyramid_max_adds and i < n - 1:
                    can_add = True
                    if strength_filter[i] == 0 or (use_volume_filter and volume_ratio[i] < vol_threshold): can_add = False
                    elif rsi[i] >= rsi_entry_max or d_natr < natr_entry_min or final_trend != 1 or risk_blocked: can_add = False

                    if can_add and c_price > next_pyramid_trigger:
                        reg_m = 1.0
                        if use_dynamic_risk:
                            if d_natr > natr_panic_threshold: reg_m = panic_regime_multiplier
                            elif d_hurst > hurst_threshold and d_natr > strong_regime_natr: reg_m = strong_regime_multiplier
                            elif d_hurst < hurst_weak_threshold: reg_m = weak_regime_multiplier
                        pending_pyramid_risk, pending_pyramid = risk_per_trade * reg_m * pyramid_risk_ratio, True

        # --- 3. SIGNAL: ENTRY DETECTION ---
        elif not in_position and not pending_entry:
            can_signal = True
            if np.isnan(entry_upper[i]) or strength_filter[i] == 0: can_signal = False
            elif use_volume_filter and volume_ratio[i] < vol_threshold: can_signal = False
            elif rsi[i] >= rsi_entry_max or d_natr < natr_entry_min or risk_blocked: can_signal = False
            
            if can_signal and final_trend == 1 and c_price > entry_upper[i]:
                reg_m = 1.0
                if use_dynamic_risk:
                    if d_natr > natr_panic_threshold: reg_m = panic_regime_multiplier
                    elif d_hurst > hurst_threshold and d_natr > strong_regime_natr: reg_m = strong_regime_multiplier
                    elif d_hurst < hurst_weak_threshold: reg_m = weak_regime_multiplier
                exec_risk = risk_per_trade * reg_m
                if i < n - 1: pending_entry = True
        
        equity_curve[i] = balance + (coin * c_price)

    return trades[:trade_count], equity_curve, balance + (coin * close[-1])
