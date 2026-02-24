
from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
import logging
from numba import njit
from config.settings import TRADING_FEE_RATE, SLIPPAGE_RATE, FUNDING_FEE_RATE

class BacktestEngineFast:
    """
    Numba-accelerated Backtest Engine (5-10x faster)
    """
    def __init__(
        self,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        strategy,
        initial_balance: float = 1_000_000,
        merge_index_map=None,
        precomputed_daily_df: Optional[pd.DataFrame] = None,
    ):
        # Shallow copy (deep=False): new DataFrame object so _prepare_data can add columns without
        # mutating the caller's DataFrame (e.g. same df reused across optimization trials).
        # Underlying array data is shared—do not mutate existing columns in place.
        self.hourly_df = hourly_df.copy(deep=False)
        self.daily_df = daily_df.copy(deep=False)
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self._precomputed_daily_df = precomputed_daily_df

        # injected by optimization script
        self.leverage = 1
        self.risk_per_trade = 0.02
        
        self.fee_rate = TRADING_FEE_RATE
        self.slippage_rate = SLIPPAGE_RATE

        # Optional fast merge index injection (must be set before _prepare_data call)
        if merge_index_map is not None:
            self._merge_index_map = merge_index_map
        
        self.risk_limits = {'max_daily_loss': 0.05}
        self.logger = logging.getLogger(__name__)
        self._prepare_data()
    
    def _prepare_data(self):
        # [OPTIMIZATION] Use pre-computed merge index if available (set by optimize script)
        # This eliminates expensive pd.merge on every trial
        if hasattr(self, '_merge_index_map'):
            # Fast path: Use pre-computed index mapping
            self._prepare_data_with_index()
        else:
            # Fallback: Traditional merge (for verify/live scripts)
            self._prepare_data_with_merge()
    
    def _prepare_data_with_index(self):
        """
        [FAST PATH] Use pre-computed merge index to avoid pd.merge overhead.
        This is 10-50x faster than pd.merge for large datasets.
        """
        # Use pre-computed signals if available (cache hit path), otherwise compute
        if self._precomputed_daily_df is not None:
            self.daily_df = self._precomputed_daily_df
        else:
            self.daily_df = self.strategy.generate_signals(self.daily_df)
        
        # Filter essential columns only
        exclude_cols = {'date_key', 'datetime', 'date', 'open', 'high', 'low', 'close', 'volume'}
        indicator_cols = [c for c in self.daily_df.columns if c not in exclude_cols]
        
        # Shift(1) to prevent lookahead
        shifted_daily = self.daily_df[indicator_cols].shift(1)
        
        # Shallow copy (deep=False): new object so we add daily_* columns to merged_df only;
        # underlying data shared with hourly_df for memory/speed. Align daily -> hourly via index map.
        self.merged_df = self.hourly_df.copy(deep=False)
        
        # Map daily indicators to hourly using pre-computed indices
        for col in indicator_cols:
            # Use NumPy array indexing for maximum speed
            daily_values = shifted_daily[col].values
            mapped_values = daily_values[self._merge_index_map]
            self.merged_df[f'daily_{col}'] = mapped_values
    
    def _prepare_data_with_merge(self):
        """
        [FALLBACK] Traditional pd.merge approach for compatibility.
        Used when merge index is not pre-computed (verify/live scripts).
        """
        # Pre-calculated keys check
        if 'date_key' not in self.daily_df.columns:
             self.daily_df['date_key'] = pd.to_datetime(self.daily_df['datetime']).dt.strftime('%Y-%m-%d')
        
        # Use pre-computed signals if available (cache hit path), otherwise compute
        if self._precomputed_daily_df is not None:
            self.daily_df = self._precomputed_daily_df
        else:
            self.daily_df = self.strategy.generate_signals(self.daily_df)
        
        # Filter essential columns
        exclude_cols = {'date_key', 'datetime', 'date', 'open', 'high', 'low', 'close', 'volume'}
        indicator_cols = [c for c in self.daily_df.columns if c not in exclude_cols]
        
        # Shift & Rename
        shifted_daily = self.daily_df[indicator_cols].shift(1)
        shifted_daily.columns = [f'daily_{c}' for c in indicator_cols]
        shifted_daily['date_key'] = self.daily_df['date_key']
        
        if 'date_key' not in self.hourly_df.columns:
            self.hourly_df['date_key'] = pd.to_datetime(self.hourly_df['datetime']).dt.strftime('%Y-%m-%d')
            
        # Left Join: Hourly <- Daily
        self.merged_df = pd.merge(self.hourly_df, shifted_daily, on='date_key', how='left')
    
    def run(self):
        self.logger.debug(f"Running FAST backtest for {self.strategy.name}...")
        
        df = self.merged_df
        
        # Daily signals are already aligned to prior day in _prepare_data_* via shift(1).
        # Do not shift again here to avoid extra 1-bar latency.
                
        # Extract all columns as numpy arrays for speed
        n = len(df)
        
        # Price columns
        open_prices = df['open'].values # [NEW] For Entry
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        # volume = df['volume'].values # Not needed for daily ratio check
        
        # Entry signals (Donchian or other)
        # Columns guaranteed by strategy.generate_signals
        entry_upper = df['daily_entry_upper'].values
        entry_lower = df['daily_entry_lower'].values
        
        # Trend filter
        trend_dir = df['daily_trend_direction'].values
        
        # Strength filter
        strength_filter = df['daily_strength_filter'].values
        
        # Volume Filter (Ratio)
        volume_ratio = df['daily_volume_ratio'].values
        
        # ATR for risk
        atr = df['daily_atr'].values
        
        # Strategy params
        exit_type_str = self.strategy.params.get('EXIT_TYPE', 'ATR')  # 'ATR', 'PARABOLIC_SAR'
        exit_type = 1 if exit_type_str == 'PARABOLIC_SAR' else 0  # 0: ATR, 1: SAR
        
        stop_loss_type = 1 if self.strategy.params.get('STOP_LOSS_TYPE', 'FIXED') == 'ATR' else 0 # 0: FIXED, 1: ATR
        stop_loss_pct = self.strategy.params.get('STOP_LOSS_PCT', 0.03)
        atr_sl_mult = self.strategy.params.get('ATR_STOP_LOSS_MULT', 1.5)
        
        atr_mult = self.strategy.params.get('ATR_MULTIPLIER', 3.0) # For Trailing Stop
        leverage = self.leverage
        
        # New Params for Volume & TP
        use_volume_filter = self.strategy.params.get('USE_VOLUME_FILTER', False)
        vol_threshold = self.strategy.params.get('VOLUME_THRESHOLD_MULT', 1.0)
        
        use_take_profit = self.strategy.params.get('USE_TAKE_PROFIT', False)
        # Use Futures specific TP if available
        tp_atr_mult = self.strategy.params.get('TAKE_PROFIT_ATR_MULT_FUTURES', self.strategy.params.get('TAKE_PROFIT_ATR_MULT', 3.0))
        
        # Parabolic SAR (optional exit signal)
        parabolic_sar = df['daily_parabolic_sar'].values
        
        # Extract timestamps for funding fee calculation
        timestamps = df["timestamp"].values  # milliseconds

        # Funding rate: time-series if column exists (direction applied in Numba via pos_side), else constant fallback
        if "funding_rate" in df.columns:
            funding_rates = np.asarray(df["funding_rate"].values, dtype=np.float64)
            # NaN at funding hour -> treat as 0 (no payment)
            funding_rates = np.nan_to_num(funding_rates, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            funding_rates = np.full(n, FUNDING_FEE_RATE, dtype=np.float64)
        
        # [NEW] Time-Based Exit & Trailing Activation
        max_holding_bars = self.strategy.params.get('MAX_HOLDING_BARS', 999999)  # Default: No limit
        trailing_activation_atr = self.strategy.params.get('TRAILING_ACTIVATION_ATR', 0.0)  # Default: Immediate activation
        time_exit_profit_threshold = self.strategy.params.get('TIME_EXIT_PROFIT_THRESHOLD', 0.5)  # Default: 0.5 ATR profit required to hold
        enable_trend_exit = self.strategy.params.get('ENABLE_TREND_EXIT', True)
        
        # [WARMUP OPTIMIZATION] Warmup in execution (hourly) bar count; fallback converts daily bars to hourly
        warmup_bars = getattr(df, "attrs", {}).get(
            "warmup_bars",
            self.strategy.get_required_warmup(freq="hourly"),
        )
        
        # [NEW] RSI for Panic Exit
        # Use existing rsi if available, else fill 50
        if 'daily_rsi' in df.columns:
            rsi = df['daily_rsi'].values
        else:
            rsi = np.full(n, 50.0)
            
        # Long/short independent; fallback to legacy RSI_EXIT_THRESHOLD (long=K, short=100-K)
        _legacy = self.strategy.params.get("RSI_EXIT_THRESHOLD", 80.0)
        rsi_long_exit = self.strategy.params.get("RSI_LONG_EXIT_THRESHOLD", _legacy)
        rsi_short_exit = self.strategy.params.get("RSI_SHORT_EXIT_THRESHOLD", 100.0 - _legacy)
        
        # [NEW] Dynamic Risk Sizing (Regime-based)
        # Extract Hurst & NATR for regime detection
        if 'daily_hurst' in df.columns:
            hurst = df['daily_hurst'].values
        else:
            hurst = np.full(n, 0.5)  # Neutral (random walk)
            
        if 'daily_natr' in df.columns:
            natr = df['daily_natr'].values
        else:
            natr = np.full(n, 1.0)  # Neutral volatility
        
        # Dynamic Risk Multipliers (Optimizable)
        use_dynamic_risk = self.strategy.params.get('USE_DYNAMIC_RISK', False)
        strong_regime_hurst = self.strategy.params.get('STRONG_REGIME_HURST', 0.6)
        strong_regime_natr = self.strategy.params.get('STRONG_REGIME_NATR', 1.5)
        strong_regime_multiplier = self.strategy.params.get('STRONG_REGIME_MULTIPLIER', 1.5)
        
        weak_regime_hurst = self.strategy.params.get('WEAK_REGIME_HURST', 0.55)
        weak_regime_multiplier = self.strategy.params.get('WEAK_REGIME_MULTIPLIER', 0.5)
        
        panic_regime_natr = self.strategy.params.get('PANIC_REGIME_NATR', 4.0)
        panic_regime_multiplier = self.strategy.params.get('PANIC_REGIME_MULTIPLIER', 0.25)
        
        # [OPTIMIZATION] Extract timestamps for fast lookup (avoid iloc)
        datetime_values = df['datetime'].values
        
        # [NEW] Compounding Control
        use_compounding = self.strategy.params.get('USE_COMPOUNDING', False)
        max_capital_usage = self.strategy.params.get('MAX_CAPITAL_USAGE', 1_000_000)
        
        # Run Numba-accelerated loop
        trades, final_balance, equity_curve = backtest_loop_numba(
            close, high, low, open_prices, volume_ratio,
            entry_upper, entry_lower,
            trend_dir, strength_filter, atr, parabolic_sar, rsi,
            hurst, natr,
            self.initial_balance,
            leverage,
            self.fee_rate,
            self.slippage_rate,
            exit_type, stop_loss_type, stop_loss_pct, atr_sl_mult,
            atr_mult,
            self.risk_per_trade,
            use_volume_filter, vol_threshold,
            use_take_profit, tp_atr_mult,
            timestamps, funding_rates,
            max_holding_bars, trailing_activation_atr, time_exit_profit_threshold,
            enable_trend_exit,
            rsi_long_exit, rsi_short_exit,
            use_dynamic_risk, strong_regime_hurst, strong_regime_natr, strong_regime_multiplier,
            weak_regime_hurst, weak_regime_multiplier,
            panic_regime_natr, panic_regime_multiplier,
            warmup_bars,
            use_compounding,    # [NEW]
            max_capital_usage   # [NEW]
        )
        
        self.balance = final_balance
        self._equity_curve = equity_curve

        # [MEMORY] Release large DataFrames before processing results
        self.merged_df = None
        self.hourly_df = None
        self.daily_df = None
        
        # Convert trades to list of dicts; Numba returns trades[:trade_count] so no padding rows
        self.trades = []
        for i in range(len(trades)):
            entry_idx = int(trades[i][0])
            exit_idx = int(trades[i][1])
            self.trades.append({
                "entry_time": datetime_values[entry_idx],
                "exit_time": datetime_values[exit_idx],
                "side": "LONG" if trades[i][2] == 1 else "SHORT",
                "entry_price": trades[i][3],
                "exit_price": trades[i][4],
                "pnl": trades[i][5],
                "amount": trades[i][6],
                "entry_fee": trades[i][7],
            })
        
        result = self.get_results()
        return result
    
    def get_results(self):
        # [ROBUSTNESS] Exploded strategy (inf/nan balance): return invalid result so optimizer penalizes it
        if not np.isfinite(self.balance):
            self.logger.warning("Balance is non-finite (exploded strategy). Returning invalid result.")
            return {
                'total_trades': 0,
                'win_trades': 0,
                'loss_trades': 0,
                'win_rate': 0,
                'total_return_pct': 0,
                'final_balance': self.initial_balance,
                'mdd_pct': 0,
                'trades_df': pd.DataFrame()
            }

        total_return = self.balance - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100

        if not self.trades:
            return {
                'total_trades': 0,
                'win_trades': 0,
                'loss_trades': 0,
                'win_rate': 0,
                'total_return_pct': 0,
                'final_balance': self.initial_balance,
                'mdd_pct': 0,
                'trades_df': pd.DataFrame()
            }
        
        trades_df = pd.DataFrame(self.trades)

        # MDD from bar-level equity (includes unrealized P&L during open positions)
        equity = getattr(self, "_equity_curve", None)
        if equity is not None and len(equity) > 0 and np.isfinite(equity).all():
            running_max = np.maximum.accumulate(equity)
            running_max[running_max == 0] = 1e-9
            drawdown = (equity - running_max) / running_max * 100
            drawdown = np.nan_to_num(drawdown, nan=0.0)
            mdd = float(drawdown.min())
        else:
            # Fallback: trade-level cumulative (no unrealized)
            cumulative = [self.initial_balance]
            for pnl in trades_df["pnl"]:
                cumulative.append(cumulative[-1] + pnl)
            cumulative = np.array(cumulative)
            if not np.isfinite(cumulative).all():
                mdd = 0.0
            else:
                running_max = np.maximum.accumulate(cumulative)
                running_max[running_max == 0] = 1e-9
                drawdown = (cumulative - running_max) / running_max * 100
                drawdown = np.nan_to_num(drawdown, nan=0.0)
                mdd = float(drawdown.min())

        # ROE per trade = (PnL - entry_fee) / Margin Used. entry_fee is deducted at entry but not in pnl.
        entry_fee_arr = trades_df["entry_fee"].values
        true_pnl = trades_df["pnl"].values - entry_fee_arr
        entry_p = trades_df["entry_price"].values
        amount_arr = trades_df["amount"].values
        leverage = float(self.leverage)
        margin_used = amount_arr * entry_p / leverage
        margin_used = np.where(np.isfinite(margin_used) & (margin_used > 0), margin_used, np.nan)

        pnl_cumsum = trades_df["pnl"].cumsum().shift(1).fillna(0)
        trades_df["balance_before"] = (self.initial_balance + pnl_cumsum).replace(0, 1e-9)
        balance_before = trades_df["balance_before"].values

        denom = np.where(
            np.isfinite(margin_used) & (margin_used > 0),
            margin_used,
            np.asarray(balance_before, dtype=np.float64),
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            trades_df["pnl_pct"] = (true_pnl / denom) * 100
        trades_df["pnl_pct"] = trades_df["pnl_pct"].replace([np.inf, -np.inf], 0).fillna(0)
        
        win_trades = len(trades_df[trades_df["pnl"] > 0])
        loss_trades = len(trades_df[trades_df["pnl"] <= 0])
        win_rate = (win_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0

        return {
            'total_trades': len(trades_df),
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'final_balance': self.balance,
            'mdd_pct': mdd,
            'trades_df': trades_df
        }



@njit(nogil=True, cache=True)
def backtest_loop_numba(
    close, high, low, open_prices, volume_ratio,
    entry_upper, entry_lower,
    trend_dir, strength_filter, atr, parabolic_sar, rsi, # [NEW]
    hurst, natr, # [NEW] Regime indicators
    initial_balance, leverage, fee_rate, slippage_rate,
    exit_type, stop_loss_type, stop_loss_pct, atr_sl_mult,
    atr_mult,
    risk_per_trade,
    use_volume_filter, vol_threshold,
    use_take_profit, tp_atr_mult,
    timestamps, funding_rates,
    max_holding_bars, trailing_activation_atr, time_exit_profit_threshold,
    enable_trend_exit,
    rsi_long_exit, rsi_short_exit,
    use_dynamic_risk, strong_regime_hurst, strong_regime_natr, strong_regime_multiplier,
    weak_regime_hurst, weak_regime_multiplier,
    panic_regime_natr, panic_regime_multiplier, # [NEW] Dynamic Risk
    warmup_bars,  # [WARMUP] Number of bars to skip at start for indicator warmup
    use_compounding,   # [NEW] Bool
    max_capital_usage  # [NEW] Float
):
    """
    Numba JIT-compiled backtest loop for Futures (Long/Short).
    Simulation premise:
    - Entry: Intra-bar breakout allowed. At bar i we use high[i]/low[i] to detect
      breakout and fill at max(open, level) (long) or min(open, level) (short).
      Real-time: you would not know bar high/low until bar close; this assumes
      tick-level touch within the bar is observable at bar processing time.
    - Exit: Slippage on all exits. Sequential stop logic. Funding at UTC 0/8/16.
    - [v16] Compounding control & capital caps.
    """
    n = len(close)
    balance = initial_balance
    equity_curve = np.zeros(n)

    # Position state
    in_position = False
    pending_entry = False 
    pending_side = 0
    
    pos_side = 0  # 1: LONG, -1: SHORT
    entry_price = 0.0
    entry_idx = 0
    amount = 0.0
    entry_fee_stored = 0.0
    highest = 0.0
    lowest = 0.0
    pos_atr = 0.0
    stop_price = 0.0 # Initial Stop Loss
    tp_price = 0.0 
    
    # Funding Fee Tracking (UTC-based: 00:00, 08:00, 16:00)
    last_funding_hour = -1
    
    # Trades storage (max 30000 trades)
    max_trades = 30000
    trades = np.zeros((max_trades, 8))  # [entry_idx, exit_idx, side, entry_p, exit_p, pnl, amount, entry_fee]
    trade_count = 0
    
    exec_risk = risk_per_trade # Local execution risk
    
    for i in range(n):
        # [WARMUP] Skip trading during warmup period
        if i < warmup_bars:
            equity_curve[i] = initial_balance
            continue

        # [SAFETY] Bankruptcy Check
        if balance <= 0:
            equity_curve[i] = balance
            break

        c_open = open_prices[i]
        c_price = close[i]
        c_high = high[i]
        c_low = low[i]
        current_timestamp = timestamps[i]
        execute_intra_bar = False
        bar_processed = False  # No re-entry on same bar after exit (Zombie fix)

        # --- 1. POSITION MANAGEMENT (Exits & Funding) ---
        if in_position:
            # A. Funding Fee (UTC 00, 08, 16)
            current_hour_utc = int((current_timestamp // 1000) % 86400 // 3600)
            is_funding_hour = (current_hour_utc in (0, 8, 16))
            
            if is_funding_hour and last_funding_hour != current_hour_utc:
                notional_value = amount * c_price
                rate = funding_rates[i]
                if np.isnan(rate):
                    rate = 0.0
                # Exchange convention: rate = what longs pay; long pays when rate>0, short receives
                funding_cost = notional_value * rate * pos_side
                balance -= funding_cost
                last_funding_hour = current_hour_utc
                
                # Bankruptcy check from Funding (forced liquidation)
                # On liquidation do NOT add margin+pnl back: balance must go to 0 (or stay <= 0).
                if balance <= 0:
                    exit_price = c_price
                    if pos_side == 1: pnl = (exit_price - entry_price) * amount
                    else:             pnl = (entry_price - exit_price) * amount
                    
                    exit_fee = amount * exit_price * fee_rate
                    pnl -= exit_fee
                    balance = 0.0
                    
                    if trade_count < max_trades:
                        trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl, amount, entry_fee_stored]
                        trade_count += 1
                    in_position = False
                    last_funding_hour = -1  # Reset so next position can be charged in same funding hour
                    break
            
            # Start-of-bar stop for same-bar exit (guard: no intra-bar time-order assumption)
            start_of_bar_stop = stop_price
            
            # [SEQ-1] Update High/Low & Trailing Stop in real time (exit_triggered-agnostic)
            if pos_side == 1:
                if c_high > highest:
                    highest = c_high
                if exit_type == 0:
                    unreal_profit = (highest - entry_price) / pos_atr if pos_atr > 0 else 0
                    if unreal_profit >= trailing_activation_atr:
                        new_stop = highest - (pos_atr * atr_mult)
                        if new_stop > stop_price:
                            stop_price = new_stop
            else:
                if c_low < lowest:
                    lowest = c_low
                if exit_type == 0:
                    unreal_profit = (entry_price - lowest) / pos_atr if pos_atr > 0 else 0
                    if unreal_profit >= trailing_activation_atr:
                        new_stop = lowest + (pos_atr * atr_mult)
                        if new_stop < stop_price:
                            stop_price = new_stop
            
            # B. Exit Checks (use start_of_bar_stop so same-bar exit does not use same-bar-updated stop)
            exit_triggered = False
            exit_price = 0.0
            
            # [SEQ-2] Stop Loss (Gap-Adjusted Realism)
            current_stop = start_of_bar_stop
            # SAR valid only when on correct side of price (bullish: SAR below price; bearish: SAR above)
            if exit_type == 1 and parabolic_sar[i] > 0:
                if pos_side == 1 and parabolic_sar[i] < c_open:
                    current_stop = max(start_of_bar_stop, parabolic_sar[i])
                elif pos_side == -1 and parabolic_sar[i] > c_open:
                    current_stop = min(start_of_bar_stop, parabolic_sar[i])
                # else: wrong side → keep ATR stop only

            if pos_side == 1:  # Long exit
                if c_open < current_stop:
                    # Gap down: liquidate at open (worst fill)
                    exit_price = c_open * (1 - slippage_rate)
                    exit_triggered = True
                elif c_low <= current_stop:
                    # Intraday touch: fill at stop level
                    exit_price = current_stop * (1 - slippage_rate)
                    exit_triggered = True
            else:  # Short exit
                if c_open > current_stop:
                    # Gap up: cover at open (worst fill)
                    exit_price = c_open * (1 + slippage_rate)
                    exit_triggered = True
                elif c_high >= current_stop:
                    exit_price = current_stop * (1 + slippage_rate)
                    exit_triggered = True

            # [SEQ-3] Take Profit (Gap-Adjusted)
            if not exit_triggered and use_take_profit and tp_price > 0:
                if pos_side == 1:
                    if c_open > tp_price:
                        # Gap up through TP: conservative fill at open with slippage
                        exit_price = c_open * (1 - slippage_rate)
                        exit_triggered = True
                    elif c_high >= tp_price:
                        exit_price = tp_price
                        exit_triggered = True
                else:
                    if c_open < tp_price:
                        exit_price = c_open * (1 + slippage_rate)
                        exit_triggered = True
                    elif c_low <= tp_price:
                        exit_price = tp_price
                        exit_triggered = True

            # [SEQ-4] Conditional Market Exits
            if not exit_triggered:
                # Time Exit
                bars_held = i - entry_idx
                if bars_held >= max_holding_bars:
                    # [FIX] Evaluate unrealized PnL at open; exit is at open → no look-ahead
                    if pos_side == 1:
                        unreal_p = (c_open - entry_price) / pos_atr if pos_atr > 0 else 0
                    else:
                        unreal_p = (entry_price - c_open) / pos_atr if pos_atr > 0 else 0
                    
                    if unreal_p < time_exit_profit_threshold:
                        exit_price = c_open  # Market at bar open
                        if pos_side == 1: exit_price *= (1 - slippage_rate)
                        else:             exit_price *= (1 + slippage_rate)
                        exit_triggered = True
                
                # RSI Panic (rsi[i] is shift(1) daily → prior day; no future reference)
                if not exit_triggered:
                    if pos_side == 1 and rsi[i] > rsi_long_exit:
                        exit_price = c_open * (1 - slippage_rate)
                        exit_triggered = True
                    elif pos_side == -1 and rsi[i] < rsi_short_exit:
                        exit_price = c_open * (1 + slippage_rate)
                        exit_triggered = True
                
                # Trend Reversal (optional to avoid over-filtered early exits)
                if not exit_triggered and enable_trend_exit:
                    # [FIX] Evaluate unrealized PnL at open; exit is at open → no look-ahead
                    unrealized_atr = 0.0
                    if pos_atr > 0:
                        if pos_side == 1:
                            unrealized_atr = (c_open - entry_price) / pos_atr
                        else:
                            unrealized_atr = (entry_price - c_open) / pos_atr
                    if pos_side == 1 and trend_dir[i] == -1:
                        # Ignore reversal exits on strong trend winners.
                        if unrealized_atr < 1.0:
                            exit_price = c_open * (1 - slippage_rate)
                            exit_triggered = True
                    elif pos_side == -1 and trend_dir[i] == 1:
                        if unrealized_atr < 1.0:
                            exit_price = c_open * (1 + slippage_rate)
                            exit_triggered = True

            if exit_triggered:
                # PnL Calc with Slippage
                if pos_side == 1:
                    pnl = (exit_price - entry_price) * amount
                else:
                    pnl = (entry_price - exit_price) * amount
                
                exit_fee = amount * exit_price * fee_rate
                pnl -= exit_fee
                
                margin = (amount * entry_price) / leverage
                balance += margin + pnl
                
                if trade_count < max_trades:
                    trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl, amount, entry_fee_stored]
                    trade_count += 1
                in_position = False
                last_funding_hour = -1  # Reset so next position can be charged in same funding hour
                pending_entry = False  # Clear any pending
                bar_processed = True   # No re-entry on this bar (Zombie/Phantom fix)

        # --- 2. ENTRY: Intra-bar breakout (simulation premise: bar i high/low known at bar i) ---
        # Only check entry when no position AND this bar did not just exit (Zombie fix)
        elif not in_position and not bar_processed:
            if np.isnan(entry_upper[i]) or np.isnan(entry_lower[i]):
                equity_curve[i] = balance
                continue
            if strength_filter[i] == 0:
                equity_curve[i] = balance
                continue
            vol_pass = True
            if use_volume_filter and volume_ratio[i] < vol_threshold:
                vol_pass = False
            if not vol_pass:
                equity_curve[i] = balance
                continue

            regime_mult = 1.0
            if use_dynamic_risk:
                if natr[i] > panic_regime_natr:
                    regime_mult = panic_regime_multiplier
                elif hurst[i] > strong_regime_hurst and natr[i] > strong_regime_natr:
                    regime_mult = strong_regime_multiplier
                elif hurst[i] < weak_regime_hurst:
                    regime_mult = weak_regime_multiplier
            exec_risk = risk_per_trade * regime_mult

            do_entry = False
            fill_price = 0.0
            # Breakout: use bar i high/low (intra-bar premise); fill at first touch (open or level)
            if c_high > entry_upper[i] and trend_dir[i] == 1:
                fill_price = max(c_open, entry_upper[i]) * (1 + slippage_rate)
                pending_side = 1
                do_entry = True
            elif c_low < entry_lower[i] and trend_dir[i] == -1:
                fill_price = min(c_open, entry_lower[i]) * (1 - slippage_rate)
                pending_side = -1
                do_entry = True

            if do_entry:
                # Use atr[i]: daily_atr is already shift(1) in _prepare_data (prior day). No extra i-1.
                current_atr = atr[i]
                if np.isnan(current_atr) or current_atr <= 0.0:
                    current_atr = atr[i - 1] if i > 0 else 0.0
                if np.isnan(current_atr) or current_atr <= 0.0:
                    current_atr = 0.0
                if stop_loss_type == 1:
                    if pending_side == 1:
                        stop_price = fill_price - (current_atr * atr_sl_mult)
                    else:
                        stop_price = fill_price + (current_atr * atr_sl_mult)
                else:
                    if pending_side == 1:
                        stop_price = fill_price * (1 - stop_loss_pct)
                    else:
                        stop_price = fill_price * (1 + stop_loss_pct)
                tp_price = 0.0
                if use_take_profit:
                    if pending_side == 1:
                        tp_price = fill_price + (current_atr * tp_atr_mult)
                    else:
                        tp_price = fill_price - (current_atr * tp_atr_mult)
                stop_distance = abs(fill_price - stop_price)
                if use_compounding:
                    current_equity = max_capital_usage if balance > max_capital_usage else balance
                else:
                    current_equity = initial_balance
                if stop_distance > 0:
                    amount = (current_equity * exec_risk) / stop_distance
                else:
                    amount = (current_equity * 0.01) / fill_price
                max_amount = (current_equity * leverage) / fill_price
                if amount > max_amount:
                    amount = max_amount
                required_margin = (amount * fill_price) / leverage
                entry_fee = amount * fill_price * fee_rate
                if balance >= required_margin + entry_fee:
                    balance -= (required_margin + entry_fee)
                    entry_fee_stored = entry_fee
                    in_position = True
                    pos_side = pending_side
                    entry_price = fill_price
                    entry_idx = i
                    highest = fill_price
                    lowest = fill_price
                    pos_atr = current_atr
                    execute_intra_bar = True

        # --- 3. INTRA-BAR SL/TP CHECK (same-bar exit after intra-bar entry; SL priority) ---
        if in_position and execute_intra_bar:
            intra_exit_triggered = False
            intra_exit_price = 0.0
            if pos_side == 1 and c_low <= stop_price:
                intra_exit_price = stop_price * (1 - slippage_rate)
                intra_exit_triggered = True
            elif pos_side == -1 and c_high >= stop_price:
                intra_exit_price = stop_price * (1 + slippage_rate)
                intra_exit_triggered = True
            elif use_take_profit and tp_price > 0:
                if pos_side == 1 and c_high >= tp_price:
                    intra_exit_price = tp_price
                    intra_exit_triggered = True
                elif pos_side == -1 and c_low <= tp_price:
                    intra_exit_price = tp_price
                    intra_exit_triggered = True
            if intra_exit_triggered:
                if pos_side == 1:
                    pnl = (intra_exit_price - entry_price) * amount
                else:
                    pnl = (entry_price - intra_exit_price) * amount
                exit_fee = amount * intra_exit_price * fee_rate
                pnl -= exit_fee
                margin = (amount * entry_price) / leverage
                balance += margin + pnl
                if trade_count < max_trades:
                    trades[trade_count] = [entry_idx, i, pos_side, entry_price, intra_exit_price, pnl, amount, entry_fee_stored]
                    trade_count += 1
                in_position = False
                last_funding_hour = -1  # Reset so next position can be charged in same funding hour
                pos_side = 0
                execute_intra_bar = False

        # [EQUITY] Bar-end equity for MDD (includes unrealized when in position)
        if in_position:
            margin = (amount * entry_price) / leverage
            unrealized = (c_price - entry_price) * amount * pos_side
            equity_curve[i] = balance + margin + unrealized
        else:
            equity_curve[i] = balance

    # --- 4. END-OF-DATA FORCED LIQUIDATION ---
    # Realize remaining position at the final bar close to avoid hidden unrealized PnL.
    if in_position and n > 0:
        last_idx = n - 1
        last_close = close[last_idx]
        if pos_side == 1:
            exit_price = last_close * (1 - slippage_rate)
            pnl = (exit_price - entry_price) * amount
        else:
            exit_price = last_close * (1 + slippage_rate)
            pnl = (entry_price - exit_price) * amount

        exit_fee = amount * exit_price * fee_rate
        pnl -= exit_fee

        margin = (amount * entry_price) / leverage
        balance += margin + pnl

        if trade_count < max_trades:
            trades[trade_count] = [entry_idx, last_idx, pos_side, entry_price, exit_price, pnl, amount, entry_fee_stored]
            trade_count += 1

    return trades[:trade_count], balance, equity_curve
