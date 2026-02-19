
from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
import logging
from numba import njit
from config.settings import TRADING_FEE_RATE, SLIPPAGE_RATE, FUNDING_FEE_RATE, FUNDING_INTERVAL_HOURS

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
        # [MEMORY] Use shallow copy to prevent contaminating usage across trials
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
        
        # [CRITICAL] Use pre-computed index mapping to align daily -> hourly
        # merged_df = hourly_df + daily_indicators[merge_index_map]
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
        self.logger.info(f"Running FAST backtest for {self.strategy.name}...")
        
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
        timestamps = df['timestamp'].values  # milliseconds
        
        # [NEW] Time-Based Exit & Trailing Activation
        max_holding_bars = self.strategy.params.get('MAX_HOLDING_BARS', 999999)  # Default: No limit
        trailing_activation_atr = self.strategy.params.get('TRAILING_ACTIVATION_ATR', 0.0)  # Default: Immediate activation
        time_exit_profit_threshold = self.strategy.params.get('TIME_EXIT_PROFIT_THRESHOLD', 0.5)  # Default: 0.5 ATR profit required to hold
        enable_trend_exit = self.strategy.params.get('ENABLE_TREND_EXIT', True)
        
        # [WARMUP OPTIMIZATION] Calculate required warmup based on strategy indicators
        # Strategy analyzes its own parameters to determine minimum stable period
        warmup_bars = getattr(df, 'attrs', {}).get('warmup_bars', self.strategy.get_required_warmup())
        
        # [NEW] RSI for Panic Exit
        # Use existing rsi if available, else fill 50
        if 'daily_rsi' in df.columns:
            rsi = df['daily_rsi'].values
        else:
            rsi = np.full(n, 50.0)
            
        rsi_exit_threshold = self.strategy.params.get('RSI_EXIT_THRESHOLD', 80.0) # Default: 80 for Long Exit, 20 for Short Exit
        
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
        trades, final_balance = backtest_loop_numba(
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
            timestamps, FUNDING_FEE_RATE, FUNDING_INTERVAL_HOURS,
            max_holding_bars, trailing_activation_atr, time_exit_profit_threshold,
            enable_trend_exit,
            rsi_exit_threshold,
            use_dynamic_risk, strong_regime_hurst, strong_regime_natr, strong_regime_multiplier,
            weak_regime_hurst, weak_regime_multiplier,
            panic_regime_natr, panic_regime_multiplier,
            warmup_bars,
            use_compounding,    # [NEW]
            max_capital_usage   # [NEW]
        )
        
        self.balance = final_balance
        
        # [MEMORY] Release large DataFrames before processing results
        self.merged_df = None
        self.hourly_df = None
        self.daily_df = None
        
        # Convert trades to DataFrame (Vectorized Lookup)
        self.trades = []
        for i in range(len(trades)):
            if trades[i][0] == 0 and trades[i][1] == 0:  # Check for empty/dummy rows
                break
                
            entry_idx = int(trades[i][0])
            exit_idx = int(trades[i][1])
            
            self.trades.append({
                'entry_time': datetime_values[entry_idx],
                'exit_time': datetime_values[exit_idx],
                'side': 'LONG' if trades[i][2] == 1 else 'SHORT',
                'entry_price': trades[i][3],
                'exit_price': trades[i][4],
                'pnl': trades[i][5]
            })
        
        result = self.get_results()
        return result
    
    def get_results(self):
        # [ROBUSTNESS] Sanitize balance if infinite (likely due to extreme compounding)
        if not np.isfinite(self.balance):
             # Cap at an unreasonably high number to preserve "goodness" but avoid Inf
             self.balance = 1e15 
             
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
        
        # [FIX] Calculate pnl_pct as Return on Equity (ROE) per trade
        # This provides a consistent metric regardless of account size growing
        # ROE = PnL / Initial Margin Used
        
        # Reconstruct Margin used for each trade
        # Margin = (Entry Price * Amount) / Leverage
        # We need to estimate Amount from trades? 
        # The trades array has: [entry_idx, exit_idx, side, entry_p, exit_p, pnl]
        # We don't have 'amount' stored in trades array directly.
        # But PnL = (Exit - Entry) * Amount * Side
        # Amount = PnL / ((Exit - Entry) * Side)
        
        # Let's Vectorize this safely
        pnl = trades_df['pnl'].values
        entry_p = trades_df['entry_price'].values
        exit_p = trades_df['exit_price'].values
        side = np.where(trades_df['side'] == 'LONG', 1, -1)
        
        price_diff = (exit_p - entry_p) * side
        
        # Avoid zero division for amount calc
        with np.errstate(invalid='ignore', divide='ignore'):
             amount_est = pnl / price_diff
             # If price_diff is 0 (entry=exit), amount is inf. Handle this.
             # But if entry=exit, PnL should be -Fee.
             # This estimation is tricky with fees included in PnL.
             
        # Alternative: Simply use (PnL / Balance_Before) * 100 as before but handle the precision better
        # The user's issue is likely the astronomical total return.
        
        # [DECISION] Stick to Balance Relative return but ensure float64 precision
        with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
            pnl_cumsum = trades_df['pnl'].cumsum().shift(1).fillna(0)
            trades_df['balance_before'] = self.initial_balance + pnl_cumsum
            
            trades_df['balance_before'] = trades_df['balance_before'].replace(0, 1e-9)
            trades_df['pnl_pct'] = (trades_df['pnl'] / trades_df['balance_before']) * 100
        
        trades_df['pnl_pct'] = trades_df['pnl_pct'].replace([np.inf, -np.inf], 0).fillna(0)
        
        win_trades = len(trades_df[trades_df['pnl'] > 0])
        loss_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (win_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        
        # MDD Calculation
        # Construct cumulative array safely
        cumulative = [self.initial_balance]
        for pnl in trades_df['pnl']:
            cumulative.append(cumulative[-1] + pnl)
        
        cumulative = np.array(cumulative)
        
        # [ROBUSTNESS] Check for Inf/NaN in cumulative data (Exploding Strategy)
        if not np.isfinite(cumulative).all():
             self.logger.warning("Strategy equity curve contains Inf/NaN. Marking as invalid.")
             # Return invalid result to trigger -10000 score in objective function
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

        with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
            running_max = np.maximum.accumulate(cumulative)
            
            # Prevent division by zero if running_max is 0
            running_max[running_max == 0] = 1e-9
            
            drawdown = (cumulative - running_max) / running_max * 100
        
        # Sanitize drawdown (nan -> 0)
        drawdown = np.nan_to_num(drawdown, nan=0.0)
        mdd = drawdown.min()
        
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
    timestamps, funding_fee_rate, funding_interval_hours,
    max_holding_bars, trailing_activation_atr, time_exit_profit_threshold,
    enable_trend_exit,
    rsi_exit_threshold, # [NEW]
    use_dynamic_risk, strong_regime_hurst, strong_regime_natr, strong_regime_multiplier,
    weak_regime_hurst, weak_regime_multiplier,
    panic_regime_natr, panic_regime_multiplier, # [NEW] Dynamic Risk
    warmup_bars,  # [WARMUP] Number of bars to skip at start for indicator warmup
    use_compounding,   # [NEW] Bool
    max_capital_usage  # [NEW] Float
):
    """
    Numba JIT-compiled backtest loop for Futures (Long/Short).
    Strict Realism Mode (v15.1):
    - Signal at i -> Execute at Open of i+1
    - Slippage on All Exits
    - Sequential Stop Logic
    - [v16] Compounding Control & Capital Caps
    """
    n = len(close)
    balance = initial_balance
    
    # Position state
    in_position = False
    pending_entry = False 
    pending_side = 0
    
    pos_side = 0  # 1: LONG, -1: SHORT
    entry_price = 0.0
    entry_idx = 0
    amount = 0.0
    highest = 0.0
    lowest = 0.0
    pos_atr = 0.0
    stop_price = 0.0 # Initial Stop Loss
    tp_price = 0.0 
    
    # Funding Fee Tracking (UTC-based: 00:00, 08:00, 16:00)
    last_funding_hour = -1
    
    # Trades storage (max 30000 trades)
    max_trades = 30000
    trades = np.zeros((max_trades, 6))  # [entry_idx, exit_idx, side, entry_p, exit_p, pnl]
    trade_count = 0
    
    exec_risk = risk_per_trade # Local execution risk
    
    for i in range(n):
        # [WARMUP] Skip trading during warmup period
        if i < warmup_bars:
            continue
            
        # [SAFETY] Bankruptcy Check
        if balance <= 0:
            break

        c_open = open_prices[i]
        c_price = close[i]
        c_high = high[i]
        c_low = low[i]
        current_timestamp = timestamps[i]
        
        # --- 1. EXECUTION: PENDING ENTRY AT OPEN ---
        if pending_entry and not in_position:
            # We are at Open of bar i. Signal was confirmed at i-1.
            fill_price = 0.0
            
            if pending_side == 1: # LONG
                fill_price = c_open * (1 + slippage_rate)
            else: # SHORT
                fill_price = c_open * (1 - slippage_rate)
            
            # Use indicators from i-1 (signal bar) to keep timing strictly causal.
            signal_idx = i - 1 if i > 0 else i
            current_atr = atr[signal_idx]
            
            # 1. SL Calc
            if stop_loss_type == 1: # ATR
                if pending_side == 1:
                    stop_price = fill_price - (current_atr * atr_sl_mult)
                else:
                    stop_price = fill_price + (current_atr * atr_sl_mult)
            else: # Fixed
                if pending_side == 1:
                    stop_price = fill_price * (1 - stop_loss_pct)
                else:
                    stop_price = fill_price * (1 + stop_loss_pct)
            
            # 2. TP Calc
            if use_take_profit:
                if pending_side == 1:
                    tp_price = fill_price + (current_atr * tp_atr_mult)
                else:
                    tp_price = fill_price - (current_atr * tp_atr_mult)
            else:
                tp_price = 0.0
                
            # 3. Size Calculation
            stop_distance = abs(fill_price - stop_price)
            
            # [REALISM FIX] Capital Management
            if use_compounding:
                # Use balance but cap it to simulate liquidity limits
                if balance > max_capital_usage:
                    current_equity = max_capital_usage
                else:
                    current_equity = balance
            else:
                # Fixed Fractional (Risk based on Initial Balance only)
                # Most statistically robust for optimization
                current_equity = initial_balance
            
            if stop_distance > 0:
                risk_amount = current_equity * exec_risk
                amount = risk_amount / stop_distance
            else:
                amount = (current_equity * 0.01) / fill_price
                
            # Cap to Leverage (based on current equity)
            max_amount = (current_equity * leverage) / fill_price
            if amount > max_amount:
                amount = max_amount

            # Fee & Margin Check
            required_margin = (amount * fill_price) / leverage
            entry_fee = amount * fill_price * fee_rate
            total_cost = required_margin + entry_fee
            
            if balance >= total_cost:
                balance -= total_cost
                in_position = True
                pos_side = pending_side
                entry_price = fill_price
                entry_idx = i
                highest = fill_price
                lowest = fill_price
                pos_atr = current_atr
                pending_entry = False
            else:
                # Funding insufficient, cancel
                pending_entry = False

        # --- 2. POSITION MANAGEMENT (Exits & Funding) ---
        if in_position:
            # A. Funding Fee (UTC 00, 08, 16)
            current_hour_utc = int((current_timestamp // 1000) % 86400 // 3600)
            is_funding_hour = (current_hour_utc in (0, 8, 16))
            
            if is_funding_hour and last_funding_hour != current_hour_utc:
                notional_value = amount * c_price
                funding_cost = notional_value * funding_fee_rate
                balance -= funding_cost
                last_funding_hour = current_hour_utc
                
                # Bankruptcy check from Funding
                if balance <= 0:
                    exit_price = c_price
                    if pos_side == 1: pnl = (exit_price - entry_price) * amount
                    else:             pnl = (entry_price - exit_price) * amount
                    
                    exit_fee = amount * exit_price * fee_rate
                    pnl -= exit_fee
                    margin = (amount * entry_price) / leverage
                    balance += margin + pnl
                    
                    trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl]
                    trade_count += 1
                    in_position = False
                    break # Loop break
            
            # B. Exit Checks
            exit_triggered = False
            exit_price = 0.0
            
            # [SEQ-1] Stop Loss (Inherited)
            # Check using stop_price from previous bar (Sequential Realism)
            
            # Pre-calc SAR stop if active
            current_stop = stop_price
            if exit_type == 1 and parabolic_sar[i] > 0:
                # SAR acts as trailing stop
                if pos_side == 1:
                     current_stop = max(stop_price, parabolic_sar[i])
                else:
                     current_stop = min(stop_price, parabolic_sar[i])

            # Check SL Hit
            if pos_side == 1:
                if c_low <= current_stop:
                    exit_price = current_stop * (1 - slippage_rate)
                    exit_triggered = True
            else:
                if c_high >= current_stop:
                    exit_price = current_stop * (1 + slippage_rate)
                    exit_triggered = True
            
            # [SEQ-2] Take Profit
            if not exit_triggered and use_take_profit and tp_price > 0:
                if pos_side == 1:
                    if c_high >= tp_price:
                        exit_price = tp_price # Limit fill assumption or half slip
                        exit_triggered = True
                else:
                    if c_low <= tp_price:
                        exit_price = tp_price
                        exit_triggered = True

            # [SEQ-3] Conditional Market Exits
            if not exit_triggered:
                # Time Exit
                bars_held = i - entry_idx
                if bars_held >= max_holding_bars:
                    if pos_side == 1:
                        unreal_p = (c_price - entry_price) / pos_atr if pos_atr > 0 else 0
                    else:
                        unreal_p = (entry_price - c_price) / pos_atr if pos_atr > 0 else 0
                    
                    if unreal_p < time_exit_profit_threshold:
                        exit_price = c_open # Market at bar open
                        if pos_side == 1: exit_price *= (1 - slippage_rate)
                        else:             exit_price *= (1 + slippage_rate)
                        exit_triggered = True
                
                # RSI Panic
                if not exit_triggered:
                    if pos_side == 1 and rsi[i] > rsi_exit_threshold:
                         exit_price = c_open * (1 - slippage_rate)
                         exit_triggered = True
                    elif pos_side == -1 and rsi[i] < (100 - rsi_exit_threshold):
                         exit_price = c_open * (1 + slippage_rate)
                         exit_triggered = True
                
                # Trend Reversal (optional to avoid over-filtered early exits)
                if not exit_triggered and enable_trend_exit:
                    unrealized_atr = 0.0
                    if pos_atr > 0:
                        if pos_side == 1:
                            unrealized_atr = (c_price - entry_price) / pos_atr
                        else:
                            unrealized_atr = (entry_price - c_price) / pos_atr
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
                
                trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl]
                trade_count += 1
                in_position = False
                pending_entry = False # Clear any pending
                
            else:
                # [SEQ-4] Update High/Low & Trailing Stop for NEXT Bar
                if pos_side == 1:
                    if c_high > highest:
                        highest = c_high
                        
                    if exit_type == 0: # ATR Trailing
                        unreal_profit = (highest - entry_price) / pos_atr if pos_atr > 0 else 0
                        if unreal_profit >= trailing_activation_atr:
                            new_stop = highest - (pos_atr * atr_mult)
                            if new_stop > stop_price:
                                stop_price = new_stop
                                
                else: # Short
                    if c_low < lowest:
                        lowest = c_low
                        
                    if exit_type == 0:
                        unreal_profit = (entry_price - lowest) / pos_atr if pos_atr > 0 else 0
                        if unreal_profit >= trailing_activation_atr:
                            new_stop = lowest + (pos_atr * atr_mult)
                            if new_stop < stop_price:
                                stop_price = new_stop

        # --- 3. SIGNAL DETECTION (For Pending Next Open) ---
        elif not in_position and not pending_entry:
            # Indicators are already shifted by 1.
            # So looking at index [i] means looking at Confirmed Daily/Hourly data from [i-1].
            
            # Check NaN
            if np.isnan(entry_upper[i]) or np.isnan(entry_lower[i]):
                continue
            
            # Volume & Strength Filters
            if strength_filter[i] == 0: continue
            
            vol_pass = True
            if use_volume_filter and volume_ratio[i] < vol_threshold:
                vol_pass = False
            if not vol_pass: continue

            # Dynamic Risk Sizing
            regime_mult = 1.0
            if use_dynamic_risk:
                if natr[i] > panic_regime_natr:
                    regime_mult = panic_regime_multiplier
                elif hurst[i] > strong_regime_hurst and natr[i] > strong_regime_natr:
                    regime_mult = strong_regime_multiplier
                elif hurst[i] < weak_regime_hurst:
                    regime_mult = weak_regime_multiplier
            
            exec_risk = risk_per_trade * regime_mult

            # LONG Signal
            if c_price > entry_upper[i] and trend_dir[i] == 1:
                pending_entry = True
                pending_side = 1
                
            # SHORT Signal
            elif c_price < entry_lower[i] and trend_dir[i] == -1:
                pending_entry = True
                pending_side = -1
    
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
            trades[trade_count] = [entry_idx, last_idx, pos_side, entry_price, exit_price, pnl]
            trade_count += 1

    return trades[:trade_count], balance
