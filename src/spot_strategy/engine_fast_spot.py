
import pandas as pd
import numpy as np
import logging
from numba import njit


class BacktestEngineFastSpot:
    """
    Numba-accelerated Backtest Engine for Spot (Long-Only)
    Reuses architecture from BacktestEngineFast (Futures) for consistency.
    """
    def __init__(self, hourly_df, daily_df, strategy, backtest_func, initial_balance=10_000_000, fee_rate=0.0005, slippage_rate=0.0003, merge_index_map=None):
        # [MEMORY] Use shallow copy to prevent contaminating the global cached dataframe
        self.hourly_df = hourly_df.copy(deep=False)
        self.daily_df = daily_df.copy(deep=False)
        self.strategy = strategy
        self.backtest_func = backtest_func
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        
        # Injected by optimization script
        self.risk_per_trade = 0.99
        
        if merge_index_map is not None:
            self._merge_index_map = merge_index_map
        
        # [WARMUP] Extract from hourly df attrs
        self._warmup_bars = getattr(hourly_df, 'attrs', {}).get('warmup_bars', 0)
        
        self.logger = logging.getLogger(__name__)
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Prepare data with Daily Trend mapping.
        """
        # [OPTIMIZATION] Use pre-computed merge index if available
        if hasattr(self, '_merge_index_map'):
            self._prepare_data_with_index()
        else:
            self._prepare_data_with_merge() # Fallback for verify/live

    def _prepare_data_with_index(self):
        """
        [FAST PATH] Use pre-computed merge index to avoid pd.merge overhead.
        """
        # 1. Generate Daily Indicators (Trend Context)
        self.daily_df = self.strategy.generate_signals(self.daily_df)
        
        # 2. Generate Hourly Indicators (Entry/Exit Signals)
        # Note: Strategy creates columns like 'trend_direction', 'rsi', etc.
        self.hourly_df = self.strategy.generate_signals(self.hourly_df)
        
        # 3. Map Daily Trend to Hourly
        # We need 'trend_direction' from Daily to act as a filter
        # But 'trend_direction' in Hourly is used for Entry trigger.
        # Let's rename Daily columns to avoid collision.
        
        exclude_cols = {'date_key', 'datetime', 'date', 'open', 'high', 'low', 'close', 'volume'}
        daily_cols = [c for c in self.daily_df.columns if c not in exclude_cols]
        
        shifted_daily = self.daily_df[daily_cols].shift(1) # Shift 1 day to avoid lookahead
        
        # Create output dataframe based on Hourly
        self.df = self.hourly_df.copy(deep=False)
        
        # Map Daily columns using index
        for col in daily_cols:
            daily_vals = shifted_daily[col].values
            mapped_vals = daily_vals[self._merge_index_map]
            self.df[f'daily_{col}'] = mapped_vals
            
        # [LOGIC MERGE] Combine Hourly Trend with Daily Trend
        # Final Trend = 1 if (Hourly Trend == 1 AND Daily Trend == 1) else ...
        # Actually, let's keep them separate and handle logic in Numba or here.
        # For simplicity and speed, let's update 'trend_direction' here.
        
        # Strategy: "Trend Alignment"
        # If Daily Trend is Uptrend (1), allow Long Entry.
        # If Daily Trend is Downtrend/Neutral, block Long Entry.
        # (Spot is Long-Only, so this is critical)
        
        if 'daily_trend_direction' in self.df.columns:
            # Overwrite hourly trend_direction to enforce Daily Filter
            # 1 (Long) only if both are 1.
            h_trend = self.df['trend_direction'].values
            d_trend = self.df['daily_trend_direction'].values
            
            # Vectorized AND logic: Both must be positive 1
            final_trend = np.where((h_trend == 1) & (d_trend == 1), 1, 0)
            self.df['trend_direction'] = final_trend
            
        # Extract Arrays for Numba
        self._extract_arrays()

    def _prepare_data_with_merge(self):
        """[FALLBACK] Slow merge for verification"""
        # Generate Indicators
        self.daily_df = self.strategy.generate_signals(self.daily_df)
        self.hourly_df = self.strategy.generate_signals(self.hourly_df)
        
        # Daily Date Key
        if 'date_key' not in self.daily_df.columns:
             self.daily_df['date_key'] = pd.to_datetime(self.daily_df['datetime']).dt.strftime('%Y-%m-%d')
        if 'date_key' not in self.hourly_df.columns:
            self.hourly_df['date_key'] = pd.to_datetime(self.hourly_df['datetime']).dt.strftime('%Y-%m-%d')
            
        # Rename Daily Cols
        exclude_cols = {'date_key', 'datetime', 'date', 'open', 'high', 'low', 'close', 'volume'}
        daily_cols = [c for c in self.daily_df.columns if c not in exclude_cols]
        shifted_daily = self.daily_df[daily_cols].shift(1)
        shifted_daily.columns = [f'daily_{c}' for c in daily_cols]
        shifted_daily['date_key'] = self.daily_df['date_key']
        
        # Merge
        self.df = pd.merge(self.hourly_df, shifted_daily, on='date_key', how='left')
        
        # Apply Logic Merge (Trend Alignment)
        if 'daily_trend_direction' in self.df.columns:
            h_trend = self.df['trend_direction'].fillna(0).values
            d_trend = self.df['daily_trend_direction'].fillna(0).values
            self.df['trend_direction'] = np.where((h_trend == 1) & (d_trend == 1), 1, 0)
            
        self._extract_arrays()

    def _extract_arrays(self):
        """Extract numpy arrays from final self.df"""
        self.close = self.df['close'].values
        self.high = self.df['high'].values
        self.low = self.df['low'].values
        self.open_prices = self.df['open'].values # Standard open prices
        
        # [CRITICAL] LOOKAHEAD PROTECTION
        # All indicators used for ENTRY/EXIT decisions must be based on CLOSED bars.
        # Shift(1) ensures that at iteration 'i', we use the signal determined at 'i-1'.
        signal_cols = [
            'entry_upper', 'trend_direction', 'strength_filter', 
            'volume_ratio', 'atr', 'parabolic_sar', 'hurst', 'natr', 'rsi'
        ]
        
        for col in signal_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].shift(1)

        self.entry_upper = self.df['entry_upper'].values
        self.trend_dir = self.df['trend_direction'].values # MTF Filtered Trend
        self.strength_filter = self.df['strength_filter'].values
        self.volume_ratio = self.df['volume_ratio'].values
        self.atr = self.df['atr'].values
        self.parabolic_sar = self.df['parabolic_sar'].values
        self.hurst = self.df['hurst'].values
        self.natr = self.df['natr'].values
        self.rsi = self.df['rsi'].values
        
        # Cleanup
        self.df = None
    
    def run(self):
        """
        Execute backtest using Numba-accelerated loop.
        """
        # Extract strategy params
        exit_type = 1 if self.strategy.params.get('EXIT_TYPE') == 'PARABOLIC_SAR' else 0
        stop_loss_type = 1 if self.strategy.params.get('STOP_LOSS_TYPE') == 'ATR' else 0
        stop_loss_pct = self.strategy.params.get('STOP_LOSS_PCT', 0.03)
        atr_sl_mult = self.strategy.params.get('ATR_STOP_LOSS_MULT', 1.5)
        atr_mult = self.strategy.params.get('ATR_MULTIPLIER', 3.0)
        
        use_volume_filter = self.strategy.params.get('USE_VOLUME_FILTER', False)
        vol_threshold = self.strategy.params.get('VOLUME_THRESHOLD_MULT', 1.0)
        
        use_take_profit = self.strategy.params.get('USE_TAKE_PROFIT', False)
        tp_atr_mult = self.strategy.params.get('TAKE_PROFIT_ATR_MULT', 3.0)
        
        # [NEW] Time-Based Exit & Trailing Activation
        max_holding_bars = self.strategy.params.get('MAX_HOLDING_BARS', 999999)
        trailing_activation_atr = self.strategy.params.get('TRAILING_ACTIVATION_ATR', 0.0)
        
        # [WARMUP OPTIMIZATION] Extract warmup period if available
        # Note: self.df was set to None after data extraction, so we need to get it from original df
        # Actually, we need to pass this through __init__ or store it before clearing
        # For now, use a safe default of 0 (no warmup) - will be set by optimize script
        warmup_bars = getattr(self, '_warmup_bars', 0)
        
        # [NEW] Extract Regime Params from Strategy
        # Defaults are set broad if not optimized
        hurst_threshold = self.strategy.params.get('STRONG_REGIME_HURST', 0.6)
        natr_panic_threshold = self.strategy.params.get('PANIC_REGIME_NATR', 4.5)
        rsi_panic_threshold = self.strategy.params.get('RSI_EXIT_THRESHOLD', 94) # Changed key to match config
        
        strong_regime_multiplier = self.strategy.params.get('STRONG_REGIME_MULTIPLIER', 1.3)
        panic_regime_multiplier = self.strategy.params.get('PANIC_REGIME_MULTIPLIER', 0.15)
        
        # [NEW] Entry Filters (Safety)
        rsi_entry_max = self.strategy.params.get('RSI_ENTRY_MAX', 100) # Default: No max limit
        natr_entry_min = self.strategy.params.get('NATR_ENTRY_MIN', 0.0) # Default: No min limit
        

        # [NEW] Weak Regime (Choppy/Correction)
        hurst_weak_threshold = self.strategy.params.get('WEAK_REGIME_HURST', 0.45)
        weak_regime_multiplier = self.strategy.params.get('WEAK_REGIME_MULTIPLIER', 0.6)

        # [NEW] Capital Management
        use_compounding = self.strategy.params.get('USE_COMPOUNDING', False)
        max_capital_usage = self.strategy.params.get('MAX_CAPITAL_USAGE', 100_000_000_000.0)

        # Run Numba loop (using injected function)
        trades, equity, final_bal = self.backtest_func(
            self.close, self.high, self.low, self.open_prices,
            self.entry_upper,
            self.trend_dir, self.strength_filter, self.volume_ratio,
            self.atr, self.parabolic_sar,
            self.hurst, self.natr, self.rsi, # [NEW] Regime Inputs
            self.initial_balance, self.fee_rate, self.slippage_rate,
            exit_type,
            stop_loss_type, stop_loss_pct, atr_sl_mult,
            atr_mult, self.risk_per_trade,
            use_volume_filter, vol_threshold,
            use_take_profit, tp_atr_mult,
            max_holding_bars, trailing_activation_atr,
            # [NEW] Regime optimization params
            hurst_threshold, natr_panic_threshold, rsi_panic_threshold,
            strong_regime_multiplier, panic_regime_multiplier,
            # [NEW] Weak Regime
            hurst_weak_threshold, weak_regime_multiplier,
            # [NEW] Entry Filters
            rsi_entry_max, natr_entry_min,
            warmup_bars,
            # [NEW] Capital Mgmt
            use_compounding, max_capital_usage
        )
        
        # Calculate metrics
        total_return_pct = (final_bal - self.initial_balance) / self.initial_balance * 100
        
        # MDD calculation
        peak = np.maximum.accumulate(equity)
        with np.errstate(divide='ignore', invalid='ignore'):
            mdd_series = np.where(peak > 0, (equity - peak) / peak * 100, 0.0)
            mdd_pct = np.min(mdd_series)
            if np.isnan(mdd_pct):
                mdd_pct = 0.0
        
        # Trade statistics
        num_trades = len(trades)
        if num_trades > 0:
            pnl_pcts = trades[:, 0]
            win_rate = (len(pnl_pcts[pnl_pcts > 0]) / num_trades * 100)
        else:
            win_rate = 0.0
        
        # Convert trades to DataFrame
        if num_trades > 0:
            trades_df = pd.DataFrame(trades, columns=['pnl_pct', 'duration', 'dummy'])
            # [FIX] calculate_score expects 'pnl' column. Add dummy since we optimize on % anyway.
            trades_df['pnl'] = trades_df['pnl_pct'] 
        else:
            trades_df = pd.DataFrame()
        
        return {
            'total_return_pct': total_return_pct,
            'mdd_pct': mdd_pct,
            'total_trades': num_trades,
            'win_rate': win_rate,
            'final_balance': final_bal,
            'trades_df': trades_df,
            'equity_curve': equity
        }


@njit(nogil=True, cache=True)
def backtest_loop_spot_numba(
    close, high, low, open_prices,
    entry_upper,
    trend_dir, strength_filter, volume_ratio, atr, parabolic_sar,
    hurst, natr, rsi,
    initial_balance, fee_rate, slippage_rate,
    exit_type,
    stop_loss_type, stop_loss_pct, atr_sl_mult,
    atr_mult, risk_per_trade,
    use_volume_filter, vol_threshold,
    use_take_profit, tp_atr_mult,
    max_holding_bars, trailing_activation_atr,
    hurst_threshold, natr_panic_threshold, rsi_panic_threshold,
    strong_regime_multiplier, panic_regime_multiplier,
    hurst_weak_threshold, weak_regime_multiplier, # [NEW] Weak Regime
    rsi_entry_max, natr_entry_min,
    warmup_bars,
    use_compounding, max_capital_usage # [NEW] Capital Mgmt
):
    """
    Numba Backtest Loop v15.2: Strict Realism & Slippage Enforcement + Regime Sizing
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
    
    max_trades = 30000
    trades = np.zeros((max_trades, 3))
    trade_count = 0
    
    equity_curve = np.zeros(n)
    exec_risk = risk_per_trade 
    
    for i in range(n):
        # [WARMUP] Skip trading during warmup period
        if i < warmup_bars:
            equity_curve[i] = balance
            continue
            
        c_open = open_prices[i]
        c_price = close[i]
        c_high = high[i]
        c_low = low[i]
        
        # --- 1. EXECUTION: BUY AT OPEN (If signaled at i-1) ---
        if pending_entry and not in_position:
            fill_price = c_open * (1 + slippage_rate)
            
            # Use ATR from signal bar (i-1)
            sig_atr = atr[i-1] if i > 0 else atr[i]
            
            # SL/TP Setup
            if stop_loss_type == 1:
                stop_price = fill_price - (sig_atr * atr_sl_mult)
            else:
                stop_price = fill_price * (1 - stop_loss_pct)
            
            tp_price = fill_price + (sig_atr * tp_atr_mult) if use_take_profit else 0.0
            
            # [CAPITAL MGMT] Dynamic Risk Sizing
            target_risk = exec_risk
            if target_risk > 0.99: target_risk = 0.99 # Spot cannot exceed 1.0 (no leverage)
            
            # [COMPOUNDING]
            current_capital = balance
            if not use_compounding:
                current_capital = min(balance, initial_balance)
                
            cost = current_capital * target_risk
            cost = min(cost, max_capital_usage) # Cap max usage
            
            # Ensure we have enough balance
            cost = min(cost, balance)
            
            if cost > 0:
                coin = (cost * (1 - fee_rate)) / fill_price
                balance -= cost
                
                in_position = True
                pending_entry = False
                entry_price = fill_price
                entry_idx = i
                highest = fill_price
                pos_atr = sig_atr
        
        # --- 2. EXECUTION: EXIT CHECKS (During Bar i) ---
        if in_position:
            exit_triggered = False
            exit_price = 0.0
            
            # [SEQ-1] Check Stop Loss Hierarchy FIRST
            current_stop = stop_price
            if exit_type == 1 and parabolic_sar[i] > 0:
                current_stop = max(stop_price, parabolic_sar[i])

            if c_low <= current_stop:
                # Market Exit Slip
                exit_price = current_stop * (1 - slippage_rate)
                exit_triggered = True
            
            # [SEQ-2] Check Take Profit
            elif not exit_triggered and use_take_profit and tp_price > 0 and c_high >= tp_price:
                # Limit Exit (No Slip for TP usually)
                exit_price = tp_price 
                exit_triggered = True

            # [SEQ-3] Conditional Market Exits (RSI, Trend, Time)
            elif not exit_triggered:
                # RSI Panic
                if rsi[i] > rsi_panic_threshold:
                    exit_price = c_price * (1 - slippage_rate)
                    exit_triggered = True
                
                # Trend Reversal
                elif i > 0 and trend_dir[i] == -1: 
                    exit_price = c_price * (1 - slippage_rate)
                    exit_triggered = True
                
                # Max Holding
                elif (i - entry_idx) >= max_holding_bars:
                    unrealized_pct = (c_price - entry_price) / entry_price * 100
                    # Only exit if profit is small (stagnant) or negative
                    if unrealized_pct < 0.2:
                        exit_price = c_price * (1 - slippage_rate)
                        exit_triggered = True

            if exit_triggered:
                revenue = coin * exit_price * (1 - fee_rate)
                balance += revenue 
                
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                if trade_count < max_trades:
                    trades[trade_count] = [pnl_pct, float(i - entry_idx), 0.0]
                    trade_count += 1
                
                coin = 0.0
                in_position = False
            else:
                # [SEQ-4] Update High and Trailing Stop for NEXT Bar (i+1)
                if c_high > highest:
                    highest = c_high
                
                if exit_type == 0: # ATR Trailing
                    unrealized_profit_atr = (highest - entry_price) / pos_atr if pos_atr > 0 else 0
                    if unrealized_profit_atr >= trailing_activation_atr:
                        new_stop = highest - (pos_atr * atr_mult)
                        if new_stop > stop_price:
                            stop_price = new_stop

        # --- 3. SIGNAL: ENTRY DETECTION (At Close of i, for Open of i+1) ---
        elif not in_position and not pending_entry:
            can_signal = True
            if np.isnan(entry_upper[i]): can_signal = False
            elif strength_filter[i] == 0: can_signal = False
            elif use_volume_filter and volume_ratio[i] < vol_threshold: can_signal = False
            elif rsi[i] >= rsi_entry_max: can_signal = False
            elif natr[i] < natr_entry_min: can_signal = False
            
            if can_signal and trend_dir[i] == 1 and c_price > entry_upper[i]:
                # [DYNAMIC RISK SIZING]
                regime_mult = 1.0
                
                # Strong Trend Regime
                if hurst[i] > hurst_threshold: 
                    regime_mult = strong_regime_multiplier
                
                # Panic Regime
                if natr[i] > natr_panic_threshold: 
                    regime_mult = panic_regime_multiplier
                    
                # Weak/Choppy Regime (Low Hurst)
                elif hurst[i] < hurst_weak_threshold:
                    regime_mult = weak_regime_multiplier
                
                exec_risk = risk_per_trade * regime_mult
                
                if i < n - 1:
                    pending_entry = True
        
        equity_curve[i] = balance + (coin * c_price)

    return trades[:trade_count], equity_curve, balance + (coin * close[-1])
