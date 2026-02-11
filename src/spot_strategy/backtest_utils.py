
import pandas as pd
import numpy as np

def run_backtest_segment(df, params, initial_balance=10000000.0, return_series=False):
    """
    Shared backtest logic (Python Loop) for Verification and Walk-Forward Analysis.
    Not for Optimization (use Numba version there for speed).
    """
    balance = initial_balance
    coin = 0.0
    in_pos = False
    entry_price = 0.0
    highest = 0.0
    pos_atr = 0.0
    stop_price = 0.0
    tp_price = 0.0
    
    # Params
    fee_rate = 0.001  # 0.1%
    slippage = 0.001  # 0.1% (total cost 0.2%)
    atr_mult = params.get('ATR_MULTIPLIER', 3.0)
    exit_type = params.get('EXIT_TYPE', 'ATR')
    sl_type = params.get('STOP_LOSS_TYPE', 'FIXED')
    sl_pct = params.get('STOP_LOSS_PCT', 0.03)
    atr_sl_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
    use_tp = params.get('USE_TAKE_PROFIT', False)
    tp_atr_mult = params.get('TAKE_PROFIT_ATR_MULT', 3.0)
    # Use the spot-specific key
    risk_per_trade = params.get('RISK_PER_TRADE_SPOT', 0.99)
    
    trades_log = [] # Holds % returns
    detailed_log = [] # Holds dicts with details
    equity_curve = []
    
    # Data extraction
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    times = df['datetime'].values
    entry_upper = df['entry_upper'].values # NaNs here are handled by 'continue'
    
    # [FIX] Handle NaNs in indicators to prevent logic breakage
    # If ATR is NaN, fill with 0.0 (prevents SL becoming NaN)
    # If Trend is NaN, fill with 0 (Neutral)
    trend_dir = np.nan_to_num(df['trend_direction'].values, nan=0).astype(int)
    strength = np.nan_to_num(df['strength_filter'].values, nan=0).astype(int)
    atr = np.nan_to_num(df['atr'].values, nan=0.0)
    sar = np.nan_to_num(df.get('parabolic_sar', pd.Series([0.0]*len(df))).values, nan=0.0)
    
    # [NEW] Regime Indicators
    hurst = np.nan_to_num(df.get('hurst', pd.Series([0.5]*len(df))).values, nan=0.5)
    natr = np.nan_to_num(df.get('natr', pd.Series([0.0]*len(df))).values, nan=0.0)
    rsi = np.nan_to_num(df.get('rsi', pd.Series([50.0]*len(df))).values, nan=50.0)
    
    # [REGIME PARAMS]
    hurst_threshold = params.get('HURST_TREND_THRESHOLD', params.get('STRONG_REGIME_HURST', 0.6))
    strong_regime_natr = params.get('STRONG_REGIME_NATR', 1.0)
    natr_panic_threshold = params.get('PANIC_REGIME_NATR', 4.5)
    rsi_panic_threshold = params.get('RSI_EXIT_THRESHOLD', 94.0)
    use_dynamic_risk = params.get('USE_DYNAMIC_RISK', True)
    
    strong_regime_multiplier = params.get('STRONG_REGIME_MULTIPLIER', 1.3)
    panic_regime_multiplier = params.get('PANIC_REGIME_MULTIPLIER', 0.15)
    weak_regime_hurst = params.get('WEAK_REGIME_HURST', 0.45)
    weak_regime_multiplier = params.get('WEAK_REGIME_MULTIPLIER', 0.6)
    
    max_holding_bars = params.get('MAX_HOLDING_BARS', 999999)
    time_exit_profit_threshold = params.get('TIME_EXIT_PROFIT_THRESHOLD', 1.4)
    rsi_entry_max = params.get('RSI_ENTRY_MAX', 100.0)
    natr_entry_min = params.get('NATR_ENTRY_MIN', 0.0)
    use_compounding = params.get('USE_COMPOUNDING', False)
    max_capital_usage = params.get('MAX_CAPITAL_USAGE', 100_000_000.0)
    
    entry_idx = 0

    for i in range(1, len(df)):
        price = close[i]
        
        if in_pos:
            exit_triggered = False
            reason = ""
            pnl_pct_current = (price - entry_price) / entry_price * 100
            
            # [NEW] Panic Exit (RSI Cut)
            if rsi[i] > rsi_panic_threshold:
                exit_price = price
                exit_triggered = True
                reason = "Panic Exit (RSI)"
            
            # [NEW] Time-Based Exit (Conditional)
            if not exit_triggered:
                bars_held = i - entry_idx
                if bars_held >= max_holding_bars:
                    unrealized_profit_atr = (price - entry_price) / pos_atr if pos_atr > 0 else 0.0
                    if unrealized_profit_atr <= time_exit_profit_threshold:
                        exit_price = price
                        exit_triggered = True
                        reason = "Time Cut (No Profit)"
            
            # --- 1. Main Trend Exit (ATR Trailing or SAR) ---
            if not exit_triggered:
                if exit_type == 'ATR':
                    # Trailing Stop Update
                    if high[i] > highest:
                        highest = high[i]
                        new_stop = highest - (pos_atr * atr_mult)
                        if new_stop > stop_price:
                            stop_price = new_stop
                    
                    if low[i] <= stop_price:
                        exit_price = stop_price if stop_price > low[i] else low[i]
                        exit_triggered = True
                        reason = "ATR Trailing Stop"
                else: # PARABOLIC_SAR
                    current_sar = sar[i]
                    if current_sar > 0 and low[i] <= current_sar:
                        exit_price = current_sar if current_sar > low[i] else low[i]
                        exit_triggered = True
                        reason = "Parabolic SAR"
                    elif low[i] <= stop_price: # Safety Stop
                        exit_price = stop_price if stop_price > low[i] else low[i]
                        exit_triggered = True
                        reason = "Safety Stop Loss"
            
            # --- 2. Take Profit (Safety Net) ---
            if not exit_triggered and use_tp and tp_price > 0 and high[i] >= tp_price:
                exit_price = tp_price
                exit_triggered = True
                reason = "Take Profit"
            
            # --- 3. Trend Reversal (Emergency) ---
            if not exit_triggered and trend_dir[i-1] == -1:
                exit_price = price
                exit_triggered = True
                reason = "Trend Reversal"
                
            if exit_triggered:
                # Sell
                revenue = coin * exit_price * (1 - fee_rate)
                balance += revenue 
                coin = 0.0
                in_pos = False
                
                ret_pct = (exit_price - entry_price) / entry_price * 100
                trades_log.append(ret_pct)
                detailed_log.append({
                    'time': times[i],
                    'type': 'SELL',
                    'price': exit_price,
                    'return': ret_pct,
                    'balance': balance,
                    'reason': reason
                })
        else:
            # Entry Logic
            if np.isnan(entry_upper[i]): continue
            if strength[i] == 0: continue
            
            if trend_dir[i] == 1 and price > entry_upper[i]:
                if rsi[i] >= rsi_entry_max:
                    continue
                if natr[i] < natr_entry_min:
                    continue
                # [NEW] Regime Detection (Position Sizing)
                regime_mult = 1.0
                if use_dynamic_risk:
                    if natr[i] > natr_panic_threshold:
                        regime_mult = panic_regime_multiplier
                    elif hurst[i] > hurst_threshold and natr[i] > strong_regime_natr:
                        regime_mult = strong_regime_multiplier
                    elif hurst[i] < weak_regime_hurst:
                        regime_mult = weak_regime_multiplier
                
                fill_price = price * (1 + slippage)
                entry_price = fill_price
                
                # Sizing based on Risk Per Trade with 100M cap
                target_risk = risk_per_trade * regime_mult
                if target_risk > 0.99: target_risk = 0.99
                
                current_capital = balance if use_compounding else min(balance, initial_balance)
                cost = current_capital * target_risk
                cost = min(cost, max_capital_usage)
                cost = min(cost, balance)
                coin = (cost * (1 - fee_rate)) / fill_price
                balance -= cost
                
                in_pos = True
                entry_idx = i
                highest = fill_price
                pos_atr = atr[i]
                
                # Calc SL/TP
                if sl_type == 'ATR':
                    stop_price = fill_price - (atr[i] * atr_sl_mult)
                else:
                    stop_price = fill_price * (1 - sl_pct)
                    
                if use_tp:
                    tp_price = fill_price + (atr[i] * tp_atr_mult)
                else:
                    tp_price = 0.0
                    
                detailed_log.append({
                    'time': times[i],
                    'type': 'BUY',
                    'price': fill_price,
                    'stop_loss': stop_price,
                    'take_profit': tp_price
                })
                
        # Update Equity
        eq_val = balance + (coin * price)
        equity_curve.append(eq_val)

    final_val = balance + (coin * close[-1])
    
    # MDD Calculation
    mdd = 0.0
    if len(equity_curve) > 0:
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            dd = (eq_arr - peak) / peak * 100
            mdd = np.nanmin(dd) if not np.all(np.isnan(dd)) else 0.0

    ret_pct = (final_val - initial_balance) / initial_balance * 100
    
    if return_series:
        return ret_pct, mdd, trades_log, detailed_log, equity_curve
    else:
        return ret_pct, mdd
