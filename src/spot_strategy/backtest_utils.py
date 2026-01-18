
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
    
    for i in range(1, len(df)):
        price = close[i]
        
        if in_pos:
            exit_triggered = False
            reason = ""
            
            # --- 1. Main Trend Exit (ATR Trailing or SAR) ---
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
                balance += revenue  # Corrected: Add revenue back to remaining balance
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
                fill_price = price * (1 + slippage)
                entry_price = fill_price
                
                # Sizing based on Risk Per Trade with 100M cap
                cost = balance * risk_per_trade
                max_position_value = 100_000_000.0  # 1억 KRW cap
                cost = min(cost, max_position_value)
                coin = (cost * (1 - fee_rate)) / fill_price
                balance -= cost
                
                in_pos = True
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
