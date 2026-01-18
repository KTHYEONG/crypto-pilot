
import pandas as pd
import numpy as np

def prepare_futures_data(hourly_df, daily_df, strategy):
    """
    Unified Data Preparation for Futures.
    Ensures consistent signal generation and merging between Verify and WFA.
    """
    # 0. Ensure daily_df has date_key
    daily_df = daily_df.copy()
    if 'date_key' not in daily_df.columns:
         daily_df['date_key'] = pd.to_datetime(daily_df['datetime']).dt.strftime('%Y-%m-%d')

    # Apply Strategy Indicators to Daily
    daily_df = strategy.generate_signals(daily_df)
    
    # Shift daily indicators forward by 1 day (prevent lookahead)
    shifted_cols = [col for col in daily_df.columns if col not in ['date_key', 'datetime', 'date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
    shifted_daily = daily_df[['date_key'] + shifted_cols].copy()
    for col in shifted_cols:
        shifted_daily[col] = shifted_daily[col].shift(1)
    
    # Merge hourly with daily
    hourly_df = hourly_df.copy()
    hourly_df['date_key'] = pd.to_datetime(hourly_df['datetime']).dt.strftime('%Y-%m-%d')
    
    merged_df = pd.merge(hourly_df, shifted_daily, on='date_key', how='left')
    merged_df.sort_values('datetime', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    return merged_df

def run_backtest_segment_futures(df, params, initial_balance=1000000.0, return_series=False):
    """
    Shared backtest logic (Python Loop) for Verification and Walk-Forward Analysis (Futures).
    Supports LONG and SHORT positions + Leverage.
    """
    balance = initial_balance
    
    # State
    in_pos = False
    pos_side = 0 # 1: Long, -1: Short
    entry_price = 0.0
    amount = 0.0
    entry_fee = 0.0
    balance_at_entry = 0.0
    highest = 0.0
    lowest = 0.0
    pos_atr = 0.0
    stop_price = 0.0
    tp_price = 0.0
    
    # Params
    fee_rate = 0.0005  # 0.05% (Binance Futures taker approx)
    slippage = 0.0005  # 0.05%
    leverage = params.get('LEVERAGE', 1)
    
    atr_mult = params.get('ATR_MULTIPLIER', 3.0)
    exit_type = params.get('EXIT_TYPE', 'ATR') # ATR or PARABOLIC_SAR
    sl_type = params.get('STOP_LOSS_TYPE', 'FIXED')
    sl_pct = params.get('STOP_LOSS_PCT', 0.03)
    atr_sl_mult = params.get('ATR_STOP_LOSS_MULT', 1.5)
    
    use_tp = params.get('USE_TAKE_PROFIT', False)
    tp_atr_mult = params.get('TAKE_PROFIT_ATR_MULT_FUTURES', params.get('TAKE_PROFIT_ATR_MULT', 3.0))
    
    risk_per_trade = params.get('RISK_PER_TRADE_FUTURES', params.get('RISK_PER_TRADE', 0.02))
    
    # Data extraction
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    times = df['datetime'].values
    
    # Indicators (Look for columns without daily_ prefix)
    entry_upper = df.get('entry_upper', df.get('donchian_high', pd.Series([np.nan]*len(df)))).values
    entry_lower = df.get('entry_lower', df.get('donchian_low', pd.Series([np.nan]*len(df)))).values
    
    trend_dir = np.nan_to_num(df.get('trend_direction', pd.Series([1]*len(df))).values, nan=0).astype(int)
    strength = np.nan_to_num(df.get('strength_filter', pd.Series([1]*len(df))).values, nan=0).astype(int)
    atr = np.nan_to_num(df.get('atr', pd.Series([0.0]*len(df))).values, nan=0.0)
    sar = np.nan_to_num(df.get('parabolic_sar', pd.Series([0.0]*len(df))).values, nan=0.0)
    
    # Logs
    trades_log = [] # Returns in %
    detailed_log = []
    equity_curve = []
    
    for i in range(1, len(df)):
        price = close[i]
        c_high = high[i]
        c_low = low[i]
        
        if in_pos:
            exit_triggered = False
            reason = ""
            exit_price = 0.0
            
            # --- Position Management ---
            if pos_side == 1: # LONG
                # Trailing Stop / SAR
                if exit_type == 'ATR':
                    if c_high > highest:
                        highest = c_high
                        new_stop = highest - (pos_atr * atr_mult)
                        if new_stop > stop_price:
                            stop_price = new_stop
                    
                    if c_low <= stop_price:
                        exit_price = stop_price if stop_price > c_low else c_low
                        exit_triggered = True
                        reason = "ATR Trailing Stop"
                else: # SAR
                    current_sar = sar[i]
                    if current_sar > 0:
                         if c_low <= current_sar:
                            exit_price = current_sar if current_sar > c_low else c_low
                            exit_triggered = True
                            reason = "Parabolic SAR"
                    
                    # Safety Stop
                    if not exit_triggered and c_low <= stop_price:
                         exit_price = stop_price if stop_price > c_low else c_low
                         exit_triggered = True
                         reason = "Safety Stop Loss"
                
                # Take Profit
                if not exit_triggered and use_tp and tp_price > 0 and c_high >= tp_price:
                    exit_price = tp_price
                    exit_triggered = True
                    reason = "Take Profit"
                    
                # Trend Reversal
                if not exit_triggered and trend_dir[i-1] == -1:
                    exit_price = price
                    exit_triggered = True
                    reason = "Trend Reversal"
            
            else: # SHORT
                # Trailing Stop / SAR
                if exit_type == 'ATR':
                    if c_low < lowest:
                        lowest = c_low
                        new_stop = lowest + (pos_atr * atr_mult)
                        if new_stop < stop_price:
                            stop_price = new_stop
                    
                    if c_high >= stop_price:
                        exit_price = stop_price if stop_price < c_high else c_high
                        exit_triggered = True
                        reason = "ATR Trailing Stop"
                else: # SAR
                    current_sar = sar[i]
                    if current_sar > 0:
                        if c_high >= current_sar:
                            exit_price = current_sar if current_sar < c_high else c_high
                            exit_triggered = True
                            reason = "Parabolic SAR"
                    
                    # Safety Stop
                    if not exit_triggered and c_high >= stop_price:
                        exit_price = stop_price if stop_price < c_high else c_high
                        exit_triggered = True
                        reason = "Safety Stop Loss"
                
                # Take Profit
                if not exit_triggered and use_tp and tp_price > 0 and c_low <= tp_price:
                    exit_price = tp_price
                    exit_triggered = True
                    reason = "Take Profit"

                # Trend Reversal
                if not exit_triggered and trend_dir[i-1] == 1:
                    exit_price = price
                    exit_triggered = True
                    reason = "Trend Reversal"

            if exit_triggered:
                # Calculate PnL
                if pos_side == 1:
                    pnl = (exit_price - entry_price) * amount
                else:
                    pnl = (entry_price - exit_price) * amount
                
                # Exit Fee
                exit_val = amount * exit_price
                pnl -= (exit_val * fee_rate)
                
                # Margin Release
                margin = (amount * entry_price) / leverage
                balance += margin + pnl
                
                # Log account-level ROI (Net PnL including both fees / Account Balance at Entry)
                net_pnl = pnl - entry_fee
                account_roi = (net_pnl / balance_at_entry) * 100 if balance_at_entry > 0 else 0.0
                trades_log.append(account_roi)
                detailed_log.append({
                    'time': times[i],
                    'type': 'SELL' if pos_side == 1 else 'COVER', 
                    'side': 'LONG' if pos_side == 1 else 'SHORT',
                    'price': exit_price,
                    'return': account_roi,
                    'balance': balance,
                    'reason': reason
                })
                
                in_pos = False
                pos_side = 0
                amount = 0.0
                
        else:
            # --- Entry Logic ---
            if np.isnan(entry_upper[i]) or np.isnan(entry_lower[i]): continue
            if strength[i] == 0: continue
            
            # LONG
            if trend_dir[i] == 1 and price > entry_upper[i]:
                fill_price = price * (1 + slippage)
                
                # SL Calc
                if sl_type == 'ATR':
                    stop_price = fill_price - (atr[i] * atr_sl_mult)
                else:
                    stop_price = fill_price * (1 - sl_pct)
                
                # TP Calc
                if use_tp:
                    tp_price = fill_price + (atr[i] * tp_atr_mult)
                else:
                    tp_price = 0.0

                # Sizing
                dist = abs(fill_price - stop_price)
                if dist > 0:
                    risk_amt = balance * risk_per_trade
                    qty = (risk_amt / dist) * leverage
                else:
                    qty = (balance * 0.01 * leverage) / fill_price
                
                # Cap
                max_qty = (balance * leverage) / fill_price
                qty = min(qty, max_qty)
                
                # Cost Check
                margin = (qty * fill_price) / leverage
                fee = qty * fill_price * fee_rate
                if balance >= (margin + fee):
                    balance_at_entry = balance # Total value before subtracting cost
                    balance -= (margin + fee)
                    amount = qty
                    entry_price = fill_price
                    entry_fee = fee
                    pos_side = 1
                    in_pos = True
                    highest = fill_price
                    lowest = fill_price
                    pos_atr = atr[i]
                    
                    detailed_log.append({
                        'time': times[i],
                        'type': 'BUY',
                        'side': 'LONG',
                        'price': fill_price,
                        'stop_loss': stop_price,
                        'take_profit': tp_price
                    })

            # SHORT
            elif trend_dir[i] == -1 and price < entry_lower[i]:
                fill_price = price * (1 - slippage)
                
                # SL Calc
                if sl_type == 'ATR':
                    stop_price = fill_price + (atr[i] * atr_sl_mult)
                else:
                    stop_price = fill_price * (1 + sl_pct)
                    
                # TP Calc
                if use_tp:
                    tp_price = fill_price - (atr[i] * tp_atr_mult)
                else:
                    tp_price = 0.0
                
                # Sizing
                dist = abs(stop_price - fill_price)
                if dist > 0:
                    risk_amt = balance * risk_per_trade
                    qty = (risk_amt / dist) * leverage
                else:
                    qty = (balance * 0.01 * leverage) / fill_price
                
                # Cap
                max_qty = (balance * leverage) / fill_price
                qty = min(qty, max_qty)
                
                # Cost Check
                margin = (qty * fill_price) / leverage
                fee = qty * fill_price * fee_rate
                if balance >= (margin + fee):
                    balance_at_entry = balance # Total value before subtracting cost
                    balance -= (margin + fee)
                    amount = qty
                    entry_price = fill_price
                    entry_fee = fee
                    pos_side = -1 
                    in_pos = True
                    highest = fill_price
                    lowest = fill_price
                    pos_atr = atr[i]
                    
                    detailed_log.append({
                        'time': times[i],
                        'type': 'SHORT',
                        'side': 'SHORT',
                        'price': fill_price,
                        'stop_loss': stop_price,
                        'take_profit': tp_price
                    })

        # Equity Curve
        if in_pos:
            if pos_side == 1:
                unrealized = (price - entry_price) * amount
            else:
                unrealized = (entry_price - price) * amount
            
            margin = (amount * entry_price) / leverage
            cur_equity = balance + margin + unrealized
            equity_curve.append(cur_equity)
        else:
            equity_curve.append(balance)
            
    # End
    final_val = equity_curve[-1] if equity_curve else balance
    ret_pct = (final_val - initial_balance) / initial_balance * 100
    
    # MDD
    mdd = 0.0
    if equity_curve:
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        with np.errstate(divide='ignore', invalid='ignore'):
            dd = (eq_arr - peak) / peak * 100
            mdd = np.nanmin(dd) if not np.all(np.isnan(dd)) else 0.0
            
    if return_series:
        return ret_pct, mdd, trades_log, detailed_log, equity_curve
    else:
        return ret_pct, mdd
