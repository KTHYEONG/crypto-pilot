import pandas as pd
import numpy as np
import logging
from numba import njit

class BacktestEngineFast:
    """
    Numba-accelerated Backtest Engine (5-10x faster)
    """
    def __init__(self, hourly_df, daily_df, strategy, initial_balance=1_000_000):
        self.hourly_df = hourly_df
        self.daily_df = daily_df
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        # injected by optimization script
        self.leverage = 1
        self.risk_per_trade = 0.02
        
        # Fees (from settings)
        from config.settings import TRADING_FEE_RATE, SLIPPAGE_RATE
        self.fee_rate = TRADING_FEE_RATE
        self.slippage_rate = SLIPPAGE_RATE
        
        self.risk_limits = {'max_daily_loss': 0.05}
        self.logger = logging.getLogger(__name__)
        self._prepare_data()
    
    def _prepare_data(self):
        # 0. Ensure daily_df has date_key
        if 'date_key' not in self.daily_df.columns:
             self.daily_df['date_key'] = pd.to_datetime(self.daily_df['datetime']).dt.strftime('%Y-%m-%d')

        # Apply Strategy Indicators
        self.daily_df = self.strategy.generate_signals(self.daily_df)
        
        # Re-ensure date_key exists (in case strategy dropped it)
        if 'date_key' not in self.daily_df.columns:
             self.daily_df['date_key'] = pd.to_datetime(self.daily_df['datetime']).dt.strftime('%Y-%m-%d')
        
        
        # Shift daily indicators forward by 1 day (prevent lookahead)
        shifted_cols = [col for col in self.daily_df.columns if col not in ['date_key', 'datetime', 'date']]
        shifted_daily = self.daily_df[['date_key'] + shifted_cols].copy()
        for col in shifted_cols:
            shifted_daily[col] = shifted_daily[col].shift(1)
        shifted_daily.columns = ['date_key'] + [f'daily_{col}' for col in shifted_cols]
        
        # Merge hourly with daily
        self.hourly_df['date_key'] = pd.to_datetime(self.hourly_df['datetime']).dt.strftime('%Y-%m-%d')
        daily_open = self.daily_df[['date_key', 'open']].rename(columns={'open': 'daily_open'})
        
        self.merged_df = pd.merge(self.hourly_df, shifted_daily, on='date_key', how='left')
        self.merged_df = pd.merge(self.merged_df, daily_open, on='date_key', how='left')
        self.merged_df.sort_values('datetime', inplace=True)
        self.merged_df.reset_index(drop=True, inplace=True)
    
    def run(self):
        self.logger.info(f"Running FAST backtest for {self.strategy.name}...")
        
        # Extract all columns as numpy arrays for speed
        df = self.merged_df
        n = len(df)
        
        # Price columns
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Entry signals (Donchian or other)
        entry_upper = df.get('daily_entry_upper', df.get('daily_donchian_high', pd.Series([np.nan]*n))).values
        entry_lower = df.get('daily_entry_lower', df.get('daily_donchian_low', pd.Series([np.nan]*n))).values
        
        # Trend filter
        trend_dir = df.get('daily_trend_direction', pd.Series([1]*n)).values
        
        # Strength filter
        strength_filter = df.get('daily_strength_filter', pd.Series([1]*n)).values
        
        # ATR for risk
        atr = df.get('daily_atr', close * 0.01).values
        
        # Strategy params
        stop_loss_type = 1 if self.strategy.params.get('STOP_LOSS_TYPE', 'FIXED') == 'ATR' else 0 # 0: FIXED, 1: ATR
        stop_loss_pct = self.strategy.params.get('STOP_LOSS_PCT', 0.03)
        atr_sl_mult = self.strategy.params.get('ATR_STOP_LOSS_MULT', 1.5)
        
        atr_mult = self.strategy.params.get('ATR_MULTIPLIER', 3.0) # For Trailing Stop
        leverage = self.leverage
        
        # Run Numba-accelerated loop
        trades, final_balance = backtest_loop_numba(
            close, high, low,
            entry_upper, entry_lower,
            trend_dir, strength_filter, atr,
            self.initial_balance,
            leverage,
            self.fee_rate,
            self.slippage_rate,
            stop_loss_type, stop_loss_pct, atr_sl_mult,
            atr_mult
        )
        
        self.balance = final_balance
        
        # Convert trades to DataFrame
        self.trades = []
        for i in range(len(trades)):
            if trades[i][0] == 0:  # 0 means no trade
                break
            self.trades.append({
                'entry_time': df.iloc[int(trades[i][0])]['datetime'],
                'exit_time': df.iloc[int(trades[i][1])]['datetime'],
                'side': 'LONG' if trades[i][2] == 1 else 'SHORT',
                'entry_price': trades[i][3],
                'exit_price': trades[i][4],
                'pnl': trades[i][5]
            })
        
        return self.get_results()
    
    def get_results(self):
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
        win_trades = len(trades_df[trades_df['pnl'] > 0])
        loss_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (win_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        
        # MDD Calculation
        cumulative = [self.initial_balance]
        for pnl in trades_df['pnl']:
            cumulative.append(cumulative[-1] + pnl)
        
        cumulative = np.array(cumulative)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
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


@njit
def backtest_loop_numba(
    close, high, low,
    entry_upper, entry_lower,
    trend_dir, strength_filter, atr,
    initial_balance, leverage, fee_rate, slippage_rate,
    stop_loss_type, stop_loss_pct, atr_sl_mult,
    atr_mult
):
    """
    Numba JIT-compiled backtest loop for maximum speed.
    stop_loss_type: 0 (Fixed %), 1 (ATR-based)
    Returns: trades array, final_balance
    """
    n = len(close)
    balance = initial_balance
    
    # Position state
    in_position = False
    pos_side = 0  # 1: LONG, -1: SHORT
    entry_price = 0.0
    entry_idx = 0
    amount = 0.0
    highest = 0.0
    lowest = 0.0
    pos_atr = 0.0
    stop_price = 0.0 # Initial Stop Loss
    
    # Trades storage (max 1000 trades)
    max_trades = 1000
    trades = np.zeros((max_trades, 6))  # [entry_idx, exit_idx, side, entry_p, exit_p, pnl]
    trade_count = 0
    
    for i in range(n):
        # Skip if NaN in entry signals
        if np.isnan(entry_upper[i]) or np.isnan(entry_lower[i]):
            continue
        
        # Exit Logic
        if in_position:
            c_high = high[i]
            c_low = low[i]
            c_close = close[i]
            
            # Update extremes
            if c_high > highest:
                highest = c_high
            if c_low < lowest:
                lowest = c_low
            
            exit_triggered = False
            exit_price = 0.0
            
            if pos_side == 1:  # LONG
                trailing_price = highest - (pos_atr * atr_mult)
                
                # Check Stop Loss First (Highest Priority)
                if c_low <= stop_price:
                    exit_price = stop_price * (1 - slippage_rate)
                    exit_triggered = True
                # Then Trailing Stop
                elif c_low <= trailing_price:
                    exit_price = trailing_price * (1 - slippage_rate)
                    exit_triggered = True
                    
            elif pos_side == -1:  # SHORT
                trailing_price = lowest + (pos_atr * atr_mult)
                
                if c_high >= stop_price:
                    exit_price = stop_price * (1 + slippage_rate)
                    exit_triggered = True
                elif c_high >= trailing_price:
                    exit_price = trailing_price * (1 + slippage_rate)
                    exit_triggered = True
            
            if exit_triggered:
                # Calculate PnL
                if pos_side == 1:
                    pnl = (exit_price - entry_price) * amount
                else:
                    pnl = (entry_price - exit_price) * amount
                
                # Fee
                exit_fee = amount * exit_price * fee_rate
                pnl -= exit_fee
                
                balance += pnl
                
                # Record trade
                if trade_count < max_trades:
                    trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl]
                    trade_count += 1
                
                in_position = False
        
        # Entry Logic
        if not in_position:
            # Strength filter check
            if strength_filter[i] == 0:
                continue
            
            c_price = close[i]
            current_atr = atr[i]
            
            # LONG Entry
            if c_price > entry_upper[i] and trend_dir[i] == 1:
                fill_price = c_price * (1 + slippage_rate)
                amount = (balance * 0.9 * leverage) / fill_price
                cost = amount * fill_price * fee_rate
                
                if balance >= cost:
                    balance -= cost
                    in_position = True
                    pos_side = 1
                    entry_price = fill_price
                    entry_idx = i
                    highest = fill_price
                    lowest = fill_price
                    pos_atr = current_atr
                    
                    # Set Initial Stop Loss
                    if stop_loss_type == 1: # ATR Based
                        stop_price = fill_price - (current_atr * atr_sl_mult)
                    else: # Fixed %
                        stop_price = fill_price * (1 - stop_loss_pct)
                    
            # SHORT Entry
            elif c_price < entry_lower[i] and trend_dir[i] == -1:
                fill_price = c_price * (1 - slippage_rate)
                amount = (balance * 0.9 * leverage) / fill_price
                cost = amount * fill_price * fee_rate
                
                if balance >= cost:
                    balance -= cost
                    in_position = True
                    pos_side = -1
                    entry_price = fill_price
                    entry_idx = i
                    highest = fill_price
                    lowest = fill_price
                    pos_atr = current_atr
                    
                    # Set Initial Stop Loss
                    if stop_loss_type == 1: # ATR Based
                        stop_price = fill_price + (current_atr * atr_sl_mult)
                    else: # Fixed %
                        stop_price = fill_price * (1 + stop_loss_pct)
    
    return trades[:trade_count], balance
