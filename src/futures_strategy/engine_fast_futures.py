
import pandas as pd
import numpy as np
import logging
from numba import njit
from config.settings import TRADING_FEE_RATE, SLIPPAGE_RATE, FUNDING_FEE_RATE, FUNDING_INTERVAL_HOURS

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
        
        self.fee_rate = TRADING_FEE_RATE
        self.slippage_rate = SLIPPAGE_RATE
        
        self.risk_limits = {'max_daily_loss': 0.05}
        self.logger = logging.getLogger(__name__)
        self._prepare_data()
    
    def _prepare_data(self):
        # [OPTIMIZATION] 1. Pre-calculated keys check
        # 'date_key' generation is expensive. Assume it exists from loader if possible.
        if 'date_key' not in self.daily_df.columns:
             self.daily_df['date_key'] = pd.to_datetime(self.daily_df['datetime']).dt.strftime('%Y-%m-%d')
        
        # Apply Strategy Indicators (Calculates Daily Indicators)
        self.daily_df = self.strategy.generate_signals(self.daily_df)
        
        # [OPTIMIZATION] 2. Fast Shift & Rename
        # Filter essential columns only to reduce copy overhead
        exclude_cols = {'date_key', 'datetime', 'date', 'open', 'high', 'low', 'close', 'volume'}
        indicator_cols = [c for c in self.daily_df.columns if c not in exclude_cols]
        
        # Use simple dictionary for rename to leverage C-speed
        # Shift(1) to prevent lookahead
        shifted_daily = self.daily_df[indicator_cols].shift(1)
        shifted_daily.columns = [f'daily_{c}' for c in indicator_cols]
        shifted_daily['date_key'] = self.daily_df['date_key'] # Restore key matching
        
        # [OPTIMIZATION] 3. Efficient Merge
        # Extract 'daily_open' separately (needed for logic?) - If not needed, skip. 
        # Assuming we just need mapped daily open?
        # daily_open = self.daily_df[['date_key', 'open']].rename(columns={'open': 'daily_open'})
        
        if 'date_key' not in self.hourly_df.columns:
            self.hourly_df['date_key'] = pd.to_datetime(self.hourly_df['datetime']).dt.strftime('%Y-%m-%d')
            
        # Left Join: Hourly <- Daily
        # Using suffix to avoid collisions if any
        self.merged_df = pd.merge(self.hourly_df, shifted_daily, on='date_key', how='left')
        
        # If daily_open is strictly needed (for some reason not covered by standard OHLC from hourly)
        # But usually hourly data has its own context. Daily open is mostly for references.
        # self.merged_df = pd.merge(self.merged_df, daily_open, on='date_key', how='left')
        
        # Sort & Reset (Only if needed - usually data comes sorted)
        # self.merged_df.sort_values('datetime', inplace=True) 
        # self.merged_df.reset_index(drop=True, inplace=True)
    
    def run(self):
        self.logger.info(f"Running FAST backtest for {self.strategy.name}...")
        
        # Extract all columns as numpy arrays for speed
        df = self.merged_df
        n = len(df)
        
        # Price columns
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
        
        # Run Numba-accelerated loop
        trades, final_balance = backtest_loop_numba(
            close, high, low, volume_ratio,
            entry_upper, entry_lower,
            trend_dir, strength_filter, atr, parabolic_sar,
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
            max_holding_bars, trailing_activation_atr  # [NEW]
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
        
        # ✅ ADD: Calculate pnl_pct (percentage return per trade)
        # pnl_pct = (pnl / position_size) * 100
        # position_size approximated as: entry_price * amount (we don't have amount, so use entry_price as proxy)
        # Since we use risk-based sizing, we normalize by initial balance instead
        trades_df['pnl_pct'] = (trades_df['pnl'] / self.initial_balance) * 100
        
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



@njit(nogil=True, cache=True)
def backtest_loop_numba(
    close, high, low, volume_ratio,
    entry_upper, entry_lower,
    trend_dir, strength_filter, atr, parabolic_sar,
    initial_balance, leverage, fee_rate, slippage_rate,
    exit_type, stop_loss_type, stop_loss_pct, atr_sl_mult,
    atr_mult,
    risk_per_trade,
    use_volume_filter, vol_threshold,
    use_take_profit, tp_atr_mult,
    timestamps, funding_fee_rate, funding_interval_hours,
    max_holding_bars, trailing_activation_atr  # [NEW] Time-based exit & Trailing activation threshold
):
    """
    Numba JIT-compiled backtest loop for maximum speed.
    펀딩비(Funding Fee) 반영: 8시간마다 포지션 가치의 0.01% 차감
    [NEW] MAX_HOLDING_BARS: 일정 시간 후 강제 청산 (기회비용 관리)
    [NEW] TRAILING_ACTIVATION_ATR: 일정 이익 이상일 때만 trailing stop 활성화 (이익 보호)
    [NEW] EXIT_TYPE: 0=ATR Trailing, 1=Parabolic SAR
    [NEW] TREND_REVERSAL: trend_dir 변화 시 즉시 청산
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
    tp_price = 0.0 
    
    # Funding Fee Tracking (UTC-based: 00:00, 08:00, 16:00)
    last_funding_hour = -1  # 마지막 펀딩비 적용 시간 (UTC hour: 0, 8, or 16)
    
    # Trades storage (max 30000 trades)
    max_trades = 30000
    trades = np.zeros((max_trades, 6))  # [entry_idx, exit_idx, side, entry_p, exit_p, pnl]
    trade_count = 0
    
    for i in range(n):
        # [SAFETY] Bankruptcy Check
        if balance <= 0:
            break

        # Skip if NaN in entry signals
        if np.isnan(entry_upper[i]) or np.isnan(entry_lower[i]):
            continue
            
        c_price = close[i]
        c_high = high[i]
        c_low = low[i]
        current_atr = atr[i]
        current_timestamp = timestamps[i]
        # --- FUNDING FEE DEDUCTION (UTC 00:00, 08:00, 16:00에만 적용) ---
        if in_position:
            # Convert timestamp (ms) to UTC hour (0-23)
            # Binance funding time: 00:00, 08:00, 16:00 UTC
            current_hour_utc = int((current_timestamp // 1000) % 86400 // 3600)  # 0-23
            
            # Check if current time is a funding hour (0, 8, 16)
            is_funding_hour = (current_hour_utc in (0, 8, 16))
            
            # Apply funding fee only once per funding period
            if is_funding_hour and last_funding_hour != current_hour_utc:
                # 펀딩비 = 포지션 가치(notional value) * funding_fee_rate
                notional_value = amount * c_price
                funding_cost = notional_value * funding_fee_rate
                balance -= funding_cost
                
                # 이번 펀딩 시간 기록 (중복 차감 방지)
                last_funding_hour = current_hour_utc
                
                # [SAFETY] 펀딩비로 인한 파산 체크
                if balance <= 0:
                    # 강제 청산 처리
                    exit_price = c_price
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
                    break
        
        # --- POSITION MANAGEMENT ---
        if in_position:
            exit_triggered = False
            exit_price = 0.0
            
            # [NEW] Time-Based Exit (Opportunity Cost Management)
            bars_held = i - entry_idx
            if bars_held >= max_holding_bars:
                exit_price = c_price  # Exit at current close
                exit_triggered = True
            
            # Update extremes and TP check
            if pos_side == 1: # LONG
                if c_high > highest:
                    highest = c_high
                    
                    # [NEW] Conditional Trailing Stop Activation
                    # Only activate trailing if profit exceeds activation threshold
                    if exit_type == 0:  # ATR Trailing mode only
                        unrealized_profit_atr = (highest - entry_price) / pos_atr if pos_atr > 0 else 0
                        
                        if unrealized_profit_atr >= trailing_activation_atr:
                            # Update Stop Price (Trailing) - ONLY when activation condition met
                            new_stop = highest - (pos_atr * atr_mult)
                            if new_stop > stop_price:
                                stop_price = new_stop
                
                # [NEW] 1. Parabolic SAR Exit (if enabled)
                if not exit_triggered and exit_type == 1:  # SAR mode
                    current_sar = parabolic_sar[i]
                    if current_sar > 0 and c_price < current_sar:
                        exit_price = c_price
                        exit_triggered = True
                
                # [NEW] 2. Trend Reversal Exit
                if not exit_triggered and trend_dir[i] == -1:
                    exit_price = c_price
                    exit_triggered = True

                # 3. Check Stop Loss
                if not exit_triggered and c_low <= stop_price:
                    exit_price = min(c_low, stop_price)
                    exit_triggered = True
                
                # 4. Then Check Take Profit
                elif not exit_triggered and use_take_profit and tp_price > 0 and c_high >= tp_price:
                    exit_price = tp_price
                    exit_triggered = True
                    
            elif pos_side == -1: # SHORT
                if c_low < lowest:
                    lowest = c_low
                    
                    # [NEW] Conditional Trailing Stop Activation
                    if exit_type == 0:  # ATR Trailing mode only
                        unrealized_profit_atr = (entry_price - lowest) / pos_atr if pos_atr > 0 else 0
                        
                        if unrealized_profit_atr >= trailing_activation_atr:
                            new_stop = lowest + (pos_atr * atr_mult)
                            if new_stop < stop_price:
                                stop_price = new_stop
                
                # [NEW] 1. Parabolic SAR Exit (if enabled)
                if not exit_triggered and exit_type == 1:  # SAR mode
                    current_sar = parabolic_sar[i]
                    if current_sar > 0 and c_price > current_sar:
                        exit_price = c_price
                        exit_triggered = True
                
                # [NEW] 2. Trend Reversal Exit
                if not exit_triggered and trend_dir[i] == 1:
                    exit_price = c_price
                    exit_triggered = True
                
                # 3. Check Stop Loss
                if not exit_triggered and c_high >= stop_price:
                    exit_price = max(c_high, stop_price)
                    exit_triggered = True
                
                # 4. Then Check Take Profit
                elif not exit_triggered and use_take_profit and tp_price > 0 and c_low <= tp_price:
                    exit_price = tp_price
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
                
                # Balance Update
                margin = (amount * entry_price) / leverage
                balance += margin + pnl  # pnl already includes exit_fee
                
                trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl]
                trade_count += 1
                in_position = False
                
                # [SAFETY] Bankruptcy check after exit
                if balance <= 0:
                    break
                    
                continue

        # --- ENTRY LOGIC ---
        else:
            if trade_count >= max_trades:
                break
            
            # Volume Filter Check (Ratio based)
            vol_pass = True
            if use_volume_filter:
                # volume_ratio[i] is shifted daily ratio
                # Meaning: "Did yesterday's volume exceed average?"
                if volume_ratio[i] < vol_threshold:
                    vol_pass = False
            
            if not vol_pass:
                continue

            # Strength Filter
            if strength_filter[i] == 0:
                continue

            # LONG Entry
            if c_price > entry_upper[i] and trend_dir[i] == 1:
                fill_price = c_price * (1 + slippage_rate)
                
                # 1. SL Calc
                if stop_loss_type == 1: # ATR Based
                    stop_price = fill_price - (current_atr * atr_sl_mult)
                else: # Fixed %
                    stop_price = fill_price * (1 - stop_loss_pct)
                
                # TP Price
                if use_take_profit:
                    tp_price = fill_price + (current_atr * tp_atr_mult)
                else:
                    tp_price = 0.0

                # 2. Risk Sizing
                stop_distance = abs(fill_price - stop_price)
                if stop_distance > 0:
                    risk_amount = balance * risk_per_trade 
                    amount = (risk_amount / stop_distance) * leverage
                else:
                    amount = (balance * 0.01 * leverage) / fill_price
                
                # [SAFETY] Cap amount to max leverage
                max_amount = (balance * leverage) / fill_price
                if amount > max_amount:
                    amount = max_amount

                # [FIX] Deduct Margin + Entry Fee
                required_margin = (amount * fill_price) / leverage
                entry_fee = amount * fill_price * fee_rate
                total_entry_cost = required_margin + entry_fee
                
                if balance >= total_entry_cost:
                    balance -= total_entry_cost
                    in_position = True
                    pos_side = 1
                    entry_price = fill_price
                    entry_idx = i
                    highest = fill_price
                    lowest = fill_price
                    pos_atr = current_atr
                    
            # SHORT Entry
            elif c_price < entry_lower[i] and trend_dir[i] == -1:
                fill_price = c_price * (1 - slippage_rate)
                
                # 1. SL Calc
                if stop_loss_type == 1: # ATR Based
                    stop_price = fill_price + (current_atr * atr_sl_mult)
                else: # Fixed %
                    stop_price = fill_price * (1 + stop_loss_pct)

                # TP Price
                if use_take_profit:
                    tp_dictance = (current_atr * tp_atr_mult)
                    tp_price = fill_price - tp_dictance
                else:
                    tp_price = 0.0
                
                # 2. Risk Sizing
                stop_distance = abs(stop_price - fill_price)
                if stop_distance > 0:
                    risk_amount = balance * risk_per_trade 
                    amount = (risk_amount / stop_distance) * leverage
                else:
                    amount = (balance * 0.01 * leverage) / fill_price
                
                # [SAFETY] Cap amount to max leverage
                max_amount = (balance * leverage) / fill_price
                if amount > max_amount:
                    amount = max_amount
                
                # [FIX] Deduct Margin + Entry Fee
                required_margin = (amount * fill_price) / leverage
                entry_fee = amount * fill_price * fee_rate
                total_entry_cost = required_margin + entry_fee
                
                if balance >= total_entry_cost:
                    balance -= total_entry_cost
                    in_position = True
                    pos_side = -1
                    entry_price = fill_price
                    entry_idx = i
                    highest = fill_price
                    lowest = fill_price
                    pos_atr = current_atr
    
    return trades[:trade_count], balance
