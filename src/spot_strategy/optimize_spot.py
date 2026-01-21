
import optuna
import pandas as pd
import numpy as np
import sys
import os
import logging
import time
from datetime import datetime
from numba import njit
import sqlite3
import pyupbit
from dotenv import load_dotenv
import threading

# Load Environment Variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


from src.spot_strategy.upbit_client import UpbitClient
from src.strategy.strategies import UltimateStrategy
from src.spot_strategy.engine_fast_spot import BacktestEngineFastSpot
from config.optimization_config_modes import GET_SEARCH_SPACE, BASE_SEARCH_SPACE
from config.settings import TRAIN_CUTOFF_DATE

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpotOptimizer")

# Data Settings
SPOT_START_DATE = "2018-01-01"
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)



def load_all_timeframes(symbols, start_date, end_date, timeframes):
    """
    Load all necessary timeframe data (Static Cache - like Futures)
    """
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    client = UpbitClient(access, secret)
    
    symbols_data = {s: {} for s in symbols}
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

    for symbol in symbols:
        for tf in timeframes:
            filename = f"{symbol.replace('/', '_')}_{tf}_{start_date.replace('-','')}_{end_date.replace('-','')}_spot.csv"
            filepath = os.path.join(DATA_DIR, filename)
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                symbols_data[symbol][tf] = df
            else:
                print(f"  ⬇️ Downloading {symbol}-{tf}...")
                all_data = []
                since = start_ts
                while since < end_ts:
                    df_batch = client.fetch_ohlcv(symbol, tf, since=since, limit=200)
                    if df_batch is None or df_batch.empty: break
                    all_data.append(df_batch)
                    since = int(df_batch.iloc[-1]['timestamp']) + 1
                    time.sleep(0.1)
                
                if all_data:
                    df = pd.concat(all_data, ignore_index=True)
                    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
                    df.to_csv(filepath, index=False)
                    symbols_data[symbol][tf] = df
                    
    return symbols_data

@njit(nogil=True, cache=True)
def backtest_loop_spot_numba(
    close, high, low,
    entry_upper,
    trend_dir, strength_filter, volume_ratio, atr, parabolic_sar,
    initial_balance, fee_rate, slippage_rate,
    exit_type, # 0: ATR, 1: SAR
    stop_loss_type, stop_loss_pct, atr_sl_mult,
    atr_mult, risk_per_trade,
    use_volume_filter, vol_threshold,
    use_take_profit, tp_atr_mult,
    max_holding_bars, trailing_activation_atr  # [NEW]
):
    """
    Numba JIT-compiled backtest loop for Spot (Long-Only).
    """
    n = len(close)
    balance = initial_balance
    coin = 0.0
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    highest = 0.0
    pos_atr = 0.0
    stop_price = 0.0
    tp_price = 0.0
    
    max_trades = 30000
    trades = np.zeros((max_trades, 3)) # [pnl_pct, duration, dummy]
    trade_count = 0
    
    equity_curve = np.zeros(n)
    
    for i in range(n):
        c_price = close[i]
        c_high = high[i]
        c_low = low[i]
        
        # --- POSITION MANAGEMENT ---
        if in_position:
            exit_triggered = False
            exit_price = 0.0
            
            # [NEW] Time-Based Exit
            bars_held = i - entry_idx
            if bars_held >= max_holding_bars:
                exit_price = c_price
                exit_triggered = True
            
            # --- 1. Main Trend Exit (ATR Trailing or SAR) ---
            if not exit_triggered:
                if exit_type == 0: # ATR Trailing Stop
                    if c_high > highest:
                        highest = c_high
                        
                        # [NEW] Conditional Trailing Activation
                        # Only activate if profit exceeds threshold (e.g., 3 ATR)
                        unrealized_profit_atr = 0.0
                        if pos_atr > 0:
                            unrealized_profit_atr = (highest - entry_price) / pos_atr
                        
                        if unrealized_profit_atr >= trailing_activation_atr:
                            new_stop = highest - (pos_atr * atr_mult)
                            if new_stop > stop_price:
                                stop_price = new_stop
                    
                    if c_low <= stop_price:
                        exit_price = stop_price if stop_price > c_low else c_low
                        exit_triggered = True
                else: # Parabolic SAR Exit
                    # Ensure SAR is valid (> 0)
                    current_sar = parabolic_sar[i]
                    if current_sar > 0:
                        if c_low <= current_sar:
                            exit_price = current_sar if current_sar > c_low else c_low
                            exit_triggered = True
                    
                    # Safety Stop (Static Initial SL)
                    if not exit_triggered and c_low <= stop_price:
                        exit_price = stop_price if stop_price > c_low else c_low
                        exit_triggered = True
            
            # --- 2. Take Profit (Safety Net) ---
            if not exit_triggered and use_take_profit and tp_price > 0 and c_high >= tp_price:
                exit_price = tp_price
                exit_triggered = True
                
            # --- 3. Trend Reversal (Emergency Exit) ---
            if not exit_triggered and i > 0 and trend_dir[i-1] == -1:
                exit_price = c_price
                exit_triggered = True
                
            if exit_triggered:
                # Sell All
                revenue = coin * exit_price * (1 - fee_rate)
                balance += revenue  # Corrected: Add revenue back to remaining balance
                coin = 0.0
                in_position = False
                
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                if trade_count < max_trades:
                    trades[trade_count] = [pnl_pct, float(i - entry_idx), 0.0]
                    trade_count += 1
        
        # --- ENTRY LOGIC ---
        else:
            # Filters
            if np.isnan(entry_upper[i]): continue
            if strength_filter[i] == 0: continue
            if use_volume_filter and volume_ratio[i] < vol_threshold: continue
            
            # Long Entry
            if trend_dir[i] == 1 and c_price > entry_upper[i]:
                fill_price = c_price * (1 + slippage_rate)
                entry_price = fill_price
                
                # Initial SL Calc (Safety Net)
                if stop_loss_type == 1: # ATR
                    stop_price = fill_price - (atr[i] * atr_sl_mult)
                else: # Fixed
                    stop_price = fill_price * (1 - stop_loss_pct)
                
                # TP Calc (Safety Net)
                if use_take_profit:
                    tp_price = fill_price + (atr[i] * tp_atr_mult)
                else:
                    tp_price = 0.0
                
                # Sizing based on Risk Per Trade with 100M cap
                cost = balance * risk_per_trade
                max_position_value = 100_000_000.0  # 1억 KRW cap for realistic sizing
                cost = min(cost, max_position_value)
                coin = (cost * (1 - fee_rate)) / fill_price
                balance -= cost
                
                in_position = True
                entry_idx = i
                highest = fill_price
                pos_atr = atr[i]
        
        equity_curve[i] = balance + (coin * c_price)

    return trades[:trade_count], equity_curve, balance + (coin * close[-1])

def suggest_params(trial, search_space):
    """
    Generate trial parameters from search space with conditional dependency pruning.
    Only suggests parameters that are actually used by the selected strategy configuration.
    
    Efficiency Gain: 60~70% reduction in search space by skipping irrelevant parameters.
    """
    params = {}
    
    # === Phase 1: Core Strategy Selection ===
    for key in ['ENTRY_TYPE', 'TREND_FILTER_TYPE', 'STRENGTH_FILTER_TYPE', 'EXIT_TYPE', 
                'STOP_LOSS_TYPE', 'USE_TAKE_PROFIT', 'USE_VOLUME_FILTER', 'TIMEFRAME']:
        if key in search_space:
            spec = search_space[key]
            if spec['type'] == 'categorical':
                params[key] = trial.suggest_categorical(key, spec['choices'])
    
    # === Phase 2: Entry-Type Dependent Parameters ===
    entry_type = params.get('ENTRY_TYPE', 'DONCHIAN')
    
    if entry_type == 'BOLLINGER':
        if 'BB_STD' in search_space:
            spec = search_space['BB_STD']
            params['BB_STD'] = trial.suggest_float('BB_STD', spec['low'], spec['high'], step=spec.get('step'))
    
    # === Phase 3: Trend-Filter Dependent Parameters ===
    trend_filter = params.get('TREND_FILTER_TYPE', 'EMA')
    
    if trend_filter == 'SUPERTREND':
        for key in ['SUPERTREND_MULT', 'SUPERTREND_PERIOD']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                if spec['type'] == 'float':
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_float(key, spec['low'], spec['high'], step=spec.get('step'))
                elif spec['type'] == 'int':
                    params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif trend_filter == 'MACD':
        for key in ['MACD_FAST', 'MACD_SLOW', 'MACD_SIGNAL']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif trend_filter == 'ICHIMOKU':
        for key in ['ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_B']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif trend_filter == 'VWAP':
        if 'VWAP_STD_MULT' in search_space:
            spec = search_space['VWAP_STD_MULT']
            params['VWAP_STD_MULT'] = trial.suggest_float('VWAP_STD_MULT', spec['low'], spec['high'], step=spec.get('step'))
    
    # === Phase 4: Strength-Filter Dependent Parameters ===
    strength_filter = params.get('STRENGTH_FILTER_TYPE', 'NONE')
    
    if strength_filter in ['ADX', 'VHF', 'MFI', 'RSI', 'STOCHASTIC', 'STOCH_RSI']:
        if 'STRENGTH_FILTER_PERIOD' in search_space:
            spec = search_space['STRENGTH_FILTER_PERIOD']
            use_log = spec.get('log', False)
            params['STRENGTH_FILTER_PERIOD'] = trial.suggest_int('STRENGTH_FILTER_PERIOD', spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_int('STRENGTH_FILTER_PERIOD', spec['low'], spec['high'], step=spec.get('step'))
    
    if strength_filter == 'VHF':
        if 'VHF_THRESHOLD' in search_space:
            spec = search_space['VHF_THRESHOLD']
            params['VHF_THRESHOLD'] = trial.suggest_float('VHF_THRESHOLD', spec['low'], spec['high'], step=spec.get('step'))
    
    elif strength_filter == 'MFI':
        if 'MFI_THRESHOLD' in search_space:
            spec = search_space['MFI_THRESHOLD']
            params['MFI_THRESHOLD'] = trial.suggest_int('MFI_THRESHOLD', spec['low'], spec['high'], step=spec.get('step'))
    
    elif strength_filter == 'RSI':
        for key in ['RSI_OVERBOUGHT', 'RSI_OVERSOLD']:
            if key in search_space:
                spec = search_space[key]
                params[key] = trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif strength_filter == 'STOCHASTIC':
        for key in ['STOCH_OVERBOUGHT', 'STOCH_OVERSOLD']:
            if key in search_space:
                spec = search_space[key]
                params[key] = trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif strength_filter == 'STOCH_RSI':
        for key in ['STOCH_RSI_OVERBOUGHT', 'STOCH_RSI_OVERSOLD']:
            if key in search_space:
                spec = search_space[key]
                params[key] = trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif strength_filter == 'CMF':
        for key in ['CMF_PERIOD', 'CMF_THRESHOLD']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                if spec['type'] == 'int':
                    params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
                else:
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif strength_filter == 'HURST':
        for key in ['HURST_PERIOD', 'HURST_TREND_THRESHOLD', 'HURST_RANDOM_THRESHOLD']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                if spec['type'] == 'int':
                    params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
                else:
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], step=spec.get('step'))
    
    # === Phase 5: Exit-Type Dependent Parameters ===
    exit_type = params.get('EXIT_TYPE', 'ATR')
    
    if exit_type == 'PARABOLIC_SAR':
        if 'SAR_STEP' in search_space:
            spec = search_space['SAR_STEP']
            params['SAR_STEP'] = trial.suggest_float('SAR_STEP', spec['low'], spec['high'], step=spec.get('step'))
    
    # === Phase 6: Common Parameters (Always Used) ===
    # [CRITICAL] Handle STOP_LOSS_TYPE logical conflict first
    stop_loss_type = params.get('STOP_LOSS_TYPE', 'FIXED')
    
    if stop_loss_type == 'FIXED':
        if 'STOP_LOSS_PCT' in search_space:
            spec = search_space['STOP_LOSS_PCT']
            params['STOP_LOSS_PCT'] = trial.suggest_float('STOP_LOSS_PCT', spec['low'], spec['high'], step=spec.get('step'))
    
    elif stop_loss_type == 'ATR':
        if 'ATR_STOP_LOSS_MULT' in search_space:
            spec = search_space['ATR_STOP_LOSS_MULT']
            use_log = spec.get('log', False)
            if use_log:
                params['ATR_STOP_LOSS_MULT'] = trial.suggest_float('ATR_STOP_LOSS_MULT', spec['low'], spec['high'], log=True)
            else:
                params['ATR_STOP_LOSS_MULT'] = trial.suggest_float('ATR_STOP_LOSS_MULT', spec['low'], spec['high'], step=spec.get('step'))
    
    # [CRITICAL] Handle USE_TAKE_PROFIT logical conflict
    use_take_profit = params.get('USE_TAKE_PROFIT', False)
    
    if use_take_profit:
        if 'TAKE_PROFIT_ATR_MULT' in search_space:
            spec = search_space['TAKE_PROFIT_ATR_MULT']
            use_log = spec.get('log', False)
            if use_log:
                params['TAKE_PROFIT_ATR_MULT'] = trial.suggest_float('TAKE_PROFIT_ATR_MULT', spec['low'], spec['high'], log=True)
            else:
                params['TAKE_PROFIT_ATR_MULT'] = trial.suggest_float('TAKE_PROFIT_ATR_MULT', spec['low'], spec['high'], step=spec.get('step'))
    
    # [CRITICAL] Handle USE_VOLUME_FILTER logical conflict
    use_volume_filter = params.get('USE_VOLUME_FILTER', False)
    
    if use_volume_filter:
        for key in ['VOLUME_THRESHOLD_MULT', 'VOLUME_MA_PERIOD']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                if spec['type'] == 'float':
                    if use_log:
                        params[key] = trial.suggest_float(key, spec['low'], spec['high'], log=True)
                    else:
                        params[key] = trial.suggest_float(key, spec['low'], spec['high'], step=spec.get('step'))
                elif spec['type'] == 'int':
                    if use_log:
                        params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=True)
                    else:
                        params[key] = trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    # Other common parameters (no conflicts)
    common_keys = [
        'ENTRY_PERIOD', 'MA_PERIOD', 'ATR_PERIOD',
        'ATR_MULTIPLIER',
        'ADX_THRESHOLD',
        'RISK_PER_TRADE', 'LEVERAGE',
        'MAX_HOLDING_BARS', 'TRAILING_ACTIVATION_ATR',
        'RISK_PER_TRADE_SPOT'
    ]
    
    for key in common_keys:
        if key in search_space:
            spec = search_space[key]
            use_log = spec.get('log', False)
            
            if spec['type'] == 'float':
                if use_log:
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], log=True)
                else:
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], step=spec.get('step'))
            elif spec['type'] == 'int':
                if use_log:
                    params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=True)
                else:
                    params[key] = trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    return params

def calculate_score(ret, mdd, trades_df, mode="DAY"):
    """
    SQN Hybrid Objective v3 (Final) for SPOT
    """
    import numpy as np

    if trades_df.empty:
        return -10000

    N = len(trades_df)
    
    # 1. Individual Trade Returns (%)
    # Engine must provide 'pnl_pct' for proper SQN calculation
    if 'pnl_pct' not in trades_df.columns:
        raise ValueError("trades_df must contain 'pnl_pct' column for SQN calculation. Check engine's run().")
    
    returns = trades_df['pnl_pct'].values

    r_avg = np.mean(returns)
    r_std = np.std(returns) if len(returns) > 1 else 100.0
    if r_std == 0: r_std = 0.001
    
    # --- Helper: Soft Sigmoid Normalization ---
    def soft_sigmoid(x, center, steepness):
        z = -steepness * (x - center)
        # Prevent overflow: clip to [-500, 500] (exp(500) is still safe)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(z))

    # --- Component 1: SQN ---
    sqn = (np.sqrt(N) * r_avg) / r_std
    sqn_score = soft_sigmoid(sqn, center=2.5, steepness=0.5)

    # --- Component 2: Calmar Ratio ---
    abs_mdd = abs(mdd) if mdd != 0 else 0.01
    calmar = ret / abs_mdd
    calmar_score = soft_sigmoid(calmar, center=2.5, steepness=0.4) # Spot is slightly more conservative

    # --- Component 3: Profit Factor ---
    pos_sum = np.sum(returns[returns > 0])
    neg_sum = abs(np.sum(returns[returns < 0]))
    pf = pos_sum / neg_sum if neg_sum > 0 else 3.0
    pf_score = soft_sigmoid(pf, center=1.3, steepness=1.5) # PF expectation is lower in Spot

    # --- Component 4: Smooth MDD Penalty ---
    # Spot: No leverage, so more tolerant of MDD
    # Center at -20% with moderate steepness
    mdd_penalty = soft_sigmoid(-abs_mdd, center=-20, steepness=0.25)

    # --- Component 5: Soft Trade Count Penalty ---
    MIN_TRADES_MAP = {'SCALP': 500, 'DAY': 150, 'SWING': 30, 'ALL': 100}
    min_trades = MIN_TRADES_MAP.get(mode.upper(), 100)
    # Replaces 'if N < min_trades: return -10000'
    # Sigmoid alone provides sufficient gradient
    trade_penalty = soft_sigmoid(N, center=min_trades, steepness=0.1)
    
    # Hard floor only for extreme cases
    if N < 10:
        return -10000

    # --- Final Score: Multiplicative ---
    final_score = sqn_score * calmar_score * pf_score * mdd_penalty * trade_penalty * 1000
    
    return final_score


def objective(trial, symbols_data, search_space, mode="DAY"):
    params = suggest_params(trial, search_space)
    
    # [VALIDATION] Enforce Logical Constraints
    if params.get('TREND_FILTER_TYPE') == 'MACD':
        if params.get('MACD_FAST', 12) >= params.get('MACD_SLOW', 26):
            return -10000 # Invalid trial penalty
            
    tf = params['TIMEFRAME']
    symbol_scores = []
    
    # [UNIVERSAL] Loop through all symbols to find Robust Params
    for symbol, data_map in symbols_data.items():
        if tf not in data_map: continue
        
        # Get data (no need to copy - Engine doesn't modify it)
        df = data_map[tf]
        
        # Create Strategy
        strategy = UltimateStrategy(f"Opt_{symbol}", params)
        
        # Create Engine (signals generated inside _prepare_data)
        engine = BacktestEngineFastSpot(
            df, 
            strategy,
            backtest_loop_spot_numba,  # Inject backtest function
            initial_balance=10_000_000,
            fee_rate=0.0005,      # Upbit: 0.05%
            slippage_rate=0.0003  # Upbit: 0.03%
        )
        # Inject risk parameter
        engine.risk_per_trade = params.get('RISK_PER_TRADE_SPOT', 0.99)
        
        # Run backtest
        try:
            result = engine.run()
        except Exception as e:
            return -10000
        
        # Extract metrics
        ret = result['total_return_pct']
        mdd = result['mdd_pct']
        num_trades = result['total_trades']
        win_rate = result['win_rate']
        
        # Calculate Profit Factor (PF)
        trades_df = result['trades_df']
        pf = 0.0
        if not trades_df.empty and 'pnl_pct' in trades_df.columns:
            gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
            gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        
        # Calculate Score per Symbol
        score = calculate_score(ret, mdd, trades_df, mode=mode)
        
        # [SPEED UP] Short-circuit: If any symbol performs poorly, fail the trial immediately
        # No need to test other symbols if this one failed
        if score < 5:
            return -10000

        symbol_scores.append(score)
        
        # Set user attrs for analysis
        trial.set_user_attr(f"ret_{symbol}", float(ret))
        trial.set_user_attr(f"mdd_{symbol}", float(mdd))
        trial.set_user_attr(f"trades_{symbol}", int(num_trades))
        trial.set_user_attr(f"winrate_{symbol}", float(win_rate))
        trial.set_user_attr(f"pf_{symbol}", float(pf))

    if not symbol_scores: return -10000
    
    # [UNIVERSAL] Combine using Harmonic Mean for Robustness
    # Formula: n / (1/x1 + 1/x2 + ... + 1/xn)
    offset = 6000
    shifted_scores = [s + offset for s in symbol_scores]
    
    if any(s <= 0 for s in shifted_scores):
        return -10000
        
    harmonic_mean = len(shifted_scores) / sum(1/s for s in shifted_scores)
    final_score = harmonic_mean - offset
    
    trial.set_user_attr("score_avg", final_score)
    return final_score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None, help="Number of optimization trials")
    parser.add_argument("--symbols", type=str, default="KRW-BTC,KRW-ETH") 
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--mode", type=str, default="DAY", choices=["SCALP", "DAY", "SWING", "ALL"])
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    mode = args.mode.upper()
    
    MODE_TRIALS_MAP = {
        'SCALP': 3000, 'DAY': 2500, 'SWING': 2700, 'ALL': 3000
    }
    
    trials = args.trials if args.trials is not None else MODE_TRIALS_MAP.get(mode, 2500)
    
    # Adjust Logging (Quiet Optuna)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("SpotOptimizer").setLevel(logging.WARNING)
    
    # Get Search Space & Timeframes
    search_space = GET_SEARCH_SPACE(mode, market_type='spot')
    timeframes = search_space['TIMEFRAME']['choices']
    
    print(f"\n{'='*70}")
    print(f"🚀 MODE: {mode} SPOT OPTIMIZATION")
    print(f"⏰ Target Timeframes: {timeframes}")
    print(f"{'='*70}\n")
    
    SPOT_END_DATE = "2026-01-16"
    
    print(f" Loading data for symbols: {', '.join(symbols)}")
    
    symbols_data = load_all_timeframes(symbols, SPOT_START_DATE, SPOT_END_DATE, timeframes)
    
    print(f"✂️  Trimming Data for Optimization (Train Period: ~ {TRAIN_CUTOFF_DATE})")
    train_data = {}
    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)
    
    for sym, tf_map in symbols_data.items():
        train_data[sym] = {}
        for tf, df_ in tf_map.items():
            train_data[sym][tf] = df_[df_['datetime'] < cutoff_ts].copy()

    # DB Setup (MySQL)
    from dotenv import load_dotenv
    from urllib.parse import quote_plus
    load_dotenv()
    
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "trading_optuna")
    
    if not all([db_user, db_pass, db_name]):
        print("❌ Error: Missing DB credentials in .env (DB_USER, DB_PASS, DB_NAME)")
        sys.exit(1)
        
    study_name = f"spot_{mode.lower()}_strategy"
    # [CRITICAL] Encode password to handle special characters like '@'
    safe_pass = quote_plus(db_pass)
    storage_url = f"mysql+pymysql://{db_user}:{safe_pass}@{db_host}:{db_port}/{db_name}"
    
    # [Clean Start] Intead of deleting file, delete study from DB
    print(f"🔄 Preparing study: {study_name}")
    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print(f"🗑️  Deleted old study: {study_name}")
    except Exception:
        pass # Study might not exist

    # [Performance] Optimize for parallel MySQL access
    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={
            "pool_size": max(30, args.jobs * 2),  # Scale with jobs
            "max_overflow": 10,                    # Allow burst connections
            "pool_recycle": 3600,
            "pool_pre_ping": True,                 # Validate connections
        }
    )
    
    # [Performance] Use ConstantLiar for parallel efficiency
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=100,     # Random exploration first
        multivariate=True,        # Consider param dependencies
        constant_liar=True,       # Avoid duplicate proposals
        warn_independent_sampling=False,
    )
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage, 
        direction="maximize",
        sampler=sampler
    )
    
    print(f"\n{'='*70}")
    print(f"🔥 STARTING OPTIMIZATION for {study_name}")
    print(f"🛢️  Storage: MySQL ({db_host}/{db_name})")
    print(f"📈 Total Trials: {trials}")
    print(f"💻 Parallel Jobs: {args.jobs}")
    print(f"{'='*70}\n")
    
    # [Performance] Numba JIT Warmup
    print("🔥 Warming up Numba JIT...", end="", flush=True)
    dummy_len = 10
    _dummy_arr = np.ones(dummy_len, dtype=np.float64)
    _dummy_int = np.zeros(dummy_len, dtype=np.int64)
    try:
        backtest_loop_spot_numba(
            _dummy_arr, _dummy_arr, _dummy_arr, # OHLC
            _dummy_arr, # Entry Upper
            _dummy_int, _dummy_int, _dummy_arr, _dummy_arr, _dummy_arr, # Trend, Strength, Vol, ATR, SAR
            10000.0, 0.001, 0.001, # Bal, Fee, Slip
            0, # Exit type
            0, 0.01, 1.5, # SL params
            3.0, 0.99, # ATR Mult, Risk
            False, 1.0, # Vol Filter
            False, 3.0, # TP
            1000, 0.0 # Max Hold, Trailing Act
        )
        print(" Done!")
    except Exception as e:
        print(f" Skipped ({e})")

    try:
        study.optimize(lambda t: objective(t, train_data, search_space, mode=mode), 
                       n_trials=trials,
                       n_jobs=args.jobs,
                       show_progress_bar=True)
                       
    except KeyboardInterrupt:
        print("\n🛑 Optimization Interrupted by User")
    
    print(f"\n{'='*70}")
    print(f"✅ {mode} Optimization Complete!")
    
    if len(study.trials) > 0:
        print(f"🏆 Best Score: {study.best_value:.2f}")
        print(f"✨ Best Params: {study.best_params}")
    print(f"{'='*70}")
    
    print("\n✅ Optimization Complete.")
