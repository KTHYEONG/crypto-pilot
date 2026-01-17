
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load Environment Variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.spot_strategy.upbit_client import UpbitClient
from src.strategy.strategies import UltimateStrategy
from config.optimization_config_ultimate import ULTIMATE_SEARCH_SPACE

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpotOptimizer")

# Data Settings
SPOT_START_DATE = "2018-01-01"
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

import threading

STOP_EVENT = threading.Event()

def fetch_full_history(client, symbol, timeframe, start_date):
    """
    Fetch OHLCV history from Upbit since start_date using CCXT forward fetching.
    Supports incremental updates.
    """
    filename = f"{symbol}_{timeframe}_{start_date.replace('-','')}_spot.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    existing_df = pd.DataFrame()
    # Default start timestamp (ms)
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    since = start_ts
    
    # 1. Load existing cache if available
    if os.path.exists(filepath):
        try:
            existing_df = pd.read_csv(filepath)
            if not existing_df.empty:
                # Ensure datetime column is proper datetime type
                if 'datetime' in existing_df.columns:
                    existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                
                # [FIX] Handle missing 'timestamp' column (compatibility with old format)
                if 'timestamp' not in existing_df.columns and 'datetime' in existing_df.columns:
                    logger.info(f"⚠️ 'timestamp' column missing in {filename}. Generating from 'datetime'...")
                    existing_df['timestamp'] = existing_df['datetime'].astype('int64') // 10**6
                
                # Resume from the last timestamp found
                if 'timestamp' in existing_df.columns:
                    last_row_ts = existing_df.iloc[-1]['timestamp']
                    since = int(last_row_ts) + 1 # Next ms
                    last_dt_str = existing_df.iloc[-1]['datetime']
                    logger.info(f"📂 Found Cache: {symbol}-{timeframe} (up to {last_dt_str})")
                else:
                    logger.warning(f"⚠️ Cache Invalid (No timestamp/datetime): {filename}")
                    existing_df = pd.DataFrame() # Really invalid

        except Exception as e:
            logger.warning(f"⚠️ Cache Error for {symbol}-{timeframe}: {e}. Re-downloading...")
            # Backup corrupt file just in case
            try:
                os.rename(filepath, filepath + ".bak")
            except: pass
            existing_df = pd.DataFrame()
            since = start_ts

    now_ts = int(datetime.now().timestamp() * 1000)
    
    logger.info(f"⬇️ Syncing: {symbol}-{timeframe} since {datetime.fromtimestamp(since/1000)}...")
    
    new_dfs = []
    retry_count = 0
    max_retries = 10
    
    # Batch size for Upbit via CCXT is usually limited (e.g. 200)
    # CCXT handles rate limits, but we loop until we reach 'now'
    
    while since < now_ts:
        if STOP_EVENT.is_set():
            logger.info(f"🛑 Stop Signal Received. Halting {symbol}-{timeframe}...")
            break
            
        try:
            # fetch_ohlcv(symbol, timeframe, since, limit)
            # Upbit max limit is 200
            df = client.fetch_ohlcv(symbol, timeframe, since=since, limit=200)
            
            if df is None or df.empty:
                # If we get empty data but haven't reached 'now', it might be a gap or end of data
                # Try moving forward a bit to skip potential gap
                if now_ts - since > 3600 * 1000: # If gap > 1 hour
                     since += 3600 * 1000 
                     continue
                break
                
            # Filter valid (sometimes ccxt returns inclusive start)
            df = df[df['timestamp'] >= since]
            
            if df.empty:
                # Same gap logic
                if now_ts - since > 3600 * 1000:
                     since += 3600 * 1000 
                     continue
                break
            
            new_dfs.append(df)
            
            # Update 'since' to the last timestamp + 1 ms or appropriate interval
            last_ts = df.iloc[-1]['timestamp']
            
            # Progress Logging (Reduced frequency)
            if len(new_dfs) % 50 == 0:
                curr_date = datetime.fromtimestamp(last_ts/1000).strftime('%Y-%m-%d')
                logger.info(f"⏳ [{symbol}-{timeframe}] Syncing forward... reaching {curr_date}")
            
            # Move pointer
            since = int(last_ts) + 1
            retry_count = 0
            
            # Increased sleep for stability (CCXT handles rate limit but we add buffer)
            time.sleep(0.4) 
            
            # Break if we are close enough to now (e.g., within 1 candle)
            if now_ts - last_ts < 60000: # less than 1 min
                break
                
        except Exception as e:
            retry_count += 1
            wait_time = (2 ** retry_count) # Exponential Backoff: 2, 4, 8, 16...
            logger.warning(f"⚠️ [{symbol}-{timeframe}] Error (Retry {retry_count}/{max_retries}): {e}. Waiting {wait_time}s...")
            
            if retry_count >= max_retries:
                logger.error(f"❌ [{symbol}-{timeframe}] SYNC FAILED after {max_retries} retries.")
                break
                
            time.sleep(wait_time)
            
    # Combine Data
    if not new_dfs:
        if existing_df.empty:
            return symbol, timeframe, None
        full_df = existing_df
    else:
        new_data = pd.concat(new_dfs)
        if not existing_df.empty:
            full_df = pd.concat([existing_df, new_data])
        else:
            full_df = new_data
            
    # Deduplicate and Sort
    full_df = full_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # Save to CSV
    full_df.to_csv(filepath, index=False)
    logger.info(f"✅ Sync Complete: {symbol}-{timeframe} (Total {len(full_df)} candles)")
    
    return symbol, timeframe, full_df

def fetch_all_data_parallel(symbols, timeframes):
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    client = UpbitClient(access, secret)
    symbols_data = {s: {} for s in symbols}
    
    # Parallel settings
    # Since we use CCXT with RateLimit, excessive parallelism might just hit rate limits harder.
    # 2-3 workers is usually safe for Upbit Public/Private mix.
    # But since we are focusing on just '3m' for few symbols, we can run them.
    
    tasks = []
    for s in symbols:
        for tf in timeframes:
            tasks.append((s, tf))
            
    logger.info(f"🚀 Starting Parallel Download for {len(tasks)} tasks...")
    
    # 2 workers are safer for Upbit Public/Private mix to avoid 429
    executor = ThreadPoolExecutor(max_workers=2)
    futures = {executor.submit(fetch_full_history, client, s, tf, SPOT_START_DATE): (s, tf) for s, tf in tasks}
    
    try:
        for future in as_completed(futures):
            symbol, tf, df = future.result()
            if df is not None:
                symbols_data[symbol][tf] = df
    except KeyboardInterrupt:
        logger.info("\n🛑 Interrupted. Shutting down...")
        STOP_EVENT.set()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=False)
                
    return symbols_data

@njit
def backtest_loop_spot_numba(
    close, high, low,
    entry_upper,
    trend_dir, strength_filter, volume_ratio, atr,
    initial_balance, fee_rate, slippage_rate,
    stop_loss_type, stop_loss_pct, atr_sl_mult,
    atr_mult, risk_per_trade,
    use_volume_filter, vol_threshold,
    use_take_profit, tp_atr_mult
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
    
    max_trades = 2000
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
            
            if c_high > highest:
                highest = c_high
                # Trailing Stop Update
                new_stop = highest - (pos_atr * atr_mult)
                if new_stop > stop_price:
                    stop_price = new_stop
            
            # 1. Stop Loss
            if c_low <= stop_price:
                exit_price = stop_price if stop_price > c_low else c_low
                exit_triggered = True
            # 2. Take Profit
            elif use_take_profit and tp_price > 0 and c_high >= tp_price:
                exit_price = tp_price
                exit_triggered = True
            # 3. Trend Reversal (Optional but common in trend following)
            elif i > 0 and trend_dir[i-1] == -1:
                exit_price = c_price
                exit_triggered = True
                
            if exit_triggered:
                # Sell All
                revenue = coin * exit_price * (1 - fee_rate)
                balance = revenue
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
                
                # SL Calc
                if stop_loss_type == 1: # ATR
                    stop_price = fill_price - (atr[i] * atr_sl_mult)
                else: # Fixed
                    stop_price = fill_price * (1 - stop_loss_pct)
                
                # TP Calc
                if use_take_profit:
                    tp_price = fill_price + (atr[i] * tp_atr_mult)
                else:
                    tp_price = 0.0
                
                # Sizing (For spot, we just use 99% of balance)
                # Note: Risk_per_trade could be used to limit size, but usually spot uses full capital
                # We'll use full capital for simplicity as 'leverage' is effectively 1.0
                cost = balance * 0.99
                coin = (cost * (1 - fee_rate)) / fill_price
                balance -= cost
                
                in_position = True
                entry_idx = i
                highest = fill_price
                pos_atr = atr[i]
        
        equity_curve[i] = balance + (coin * c_price)

    return trades[:trade_count], equity_curve, balance + (coin * close[-1])

def suggest_params(trial, search_space):
    params = {}
    for key, spec in search_space.items():
        if spec['type'] == 'float':
            params[key] = trial.suggest_float(key, spec['low'], spec['high'], step=spec.get('step'))
        elif spec['type'] == 'int':
            params[key] = trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
        elif spec['type'] == 'categorical':
            params[key] = trial.suggest_categorical(key, spec['choices'])
    return params

def calculate_score(ret, mdd, num_trades, win_rate):
    """
    Atomic scoring logic to reduce objective function mass (!Clean).
    """
    if np.isnan(ret) or np.isnan(mdd):
        return -10000

    # Base Score (Profitability Adjusted for MDD)
    abs_mdd = abs(mdd)
    efficiency = ret / (abs_mdd + 5.0) 
    score = (efficiency * 50) + (ret * 0.5)
    
    # Penalties (Ruin Prevention)
    if mdd < -30: score -= 1000
    elif mdd < -20: score -= 200
    elif mdd < -15: score -= 50
    
    # Trade Count (Statistical Significance)
    if num_trades < 15: score -= 2000
    elif num_trades < 30: score -= 500
    elif num_trades >= 100: score += 150
    elif num_trades >= 70: score += 100
    elif num_trades >= 40: score += 50
    
    # Win Rate Buffer
    if win_rate < 35: score -= 300
    elif win_rate >= 55: score += 50
    
    return max(score, -5000)

def objective(trial, symbols_data):
    # 0. Custom Search Space for Spot (Exclude 2h)
    search_space = ULTIMATE_SEARCH_SPACE.copy()
    original_choices = search_space['TIMEFRAME']['choices']
    search_space['TIMEFRAME'] = {
        'type': 'categorical', 
        'choices': [c for c in original_choices if c != '2h']
    }

    params = suggest_params(trial, search_space)
    tf = params['TIMEFRAME']
    symbol_scores = []
    
    for symbol, data_map in symbols_data.items():
        if tf not in data_map: continue
            
        df = data_map[tf].copy()
        strategy = UltimateStrategy(f"Opt_{symbol}", params)
        df = strategy.generate_signals(df)
        
        # Run Backtest
        trades, equity, final_bal = backtest_loop_spot_numba(
            df['close'].values, df['high'].values, df['low'].values, df['entry_upper'].values,
            df['trend_direction'].values, df['strength_filter'].values, 
            df.get('volume_ratio', pd.Series([1.0]*len(df))).fillna(1.0).values, 
            df['atr'].fillna(0.0).values,
            10000000.0, 0.0005, 0.0002,
            (1 if params.get('STOP_LOSS_TYPE') == 'ATR' else 0), 
            params['STOP_LOSS_PCT'], params['ATR_STOP_LOSS_MULT'],
            params['ATR_MULTIPLIER'], params['RISK_PER_TRADE'],
            params['USE_VOLUME_FILTER'], params['VOLUME_THRESHOLD_MULT'],
            params['USE_TAKE_PROFIT'], params['TAKE_PROFIT_ATR_MULT']
        )
        
        ret = (final_bal - 10000000.0) / 10000000.0 * 100
        peak = np.maximum.accumulate(equity)
        
        # Safe MDD calculation to avoid RuntimeWarning: invalid value encountered in divide
        with np.errstate(divide='ignore', invalid='ignore'):
            mdd_series = np.where(peak > 0, (equity - peak) / peak * 100, 0.0)
            mdd = np.min(mdd_series)
            if np.isnan(mdd):
                mdd = 0.0
        num_trades = len(trades)
        win_rate = (len(trades[trades[:, 0] > 0]) / num_trades * 100) if num_trades > 0 else 0
        
        # [!Clean] Use extracted scoring function
        score = calculate_score(ret, mdd, num_trades, win_rate)
        symbol_scores.append(score)
        
        # Set user attrs
        trial.set_user_attr(f"ret_{symbol}", float(ret))
        trial.set_user_attr(f"mdd_{symbol}", float(mdd))
        trial.set_user_attr(f"trades_{symbol}", int(num_trades))
        trial.set_user_attr(f"winrate_{symbol}", float(win_rate))

    if not symbol_scores: return -10000
    
    # 2. Combine scores using HARMONIC MEAN (For Universal Robustness)
    # Allows finding params that work well on BOTH BTC and ETH
    
    offset = 6000  # Shift to make scores positive
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
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--symbols", type=str, default="KRW-BTC,KRW-ETH") # Updated default
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # [UPDATE] Use ALL timeframes like futures optimization
    # [UPDATE] Use ALL timeframes for optimization (Removed '2h' as it is not supported by Upbit)
    timeframes = ['3m', '5m', '15m', '30m', '1h', '4h', '1d']
    # If 2h is excluded in optimize, we can still fetch it or just sync with SEARCH_SPACE
    # But usually more data is fine.
    
    print(f"🚀 Loading FULL HISTORY ({SPOT_START_DATE} ~ Now) for {symbols}...")
    try:
        symbols_data = fetch_all_data_parallel(symbols, timeframes)
        
        study_name = "spot_strategy"
        db_file = f"{study_name}.db"
        storage_name = f"sqlite:///{db_file}"
        
        # [CRITICAL] Delete existing DB for a fresh start
        if os.path.exists(db_file):
            logger.info(f"🗑️ Deleting existing database: {db_file} for a fresh start...")
            try:
                os.remove(db_file)
                # Also remove WAL files if they exist
                for ext in ['-wal', '-shm']:
                    if os.path.exists(db_file + ext):
                        os.remove(db_file + ext)
            except Exception as e:
                logger.error(f"Failed to delete old DB: {e}")
        
        # [CRITICAL UPDATE] DB Locking Fix & Storage Setup
        storage = optuna.storages.RDBStorage(
            url=storage_name,
            engine_kwargs={
                "connect_args": {"timeout": 120},
                "pool_size": 20,
                "max_overflow": 0,
            }
        )
        
        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=False)
        
        print(f"🔥 Starting Universal Strategy Discovery ({args.trials} trials, {args.jobs} jobs)...")
        study.optimize(lambda t: objective(t, symbols_data), n_trials=args.trials, n_jobs=args.jobs)
        
    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("🛑 STOPPED BY USER. Exiting gracefully...")
        print("="*50)
        sys.exit(0)
    
    print("\n" + "="*50)
    print("🏆 BEST UNIVERSAL STRATEGY FOUND")
    print("="*50)
    print(f"Best Score : {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    print("="*50)

