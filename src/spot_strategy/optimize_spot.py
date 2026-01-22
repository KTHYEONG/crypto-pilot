
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
from src.optimization.opt_utils import suggest_params, calculate_score

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
    max_holding_bars, trailing_activation_atr,  # [NEW]
    warmup_bars  # [WARMUP] Skip trading during this period
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
        # [WARMUP] Skip trading during warmup period
        if i < warmup_bars:
            equity_curve[i] = balance
            continue
            
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




def objective(trial, symbols_data, search_space, mode="DAY"):
    import gc
    params = suggest_params(trial, search_space)
    
    # [FUTURE-PROOF] Merge common search space (currently empty, but prepared for future use)
    # This ensures consistency with Futures optimization structure
    common_params = {}  # Reserved for future common parameters
    full_params = {**params, **common_params}
    
    # [VALIDATION] Enforce Logical Constraints
    if full_params.get('TREND_FILTER_TYPE') == 'MACD':
        if full_params.get('MACD_FAST', 12) >= full_params.get('MACD_SLOW', 26):
            return -10000 # Invalid trial penalty
            
    tf = full_params['TIMEFRAME']
    symbol_scores = []
    
    # [UNIVERSAL] Loop through all symbols to find Robust Params
    for symbol, data_map in symbols_data.items():
        if tf not in data_map: continue
        
        # Get data (no need to copy - Engine doesn't modify it)
        df = data_map[tf]
        
        # Create Strategy
        strategy = UltimateStrategy(f"Opt_{symbol}", full_params)
        
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
        engine.risk_per_trade = full_params.get('RISK_PER_TRADE_SPOT', 0.99)
        
        # Run backtest
        try:
            result = engine.run()
        except MemoryError as e:
            gc.collect()
            return -10000
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
        
        # [MEMORY] Explicit cleanup per symbol
        del engine, result, strategy

    if not symbol_scores:
        gc.collect()
        return -10000
    
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
    parser.add_argument("--jobs", type=int, default=10) 
    parser.add_argument("--mode", type=str, default="DAY", choices=["SCALP", "DAY", "SWING", "ALL"])
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    mode = args.mode.upper()
    
    # [UPDATED] Trials adjusted based on search space complexity and data volume analysis
    MODE_TRIALS_MAP = {
        'SCALP': 3600,  # High data volume but narrow param range
        'DAY': 4200,    # Balanced - most commonly used mode
        'SWING': 5000,  # Wide param range + low data volume (overfitting risk)
        'ALL': 4500     # Catch-all (highest complexity)
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
    
    # [WARMUP OPTIMIZATION] Include buffer period for indicator calculation
    WARMUP_BUFFER_BARS = {
        '1m': 600,   # ~10 hours
        '3m': 500,   # ~25 hours
        '5m': 500,   # ~42 hours
        '15m': 400,  # ~100 hours
        '30m': 350,  # ~220 hours
        '1h': 300,   # ~12.5 days
        '4h': 200,   # ~33 days
        '1d': 150,   # ~5 months
    }
    
    for sym, tf_map in symbols_data.items():
        train_data[sym] = {}
        for tf, df_ in tf_map.items():
            # Find the end of training period (data before TRAIN_CUTOFF_DATE)
            cutoff_mask = df_['datetime'] < cutoff_ts
            train_end_idx = cutoff_mask.sum()
            
            if train_end_idx == 0:
                print(f"⚠️  Warning: {sym}-{tf} has no data before cutoff date. Skipping.")
                continue
            
            # Get desired warmup period for this timeframe
            desired_warmup = WARMUP_BUFFER_BARS.get(tf, 200)
            
            # Slice all data from start to cutoff (entire training period)
            sliced_df = df_.iloc[:train_end_idx].copy()
            
            # Set warmup_bars: first N bars are for indicator warmup only
            # Trading will start after these warmup bars
            sliced_df.attrs['warmup_bars'] = min(desired_warmup, train_end_idx)
            
            train_data[sym][tf] = sliced_df

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
            _dummy_arr, _dummy_arr, _dummy_arr, # OHLC (close, high, low)
            _dummy_arr, # Entry Upper
            _dummy_int, _dummy_int, _dummy_arr, _dummy_arr, _dummy_arr, # Trend, Strength, Vol, ATR, SAR
            10000.0, 0.001, 0.001, # Bal, Fee, Slip
            0, # Exit type (New: 0=Trailing, 1=SAR)
            0, 0.01, 1.5, # SL Type, Pct, Mult
            3.0, # ATR Mult
            0.99, # Risk
            False, 1.0, # Vol Filter
            False, 3.0, # TP
            1000, 0.0, # Max Hold, Trailing Act
            0 # [WARMUP] Missing argument fixed
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
