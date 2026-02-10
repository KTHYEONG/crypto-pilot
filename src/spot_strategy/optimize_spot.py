
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
from src.spot_strategy.engine_fast_spot import BacktestEngineFastSpot, backtest_loop_spot_numba
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
    
    # [VALIDATION] Check API credentials
    if not access or not secret:
        print("❌ Error: UPBIT API credentials not found in .env")
        print("   Please set UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY")
        sys.exit(1)
    
    client = UpbitClient(access, secret)
    
    symbols_data = {s: {} for s in symbols}
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

    for symbol in symbols:
        for tf in timeframes:
            filename = f"{symbol.replace('/', '_')}_{tf}_{start_date.replace('-','')}_{end_date.replace('-','')}_spot.csv"
            filepath = os.path.join(DATA_DIR, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.sort_values('datetime', inplace=True) # [CRITICAL] Enforce Order
                    df.reset_index(drop=True, inplace=True)
                    symbols_data[symbol][tf] = df
                except Exception as e:
                    print(f"⚠️  Warning: Failed to load {filepath}: {e}")
                    print(f"   Will attempt to re-download...")
                    # Fall through to download logic
            
            if tf not in symbols_data[symbol] or symbols_data[symbol][tf].empty:
                print(f"  ⬇️ Downloading {symbol}-{tf}...")
                
                try:
                    df = client.fetch_ohlcv(symbol, tf, since=start_ts, end=end_ts)
                    
                    if df is not None and not df.empty:
                        try:
                            df.sort_values('timestamp', inplace=True) # [CRITICAL] Enforce Order
                            df.to_csv(filepath, index=False)
                        except Exception as e:
                            print(f"⚠️  Warning: Failed to save {filepath}: {e}")
                            print(f"   Continuing with in-memory data...")
                        
                        symbols_data[symbol][tf] = df
                    else:
                         print(f"❌ Error: No data downloaded for {symbol}-{tf}")
                         sys.exit(1)
                except Exception as e:
                    print(f"❌ Error: Failed to download {symbol}-{tf}: {e}")
                    print(f"   Please check your API credentials and network connection")
                    sys.exit(1)
                    
    return symbols_data

def compute_merge_indices(data_maps):
    """
    Pre-compute merge index mappings matches optimize_futures.py
    """
    merge_indices = {}
    
    for symbol, data_map in data_maps.items():
        merge_indices[symbol] = {}
        
        if '1d' not in data_map:
            continue
            
        daily_df = data_map['1d']
        # Ensure date_key exists
        if 'date_key' not in daily_df.columns:
            daily_df['date_key'] = pd.to_datetime(daily_df['datetime']).dt.strftime('%Y-%m-%d')
            
        daily_date_keys = daily_df['date_key'].values
        date_to_daily_idx = {date_key: idx for idx, date_key in enumerate(daily_date_keys)}
        
        for tf, tf_df in data_map.items():
            if tf == '1d': continue
            
            if 'date_key' not in tf_df.columns:
                tf_df['date_key'] = pd.to_datetime(tf_df['datetime']).dt.strftime('%Y-%m-%d')
            
            hourly_date_keys = tf_df['date_key'].values
            
            merge_index = np.array([
                date_to_daily_idx.get(date_key, -1) 
                for date_key in hourly_date_keys
            ], dtype=np.int32)
            
            if np.any(merge_index == -1):
                # Fallback to 0 to prevent crash, but warn
                merge_index[merge_index == -1] = 0
                
            merge_indices[symbol][tf] = merge_index
            
    return merge_indices

def objective(trial, symbols_data, search_space, mode="DAY", merge_indices=None):
    import gc
    params = suggest_params(trial, search_space)
    
    full_params = params.copy()
    tf = full_params['TIMEFRAME']
    
    # [VALIDATION] Enforce Logical Constraints
    if full_params.get('TREND_FILTER_TYPE') == 'MACD':
        if full_params.get('MACD_FAST', 12) >= full_params.get('MACD_SLOW', 26):
            return -10000 
            
    symbol_scores = []
    
    # [UNIVERSAL] Loop through all symbols to find Robust Params
    for symbol, data_map in symbols_data.items():
        if tf not in data_map: continue

        # [CAPITAL MGMT] Symbol-Specific Max Capital Usage
        is_major = any(m in symbol for m in ["BTC", "ETH"])
        max_capital = 100_000_000_000.0 if is_major else 20_000_000.0
        full_params['MAX_CAPITAL_USAGE'] = max_capital
        
        # Get data 
        df = data_map[tf]
        daily_df = data_map.get('1d') # Need this now
        
        # Create Strategy
        strategy = UltimateStrategy(f"Opt_{symbol}", full_params)
        
        # Prepare merge index
        current_merge_index = None
        if merge_indices and symbol in merge_indices and tf in merge_indices[symbol]:
            current_merge_index = merge_indices[symbol][tf]

        # Create Engine 
        engine = BacktestEngineFastSpot(
            df, daily_df,  # Pass both DFs
            strategy,
            backtest_loop_spot_numba,  
            initial_balance=1_000_000,
            fee_rate=0.0005,      
            slippage_rate=0.0003,
            merge_index_map=current_merge_index
        )
            
        # Inject risk parameter
        engine.risk_per_trade = full_params.get('RISK_PER_TRADE_SPOT', 0.99)
        
        # Run backtest
        try:
            result = engine.run()
            
            # [DEBUG] Diagnose Score Issue (Print on First Trial Only)
            if trial.number == 0:
                print(f"\n🔍 [DEBUG] Symbol: {symbol}, Timeframe: {tf}")
                print(f"   - Trades: {result['total_trades']}, Return: {result['total_return_pct']:.2f}%, MDD: {result['mdd_pct']:.2f}%")
        except MemoryError as e:
            # [MEMORY] Cleanup on OOM
            del engine, strategy
            gc.collect()
            return -10000
        except Exception as e:
            # [MEMORY] Cleanup on error
            del engine, strategy
            gc.collect()
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
        score = calculate_score(ret, mdd, trades_df, mode=mode, market_type="spot", timeframe=tf)
        
        # [DEBUG] Check Score Value
        if trial.number == 0:
             print(f"   - Calculated Score: {score:.2f}")

        # [SPEED UP] Short-circuit (Relaxed threshold)
        if score < 0:
            del engine, result, strategy, trades_df
            gc.collect()
            return -10000

        symbol_scores.append(score)
        
        # Set user attrs for analysis
        trial.set_user_attr(f"ret_{symbol}", float(ret))
        trial.set_user_attr(f"mdd_{symbol}", float(mdd))
        
        del engine, result, strategy, trades_df

    # [MEMORY] Explicit cleanup per symbol
    if not symbol_scores:
        gc.collect()
        return -10000
    
    # 4. Combine scores using HARMONIC MEAN
    offset = 200  # [ADJUSTED] Reduced from 6000 for Simple Interest scale (matches Futures)
    shifted_scores = [s + offset for s in symbol_scores]
    
    if any(s <= 0 for s in shifted_scores):
        final_score = -10000
    else:
        harmonic_mean = len(shifted_scores) / sum(1 / s for s in shifted_scores)
        final_score = harmonic_mean - offset
    
    # Record Average Attributes (for analysis)
    trial.set_user_attr("score_avg", final_score)
    
    # [MEMORY] Force garbage collection after processing all symbols
    gc.collect()
    
    return final_score





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None, help="Number of optimization trials")
    parser.add_argument("--symbols", type=str, default="KRW-BTC,KRW-ETH") 
    parser.add_argument("--jobs", type=int, default=10) 
    parser.add_argument("--mode", type=str, default="UNIFIED", choices=["SCALP", "DAY", "SWING", "UNIFIED", "ALL"],
                        help="Trading Mode: SCALP, DAY, SWING, or UNIFIED (recommended - auto-selects best timeframe)")
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    mode = args.mode.upper()
    
    # Auto-set trials based on mode if not specified
    # Formula: Parameters × 150 (TPE rule of thumb)
    # All modes have 3 timeframes, adjusted by backtest speed and trade frequency
    # [UPDATED] Trials adjusted based on search space complexity and data volume analysis
    MODE_TRIALS_MAP = {
        'SCALP': 3600,   # 3 timeframes (5m,15m,30m), high data volume but narrow param range
        'DAY': 4200,     # 3 timeframes (30m,1h,4h), balanced - most commonly used mode
        'SWING': 5000,   # 3 timeframes (4h,1d,1w), wide param range + low data volume (overfitting risk)
        'UNIFIED': 8000, # Increased for 55+ parameters and more timeframes (15m, 30m)
        'ALL': 8000      # Alias for UNIFIED
    }
    
    if args.trials is None:
        trials = MODE_TRIALS_MAP.get(mode, 2500)
        print(f"ℹ️  Auto-setting trials for {mode} mode: {trials}")
    else:
        trials = args.trials
        print(f"ℹ️  Using custom trials: {trials}")
    
    # Adjust Logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("SpotOptimizer").setLevel(logging.WARNING)
    
    # Get Search Space & Timeframes
    try:
        search_space = GET_SEARCH_SPACE(mode, market_type='spot')
        timeframes = search_space['TIMEFRAME']['choices']
    except Exception as e:
        print(f"❌ Error: Failed to load search space for mode '{mode}'")
        print(f"   Details: {e}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"🚀 MODE: {mode} SPOT OPTIMIZATION")
    print(f"⏰ Target Timeframes: {timeframes}")
    print(f"{'='*70}\n")
    
    SPOT_END_DATE = "2026-01-16"
    
    # [FIX] Always include '1d' for MTF (Multi-Timeframe) Logic
    # BacktestEngineFastSpot requires '1d' data for daily trend filtering.
    load_tfs = list(set(timeframes + ['1d']))
    
    symbols_data = load_all_timeframes(symbols, SPOT_START_DATE, SPOT_END_DATE, load_tfs)
    
    # [WARMUP OPTIMIZATION]
    WARMUP_BUFFER_BARS = {
        '1h': 300, '4h': 200, '1d': 150,
    }
    
    print(f"✂️  Trimming Data for Optimization (Train Period: ~ {TRAIN_CUTOFF_DATE})")
    train_data = {}
    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)
    
    for sym, tf_map in symbols_data.items():
        train_data[sym] = {}
        for tf, df_ in tf_map.items():
            cutoff_mask = df_['datetime'] < cutoff_ts
            train_end_idx = cutoff_mask.sum()
            if train_end_idx == 0: continue
            
            desired_warmup = WARMUP_BUFFER_BARS.get(tf, 200)
            sliced_df = df_.iloc[:train_end_idx].copy()
            sliced_df.attrs['warmup_bars'] = min(desired_warmup, train_end_idx)
            train_data[sym][tf] = sliced_df
    
    # [OPTIMIZATION] Pre-compute merge indices to eliminate pd.merge overhead
    print(f"🔗 Pre-computing merge indices for fast data alignment...")
    merge_indices = compute_merge_indices(train_data)
    print(f"✅ Merge indices computed for {len(merge_indices)} symbols")

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
    safe_pass = quote_plus(db_pass)
    storage_url = f"mysql+pymysql://{db_user}:{safe_pass}@{db_host}:{db_port}/{db_name}"
    
    print(f"🔄 Preparing study: {study_name}")
    
    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print(f"🗑️  Deleted old study: {study_name}")
    except Exception:
        pass

    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={
            "pool_size": max(30, args.jobs * 2),
            "max_overflow": 10,
            "pool_recycle": 3600,
            "pool_pre_ping": True,
        }
    )
    
    
    # [Performance] Use ConstantLiar for parallel efficiency
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=2000,  # 25% of 8000 trials for thorough random exploration (matches Futures)
        multivariate=True,  # Consider param dependencies
        constant_liar=True,  # Avoid duplicate proposals
        warn_independent_sampling=False,
    )
    
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize", sampler=sampler
    )
    
    print(f"\n{'='*70}")
    print(f"🔥 STARTING OPTIMIZATION for {study_name}")
    print(f"🛢️  Storage: MySQL ({db_host}/{db_name})")
    print(f"📈 Total Trials: {trials}")
    print(f"💻 Parallel Jobs: {args.jobs}")
    print(f"{'='*70}\n")
    
    # Numba Warmup
    print("🔥 Warming up Numba JIT...", end="", flush=True)
    try:
        dummy_len = 10
        _dummy_arr = np.ones(dummy_len, dtype=np.float64)
        _dummy_int = np.zeros(dummy_len, dtype=np.int64)
        backtest_loop_spot_numba(
            _dummy_arr, _dummy_arr, _dummy_arr, _dummy_arr, 
            _dummy_arr, # entry_upper
            _dummy_int, _dummy_int, _dummy_arr, # trend, strength, vol
            _dummy_arr, _dummy_arr, # atr, sar
            _dummy_arr, _dummy_arr, _dummy_arr, # regime: hurst, natr, rsi
            10000.0, 0.001, 0.001, 
            0, 0, 0.01, 1.5, 3.0, 0.99, # exit, sl, risk
            False, 1.0, False, 3.0, # vol, tp
            1000, 0.0, # hold, trailing
            0.6, 4.5, 94.0, 1.3, 0.15, # regime params (strong/panic)
            0.45, 0.6, # [NEW] Weak regime
            90, 0.1, 0, # entry filters, warmup
            False, 1000000.0 # [NEW] compounding
        )
        print(" Done!")
    except Exception as e:
        print(f"\n⚠️  Numba warmup failed: {e}")

    try:
        study.optimize(lambda t: objective(t, train_data, search_space, mode, merge_indices), 
                       n_trials=trials,
                       n_jobs=args.jobs,
                       show_progress_bar=True)
                       
    except KeyboardInterrupt:
        print("\n🛑 Optimization Interrupted by User")
        print(f"💾 Progress saved: {len(study.trials)} trials completed")
    except Exception as e:
        print(f"\n❌ Optimization failed with error: {e}")
        print(f"💾 Progress saved: {len(study.trials)} trials completed before failure")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"✅ {mode} Optimization Complete!")
    
    if len(study.trials) > 0:
        print(f"🏆 Best Score: {study.best_value:.2f}")
        print(f"✨ Best Params: {study.best_params}")
        
        # [NEW] Detailed Report for TRAIN Period
        print(f"\n{'='*70}")
        print(f"📊 TRAIN PERIOD PERFORMANCE (Best Strategy)")
        print(f"{'='*70}")
        
        best_params = study.best_params
        tf = best_params.get('TIMEFRAME', '1d')
        
        for symbol, data_map in train_data.items():
            if tf not in data_map: continue
            
            df = data_map[tf]
            daily_df = data_map.get('1d') # Get Daily Data
            
            # Re-create Strategy & Engine
            strategy = UltimateStrategy(f"Best_{symbol}", best_params)
            
            # Pass daily_df and check if merge_indices available
            engine = BacktestEngineFastSpot(
                df, daily_df, strategy, backtest_loop_spot_numba,
                initial_balance=1_000_000,
                fee_rate=0.0005,
                slippage_rate=0.0003
            )
            
            # Inject merge index manually for best result view if needed
            if symbol in merge_indices and tf in merge_indices[symbol]:
                engine._merge_index_map = merge_indices[symbol][tf]
            
            engine.risk_per_trade = best_params.get('RISK_PER_TRADE_SPOT', 0.99)
            
            try:
                res = engine.run()
                
                ret = res['total_return_pct']
                mdd = res['mdd_pct']
                cnt = res['total_trades']
                win = res['win_rate']
                
                trades_df = res['trades_df']
                pf = 0.0
                if not trades_df.empty and 'pnl_pct' in trades_df.columns:
                    gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
                    gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
                    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
                
                print(f"   - {symbol:<9} : Return {ret:>7.2f}% | MDD {mdd:>6.2f}% | Trades {cnt:>3} | Win {win:>5.1f}% | PF {pf:.2f}")
            
            except Exception as e:
                print(f"   - {symbol:<9} : Error calculating performance: {e}")
                
    print(f"{'='*70}")
    
    print("\n✅ Optimization Complete.")
