import argparse
import pandas as pd
import os
import sys
import optuna
import logging
import sqlite3
import numpy as np
from pathlib import Path

# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    # Fallback if specific path resolution fails (unlikely in correct structure)
    sys.path.append(os.getcwd())

from config.settings import DATA_DIR, BACKTEST_START_DATE, BACKTEST_END_DATE, TRAIN_CUTOFF_DATE
from config.optimization_config_ultimate import ULTIMATE_SEARCH_SPACE, COMMON_SEARCH_SPACE
from src.data.collector import DataCollector
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.futures_strategy.engine_fast_futures import BacktestEngineFast  # Using Numba-accelerated engine

def load_all_timeframes(symbol, start_date, end_date, timeframes):
    """Load all necessary timeframe data into memory"""
    data_map = {}
    collector = DataCollector()
    
    # Daily Data (Required for Indicators)
    daily_file = DATA_DIR / f"{symbol.replace('/', '_')}_1d_{start_date}_{end_date}.csv"
    if not daily_file.exists():
         collector.collect_and_save(symbol, '1d', start_date, end_date)
    
    df = pd.read_csv(daily_file)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    data_map['1d'] = df
    
    for tf in timeframes:
        tf_file = DATA_DIR / f"{symbol.replace('/', '_')}_{tf}_{start_date}_{end_date}.csv"
        if not tf_file.exists():
            print(f"Downloading {tf} data...")
            collector.collect_and_save(symbol, tf, start_date, end_date)
            
        df = pd.read_csv(tf_file)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        data_map[tf] = df
        
    return data_map

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

def calculate_score(ret, mdd, num_trades, win_rate, pf):
    """
    Advanced Professional Scoring (Aligned with Spot):
    Focuses on Robustness, Safety (Exp. MDD), and Quality (Profit Factor).
    """
    if np.isnan(ret) or np.isnan(mdd):
        return -10000

    # 1. Statistical Floor
    # Futures strategies often trade more frequently, so we can expect reasonable sample size.
    if num_trades < 100:
        return -5000 + num_trades 
    
    # 2. Risk Adjustment: Exponential MDD Penalty
    # -15% is the pivot point for 'pain'. Beyond that, we penalize exponentially.
    abs_mdd = abs(mdd)
    mdd_penalty = 0
    if abs_mdd > 15:
        # Penalize hard for deeper drawdowns
        mdd_penalty = ((abs_mdd - 15) ** 2.2) * 5.0
    
    # 3. Efficiency & Quality (Profit Factor & Ret)
    pf_bonus = 0
    if pf > 1.1:
        pf_bonus = (pf - 1.0) * 1500 
    
    # 4. Base Equity Performance
    # Return adjusted by MDD
    efficiency = ret / (abs_mdd + 5.0) * 50
    
    score = (ret * 0.2) + efficiency + pf_bonus - mdd_penalty
    
    # 5. Over-trading Ceiling
    if num_trades > 2000:
        score -= (num_trades - 2000) * 2.0
        
    return max(score, -10000)

def objective(trial, strategy_cls, strategy_name, data_maps, base_search_space, common_search_space):
    """
    Multi-symbol objective function.
    Returns harmonic mean of scores across all symbols to ensure universal performance.
    """
    # 1. Generate Params
    strategy_params = suggest_params(trial, base_search_space)
    common_params = suggest_params(trial, common_search_space)
    full_params = {**strategy_params, **common_params}
    
    # [VALIDATION] Enforce Logical Constraints
    if full_params.get('TREND_FILTER_TYPE') == 'MACD':
        if full_params.get('MACD_FAST', 12) >= full_params.get('MACD_SLOW', 26):
            return -10000 # Invalid trial penalty
    
    # 2. Select Timeframe
    selected_tf = full_params.get('TIMEFRAME', '1h')
    
    # 3. Run backtest for EACH symbol
    symbol_scores = []
    symbol_results = {}
    
    for symbol, data_map in data_maps.items():
        # Ensure timeframe exists
        if selected_tf not in data_map:
            # Fallback (though data should be loaded)
            return -10000
            
        hourly_df = data_map[selected_tf].copy() 
        daily_df = data_map['1d'].copy()
        
        # Create Strategy
        strategy = strategy_cls(f"{strategy_name}_{symbol}", full_params)
        
        # Engine Execution
        # 1M KRW approx 750 USDT
        engine = BacktestEngineFast(hourly_df, daily_df, strategy, initial_balance=750)
        
        # Inject Leverage/Risk
        engine.leverage = full_params.get('LEVERAGE', 1)
        engine.risk_per_trade = full_params.get('RISK_PER_TRADE', 0.02)
        
        try:
            result = engine.run()
        except Exception as e:
            # print(f"Trial failed for {symbol}: {e}")
            return -10000  # Severe penalty if ANY symbol fails
        
        # Extract Metrics
        ret = result['total_return_pct']
        mdd = result['mdd_pct']
        trades = result['total_trades']
        win_rate = result['win_rate']
        
        # Calculate Profit Factor (PF)
        trades_df = result['trades_df']
        pf = 0.0
        if not trades_df.empty and 'pnl' in trades_df.columns:
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        
        # Store results
        symbol_results[symbol] = {
            'return': ret,
            'mdd': mdd,
            'trades': trades,
            'win_rate': win_rate,
            'pf': pf
        }
        
        # Calculate Score
        score = calculate_score(ret, mdd, trades, win_rate, pf)
        symbol_scores.append(score)
        
        # Set individual attrs
        trial.set_user_attr(f"ret_{symbol.replace('/', '_')}", float(ret))
        trial.set_user_attr(f"mdd_{symbol.replace('/', '_')}", float(mdd))
        trial.set_user_attr(f"pf_{symbol.replace('/', '_')}", float(pf))
    
    # 4. Combine scores using HARMONIC MEAN
    offset = 6000
    shifted_scores = [s + offset for s in symbol_scores]
    
    if any(s <= 0 for s in shifted_scores):
        final_score = -10000
    else:
        harmonic_mean = len(shifted_scores) / sum(1/s for s in shifted_scores)
        final_score = harmonic_mean - offset
    
    # Record Average Attributes
    avg_ret = np.mean([r['return'] for r in symbol_results.values()])
    avg_mdd = np.mean([r['mdd'] for r in symbol_results.values()])
    avg_pf = np.mean([r['pf'] for r in symbol_results.values()])
    
    trial.set_user_attr("return_avg", avg_ret)
    trial.set_user_attr("mdd_avg", avg_mdd)
    trial.set_user_attr("pf_avg", avg_pf)
    
    return final_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Adjust Logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger('src.backtest.engine').setLevel(logging.WARNING)
    
    # Load Data for ALL symbols
    # Exclude ultra-short timeframes (3m, 5m) and daily (1d) to reduce noise and improve performance
    timeframes = ['15m', '30m', '1h', '2h', '4h']
    print(f"Loading data for timeframes: {timeframes}")
    
    data_maps = {}
    
    print(f"\n{'='*70}")
    print(f"📡 Loading data for symbols: {', '.join(symbols)}")
    print(f"{'='*70}\n")
    
    for symbol in symbols:
        print(f"Loading {symbol}...")
        data_maps[symbol] = load_all_timeframes(symbol, BACKTEST_START_DATE, BACKTEST_END_DATE, timeframes)
    
    # [CRITICAL] Slice Data for Optimization (Train Set)
    print(f"✂️  Trimming Data for Optimization (Train Period: ~ {TRAIN_CUTOFF_DATE})")
    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)
    
    for sym in data_maps:
        for tf in data_maps[sym]:
            original_len = len(data_maps[sym][tf])
            data_maps[sym][tf] = data_maps[sym][tf][data_maps[sym][tf]['datetime'] < cutoff_ts].copy()
            new_len = len(data_maps[sym][tf])
            if tf == '1h': # Log once per symbol usually
                print(f"  [{sym}] Train Size: {new_len} (Original: {original_len})")
    
    # [UPDATE] Standardized Study Name
    study_name = "futures_strategy"
    db_path = f"{study_name}.db"
    storage_name = f"sqlite:///{db_path}"
    
    # [CRITICAL] Delete existing DB for a fresh start
    if os.path.exists(db_path):
        print(f"🗑️ Deleting existing database: {db_path} for a fresh start...")
        try:
            os.remove(db_path)
            for ext in ['-wal', '-shm']:
                if os.path.exists(db_path + ext):
                    os.remove(db_path + ext)
        except Exception as e:
            print(f"⚠️ Warning: Could not delete old DB: {e}")

    # [CRITICAL UPDATE] DB Locking Fix & Storage Setup
    # Removed redundant manual sqlite3 connection which can cause file locking issues on Windows
    storage = optuna.storages.RDBStorage(
        url=storage_name,
        engine_kwargs={
            "connect_args": {"timeout": 120},
            "pool_size": 20,
            "max_overflow": 0,
        }
    )
    
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=False)
    
    print(f"\n{'='*70}")
    print(f"🚀 STRATEGY DISCOVERY STARTING for {study_name}")
    print(f"📊 Target Symbols: {', '.join(symbols)}")
    print(f"💻 Parallel Jobs: {args.jobs}")
    print(f"📈 Total Trials: {args.trials}")
    print(f"{'='*70}\n")
    
    try:
        study.optimize(
            lambda trial: objective(trial, UltimateStrategy, "Ultimate_Universal", data_maps, ULTIMATE_SEARCH_SPACE, COMMON_SEARCH_SPACE),
            n_trials=args.trials,
            n_jobs=args.jobs,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Optimization Interrupted by User")
    
    
    print(f"\n{'='*70}")
    print("✅ Universal Optimization Complete!")
    print(f"📊 Results saved to DB: {db_path}")
    if len(study.trials) > 0:
        print(f"🏆 Best Score: {study.best_value:.2f}")
        print(f"✨ Best Params: {study.best_params}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
