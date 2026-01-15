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
sys.path.append(os.path.join(os.path.dirname(__file__)))

from config.settings import DATA_DIR, BACKTEST_START_DATE, BACKTEST_END_DATE
from config.optimization_config_ultimate import ULTIMATE_SEARCH_SPACE, COMMON_SEARCH_SPACE
from src.data.collector import DataCollector
from src.strategy.strategies import UltimateStrategy
from src.backtest.engine_fast import BacktestEngineFast  # Using Numba-accelerated engine

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

def objective(trial, strategy_cls, strategy_name, data_maps, base_search_space, common_search_space):
    """
    Multi-symbol objective function.
    data_maps: dict of {symbol: data_map} where each data_map contains timeframe data
    Returns harmonic mean of scores across all symbols to ensure universal performance
    """
    # 1. Generate Params
    strategy_params = suggest_params(trial, base_search_space)
    common_params = suggest_params(trial, common_search_space)
    full_params = {**strategy_params, **common_params}
    
    # 2. Select Timeframe
    selected_tf = full_params.get('TIMEFRAME', '1h')
    
    # 3. Run backtest for EACH symbol
    symbol_scores = []
    symbol_results = {}
    
    for symbol, data_map in data_maps.items():
        # Ensure timeframe exists
        if selected_tf not in data_map:
            selected_tf = '1h'
            
        hourly_df = data_map[selected_tf].copy() 
        daily_df = data_map['1d'].copy()
        
        # Create Strategy
        strategy = strategy_cls(f"{strategy_name}_{symbol}", full_params)
        
        # Engine Execution
        engine = BacktestEngineFast(hourly_df, daily_df, strategy, initial_balance=1_000_000)
        
        # Inject Leverage/Risk
        engine.leverage = full_params.get('LEVERAGE', 1)
        engine.risk_per_trade = full_params.get('RISK_PER_TRADE', 0.02)
        
        try:
            result = engine.run()
        except Exception as e:
            print(f"Trial failed for {symbol}: {e}")
            return -10000  # Severe penalty if ANY symbol fails
        
        # Calculate individual score for this symbol
        ret = result['total_return_pct']
        mdd = result['mdd_pct']
        trades = result['total_trades']
        win_rate = result['win_rate']
        
        # Store results
        symbol_results[symbol] = {
            'return': ret,
            'mdd': mdd,
            'trades': trades,
            'win_rate': win_rate
        }
        
        # Calculate score for this symbol
        abs_mdd = abs(mdd)
        efficiency = ret / (abs_mdd + 5.0) 
        score = (efficiency * 50) + (ret * 0.5)
        
        # Penalties & Bonuses
        # 1. Ruin Prevention (MDD)
        if mdd < -30: 
            score -= 1000
        elif mdd < -20:
            score -= 200
            
        # 2. Trade Count (Statistical Significance)
        if trades < 30: 
            score -= 2000
        elif trades < 50:
            score -= 500
        elif trades >= 120:
            score += 150
        elif trades >= 80:
            score += 100
        elif trades >= 50:
            score += 50
            
        # 3. Win Rate Buffer
        if win_rate < 35:
            score -= 300
        elif win_rate >= 55:
            score += 50
        
        symbol_scores.append(max(score, -5000))  # Floor at -5000 to prevent extreme outliers
    
    # 4. Combine scores using HARMONIC MEAN
    # Harmonic mean heavily penalizes if one symbol performs poorly
    # Formula: n / (1/x1 + 1/x2 + ... + 1/xn)
    # To handle negative scores, we shift scores up by adding offset
    
    offset = 6000  # Shift to make all scores positive
    shifted_scores = [s + offset for s in symbol_scores]
    
    if any(s <= 0 for s in shifted_scores):
        # If any score is too negative even after offset, return severe penalty
        final_score = -10000
    else:
        # Harmonic mean
        harmonic_mean = len(shifted_scores) / sum(1/s for s in shifted_scores)
        final_score = harmonic_mean - offset  # Shift back
    
    # Record Attributes for Analysis (Average across symbols)
    avg_ret = np.mean([r['return'] for r in symbol_results.values()])
    avg_mdd = np.mean([r['mdd'] for r in symbol_results.values()])
    avg_trades = np.mean([r['trades'] for r in symbol_results.values()])
    avg_win_rate = np.mean([r['win_rate'] for r in symbol_results.values()])
    
    trial.set_user_attr("return_avg", avg_ret)
    trial.set_user_attr("mdd_avg", avg_mdd)
    trial.set_user_attr("trades_avg", avg_trades)
    trial.set_user_attr("win_rate_avg", avg_win_rate)
    trial.set_user_attr("timeframe", selected_tf)
    trial.set_user_attr("leverage", full_params.get('LEVERAGE', 1))
    trial.set_user_attr("entry_type", full_params.get('ENTRY_TYPE'))
    trial.set_user_attr("trend_filter", full_params.get('TREND_FILTER_TYPE'))
    
    # Also store individual symbol results
    for symbol, res in symbol_results.items():
        trial.set_user_attr(f"return_{symbol.replace('/', '_')}", res['return'])
        trial.set_user_attr(f"mdd_{symbol.replace('/', '_')}", res['mdd'])
        trial.set_user_attr(f"trades_{symbol.replace('/', '_')}", res['trades'])
        trial.set_user_attr(f"winrate_{symbol.replace('/', '_')}", res['win_rate'])
    
    return final_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000) # Default updated to 1000
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT", help="Comma-separated symbols for multi-symbol optimization")
    parser.add_argument("--jobs", type=int, default=6, help="Parallel jobs (Safe: 6)")
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Adjust Logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger('src.backtest.engine').setLevel(logging.WARNING)
    
    # Load Data for ALL symbols
    timeframes = ['3m', '5m', '15m', '30m', '1h', '2h', '4h']
    print(f"Loading data for timeframes: {timeframes}")
    
    data_maps = {}
    
    print(f"\n{'='*70}")
    print(f"📡 Loading data for symbols: {', '.join(symbols)}")
    print(f"{'='*70}\n")
    
    for symbol in symbols:
        print(f"Loading {symbol}...")
        data_maps[symbol] = load_all_timeframes(symbol, BACKTEST_START_DATE, BACKTEST_END_DATE, timeframes)
    
    study_name = "optimize_Ultimate_Universal"
    storage_name = f"sqlite:///{study_name}.db"
    
    # [CRITICAL UPDATE] DB Locking Fix
    storage = optuna.storages.RDBStorage(
        url=storage_name,
        engine_kwargs={
            "connect_args": {"timeout": 120},
            "pool_size": 20,
            "max_overflow": 0,
        }
    )
    
    # SQLite WAL Mode Setup (Manual)
    db_path = f"{study_name}.db"
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=120000;")
        print("✅ SQLite WAL mode & Tuning applied.")
    except Exception as e:
        print(f"⚠️ Warning: Could not tune SQLite: {e}")
    
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 UNIVERSAL STRATEGY DISCOVERY STARTING")
    print(f"🌌 Multi-Symbol Optimization: {', '.join(symbols)}")
    print(f"📊 Target: Find strategy that works across ALL symbols")
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
    
    # Save Results
    df = study.trials_dataframe()
    df.to_csv("optimization_results_Ultimate_Universal.csv", index=False)
    
    print(f"\n{'='*70}")
    print("✅ Universal Optimization Complete!")
    print(f"📊 Results saved to: optimization_results_Ultimate_Universal.csv")
    if len(study.trials) > 0:
        print(f"🏆 Best Score: {study.best_value:.2f}")
        print(f"✨ Best Params: {study.best_params}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
