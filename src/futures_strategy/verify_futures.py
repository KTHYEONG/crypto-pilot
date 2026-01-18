
import argparse
import pandas as pd
import sys
import os
import logging
import json
from pathlib import Path
import numpy as np

# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())

from config.settings import DATA_DIR, BACKTEST_START_DATE, BACKTEST_END_DATE, TRAIN_CUTOFF_DATE
from src.data.collector import DataCollector
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.futures_strategy.backtest_utils_futures import run_backtest_segment_futures, prepare_futures_data

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FuturesVerifier")

def load_data(symbol, start_date, end_date, timeframe):
    """Load Data Helper"""
    collector = DataCollector()
    
    # Daily Data
    daily_file = DATA_DIR / f"{symbol.replace('/', '_')}_1d_{start_date}_{end_date}.csv"
    if not daily_file.exists():
         logger.info("Downloading Daily data...")
         collector.collect_and_save(symbol, '1d', start_date, end_date)
    daily_df = pd.read_csv(daily_file)
    daily_df['datetime'] = pd.to_datetime(daily_df['timestamp'], unit='ms')

    # Timeframe Data
    tf_file = DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}_{start_date}_{end_date}.csv"
    if not tf_file.exists():
        logger.info(f"Downloading {timeframe} data...")
        collector.collect_and_save(symbol, timeframe, start_date, end_date)
    hourly_df = pd.read_csv(tf_file)
    hourly_df['datetime'] = pd.to_datetime(hourly_df['timestamp'], unit='ms')
    
    # Merge for Strategy Signals
    # (Strategy typically expects specific structure or merges internally. 
    # Here we simulate what EngineFast does but for Python loop we might need pre-merged if strategy relies on it.
    # Actually UltimateStrategy generates signals on 'df'. We usually pass hourly_df and let it access daily if needed.
    # But UltimateStrategy in this repo seems to handle daily merging or requires daily_df passed?
    # Let's check Optimize: it passes hourly and daily to Engine. Engine merges them.
    # So we need to merge here before passing to 'run_backtest_segment_futures' if that function expects merged cols.
    # run_backtest_segment_futures expects 'entry_upper' etc in the df.
    
    return hourly_df, daily_df


def detailed_backtest_futures(hourly_df, daily_df, params):
    """
    Detailed Backtest for Futures (Long/Short)
    Matches Optimize logic but with logging.
    """
    logger.info(f"--- Starting Detailed Backtest (Futures) ---")
    
    strategy = UltimateStrategy("Verify", params)
    
    # Prepare Data (Merge signals)
    df = prepare_futures_data(hourly_df, daily_df, strategy)
    
    initial_balance = 750.0
    
    # Run Shared Backtest
    ret_pct, mdd, trades_log, detailed_log, equity_curve = run_backtest_segment_futures(
        df, params, initial_balance=initial_balance, return_series=True
    )
    
    final_val = initial_balance * (1 + ret_pct/100)
    
    print("\n" + "="*50)
    print("BACKTEST RESULT")
    print("="*50)
    print(f"Final Balance: {final_val:,.0f} (Initial: {initial_balance:,.0f})")
    print(f"Total Return : {ret_pct:.2f}%")
    print(f"Trade Count  : {len(trades_log)}")
    
    if trades_log:
        # trades_log is list of ROI percentages per trade
        # Win Rate
        wins = [t for t in trades_log if t > 0]
        losses = [t for t in trades_log if t <= 0]
        win_rate = len(wins)/len(trades_log)*100
        
        # Profit Factor (Sum of +ROI / Sum of | -ROI |) 
        # Note: This is ROI based PF, approximate but useful.
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        
        print(f"Win Rate      : {win_rate:.2f}%")
        print(f"Profit Factor : {pf:.2f}")
        print(f"Max drawdown  : {mdd:.2f}%")
    else:
        print("No Trades Executed.")
        
    print("="*50)
    
    # --- 2. Walk Forward Analysis (Robustness) ---
    from src.futures_strategy.walk_forward_futures import FuturesWalkForwardAnalyzer
    print(f"\n🚀 Running Walk-Forward Analysis (5 Splits)...")
    
    # WFA usually needs raw data and does its own sliding window + engine run
    # We pass the raw DFs + params
    wfa = FuturesWalkForwardAnalyzer(hourly_df, daily_df, params)
    wfa_results = wfa.run(n_splits=5)
    
    print(f"{'='*50}")
    print(f"WALK FORWARD ANALYSIS RESULT")
    print(f"{'='*50}")
    if wfa_results.empty:
        print("⚠️ Not enough data to run Walk-Forward Analysis.")
    else:
        print(wfa_results.to_markdown(index=False, floatfmt=".2f"))
        
        avg_wfa_ret = wfa_results['Return'].mean()
        print(f"\nAverage Return per Split: {avg_wfa_ret:.2f}%")
        consistency = len(wfa_results[wfa_results['Return'] > 0]) / len(wfa_results) * 100
        print(f"Consistency (Positive Segments): {consistency:.0f}%")

    # --- 3. Monte Carlo Simulation (Probability) ---
    from src.futures_strategy.monte_carlo_futures import FuturesMonteCarloSimulator
    print(f"\n🎲 Running Monte Carlo Simulation (10,000 runs)...")
    
    if trades_log:
        # MC uses list of returns %
        mc = FuturesMonteCarloSimulator(trades_log)
        mc_res = mc.run(n_simulations=10000, initial_balance=initial_balance)
        
        print(f"{'='*50}")
        print(f"MONTE CARLO SIMULATION RESULT (95% Confidence)")
        print(f"{'='*50}")
        print(f"Probability of Profit : {mc_res['prob_profit']:.2f}%")
        print(f"Expected Return       : {mc_res['mean_return_pct']:.2f}% (Median: {mc_res['median_return_pct']:.2f}%)")
        print(f"Worst Case MDD (5%)   : {mc_res['worst_case_mdd']:.2f}%")
        print(f"Return Range (95%)    : {mc_res['lower_bound_95']:.2f}% ~ {mc_res['upper_bound_95']:.2f}%")
        print("="*50)
    else:
        print("Not enough trades for Monte Carlo.")
    
    return
    
if __name__ == "__main__":
    import argparse
    import optuna
    
    parser = argparse.ArgumentParser()
    # Matches the default DB name and symbols in optimize_futures.py
    parser.add_argument("--db", type=str, default="futures_strategy")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # 1. Load Universal Best Params
    db_name = args.db
    db_path = f"{db_name}.db"
    storage = f"sqlite:///{db_path}"
    study_name = "futures_strategy"
    
    best_params = {}
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        best_params = study.best_params
        print("\n" + "="*70)
        print(f"🚀 VERIFYING UNIVERSAL STRATEGY (Score: {study.best_value:.4f})")
        print(f"Params: {best_params}")
        print("="*70)
    except Exception as e:
        print(f"❌ Failed to load Universal DB ({db_path}): {e}")
        print("Please run optimize_futures.py first.")
        sys.exit(1)
        
    for symbol in symbols:
        print(f"\n👉 Analyzing {symbol}...")

        # Load from History using optimized timeframe
        tf = best_params.get('TIMEFRAME', '1h')
        
        try:
            hourly_df, daily_df = load_data(symbol, BACKTEST_START_DATE, BACKTEST_END_DATE, tf)
            
            # Apply Split (Verification must be on Out-of-Sample data)
            cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)
            
            test_hourly = hourly_df[hourly_df['datetime'] >= cutoff_ts].copy()
            test_daily = daily_df[daily_df['datetime'] >= cutoff_ts].copy()
            
            if not test_hourly.empty:
                print(f"\n🔵 >>> Running Verification on TEST DATA (OOS: {TRAIN_CUTOFF_DATE} ~ Now) <<<")
                detailed_backtest_futures(test_hourly, test_daily, best_params)
            else:
                print(f"⚠️ No Test Data found after {TRAIN_CUTOFF_DATE}! Check settings.py.")
                print("Running on Full Data as fallback...")
                detailed_backtest_futures(hourly_df, daily_df, best_params)
                
        except Exception as e:
            print(f"❌ Error verifying {symbol}: {e}")
            import traceback
            traceback.print_exc()
