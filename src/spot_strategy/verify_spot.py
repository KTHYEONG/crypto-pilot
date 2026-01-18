import optuna
import pandas as pd
import numpy as np
import sys
import os
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.strategy.strategies import UltimateStrategy

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpotVerifier")

from src.spot_strategy.backtest_utils import run_backtest_segment

def detailed_backtest_spot(df, params):
    """
    Detailed Backtest for Spot (Long-Only)
    Matches Optimize logic but with logging.
    """
    logger.info(f"--- Starting Detailed Backtest (Spot) ---")
    
    strategy = UltimateStrategy("Verify", params)
    df = strategy.generate_signals(df.copy())
    
    initial_balance = 10000000.0
    
    # Run Shared Backtest
    ret_pct, mdd, trades_log, detailed_log, equity_curve = run_backtest_segment(
        df, params, initial_balance=initial_balance, return_series=True
    )
    
    # Print Logs (Disabled per user request)
    # for log in detailed_log:
    #     t_str = log['time']
    #     if log['type'] == 'BUY':
    #         print(f"[{t_str}] 🟢 BUY  @ {log['price']:,.0f} | SL: {log['stop_loss']:,.0f} | TP: {log['take_profit']:,.0f}")
    #     elif log['type'] == 'SELL':
    #         print(f"[{t_str}] 🔴 SELL @ {log['price']:,.0f} | Ret: {log['return']:.2f}% | Bal: {log['balance']:,.0f} | {log['reason']}")

    final_val = initial_balance * (1 + ret_pct/100)
    
    print("\n" + "="*50)
    print("BACKTEST RESULT")
    print("="*50)
    print(f"Final Balance: {final_val:,.0f} KRW (Initial: {initial_balance:,.0f})")
    print(f"Total Return : {ret_pct:.2f}%")
    print(f"Trade Count  : {len(trades_log)}")
    if trades_log:
        win_rate = len([t for t in trades_log if t > 0])/len(trades_log)*100
        gross_profit = sum([t for t in trades_log if t > 0])
        gross_loss = abs(sum([t for t in trades_log if t < 0]))
        pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        
        print(f"Win Rate      : {win_rate:.2f}%")
        print(f"Profit Factor : {pf:.2f}")
        print(f"Max drawdown  : {mdd:.2f}%")
    print("="*50)
    
    # --- 2. Walk Forward Analysis (Robustness) ---
    from src.spot_strategy.walk_forward_spot import SpotWalkForwardAnalyzer
    print(f"\n🚀 Running Walk-Forward Analysis (5 Splits)...")
    wfa = SpotWalkForwardAnalyzer(df, params)
    wfa_results = wfa.run(n_splits=5)
    
    print(f"{'='*50}")
    print(f"WALK FORWARD ANALYSIS RESULT")
    print(f"{'='*50}")
    if wfa_results.empty:
        print("⚠️ Not enough data to run Walk-Forward Analysis (Segments too short).")
    else:
        print(wfa_results.to_markdown(index=False, floatfmt=".2f"))
        
        avg_wfa_ret = wfa_results['Return'].mean()
        print(f"\nAverage Return per Split: {avg_wfa_ret:.2f}%")
        consistency = len(wfa_results[wfa_results['Return'] > 0]) / len(wfa_results) * 100
        print(f"Consistency (Positive Segments): {consistency:.0f}%")

    # --- 3. Monte Carlo Simulation (Probability) ---
    from src.spot_strategy.monte_carlo_spot import SpotMonteCarloSimulator
    print(f"\n🎲 Running Monte Carlo Simulation (10,000 runs)...")
    
    if trades_log:
        mc = SpotMonteCarloSimulator(trades_log) # trades_log is list of % returns
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

if __name__ == "__main__":
    import argparse
    import optuna
    
    parser = argparse.ArgumentParser()
    # Default matches the new optimize_spot.py output
    parser.add_argument("--db", type=str, default="spot_strategy.db")
    parser.add_argument("--symbols", type=str, default="KRW-BTC,KRW-ETH")
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    


    # 2. Run Verification
    print("\n" + "="*70)
    print(f"🚀 VERIFYING UNIVERSAL STRATEGY ON: {symbols}")
    print("="*70)
    
    # 2-1. Load Universal Best Params
    db_path = "spot_strategy.db"
    storage = f"sqlite:///{db_path}"
    study_name = "spot_strategy"
    
    best_params = {}
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        best_params = study.best_params
        print(f"✅ Loaded Universal Params (Score: {study.best_value:.4f})")
    except Exception as e:
        print(f"❌ Failed to load Universal DB ({db_path}): {e}")
        sys.exit(1)
        
    for symbol in symbols:
        print(f"\n👉 Analyzing {symbol}...")

        # Load from Full History Cache
        tf = best_params.get('TIMEFRAME', '1h')
        # Reuse the logic: check data folder for CSV
        filename = f"{symbol}_{tf}_20180101_spot.csv"
        filepath = os.path.join(os.path.dirname(__file__), '../../data', filename)
        
        if os.path.exists(filepath):
            logger.info(f"📂 Loading cached full history: {filename}")
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Run Detailed Backtest
            from config.settings import TRAIN_CUTOFF_DATE
            cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)
            
            train_df = df[df['datetime'] < cutoff_ts].copy()
            test_df = df[df['datetime'] >= cutoff_ts].copy()
            
            print(f"\n[DATA SPLIT] Train: {len(train_df)} candles | Test (OOS): {len(test_df)} candles (Cutoff: {TRAIN_CUTOFF_DATE})")
            
            if not test_df.empty:
                print(f"\n🔵 >>> Running Verification on TEST DATA (Out-of-Sample: {TRAIN_CUTOFF_DATE} ~ Now) <<<")
                detailed_backtest_spot(test_df, best_params)
            else:
                print(f"⚠️ No Test Data found after {TRAIN_CUTOFF_DATE}! Check your data files.")
                print("Running on Full Data as fallback...")
                detailed_backtest_spot(df, best_params)
        else:
            print(f"❌ Full history cache not found at {filepath}")
            print("Please run optimize_spot.py first to collect full data.")
