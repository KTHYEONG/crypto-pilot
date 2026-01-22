
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
    import shutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # PRIMARY SYMBOLS: Only these affect final strategy selection
    PRIMARY_SYMBOLS = ['BTC/USDT', 'ETH/USDT']
    
    # Modes to check
    MODES = ['SCALP', 'DAY', 'SWING']
    results = []
    
    print("\n" + "="*80)
    print(f"🚀 INTEGRATED STRATEGY VERIFICATION (Futures)")
    print(f"   Searching for optimized strategies: {MODES}")
    print("="*80)
    
    best_overall_score = -float('inf')
    best_mode = None
    # MySQL Setup
    from dotenv import load_dotenv
    from urllib.parse import quote_plus
    load_dotenv()
    
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "trading_optuna")
    
    if not all([db_user, db_pass, db_name]):
        print("❌ Error: Missing DB credentials in .env")
        sys.exit(1)
        
    safe_pass = quote_plus(db_pass)
    storage_url = f"mysql+pymysql://{db_user}:{safe_pass}@{db_host}:{db_port}/{db_name}"
    
    best_overall_score = -float('inf')
    best_mode = None
    best_study_name = None
    
    for mode in MODES:
        study_name = f"futures_{mode.lower()}_strategy"
        
        print(f"\n🔎 Verifying {mode} Mode Strategy (from MySQL)...")
        
        try:
            # Check if study exists
            try:
                optuna.load_study(study_name=study_name, storage=storage_url)
            except KeyError:
                print(f"⚠️  {mode} strategy not found in MySQL. Skipping...")
                continue
                
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            best_params = study.best_params
            train_score = study.best_value
            
            print(f"   ✅ Loaded Params (Train Score: {train_score:.4f})")
            print(f"   Timeframe: {best_params.get('TIMEFRAME')}")
            
            # --- OOS Verification Loop ---
            # Separate tracking for PRIMARY (affects strategy selection) and ALL (display only)
            primary_total_ret = 0
            primary_count = 0
            all_results = []  # Store all symbol results for display
            
            for symbol in symbols:
                tf = best_params.get('TIMEFRAME', '1h')
                
                # Load Data
                try:
                    hourly_df, daily_df = load_data(symbol, BACKTEST_START_DATE, BACKTEST_END_DATE, tf)
                except Exception as e:
                    print(f"   ❌ Data load error for {symbol}: {e}")
                    continue
                
                # Slice for OOS (Test Data)
                cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)
                test_hourly = hourly_df[hourly_df['datetime'] >= cutoff_ts].copy()
                test_daily = daily_df[daily_df['datetime'] >= cutoff_ts].copy()
                
                if test_hourly.empty:
                    print(f"   ⚠️ No OOS data for {symbol}. Skipping verification.")
                    continue
                
                # Run Backtest
                strategy = UltimateStrategy("Verify", best_params)
                df = prepare_futures_data(test_hourly, test_daily, strategy)
                ret_pct, mdd, _, _, _ = run_backtest_segment_futures(
                    df, best_params, initial_balance=750.0, return_series=True
                )
                
                # Track for PRIMARY symbols (strategy selection)
                is_primary = symbol in PRIMARY_SYMBOLS
                if is_primary:
                    primary_total_ret += ret_pct
                    primary_count += 1
                
                # Store all results for display
                all_results.append({
                    'symbol': symbol,
                    'return': ret_pct,
                    'mdd': mdd,
                    'is_primary': is_primary
                })
                
                # Display with PRIMARY/REFERENCE indicator
                indicator = "🎯 PRIMARY" if is_primary else "📊 REFERENCE"
                print(f"   - {symbol} [{indicator}]: Return {ret_pct:.2f}% | MDD {mdd:.2f}%")
            
            # Calculate average using PRIMARY symbols only
            if primary_count > 0:
                avg_ret = primary_total_ret / primary_count
                print(f"\n   � Results Summary:")
                print(f"   - PRIMARY Avg Return (BTC/ETH): {avg_ret:.2f}%")
                if len(all_results) > primary_count:
                    ref_total = sum([r['return'] for r in all_results if not r['is_primary']])
                    ref_count = len([r for r in all_results if not r['is_primary']])
                    ref_avg = ref_total / ref_count if ref_count > 0 else 0
                    print(f"   - REFERENCE Avg Return (Alts): {ref_avg:.2f}%")
                
                results.append({
                    'mode': mode,
                    'study_name': study_name,
                    'return': avg_ret,  # Only PRIMARY symbols affect this
                    'score': train_score,
                    'all_results': all_results  # Keep for detailed view
                })
                
                if avg_ret > best_overall_score:
                    best_overall_score = avg_ret
                    best_mode = mode
                    best_study_name = study_name
            
        except Exception as e:
            print(f"   ❌ Error processing {mode}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("🏆 FINAL RESULTS")
    print("="*80)
    
    if not results:
        print("❌ No valid strategies found/verified.")
        sys.exit(1)
        
    results.sort(key=lambda x: x['return'], reverse=True)
    
    for res in results:
        mark = "👑" if res['mode'] == best_mode else "  "
        print(f"{mark} {res['mode']:<6} | OOS Return: {res['return']:>7.2f}% | Train Score: {res['score']:.4f}")
        
    print("-" * 80)
    
    target_db = "futures_strategy.db"
    
    if best_study_name:
        print(f"💾 Saving Best Strategy ({best_mode}) from MySQL to '{target_db}'...")
        
        try:
            # 1. Remove old target
            if os.path.exists(target_db):
                os.remove(target_db)
            
            # 2. Rename study inside the new DB to 'futures_strategy' for standard access
            target_storage = f"sqlite:///{target_db}"
            
            print(f"   ⚙️  Migrating best params to standard 'futures_strategy' study...")
            
            # 3. Load Source (MySQL)
            src_study = optuna.load_study(study_name=best_study_name, storage=storage_url)
            best_trial = src_study.best_trial
            
            # 4. Create Target (Local)
            # Create new standard study "futures_strategy"
            optuna.create_study(study_name="futures_strategy", storage=target_storage, direction="maximize", load_if_exists=True)
            study_dest = optuna.load_study(study_name="futures_strategy", storage=target_storage)
            
            # 5. Create Frozen Trial matching the best trial
            frozen_trial = optuna.trial.create_trial(
                params=best_trial.params,
                distributions=best_trial.distributions,
                value=best_trial.value,
            )
            
            # 6. Add trial directly to destination study
            study_dest.add_trial(frozen_trial)
            
            print(f"✅ SUCCESSFULLY DEPLOYED {best_mode} STRATEGY!")
            print(f"   The bot will now use the OOS-verified best strategy from '{target_db}'.")
            
        except Exception as e:
            print(f"❌ Error Saving Strategy: {e}")
    else:
        print("❌ No best strategy selected.")
