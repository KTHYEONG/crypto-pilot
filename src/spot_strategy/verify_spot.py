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
from src.spot_strategy.upbit_client import UpbitClient

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

def load_best_params_from_mysql(mode, storage_url):
    """MySQL에서 최적 파라미터 로드"""
    study_name = f"spot_{mode.lower()}_strategy"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        return study_name, study.best_params, study.best_value
    except KeyError:
        return None, None, None

def load_data_spot(symbol, tf):
    """
    Load Hourly and Daily data for Multi-Timeframe Verification using UpbitClient.
    [IMPROVED] Search for existing files with glob pattern to maximize cache usage.
    """
    from config.settings import BACKTEST_START_DATE, BACKTEST_END_DATE
    import glob
    
    # Setup Upbit Client
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    client = UpbitClient(access, secret)
    
    start_ts = int(pd.Timestamp(BACKTEST_START_DATE).timestamp() * 1000)
    end_ts = int(pd.Timestamp(BACKTEST_END_DATE).timestamp() * 1000)
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    
    def get_or_download(timeframe, start_cfg, end_cfg):
        sym_clean = symbol.replace('/', '_')
        # Target exact filename
        target_name = f"{sym_clean}_{timeframe}_{start_cfg.replace('-','')}_{end_cfg.replace('-','')}_spot.csv"
        target_path = os.path.join(data_dir, target_name)
        
        # 1. Try to find EXISTING files (Pattern Match)
        # Pattern: KRW-BTC_1h_*_spot.csv
        pattern = os.path.join(data_dir, f"{sym_clean}_{timeframe}_*_spot.csv")
        candidates = glob.glob(pattern)
        
        df = None
        loaded_path = None
        
        if candidates:
            # Sort by size (assume larger file has more data) or creation time
            # Let's pick the largest file
            candidates.sort(key=lambda x: os.path.getsize(x), reverse=True)
            best_candidate = candidates[0]
            
            try:
                temp_df = pd.read_csv(best_candidate)
                if 'timestamp' in temp_df.columns:
                    # Check coverage
                    min_ts = temp_df['timestamp'].min()
                    max_ts = temp_df['timestamp'].max()
                    
                    if max_ts >= (end_ts - 86400000 * 5): # Allow 5 days lag for end date (e.g. recent weekend)
                        # Check start date
                        if min_ts <= (start_ts + 86400000 * 30): # 30 days tolerance
                            # logger.info(f"   ✅ Found compatible cached data: {os.path.basename(best_candidate)}")
                            df = temp_df
                            loaded_path = best_candidate
                        else:
                            # Start date is later than requested -> Likely Listing Date issue
                            # Accept it silently per user request
                            # logger.info(f"   ⚠️ Cached data starts later ({pd.to_datetime(min_ts, unit='ms').strftime('%Y-%m-%d')}) than requested. Using cached file.")
                            df = temp_df
                            loaded_path = best_candidate
                    else:
                        file_end_str = pd.to_datetime(max_ts, unit='ms').strftime('%Y-%m-%d')
                        logger.info(f"   ⚠️ Cached data too old (Ends {file_end_str}). Downloading fresh...")
            except Exception as e:
                logger.error(f"Error reading candidate {best_candidate}: {e}")
        
        # 2. If not found or insufficient, Download
        if df is None:
            start_str = pd.to_datetime(start_ts, unit='ms').strftime('%Y-%m-%d')
            end_str = pd.to_datetime(end_ts, unit='ms').strftime('%Y-%m-%d')
            logger.info(f"   📥 [API START] Downloading {timeframe} data for {symbol} ({start_str} ~ {end_str})... This may take a while.")
            
            try:
                import time
                t0 = time.time()
                df = client.fetch_ohlcv(symbol, timeframe, since=start_ts, end=end_ts)
                elapsed = time.time() - t0
                
                if df is not None and not df.empty:
                    try:
                        df.to_csv(target_path, index=False)
                        loaded_path = target_path
                        logger.info(f"   ✅ [API DONE] Downloaded {len(df)} rows in {elapsed:.1f}s. Saved to: {os.path.basename(target_path)}")
                    except Exception as save_err:
                        logger.error(f"   ❌ Failed to save file {target_path}: {save_err}")
                else:
                    logger.warning(f"   ⚠️ Download returned empty/None for {symbol} {timeframe}")
            except Exception as e:
                logger.error(f"   ❌ Failed to download {timeframe} data: {e}")
                
        if df is not None:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
        return df

    # Load Daily
    daily_df = get_or_download('1d', BACKTEST_START_DATE, BACKTEST_END_DATE)
    
    # Load Hourly
    hourly_df = get_or_download(tf, BACKTEST_START_DATE, BACKTEST_END_DATE)
    
    return hourly_df, daily_df

def prepare_spot_data(hourly_df, daily_df, strategy):
    """
    Merge Daily indicators into Hourly data for backtesting.
    Simulates what EngineFastSpot does.
    """
    # 1. Calculate Indicators
    daily_df = strategy.generate_signals(daily_df.copy())
    hourly_df = strategy.generate_signals(hourly_df.copy())
    
    # 2. Prepare Merge Keys
    daily_df['date_key'] = daily_df['datetime'].dt.strftime('%Y-%m-%d')
    hourly_df['date_key'] = hourly_df['datetime'].dt.strftime('%Y-%m-%d')
    
    # 3. Select Daily Columns to Merge
    exclude_cols = {'date_key', 'datetime', 'date', 'open', 'high', 'low', 'close', 'volume'}
    daily_cols = [c for c in daily_df.columns if c not in exclude_cols]
    
    # 4. Shift Daily Data (Avoid Lookahead) & Rename
    shifted_daily = daily_df[daily_cols + ['date_key']].copy()
    shifted_daily[daily_cols] = shifted_daily[daily_cols].shift(1)
    shifted_daily = shifted_daily.rename(columns={c: f'daily_{c}' for c in daily_cols})
    
    # 5. Merge
    merged_df = pd.merge(hourly_df, shifted_daily, on='date_key', how='left')
    
    # 6. Apply Logic Merge (Trend Alignment)
    # Spot Engine Logic: Trend is LONG(1) only if both Hourly and Daily agree
    if 'daily_trend_direction' in merged_df.columns:
        h_trend = merged_df['trend_direction'].fillna(0).values
        d_trend = merged_df['daily_trend_direction'].fillna(0).values
        merged_df['trend_direction'] = np.where((h_trend == 1) & (d_trend == 1), 1, 0)
        
    return merged_df

def verify_single_symbol(symbol, best_params, primary_symbols):
    """단일 심볼 백테스트 실행 (Multi-Timeframe)"""
    from config.settings import TRAIN_CUTOFF_DATE
    
    tf = best_params.get('TIMEFRAME', '1h')
    
    # Load BOTH Hourly and Daily
    hourly_df, daily_df = load_data_spot(symbol, tf)
    
    if hourly_df is None or daily_df is None:
        print(f"   ⚠️ Data missing for {symbol}. Skipping.")
        return None
    
    # OOS 슬라이싱
    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)
    test_hourly = hourly_df[hourly_df['datetime'] >= cutoff_ts].copy()
    test_daily = daily_df[daily_df['datetime'] >= cutoff_ts].copy()
    
    if test_hourly.empty:
        print(f"   ⚠️ No OOS data for {symbol}. Skipping.")
        return None
    
    # 전략 준비 및 데이터 병합
    strategy = UltimateStrategy("Verify", best_params)
    test_df = prepare_spot_data(test_hourly, test_daily, strategy)
    
    # 백테스트 실행
    ret_pct, mdd, trades_log, _, _ = run_backtest_segment(
        test_df, best_params, initial_balance=10000000.0, return_series=True
    )
    
    trade_count = len(trades_log) if trades_log else 0
    win_rate = 0.0
    pf = 0.0
    if trade_count > 0:
        wins = [t for t in trades_log if t > 0]
        # Spot trades_log contains Profit amounts, not %? 
        # run_backtest_segment returns trades_log as list of PnL amounts (based on code context)
        # Let's verify backtest_utils return. Usually it returns PnL amounts.
        losses = [t for t in trades_log if t <= 0]
        win_rate = len(wins) / trade_count * 100
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    is_primary = symbol in primary_symbols
    indicator = "🎯 PRIMARY" if is_primary else "📊 REFERENCE"
    print(f"   - {symbol} [{indicator}]: Return {ret_pct:.2f}% | MDD {mdd:.2f}% | Trades {trade_count} | Win {win_rate:.1f}% | PF {pf:.2f}")
    
    return {'symbol': symbol, 'return': ret_pct, 'mdd': mdd, 'trades': trade_count, 'win_rate': win_rate, 'pf': pf, 'is_primary': is_primary}
def calculate_mode_performance(all_results):
    """PRIMARY/REFERENCE 성능 계산"""
    primary_results = [r for r in all_results if r['is_primary']]
    if not primary_results:
        return None
    
    avg_ret = sum(r['return'] for r in primary_results) / len(primary_results)
    print(f"\n   📊 Results Summary:")
    print(f"   - PRIMARY Avg Return (BTC/ETH): {avg_ret:.2f}%")
    
    ref_results = [r for r in all_results if not r['is_primary']]
    if ref_results:
        ref_avg = sum(r['return'] for r in ref_results) / len(ref_results)
        print(f"   - REFERENCE Avg Return (Alts): {ref_avg:.2f}%")
    
    return avg_ret

if __name__ == "__main__":
    import argparse
    import optuna
    import shutil
    from config.settings import TRAIN_CUTOFF_DATE
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="KRW-BTC,KRW-ETH", 
                        help="Comma-separated list of symbols to verify")
    parser.add_argument("--alt", type=int, default=0, choices=[0, 1],
                        help="Include altcoins for validation (1=yes, 0=no). Adds SOL, XRP, DOGE, ADA")
    parser.add_argument("--all-modes", action="store_true",
                        help="Verify all modes (SCALP, DAY, SWING, UNIFIED). Default: UNIFIED only")
    args = parser.parse_args()
    
    # 심볼 목록 빌드
    base_symbols = [s.strip() for s in args.symbols.split(',')]
    if args.alt == 1:
        alt_symbols = ['KRW-SOL', 'KRW-XRP', 'KRW-DOGE', 'KRW-ADA']
        for alt in alt_symbols:
            if alt not in base_symbols:
                base_symbols.append(alt)
        print(f"📊 Altcoin validation enabled. Added: {', '.join(alt_symbols)}")
    
    symbols = base_symbols
    PRIMARY_SYMBOLS = ['KRW-BTC', 'KRW-ETH']
    
    # [DEFAULT] Verify UNIFIED only (fastest, most flexible)
    # Use --all-modes to compare all strategies
    if args.all_modes:
        MODES = ['SCALP', 'DAY', 'SWING', 'UNIFIED']
        print(f"🔍 All-Modes Verification enabled. Testing: {MODES}")
    else:
        MODES = ['UNIFIED']
        print(f"⚡ Quick Verification: UNIFIED mode only (use --all-modes to compare all strategies)")
    
    results = []
    
    print("\n" + "="*80)
    print(f"🚀 INTEGRATED STRATEGY VERIFICATION (Spot)")
    print(f"   Searching for optimized strategies: {MODES}")
    print("="*80)
    
    # MySQL 설정
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
        print(f"\n🔎 Verifying {mode} Mode Strategy (from MySQL)...")
        
        try:
            study_name, best_params, train_score = load_best_params_from_mysql(mode, storage_url)
            if study_name is None:
                print(f"⚠️  {mode} strategy not found in MySQL. Skipping...")
                continue
            
            print(f"   ✅ Loaded Params (Train Score: {train_score:.4f})")
            print(f"   Timeframe: {best_params.get('TIMEFRAME', '1h')}")
            
            # OOS 검증
            all_results = []
            for symbol in symbols:
                result = verify_single_symbol(symbol, best_params, PRIMARY_SYMBOLS)
                if result:
                    all_results.append(result)
            
            avg_ret = calculate_mode_performance(all_results)
            if avg_ret is None:
                continue
            
            results.append({
                'mode': mode,
                'study_name': study_name,
                'return': avg_ret,
                'score': train_score,
                'all_results': all_results
            })
            
            if avg_ret > best_overall_score:
                best_overall_score = avg_ret
                best_mode = mode
                best_study_name = study_name
                    
        except Exception as e:
            print(f"   ❌ Error processing {mode}: {e}")

    print("\n" + "="*80)
    print("🏆 FINAL RESULTS (Spot)")
    print("="*80)
    
    if not results:
        print("❌ No valid strategies found/verified.")
        sys.exit(1)
        
    results.sort(key=lambda x: x['return'], reverse=True)
    
    for res in results:
        mark = "👑" if res['mode'] == best_mode else "  "
        print(f"{mark} {res['mode']:<6} | OOS Return: {res['return']:>7.2f}% | Train Score: {res['score']:.4f}")
        
    print("-" * 80)
    
    target_db = "spot_strategy.db"
    target_study_name = "spot_strategy"
    
    if best_study_name:
        print(f"💾 Saving Best Strategy ({best_mode}) from MySQL to '{target_db}'...")
        
        try:
            # 1. Remove old target
            if os.path.exists(target_db):
                os.remove(target_db)
            
            # 2. Load Source Study (MySQL)
            src_study = optuna.load_study(study_name=best_study_name, storage=storage_url)
            best_trial = src_study.best_trial
            
            # 3. Create Target Storage (Local SQLite)
            target_storage = f"sqlite:///{target_db}"
            
            # 4. Create Standard Study
            print(f"   ⚙️  Migrating best params to standard '{target_study_name}' study...")
            optuna.create_study(study_name=target_study_name, storage=target_storage, direction="maximize", load_if_exists=True)
            study_dest = optuna.load_study(study_name=target_study_name, storage=target_storage)
            
            # 5. Create & Add Frozen Trial
            frozen_trial = optuna.trial.create_trial(
                params=best_trial.params,
                distributions=src_study.trials[best_trial.number].distributions, # Load distributions from source
                value=best_trial.value,
            )
            study_dest.add_trial(frozen_trial)
            
            print(f"✅ SUCCESSFULLY DEPLOYED {best_mode} STRATEGY!")
            print(f"   The bot will now use the OOS-verified best strategy from '{target_db}'.")
            
        except Exception as e:
            print(f"❌ Error Saving Strategy: {e}")
    else:
        print("❌ No best strategy selected.")
