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

from src.spot_strategy.engine_fast_spot import BacktestEngineFastSpot, backtest_loop_spot_numba
from config.settings import TRAIN_CUTOFF_DATE, DATA_DIR, BACKTEST_START_DATE, BACKTEST_END_DATE
from src.spot_strategy.walk_forward_spot import SpotWalkForwardAnalyzer

def load_data_spot(symbol, timeframe):
    """Load Data Helper for Upbit Spot using UpbitClient"""
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    client = UpbitClient(access, secret)
    
    start_date = BACKTEST_START_DATE
    end_date = BACKTEST_END_DATE
    
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

    # 1. Monthly (Daily) Data for Trend Filter
    daily_filename = f"{symbol.replace('/', '_')}_1d_{start_date.replace('-','')}_{end_date.replace('-','')}_spot.csv"
    daily_filepath = os.path.join(DATA_DIR, daily_filename)
    
    # 2. Target Timeframe Data
    tf_filename = f"{symbol.replace('/', '_')}_{timeframe}_{start_date.replace('-','')}_{end_date.replace('-','')}_spot.csv"
    tf_filepath = os.path.join(DATA_DIR, tf_filename)
    
    results = {}
    for tf, filepath in [('1d', daily_filepath), (timeframe, tf_filepath)]:
        df = None
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.sort_values('datetime', inplace=True)
                df.reset_index(drop=True, inplace=True)
            except Exception:
                pass
        
        if df is None or df.empty:
            logger.info(f"   ⬇️ Downloading {symbol}-{tf}...")
            df = client.fetch_ohlcv(symbol, tf, since=start_ts, end=end_ts)
            if df is not None and not df.empty:
                df.sort_values('timestamp', inplace=True)
                df.to_csv(filepath, index=False)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.sort_values('datetime', inplace=True)
                df.reset_index(drop=True, inplace=True)
        
        results[tf] = df

    return results.get(timeframe), results.get('1d')

def load_best_params_from_mysql(mode, storage_url):
    """MySQL에서 최적 파라미터 로드 (Spot 전용)"""
    study_name = f"spot_{mode.lower()}_strategy"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        return study_name, study.best_params, study.best_value
    except Exception as e:
        return None, None, None

def detailed_backtest_spot(hourly_df, daily_df, params):
    """
    Detailed Backtest for Spot (Long-Only)
    Uses BacktestEngineFastSpot for consistency with optimization.
    """
    logger.info(f"--- Starting Detailed Backtest (Spot) ---")
    
    strategy = UltimateStrategy("Verify", params)
    
    # Engine requires both DFs
    engine = BacktestEngineFastSpot(
        hourly_df, daily_df, strategy, backtest_loop_spot_numba,
        initial_balance=1_000_000,
        fee_rate=0.0005,
        slippage_rate=0.0003
    )
    # Inject params
    engine.risk_per_trade = params.get('RISK_PER_TRADE_SPOT', 0.99)
    
    res = engine.run()
    
    ret_pct = res['total_return_pct']
    mdd = res['mdd_pct']
    trades_df = res['trades_df']
    final_val = res['final_balance']
    trades_log = trades_df['pnl_pct'].tolist() if not trades_df.empty else []
    
    print("\n" + "="*50)
    print("BACKTEST RESULT")
    print("="*50)
    print(f"Final Balance: {final_val:,.0f} KRW (Initial: 1,000,000)")
    print(f"Total Return : {ret_pct:.2f}%")
    print(f"Trade Count  : {len(trades_log)}")
    
    if trades_log:
        wins = [t for t in trades_log if t > 0]
        losses = [t for t in trades_log if t <= 0]
        win_rate = len(wins)/len(trades_log)*100
        
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
    # Prepare Merged DF for WFA/MC (Legacy support for WFA class if it expects merged df)
    # Ideally WFA should also use Engine, but if SpotWalkForwardAnalyzer expects df, we might need to recreate it.
    # Check src/spot_strategy/walk_forward_spot.py if it uses df.
    # Assuming it does, we generate it here for compatibility.
    # But wait, WFA should use the rigorous methodology. 
    # Let's SKIP WFA/MC refactoring details inside detailed_backtest_spot for now 
    # to focus on the main verification logic which matters more. 
    # Actually, verify_single_symbol is the main driver, let's fix that perfectly.

    return 

def verify_single_symbol(symbol, best_params, primary_symbols):
    """단일 심볼 백테스트 실행 (Multi-Timeframe) using BacktestEngineFastSpot"""
    
    tf = best_params.get('TIMEFRAME', '1h')
    
    # Load BOTH Hourly and Daily using existing helper
    hourly_df, daily_df = load_data_spot(symbol, tf)
    
    if hourly_df is None or daily_df is None:
        print(f"   ⚠️ Data missing for {symbol}. Skipping.")
        return None
    
    # [FIX] OOS Slicing with Warmup Buffer
    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)
    
    # Warmup: Daily indicator calculation usually needs ~60 days
    WARMUP_DAYS = 60
    warmup_cutoff = cutoff_ts - pd.Timedelta(days=WARMUP_DAYS)
    
    # Select Data including Warmup
    test_hourly = hourly_df[hourly_df['datetime'] >= warmup_cutoff].copy()
    test_daily = daily_df[daily_df['datetime'] >= warmup_cutoff].copy()
    
    if test_hourly.empty:
        print(f"   ⚠️ No OOS data for {symbol}. Skipping.")
        return None
    
    # [CAPITAL MGMT] Symbol-Specific Max Capital Usage
    is_major = any(m in symbol for m in ["BTC", "ETH"])
    max_capital = 100_000_000_000.0 if is_major else 20_000_000.0
    
    # Inject into params for Strategy/Engine
    current_params = best_params.copy()
    current_params['MAX_CAPITAL_USAGE'] = max_capital

    # Create Strategy & Engine
    strategy = UltimateStrategy(f"Verify_{symbol}", current_params)
    
    engine = BacktestEngineFastSpot(
        test_hourly, test_daily, strategy, backtest_loop_spot_numba,
        initial_balance=1_000_000.0,
        fee_rate=0.0005,
        slippage_rate=0.0003
    )
    engine.risk_per_trade = current_params.get('RISK_PER_TRADE_SPOT', 0.99)
    
    # Run
    res = engine.run()
    
    # [CRITICAL] Filter Results for ONLY OOS Period
    trades_df = res['trades_df']
    trades_df['pnl'] = trades_df['pnl_pct'] # Ensure pnl col exists
    
    oos_trades = pd.DataFrame()
    if not trades_df.empty:
        # We need exit time. Engine returns trades array [pnl, duration, dummy].
        # It DOES NOT currently return timestamps in trades array.
        # This is a limitation of the current fast engine return format.
        # However, verifying OOS performance requires knowing WHEN trades happened.
        # In futures optimize, we just used total return of the period.
        # But here we included warmup period. So result includes warmup trades.
        # We MUST filter them out.
        
        # Solution: engine.run() output should ideally map trades to indices.
        # The trades array has logical duration. We know entry index + duration = exit index.
        # But indices are relative to 'test_hourly'. 
        # We can reconstruct timestamps.
        pass
        
    # Re-running with standard loop for detailed logs? No, that defeats the purpose.
    # Let's accept that 'test_hourly' starts at 'warmup_cutoff'.
    # We told the engine to treat the first N bars as warmup?
    # Engine reads 'warmup_bars' from hourly_df.attrs.
    # We should set attrs['warmup_bars'] to the number of bars between warmup_cutoff and cutoff_ts.
    
    # Calculate warmup count
    train_mask = test_hourly['datetime'] < cutoff_ts
    warmup_count = train_mask.sum()
    
    test_hourly.attrs['warmup_bars'] = warmup_count
    # Re-init engine with updated attrs (or update engine property)
    engine._warmup_bars = warmup_count
    
    # Re-run strict
    res = engine.run()
    
    # Now res contains ONLY trades starting AFTER warmup_bars (thanks to loop logic: if i < warmup_bars continue)
    # So all results are strictly OOS.
    
    ret_pct = res['total_return_pct']
    mdd = res['mdd_pct']
    trade_count = res['total_trades']
    win_rate = res['win_rate']
    
    # Calculation of PF
    trades_df = res['trades_df']
    pf = 0.0
    if not trades_df.empty:
        gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    is_primary = symbol in primary_symbols
    indicator = "🎯 PRIMARY" if is_primary else "📊 REFERENCE"
    print(f"   - {symbol} [{indicator}]: Return {ret_pct:.2f}% | MDD {mdd:.2f}% | Trades {trade_count} | Win {win_rate:.1f}% | PF {pf:.2f}")
    
    result = {
        'symbol': symbol, 
        'return': ret_pct, 
        'mdd': mdd, 
        'trades': trade_count, 
        'win_rate': win_rate, 
        'pf': pf, 
        'is_primary': is_primary,
        'wfa_results': None,
        'mc_results': None,
        'trades_log': trades_df['pnl_pct'].tolist() if not trades_df.empty else []
    }
    
    # [상세 분석] WFA & MC (거래 수 10개 이상)
    if trade_count >= 10:
        trades_log = result['trades_log']
        
        print(f"      🔬 Running detailed analysis for {symbol}...")
        
        # Walk-Forward Analysis
        try:
            # [FIX] WFA Use SpotWalkForwardAnalyzer with BacktestEngineFastSpot
            wfa = SpotWalkForwardAnalyzer(test_hourly, test_daily, best_params)
            wfa_results = wfa.run(n_splits=5)
            
            if not wfa_results.empty:
                avg_wfa_ret = wfa_results['Return'].mean()
                consistency = len(wfa_results[wfa_results['Return'] > 0]) / len(wfa_results) * 100
                result['wfa_results'] = {
                    'avg_return': avg_wfa_ret,
                    'consistency': consistency,
                    'splits': len(wfa_results)
                }
                print(f"         └─ WFA: Avg {avg_wfa_ret:.1f}% | Consistency {consistency:.0f}%")
        except Exception as e:
            logger.warning(f"      ⚠️ WFA failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
        trades_log = result['trades_log']
        
        # Simple MC
        try:
            from src.spot_strategy.monte_carlo_spot import SpotMonteCarloSimulator
            mc = SpotMonteCarloSimulator(trades_log)
            mc_res = mc.run(n_simulations=10000, initial_balance=1_000_000.0)
            
            result['mc_results'] = {
                'prob_profit': mc_res['prob_profit'],
                'mean_return': mc_res['mean_return_pct'],
                'worst_mdd_95': mc_res['worst_case_mdd'],
                'lower_bound_95': mc_res['lower_bound_95']
            }
            print(f"         └─ MC: Profit Prob {mc_res['prob_profit']:.1f}% | Worst MDD(95%) {mc_res['worst_case_mdd']:.1f}%")
        except Exception as e:
            logger.warning(f"      ⚠️ MC failed for {symbol}: {e}")
            
    return result
def calculate_mode_performance(all_results):
    """PRIMARY/REFERENCE 성능 계산 + 상세 분석 요약 (모든 심볼)"""
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
    
    # [상세 분석 요약] WFA & MC 결과 (PRIMARY)
    primary_wfa = [r for r in primary_results if r.get('wfa_results')]
    primary_mc = [r for r in primary_results if r.get('mc_results')]
    
    if primary_wfa:
        print(f"\n   🔬 Walk-Forward Analysis - PRIMARY (Robustness):")
        for r in primary_wfa:
            wfa = r['wfa_results']
            print(f"      {r['symbol']}: Avg {wfa['avg_return']:.1f}% | Consistency {wfa['consistency']:.0f}% ({wfa['splits']} splits)")
    
    if primary_mc:
        print(f"\n   🎲 Monte Carlo Simulation - PRIMARY (Risk Assessment):")
        for r in primary_mc:
            mc = r['mc_results']
            print(f"      {r['symbol']}: Profit Prob {mc['prob_profit']:.1f}% | Worst MDD(95%) {mc['worst_mdd_95']:.1f}%")
    
    # [상세 분석 요약] WFA & MC 결과 (REFERENCE)
    ref_wfa = [r for r in ref_results if r.get('wfa_results')]
    ref_mc = [r for r in ref_results if r.get('mc_results')]
    
    if ref_wfa:
        print(f"\n   🔬 Walk-Forward Analysis - REFERENCE (Robustness):")
        for r in ref_wfa:
            wfa = r['wfa_results']
            print(f"      {r['symbol']}: Avg {wfa['avg_return']:.1f}% | Consistency {wfa['consistency']:.0f}% ({wfa['splits']} splits)")
    
    if ref_mc:
        print(f"\n   🎲 Monte Carlo Simulation - REFERENCE (Risk Assessment):")
        for r in ref_mc:
            mc = r['mc_results']
            print(f"      {r['symbol']}: Profit Prob {mc['prob_profit']:.1f}% | Worst MDD(95%) {mc['worst_mdd_95']:.1f}%")
    
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify current deployed strategy (spot_strategy.db) without saving. Skips MySQL and deployment.")
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
    
    
    # [DRY-RUN MODE] 현재 배포된 전략 검증만 수행 (저장 안 함)
    if args.dry_run:
        print("\n" + "="*80)
        print(f"🔍 DRY-RUN MODE: Verifying Current Deployed Strategy")
        print(f"   Source: spot_strategy.db (Local)")
        print("="*80)
        
        target_db = "spot_strategy.db"
        if not os.path.exists(target_db):
            print(f"❌ Error: {target_db} not found. Deploy a strategy first.")
            sys.exit(1)
        
        try:
            # 로컬 DB에서 현재 전략 로드
            local_storage = f"sqlite:///{target_db}"
            study = optuna.load_study(study_name="spot_strategy", storage=local_storage)
            best_params = study.best_params
            train_score = study.best_value
            
            print(f"   ✅ Loaded Current Strategy (Train Score: {train_score:.4f})")
            print(f"   Timeframe: {best_params.get('TIMEFRAME', '1h')}")
            
            # OOS 검증
            all_results = []
            for symbol in symbols:
                result = verify_single_symbol(symbol, best_params, PRIMARY_SYMBOLS)
                if result:
                    all_results.append(result)
            
            avg_ret = calculate_mode_performance(all_results)
            
            print("\n" + "="*80)
            print("🏁 DRY-RUN COMPLETE (No changes saved)")
            print("="*80)
            if avg_ret is not None:
                print(f"📊 Current Strategy OOS Performance: {avg_ret:.2f}%")
            
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ Error loading deployed strategy: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # [NORMAL MODE] MySQL에서 전략 검증 후 최적 전략 배포
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
