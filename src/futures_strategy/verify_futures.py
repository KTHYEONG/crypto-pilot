
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

from config.settings import (
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
    TRAIN_CUTOFF_DATE,
    FUTURES_INITIAL_BALANCE,
)
from src.data.collector import DataCollector
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.futures_strategy.backtest_utils_futures import run_backtest_segment_futures, prepare_futures_data

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FuturesVerifier")
# [LOGGING] 엔진 로그 레벨 상향 (INFO -> WARNING)하여 불필요한 출력 억제
logging.getLogger("src.futures_strategy.engine_fast_futures").setLevel(logging.WARNING)

def load_data(symbol, start_date, end_date, timeframe):
    """Load Data Helper"""
    collector = DataCollector()

    # Daily Data (Parquet range cache + incremental fetch)
    daily_df = collector.ensure_data(symbol, "1d", start_date, end_date)
    daily_df['datetime'] = pd.to_datetime(daily_df['timestamp'], unit='ms')

    # Timeframe Data (Parquet range cache + incremental fetch)
    hourly_df = collector.ensure_data(symbol, timeframe, start_date, end_date)
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
    
    initial_balance = FUTURES_INITIAL_BALANCE
    
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
    
def load_best_params_from_mysql(mode, storage_url):
    """MySQL에서 최적 파라미터 로드"""
    study_name = f"futures_{mode.lower()}_strategy"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        return study_name, study.best_params, study.best_value
    except KeyError:
        return None, None, None

def verify_single_symbol_futures(symbol, best_params, primary_symbols):
    """단일 심볼 백테스트 실행 (Futures) + 상세 분석 (모든 심볼)"""
    try:
        tf = best_params.get('TIMEFRAME', '1h')
        hourly_df, daily_df = load_data(symbol, BACKTEST_START_DATE, BACKTEST_END_DATE, tf)
    except Exception as e:
        print(f"   ❌ Data load error for {symbol}: {e}")
        return None
    
    # [FIX] OOS 슬라이싱 with Warmup Buffer
    # Daily indicator shift(1)로 인한 NaN 방지를 위해 앞에 warmup 기간 추가
    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)
    
    # Warmup: Daily indicator 계산을 위한 최소 기간 (일별 60일 = 약 2개월 버퍼)
    WARMUP_DAYS = 60
    warmup_cutoff = cutoff_ts - pd.Timedelta(days=WARMUP_DAYS)
    
    # Warmup 기간을 포함하여 슬라이싱
    test_hourly = hourly_df[hourly_df['datetime'] >= warmup_cutoff].copy()
    test_daily = daily_df[daily_df['datetime'] >= warmup_cutoff].copy()
    
    if test_hourly.empty:
        print(f"   ⚠️ No OOS data for {symbol}. Skipping verification.")
        return None
    
    # [FIX] 최적화 엔진(BacktestEngineFast)을 직접 사용하여 검증 일관성 확보
    from src.futures_strategy.engine_fast_futures import BacktestEngineFast
    
    strategy = UltimateStrategy(f"Verify_{symbol}", best_params)
    # 전체 데이터를 넣되, 엔진 내부에서 cutoff_ts 이전은 warmup으로 처리되도록 유도하거나 
    # 실행 후 OOS 기간만 별도 집계합니다.
    engine = BacktestEngineFast(
        test_hourly, test_daily, strategy, initial_balance=FUTURES_INITIAL_BALANCE
    )
    engine.leverage = best_params.get('LEVERAGE', 1)
    engine.risk_per_trade = best_params.get('RISK_PER_TRADE', 0.02)
    
    # 실행
    res = engine.run()
    
    trades_df = res['trades_df']
    
    # [중요] OOS 기간(cutoff_ts 이후)의 거래만 필터링하여 성과 재계산
    if not trades_df.empty:
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        oos_trades = trades_df[trades_df['exit_time'] >= cutoff_ts].copy()
    else:
        oos_trades = pd.DataFrame()

    trade_count = len(oos_trades)
    if trade_count > 0:
        # OOS 수익률 계산 (OOS 기간의 PnL 합계 / 초기 자본)
        oos_pnl = oos_trades['pnl'].sum()
        ret_pct = (oos_pnl / FUTURES_INITIAL_BALANCE) * 100
        
        # OOS MDD는 간소화하여 전체 MDD 사용하거나 재계산 가능 (여기선 전체 MDD 활용)
        mdd = res['mdd_pct'] 
        
        win_trades = oos_trades[oos_trades['pnl'] > 0]
        win_rate = len(win_trades) / trade_count * 100
        
        pos_pnl = win_trades['pnl'].sum()
        neg_pnl = abs(oos_trades[oos_trades['pnl'] < 0]['pnl'].sum())
        pf = pos_pnl / neg_pnl if neg_pnl > 0 else (pos_pnl if pos_pnl > 0 else 0.0)
    else:
        ret_pct = 0.0
        mdd = 0.0
        win_rate = 0.0
        pf = 0.0

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
        'trades_log': oos_trades['pnl'].tolist() if not oos_trades.empty else []
    }
    
    # [상세 분석] 모든 심볼에 대해 WFA + MC 수행 (거래 수 10개 이상)
    if trade_count >= 10:
        from src.futures_strategy.monte_carlo_futures import FuturesMonteCarloSimulator
        trades_log = result['trades_log'] # ROI 대신 PnL 기반이나 MC 수정 필요할 수 있음
        # 기존 MC는 ROI(%)를 기대하므로 변환
        roi_log = [(pnl / FUTURES_INITIAL_BALANCE) * 100 for pnl in trades_log]

        print(f"      🔬 Running detailed analysis for {symbol}...")
        
        # Walk-Forward Analysis
        try:
            from src.futures_strategy.walk_forward_futures import FuturesWalkForwardAnalyzer
            # [FIX] WFA에 Warmup 포함 데이터 전달 (Indicator 계산 안정화)
            wfa = FuturesWalkForwardAnalyzer(test_hourly, test_daily, best_params)
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
        
        # Monte Carlo Simulation
        try:
            from src.futures_strategy.monte_carlo_futures import FuturesMonteCarloSimulator
            # roi_log는 [(pnl / FUTURES_INITIAL_BALANCE) * 100]으로 이미 계산됨
            mc = FuturesMonteCarloSimulator(roi_log)
            mc_res = mc.run(
                n_simulations=10000, initial_balance=FUTURES_INITIAL_BALANCE
            )
            
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT",
                        help="Comma-separated list of symbols to verify")
    parser.add_argument("--alt", type=int, default=0, choices=[0, 1],
                        help="Include altcoins for validation (1=yes, 0=no). Adds SOL, XRP, DOGE, BNB")
    parser.add_argument("--all-modes", action="store_true",
                        help="Verify all modes (SCALP, DAY, SWING, UNIFIED). Default: UNIFIED only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify current deployed strategy (futures_strategy.db) without saving. Skips MySQL and deployment.")
    args = parser.parse_args()
    
    # 심볼 목록 빌드
    base_symbols = [s.strip() for s in args.symbols.split(',')]
    if args.alt == 1:
        alt_symbols = ['SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'BNB/USDT']
        for alt in alt_symbols:
            if alt not in base_symbols:
                base_symbols.append(alt)
        print(f"📊 Altcoin validation enabled. Added: {', '.join(alt_symbols)}")
    
    symbols = base_symbols
    PRIMARY_SYMBOLS = ['BTC/USDT', 'ETH/USDT']
    
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
        print(f"   Source: futures_strategy.db (Local)")
        print("="*80)
        
        target_db = "futures_strategy.db"
        if not os.path.exists(target_db):
            print(f"❌ Error: {target_db} not found. Deploy a strategy first.")
            sys.exit(1)
        
        try:
            # 로컬 DB에서 현재 전략 로드
            local_storage = f"sqlite:///{target_db}"
            study = optuna.load_study(study_name="futures_strategy", storage=local_storage)
            best_params = study.best_params
            train_score = study.best_value
            
            print(f"   ✅ Loaded Current Strategy (Train Score: {train_score:.4f})")
            print(f"   Timeframe: {best_params.get('TIMEFRAME')}")
            print(f"   Leverage: {best_params.get('LEVERAGE', 1)}x")
            
            # OOS 검증
            all_results = []
            for symbol in symbols:
                result = verify_single_symbol_futures(symbol, best_params, PRIMARY_SYMBOLS)
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
    print("\n" + "="*80)
    print(f"🚀 INTEGRATED STRATEGY VERIFICATION (Futures)")
    print(f"   Searching for optimized strategies: {MODES}")
    print("="*80)
    
    results = []
    
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
            print(f"   Timeframe: {best_params.get('TIMEFRAME')}")
            
            # OOS 검증
            all_results = []
            for symbol in symbols:
                result = verify_single_symbol_futures(symbol, best_params, PRIMARY_SYMBOLS)
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
