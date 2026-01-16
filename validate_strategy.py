
import argparse
import pandas as pd
import sys
import os
import logging
import json

# 프로젝트 루트 경로 추가 (모듈 임포트 문제 방지)
sys.path.append(os.getcwd())

from config.settings import DATA_DIR, BACKTEST_START_DATE, BACKTEST_END_DATE
from src.data.collector import DataCollector
from src.strategy.strategies import UltimateStrategy
from src.backtest.engine_fast import BacktestEngineFast
from src.validation.evaluator import StrategyEvaluator
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Validator")

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
    
    return hourly_df, daily_df

def run_validation(params, symbol):
    logger.info(f"🚀 Starting Validation for {symbol}")
    
    # 1. 데이터 로드
    timeframe = params.get('TIMEFRAME', '1h')
    try:
        hourly_df, daily_df = load_data(symbol, BACKTEST_START_DATE, BACKTEST_END_DATE, timeframe)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 2. 초기 백테스트 실행 (Baseline)
    strategy = UltimateStrategy("Validation_Base", params)
    engine = BacktestEngineFast(hourly_df, daily_df, strategy, initial_balance=1_000_000)
    engine.leverage = params.get('LEVERAGE', 1)
    engine.risk_per_trade = params.get('RISK_PER_TRADE', 0.02)
    
    initial_result = engine.run()
    
    logger.info(f"📊 Baseline Return: {initial_result['total_return_pct']:.2f}% | MDD: {initial_result['mdd_pct']:.2f}%")
    
    # 3. 종합 검증기 실행
    evaluator = StrategyEvaluator()
    report = evaluator.evaluate(hourly_df, daily_df, UltimateStrategy, "Validation", params, {}, initial_result)
    
    # 4. 결과 출력
    print(f"\n{'='*60}")
    print(f"✅ VALIDATION REPORT: {symbol}")
    print(f"{'='*60}")
    
    print(f"📡 Status: {report['status']}")
    
    if report['status'] == 'FAILED':
        print(f"❌ Reasons:")
        for reason in report['reason']:
            print(f"  - {reason}")
    else:
        print(f"🎉 Strategy Passed all validation checks!")
        
    print(f"\n📊 Key Metrics:")
    print(f"  - Sharpe Ratio : {report['metrics'].get('sharpe', 0):.2f}")
    print(f"  - Sortino Ratio: {report['metrics'].get('sortino', 0):.2f}")
    print(f"  - p-value      : {report['metrics'].get('p_value', 1.0):.4f}")
    
    print(f"\n🔄 Walk-Forward Analysis:")
    print(f"  - Robustness Score: {report['wfa'].get('score', 0):.2f}")
    
    print(f"\n🎲 Monte Carlo Simulation:")
    print(f"  - Prob. of Profit: {report['monte_carlo'].get('prob_profit', 0):.1f}%")
    print(f"  - Worst Case MDD : {report['monte_carlo'].get('worst_case_mdd', 0):.1f}%")
    
    print(f"{'='*60}\n")
    
    return report

# --- PRESETS (For testing) ---
PRESETS = {
    "ultimate_v1": {
        "ADX_THRESHOLD": 26,
        "ATR_MULTIPLIER": 2.4,
        "ATR_STOP_LOSS_MULT": 3.0,
        "BB_STD": 1.5,
        "ENTRY_PERIOD": 137,
        "ENTRY_TYPE": "BOLLINGER",
        "EXIT_TYPE": "ATR",
        "LEVERAGE": 2.25,
        "MA_PERIOD": 122,
        "MFI_THRESHOLD": 16,
        "MFI_WINDOW": 10,
        "RISK_PER_TRADE": 0.028999999999999998,
        "RSI_OVERBOUGHT": 66,
        "RSI_OVERSOLD": 34,
        "RSI_WINDOW": 13,
        "SAR_STEP": 0.012,
        "STOCH_OVERBOUGHT": 83,
        "STOCH_OVERSOLD": 10,
        "STOCH_WINDOW": 11,
        "STOP_LOSS_PCT": 0.041,
        "STOP_LOSS_TYPE": "ATR",
        "SUPERTREND_MULT": 2.8,
        "SUPERTREND_PERIOD": 34,
        "TIMEFRAME": "3m",
        "TREND_FILTER_TYPE": "SUPERTREND",
        "USE_ADX": False,
        "USE_MFI": False,
        "USE_RSI": False,
        "USE_STOCHASTIC": False,
        "USE_VHF": False,
        "VHF_THRESHOLD": 0.41
    },
    "ultimate_v2" : {
        "ADX_THRESHOLD": 30,
        "ATR_MULTIPLIER": 2.4,
        "ATR_STOP_LOSS_MULT": 1.8,
        "BB_STD": 1.6,
        "ENTRY_PERIOD": 31,
        "ENTRY_TYPE": "BOLLINGER",
        "EXIT_TYPE": "PARABOLIC_SAR",
        "LEVERAGE": 1.3,
        "MA_PERIOD": 105,
        "MFI_THRESHOLD": 32,
        "MFI_WINDOW": 20,
        "RISK_PER_TRADE": 0.024, # Safe Level (Validated)
        "RSI_OVERBOUGHT": 75,
        "RSI_OVERSOLD": 20,
        "RSI_WINDOW": 20,
        "SAR_STEP": 0.045,
        "STOCH_OVERBOUGHT": 82,
        "STOCH_OVERSOLD": 22,
        "STOCH_WINDOW": 11,
        "STOP_LOSS_PCT": 0.026,
        "STOP_LOSS_TYPE": "ATR",
        "SUPERTREND_MULT": 4.9,
        "SUPERTREND_PERIOD": 44,
        "TAKE_PROFIT_ATR_MULT": 2.5,
        "TIMEFRAME": "5m",
        "TREND_FILTER_TYPE": "SUPERTREND",
        "USE_ADX": False,
        "USE_MFI": True,
        "USE_RSI": False,
        "USE_STOCHASTIC": False,
        "USE_TAKE_PROFIT": False,
        "USE_VHF": False,
        "USE_VOLUME_FILTER": False,
        "VHF_THRESHOLD": 0.52,
        "VOLUME_MA_PERIOD": 30,
        "VOLUME_THRESHOLD_MULT": 1.2
    },
    "eth" : {
        "ADX_THRESHOLD": 10,
        "ATR_MULTIPLIER": 2.1,
        "ATR_STOP_LOSS_MULT": 2.5,
        "BB_STD": 1.7,
        "ENTRY_PERIOD": 91,
        "ENTRY_TYPE": "CCI",
        "EXIT_TYPE": "PARABOLIC_SAR",
        "LEVERAGE": 2.5000000000000004,
        "MA_PERIOD": 60,
        "MFI_THRESHOLD": 34,
        "MFI_WINDOW": 11,
        "RISK_PER_TRADE": 0.035500000000000004,
        "RSI_OVERBOUGHT": 65,
        "RSI_OVERSOLD": 23,
        "RSI_WINDOW": 20,
        "SAR_STEP": 0.019000000000000003,
        "STOCH_OVERBOUGHT": 78,
        "STOCH_OVERSOLD": 20,
        "STOCH_WINDOW": 12,
        "STOP_LOSS_PCT": 0.011,
        "STOP_LOSS_TYPE": "ATR",
        "SUPERTREND_MULT": 2.9000000000000004,
        "SUPERTREND_PERIOD": 30,
        "TAKE_PROFIT_ATR_MULT": 7.0,
        "TIMEFRAME": "5m",
        "TREND_FILTER_TYPE": "EMA",
        "USE_ADX": True,
        "USE_MFI": True,
        "USE_RSI": False,
        "USE_STOCHASTIC": False,
        "USE_TAKE_PROFIT": False,
        "USE_VHF": True,
        "USE_VOLUME_FILTER": False,
        "VHF_THRESHOLD": 0.33999999999999997,
        "VOLUME_MA_PERIOD": 50,
        "VOLUME_THRESHOLD_MULT": 1.6
    }
    
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [UPDATE] Multiple symbols support
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT", help="Comma separated symbols")
    parser.add_argument("--preset", type=str, default="ultimate_v2", help="Preset name")
    parser.add_argument("--params", type=str, help="JSON string of params")
    
    args = parser.parse_args()
    
    params = PRESETS.get(args.preset)
    if args.params:
         params = json.loads(args.params)
         
    if not params:
         print(f"❌ Preset {args.preset} not found.")
         sys.exit(1)
         
    # [UPDATE] Loop through symbols and collect reports
    symbols = [s.strip() for s in args.symbols.split(",")]
    all_reports = {}
    
    print(f"\n🚀 Starting Multi-Symbol Validation: {', '.join(symbols)}")
    print(f"{'='*70}\n")
    
    for symbol in symbols:
        print(f"🔹 Using Preset: {args.preset} for {symbol}")
        report = run_validation(params, symbol)
        all_reports[symbol] = report

    # [UPDATE] Final Summary Header
    print("\n" + "="*70)
    print("🏁 FINAL CONSOLIDATED VALIDATION REPORT")
    print("="*70)
    
    for symbol, report in all_reports.items():
        status_icon = "✅" if report['status'] == 'PASSED' else "❌"
        ret = report['metrics'].get('total_return_pct', 0)
        mdd = report['metrics'].get('mdd_pct', 0)
        sharpe = report['metrics'].get('sharpe', 0)
        p_val = report['metrics'].get('p_value', 1.0)
        
        print(f"{status_icon} {symbol:10s} | Status: {report['status']:7s} | Return: {ret:6.1f}% | MDD: {mdd:5.1f}% | Sharpe: {sharpe:4.2f} | p-value: {p_val:.4f}")
    
    print("="*70 + "\n")
