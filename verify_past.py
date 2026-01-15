import argparse
import pandas as pd
from config.settings import DATA_DIR
from src.data.collector import DataCollector
from src.strategy.strategies import UltimateStrategy
from src.backtest.engine_fast import BacktestEngineFast

def run_past_verification(params, symbols=None):
    """
    Run Out-of-Sample verification for multiple symbols
    symbols: list of symbols to test (default: ["BTC/USDT", "ETH/USDT"])
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT"]
    
    # 2025년 검증 (Out-of-Sample Test)
    start_date = "2025-01-01"
    end_date = "2025-12-31"
    
    print(f"\n{'='*70}")
    print(f"  🔬 OUT-OF-SAMPLE VERIFICATION (2025)")
    print(f"  Testing strategy on UNSEEN data: {start_date} ~ {end_date}")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"{'='*70}\n")
    
    collector = DataCollector()
    timeframe = params.get('TIMEFRAME', '2h')
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'─'*70}")
        print(f"📊 Testing {symbol}...")
        print(f"{'─'*70}")
        
        # 1. 데이터 로드
        daily_df = collector.collect_and_save(symbol, '1d', start_date, end_date)
        hourly_df = collector.collect_and_save(symbol, timeframe, start_date, end_date)
        
        daily_df['datetime'] = pd.to_datetime(daily_df['timestamp'], unit='ms')
        hourly_df['datetime'] = pd.to_datetime(hourly_df['timestamp'], unit='ms')

        # 2. 전략 및 엔진 설정
        strategy = UltimateStrategy(f"Verification_{symbol}", params)
        engine = BacktestEngineFast(hourly_df, daily_df, strategy, initial_balance=500_000)
        
        # 레버리지 및 리스크 주입
        engine.leverage = params.get('LEVERAGE', 1.0)
        engine.risk_per_trade = params.get('RISK_PER_TRADE', 0.02)
        
        # 3. 백테스트 실행
        result = engine.run()
        all_results[symbol] = result
        
        # 4. 개별 결과 출력
        print(f"\n[{symbol} 결과]")
        print(f"  초기 자본         : {engine.initial_balance:,.0f}원")
        print(f"  최종 자산         : {result['final_balance']:,.0f}원")
        print(f"  수익금            : {result['final_balance'] - engine.initial_balance:+,.0f}원")
        print(f"  수익률 (Return)   : {result['total_return_pct']:.2f}%")
        print(f"  MDD               : {result['mdd_pct']:.2f}%")
        print(f"  거래 횟수         : {result['total_trades']}")
        print(f"  승률 (Win Rate)   : {result['win_rate']:.2f}%")
        
        # 거래 내역 상세 출력
        if result['total_trades'] > 0 and 'trades_df' in result:
            trades_df = result['trades_df']
            print(f"\n📋 거래 내역 (총 {len(trades_df)}건):")
            print(f"{'─'*110}")
            print(f"{'No':>3} {'진입시간':^19} {'청산시간':^19} {'방향':^5} {'진입가':>12} {'청산가':>12} {'손익':>15} {'잔고':>15}")
            print(f"{'─'*110}")
            
            cumulative_balance = engine.initial_balance
            for idx, trade in trades_df.iterrows():
                cumulative_balance += trade['pnl']
                print(f"{idx+1:3d} {str(trade['entry_time']):19s} {str(trade['exit_time']):19s} "
                      f"{trade['side']:^5s} {trade['entry_price']:>12.2f} {trade['exit_price']:>12.2f} "
                      f"{trade['pnl']:>+15,.0f} {cumulative_balance:>15,.0f}")
            print(f"{'─'*110}")
    
    # 5. 종합 결과 출력
    print(f"\n{'='*70}")
    print(f"📈 VERIFICATION SUMMARY")
    print(f"{'='*70}")
    
    avg_return = sum(r['total_return_pct'] for r in all_results.values()) / len(all_results)
    avg_mdd = sum(r['mdd_pct'] for r in all_results.values()) / len(all_results)
    avg_trades = sum(r['total_trades'] for r in all_results.values()) / len(all_results)
    avg_winrate = sum(r['win_rate'] for r in all_results.values()) / len(all_results)
    
    print(f"  평균 수익률     : {avg_return:.2f}%")
    print(f"  평균 MDD        : {avg_mdd:.2f}%")
    print(f"  평균 거래 횟수  : {avg_trades:.1f}")
    print(f"  평균 승률       : {avg_winrate:.2f}%")
    
    # 성공 판정
    success = True
    for symbol, result in all_results.items():
        if result['total_return_pct'] < 0:
            success = False
            print(f"\n  ⚠️  {symbol}: 마이너스 수익률 ({result['total_return_pct']:.2f}%)")
        if result['mdd_pct'] < -40:
            success = False
            print(f"\n  ⚠️  {symbol}: MDD 과다 ({result['mdd_pct']:.2f}%)")
    
    if success:
        print(f"\n  ✅ 검증 통과! 모든 종목에서 양호한 성과")
    else:
        print(f"\n  ❌ 검증 실패! 일부 종목에서 부진")
    
    print(f"{'='*70}\n")


# --- 전략 프리셋 저장소 ---
STRATEGY_PRESETS = {
    "New_Master_15m": {
        'TIMEFRAME': '15m',
        'LEVERAGE': 2.0,
        'CHANNEL_PERIOD': 125,
        'REGIME_FILTER': 'EMA',
        'TREND_EMA_WINDOW': 110,
        'HMA_WINDOW': 180, # 사용 안함 (EMA 모드)
        'USE_ADX': False,
        'ADX_THRESHOLD': 26, # 사용 안함
        'STOP_LOSS_PCT': 0.035,
        'ATR_MULTIPLIER': 2.0,
        'RISK_PER_TRADE': 0.03
    },
    "Grand_Finale_2h": {
        'TIMEFRAME': '2h',
        'LEVERAGE': 1.6,
        'CHANNEL_PERIOD': 75,
        'REGIME_FILTER': 'EMA',
        'TREND_EMA_WINDOW': 50,
        'USE_ADX': True,
        'ADX_THRESHOLD': 20,
        'STOP_LOSS_PCT': 0.015,
        'ATR_MULTIPLIER': 5.5,
        'RISK_PER_TRADE': 0.01
    },
    "Legendary_Safe_1h": {
        'TIMEFRAME': '1h',
        'LEVERAGE': 1.0,
        'CHANNEL_PERIOD': 95,
        'REGIME_FILTER': 'EMA',
        'TREND_EMA_WINDOW': 160,
        'USE_ADX': False, # 당시엔 없었음
        'STOP_LOSS_PCT': 0.04,
        'ATR_MULTIPLIER': 2.5,
        'RISK_PER_TRADE': 0.03
    },
    "Latest_1h": {
        'TIMEFRAME': '1h',
        'LEVERAGE': 1.1,
        'CHANNEL_PERIOD': 45,
        'REGIME_FILTER': 'HMA',
        'TREND_EMA_WINDOW': 170,
        'HMA_WINDOW': 70,
        'USE_ADX': True,
        'ADX_THRESHOLD': 24,    
        'STOP_LOSS_PCT': 0.025,
        'ATR_MULTIPLIER': 5.0,
        'RISK_PER_TRADE': 0.01
    },
    "ultimate" : {
        'ADX_THRESHOLD': 26,
        'ATR_MULTIPLIER': 2.4,
        'ATR_STOP_LOSS_MULT': 3.0,
        'BB_STD': 1.5,
        'ENTRY_PERIOD': 137,
        'ENTRY_TYPE': 'BOLLINGER',
        'EXIT_TYPE': 'ATR',
        'LEVERAGE': 2.25,
        'MA_PERIOD': 122,
        'MFI_THRESHOLD': 16,
        'MFI_WINDOW': 10,
        'RISK_PER_TRADE': 0.029,
        'RSI_OVERBOUGHT': 66,
        'RSI_OVERSOLD': 34,
        'RSI_WINDOW': 13,
        'SAR_STEP': 0.012,
        'STOCH_OVERBOUGHT': 83,
        'STOCH_OVERSOLD': 10,
        'STOCH_WINDOW': 11,
        'STOP_LOSS_PCT': 0.041,
        'STOP_LOSS_TYPE': 'ATR',
        'SUPERTREND_MULT': 2.8,
        'SUPERTREND_PERIOD': 34,
        'TIMEFRAME': '3m',
        'TREND_FILTER_TYPE': 'SUPERTREND',
        'USE_ADX': False,
        'USE_MFI': False,
        'USE_RSI': False,
        'USE_STOCHASTIC': False,
        'USE_VHF': False,
        'VHF_THRESHOLD': 0.41
    }
}

if __name__ == "__main__":
    # --------------------------------------------------------
    # [사용자 설정] 테스트할 전략을 선택하세요
    # 선택지: "New_Master_15m", "Grand_Finale_2h", "Legendary_Safe_1h", "Latest_1h", "ultimate"
    SELECTED_PRESET = "ultimate"
    # --------------------------------------------------------
    
    if SELECTED_PRESET not in STRATEGY_PRESETS:
        print(f"❌ Error: '{SELECTED_PRESET}' is not in presets.")
        print(f"Available: {list(STRATEGY_PRESETS.keys())}")
    else:
        print(f"🚀 Verifying Preset: {SELECTED_PRESET}")
        best_params = STRATEGY_PRESETS[SELECTED_PRESET]
        run_past_verification(best_params)
