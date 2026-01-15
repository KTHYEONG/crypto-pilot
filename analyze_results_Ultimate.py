import optuna
import os
import pandas as pd
import sqlite3

def analyze_ultimate_study():
    """Universal Strategy (BTC+ETH Multi-Symbol) Results Analysis"""
    db_path = "optimize_Ultimate_Universal.db"
    
    if not os.path.exists(db_path):
        print(f"❌ {db_path} not found. Run 'run_optimization_ultimate.py' first.")
        return
    
    storage_name = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name="optimize_Ultimate_Universal", storage=storage_name)
    except Exception as e:
        print(f"Error loading study: {e}")
        return
        
    print("="*70)
    print("  🌌 UNIVERSAL STRATEGY DISCOVERY ANALYSIS 🌌")
    print("  Multi-Symbol Optimization (BTC + ETH)")
    print("="*70)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Total Trials: {len(study.trials)}")
    print(f"Completed: {len(completed_trials)}")
    
    if not completed_trials:
        print("No completed trials yet.")
        return

    best = study.best_trial
    p = best.params
    attrs = best.user_attrs
    
    print(f"\n{'='*70}")
    print(f"🏆 THE UNIVERSAL STRATEGY (Best Score)")
    print(f"{'='*70}")
    print(f"Score (Harmonic Mean): {best.value:.4f}")
    print(f"\n📊 AVERAGE PERFORMANCE:")
    print(f"  Return (Avg)  : {attrs.get('return_avg', 0):.2f}%")
    print(f"  MDD (Avg)     : {attrs.get('mdd_avg', 0):.2f}%")
    print(f"  Win Rate (Avg): {attrs.get('win_rate_avg', 0):.2f}%")
    print(f"  Trades (Avg)  : {attrs.get('trades_avg', 0):.0f}")
    
    # Individual Symbol Performance
    print(f"\n📈 INDIVIDUAL SYMBOL PERFORMANCE:")
    btc_return = attrs.get('return_BTC_USDT', attrs.get('return', 0))
    btc_mdd = attrs.get('mdd_BTC_USDT', attrs.get('mdd', 0))
    btc_trades = attrs.get('trades_BTC_USDT', attrs.get('trades', 0))
    btc_winrate = attrs.get('winrate_BTC_USDT', attrs.get('win_rate', 0))
    
    eth_return = attrs.get('return_ETH_USDT', 0)
    eth_mdd = attrs.get('mdd_ETH_USDT', 0)
    eth_trades = attrs.get('trades_ETH_USDT', 0)
    eth_winrate = attrs.get('winrate_ETH_USDT', 0)
    
    print(f"  🟠 BTC/USDT:")
    print(f"     Return: {btc_return:.2f}% | MDD: {btc_mdd:.2f}% | Trades: {btc_trades} | Win Rate: {btc_winrate:.2f}%")
    print(f"  🔵 ETH/USDT:")
    print(f"     Return: {eth_return:.2f}% | MDD: {eth_mdd:.2f}% | Trades: {eth_trades} | Win Rate: {eth_winrate:.2f}%")
    
    print(f"\n⚙️  CONFIG:")
    print(f"  Timeframe: {attrs.get('timeframe', 'N/A')}")
    print(f"  Leverage: {attrs.get('leverage', 1)}x")
    
    print(f"\n✨ WINNING COMBINATION:")
    print(f"  🚦 Entry Signal      : {p.get('ENTRY_TYPE')} (Period: {p.get('ENTRY_PERIOD')})") 
    if p.get('ENTRY_TYPE') == 'BOLLINGER':
        print(f"     └─ Std Dev       : {p.get('BB_STD')}")
        
    print(f"  🌊 Trend Filter      : {p.get('TREND_FILTER_TYPE')} (Period: {p.get('MA_PERIOD')})") 
    if p.get('TREND_FILTER_TYPE') == 'SUPERTREND':
        print(f"     └─ Multiplier    : {p.get('SUPERTREND_MULT')}")
        print(f"     └─ Period        : {p.get('SUPERTREND_PERIOD')}")
    elif p.get('TREND_FILTER_TYPE') in ['MACD', 'ICHIMOKU']:
        print(f"     └─ Type          : {p.get('TREND_FILTER_TYPE')}")
        
    print(f"  💪 Strength Filters  : ")
    print(f"     ├─ ADX           : {'ON' if p.get('USE_ADX') else 'OFF'} (Thresh: {p.get('ADX_THRESHOLD')})")
    print(f"     ├─ VHF           : {'ON' if p.get('USE_VHF') else 'OFF'} (Thresh: {p.get('VHF_THRESHOLD')})")
    print(f"     ├─ MFI           : {'ON' if p.get('USE_MFI') else 'OFF'} (Thresh: {p.get('MFI_THRESHOLD', 'N/A')})")
    print(f"     ├─ RSI           : {'ON' if p.get('USE_RSI') else 'OFF'}")
    print(f"     └─ Stochastic    : {'ON' if p.get('USE_STOCHASTIC') else 'OFF'}")
    
    print(f"  🛡️ Exit Logic        : {p.get('EXIT_TYPE')}")
    if p.get('EXIT_TYPE') == 'ATR':
        print(f"     └─ ATR Multiplier: {p.get('ATR_MULTIPLIER')}")
    elif p.get('EXIT_TYPE') == 'PARABOLIC_SAR':
        print(f"     └─ SAR Step      : {p.get('SAR_STEP')}")
    
    print(f"\n🔍 All Parameters:")
    for k, v in sorted(p.items()):
        print(f"  {k:25s}: {v}")
    
    print(f"\n{'='*70}")
    print(f"📊 Top 5 Universal Strategies")
    print(f"{'='*70}")
    
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
    for i, t in enumerate(sorted_trials, 1):
        tp = t.params
        ta = t.user_attrs
        avg_ret = ta.get('return_avg', 0)
        avg_mdd = ta.get('mdd_avg', 0)
        combo = f"{tp.get('ENTRY_TYPE')[:3]} + {tp.get('TREND_FILTER_TYPE')} + {tp.get('EXIT_TYPE')}"
        print(f"{i}. Score {t.value:.2f} | Avg Ret {avg_ret:.2f}% | Avg MDD {avg_mdd:.2f}% | {combo}")

    # Stat Analysis
    print(f"\n{'='*70}")
    print(f"📈 Component Popularity in Top 20")
    print(f"{'='*70}")
    top_20 = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:20]
    entry_types = [t.params.get('ENTRY_TYPE') for t in top_20]
    trend_types = [t.params.get('TREND_FILTER_TYPE') for t in top_20]
    exit_types = [t.params.get('EXIT_TYPE') for t in top_20]
    
    from collections import Counter
    print(f"Entries : {Counter(entry_types)}")
    print(f"Trends  : {Counter(trend_types)}")
    print(f"Exits   : {Counter(exit_types)}")

if __name__ == "__main__":
    analyze_ultimate_study()
