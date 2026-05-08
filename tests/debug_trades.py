import numpy as np
import pandas as pd
from src.domain.futures.backtest_engine import backtest_loop_multi_numba as backtest_portfolio_numba
from config.settings import FUTURES_DATA_DIR

def run_test():
    files = list(FUTURES_DATA_DIR.glob('*_1h.parquet'))
    if not files:
        print('No data')
        return
        
    df = None
    for f in files:
        tmp = pd.read_parquet(f)
        if len(tmp) > 2000:
            df = tmp
            break
            
    if df is None: return
    n_bars = 2000
    n_syms = 1
    df = df.iloc[-n_bars:].copy()
    
    close_2d = df['close'].to_numpy(dtype=np.float64).reshape(-1, 1)
    high_2d = df['high'].to_numpy(dtype=np.float64).reshape(-1, 1)
    low_2d = df['low'].to_numpy(dtype=np.float64).reshape(-1, 1)
    open_2d = df['open'].to_numpy(dtype=np.float64).reshape(-1, 1)
    
    if 'atr' not in df.columns:
        df['atr'] = df['close'] * 0.01
    atr_2d = df['atr'].to_numpy(dtype=np.float64).reshape(-1, 1)
    
    entry_upper = np.zeros((n_bars, n_syms))
    entry_lower = np.ones((n_bars, n_syms)) * 999999.0
    
    # 숏 테스트
    trend_dir = np.ones((n_bars, n_syms)) * -1.0
    strength_filter_raw = np.ones((n_bars, n_syms))
    garch_kelly_f = np.ones((n_bars, n_syms)) * 0.01 # 강제 dust 발생 유도
    kill_signal = np.zeros((n_bars, n_syms))
    funding_rate = np.zeros((n_bars, n_syms))
    slot_rank_score = np.ones((n_bars, n_syms)) * 0.8
    
    xs_long = np.zeros((n_bars, n_syms))
    xs_short = np.ones((n_bars, n_syms)) * -0.8
    hmm_crisis = np.zeros((n_bars, n_syms))
    hmm_mod_long = np.ones((n_bars, n_syms))
    hmm_mod_short = np.ones((n_bars, n_syms))
    lev_2d = np.ones((n_bars, n_syms)) * 5.0
    
    initial_balance = 1000.0 # 1000불 소액 계좌 테스트

    trades, bal, eq, diag = backtest_portfolio_numba(
        close_2d, high_2d, low_2d, open_2d, entry_upper, entry_lower, trend_dir,
        strength_filter_raw, atr_2d, garch_kelly_f, kill_signal, funding_rate,
        slot_rank_score, xs_long, xs_short, hmm_crisis, hmm_mod_long, hmm_mod_short,
        initial_balance, lev_2d, 0.0004, 0.001, 0.02,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        max_concurrent=1, max_exposure=0.8, max_exp_per_coin=1.5,
        dd_scaling_threshold=0.15, k_long=1, k_short=1, rebalance_bars=6,
        crisis_gamma=1.0, use_cs_rank=1
    )
    
    print('Short Trades:', len(trades))
    print('Diag:', diag)
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades, columns=['sym', 'ent', 'ex', 'side', 'ep', 'xp', 'pnl', 'amt', 'efee', 'ffee'])
        print('Total PnL:', trades_df['pnl'].sum())
        print('First Trade Amount (Qty) * Entry Price:', trades_df['amt'].iloc[0] * trades_df['ep'].iloc[0])

run_test()
