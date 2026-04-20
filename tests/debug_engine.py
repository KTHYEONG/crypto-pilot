import numpy as np
import pandas as pd
from src.domain.futures.engine_multi_futures import backtest_portfolio_numba

n_bars = 1000
n_syms = 1

close_2d = np.ones((n_bars, n_syms)) * 100.0
close_2d[:, 0] = np.linspace(100.0, 50.0, n_bars)
high_2d = close_2d + 1.0
low_2d = close_2d - 1.0
open_2d = close_2d + 0.5

entry_upper = np.zeros((n_bars, n_syms))
entry_lower = np.ones((n_bars, n_syms)) * 999999.0
trend_dir = np.ones((n_bars, n_syms))
strength_filter_raw = np.ones((n_bars, n_syms))
atr_2d = np.ones((n_bars, n_syms)) * 2.0
garch_kelly_f = np.ones((n_bars, n_syms))
kill_signal = np.zeros((n_bars, n_syms))
funding_rate = np.zeros((n_bars, n_syms))
slot_rank_score = np.ones((n_bars, n_syms))
xs_long = np.ones((n_bars, n_syms)) * 0.8
xs_short = np.zeros((n_bars, n_syms))
hmm_crisis = np.zeros((n_bars, n_syms))
hmm_mod_long = np.ones((n_bars, n_syms))
hmm_mod_short = np.ones((n_bars, n_syms))
lev_2d = np.ones((n_bars, n_syms)) * 5.0

initial_balance = 10000.0

trades, bal, eq, diag = backtest_portfolio_numba(
    close_2d, high_2d, low_2d, open_2d, entry_upper, entry_lower, trend_dir,
    strength_filter_raw, atr_2d, garch_kelly_f, kill_signal, funding_rate,
    slot_rank_score, xs_long, xs_short, hmm_crisis, hmm_mod_long, hmm_mod_short,
    initial_balance, lev_2d, 0.0004, 0.001, 0.02,
    3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
    max_concurrent=2, max_exposure=0.8, max_exp_per_coin=1.5,
    dd_scaling_threshold=0.15, k_long=1, k_short=1, rebalance_bars=6,
    crisis_gamma=1.0, use_cs_rank=1
)

print('Trades length:', len(trades))
if len(trades) > 0:
    print('First trade:', trades[0])
print('Final Balance:', bal)
print('Diag:', diag)
