from __future__ import annotations

import numpy as np

from src.domain.futures.legacy.strategy_sleev.normalize import winsorized_cs_zscore
from src.domain.futures.legacy.strategy_sleev.sleeves.xs_reversal import XSReversalSleeve
from src.domain.futures.strategy.diagnostics import ic_summary, rolling_ic


def test_sleeve_reversal_ic_and_lookahead() -> None:
    """Tests XSReversalSleeve edge, look-ahead, and turnover on mean-reverting data."""
    # 1. Synthesize 300 steps and 10 assets with strong mean-reversion
    # Simple AR(1) process with negative phi: x_t = -0.5 * x_{t-1} + noise
    np.random.seed(42)
    t_len = 300
    n_syms = 10
    prices = np.zeros((t_len, n_syms))
    prices[0] = 100.0

    for t in range(1, t_len):
        # Generate mean-reverting changes relative to a baseline
        prev_log = np.log(prices[t - 1])
        # Force reversion to log(100.0)
        reversion = -0.4 * (prev_log - np.log(100.0))
        noise = np.random.normal(0, 0.02, n_syms)
        prices[t] = np.exp(prev_log + reversion + noise)

    # 2. Compute XSReversalSleeve signals
    sleeve = XSReversalSleeve(lookback_bars=1)
    raw_sig = sleeve.compute_raw(prices, aux={})
    z_sig = winsorized_cs_zscore(raw_sig, min_symbols=5)

    # 3. Calculate forward 1-bar returns
    # fwd_ret[t] = close[t+1]/close[t] - 1
    fwd_ret = np.full((t_len, n_syms), np.nan, dtype=np.float64)
    fwd_ret[:-1] = prices[1:] / prices[:-1] - 1.0

    # 4. Chronological rolling IC
    ic = rolling_ic(z_sig, fwd_ret, method="spearman")
    summary = ic_summary(ic)

    # Reversal signal should yield POSITIVE IC on mean-reverting data
    # (Since signal is negative of recent return, and return is about to reverse)
    print(
        f"\n[REVERSAL IC SUMMARY] mean_ic={summary['mean_ic']:.4f}, t_stat={summary['t_stat']:.4f}"
    )
    assert summary["mean_ic"] > 0.02, (
        f"Mean IC must pass gate (>0.02), got {summary['mean_ic']:.4f}"
    )
    assert summary["t_stat"] > 2.0, f"T-stat must pass gate (>2.0), got {summary['t_stat']:.4f}"

    # 5. Look-ahead leak detection
    # If look-ahead is absent, the correlation of sig[t] with fwd_ret[t]
    # (close[t+1]/close[t] - 1) should be high due to mean-reversion.
    # The correlation of sig[t] with hist_ret[t] (close[t]/close[t-1] - 1)
    # should not leak future information.
    # More formally, let's verify that future data changes (e.g. at t+2) do not impact sig[t].
    prices_mod = prices.copy()
    prices_mod[150:] += 10.0  # Shock the future

    raw_sig_mod = sleeve.compute_raw(prices_mod, aux={})
    # The signal at t < 150 must remain exactly identical
    assert np.allclose(raw_sig[:149], raw_sig_mod[:149], equal_nan=True)

    # 6. Turnover analysis
    # Turnover measures average absolute change in standardized scores
    # TO_t = mean(|z_t - z_{t-1}|)
    valid_z = z_sig[~np.isnan(z_sig).any(axis=1)]
    if len(valid_z) > 1:
        diff = np.abs(valid_z[1:] - valid_z[:-1])
        avg_turnover = float(np.mean(diff))
        print(f"[REVERSAL TURNOVER] avg_turnover={avg_turnover:.4f}")
        # Turnover should be bounded and non-zero
        assert 0.0 < avg_turnover < 4.0
