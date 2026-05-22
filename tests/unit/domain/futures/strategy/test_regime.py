from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.config import RegimeConfig
from src.domain.futures.strategy.regime.provider import compute_regime_posterior


def test_regime_posterior_sum_one() -> None:
    # 50 steps, 8 assets
    np.random.seed(42)
    close = np.random.normal(100, 5, (50, 8))

    cfg = RegimeConfig()
    probs = compute_regime_posterior(close, cfg)

    # 5 columns
    assert len(probs) == 5
    for c in probs.values():
        assert len(c) == 50

    # Sum of probabilities at each step must equal 1.0
    for t in range(50):
        val = sum(probs[name][t] for name in probs)
        assert np.allclose(val, 1.0, atol=1e-6)


def test_regime_crisis_detection() -> None:
    # Build prices with massive drop and huge correlation to trigger crisis state
    t_len = 100
    n_syms = 6
    close = np.ones((t_len, n_syms)) * 100.0

    # Normal regime
    for t in range(1, 80):
        close[t] = close[t - 1] + np.random.normal(0, 0.1, n_syms)

    # Shock regime: massive drop (drawdown) and high correlation
    for t in range(80, 100):
        close[t] = close[t - 1] - 10.0  # huge down trend & drop

    cfg = RegimeConfig(vol_window=5, dd_crisis_thr=-0.05, corr_crisis_thr=0.7)
    probs = compute_regime_posterior(close, cfg)

    # Crisis prob at shock period (near end) should be significantly higher than normal period
    normal_crisis = np.mean(probs["hmm_prob_crisis"][:50])
    shock_crisis = np.mean(probs["hmm_prob_crisis"][90:])

    print(f"\n[CRISIS TEST] normal={normal_crisis:.4f}, shock={shock_crisis:.4f}")
    assert shock_crisis > normal_crisis
    assert shock_crisis > 0.5  # Should converge towards 1.0


def test_regime_no_lookahead() -> None:
    np.random.seed(42)
    close = np.random.normal(100, 5, (40, 6))

    cfg = RegimeConfig()
    probs1 = compute_regime_posterior(close, cfg)

    # Shock at the end (t=39)
    close_mod = close.copy()
    close_mod[-1] += 50.0

    probs2 = compute_regime_posterior(close_mod, cfg)

    # For t < 39, the probabilities must be exactly identical
    for name in probs1:
        assert np.allclose(probs1[name][:-1], probs2[name][:-1], atol=1e-12)
