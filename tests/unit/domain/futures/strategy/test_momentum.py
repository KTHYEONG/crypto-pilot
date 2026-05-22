from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.config import MomentumConfig
from src.domain.futures.strategy.momentum import compute_xs_momentum_alpha


def test_warmup_zero() -> None:
    close = np.tile(np.arange(1.0, 25.0).reshape(-1, 1), (1, 5))
    cfg = MomentumConfig(lookback_bars=6, min_symbols_for_xs=3)
    alpha_l, alpha_s = compute_xs_momentum_alpha(close, cfg)
    assert np.all(alpha_l[:6] == 0.0)
    assert np.all(alpha_s[:6] == 0.0)


def test_top_only_long() -> None:
    t = np.arange(30.0)
    close = np.column_stack([100.0 + t * f for f in (0.2, 0.5, 1.0, 1.7, 2.5)])
    cfg = MomentumConfig(lookback_bars=3, top_ratio=0.3, bottom_ratio=0.3, min_symbols_for_xs=5)
    alpha_l, _ = compute_xs_momentum_alpha(close, cfg)
    winners = np.argmax(alpha_l[3:], axis=1)
    assert np.all(winners == 4)
    assert np.all(alpha_l[3:, 4] > 0.0)


def test_bottom_only_short() -> None:
    t = np.arange(30.0)
    close = np.column_stack([100.0 + t * f for f in (2.5, 1.7, 1.0, 0.5, 0.2)])
    cfg = MomentumConfig(lookback_bars=3, top_ratio=0.3, bottom_ratio=0.3, min_symbols_for_xs=5)
    _, alpha_s = compute_xs_momentum_alpha(close, cfg)
    losers = np.argmax(alpha_s[3:], axis=1)
    assert np.all(losers == 4)
    assert np.all(alpha_s[3:, 4] > 0.0)


def test_nan_skip() -> None:
    close = np.tile(np.arange(1.0, 31.0).reshape(-1, 1), (1, 5))
    close[10:, 2] = np.nan
    cfg = MomentumConfig(lookback_bars=3, min_symbols_for_xs=3)
    alpha_l, alpha_s = compute_xs_momentum_alpha(close, cfg)
    assert np.all(alpha_l[10:, 2] == 0.0)
    assert np.all(alpha_s[10:, 2] == 0.0)


def test_min_symbols_gate() -> None:
    close = np.tile(np.arange(1.0, 25.0).reshape(-1, 1), (1, 4))
    cfg = MomentumConfig(lookback_bars=2, min_symbols_for_xs=5)
    alpha_l, alpha_s = compute_xs_momentum_alpha(close, cfg)
    assert np.all(alpha_l == 0.0)
    assert np.all(alpha_s == 0.0)


def test_no_lookahead() -> None:
    close = np.tile(np.arange(1.0, 41.0).reshape(-1, 1), (1, 6))
    cfg = MomentumConfig(lookback_bars=4, min_symbols_for_xs=5)
    alpha_l_a, alpha_s_a = compute_xs_momentum_alpha(close.copy(), cfg)

    changed = close.copy()
    changed[30, :] *= 10.0
    alpha_l_b, alpha_s_b = compute_xs_momentum_alpha(changed, cfg)

    assert np.allclose(alpha_l_a[:29], alpha_l_b[:29])
    assert np.allclose(alpha_s_a[:29], alpha_s_b[:29])


def test_dtype_shape() -> None:
    close = np.random.default_rng(7).uniform(1.0, 10.0, size=(50, 7)).astype(np.float64)
    cfg = MomentumConfig(lookback_bars=6, min_symbols_for_xs=5)
    alpha_l, alpha_s = compute_xs_momentum_alpha(close, cfg)
    assert alpha_l.shape == close.shape
    assert alpha_s.shape == close.shape
    assert alpha_l.dtype == np.float64
    assert alpha_s.dtype == np.float64
