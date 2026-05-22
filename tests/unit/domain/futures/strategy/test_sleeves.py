from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.sleeves.carry import CarrySleeve
from src.domain.futures.strategy.sleeves.ts_momentum import TSMomentumSleeve
from src.domain.futures.strategy.sleeves.xs_reversal import XSReversalSleeve


def test_xs_reversal_sleeve_basic() -> None:
    # 10 steps, 3 assets
    close = np.array(
        [
            [10.0, 20.0, 30.0],
            [10.1, 19.8, 30.2],
            [10.2, 19.5, 30.5],
            [10.3, 19.2, 30.8],
            [10.4, 18.9, 31.1],
            [10.5, 18.6, 31.4],
            [10.6, 18.3, 31.7],  # lookback 6 is here
            [10.7, 18.0, 32.0],
            [10.8, 17.7, 32.3],
            [10.9, 17.4, 32.6],
        ],
        dtype=np.float64,
    )

    sleeve = XSReversalSleeve(lookback_bars=6)
    sig = sleeve.compute_raw(close, aux={})

    assert sig.shape == (10, 3)
    # Warmup should be NaN
    assert np.all(np.isnan(sig[:6]))
    # Valid from idx=6
    assert np.all(~np.isnan(sig[6:]))

    # Asset 0: steadily goes up (10.0 -> 10.6 at idx=6). Reversal signal should be negative (bearish)
    assert sig[6, 0] < 0.0
    # Asset 1: steadily goes down (20.0 -> 18.3 at idx=6). Reversal signal should be positive (bullish)
    assert sig[6, 1] > 0.0


def test_xs_reversal_no_lookahead() -> None:
    # Causality test: changing close[t+1] must not affect sig[t]
    close = np.random.normal(100, 5, (20, 4))
    sleeve = XSReversalSleeve(lookback_bars=6)

    sig1 = sleeve.compute_raw(close, aux={})

    # Modify last row
    close_modified = close.copy()
    close_modified[-1] += 50.0

    sig2 = sleeve.compute_raw(close_modified, aux={})

    # All signals except the last one must be identical
    assert np.allclose(sig1[:-1], sig2[:-1], equal_nan=True)


def test_ts_momentum_sleeve_basic() -> None:
    close = np.array(
        [
            [10.0, 10.0],
            [10.1, 9.9],
            [10.2, 9.8],
            [10.3, 9.7],
            [10.4, 9.6],
        ],
        dtype=np.float64,
    )
    # total lag = 2 + 1 = 3
    sleeve = TSMomentumSleeve(lookback_bars=2, skip_bars=1)
    sig = sleeve.compute_raw(close, aux={})

    assert sig.shape == (5, 2)
    assert np.all(np.isnan(sig[:3]))
    assert np.all(~np.isnan(sig[3:]))

    # For t=3: return from close[0] to close[2] (10.0 -> 10.2 for asset 0 (momentum > 0), 10.0 -> 9.8 for asset 1 (momentum < 0))
    assert sig[3, 0] > 0.0
    assert sig[3, 1] < 0.0


def test_carry_sleeve_basic() -> None:
    close = np.ones((5, 2))
    funding = np.array(
        [
            [-0.001, 0.002],  # Asset 0: negative (bullish for carry), Asset 1: positive (bearish)
            [-0.0015, 0.0025],
            [-0.0012, 0.0022],
            [np.nan, np.nan],  # NaN handling
            [-0.001, 0.002],
        ],
        dtype=np.float64,
    )

    sleeve = CarrySleeve(smooth_bars=3)
    sig = sleeve.compute_raw(close, aux={"funding_2d": funding})

    assert sig.shape == (5, 2)
    # Negative funding leads to positive carry signal
    assert sig[0, 0] > 0.0
    assert sig[0, 1] < 0.0

    # Ensure NaN values simply carry forward previous state
    assert np.allclose(sig[3], sig[2])

