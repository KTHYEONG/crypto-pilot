"""rank_score direction vs entry_signal (tmp.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.spot.signals.kc_pullback import KCPullbackSignal
from src.domain.spot.signals.rs_momentum import RSMomentumSignal


def _minimal_ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + rng.random(n) * 0.5
    low = close - rng.random(n) * 0.5
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.random(n) * 1e6 + 1e5,
        }
    )


def test_rs_momentum_rank_is_positive_z_score() -> None:
    df = _minimal_ohlcv(400)
    params = {
        "MOMENTUM_PERIOD": 20,
        "MOMENTUM_LOOKBACK": 60,
        "MOMENTUM_THRESHOLD": 0.5,
    }
    out = RSMomentumSignal().compute(df, params)
    # rank_score must equal +z (not inverted)
    close = df["close"].to_numpy(dtype=np.float64)
    ret = pd.Series(close).pct_change(int(params["MOMENTUM_PERIOD"]))
    vol = pd.Series(close).pct_change().rolling(int(params["MOMENTUM_PERIOD"])).std()
    momentum_score = (ret / (vol + 1e-9)).to_numpy(dtype=np.float64)
    momentum_score = np.nan_to_num(momentum_score, nan=0.0, posinf=0.0, neginf=0.0)
    lookback = int(params["MOMENTUM_LOOKBACK"])
    min_periods = max(10, lookback // 3)
    rolling_mean = (
        pd.Series(momentum_score)
        .rolling(lookback, min_periods=min_periods)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    rolling_std = (
        pd.Series(momentum_score)
        .rolling(lookback, min_periods=min_periods)
        .std()
        .to_numpy(dtype=np.float64)
    )
    z = (momentum_score - rolling_mean) / np.where(rolling_std > 1e-9, rolling_std, 1e-9)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.allclose(out.rank_score, z, rtol=0.0, atol=1e-6)


def test_kc_pullback_rank_is_100_minus_rsi() -> None:
    df = _minimal_ohlcv(400)
    params = {
        "EMA_SLOW_PERIOD": 100,
        "KC_PERIOD": 20,
        "KC_MULT": 2.0,
        "RSI_PERIOD": 14,
        "RSI_LOW_THRESH": 35.0,
        "TP_MEAN_PERIOD": 20,
        "EMA_SLOPE_LAG": 10,
    }
    from src.core.indicators.numpy_ops_spot import compute_rsi_numpy

    out = KCPullbackSignal().compute(df, params)
    close = df["close"].to_numpy(dtype=np.float64)
    rsi = compute_rsi_numpy(close, int(params["RSI_PERIOD"]))
    expected = 100.0 - rsi
    assert np.allclose(out.rank_score, expected, rtol=0.0, atol=1e-6)
