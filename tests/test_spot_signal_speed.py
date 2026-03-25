"""Regression checks for spot signal generation after speed optimizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.spot_strategy.strategies_spot import UltimateSpotStrategy


def _sample_ohlcv(n: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="4h")
    price = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "close": price,
            "volume": rng.random(n) * 1e6 + 1e3,
        }
    )


def _default_params() -> dict[str, object]:
    return {
        "SUPERTREND_PERIOD": 10,
        "SUPERTREND_MULT": 3.0,
        "ATR_RATIO_PERIOD": 14,
        "ATR_RATIO_LONG_PERIOD": 42,
        "ATR_EXPANSION_THRESHOLD": 1.2,
        "EMA_TREND_PERIOD": 100,
        "MOMENTUM_ROC_PERIOD": 14,
        "RSI_PERIOD": 14,
        "HMM_TRAIN_WINDOW": 240,
        "HMM_RETRAIN_FREQ": 24,
        "GARCH_WINDOW": 240,
        "GARCH_RETRAIN_FREQ": 24,
        "KILL_ATR_K": 4.0,
        "HURST_WINDOW": 40,
        "USE_HMM_REGIME": True,
    }


def test_generate_signals_deterministic_repeated() -> None:
    """Same inputs must yield identical outputs (no hidden RNG in signal path)."""
    df = _sample_ohlcv()
    s = UltimateSpotStrategy("spot_det", _default_params())
    a = s.generate_signals(df.copy())
    b = s.generate_signals(df.copy())
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True),
        b.reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )


def test_generate_signals_key_columns_finite() -> None:
    df = _sample_ohlcv()
    s = UltimateSpotStrategy("spot_fin", _default_params())
    out = s.generate_signals(df.copy())
    for col in ("garch_kelly_f", "regime_risk_mult", "p_bull", "p_side", "long_entry_signal"):
        assert np.isfinite(out[col].astype(np.float64).values).all(), col
