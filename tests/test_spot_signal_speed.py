"""Regression checks for spot signal generation after speed optimizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.spot.strategies_spot import UltimateSpotStrategy


def _sample_ohlcv(n: int = 480, seed: int = 7) -> pd.DataFrame:
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
        "SIGNAL_TYPE": "ADX_BREAKOUT",
        "REGIME_TYPE": "EMA_ATR",
        "ATR_PERIOD": 14,
        "KC_PERIOD": 20,
        "KC_MULT": 2.0,
        "SIZING_METHOD": "vol_target",
        "RISK_PER_TRADE": 0.02,
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
    for col in ("garch_kelly_f", "regime_risk_mult", "long_entry_signal"):
        assert np.isfinite(out[col].astype(np.float64).values).all(), col
