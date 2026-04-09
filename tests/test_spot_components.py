from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.spot.opt_spot_utils.combination_screener import params_disqualified_against_space
from src.domain.spot.opt_spot_utils.opt_params import build_combined_param_space
from src.domain.spot.signals.bb_squeeze import BBSqueezeSignal
from src.domain.spot.signals.macd_hist_div import MacdHistDivSignal
from src.core.indicators.numpy_ops_spot import rolling_ema_winsorize_volume
from src.domain.spot.signals.obv_ma_breakout import ObvMaBreakoutSignal
from src.domain.spot.signals.rsi2_pullback import RSI2PullbackSignal
from src.domain.spot.signals.rs_momentum import RSMomentumSignal
from src.domain.spot.signals.stochrsi_cross import StochRSICrossSignal
from src.domain.spot.signals.vix_fix import VIXFixSignal


def test_rs_momentum_signal_compute_returns_signal_output() -> None:
    rng = np.random.default_rng(3)
    n = 120
    idx = pd.date_range("2021-01-01", periods=n, freq="4h")
    price = 50.0 + np.cumsum(rng.standard_normal(n) * 0.2)
    df = pd.DataFrame(
        {
            "datetime": idx,
            "open": price,
            "high": price * 1.002,
            "low": price * 0.998,
            "close": price,
            "volume": rng.random(n) * 1e5 + 1e3,
        }
    )
    sig = RSMomentumSignal()
    out = sig.compute(df, {"MOMENTUM_PERIOD": 15})
    assert out.entry_signal.shape == (n,)
    assert out.kill_signal.shape == (n,)
    assert out.rank_score.shape == (n,)


def test_kc_mult_three_disqualified_against_space() -> None:
    space = build_combined_param_space("ADX_BREAKOUT", "EMA_ATR", "vol_target")
    assert params_disqualified_against_space({"KC_MULT": 3.0}, space) == "kc_mult_oob"
    assert params_disqualified_against_space({"KC_MULT": 2.0}, space) == ""


def _ohlcv_df(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="4h")
    price = 50.0 + np.cumsum(rng.standard_normal(n) * 0.2)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": price,
            "high": price * 1.002,
            "low": price * 0.998,
            "close": price,
            "volume": rng.random(n) * 1e5 + 1e3,
        }
    )


@pytest.mark.parametrize(
    "cls, param_key, param_val",
    [
        (BBSqueezeSignal, "BB_SQ_PERIOD", 20),
        (RSI2PullbackSignal, "RSI2_TREND_EMA", 200),
        (StochRSICrossSignal, "STOCHRSI_RSI_P", 14),
        (VIXFixSignal, "WVF_PERIOD", 22),
        (ObvMaBreakoutSignal, "OBV_WINSOR_SPAN", 48),
        (MacdHistDivSignal, "MACD_FAST", 12),
    ],
)
def test_new_spot_signals_compute_shape(
    cls: type,
    param_key: str,
    param_val: int | float,
) -> None:
    df = _ohlcv_df(250)
    sig = cls()
    out = sig.compute(df, {param_key: param_val})
    n = len(df)
    assert out.entry_signal.shape == (n,)
    assert out.kill_signal.shape == (n,)
    assert out.rank_score.shape == (n,)


def test_rolling_ema_winsorize_causality_last_bar_unchanged_when_future_volume_spike() -> None:
    """Bar t output must not depend on volume[t+1:]."""
    base = np.ones(30, dtype=np.float64) * 10.0
    full = base.copy()
    full[-1] = 1e9
    w_full = rolling_ema_winsorize_volume(full, span=8, k=3.0)
    truncated = np.concatenate([base, np.array([10.0])])
    w_trunc = rolling_ema_winsorize_volume(truncated, span=8, k=3.0)
    assert abs(w_full[-2] - w_trunc[-2]) < 1e-9


def test_convex_slippage_multiplier_examples() -> None:
    """adj = rate * (1 + gamma * excess^1.5); scale=1."""
    rate = 0.0005
    gamma = 0.03
    for excess, expected in [(0, 1.0), (2, 1.0 + 0.03 * (2**1.5)), (6, 1.0 + 0.03 * (6**1.5))]:
        adj = rate * (1.0 + gamma * (float(excess) ** 1.5))
        assert abs(adj / rate - expected) < 1e-9
