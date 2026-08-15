from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.optimization.ml_context import _build_beta_2d_full


def _df_from_close(close: np.ndarray, beta: np.ndarray | None = None) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=close.size, freq="4h", tz="UTC"),
            "close": close,
        }
    )
    if beta is not None:
        out["beta"] = beta
    return out


def test_build_beta_2d_full_uses_existing_beta_column() -> None:
    n = 64
    beta_vals = np.full((n,), 1.25, dtype=np.float64)
    data_maps = {
        "BTCUSDT": {"4h": _df_from_close(np.linspace(100.0, 120.0, n), beta_vals)},
        "ETHUSDT": {"4h": _df_from_close(np.linspace(200.0, 260.0, n), np.full((n,), 0.75))},
    }
    info = {"alignment_offsets": {"BTCUSDT": 0, "ETHUSDT": 0}}
    beta_2d = _build_beta_2d_full(data_maps, ["BTCUSDT", "ETHUSDT"], "4h", info, n)
    assert beta_2d is not None
    assert beta_2d.shape == (n, 2)
    np.testing.assert_allclose(beta_2d[:, 0], beta_vals, atol=0.0, rtol=0.0)


def test_build_beta_2d_full_falls_back_to_causal_trailing_beta() -> None:
    n = 96
    rng = np.random.default_rng(7)
    btc_ret = rng.normal(loc=0.001, scale=0.01, size=n)
    eth_ret = 2.0 * btc_ret
    btc_close = 100.0 * np.cumprod(1.0 + btc_ret)
    eth_close = 200.0 * np.cumprod(1.0 + eth_ret)
    data_maps = {
        "BTCUSDT": {"4h": _df_from_close(btc_close)},
        "ETHUSDT": {"4h": _df_from_close(eth_close)},
    }
    info = {"alignment_offsets": {"BTCUSDT": 0, "ETHUSDT": 0}}
    beta_2d = _build_beta_2d_full(data_maps, ["BTCUSDT", "ETHUSDT"], "4h", info, n)
    assert beta_2d is not None
    assert beta_2d.shape == (n, 2)
    assert np.isfinite(beta_2d).all()
    assert beta_2d[-1, 0] > 0.7
    assert beta_2d[-1, 1] > 1.4
