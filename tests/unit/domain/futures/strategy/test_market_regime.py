from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.market_regime import compute_market_regime_context


def _make_aligned() -> AlignedMarketData:
    t = 240
    base = np.linspace(100.0, 140.0, t, dtype=np.float64)
    shock = np.zeros(t, dtype=np.float64)
    shock[-40:] = np.where(np.arange(40) % 2 == 0, 10.0, -10.0)
    close = (base + shock).reshape(t, 1)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full((t, 1), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, 1), dtype=np.float64),
        active_mask=np.ones((t, 1), dtype=bool),
        warm_mask=np.ones((t, 1), dtype=bool),
        entry_block_mask=np.zeros((t, 1), dtype=bool),
        kill_mask=np.zeros((t, 1), dtype=bool),
        execution_cost_bps_2d=np.zeros((t, 1), dtype=np.float64),
    )


def test_compute_market_regime_context_shapes_and_names() -> None:
    aligned = _make_aligned()
    regime = compute_market_regime_context(aligned=aligned)

    assert regime.code_1d.shape == (aligned.close_2d.shape[0],)
    assert regime.trend_score_1d.shape == (aligned.close_2d.shape[0],)
    assert regime.vol_z_1d.shape == (aligned.close_2d.shape[0],)
    assert regime.dispersion_z_1d.shape == (aligned.close_2d.shape[0],)
    assert regime.names().shape == (aligned.close_2d.shape[0],)
    assert set(np.unique(regime.names())).issubset(set(regime.name_by_code))


def test_compute_market_regime_context_detects_high_vol_regime() -> None:
    aligned = _make_aligned()
    regime = compute_market_regime_context(aligned=aligned)

    tail_codes = {int(code) for code in regime.code_1d[-20:]}
    assert tail_codes & {1, 3, 5}
