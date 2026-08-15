from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.market_regime import (
    _REGIME_MEMO,
    compute_market_regime_context,
)


@pytest.fixture(autouse=True)
def clear_memo():
    _REGIME_MEMO.clear()
    return


def _aligned(n_bars: int = 500, n_sym: int = 10) -> AlignedMarketData:
    import pandas as pd

    base_dt = pd.date_range("2026-01-01", periods=n_bars, freq="1h", tz="UTC").to_numpy(dtype="datetime64[ns]")
    close = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, (n_bars, n_sym)), axis=0)
    close = np.maximum(close, 1.0)
    mask = np.ones((n_bars, n_sym), dtype=bool)
    return AlignedMarketData(
        datetimes=base_dt,
        symbols=("BTCUSDT", *(f"SYM{i}" for i in range(n_sym - 1))),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.ones((n_bars, n_sym), dtype=np.float64) * 1000.0,
        funding_2d=np.zeros((n_bars, n_sym), dtype=np.float64),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=mask,
        kill_mask=~mask,
    )


def test_regime_memo_hit_on_same_aligned():
    aligned = _aligned(n_bars=500, n_sym=10)
    ctx1 = compute_market_regime_context(aligned=aligned)
    ctx2 = compute_market_regime_context(aligned=aligned)
    assert ctx1 is ctx2
    assert len(_REGIME_MEMO) == 1


def test_regime_memo_miss_on_different_aligned():
    aligned_a = _aligned(n_bars=500, n_sym=10)
    aligned_b = _aligned(n_bars=300, n_sym=5)
    ctx1 = compute_market_regime_context(aligned=aligned_a)
    ctx2 = compute_market_regime_context(aligned=aligned_b)
    assert ctx1 is not ctx2
    assert len(_REGIME_MEMO) == 2


def test_regime_memo_eviction():
    aligned_list = [_aligned(n_bars=100 * i, n_sym=3) for i in range(1, 12)]
    for a in aligned_list[:10]:
        compute_market_regime_context(aligned=a)
    ctx_before = compute_market_regime_context(aligned=aligned_list[0])
    compute_market_regime_context(aligned=aligned_list[10])
    assert len(_REGIME_MEMO) <= 8
