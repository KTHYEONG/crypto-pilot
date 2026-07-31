from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.types import CostModel, StrategySpec


@pytest.fixture
def spec() -> StrategySpec:
    return StrategySpec()


@pytest.fixture
def costs() -> CostModel:
    return CostModel()


@pytest.fixture
def bars_flat() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
    return pd.DataFrame({
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000.0,
    }, index=index)


@pytest.fixture
def bars_ramp() -> pd.DataFrame:
    n = 300
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    opens = np.arange(100.0, 100.0 + n, dtype=np.float64)
    return pd.DataFrame({
        "open": opens,
        "high": opens + 1.0,
        "low": opens - 1.0,
        "close": opens + 0.5,
        "volume": 1000.0,
    }, index=index)


@pytest.fixture
def bars_stop_gap() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
    return pd.DataFrame({
        "open": [100.0, 102.0, 101.0, 103.0, 95.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        "high": [101.0, 103.0, 102.0, 104.0, 96.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "low":  [99.0,  101.0, 100.0, 102.0, 94.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        "close":[101.0, 103.0, 102.0, 104.0, 95.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "volume": 1000.0,
    }, index=index)


@pytest.fixture
def bars_breakout_sparse() -> pd.DataFrame:
    """Isolated breakout cycles: each produces exactly one trade with no overlap.

    Designed so a one-bar signal delay shifts every fill by exactly one bar
    (SC-STRESS-02) -- two well-separated breakout/pullback cycles on a flat
    base, each completing before the next signal fires.
    """
    n = 220
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    o = np.full(n, 100.0)
    h = np.full(n, 100.6)
    l_ = np.full(n, 99.4)
    c = np.full(n, 100.0)

    def cycle(start: int, jump: float) -> None:
        h[start] = 100.0 + jump + 0.6
        c[start] = 100.0 + jump
        for t in range(start + 1, start + 4):
            o[t] = 100.0 + jump
            h[t] = 100.0 + jump + 0.6
            l_[t] = 100.0 + jump - 0.6
            c[t] = 100.0 + jump
        for t in range(start + 4, start + 12):
            o[t] = 100.0 + jump
            h[t] = 100.0 + jump + 0.5
            l_[t] = 100.0 + jump - 0.5
            c[t] = 100.0 + jump
        for t in range(start + 12, start + 16):
            o[t] = 100.0 + jump - 2.0
            h[t] = 100.0 + jump - 1.5
            l_[t] = 100.0 + jump - 2.5
            c[t] = 100.0 + jump - 2.0
        for t in range(start + 16, start + 40):
            o[t] = 100.0 + jump - 2.0
            h[t] = 100.0 + jump - 1.4
            l_[t] = 100.0 + jump - 2.6
            c[t] = 100.0 + jump - 2.0

    cycle(60, 6.0)
    cycle(130, 9.0)
    return pd.DataFrame({
        "open": o, "high": h, "low": l_, "close": c, "volume": 1000.0,
    }, index=index)


@pytest.fixture
def bars_both_touch() -> pd.DataFrame:
    """A bar where low touches stop price AND close triggers channel exit."""
    index = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
    o = np.full(300, 100.0)
    h = np.full(300, 101.0)
    l_ = np.full(300, 99.0)
    c = np.full(300, 100.0)
    o[210:] = 90.0
    h[210:] = 91.0
    l_[210:] = 89.0
    c[210:] = 90.0
    l_[210] = 85.0
    return pd.DataFrame({
        "open": o, "high": h, "low": l_, "close": c, "volume": 1000.0,
    }, index=index)


@pytest.fixture
def btc_4h_slice() -> pd.DataFrame:
    from src.data.loader import load_ohlcv_4h
    path = Path("data/futures/ohlcv/1h/BTCUSDT.parquet")
    return load_ohlcv_4h(path, end="2025-12-31")


@pytest.fixture
def make_carry_data():
    """Factory building a complete, validator-ready CarryMarketData fixture.

    ``funding`` maps "YYYY-MM-DD HH:MM" to a rate; when None, one zero-rate
    event is emitted on every bar (no funding gaps). ``borrow`` is one rate per
    bar; defaults to all zero. Prices default to a flat 100 for both legs.
    """
    from src.data.carry_data import CarryMarketData

    def _build(
        *,
        n_bars: int = 6,
        freq: str = "4h",
        start: str = "2024-01-01",
        spot_open: list[float] | None = None,
        spot_close: list[float] | None = None,
        perp_open: list[float] | None = None,
        perp_close: list[float] | None = None,
        funding: dict[str, float] | None = None,
        borrow: list[float] | None = None,
    ) -> CarryMarketData:
        grid = pd.date_range(start, periods=n_bars, freq=freq, tz="UTC")

        def _frame(o: list[float] | None, c: list[float] | None) -> pd.DataFrame:
            oa = np.full(n_bars, 100.0) if o is None else np.asarray(o, dtype=float)
            ca = np.full(n_bars, 100.0) if c is None else np.asarray(c, dtype=float)
            if len(oa) != n_bars or len(ca) != n_bars:
                raise ValueError("price arrays must match n_bars")
            return pd.DataFrame({
                "open": oa,
                "high": np.maximum(oa, ca) + 1.0,
                "low": np.minimum(oa, ca) - 1.0,
                "close": ca,
                "volume": 1000.0,
            }, index=grid)

        if funding is None:
            funding_series = pd.Series(0.0, index=grid, dtype=float)
        else:
            idx = pd.DatetimeIndex(
                [pd.Timestamp(key, tz="UTC") for key in funding]
            )
            funding_series = pd.Series(
                list(funding.values()), index=idx, dtype=float,
            ).sort_index()

        borrow_arr = np.zeros(n_bars) if borrow is None else np.asarray(borrow, dtype=float)
        borrow_series = pd.Series(borrow_arr, index=grid, dtype=float)
        return CarryMarketData(
            symbol="BTCUSDT",
            spot=_frame(spot_open, spot_close),
            perp=_frame(perp_open, perp_close),
            funding=funding_series,
            borrow=borrow_series,
        )

    return _build
