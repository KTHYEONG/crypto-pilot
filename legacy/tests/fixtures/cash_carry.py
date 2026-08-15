from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_carry_data():
    """Factory building a complete, validator-ready CarryMarketData fixture.

    ``funding`` maps "YYYY-MM-DD HH:MM" to a rate; when None, one zero-rate
    event is emitted on every bar (no funding gaps). ``borrow`` is one rate per
    bar; defaults to all zero. Prices default to a flat 100 for both legs.
    """
    from src.research.cash_carry.contracts import CarryMarketData

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
