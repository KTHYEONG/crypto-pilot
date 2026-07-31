from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.data.carry_data import CarryMarketData, validate_carry_market_data
from src.data.loader import DataIntegrityError


class TestCarryMarketDataValidation:
    def test_accepts_complete_two_bar_fixture_without_interpolation(
        self,
        make_carry_data,
    ) -> None:
        # SC-CARRY-DATA-01: complete event coverage is accepted as-is; no
        # missing funding/borrow observation is synthesized.
        data = make_carry_data(
            n_bars=2,
            funding={"2024-01-01 00:00": 0.001},
            borrow=[0.0001, 0.0001],
        )
        validate_carry_market_data(data)
        assert data.spot.index.equals(data.perp.index)
        assert len(data.funding) == 1
        assert data.borrow.notna().all()

    def test_rejects_funding_gap_over_eight_hours(self, make_carry_data) -> None:
        # SC-CARRY-DATA-02: a 12h funding gap is never accepted as zero funding.
        data = make_carry_data(
            n_bars=4,
            funding={
                "2024-01-01 00:00": 0.001,
                "2024-01-01 12:00": 0.001,
            },
            borrow=[0.0, 0.0, 0.0, 0.0],
        )
        with pytest.raises(DataIntegrityError, match="funding gap"):
            validate_carry_market_data(data)

    def test_rejects_nan_borrow(self, make_carry_data) -> None:
        # SC-CARRY-DATA-02: NaN borrow is never replaced with a zero cost.
        data = make_carry_data(
            n_bars=2,
            funding={"2024-01-01 00:00": 0.001},
            borrow=[0.0, np.nan],
        )
        with pytest.raises(DataIntegrityError, match="borrow"):
            validate_carry_market_data(data)

    def test_rejects_missing_spot_bar(self) -> None:
        # SC-CARRY-DATA-02: a non-uniform spot grid is a missing interval.
        grid = pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 04:00", "2024-01-01 09:00"],
            tz="UTC",
        )
        spot = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0},
            index=grid,
        )
        perp = spot.copy()
        funding = pd.Series([0.001, 0.0, 0.0], index=grid)
        borrow = pd.Series([0.0, 0.0, 0.0], index=grid)
        data = CarryMarketData(symbol="BTCUSDT", spot=spot, perp=perp, funding=funding, borrow=borrow)
        with pytest.raises(DataIntegrityError, match="missing bars"):
            validate_carry_market_data(data)

    def test_rejects_missing_funding_events(self, make_carry_data) -> None:
        # An empty funding series is incomplete, never a zero-cost assumption.
        data = dataclasses.replace(
            make_carry_data(n_bars=2, borrow=[0.0, 0.0]),
            funding=pd.Series(dtype=float),
        )
        with pytest.raises(DataIntegrityError, match="funding"):
            validate_carry_market_data(data)

    def test_rejects_non_positive_price(self, make_carry_data) -> None:
        data = make_carry_data(
            n_bars=2,
            funding={"2024-01-01 00:00": 0.001},
            borrow=[0.0, 0.0],
            spot_close=[100.0, 0.0],
        )
        with pytest.raises(DataIntegrityError, match="strictly positive"):
            validate_carry_market_data(data)

    def test_rejects_misaligned_spot_perp_grid(self, make_carry_data) -> None:
        data = make_carry_data(
            n_bars=2,
            funding={"2024-01-01 00:00": 0.001},
            borrow=[0.0, 0.0],
        )
        perp_shifted = data.perp.copy()
        perp_shifted.index = perp_shifted.index + pd.Timedelta(hours=1)
        shifted = dataclasses.replace(data, perp=perp_shifted)
        with pytest.raises(DataIntegrityError, match="identical"):
            validate_carry_market_data(shifted)
