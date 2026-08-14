"""Contract coverage for the additive time-series trend sleeve.

Scenario ids (docs/specs/mhs_directional_trend_sleeve_contract.json):
SCENARIO_TREND_BASKET_IS_CAUSAL_AND_EQUAL_WEIGHT
SCENARIO_TREND_POSITION_CLIPPED_AND_VOL_NORMALIZED
SCENARIO_TREND_POSITION_HELD_ON_DECISION_GRID
SCENARIO_TREND_SLEEVE_WEIGHTS_ARE_DIRECTIONAL_NOT_DOLLAR_NEUTRAL
SCENARIO_TREND_SLEEVE_GROSS_BUDGET_IS_AN_UPPER_BOUND
SCENARIO_TREND_SLEEVE_IS_ADDITIVE_NOT_CANNIBALIZING
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from src.mhs.regime import crash_regime_tilt_weights
from src.mhs.trend_sleeve import (
    market_basket_log_price,
    time_series_trend_position,
    trend_sleeve_weights,
)


class TestMarketBasketLogPrice:
    """SCENARIO_TREND_BASKET_IS_CAUSAL_AND_EQUAL_WEIGHT"""

    def test_flat_index_when_cross_section_offsets(self) -> None:
        idx = pd.date_range("2021-01-01", periods=5, freq="1h", tz="UTC")
        step = 0.01
        log_close = pd.DataFrame(
            {"A": np.cumsum([step] * 5), "B": np.cumsum([-step] * 5)}, index=idx,
        )
        eligible = pd.DataFrame(True, index=idx, columns=log_close.columns)
        basket = market_basket_log_price(log_close, eligible)
        pd.testing.assert_series_equal(basket, pd.Series(0.0, index=idx))

    def test_truncation_leaves_prefix_bit_identical(self) -> None:
        # Truncating the panel after bar k must leave the first k index values
        # bit-identical -- proof that no bar after t is ever read.
        rng = np.random.default_rng(1)
        n = 40
        idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
        log_close = pd.DataFrame(
            {
                "A": np.cumsum(rng.normal(0.0, 0.01, n)),
                "B": np.cumsum(rng.normal(0.0, 0.01, n)),
                "C": np.cumsum(rng.normal(0.0, 0.01, n)),
            },
            index=idx,
        )
        eligible = pd.DataFrame(True, index=idx, columns=log_close.columns)
        full = market_basket_log_price(log_close, eligible)
        for k in (5, 17, 33):
            truncated = market_basket_log_price(
                log_close.iloc[:k], eligible.iloc[:k],
            )
            pd.testing.assert_series_equal(truncated, full.iloc[:k])

    def test_ineligible_symbols_never_contribute(self) -> None:
        idx = pd.date_range("2021-01-01", periods=5, freq="1h", tz="UTC")
        log_close = pd.DataFrame(
            {"A": np.cumsum([0.01] * 5), "B": np.cumsum([-0.05] * 5)}, index=idx,
        )
        eligible = pd.DataFrame(
            {"A": True, "B": False}, index=idx, columns=log_close.columns,
        )
        basket = market_basket_log_price(log_close, eligible)
        expected = pd.Series([0.0, 0.01, 0.02, 0.03, 0.04], index=idx)
        pd.testing.assert_series_equal(basket, expected)

    def test_mismatched_panels_raise_value_error(self) -> None:
        idx = pd.date_range("2021-01-01", periods=5, freq="1h", tz="UTC")
        log_close = pd.DataFrame({"A": [0.0] * 5, "B": [0.0] * 5}, index=idx)
        bad_idx = pd.date_range("2021-01-02", periods=5, freq="1h", tz="UTC")
        with pytest.raises(ValueError, match="indexed"):
            market_basket_log_price(
                log_close, pd.DataFrame(True, index=bad_idx, columns=["A", "B"]),
            )
        with pytest.raises(ValueError, match="columned"):
            market_basket_log_price(
                log_close, pd.DataFrame(True, index=idx, columns=["A", "X"]),
            )


class TestTimeSeriesTrendPosition:
    """SCENARIO_TREND_POSITION_CLIPPED_AND_VOL_NORMALIZED
    SCENARIO_TREND_POSITION_HELD_ON_DECISION_GRID"""

    def test_bounded_finite_and_zero_in_lead_in(self) -> None:
        idx = pd.date_range("2021-01-01", periods=200, freq="1h", tz="UTC")
        rising = pd.Series(np.cumsum(np.full(200, 0.0005)), index=idx)
        grid = pd.date_range(idx[0], idx[-1], freq="24h", tz="UTC")
        pos = time_series_trend_position(rising, (24, 48), grid)
        assert np.isfinite(pos).all()
        assert (pos >= -1.0).all()
        assert (pos <= 1.0).all()
        # The insufficient-history lead-in (the smallest horizon's warmup) is
        # exactly 0.0 -- never NaN, never inf.
        assert (pos.iloc[:24] == 0.0).all()

    def test_rising_basket_positive_falling_negative_after_lead_in(self) -> None:
        n = 400
        idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
        grid = pd.date_range(idx[0], idx[-1], freq="24h", tz="UTC")
        rising = pd.Series(np.cumsum(np.full(n, 0.0005)), index=idx)
        pos_up = time_series_trend_position(rising, (24, 48, 72), grid)
        assert (pos_up.iloc[72:] > 0.0).all()
        falling = pd.Series(np.cumsum(np.full(n, -0.0005)), index=idx)
        pos_dn = time_series_trend_position(falling, (24, 48, 72), grid)
        assert (pos_dn.iloc[72:] < 0.0).all()

    def test_doubling_volatility_leaves_position_unchanged(self) -> None:
        # Self-vol normalization: scaling the basket's volatility by 2 while
        # preserving its trend shape must leave every value identical.
        rng = np.random.default_rng(3)
        n = 600
        idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
        trend = np.cumsum(rng.normal(0.0002, 0.001, n))
        grid = pd.date_range(idx[0], idx[-1], freq="24h", tz="UTC")
        pos_low = time_series_trend_position(
            pd.Series(trend, index=idx), (48, 120), grid,
        )
        pos_high = time_series_trend_position(
            pd.Series(2.0 * trend, index=idx), (48, 120), grid,
        )
        pd.testing.assert_series_equal(pos_low, pos_high)

    def test_held_piecewise_constant_between_grid_stamps(self) -> None:
        rng = np.random.default_rng(9)
        n = 240
        idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
        basket = pd.Series(np.cumsum(rng.normal(0.0002, 0.001, n)), index=idx)
        grid = pd.date_range(idx[0], idx[-1], freq="24h", tz="UTC")
        pos = time_series_trend_position(basket, (24, 48), grid)
        assert pos.index.equals(idx)
        for lo, hi in pairwise(grid):
            inside = pos.loc[lo : hi - pd.Timedelta(hours=1)]
            assert len(inside) == 24
            assert (inside == inside.iloc[0]).all()
        # A decision made at a grid stamp is held, not recomputed intra-interval.
        assert pos.loc[grid[0]] == pos.loc[grid[0] + pd.Timedelta(hours=1)]

    def test_validation(self) -> None:
        idx = pd.date_range("2021-01-01", periods=100, freq="1h", tz="UTC")
        basket = pd.Series(np.cumsum(np.full(100, 0.001)), index=idx)
        grid = pd.date_range(idx[0], idx[-1], freq="24h", tz="UTC")
        with pytest.raises(ValueError, match="horizons_hours"):
            time_series_trend_position(basket, (), grid)
        with pytest.raises(ValueError, match="horizons_hours"):
            time_series_trend_position(basket, (24, 0), grid)
        with pytest.raises(ValueError, match="decision_grid"):
            time_series_trend_position(basket, (24,), grid[::-1])


class TestTrendSleeveWeights:
    """SCENARIO_TREND_SLEEVE_WEIGHTS_ARE_DIRECTIONAL_NOT_DOLLAR_NEUTRAL
    SCENARIO_TREND_SLEEVE_GROSS_BUDGET_IS_AN_UPPER_BOUND"""

    def test_directional_not_dollar_neutral(self) -> None:
        idx = pd.date_range("2021-01-01", periods=2, freq="1h", tz="UTC")
        cols = [f"S{i}" for i in range(30)]
        position = pd.Series([1.0, -1.0], index=idx)
        mask = pd.DataFrame(True, index=idx, columns=cols)
        w = trend_sleeve_weights(position, mask, 0.3)
        # position=+1.0: every eligible cell is +0.3/30, the row SUM is +0.3
        # (deliberately NOT zero -- the net directional exposure this feature
        # exists to express) and the row gross equals 0.3.
        assert w.iloc[0].to_numpy() == pytest.approx(0.3 / 30)
        assert w.iloc[0].sum() == pytest.approx(0.3)
        assert w.iloc[0].abs().sum() == pytest.approx(0.3)
        # position=-1.0 flips every sign.
        assert w.iloc[1].to_numpy() == pytest.approx(-0.3 / 30)
        assert w.iloc[1].sum() == pytest.approx(-0.3)

    def test_ineligible_cells_exactly_zero(self) -> None:
        idx = pd.date_range("2021-01-01", periods=1, freq="1h", tz="UTC")
        cols = [f"S{i}" for i in range(30)]
        position = pd.Series([1.0], index=idx)
        mask = pd.DataFrame(True, index=idx, columns=cols)
        mask.loc[idx[0], "S3"] = False
        w = trend_sleeve_weights(position, mask, 0.3)
        assert w.loc[idx[0], "S3"] == 0.0
        assert w.loc[idx[0], "S0"] == pytest.approx(0.3 / 29)
        assert w.loc[idx[0]].sum() == pytest.approx(0.3)

    def test_fails_closed_below_min_symbols(self) -> None:
        idx = pd.date_range("2021-01-01", periods=1, freq="1h", tz="UTC")
        cols = [f"S{i}" for i in range(30)]
        position = pd.Series([1.0], index=idx)
        mask = pd.DataFrame(True, index=idx, columns=cols)
        mask.iloc[0, :5] = False  # 25 active cells
        w = trend_sleeve_weights(position, mask, 0.3, min_symbols=26)
        assert (w.iloc[0] == 0.0).all()
        w_default = trend_sleeve_weights(position, mask, 0.3)
        assert w_default.iloc[0].sum() == pytest.approx(0.3)

    def test_gross_budget_is_an_upper_bound(self) -> None:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2021-01-01", periods=100, freq="1h", tz="UTC")
        cols = [f"S{i}" for i in range(30)]
        position = pd.Series(rng.uniform(-1.0, 1.0, len(idx)), index=idx)
        mask = pd.DataFrame(True, index=idx, columns=cols)
        w = trend_sleeve_weights(position, mask, 0.3)
        gross = w.abs().sum(axis=1)
        assert (gross <= 0.3 + 1e-12).all()
        pd.testing.assert_series_equal(gross, 0.3 * position.abs())

    def test_zero_budget_is_all_zero_book(self) -> None:
        idx = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
        cols = [f"S{i}" for i in range(30)]
        position = pd.Series([1.0, -0.5, 0.3], index=idx)
        mask = pd.DataFrame(True, index=idx, columns=cols)
        w = trend_sleeve_weights(position, mask, 0.0)
        assert (w == 0.0).all().all()

    def test_validation(self) -> None:
        idx = pd.date_range("2021-01-01", periods=2, freq="1h", tz="UTC")
        position = pd.Series([1.0, 1.0], index=idx)
        mask = pd.DataFrame(True, index=idx, columns=["A", "B", "C"])
        with pytest.raises(ValueError, match="gross_budget"):
            trend_sleeve_weights(position, mask, -0.1)
        with pytest.raises(ValueError, match="gross_budget"):
            trend_sleeve_weights(position, mask, 1.5)
        with pytest.raises(ValueError, match="min_symbols"):
            trend_sleeve_weights(position, mask, 0.3, min_symbols=1)
        with pytest.raises(ValueError, match="index"):
            trend_sleeve_weights(position.iloc[:1], mask, 0.3)


class TestAdditiveNotCannibalizing:
    """SCENARIO_TREND_SLEEVE_IS_ADDITIVE_NOT_CANNIBALIZING"""

    def test_additive_sleeve_leaves_base_book_untouched(self) -> None:
        idx = pd.date_range("2021-01-01", periods=120, freq="1h", tz="UTC")
        t = np.arange(120, dtype=float)
        columns = ["BTCUSDT", "A", "B", "C", "D", "E", "F", "G"]
        log_price = pd.DataFrame(
            {
                "BTCUSDT": 0.05 * t**2,
                "A": 0.01 * t,
                "B": -0.01 * t,
                "C": 0.02 * t,
                "D": 0.03 * t,
                "E": -0.02 * t,
                "F": 0.01 * t,
                "G": -0.03 * t,
            },
            index=idx,
        )
        eligible = pd.DataFrame(True, index=idx, columns=columns)
        w_base = pd.DataFrame(
            {
                "BTCUSDT": 0.0, "A": 0.5, "B": 0.0, "C": 0.0,
                "D": 0.0, "E": -0.5, "F": 0.0, "G": 0.0,
            },
            index=idx,
        )
        # The base book is exactly dollar-neutral and unit-gross.
        assert (w_base.sum(axis=1).abs() < 1e-12).all()
        assert np.allclose(w_base.abs().sum(axis=1).to_numpy(), 1.0)

        basket = market_basket_log_price(log_price, eligible)
        grid = pd.date_range(idx[0], idx[-1], freq="24h", tz="UTC")
        position = time_series_trend_position(basket, (24, 48, 72), grid)
        sleeve = trend_sleeve_weights(position, eligible, 0.3)

        # w_total = w_base + sleeve: the sleeve is added on, never a convex
        # blend, so the base book's own weights are untouched.
        w_total = w_base.add(sleeve)
        assert np.allclose(
            w_total.sub(sleeve).to_numpy(), w_base.to_numpy(), atol=1e-12,
        )
        # The combined row sum equals the sleeve's net exposure (the base book
        # is dollar-neutral, contributing zero net).
        assert np.allclose(
            w_total.sum(axis=1).to_numpy(), sleeve.sum(axis=1).to_numpy(), atol=1e-12,
        )

    def test_crash_tilt_convex_blend_shrinks_base_book(self) -> None:
        # The explicit contrast: regime.crash_regime_tilt_weights' convex blend
        # (1-alpha)*book + alpha*tilt DOES shrink the base book on the same
        # fixture, while the additive sleeve does not.
        idx = pd.date_range("2021-01-01", periods=120, freq="1h", tz="UTC")
        t = np.arange(120, dtype=float)
        columns = ["BTCUSDT", "A", "B", "C", "D", "E", "F", "G"]
        log_price = pd.DataFrame(
            {
                "BTCUSDT": 0.05 * t**2,
                "A": 0.01 * t,
                "B": -0.01 * t,
                "C": 0.02 * t,
                "D": 0.03 * t,
                "E": -0.02 * t,
                "F": 0.01 * t,
                "G": -0.03 * t,
            },
            index=idx,
        )
        eligible = pd.DataFrame(True, index=idx, columns=columns)
        w_base = pd.DataFrame(
            {
                "BTCUSDT": 0.0, "A": 0.5, "B": 0.0, "C": 0.0,
                "D": 0.0, "E": -0.5, "F": 0.0, "G": 0.0,
            },
            index=idx,
        )
        tilt = crash_regime_tilt_weights(
            w_base, log_price, eligible, ("BTCUSDT",), 24, 0.5,
        )
        base_gross = w_base.abs().sum(axis=1)
        tilt_gross = tilt.abs().sum(axis=1)
        assert (tilt_gross <= base_gross + 1e-12).all()
        assert (tilt_gross < base_gross - 1e-9).any()
