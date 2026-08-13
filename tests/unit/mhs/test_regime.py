from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.mhs.regime import (
    beta_neutralize_weights,
    causal_market_beta,
    crash_regime_tilt_weights,
    reference_basket_drawdown,
    reference_basket_trend,
    trend_efficiency_scale,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _basket(log_price: pd.DataFrame, reference_symbols: tuple[str, ...]) -> pd.Series:
    return log_price[list(reference_symbols)].mean(axis=1)


class TestReferenceBasketTrend:
    """SCENARIO_MHS_REGIME_TREND_BASIC_01"""

    def test_matches_hand_computed_shifted_lookback(self) -> None:
        log_price = pd.DataFrame(
            {
                "BTCUSDT": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "ETHUSDT": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                "OTHER": [9.0, 9.1, 9.2, 9.3, 9.4, 9.5],
            }
        )
        result = reference_basket_trend(log_price, ("BTCUSDT", "ETHUSDT"), 2)
        assert isinstance(result, pd.Series)
        assert result.index.equals(log_price.index)
        expected = _basket(log_price, ("BTCUSDT", "ETHUSDT")) - _basket(
            log_price, ("BTCUSDT", "ETHUSDT")
        ).shift(2)
        pd.testing.assert_series_equal(result, expected)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(
            _basket(log_price, ("BTCUSDT", "ETHUSDT")).iloc[2]
        )

    def test_fails_closed_on_zero_horizon(self) -> None:
        log_price = pd.DataFrame({"BTCUSDT": [0.0, 1.0], "ETHUSDT": [0.0, 2.0]})
        with pytest.raises(ValueError, match="horizon_bars"):
            reference_basket_trend(log_price, ("BTCUSDT", "ETHUSDT"), 0)


class TestReferenceBasketDrawdown:
    """SCENARIO_MHS_REGIME_DRAWDOWN_BASIC_02"""

    def test_zero_at_peak_then_negative_during_decline(self) -> None:
        ramp = [float(i) for i in range(20)] + [float(39 - i) for i in range(20, 30)]
        log_price = pd.DataFrame(
            {"BTCUSDT": ramp, "ETHUSDT": ramp, "OTHER": [5.0] * 30}
        )
        result = reference_basket_drawdown(log_price, ("BTCUSDT", "ETHUSDT"), 5)
        assert (result <= 1e-12).all()
        expected = _basket(log_price, ("BTCUSDT", "ETHUSDT")) - _basket(
            log_price, ("BTCUSDT", "ETHUSDT")
        ).rolling(5, min_periods=1).max()
        pd.testing.assert_series_equal(result, expected)
        assert result.iloc[:21].abs().max() == pytest.approx(0.0)
        assert result.iloc[21] == pytest.approx(-1.0)
        assert result.iloc[22] == pytest.approx(-2.0)
        assert result.iloc[23] == pytest.approx(-3.0)
        assert result.iloc[29] == pytest.approx(-4.0)

    def test_fails_closed_on_zero_lookback(self) -> None:
        log_price = pd.DataFrame({"BTCUSDT": [0.0, 1.0], "ETHUSDT": [0.0, 2.0]})
        with pytest.raises(ValueError, match="lookback_bars"):
            reference_basket_drawdown(log_price, ("BTCUSDT", "ETHUSDT"), 0)


class TestRegimeValidation:
    """SCENARIO_MHS_REGIME_VALIDATION_03"""

    def test_empty_reference_symbols(self) -> None:
        log_price = pd.DataFrame({"BTCUSDT": [0.0, 1.0]})
        with pytest.raises(ValueError, match="reference_symbols"):
            reference_basket_trend(log_price, (), 2)
        with pytest.raises(ValueError, match="reference_symbols"):
            reference_basket_drawdown(log_price, (), 2)

    def test_unknown_symbol_names_the_missing(self) -> None:
        log_price = pd.DataFrame({"BTCUSDT": [0.0, 1.0]})
        with pytest.raises(ValueError, match=r"ETHUSDT.*DOGEUSDT"):
            reference_basket_trend(log_price, ("BTCUSDT", "ETHUSDT", "DOGEUSDT"), 2)
        with pytest.raises(ValueError, match="DOGEUSDT"):
            reference_basket_drawdown(log_price, ("ETHUSDT", "DOGEUSDT"), 2)


class TestCausalMarketBeta:
    """SCENARIO_MHS_ALPHA_ENGINE_04: rolling OLS beta against the eligible-
    universe equal-weight market, causal, NaN (never inf) on zero market
    variance, and clipped."""

    def test_reproduces_known_synthetic_beta(self) -> None:
        rng = np.random.default_rng(5)
        n = 800
        idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
        market_ret = rng.normal(0.0, 0.001, n)
        # Equal-weight two-symbol market with mean beta 1.0, so the measured
        # beta of A equals its true beta 2.0 exactly (plus tiny idio noise) and
        # B's true beta 0.0 stays near zero.
        rets = pd.DataFrame(
            {
                "A": 2.0 * market_ret + rng.normal(0.0, 1e-4, n),
                "B": 0.0 * market_ret + rng.normal(0.0, 1e-4, n),
            },
            index=idx,
        )
        log_price = rets.cumsum()
        eligible = pd.DataFrame(True, index=idx, columns=log_price.columns)
        beta = causal_market_beta(log_price, eligible, lookback_bars=200, min_periods=100)
        warmed = beta.iloc[200:]
        assert float(warmed["A"].median()) == pytest.approx(2.0, abs=0.15)
        assert float(warmed["B"].median()) == pytest.approx(0.0, abs=0.1)
        assert bool(np.isfinite(warmed.to_numpy()).all())

    def test_zero_variance_market_window_is_nan_never_inf(self) -> None:
        n = 60
        idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
        # All-zero one-bar returns: the equal-weight market is exactly constant,
        # its rolling variance is exactly 0.0, and the beta must be NaN (never
        # inf and never a spurious clipped value).
        rets = pd.DataFrame({"A": [0.0] * n, "B": [0.0] * n}, index=idx)
        log_price = rets.cumsum()
        eligible = pd.DataFrame(True, index=idx, columns=log_price.columns)
        beta = causal_market_beta(log_price, eligible, lookback_bars=20, min_periods=10)
        assert not np.isinf(beta.to_numpy()).any()
        assert beta.isna().all().all()

    def test_clips_to_bounded_range(self) -> None:
        rng = np.random.default_rng(8)
        n = 500
        idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
        market_ret = rng.normal(0.0, 0.001, n)
        rets = pd.DataFrame(
            {"A": 20.0 * market_ret + rng.normal(0.0, 1e-4, n), "B": 0.0 * market_ret + rng.normal(0.0, 1e-4, n)},
            index=idx,
        )
        log_price = rets.cumsum()
        eligible = pd.DataFrame(True, index=idx, columns=log_price.columns)
        beta = causal_market_beta(log_price, eligible, lookback_bars=200, min_periods=100, clip_abs=3.0)
        assert float(beta.abs().max().max()) <= 3.0

    def test_is_causal_and_reads_no_bar_after_t(self) -> None:
        rng = np.random.default_rng(11)
        n = 400
        idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
        market_ret = rng.normal(0.0, 0.001, n)
        rets = pd.DataFrame(
            {"A": 2.0 * market_ret + rng.normal(0.0, 1e-4, n), "B": -0.5 * market_ret},
            index=idx,
        )
        log_price = rets.cumsum()
        eligible = pd.DataFrame(True, index=idx, columns=log_price.columns)
        base = causal_market_beta(log_price, eligible, lookback_bars=100, min_periods=50)
        shocked = log_price.copy()
        shocked.iloc[-20:] += 100.0
        future = causal_market_beta(shocked, eligible, lookback_bars=100, min_periods=50)
        pd.testing.assert_frame_equal(base.iloc[:-20], future.iloc[:-20])
        assert not base.iloc[-20:].equals(future.iloc[-20:])

    def test_validation(self) -> None:
        idx = pd.date_range("2021-01-01", periods=50, freq="1h", tz="UTC")
        log_price = pd.DataFrame({"A": np.arange(50.0), "B": np.arange(50.0) * 2.0}, index=idx)
        eligible = pd.DataFrame(True, index=idx, columns=log_price.columns)
        with pytest.raises(ValueError, match="lookback_bars"):
            causal_market_beta(log_price, eligible, 1, 1)
        with pytest.raises(ValueError, match="min_periods"):
            causal_market_beta(log_price, eligible, 20, 1)
        with pytest.raises(ValueError, match="min_periods"):
            causal_market_beta(log_price, eligible, 10, 20)
        with pytest.raises(ValueError, match="clip_abs"):
            causal_market_beta(log_price, eligible, 20, 10, clip_abs=0.0)
        bad_elig = eligible.rename(columns={"B": "C"})
        with pytest.raises(ValueError, match="identically indexed"):
            causal_market_beta(log_price, bad_elig, 20, 10)


class TestBetaNeutralizeWeights:
    """SCENARIO_MHS_ALPHA_ENGINE_05: the book is projected onto the subspace
    orthogonal to both the constant vector and beta, so sum(w)==0 and
    sum(w*beta)==0 hold by construction; degenerate rows fail closed to zeros."""

    def test_removes_beta_exposure_and_keeps_dollar_neutral(self) -> None:
        w = pd.DataFrame(
            {"A": [0.4, 0.2], "B": [-0.3, 0.3], "C": [0.1, -0.4], "D": [-0.2, -0.1]},
        )
        b = pd.DataFrame(
            {"A": [1.0, 2.0], "B": [0.5, 1.0], "C": [2.0, 0.5], "D": [1.5, 3.0]},
        )
        m = pd.DataFrame(True, index=w.index, columns=w.columns)
        out = beta_neutralize_weights(w, b, m, 2)
        assert out.sum(axis=1).abs().max() < 1e-12
        assert (out * b).sum(axis=1).abs().max() < 1e-10
        assert out.abs().sum(axis=1).sub(1.0).abs().max() < 1e-9

    def test_fails_closed_on_zero_beta_dispersion_or_all_nan_beta(self) -> None:
        w = pd.DataFrame({"A": [0.5], "B": [-0.5], "C": [0.0], "D": [0.0]})
        flat = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [1.0], "D": [1.0]})
        m = pd.DataFrame({"A": [True], "B": [True], "C": [True], "D": [True]})
        out = beta_neutralize_weights(w, flat, m, 2)
        assert out.to_numpy().tolist() == [[0.0, 0.0, 0.0, 0.0]]
        all_nan = pd.DataFrame(
            {"A": [np.nan], "B": [np.nan], "C": [np.nan], "D": [np.nan]},
        )
        out_nan = beta_neutralize_weights(w, all_nan, m, 2)
        assert out_nan.to_numpy().tolist() == [[0.0, 0.0, 0.0, 0.0]]

    def test_fails_closed_below_min_symbols(self) -> None:
        w = pd.DataFrame({"A": [0.5], "B": [-0.5], "C": [0.0], "D": [0.0]})
        b = pd.DataFrame({"A": [1.0], "B": [1.5], "C": [2.0], "D": [0.5]})
        m = pd.DataFrame({"A": [True], "B": [True], "C": [False], "D": [False]})
        out = beta_neutralize_weights(w, b, m, 4)
        assert out.to_numpy().tolist() == [[0.0, 0.0, 0.0, 0.0]]

    def test_ignores_masked_out_columns(self) -> None:
        w = pd.DataFrame({"A": [0.4], "B": [-0.4], "C": [0.2], "D": [-0.2]})
        b = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [5.0], "D": [6.0]})
        m = pd.DataFrame({"A": [True], "B": [True], "C": [False], "D": [False]})
        out = beta_neutralize_weights(w, b, m, 2)
        assert out.iloc[0]["C"] == 0.0
        assert out.iloc[0]["D"] == 0.0
        assert abs(float(out.iloc[0][["A", "B"]].sum())) < 1e-9
        assert abs((out.iloc[0][["A", "B"]] * b.iloc[0][["A", "B"]]).sum()) < 1e-10

    def test_fails_closed_on_misaligned_frames(self) -> None:
        w = pd.DataFrame({"A": [0.5], "B": [-0.5]})
        b = pd.DataFrame({"A": [1.0], "C": [1.0]})
        m = pd.DataFrame({"A": [True], "B": [True]})
        with pytest.raises(ValueError, match="identically indexed"):
            beta_neutralize_weights(w, b, m, 2)
        with pytest.raises(ValueError, match="min_symbols"):
            beta_neutralize_weights(
                pd.DataFrame({"A": [0.5]}), pd.DataFrame({"A": [1.0]}),
                pd.DataFrame({"A": [True]}), 1,
            )


class TestNoProductionWiring:
    """Wiring guard after mhs_crash_regime_tilt_overlay: the ONLY production
    call site that may import src.mhs.regime is the fold-target-weights path in
    src/application/research/mhs/evaluation.py (deliberately scoped to
    ``_build_fold_target_weights``; the diagnostic-only ``blend_1h`` and the
    books/discovery layers stay untouched per docs/specs/mhs_crash_regime_tilt_overlay.md
    §5). The non-wired production layers below must never import it.
    """

    _PRODUCTION_FILES = (
        "src/mhs/books.py",
        "src/mhs/discovery.py",
    )

    def test_no_production_module_imports_regime(self) -> None:
        for rel in self._PRODUCTION_FILES:
            content = (_REPO_ROOT / rel).read_text()
            assert "src.mhs.regime" not in content, rel
            assert re.search(r"import\s+regime\b", content) is None, rel

class TestTrendEfficiencyScale:
    """SCENARIO_MHS_TREND_EFFICIENCY_SCALE_NEVER_LEVERS_UP_01
    SCENARIO_MHS_TREND_EFFICIENCY_SCALE_DERISKS_ON_LOW_ER_02
    SCENARIO_MHS_TREND_EFFICIENCY_SCALE_NEVER_ZERO_DIV_03"""

    def _series(self, n: int = 800) -> pd.Series:
        idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
        return pd.Series(0.4, index=idx)

    def test_never_levers_above_one(self) -> None:
        rng = np.random.default_rng(20260807)
        mean_er = pd.Series(rng.uniform(0.1, 0.9, 800), index=self._series().index)
        mean_er.iloc[700:] = 0.9
        out = trend_efficiency_scale(mean_er, window_hours=720, floor=0.5)
        assert (out <= 1.0 + 1e-12).all()
        assert not out.isna().all()

    def test_derisks_when_er_drops_below_own_median(self) -> None:
        mean_er = self._series()
        mean_er.iloc[750:] = 0.05
        out = trend_efficiency_scale(mean_er, window_hours=720, floor=0.5)
        assert float(out.iloc[799]) < 1.0
        assert float(out.iloc[799]) >= 0.5
        assert float(out.iloc[799]) == pytest.approx(0.5)

    def test_flat_series_is_full_exposure(self) -> None:
        out = trend_efficiency_scale(self._series(), window_hours=720, floor=0.5)
        assert (out.dropna() == 1.0).all()

    def test_insufficient_history_and_all_nan_are_full_exposure(self) -> None:
        short = pd.Series(
            0.3, index=pd.date_range("2021-01-01", periods=10, freq="1h", tz="UTC"),
        )
        out_short = trend_efficiency_scale(short, window_hours=720, floor=0.5)
        assert (out_short == 1.0).all()

        all_nan = pd.Series(
            np.nan, index=pd.date_range("2021-01-01", periods=50, freq="1h", tz="UTC"),
        )
        out_nan = trend_efficiency_scale(all_nan, window_hours=720, floor=0.5)
        assert (out_nan == 1.0).all()

    def test_empty_input_returns_empty(self) -> None:
        out = trend_efficiency_scale(pd.Series(dtype="float64"), floor=0.5)
        assert out.empty

    def test_validation(self) -> None:
        mean_er = self._series()
        with pytest.raises(ValueError, match="floor"):
            trend_efficiency_scale(mean_er, floor=0.0)
        with pytest.raises(ValueError, match="floor"):
            trend_efficiency_scale(mean_er, floor=1.5)
        with pytest.raises(ValueError, match="window_hours"):
            trend_efficiency_scale(mean_er, window_hours=0)


def _tilt_fixture(bars: int = 120, horizon: int = 24):
    """Synthetic deterministic panel for tilt unit tests.

    BTCUSDT grows quadratically (``0.05 * t**2``), so its ``horizon``-bar
    trend is strictly positive and increasing, and once the rolling vol window
    warms up the self-normalized z-score clips to exactly ``+1.0`` on every
    remaining row. ``rank_neutral_weights`` is a dollar-neutral long/short
    book (unit gross, zero net) over an 8-symbol roster -- >= the function's
    default ``min_symbols=8``, so the tilt stays active. ``eligible`` marks the
    full roster active.
    """
    idx = pd.date_range("2021-01-01", periods=bars, freq="1h", tz="UTC")
    t = np.arange(bars, dtype=float)
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
    rank_neutral = pd.DataFrame(
        {"BTCUSDT": 0.0, "A": 0.5, "B": 0.0, "C": 0.0, "D": 0.0, "E": -0.5, "F": 0.0, "G": 0.0},
        index=idx,
    )
    return log_price, eligible, rank_neutral


class TestCrashRegimeTiltWeights:
    """SCENARIO_MHS_CRASH_TILT_ALPHA_ZERO_IDENTITY_01
    SCENARIO_MHS_CRASH_TILT_ALPHA_ONE_PURE_TILT_02
    SCENARIO_MHS_CRASH_TILT_GROSS_INVARIANT_03
    SCENARIO_MHS_CRASH_TILT_VALIDATION_04"""

    def test_alpha_zero_returns_input_unchanged(self) -> None:
        log_price, eligible, w = _tilt_fixture()
        out = crash_regime_tilt_weights(
            w, log_price, eligible, ("BTCUSDT",), 24, 0.0,
        )
        pd.testing.assert_frame_equal(out, w)

    def test_alpha_one_is_the_pure_tilt_book(self) -> None:
        log_price, eligible, w = _tilt_fixture()
        horizon = 24
        out = crash_regime_tilt_weights(
            w, log_price, eligible, ("BTCUSDT",), horizon, 1.0,
        )
        trend = reference_basket_trend(log_price, ("BTCUSDT",), horizon)
        trend_vol = trend.rolling(horizon, min_periods=horizon).std()
        trend_z = (trend / trend_vol.where(trend_vol > 0)).clip(-1.0, 1.0).fillna(0.0)
        n_active = eligible.sum(axis=1)
        expected_tilt = (
            eligible.astype("float64")
            .div(n_active.where(n_active > 0), axis=0)
            .mul(trend_z, axis=0)
            .fillna(0.0)
        )
        pd.testing.assert_frame_equal(out, expected_tilt)
        clipped = trend_z.abs() > 0
        assert clipped.any(), "fixture must reach the clipped +1.0 regime"
        for idx in log_price.index[clipped]:
            row = out.loc[idx]
            assert (row.to_numpy() > 0).all(), "pure tilt must follow the up-trend sign"
            assert row.to_numpy().sum() == pytest.approx(1.0)
            assert (row == row.iloc[0]).all(), "tilt allocates uniformly across the roster"

    def test_gross_invariant_across_intermediate_alpha(self) -> None:
        log_price, eligible, _w = _tilt_fixture()
        # The convex-combination invariant holds exactly when the blended books
        # are sign-consistent per column (both non-negative here): then
        # |(1-a)w + a*tilt|_1 = (1-a)|w|_1 + a|tilt|_1 = 1.0 on active rows.
        w_aligned = _w.abs()
        horizon = 24
        trend = reference_basket_trend(log_price, ("BTCUSDT",), horizon)
        trend_vol = trend.rolling(horizon, min_periods=horizon).std()
        trend_z = (trend / trend_vol.where(trend_vol > 0)).clip(-1.0, 1.0).fillna(0.0)
        active = trend_z.abs() > 0
        for alpha in (0.1, 0.3, 0.7):
            out = crash_regime_tilt_weights(
                w_aligned, log_price, eligible, ("BTCUSDT",), horizon, alpha,
            )
            gross = out.abs().sum(axis=1)
            assert gross.where(active).sub(1.0).abs().max() < 1e-9

    def test_gross_never_exceeds_unit_when_books_oppose(self) -> None:
        # SCENARIO_MHS_CRASH_TILT_GROSS_INVARIANT_03 (conservative side): a
        # genuine dollar-neutral book has shorts opposing the long tilt, so by
        # the triangle inequality the blend gross is bounded above by 1.0 --
        # the tilt offsets shorts instead of amplifying them.
        log_price, eligible, w_neutral = _tilt_fixture()
        horizon = 24
        for alpha in (0.1, 0.3, 0.7):
            out = crash_regime_tilt_weights(
                w_neutral, log_price, eligible, ("BTCUSDT",), horizon, alpha,
            )
            gross = out.abs().sum(axis=1)
            assert gross.max() <= 1.0 + 1e-9

    def test_validation(self) -> None:
        log_price, eligible, w = _tilt_fixture()
        with pytest.raises(ValueError, match="alpha"):
            crash_regime_tilt_weights(w, log_price, eligible, ("BTCUSDT",), 24, -0.1)
        with pytest.raises(ValueError, match="alpha"):
            crash_regime_tilt_weights(w, log_price, eligible, ("BTCUSDT",), 24, 1.1)
        with pytest.raises(ValueError, match="index"):
            crash_regime_tilt_weights(
                w.iloc[:5], log_price, eligible, ("BTCUSDT",), 24, 0.2,
            )
        mismatched = eligible.rename(columns={"A": "OTHER"})
        with pytest.raises(ValueError, match="eligible"):
            crash_regime_tilt_weights(
                w, log_price, mismatched, ("BTCUSDT",), 24, 0.2,
            )
        with pytest.raises(ValueError, match="min_symbols"):
            crash_regime_tilt_weights(
                w, log_price, eligible, ("BTCUSDT",), 24, 0.2, min_symbols=1,
            )
