from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.mhs.regime import (
    crash_regime_tilt_weights,
    reference_basket_drawdown,
    reference_basket_trend,
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
