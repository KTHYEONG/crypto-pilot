"""Contract coverage for the MHS compounding growth feature."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.mhs.evaluation import (
    MhsDiagnosticRequest,
    _committee_execution_book,
)
from src.mhs.scaling import (
    _committee_capital_replay_scale,
    _exante_vol_target_scale,
    _pnl_vol_target_scale,
    _replay_exposure_scale,
)


class TestExanteVolTargetScale:
    """SCENARIO_EXANTE_SCALE_HITS_TARGET_VOL / NEVER_LEVERS_UP / CAUSAL_SHIFT / FAILS_CLOSED."""

    def test_hits_target_vol(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-01", periods=600, freq="1D", tz="UTC")
        returns = pd.Series(rng.normal(0, 0.02, 600), index=idx)
        scale = _exante_vol_target_scale(returns, target_vol=0.20)
        sigma = returns.ewm(halflife=20, min_periods=90).std().shift(1) * np.sqrt(365.0)
        expected_last = 0.20 / sigma.iloc[-1]
        assert abs(scale.iloc[-1] - expected_last) < 1e-12
        assert scale.iloc[-1] > 0.2
        assert scale.iloc[-1] < 1.0

    def test_never_levers_up(self) -> None:
        idx = pd.date_range("2021-01-01", periods=400, freq="1D", tz="UTC")
        # Constant series -> zero std -> sigma=0 -> fillna(1.0) (fail open)
        low_vol = pd.Series(0.001, index=idx)
        scale_low = _exante_vol_target_scale(low_vol, target_vol=0.20)
        assert (scale_low == 1.0).all()
        assert np.all(np.isfinite(scale_low))
        assert scale_low.max() <= 1.0

        # High vol with actual variation -> floor (0.2)
        rng = np.random.default_rng(99)
        high_vol = pd.Series(rng.normal(0, 0.15, 400), index=idx)  # annualized ~2.87
        scale_high = _exante_vol_target_scale(high_vol, target_vol=0.20)
        # Non-warmup rows hit the floor
        assert scale_high.iloc[100:].min() >= 0.2 - 1e-12
        assert np.all(np.isfinite(scale_high))
        assert scale_high.max() <= 1.0

    def test_causal_shift(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-01", periods=400, freq="1D", tz="UTC")
        returns = pd.Series(rng.normal(0, 0.02, 400), index=idx)
        baseline = _exante_vol_target_scale(returns, target_vol=0.20)
        perturbed = returns.copy()
        perturbed.iloc[-1] = 1e6
        after = _exante_vol_target_scale(perturbed, target_vol=0.20)
        pd.testing.assert_series_equal(baseline, after)

    def test_fails_closed(self) -> None:
        idx = pd.date_range("2021-01-01", periods=400, freq="1D", tz="UTC")
        returns = pd.Series(0.01, index=idx)
        with pytest.raises(ValueError, match="target_vol"):
            _exante_vol_target_scale(returns, target_vol=0.0)
        with pytest.raises(ValueError, match="target_vol"):
            _exante_vol_target_scale(returns, target_vol=-0.1)
        with pytest.raises(ValueError, match="halflife_days"):
            _exante_vol_target_scale(returns, halflife_days=0)
        with pytest.raises(ValueError, match="min_days"):
            _exante_vol_target_scale(returns, min_days=0)
        with pytest.raises(ValueError, match="floor"):
            _exante_vol_target_scale(returns, floor=0.0)
        with pytest.raises(ValueError, match="floor"):
            _exante_vol_target_scale(returns, floor=1.5)

    def test_empty_series(self) -> None:
        empty = pd.Series(dtype=float)
        result = _exante_vol_target_scale(empty)
        assert result.empty

    def test_zero_variance_fails_open(self) -> None:
        idx = pd.date_range("2021-01-01", periods=400, freq="1D", tz="UTC")
        zero = pd.Series(0.0, index=idx)
        result = _exante_vol_target_scale(zero)
        assert (result == 1.0).all()
        assert float(result.iloc[-1]) == 1.0


class TestExanteVolTargetScaleWarmup:
    """SCENARIO_MHS_FOLD_WARMUP_BURN_IN_DEAD_ZONE_ELIMINATED / _LEAK_GUARD.

    Root cause of the real-pipeline FOLD_BLEND_PATH_DIVERGENCE regression:
    min_days=90 (PNL_VOL_TARGET_BURN_IN_DAYS) exceeds a quarterly fold's own
    ~83-91 row validation window, so every row fell back to scale=1.0 without
    warmup_returns while the continuously-running blend never hit that wall.

    SCENARIO_MHS_FOLD_BLEND_DIVERGENCE_RESOLVED_REAL_PIPELINE is verified
    separately by a real full-pipeline run (not a unit test here): pre-fix
    fold_blend_parity.max_abs_log_deployed_gross_ratio=0.836 (over the 0.25
    tolerance, FOLD_BLEND_PATH_DIVERGENCE in research_go.reason_codes) ->
    post-fix 0.180 (under tolerance, code no longer present). See
    docs/specs/mhs_fold_exposure_warmup.md section 5.
    """

    def test_warmup_eliminates_burn_in_dead_zone(self) -> None:
        rng = np.random.default_rng(0)
        warmup_idx = pd.date_range("2021-01-01", periods=370, freq="1D", tz="UTC")
        warmup = pd.Series(rng.normal(0, 0.02, len(warmup_idx)), index=warmup_idx)
        fold_idx = pd.date_range(
            warmup_idx[-1] + pd.Timedelta(days=1), periods=85, freq="1D", tz="UTC",
        )
        fold_returns = pd.Series(rng.normal(0, 0.02, len(fold_idx)), index=fold_idx)

        no_warmup = _exante_vol_target_scale(fold_returns, target_vol=0.20)
        assert (no_warmup == 1.0).mean() == 1.0  # today's bug: 100% fallback

        with_warmup = _exante_vol_target_scale(fold_returns, target_vol=0.20, warmup_returns=warmup)
        assert (with_warmup == 1.0).mean() < 0.5

    def test_warmup_matches_continuous_history_bit_identical(self) -> None:
        rng = np.random.default_rng(1)
        warmup_idx = pd.date_range("2021-01-01", periods=370, freq="1D", tz="UTC")
        warmup = pd.Series(rng.normal(0, 0.02, len(warmup_idx)), index=warmup_idx)
        fold_idx = pd.date_range(
            warmup_idx[-1] + pd.Timedelta(days=1), periods=85, freq="1D", tz="UTC",
        )
        fold_returns = pd.Series(rng.normal(0, 0.02, len(fold_idx)), index=fold_idx)

        with_warmup = _exante_vol_target_scale(fold_returns, target_vol=0.20, warmup_returns=warmup)
        full_history = pd.concat([warmup, fold_returns])
        continuous_reference = _exante_vol_target_scale(full_history, target_vol=0.20).loc[fold_idx]
        pd.testing.assert_series_equal(with_warmup, continuous_reference)

    def test_warmup_none_is_backward_compatible(self) -> None:
        # SCENARIO_MHS_FOLD_WARMUP_BACKWARD_COMPAT
        idx = pd.date_range("2021-01-01", periods=600, freq="1D", tz="UTC")
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 600), index=idx)
        without_kwarg = _exante_vol_target_scale(returns, target_vol=0.20)
        with_none = _exante_vol_target_scale(returns, target_vol=0.20, warmup_returns=None)
        pd.testing.assert_series_equal(without_kwarg, with_none)

    def test_warmup_leak_guard_raises(self) -> None:
        # SCENARIO_MHS_FOLD_WARMUP_LEAK_GUARD
        idx = pd.date_range("2021-01-01", periods=100, freq="1D", tz="UTC")
        returns = pd.Series(0.01, index=idx)
        overlapping_warmup = pd.Series(0.01, index=idx[50:60])
        with pytest.raises(ValueError, match="must precede"):
            _exante_vol_target_scale(returns, warmup_returns=overlapping_warmup)


class TestExanteVolTargetScaleCap:
    """SCENARIO_MHS_FILL_MARK_PARITY_03: _exante_vol_target_scale cap parameter."""

    def test_default_cap_levers_to_one(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-01", periods=400, freq="1D", tz="UTC")
        low_vol = pd.Series(rng.normal(0, 0.005, 400), index=idx)  # annualized ~0.095
        scale_default = _exante_vol_target_scale(low_vol, target_vol=0.20)
        assert scale_default.max() == pytest.approx(1.0)
        post_burn = scale_default.iloc[90:]
        assert (post_burn == 1.0).all()

    def test_cap_two_sided(self) -> None:
        from src.mhs.types import PNL_VOL_TARGET_MAX_SCALE

        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-01", periods=400, freq="1D", tz="UTC")
        low_vol = pd.Series(rng.normal(0, 0.005, 400), index=idx)
        scale_capped = _exante_vol_target_scale(low_vol, target_vol=0.20, cap=PNL_VOL_TARGET_MAX_SCALE)
        post_burn = scale_capped.iloc[90:]
        assert (post_burn > 1.0).all()
        assert scale_capped.max() <= PNL_VOL_TARGET_MAX_SCALE + 1e-12

    def test_high_vol_both_identical(self) -> None:
        from src.mhs.types import PNL_VOL_TARGET_MAX_SCALE

        rng = np.random.default_rng(99)
        idx = pd.date_range("2021-01-01", periods=400, freq="1D", tz="UTC")
        high_vol = pd.Series(rng.normal(0, 0.15, 400), index=idx)
        scale_default = _exante_vol_target_scale(high_vol, target_vol=0.20)
        scale_capped = _exante_vol_target_scale(high_vol, target_vol=0.20, cap=PNL_VOL_TARGET_MAX_SCALE)
        pd.testing.assert_series_equal(scale_default, scale_capped)

    def test_cap_below_one_raises(self) -> None:
        idx = pd.date_range("2021-01-01", periods=10, freq="1D", tz="UTC")
        data = pd.Series(0.01, index=idx)
        with pytest.raises(ValueError, match="cap"):
            _exante_vol_target_scale(data, target_vol=0.20, cap=0.9)


class TestReplayExposureScale:
    """SCENARIO_REPLAY_EXPOSURE_SCALE_DEFAULT_BYTE_IDENTICAL."""

    def test_default_byte_identical(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-01", periods=400, freq="1D", tz="UTC")
        daily = pd.Series(rng.normal(0, 0.02, 400), index=idx)
        request = MhsDiagnosticRequest()
        result = _replay_exposure_scale(daily, request)
        expected = _committee_capital_replay_scale(
            _pnl_vol_target_scale(daily), daily,
            request.committee_capital, request.committee_kelly_sizing,
        )
        pd.testing.assert_series_equal(result, expected)

    def test_exante_target_mode(self) -> None:
        # SCENARIO_MHS_FOLD_WARMUP_CONSERVATIVE_ENVELOPE_UNCHANGED: default
        # request resolves the I4-protected conservative envelope, which must
        # stay byte-identical (no warmup_returns wiring) after the fix.
        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-01", periods=400, freq="1D", tz="UTC")
        daily = pd.Series(rng.normal(0, 0.02, 400), index=idx)
        request = MhsDiagnosticRequest(pnl_vol_target_mode="exante_target")
        result = _replay_exposure_scale(daily, request)
        expected = _committee_capital_replay_scale(
            _exante_vol_target_scale(daily), daily,
            request.committee_capital, request.committee_kelly_sizing,
        )
        pd.testing.assert_series_equal(result, expected)

    def test_unknown_mode_raises(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-01", periods=400, freq="1D", tz="UTC")
        daily = pd.Series(rng.normal(0, 0.02, 400), index=idx)
        request = MhsDiagnosticRequest()
        object.__setattr__(request, "pnl_vol_target_mode", "nope")
        with pytest.raises(ValueError, match="unknown pnl_vol_target_mode"):
            _replay_exposure_scale(daily, request)


class TestCommitteeBookCarryMix:
    """SCENARIO_COMMITTEE_BOOK_CARRY_MIX_PRESERVES_INVARIANTS."""

    def _mock_build(self, decision_grid, cols):
        """Return a mock build_feature_books that yields a dollar-neutral unit-gross book."""
        def _build(specs, data, mask, grid, min_symbols=8, coverage_cutoff=None):
            # Simple dollar-neutral book: long A, short B, zero C
            n = len(grid)
            book = pd.DataFrame(0.0, index=grid, columns=cols)
            book[cols[0]] = 0.5
            book[cols[1]] = -0.5
            return {"mock_member": book}
        return _build

    def test_carry_mix_preserves_invariants(self) -> None:
        idx = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")
        cols = ["A", "B", "C"]
        rng = np.random.default_rng(42)
        close = pd.DataFrame(rng.uniform(90, 110, (400, 3)), index=idx, columns=cols)
        quote_vol = pd.DataFrame(1000.0, index=idx, columns=cols)
        taker_buy = pd.DataFrame(500.0, index=idx, columns=cols)
        mask = pd.DataFrame(True, index=idx, columns=cols)
        decision_grid = idx[::24]
        # Dollar-neutral carry book: long A, short B, zero C
        carry = pd.DataFrame(0.0, index=idx, columns=cols)
        carry["A"] = 0.5
        carry["B"] = -0.5

        with patch(
            "src.mhs.evaluation.build_feature_books",
            side_effect=self._mock_build(decision_grid, cols),
        ):
            result = _committee_execution_book(
                close, quote_vol, taker_buy, mask, decision_grid,
                min_symbols=2, tranche_count=1, target_gross=0.92,
                carry_book=carry, carry_weight=0.30,
            )
        assert (result.sum(axis=1).abs() < 1e-12).all()
        assert ((result.abs().sum(axis=1) - 0.92).abs() < 1e-12).all()

    def test_carry_weight_zero_identical(self) -> None:
        idx = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")
        cols = ["A", "B", "C"]
        rng = np.random.default_rng(42)
        close = pd.DataFrame(rng.uniform(90, 110, (400, 3)), index=idx, columns=cols)
        quote_vol = pd.DataFrame(1000.0, index=idx, columns=cols)
        taker_buy = pd.DataFrame(500.0, index=idx, columns=cols)
        mask = pd.DataFrame(True, index=idx, columns=cols)
        decision_grid = idx[::24]
        carry = pd.DataFrame(0.0, index=idx, columns=cols)
        carry["A"] = 0.5
        carry["B"] = -0.5

        with patch(
            "src.mhs.evaluation.build_feature_books",
            side_effect=self._mock_build(decision_grid, cols),
        ):
            without_carry = _committee_execution_book(
                close, quote_vol, taker_buy, mask, decision_grid,
                min_symbols=2, tranche_count=1, target_gross=0.92,
            )
            with_zero_weight = _committee_execution_book(
                close, quote_vol, taker_buy, mask, decision_grid,
                min_symbols=2, tranche_count=1, target_gross=0.92,
                carry_book=carry, carry_weight=0.0,
            )
        pd.testing.assert_frame_equal(without_carry, with_zero_weight)

    def test_carry_requires_target_gross(self) -> None:
        idx = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")
        cols = ["A", "B", "C"]
        rng = np.random.default_rng(42)
        close = pd.DataFrame(rng.uniform(90, 110, (400, 3)), index=idx, columns=cols)
        quote_vol = pd.DataFrame(1000.0, index=idx, columns=cols)
        taker_buy = pd.DataFrame(500.0, index=idx, columns=cols)
        mask = pd.DataFrame(True, index=idx, columns=cols)
        decision_grid = idx[::24]
        carry = pd.DataFrame(0.0, index=idx, columns=cols)
        carry["A"] = 0.5
        carry["B"] = -0.5

        with patch(
            "src.mhs.evaluation.build_feature_books",
            side_effect=self._mock_build(decision_grid, cols),
        ):
            with pytest.raises(ValueError, match="target_gross"):
                _committee_execution_book(
                    close, quote_vol, taker_buy, mask, decision_grid,
                    min_symbols=2, tranche_count=1, target_gross=None,
                    carry_book=carry, carry_weight=0.30,
                )
            with pytest.raises(ValueError, match="carry_weight"):
                _committee_execution_book(
                    close, quote_vol, taker_buy, mask, decision_grid,
                    min_symbols=2, tranche_count=1, target_gross=0.92,
                    carry_book=carry, carry_weight=1.0,
                )
            # carry_weight < 0 is validated at MhsDiagnosticRequest level, not here
            # (the function guard is `carry_weight > 0.0`, so negative weight is inert)


class TestRequestValidationCarrySleeve:
    """SCENARIO_REQUEST_VALIDATION_CARRY_SLEEVE."""

    def test_carry_sleeve_requires_committee_capital(self) -> None:
        with pytest.raises(ValueError, match="funding_carry_sleeve requires committee_capital"):
            MhsDiagnosticRequest(funding_carry_sleeve=True)

    def test_carry_weight_requires_sleeve(self) -> None:
        with pytest.raises(ValueError, match=r"funding_carry_weight > 0.0 requires funding_carry_sleeve"):
            MhsDiagnosticRequest(funding_carry_weight=0.3)

    def test_carry_sleeve_requires_target_gross(self) -> None:
        with pytest.raises(ValueError, match="funding_carry_sleeve is mutually exclusive"):
            MhsDiagnosticRequest(
                committee_capital=True,
                funding_carry_sleeve=True,
                committee_target_gross=None,
            )

    def test_unknown_pnl_vol_target_mode(self) -> None:
        with pytest.raises(ValueError, match="unknown pnl_vol_target_mode"):
            MhsDiagnosticRequest(pnl_vol_target_mode="nope")

    def test_defaults(self) -> None:
        req = MhsDiagnosticRequest()
        assert req.pnl_vol_target_mode == "median_relative"
        assert req.funding_carry_sleeve is False
        assert req.funding_carry_weight == 0.0
