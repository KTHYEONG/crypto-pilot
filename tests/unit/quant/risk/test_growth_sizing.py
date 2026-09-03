from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.mhs.committee import growth_budget_annual_vol
from src.mhs.params import GROWTH_RISK_ENVELOPES
from src.research.risk.growth_sizing import (
    FrontierScanPoint,
    GrowthHeadroomDiagnostic,
    GrowthSizingConfig,
    GrowthSizingResult,
    apply_realised_risk_overlay,
    apply_vol_target_overlay,
    compute_discovery_target_vol,
    diagnose_growth_headroom,
    drawdown_risk_multiplier,
    scan_leverage_frontier,
    solve_growth_optimal_risk,
)


def _crash_stream(seed: int = 42, n: int = 4000) -> np.ndarray:
    """Positive-drift returns with rare large drawdown shocks."""
    rng = np.random.default_rng(seed)
    return np.where(
        rng.random(n) < 0.001,
        rng.normal(-0.08, 0.01, n),
        rng.normal(0.0012, 0.004, n),
    )


class TestGrowthSizingConfig:
    def test_defaults(self) -> None:
        config = GrowthSizingConfig(risk_grid=(0.0005, 0.001, 0.005))
        assert config.bars_per_year == 2190
        assert config.max_drawdown == 0.20
        assert config.plateau_fraction == 0.95
        assert config.reference_risk == 0.005

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"risk_grid": ()},
            {"risk_grid": (0.005, 0.001)},
            {"risk_grid": (0.001, 0.001)},
            {"reference_risk": 0.0},
            {"reference_risk": -1.0},
            {"n_paths": 99},
            {"max_drawdown_prob": 0.0},
            {"max_ruin_prob": 0.0},
            {"max_ruin_prob": 1.0},
        ],
    )
    def test_validation_raises(self, kwargs: dict[str, object]) -> None:
        kwargs.setdefault("risk_grid", (0.001, 0.002, 0.005))
        with pytest.raises(ValueError, match="must"):
            GrowthSizingConfig(**kwargs)


class TestGrowthSizingResult:
    def test_none_selected_when_infeasible(self) -> None:
        result = GrowthSizingResult(None, 0.0, 1.0, 0.5, (), "infeasible", 19)
        assert result.selected_risk is None
        assert result.feasible_risks == ()


class TestDrawdownRiskMultiplier:
    # GEV2-10-DD-LADDER
    def test_reproduces_published_table(self) -> None:
        dd = np.array([0.0, 0.05, 0.10, 0.15, 0.175, 0.20, 0.30])
        out = drawdown_risk_multiplier(dd)
        assert np.allclose(out, np.array([1.0, 1.0, 0.625, 0.25, 0.125, 0.0, 0.0]))

    def test_bounded_to_unit_interval(self) -> None:
        dd = np.linspace(0.0, 0.50, 101)
        out = drawdown_risk_multiplier(dd)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_monotonically_non_increasing(self) -> None:
        dd = np.linspace(0.0, 0.30, 61)
        out = drawdown_risk_multiplier(dd)
        assert all(out[i] >= out[i + 1] for i in range(len(out) - 1))

    def test_zero_past_the_ladder_end(self) -> None:
        assert np.all(drawdown_risk_multiplier(np.array([0.20, 0.5, 1.0])) == 0.0)

    def test_rejects_negative_drawdown(self) -> None:
        with pytest.raises(ValueError, match="drawdown"):
            drawdown_risk_multiplier(np.array([-0.1]))

    def test_rejects_non_finite_drawdown(self) -> None:
        with pytest.raises(ValueError, match="drawdown"):
            drawdown_risk_multiplier(np.array([0.1, np.nan]))


class TestSolveGrowthOptimalRisk:
    def test_contract_assertion_surface(self) -> None:
        rng = np.random.default_rng(0)
        sample = rng.normal(0.0004, 0.006, 4000)
        config = GrowthSizingConfig(
            risk_grid=(0.001, 0.005, 0.02), horizon_years=1.0, n_paths=200,
        )
        result = solve_growth_optimal_risk(sample, config)
        assert result.selected_risk is None or result.selected_risk in config.risk_grid
        assert result.binding_constraint in ("none", "infeasible")
        assert result.block_size_used >= 1
        assert solve_growth_optimal_risk(sample, config).selected_risk == result.selected_risk

    # GEV2-09-CONSTRAINT-ORDER
    def test_constraints_define_feasible_set_before_plateau_rule(self) -> None:
        rets = _crash_stream()
        strict = GrowthSizingConfig(
            risk_grid=(0.001, 0.003, 0.006, 0.012, 0.025),
            horizon_years=1.0, n_paths=400, seed=0,
            max_drawdown_prob=0.05, max_ruin_prob=0.001,
        )
        relaxed = GrowthSizingConfig(
            risk_grid=(0.001, 0.003, 0.006, 0.012, 0.025),
            horizon_years=1.0, n_paths=400, seed=0,
            max_drawdown_prob=0.99, max_ruin_prob=0.99,
        )
        strict_result = solve_growth_optimal_risk(rets, strict, use_drawdown_overlay=False)
        relaxed_result = solve_growth_optimal_risk(rets, relaxed, use_drawdown_overlay=False)
        assert strict_result.selected_risk is not None
        assert relaxed_result.selected_risk is not None
        assert strict_result.selected_risk < relaxed_result.selected_risk
        assert relaxed_result.selected_risk not in strict_result.feasible_risks

    # GEV2-11-INFEASIBLE-FAILS-CLOSED
    def test_pure_loss_stream_fails_closed(self) -> None:
        config = GrowthSizingConfig(
            risk_grid=(0.001, 0.005, 0.02), horizon_years=1.0, n_paths=200,
        )
        loss = np.full(2000, -0.01)
        result = solve_growth_optimal_risk(loss, config)
        assert result.selected_risk is None
        assert result.binding_constraint == "infeasible"

    def test_feasible_set_drives_ruin_probability(self) -> None:
        # A pure-loss stream with the overlay capped stays under the MDD bound but
        # still never deploys: the selected risk must be None, not a least-bad point.
        config = GrowthSizingConfig(
            risk_grid=(0.001, 0.005, 0.02), horizon_years=1.0, n_paths=200,
        )
        loss = np.full(2000, -0.01)
        result = solve_growth_optimal_risk(loss, config, use_drawdown_overlay=True)
        assert result.selected_risk is None

    def test_seeded_reproducibility(self) -> None:
        rng = np.random.default_rng(3)
        sample = rng.normal(0.0005, 0.005, 3000)
        config = GrowthSizingConfig(
            risk_grid=(0.001, 0.005, 0.01), horizon_years=1.0, n_paths=300, seed=7,
        )
        first = solve_growth_optimal_risk(sample, config)
        second = solve_growth_optimal_risk(sample, config)
        assert first == second

    def test_rejects_empty_returns(self) -> None:
        config = GrowthSizingConfig(risk_grid=(0.001,), n_paths=200)
        with pytest.raises(ValueError, match="empty"):
            solve_growth_optimal_risk(np.array([]), config)

    def test_rejects_non_finite_returns(self) -> None:
        config = GrowthSizingConfig(risk_grid=(0.001,), n_paths=200)
        with pytest.raises(ValueError, match="finite"):
            solve_growth_optimal_risk(np.array([0.001, np.nan]), config)

class TestDiagnoseGrowthHeadroom:
    """Observability-only headroom diagnostic over the risk grid (never re-selects)."""

    @staticmethod
    def _plateau_series() -> np.ndarray:
        # Deterministic Kelly-concave normal series (peak scale ~2.0); the config
        # below relaxes the tail gates so the whole grid stays feasible, isolating
        # the plateau rule from tail-risk gating (same relaxed-constraint pattern
        # as test_constraints_define_feasible_set_before_plateau_rule).
        rng = np.random.default_rng(7)
        return rng.normal(8e-4, 2e-2, 50000)

    @staticmethod
    def _plateau_config() -> GrowthSizingConfig:
        return GrowthSizingConfig(
            risk_grid=(1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0),
            reference_risk=1.0, horizon_years=1.0, n_paths=300, seed=0,
            max_drawdown=1.5, max_ruin_prob=0.99,
        )

    # SCENARIO_HEADROOM_NULL_WHEN_INFEASIBLE
    def test_headroom_null_when_infeasible(self) -> None:
        # selected_risk=None short-circuits before the bootstrap: block_size is
        # echoed from the passed result, never re-derived from unit_returns.
        selected = GrowthSizingResult(None, 0.0, 1.0, 0.5, (), "infeasible", 19)
        diag = diagnose_growth_headroom(self._plateau_series(), self._plateau_config(), selected)
        assert isinstance(diag, GrowthHeadroomDiagnostic)
        assert diag.selected_risk is None
        assert diag.selected_median_log_growth == 0.0
        assert diag.peak_feasible_risk is None
        assert diag.peak_feasible_median_log_growth == 0.0
        assert diag.headroom_ratio == 0.0
        assert diag.risk_constrained is False
        assert diag.block_size_used == 19

    # SCENARIO_HEADROOM_ZERO_AT_TRUE_PEAK
    def test_headroom_zero_at_true_peak(self) -> None:
        # A single-point grid: the selected point is the only grid point, so
        # there is nothing higher to compare against -- headroom is exactly 0
        # and nothing can be tail-constrained.
        cfg = GrowthSizingConfig(
            risk_grid=(1.0,), reference_risk=1.0, horizon_years=1.0,
            n_paths=300, seed=0, max_drawdown=1.5, max_ruin_prob=0.99,
        )
        selected = solve_growth_optimal_risk(
            self._plateau_series(), cfg, use_drawdown_overlay=False,
        )
        assert selected.selected_risk == 1.0
        diag = diagnose_growth_headroom(
            self._plateau_series(), cfg, selected, use_drawdown_overlay=False,
        )
        assert diag.headroom_ratio == 0.0
        assert diag.risk_constrained is False
        assert diag.peak_feasible_risk == selected.selected_risk
        assert diag.peak_feasible_median_log_growth == selected.median_log_growth

    # SCENARIO_HEADROOM_DETECTS_TAIL_RISK_CONSTRAINT
    def test_headroom_detects_tail_risk_constraint(self) -> None:
        # v8_sized shape: median log growth climbs monotonically with risk while
        # mdd_breach_prob explodes past the selected point, so the best feasible
        # grid point sits slightly above selection and everything beyond it is
        # walled off by tail risk (not the plateau rule).
        config = GrowthSizingConfig(
            risk_grid=(1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5),
            reference_risk=1.0, horizon_years=1.0, n_paths=400, seed=0,
        )
        selected = solve_growth_optimal_risk(
            _crash_stream(), config, use_drawdown_overlay=False,
        )
        assert selected.selected_risk is not None
        diag = diagnose_growth_headroom(
            _crash_stream(), config, selected, use_drawdown_overlay=False,
        )
        assert diag.risk_constrained is True
        assert diag.headroom_ratio > 0.0
        assert diag.peak_feasible_risk is not None
        assert diag.peak_feasible_risk > diag.selected_risk
        assert diag.selected_risk == selected.selected_risk
        assert diag.selected_median_log_growth == selected.median_log_growth

    # SCENARIO_HEADROOM_PLATEAU_NOT_RISK_CONSTRAINED
    def test_headroom_plateau_not_risk_constrained(self) -> None:
        # v6_growth_sized shape: every higher grid point stays feasible and the
        # median declines past the peak, so the selected point already captures
        # ~99% of the best feasible median and nothing is tail-constrained.
        config = self._plateau_config()
        selected = solve_growth_optimal_risk(
            self._plateau_series(), config, use_drawdown_overlay=False,
        )
        assert selected.selected_risk is not None
        diag = diagnose_growth_headroom(
            self._plateau_series(), config, selected, use_drawdown_overlay=False,
        )
        assert diag.risk_constrained is False
        assert 0.0 < diag.headroom_ratio < 0.05
        assert len(selected.feasible_risks) == len(config.risk_grid)
        assert diag.selected_risk == selected.selected_risk
        assert diag.selected_median_log_growth == selected.median_log_growth

    # SCENARIO_HEADROOM_REJECTS_EMPTY_INPUT
    def test_headroom_rejects_empty_input(self) -> None:
        config = self._plateau_config()
        selected = solve_growth_optimal_risk(
            self._plateau_series(), config, use_drawdown_overlay=False,
        )
        with pytest.raises(ValueError, match="empty"):
            diagnose_growth_headroom(np.array([]), config, selected)
        with pytest.raises(ValueError, match="finite"):
            diagnose_growth_headroom(
                np.array([0.001, np.nan]), config, selected,
            )

    def test_headroom_with_drawdown_overlay_branch(self) -> None:
        # The path-dependent de-risk ladder (use_drawdown_overlay=True) is a
        # distinct simulation branch; the diagnostic must run it without ever
        # re-selecting or mutating the passed result.
        config = GrowthSizingConfig(
            risk_grid=(1.0, 1.5, 2.0, 2.5, 3.0), reference_risk=1.0,
            horizon_years=1.0, n_paths=300, seed=0,
        )
        selected = solve_growth_optimal_risk(
            self._plateau_series(), config, use_drawdown_overlay=True,
        )
        assert selected.selected_risk is not None
        diag = diagnose_growth_headroom(
            self._plateau_series(), config, selected, use_drawdown_overlay=True,
        )
        assert diag.selected_risk == selected.selected_risk
        assert diag.selected_median_log_growth == selected.median_log_growth
        assert diag.peak_feasible_median_log_growth >= selected.median_log_growth - 1e-9
        assert diag.headroom_ratio >= 0.0

class TestApplyRealisedRiskOverlay:
    def _index(self, n: int = 5) -> pd.DatetimeIndex:
        return pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")

    # GPR-02-RISK-OVERLAY-CHANGES-EQUITY
    def test_changing_selected_risk_changes_realised_equity_and_weights(self) -> None:
        index = self._index()
        net = pd.Series([0.01, 0.02, -0.005, 0.01, 0.005], index=index)
        weights = pd.DataFrame({"A": [1.0] * 5, "B": [-1.0] * 5}, index=index)
        low = apply_realised_risk_overlay(net, weights, 0.0025, 0.005)
        full = apply_realised_risk_overlay(net, weights, 0.005, 0.005)
        # the first multiplier is exactly one; scale = selected/reference
        assert low[0].iloc[0] == pytest.approx(0.5 * net.iloc[0])
        assert full[0].iloc[0] == pytest.approx(net.iloc[0])
        assert low[1].iloc[0, 0] == pytest.approx(0.5)
        assert full[1].iloc[0, 0] == pytest.approx(1.0)
        assert not np.allclose(low[0].to_numpy(), full[0].to_numpy())
        assert not np.allclose(low[1].to_numpy(), full[1].to_numpy())

    # GPR-02-RISK-OVERLAY-CHANGES-EQUITY
    def test_drawdown_overlay_never_increases_exposure(self) -> None:
        index = self._index(7)
        # a drawdown early in the ledger must de-risk every later bar
        net = pd.Series([0.05, -0.18, 0.10, 0.02, -0.02, 0.01, 0.01], index=index)
        weights = pd.DataFrame({"A": [1.0] * 7, "B": [-1.0] * 7}, index=index)
        _scaled_net, scaled_weights = apply_realised_risk_overlay(net, weights, 0.005, 0.005)
        magnitudes = np.abs(scaled_weights.to_numpy())
        assert np.all(magnitudes <= 1.0 + 1e-12)
        # after the drawdown, exposure is strictly below the full-scale first bar
        assert np.all(magnitudes[2:] < magnitudes[0])

    def test_first_multiplier_is_one(self) -> None:
        index = self._index(3)
        net = pd.Series([0.01, 0.01, 0.01], index=index)
        weights = pd.DataFrame({"A": [1.0] * 3}, index=index)
        scaled, _ = apply_realised_risk_overlay(net, weights, 0.005, 0.005)
        assert scaled.iloc[0] == pytest.approx(0.01)

    def test_rejects_non_finite_returns(self) -> None:
        index = self._index(3)
        net = pd.Series([0.01, np.nan, 0.01], index=index)
        weights = pd.DataFrame({"A": [1.0] * 3}, index=index)
        with pytest.raises(ValueError, match="finite"):
            apply_realised_risk_overlay(net, weights, 0.005, 0.005)

    def test_rejects_shared_index_mismatch(self) -> None:
        index = self._index(3)
        other = self._index(4)
        net = pd.Series([0.01] * 3, index=index)
        weights = pd.DataFrame({"A": [1.0] * 4}, index=other)
        with pytest.raises(ValueError, match="identical index"):
            apply_realised_risk_overlay(net, weights, 0.005, 0.005)

    def test_rejects_non_positive_risk(self) -> None:
        index = self._index(2)
        net = pd.Series([0.01, 0.01], index=index)
        weights = pd.DataFrame({"A": [1.0] * 2}, index=index)
        with pytest.raises(ValueError, match="> 0"):
            apply_realised_risk_overlay(net, weights, 0.0, 0.005)

    def test_rejects_non_monotonic_index(self) -> None:
        index = self._index(4)
        shuffled = pd.DatetimeIndex([index[2], index[0], index[3], index[1]])
        net = pd.Series([0.01] * 4, index=shuffled)
        weights = pd.DataFrame({"A": [1.0] * 4}, index=shuffled)
        with pytest.raises(ValueError, match="monotonic"):
            apply_realised_risk_overlay(net, weights, 0.005, 0.005)

    def test_rejects_non_datetime_index(self) -> None:
        index = pd.RangeIndex(3)
        net = pd.Series([0.01] * 3, index=index)
        weights = pd.DataFrame({"A": [1.0] * 3}, index=index)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            apply_realised_risk_overlay(net, weights, 0.005, 0.005)


class TestVolTargetOverlay:
    def _index(self, n: int = 100) -> pd.DatetimeIndex:
        return pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")

    def _series(self, n: int = 100, seed: int = 7) -> tuple[pd.Series, pd.DataFrame]:
        idx = self._index(n)
        rng = np.random.default_rng(seed)
        r = np.concatenate([rng.normal(0.0, 0.001, n // 2), rng.normal(0.0, 0.02, n // 2)])
        net = pd.Series(r, index=idx)
        weights = pd.DataFrame({"A": 0.5, "B": -0.5}, index=idx)
        return net, weights

    # SCENARIO_VOLTARGET_01_CAUSAL_NO_LOOKAHEAD
    def test_voltarget_01_multiplier_never_looks_ahead(self) -> None:
        net, weights = self._series()
        scaled_net, _scaled_weights = apply_vol_target_overlay(
            net, weights, window=10, target_vol=0.005, multiplier_bounds=(0.25, 4.0),
        )
        net_mut = net.copy()
        net_mut.iloc[90] = 5.0
        scaled_net_mut, _w = apply_vol_target_overlay(
            net_mut, weights, window=10, target_vol=0.005, multiplier_bounds=(0.25, 4.0),
        )
        # bar t's multiplier uses strictly prior bars, so mutating bar 90 can
        # only affect the multiplier from bar 91 onward; bars 0..89 (including
        # their unmutated raw values) are reproduced exactly.
        assert np.allclose(scaled_net_mut.iloc[:90].to_numpy(), scaled_net.iloc[:90].to_numpy())

    # SCENARIO_VOLTARGET_02_FALLBACK_INSUFFICIENT_HISTORY
    def test_voltarget_02_insufficient_history_falls_back_to_one(self) -> None:
        net, weights = self._series()
        scaled_net, scaled_weights = apply_vol_target_overlay(
            net, weights, window=10, target_vol=0.005, multiplier_bounds=(0.25, 4.0),
        )
        # bars 0..window-1 have fewer than `window` strictly-prior bars: multiplier is exactly 1.0.
        assert np.allclose(scaled_net.iloc[:10].to_numpy(), net.iloc[:10].to_numpy())
        assert scaled_weights.iloc[:10].equals(weights.iloc[:10])

    # SCENARIO_VOLTARGET_03_BOUNDS_CLIPPING
    def test_voltarget_03_bounds_clip_unbounded_scale(self) -> None:
        net, weights = self._series()
        scaled_hi, _w = apply_vol_target_overlay(
            net, weights, window=10, target_vol=10.0, multiplier_bounds=(0.25, 4.0),
        )
        assert np.allclose(
            scaled_hi.iloc[15:].to_numpy(), 4.0 * net.iloc[15:].to_numpy(), atol=1e-9,
        )
        scaled_lo, _w2 = apply_vol_target_overlay(
            net, weights, window=10, target_vol=1e-9, multiplier_bounds=(0.25, 4.0),
        )
        assert np.allclose(
            scaled_lo.iloc[15:].to_numpy(), 0.25 * net.iloc[15:].to_numpy(), atol=1e-9,
        )

    # SCENARIO_VOLTARGET_04_TARGET_VOL_DISCOVERY_ONLY_FORMULA
    def test_voltarget_04_target_vol_matches_causal_median_formula(self) -> None:
        idx = pd.date_range("2024-01-01", periods=60, freq="4h", tz="UTC")
        rng = np.random.default_rng(1)
        sigma_envelope = np.concatenate([np.full(30, 0.001), np.full(30, 0.02)])
        net = pd.Series(rng.normal(0.0, 1.0, 60) * sigma_envelope, index=idx)
        tv = compute_discovery_target_vol(net, window=10)
        expected = net.rolling(10, min_periods=10).std().shift(1).dropna().median()
        assert isinstance(tv, float)
        assert np.isfinite(tv)
        assert tv > 0
        assert abs(tv - float(expected)) < 1e-12
        with pytest.raises(ValueError, match="window"):
            compute_discovery_target_vol(net, window=1)
        with pytest.raises(ValueError, match=r"window \+ 1"):
            compute_discovery_target_vol(net, window=100)
        with pytest.raises(ValueError, match="finite"):
            compute_discovery_target_vol(
                pd.Series(np.where(np.arange(30) == 5, np.nan, 0.001), index=idx[:30]),
                window=10,
            )

    def test_voltarget_overlay_rejects_invalid_inputs(self) -> None:
        net, weights = self._series()
        with pytest.raises(ValueError, match="DatetimeIndex"):
            apply_vol_target_overlay(
                pd.Series(net.to_numpy(), index=pd.RangeIndex(100)),
                weights, 10, 0.005, (0.25, 4.0),
            )
        with pytest.raises(ValueError, match="identical index"):
            apply_vol_target_overlay(net.iloc[1:], weights, 10, 0.005, (0.25, 4.0))
        shuffled = pd.DatetimeIndex(
            [self._index(100)[i] for i in [5, 0, 9, 1, 2, 3, 4, 6, 7, 8, *range(10, 100)]]
        )
        with pytest.raises(ValueError, match="monotonic"):
            apply_vol_target_overlay(
                pd.Series(net.to_numpy(), index=shuffled),
                weights.reindex(shuffled), 10, 0.005, (0.25, 4.0),
            )
        bad_net = net.copy()
        bad_net.iloc[3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            apply_vol_target_overlay(bad_net, weights, 10, 0.005, (0.25, 4.0))
        with pytest.raises(ValueError, match="window"):
            apply_vol_target_overlay(net, weights, 1, 0.005, (0.25, 4.0))
        with pytest.raises(ValueError, match="target_vol"):
            apply_vol_target_overlay(net, weights, 10, -1.0, (0.25, 4.0))
        with pytest.raises(ValueError, match="target_vol"):
            apply_vol_target_overlay(net, weights, 10, np.nan, (0.25, 4.0))
        with pytest.raises(ValueError, match="multiplier_bounds"):
            apply_vol_target_overlay(net, weights, 10, 0.005, (4.0, 0.25))
        with pytest.raises(ValueError, match="multiplier_bounds"):
            apply_vol_target_overlay(net, weights, 10, 0.005, (0.0, 4.0))


# SCENARIO_GROWTH_ENVELOPE_LADDER_SELECTS_MONOTONE_RISK
class TestGrowthEnvelopeLadder:
    """On a fixed seeded synthetic series, growth_budget_annual_vol is non-decreasing
    across the conservative -> balanced -> growth ladder."""

    @pytest.fixture(autouse=True)
    def _seeded_returns(self) -> None:
        rng = np.random.default_rng(0)
        self.returns = pd.Series(rng.normal(0.0015, 0.018, 750))

    def test_ladder_non_decreasing(self) -> None:
        vols = []
        for key in ("conservative", "balanced", "growth"):
            env = GROWTH_RISK_ENVELOPES[key]
            vol = growth_budget_annual_vol(self.returns, envelope=env)
            vols.append(vol)
        # conservative <= balanced <= growth
        assert vols[0] <= vols[1] + 1e-9
        assert vols[1] <= vols[2] + 1e-9
        # growth > conservative strictly
        assert vols[2] > vols[0]

    def test_envelope_none_defaults_to_conservative(self) -> None:
        vol_default = growth_budget_annual_vol(self.returns)
        vol_conservative = growth_budget_annual_vol(
            self.returns, envelope=GROWTH_RISK_ENVELOPES["conservative"],
        )
        assert vol_default == pytest.approx(vol_conservative)


class TestScanLeverageFrontier:
    """Diagnostic-only per-multiple frontier scan (no solver, no plateau rule)."""

    # SCENARIO_MHS_LEVERAGE_SCAN_01
    def test_matches_solver_inner_loop_positive_drift_scenario_mhs_leverage_scan_01(self) -> None:
        # Positive median growth => the plateau rule cannot interfere, so the
        # duplicated arithmetic must equal the frozen solver's inner loop to
        # the bit (same seed, same single bootstrap draw).
        rng = np.random.default_rng(11)
        arr = rng.normal(0.002, 0.01, 3000)
        config = GrowthSizingConfig(
            risk_grid=(0.001, 0.002), reference_risk=0.002,
            horizon_years=1.0, bars_per_year=365, n_paths=500, seed=7,
        )
        points = scan_leverage_frontier(arr, config, (1.0,))
        solver = solve_growth_optimal_risk(
            arr, dataclasses.replace(config, risk_grid=(config.reference_risk,)),
            use_drawdown_overlay=False,
        )
        assert len(points) == 1
        point = points[0]
        assert isinstance(point, FrontierScanPoint)
        assert point.multiple == 1.0
        assert point.mdd_breach_prob == solver.mdd_breach_prob
        assert point.ruin_prob == solver.ruin_prob
        assert point.feasible == (solver.selected_risk is not None)

    # SCENARIO_MHS_LEVERAGE_SCAN_02
    def test_reports_real_feasibility_where_plateau_masks_scenario_mhs_leverage_scan_02(self) -> None:
        # Losing stream: the frozen solver's plateau rule excludes every
        # negative-median-growth candidate (even constraint-feasible ones) and
        # returns selected_risk=None for ALL grid points; the scan must instead
        # report each candidate's TRUE mdd_breach_prob/ruin_prob/feasible.
        rng = np.random.default_rng(5)
        arr = rng.normal(-0.0008, 0.005, 3000)
        config = GrowthSizingConfig(
            risk_grid=(0.5, 1.0, 2.0), reference_risk=1.0,
            max_drawdown=0.5, max_drawdown_prob=1.0,
            ruin_fraction=0.6, max_ruin_prob=0.01,
            horizon_years=1.0, bars_per_year=365, n_paths=500, seed=3,
        )
        points = scan_leverage_frontier(arr, config, (0.5, 1.0, 2.0))
        assert [p.multiple for p in points] == [0.5, 1.0, 2.0]
        assert points[0].feasible is True
        assert points[-1].feasible is False
        assert any(p.feasible is False for p in points)
        solver = solve_growth_optimal_risk(arr, config, use_drawdown_overlay=False)
        assert solver.selected_risk is None

    # SCENARIO_MHS_LEVERAGE_SCAN_03
    def test_rejects_invalid_inputs_scenario_mhs_leverage_scan_03(self) -> None:
        arr = np.random.default_rng(0).normal(0.001, 0.01, 500)
        config = GrowthSizingConfig(
            risk_grid=(0.001,), reference_risk=0.001,
            horizon_years=1.0, bars_per_year=365, n_paths=200,
        )
        with pytest.raises(ValueError, match="empty"):
            scan_leverage_frontier(np.array([]), config, (1.0,))
        with pytest.raises(ValueError, match="finite"):
            scan_leverage_frontier(np.array([0.001, np.nan]), config, (1.0,))
        with pytest.raises(ValueError, match="finite"):
            scan_leverage_frontier(np.array([0.001, np.inf]), config, (1.0,))
        with pytest.raises(ValueError, match="empty"):
            scan_leverage_frontier(arr, config, ())
        with pytest.raises(ValueError, match=r"-0\.5"):
            scan_leverage_frontier(arr, config, (1.0, -0.5))
        with pytest.raises(ValueError, match="positive"):
            scan_leverage_frontier(arr, config, (0.0,))
