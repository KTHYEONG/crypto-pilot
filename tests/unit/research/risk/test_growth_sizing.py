from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.risk.growth_sizing import (
    GrowthSizingConfig,
    GrowthSizingResult,
    apply_realised_risk_overlay,
    drawdown_risk_multiplier,
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
            {"max_drawdown_prob": 1.0},
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
