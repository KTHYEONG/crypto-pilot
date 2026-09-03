"""Cross-sectional weights tests (remainder after domain splits)."""

from __future__ import annotations




"""Cross-sectional expert contract tests (split by behavioral domain)."""
"""Contract scenarios XSC-01..XSC-05, XSC-07, XSA-02, and XSV3-01 for the cross-sectional module.

XSC-01-NO-TRADE-BAND-STATEFUL, XSC-02-WEIGHTS-DOLLAR-NEUTRAL,
XSC-03-SPEC-FROZEN-BOUNDS, XSC-04-LEDGER-EXECUTION-LAG,
XSC-05-ADMISSION-SCALE-INVARIANT, XSC-07-COMPOSITE-BEATS-SINGLE-FAMILY,
XSA-02-COMPOSITE-PRESERVATION, XSV3-01-FAMILY-SUM,
SCENARIO_XSV5_01_DUAL_FAMILY_EXCLUDES_FUNDING,
SCENARIO_XSV6_01_CAUSAL_VOL_WEIGHTS_EXCLUDE_CURRENT_BAR,
SCENARIO_XSV6_02_VOL_WEIGHTED_MATCHES_MANUAL_RECOMPUTE,
SCENARIO_XS_POSITIONING_WEIGHTS_01,
SCENARIO_XSV6SIZE_01_DISCOVERY_ONLY_SIZING_NO_LEAKAGE,
SCENARIO_XSV6SIZE_02_INFEASIBLE_SIZING_FAILS_CLOSED, and
SCENARIO_COSTFIX_01..07 (honest turnover-cost repricing of the vol-target
overlay stack).
"""
import numpy as np
import pandas as pd
import pytest
from src.quant.risk.growth_sizing import (
    GrowthSizingConfig,
    GrowthSizingResult,
    apply_realised_risk_overlay,
    apply_vol_target_overlay,
    compute_discovery_target_vol,
    solve_growth_optimal_risk,
)
from src.quant.technical_experts.cross_sectional import (
    XsAlphaCompositeSpec,
    XsCompositeSpec,
    _basis_score,
    _ledger_pnl,
    _true_realized_net,
    run_xs_composite_ledger,
    select_vol_target_window,
    size_xs_alpha_growth_optimal,
)
from tests.unit.quant.technical_experts.test_cross_sectional import (  # noqa: F401
    _alpha_inputs,
    _ref_zscore,
    _score_frame,
)

class TestBasisScore:
    def test_scenario_basis_score_contrarian_sign(self) -> None:
        # SCENARIO_BASIS_SCORE_CONTRARIAN_SIGN: one symbol's rolling-mean basis
        # uniformly higher than another's across the full signal_windows history
        # gets a strictly lower (more negative) score -- the contrarian sign is
        # wired exactly like _positioning_score (high premium -> short).
        idx = pd.date_range("2024-01-01", periods=400, freq="4h", tz="UTC")
        cols = ["A", "B", "C", "D"]
        path = 100.0 * np.exp(np.linspace(0.0, 0.05, len(idx)))
        closes = pd.DataFrame(np.tile(path, (4, 1)).T, index=idx, columns=cols)
        basis = pd.DataFrame(0.0, index=idx, columns=cols)
        basis.loc[idx[170]:, "A"] = 0.001
        basis.loc[idx[170]:, "B"] = 0.0005
        basis.loc[idx[170]:, "C"] = -0.0005
        basis.loc[idx[170]:, "D"] = -0.001
        alpha_spec = XsAlphaCompositeSpec()
        score = _basis_score(closes, basis, alpha_spec)
        late = score.loc[idx[380]:]
        assert bool((late["A"] < late["C"]).all())
        assert bool((late["B"] < late["D"]).all())
        assert bool((late["A"] < 0.0).all())
        assert bool((late["D"] > 0.0).all())

    def test_scenario_basis_score_zero_variance_safe(self) -> None:
        # SCENARIO_BASIS_SCORE_ZERO_VARIANCE_SAFE: a row with fewer than two
        # finite observations (all-NaN basis bar) yields score 0.0 -- no NaN, no
        # exception, matching _cross_sectional_zscore's finite-only contract.
        idx = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
        cols = ["A", "B", "C"]
        rng = np.random.default_rng(7)
        closes = pd.DataFrame(
            {c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))) for c in cols},
            index=idx,
        )
        basis = pd.DataFrame(
            rng.normal(0.0, 0.001, (300, 3)), index=idx, columns=cols,
        )
        basis.loc[idx[150], :] = np.nan
        alpha_spec = XsAlphaCompositeSpec()
        score = _basis_score(closes, basis, alpha_spec)
        assert bool(np.isfinite(score.to_numpy()).all())
        assert float(score.loc[idx[150]].sum()) == 0.0
        assert bool(np.allclose(score.loc[idx[150]].to_numpy(), 0.0))

    def test_scenario_basis_score_shape_parity(self) -> None:
        # SCENARIO_BASIS_SCORE_SHAPE_PARITY: output index and ordered columns
        # mirror the input closes frame for any basis panel sharing that index.
        idx = pd.date_range("2024-01-01", periods=250, freq="4h", tz="UTC")
        cols = ["Z", "A", "M"]
        closes = pd.DataFrame(
            {c: 100.0 * np.exp(np.linspace(0.0, 0.02, 250)) for c in cols},
            index=idx,
        )
        basis = pd.DataFrame(
            np.random.default_rng(11).normal(0.0, 0.001, (250, 3)),
            index=idx, columns=cols,
        )
        alpha_spec = XsAlphaCompositeSpec()
        score = _basis_score(closes, basis, alpha_spec)
        assert list(score.columns) == list(closes.columns) == cols
        assert score.index.equals(closes.index)
        assert score.shape == closes.shape

class TestGrowthOptimalSizing:
    def _sizing_inputs(self, rows: int = 300) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        # A constant long-A/short-B book earns positive drift (A up, B down); a
        # single -80% A open inside the discovery window provides the tail event
        # that makes the hard 1e-6 constraints infeasible while the easy 0.9
        # constraints stay feasible.
        idx = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC")
        rng = np.random.default_rng(3)
        closes = pd.DataFrame({
            "A": 100 * np.exp(np.cumsum(rng.normal(0.0015, 0.008, rows))),
            "B": 100 * np.exp(np.cumsum(rng.normal(-0.0015, 0.008, rows))),
        }, index=idx)
        opens = closes.shift(1).bfill()
        opens.loc[idx[40], "A"] = opens.loc[idx[39], "A"] * 0.2
        funding = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
        weights = pd.DataFrame({"A": 0.5, "B": -0.5}, index=idx)
        return weights, opens, funding, idx

    def _easy_config(self) -> GrowthSizingConfig:
        return GrowthSizingConfig(
            risk_grid=(0.5, 1.0, 2.0), reference_risk=1.0, horizon_years=0.5,
            n_paths=100, max_drawdown_prob=0.9, max_ruin_prob=0.9,
        )

    def test_xsv6size_01_sizing_reads_only_discovery_window(self) -> None:
        # SCENARIO_XSV6SIZE_01_DISCOVERY_ONLY_SIZING_NO_LEAKAGE
        weights, opens, funding, idx = self._sizing_inputs()
        discovery_start, discovery_end = idx[0], idx[149]
        spec = XsCompositeSpec()

        baseline_net, _baseline_weights, baseline = size_xs_alpha_growth_optimal(
            weights, opens, funding, spec, discovery_start, discovery_end, self._easy_config(),
        )
        assert baseline.selected_risk is not None

        mutated_opens = opens.copy()
        mutated_opens.loc[idx[150]:] *= (
            1.0 - 0.0005 * np.arange(1, len(idx) - 149)[:, None]
        )
        _net2, _w2, mutated = size_xs_alpha_growth_optimal(
            weights, mutated_opens, funding, spec, discovery_start, discovery_end, self._easy_config(),
        )
        assert mutated == baseline

        crashed_opens = opens.copy()
        crashed_opens.loc[idx[:150]] *= (1.0 - 0.001 * np.arange(150))[:, None]
        _net3, _w3, crashed = size_xs_alpha_growth_optimal(
            weights, crashed_opens, funding, spec, discovery_start, discovery_end, self._easy_config(),
        )
        assert crashed != baseline
        assert baseline_net.index.equals(_baseline_weights.index)

    def test_xsv6size_02_infeasible_sizing_fails_closed(self) -> None:
        # SCENARIO_XSV6SIZE_02_INFEASIBLE_SIZING_FAILS_CLOSED
        weights, opens, funding, idx = self._sizing_inputs()
        discovery_start, discovery_end = idx[0], idx[149]
        hard_config = GrowthSizingConfig(
            risk_grid=(0.5, 1.0, 2.0), reference_risk=1.0, horizon_years=0.5,
            n_paths=100, max_drawdown_prob=1e-6, max_ruin_prob=1e-6,
        )
        net, weights_out, sizing = size_xs_alpha_growth_optimal(
            weights, opens, funding, XsCompositeSpec(), discovery_start, discovery_end, hard_config,
        )
        assert sizing.selected_risk is None
        assert isinstance(sizing, GrowthSizingResult)
        unscaled_equity, _unscaled_turnover = run_xs_composite_ledger(
            weights, opens, funding, XsCompositeSpec(),
        )
        unscaled_net = unscaled_equity.pct_change().dropna()
        assert np.allclose(net.to_numpy(), unscaled_net.to_numpy(), atol=1e-12)
        assert weights_out.equals(weights)

class TestVolTargetGrowthOptimalSizing:
    def _sizing_inputs(self, rows: int = 300, crash_factor: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        idx = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC")
        rng = np.random.default_rng(3)
        closes = pd.DataFrame({
            "A": 100 * np.exp(np.cumsum(rng.normal(0.0015, 0.008, rows))),
            "B": 100 * np.exp(np.cumsum(rng.normal(-0.0015, 0.008, rows))),
        }, index=idx)
        opens = closes.shift(1).bfill()
        opens.loc[idx[40], "A"] = opens.loc[idx[39], "A"] * crash_factor
        funding = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
        weights = pd.DataFrame({"A": 0.5, "B": -0.5}, index=idx)
        return weights, opens, funding, idx

    def _easy_config(self) -> GrowthSizingConfig:
        return GrowthSizingConfig(
            risk_grid=(0.5, 1.0, 2.0), reference_risk=1.0, horizon_years=0.5,
            n_paths=100, max_drawdown_prob=0.9, max_ruin_prob=0.9,
        )

    # SCENARIO_VOLTARGET_05_DEFAULT_DISABLED_BACKWARD_COMPATIBLE
    def test_voltarget_05_default_disabled_is_backward_compatible(self) -> None:
        weights, opens, funding, idx = self._sizing_inputs()
        discovery_start, discovery_end = idx[0], idx[149]
        config = self._easy_config()
        net_off, _w_off, sizing_off = size_xs_alpha_growth_optimal(
            weights, opens, funding, XsCompositeSpec(), discovery_start, discovery_end, config,
        )
        net_off2, _w_off2, sizing_off2 = size_xs_alpha_growth_optimal(
            weights, opens, funding, XsCompositeSpec(), discovery_start, discovery_end, config,
            vol_target_window=None,
        )
        assert np.allclose(net_off.to_numpy(), net_off2.to_numpy())
        assert sizing_off.selected_risk == sizing_off2.selected_risk

    # SCENARIO_VOLTARGET_06_ENABLED_DISCOVERY_ONLY_NO_LEAKAGE
    def test_voltarget_06_enabled_anchor_is_discovery_only(self) -> None:
        weights, opens, funding, idx = self._sizing_inputs()
        discovery_start, discovery_end = idx[0], idx[149]
        config = self._easy_config()
        net_off, _w_off, _sizing_off = size_xs_alpha_growth_optimal(
            weights, opens, funding, XsCompositeSpec(), discovery_start, discovery_end, config,
        )
        net_vt, w_vt, sizing_vt = size_xs_alpha_growth_optimal(
            weights, opens, funding, XsCompositeSpec(), discovery_start, discovery_end, config,
            vol_target_window=20,
        )
        assert net_vt.index.equals(net_off.index)
        assert not np.allclose(net_vt.to_numpy(), net_off.to_numpy())
        # mutating holdout-window opens must not change the anchor, the selected
        # risk, or any discovery-window bar of the returned net.
        opens_mut = opens.copy()
        opens_mut.loc[idx[200]:, "A"] = opens_mut.loc[idx[200]:, "A"] * 1.5
        net_vt_mut, _w_mut, sizing_vt_mut = size_xs_alpha_growth_optimal(
            weights, opens_mut, funding, XsCompositeSpec(), discovery_start, discovery_end, config,
            vol_target_window=20,
        )
        assert sizing_vt_mut.selected_risk == sizing_vt.selected_risk
        assert np.allclose(net_vt_mut.iloc[:150].to_numpy(), net_vt.iloc[:150].to_numpy())

    # SCENARIO_VOLTARGET_07_INFEASIBLE_BYPASSES_VOL_TARGETING_ENTIRELY
    def test_voltarget_07_infeasible_bypasses_vol_targeting(self) -> None:
        # A severe enough crash stays infeasible even after vol-targeting
        # compresses the tail (the causal multiplier can never know the shock
        # is coming), so the infeasible scalar branch is genuinely exercised.
        weights, opens, funding, idx = self._sizing_inputs(crash_factor=0.05)
        discovery_start, discovery_end = idx[0], idx[149]
        hard_config = GrowthSizingConfig(
            risk_grid=(0.5, 1.0, 2.0), reference_risk=1.0, horizon_years=0.5,
            n_paths=100, max_drawdown_prob=1e-6, max_ruin_prob=1e-6,
        )
        net, weights_out, sizing = size_xs_alpha_growth_optimal(
            weights, opens, funding, XsCompositeSpec(), discovery_start, discovery_end,
            hard_config, vol_target_window=20,
        )
        assert sizing.selected_risk is None
        unscaled_equity, _unscaled_turnover = run_xs_composite_ledger(
            weights, opens, funding, XsCompositeSpec(),
        )
        unscaled_net = unscaled_equity.pct_change().dropna()
        assert np.allclose(net.to_numpy(), unscaled_net.to_numpy(), atol=1e-12)
        assert weights_out.equals(weights)

class TestVolTargetWindowSelection:
    """SCENARIO_WINDOWSEARCH_01/02/03 for select_vol_target_window."""

    def _sizing_inputs(
        self, rows: int = 400, seed: int = 5, crash_factor: float | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        idx = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC")
        rng = np.random.default_rng(seed)
        closes = pd.DataFrame({
            "A": 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, rows))),
            "B": 100 * np.exp(np.cumsum(rng.normal(-0.001, 0.01, rows))),
        }, index=idx)
        opens = closes.shift(1).bfill()
        if crash_factor is not None:
            opens.loc[idx[40], "A"] = opens.loc[idx[39], "A"] * crash_factor
        funding = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
        weights = pd.DataFrame({"A": 0.5, "B": -0.5}, index=idx)
        return weights, opens, funding, idx

    def _easy_config(self) -> GrowthSizingConfig:
        return GrowthSizingConfig(
            risk_grid=(0.5, 1.0, 2.0), reference_risk=1.0, horizon_years=0.5,
            n_paths=100, max_drawdown_prob=0.9, max_ruin_prob=0.9,
        )

    def _hard_config(self) -> GrowthSizingConfig:
        return GrowthSizingConfig(
            risk_grid=(0.5, 1.0, 2.0), reference_risk=1.0, horizon_years=0.5,
            n_paths=100, max_drawdown_prob=1e-6, max_ruin_prob=1e-6,
        )

    # SCENARIO_WINDOWSEARCH_01_ARGMAX_MEDIAN_LOG_GROWTH
    def test_windowsearch_01_argmax_median_log_growth(self) -> None:
        weights, opens, funding, idx = self._sizing_inputs(seed=5)
        discovery_start, discovery_end = idx[0], idx[249]
        config = self._easy_config()
        grid = (20, 40, 60)
        net_sel, w_sel, sizing_sel, window_sel = select_vol_target_window(
            weights, opens, funding, XsCompositeSpec(), discovery_start,
            discovery_end, config, window_grid=grid,
        )
        assert window_sel in grid
        assert sizing_sel.selected_risk is not None
        # The winner is the strictly-highest median_log_growth feasible candidate.
        growth_by_window: dict[int, float] = {}
        for w in grid:
            _n, _w2, s_w = size_xs_alpha_growth_optimal(
                weights, opens, funding, XsCompositeSpec(), discovery_start,
                discovery_end, config, vol_target_window=w,
            )
            assert s_w.selected_risk is not None
            growth_by_window[w] = s_w.median_log_growth
        assert window_sel == max(growth_by_window, key=growth_by_window.get)
        assert sizing_sel.median_log_growth == growth_by_window[window_sel]
        # No other feasible candidate scores higher.
        assert max(growth_by_window.values()) <= sizing_sel.median_log_growth + 1e-9
        # The returned result reproduces a direct call with the selected window.
        net_direct, w_direct, sizing_direct = size_xs_alpha_growth_optimal(
            weights, opens, funding, XsCompositeSpec(), discovery_start,
            discovery_end, config, vol_target_window=window_sel,
        )
        assert np.allclose(net_sel.to_numpy(), net_direct.to_numpy())
        assert w_sel.equals(w_direct)
        assert sizing_sel == sizing_direct

    # SCENARIO_WINDOWSEARCH_02_DISCOVERY_ONLY_LEAKAGE
    def test_windowsearch_02_discovery_only_no_leakage(self) -> None:
        weights, opens, funding, idx = self._sizing_inputs(seed=5)
        discovery_start, discovery_end = idx[0], idx[249]
        config = self._easy_config()
        grid = (20, 40, 60)
        net_sel, _w_sel, _sizing_sel, window_sel = select_vol_target_window(
            weights, opens, funding, XsCompositeSpec(), discovery_start,
            discovery_end, config, window_grid=grid,
        )
        opens_mut = opens.copy()
        opens_mut.loc[idx[300]:] *= 1.5
        net_mut, _w_mut, _sizing_mut, window_mut = select_vol_target_window(
            weights, opens_mut, funding, XsCompositeSpec(), discovery_start,
            discovery_end, config, window_grid=grid,
        )
        assert window_mut == window_sel
        discovery_bars = (net_sel.index >= discovery_start) & (net_sel.index <= discovery_end)
        assert np.allclose(
            net_mut.loc[discovery_bars].to_numpy(), net_sel.loc[discovery_bars].to_numpy(),
        )

    # SCENARIO_WINDOWSEARCH_03_ALL_INFEASIBLE_FALLS_BACK_TO_SCALAR_ONLY
    def test_windowsearch_03_all_infeasible_falls_back_to_scalar_only(self) -> None:
        weights, opens, funding, idx = self._sizing_inputs(
            seed=3, crash_factor=0.05,
        )
        discovery_start, discovery_end = idx[0], idx[249]
        hard_config = self._hard_config()
        grid = (20, 40, 60)
        net_sel, w_sel, sizing_sel, window_sel = select_vol_target_window(
            weights, opens, funding, XsCompositeSpec(), discovery_start,
            discovery_end, hard_config, window_grid=grid,
        )
        assert window_sel is None
        net_direct, w_direct, sizing_direct = size_xs_alpha_growth_optimal(
            weights, opens, funding, XsCompositeSpec(), discovery_start,
            discovery_end, hard_config, vol_target_window=None,
        )
        assert np.allclose(net_sel.to_numpy(), net_direct.to_numpy())
        assert w_sel.equals(w_direct)
        assert sizing_sel == sizing_direct

    def test_windowsearch_empty_grid_raises(self) -> None:
        weights, opens, funding, idx = self._sizing_inputs(seed=5)
        with pytest.raises(ValueError, match="window_grid"):
            select_vol_target_window(
                weights, opens, funding, XsCompositeSpec(), idx[0], idx[249],
                self._easy_config(), window_grid=(),
            )

class TestCostRepricing:
    """SCENARIO_COSTFIX_01..07: honest turnover-cost repricing of the overlay stack."""

    def _sizing_inputs(
        self, rows: int = 300, crash_factor: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        idx = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC")
        rng = np.random.default_rng(3)
        closes = pd.DataFrame({
            "A": 100 * np.exp(np.cumsum(rng.normal(0.0015, 0.008, rows))),
            "B": 100 * np.exp(np.cumsum(rng.normal(-0.0015, 0.008, rows))),
        }, index=idx)
        opens = closes.shift(1).bfill()
        opens.loc[idx[40], "A"] = opens.loc[idx[39], "A"] * crash_factor
        funding = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
        weights = pd.DataFrame({"A": 0.5, "B": -0.5}, index=idx)
        return weights, opens, funding, idx

    def _easy_config(self) -> GrowthSizingConfig:
        return GrowthSizingConfig(
            risk_grid=(0.5, 1.0, 2.0), reference_risk=1.0, horizon_years=0.5,
            n_paths=100, max_drawdown_prob=0.9, max_ruin_prob=0.9,
        )

    # SCENARIO_COSTFIX_01_LEDGER_PNL_REGRESSION
    def test_costfix_01_ledger_pnl_extraction_is_regression_free(self) -> None:
        weights, opens, funding, _ = self._sizing_inputs()
        spec = XsCompositeSpec()
        equity, turnover_series = run_xs_composite_ledger(weights, opens, funding, spec)
        lag = 1 + spec.execution_delay_bars
        lagged = weights.shift(lag).fillna(0.0).to_numpy(dtype=np.float64)
        o = opens.to_numpy(dtype=np.float64)
        f = funding.to_numpy(dtype=np.float64)
        o2o = np.zeros_like(o)
        with np.errstate(divide="ignore", invalid="ignore"):
            o2o[1:] = o[1:] / o[:-1] - 1.0
        net_returns, turnover = _ledger_pnl(lagged, o2o, f, spec.round_trip_cost_rate())
        assert np.allclose(turnover, turnover_series.to_numpy())
        assert np.allclose(
            net_returns, equity.pct_change().fillna(0.0).to_numpy(), atol=1e-12,
        )

    # SCENARIO_COSTFIX_02_TRUE_REALIZED_NET_MATCHES_LEDGER
    def test_costfix_02_true_realized_net_matches_ledger(self) -> None:
        weights, opens, funding, _ = self._sizing_inputs()
        spec = XsCompositeSpec()
        equity, _turnover = run_xs_composite_ledger(weights, opens, funding, spec)
        expected_net = equity.pct_change().fillna(0.0)
        lag = 1 + spec.execution_delay_bars
        lagged = weights.shift(lag).fillna(0.0)
        true_net = _true_realized_net(lagged, opens, funding, spec.round_trip_cost_rate())
        assert true_net.index.equals(expected_net.index)
        assert np.allclose(true_net.to_numpy(), expected_net.to_numpy(), atol=1e-10)

    # SCENARIO_COSTFIX_03_OUTPUT_INVARIANT_HOLDS_FEASIBLE
    def test_costfix_03_output_invariant_holds_feasible(self) -> None:
        weights, opens, funding, idx = self._sizing_inputs()
        spec = XsCompositeSpec()
        config = self._easy_config()
        cost_rate = spec.round_trip_cost_rate()
        net_vt, w_vt, sizing_vt = size_xs_alpha_growth_optimal(
            weights, opens, funding, spec, idx[0], idx[149], config,
            vol_target_window=20,
        )
        assert sizing_vt.selected_risk is not None
        assert np.allclose(
            net_vt.to_numpy(),
            _true_realized_net(w_vt, opens, funding, cost_rate).to_numpy(),
            atol=1e-10,
        )
        net_scalar, w_scalar, sizing_scalar = size_xs_alpha_growth_optimal(
            weights, opens, funding, spec, idx[0], idx[149], config,
            vol_target_window=None,
        )
        assert sizing_scalar.selected_risk is not None
        assert np.allclose(
            net_scalar.to_numpy(),
            _true_realized_net(w_scalar, opens, funding, cost_rate).to_numpy(),
            atol=1e-10,
        )

    # SCENARIO_COSTFIX_04_OUTPUT_INVARIANT_HOLDS_INFEASIBLE
    def test_costfix_04_output_invariant_holds_infeasible(self) -> None:
        # A severe crash keeps even the vol-targeted series infeasible, so the
        # fail-closed branch is genuinely exercised; the returned net must still
        # be the true ledger P&L of the lag-reconstructed raw input weights.
        weights, opens, funding, idx = self._sizing_inputs(crash_factor=0.05)
        spec = XsCompositeSpec()
        hard_config = GrowthSizingConfig(
            risk_grid=(0.5, 1.0, 2.0), reference_risk=1.0, horizon_years=0.5,
            n_paths=100, max_drawdown_prob=1e-6, max_ruin_prob=1e-6,
        )
        net, weights_out, sizing = size_xs_alpha_growth_optimal(
            weights, opens, funding, spec, idx[0], idx[149], hard_config,
            vol_target_window=20,
        )
        assert sizing.selected_risk is None
        assert weights_out.equals(weights)
        lag = 1 + spec.execution_delay_bars
        lagged_inf = weights_out.shift(lag).fillna(0.0)
        true_net_inf = _true_realized_net(
            lagged_inf, opens, funding, spec.round_trip_cost_rate(),
        )
        assert np.allclose(
            net.to_numpy(), true_net_inf.reindex(net.index).to_numpy(), atol=1e-10,
        )

    # SCENARIO_COSTFIX_05_SCALAR_SEARCH_USES_TRUE_COST
    def test_costfix_05_scalar_search_uses_true_cost(self) -> None:
        weights, opens, funding, idx = self._sizing_inputs()
        spec = XsCompositeSpec()
        config = self._easy_config()
        _net, _w, sizing = size_xs_alpha_growth_optimal(
            weights, opens, funding, spec, idx[0], idx[149], config,
            vol_target_window=20,
        )
        assert sizing.selected_risk is not None
        equity, _turnover = run_xs_composite_ledger(weights, opens, funding, spec)
        net = equity.pct_change().dropna()
        discovery_net = net[(net.index >= idx[0]) & (net.index <= idx[149])]
        target_vol = compute_discovery_target_vol(discovery_net, 20)
        naive_full = equity.pct_change().fillna(0.0)
        realized_weights = weights.shift(1 + spec.execution_delay_bars).fillna(0.0)
        vt_net_naive, _vt_w = apply_vol_target_overlay(
            naive_full, realized_weights, 20, target_vol, (0.25, 4.0),
        )
        vt_disc_naive = vt_net_naive[
            (vt_net_naive.index >= idx[0]) & (vt_net_naive.index <= idx[149])
        ]
        naive_sizing = solve_growth_optimal_risk(vt_disc_naive.to_numpy(), config)
        assert naive_sizing.selected_risk is not None
        # The honest series charges the turnover the vol-target factor actually
        # causes, so the fixed search's median log growth must be below the
        # naive value computed from apply_vol_target_overlay's own net output.
        assert sizing.median_log_growth < naive_sizing.median_log_growth

    # SCENARIO_COSTFIX_06_DRAWDOWN_LADDER_USES_TRUE_COST_NET
    def test_costfix_06_drawdown_ladder_uses_true_cost_net(self) -> None:
        # Vol-regime blocks make the vol-target factor swing bar-to-bar, so the
        # honest net (with true turnover costs) drifts below the naive net;
        # after bar 120 a sustained book decline pushes both through the
        # ladder's 5% threshold at different bars, de-risking the honest path
        # first. The returned weights must reflect de-risking timed to the
        # honestly-repriced equity path, not the naive one.
        rows = 400
        idx = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC")
        rng = np.random.default_rng(0)
        sigma = np.full(rows, 0.001)
        block = 30
        k = 0
        while k < rows:
            sigma[k:k + block] = 0.015 if (k // block) % 2 == 0 else 0.001
            k += block
        mu_a = np.where(np.arange(rows) < 120, 0.0015, -0.002)
        mu_b = np.full(rows, 0.0005)
        closes = pd.DataFrame({
            "A": 100 * np.exp(np.cumsum(rng.normal(mu_a, sigma))),
            "B": 100 * np.exp(np.cumsum(rng.normal(mu_b, sigma))),
        }, index=idx)
        opens = closes.shift(1).bfill()
        weights = pd.DataFrame({"A": 0.5, "B": -0.5}, index=idx)
        funding = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
        spec = XsCompositeSpec()
        config = self._easy_config()
        net_fixed, w_fixed, sizing = size_xs_alpha_growth_optimal(
            weights, opens, funding, spec, idx[0], idx[149], config,
            vol_target_window=20,
        )
        assert sizing.selected_risk is not None

        equity, _turnover = run_xs_composite_ledger(weights, opens, funding, spec)
        net = equity.pct_change().dropna()
        discovery_net = net[(net.index >= idx[0]) & (net.index <= idx[149])]
        target_vol = compute_discovery_target_vol(discovery_net, 20)
        naive_full = equity.pct_change().fillna(0.0)
        realized_weights = weights.shift(1 + spec.execution_delay_bars).fillna(0.0)
        vt_net_naive, vt_weights = apply_vol_target_overlay(
            naive_full, realized_weights, 20, target_vol, (0.25, 4.0),
        )
        vt_true_net = _true_realized_net(
            vt_weights, opens, funding, spec.round_trip_cost_rate(),
        )
        honest_scaled_net, honest_scaled_w = apply_realised_risk_overlay(
            vt_true_net, vt_weights, sizing.selected_risk, config.reference_risk,
        )
        naive_scaled_net, naive_scaled_w = apply_realised_risk_overlay(
            vt_net_naive, vt_weights, sizing.selected_risk, config.reference_risk,
        )

        def _mdd(net_series: pd.Series) -> np.ndarray:
            eq = (1.0 + net_series.to_numpy()).cumprod()
            return 1.0 - eq / np.maximum.accumulate(eq)

        mdd_honest = _mdd(honest_scaled_net)
        mdd_naive = _mdd(naive_scaled_net)
        cross_honest = int(np.argmax(mdd_honest > 0.05))
        cross_naive = int(np.argmax(mdd_naive > 0.05))
        assert cross_honest < cross_naive
        # The fixed path consumes the honest net for the drawdown ladder.
        assert np.allclose(w_fixed.to_numpy(), honest_scaled_w.to_numpy(), atol=1e-12)
        divergent = np.abs(
            honest_scaled_w["A"].to_numpy() - naive_scaled_w["A"].to_numpy(),
        ) > 1e-12
        assert divergent.any()
        first = int(np.argmax(divergent))
        # At the first divergent bar the honest path has de-risked more than the
        # naive one (it crossed the ladder threshold earlier).
        assert honest_scaled_w["A"].iloc[first] < naive_scaled_w["A"].iloc[first]
        # The returned net remains the true P&L of the returned weights.
        assert np.allclose(
            net_fixed.to_numpy(),
            _true_realized_net(w_fixed, opens, funding, spec.round_trip_cost_rate()).to_numpy(),
            atol=1e-10,
        )

    # SCENARIO_COSTFIX_07_DISCOVERY_ONLY_LEAKAGE_STILL_HOLDS
    def test_costfix_07_discovery_only_leakage_still_holds(self) -> None:
        weights, opens, funding, idx = self._sizing_inputs()
        spec = XsCompositeSpec()
        config = self._easy_config()
        for vol_target_window in (None, 20):
            net_base, _w_base, sizing_base = size_xs_alpha_growth_optimal(
                weights, opens, funding, spec, idx[0], idx[149], config,
                vol_target_window=vol_target_window,
            )
            assert sizing_base.selected_risk is not None
            mutated_opens = opens.copy()
            mutated_opens.loc[idx[200]:] *= (
                1.0 - 0.0005 * np.arange(1, len(idx) - 199)[:, None]
            )
            mutated_funding = funding.copy()
            mutated_funding.loc[idx[200]:] = 0.0005
            net_mut, _w_mut, sizing_mut = size_xs_alpha_growth_optimal(
                weights, mutated_opens, mutated_funding, spec, idx[0], idx[149],
                config, vol_target_window=vol_target_window,
            )
            assert sizing_mut == sizing_base
            assert np.allclose(
                net_mut.iloc[:150].to_numpy(), net_base.iloc[:150].to_numpy(),
                atol=1e-12,
            )
