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
from src.quant.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateConfig,
    ReliabilityGateResult,
    count_closed_trades,
    derive_block_size,
)
from src.quant.technical_experts.cross_sectional import (
    XsAdmissionConfig,
    XsReliabilityResult,
    apply_no_trade_band,
    evaluate_xs_admission,
    evaluate_xs_reliability,
)
from tests.unit.quant.technical_experts.test_cross_sectional import (  # noqa: F401
    _alpha_inputs,
    _ref_zscore,
    _score_frame,
)

class TestNoTradeBand:
    def test_xsc_01_holds_until_target_drifts_past_band_then_snaps_only_that_name(self) -> None:
        targets = np.array(
            [[0.10, -0.10], [0.12, -0.12], [0.20, -0.20], [0.21, -0.21]],
            dtype=np.float64,
        )
        held = apply_no_trade_band(targets, 0.05)
        assert np.allclose(held[0], [0.10, -0.10])
        assert np.allclose(held[1], [0.10, -0.10])
        assert np.allclose(held[2], [0.20, -0.20])
        assert np.allclose(held[3], [0.20, -0.20])

    def test_xsc_01_band_at_most_zero_is_pass_through(self) -> None:
        targets = np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float64)
        assert np.array_equal(apply_no_trade_band(targets, 0.0), targets)
        assert np.array_equal(apply_no_trade_band(targets, -1.0), targets)

    def test_xsc_01_band_application_is_strictly_causal(self) -> None:
        targets = np.array(
            [[0.10, -0.10], [0.12, -0.12], [0.20, -0.20], [0.21, -0.21]],
            dtype=np.float64,
        )
        held = apply_no_trade_band(targets, 0.05)
        assert np.allclose(held[:2], apply_no_trade_band(targets[:2], 0.05))

    def test_xsc_01_rejects_malformed_inputs(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            apply_no_trade_band(np.array([0.1, -0.1]), 0.05)
        with pytest.raises(ValueError, match="float"):
            apply_no_trade_band(np.array([[1, -1], [2, -2]]), 0.05)
        with pytest.raises(ValueError, match="finite"):
            apply_no_trade_band(np.array([[0.1, -0.1]]), np.nan)

class TestAdmission:
    def _admission_inputs(self, n: int = 2200) -> tuple[pd.Series, pd.Series, pd.Series]:
        index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        rng = np.random.default_rng(11)
        r = rng.normal(0.0006, 0.0045, n)
        equity = pd.Series(10000.0 * np.cumprod(1.0 + r), index=index)
        turnover = pd.Series(0.03, index=index)
        benchmark = pd.Series(rng.normal(0.0, 0.02, n), index=index)
        return equity, turnover, benchmark

    def test_xsc_05_verdict_is_invariant_to_constant_leverage(self) -> None:
        equity, turnover, benchmark = self._admission_inputs()
        res = evaluate_xs_admission(equity, turnover, benchmark, XsAdmissionConfig())
        assert hasattr(res, "admitted")
        assert hasattr(res, "binding_constraint")
        assert hasattr(res, "sharpe")
        assert hasattr(res, "beta")
        levered = pd.Series(10000.0 * np.cumprod(1.0 + 2.0 * equity.pct_change().fillna(0).to_numpy()), index=equity.index)
        res_lev = evaluate_xs_admission(levered, turnover, benchmark, XsAdmissionConfig())
        assert res_lev.admitted == res.admitted

    def test_xsc_05_losing_ledger_fails_closed_named(self) -> None:
        equity, turnover, benchmark = self._admission_inputs()
        rng = np.random.default_rng(11)
        r = rng.normal(0.0006, 0.0045, len(equity))
        losing = pd.Series(10000.0 * np.cumprod(1.0 - np.abs(r)), index=equity.index)
        res = evaluate_xs_admission(losing, turnover, benchmark, XsAdmissionConfig())
        assert res.admitted is False
        assert res.binding_constraint

    def test_xsc_05_high_beta_book_names_beta_abs_max(self) -> None:
        # A book whose return stream is a 2x copy of the benchmark carries
        # realized beta 2.0 and must fail the structure gate by name.
        index = pd.date_range("2024-01-01", periods=800, freq="4h", tz="UTC")
        rng = np.random.default_rng(13)
        bm = rng.normal(0.0, 0.0005, len(index))
        benchmark = pd.Series(bm, index=index)
        levered = pd.Series(10000.0 * np.cumprod(1.0 + 2.0 * bm), index=index)
        turnover = pd.Series(0.03, index=index)
        res = evaluate_xs_admission(levered, turnover, benchmark, XsAdmissionConfig())
        assert res.admitted is False
        assert "beta_abs_max" in res.binding_constraint

    def test_xsc_05_negative_year_names_annual_sub_sharpe(self) -> None:
        index = pd.date_range("2024-01-01", periods=4000, freq="4h", tz="UTC")
        rng = np.random.default_rng(5)
        r = rng.normal(0.0006, 0.0045, len(index))
        r[index.year == 2025] = rng.normal(-0.0006, 0.0045, int((index.year == 2025).sum()))
        equity = pd.Series(10000.0 * np.cumprod(1.0 + r), index=index)
        turnover = pd.Series(0.03, index=index)
        benchmark = pd.Series(rng.normal(0.0, 0.02, len(index)), index=index)
        res = evaluate_xs_admission(equity, turnover, benchmark, XsAdmissionConfig())
        assert res.admitted is False
        assert "annual_sub_sharpe" in res.binding_constraint

    def test_xsc_05_high_turnover_names_turnover_max(self) -> None:
        # TMAX-03: constant per-bar turnover 1.0 annualizes to ~2190x/yr --
        # far above even the recalibrated 200.0 default -- so the gate still
        # catches genuinely extreme turnover; only the 150-200x band moved.
        equity, turnover, benchmark = self._admission_inputs()
        high = pd.Series(1.0, index=turnover.index)
        res = evaluate_xs_admission(equity, high, benchmark, XsAdmissionConfig())
        assert res.admitted is False
        assert "turnover_max" in res.binding_constraint

    def test_xsc_05_turnover_max_default_recalibrated_to_200(self) -> None:
        # TMAX-01: the recalibration moved only the class default --
        # 150.0 -> 200.0 -- with every other gate and the gate mechanism
        # untouched.
        assert XsAdmissionConfig().turnover_max == 200.0
        assert XsAdmissionConfig().sharpe_floor == 0.80
        assert XsAdmissionConfig().beta_abs_max == 0.15
        assert XsAdmissionConfig().annual_bars_min == 60
        assert XsAdmissionConfig().cost_breakeven_min == 0.0024
        assert XsAdmissionConfig().round_trip_cost_rate == 0.0008

    def test_xsc_05_turnover_max_recalibration_blast_radius_is_isolated(self) -> None:
        # TMAX-04: the study's measured history shows every other admitted XS
        # profile sits well under both caps (max observed ~83x,
        # xs_neutral_composite_v1 discovery), so raising the default 150 -> 200
        # flips exactly one verdict -- xs_alpha_baseline_blend_v8_joint's
        # qualification (175.42x/yr). This is a regression guard documenting
        # that isolation, not a runtime check (it's a property of the historical
        # data the study verified, not something to re-derive here).
        default = XsAdmissionConfig().turnover_max
        assert default > 175.42  # admits the v8_joint qualification candidate
        assert default < 216.5  # stays below the one known disaster zone

    def test_xsc_05_no_absolute_cagr_hurdle(self) -> None:
        # A book with a high CAGR but sub-floor Sharpe must fail on sharpe_floor,
        # never on a CAGR threshold, and CAGR is reported as a diagnostic.
        index = pd.date_range("2024-01-01", periods=1000, freq="4h", tz="UTC")
        rng = np.random.default_rng(7)
        r = rng.normal(0.0002, 0.01, len(index))
        equity = pd.Series(10000.0 * np.cumprod(1.0 + r), index=index)
        turnover = pd.Series(0.03, index=index)
        benchmark = pd.Series(rng.normal(0.0, 0.02, len(index)), index=index)
        res = evaluate_xs_admission(equity, turnover, benchmark, XsAdmissionConfig())
        assert "sharpe_floor" in (res.binding_constraint or "")
        assert isinstance(res.cagr, float)

    def test_xsc_05_malformed_inputs_raise(self) -> None:
        equity, turnover, benchmark = self._admission_inputs()
        with pytest.raises(ValueError, match="2 marks"):
            evaluate_xs_admission(equity.iloc[:1], turnover.iloc[:1], benchmark.iloc[:1], XsAdmissionConfig())
        with pytest.raises(ValueError, match="identical index"):
            evaluate_xs_admission(
                equity, turnover, benchmark.iloc[1:], XsAdmissionConfig(),
            )
        with pytest.raises(ValueError, match="identical index"):
            evaluate_xs_admission(
                equity, turnover.iloc[1:], benchmark, XsAdmissionConfig(),
            )

    def test_xsc_05_zero_turnover_book_fails_closed_on_breakeven(self) -> None:
        # A book that never trades has no gross return per unit turnover, so
        # the breakeven-cost floor fails by name instead of producing NaN.
        equity, _turnover, benchmark = self._admission_inputs()
        zero_turnover = pd.Series(0.0, index=equity.index)
        res = evaluate_xs_admission(equity, zero_turnover, benchmark, XsAdmissionConfig())
        assert res.admitted is False
        assert res.breakeven_cost == 0.0
        assert "cost_breakeven_min" in res.binding_constraint

    def test_xsc_05_equity_without_usable_returns_raises(self) -> None:
        # A flat 2-mark ledger has no usable return stream and fails closed.
        index = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
        equity = pd.Series([100.0, 100.0], index=index)
        turnover = pd.Series(0.0, index=index)
        benchmark = pd.Series(0.0, index=index)
        with pytest.raises(ValueError, match="usable return"):
            evaluate_xs_admission(equity, turnover, benchmark, XsAdmissionConfig())

    def test_xsc_05_constant_benchmark_yields_zero_beta(self) -> None:
        # A constant benchmark has no variance to attribute, so realized beta is
        # exactly zero rather than NaN.
        equity, turnover, _benchmark = self._admission_inputs()
        flat_bm = pd.Series(0.0, index=equity.index)
        res = evaluate_xs_admission(equity, turnover, flat_bm, XsAdmissionConfig())
        assert res.beta == 0.0

    def test_xsc_05_flat_equity_fails_closed_zero_sharpe(self) -> None:
        # A constant ledger has zero volatility and therefore zero Sharpe,
        # failing the sharpe_floor gate by name.
        index = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
        equity = pd.Series([100.0] * 10, index=index)
        turnover = pd.Series(0.0, index=index)
        benchmark = pd.Series(0.0, index=index)
        res = evaluate_xs_admission(equity, turnover, benchmark, XsAdmissionConfig())
        assert res.sharpe == 0.0
        assert res.admitted is False
        assert "sharpe_floor" in res.binding_constraint

class TestXsReliability:
    """SCENARIO_XS_RELIABILITY_03: thin composition of the reliability primitives."""

    def _fixture(self, n: int = 2200) -> tuple[pd.Series, pd.DataFrame]:
        index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        rng = np.random.default_rng(11)
        r = rng.normal(0.0006, 0.0045, n)
        equity = pd.Series(10000.0 * np.cumprod(1.0 + r), index=index)
        weights = pd.DataFrame(
            rng.normal(0.0, 0.2, (n, 3)),
            index=index, columns=["A", "B", "C"],
        )
        return equity, weights

    def test_xs_reliability_03_composes_existing_primitives(self) -> None:
        # SCENARIO_XS_RELIABILITY_03
        equity, weights = self._fixture()
        result = evaluate_xs_reliability(equity, weights, ReliabilityGateConfig())
        assert isinstance(result, XsReliabilityResult)
        assert isinstance(result.lcb, ReliabilityGateResult)
        assert result.lcb.verdict in ("PASS", "FAIL", "PENDING")
        returns = equity.pct_change().dropna().to_numpy(dtype=np.float64)
        assert result.lcb.block_size_used == derive_block_size(returns)
        assert isinstance(result.fold, FoldDistributionResult)
        assert isinstance(result.fold.gate_pass, bool)
        assert result.fold.n_folds >= 2
        assert 1.0 / result.fold.n_folds <= result.fold.fold_concentration <= 1.0

    def test_xs_reliability_03_trade_count_is_the_realized_transition_proxy(self) -> None:
        equity, weights = self._fixture()
        result = evaluate_xs_reliability(equity, weights, ReliabilityGateConfig())
        assert result.lcb.trade_count == count_closed_trades(weights)
