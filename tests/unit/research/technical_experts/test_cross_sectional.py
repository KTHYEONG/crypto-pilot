"""Contract scenarios XSC-01..XSC-05, XSC-07, XSA-02, and XSV3-01 for the cross-sectional module.

XSC-01-NO-TRADE-BAND-STATEFUL, XSC-02-WEIGHTS-DOLLAR-NEUTRAL,
XSC-03-SPEC-FROZEN-BOUNDS, XSC-04-LEDGER-EXECUTION-LAG,
XSC-05-ADMISSION-SCALE-INVARIANT, XSC-07-COMPOSITE-BEATS-SINGLE-FAMILY,
XSA-02-COMPOSITE-PRESERVATION, XSV3-01-FAMILY-SUM,
SCENARIO_XSV5_01_DUAL_FAMILY_EXCLUDES_FUNDING,
SCENARIO_XSV6_01_CAUSAL_VOL_WEIGHTS_EXCLUDE_CURRENT_BAR,
SCENARIO_XSV6_02_VOL_WEIGHTED_MATCHES_MANUAL_RECOMPUTE,
SCENARIO_XSV6SIZE_01_DISCOVERY_ONLY_SIZING_NO_LEAKAGE, and
SCENARIO_XSV6SIZE_02_INFEASIBLE_SIZING_FAILS_CLOSED.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.risk.growth_sizing import GrowthSizingConfig, GrowthSizingResult
from src.research.technical_experts.cross_sectional import (
    XsAdmissionConfig,
    XsAlphaCompositeSpec,
    XsCompositeSpec,
    _causal_family_inverse_vol_weights,
    apply_no_trade_band,
    build_xs_alpha_composite_score,
    build_xs_alpha_dual_family_weights,
    build_xs_alpha_family_scores,
    build_xs_alpha_family_weights,
    build_xs_alpha_vol_weighted_weights,
    build_xs_alpha_weights,
    build_xs_neutral_weights,
    evaluate_xs_admission,
    run_xs_composite_ledger,
    size_xs_alpha_growth_optimal,
)


def _score_frame(rows: int = 40, cols: int = 5, seed: int = 3) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(rows, cols)),
        index=index,
        columns=[chr(ord("A") + i) for i in range(cols)],
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


class TestNeutralWeights:
    def test_xsc_02_pre_band_rows_are_dollar_neutral_unit_gross(self) -> None:
        score = _score_frame()
        unbanded = build_xs_neutral_weights(score, 6, 0.0)
        invested = unbanded[unbanded.abs().sum(axis=1) > 1e-12]
        assert (invested.sum(axis=1).abs() < 1e-9).all()
        assert ((invested.abs().sum(axis=1) - 1.0).abs() < 1e-9).all()

    def test_xsc_02_band_changes_output_without_renormalizing(self) -> None:
        # Re-normalizing after the band would rescale band-held names on every
        # bar and defeat the band's purpose. Because names snap asynchronously,
        # neither sum(w) == 0 nor sum(abs(w)) == 1 is exact per row once banded
        # (only the pre-band normalization guarantees them) -- realized
        # neutrality and gross exposure are validated at the portfolio level
        # via realized beta and turnover, not via a per-row equality here.
        score = _score_frame()
        weights = build_xs_neutral_weights(score, 6, 0.05)
        assert weights.shape == score.shape
        assert weights.index.equals(score.index)
        assert list(weights.columns) == list(score.columns)

        unbanded = build_xs_neutral_weights(score, 6, 0.0)
        assert not weights.equals(unbanded)

    def test_xsc_02_flat_score_produces_no_position(self) -> None:
        score = pd.DataFrame(1.0, index=_score_frame().index, columns=list("ABCDE"))
        weights = build_xs_neutral_weights(score, 6, 0.05)
        assert float(weights.abs().to_numpy().max()) < 1e-12
        assert not weights.isna().any().any()

    def test_xsc_02_halflife_zero_skips_smoothing(self) -> None:
        score = _score_frame()
        raw = build_xs_neutral_weights(score, 0, 0.0)
        smoothed = build_xs_neutral_weights(score, 6, 0.0)
        assert not raw.equals(smoothed)

    def test_xsc_02_never_shifts_the_frame(self) -> None:
        score = _score_frame()
        weights = build_xs_neutral_weights(score, 6, 0.05)
        assert weights.index.equals(score.index)

    def test_xsc_02_band_applies_on_weights_not_score(self) -> None:
        score = _score_frame()
        banded = build_xs_neutral_weights(score, 6, 0.05)
        no_band = build_xs_neutral_weights(score, 6, 0.0)
        assert not np.allclose(banded.to_numpy(), no_band.to_numpy())


class TestCompositeSpec:
    def test_xsc_03_frozen_defaults_and_cost_rate(self) -> None:
        spec = XsCompositeSpec()
        assert (spec.halflife_bars, spec.no_trade_band, spec.execution_delay_bars) == (
            6, 0.05, 1,
        )
        assert abs(spec.round_trip_cost_rate() - 0.0008) < 1e-12
        assert dataclasses.is_dataclass(spec)

    def test_xsc_03_out_of_range_fields_fail_closed(self) -> None:
        with pytest.raises(ValueError, match="no_trade_band"):
            XsCompositeSpec(no_trade_band=1.0)
        with pytest.raises(ValueError, match="no_trade_band"):
            XsCompositeSpec(no_trade_band=-0.1)
        with pytest.raises(ValueError, match="halflife_bars"):
            XsCompositeSpec(halflife_bars=-1)
        with pytest.raises(ValueError, match="execution_delay_bars"):
            XsCompositeSpec(execution_delay_bars=-1)
        with pytest.raises(ValueError, match="fee_rate"):
            XsCompositeSpec(fee_rate=-0.1)
        with pytest.raises(ValueError, match="slippage_rate"):
            XsCompositeSpec(slippage_rate=-0.1)


class TestCompositeLedger:
    def _ledger_inputs(self, bars: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        index = pd.date_range("2024-01-01", periods=bars, freq="4h", tz="UTC")
        opens = pd.DataFrame(
            {
                "A": np.linspace(100.0, 130.0, bars),
                "B": np.linspace(100.0, 70.0, bars),
            },
            index=index,
        )
        weights = pd.DataFrame({"A": [0.5] * bars, "B": [-0.5] * bars}, index=index)
        funding = pd.DataFrame(0.0, index=index, columns=["A", "B"])
        return weights, opens, funding

    def test_xsc_04_execution_lag_leaves_first_bars_flat(self) -> None:
        weights, opens, funding = self._ledger_inputs()
        equity, turnover = run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec())
        assert equity.index.equals(opens.index)
        assert turnover.index.equals(opens.index)
        assert float(turnover.iloc[:2].sum()) == 0.0
        assert float(turnover.sum()) > 0.0

    def test_xsc_04_long_riser_short_faller_profits(self) -> None:
        weights, opens, funding = self._ledger_inputs()
        equity, _turnover = run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec())
        assert bool((equity > 0).all())
        assert float(equity.iloc[-1]) > float(equity.iloc[0])

    def test_xsc_04_mismatched_index_raises(self) -> None:
        weights, opens, funding = self._ledger_inputs()
        shifted = opens.iloc[1:].copy()
        with pytest.raises(DataIntegrityError):
            run_xs_composite_ledger(weights, shifted, funding, XsCompositeSpec())

    def test_xsc_04_mismatched_columns_raise(self) -> None:
        weights, opens, funding = self._ledger_inputs()
        opens = opens.rename(columns={"B": "C"})
        with pytest.raises(DataIntegrityError):
            run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec())

    def test_xsc_04_equity_would_reach_zero_raises(self) -> None:
        index = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
        opens = pd.DataFrame({"A": [100.0] * 5}, index=index)
        weights = pd.DataFrame({"A": [1.0] * 5}, index=index)
        funding = pd.DataFrame(0.0, index=index, columns=["A"])
        opens.iloc[2] = 0.0
        with pytest.raises(DataIntegrityError):
            run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec())

    def test_xsc_04_funding_debits_long_credits_short(self) -> None:
        index = pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC")
        opens = pd.DataFrame({"A": [100.0] * 8, "B": [100.0] * 8}, index=index)
        weights = pd.DataFrame({"A": [0.5] * 8, "B": [-0.5] * 8}, index=index)
        funding = pd.DataFrame(0.001, index=index, columns=["A", "B"])
        no_funding = pd.DataFrame(0.0, index=index, columns=["A", "B"])
        with_funding, _t = run_xs_composite_ledger(weights, opens, funding, XsCompositeSpec())
        without, _t = run_xs_composite_ledger(weights, opens, no_funding, XsCompositeSpec())
        # A long pays positive funding, a short is credited: net funding PnL is
        # -w_long*f + -w_short*f = -(0.5 - 0.5) = 0 here, so the two ledgers must
        # only differ by the turnover cost term, which is identical.
        assert float(with_funding.iloc[-1]) == pytest.approx(float(without.iloc[-1]))


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
        equity, turnover, benchmark = self._admission_inputs()
        high = pd.Series(1.0, index=turnover.index)
        res = evaluate_xs_admission(equity, high, benchmark, XsAdmissionConfig())
        assert res.admitted is False
        assert "turnover_max" in res.binding_constraint

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


class TestCompositeEnsemble:
    def test_xsc_07_composite_tracks_signal_better_than_any_single_family(self) -> None:
        # Each family's score is the true latent signal plus independent noise.
        # Averaging 15 such families preserves the signal and cancels the noise,
        # so the composite weights must track the latent signal strictly more
        # closely than any single family's weights.
        index = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
        n_symbols = 8
        columns = [f"S{i}" for i in range(n_symbols)]
        rng = np.random.default_rng(17)
        true_signal = pd.DataFrame(
            rng.normal(size=(len(index), n_symbols)), index=index, columns=columns,
        )
        n_families = 15
        noise_scale = 1.2

        def weights_for(family_scores: pd.DataFrame) -> np.ndarray:
            return build_xs_neutral_weights(family_scores, 6, 0.05).to_numpy()

        latent = weights_for(true_signal)
        per_family: list[np.ndarray] = []
        family_scores: list[pd.DataFrame] = []
        for _f in range(n_families):
            noisy = true_signal + rng.normal(0.0, noise_scale, size=true_signal.shape)
            noisy = pd.DataFrame(noisy, index=index, columns=columns)
            family_scores.append(noisy)
            per_family.append(weights_for(noisy))
        composite_score = sum(family_scores) / n_families
        composite = weights_for(composite_score)

        def correlation(a: np.ndarray, b: np.ndarray) -> float:
            flat_a = a.reshape(-1)
            flat_b = b.reshape(-1)
            if np.std(flat_a) == 0.0 or np.std(flat_b) == 0.0:
                return 0.0
            return float(np.corrcoef(flat_a, flat_b)[0, 1])

        best_single = max(correlation(latent, fam) for fam in per_family)
        composite_corr = correlation(latent, composite)
        assert composite_corr > best_single


def _alpha_inputs(
    rows: int = 300, cols: int = 5, seed: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic strictly-positive closes, in-[0,1] taker ratios, finite funding."""
    index = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC")
    columns = [chr(ord("A") + i) for i in range(cols)]
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=(rows, cols)), axis=0))
    closes = pd.DataFrame(closes, index=index, columns=columns)
    taker = pd.DataFrame(
        0.5 + 0.1 * np.sin(np.arange(rows)[:, None] / 9.0 + np.arange(cols)),
        index=index, columns=columns,
    )
    funding = pd.DataFrame(
        0.0001 * np.cos(np.arange(rows)[:, None] / 5.0 + np.arange(cols)),
        index=index, columns=columns,
    )
    return closes, taker, funding


def _ref_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Reference finite-only cross-sectional z-score (see XSA-02 docstring)."""
    values = frame.to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    mean = np.where(finite, values, 0.0).sum(axis=1, keepdims=True) / np.maximum(count, 1)[:, None]
    demeaned = np.where(finite, values - mean, 0.0)
    var = (demeaned ** 2).sum(axis=1, keepdims=True) / np.maximum(count - 1, 1)[:, None]
    std = np.sqrt(np.maximum(var, 0.0))
    out = np.zeros_like(values)
    np.divide(
        demeaned, std, out=out,
        where=(count[:, None] >= 2) & (std > 0.0),
    )
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


class TestAlphaCompositeSpec:
    def test_xsa_01_windows_frozen_and_rejects_others(self) -> None:
        spec = XsAlphaCompositeSpec()
        assert spec.signal_windows == (42, 84, 168)
        assert spec.components == ("trend", "funding_contrarian", "taker_imbalance")
        with pytest.raises(ValueError, match="signal_windows"):
            XsAlphaCompositeSpec(signal_windows=(21, 42, 84))
        with pytest.raises(ValueError, match="components"):
            XsAlphaCompositeSpec(components=("a", "b", "c"))


class TestAlphaCompositeScore:
    def test_xsa_01_components_are_causal_prefix_invariant(self) -> None:
        closes, taker, funding = _alpha_inputs()
        full = build_xs_alpha_composite_score(closes, taker, funding, XsAlphaCompositeSpec())
        cutoff = 200
        prefix = build_xs_alpha_composite_score(
            closes.iloc[:cutoff], taker.iloc[:cutoff], funding.iloc[:cutoff],
            XsAlphaCompositeSpec(),
        )
        assert np.allclose(
            full.iloc[:cutoff].to_numpy(), prefix.to_numpy(), atol=1e-12,
        )

    def test_xsa_01_early_incomplete_horizons_are_zero(self) -> None:
        closes, taker, funding = _alpha_inputs()
        score = build_xs_alpha_composite_score(closes, taker, funding, XsAlphaCompositeSpec())
        assert score.index.equals(closes.index)
        assert list(score.columns) == list(closes.columns)
        # The taker imbalance window completes one bar earlier than trend/carry,
        # so the score is exactly zero for the first min(window) - 1 rows.
        assert float(score.iloc[:41].abs().sum().sum()) == 0.0
        assert float(score.iloc[41:].abs().sum().sum()) > 0.0

    def test_xsa_01_malformed_panels_fail_closed(self) -> None:
        closes, taker, funding = _alpha_inputs()

        bad_close = closes.copy()
        bad_close.iloc[5, 0] = np.nan
        with pytest.raises(DataIntegrityError, match="strictly positive"):
            build_xs_alpha_composite_score(bad_close, taker, funding, XsAlphaCompositeSpec())

        bad_close = closes.copy()
        bad_close.iloc[5, 0] = 0.0
        with pytest.raises(DataIntegrityError, match="strictly positive"):
            build_xs_alpha_composite_score(bad_close, taker, funding, XsAlphaCompositeSpec())

        bad_funding = funding.copy()
        bad_funding.iloc[5, 0] = np.nan
        with pytest.raises(DataIntegrityError, match="bar_funding must be finite"):
            build_xs_alpha_composite_score(closes, taker, bad_funding, XsAlphaCompositeSpec())

        bad_taker = taker.copy()
        bad_taker.iloc[5, 0] = 1.5
        with pytest.raises(DataIntegrityError, match=r"in \[0, 1\]"):
            build_xs_alpha_composite_score(closes, bad_taker, funding, XsAlphaCompositeSpec())

        bad_taker = taker.copy()
        bad_taker.iloc[5, 0] = np.nan
        with pytest.raises(DataIntegrityError, match="finite and in"):
            build_xs_alpha_composite_score(closes, bad_taker, funding, XsAlphaCompositeSpec())

        with pytest.raises(DataIntegrityError, match="identical index"):
            build_xs_alpha_composite_score(
                closes.iloc[1:], taker, funding, XsAlphaCompositeSpec(),
            )

        with pytest.raises(DataIntegrityError, match="ordered column"):
            build_xs_alpha_composite_score(
                closes.rename(columns={"A": "Z"}), taker, funding, XsAlphaCompositeSpec(),
            )

        dup = closes.copy()
        dup.index = pd.DatetimeIndex([closes.index[0]] * len(closes))
        with pytest.raises(DataIntegrityError, match="unique"):
            build_xs_alpha_composite_score(dup, taker, funding, XsAlphaCompositeSpec())

        unsorted = closes.sort_index(ascending=False)
        with pytest.raises(DataIntegrityError, match="monotonic"):
            build_xs_alpha_composite_score(
                unsorted, taker.reindex(unsorted.index), funding.reindex(unsorted.index),
                XsAlphaCompositeSpec(),
            )

        naive = closes.copy()
        naive.index = naive.index.tz_localize(None)
        with pytest.raises(DataIntegrityError, match="tz-aware"):
            build_xs_alpha_composite_score(naive, taker, funding, XsAlphaCompositeSpec())


class TestAlphaMultihorizonConstruction:
    def test_xsa_02_score_is_deterministic(self) -> None:
        closes, taker, funding = _alpha_inputs()
        spec = XsAlphaCompositeSpec()
        first = build_xs_alpha_composite_score(closes, taker, funding, spec)
        second = build_xs_alpha_composite_score(closes, taker, funding, spec)
        assert first.equals(second)

    def test_xsa_02_nine_equally_weighted_component_panels(self) -> None:
        # The score is the equal-weight (coefficient-one) sum of the nine z-scored
        # component panels -- three horizons for each of trend, funding carry, and
        # taker imbalance. A reference that reproduces the documented panel formulas
        # must match the construction exactly.
        closes, taker, funding = _alpha_inputs()
        score = build_xs_alpha_composite_score(closes, taker, funding, XsAlphaCompositeSpec())

        log_close = np.log(closes)
        dlog = log_close.diff()
        total: pd.DataFrame | None = None
        for window in (42, 84, 168):
            trend = np.log(closes / closes.shift(window)) / dlog.rolling(window).std()
            carry = -funding.shift(1).rolling(window).sum()
            taker_imbalance = taker.rolling(window).mean() - 0.5
            for component in (trend, carry, taker_imbalance):
                z = _ref_zscore(component)
                total = z if total is None else total + z
        assert total is not None
        assert np.allclose(score.to_numpy(), total.to_numpy(), atol=1e-12)

    def test_xsa_02_shared_construction_retains_preband_neutrality(self) -> None:
        closes, taker, funding = _alpha_inputs()
        weights = build_xs_alpha_weights(
            closes, taker, funding, XsAlphaCompositeSpec(),
            XsCompositeSpec(no_trade_band=0.0),
        )
        assert weights.index.equals(closes.index)
        assert list(weights.columns) == list(closes.columns)
        invested = weights[weights.abs().sum(axis=1) > 1e-12]
        assert (invested.sum(axis=1).abs() < 1e-9).all()
        assert ((invested.abs().sum(axis=1) - 1.0).abs() < 1e-9).all()

    def test_xsa_02_band_applies_on_weights_never_score(self) -> None:
        closes, taker, funding = _alpha_inputs()
        banded = build_xs_alpha_weights(
            closes, taker, funding, XsAlphaCompositeSpec(), XsCompositeSpec(),
        )
        no_band = build_xs_alpha_weights(
            closes, taker, funding, XsAlphaCompositeSpec(),
            XsCompositeSpec(no_trade_band=0.0),
        )
        assert not np.allclose(banded.to_numpy(), no_band.to_numpy())


class TestAlphaFamilyDecomposition:
    def test_xsv3_01_three_family_scores_sum_to_legacy_composite(self) -> None:
        closes, taker, funding = _alpha_inputs()
        spec = XsAlphaCompositeSpec()
        composite = build_xs_alpha_composite_score(closes, taker, funding, spec)
        families = build_xs_alpha_family_scores(closes, taker, funding, spec)
        assert list(families) == ["trend", "funding_contrarian", "taker_imbalance"]
        total = sum(families.values())
        assert np.allclose(total.to_numpy(), composite.to_numpy(), atol=1e-12)
        for name in ("trend", "funding_contrarian", "taker_imbalance"):
            frame = families[name]
            assert not frame.isna().any().any()
            assert frame.index.equals(closes.index)
            assert list(frame.columns) == list(closes.columns)

    def test_xsv3_01_family_scores_are_prefix_invariant(self) -> None:
        closes, taker, funding = _alpha_inputs()
        cutoff = 200
        full = build_xs_alpha_family_scores(
            closes, taker, funding, XsAlphaCompositeSpec(),
        )
        prefix = build_xs_alpha_family_scores(
            closes.iloc[:cutoff], taker.iloc[:cutoff], funding.iloc[:cutoff],
            XsAlphaCompositeSpec(),
        )
        for name, frame in full.items():
            assert np.allclose(
                frame.iloc[:cutoff].to_numpy(), prefix[name].to_numpy(), atol=1e-12,
            )

    def test_xsv3_01_family_weights_map_exactly_the_three_families(self) -> None:
        closes, taker, funding = _alpha_inputs()
        weights = build_xs_alpha_family_weights(
            closes, taker, funding, XsAlphaCompositeSpec(), XsCompositeSpec(),
        )
        assert list(weights) == ["trend", "funding_contrarian", "taker_imbalance"]
        for frame in weights.values():
            assert frame.index.equals(closes.index)
            assert list(frame.columns) == list(closes.columns)
            assert not frame.isna().any().any()


class TestDualFamilyAlphaWeights:
    def test_xsv5_01_equals_neutral_on_trend_plus_taker_only(self) -> None:
        idx = pd.date_range("2024-01-01", periods=200, freq="4h", tz="UTC")
        rng = np.random.default_rng(0)
        closes = pd.DataFrame({
            "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))),
            "B": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))),
        }, index=idx)
        taker = pd.DataFrame(
            np.clip(0.5 + rng.normal(0, 0.05, (200, 2)), 0.0, 1.0),
            index=idx, columns=["A", "B"],
        )
        funding = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
        alpha_spec = XsAlphaCompositeSpec()
        exec_spec = XsCompositeSpec()
        scores = build_xs_alpha_family_scores(closes, taker, funding, alpha_spec)
        expected = build_xs_neutral_weights(
            scores["trend"] + scores["taker_imbalance"],
            exec_spec.halflife_bars, exec_spec.no_trade_band,
        )
        actual = build_xs_alpha_dual_family_weights(
            closes, taker, funding, alpha_spec, exec_spec,
        )
        assert actual.equals(expected)
        assert list(actual.columns) == ["A", "B"]
        assert actual.index.equals(closes.index)

    def test_xsv5_01_bar_funding_input_does_not_change_output(self) -> None:
        # funding_contrarian is the only family driven by bar_funding; with it
        # dropped, the dual-family weights must be identical under any finite
        # funding panel.
        closes, taker, funding = _alpha_inputs()
        spec = XsAlphaCompositeSpec()
        exec_spec = XsCompositeSpec()
        baseline = build_xs_alpha_dual_family_weights(
            closes, taker, funding, spec, exec_spec,
        )
        other_funding = pd.DataFrame(
            np.zeros_like(funding.to_numpy(dtype=np.float64)),
            index=funding.index, columns=funding.columns,
        )
        other = build_xs_alpha_dual_family_weights(
            closes, taker, other_funding, spec, exec_spec,
        )
        assert other.equals(baseline)

    def test_xsv5_01_malformed_panels_fail_closed(self) -> None:
        closes, taker, funding = _alpha_inputs()
        bad_taker = taker.copy()
        bad_taker.iloc[5, 0] = np.nan
        with pytest.raises(DataIntegrityError, match="finite and in"):
            build_xs_alpha_dual_family_weights(
                closes, bad_taker, funding, XsAlphaCompositeSpec(), XsCompositeSpec(),
            )

class TestCausalFamilyInverseVolWeights:
    def test_xsv6_01_prior_window_fallback_and_causal_exclusion(self) -> None:
        # SCENARIO_XSV6_01_CAUSAL_VOL_WEIGHTS_EXCLUDE_CURRENT_BAR
        idx = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
        rng = np.random.default_rng(1)
        cols = ["trend", "funding_contrarian", "taker_imbalance"]
        base = pd.DataFrame(0.0, index=idx, columns=cols)
        base["trend"] = rng.normal(0, 0.02, 100)
        base["funding_contrarian"] = rng.normal(0, 0.001, 100)
        base["taker_imbalance"] = rng.normal(0, 0.005, 100)
        weights = _causal_family_inverse_vol_weights(base, 42)
        assert list(weights.columns) == cols
        # Rows before a full 42-bar strictly-prior window are equal-weight 1/3.
        assert np.allclose(weights.iloc[:42].to_numpy(), 1.0 / 3.0, atol=1e-9)
        assert bool(np.allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-9))
        # Mutating bar 80's own return must not change bar 80's weight but must
        # change bar 81's (its strictly-prior window includes bar 80).
        mutated = base.copy()
        mutated.iloc[80, 0] = 5.0
        w2 = _causal_family_inverse_vol_weights(mutated, 42)
        assert np.allclose(weights.iloc[80].to_numpy(), w2.iloc[80].to_numpy(), atol=1e-9)
        assert not np.allclose(weights.iloc[81].to_numpy(), w2.iloc[81].to_numpy(), atol=1e-9)

    def test_xsv6_01_zero_std_family_falls_back_and_row_still_normalizes(self) -> None:
        idx = pd.date_range("2024-01-01", periods=60, freq="4h", tz="UTC")
        rng = np.random.default_rng(7)
        cols = ["trend", "funding_contrarian", "taker_imbalance"]
        frame = pd.DataFrame(0.0, index=idx, columns=cols)
        frame["trend"] = rng.normal(0, 0.01, 60)
        # funding_contrarian and taker_imbalance stay constant at 0.0: their
        # trailing std is zero after the fallback window, so only trend tilts.
        weights = _causal_family_inverse_vol_weights(frame, 42)
        assert np.allclose(weights.iloc[:42].to_numpy(), 1.0 / 3.0, atol=1e-9)
        assert bool(np.allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-9))
        late = weights.iloc[50:]
        # Both constant families fall back to the same shared default and are
        # renormalized identically, so they must stay equal to each other while
        # the single non-degenerate family carries the bulk of the tilt.
        assert np.allclose(late["funding_contrarian"].to_numpy(), late["taker_imbalance"].to_numpy(), atol=1e-9)
        assert bool((late["trend"] > late["funding_contrarian"]).all())


class TestVolWeightedAlphaWeights:
    def test_xsv6_02_matches_manual_recompute(self) -> None:
        # SCENARIO_XSV6_02_VOL_WEIGHTED_MATCHES_MANUAL_RECOMPUTE
        idx = pd.date_range("2024-01-01", periods=250, freq="4h", tz="UTC")
        rng = np.random.default_rng(2)
        cols = ["A", "B", "C"]
        closes = pd.DataFrame(
            {c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 250))) for c in cols},
            index=idx,
        )
        opens = closes.shift(1).bfill()
        taker = pd.DataFrame(
            np.clip(0.5 + rng.normal(0, 0.05, (250, 3)), 0.0, 1.0),
            index=idx, columns=cols,
        )
        funding = pd.DataFrame(0.0, index=idx, columns=cols)
        alpha_spec = XsAlphaCompositeSpec()
        exec_spec = XsCompositeSpec()
        family_scores = build_xs_alpha_family_scores(closes, taker, funding, alpha_spec)
        family_weights = build_xs_alpha_family_weights(closes, taker, funding, alpha_spec, exec_spec)
        sleeve_returns = {}
        for name, fw in family_weights.items():
            equity, _turnover = run_xs_composite_ledger(fw, opens, funding, exec_spec)
            sleeve_returns[name] = equity.pct_change()
        sleeve_returns_frame = pd.DataFrame(sleeve_returns, index=idx)
        vol_weights = _causal_family_inverse_vol_weights(
            sleeve_returns_frame, alpha_spec.signal_windows[0],
        )
        combined = sum(
            vol_weights[name].to_numpy()[:, None] * family_scores[name].to_numpy()
            for name in family_scores
        )
        expected = build_xs_neutral_weights(
            pd.DataFrame(combined, index=idx, columns=cols),
            exec_spec.halflife_bars, exec_spec.no_trade_band,
        )
        actual = build_xs_alpha_vol_weighted_weights(
            closes, taker, funding, opens, alpha_spec, exec_spec,
        )
        assert actual.equals(expected)

    def test_xsv6_02_malformed_panels_fail_closed(self) -> None:
        closes, taker, funding = _alpha_inputs(rows=250)
        opens = closes.shift(1).bfill()
        bad_taker = taker.copy()
        bad_taker.iloc[5, 0] = np.nan
        with pytest.raises(DataIntegrityError, match="finite and in"):
            build_xs_alpha_vol_weighted_weights(
                closes, bad_taker, funding, opens, XsAlphaCompositeSpec(), XsCompositeSpec(),
            )


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
