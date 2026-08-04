"""Contract scenarios XSC-01..XSC-05 and XSC-07 for the cross-sectional module.

XSC-01-NO-TRADE-BAND-STATEFUL, XSC-02-WEIGHTS-DOLLAR-NEUTRAL,
XSC-03-SPEC-FROZEN-BOUNDS, XSC-04-LEDGER-EXECUTION-LAG,
XSC-05-ADMISSION-SCALE-INVARIANT, XSC-07-COMPOSITE-BEATS-SINGLE-FAMILY.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.technical_experts.cross_sectional import (
    XsAdmissionConfig,
    XsCompositeSpec,
    apply_no_trade_band,
    build_xs_neutral_weights,
    evaluate_xs_admission,
    run_xs_composite_ledger,
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
