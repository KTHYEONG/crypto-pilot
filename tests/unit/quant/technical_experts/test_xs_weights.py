"""Cross-sectional expert contract tests (split by behavioral domain)."""

from __future__ import annotations




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
import dataclasses
import numpy as np
import pandas as pd
import pytest
from src.common.errors import DataIntegrityError
from src.quant.technical_experts.cross_sectional import (
    XsAlphaCompositeSpec,
    XsCompositeSpec,
    _causal_family_inverse_vol_weights,
    _cross_sectional_zscore,
    _positioning_score,
    build_xs_alpha_composite_score,
    build_xs_alpha_dual_family_weights,
    build_xs_alpha_family_scores,
    build_xs_alpha_family_weights,
    build_xs_alpha_positioning_only_weights,
    build_xs_alpha_positioning_weights,
    build_xs_alpha_vol_weighted_weights,
    build_xs_alpha_weights,
    build_xs_neutral_weights,
    run_xs_composite_ledger,
)

from tests.unit.quant.technical_experts.test_cross_sectional import (  # noqa: F401
    _alpha_inputs,
    _ref_zscore,
    _score_frame,
)


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

class TestPositioningAlphaWeights:
    def test_xsp_01_dollar_neutral_unit_gross_on_synthetic_fixture(self) -> None:
        # SCENARIO_XS_POSITIONING_WEIGHTS_01
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
        long_short_ratio = pd.DataFrame(
            0.5 + 0.3 * rng.normal(0, 1.0, (250, 3)),
            index=idx, columns=cols,
        )
        alpha_spec = XsAlphaCompositeSpec()
        exec_spec = dataclasses.replace(XsCompositeSpec(), no_trade_band=0.0)
        weights = build_xs_alpha_positioning_weights(
            closes, taker, funding, long_short_ratio, opens, alpha_spec, exec_spec,
        )
        assert weights.index.equals(idx)
        assert list(weights.columns) == cols
        invested = weights[weights.abs().sum(axis=1) > 1e-12]
        assert (invested.sum(axis=1).abs() < 1e-9).all()
        assert ((invested.abs().sum(axis=1) - 1.0).abs() < 1e-9).all()

    def test_xsp_01_matches_manual_recompute(self) -> None:
        # SCENARIO_XS_POSITIONING_WEIGHTS_01 (construction lock)
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
        long_short_ratio = pd.DataFrame(
            0.5 + 0.3 * rng.normal(0, 1.0, (250, 3)),
            index=idx, columns=cols,
        )
        alpha_spec = XsAlphaCompositeSpec()
        exec_spec = XsCompositeSpec()
        family_scores = build_xs_alpha_family_scores(closes, taker, funding, alpha_spec)
        family_weights = build_xs_alpha_family_weights(
            closes, taker, funding, alpha_spec, exec_spec,
        )
        positioning_score = np.zeros((250, 3), dtype=np.float64)
        for window in alpha_spec.signal_windows:
            positioning_score += _cross_sectional_zscore(
                long_short_ratio.rolling(window).mean().to_numpy(dtype=np.float64),
            )
        family_scores["positioning"] = pd.DataFrame(
            -positioning_score, index=idx, columns=cols,
        )
        family_weights["positioning"] = build_xs_neutral_weights(
            family_scores["positioning"],
            exec_spec.halflife_bars, exec_spec.no_trade_band,
        )
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
        actual = build_xs_alpha_positioning_weights(
            closes, taker, funding, long_short_ratio, opens, alpha_spec, exec_spec,
        )
        assert actual.equals(expected)

    def test_xsp_01_high_lsr_precedes_negative_tilt_causally(self) -> None:
        # SCENARIO_XS_POSITIONING_WEIGHTS_01 (contrarian sign + causality)
        # The other families are cross-sectionally flat, so only the positioning
        # sleeve tilts: high long_short_ratio -> short, low -> long. The band
        # is disabled so the causality check reads the raw construction instead
        # of being absorbed by the stateful no-trade deadband.
        idx = pd.date_range("2024-01-01", periods=400, freq="4h", tz="UTC")
        cols = ["A", "B", "C", "D"]
        path = 100.0 * np.exp(np.linspace(0.0, 0.05, len(idx)))
        closes = pd.DataFrame(np.tile(path, (4, 1)).T, index=idx, columns=cols)
        opens = closes.shift(1).bfill()
        taker = pd.DataFrame(0.5, index=idx, columns=cols)
        funding = pd.DataFrame(0.0, index=idx, columns=cols)
        lsr = pd.DataFrame(1.0, index=idx, columns=cols)
        lsr.loc[idx[170]:, "A"] = 2.0
        lsr.loc[idx[170]:, "B"] = 1.8
        lsr.loc[idx[170]:, "C"] = 0.4
        lsr.loc[idx[170]:, "D"] = 0.4
        alpha_spec = XsAlphaCompositeSpec()
        exec_spec = dataclasses.replace(XsCompositeSpec(), no_trade_band=0.0)
        base = build_xs_alpha_positioning_weights(
            closes, taker, funding, lsr, opens, alpha_spec, exec_spec,
        )
        late = base.loc[idx[380]:]
        assert bool((late["A"] < 0.0).all())
        assert bool((late["B"] < 0.0).all())
        assert bool((late["C"] > 0.0).all())
        assert bool((late["D"] > 0.0).all())
        # Causality: lifting A's lsr to 4.0 from bar 301 onward must leave
        # every weight at or before bar 300 bit-identical (no lookahead), push
        # A's tilt more negative on the following bar 301, and never touch the
        # same bar 300.
        raised = lsr.copy()
        raised.loc[idx[301]:, "A"] = 4.0
        altered = build_xs_alpha_positioning_weights(
            closes, taker, funding, raised, opens, alpha_spec, exec_spec,
        )
        assert base.loc[:idx[300]].equals(altered.loc[:idx[300]])
        assert base.loc[idx[300], "A"] == altered.loc[idx[300], "A"]
        assert bool(altered.loc[idx[301], "A"] < base.loc[idx[301], "A"])

class TestPositioningOnlyAlphaWeights:
    def test_xatb_01_positioning_score_extraction_parity(self) -> None:
        # XATB-01-POSITIONING-SCORE-EXTRACTION-PARITY: the _positioning_score
        # helper is a byte-for-byte extraction of the inline block that used to
        # live inside build_xs_alpha_positioning_weights, so the already-admitted
        # v7 profile's output must be identical whether the inline block or the
        # shared helper is used.
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
        long_short_ratio = pd.DataFrame(
            0.5 + 0.3 * rng.normal(0, 1.0, (250, 3)),
            index=idx, columns=cols,
        )
        alpha_spec = XsAlphaCompositeSpec()
        exec_spec = dataclasses.replace(XsCompositeSpec(), no_trade_band=0.0)
        family_scores = build_xs_alpha_family_scores(closes, taker, funding, alpha_spec)
        family_weights = build_xs_alpha_family_weights(
            closes, taker, funding, alpha_spec, exec_spec,
        )
        family_scores["positioning"] = _positioning_score(closes, long_short_ratio, alpha_spec)
        family_weights["positioning"] = build_xs_neutral_weights(
            family_scores["positioning"],
            exec_spec.halflife_bars, exec_spec.no_trade_band,
        )
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
        actual = build_xs_alpha_positioning_weights(
            closes, taker, funding, long_short_ratio, opens, alpha_spec, exec_spec,
        )
        assert actual.equals(expected)

    def test_xatb_02_positioning_only_contrarian_sign(self) -> None:
        # XATB-02-POSITIONING-ONLY-CONTRARIAN-SIGN: the isolated positioning-only
        # leg must carry the same contrarian sign as the v7 family: high
        # long_short_ratio -> short (negative weight), low -> long (positive),
        # on the final bars.
        idx = pd.date_range("2024-01-01", periods=400, freq="4h", tz="UTC")
        cols = ["A", "B", "C", "D"]
        path = 100.0 * np.exp(np.linspace(0.0, 0.05, len(idx)))
        closes = pd.DataFrame(np.tile(path, (4, 1)).T, index=idx, columns=cols)
        lsr = pd.DataFrame(1.0, index=idx, columns=cols)
        lsr.loc[idx[170]:, "A"] = 2.0
        lsr.loc[idx[170]:, "B"] = 1.8
        lsr.loc[idx[170]:, "C"] = 0.4
        lsr.loc[idx[170]:, "D"] = 0.4
        alpha_spec = XsAlphaCompositeSpec()
        exec_spec = dataclasses.replace(XsCompositeSpec(), no_trade_band=0.0)
        weights = build_xs_alpha_positioning_only_weights(
            closes, lsr, alpha_spec, exec_spec,
        )
        late = weights.loc[idx[380]:]
        assert bool((late["A"] < 0.0).all())
        assert bool((late["B"] < 0.0).all())
        assert bool((late["C"] > 0.0).all())
        assert bool((late["D"] > 0.0).all())

    def test_xatb_03_positioning_only_fail_closed(self) -> None:
        # XATB-03-POSITIONING-ONLY-FAIL-CLOSED: malformed input (mismatched
        # indices, non-finite panels) raises DataIntegrityError/ValueError,
        # matching every sibling builder's fail-closed contract -- never a
        # silent reindex or zero-filling fallback.
        idx = pd.date_range("2022-01-01", periods=200, freq="4h", tz="UTC")
        closes = pd.DataFrame(
            {"HIGH": np.linspace(100.0, 110.0, 200), "LOW": np.linspace(100.0, 110.0, 200)},
            index=idx,
        )
        ratio = pd.DataFrame(
            {"HIGH": np.linspace(1.0, 3.0, 200), "LOW": np.full(200, 1.0)},
            index=idx,
        )
        alpha_spec = XsAlphaCompositeSpec()
        exec_spec = XsCompositeSpec()
        valid = build_xs_alpha_positioning_only_weights(closes, ratio, alpha_spec, exec_spec)
        assert bool(np.isfinite(valid.to_numpy()).all())
        with pytest.raises((DataIntegrityError, ValueError)):
            build_xs_alpha_positioning_only_weights(
                closes, ratio.iloc[:100], alpha_spec, exec_spec,
            )
        bad_closes = closes.copy()
        bad_closes.iloc[50, 0] = np.nan
        with pytest.raises((DataIntegrityError, ValueError)):
            build_xs_alpha_positioning_only_weights(
                bad_closes, ratio, alpha_spec, exec_spec,
            )
        # Non-finite long_short_ratio bars are tolerated (finite-only
        # cross-sectional z-score), matching the already-admitted v7 family --
        # never a silent reindex, but never a hard failure on real PIT panels
        # with leading-metrics gaps either.
        sparse_ratio = ratio.copy()
        sparse_ratio.iloc[:80, 0] = np.nan
        sparse_weights = build_xs_alpha_positioning_only_weights(
            closes, sparse_ratio, alpha_spec, exec_spec,
        )
        assert bool(np.isfinite(sparse_weights.to_numpy()).all())





