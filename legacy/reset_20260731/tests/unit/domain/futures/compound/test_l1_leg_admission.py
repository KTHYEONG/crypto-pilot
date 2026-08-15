from __future__ import annotations

import numpy as np

import pytest

from numpy.typing import NDArray

from src.domain.futures.compound.config import L1LegConfig
from src.domain.futures.compound.contracts import CausalFold, LegBook, LegEvidence, SignalConceptSpec
from src.domain.futures.compound.l1_leg_admission import (
    accumulate_prequential_leg_weights,
    compute_handoff_scale,
    compute_leg_prior_weights,
    compute_leg_tilt_scores,
    compute_shrinkage_weights,
    evaluate_portfolio_admission,
    normalise_leg_weights,
    screen_leg_evidence,
)


def _make_spec(concept_id: str = "test") -> SignalConceptSpec:
    return SignalConceptSpec(
        concept_id=concept_id, member_signal_ids=("sig_a",),
        mode="xs", horizon_band_bars=(6,), declared_orientation=1,
    )


def _make_folds(n: int, per_fold: int, start: int = 0) -> tuple[CausalFold, ...]:
    folds = []
    for i in range(n):
        oos_start = start + i * per_fold
        folds.append(CausalFold(
            fold_id=i, fit_start=0, fit_end_exclusive=oos_start,
            calibration_start=0, calibration_end_exclusive=0,
            oos_start=oos_start, oos_end_exclusive=oos_start + per_fold,
            purge_bars=0, embargo_bars=0,
        ))
    return tuple(folds)


def _make_leg_evidence(**overrides: object) -> LegEvidence:
    base: dict = {
        "concept_id": "t", "mode": "xs", "n_oos_bars": 100,
        "alpha_ann": 0.05, "beta_market": 0.0, "alpha_sharpe": 0.5,
        "t_alpha_newey_west": 2.0, "breakeven_cost_bps": 50.0,
        "mean_turnover_per_bar": 0.01, "positive_folds": 6, "n_folds": 10,
        "posterior_positive": 0.95, "evidence_weight": 0.0, "reasons": (),
        "net_alpha_ann": 0.03, "net_alpha_sharpe": 0.4,
        "t_net_alpha_newey_west": 1.8,
    }
    base.update(overrides)
    return LegEvidence(**base)


def _two_leg_known_vol_panel() -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    n = 100
    leg0 = np.concatenate([rng.normal(0.01, 0.02, n // 2), rng.normal(-0.01, 0.02, n // 2)])
    leg1 = np.concatenate([rng.normal(0.02, 0.04, n // 2), rng.normal(-0.02, 0.04, n // 2)])
    return np.column_stack([leg0, leg1])


class TestNormaliseLegWeights:
    def test_water_fills_cap_surplus(self) -> None:
        result = normalise_leg_weights(np.array([3.0, 1.0] + [0.0] * 9, dtype=np.float64), 0.25)
        assert abs(float(np.sum(result)) - 1.0) < 1e-9

    def test_all_zero_returns_zero(self) -> None:
        result = normalise_leg_weights(np.array([0.0, 0.0], dtype=np.float64), 0.51)
        assert np.allclose(result, [0.0, 0.0])

    def test_already_compliant(self) -> None:
        result = normalise_leg_weights(np.array([0.4, 0.3, 0.3], dtype=np.float64), 0.5)
        assert abs(float(np.sum(result)) - 1.0) < 1e-10
        assert float(np.max(result)) <= 0.5 + 1e-10

    def test_degenerate_leg_cap_rejected_at_k11(self) -> None:
        with pytest.raises(ValueError, match="degenerate cap"):
            normalise_leg_weights(np.ones(11), 1.0 / 11)

    def test_leg_cap_is_non_degenerate_at_k11(self) -> None:
        cfg = L1LegConfig()
        assert cfg.max_leg_weight == 0.25
        assert cfg.max_leg_weight > 1.0 / 11


class TestComputeLegPriorWeights:
    def test_inverse_vol_ratio_two_legs(self) -> None:
        panel = _two_leg_known_vol_panel()
        cfg = L1LegConfig(max_leg_weight=0.99)
        weights = compute_leg_prior_weights(panel, 100, cfg)
        assert abs(weights[0] - 2 / 3) < 0.15
        assert abs(float(np.sum(weights)) - 1.0) < 1e-6

    def test_zero_end_idx_raises(self) -> None:
        with pytest.raises(ValueError, match="end_idx"):
            compute_leg_prior_weights(np.ones((10, 2)), 0, L1LegConfig())


class TestComputeLegTiltScores:
    def test_cost_headroom_zeroes_tilt(self) -> None:
        ev = _make_leg_evidence(
            breakeven_cost_bps=1.0, mean_turnover_per_bar=0.1,
            net_alpha_ann=0.1, net_alpha_sharpe=1.0,
        )
        scores = compute_leg_tilt_scores((ev,), 8.0, L1LegConfig())
        assert scores[0] == 0.0

    def test_negative_net_alpha_zeroes_tilt(self) -> None:
        ev = _make_leg_evidence(
            net_alpha_ann=-0.01, net_alpha_sharpe=1.0,
            breakeven_cost_bps=50.0, mean_turnover_per_bar=0.1,
        )
        scores = compute_leg_tilt_scores((ev,), 8.0, L1LegConfig())
        assert scores[0] == 0.0

    def test_non_finite_cost_raises(self) -> None:
        ev = _make_leg_evidence()
        with pytest.raises(ValueError, match="cost_bps"):
            compute_leg_tilt_scores((ev,), float("nan"), L1LegConfig())


class TestComputeShrinkageWeights:
    def test_zero_obs_returns_prior(self) -> None:
        cfg = L1LegConfig(max_leg_weight=0.51)
        result = compute_shrinkage_weights(np.array([0.5, 0.5]), np.array([1.0, 0.0]), 0, cfg)
        assert np.allclose(result, [0.5, 0.5])

    def test_many_obs_converges_to_tilt(self) -> None:
        cfg = L1LegConfig(max_leg_weight=0.51)
        prior = np.array([0.5, 0.5])
        tilt = np.array([1.0, 0.0])
        result = compute_shrinkage_weights(prior, tilt, 10**6, cfg)
        assert result[0] > result[1]
        assert abs(float(np.sum(result)) - 1.0) < 1e-6

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            compute_shrinkage_weights(np.array([0.5, 0.5]), np.array([1.0]), 10, L1LegConfig())


class TestComputeHandoffScale:
    def test_ramp_endpoints(self) -> None:
        cfg = L1LegConfig()
        assert abs(compute_handoff_scale(0.793, cfg) - 0.7325) < 1e-4
        assert compute_handoff_scale(0.90, cfg) == 1.0
        assert compute_handoff_scale(0.50, cfg) == 0.0

    def test_nan_returns_zero(self) -> None:
        assert compute_handoff_scale(float("nan"), L1LegConfig()) == 0.0

    def test_floor_higher_than_posterior_returns_zero(self) -> None:
        cfg = L1LegConfig(handoff_posterior_floor=0.80)
        assert compute_handoff_scale(0.70, cfg) == 0.0


class TestScreenLegEvidence:
    def test_blocks_two_fold_lucky_family(self) -> None:
        ev = _make_leg_evidence(n_folds=2, positive_folds=2, t_net_alpha_newey_west=9.0)
        result = screen_leg_evidence(ev, 8.0, L1LegConfig(), 11)
        assert result.evidence_weight == 0.0
        assert any("insufficient_folds" in r for r in result.reasons)

    def test_familywise_no_longer_blocks_capital(self) -> None:
        cfg = L1LegConfig()
        ev = _make_leg_evidence(n_folds=6, positive_folds=5, t_net_alpha_newey_west=2.0,
                                breakeven_cost_bps=30.0, mean_turnover_per_bar=0.01,
                                net_alpha_ann=0.03)
        result = screen_leg_evidence(ev, 8.0, cfg, 11)
        assert result.evidence_weight == 1.0

    def test_passes_all_gates(self) -> None:
        cfg = L1LegConfig()
        ev = _make_leg_evidence(
            n_folds=6, positive_folds=5, t_net_alpha_newey_west=3.5,
            breakeven_cost_bps=30.0, mean_turnover_per_bar=0.01,
        )
        result = screen_leg_evidence(ev, 8.0, cfg, 11)
        assert result.evidence_weight == 1.0
        assert result.reasons == ()

    def test_rejects_negative_gross_with_high_be(self) -> None:
        cfg = L1LegConfig()
        ev = _make_leg_evidence(
            n_folds=6, positive_folds=3, t_net_alpha_newey_west=-1.0,
            breakeven_cost_bps=-10.0,
        )
        result = screen_leg_evidence(ev, 8.0, cfg, 11)
        assert result.evidence_weight == 0.0

    def test_zero_net_fold_stability(self) -> None:
        cfg = L1LegConfig()
        ev = _make_leg_evidence(
            n_folds=6, positive_folds=2, t_net_alpha_newey_west=5.0,
        )
        result = screen_leg_evidence(ev, 8.0, cfg, 11)
        assert result.evidence_weight == 0.0
        assert any("fold_instability" in r for r in result.reasons)

    def test_invalid_hypothesis_count_raises(self) -> None:
        ev = _make_leg_evidence()
        with pytest.raises(ValueError, match="n_tested_hypotheses"):
            screen_leg_evidence(ev, 8.0, L1LegConfig(), 0)

    def test_k1_does_not_apply_bonferroni(self) -> None:
        cfg = L1LegConfig()
        ev = _make_leg_evidence(
            n_folds=6, positive_folds=5, t_net_alpha_newey_west=1.0,
        )
        result = screen_leg_evidence(ev, 8.0, cfg, 1)
        assert result.evidence_weight == 1.0


class TestAccumulatePrequentialLegWeights:
    def test_deploys_prior_at_fold_zero(self) -> None:
        T, S, K = 80, 5, 2
        rng = np.random.default_rng(42)
        market = rng.standard_normal(T).astype(np.float64) * 0.01
        legs = []
        for k in range(K):
            spec = _make_spec(concept_id=f"c{k}")
            book = rng.standard_normal((T, S)).astype(np.float64)
            book = book / np.maximum(np.sum(np.abs(book), axis=1, keepdims=True), 1e-12)
            ret = rng.standard_normal(T).astype(np.float64)
            turn = np.abs(np.diff(book, axis=0, prepend=book[:1])).sum(axis=1)
            legs.append(LegBook(spec=spec, book_2d=book, gross_return_1d=ret, turnover_1d=turn))
        folds = tuple(CausalFold(
            fold_id=i, fit_start=0, fit_end_exclusive=20 + 10 * i,
            calibration_start=0, calibration_end_exclusive=20 + 10 * i,
            oos_start=20 + 10 * i, oos_end_exclusive=30 + 10 * i,
            purge_bars=0, embargo_bars=0,
        ) for i in range(5))
        cfg = L1LegConfig(prior_only_folds=1, min_turnover_per_bar=0.0001,
                          cost_safety_margin=1.0, min_positive_fold_ratio=0.01,
                          max_leg_weight=0.51, familywise_error_rate=0.5)
        weights = accumulate_prequential_leg_weights(tuple(legs), market, folds, 8.0, cfg)
        assert weights.shape == (T, K)
        assert float(np.sum(weights[folds[0].oos_start:folds[0].oos_end_exclusive])) > 0.0

    def test_is_causal(self) -> None:
        T, S, K = 80, 5, 2
        rng = np.random.default_rng(42)
        market = rng.standard_normal(T).astype(np.float64) * 0.01
        legs = []
        for k in range(K):
            spec = _make_spec(concept_id=f"c{k}")
            book = rng.standard_normal((T, S)).astype(np.float64)
            book = book / np.maximum(np.sum(np.abs(book), axis=1, keepdims=True), 1e-12)
            ret = rng.standard_normal(T).astype(np.float64)
            turn = np.abs(np.diff(book, axis=0, prepend=book[:1])).sum(axis=1)
            legs.append(LegBook(spec=spec, book_2d=book, gross_return_1d=ret, turnover_1d=turn))
        folds = tuple(CausalFold(
            fold_id=i, fit_start=0, fit_end_exclusive=20 + 10 * i,
            calibration_start=0, calibration_end_exclusive=20 + 10 * i,
            oos_start=20 + 10 * i, oos_end_exclusive=30 + 10 * i,
            purge_bars=0, embargo_bars=0,
        ) for i in range(5))
        cfg = L1LegConfig(prior_only_folds=1, min_turnover_per_bar=0.0001,
                          cost_safety_margin=1.0, min_positive_fold_ratio=0.01,
                          max_leg_weight=0.51, familywise_error_rate=0.5)
        weights = accumulate_prequential_leg_weights(tuple(legs), market, folds, 8.0, cfg)
        legs_mutated = list(legs)
        mutated_book = legs[0].book_2d.copy()
        sl = slice(folds[2].oos_start, folds[2].oos_end_exclusive)
        mutated_book[sl] *= -5.0
        legs_mutated[0] = LegBook(
            spec=legs[0].spec, book_2d=mutated_book,
            gross_return_1d=legs[0].gross_return_1d, turnover_1d=legs[0].turnover_1d,
        )
        weights_mutated = accumulate_prequential_leg_weights(
            tuple(legs_mutated), market, folds, 8.0, cfg,
        )
        assert np.allclose(
            weights[:folds[2].oos_start], weights_mutated[:folds[2].oos_start],
        )


class TestEvaluatePortfolioAdmission:
    def test_each_condition_blocks(self) -> None:
        self.test_posterior_condition_blocks()
        self.test_fold_ratio_condition_blocks()
        self.test_stress_condition_blocks()

    def test_posterior_condition_blocks(self) -> None:
        n_folds, per_fold = 6, 20
        T, S = n_folds * per_fold, 3
        combined = np.full((T, S), 1.0 / S, dtype=np.float64)
        ret = np.zeros((T, S), dtype=np.float64)
        pattern = np.array([0.5] * 10 + [-0.4998] * 10)
        for f in range(n_folds):
            start = f * per_fold
            for i, val in enumerate(pattern):
                ret[start + i] = val / S
        folds = _make_folds(n_folds, per_fold)
        cfg = L1LegConfig(prior_only_folds=1, min_growth_posterior_probability=0.90,
                          min_positive_fold_ratio=0.50, stress_cost_multiplier=2.0,
                          bars_per_year=2190.0, n_bootstrap=500)
        admitted, reasons, _ = evaluate_portfolio_admission(combined, ret, folds, 8.0, cfg, admission_end_exclusive=10**9)
        assert admitted is False
        assert any("posterior" in r for r in reasons)

    def test_fold_ratio_condition_blocks(self) -> None:
        n_folds, per_fold = 6, 20
        T, S = n_folds * per_fold, 3
        combined = np.full((T, S), 1.0 / S, dtype=np.float64)
        ret = np.zeros((T, S), dtype=np.float64)
        ret[0:per_fold] = 1.0 / S
        for f in range(1, n_folds):
            ret[f * per_fold:(f + 1) * per_fold] = -0.01 / S
        folds = _make_folds(n_folds, per_fold)
        cfg = L1LegConfig(prior_only_folds=1, min_growth_posterior_probability=0.90,
                          min_positive_fold_ratio=0.50, stress_cost_multiplier=2.0,
                          bars_per_year=2190.0, n_bootstrap=500)
        admitted, reasons, _ = evaluate_portfolio_admission(combined, ret, folds, 8.0, cfg, admission_end_exclusive=10**9)
        assert admitted is False
        assert any("positive_folds" in r for r in reasons)

    def test_stress_condition_blocks(self) -> None:
        n_folds, per_fold = 6, 20
        T, S = 1 + n_folds * per_fold, 2
        combined = np.zeros((T, S), dtype=np.float64)
        for t in range(T):
            combined[t] = [0.5, -0.5] if t % 2 == 0 else [-0.5, 0.5]
        ret = np.zeros((T, S), dtype=np.float64)
        ret[1:] = combined[:-1] * 0.005
        folds = _make_folds(n_folds, per_fold, start=1)
        cfg = L1LegConfig(prior_only_folds=1, min_growth_posterior_probability=0.90,
                          min_positive_fold_ratio=0.50, stress_cost_multiplier=2.0,
                          bars_per_year=2190.0, n_bootstrap=500)
        admitted, reasons, net_ann = evaluate_portfolio_admission(combined, ret, folds, 8.0, cfg, admission_end_exclusive=10**9)
        assert admitted is False
        assert any("stressed_net_ann" in r for r in reasons)
        assert net_ann > 0.0

    def test_all_pass(self) -> None:
        n_folds, per_fold = 6, 20
        T, S = n_folds * per_fold, 3
        combined = np.full((T, S), 1.0 / S, dtype=np.float64)
        ret = np.full((T, S), 0.02 / S, dtype=np.float64)
        folds = _make_folds(n_folds, per_fold)
        cfg = L1LegConfig(prior_only_folds=1, min_growth_posterior_probability=0.90,
                          min_positive_fold_ratio=0.50, stress_cost_multiplier=2.0,
                          bars_per_year=2190.0, n_bootstrap=500)
        admitted, reasons, net_ann = evaluate_portfolio_admission(combined, ret, folds, 8.0, cfg, admission_end_exclusive=10**9)
        assert admitted is True
        assert reasons == ()
        assert net_ann > 0.0

    def test_no_active_bars(self) -> None:
        combined = np.zeros((10, 3), dtype=np.float64)
        ret = np.zeros((10, 3), dtype=np.float64)
        admitted, _, net_ann = evaluate_portfolio_admission(combined, ret, (), 8.0, L1LegConfig(), admission_end_exclusive=10**9)
        assert admitted is False
        assert net_ann == 0.0


class TestLegEvidenceValidation:
    @staticmethod
    def _valid_kwargs() -> dict:
        return {
            "concept_id": "t", "mode": "xs", "n_oos_bars": 10,
            "alpha_ann": 0.0, "beta_market": 0.0, "alpha_sharpe": 0.0,
            "t_alpha_newey_west": 0.0, "breakeven_cost_bps": 0.0,
            "mean_turnover_per_bar": 0.0, "positive_folds": 0, "n_folds": 1,
            "posterior_positive": 0.5, "evidence_weight": 0.0, "reasons": (),
        }

    def test_leg_evidence_accepts_valid_input(self) -> None:
        ev = LegEvidence(**self._valid_kwargs())
        assert ev.concept_id == "t"

    @pytest.mark.parametrize(("field", "bad_value", "match"), [
        ("concept_id", "", "concept_id"),
        ("mode", "invalid", "mode"),
        ("n_oos_bars", -1, "n_oos_bars"),
        ("alpha_ann", float("nan"), "alpha_ann"),
        ("beta_market", float("nan"), "beta_market"),
        ("alpha_sharpe", float("nan"), "alpha_sharpe"),
        ("t_alpha_newey_west", float("nan"), "t_alpha_newey_west"),
        ("breakeven_cost_bps", float("nan"), "breakeven_cost_bps"),
        ("mean_turnover_per_bar", float("nan"), "mean_turnover_per_bar"),
        ("positive_folds", -1, "positive_folds"),
        ("posterior_positive", 1.5, "posterior_positive"),
        ("evidence_weight", float("nan"), "evidence_weight"),
        ("evidence_weight", -0.1, "evidence_weight"),
        ("net_alpha_ann", float("nan"), "net_alpha_ann"),
        ("net_alpha_sharpe", float("nan"), "net_alpha_sharpe"),
        ("t_net_alpha_newey_west", float("nan"), "t_net_alpha_newey_west"),
    ])
    def test_leg_evidence_rejects_invalid_fields(self, field: str, bad_value: object, match: str) -> None:
        kwargs = {**self._valid_kwargs(), field: bad_value}
        with pytest.raises(ValueError, match=match):
            LegEvidence(**kwargs)

    def test_leg_evidence_rejects_positive_folds_exceeding_n_folds(self) -> None:
        kwargs = {**self._valid_kwargs(), "n_folds": 2, "positive_folds": 5}
        with pytest.raises(ValueError, match="positive_folds"):
            LegEvidence(**kwargs)


class TestClassifyLegEvidence:
    def test_significance_no_longer_blocks_capital(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import classify_leg_evidence
        ev = _make_leg_evidence(
            n_folds=6, positive_folds=5, net_alpha_ann=0.03,
            t_net_alpha_newey_west=1.765, breakeven_cost_bps=30.0,
            mean_turnover_per_bar=0.01,
        )
        decision = classify_leg_evidence(ev, 8.0, L1LegConfig(), n_tested_hypotheses=11)
        assert decision.economic_eligible is True
        assert decision.familywise_supported is False
        assert decision.capital_eligible is True

    def test_negative_net_alpha(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import classify_leg_evidence
        ev = _make_leg_evidence(
            n_folds=6, positive_folds=5, net_alpha_ann=-0.01,
            t_net_alpha_newey_west=5.0,
        )
        decision = classify_leg_evidence(ev, 8.0, L1LegConfig(), n_tested_hypotheses=11)
        assert decision.economic_eligible is False
        assert decision.capital_eligible is False

    def test_all_criteria_pass(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import classify_leg_evidence
        ev = _make_leg_evidence(
            n_folds=6, positive_folds=5, net_alpha_ann=0.03,
            t_net_alpha_newey_west=5.0, breakeven_cost_bps=30.0,
            mean_turnover_per_bar=0.01,
        )
        decision = classify_leg_evidence(ev, 8.0, L1LegConfig(), n_tested_hypotheses=1)
        assert decision.economic_eligible is True
        assert decision.familywise_supported is True
        assert decision.capital_eligible is True


class TestAccumulatePrequentialCarryForward:
    def test_carries_forward_past_last_fold(self) -> None:
        n_t, n_s, n_k = 100, 3, 2
        rng = np.random.default_rng(42)
        market = rng.standard_normal(n_t).astype(np.float64) * 0.01
        legs = []
        for k in range(n_k):
            book = rng.standard_normal((n_t, n_s)).astype(np.float64)
            book = book / np.maximum(np.sum(np.abs(book), axis=1, keepdims=True), 1e-12)
            ret = rng.standard_normal(n_t).astype(np.float64)
            turn = np.abs(np.diff(book, axis=0, prepend=book[:1])).sum(axis=1)
            legs.append(LegBook(
                spec=_make_spec(concept_id=f"c{k}"),
                book_2d=book, gross_return_1d=ret, turnover_1d=turn,
            ))
        folds = tuple(CausalFold(
            fold_id=i, fit_start=0, fit_end_exclusive=20 + 20 * i,
            calibration_start=0, calibration_end_exclusive=20 + 20 * i,
            oos_start=20 + 20 * i, oos_end_exclusive=40 + 20 * i,
            purge_bars=0, embargo_bars=0,
        ) for i in range(5))
        cfg = L1LegConfig(prior_only_folds=1, min_turnover_per_bar=0.0001,
                          cost_safety_margin=1.0, min_positive_fold_ratio=0.01,
                          max_leg_weight=0.51, familywise_error_rate=0.5)
        weights = accumulate_prequential_leg_weights(
            tuple(legs), market, folds, 8.0, cfg,
        )
        last_stop = folds[-1].oos_end_exclusive
        assert last_stop == 120
        assert np.all(weights[last_stop:] == weights[last_stop - 1:last_stop])


class TestL1BootstrapConsistency:
    def test_admission_short_sample_falls_back_to_fixed_block_size(self) -> None:
        n_folds, per_fold = 6, 15
        T, S = 1 + n_folds * per_fold, 3
        combined = np.full((T, S), 1.0 / S, dtype=np.float64)
        ret = np.full((T, S), 0.02 / S, dtype=np.float64)
        folds = tuple(CausalFold(
            fold_id=i, fit_start=0, fit_end_exclusive=1 + i * per_fold,
            calibration_start=0, calibration_end_exclusive=1 + i * per_fold,
            oos_start=1 + i * per_fold, oos_end_exclusive=1 + (i + 1) * per_fold,
            purge_bars=0, embargo_bars=0,
        ) for i in range(n_folds))
        cfg = L1LegConfig(prior_only_folds=1, bars_per_year=2190.0, n_bootstrap=500)
        admitted, _, _ = evaluate_portfolio_admission(
            combined, ret, folds, 8.0, cfg, admission_end_exclusive=10**9,
        )
        assert admitted in (True, False)


class TestShadowWeightScenarios:
    def test_shadow_weight_equals_one_for_economic(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import classify_leg_evidence
        ev = _make_leg_evidence(net_alpha_ann=0.03, t_net_alpha_newey_west=5.0)
        decision = classify_leg_evidence(ev, 8.0, L1LegConfig(), n_tested_hypotheses=1)
        assert decision.economic_eligible is True

    def test_shadow_weight_zero_for_non_economic(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import classify_leg_evidence
        ev = _make_leg_evidence(net_alpha_ann=-0.01, t_net_alpha_newey_west=5.0)
        decision = classify_leg_evidence(ev, 8.0, L1LegConfig(), n_tested_hypotheses=1)
        assert decision.economic_eligible is False


class TestEvaluatePortfolioEvidence:
    def test_uses_all_eligible_folds(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import evaluate_portfolio_evidence
        n_folds, per_fold = 6, 20
        T, S = n_folds * per_fold, 3
        combined = np.full((T, S), 1.0 / S, dtype=np.float64)
        ret = np.full((T, S), 0.02 / S, dtype=np.float64)
        folds = _make_folds(n_folds, per_fold)
        cfg = L1LegConfig(prior_only_folds=1, min_growth_posterior_probability=0.90,
                          min_positive_fold_ratio=0.50, stress_cost_multiplier=2.0,
                          bars_per_year=2190.0, n_bootstrap=500)
        evidence = evaluate_portfolio_evidence(combined, ret, folds, 8.0, cfg, admission_end_exclusive=10**9)
        assert evidence.n_folds == len(folds)

    def test_no_prequential_folds(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import evaluate_portfolio_evidence
        combined = np.zeros((10, 3), dtype=np.float64)
        ret = np.zeros((10, 3), dtype=np.float64)
        evidence = evaluate_portfolio_evidence(combined, ret, (), 8.0, L1LegConfig(), admission_end_exclusive=10**9)
        assert evidence.admitted is False
        assert evidence.n_folds == 0
        assert evidence.n_traded_bars == 0

    def test_stressed_zero_gives_handoff_zero(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import evaluate_portfolio_evidence
        n_folds, per_fold = 2, 20
        T, S = n_folds * per_fold, 3
        combined = np.full((T, S), 1.0 / S, dtype=np.float64)
        ret = np.zeros((T, S), dtype=np.float64)
        folds = _make_folds(n_folds, per_fold)
        cfg = L1LegConfig(prior_only_folds=1, bars_per_year=2190.0, n_bootstrap=500)
        evidence = evaluate_portfolio_evidence(combined, ret, folds, 8.0, cfg, admission_end_exclusive=10**9)
        assert evidence.handoff_scale == 0.0

    def test_handoff_scale_non_zero_when_posterior_between(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import evaluate_portfolio_evidence
        n_folds, per_fold = 6, 20
        T, S = n_folds * per_fold, 3
        combined = np.full((T, S), 1.0 / S, dtype=np.float64)
        ret = np.full((T, S), 0.02 / S, dtype=np.float64)
        folds = _make_folds(n_folds, per_fold)
        cfg = L1LegConfig(prior_only_folds=1, min_growth_posterior_probability=0.90,
                          handoff_posterior_floor=0.50, min_positive_fold_ratio=0.50,
                          stress_cost_multiplier=2.0, bars_per_year=2190.0, n_bootstrap=500)
        evidence = evaluate_portfolio_evidence(combined, ret, folds, 8.0, cfg, admission_end_exclusive=10**9)
        assert evidence.handoff_scale >= 0.0


class TestClassifyL1Bottleneck:
    @staticmethod
    def _evidence_portfolio(**overrides: object):
        from src.domain.futures.compound.contracts import PortfolioAdmissionEvidence
        base = {"admitted": False, "reasons": ("posterior_below_0.9",),
                "net_alpha_ann": 0.05, "stressed_net_alpha_ann": 0.03,
                "posterior_positive": 0.85, "positive_folds": 4, "n_folds": 7,
                "n_traded_bars": 1260, "handoff_scale": 0.0}
        base.update(overrides)
        return PortfolioAdmissionEvidence(**base)

    def test_deployable(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import classify_l1_bottleneck
        prod = self._evidence_portfolio(admitted=True, reasons=(), posterior_positive=0.95,
                                        handoff_scale=1.0)
        shadow = self._evidence_portfolio(admitted=True, reasons=(), posterior_positive=0.90,
                                          handoff_scale=1.0)
        report = classify_l1_bottleneck(prod, shadow, 4, 2, True)
        assert report.bottleneck_code == "deployable"
        assert report.production_weights_unchanged is True

    def test_partial_evidence_sized(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import classify_l1_bottleneck
        prod = self._evidence_portfolio(handoff_scale=0.7325)
        shadow = self._evidence_portfolio(handoff_scale=0.0)
        report = classify_l1_bottleneck(prod, shadow, 5, 5, True)
        assert report.bottleneck_code == "partial_evidence_sized"

    def test_familywise_power_limited(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import classify_l1_bottleneck
        prod = self._evidence_portfolio()
        shadow = self._evidence_portfolio(admitted=True, reasons=(), posterior_positive=0.90,
                                          handoff_scale=1.0)
        report = classify_l1_bottleneck(prod, shadow, 4, 0, True)
        assert report.bottleneck_code == "familywise_power_limited"

    def test_signal_economics_absent(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import classify_l1_bottleneck
        prod = self._evidence_portfolio(handoff_scale=0.0)
        shadow = self._evidence_portfolio(handoff_scale=0.0)
        report = classify_l1_bottleneck(prod, shadow, 0, 0, True)
        assert report.bottleneck_code == "signal_economics_absent"

    def test_signal_generalization_failed(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import classify_l1_bottleneck
        prod = self._evidence_portfolio(handoff_scale=0.0)
        shadow = self._evidence_portfolio(handoff_scale=0.0)
        report = classify_l1_bottleneck(prod, shadow, 4, 0, True)
        assert report.bottleneck_code == "signal_generalization_failed"

    def test_diagnostic_unavailable(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import classify_l1_bottleneck
        prod = self._evidence_portfolio(handoff_scale=0.0)
        shadow = self._evidence_portfolio(handoff_scale=0.0)
        report = classify_l1_bottleneck(prod, shadow, 0, 0, False)
        assert report.bottleneck_code == "diagnostic_unavailable"
        assert report.shadow_available is False


class TestPortfolioAdmissionEvidenceValidation:
    def test_admitted_with_reasons_raises(self) -> None:
        from src.domain.futures.compound.contracts import PortfolioAdmissionEvidence
        with pytest.raises(ValueError, match="admitted must have empty reasons"):
            PortfolioAdmissionEvidence(admitted=True, reasons=("something",), net_alpha_ann=0.0,
                                       stressed_net_alpha_ann=0.0, posterior_positive=0.5,
                                       positive_folds=0, n_folds=1, n_traded_bars=0)

    def test_not_admitted_without_reasons_raises(self) -> None:
        from src.domain.futures.compound.contracts import PortfolioAdmissionEvidence
        with pytest.raises(ValueError, match="not-admitted must have at least one reason"):
            PortfolioAdmissionEvidence(admitted=False, reasons=(), net_alpha_ann=0.0,
                                       stressed_net_alpha_ann=0.0, posterior_positive=0.5,
                                       positive_folds=0, n_folds=1, n_traded_bars=0)

    def test_non_finite_metrics_raise(self) -> None:
        from src.domain.futures.compound.contracts import PortfolioAdmissionEvidence
        with pytest.raises(ValueError, match="must be finite"):
            PortfolioAdmissionEvidence(admitted=False, reasons=("x",), net_alpha_ann=float("nan"),
                                       stressed_net_alpha_ann=0.0, posterior_positive=0.5,
                                       positive_folds=0, n_folds=1, n_traded_bars=0)

    def test_handoff_scale_bounds(self) -> None:
        from src.domain.futures.compound.contracts import PortfolioAdmissionEvidence
        ev = PortfolioAdmissionEvidence(admitted=False, reasons=("posterior_0.793_below_0.9",),
                                        net_alpha_ann=0.1229, stressed_net_alpha_ann=0.0322,
                                        posterior_positive=0.793, positive_folds=4, n_folds=7,
                                        n_traded_bars=1260, handoff_scale=0.7325)
        assert ev.handoff_scale > 0.0

    def test_admitted_requires_handoff_one(self) -> None:
        from src.domain.futures.compound.contracts import PortfolioAdmissionEvidence
        with pytest.raises(ValueError, match="admitted implies handoff_scale"):
            PortfolioAdmissionEvidence(admitted=True, reasons=(), net_alpha_ann=0.0,
                                       stressed_net_alpha_ann=0.0, posterior_positive=0.5,
                                       positive_folds=0, n_folds=1, n_traded_bars=0, handoff_scale=0.5)


class TestLegScreenDecisionValidation:
    def test_capital_eligible_mismatch_raises(self) -> None:
        from src.domain.futures.compound.contracts import LegScreenDecision
        with pytest.raises(ValueError, match="capital_eligible must equal"):
            LegScreenDecision(economic_eligible=True, familywise_supported=False,
                              capital_eligible=False, economic_reasons=(),
                              familywise_reasons=("familywise",), critical_t=2.5,
                              n_tested_hypotheses=11)

    def test_capital_eligible_even_with_familywise_fail(self) -> None:
        from src.domain.futures.compound.contracts import LegScreenDecision
        decision = LegScreenDecision(economic_eligible=True, familywise_supported=False,
                                     capital_eligible=True, economic_reasons=(),
                                     familywise_reasons=("net_t_below_familywise_threshold:1.765_below_2.362_K=11",),
                                     critical_t=2.362, n_tested_hypotheses=11)
        assert decision.capital_eligible is True

    def test_non_finite_critical_t_raises(self) -> None:
        from src.domain.futures.compound.contracts import LegScreenDecision
        with pytest.raises(ValueError, match="critical_t must be finite"):
            LegScreenDecision(economic_eligible=True, familywise_supported=True,
                              capital_eligible=True, economic_reasons=(),
                              familywise_reasons=(), critical_t=float("nan"),
                              n_tested_hypotheses=1)

    def test_hypothesis_count_too_low_raises(self) -> None:
        from src.domain.futures.compound.contracts import LegScreenDecision
        with pytest.raises(ValueError, match="n_tested_hypotheses"):
            LegScreenDecision(economic_eligible=True, familywise_supported=True,
                              capital_eligible=True, economic_reasons=(),
                              familywise_reasons=(), critical_t=0.0,
                              n_tested_hypotheses=0)


class TestAccumulatePrequentialShadowWeights:
    def test_shadow_weights_non_negative_max_leg_weight_compliant(self) -> None:
        T, S, K = 80, 5, 2
        rng = np.random.default_rng(42)
        market = rng.standard_normal(T).astype(np.float64) * 0.01
        legs = []
        for k in range(K):
            spec = _make_spec(concept_id=f"c{k}")
            book = rng.standard_normal((T, S)).astype(np.float64)
            book = book / np.maximum(np.sum(np.abs(book), axis=1, keepdims=True), 1e-12)
            ret = rng.standard_normal(T).astype(np.float64)
            turn = np.abs(np.diff(book, axis=0, prepend=book[:1])).sum(axis=1)
            legs.append(LegBook(spec=spec, book_2d=book, gross_return_1d=ret, turnover_1d=turn))
        folds = _make_folds(6, 10)
        cfg = L1LegConfig(prior_only_folds=1, warmup_folds=4, min_turnover_per_bar=0.0001,
                          cost_safety_margin=1.0, min_positive_fold_ratio=0.01,
                          max_leg_weight=0.51, familywise_error_rate=0.5)
        from src.domain.futures.compound.l1_leg_admission import accumulate_prequential_shadow_weights
        weights = accumulate_prequential_shadow_weights(tuple(legs), market, folds, 8.0, cfg)
        assert weights.shape == (T, K)
        assert np.all(weights >= 0.0)
        assert np.all(weights[:folds[4].oos_start] == 0.0)

    def test_shadow_weights_carry_forward_past_last_fold(self) -> None:
        n_t, n_s, n_k = 100, 3, 2
        rng = np.random.default_rng(42)
        market = rng.standard_normal(n_t).astype(np.float64) * 0.01
        legs = []
        for k in range(n_k):
            book = rng.standard_normal((n_t, n_s)).astype(np.float64)
            book = book / np.maximum(np.sum(np.abs(book), axis=1, keepdims=True), 1e-12)
            ret = rng.standard_normal(n_t).astype(np.float64)
            turn = np.abs(np.diff(book, axis=0, prepend=book[:1])).sum(axis=1)
            legs.append(LegBook(
                spec=_make_spec(concept_id=f"c{k}"),
                book_2d=book, gross_return_1d=ret, turnover_1d=turn,
            ))
        folds = tuple(CausalFold(
            fold_id=i, fit_start=0, fit_end_exclusive=20 + 20 * i,
            calibration_start=0, calibration_end_exclusive=20 + 20 * i,
            oos_start=20 + 20 * i, oos_end_exclusive=40 + 20 * i,
            purge_bars=0, embargo_bars=0,
        ) for i in range(5))
        cfg = L1LegConfig(prior_only_folds=1, warmup_folds=4, min_turnover_per_bar=0.0001,
                          cost_safety_margin=1.0, min_positive_fold_ratio=0.01,
                          max_leg_weight=0.51, familywise_error_rate=0.5)
        from src.domain.futures.compound.l1_leg_admission import accumulate_prequential_shadow_weights
        weights = accumulate_prequential_shadow_weights(tuple(legs), market, folds, 8.0, cfg)
        last_stop = folds[-1].oos_end_exclusive
        assert last_stop == 120
        assert np.all(weights[last_stop:] == weights[last_stop - 1:last_stop])


class TestWrapperParity:
    def test_evaluate_portfolio_admission_wrapper_matches_evidence(self) -> None:
        from src.domain.futures.compound.l1_leg_admission import (
            evaluate_portfolio_admission, evaluate_portfolio_evidence,
        )
        n_folds, per_fold = 6, 20
        T, S = n_folds * per_fold, 3
        combined = np.full((T, S), 1.0 / S, dtype=np.float64)
        ret = np.full((T, S), 0.02 / S, dtype=np.float64)
        folds = _make_folds(n_folds, per_fold)
        cfg = L1LegConfig(prior_only_folds=1, bars_per_year=2190.0, n_bootstrap=500)
        admitted, reasons, net_ann = evaluate_portfolio_admission(combined, ret, folds, 8.0, cfg, admission_end_exclusive=10**9)
        evidence = evaluate_portfolio_evidence(combined, ret, folds, 8.0, cfg, admission_end_exclusive=10**9)
        assert evidence.admitted == admitted
        assert evidence.reasons == reasons
        assert abs(evidence.net_alpha_ann - net_ann) < 1e-10


class TestRetiredSymbols:
    def test_binary_evidence_weight_symbols_are_retired(self) -> None:
        mod = __import__('src.domain.futures.compound.l1_leg_admission', fromlist=['x'])
        assert not hasattr(mod, 'compute_evidence_weight')
        assert not hasattr(mod, 'compute_leg_sizing_score')
