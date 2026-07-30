from __future__ import annotations

import numpy as np

import pytest

from src.domain.futures.compound.config import L1LegConfig
from src.domain.futures.compound.contracts import CausalFold, LegBook, LegEvidence, SignalConceptSpec
from src.domain.futures.compound.l1_leg_admission import (
    accumulate_prequential_leg_weights,
    compute_evidence_weight,
    evaluate_portfolio_admission,
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


class TestComputeEvidenceWeight:
    def test_compute_evidence_weight_state_table(self) -> None:
        cfg = L1LegConfig(min_turnover_per_bar=0.00105, cost_safety_margin=1.5,
                          min_positive_fold_ratio=0.50, max_leg_weight=0.50)
        base = {
            "concept_id": "t", "mode": "xs", "n_oos_bars": 100,
            "alpha_ann": 0.05, "beta_market": 0.0, "alpha_sharpe": 0.5,
            "t_alpha_newey_west": 2.0, "positive_folds": 6, "n_folds": 10,
            "posterior_positive": 0.95, "evidence_weight": 0.0, "reasons": (),
        }
        # precondition 1: turnover below floor
        ev_low_turn = LegEvidence(**{**base, "breakeven_cost_bps": 15.0, "mean_turnover_per_bar": 0.001})
        assert compute_evidence_weight(ev_low_turn, 8.0, cfg) == 0.0
        # precondition 2: breakeven below cost*safety_margin
        ev_low_be = LegEvidence(**{**base, "breakeven_cost_bps": 5.0, "mean_turnover_per_bar": 0.01})
        assert compute_evidence_weight(ev_low_be, 8.0, cfg) == 0.0
        # precondition 3: positive_folds ratio below floor
        ev_unstable = LegEvidence(**{
            **base, "breakeven_cost_bps": 30.0, "mean_turnover_per_bar": 0.01,
            "positive_folds": 2, "n_folds": 10,
        })
        assert compute_evidence_weight(ev_unstable, 8.0, cfg) == 0.0

    def test_compute_evidence_weight_pass_branch_capped(self) -> None:
        cfg = L1LegConfig(min_turnover_per_bar=0.00105, max_leg_weight=0.30)
        ev = LegEvidence(
            concept_id="t", mode="xs", n_oos_bars=100,
            alpha_ann=0.10, beta_market=0.0, alpha_sharpe=1.0,
            t_alpha_newey_west=4.0, breakeven_cost_bps=50.0,
            mean_turnover_per_bar=0.01, positive_folds=8, n_folds=10,
            posterior_positive=0.99, evidence_weight=0.0, reasons=(),
        )
        weight = compute_evidence_weight(ev, 8.0, cfg)
        assert 0.0 < weight <= cfg.max_leg_weight
        assert weight == cfg.max_leg_weight


class TestAccumulatePrequentialLegWeights:
    def test_accumulate_prequential_leg_weights_is_causal(self) -> None:
        T, S, K = 40, 5, 2
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
        folds = _make_folds(4, 10)
        cfg = L1LegConfig(warmup_folds=1, min_turnover_per_bar=0.001,
                          cost_safety_margin=1.0, min_positive_fold_ratio=0.1)
        weights = accumulate_prequential_leg_weights(tuple(legs), market, folds, 8.0, cfg)
        assert weights.shape == (T, K)
        assert np.all(weights[:folds[1].oos_start] == 0.0)

        # causality: mutating fold 3's OOS returns must not change any weight
        # row computed before fold 3 starts (those weights only used folds 0-2).
        legs_mutated = list(legs)
        mutated_book = legs[0].book_2d.copy()
        sl = slice(folds[3].oos_start, folds[3].oos_end_exclusive)
        mutated_book[sl] *= -5.0
        legs_mutated[0] = LegBook(
            spec=legs[0].spec, book_2d=mutated_book,
            gross_return_1d=legs[0].gross_return_1d, turnover_1d=legs[0].turnover_1d,
        )
        weights_mutated = accumulate_prequential_leg_weights(
            tuple(legs_mutated), market, folds, 8.0, cfg,
        )
        assert np.allclose(
            weights[:folds[3].oos_start], weights_mutated[:folds[3].oos_start],
        )


class TestEvaluatePortfolioAdmission:
    def test_evaluate_portfolio_admission_each_condition_blocks(self) -> None:
        """[RULE-09] all three sub-conditions independently flip admitted to
        False with the matching reason string -- delegates to the three
        scenario-specific tests below so each can also be run/read standalone."""
        self.test_evaluate_portfolio_admission_posterior_condition_blocks()
        self.test_evaluate_portfolio_admission_fold_ratio_condition_blocks()
        self.test_evaluate_portfolio_admission_stress_condition_blocks()

    def test_evaluate_portfolio_admission_posterior_condition_blocks(self) -> None:
        """Turnover is zero (constant book) so cost never enters; the pooled
        bar-level return is a near-zero-mean coin flip (dispersion keeps
        bootstrap P(net>0) well under the 0.90 threshold) while every fold's
        own sample mean stays barely positive."""
        n_folds, per_fold = 4, 20
        T, S = n_folds * per_fold, 3
        combined = np.full((T, S), 1.0 / S, dtype=np.float64)  # constant book, zero turnover
        ret = np.zeros((T, S), dtype=np.float64)
        pattern = np.array([0.5] * 10 + [-0.4998] * 10)
        for f in range(n_folds):
            start = f * per_fold
            for i, val in enumerate(pattern):
                ret[start + i] = val / S
        folds = _make_folds(n_folds, per_fold)
        cfg = L1LegConfig(warmup_folds=0, min_growth_posterior_probability=0.90,
                          min_positive_fold_ratio=0.50, stress_cost_multiplier=2.0,
                          bars_per_year=2190.0, n_bootstrap=2000)
        admitted, reasons, net_ann = evaluate_portfolio_admission(combined, ret, folds, 8.0, cfg)
        assert admitted is False
        assert any("posterior" in r for r in reasons)

    def test_evaluate_portfolio_admission_fold_ratio_condition_blocks(self) -> None:
        """Fold 0 has a large positive mean that dominates the pooled bootstrap
        (posterior stays high) while the other 3 folds are individually
        negative, so positive_folds/n_folds < 0.5."""
        n_folds, per_fold = 4, 20
        T, S = n_folds * per_fold, 3
        combined = np.full((T, S), 1.0 / S, dtype=np.float64)
        ret = np.zeros((T, S), dtype=np.float64)
        ret[0:per_fold] = 1.0 / S
        for f in range(1, n_folds):
            ret[f * per_fold:(f + 1) * per_fold] = -0.01 / S
        folds = _make_folds(n_folds, per_fold)
        cfg = L1LegConfig(warmup_folds=0, min_growth_posterior_probability=0.90,
                          min_positive_fold_ratio=0.50, stress_cost_multiplier=2.0,
                          bars_per_year=2190.0, n_bootstrap=2000)
        admitted, reasons, net_ann = evaluate_portfolio_admission(combined, ret, folds, 8.0, cfg)
        assert admitted is False
        assert any("positive_folds" in r for r in reasons)

    def test_evaluate_portfolio_admission_stress_condition_blocks(self) -> None:
        """Deterministic, high-turnover book with a thin positive margin over
        normal cost that flips negative once stress_cost_multiplier doubles
        the cost -- posterior and fold-ratio both pass cleanly since every
        bar/fold is identically positive at normal cost."""
        n_folds, per_fold = 4, 20
        T, S = 1 + n_folds * per_fold, 2
        combined = np.zeros((T, S), dtype=np.float64)
        for t in range(T):
            combined[t] = [0.5, -0.5] if t % 2 == 0 else [-0.5, 0.5]
        # ret co-moves with the weight vector itself (dot(w, k*w) = k*sum(w^2)
        # is sign-invariant), giving a constant +0.0025 port_ret every bar
        # regardless of which alternating weight pattern is active.
        ret = combined * 0.005
        folds = _make_folds(n_folds, per_fold, start=1)
        cfg = L1LegConfig(warmup_folds=0, min_growth_posterior_probability=0.90,
                          min_positive_fold_ratio=0.50, stress_cost_multiplier=2.0,
                          bars_per_year=2190.0, n_bootstrap=2000)
        admitted, reasons, net_ann = evaluate_portfolio_admission(combined, ret, folds, 8.0, cfg)
        assert admitted is False
        assert any("stressed_net_ann" in r for r in reasons)
        assert net_ann > 0.0  # normal (unstressed) net_ann is positive

    def test_evaluate_portfolio_admission_all_pass(self) -> None:
        n_folds, per_fold = 4, 20
        T, S = n_folds * per_fold, 3
        combined = np.full((T, S), 1.0 / S, dtype=np.float64)
        ret = np.full((T, S), 0.02 / S, dtype=np.float64)
        folds = _make_folds(n_folds, per_fold)
        cfg = L1LegConfig(warmup_folds=0, min_growth_posterior_probability=0.90,
                          min_positive_fold_ratio=0.50, stress_cost_multiplier=2.0,
                          bars_per_year=2190.0, n_bootstrap=2000)
        admitted, reasons, net_ann = evaluate_portfolio_admission(combined, ret, folds, 8.0, cfg)
        assert admitted is True
        assert reasons == ()
        assert net_ann > 0.0

    def test_evaluate_portfolio_admission_no_active_bars(self) -> None:
        combined = np.zeros((10, 3), dtype=np.float64)
        ret = np.zeros((10, 3), dtype=np.float64)
        admitted, reasons, net_ann = evaluate_portfolio_admission(combined, ret, (), 8.0, L1LegConfig())
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
    ])
    def test_leg_evidence_rejects_invalid_fields(self, field: str, bad_value: object, match: str) -> None:
        kwargs = {**self._valid_kwargs(), field: bad_value}
        with pytest.raises(ValueError, match=match):
            LegEvidence(**kwargs)

    def test_leg_evidence_rejects_positive_folds_exceeding_n_folds(self) -> None:
        kwargs = {**self._valid_kwargs(), "n_folds": 2, "positive_folds": 5}
        with pytest.raises(ValueError, match="positive_folds"):
            LegEvidence(**kwargs)
