from __future__ import annotations

from itertools import pairwise

import pytest
from scipy.stats import norm

from src.research.evaluation.falsification import (
    FalsificationConfig,
    FalsificationVerdict,
    PlateauResult,
    evaluate_falsification,
    evaluate_parameter_plateau,
    multiplicity_adjusted_t_floor,
)


class TestFalsificationConfig:
    def test_defaults(self) -> None:
        config = FalsificationConfig()
        assert (config.plateau_ratio, config.min_neighbors, config.base_t_floor) == (0.70, 2, 2.0)
        assert config.holdout_retention == 0.50

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"plateau_ratio": 0.0}, "plateau_ratio"),
            ({"plateau_ratio": 1.5}, "plateau_ratio"),
            ({"min_neighbors": 0}, "min_neighbors"),
            ({"base_t_floor": -0.1}, "base_t_floor"),
            ({"holdout_retention": 0.0}, "holdout_retention"),
            ({"holdout_retention": 1.5}, "holdout_retention"),
        ],
    )
    def test_validation_raises(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            FalsificationConfig(**kwargs)


class TestMultiplicityAdjustedTFloor:
    # GEV2-06-MULTIPLICITY
    def test_family_size_one_is_base_floor(self) -> None:
        assert abs(multiplicity_adjusted_t_floor(1) - 2.0) < 1e-9

    def test_nine_hypotheses_reject_t_of_230(self) -> None:
        floor = multiplicity_adjusted_t_floor(9)
        assert abs(floor - 2.8035) < 1e-3
        assert floor > 2.30

    def test_non_decreasing_in_family_size(self) -> None:
        values = [multiplicity_adjusted_t_floor(k) for k in range(1, 25)]
        assert all(a <= b for a, b in pairwise(values))
        assert multiplicity_adjusted_t_floor(20) > multiplicity_adjusted_t_floor(5)

    def test_matches_bonferroni_in_t_space(self) -> None:
        alpha = 2 * norm.sf(2.0)
        assert abs(multiplicity_adjusted_t_floor(9) - norm.isf(alpha / 18)) < 1e-12

    def test_rejects_invalid_family_size(self) -> None:
        with pytest.raises(ValueError, match="family_size"):
            multiplicity_adjusted_t_floor(0)

    def test_rejects_negative_base_floor(self) -> None:
        with pytest.raises(ValueError, match="base_t_floor"):
            multiplicity_adjusted_t_floor(5, base_t_floor=-1.0)


class TestEvaluateParameterPlateau:
    # GEV2-05-PLATEAU-REJECTS-SPIKE
    def test_measured_momentum_spike_is_rejected(self) -> None:
        scores = {12.0: 0.540, 14.0: 1.291, 16.0: 0.976, 18.0: 0.435}
        result = evaluate_parameter_plateau(scores, 14.0, FalsificationConfig())
        assert result.passed is False
        assert abs(result.neighbor_ratio - (0.758 / 1.291)) < 1e-6

    def test_flat_surface_passes(self) -> None:
        result = evaluate_parameter_plateau(
            {1.0: 0.95, 2.0: 1.00, 3.0: 0.98}, 2.0, FalsificationConfig(),
        )
        assert result.passed is True

    def test_fails_closed_when_neighbors_missing(self) -> None:
        result = evaluate_parameter_plateau({1.0: 1.0, 2.0: 1.0}, 1.0, FalsificationConfig())
        assert result.passed is False
        assert result.neighbor_scores == ()

    def test_fails_closed_on_non_positive_chosen_score(self) -> None:
        result = evaluate_parameter_plateau(
            {1.0: 0.5, 2.0: 0.0, 3.0: 0.5}, 2.0, FalsificationConfig(),
        )
        assert result.passed is False
        assert result.neighbor_scores == (0.5, 0.5)

    def test_raises_key_error_when_chosen_absent(self) -> None:
        with pytest.raises(KeyError):
            evaluate_parameter_plateau({1.0: 1.0, 2.0: 1.0}, 3.0, FalsificationConfig())

    def test_neighbour_ties_broken_by_smaller_key(self) -> None:
        result = evaluate_parameter_plateau(
            {12.0: 0.5, 14.0: 1.0, 16.0: 1.0}, 14.0, FalsificationConfig(),
        )
        assert result.neighbor_scores == (0.5, 1.0)


class TestFalsificationVerdict:
    def test_fields(self) -> None:
        plateau = PlateauResult(14.0, 1.291, (0.540, 0.976), 0.587, False)
        verdict = FalsificationVerdict(False, "plateau", plateau, 2.30, 2.8035, 0.874)
        assert verdict.passed is False
        assert verdict.binding_constraint == "plateau"

    def test_rejects_unknown_binding_constraint(self) -> None:
        plateau = PlateauResult(14.0, 1.291, (0.540, 0.976), 0.587, False)
        with pytest.raises(ValueError, match="binding_constraint"):
            FalsificationVerdict(False, "time_oos", plateau, 2.30, 2.8035, 0.874)


class TestEvaluateFalsification:
    # GEV2-05-PLATEAU-REJECTS-SPIKE
    def test_measured_surface_fails_on_plateau(self) -> None:
        verdict = evaluate_falsification(
            parameter_scores={12.0: 0.540, 14.0: 1.291, 16.0: 0.976},
            chosen_parameter=14.0,
            oos_t_stat=2.30,
            family_size=9,
            dev_score=0.674,
            holdout_score=0.589,
        )
        assert verdict.passed is False
        assert verdict.binding_constraint == "plateau"

    # GEV2-06-MULTIPLICITY
    def test_flat_surface_fails_on_multiplicity(self) -> None:
        verdict = evaluate_falsification(
            parameter_scores={1.0: 0.95, 2.0: 1.00, 3.0: 0.98},
            chosen_parameter=2.0,
            oos_t_stat=2.30,
            family_size=9,
            dev_score=1.0,
            holdout_score=0.9,
        )
        assert verdict.binding_constraint == "multiplicity"
        assert verdict.passed is False

    def test_symbol_holdout_gate_fails_closed_on_non_positive_dev(self) -> None:
        verdict = evaluate_falsification(
            parameter_scores={1.0: 0.95, 2.0: 1.00, 3.0: 0.98},
            chosen_parameter=2.0,
            oos_t_stat=3.0,
            family_size=1,
            dev_score=0.0,
            holdout_score=1.0,
        )
        assert verdict.binding_constraint == "symbol_holdout"

    def test_symbol_holdout_gate_fails_on_retention(self) -> None:
        verdict = evaluate_falsification(
            parameter_scores={1.0: 0.95, 2.0: 1.00, 3.0: 0.98},
            chosen_parameter=2.0,
            oos_t_stat=3.0,
            family_size=1,
            dev_score=1.0,
            holdout_score=0.4,
        )
        assert verdict.binding_constraint == "symbol_holdout"

    def test_all_gates_pass_yields_none(self) -> None:
        verdict = evaluate_falsification(
            parameter_scores={1.0: 0.95, 2.0: 1.00, 3.0: 0.98},
            chosen_parameter=2.0,
            oos_t_stat=3.5,
            family_size=1,
            dev_score=1.0,
            holdout_score=0.9,
        )
        assert verdict.passed is True
        assert verdict.binding_constraint == "none"
        assert abs(verdict.required_t_floor - 2.0) < 1e-9
