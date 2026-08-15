from __future__ import annotations

import sys

import pytest

from tools.research.structural_tuner import (
    StructuralSearchConfig,
    StructuralSearchResult,
    check_plateau_stability,
    run_structural_search,
)


def _concave(p: dict[str, float]) -> float:
    return 6.0 - 0.5 * (p["x"] - 1.0) ** 2 - 0.5 * (p["y"] - 2.0) ** 2


class TestCheckPlateauStability:
    # GEV3-02-PLATEAU-MULTIDIM: no optuna required -- pure function over a callable.
    def test_passes_at_peak_of_smooth_concave_objective(self) -> None:
        ratio, passed = check_plateau_stability(
            _concave, {"x": 1.0, "y": 2.0}, {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        )
        assert passed is True
        assert 0.0 < ratio <= 1.0

    def test_fails_for_point_far_from_peak(self) -> None:
        ratio, passed = check_plateau_stability(
            _concave, {"x": 4.9, "y": 2.0}, {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        )
        assert passed is False
        assert ratio == 0.0

    def test_fails_closed_on_non_positive_baseline(self) -> None:
        def flat(p: dict[str, float]) -> float:
            return -abs(p["x"])

        _, passed = check_plateau_stability(flat, {"x": -3.0}, {"x": (-5.0, 5.0)})
        assert passed is False

    def test_raises_on_empty_search_space(self) -> None:
        with pytest.raises(ValueError, match="at least one parameter"):
            check_plateau_stability(_concave, {}, {})

    def test_raises_when_best_params_missing_search_space_key(self) -> None:
        with pytest.raises(ValueError, match="missing keys"):
            check_plateau_stability(_concave, {"x": 1.0}, {"x": (-5.0, 5.0), "y": (-5.0, 5.0)})


class TestStructuralSearchConfig:
    # GEV3-04-BUDGET-CAP: dense Bayesian search over hundreds/thousands of trials
    # is its own overfitting risk, so the budget is capped grid-comparable.
    def test_defaults(self) -> None:
        config = StructuralSearchConfig()
        assert (config.max_trials, config.seed, config.plateau_ratio) == (20, 0, 0.70)

    def test_rejects_max_trials_above_grid_cap(self) -> None:
        with pytest.raises(ValueError, match="max_trials"):
            StructuralSearchConfig(max_trials=200)

    def test_rejects_max_trials_below_minimum(self) -> None:
        with pytest.raises(ValueError, match="max_trials"):
            StructuralSearchConfig(max_trials=2)

    def test_rejects_invalid_plateau_step_fraction(self) -> None:
        with pytest.raises(ValueError, match="plateau_step_fraction"):
            StructuralSearchConfig(plateau_step_fraction=0.0)
        with pytest.raises(ValueError, match="plateau_step_fraction"):
            StructuralSearchConfig(plateau_step_fraction=1.0)

    def test_rejects_invalid_plateau_ratio(self) -> None:
        with pytest.raises(ValueError, match="plateau_ratio"):
            StructuralSearchConfig(plateau_ratio=0.0)
        with pytest.raises(ValueError, match="plateau_ratio"):
            StructuralSearchConfig(plateau_ratio=1.1)


class TestStructuralSearchResult:
    def test_result_fields(self) -> None:
        result = StructuralSearchResult({"x": 1.0}, 0.5, 0.9, True, 20)
        assert result.plateau_passed is True
        assert result.n_trials == 20


class TestRunStructuralSearch:
    # GEV3-03-OPTUNA-OPTIONAL-FAILS-CLOSED: optuna is a lazy `tuning` extra, never
    # a core runtime dependency; absence must raise an actionable ImportError.
    def test_raises_actionable_import_error_when_optuna_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "optuna", None)
        with pytest.raises(ImportError) as exc_info:
            run_structural_search(
                lambda p: -(p["x"] - 1.0) ** 2,
                {"x": (-5.0, 5.0)},
                StructuralSearchConfig(max_trials=4),
            )
        message = str(exc_info.value)
        assert "optuna" in message
        assert "uv sync --extra tuning" in message

    def test_raises_on_empty_search_space(self) -> None:
        with pytest.raises(ValueError, match="at least one parameter"):
            run_structural_search(_concave, {})
