from __future__ import annotations

import pytest

optuna = pytest.importorskip("optuna")

from tools.research.structural_tuner import (  # noqa: E402
    StructuralSearchConfig,
    run_structural_search,
)


def _interaction(p: dict[str, float]) -> float:
    """Smooth concave objective whose joint optimum (3, 2) is missed by
    axis-wise combination: the dominant ((x-3) + 3*(y-2))^2 term couples the
    axes, so combining independently-optimal single-axis values (each swept with
    the other axis held at the base) lands far below the joint optimum
    (growth_engine_v3.md section 3.3, E8-A/E8-B finding)."""
    x, y = p["x"], p["y"]
    u = (x - 3.0) + 3.0 * (y - 2.0)
    v = y - 2.0
    return 10.0 - 2.0 * u * u - 0.5 * v * v


def _single_axis_combination(
    objective,
    search_space: dict[str, tuple[float, float]],
    base: dict[str, float],
    grid: int = 61,
) -> tuple[dict[str, float], float]:
    """The historical project practice (growth_engine_v3.md section 4): sweep
    each axis independently holding the others fixed at base, then combine the
    per-axis best values. Evaluates objective once per grid point per axis."""
    best_combo: dict[str, float] = dict(base)
    base_score = objective(base)
    for param, (lo, hi) in search_space.items():
        best_value = base[param]
        best_score = base_score
        for step in range(grid):
            value = lo + (hi - lo) * step / (grid - 1)
            candidate = {**base, param: value}
            score = objective(candidate)
            if score > best_score:
                best_score = score
                best_value = value
        best_combo[param] = best_value
    return best_combo, objective(best_combo)


def test_joint_tpe_result_scores_at_least_as_well_as_axis_combination() -> None:
    # GEV3-05-OPTUNA-BEATS-AXIS-COMBINE: with a strong parameter interaction the
    # joint TPE search must not underperform the best single-axis-combination
    # point -- axis-wise combination can miss the joint optimum entirely.
    search_space: dict[str, tuple[float, float]] = {"x": (0.0, 6.0), "y": (0.0, 4.0)}
    base = {"x": 5.0, "y": 4.0}
    combination_params, combination_score = _single_axis_combination(_interaction, search_space, base)
    assert combination_score < _interaction({"x": 3.0, "y": 2.0})

    result = run_structural_search(
        _interaction, search_space, StructuralSearchConfig(max_trials=20, seed=7),
    )
    assert result.n_trials == 20
    assert result.best_is_score >= combination_score
