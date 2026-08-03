"""Structural hyperparameter search for portfolio-construction parameters.

Dev tool (tools/, not runtime src/): joint TPE search over portfolio-construction
structural variables (universe_size, quantile, rebalance_bars, no_trade_band,
rebalance period) with the growth_engine_v3.md section 3.4 safeguards coded in --
an IS-only objective (the caller must restrict the objective to in-sample data;
this module never sees or touches an OOS split), a grid-comparable trial budget
cap, and a mandatory plateau-stability gate that reuses
``evaluate_parameter_plateau`` before any point can be adopted.

optuna is an optional ``tuning`` extra, never a core runtime dependency, and is
therefore imported lazily inside ``run_structural_search``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.research.evaluation.falsification import FalsificationConfig, evaluate_parameter_plateau


@dataclass(frozen=True, slots=True)
class StructuralSearchConfig:
    max_trials: int = 20
    seed: int = 0
    plateau_step_fraction: float = 0.15
    plateau_ratio: float = 0.70

    def __post_init__(self) -> None:
        if not 4 <= self.max_trials <= 50:
            raise ValueError(
                f"max_trials must be in [4, 50] (grid-comparable budget cap), got {self.max_trials}"
            )
        if not 0 < self.plateau_step_fraction < 1:
            raise ValueError(
                f"plateau_step_fraction must be in (0, 1), got {self.plateau_step_fraction}"
            )
        if not 0 < self.plateau_ratio <= 1:
            raise ValueError(
                f"plateau_ratio must be in (0, 1], got {self.plateau_ratio}"
            )


@dataclass(frozen=True, slots=True)
class StructuralSearchResult:
    best_params: dict[str, float]
    best_is_score: float
    plateau_neighbor_ratio: float
    plateau_passed: bool
    n_trials: int


def check_plateau_stability(
    objective: Callable[[dict[str, float]], float],
    best_params: Mapping[str, float],
    search_space: Mapping[str, tuple[float, float]],
    config: StructuralSearchConfig = StructuralSearchConfig(),  # noqa: B008 -- contract-mandated signature
) -> tuple[float, bool]:
    """Perturb every search-space axis around ``best_params`` and test the local
    surface for plateau stability using ``evaluate_parameter_plateau`` per axis.

    Each axis yields a three-point score map ``{perturbed_low, best,
    perturbed_high}`` (perturbations clipped to the axis bounds); the returned
    ``neighbor_ratio`` is the minimum per-axis neighbour ratio and ``passed`` is
    True only when every axis passes. Fails closed (ratio 0.0, ``passed`` False)
    when the baseline objective value is non-positive -- never divides by a
    non-positive baseline. Pure function: no optuna import, always runnable.
    """
    if not search_space:
        raise ValueError("search_space must contain at least one parameter")
    missing = sorted(set(search_space).difference(best_params))
    if missing:
        raise ValueError(f"best_params missing keys from search_space: {missing}")

    baseline = objective(dict(best_params))
    if baseline <= 0:
        return (0.0, False)

    falsification_config = FalsificationConfig(
        plateau_ratio=config.plateau_ratio, min_neighbors=2
    )
    axis_ratios: list[float] = []
    passed = True
    for param, (lo, hi) in search_space.items():
        step = (hi - lo) * config.plateau_step_fraction
        perturbed_low = min(max(best_params[param] - step, lo), hi)
        perturbed_high = min(max(best_params[param] + step, lo), hi)
        scores = {
            perturbed_low: objective({**best_params, param: perturbed_low}),
            best_params[param]: baseline,
            perturbed_high: objective({**best_params, param: perturbed_high}),
        }
        result = evaluate_parameter_plateau(scores, best_params[param], falsification_config)
        axis_ratios.append(result.neighbor_ratio)
        passed = passed and result.passed
    return (min(axis_ratios), passed)


def run_structural_search(
    objective: Callable[[dict[str, float]], float],
    search_space: Mapping[str, tuple[float, float]],
    config: StructuralSearchConfig = StructuralSearchConfig(),  # noqa: B008 -- contract-mandated signature
) -> StructuralSearchResult:
    """Joint TPE search over ``search_space`` (uniform float ranges only).

    The caller's ``objective`` must already be restricted to in-sample data --
    this function never sees or touches an OOS split (growth_engine_v3.md
    section 3.4 mandatory safeguard against directly optimizing OOS). Runs
    exactly ``config.max_trials`` trials, then gates the best point through
    ``check_plateau_stability``; callers must not skip that gate.
    """
    if not search_space:
        raise ValueError("search_space must contain at least one parameter")
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "optuna is required for structural search; install via `uv sync --extra tuning`"
        ) from None

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )

    def _suggested_params(trial: Any) -> dict[str, float]:
        return {
            name: trial.suggest_float(name, lo, hi)
            for name, (lo, hi) in search_space.items()
        }

    for _ in range(config.max_trials):
        study.optimize(
            lambda trial: objective(_suggested_params(trial)), n_trials=1
        )

    best_params = dict(study.best_trial.params)
    plateau_neighbor_ratio, plateau_passed = check_plateau_stability(
        objective, dict(study.best_trial.params), search_space, config
    )
    return StructuralSearchResult(
        best_params=best_params,
        best_is_score=float(study.best_value),
        plateau_neighbor_ratio=plateau_neighbor_ratio,
        plateau_passed=plateau_passed,
        n_trials=config.max_trials,
    )
