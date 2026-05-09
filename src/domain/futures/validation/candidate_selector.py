from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
from optuna.trial import TrialState


@dataclass(frozen=True)
class CandidateSelectorConfig:
    elite_top_n: int = 30
    basin_iqr_mult: float = 1.0


@dataclass(frozen=True)
class CandidateSelectionResult:
    params: dict[str, Any]
    representative_trial: optuna.trial.FrozenTrial
    n_completed: int
    n_passing: int
    n_basin: int
    method: str


def _aggregate_params_from_trials(
    trials: list[optuna.trial.FrozenTrial],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not trials:
        return out
    keys = trials[0].params.keys()
    for key in keys:
        vals = [t.params[key] for t in trials if key in t.params]
        if not vals:
            continue
        if isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
            med = float(np.median(np.asarray(vals, dtype=np.float64)))
            if isinstance(vals[0], int):
                out[key] = int(round(med))
            else:
                out[key] = med
        else:
            out[key] = max(set(vals), key=vals.count)
    return out


def _robust_basin(
    trials: list[optuna.trial.FrozenTrial],
    cfg: CandidateSelectorConfig,
) -> list[optuna.trial.FrozenTrial]:
    if not trials:
        return []
    sorted_trials = sorted(trials, key=lambda t: float(t.value if t.value is not None else 1e18))
    elite = sorted_trials[: max(3, min(int(cfg.elite_top_n), len(sorted_trials)))]
    if len(elite) < 3:
        return elite

    basin = elite[:]
    num_keys = [k for k, v in elite[0].params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for key in num_keys:
        vals = np.asarray([float(t.params[key]) for t in elite], dtype=np.float64)
        q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        iqr = q3 - q1
        lo = q1 - float(cfg.basin_iqr_mult) * iqr
        hi = q3 + float(cfg.basin_iqr_mult) * iqr
        basin = [t for t in basin if lo <= float(t.params[key]) <= hi]
        if len(basin) < 3:
            basin = elite[:]
            break
    return basin


def select_candidate_by_robust_basin(
    all_trials: list[optuna.trial.FrozenTrial],
    is_passing: Callable[[optuna.trial.FrozenTrial], bool],
    build_params: Callable[[dict[str, Any]], dict[str, Any]],
    cfg: CandidateSelectorConfig | None = None,
) -> CandidateSelectionResult:
    use_cfg = cfg or CandidateSelectorConfig()
    completed = [t for t in all_trials if t.state == TrialState.COMPLETE and t.value is not None]
    passing = [t for t in completed if is_passing(t)]
    base = passing if passing else completed
    if not base:
        raise ValueError("No completed trials available for candidate selection.")

    basin = _robust_basin(base, use_cfg)
    representative = min(basin, key=lambda t: float(t.value if t.value is not None else 1e18))
    agg_raw = _aggregate_params_from_trials(basin)
    params = build_params(agg_raw if agg_raw else dict(representative.params))
    method = "robust_basin_median" if passing else "fallback_basin_median"
    return CandidateSelectionResult(
        params=params,
        representative_trial=representative,
        n_completed=len(completed),
        n_passing=len(passing),
        n_basin=len(basin),
        method=method,
    )


def select_orthogonal_ensemble(
    all_trials: list[optuna.trial.FrozenTrial],
    is_passing: Callable[[optuna.trial.FrozenTrial], bool],
    build_params: Callable[[dict[str, Any]], dict[str, Any]],
    max_size: int = 3,
    corr_threshold: float = 0.6,
) -> list[CandidateSelectionResult]:
    """Select an ensemble of high-performing candidates that are minimally correlated.

    Selection Logic:
    1. Filter trials that pass the hard gates (is_passing(t)).
    2. Sort them by their objective value (best first).
    3. Iteratively select up to max_size trials. A trial is selected if its Pearson correlation
       with already selected trials is below corr_threshold.
    4. Correlation is based on 'cpcv_path_oos_log_tw' user attribute.
    """
    completed = [t for t in all_trials if t.state == TrialState.COMPLETE and t.value is not None]
    passing = [t for t in completed if is_passing(t)]
    base = passing if passing else completed
    if not base:
        raise ValueError("No completed trials available for ensemble selection.")

    # Sort by value (best first). Optuna minimizes by default for our objective.
    sorted_trials = sorted(base, key=lambda t: float(t.value if t.value is not None else 1e18))

    selected_trials: list[optuna.trial.FrozenTrial] = []

    for trial in sorted_trials:
        if len(selected_trials) >= max_size:
            break

        if not selected_trials:
            selected_trials.append(trial)
            continue

        current_returns = trial.user_attrs.get("cpcv_path_oos_log_tw", [])
        if not current_returns or len(current_returns) < 2:
            # If we don't have enough data for correlation, we skip to be safe
            continue

        is_orthogonal = True
        curr_arr = np.asarray(current_returns, dtype=np.float64)
        for sel in selected_trials:
            sel_returns = sel.user_attrs.get("cpcv_path_oos_log_tw", [])
            if not sel_returns or len(sel_returns) != len(current_returns):
                continue

            sel_arr = np.asarray(sel_returns, dtype=np.float64)
            # Use Pearson correlation
            if np.std(curr_arr) < 1e-9 or np.std(sel_arr) < 1e-9:
                # Constant returns means high correlation/redundancy or dead trial
                is_orthogonal = False
                break

            corr = np.corrcoef(curr_arr, sel_arr)[0, 1]
            if corr > corr_threshold:
                is_orthogonal = False
                break

        if is_orthogonal:
            selected_trials.append(trial)

    return [
        CandidateSelectionResult(
            params=build_params(t.params),
            representative_trial=t,
            n_completed=len(completed),
            n_passing=len(passing),
            n_basin=1,
            method="orthogonal_ensemble"
        )
        for t in selected_trials
    ]

