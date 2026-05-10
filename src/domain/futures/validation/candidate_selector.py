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

