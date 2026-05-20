from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna

from src.domain.futures.optimization.phase_param_space import V43_CORE_PARAM_KEYS

_NON_FIXABLE_PARAMS = {
    "K_LONG",
    "K_SHORT",
    "EV_HURDLE_BPS",
    "PORTFOLIO_KAPPA",
    "MAX_EXPOSURE",
    "MAX_EXPOSURE_PER_COIN",
}

_V43_BOUNDS: dict[str, tuple[float, float]] = {
    "BETA_REGIME_BEAR": (0.0, 1.5),
    "BETA_REGIME_CHOP": (0.0, 1.0),
    "K_LONG": (1.0, 8.0),
    "K_SHORT": (0.0, 5.0),
    "REBALANCE_BARS": (1.0, 24.0),
    "EV_HURDLE_BPS": (5.0, 100.0),
    "PORTFOLIO_KAPPA": (0.05, 0.50),
    "TARGET_ANN_VOL": (0.05, 0.40),
    "MAX_EXPOSURE": (0.50, 3.00),
    "MAX_EXPOSURE_PER_COIN": (0.05, 0.40),
}


@dataclass(frozen=True)
class PhaseBPlan:
    fixed_params: dict[str, Any]
    shrunk_ranges: dict[str, tuple[float, float]]
    seed_combos: list[dict[str, Any]]
    importance_report: dict[str, Any]


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    from optuna.trial import TrialState

    return [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]


def _make_evaluator() -> Any:
    try:
        return optuna.importance.PedAnovaImportanceEvaluator()
    except Exception:
        return optuna.importance.FanovaImportanceEvaluator()


def _target_from_user_attr(metric_key: str) -> Callable[[optuna.trial.FrozenTrial], float]:
    def _target(t: optuna.trial.FrozenTrial) -> float:
        raw = t.user_attrs.get(metric_key)
        if raw is None:
            # Missing metrics should be pessimistic for ranking/importance influence.
            return -1e9
        try:
            return float(raw)
        except Exception:
            return -1e9

    return _target


def _importances_for(
    study: optuna.Study, target: Callable[[optuna.trial.FrozenTrial], float] | None = None
) -> dict[str, float]:
    try:
        vals = optuna.importance.get_param_importances(
            study,
            evaluator=_make_evaluator(),
            params=list(V43_CORE_PARAM_KEYS),
            target=target,
        )
        return {k: float(v) for k, v in vals.items()}
    except Exception:
        return {}


def _merge_importances(parts: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for p in parts:
        for k, v in p.items():
            merged[k] = max(float(v), merged.get(k, 0.0))
    return merged


def _best_completed_by_value(study: optuna.Study, top_k: int) -> list[optuna.trial.FrozenTrial]:
    trials = [t for t in _completed_trials(study) if t.value is not None]
    trials.sort(key=lambda t: float(t.value), reverse=True)
    return trials[:top_k]


def _a2_diverse_trials(study: optuna.Study, top_k: int) -> list[optuna.trial.FrozenTrial]:
    trials = _completed_trials(study)
    if not trials:
        return []
    score = [t for t in trials if t.user_attrs.get("sortino_lcb") is not None]
    risk = [t for t in trials if t.user_attrs.get("mdd_ucb") is not None]
    out: list[optuna.trial.FrozenTrial] = []
    if score:
        score.sort(key=lambda t: float(t.user_attrs.get("sortino_lcb", -1e9)), reverse=True)
        out.extend(score[:top_k])
        out.append(score[0])
        out.append(score[-1])
    if risk:
        risk.sort(key=lambda t: float(t.user_attrs.get("mdd_ucb", 1e9)))
        out.append(risk[0])
        out.append(risk[-1])
    seen: set[int] = set()
    uniq: list[optuna.trial.FrozenTrial] = []
    for t in out:
        if t.number in seen:
            continue
        seen.add(t.number)
        uniq.append(t)
    return uniq[: max(top_k, 5)]


def _v43_param_subset(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k in V43_CORE_PARAM_KEYS}


def _make_shrunk_range(param: str, best_val: float) -> tuple[float, float]:
    lo, hi = _V43_BOUNDS[param]
    span = hi - lo
    half = span * 0.2
    return (max(lo, best_val - half), min(hi, best_val + half))


def build_phase_b_plan(
    study_a1: optuna.Study,
    study_a2: optuna.Study,
    *,
    top_k_a1: int = 5,
    top_k_a2: int = 5,
) -> PhaseBPlan:
    imp_a1 = _importances_for(study_a1)
    imp_a2_sortino = _importances_for(study_a2, target=_target_from_user_attr("sortino_lcb"))
    imp_a2_mdd = _importances_for(
        study_a2,
        target=lambda t: -_target_from_user_attr("mdd_ucb")(t),
    )
    merged = _merge_importances([imp_a1, imp_a2_sortino, imp_a2_mdd])

    a1_best = _best_completed_by_value(study_a1, top_k=top_k_a1)
    a2_best = _a2_diverse_trials(study_a2, top_k=top_k_a2)

    base_params: dict[str, Any] = {}
    if a1_best:
        base_params = _v43_param_subset(a1_best[0].params)

    fixed_params: dict[str, Any] = {}
    shrunk_ranges: dict[str, tuple[float, float]] = {}
    for p in V43_CORE_PARAM_KEYS:
        if p not in base_params:
            continue
        imp = float(merged.get(p, 0.0))
        if imp < 0.02 and p not in _NON_FIXABLE_PARAMS:
            fixed_params[p] = base_params[p]
        elif 0.02 <= imp < 0.05:
            shrunk_ranges[p] = _make_shrunk_range(p, float(base_params[p]))

    seed_combos: list[dict[str, Any]] = []
    for a1 in a1_best:
        a1p = _v43_param_subset(a1.params)
        seed_combos.append(dict(a1p))
        for a2 in a2_best:
            a2p = _v43_param_subset(a2.params)
            merged_seed = dict(a1p)
            for rk in ("PORTFOLIO_KAPPA", "TARGET_ANN_VOL", "MAX_EXPOSURE", "MAX_EXPOSURE_PER_COIN"):
                if rk in a2p:
                    merged_seed[rk] = a2p[rk]
            seed_combos.append(merged_seed)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for seed in seed_combos:
        key = tuple(sorted(seed.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(seed)

    report = {
        "a1_importance": imp_a1,
        "a2_sortino_importance": imp_a2_sortino,
        "a2_mdd_importance": imp_a2_mdd,
        "merged_importance": merged,
        "non_fixable_params": sorted(_NON_FIXABLE_PARAMS),
        "a1_completed": len(_completed_trials(study_a1)),
        "a2_completed": len(_completed_trials(study_a2)),
    }

    return PhaseBPlan(
        fixed_params=fixed_params,
        shrunk_ranges=shrunk_ranges,
        seed_combos=deduped[:25],
        importance_report=report,
    )
