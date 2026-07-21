"""Deterministic, feasibility-first Layer 2 search helpers."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from src.domain.futures.optimization.l2_search_space import L2_SEARCH_SPACE

L2_FIXED_ROBUST_PARAMS: dict[str, Any] = {
    "l2_regime_bucket_side_split_enabled": True,
    "l2_regime_scoped_fold_override_enabled": True,
    "l2_crisis_replay_routing_parity_enabled": True,
    "l2_regime_policy_mode": "soft",
    "l2_regime_hard_block_enabled": False,
    "l2_regime_pooled_is_passthrough": True,
}


@dataclass(slots=True, frozen=True)
class L2RobustSearchBudget:
    anchors: int
    adaptive: int
    refinement: int

    @property
    def total(self) -> int:
        return self.anchors + self.adaptive + self.refinement


def resolve_l2_robust_search_budget(n_trials: int) -> L2RobustSearchBudget:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    anchors = min(24, max(1, n_trials // 5))
    refinement = min(24, n_trials // 5)
    adaptive = n_trials - anchors - refinement
    if adaptive < 0:
        adaptive = 0
        anchors = n_trials - refinement
    return L2RobustSearchBudget(anchors, adaptive, refinement)


def derive_l2_search_seed(experiment_key: str, search_space_hash: str) -> int:
    digest = hashlib.sha256(f"{experiment_key}:{search_space_hash}".encode()).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def materialize_l2_robust_params(params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(params)
    merged.update(L2_FIXED_ROBUST_PARAMS)
    return merged


def suggest_l2_robust_params(trial: Any) -> dict[str, Any]:
    """Suggest the configured search space and inject fixed routing controls."""
    params: dict[str, Any] = {}
    for key, spec in L2_SEARCH_SPACE.items():
        kind = spec["type"]
        if kind == "int":
            params[key] = trial.suggest_int(key, spec["low"], spec["high"], step=spec.get("step", 1))
        elif kind == "float":
            params[key] = trial.suggest_float(key, spec["low"], spec["high"], step=spec.get("step"))
        else:
            params[key] = trial.suggest_categorical(key, list(spec["choices"]))
    return materialize_l2_robust_params(params)


def build_l2_feasibility_anchors(*, count: int = 24) -> tuple[dict[str, Any], ...]:
    if count < 0:
        raise ValueError("count must be non-negative")
    keys = tuple(L2_SEARCH_SPACE)
    anchors: list[dict[str, Any]] = []
    for index in range(count):
        params: dict[str, Any] = {}
        for key in keys:
            spec = L2_SEARCH_SPACE[key]
            if spec["type"] == "categorical":
                choices = tuple(spec["choices"])
                params[key] = choices[index % len(choices)]
            else:
                low, high = float(spec["low"]), float(spec["high"])
                ratio = (index + 0.5) / max(count, 1)
                value = low + (high - low) * ratio
                step = float(spec.get("step", 1))
                value = round(round((value - low) / step) * step + low, 10)
                params[key] = int(value) if spec["type"] == "int" else value
        if not params.get("l2_regime_long_short_asymmetry_enabled", True):
            params.pop("l2_regime_bear_long_extra_mult", None)
            params.pop("l2_regime_crisis_long_extra_mult", None)
        anchors.append(materialize_l2_robust_params(params))
    return tuple(anchors)


def build_l2_refinement_trials(*, trials: tuple[Any, ...], count: int = 24) -> tuple[dict[str, Any], ...]:
    if count <= 0 or not trials:
        return ()
    ranked = sorted(trials, key=lambda t: (not bool(t.user_attrs.get("l2_joint_feasible", False)), float(t.value or 0.0), t.number))
    source = dict(ranked[0].params)
    key = "K_RANK"
    spec = L2_SEARCH_SPACE.get(key)
    if spec is None:
        return (materialize_l2_robust_params(source),)
    values: list[dict[str, Any]] = []
    current = int(source.get(key, spec["low"]))
    for delta in (-1, 1):
        candidate = max(int(spec["low"]), min(int(spec["high"]), current + delta))
        item = dict(source)
        item[key] = candidate
        values.append(materialize_l2_robust_params(item))
    return tuple(values[:count])
