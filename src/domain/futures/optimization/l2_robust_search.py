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
    "l2_deploy_mdd_margin": 0.30,
    "l2_deploy_crisis_mdd_margin": 0.30,
    "l2_min_crisis_cagr": -0.05,
    "l2_regime_bull_leverage_boost_enabled": False,
    "l2_regime_long_short_asymmetry_enabled": False,
    "l2_regime_severity_gating_enabled": False,
    "l2_regime_bear_gross_cap": 0.35,
    "l2_regime_crisis_gross_cap": 0.25,
    "l2_regime_cap_release_cooldown_bars": 0,
    "l2_regime_bear_long_extra_mult": 1.0,
    "l2_regime_crisis_long_extra_mult": 1.0,
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
    anchors = 24
    refinement = 24
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
    for key, val in L2_FIXED_ROBUST_PARAMS.items():
        if key not in merged:
            merged[key] = val
    return merged


def suggest_l2_robust_params(trial: Any) -> dict[str, Any]:
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
                params[key] = int(spec["choices"][index % len(spec["choices"])])
            else:
                low = float(spec["low"])
                high = float(spec["high"])
                ratio = (index + 0.5) / float(max(count, 1))
                if spec.get("step"):
                    step = float(spec["step"])
                    n_steps = max(round((high - low) / step), 1)
                    step_idx = min(round(ratio * n_steps), n_steps - 1)
                    params[key] = float(low + step_idx * step)
                else:
                    params[key] = float(low + ratio * (high - low))
        anchors.append(materialize_l2_robust_params(params))
    return tuple(anchors)


def compute_search_space_hash() -> str:
    hasher = hashlib.sha256()
    for key in sorted(L2_SEARCH_SPACE.keys()):
        spec = L2_SEARCH_SPACE[key]
        hasher.update(key.encode("utf-8"))
        hasher.update(str(spec).encode("utf-8"))
    for key in sorted(L2_FIXED_ROBUST_PARAMS.keys()):
        hasher.update(key.encode("utf-8"))
        hasher.update(str(L2_FIXED_ROBUST_PARAMS[key]).encode("utf-8"))
    return hasher.hexdigest()


def build_refinement_neighbors(
    base_params: dict[str, Any],
    *,
    previous_hashes: set[str],
    max_neighbors: int = 24,
) -> tuple[dict[str, Any], ...]:
    """Generate discrete neighbors around a base parameter set.

    Only returns configs whose hashes are not in previous_hashes.
    """
    neighbors: list[dict[str, Any]] = []
    for key, spec in L2_SEARCH_SPACE.items():
        if len(neighbors) >= max_neighbors:
            break
        base_val = base_params.get(key)
        if base_val is None:
            continue
        variants: list[Any] = []
        if spec["type"] == "int":
            low, high = int(spec["low"]), int(spec["high"])
            step = int(spec.get("step", 1))
            for delta in (-step, step):
                iv = int(base_val) + delta
                if low <= iv <= high:
                    variants.append(iv)
        elif spec["type"] == "float":
            flow, fhigh = float(spec["low"]), float(spec["high"])
            fstep = float(spec.get("step", 1.0))
            for fdelta in (-fstep, fstep):
                fv = float(base_val) + fdelta
                if flow <= fv <= fhigh + 1e-12:
                    variants.append(fv)
        else:
            cat_choices = list(spec["choices"])
            idx = cat_choices.index(base_val) if base_val in cat_choices else -1
            for offset in (-1, 1):
                candidate_idx = idx + offset
                if 0 <= candidate_idx < len(cat_choices):
                    variants.append(cat_choices[candidate_idx])
        for v in variants:
            neighbor = dict(base_params)
            neighbor[key] = v
            h = hashlib.sha256(str(sorted(neighbor.items())).encode()).hexdigest()
            if h not in previous_hashes:
                previous_hashes.add(h)
                neighbors.append(materialize_l2_robust_params(neighbor))
                if len(neighbors) >= max_neighbors:
                    break
    return tuple(neighbors)


def build_l2_refinement_trials(
    *,
    trials: tuple[Any, ...],
    count: int = 24,
) -> tuple[dict[str, Any], ...]:
    """Compatibility wrapper for the staged refinement API.

    The public helper existed before the neighbor generator was split out.  Keep
    it as a thin deterministic adapter so callers and regression tests do not
    silently lose refinement after the search helper refactor.
    """
    if count <= 0 or not trials:
        return ()
    ranked = sorted(
        trials,
        key=lambda trial: (
            not bool(getattr(trial, "user_attrs", {}).get("l2_joint_feasible", False)),
            -(float(trial.value) if getattr(trial, "value", None) is not None else float("-inf")),
            int(getattr(trial, "number", 0)),
        ),
    )
    source = dict(getattr(ranked[0], "params", {}))
    previous_hashes: set[str] = set()
    return build_refinement_neighbors(
        source,
        previous_hashes=previous_hashes,
        max_neighbors=count,
    )
