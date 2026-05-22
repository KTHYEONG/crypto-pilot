from __future__ import annotations

import inspect
import warnings
from typing import Any, cast

import optuna

from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

# Optuna Experimental Warning suppression at code level
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def build_phase_study_name(base_study_name: str, phase: str) -> str:
    phase_suffix = phase.strip().lower()
    return f"{base_study_name}_{phase_suffix}"


def _ua_float(trial: optuna.trial.FrozenTrial, key: str) -> float | None:
    raw = trial.user_attrs.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _mark_proxy_used(trial: optuna.trial.FrozenTrial, key: str) -> None:
    try:
        trial.user_attrs[f"{key}_proxy_used"] = 1
    except Exception:
        return


def _resolve_metric(
    trial: optuna.trial.FrozenTrial,
    key: str,
    aliases: tuple[str, ...] = (),
    *,
    conservative_proxy: float | None = None,
) -> float | None:
    v = _ua_float(trial, key)
    if v is not None:
        return v
    for alias in aliases:
        v = _ua_float(trial, alias)
        if v is not None:
            _mark_proxy_used(trial, key)
            return v
    if conservative_proxy is not None:
        _mark_proxy_used(trial, key)
        return conservative_proxy
    return None


def _min_constraint(
    trial: optuna.trial.FrozenTrial,
    key: str,
    threshold: float,
    aliases: tuple[str, ...] = (),
    *,
    missing_violation: float = 1.0,
    conservative_proxy: float | None = None,
) -> float:
    v = _resolve_metric(
        trial,
        key,
        aliases,
        conservative_proxy=conservative_proxy,
    )
    if v is None:
        return missing_violation
    return threshold - v


def _max_constraint(
    trial: optuna.trial.FrozenTrial,
    key: str,
    threshold: float,
    aliases: tuple[str, ...] = (),
    *,
    missing_violation: float = 1.0,
    conservative_proxy: float | None = None,
) -> float:
    v = _resolve_metric(
        trial,
        key,
        aliases,
        conservative_proxy=conservative_proxy,
    )
    if v is None:
        return missing_violation
    return v - threshold


def _resolve_cvar_mdd_constraint(trial: optuna.trial.FrozenTrial) -> float:
    mdd_val = _resolve_metric(
        trial,
        "mdd_ucb",
        ("mdd", "awf_worst_mdd_pct", "oos_mdd"),
    )
    cvar_val = _resolve_metric(
        trial,
        "cvar",
        ("awf_cvar",),
    )
    if mdd_val is None or cvar_val is None:
        return 1.0
    mdd_cmp = float(mdd_val)
    cvar_cmp = float(cvar_val)
    # Align units before applying cvar <= 1.3 * mdd:
    # if one metric looks fractional and the other percent-like, scale fraction by 100.
    if mdd_cmp <= 1.0 < cvar_cmp:
        mdd_cmp *= 100.0
    elif cvar_cmp <= 1.0 < mdd_cmp:
        cvar_cmp *= 100.0
    return cvar_cmp - (1.3 * mdd_cmp)


def phase_a1_constraints(trial: optuna.trial.FrozenTrial) -> list[float]:
    min_trades = float(OPT_FUTURES_CONFIG.get("FUTURES_A1_MIN_TRADES_FLOOR", 30.0))
    active_floor = float(OPT_FUTURES_CONFIG.get("FUTURES_A1_ACTIVE_MONTH_RATIO_FLOOR", 0.55))
    turnover_ref = float(OPT_FUTURES_CONFIG.get("FUTURES_A1_TURNOVER_COST_RATIO_MAX", 0.20))
    return [
        _min_constraint(trial, "oos_ic", 0.015),
        _min_constraint(trial, "short_side_ic", 0.010),
        _min_constraint(trial, "n_trades", min_trades),
        _min_constraint(trial, "active_month_ratio", active_floor),
        _max_constraint(trial, "turnover_cost_ratio", turnover_ref),
    ]


def phase_a2_constraints(trial: optuna.trial.FrozenTrial) -> list[float]:
    return [
        _min_constraint(
            trial,
            "sortino_lcb",
            1.8,
            ("sortino", "awf_sortino", "oos_sortino"),
        ),
        _min_constraint(
            trial,
            "calmar_lcb",
            1.5,
            ("calmar", "awf_calmar", "oos_calmar"),
        ),
        _max_constraint(
            trial,
            "mdd_ucb",
            20.0,
            ("mdd", "awf_worst_mdd_pct", "oos_mdd"),
        ),
    ]


def phase_b_constraints(trial: optuna.trial.FrozenTrial) -> list[float]:
    return [
        _min_constraint(
            trial,
            "ev_cost_ratio",
            3.0,
            ("ev_cost",),
        ),
        _max_constraint(
            trial,
            "funding_drag_ratio",
            0.25,
            ("funding_drag",),
        ),
        _min_constraint(
            trial,
            "cagr_lcb",
            30.0,
            ("cagr", "oos_cagr"),
        ),
        _min_constraint(
            trial,
            "sortino_lcb",
            1.8,
            ("sortino", "awf_sortino", "oos_sortino"),
        ),
        _min_constraint(
            trial,
            "calmar_lcb",
            1.5,
            ("calmar", "awf_calmar", "oos_calmar"),
        ),
        _max_constraint(
            trial,
            "mdd_ucb",
            20.0,
            ("mdd", "awf_worst_mdd_pct", "oos_mdd"),
        ),
        _max_constraint(trial, "mdd_duration", 180.0),
        _resolve_cvar_mdd_constraint(trial),
        _min_constraint(
            trial,
            "minority_side_ratio",
            0.15,
            ("ls_minority_share", "minority", "minority_ratio"),
        ),
    ]


def build_phase_a1_sampler(seed: int) -> optuna.samplers.BaseSampler:
    if OPT_FUTURES_CONFIG.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False):
        return optuna.samplers.NSGAIISampler(
            seed=seed,
            population_size=int(OPT_FUTURES_CONFIG.get("FUTURES_NSGA2_POPULATION_SIZE", 30)),
            constraints_func=phase_a1_constraints,
        )
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=20,
        multivariate=True,
        group=True,
        constant_liar=True,
        n_ei_candidates=48,
    )


def build_phase_a2_sampler(seed: int) -> optuna.samplers.BaseSampler:
    if OPT_FUTURES_CONFIG.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False):
        return optuna.samplers.NSGAIISampler(
            seed=seed,
            population_size=int(OPT_FUTURES_CONFIG.get("FUTURES_NSGA2_POPULATION_SIZE", 30)),
            constraints_func=phase_a2_constraints,
        )
    try:
        from optuna.integration import BoTorchSampler

        kwargs: dict[str, Any] = {
            "seed": seed,
            "n_startup_trials": 40,
            "consider_running_trials": True,
        }
        try:
            sig = inspect.signature(BoTorchSampler.__init__)
            if "constraints_func" in sig.parameters:
                kwargs["constraints_func"] = phase_a2_constraints
        except Exception:  # noqa: S110
            pass
        return cast(optuna.samplers.BaseSampler, BoTorchSampler(**kwargs))
    except Exception:
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=40,
            multivariate=True,
            group=True,
            constant_liar=True,
            n_ei_candidates=48,
        )


def build_phase_b_sampler(seed: int) -> optuna.samplers.BaseSampler:
    if OPT_FUTURES_CONFIG.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False):
        return optuna.samplers.NSGAIISampler(
            seed=seed,
            population_size=int(OPT_FUTURES_CONFIG.get("FUTURES_NSGA2_POPULATION_SIZE", 30)),
            constraints_func=phase_b_constraints,
        )
    kwargs: dict[str, Any] = {"seed": seed, "restart_strategy": "ipop"}
    try:
        return optuna.samplers.CmaEsSampler(with_margin=True, lr_adapt=True, **kwargs)
    except (TypeError, ValueError):
        try:
            return optuna.samplers.CmaEsSampler(with_margin=True, **kwargs)
        except (TypeError, ValueError):
            pass
    try:
        return optuna.samplers.CmaEsSampler(lr_adapt=True, **kwargs)
    except (TypeError, ValueError):
        return optuna.samplers.CmaEsSampler(**kwargs)


def build_phase_a1_pruner() -> optuna.pruners.BasePruner:
    return optuna.pruners.WilcoxonPruner(p_threshold=0.10)
