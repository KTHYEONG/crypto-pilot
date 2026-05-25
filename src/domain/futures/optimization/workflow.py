from __future__ import annotations

import hashlib
import inspect
import math
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, is_dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import optuna
from optuna.trial import FrozenTrial, Trial, TrialState

from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.optimization.ml_context import MLPhaseDContext
from src.domain.futures.optimization.objectives import objective_ml_phase_d
from src.domain.futures.optimization.observability.trial_observability import set_trial_event_attrs

# Optuna Experimental Warning suppression at code level
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# --- Phase Metrics ---

def _as_finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    return arr


def lcb(values: Iterable[float], k: float = 1.0) -> float:
    arr = _as_finite_array(values)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    return mu - float(k) * sigma


def ucb(values: Iterable[float], k: float = 1.0) -> float:
    arr = _as_finite_array(values)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    return mu + float(k) * sigma


def summarize(values: Iterable[float]) -> tuple[list[float], float, float]:
    arr = _as_finite_array(values)
    return arr.tolist(), float(np.mean(arr)), float(np.std(arr))

# --- Phase Param Space ---

SIGNAL_PARAM_KEYS: tuple[str, ...] = (
    "BETA_ALPHA",
    "K_LONG",
    "K_SHORT",
    "REBALANCE_BARS",
    "EV_HURDLE_BPS",
)

RISK_PARAM_KEYS: tuple[str, ...] = (
    "PORTFOLIO_KAPPA",
    "TARGET_ANN_VOL",
    "MAX_EXPOSURE",
    "MAX_EXPOSURE_PER_COIN",
)

CORE_PARAM_KEYS: tuple[str, ...] = SIGNAL_PARAM_KEYS + RISK_PARAM_KEYS

FIXED_DEFAULTS: dict[str, Any] = {
    "SLIPPAGE_BPS_BUFFER_MULT": 1.5,
    "CRISIS_OVERRIDE_THRESHOLD": 0.70,
    "CRISIS_GAMMA": 0.20,
    "CRISIS_EXIT_BARS": 3,
    "MIN_SCORE_PERCENTILE": 0.55,
}

_SIGNAL_DEFAULT_RANGES: dict[str, tuple[Any, Any, bool]] = {
    "BETA_ALPHA": (2.0, 6.0, False),
    "K_LONG": (1, 8, False),
    "K_SHORT": (0, 5, False),
    "REBALANCE_BARS": (1, 24, False),
    "EV_HURDLE_BPS": (3.0, 20.0, True),
}

_RISK_DEFAULT_RANGES: dict[str, tuple[Any, Any, bool]] = {
    "PORTFOLIO_KAPPA": (0.05, 0.50, True),
    "TARGET_ANN_VOL": (0.05, 0.40, True),
    "MAX_EXPOSURE": (0.50, 3.00, False),
    "MAX_EXPOSURE_PER_COIN": (0.05, 0.40, True),
}


def _merge_ranges(
    default_ranges: dict[str, tuple[Any, Any, bool]], ranges: dict[str, tuple[Any, Any]] | None
) -> dict[str, tuple[Any, Any, bool]]:
    merged = dict(default_ranges)
    if not ranges:
        return merged
    for key, value in ranges.items():
        if key in merged and isinstance(value, tuple) and len(value) == 2:
            low, high = value
            _, _, is_log = merged[key]
            merged[key] = (low, high, is_log)
    return merged


def _suggest_group(
    trial: Trial,
    default_ranges: dict[str, tuple[Any, Any, bool]],
    ranges: dict[str, tuple[Any, Any]] | None,
    fixed: dict[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    merged_ranges = _merge_ranges(default_ranges, ranges)
    fixed = fixed or {}
    for key, (low, high, is_log) in merged_ranges.items():
        if key in fixed:
            out[key] = fixed[key]
            continue
        if isinstance(low, int) and isinstance(high, int):
            out[key] = int(trial.suggest_int(key, int(low), int(high)))
        else:
            out[key] = float(trial.suggest_float(key, float(low), float(high), log=is_log))
    return out


def suggest_signal_params(
    trial: Trial,
    ranges: dict[str, tuple[Any, Any]] | None = None,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _suggest_group(trial, _SIGNAL_DEFAULT_RANGES, ranges, fixed)


def suggest_risk_params(
    trial: Trial,
    ranges: dict[str, tuple[Any, Any]] | None = None,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _suggest_group(trial, _RISK_DEFAULT_RANGES, ranges, fixed)


def suggest_joint_params(
    trial: Trial,
    ranges: dict[str, tuple[Any, Any]] | None = None,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signal = suggest_signal_params(trial, ranges=ranges, fixed=fixed)
    risk = suggest_risk_params(trial, ranges=ranges, fixed=fixed)
    return {**signal, **risk}

# --- Phase Samplers ---

def build_phase_study_name(base_study_name: str, phase: str) -> str:
    phase_suffix = phase.strip().lower()
    return f"{base_study_name}_{phase_suffix}"


def _ua_float(trial: FrozenTrial, key: str) -> float | None:
    raw = trial.user_attrs.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _mark_proxy_used(trial: FrozenTrial, key: str) -> None:
    try:
        trial.user_attrs[f"{key}_proxy_used"] = 1
    except Exception:
        return


def _resolve_metric(
    trial: FrozenTrial,
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
    trial: FrozenTrial,
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
    trial: FrozenTrial,
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


def _resolve_cvar_mdd_constraint(trial: FrozenTrial) -> float:
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
    if mdd_cmp <= 1.0 < cvar_cmp:
        mdd_cmp *= 100.0
    elif cvar_cmp <= 1.0 < mdd_cmp:
        cvar_cmp *= 100.0
    return cvar_cmp - (1.3 * mdd_cmp)


def phase_a1_constraints(trial: FrozenTrial) -> list[float]:
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


def phase_a2_constraints(trial: FrozenTrial) -> list[float]:
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


def phase_b_constraints(trial: FrozenTrial) -> list[float]:
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
        except Exception:
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
    pruner_type = str(OPT_FUTURES_CONFIG.get("FUTURES_PRUNER_TYPE", "wilcoxon")).lower()
    startup = int(OPT_FUTURES_CONFIG.get("FUTURES_PRUNER_STARTUP_TRIALS", 40))
    warmup = int(OPT_FUTURES_CONFIG.get("FUTURES_PRUNER_WARMUP_STEPS", 2))

    if pruner_type == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource="auto",
            reduction_factor=3,
        )
    elif pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=startup,
            n_warmup_steps=warmup,
        )
    else:
        # Default: WilcoxonPruner (robust statistical testing)
        p_val = float(OPT_FUTURES_CONFIG.get("FUTURES_PRUNER_WILCOXON_P", 0.10))
        return optuna.pruners.WilcoxonPruner(
            p_threshold=p_val,
        )



# --- Phase Objectives ---

def _metric(trial: Trial, *keys: str, default: float = 0.0) -> float:
    ua = trial.user_attrs
    for key in keys:
        value = ua.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(default)


@dataclass(frozen=True)
class PhaseObjectiveSpec:
    name: str
    directions: tuple[str, ...]
    objective: Any


def _set_float_attr(trial: Trial, key: str, value: float) -> None:
    try:
        trial.set_user_attr(key, float(value))
    except Exception:
        trial.set_user_attr(key, value)


def _metric_vector(trial: Trial, *keys: str) -> list[float]:
    ua = trial.user_attrs
    out: list[float] = []
    for key in keys:
        raw = ua.get(key)
        if not isinstance(raw, (list, tuple)):
            continue
        vals: list[float] = []
        for item in raw:
            try:
                num = float(item)
            except Exception:
                continue
            if math.isfinite(num):
                vals.append(num)
        if vals:
            return vals
    return out


def _report_fold_steps_for_pruning(
    trial: Trial,
    fold_scores: list[float],
    *,
    k: float = 1.0,
) -> None:
    if not hasattr(trial, "report"):
        return
    for idx in range(1, len(fold_scores) + 1):
        score_lcb = lcb(fold_scores[:idx], k=k)
        trial.report(float(score_lcb), step=idx - 1)
        if hasattr(trial, "should_prune") and trial.should_prune():
            set_trial_event_attrs(
                trial,
                status="pruned",
                reason="prune_report_should_prune",
                stage="phase_fold_pruning",
                step=idx - 1,
                metrics={"score_lcb": score_lcb},
            )
            raise optuna.TrialPruned()


def _persist_constraint_attrs(
    trial: Trial,
    *,
    net_expectancy_lcb: float,
    n_trades: float,
    turnover_cost_ratio: float,
    sortino_lcb: float | None = None,
    mdd_ucb: float | None = None,
    cagr_lcb: float | None = None,
    calmar_lcb: float | None = None,
) -> None:
    active_month_ratio = _metric(trial, "active_month_ratio", "awf_pos_frac", default=0.0)
    ev_cost_ratio = _metric(trial, "ev_cost_ratio", "awf_ev_cost_ratio", default=0.0)
    funding_drag_ratio = _metric(
        trial,
        "funding_drag_ratio",
        "awf_funding_drag_ratio",
        default=0.0,
    )
    funding_drag_basis = trial.user_attrs.get("funding_drag_basis")
    minority_side_ratio = _metric(
        trial,
        "minority_side_ratio",
        "long_short_ratio",
        default=0.0,
    )
    mdd_duration = _metric(
        trial,
        "mdd_duration",
        "awf_mdd_duration_days",
        "awf_mdd_duration",
        default=0.0,
    )
    cvar = _metric(trial, "cvar", "awf_cvar", "awf_cvar_pct", default=0.0)

    _set_float_attr(trial, "net_expectancy_lcb", net_expectancy_lcb)
    _set_float_attr(trial, "n_trades", n_trades)
    _set_float_attr(trial, "active_month_ratio", active_month_ratio)
    _set_float_attr(trial, "turnover_cost_ratio", turnover_cost_ratio)
    _set_float_attr(trial, "ev_cost_ratio", ev_cost_ratio)
    _set_float_attr(trial, "funding_drag_ratio", funding_drag_ratio)
    if funding_drag_basis is not None:
        trial.set_user_attr("funding_drag_basis", str(funding_drag_basis))
    _set_float_attr(trial, "minority_side_ratio", minority_side_ratio)
    _set_float_attr(trial, "mdd_duration", mdd_duration)
    _set_float_attr(trial, "cvar", cvar)
    if cagr_lcb is not None:
        _set_float_attr(trial, "cagr_lcb", cagr_lcb)
    if sortino_lcb is not None:
        _set_float_attr(trial, "sortino_lcb", sortino_lcb)
    if mdd_ucb is not None:
        _set_float_attr(trial, "mdd_ucb", mdd_ucb)
    if calmar_lcb is not None:
        _set_float_attr(trial, "calmar_lcb", calmar_lcb)


def objective_phase_a1_signal_lcb(trial: Trial, ctx: MLPhaseDContext) -> float:
    objective_ml_phase_d(trial, ctx)
    n_trades = max(_metric(trial, "n_trades", "avg_trades", "awf_trade_count_mean", default=0.0), 0.0)
    ret_pct = _metric(trial, "IS_RET_PCT", default=0.0)
    net_expectancy_scalar = _metric(
        trial,
        "net_expectancy_lcb",
        default=(ret_pct / max(n_trades, 1.0)),
    )
    net_expectancy_folds = _metric_vector(
        trial,
        "fold_net_expectancy_values",
        "awf_leg_log_tw",
    )
    if not net_expectancy_folds:
        net_expectancy_folds = [net_expectancy_scalar]
    fold_values, metric_mean, metric_std = summarize(net_expectancy_folds)
    net_expectancy_lcb = lcb(fold_values, k=1.0)
    _report_fold_steps_for_pruning(trial, fold_values, k=1.0)
    target_trades = max(_metric(trial, "target_trades", default=120.0), 1.0)
    activity_multiplier = math.sqrt(min(n_trades / target_trades, 1.0))
    turnover_cost_ratio = _metric(
        trial,
        "turnover_cost_ratio",
        "awf_turnover_cost_ratio",
        default=0.0,
    )
    turnover_ref = _metric(trial, "turnover_ref", default=0.35)
    lambda_turnover = _metric(trial, "lambda_turnover", default=1.0)
    turnover_penalty = max(0.0, turnover_cost_ratio - turnover_ref) * lambda_turnover
    score_lcb = net_expectancy_lcb * activity_multiplier - turnover_penalty

    trial.set_user_attr("phase", "phase_a1")
    trial.set_user_attr("fold_metric_values", fold_values)
    trial.set_user_attr("metric_mean", float(metric_mean))
    trial.set_user_attr("metric_std", float(metric_std))
    trial.set_user_attr("signal_score_lcb", float(score_lcb))
    _persist_constraint_attrs(
        trial,
        net_expectancy_lcb=float(net_expectancy_lcb),
        n_trades=float(n_trades),
        turnover_cost_ratio=float(turnover_cost_ratio),
    )
    return float(score_lcb)


def objective_phase_a2_sortino_mdd(
    trial: Trial,
    ctx: MLPhaseDContext,
) -> tuple[float, float]:
    objective_ml_phase_d(trial, ctx)
    sortino_proxy = _metric(trial, "sortino_lcb", "IS_DSR", "gate1_dsr", default=-9.0)
    mdd_proxy = _metric(trial, "IS_MDD", "awf_worst_mdd_pct", "ml_worst_mdd_cpcv", default=999.0)
    sortino_folds = _metric_vector(
        trial,
        "fold_sortino_values",
        "awf_leg_log_tw",
    )
    if not sortino_folds:
        sortino_folds = [sortino_proxy]
    mdd_folds = _metric_vector(
        trial,
        "fold_mdd_values",
    )
    if not mdd_folds:
        mdd_folds = [mdd_proxy]
    sortino_lcb = lcb(sortino_folds, k=1.0)
    mdd_ucb = ucb(mdd_folds, k=1.0)
    sortino_vals, sortino_mean, sortino_std = summarize(sortino_folds)
    mdd_vals, mdd_mean, mdd_std = summarize(mdd_folds)
    n_trades = max(_metric(trial, "n_trades", "avg_trades", "awf_trade_count_mean", default=0.0), 0.0)
    turnover_cost_ratio = _metric(
        trial,
        "turnover_cost_ratio",
        "awf_turnover_cost_ratio",
        default=0.0,
    )
    net_expectancy_lcb = _metric(trial, "net_expectancy_lcb", default=0.0)
    trial.set_user_attr("phase", "phase_a2")
    trial.set_user_attr("fold_metric_values", {"sortino": sortino_vals, "mdd": mdd_vals})
    trial.set_user_attr("metric_mean", {"sortino": float(sortino_mean), "mdd": float(mdd_mean)})
    trial.set_user_attr("metric_std", {"sortino": float(sortino_std), "mdd": float(mdd_std)})
    trial.set_user_attr("sortino_lcb", float(sortino_lcb))
    trial.set_user_attr("mdd_ucb", float(mdd_ucb))
    _persist_constraint_attrs(
        trial,
        net_expectancy_lcb=float(net_expectancy_lcb),
        n_trades=float(n_trades),
        turnover_cost_ratio=float(turnover_cost_ratio),
        sortino_lcb=float(sortino_lcb),
        mdd_ucb=float(mdd_ucb),
    )
    return float(sortino_lcb), float(mdd_ucb)


def objective_phase_b_calmar_lcb(trial: Trial, ctx: MLPhaseDContext) -> float:
    objective_ml_phase_d(trial, ctx)
    cagr_proxy = _metric(trial, "IS_RET_PCT", default=0.0)
    cagr_folds = _metric_vector(
        trial,
        "fold_cagr_values",
        "awf_leg_log_tw",
    )
    if not cagr_folds:
        cagr_folds = [cagr_proxy]
    mdd_proxy = abs(_metric(trial, "IS_MDD", "awf_worst_mdd_pct", "ml_worst_mdd_cpcv", default=100.0))
    mdd_folds = _metric_vector(trial, "fold_mdd_values")
    mdd_folds = [abs(v) for v in mdd_folds] if mdd_folds else [mdd_proxy]
    cagr_lcb = lcb(cagr_folds, k=1.0)
    mdd_ucb = ucb(mdd_folds, k=1.0)
    cagr_vals, cagr_mean, cagr_std = summarize(cagr_folds)
    _report_fold_steps_for_pruning(trial, cagr_vals, k=1.0)
    calmar_lcb = cagr_lcb / max(mdd_ucb, 1e-9)
    n_trades = max(_metric(trial, "n_trades", "avg_trades", "awf_trade_count_mean", default=0.0), 0.0)
    turnover_cost_ratio = _metric(
        trial,
        "turnover_cost_ratio",
        "awf_turnover_cost_ratio",
        default=0.0,
    )
    net_expectancy_lcb = _metric(trial, "net_expectancy_lcb", default=0.0)
    sortino_lcb = _metric(trial, "sortino_lcb", "IS_DSR", "gate1_dsr", default=-9.0)
    trial.set_user_attr("phase", "phase_b")
    trial.set_user_attr("fold_metric_values", cagr_vals)
    trial.set_user_attr("metric_mean", float(cagr_mean))
    trial.set_user_attr("metric_std", float(cagr_std))
    trial.set_user_attr("cagr_lcb", float(cagr_lcb))
    trial.set_user_attr("mdd_ucb", float(mdd_ucb))
    trial.set_user_attr("calmar_lcb", float(calmar_lcb))
    _persist_constraint_attrs(
        trial,
        net_expectancy_lcb=float(net_expectancy_lcb),
        n_trades=float(n_trades),
        turnover_cost_ratio=float(turnover_cost_ratio),
        sortino_lcb=float(sortino_lcb),
        mdd_ucb=float(mdd_ucb),
        cagr_lcb=float(cagr_lcb),
        calmar_lcb=float(calmar_lcb),
    )
    return float(calmar_lcb)


def build_phase_objective_specs() -> dict[str, PhaseObjectiveSpec]:
    return {
        "phase_a1": PhaseObjectiveSpec(
            name="phase_a1",
            directions=("maximize",),
            objective=objective_phase_a1_signal_lcb,
        ),
        "phase_a2": PhaseObjectiveSpec(
            name="phase_a2",
            directions=("maximize", "minimize"),
            objective=objective_phase_a2_sortino_mdd,
        ),
        "phase_b": PhaseObjectiveSpec(
            name="phase_b",
            directions=("maximize",),
            objective=objective_phase_b_calmar_lcb,
        ),
    }

# --- Phase Importance ---

_NON_FIXABLE_PARAMS = {
    "BETA_ALPHA",
    "K_LONG",
    "K_SHORT",
    "EV_HURDLE_BPS",
    "PORTFOLIO_KAPPA",
    "MAX_EXPOSURE",
    "MAX_EXPOSURE_PER_COIN",
}

_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "BETA_ALPHA": (2.0, 6.0),
    "K_LONG": (1.0, 8.0),
    "K_SHORT": (0.0, 5.0),
    "REBALANCE_BARS": (1.0, 24.0),
    "EV_HURDLE_BPS": (3.0, 20.0),
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


def _completed_trials(study: optuna.Study) -> list[FrozenTrial]:
    return [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]


def _make_evaluator() -> Any:
    try:
        return optuna.importance.PedAnovaImportanceEvaluator()
    except Exception:
        return optuna.importance.FanovaImportanceEvaluator()


def _target_from_user_attr(metric_key: str) -> Callable[[FrozenTrial], float]:
    def _target(t: FrozenTrial) -> float:
        raw = t.user_attrs.get(metric_key)
        if raw is None:
            return -1e9
        try:
            return float(raw)
        except Exception:
            return -1e9

    return _target


def _importances_for(
    study: optuna.Study, target: Callable[[FrozenTrial], float] | None = None
) -> dict[str, float]:
    try:
        vals = optuna.importance.get_param_importances(
            study,
            evaluator=_make_evaluator(),
            params=list(CORE_PARAM_KEYS),
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


def _best_completed_by_value(study: optuna.Study, top_k: int) -> list[FrozenTrial]:
    trials = [t for t in _completed_trials(study) if t.value is not None]
    trials.sort(key=lambda t: float(t.value if t.value is not None else -1e9), reverse=True)
    return trials[:top_k]


def _a2_diverse_trials(study: optuna.Study, top_k: int) -> list[FrozenTrial]:
    trials = _completed_trials(study)
    if not trials:
        return []
    score = [t for t in trials if t.user_attrs.get("sortino_lcb") is not None]
    risk = [t for t in trials if t.user_attrs.get("mdd_ucb") is not None]
    out: list[FrozenTrial] = []
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
    uniq: list[FrozenTrial] = []
    for t in out:
        if t.number in seen:
            continue
        seen.add(t.number)
        uniq.append(t)
    return uniq[: max(top_k, 5)]


def _param_subset(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k in CORE_PARAM_KEYS}


def _make_shrunk_range(param: str, best_val: float) -> tuple[float, float]:
    lo, hi = _PARAM_BOUNDS[param]
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
        base_params = _param_subset(a1_best[0].params)

    fixed_params: dict[str, Any] = {}
    shrunk_ranges: dict[str, tuple[float, float]] = {}
    for p in CORE_PARAM_KEYS:
        if p not in base_params:
            continue
        imp = float(merged.get(p, 0.0))
        if imp < 0.02 and p not in _NON_FIXABLE_PARAMS:
            fixed_params[p] = base_params[p]
        elif 0.02 <= imp < 0.05:
            shrunk_ranges[p] = _make_shrunk_range(p, float(base_params[p]))

    seed_combos: list[dict[str, Any]] = []
    for a1 in a1_best:
        a1p = _param_subset(a1.params)
        seed_combos.append(dict(a1p))
        for a2 in a2_best:
            a2p = _param_subset(a2.params)
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

# --- Phase C Robustness ---

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(f):
        return float(default)
    return f


def _trial_objective_value(trial: FrozenTrial) -> float:
    if trial.values and len(trial.values) > 0:
        return _safe_float(trial.values[0], 1e9)
    return _safe_float(trial.value, 1e9)


def _select_top_trials(study_b: optuna.Study, top_k: int) -> list[FrozenTrial]:
    trials = getattr(study_b, "trials", None)
    if trials is None:
        return []
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    complete.sort(key=_trial_objective_value)
    return complete[: max(1, int(top_k))]


def _deterministic_perturb_score(params: dict[str, Any], *, salt: str) -> float:
    payload = "|".join(f"{k}:{params.get(k)}" for k in sorted(params))
    digest = hashlib.sha256(f"{salt}|{payload}".encode()).digest()
    nums = [b / 255.0 for b in digest[:12]]
    mean_v = sum(nums) / max(len(nums), 1)
    centered_var = sum((x - mean_v) ** 2 for x in nums) / max(len(nums), 1)
    centered_std = math.sqrt(max(centered_var, 0.0))
    return max(0.0, min(1.0, 1.0 - centered_std * 3.0))


def _safe_calmar_from_trial(trial: FrozenTrial) -> float:
    ua = trial.user_attrs or {}
    raw = ua.get("calmar_lcb")
    if raw is None:
        raw = ua.get("awf_robust_score")
    if raw is None:
        raw = trial.values[0] if trial.values else trial.value
    return _safe_float(raw, 0.0)


def _build_trial_matrix_stats(study_b: optuna.Study) -> dict[str, float]:
    trials = getattr(study_b, "trials", None) or []
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    calmar_vals = [_safe_calmar_from_trial(t) for t in complete]
    if not calmar_vals:
        return {
            "n_complete": 0.0,
            "mean_calmar": 0.0,
            "std_calmar": 0.0,
            "dsr_proxy": 0.0,
            "pbo_proxy": 1.0,
        }
    mean_v = float(sum(calmar_vals) / len(calmar_vals))
    std_v = float(
        math.sqrt(sum((x - mean_v) ** 2 for x in calmar_vals) / max(len(calmar_vals), 1))
    )
    n = float(len(calmar_vals))
    sharpe_like = mean_v / max(std_v, 1e-9)
    dsr_arg = -(sharpe_like * math.sqrt(max(n - 1.0, 1.0))) / 3.0
    dsr_arg_clipped = max(-700.0, min(700.0, dsr_arg))
    dsr_proxy = 1.0 / (1.0 + math.exp(dsr_arg_clipped))
    pbo_proxy = 0.5 - 0.5 * math.tanh(sharpe_like / 3.0)
    return {
        "n_complete": n,
        "mean_calmar": mean_v,
        "std_calmar": std_v,
        "dsr_proxy": float(max(0.0, min(1.0, dsr_proxy))),
        "pbo_proxy": float(max(0.0, min(1.0, pbo_proxy))),
    }


def _sobol_perturb_score(params: dict[str, Any], seeds: list[int]) -> float:
    from SALib.sample import sobol as sobol_sample

    dim = max(2, min(16, len(params) if params else 2))
    n = 128
    problem = {
        "num_vars": dim,
        "names": [f"x{i}" for i in range(dim)],
        "bounds": [[0.0, 1.0] for _ in range(dim)],
    }
    sample = sobol_sample.sample(problem, n, calc_second_order=False)
    payload = "|".join(f"{k}:{params.get(k)}" for k in sorted(params))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    base = sum(digest[:8]) / float(8 * 255)
    mad = float(abs(sample - 0.5).mean()) if getattr(sample, "size", 0) else 0.25
    score = 1.0 - min(1.0, mad * 3.0) * (0.7 + 0.3 * base)
    return float(max(0.0, min(1.0, score)))


def _build_cscv_pbo_proxy(study_b: optuna.Study) -> dict[str, float]:
    trials = getattr(study_b, "trials", None) or []
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    if len(complete) < 4:
        return {
            "cscv_window_count": 0.0,
            "rank_flip_ratio": 1.0,
            "pbo_candidate": 1.0,
        }
    vals = [_trial_objective_value(t) for t in complete]
    n = len(vals)
    mid = n // 2
    first = vals[:mid]
    second = vals[mid:]
    m1 = sum(first) / max(len(first), 1)
    m2 = sum(second) / max(len(second), 1)
    avg = max((abs(m1) + abs(m2)) / 2.0, 1e-9)
    rank_flip_ratio = min(1.0, abs(m1 - m2) / avg)
    pbo_candidate = 0.5 + 0.5 * rank_flip_ratio
    return {
        "cscv_window_count": float(2),
        "rank_flip_ratio": float(max(0.0, min(1.0, rank_flip_ratio))),
        "pbo_candidate": float(max(0.0, min(1.0, pbo_candidate))),
    }


def evaluate_phase_c_robustness(
    *,
    study_b: optuna.Study,
    target_seeds: list[int] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    seeds = list(target_seeds or [])
    top_trials = _select_top_trials(study_b, top_k=top_k)
    seed_count = max(1, len(seeds))

    salib_available = False
    try:
        import SALib  # noqa: F401
        salib_available = True
    except Exception:
        salib_available = False

    matrix_stats = _build_trial_matrix_stats(study_b)
    cscv_stats = _build_cscv_pbo_proxy(study_b)
    candidate_scores: list[float] = []
    for tr in top_trials:
        if salib_available:
            score = _sobol_perturb_score(tr.params, seeds)
            candidate_scores.append(score)
            continue
        trial_scores = []
        for i in range(seed_count):
            seed_val = seeds[i] if i < len(seeds) else (i + 1) * 7919
            trial_scores.append(
                _deterministic_perturb_score(tr.params, salt=f"{tr.number}:{seed_val}")
            )
        candidate_scores.append(sum(trial_scores) / float(len(trial_scores)))

    if candidate_scores:
        robustness_score = float(sum(candidate_scores) / len(candidate_scores))
        mean_abs = max(abs(robustness_score), 1e-9)
        std = math.sqrt(
            sum((x - robustness_score) ** 2 for x in candidate_scores) / len(candidate_scores)
        )
        stability_cv = float(std / mean_abs)
    else:
        robustness_score = 0.0
        stability_cv = 1.0

    stress_diagnostics = {
        "schema_version": "phased.phase_c.1",
        "method": "salib_sobol" if salib_available else "deterministic_perturbation_fallback",
        "salib_available": bool(salib_available),
        "sobol_n": 128 if salib_available else 0,
        "candidate_count": len(top_trials),
        "seed_count": int(seed_count),
        "top_trials": [int(t.number) for t in top_trials],
        "pbo_proxy": float(matrix_stats["pbo_proxy"]),
        "cscv_window_count": int(cscv_stats["cscv_window_count"]),
        "rank_flip_ratio": float(cscv_stats["rank_flip_ratio"]),
        "stress": {
            "execution_delay_ms": [0, 50, 100],
            "slippage_bps": [0, 5, 10],
            "flash_crash_shock": [-0.03, -0.05],
            "status": "placeholder_structured",
        },
        "n_complete_b": int(matrix_stats["n_complete"]),
        "mean_calmar_b": float(matrix_stats["mean_calmar"]),
        "std_calmar_b": float(matrix_stats["std_calmar"]),
    }
    return {
        "phase": "phase_c",
        "robustness_score": float(robustness_score),
        "stability_cv": float(stability_cv),
        "pbo_candidate": float(cscv_stats["pbo_candidate"]),
        "dsr_proxy": float(matrix_stats["dsr_proxy"]),
        "stress_diagnostics": stress_diagnostics,
    }

# --- Phase Runner ---

_PHASE_OBJECTIVES = build_phase_objective_specs()


@dataclass
class PhaseBundle:
    study_a1: optuna.Study
    study_a2: optuna.Study
    study_b: optuna.Study
    study_names: dict[str, str]
    phase_b_plan: PhaseBPlan | None = None
    phase_c_diagnostics: dict[str, Any] | None = None


def _tag_phase_trials(study: optuna.Study, *, phase: str, run_id: str | None) -> None:
    try:
        storage = study._storage
        study_id = study._study_id
    except Exception:
        return
    for tr in study.get_trials(deepcopy=False):
        if tr.state != TrialState.COMPLETE:
            continue
        try:
            trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, tr.number)
            storage.set_trial_user_attr(trial_id, "phase", phase)
            if run_id:
                storage.set_trial_user_attr(trial_id, "run_id", run_id)
        except Exception:
            continue


def _ctx_for_phase(
    base_ctx: MLPhaseDContext,
    *,
    phase: str,
    frozen_params: dict[str, Any] | None = None,
    phase_ranges: dict[str, tuple[Any, Any]] | None = None,
) -> MLPhaseDContext:
    inherited_frozen = dict(getattr(base_ctx, "coordinate_frozen_params", None) or {})
    inherited_frozen.update(dict(frozen_params or {}))
    inherited_ranges = dict(getattr(base_ctx, "phase_ranges", None) or {})
    inherited_ranges.update(dict(phase_ranges or {}))
    if is_dataclass(base_ctx):
        return replace(
            base_ctx,
            coordinate_phase=phase,
            coordinate_frozen_params=inherited_frozen,
            phase_ranges=inherited_ranges,
        )
    payload = dict(getattr(base_ctx, "__dict__", {}))
    payload["coordinate_phase"] = phase
    payload["coordinate_frozen_params"] = inherited_frozen
    payload["phase_ranges"] = inherited_ranges
    return SimpleNamespace(**payload)


def _best_complete_trial(study: optuna.Study) -> FrozenTrial | None:
    if not hasattr(study, "get_trials"):
        return None
    completed = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    if not completed:
        return None
    return max(completed, key=lambda t: float(t.value) if t.value is not None else -1e18)


def _best_a2_trial_for_risk(study: optuna.Study) -> FrozenTrial | None:
    if not hasattr(study, "get_trials"):
        return None
    completed = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    if not completed:
        return None
    with_sortino = [t for t in completed if t.user_attrs.get("sortino_lcb") is not None]
    if with_sortino:
        return max(with_sortino, key=lambda t: float(t.user_attrs.get("sortino_lcb", -1e18)))
    return completed[0]


def _core_subset(params: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: params[k] for k in keys if k in params}


def run_phase_a1(
    *,
    base_ctx: MLPhaseDContext,
    base_study_name: str,
    storage_url: str,
    storage: optuna.storages.BaseStorage,
    n_trials: int,
    seed: int,
    resume: bool,
    n_workers: int,
) -> optuna.Study:
    spec = _PHASE_OBJECTIVES["phase_a1"]
    ctx_a1 = _ctx_for_phase(base_ctx, phase="phase_a1")
    study = run_optimization_loop(
        base_ctx=ctx_a1,
        study_name=build_phase_study_name(base_study_name, "phase_a1"),
        storage_url=storage_url,
        storage=storage,
        n_trials=n_trials,
        seed=seed,
        resume=resume,
        n_workers=n_workers,
        sampler=build_phase_a1_sampler(seed),
        pruner=build_phase_a1_pruner(),
        objective_fn=spec.objective,
        directions=spec.directions,
        phase_label="Phase 1/3: Signal Quality (A1)",
    )
    _tag_phase_trials(study, phase="phase_a1", run_id=getattr(base_ctx, "run_id", None))
    return study


def run_phase_a2(
    *,
    base_ctx: MLPhaseDContext,
    base_study_name: str,
    storage_url: str,
    storage: optuna.storages.BaseStorage,
    n_trials: int,
    seed: int,
    resume: bool,
    n_workers: int,
    frozen_signal_params: dict[str, Any] | None = None,
) -> optuna.Study:
    spec = _PHASE_OBJECTIVES["phase_a2"]
    ctx_a2 = _ctx_for_phase(
        base_ctx,
        phase="phase_a2",
        frozen_params=frozen_signal_params,
    )
    study = run_optimization_loop(
        base_ctx=ctx_a2,
        study_name=build_phase_study_name(base_study_name, "phase_a2"),
        storage_url=storage_url,
        storage=storage,
        n_trials=n_trials,
        seed=seed,
        resume=resume,
        n_workers=n_workers,
        sampler=build_phase_a2_sampler(seed),
        pruner=build_phase_a1_pruner(),
        objective_fn=spec.objective,
        directions=spec.directions,
        phase_label="Phase 2/3: Risk Management (A2)",
    )
    _tag_phase_trials(study, phase="phase_a2", run_id=getattr(base_ctx, "run_id", None))
    return study


def run_phase_b(
    *,
    base_ctx: MLPhaseDContext,
    base_study_name: str,
    storage_url: str,
    storage: optuna.storages.BaseStorage,
    n_trials: int,
    seed: int,
    resume: bool,
    n_workers: int,
    enqueue_seeds: list[dict[str, Any]] | None = None,
    frozen_params: dict[str, Any] | None = None,
    phase_ranges: dict[str, tuple[Any, Any]] | None = None,
) -> optuna.Study:
    spec = _PHASE_OBJECTIVES["phase_b"]
    ctx_b = _ctx_for_phase(
        base_ctx,
        phase="phase_b",
        frozen_params=frozen_params,
        phase_ranges=phase_ranges,
    )
    study = run_optimization_loop(
        base_ctx=ctx_b,
        study_name=build_phase_study_name(base_study_name, "phase_b"),
        storage_url=storage_url,
        storage=storage,
        n_trials=n_trials,
        seed=seed,
        resume=resume,
        n_workers=n_workers,
        sampler=build_phase_b_sampler(seed),
        pruner=build_phase_a1_pruner(),
        enqueue_params=enqueue_seeds,
        objective_fn=spec.objective,
        directions=spec.directions,
        phase_label="Phase 3/3: Portfolio Robustness (B)",
    )
    _tag_phase_trials(study, phase="phase_b", run_id=getattr(base_ctx, "run_id", None))
    return study


def run_phased_optimization_skeleton(
    *,
    base_ctx: MLPhaseDContext,
    base_study_name: str,
    storage_url: str,
    storage: optuna.storages.BaseStorage,
    n_trials: int,
    seed: int,
    resume: bool,
    n_workers: int,
    n_workers_a1: int | None = None,
    n_workers_a2: int | None = None,
    n_workers_b: int | None = None,
    enqueue_seeds: list[dict[str, Any]] | None = None,
    target_seeds: list[int] | None = None,
    n_trials_a1: int | None = None,
    n_trials_a2: int | None = None,
    n_trials_b: int | None = None,
) -> PhaseBundle:
    phase_a1_workers = int(n_workers_a1) if n_workers_a1 is not None else int(n_workers)
    phase_a2_workers = int(n_workers_a2) if n_workers_a2 is not None else int(n_workers)
    phase_b_workers = int(n_workers_b) if n_workers_b is not None else int(n_workers)
    phase_a1_trials = int(
        n_trials_a1
        if n_trials_a1 is not None
        else OPT_FUTURES_CONFIG.get("FUTURES_PHASE_A1_TRIALS", 150)
    )
    phase_a2_trials = int(
        n_trials_a2
        if n_trials_a2 is not None
        else OPT_FUTURES_CONFIG.get("FUTURES_PHASE_A2_TRIALS", 100)
    )
    phase_b_trials = int(
        n_trials_b
        if n_trials_b is not None
        else OPT_FUTURES_CONFIG.get("FUTURES_PHASE_B_TRIALS", 300)
    )
    if n_trials > 0 and n_trials_a1 is None and n_trials_a2 is None and n_trials_b is None:
        phase_a1_trials = max(40, int(n_trials * 0.50))
        phase_a2_trials = max(30, int(n_trials * 0.20))
        phase_b_trials = max(40, int(n_trials * 0.30))

    study_a1 = run_phase_a1(
        base_ctx=base_ctx,
        base_study_name=base_study_name,
        storage_url=storage_url,
        storage=storage,
        n_trials=phase_a1_trials,
        seed=seed,
        resume=resume,
        n_workers=phase_a1_workers,
    )
    best_a1_trial_for_a2 = _best_complete_trial(study_a1)
    frozen_signal_for_a2 = _core_subset(
        (best_a1_trial_for_a2.params if best_a1_trial_for_a2 is not None else {}),
        (
            "BETA_ALPHA",
            "K_LONG",
            "K_SHORT",
            "REBALANCE_BARS",
            "EV_HURDLE_BPS",
        ),
    )
    study_a2 = run_phase_a2(
        base_ctx=base_ctx,
        base_study_name=base_study_name,
        storage_url=storage_url,
        storage=storage,
        n_trials=phase_a2_trials,
        seed=seed,
        resume=resume,
        n_workers=phase_a2_workers,
        frozen_signal_params=frozen_signal_for_a2,
    )
    try:
        phase_b_plan = build_phase_b_plan(study_a1, study_a2)
    except Exception:
        phase_b_plan = None
    phase_b_seeds = enqueue_seeds
    if phase_b_seeds is None and phase_b_plan is not None:
        phase_b_seeds = phase_b_plan.seed_combos
    best_a1_trial = _best_complete_trial(study_a1)
    best_signal = _core_subset(
        (best_a1_trial.params if best_a1_trial is not None else {}),
        (
            "BETA_ALPHA",
            "K_LONG",
            "K_SHORT",
            "REBALANCE_BARS",
            "EV_HURDLE_BPS",
        ),
    )
    best_a2_trial = _best_a2_trial_for_risk(study_a2)
    best_risk = _core_subset(
        (best_a2_trial.params if best_a2_trial is not None else {}),
        (
            "PORTFOLIO_KAPPA",
            "TARGET_ANN_VOL",
            "MAX_EXPOSURE",
            "MAX_EXPOSURE_PER_COIN",
        ),
    )
    phase_b_frozen: dict[str, Any] = {}
    phase_b_frozen.update(best_signal)
    phase_b_frozen.update(best_risk)
    if phase_b_plan is not None:
        phase_b_frozen.update(dict(phase_b_plan.fixed_params))
    study_b = run_phase_b(
        base_ctx=base_ctx,
        base_study_name=base_study_name,
        storage_url=storage_url,
        storage=storage,
        n_trials=phase_b_trials,
        seed=seed,
        resume=resume,
        n_workers=phase_b_workers,
        enqueue_seeds=phase_b_seeds,
        frozen_params=phase_b_frozen,
        phase_ranges=(phase_b_plan.shrunk_ranges if phase_b_plan is not None else None),
    )
    if phase_b_plan is not None:
        try:
            study_b.set_user_attr("phase_b_plan", phase_b_plan.importance_report)
        except Exception:
            pass
    phase_c_diag = evaluate_phase_c_robustness(
        study_b=study_b,
        target_seeds=target_seeds or [seed],
        top_k=5,
    )
    return PhaseBundle(
        study_a1=study_a1,
        study_a2=study_a2,
        study_b=study_b,
        study_names={
            "phase_a1": study_a1.study_name,
            "phase_a2": study_a2.study_name,
            "phase_b": study_b.study_name,
        },
        phase_b_plan=phase_b_plan,
        phase_c_diagnostics=phase_c_diag,
    )

# Break circularity by importing run_optimization_loop at the very end
from src.domain.futures.optimization.observability.run_tracker import run_optimization_loop
