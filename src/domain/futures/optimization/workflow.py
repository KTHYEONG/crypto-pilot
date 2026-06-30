from __future__ import annotations

import hashlib
import inspect
import logging
import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, is_dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import optuna
from numpy.typing import NDArray
from optuna.trial import FrozenTrial, Trial, TrialState

from src.domain.futures.optimization.ml_context import MLPhaseDContext
from src.domain.futures.optimization.objectives import objective_ml_phase_d
from src.domain.futures.optimization.observability.trial_observability import set_trial_event_attrs
from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.strategy.tiered_workflow.deployable_score import (
    build_layer2_deployable_score,
)
from src.domain.futures.strategy.tiered_workflow.diagnostics import (
    build_layer_universe_audit,
)

if TYPE_CHECKING:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache

_logger = logging.getLogger(__name__)

# Optuna Experimental Warning suppression at code level
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# --- Phase Metrics ---

def _as_finite_array(values: Iterable[float]) -> NDArray[np.float64]:
    arr: NDArray[np.float64] = cast(NDArray[np.float64], np.asarray(list(values), dtype=np.float64))
    if arr.size == 0:
        return cast(NDArray[np.float64], np.array([0.0], dtype=np.float64))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return cast(NDArray[np.float64], np.array([0.0], dtype=np.float64))
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
        except Exception as exc:
            _logger.debug("BoTorchSampler signature inspection failed: %s", exc)
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
        except Exception:  # noqa: S112
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
            except Exception:  # noqa: S112
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
        except Exception as exc:
            _logger.debug("failed to tag trial metadata for %s: %s", tr.number, exc)
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
        dataclass_ctx = cast(Any, base_ctx)
        return cast(
            MLPhaseDContext,
            replace(
                dataclass_ctx,
                coordinate_phase=phase,
                coordinate_frozen_params=inherited_frozen,
                phase_ranges=inherited_ranges,
            ),
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

    import time
    t_phase_a1 = time.perf_counter()
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
    _logger.debug("[PROF] PhasedOpt phase_a1 elapsed_s=%.4f", time.perf_counter() - t_phase_a1)
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
    t_phase_a2 = time.perf_counter()
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
    _logger.debug("[PROF] PhasedOpt phase_a2 elapsed_s=%.4f", time.perf_counter() - t_phase_a2)
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
    t_phase_b = time.perf_counter()
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
    _logger.debug("[PROF] PhasedOpt phase_b elapsed_s=%.4f", time.perf_counter() - t_phase_b)
    if phase_b_plan is not None:
        with suppress(Exception):
            study_b.set_user_attr("phase_b_plan", phase_b_plan.importance_report)
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

# ---------------------------------------------------------------------------
# Tiered Pipeline: Decoupled Optuna Study Support
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TieredContext:
    """Optuna study context for the Tiered hybrid pipeline."""

    labeled_events: Any
    aligned: Any          # AlignedMarketData
    cfg: Any              # CandidateStrategyConfig
    window: Any           # LayeredWindow
    caps: Any             # PortfolioCaps
    tf: str
    fixed_l1_params: dict[str, Any] | None = None   # L2 study 시 L1 best params 고정
    l2_sim_cache: L2SimulationCache | None = None
    awf_folds: tuple[Any, ...] | None = None


def suggest_layered_params(
    trial: Trial,
    layer: Literal["L1", "L2"],
    *,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """L1_ALPHA_SPACE 또는 L2_ALLOC_SPACE를 Optuna trial로 suggest.

    Args:
        trial: Optuna Trial 객체.
        layer: "L1" 또는 "L2" 레이어 구분.
        fixed: suggest 생략하고 직접 주입할 키-값 딕셔너리.

    Returns:
        파라미터 딕셔너리 (layer의 전체 키 포함).
    """
    from src.domain.futures.optimization.opt_config import L1_ALPHA_SPACE, L2_ALLOC_SPACE

    space = L1_ALPHA_SPACE if layer == "L1" else L2_ALLOC_SPACE
    result: dict[str, Any] = {}
    fixed = fixed or {}

    for key, spec in space.items():
        if key in fixed:
            result[key] = fixed[key]
            continue
        t = spec["type"]
        if t == "categorical":
            result[key] = trial.suggest_categorical(key, spec["choices"])
        elif t == "int":
            result[key] = trial.suggest_int(
                key,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
            )
        elif t == "float":
            log = bool(spec.get("log", False))
            result[key] = trial.suggest_float(
                key,
                float(spec["low"]),
                float(spec["high"]),
                step=float(spec["step"]) if "step" in spec else None,
                log=log,
            )
    return result


def objective_l1_ic(trial: Trial, ctx: TieredContext) -> float:
    """L1 SWF-K study objective: pooled IC 최대화. Sharpe 미참조(decoupling 보장).

    IS window [l1_start, l2_start)를 n_folds SWF-K fold로 분할하여
    purged sequential walk-forward 검증 수행.

    Args:
        trial: Optuna Trial 객체.
        ctx: Tiered 파이프라인 컨텍스트.

    Returns:
        pooled_ic (gate 미통과 시 -inf).
    """
    import pandas as pd

    from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l1_swf
    from src.domain.futures.strategy.walk_forward import build_l1_swf_folds

    l1_params = suggest_layered_params(trial, "L1")
    purge_bars, embargo_bars = resolve_purge_and_embargo_bars(ctx.cfg)
    n_bars = len(ctx.aligned.datetimes) if hasattr(ctx.aligned, "datetimes") else 0
    if n_bars < 10:
        return float("-inf")

    # LayeredWindow.l1_start → L1 fit 시작 bar index
    # LayeredWindow.l2_start → l1_end bar index (OOS 경계; production OOS는 L3 전용)
    _is_ts = pd.Timestamp(ctx.window.l1_start, tz="UTC")
    _oos_ts = pd.Timestamp(ctx.window.l2_start, tz="UTC")
    _l1_start = int(np.searchsorted(ctx.aligned.datetimes, _is_ts))
    _l1_end = int(np.searchsorted(ctx.aligned.datetimes, _oos_ts))

    folds = build_l1_swf_folds(
        n_bars=n_bars,
        n_folds=5,
        l1_start_bars=_l1_start,
        l1_end_bars=_l1_end,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    result = run_l1_swf(
        labeled_events=ctx.labeled_events,
        aligned=ctx.aligned,
        cfg=ctx.cfg,
        folds=folds,
        l1_params=l1_params,
        tf=ctx.tf,
    )
    # 오직 IC만 반환 — Sharpe 미참조
    return float(result.pooled_ic) if result.gate_passed else float("-inf")


def _resolve_l2_signal_batch_and_folds(ctx: TieredContext) -> tuple[Any, tuple[Any, ...]]:
    import numpy as np

    from src.domain.futures.strategy.tiered_workflow.pipeline import _date_to_idx, _to_utc_timestamp
    from src.domain.futures.strategy.tiered_workflow.signal_selection import predict_layer1_signals
    from src.domain.futures.strategy.walk_forward import WFFold, build_walk_forward_folds

    ho_start_idx_l2 = _date_to_idx(ctx.aligned.datetimes, ctx.window.holdout_start)

    signal_batch = None
    if ctx.fixed_l1_params:
        signal_batch = ctx.fixed_l1_params.get("signal_batch")
    if signal_batch is None:
        artifact = (ctx.fixed_l1_params or {}).get("inference_artifact")
        if artifact is None:
            return None, ()
        awf_folds = ctx.awf_folds or build_walk_forward_folds(n_bars=ho_start_idx_l2, cfg=ctx.cfg)
        if not awf_folds:
            return None, ()
        start_idx = min(fold.oos_start for fold in awf_folds)
        end_idx = max(fold.oos_end for fold in awf_folds)
        signal_batch = predict_layer1_signals(
            artifact=artifact,
            candidate_events=ctx.labeled_events,
            aligned=ctx.aligned,
            start_idx=start_idx,
            end_idx=end_idx,
            cfg=ctx.cfg,
        )
    if signal_batch is None:
        return None, ()

    awf_folds = ctx.awf_folds or build_walk_forward_folds(n_bars=ho_start_idx_l2, cfg=ctx.cfg)

    l2_start_ts = _to_utc_timestamp(ctx.window.l2_start)
    l1_end_bars = int(
        np.searchsorted(
            ctx.aligned.datetimes,
            np.datetime64(l2_start_ts.tz_localize(None), "ns"),
        )
    )

    bounded_folds = tuple(
        fold
        for fold in awf_folds
        if fold.oos_start >= l1_end_bars and fold.oos_end <= ho_start_idx_l2
    )
    if bounded_folds:
        return signal_batch, bounded_folds
    cal_end = max(l1_end_bars - 1, 1)
    return signal_batch, (
        WFFold(
            fit_start=0,
            fit_end=cal_end,
            cal_start=max(0, cal_end - max(1, cal_end // 5)),
            cal_end=cal_end,
            oos_start=l1_end_bars,
            oos_end=ho_start_idx_l2,
        ),
    )


def _shape_efficiency_l2_objective(
    *,
    sortino_hac_unit: float,
    worst_fold_sortino: float,
    worst_fold_threshold: float,
    worst_fold_weight: float,
    downside_dispersion: float,
    lambda_w: float = 0.0,
    risk_util_realized: float = 0.0,
    risk_util_target: float = 0.50,
    risk_util_weight: float = 0.03,
    trade_count: int = 0,
    trade_target: int = 90,
    trade_weight: float = 0.02,
    mean_turnover: float = 0.0,
    turnover_penalty_weight: float = 0.0,
) -> float:
    """Scale-invariant Sortino_HAC_unit 기반 shape 최적화 목적함수.

    J = Sortino_HAC_unit - lambda_w * max(0, tau_wf - worst_fold_sortino)
        - lambda_d * downside_dispersion
        - risk_util_weight * max(0, risk_util_target - risk_util_realized)  [RC-2 soft penalty]
        - trade_weight * max(0, trade_target - trade_count) / trade_target  [RC-2 soft penalty]
        - turnover_penalty_weight * mean_turnover                        [C4: cost-aware]

    scale-invariant 1차항 유지 + soft 2차항으로 배치 가능성 약한 gradient 부여.
    weight ≤ 0.03 유지 → 1차 shape 압도 방지 (quant.md §0 Anti-Overfitting).

    Args:
        sortino_hac_unit: HAC 조정 unit-vol Sortino (1차 목적, scale-invariant).
        worst_fold_sortino: 최악 fold Sortino (단조 패널티 입력).
        worst_fold_threshold: worst-fold 페널티 임계값 (≤ → 패널티).
        worst_fold_weight: worst-fold 페널티 가중치 λ_w.
        downside_dispersion: 하방 분산 λ_d·dispersion 항.
        lambda_w: 미사용 호환 파라미터 (worst_fold_weight 우선).
        risk_util_realized: 실현 리스크 활용도 (MDD/MDD_cap). RC-2 dead param 활성화.
        risk_util_target: 리스크 활용 목표 (기본 0.50). soft 패널티 기준.
        risk_util_weight: risk_util soft 패널티 가중치 (≤ 0.03 유지).
        trade_count: 실현 거래 횟수. scale 정합 soft 패널티 입력.
        trade_target: 목표 거래 횟수 (기본 90). soft 패널티 기준.
        trade_weight: trade_count soft 패널티 가중치 (≤ 0.02 유지).
        mean_turnover: 평균 리밸런싱 turnover 비율 (C4 turnover penalty 입력).
        turnover_penalty_weight: turnover 페널티 가중치 λ_t (0=off, 기본 0.0).

    Returns:
        float: 목적함수 값. 비정상 입력 시 -1e6 fail-fast 반환.

    Time Complexity: O(1). Space Complexity: O(1).
    """
    if not np.isfinite(sortino_hac_unit):
        return -1e6
    worst_fold_penalty = (
        max(0.0, float(worst_fold_threshold) - float(worst_fold_sortino))
        * float(worst_fold_weight)
    )
    risk_util_penalty = float(risk_util_weight) * max(
        0.0, float(risk_util_target) - float(risk_util_realized)
    )
    trade_penalty = float(trade_weight) * max(
        0.0, (float(trade_target) - float(trade_count)) / max(float(trade_target), 1.0)
    )
    turnover_penalty = float(turnover_penalty_weight) * float(mean_turnover)
    return float(
        sortino_hac_unit
        - worst_fold_penalty
        - float(downside_dispersion)
        - risk_util_penalty
        - trade_penalty
        - turnover_penalty
    )


def _deployment_shaped_l2_objective(
    *,
    growth_lcb: float,
    block_log_growth: Sequence[float],
    worst_fold_sharpe: float,
    worst_fold_threshold: float,
    worst_fold_weight: float,
) -> float:
    """Deprecated: growth_lcb 기반 목적함수 (RC-2로 인해 _shape_efficiency_l2_objective로 대체).

    backward-compat 유지용 shim. 직접 호출 금지.
    """
    if not np.isfinite(growth_lcb):
        return -1e6
    arr = np.asarray(list(block_log_growth), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        arr = np.array([0.0], dtype=np.float64)
    downside_lpm = float(np.mean(np.maximum(0.0, -arr)))
    median = float(np.median(arr))
    mad = float(np.mean(np.abs(arr - median)))
    worst_fold_penalty = (
        max(0.0, float(worst_fold_threshold) - float(worst_fold_sharpe))
        * float(worst_fold_weight)
    )
    return float(growth_lcb - 0.10 * downside_lpm - 0.05 * mad - worst_fold_penalty)


def evaluate_l2_trial(
    *,
    cache: L2SimulationCache,
    signal_batch: Any,
    aligned: Any,
    awf_folds: tuple[Any, ...],
    config: Any,
    caps: Any,
    tf: str,
    deploy_leverage_override: float | None = None,
    eval_tag: str = "unspecified",
) -> Any:
    from src.domain.futures.portfolio.signal_composer import hours_per_bar_tf
    from src.domain.futures.strategy.tiered_workflow.awf_sim import _run_awf_simulation
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2BlockMetric,
        Layer2TrialEvaluation,
    )
    from src.domain.futures.strategy.tiered_workflow.l2_gate import evaluate_layer2_gate
    from src.domain.futures.strategy.tiered_workflow.metrics import (
        _bars_per_year_for_tf,
        _cagr,
        _contiguous_block_log_growth,
        _growth_lower_confidence_bound,
        _mdd,
        _psr,
        _sharpe,
        _sharpe_hac,
        _sortino,
        _sortino_hac_unit,
    )
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
        apply_deployment,
        calibrate_deployment_leverage,
        compute_layer2_fold_diagnostics,
    )

    sim = _run_awf_simulation(
        cache=cache,
        signal_batch=signal_batch,
        aligned=aligned,
        awf_folds=awf_folds,
        config=config,
        caps=caps,
        tf=tf,
        sim_origin="champion_eval",
    )
    hours_per_bar = max(float(hours_per_bar_tf(tf)), 1e-9)
    bars_per_year = _bars_per_year_for_tf(tf)
    block_size = max(1, round((24.0 * 30.0) / hours_per_bar))
    rets_hybrid = list(sim.rets_hybrid)
    rets_baseline = list(sim.rets_baseline)

    cagr_baseline = _cagr(rets_baseline, bars_per_year=bars_per_year)
    sharpe_hybrid = _sharpe(rets_hybrid, bars_per_year=bars_per_year)
    sharpe_hac_hybrid = _sharpe_hac(rets_hybrid, bars_per_year=bars_per_year)
    sharpe_hac_baseline = _sharpe_hac(rets_baseline, bars_per_year=bars_per_year)
    psr_hybrid = _psr(rets_hybrid, bars_per_year=bars_per_year)
    sortino_hybrid = _sortino(rets_hybrid, bars_per_year=bars_per_year)
    trade_count = int(sim.trade_count)

    # C2: L* 산정 — fit-leg book 수익률 우선, 없으면 OOS proxy fallback.
    # cagr/mdd/cvar_hybrid는 deployed 값으로 교체. sortino/sharpe/calmar는 unit-vol 유지.
    _deploy_enabled = bool(getattr(config, "l2_deploy_enabled", True))
    _fit_rets_raw = sim.fit_rets_hybrid
    _l_star: float = 1.0
    _l_binding: str = "none"
    if deploy_leverage_override is not None and deploy_leverage_override > 1.0:
        _l_star = float(deploy_leverage_override)
        _l_binding = "champion"
        _logger.debug("[L2-EVAL] L*=%.3f (binding=%s, override=champion)", _l_star, _l_binding)
    elif _deploy_enabled:
        if _fit_rets_raw:
            _calib_rets = np.asarray(_fit_rets_raw, dtype=np.float64)
            _calib_src = "fit_leg"
        else:
            # fallback: OOS proxy (mdd_margin 완충 + hard_cap=20 → look-ahead 실질 영향 최소)
            _calib_rets = np.asarray(rets_hybrid, dtype=np.float64)
            _calib_src = "oos_proxy"
        if _calib_rets.size >= 2:
            _l_star, _l_binding, _l_cross_mdd = calibrate_deployment_leverage(
                fit_rets=_calib_rets,
                oos_rets=np.asarray(rets_hybrid, dtype=np.float64) if _calib_src == "fit_leg" else None,
                mdd_cap=float(config.l2_max_mdd_abs),
                cvar_cap=float(config.l2_max_cvar_95),
                mdd_margin=float(config.l2_deploy_mdd_margin),
                cvar_margin=float(config.l2_deploy_cvar_margin),
                l_hard_cap=float(config.l2_deploy_l_hard_cap),
                exchange_leverage_cap=getattr(config, "l2_max_exchange_leverage", None),
            )
            _logger.debug(
                "[L2-EVAL] L*=%.3f (binding=%s, src=%s)",
                _l_star, _l_binding, _calib_src,
            )
            if _l_cross_mdd > 0.0:
                _oos_risk = _l_cross_mdd / max(float(config.l2_max_mdd_abs), 1e-9)
                _logger.debug(
                    "[L2-OOS-CAP] OOS_RiskUtil=%.3f cap=%.2f (L*=%.3f)",
                    _oos_risk, config.l2_max_mdd_abs, _l_star,
                )
                if _oos_risk > 1.0:
                    _logger.debug(
                        "[L2-OOS-CAP] L* over-deployed: OOS_RiskUtil=%.3f > 1.0 (L*=%.3f)",
                        _oos_risk, _l_star,
                    )

    _dep = apply_deployment(
        rets=np.asarray(rets_hybrid, dtype=np.float64),
        leverage=_l_star,
        bars_per_year=bars_per_year,
    )
    cagr_hybrid = _dep.cagr
    mdd_hybrid = _dep.mdd
    cvar_95_hybrid = _dep.cvar_95
    mar_hybrid = cagr_hybrid / (mdd_hybrid + 1e-9)

    # Phase A 진단: tag별 parity 지문 (구조 원자값 → divergence 귀속용, DEBUG-only).
    if _logger.isEnabledFor(logging.DEBUG):
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            _content_hash_dataclass as _fp_content_hash,
        )

        _fp_rets = np.asarray(rets_hybrid, dtype=np.float64)
        _fp_oos_cagr_unit = (
            _cagr(list(rets_hybrid), bars_per_year=bars_per_year) if _fp_rets.size >= 2 else 0.0
        )
        _fp_sum_log1p = (
            float(np.sum(np.log1p(np.clip(_l_star * _fp_rets, -1.0 + 1e-9, None))))
            if _fp_rets.size >= 1
            else 0.0
        )
        _fp_cfg_ch = _fp_content_hash(config)
        _logger.debug(
            "[L2-PARITY-FP] tag=%s cfg_ch=%s n_rets=%d l_star=%.6f binding=%s "
            "oos_cagr_unit=%.6f sum_log1p_scaled=%.8f cagr_dep=%.6f mdd_dep=%.6f override=%s",
            eval_tag,
            _fp_cfg_ch,
            int(_fp_rets.size),
            float(_l_star),
            _l_binding,
            float(_fp_oos_cagr_unit),
            _fp_sum_log1p,
            float(cagr_hybrid),
            float(mdd_hybrid),
            "none" if deploy_leverage_override is None else f"{deploy_leverage_override:.6f}",
        )

    # 진단: fit-rets vs OOS-rets 분포 이격 (L* inflation 감지)
    if _deploy_enabled and _fit_rets_raw and rets_hybrid:
        _diag_fit_list = list(_fit_rets_raw)
        _diag_oos_list = list(rets_hybrid)
        if len(_diag_fit_list) >= 2 and len(_diag_oos_list) >= 2:
            _diag_fit_cagr = _cagr(_diag_fit_list, bars_per_year=bars_per_year)
            _diag_fit_mdd = _mdd(_diag_fit_list)
            _diag_oos_cagr = _cagr(_diag_oos_list, bars_per_year=bars_per_year)
            _diag_oos_mdd = _mdd(_diag_oos_list)
            _logger.debug(
                "[L2-TRIAL-DIAG] fit_CAGR_vol1=%.4f fit_MDD_vol1=%.4f | "
                "OOS_CAGR_vol1=%.4f OOS_MDD_vol1=%.4f | "
                "L*=%.3f(%s) | deployed_CAGR=%.4f deployed_MDD=%.4f",
                _diag_fit_cagr, _diag_fit_mdd,
                _diag_oos_cagr, _diag_oos_mdd,
                _l_star, _l_binding, cagr_hybrid, mdd_hybrid,
            )

    block_growth_hybrid = _contiguous_block_log_growth(
        rets_hybrid,
        block_bars=block_size,
    )
    block_growth_baseline = _contiguous_block_log_growth(
        rets_baseline,
        block_bars=block_size,
    )
    blocks_per_year = bars_per_year / float(block_size)
    growth_lcb_hybrid = _growth_lower_confidence_bound(
        block_growth_hybrid,
        blocks_per_year=blocks_per_year,
        z_value=float(config.l2_growth_lcb_z),
    )
    growth_lcb_baseline = _growth_lower_confidence_bound(
        block_growth_baseline,
        blocks_per_year=blocks_per_year,
        z_value=float(config.l2_growth_lcb_z),
    )

    block_metrics: list[Layer2BlockMetric] = []
    n_blocks = max(len(block_growth_hybrid), len(block_growth_baseline))
    for block_idx in range(n_blocks):
        start_idx = block_idx * block_size
        end_idx = min((block_idx + 1) * block_size, len(rets_hybrid))
        turnover_slice = sim.all_turnovers[block_idx:block_idx + 1]
        block_metrics.append(
            Layer2BlockMetric(
                start_idx=start_idx,
                end_idx=end_idx,
                log_growth_hybrid=(
                    float(block_growth_hybrid[block_idx])
                    if block_idx < len(block_growth_hybrid)
                    else 0.0
                ),
                log_growth_baseline=(
                    float(block_growth_baseline[block_idx])
                    if block_idx < len(block_growth_baseline)
                    else 0.0
                ),
                mdd_hybrid=_mdd(rets_hybrid[start_idx:end_idx]),
                turnover_hybrid=float(np.mean(turnover_slice)) if turnover_slice else 0.0,
                active_rebalances=int(sum(1 for value in turnover_slice if abs(value) > 0.0)),
            )
        )

    fold_diag = compute_layer2_fold_diagnostics(
        fold_rets_hybrid=sim.fold_rets_hybrid,
        fold_selected_symbols=sim.fold_selected_symbols,
        leverage=float(_l_star),
        bars_per_year=bars_per_year,
    )
    fold_pass_ratio = float(fold_diag.fold_pass_ratio)
    finite_fold_cagrs = [
        float(value)
        for value in fold_diag.fold_deployed_cagrs
        if value is not None and np.isfinite(float(value))
    ]
    worst_fold_cagr = min(finite_fold_cagrs) if finite_fold_cagrs else 0.0
    break_even_pass_pct = (
        float(sim.friction_pass_total) / float(sim.signal_total)
        if sim.signal_total > 0
        else 0.0
    )
    average_gross_exposure = (
        float(np.mean(sim.all_gross_exposures)) if sim.all_gross_exposures else 0.0
    )
    total_cost_bps = float(sim.total_cost_hybrid * 1e4)
    # FIX-1: active_block_count 정의 통일 — 최종 게이트(pipeline.py)와 동일한 AWF fold 기반.
    # contiguous-block 기반(이전)과 달리 pipeline 게이트가 보는 len(block_metrics)와 일치.
    active_block_count = len([m for m in block_metrics if m.active_rebalances > 0])
    # FIX-2/4: 순수 1/N EW baseline Sharpe (uplift 제약 전용)
    sharpe_hac_baseline_ew = _sharpe_hac(list(sim.rets_baseline_ew), bars_per_year=bars_per_year)
    cap_saturation_ratio = (
        float(sim.cap_saturation_count) / float(sim.rebalance_count)
        if sim.rebalance_count > 0
        else 0.0
    )
    positive_block_delta_ratio = (
        float(
            sum(
                1
                for metric in block_metrics
                if float(metric.log_growth_hybrid) > float(metric.log_growth_baseline)
            )
        )
        / float(len(block_metrics))
        if block_metrics
        else 0.0
    )
    reliability_values = [
        float(getattr(policy, "reliability", policy.confidence))
        for fold_policy in getattr(cache, "regime_policy_by_fold", ())
        for policy in fold_policy.values()
        if np.isfinite(float(getattr(policy, "reliability", policy.confidence)))
    ]
    bucket_reliability_mean = float(np.mean(reliability_values)) if reliability_values else 0.0
    entry_audit = build_layer_universe_audit(
        aligned=aligned,
        layer="L2",
        start_idx=int(signal_batch.start_idx),
        end_idx=int(signal_batch.end_idx),
    )
    entry_spike_penalty = (
        float(config.l2_entry_spike_penalty_weight)
        if "entry_block_spike" in entry_audit.warnings
        else 0.0
    )

    finite_score = float(growth_lcb_hybrid) if np.isfinite(growth_lcb_hybrid) else -1e6
    risk_utilization = float(mdd_hybrid) / max(float(config.l2_max_mdd_abs), 1e-9)

    # STEP 5: worst-fold Sortino soft penalty (비정상성 방어, D1 shape 목적 정합).
    # fold_rets_hybrid: list of per-fold OOS return sequences.
    # Time Complexity: O(F·T) where F=n_folds, T=fold OOS bars.
    _fold_sortinos: list[float] = [
        float(_sortino(list(fr), bars_per_year=bars_per_year)) if fr else 0.0
        for fr in sim.fold_rets_hybrid
    ]
    _fold_sharpes: list[float] = [
        float(_sharpe_hac(list(fr), bars_per_year=bars_per_year)) if fr else 0.0
        for fr in sim.fold_rets_hybrid
    ]
    worst_fold_sortino: float = min(_fold_sortinos) if _fold_sortinos else 0.0
    worst_fold_sharpe: float = min(_fold_sharpes) if _fold_sharpes else 0.0
    _wf_threshold = float(config.l2_worst_fold_penalty_threshold)
    _wf_weight = float(config.l2_worst_fold_penalty_weight)

    # D1: Sortino_HAC_unit 기반 scale-invariant shape 목적함수
    sortino_hac_unit = _sortino_hac_unit(rets_hybrid, bars_per_year=bars_per_year)
    # downside_dispersion: block-level 하방 변동성 (안정성 패널티)
    _block_arr = np.asarray(list(block_growth_hybrid), dtype=np.float64)
    _block_arr = _block_arr[np.isfinite(_block_arr)]
    _block_downside = _block_arr[_block_arr < 0.0]
    downside_dispersion = float(np.std(_block_downside, ddof=1)) * 0.05 if _block_downside.size > 1 else 0.0

    objective_value = _shape_efficiency_l2_objective(
        sortino_hac_unit=sortino_hac_unit,
        worst_fold_sortino=worst_fold_sortino,
        worst_fold_threshold=_wf_threshold,
        worst_fold_weight=_wf_weight,
        downside_dispersion=downside_dispersion,
        risk_util_realized=float(risk_utilization),
        risk_util_target=float(config.l2_objective_risk_util_target),
        risk_util_weight=float(config.l2_objective_risk_util_weight),
        trade_count=int(trade_count),
        trade_target=int(config.l2_objective_trade_target),
        trade_weight=float(config.l2_objective_trade_weight),
        mean_turnover=float(np.mean(sim.all_turnovers)) if sim.all_turnovers else 0.0,
        turnover_penalty_weight=float(config.l2_turnover_penalty_weight),
    )
    # growth_lcb는 diagnostic으로 강등 — objective에서 제외 (RC-2 해소)
    deployment_objective_bonus = float(objective_value - finite_score)
    deployment_failed = (
        sim.signal_total <= 0
        or sim.support_leak_count > 0
        or not np.isfinite(cagr_hybrid)
        or not np.isfinite(sharpe_hac_hybrid)
    )
    gate = evaluate_layer2_gate(
        deployment_failed=deployment_failed,
        support_leak_count=int(sim.support_leak_count),
        cagr_hybrid=float(cagr_hybrid),
        sharpe_hybrid=float(sharpe_hybrid),
        sharpe_hac_hybrid=float(sharpe_hac_hybrid),
        sharpe_hac_baseline=float(sharpe_hac_baseline_ew),
        sortino_hybrid=float(sortino_hybrid),
        mar_hybrid=float(mar_hybrid),
        mdd_hybrid=float(mdd_hybrid),
        cvar_95_hybrid=float(cvar_95_hybrid),
        fold_pass_ratio=float(fold_pass_ratio),
        active_block_count=int(active_block_count),
        friction_pass_pct=float(break_even_pass_pct),
        trade_count=int(trade_count),
        growth_lcb_hybrid=float(growth_lcb_hybrid),
        growth_lcb_baseline=float(growth_lcb_baseline),
        dsr_hybrid=None,
        psr_hybrid=float(psr_hybrid),
        recent_fold_passed=fold_diag.recent_fold_passed,
        recent_fold_sharpe=fold_diag.recent_fold_sharpe,
        worst_fold_cagr=float(worst_fold_cagr),
        positive_block_delta_ratio=float(positive_block_delta_ratio),
        fold_attributions=sim.fold_attributions,
        config=config,
    )
    deployable_score = build_layer2_deployable_score(
        cagr=float(cagr_hybrid),
        sortino=float(sortino_hybrid),
        sharpe=float(sharpe_hybrid),
        mdd=float(mdd_hybrid),
        fold_pass_ratio=float(fold_pass_ratio),
        worst_fold_cagr=float(worst_fold_cagr),
        positive_block_delta_ratio=float(positive_block_delta_ratio),
        total_cost_bps=float(total_cost_bps),
        bucket_reliability_mean=float(bucket_reliability_mean),
        entry_spike_penalty=float(entry_spike_penalty),
        config=config,
    )
    # deployment extras raw data (SSOT 위임용)
    _last_selected: frozenset[str] = getattr(sim, 'last_selected', frozenset())
    _symbols = getattr(aligned, 'symbols', ())
    _sym_to_idx = {s: i for i, s in enumerate(_symbols)}
    _last_weights = tuple(
        float(getattr(sim, 'last_w', np.array([]))[_sym_to_idx[s]])
        for s in _last_selected if s in _sym_to_idx
    )
    _last_selected_tuple = tuple(sorted(_last_selected))

    return Layer2TrialEvaluation(
        objective_value=float(objective_value),
        constraint_values=gate.optuna_constraint_values,
        cagr_hybrid=float(cagr_hybrid),
        cagr_baseline=float(cagr_baseline),
        growth_lcb_hybrid=float(growth_lcb_hybrid),
        growth_lcb_baseline=float(growth_lcb_baseline),
        sharpe_hac_hybrid=float(sharpe_hac_hybrid),
        sharpe_hac_baseline=float(sharpe_hac_baseline),
        psr_hybrid=float(psr_hybrid),
        mdd_hybrid=float(mdd_hybrid),
        cvar_95_hybrid=float(cvar_95_hybrid),
        fold_pass_ratio=float(fold_pass_ratio),
        break_even_pass_pct=float(break_even_pass_pct),
        average_gross_exposure=float(average_gross_exposure),
        cap_saturation_ratio=float(cap_saturation_ratio),
        total_cost_bps=float(total_cost_bps),
        block_metrics=tuple(block_metrics),
        returns_hybrid=tuple(rets_hybrid),
        returns_baseline=tuple(rets_baseline),
        sharpe_hybrid=float(sharpe_hybrid),
        sharpe_hac_baseline_ew=float(sharpe_hac_baseline_ew),
        sortino_hybrid=float(sortino_hybrid),
        trade_count=trade_count,
        risk_utilization=float(risk_utilization),
        deployment_objective_bonus=float(deployment_objective_bonus),
        worst_fold_sharpe=worst_fold_sharpe,
        gate=gate,
        fit_returns_hybrid=tuple(sim.fit_rets_hybrid),
        deploy_leverage=float(_l_star),
        deploy_binding=str(_l_binding),
        recent_fold_passed=fold_diag.recent_fold_passed,
        recent_fold_sharpe=float(fold_diag.recent_fold_sharpe or 0.0),
        recent_fold_cagr=float(fold_diag.recent_fold_cagr),
        recent_fold_mdd=float(fold_diag.recent_fold_mdd),
        latest_to_median_cagr=float(fold_diag.latest_to_median_cagr),
        fold_deployed_cagrs=tuple(fold_diag.fold_deployed_cagrs),
        fold_deployed_mdds=tuple(getattr(fold_diag, 'fold_deployed_mdds', ())),
        fold_selected_symbols=tuple(fold_diag.fold_selected_symbols),
        worst_fold_cagr=float(worst_fold_cagr),
        positive_block_delta_ratio=float(positive_block_delta_ratio),
        bucket_reliability_mean=float(bucket_reliability_mean),
        entry_spike_penalty=float(entry_spike_penalty),
        # deployment extras raw data
        last_selected_symbols=_last_selected_tuple,
        last_weights=_last_weights,
        all_turnovers=tuple(sim.all_turnovers),
        rebalance_count=int(sim.rebalance_count),
        all_net_exposures=tuple(sim.all_net_exposures),
        rets_baseline_ew=tuple(sim.rets_baseline_ew),
        fold_attributions=tuple(getattr(sim, 'fold_attributions', ())),
        deployable_score=deployable_score,
    )


def _build_l2_user_attrs(evaluation: Any) -> dict[str, Any]:
    user_attrs: dict[str, Any] = {}
    user_attrs["l2_objective_value"] = float(evaluation.objective_value)
    user_attrs["cagr_hybrid"] = float(evaluation.cagr_hybrid)
    user_attrs["cagr_baseline"] = float(evaluation.cagr_baseline)
    user_attrs["growth_lcb_hybrid"] = float(evaluation.growth_lcb_hybrid)
    user_attrs["growth_lcb_baseline"] = float(evaluation.growth_lcb_baseline)
    user_attrs["sharpe_hac_hybrid"] = float(evaluation.sharpe_hac_hybrid)
    user_attrs["sharpe_hac_baseline"] = float(evaluation.sharpe_hac_baseline)
    user_attrs["psr_hybrid"] = float(evaluation.psr_hybrid)
    user_attrs["mdd_hybrid"] = float(evaluation.mdd_hybrid)
    user_attrs["cvar_95_hybrid"] = float(evaluation.cvar_95_hybrid)
    user_attrs["fold_pass_ratio"] = float(evaluation.fold_pass_ratio)
    user_attrs["break_even_pass_pct"] = float(evaluation.break_even_pass_pct)
    user_attrs["average_gross_exposure"] = float(evaluation.average_gross_exposure)
    user_attrs["cap_saturation_ratio"] = float(evaluation.cap_saturation_ratio)
    user_attrs["total_cost_bps"] = float(evaluation.total_cost_bps)
    user_attrs["sortino_hybrid"] = float(getattr(evaluation, "sortino_hybrid", 0.0))
    user_attrs["risk_utilization"] = float(getattr(evaluation, "risk_utilization", 0.0))
    user_attrs["recent_fold_sharpe"] = float(getattr(evaluation, "recent_fold_sharpe", 0.0))
    user_attrs["recent_fold_cagr"] = float(getattr(evaluation, "recent_fold_cagr", 0.0))
    user_attrs["latest_to_median_cagr"] = float(getattr(evaluation, "latest_to_median_cagr", 0.0))
    user_attrs["deploy_leverage"] = float(getattr(evaluation, "deploy_leverage", 1.0))
    user_attrs["deployment_objective_bonus"] = float(getattr(evaluation, "deployment_objective_bonus", 0.0))
    user_attrs["worst_fold_sharpe"] = float(getattr(evaluation, "worst_fold_sharpe", 0.0))
    user_attrs["trade_count"] = int(getattr(evaluation, "trade_count", 0))
    user_attrs["recent_fold_passed"] = getattr(evaluation, "recent_fold_passed", None)
    user_attrs["deploy_binding"] = str(getattr(evaluation, "deploy_binding", ""))
    gate = getattr(evaluation, "gate", None)
    user_attrs["l2_constraint_values"] = list(evaluation.constraint_values)
    user_attrs["l2_optuna_constraint_values"] = list(evaluation.constraint_values)
    if gate is not None:
        user_attrs["l2_promotion_constraint_values"] = list(gate.promotion_constraint_values)
        user_attrs["l2_promotion_passed"] = bool(gate.promotion_passed)
        user_attrs["l2_promotion_blocker"] = gate.promotion_blocker
    user_attrs["l2_block_log_growth_signature"] = [metric.log_growth_hybrid for metric in evaluation.block_metrics]
    return user_attrs


def evaluate_l2_trial_cached(
    *,
    cache: Any,
    signal_batch: Any,
    aligned: Any,
    awf_folds: tuple[Any, ...],
    config: Any,
    caps: Any,
    tf: str,
    deploy_leverage_override: float | None = None,
    eval_tag: str = "unspecified",
    _memo: dict[tuple[Any, ...], Any],
) -> Any:
    from src.domain.futures.strategy.tiered_workflow.awf_sim import _content_hash_dataclass

    cfg_ch = _content_hash_dataclass(config)
    # memo 키는 eval_tag 제외 (tag는 진단용 라벨, 결과 불변 → 캐시 무력화 방지).
    key = (id(cache), cfg_ch, id(signal_batch), id(caps), tf, deploy_leverage_override)
    cached = _memo.get(key)
    if cached is not None:
        return cached
    result = evaluate_l2_trial(
        cache=cache,
        signal_batch=signal_batch,
        aligned=aligned,
        awf_folds=awf_folds,
        config=config,
        caps=caps,
        tf=tf,
        deploy_leverage_override=deploy_leverage_override,
        eval_tag=eval_tag,
    )
    _memo[key] = result
    return result


def _evaluate_l2_params(
    l2_params: dict[str, Any],
    ctx: TieredContext,
) -> tuple[float, dict[str, Any], float]:
    """L2 parameter evaluation logic extracted for parallel optimization.

    Returns:
        A tuple of (objective_value, user_attributes_dict, elapsed_seconds).
    """
    import time

    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

    if not ctx.fixed_l1_params:
        return -1e6, {}, 0.0

    signal_batch, awf_folds = _resolve_l2_signal_batch_and_folds(ctx)
    if signal_batch is None or not awf_folds:
        return -1e6, {}, 0.0

    # Ensure cache is built if not already present
    if getattr(ctx, "l2_sim_cache", None) is None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache
        object.__setattr__(ctx, "l2_sim_cache", build_l2_simulation_cache(ctx.aligned, signal_batch, ctx.tf))

    cache = ctx.l2_sim_cache
    assert cache is not None

    t_start = time.perf_counter()
    evaluation = evaluate_l2_trial(
        cache=cache,
        signal_batch=signal_batch,
        aligned=ctx.aligned,
        awf_folds=awf_folds,
        config=Layer2AllocationConfig.from_mapping(l2_params),
        caps=ctx.caps,
        tf=ctx.tf,
    )
    t_elapsed = time.perf_counter() - t_start

    user_attrs = _build_l2_user_attrs(evaluation)
    return float(evaluation.objective_value), user_attrs, t_elapsed


def _evaluate_l2_params_threadsafe(
    l2_params: dict[str, Any],
    ctx: TieredContext,
) -> tuple[float, dict[str, Any], float]:
    """Thread-safe variant: ctx.l2_sim_cache is assumed pre-built (no __setattr__ mutation)."""
    import time

    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

    if not ctx.fixed_l1_params:
        return -1e6, {}, 0.0
    assert ctx.l2_sim_cache is not None, "l2_sim_cache must be pre-built for thread-safe evaluation"

    signal_batch, awf_folds = _resolve_l2_signal_batch_and_folds(ctx)
    if signal_batch is None or not awf_folds:
        return -1e6, {}, 0.0

    t_start = time.perf_counter()
    evaluation = evaluate_l2_trial(
        cache=ctx.l2_sim_cache,
        signal_batch=signal_batch,
        aligned=ctx.aligned,
        awf_folds=awf_folds,
        config=Layer2AllocationConfig.from_mapping(l2_params),
        caps=ctx.caps,
        tf=ctx.tf,
    )
    t_elapsed = time.perf_counter() - t_start

    user_attrs = _build_l2_user_attrs(evaluation)
    return float(evaluation.objective_value), user_attrs, t_elapsed


def objective_l2_growth(trial: Trial, ctx: TieredContext) -> float:
    """L2 AWF study objective: 보수적 log-growth LCB 최대화."""
    l2_params = suggest_layered_params(trial, "L2", fixed=ctx.fixed_l1_params or {})
    value, attrs, t_elapsed = _evaluate_l2_params(l2_params, ctx)

    for k, v in attrs.items():
        trial.set_user_attr(k, v)

    _logger.log(
        logging.DEBUG,
        "[perf-optuna] Trial %d evaluate_l2_trial took %.4fs | Objective: %.6f",
        trial.number,
        t_elapsed,
        value,
    )
    return value


def layer2_constraints_from_trial(trial: FrozenTrial) -> tuple[float, ...]:
    raw = trial.user_attrs.get("l2_optuna_constraint_values")
    if not isinstance(raw, (list, tuple)):
        raw = trial.user_attrs.get("l2_constraint_values")
    if not isinstance(raw, (list, tuple)):
        return (1.0,) * 9
    resolved: list[float] = []
    for item in raw:
        try:
            resolved.append(float(item))
        except Exception:
            resolved.append(1.0)
    while len(resolved) < 9:
        resolved.append(1.0)
    return tuple(resolved)


def objective_l2_sharpe(trial: Trial, ctx: TieredContext) -> float:
    """Deprecated wrapper. 유지 기간 동안 growth objective로 위임.

    Args:
        trial: Optuna Trial 객체.
        ctx: Tiered 파이프라인 컨텍스트 (fixed_l1_params에 oos_stacked 포함 필수).

    Returns:
        growth_lcb objective value.
    """
    warnings.warn(
        "objective_l2_sharpe is deprecated; use objective_l2_growth instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return float(objective_l2_growth(trial, ctx))


# Break circularity by importing run_optimization_loop at the very end
from src.domain.futures.optimization.observability.run_tracker import run_optimization_loop  # noqa: E402
