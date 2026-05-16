from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import optuna

from src.domain.futures.optimization.optimizer import MLPhaseDContext, objective_ml_phase_d
from src.domain.futures.optimization.phase_metrics import lcb, summarize, ucb


def _metric(trial: optuna.Trial, *keys: str, default: float = 0.0) -> float:
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


def _set_float_attr(trial: optuna.Trial, key: str, value: float) -> None:
    try:
        trial.set_user_attr(key, float(value))
    except Exception:
        trial.set_user_attr(key, value)


def _metric_vector(trial: optuna.Trial, *keys: str) -> list[float]:
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
    trial: optuna.Trial,
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
            raise optuna.TrialPruned()


def _persist_constraint_attrs(
    trial: optuna.Trial,
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


def objective_phase_a1_signal_lcb(trial: optuna.Trial, ctx: MLPhaseDContext) -> float:
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
    trial: optuna.Trial,
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


def objective_phase_b_calmar_lcb(trial: optuna.Trial, ctx: MLPhaseDContext) -> float:
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
