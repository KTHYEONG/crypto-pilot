from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optuna


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if out == out else None


def build_metrics_snapshot(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    *,
    keys: tuple[str, ...] = (
        "awf_robust_score",
        "awf_mu_log",
        "awf_worst_leg_log_tw",
        "awf_worst_mdd_pct",
        "awf_pos_frac",
        "gate1_dsr",
        "avg_trades",
        "n_trades",
        "calmar_lcb",
        "sortino_lcb",
        "mdd_ucb",
    ),
) -> dict[str, float]:
    ua = getattr(trial, "user_attrs", {}) or {}
    snap: dict[str, float] = {}
    for key in keys:
        val = _safe_float(ua.get(key))
        if val is not None:
            snap[key] = val
    return snap


def set_trial_event_attrs(
    trial: optuna.Trial,
    *,
    status: str,
    reason: str,
    stage: str,
    step: int | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    trial.set_user_attr("obs_status", str(status))
    trial.set_user_attr("obs_reason", str(reason))
    trial.set_user_attr("obs_stage", str(stage))
    if step is not None:
        trial.set_user_attr("obs_step", int(step))
    snap = dict(build_metrics_snapshot(trial))
    if metrics:
        for key, value in metrics.items():
            num = _safe_float(value)
            if num is not None:
                snap[str(key)] = num
    trial.set_user_attr("obs_metrics", snap)


def classify_no_valid_candidates(
    *,
    selection_summary: Mapping[str, Any] | None,
    completed_trials: list[optuna.trial.FrozenTrial],
) -> str:
    if not completed_trials:
        return "no_completed_trials"
    if selection_summary is None:
        return "unknown"
    reject = selection_summary.get("selection_reject_reason_count")
    if isinstance(reject, Mapping) and reject:
        return "gate_reject_all"

    nan_like = 0
    constraint_violation = 0
    zero_alpha = 0
    for trial in completed_trials:
        ua = trial.user_attrs or {}
        value = getattr(trial, "value", None)
        if value is None:
            nan_like += 1
        elif _safe_float(value) is None:
            nan_like += 1
        obs_reason = str(ua.get("obs_reason", ""))
        if obs_reason.startswith("constraint_"):
            constraint_violation += 1
        n_trades = _safe_float(ua.get("avg_trades"))
        awf_mu = _safe_float(ua.get("awf_mu_log"))
        if (n_trades is not None and n_trades <= 0.0) or (awf_mu is not None and awf_mu <= -9.0):
            zero_alpha += 1

    if nan_like == len(completed_trials):
        return "nan_metrics"
    if constraint_violation > 0:
        return "constraint_violation"
    if zero_alpha == len(completed_trials):
        return "zero_alpha_components"
    return "unknown"


def build_compact_trial_summary(
    trial: optuna.trial.FrozenTrial,
    *,
    elapsed_sec: float | None = None,
) -> str:
    ua = trial.user_attrs or {}
    seed = ua.get("seed", ua.get("run_seed", "-"))
    reason = ua.get("obs_reason", "-")
    stage = ua.get("obs_stage", ua.get("phase", "-"))
    status = str(trial.state.name).lower()
    elapsed = elapsed_sec if elapsed_sec is not None else 0.0
    key_params = ("HMM_CRISIS_THRESHOLD", "KELLY_FRACTION", "STOP_LOSS_ATR_MULT")
    compact_params = [f"{key}={trial.params[key]}" for key in key_params if key in trial.params]
    params_text = ",".join(compact_params) if compact_params else "-"
    return (
        f"[TRIAL] n={trial.number} status={status} seed={seed} phase={stage} "
        f"reason={reason} t={elapsed:.2f}s params={params_text}"
    )


def trial_elapsed_seconds(trial: optuna.trial.FrozenTrial) -> float:
    if trial.datetime_start is None or trial.datetime_complete is None:
        return 0.0
    return max((trial.datetime_complete - trial.datetime_start).total_seconds(), 0.0)


def init_failure_reason_counts() -> dict[str, int]:
    return {}


def increment_failure_reason_count(
    counts: dict[str, int],
    reason: str,
) -> None:
    counts[reason] = int(counts.get(reason, 0)) + 1
