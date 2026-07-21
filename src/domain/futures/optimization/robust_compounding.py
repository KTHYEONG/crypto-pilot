from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class RobustnessWindow:
    label: str
    fit_start: int
    fit_end: int
    cal_start: int
    cal_end: int
    oos_start: int
    oos_end: int
    embargo_bars: int


@dataclass(slots=True, frozen=True)
class WindowCompoundingMetrics:
    label: str
    cagr: float
    mdd: float
    cvar_95: float
    growth_lcb: float
    annualized_cost_drag: float
    trade_count: int


@dataclass(slots=True, frozen=True)
class FoldLeverageDecision:
    label: str
    requested_leverage: float
    projected_leverage: float
    applied_leverage: float
    binding_reason: str
    calibration_mdd: float
    calibration_cagr: float


@dataclass(slots=True, frozen=True)
class Layer2CandidateArtifact:
    candidate_hash: str
    params: Mapping[str, Any]
    data_fingerprint: str
    handoff_fingerprint: str
    routing_hash: str
    window_plan_hash: str
    window_metrics: tuple[WindowCompoundingMetrics, ...]
    leverage_schedule: tuple[FoldLeverageDecision, ...]
    robust_score: float
    median_growth_lcb: float
    q10_growth_lcb: float
    positive_window_ratio: float
    worst_window_cagr: float
    hard_constraint_names: tuple[str, ...]
    hard_constraint_values: tuple[float, ...]
    admitted: bool
    blocker_reason: str


class CandidateArtifactParityError(RuntimeError):
    """Stored and replayed candidate artifacts do not represent the same evaluation."""


def build_robustness_windows(
    *,
    l2_start_idx: int,
    holdout_start_idx: int,
    max_holding_bars: int,
    n_windows: int = 3,
) -> tuple[RobustnessWindow, ...]:
    if n_windows < 3:
        raise ValueError(f"n_windows must be >= 3, got {n_windows}")
    total_bars = holdout_start_idx - l2_start_idx
    if total_bars < n_windows * 2:
        raise ValueError(
            f"insufficient bars for {n_windows} windows: {total_bars} available "
            f"(need at least {n_windows * 2})"
        )

    embargo_bars = max(max_holding_bars, 1)
    window_size = total_bars // n_windows

    windows: list[RobustnessWindow] = []
    for i in range(n_windows):
        oos_start = l2_start_idx + i * window_size
        oos_end = l2_start_idx + (i + 1) * window_size
        if i == n_windows - 1:
            oos_end = holdout_start_idx
        if oos_end - oos_start < 2:
            raise ValueError(
                f"window {i} too small: {oos_end - oos_start} bars"
            )

        fit_start = l2_start_idx
        fit_end = oos_start
        cal_start = max(fit_start, oos_start - max_holding_bars)
        cal_end = oos_start
        if cal_end - cal_start < 1:
            cal_start = fit_start
            cal_end = fit_end

        windows.append(RobustnessWindow(
            label=f"window_{i}",
            fit_start=fit_start,
            fit_end=fit_end,
            cal_start=cal_start,
            cal_end=cal_end,
            oos_start=oos_start,
            oos_end=oos_end,
            embargo_bars=embargo_bars,
        ))

    return tuple(windows)


def compute_robust_compounding_score(
    *,
    growth_lcbs: tuple[float, ...],
    annualized_cost_drags: tuple[float, ...],
) -> float:
    if not growth_lcbs:
        return float("-inf")
    arr = np.asarray(growth_lcbs, dtype=np.float64)
    costs = np.asarray(annualized_cost_drags, dtype=np.float64)
    median = float(np.median(arr))
    q10 = float(np.quantile(arr, 0.10))
    mad = float(np.median(np.abs(arr - median)))
    median_cost = float(np.median(costs)) if costs.size > 0 else 0.0
    return median + 0.50 * q10 - 0.25 * mad - 0.10 * median_cost


def _canonicalize_metric(value: float) -> float:
    if not np.isfinite(value):
        return -1e6
    return value


def _candidate_hash(params: Mapping[str, Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"layer2-candidate-v1")
    for key in sorted(params.keys()):
        hasher.update(key.encode("utf-8"))
        val = params[key]
        if isinstance(val, float):
            hasher.update(float(val).hex().encode("utf-8"))
        elif isinstance(val, int):
            hasher.update(val.to_bytes(8, "big", signed=True))
        elif isinstance(val, str):
            hasher.update(val.encode("utf-8"))
        elif isinstance(val, bool):
            hasher.update(b"1" if val else b"0")
        else:
            hasher.update(str(val).encode("utf-8"))
    return hasher.hexdigest()


def _compute_hard_constraint_check(
    window_metrics: tuple[WindowCompoundingMetrics, ...],
    mdd_budget: float = 0.30,
    cvar_budget: float = 0.06,
    min_trades: int = 30,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    names: list[str] = []
    values: list[float] = []
    if not window_metrics:
        return ("no_windows",), (1.0,)
    for wm in window_metrics:
        if wm.mdd > mdd_budget:
            names.append(f"mdd_exceeded_{wm.label}")
            values.append(float(wm.mdd))
        if wm.cvar_95 > cvar_budget:
            names.append(f"cvar_exceeded_{wm.label}")
            values.append(float(wm.cvar_95))
        if wm.trade_count < min_trades:
            names.append(f"low_trades_{wm.label}")
            values.append(float(wm.trade_count))
    if not names:
        return ("passed",), (0.0,)
    return tuple(names), tuple(values)


def evaluate_l2_candidate_artifact(
    *,
    params: dict[str, Any],
    ctx: Any,
    robustness_windows: tuple[RobustnessWindow, ...],
    data_fingerprint: str,
    handoff_fingerprint: str,
    routing_hash: str,
    window_plan_hash: str,
    window_metrics: tuple[WindowCompoundingMetrics, ...] = (),
    leverage_schedule: tuple[FoldLeverageDecision, ...] = (),
) -> Layer2CandidateArtifact:
    c_hash = _candidate_hash(params)

    if len(robustness_windows) < 3:
        return Layer2CandidateArtifact(
            candidate_hash=c_hash,
            params=params,
            data_fingerprint=data_fingerprint,
            handoff_fingerprint=handoff_fingerprint,
            routing_hash=routing_hash,
            window_plan_hash=window_plan_hash,
            window_metrics=(),
            leverage_schedule=(),
            robust_score=float("-inf"),
            median_growth_lcb=float("-inf"),
            q10_growth_lcb=float("-inf"),
            positive_window_ratio=0.0,
            worst_window_cagr=float("-inf"),
            hard_constraint_names=("insufficient_robustness_windows",),
            hard_constraint_values=(float(len(robustness_windows)),),
            admitted=False,
            blocker_reason="insufficient_robustness_windows",
        )

    if not window_metrics:
        return Layer2CandidateArtifact(
            candidate_hash=c_hash,
            params=params,
            data_fingerprint=data_fingerprint,
            handoff_fingerprint=handoff_fingerprint,
            routing_hash=routing_hash,
            window_plan_hash=window_plan_hash,
            window_metrics=(),
            leverage_schedule=(),
            robust_score=float("-inf"),
            median_growth_lcb=float("-inf"),
            q10_growth_lcb=float("-inf"),
            positive_window_ratio=0.0,
            worst_window_cagr=float("-inf"),
            hard_constraint_names=("no_window_metrics",),
            hard_constraint_values=(0.0,),
            admitted=False,
            blocker_reason="no_window_metrics",
        )

    any_nonfinite = any(
        not np.isfinite(wm.cagr) or not np.isfinite(wm.growth_lcb)
        or not np.isfinite(wm.mdd) or not np.isfinite(wm.cvar_95)
        for wm in window_metrics
    )

    window_metrics = tuple(
        WindowCompoundingMetrics(
            label=wm.label,
            cagr=_canonicalize_metric(wm.cagr),
            mdd=_canonicalize_metric(wm.mdd),
            cvar_95=_canonicalize_metric(wm.cvar_95),
            growth_lcb=_canonicalize_metric(wm.growth_lcb),
            annualized_cost_drag=_canonicalize_metric(wm.annualized_cost_drag),
            trade_count=wm.trade_count if np.isfinite(float(wm.trade_count)) else 0,
        )
        for wm in window_metrics
    )

    growth_lcbs = tuple(wm.growth_lcb for wm in window_metrics)
    cost_drags = tuple(wm.annualized_cost_drag for wm in window_metrics)
    worst_cagr = min(wm.cagr for wm in window_metrics)
    positive_ratio = sum(1 for g in growth_lcbs if g > 0) / max(len(growth_lcbs), 1)

    robust_score = compute_robust_compounding_score(
        growth_lcbs=growth_lcbs,
        annualized_cost_drags=cost_drags,
    )
    median_lcb = float(np.median(np.asarray(growth_lcbs, dtype=np.float64)))
    q10_lcb = float(np.quantile(np.asarray(growth_lcbs, dtype=np.float64), 0.10))

    hard_names, hard_values = _compute_hard_constraint_check(window_metrics)

    if any_nonfinite:
        return Layer2CandidateArtifact(
            candidate_hash=c_hash,
            params=params,
            data_fingerprint=data_fingerprint,
            handoff_fingerprint=handoff_fingerprint,
            routing_hash=routing_hash,
            window_plan_hash=window_plan_hash,
            window_metrics=window_metrics,
            leverage_schedule=leverage_schedule,
            robust_score=robust_score,
            median_growth_lcb=median_lcb,
            q10_growth_lcb=q10_lcb,
            positive_window_ratio=positive_ratio,
            worst_window_cagr=worst_cagr,
            hard_constraint_names=(*hard_names, "nonfinite_metric"),
            hard_constraint_values=(*hard_values, 1.0),
            admitted=False,
            blocker_reason="nonfinite_candidate_metric",
        )

    admitted = True
    blocker = ""

    if len(growth_lcbs) < 3:
        admitted = False
        blocker = "insufficient_robustness_windows"
    elif positive_ratio < 2.0 / 3.0:
        admitted = False
        blocker = "low_positive_window_ratio"
    elif median_lcb <= 0.0:
        admitted = False
        blocker = "non_positive_median_growth_lcb"
    elif worst_cagr < -0.05:
        admitted = False
        blocker = "worst_window_cagr_below_floor"
    elif "mdd_exceeded" in str(hard_names):
        admitted = False
        blocker = "mdd_budget_exceeded"
    elif "cvar_exceeded" in str(hard_names):
        admitted = False
        blocker = "cvar_budget_exceeded"
    elif any("low_trades" in n for n in hard_names):
        admitted = False
        blocker = "minimum_trades_not_met"

    return Layer2CandidateArtifact(
        candidate_hash=c_hash,
        params=params,
        data_fingerprint=data_fingerprint,
        handoff_fingerprint=handoff_fingerprint,
        routing_hash=routing_hash,
        window_plan_hash=window_plan_hash,
        window_metrics=window_metrics,
        leverage_schedule=leverage_schedule,
        robust_score=robust_score,
        median_growth_lcb=median_lcb,
        q10_growth_lcb=q10_lcb,
        positive_window_ratio=positive_ratio,
        worst_window_cagr=worst_cagr,
        hard_constraint_names=hard_names,
        hard_constraint_values=hard_values,
        admitted=admitted,
        blocker_reason=blocker,
    )


def _validate_float_parity(
    stored: float,
    replayed: float,
    atol: float,
    rtol: float,
    name: str,
    errors: list[str],
) -> None:
    if not (np.isfinite(stored) and np.isfinite(replayed)):
        if not (not np.isfinite(stored) and not np.isfinite(replayed)):
            errors.append(f"{name}: finite parity mismatch ({stored} vs {replayed})")
        return
    if not np.isclose(stored, replayed, rtol=rtol, atol=atol):
        errors.append(f"{name}: {stored} vs {replayed} exceeds tolerance (atol={atol}, rtol={rtol})")


def validate_candidate_artifact_parity(
    *,
    stored: Layer2CandidateArtifact,
    replayed: Layer2CandidateArtifact,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> None:
    errors: list[str] = []

    for fp_field in ("data_fingerprint", "handoff_fingerprint", "routing_hash", "window_plan_hash"):
        stored_val = getattr(stored, fp_field)
        replayed_val = getattr(replayed, fp_field)
        if stored_val != replayed_val:
            errors.append(f"{fp_field}: {stored_val!r} vs {replayed_val!r}")

    _validate_float_parity(stored.robust_score, replayed.robust_score, atol, rtol, "robust_score", errors)
    _validate_float_parity(stored.median_growth_lcb, replayed.median_growth_lcb, atol, rtol, "median_growth_lcb", errors)
    _validate_float_parity(stored.q10_growth_lcb, replayed.q10_growth_lcb, atol, rtol, "q10_growth_lcb", errors)
    _validate_float_parity(stored.worst_window_cagr, replayed.worst_window_cagr, atol, rtol, "worst_window_cagr", errors)
    _validate_float_parity(stored.positive_window_ratio, replayed.positive_window_ratio, atol, rtol, "positive_window_ratio", errors)

    if stored.admitted != replayed.admitted:
        errors.append(f"admitted: {stored.admitted} vs {replayed.admitted}")

    if errors:
        raise CandidateArtifactParityError(
            f"CandidateArtifactParityError: {'; '.join(errors)}"
        )
