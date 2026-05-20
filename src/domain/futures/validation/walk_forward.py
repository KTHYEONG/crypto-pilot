from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.domain.futures.optimization.validation import wf_path_ergodicity_deviation_pct


@dataclass(frozen=True)
class WalkForwardConfig:
    """User thresholds for multi-leg walk-forward on OOS.

    v3.0 상수 기준:
        n_legs: 8 (기존 10 → 8)
        min_positive_leg_ratio: 0.55 (기존 0.70 → 0.55)
        worst_leg_tw_floor: 0.85 (기존 0.95 → 0.85)
        mean_leg_tw_floor: 1.015 (기존 1.00 → 1.015)
        dsr_floor: 0.60 (신규)
        funding_drag_ceiling: 0.30 (신규)
    """

    n_legs: int = 8
    purge_bars: int = 24
    min_positive_leg_ratio: float = 0.55
    worst_leg_tw_floor: float = 0.85
    mean_leg_tw_floor: float = 1.015
    ergodicity_guideline_pct: float = 15.0
    dsr_floor: float = 0.60
    funding_drag_ceiling: float = 0.30
    ergodicity_hard_gate_enabled: bool = True
    # When True, tile OOS using the same anchored WF geometry as ML objective (IS-pool + embargo).
    use_anchored_awf_geometry: bool = False
    anchored_is_pool_frac: float = 0.70
    anchored_embargo_bars: int = 0


@dataclass(frozen=True)
class WalkForwardResult:
    """WF summary plus optional per-leg drift rows (alpha / meta-calib / crisis vs prior leg)."""

    tw_legs: list[float]
    positive_leg_ratio: float
    worst_leg_tw: float
    mean_leg_tw: float
    ergodicity_dev_pct: float
    passed: bool
    failures: list[str]
    dsr: float = 1.0
    funding_drag_ratio: float = 0.0
    leg_adaptation_logs: tuple[dict[str, Any], ...] = ()


def mirror_walk_forward_result_from_awf_user_attrs(
    user_attrs: dict[str, Any],
    cfg: WalkForwardConfig,
) -> WalkForwardResult:
    """Post-opt WF summary from trial AWF leg stats (mirrored; no extra OOS reruns).

    Applies positivity / TW floors / ergodicity checks using leg log-TW series on the trial.
    """
    raw = (
        user_attrs.get("awf_path_leg_log_tw")
        or user_attrs.get("awf_leg_log_tw")
        or user_attrs.get("cpcv_path_oos_log_tw")
        or []
    )
    log_tw = [float(x) for x in raw]
    tw_legs = [float(np.exp(v)) for v in log_tw]
    if not tw_legs:
        return WalkForwardResult(
            tw_legs=[],
            positive_leg_ratio=0.0,
            worst_leg_tw=0.0,
            mean_leg_tw=0.0,
            ergodicity_dev_pct=0.0,
            passed=False,
            failures=["WF_AWFMIRROR_EMPTY"],
            dsr=1.0,
            funding_drag_ratio=0.0,
            leg_adaptation_logs=(),
        )

    arr = np.asarray(tw_legs, dtype=np.float64)
    pos_ratio = float(np.mean(arr >= 1.0))
    worst_tw = float(np.min(arr))
    mean_tw = float(np.mean(arr))
    erg_dev = float(wf_path_ergodicity_deviation_pct(tw_legs)) if arr.size >= 2 else 0.0

    dsr = float(user_attrs.get("dsr", 1.0))
    funding_drag = float(user_attrs.get("funding_drag_ratio", 0.0))

    failures: list[str] = []
    if pos_ratio < float(cfg.min_positive_leg_ratio):
        failures.append("WF_POSITIVE_LEG_RATIO")
    if worst_tw < float(cfg.worst_leg_tw_floor):
        failures.append("WF_WORST_LEG_TW")
    if mean_tw < float(cfg.mean_leg_tw_floor):
        failures.append("WF_MEAN_LEG_TW")
    if cfg.ergodicity_hard_gate_enabled and erg_dev > float(cfg.ergodicity_guideline_pct):
        failures.append("WF_ERGODICITY")
    if dsr < float(cfg.dsr_floor):
        failures.append("WF_DSR_FLOOR")
    if funding_drag > float(cfg.funding_drag_ceiling):
        failures.append("WF_FUNDING_DRAG")

    return WalkForwardResult(
        tw_legs=tw_legs,
        positive_leg_ratio=pos_ratio,
        worst_leg_tw=worst_tw,
        mean_leg_tw=mean_tw,
        ergodicity_dev_pct=erg_dev,
        passed=len(failures) == 0,
        failures=failures,
        dsr=dsr,
        funding_drag_ratio=funding_drag,
        leg_adaptation_logs=(),
    )
