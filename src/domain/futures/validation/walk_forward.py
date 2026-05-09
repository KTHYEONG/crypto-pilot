from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.optimization.validation import wf_path_ergodicity_deviation_pct


@dataclass(frozen=True)
class WalkForwardConfig:
    """User thresholds for multi-leg walk-forward on OOS."""

    n_legs: int = 10
    purge_bars: int = 24
    min_positive_leg_ratio: float = 0.70
    worst_leg_tw_floor: float = 0.95
    mean_leg_tw_floor: float = 1.00
    ergodicity_guideline_pct: float = 15.0
    ergodicity_hard_gate_enabled: bool = True


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
    leg_adaptation_logs: tuple[dict[str, Any], ...] = ()


def _leg_window_drift(
    df: pd.DataFrame | None,
    ls_eval: int,
    le: int,
    prev_lo: int | None,
    prev_hi: int | None,
) -> dict[str, float]:
    if df is None or len(df) == 0:
        return {}
    out: dict[str, float] = {}
    # Current leg eval window [ls_eval, le)
    i0, i1 = max(0, ls_eval), max(ls_eval + 1, le)
    if i1 <= i0 or i1 > len(df):
        return out
    sub = df.iloc[i0:i1]
    prev_sub = None
    if prev_lo is not None and prev_hi is not None and prev_hi > prev_lo >= 0:
        p0, p1 = max(0, prev_lo), min(prev_hi, len(df))
        if p1 > p0:
            prev_sub = df.iloc[p0:p1]

    def _col_mean(frame: pd.DataFrame, name: str) -> float:
        if name not in frame.columns:
            return float("nan")
        return float(pd.to_numeric(frame[name], errors="coerce").mean())

    for col in ("ml_alpha_00", "ml_calib_prob", "hmm_prob_crisis"):
        cm = _col_mean(sub, col)
        out[f"{col}_mean"] = cm
        if prev_sub is not None and col in prev_sub.columns:
            pm = _col_mean(prev_sub, col)
            delta = cm - pm if np.isfinite(cm) and np.isfinite(pm) else float("nan")
            out[f"{col}_delta_vs_prev"] = delta
        else:
            out[f"{col}_delta_vs_prev"] = float("nan")

    return out


def evaluate_walk_forward(
    ref_len: int,
    oos_start_idx: int,
    run_leg: Callable[..., dict[str, Any]],
    cfg: WalkForwardConfig,
    *,
    leg_indexed_runner: bool = False,
    before_each_leg: Callable[[int, int, int], None] | None = None,
    drift_signal_df: pd.DataFrame | None = None,
) -> WalkForwardResult:
    """Evaluate walk-forward legs over OOS span.

    Args:
        ref_len: Length of reference symbol OHLCV index.
        oos_start_idx: First OOS bar index.
        run_leg: Backtest callable; see ``leg_indexed_runner``.
        cfg: Thresholds and leg count.
        leg_indexed_runner: If True, call ``run_leg(leg_idx, ls_eval, le)``; else
            ``run_leg(ls_eval, le)``.
        before_each_leg: Optional callback before each leg (HMM/meta adaptation hook).
        drift_signal_df: Optional frame for per-leg drift stats vs previous leg.

    """
    if ref_len <= oos_start_idx or cfg.n_legs <= 1:
        return WalkForwardResult(
            tw_legs=[],
            positive_leg_ratio=0.0,
            worst_leg_tw=0.0,
            mean_leg_tw=0.0,
            ergodicity_dev_pct=0.0,
            passed=False,
            failures=["WF_INVALID_SPAN"],
            leg_adaptation_logs=(),
        )

    span = max(0, int(ref_len) - int(oos_start_idx))
    leg_w = max(1, span // int(cfg.n_legs))
    tw_legs: list[float] = []
    adaptation_logs: list[dict[str, Any]] = []
    prev_bounds: tuple[int | None, int | None] = (None, None)

    for leg in range(int(cfg.n_legs)):
        ls = int(oos_start_idx + leg * leg_w)
        le = int(oos_start_idx + (leg + 1) * leg_w) if leg < cfg.n_legs - 1 else int(ref_len)
        if le <= ls:
            continue
        ls_eval = min(ls + int(cfg.purge_bars), le - 1)
        if before_each_leg is not None:
            before_each_leg(leg, ls_eval, le)

        if leg_indexed_runner:
            port = run_leg(leg, ls_eval, le)
        else:
            port = run_leg(ls_eval, le)

        tw = float(port.get("terminal_wealth_ratio", 1.0))
        if not np.isfinite(tw):
            tw = 0.0
        tw_legs.append(tw)

        drift = _leg_window_drift(drift_signal_df, ls_eval, le, prev_bounds[0], prev_bounds[1])
        adaptation_logs.append(
            {
                "leg": int(leg),
                "ls_eval": int(ls_eval),
                "le": int(le),
                "tw": float(tw),
                **drift,
            }
        )
        prev_bounds = (int(ls_eval), int(le))

    if not tw_legs:
        return WalkForwardResult(
            tw_legs=[],
            positive_leg_ratio=0.0,
            worst_leg_tw=0.0,
            mean_leg_tw=0.0,
            ergodicity_dev_pct=0.0,
            passed=False,
            failures=["WF_EMPTY_LEGS"],
            leg_adaptation_logs=(),
        )

    arr = np.asarray(tw_legs, dtype=np.float64)
    pos_ratio = float(np.mean(arr >= 1.0))
    worst_tw = float(np.min(arr))
    mean_tw = float(np.mean(arr))
    erg_dev = float(wf_path_ergodicity_deviation_pct(tw_legs)) if arr.size >= 2 else 0.0

    failures: list[str] = []
    if pos_ratio < float(cfg.min_positive_leg_ratio):
        failures.append("WF_POSITIVE_LEG_RATIO")
    if worst_tw < float(cfg.worst_leg_tw_floor):
        failures.append("WF_WORST_LEG_TW")
    if mean_tw < float(cfg.mean_leg_tw_floor):
        failures.append("WF_MEAN_LEG_TW")
    if cfg.ergodicity_hard_gate_enabled and erg_dev > float(cfg.ergodicity_guideline_pct):
        failures.append("WF_ERGODICITY")

    return WalkForwardResult(
        tw_legs=tw_legs,
        positive_leg_ratio=pos_ratio,
        worst_leg_tw=worst_tw,
        mean_leg_tw=mean_tw,
        ergodicity_dev_pct=erg_dev,
        passed=len(failures) == 0,
        failures=failures,
        leg_adaptation_logs=tuple(adaptation_logs),
    )
