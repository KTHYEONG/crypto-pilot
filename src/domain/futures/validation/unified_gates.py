"""Single entry point for futures promotion research gates (Phase-3, WF, IS, PF, survival).

Champion guard stays a separate economic promotion check; use `champion_registry` for that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.domain.futures.optimization.optimizer import check_hard_gates_ml

# ---------------------------------------------------------------------------
# Phase 3: V3HardGates 8-gate 체계
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V3HardGates:
    """v3.0 확정 상수 — 8-gate 평가 기준."""

    MIN_POSITIVE_LEG_RATIO: float = 0.55
    WORST_LEG_TW_FLOOR: float = 0.85
    MEAN_LEG_TW_FLOOR: float = 1.015
    ERGODICITY_PCT: float = 15.0
    EV_COST_FLOOR: float = 3.0
    DSR_FLOOR: float = 0.60
    FUNDING_DRAG_CEILING: float = 0.30
    CAPACITY_REQUIRED_TIERS: tuple[int, ...] = (50_000, 100_000, 250_000)


@dataclass
class GateResult:
    """Gate 평가 결과."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def evaluate_v3_hard_gates(
    leg_log_tw: np.ndarray,
    worst_mdd: float,
    dsr: float,
    ev_cost: float,
    funding_drag_ratio: float,
    ergodicity_dev_pct: float,
    capacity_results: dict[int, bool],
    gates: V3HardGates = V3HardGates(),
) -> GateResult:
    """8-gate v3.0 평가.

    Args:
        leg_log_tw: shape [K] — leg별 log Terminal Wealth.
        worst_mdd: 최대 낙폭 (0~1).
        dsr: Deflated Sharpe Ratio (0~1).
        ev_cost: EV/Cost 비율.
        funding_drag_ratio: funding_drag / gross_return (0~1).
        ergodicity_dev_pct: ergodicity deviation (%).
        capacity_results: {aum: pass/fail} — CAPACITY_REQUIRED_TIERS 전부 필요.
        gates: V3HardGates 상수 컨테이너.

    Returns:
        GateResult(passed, failures, metrics).

    """
    arr = np.asarray(leg_log_tw, dtype=np.float64)
    tw_arr = np.exp(arr)

    failures: list[str] = []

    # Gate 1: min positive leg ratio
    pos_ratio = float(np.mean(tw_arr >= 1.0))
    if pos_ratio < gates.MIN_POSITIVE_LEG_RATIO:
        failures.append("WF_POSITIVE_LEG_RATIO")

    # Gate 2: worst leg TW floor
    worst_tw = float(np.min(tw_arr)) if arr.size > 0 else 0.0
    if worst_tw < gates.WORST_LEG_TW_FLOOR:
        failures.append("WF_WORST_LEG_TW")

    # Gate 3: mean leg TW floor
    mean_tw = float(np.mean(tw_arr)) if arr.size > 0 else 0.0
    if mean_tw < gates.MEAN_LEG_TW_FLOOR:
        failures.append("WF_MEAN_LEG_TW")

    # Gate 4: DSR floor
    if float(dsr) < gates.DSR_FLOOR:
        failures.append("DSR_FLOOR")

    # Gate 5: funding drag ceiling
    if float(funding_drag_ratio) > gates.FUNDING_DRAG_CEILING:
        failures.append("FUNDING_DRAG")

    # Gate 6: capacity — CAPACITY_REQUIRED_TIERS 전부 통과 필요
    required_tiers = set(gates.CAPACITY_REQUIRED_TIERS)
    cap_fail = any(
        not capacity_results.get(tier, False)
        for tier in required_tiers
        if tier in capacity_results
    ) or any(
        tier not in capacity_results
        for tier in required_tiers
    )
    if cap_fail:
        failures.append("CAPACITY")

    # Gate 7: ergodicity
    if float(ergodicity_dev_pct) > gates.ERGODICITY_PCT:
        failures.append("WF_ERGODICITY")

    # Gate 8: EV/Cost
    if float(ev_cost) < gates.EV_COST_FLOOR:
        failures.append("EV_COST")

    metrics: dict[str, float] = {
        "pos_ratio": pos_ratio,
        "worst_tw": worst_tw,
        "mean_tw": mean_tw,
        "dsr": float(dsr),
        "funding_drag_ratio": float(funding_drag_ratio),
        "ergodicity_dev_pct": float(ergodicity_dev_pct),
        "ev_cost": float(ev_cost),
    }

    return GateResult(
        passed=len(failures) == 0,
        failures=failures,
        metrics=metrics,
    )


@dataclass(frozen=True)
class FuturesResearchGateInput:
    """Thresholds and observations for one-shot research gate evaluation."""

    phase3_enabled: bool
    pbo_max: float
    dsr_min: float
    is_precision: float
    oos_port: dict[str, Any]
    pbo_obs: float
    dsr_obs: float
    wf_failures: tuple[str, ...]
    min_is_net_alpha_pct: float
    is_net_alpha_pct: float
    min_long_pf: float
    min_short_pf: float
    oos_long_pf: float
    oos_short_pf: float
    is_cagr_pct: float
    is_sharpe: float
    is_survival_min_cagr: float
    is_survival_min_sharpe: float
    worst_leg_log_tw: float
    awf_p10_log_tw_floor: float
    # V3.1 Mechanical Additions
    oos_mdd_duration: float = 0.0
    max_mdd_duration: float = 180.0
    oos_expectancy: float = 0.0
    min_expectancy: float = 0.40
    is_expectancy: float = 0.0
    min_oos_retention_expectancy_pct: float = 50.0
    # Auxiliary diagnostics only (non-blocking): legacy CAGR retention.
    oos_cagr_pct: float = 0.0
    is_cagr_ref_pct: float = 0.0


def evaluate_research_gates(inp: FuturesResearchGateInput) -> tuple[bool, list[str]]:
    """Evaluate ordered research gates; return (ok, failure codes).

    Short-circuits on first blocking group.
    """
    failures: list[str] = []

    if inp.phase3_enabled:
        if not check_hard_gates_ml(
            inp.oos_port,
            float(inp.pbo_obs),
            float(inp.dsr_obs),
            inp.is_precision,
            pbo_max_override=inp.pbo_max,
            dsr_min_override=inp.dsr_min,
        ):
            failures.append("PHASE3_HARD_GATE")
            return False, failures

    if inp.wf_failures:
        failures.extend(inp.wf_failures)
        return False, failures

    # V3.1 Mechanical: Expectancy Gate
    if inp.oos_expectancy < float(inp.min_expectancy):
        failures.append("EXPECTANCY_GATE")
        return False, failures

    # V4.3: OOS retention gate based on expectancy (not CAGR)
    if abs(float(inp.is_expectancy)) > 1e-9:
        _exp_ret = float(inp.oos_expectancy) / float(inp.is_expectancy) * 100.0
    else:
        _exp_ret = 0.0
    if _exp_ret < float(inp.min_oos_retention_expectancy_pct):
        failures.append("OOS_RETENTION_EXPECTANCY_GATE")
        return False, failures

    # V3.1 Mechanical: MDD Duration Gate
    if inp.oos_mdd_duration > float(inp.max_mdd_duration):
        failures.append("MDD_DURATION_GATE")
        return False, failures

    if inp.is_net_alpha_pct <= float(inp.min_is_net_alpha_pct):
        failures.append("IS_ALPHA_GATE")
        return False, failures

    if inp.oos_long_pf < float(inp.min_long_pf) or inp.oos_short_pf < float(inp.min_short_pf):
        failures.append("DIRECTIONAL_PF_GATE")
        return False, failures

    if not (
        inp.is_cagr_pct > float(inp.is_survival_min_cagr)
        and inp.is_sharpe > float(inp.is_survival_min_sharpe)
    ):
        failures.append("IS_SURVIVAL_GATE")
        return False, failures

    if inp.worst_leg_log_tw <= float(inp.awf_p10_log_tw_floor):
        failures.append("AWF_HARDENING_GATE")
        return False, failures

    return True, []


# Human-readable map for logs / ops automation
GATE_CODE_DESCRIPTIONS: dict[str, str] = {
    "PHASE3_HARD_GATE": "PBO/DSR/WR/Mdd/Dir-PF Phase-3 composite gate failed",
    "WF_POSITIVE_LEG_RATIO": "Walk-forward: insufficient fraction of legs with TW>=1",
    "WF_WORST_LEG_TW": "Walk-forward: worst-leg terminal wealth below floor",
    "WF_MEAN_LEG_TW": "Walk-forward: mean leg TW below floor",
    "WF_ERGODICITY": "Walk-forward: path ergodicity deviation above guideline",
    "WF_EMPTY_LEGS": "Walk-forward: no evaluable legs",
    "WF_INVALID_SPAN": "Walk-forward: OOS span or n_legs invalid",
    "IS_ALPHA_GATE": "In-sample net alpha vs BTC below policy floor",
    "DIRECTIONAL_PF_GATE": "OOS long/short profit factor below policy minima",
    "IS_SURVIVAL_GATE": "IS CAGR or IS Sharpe below survival cut",
    "AWF_HARDENING_GATE": "Worst AWF leg log-TW below distributional floor",
    "EXPECTANCY_GATE": "OOS Mean Return per Trade below 0.40% mechanical hurdle",
    "OOS_RETENTION_EXPECTANCY_GATE": "OOS/IS expectancy retention below policy floor",
    "MDD_DURATION_GATE": "OOS Max Drawdown Duration exceeds 180 days",
}
