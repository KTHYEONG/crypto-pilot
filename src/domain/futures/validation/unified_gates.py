"""Single entry point for futures promotion research gates (Phase-3, WF, IS, PF, survival).

Champion guard stays a separate economic promotion check; use `champion_registry` for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.futures.optimization.optimizer import check_hard_gates_ml


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
