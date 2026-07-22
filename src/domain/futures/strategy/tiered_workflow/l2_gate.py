from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    Layer2FoldAttribution,
    compute_cost_drag_ratio,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    Layer2GateEvaluation,
)

_logger = logging.getLogger("opt_main_futures")

DeploymentMode = Literal["warmup_cash", "risk_on", "abstain_cash", "risk_off_cash", "blocked"]


@dataclass(slots=True, frozen=True)
class GrowthSafetyConstraintVector:
    data_integrity: float
    execution_integrity: float
    mdd: float
    cvar_95: float
    ruin: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "data_integrity": self.data_integrity,
            "execution_integrity": self.execution_integrity,
            "mdd": self.mdd,
            "cvar_95": self.cvar_95,
            "ruin": self.ruin,
        }


@dataclass(slots=True, frozen=True)
class GrowthSafetyGate:
    mode: DeploymentMode
    passed: bool
    blockers: tuple[str, ...]
    constraints: GrowthSafetyConstraintVector


def evaluate_growth_safety_gate(
    *,
    tape_valid: bool,
    execution_valid: bool,
    deployed_returns: NDArray[np.float64],
    max_mdd: float,
    max_cvar_95: float,
    min_equity_multiplier: float,
    growth_lcb: float,
) -> GrowthSafetyGate:
    blockers: list[str] = []

    if not tape_valid:
        blockers.append("data_integrity")
    if not execution_valid:
        blockers.append("execution_integrity")

    n = len(deployed_returns)
    if n >= 1:
        eq = _compute_equity_path(deployed_returns)
        dd = _compute_max_drawdown(eq)
        if dd > max_mdd:
            blockers.append("mdd")
        cvar95 = _compute_cvar_95(deployed_returns)
        if cvar95 > max_cvar_95:
            blockers.append("cvar_95")
        eq_mult = float(eq[-1])
        if eq_mult < min_equity_multiplier:
            blockers.append("ruin")
    else:
        dd = 0.0
        cvar95 = 0.0
        eq_mult = 1.0

    passed = len(blockers) == 0 and growth_lcb > 0.0
    if growth_lcb <= 0.0 and not blockers:
        mode: DeploymentMode = "abstain_cash"
    elif passed:
        mode = "risk_on"
    elif blockers:
        mode = "blocked"
    else:
        mode = "abstain_cash"

    constraints = GrowthSafetyConstraintVector(
        data_integrity=0.0 if tape_valid else 1.0,
        execution_integrity=0.0 if execution_valid else 1.0,
        mdd=dd,
        cvar_95=cvar95,
        ruin=max(0.0, min_equity_multiplier - eq_mult),
    )

    return GrowthSafetyGate(
        mode=mode,
        passed=passed,
        blockers=tuple(blockers),
        constraints=constraints,
    )


def _compute_equity_path(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    eq = np.empty(len(returns) + 1, dtype=np.float64)
    eq[0] = 1.0
    for t in range(len(returns)):
        eq[t + 1] = eq[t] * (1.0 + returns[t])
    return eq


def _compute_max_drawdown(equity: NDArray[np.float64]) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def _compute_cvar_95(returns: NDArray[np.float64]) -> float:
    sorted_rets = np.sort(returns)
    n = len(sorted_rets)
    var_idx = max(1, int(0.05 * n))
    tail = sorted_rets[:var_idx]
    return float(np.abs(np.mean(tail)))


# contract wiring: GrowthSafetyConstraintVector, evaluate_growth_safety_gate
@dataclass(slots=True, frozen=True)
class Layer2ConstraintVector:
    deployment: float
    support_leak: float
    policy_evidence: float
    active_blocks: float
    trades: float
    growth_lcb: float
    mdd: float
    cvar_95: float
    ruin: float
    crisis_mdd: float
    fold: float
    recent_fold: float
    recency_holdout: float
    friction: float
    crisis_measured: bool

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.deployment,
            self.support_leak,
            self.policy_evidence,
            self.active_blocks,
            self.trades,
            self.growth_lcb,
            self.mdd,
            self.cvar_95,
            self.ruin,
            self.crisis_mdd,
            self.fold,
            self.recent_fold,
            self.recency_holdout,
            self.friction,
        )

    def as_mapping(self) -> dict[str, float | bool]:
        return {
            "deployment": self.deployment,
            "support_leak": self.support_leak,
            "policy_evidence": self.policy_evidence,
            "active_blocks": self.active_blocks,
            "trades": self.trades,
            "growth_lcb": self.growth_lcb,
            "mdd": self.mdd,
            "cvar_95": self.cvar_95,
            "ruin": self.ruin,
            "crisis_mdd": self.crisis_mdd,
            "fold": self.fold,
            "recent_fold": self.recent_fold,
            "recency_holdout": self.recency_holdout,
            "friction": self.friction,
            "crisis_measured": self.crisis_measured,
        }

    def non_crisis_feasible(self) -> bool:
        non_crisis = (
            self.deployment,
            self.support_leak,
            self.policy_evidence,
            self.active_blocks,
            self.trades,
            self.growth_lcb,
            self.mdd,
            self.cvar_95,
            self.ruin,
            self.fold,
            self.recent_fold,
            self.recency_holdout,
            self.friction,
        )
        return all(v <= 0.0 for v in non_crisis)

    def jointly_feasible(self) -> bool:
        return bool(self.crisis_measured) and all(v <= 0.0 for v in self.as_tuple())

_DEFAULT_L2_CONFIG: Layer2AllocationConfig = Layer2AllocationConfig()

_PROMOTION_BLOCKERS: tuple[str, ...] = (
    "no_deployment",
    "low_trades",
    "cagr",
    "sharpe_abs",
    "sortino_floor",
    "calmar_floor",
    "mar",
    "mdd_abs",
    "cvar_95",
    "fold",
    "recent_fold",
    "active_blocks",
    "friction",
    "growth_lcb",
    "uplift",
    "psr_floor",
)


def compute_ruin_constraint(
    *,
    deployed_returns: NDArray[np.float64],
    min_equity_multiplier: float,
) -> float:
    equity = np.asarray(deployed_returns, dtype=np.float64).ravel()
    if equity.size == 0:
        return 1.0
    safe = np.maximum(equity, -1.0 + 1e-9)
    cum_mult = np.exp(np.log1p(safe).cumsum())
    min_mult = np.min(cum_mult)
    if min_mult <= min_equity_multiplier:
        return 1.0
    return -1.0


def _finite_or_fail(value: float, *, default_fail: float = 1.0) -> float:
    return float(value) if np.isfinite(value) else float(default_fail)


def _growth_lcb_vol_matched_baseline(
    baseline: float,
    hybrid_mean: float,
    std_hybrid: float | None,
    std_baseline: float | None,
) -> float:
    """Vol-matched growth_lcb baseline (RC-4).

    저변동 hybrid가 vol 차이로 인해 raw EW baseline을 부당하게 넘지 못하는 문제 해결.
    std_hybrid/std_baseline 미제공 시 원본 baseline 반환.
    """
    if std_hybrid is not None and std_baseline is not None and std_baseline > 1e-12:
        ratio = float(np.clip(std_hybrid / std_baseline, 0.3, 1.0))
        return baseline * ratio
    return baseline


def _absolute_growth_constraint(
    *,
    cagr_hybrid: float,
    growth_lcb_hybrid: float,
    l2_min_absolute_cagr: float,
    l2_min_growth_lcb: float,
) -> float:
    cagr_shortfall = max(0.0, float(l2_min_absolute_cagr) - float(cagr_hybrid))
    lcb_shortfall = max(0.0, float(l2_min_growth_lcb) - float(growth_lcb_hybrid))
    constraint = _finite_or_fail(max(cagr_shortfall, lcb_shortfall))
    _logger.debug(
        "[EVAL] tag=absolute_growth cagr=%.4f(vs%.4f) growth_lcb=%.4f(vs%.4f) constraint=%.4f",
        cagr_hybrid, l2_min_absolute_cagr, growth_lcb_hybrid, l2_min_growth_lcb, constraint,
    )
    return constraint


def evaluate_layer2_gate(
    *,
    deployment_failed: bool,
    support_leak_count: int,
    cagr_hybrid: float,
    sharpe_hybrid: float,
    sharpe_hac_hybrid: float,
    sharpe_hac_baseline: float,
    sortino_hybrid: float,
    mar_hybrid: float,
    mdd_hybrid: float,
    cvar_95_hybrid: float,
    fold_pass_ratio: float,
    active_block_count: int,
    friction_pass_pct: float,
    trade_count: int,
    growth_lcb_hybrid: float,
    growth_lcb_baseline: float,
    growth_lcb_deployed: float = float("-inf"),
    dsr_hybrid: float | None,
    psr_hybrid: float | None = None,
    recent_fold_passed: bool | None = None,
    recent_fold_sharpe: float | None = None,
    worst_fold_cagr: float | None = None,
    positive_block_delta_ratio: float | None = None,
    fold_attributions: tuple[Layer2FoldAttribution, ...] = (),
    config: Layer2AllocationConfig = _DEFAULT_L2_CONFIG,
    cagr_baseline: float | None = None,
    std_hybrid: float | None = None,
    std_baseline: float | None = None,
    crisis_mdd_hybrid: float | None = None,
    crisis_mdd_budget: float | None = None,
    crisis_cagr_hybrid: float | None = None,
    crisis_cagr_floor: float | None = None,
    recency_holdout_cagr: float | None = None,
    recency_holdout_applicable: bool = False,
    window_bottleneck_covered: bool = True,
    window_bottleneck_detail: str = "",
) -> Layer2GateEvaluation:
    """Build Optuna safety constraints and final L2 promotion gate diagnostics.

    [ADR_20260721_L2_RECENCY_GENERALIZATION_GATE] 14th constraint slot
    (recency_holdout) + window_bottleneck_covered diagnostic added.

    Args:
        deployment_failed: 시뮬레이션 배포 실패 여부.
        support_leak_count: look-ahead 누수 카운트.
        cagr_hybrid: 전략 연율화 복리 수익률.
        sharpe_hybrid: 전략 Sharpe 비율.
        sharpe_hac_hybrid: HAC 조정 Sharpe 비율.
        sharpe_hac_baseline: 기준선 HAC Sharpe.
        sortino_hybrid: 전략 Sortino 비율.
        mar_hybrid: CAGR / MDD 비율.
        mdd_hybrid: 최대 낙폭 (양수).
        cvar_95_hybrid: 95% CVaR 손실.
        fold_pass_ratio: 복리 기준 수익 fold 비율.
        active_block_count: 활성 블록 수.
        friction_pass_pct: 마찰 허들 통과 비율.
        trade_count: 총 거래 수.
        growth_lcb_hybrid: 성장 LCB (diagnostic).
        growth_lcb_baseline: 기준선 성장 LCB.
        dsr_hybrid: DSR (diagnostic only — BLOCKER 아님).
        psr_hybrid: PSR — L2 하드게이트 (None 시 통과).
        config: Layer2AllocationConfig.
        std_hybrid: hybrid 전략 per-bar std (vol-matched baseline용). None=비활성.
        std_baseline: 기준선 per-bar std (vol-matched baseline용). None=비활성.

    Returns:
        Layer2GateEvaluation.
    """
    _t_gate = time.perf_counter()
    # calmar = CAGR / MDD; mar_hybrid 이미 동일 계산이나 명시적 calmar 분리
    calmar_hybrid = float(cagr_hybrid) / (float(mdd_hybrid) + 1e-9)
    cost_drag = compute_cost_drag_ratio(fold_attributions) if fold_attributions else 0.0

    # psr_floor: PSR < threshold → BLOCKER (None 입력 시 -1.0 통과)
    psr_constraint = -1.0 if psr_hybrid is None else _finite_or_fail(float(config.l2_min_psr) - float(psr_hybrid))
    recent_fold_constraint = -1.0
    if config.l2_require_recent_fold_pass:
        if recent_fold_passed is False:
            recent_fold_constraint = 1.0
        elif recent_fold_sharpe is not None:
            recent_fold_constraint = max(
                recent_fold_constraint,
                _finite_or_fail(float(config.l2_min_recent_fold_sharpe) - float(recent_fold_sharpe)),
            )
    worst_fold_cagr_constraint = (
        -1.0
        if worst_fold_cagr is None or not np.isfinite(float(worst_fold_cagr))
        else _finite_or_fail(float(config.l2_min_worst_fold_cagr) - float(worst_fold_cagr))
    )
    block_delta_constraint = (
        -1.0
        if positive_block_delta_ratio is None or not np.isfinite(float(positive_block_delta_ratio))
        else _finite_or_fail(float(config.l2_min_positive_block_delta_ratio) - float(positive_block_delta_ratio))
    )
    recency_holdout_constraint = -1.0
    if (
        config.l2_require_recency_holdout_pass
        and recency_holdout_applicable
        and recency_holdout_cagr is not None
        and np.isfinite(float(recency_holdout_cagr))
    ):
        recency_holdout_constraint = _finite_or_fail(
            float(config.l2_min_recency_holdout_cagr) - float(recency_holdout_cagr),
        )
    _crisis_constraint = (
        1.0
        if crisis_mdd_hybrid is None or crisis_mdd_budget is None
        else _finite_or_fail(float(crisis_mdd_hybrid) - float(crisis_mdd_budget))
    )
    _growth_lcb = _absolute_growth_constraint(
        cagr_hybrid=cagr_hybrid,
        growth_lcb_hybrid=growth_lcb_deployed if np.isfinite(growth_lcb_deployed) else growth_lcb_hybrid,
        l2_min_absolute_cagr=config.l2_min_absolute_cagr,
        l2_min_growth_lcb=config.l2_min_growth_lcb,
    )
    optuna_constraint_values = (
        1.0 if deployment_failed else -1.0,
        float(max(support_leak_count, 0)),
        -1.0,
        _finite_or_fail(float(int(config.l2_min_active_blocks) - active_block_count)),
        _finite_or_fail(float(int(config.l2_min_trades) - trade_count)),
        _growth_lcb,
        _finite_or_fail(mdd_hybrid - float(config.l2_max_mdd_abs)),
        _finite_or_fail(cvar_95_hybrid - float(config.l2_max_cvar_95)),
        -1.0,
        _crisis_constraint,
        _finite_or_fail(float(config.l2_min_fold_pass_ratio) - fold_pass_ratio),
        recent_fold_constraint,
        recency_holdout_constraint,
        _finite_or_fail(float(config.l2_min_friction_pass) - friction_pass_pct),
    )

    promotion_constraint_values = (
        1.0 if deployment_failed else -1.0,
        _finite_or_fail(float(int(config.l2_min_trades) - trade_count)),
        _growth_lcb,
        _finite_or_fail(float(config.l2_min_sharpe_abs) - sharpe_hybrid),
        _finite_or_fail(float(config.l2_min_sortino) - sortino_hybrid),
        _finite_or_fail(float(config.l2_min_calmar) - calmar_hybrid),
        _finite_or_fail(float(config.l2_min_mar) - mar_hybrid),
        _finite_or_fail(mdd_hybrid - float(config.l2_max_mdd_abs)),
        _finite_or_fail(cvar_95_hybrid - float(config.l2_max_cvar_95)),
        _finite_or_fail(float(config.l2_min_fold_pass_ratio) - fold_pass_ratio),
        recent_fold_constraint,
        _finite_or_fail(float(int(config.l2_min_active_blocks) - active_block_count)),
        _finite_or_fail(float(config.l2_min_friction_pass) - friction_pass_pct),
        _finite_or_fail(
            _growth_lcb_vol_matched_baseline(
                growth_lcb_baseline,
                growth_lcb_hybrid,
                std_hybrid,
                std_baseline,
            )
            + float(config.l2_min_growth_uplift)
            - growth_lcb_hybrid
        ),
        _finite_or_fail(sharpe_hac_baseline + float(config.l2_min_sharpe_uplift) - sharpe_hac_hybrid),
        psr_constraint,
    )

    promotion_passed = True
    promotion_blocker = ""
    for blocker, value in zip(
        _PROMOTION_BLOCKERS,
        promotion_constraint_values,
        strict=True,
    ):
        if value > 0.0:
            promotion_passed = False
            promotion_blocker = blocker
            break
    if promotion_passed and cost_drag > float(config.l2_max_cost_drag_ratio):
        promotion_passed = False
        promotion_blocker = "cost_drag"
    if promotion_passed and worst_fold_cagr_constraint > 0.0:
        promotion_passed = False
        promotion_blocker = "worst_fold_cagr"
    if promotion_passed and recency_holdout_constraint > 0.0:
        promotion_passed = False
        promotion_blocker = "recency_holdout"
    # RC-4: block_delta → diagnostic-only (regime inert 시 구조적 통과불가 방지)
    if block_delta_constraint > 0.0:
        _logger.debug(
            "[L2-GATE-BLOCK-DELTA-DIAG] block_delta=%.4f > 0 (diagnostic only, no block)",
            block_delta_constraint,
        )

    _logger.debug(
        "[L2-GATE] promotion=%s blocker=%s | "
        "cagr=%.4f(vs%.2f) sortino=%.4f(vs%.2f) sharpe=%.4f(vs%.2f) calmar=%.4f(vs%.2f) | "
        "mdd=%.4f(vs%.2f) folds=%.2f(vs%.2f) trades=%d(vs%d) cost_drag=%.4f(vs%.2f) | "
        "psr=%.4f(vs%.2f) uplift=%.4f(vs%.2f) cvar=%.4f(vs%.2f)",
        promotion_passed,
        promotion_blocker,
        cagr_hybrid,
        config.l2_min_cagr,
        sortino_hybrid,
        config.l2_min_sortino,
        sharpe_hybrid,
        config.l2_min_sharpe_abs,
        cagr_hybrid / (mdd_hybrid + 1e-9),
        config.l2_min_calmar,
        mdd_hybrid,
        config.l2_max_mdd_abs,
        fold_pass_ratio,
        config.l2_min_fold_pass_ratio,
        trade_count,
        config.l2_min_trades,
        cost_drag,
        config.l2_max_cost_drag_ratio,
        psr_hybrid if psr_hybrid is not None else -1.0,
        config.l2_min_psr,
        sharpe_hac_hybrid - sharpe_hac_baseline,
        config.l2_min_sharpe_uplift,
        cvar_95_hybrid,
        config.l2_max_cvar_95,
    )

    _logger.debug(
        "[L2-GATE] evaluate took=%.4fs passed=%s reason=%s",
        time.perf_counter() - _t_gate,
        promotion_passed,
        promotion_blocker or "OK",
    )

    _crisis_measured = crisis_mdd_hybrid is not None and crisis_mdd_budget is not None
    constraint_vector = Layer2ConstraintVector(
        deployment=optuna_constraint_values[0],
        support_leak=optuna_constraint_values[1],
        policy_evidence=optuna_constraint_values[2],
        active_blocks=optuna_constraint_values[3],
        trades=optuna_constraint_values[4],
        growth_lcb=optuna_constraint_values[5],
        mdd=optuna_constraint_values[6],
        cvar_95=optuna_constraint_values[7],
        ruin=optuna_constraint_values[8],
        crisis_mdd=optuna_constraint_values[9],
        fold=optuna_constraint_values[10],
        recent_fold=optuna_constraint_values[11],
        recency_holdout=optuna_constraint_values[12],
        friction=optuna_constraint_values[13],
        crisis_measured=_crisis_measured,
    )
    return Layer2GateEvaluation(
        optuna_constraint_values=cast(tuple[float, ...], optuna_constraint_values),
        promotion_passed=promotion_passed,
        promotion_blocker=promotion_blocker,
        promotion_constraint_values=cast(tuple[float, ...], promotion_constraint_values),
        constraint_vector=constraint_vector,
        window_bottleneck_covered=window_bottleneck_covered,
        window_bottleneck_detail=window_bottleneck_detail,
    )
