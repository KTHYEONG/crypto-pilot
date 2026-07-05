# src/domain/futures/strategy/tiered_workflow/dataclasses.py

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.domain.futures.strategy.candidate_contracts import (
        FoldFitStatus,
        Layer1FoldReadiness,
        Layer1GateReport,
        Layer1InferenceArtifact,
        QualifiedSignalRegistry,
        SymbolStrategyEvidence,
        ValidatedSignalEvent,
    )
    from src.domain.futures.strategy.cs_rank import SymbolSignal
    from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    DirectionalVetoSummary,
    Layer2FoldAttribution,
    MajorSymbolIncoherenceSummary,
    MajorSymbolSignalSizingSummary,
    MajorSymbolSleeveContributionSummary,
    ReversalEpisode,
)
    from src.domain.futures.strategy.tiered_workflow.tf_validation_repair import (
        ValidationParityCapture,
        ValidationParityReport,
    )

def _validate_directional_veto_action(value: str) -> Literal["drop_long", "zero_mu", "cap_mu"]:
    if value not in {"drop_long", "zero_mu", "cap_mu"}:
        raise ValueError(
            f"l2_regime_directional_veto_action must be one of drop_long/zero_mu/cap_mu, got {value!r}"
        )
    return value  # type: ignore[return-value]


def _validate_directional_veto_mode(value: str) -> Literal["adverse_only", "contextual"]:
    if value not in {"adverse_only", "contextual"}:
        raise ValueError(
            f"l2_regime_directional_veto_mode must be one of adverse_only/contextual, got {value!r}"
        )
    return value  # type: ignore[return-value]


def _validate_directional_veto_symbols(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("l2_regime_directional_veto_symbols must be a sequence")
    seen: set[str] = set()
    result: list[str] = []
    for s in value:
        sym = str(s).strip().upper()
        if not sym:
            raise ValueError("l2_regime_directional_veto_symbols contains empty symbol")
        if sym not in seen:
            seen.add(sym)
            result.append(sym)
    return tuple(result)


def _validate_directional_veto_adverse_codes(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("l2_regime_directional_veto_adverse_codes must be a sequence")
    result: list[int] = []
    for c in value:
        code = int(c)
        if code not in {1, 2}:
            raise ValueError(
                f"l2_regime_directional_veto_adverse_codes must contain only 1 or 2, got {code}"
            )
        result.append(code)
    return tuple(sorted(set(result)))


AllocationPolicy = Literal["diagonal_kelly", "directional_equal_weight"]
RegimePolicyMode = Literal["filter", "observe", "soft", "hybrid"]
RegimePolicyAction = Literal["pooled", "allow", "downweight", "block"]
RegimeBucketAction = Literal["allow", "downweight", "pool"]
RegimePolicyReason = Literal[
    "legacy_filter",
    "observe_only",
    "global_unreliable",
    "insufficient_fit",
    "insufficient_cal",
    "cal_sign_unstable",
    "negative_cal_lift",
    "positive_cal_lift",
    "neutral",
    "pooled_passthrough",
    "insufficient_fit_but_good_cal",
    "insufficient_cal_partial",
]


@dataclass(frozen=True, slots=True)
class SymbolLifecycleRecord:
    """Per-symbol L1 fold lifecycle tracking for PIT promotion gate.

    Attributes:
        symbol: 심볼 식별자.
        fold_status: L1 fold 내 심볼 수명주기 상태.
            - ``not_evaluated``: active_mask 전구간 False — L2 제외.
            - ``not_ready``: promotion_available_at > l2_start — 아직 미승인.
            - ``evaluated``: oos_stacked 진입했으나 ready_symbols 미포함.
            - ``failed``: eligible bars 존재하나 oos_stacked 미진입.
            - ``promoted``: deployment_registry.ready_symbols 포함 + promotion_available_at <= l2_start.
        promotion_available_at: 첫 eligible bar 날짜. not_evaluated 시 ``None``.
    """

    symbol: str
    fold_status: Literal["promoted", "evaluated", "failed", "not_ready", "not_evaluated"]
    promotion_available_at: date | None  # None iff not_evaluated / no active bars in L1


@dataclass(slots=True, frozen=True)
class Layer1Result:
    """Layer1 SWF-K 검증 결과.

    [ADR_20260705_TF_VALIDATION_ROOT_CAUSE_CAPTURE]

    Attributes:
        signals_per_fold: fold별 symbol→SymbolSignal 매핑 튜플.
        oos_stacked: fold 횡단 합본 (L2 입력용, look-ahead 없음).
        pooled_ic: 전 fold OOS pooled Spearman IC (Σ events 기준).
        pooled_tstat: Newey-West HAC t-stat (autocorrelation 보정).
        breadth: 평균 valid 심볼 비율 (per fold).
        valid_coverage: valid 심볼 비율 ≥ 0.5인 fold 비율.
        fold_pass_ratio: event-weighted fold pass ratio (진단용, gate 미포함).
        gate_passed: L1 통과 여부.
        n_valid: 마지막 fold 기준 valid 심볼 수.
        n_total: 전체 심볼 수 (aligned width).
        n_trade_scope: tiered aligned scope 크기 (bridge와 동일한 Stage6 OOS ∩ data-valid).
    """

    signals_per_fold: tuple[dict[str, SymbolSignal], ...]
    oos_stacked: dict[str, SymbolSignal]
    pooled_ic: float
    pooled_tstat: float
    breadth: float
    valid_coverage: float
    fold_pass_ratio: float
    gate_passed: bool
    n_valid: int
    n_total: int
    n_trade_scope: int = 0
    cs_ic_mean: float = 0.0
    cs_ic_tstat: float = 0.0
    cs_ic_fold_pass_ratio: float = 0.0
    decile_lift_bps: float = 0.0
    strategy_panel: tuple[StrategySignal, ...] = ()
    n_valid_strategies: int = 0
    panel_diversity: float = 0.0
    outer_fold_reports: tuple[Layer1FoldReadiness, ...] = ()
    deployment_evidence: tuple[SymbolStrategyEvidence, ...] = ()
    gate_report: Layer1GateReport | None = None
    deployment_registry: QualifiedSignalRegistry | None = None
    symbol_lifecycle: tuple[SymbolLifecycleRecord, ...] = ()
    inference_artifact: Layer1InferenceArtifact | None = None
    artifacts_by_tf: dict[str, Layer1InferenceArtifact] = field(default_factory=dict)
    validation_parity_capture: ValidationParityCapture | None = None
    validation_parity_report: ValidationParityReport | None = None


@dataclass(frozen=True, slots=True)
class StrategySignal:
    """전략별 OOS 독립검증 결과."""

    strategy_id: str
    oos_edge_bps: float
    oos_nw_tstat: float
    hit_rate: float
    fold_sign_consistency: float
    n_obs: int
    n_folds: int
    valid: bool
    _fold_edges: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True, slots=True)
class PredictionDecompositionDiag:
    """C0 진단: 예측의 정적/동적 분산 분해 + archetype 실현엣지 + decile lift.

    Attributes:
        static_variance_share: Var 설명 비율 — (arch,regime,variant) 그룹평균으로 설명되는 비율.
        dynamic_variance_share: 1 - static_variance_share (이벤트 내 잔차 변동).
        score_cal_valid_ratio: valid regime 비율 (fold 평균).
        per_archetype_oos_edge: archetype → (mean_bps, nw_tstat) OOS 실현엣지.
        decile_lift_bps: top10% - bottom10% 실현엣지 (expected_net_bps 정렬 기준).
    """

    static_variance_share: float
    dynamic_variance_share: float
    score_cal_valid_ratio: float
    per_archetype_oos_edge: dict[str, tuple[float, float]]
    decile_lift_bps: float


@dataclass(frozen=True, slots=True)
class SymbolRealizedStat:
    """심볼별 실현 수익 기반 QC 통계 (예측값 독립).

    Attributes:
        realized_mu_bps: 실현 y_return_bps fold-pooled 평균.
        t_stat: 실현 엣지 Bartlett NW HAC t-stat.
        n_obs: fold 합산 유효 이벤트 수.
        ic: per-symbol TS rank IC (compute_per_symbol_ic 결과).
        valid: n_obs>=min_obs ∧ |t_stat|>=floor ∧ isfinite ∧ ic>0.
    """

    realized_mu_bps: float
    t_stat: float
    n_obs: int
    ic: float
    valid: bool


@dataclass(slots=True, frozen=True)
class Layer2BlockMetric:
    """Layer2 연속 블록 단위 성장/리스크 요약."""

    start_idx: int
    end_idx: int
    log_growth_hybrid: float
    log_growth_baseline: float
    mdd_hybrid: float
    turnover_hybrid: float
    active_rebalances: int


@dataclass(slots=True, frozen=True)
class LayerUniverseAudit:
    """Layer universe contract audit payload."""

    layer: str
    start_idx: int
    end_idx: int
    start_date: str
    end_date: str
    symbol_count: int
    active_symbol_count_min: int
    active_symbol_count_median: float
    active_symbol_count_max: int
    entry_block_count: int
    kill_count: int
    symbols: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class Layer2FoldDiagnostics:
    """Layer2 fold-level deployment diagnostics."""

    fold_pass_ratio: float
    fold_compound_pass: tuple[bool | None, ...]
    fold_unit_sharpes: tuple[float, ...]
    fold_deployed_cagrs: tuple[float | None, ...]
    fold_deployed_mdds: tuple[float | None, ...]
    fold_selected_symbols: tuple[tuple[str, ...], ...]
    recent_fold_passed: bool | None
    recent_fold_sharpe: float | None
    recent_fold_cagr: float
    recent_fold_mdd: float
    latest_to_median_cagr: float


@dataclass(slots=True, frozen=True)
class Layer2TrialEvaluation:
    """Layer2 단일 trial 성장/제약 평가 결과."""

    objective_value: float
    constraint_values: tuple[float, ...]
    cagr_hybrid: float
    cagr_baseline: float
    growth_lcb_hybrid: float
    growth_lcb_baseline: float
    sharpe_hac_hybrid: float
    sharpe_hac_baseline: float
    psr_hybrid: float
    mdd_hybrid: float
    cvar_95_hybrid: float
    fold_pass_ratio: float
    break_even_pass_pct: float
    average_gross_exposure: float
    cap_saturation_ratio: float
    total_cost_bps: float
    block_metrics: tuple[Layer2BlockMetric, ...]
    master_tf: str = "4h"  # annualization tf SSOT — bars_per_year 재구성용
    returns_hybrid: tuple[float, ...] = ()
    returns_baseline: tuple[float, ...] = ()
    sharpe_hybrid: float = 0.0
    sharpe_hac_baseline_ew: float = 0.0
    sortino_hybrid: float = 0.0
    trade_count: int = 0
    risk_utilization: float = 0.0
    deployment_objective_bonus: float = 0.0
    worst_fold_sharpe: float = 0.0
    gate: Layer2GateEvaluation | None = None
    fit_returns_hybrid: tuple[float, ...] = ()
    deploy_leverage: float = 1.0
    deploy_binding: str = ""
    recent_fold_passed: bool | None = None
    recent_fold_sharpe: float = 0.0
    recent_fold_cagr: float = 0.0
    recent_fold_mdd: float = 0.0
    latest_to_median_cagr: float = 0.0
    fold_deployed_cagrs: tuple[float | None, ...] = ()
    fold_deployed_mdds: tuple[float | None, ...] = ()
    fold_deployed_sharpes: tuple[float, ...] = ()
    fold_selected_symbols: tuple[tuple[str, ...], ...] = ()
    worst_fold_cagr: float = float("nan")
    positive_block_delta_ratio: float = float("nan")
    bucket_reliability_mean: float = 0.0
    entry_spike_penalty: float = 0.0
    # deployment extras (SSOT: run_l2_awf가 evaluate_l2_trial에 위임하기 위한 raw data)
    last_selected_symbols: tuple[str, ...] = ()
    last_weights: tuple[float, ...] = ()
    all_turnovers: tuple[float, ...] = ()
    rebalance_count: int = 0
    all_net_exposures: tuple[float, ...] = ()
    rets_baseline_ew: tuple[float, ...] = ()
    fold_attributions: tuple[Layer2FoldAttribution, ...] = ()
    deployable_score: Layer2DeployableScore | None = None


@dataclass(slots=True, frozen=True)
class Layer2DeployableScore:
    """Blocked fallback candidate deployability diagnostic."""

    cagr: float
    sortino: float
    sharpe: float
    calmar: float
    mdd: float
    fold_pass_ratio: float
    score: float
    worst_fold_cagr: float
    positive_block_delta_ratio: float
    cost_drag: float
    bucket_reliability_mean: float
    entry_spike_penalty: float


@dataclass(slots=True, frozen=True)
class RegimeBucketReliability:
    regime: int
    family: str
    tf: str
    fit_edge_bps: float
    cal_edge_bps: float
    n_fit: int
    n_cal: int
    sign_consistent: bool
    reliability: float
    action: RegimeBucketAction


@dataclass(slots=True, frozen=True)
class RegimePolicyEffectSummary:
    n_bars: int
    n_sleeves: int
    action_ratio: float
    pooled_ratio: float
    block_ratio: float
    mu_abs_ratio: float
    quality_weight_ratio: float
    edge_abs_ratio: float


@dataclass(slots=True, frozen=True)
class Layer2GateEvaluation:
    """Layer2 Optuna safety constraints and final promotion gate diagnostics."""

    optuna_constraint_values: tuple[float, ...]
    promotion_passed: bool
    promotion_blocker: str
    promotion_constraint_values: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class Layer2StudyResult:
    """Layer2 study 최종 챔피언 선택 결과."""

    best_params: dict[str, object]
    best_trial_number: int | None
    best_evaluation: Layer2TrialEvaluation | None
    dsr: float
    effective_trial_count: float
    completed_trials: int
    feasible_trials: int
    blocker_reason: str
    sim_cache: object | None = None
    awf_folds: Any = None
    eval_memo: dict[Any, Any] | None = None



@dataclass(slots=True, frozen=True)
class Layer2Result:
    """Layer2 AWF 포트폴리오 검증 결과. [ADR_20260704_L2_DIRECTIONAL_VETO]

    [ADR_20260705_TF_VALIDATION_ROOT_CAUSE_CAPTURE]

    Attributes:
        selected_last: 마지막 리밸런스 선택 심볼 집합.
        weights_last: 마지막 리밸런스 비중 (symbol→weight).
        sharpe_hybrid: Diagonal Kelly 전략 Sharpe.
        sharpe_baseline: Equal-weight 기준 Sharpe.
        mdd_hybrid: 전략 최대 낙폭 (양수).
        mdd_baseline: 기준 최대 낙폭 (양수).
        cagr_hybrid: 전략 연율화 복리 수익률 (비용 차감 후).
        cagr_baseline: 1/N 기준 CAGR.
        mar_hybrid: 전략 MAR ratio (CAGR / MDD).
        mar_baseline: 1/N 기준 MAR ratio.
        fold_pass_ratio: 복리 기준 수익 fold 비율 (prod(1+r)>1).
        turnover: 평균 단방향 회전율.
        friction_pass_pct: 마찰 허들 통과 심볼 비율 (진단용).
        gate_passed: L2 통과 여부.
        blocker_reason: 실패 원인 키. "" = 통과. 값: no_deployment/low_trades/cagr/sharpe_abs/
            sortino/mar/mdd_abs/cvar_95/fold/active_blocks/friction/growth_lcb/uplift.
        deploy_leverage: champion L* applied to hybrid holdout returns.
        recent_fold_passed: 최신 non-empty fold deployed CAGR 양수 여부.
        recent_fold_sharpe: 최신 non-empty fold Sharpe.
        recent_fold_cagr: 최신 non-empty fold deployed CAGR.
        recent_fold_mdd: 최신 non-empty fold deployed MDD.
        master_tf: Annualization timeframe used for deployment metrics (SSOT).
        mean_trend_efficiency: [ADR_20260704_L3_REGIME] OOS-bar-weighted mean Kaufman
            ER across fit/cal AWF folds (diagnostics-only). Requires L2_DIAG_ATTR=1,
            else 0.0 (uncollected).
        trend_efficiency_corr: OOS-bar-weighted mean ER-return correlation across
            fit/cal AWF folds (diagnostics-only). Requires L2_DIAG_ATTR=1.
        realized_price_long: [ADR_20260704_L2L3_LONGSHORT] Summed long-leg realized
            price P&L across fit/cal AWF folds (diagnostics-only, always-on).
        realized_price_short: Summed short-leg realized price P&L across fit/cal
            AWF folds (diagnostics-only, always-on).
        realized_price_long_by_symbol: [ADR_20260704_L2L3_PERSYMBOL] Per-symbol
            long-leg realized price P&L, merged across fit/cal AWF folds
            (diagnostics-only, always-on).
        realized_price_short_by_symbol: Per-symbol short-leg realized price P&L,
            merged across fit/cal AWF folds (diagnostics-only, always-on).
        major_symbol_diag: [ADR_20260704_L3_MAJORDIAG] Per-symbol signal-vs-sizing
            mismatch ratios (mu_bullish_pct, weight_long_pct, stale_long_pct,
            regime_cap_engaged_pct, mean_regime_risk_mult_when_long) for
            MAJOR_DIAG_SYMBOLS, merged across fit/cal AWF folds (diagnostics-only,
            always-on).
    """

    selected_last: frozenset[str]
    weights_last: dict[str, float]
    sharpe_hybrid: float
    sharpe_baseline: float
    mdd_hybrid: float
    mdd_baseline: float
    cagr_hybrid: float
    cagr_baseline: float
    mar_hybrid: float
    mar_baseline: float
    fold_pass_ratio: float
    turnover: float
    friction_pass_pct: float
    gate_passed: bool
    blocker_reason: str
    allocation_policy: AllocationPolicy = "diagonal_kelly"
    deploy_leverage: float = 1.0
    psr_hybrid: float = 0.0
    growth_lcb_hybrid: float = 0.0
    growth_lcb_baseline: float = 0.0
    sharpe_hac_hybrid: float = 0.0
    sharpe_hac_baseline: float = 0.0
    dsr_hybrid: float = 0.0
    cvar_95_hybrid: float = 0.0
    average_gross_exposure: float = 0.0
    average_net_exposure: float = 0.0
    cap_saturation_ratio: float = 0.0
    total_cost_bps: float = 0.0
    n_rebalances: int = 0
    block_metrics: tuple[Layer2BlockMetric, ...] = ()
    sortino_hybrid: float = 0.0
    terminal_multiple: float = 1.0
    total_pnl_pct: float = 0.0
    trade_count: int = 0
    risk_utilization: float = 0.0
    recent_fold_passed: bool | None = None
    recent_fold_sharpe: float = 0.0
    recent_fold_cagr: float = 0.0
    recent_fold_mdd: float = 0.0
    master_tf: str = "4h"
    mean_trend_efficiency: float = 0.0
    trend_efficiency_corr: float = 0.0
    realized_price_long: float = 0.0
    realized_price_short: float = 0.0
    realized_price_long_by_symbol: tuple[tuple[str, float], ...] = ()
    realized_price_short_by_symbol: tuple[tuple[str, float], ...] = ()
    major_symbol_diag: tuple[MajorSymbolSignalSizingSummary, ...] = ()
    major_symbol_sleeve_diag: tuple[MajorSymbolSleeveContributionSummary, ...] = ()
    major_symbol_incoherence: tuple[MajorSymbolIncoherenceSummary, ...] = ()
    directional_veto_summary: tuple[DirectionalVetoSummary, ...] = ()
    validation_parity_report: ValidationParityReport | None = None


@dataclass(slots=True, frozen=True)
class Layer2AllocationConfig:
    """Typed Layer2 allocation and gate configuration. [ADR_20260704_L2_DIRECTIONAL_VETO]"""

    k_rank: int = 3
    rebalance_bars: int = 3
    kelly_fraction: float = 0.25
    min_abs_rank_z: float = 0.0
    rank_buffer: int = 1
    no_trade_band: float = 0.01
    max_ann_vol: float | None = None
    l2_min_cagr: float = 0.30
    l2_min_mar: float = 1.0
    l2_min_sortino: float = 1.5
    l2_min_sharpe_abs: float = 0.7
    l2_min_calmar: float = 0.5
    l2_max_mdd_abs: float = 0.30
    l2_mdd_material_floor: float = 0.05
    l2_mdd_rel_tol: float = 0.25
    l2_min_fold_pass_ratio: float = 0.60
    l2_min_sharpe_uplift: float = 0.05
    l2_min_growth_uplift: float = 0.0
    l2_min_psr: float = 0.90
    l2_min_friction_pass: float = 0.50
    fixed_cost_safety_mult: float = 1.25
    l2_min_dsr: float = 0.60
    l2_max_cvar_95: float = 0.06
    l2_min_active_blocks: int = 3
    l2_min_sortino_abs: float = 1.5
    l2_min_trades: int = 30
    l2_growth_lcb_z: float = 0.5
    edge_throttle_enabled: bool = True
    edge_floor_bps: float = 0.0
    edge_ref_bps: float = 5.0
    edge_throttle_gamma: float = 1.0
    deploy_cost_safety_mult: float = 1.25
    edge_throttle_min_active_mult: float = 0.0
    risk_budget_floor_ratio: float = 0.0
    risk_budget_max_scale: float = 3.0
    adaptive_breadth_enabled: bool = False
    adaptive_k_extra: int = 0
    adaptive_expand_below_vol_ratio: float = 0.0
    l2_objective_risk_util_target: float = 0.50
    l2_objective_risk_util_weight: float = 0.03
    l2_objective_trade_target: int = 90
    l2_objective_trade_weight: float = 0.02
    l2_replay_max_fallbacks: int = 24
    l2_worst_fold_penalty_threshold: float = -0.30
    l2_worst_fold_penalty_weight: float = 0.005
    l2_min_worst_fold_cagr: float = -0.05
    l2_min_positive_block_delta_ratio: float = 0.45
    l2_worst_fold_cagr_penalty_weight: float = 0.50
    l2_block_delta_penalty_weight: float = 0.25
    # D3: 결정론적 리스크 배치 파라미터 (fit-leg 기반, look-ahead-free)
    l2_deploy_enabled: bool = True
    l2_deploy_mdd_margin: float = 0.30
    l2_deploy_cvar_margin: float = 0.20
    # fit-leg 사용 시 MDD/CVaR 예산이 실제 binding → hard_cap 완화해도 안전.
    # OOS 대리(fallback) 시에도 mdd_margin=0.30이 완충.
    l2_deploy_l_hard_cap: float = 20.0
    # RC-2 crisis gate: fit-leg unit-vol MDD 이 값 이상이면 oos_blend 억제. None=비활성.
    l2_deploy_fit_mdd_crisis_gate: float | None = None
    # 거래소 실행가능 notional 레버리지 상한 (None=무제한). Binance perp 기본 10x.
    l2_max_exchange_leverage: float | None = 10.0
    l2_require_recent_fold_pass: bool = True
    l2_min_recent_fold_sharpe: float = 0.0
    l2_is_expansion_bars: int = 0
    l2_sleeve_combine_method: str = "precision_weighted"
    l2_sleeve_conviction_cap_mult: float = 1.5
    l2_diag_attribution_enabled: bool = False
    l2_diag_sleeve_top_k: int = 15
    l2_diag_sleeve_sample_every: int = 0
    l2_max_cost_drag_ratio: float = 0.60
    l2_turnover_penalty_weight: float = 0.0
    l2_tf_inclusion_enabled: bool = True
    l2_tf_inclusion_min_edge: float = 0.0
    l2_routing_mode: Literal["pool", "bucket"] = "bucket"
    l2_bucket_cost_bps: float = 6.0
    l2_bucket_min_n: int = 15
    l2_bucket_shrinkage: float = 0.3
    l2_bucket_edge_floor_bps: float = 50.0
    l2_bucket_min_reliability: float = 0.55
    # Portfolio covariance mode for diagonal_kelly_weights
    l2_portfolio_cov_mode: Literal["diagonal", "correlated"] = "diagonal"
    l2_portfolio_cov_lookback_bars: int = 180
    l2_portfolio_cov_min_obs: int = 20
    # L* concentration gate (correlation-clustering 기반, cov_mode와 독립)
    l2_leverage_diversification_gate_enabled: bool = False
    l2_leverage_concentration_recent_window_bars: int = 60
    l2_leverage_concentration_floor: float | None = None
    # CS Score Amplification (anti-Kelly=EW-convergence) — 중단 (효과 없음 입증됨)
    l2_cs_amp_enabled: bool = False
    # Breadth-selection mode: True=use all valid symbols (no rank_and_select alpha sorting)
    l2_selection_breadth_mode: bool = False
    l2_cs_amp_alpha: float = 2.0
    l2_cs_amp_mode: str = "power"
    l2_cs_amp_power: float = 2.0
    # Regime State Compression (6→3) for quality improvement
    l2_regime_compression_enabled: bool = True
    l2_regime_proof_enabled: bool = True
    l2_regime_proof_nw_tstat: float = 1.5
    l2_regime_proof_fold_pass_ratio: float = 0.60
    l2_regime_fallback_mode: Literal["pooled", "empty"] = "pooled"
    l2_regime_policy_mode: RegimePolicyMode = "soft"
    l2_regime_cal_min_n: int = 20
    l2_regime_min_cal_lift_bps: float = 8.0
    l2_regime_block_lift_bps: float = -12.0
    l2_regime_soft_downweight_min: float = 0.50
    l2_regime_soft_downweight_max: float = 1.0
    l2_regime_min_policy_confidence: float = 0.55
    l2_regime_hard_block_enabled: bool = False
    l2_regime_block_min_confidence: float = 0.80
    l2_regime_require_sign_consistency: bool = True
    l2_regime_scale_signal_mu: bool = True
    l2_regime_scale_quality_weight: bool = True
    l2_regime_max_pooled_ratio_for_effective: float = 0.80
    l2_regime_min_action_ratio_for_effective: float = 0.10
    l2_regime_min_mu_abs_change: float = 0.03
    l2_regime_risk_cap_enabled: bool = True
    l2_regime_bull_gross_cap: float = 1.0
    l2_regime_bear_gross_cap: float = 0.35
    l2_regime_crisis_gross_cap: float = 0.25
    # L3: Walk-forward 적응형 Regime-Reliability (A/B baseline=off)
    l2_regime_reliability_enabled: bool = False
    l2_regime_reliability_window: int = 2
    l2_regime_reliability_floor: float = 0.2
    l2_entry_cooldown_bars: int = 12
    l2_entry_spike_penalty_weight: float = 0.05
    l2_entry_spike_warn_threshold: float = 0.20
    # Regime policy conservatism relaxation
    l2_regime_pooled_is_passthrough: bool = True
    l2_regime_min_fit_n_floor: int = 5
    l2_regime_require_fit_n_for_downweight: bool = False
    # L2 positioning-crowding dampener (l1-positioning-crowding-dampener.md)
    l2_crowding_persistence_bars: int = 3
    l2_crowding_recovery_cooldown_bars: int = 3
    l2_crowding_floor_mult: float | None = None
    # L2 regime directional veto
    l2_regime_directional_veto_enabled: bool = False
    l2_regime_directional_veto_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    l2_regime_directional_veto_adverse_codes: tuple[int, ...] = (1, 2)
    l2_regime_directional_veto_long_eps_bps: float = 0.0
    l2_regime_directional_veto_action: Literal["drop_long", "zero_mu", "cap_mu"] = "drop_long"
    l2_regime_directional_veto_mode: Literal["adverse_only", "contextual"] = "adverse_only"
    l2_regime_directional_veto_persistence_bars: int = 3
    l2_regime_directional_veto_loss_lookback_bars: int = 18
    l2_regime_directional_veto_loss_trigger_bps: float = 150.0
    l2_regime_directional_veto_cap_mu_bps: float = 0.0
    l2_regime_directional_veto_release_raw_mu_nonpos: bool = True
    l2_regime_directional_veto_release_regime_bull_bars: int = 2
    l2_regime_directional_veto_cooldown_bars: int = 3
    l2_regime_directional_veto_max_fit_net_value_loss: float = 0.0
    l2_regime_directional_veto_min_l3_total_return_delta: float = 0.02
    l2_regime_directional_veto_max_l2_cagr_delta_loss: float = 0.005
    l2_regime_directional_veto_max_fit_false_positive_rate: float = 0.50
    l2_regime_directional_veto_max_turnover_delta: float = 0.05
    l2_regime_directional_veto_min_gross_ratio: float = 0.90
    # L1 Intra-Symbol Divergence Dampener (Track 1 — BTC)
    l2_intra_symbol_divergence_enabled: bool = False
    l2_intra_symbol_divergence_symbols: tuple[str, ...] = ("BTCUSDT",)
    l2_intra_symbol_divergence_dominant_families: tuple[str, ...] = ("dual_momentum", "supertrend")
    l2_intra_symbol_divergence_persistence_bars: int = 3
    l2_intra_symbol_divergence_release_bars: int = 2
    l2_intra_symbol_divergence_cooldown_bars: int = 3
    l2_intra_symbol_divergence_dominant_damp_mult: float = 0.5
    l2_intra_symbol_divergence_dissent_boost_mult: float = 2.0

    @staticmethod
    def _as_int(value: object, default: int) -> int:
        if isinstance(value, (int, float, str)):
            return int(value)
        return default

    @staticmethod
    def _as_float(value: object, default: float) -> float:
        if isinstance(value, (int, float, str)):
            return float(value)
        return default

    @staticmethod
    def _validate_range(name: str, value: float, lower: float, upper: float | None = None) -> float:
        if value < lower or (upper is not None and value > upper):
            suffix = f", {upper}" if upper is not None else ", inf"
            raise ValueError(f"{name} must be in range [{lower}{suffix}]")
        return value

    @classmethod
    def from_mapping(cls, params: dict[str, object] | None) -> Layer2AllocationConfig:
        _dc = _L2_DEFAULT_CONFIG  # SSOT shortcut
        params = params or {}
        if "friction_safety_mult" in params:
            raise ValueError("friction_safety_mult is deprecated; use fixed_cost_safety_mult")
        raw_vol_target = params.get("max_ann_vol", params.get("vol_target"))
        vol_target = float(raw_vol_target) if isinstance(raw_vol_target, (int, float)) else None
        min_abs_rank_z = params.get("min_abs_rank_z", params.get("CS_Z_SCORE_THRESHOLD", 0.0))
        fixed_cost_safety_mult = cls._validate_range(
            "fixed_cost_safety_mult",
            cls._as_float(params.get("fixed_cost_safety_mult", 1.25), 1.25),
            1.0,
        )
        deploy_cost_safety_mult = cls._validate_range(
            "deploy_cost_safety_mult",
            cls._as_float(params.get("deploy_cost_safety_mult", 1.25), 1.25),
            1.0,
        )
        edge_throttle_min_active_mult = cls._validate_range(
            "edge_throttle_min_active_mult",
            cls._as_float(params.get("edge_throttle_min_active_mult", 0.0), 0.0),
            0.0,
            1.0,
        )
        risk_budget_floor_ratio = cls._validate_range(
            "risk_budget_floor_ratio",
            cls._as_float(params.get("risk_budget_floor_ratio", 0.0), 0.0),
            0.0,
            1.0,
        )
        risk_budget_max_scale = cls._validate_range(
            "risk_budget_max_scale",
            cls._as_float(params.get("risk_budget_max_scale", 3.0), 3.0),
            1.0,
        )
        adaptive_k_extra = int(
            cls._validate_range(
                "adaptive_k_extra",
                cls._as_int(params.get("adaptive_k_extra", 0), 0),
                0,
            )
        )
        adaptive_expand_below_vol_ratio = cls._validate_range(
            "adaptive_expand_below_vol_ratio",
            cls._as_float(params.get("adaptive_expand_below_vol_ratio", 0.0), 0.0),
            0.0,
            1.0,
        )
        raw_exchange_cap = params.get("l2_max_exchange_leverage", 10.0)
        if "l2_max_exchange_leverage" not in params:
            l2_max_exchange_leverage: float | None = 10.0
        elif raw_exchange_cap is None:
            l2_max_exchange_leverage = None
        else:
            l2_max_exchange_leverage = cls._validate_range(
                "l2_max_exchange_leverage",
                cls._as_float(raw_exchange_cap, 10.0),
                0.0,
            )
        _raw_fit_mdd_crisis_gate = params.get("l2_deploy_fit_mdd_crisis_gate")
        l2_deploy_fit_mdd_crisis_gate: float | None = (
            cls._validate_range(
                "l2_deploy_fit_mdd_crisis_gate",
                cls._as_float(_raw_fit_mdd_crisis_gate, 0.0),
                0.0,
                1.0,
            )
            if _raw_fit_mdd_crisis_gate is not None
            else None
        )
        l2_objective_risk_util_target = cls._validate_range(
            "l2_objective_risk_util_target",
            cls._as_float(params.get("l2_objective_risk_util_target", 0.50), 0.50),
            0.0,
            1.0,
        )
        if l2_objective_risk_util_target <= 0.0:
            raise ValueError("l2_objective_risk_util_target must be in range (0.0, 1.0]")
        l2_objective_risk_util_weight = cls._validate_range(
            "l2_objective_risk_util_weight",
            cls._as_float(params.get("l2_objective_risk_util_weight", 0.03), 0.03),
            0.0,
        )
        l2_objective_trade_target = int(
            cls._validate_range(
                "l2_objective_trade_target",
                cls._as_int(params.get("l2_objective_trade_target", 90), 90),
                1,
            )
        )
        l2_objective_trade_weight = cls._validate_range(
            "l2_objective_trade_weight",
            cls._as_float(params.get("l2_objective_trade_weight", 0.02), 0.02),
            0.0,
        )
        l2_replay_max_fallbacks = int(
            cls._validate_range(
                "l2_replay_max_fallbacks",
                cls._as_int(params.get("l2_replay_max_fallbacks", 24), 24),
                1,
            )
        )
        l2_min_worst_fold_cagr = cls._as_float(
            params.get("l2_min_worst_fold_cagr", _dc.l2_min_worst_fold_cagr),
            _dc.l2_min_worst_fold_cagr,
        )
        l2_min_positive_block_delta_ratio = cls._validate_range(
            "l2_min_positive_block_delta_ratio",
            cls._as_float(
                params.get(
                    "l2_min_positive_block_delta_ratio",
                    _dc.l2_min_positive_block_delta_ratio,
                ),
                _dc.l2_min_positive_block_delta_ratio,
            ),
            0.0,
            1.0,
        )
        l2_worst_fold_cagr_penalty_weight = cls._validate_range(
            "l2_worst_fold_cagr_penalty_weight",
            cls._as_float(
                params.get(
                    "l2_worst_fold_cagr_penalty_weight",
                    _dc.l2_worst_fold_cagr_penalty_weight,
                ),
                _dc.l2_worst_fold_cagr_penalty_weight,
            ),
            0.0,
        )
        l2_block_delta_penalty_weight = cls._validate_range(
            "l2_block_delta_penalty_weight",
            cls._as_float(
                params.get(
                    "l2_block_delta_penalty_weight",
                    _dc.l2_block_delta_penalty_weight,
                ),
                _dc.l2_block_delta_penalty_weight,
            ),
            0.0,
        )
        combine_method = str(
            os.environ.get("L2_SLEEVE_COMBINE")
            or params.get("l2_sleeve_combine_method", "precision_weighted")
        )
        if combine_method not in {"precision_weighted", "equal", "max_edge"}:
            raise ValueError(
                f"l2_sleeve_combine_method must be one of precision_weighted/equal/max_edge, "
                f"got {combine_method!r}"
            )
        conviction_cap_mult = cls._validate_range(
            "l2_sleeve_conviction_cap_mult",
            cls._as_float(params.get("l2_sleeve_conviction_cap_mult", 1.5), 1.5),
            1.0,
            3.0,
        )
        raw_fallback_mode = str(params.get("l2_regime_fallback_mode", "pooled"))
        if raw_fallback_mode not in {"pooled", "empty"}:
            raise ValueError("l2_regime_fallback_mode must be one of pooled/empty")
        fallback_mode = cast(Literal["pooled", "empty"], raw_fallback_mode)
        raw_policy_mode = str(params.get("l2_regime_policy_mode", _dc.l2_regime_policy_mode))
        if raw_policy_mode not in {"filter", "observe", "soft", "hybrid"}:
            raise ValueError("l2_regime_policy_mode must be one of filter/observe/soft/hybrid")
        policy_mode = cast(RegimePolicyMode, raw_policy_mode)
        l2_regime_cal_min_n = int(
            cls._validate_range(
                "l2_regime_cal_min_n",
                cls._as_int(params.get("l2_regime_cal_min_n", 20), 20),
                1,
            )
        )
        l2_regime_soft_downweight_min = cls._validate_range(
            "l2_regime_soft_downweight_min",
            cls._as_float(
                params.get("l2_regime_soft_downweight_min", _dc.l2_regime_soft_downweight_min),
                _dc.l2_regime_soft_downweight_min,
            ),
            0.0,
            1.0,
        )
        l2_regime_soft_downweight_max = cls._validate_range(
            "l2_regime_soft_downweight_max",
            cls._as_float(
                params.get("l2_regime_soft_downweight_max", _dc.l2_regime_soft_downweight_max),
                _dc.l2_regime_soft_downweight_max,
            ),
            0.0,
            1.0,
        )
        if l2_regime_soft_downweight_min > l2_regime_soft_downweight_max:
            raise ValueError(
                "l2_regime_soft_downweight_min must be <= l2_regime_soft_downweight_max"
            )
        l2_regime_block_min_confidence = cls._validate_range(
            "l2_regime_block_min_confidence",
            cls._as_float(
                params.get("l2_regime_block_min_confidence", _dc.l2_regime_block_min_confidence),
                _dc.l2_regime_block_min_confidence,
            ),
            0.0,
            1.0,
        )
        l2_regime_bull_gross_cap = cls._validate_range(
            "l2_regime_bull_gross_cap",
            cls._as_float(
                params.get("l2_regime_bull_gross_cap", _dc.l2_regime_bull_gross_cap),
                _dc.l2_regime_bull_gross_cap,
            ),
            0.0,
            1.0,
        )
        l2_regime_bear_gross_cap = cls._validate_range(
            "l2_regime_bear_gross_cap",
            cls._as_float(
                params.get("l2_regime_bear_gross_cap", _dc.l2_regime_bear_gross_cap),
                _dc.l2_regime_bear_gross_cap,
            ),
            0.0,
            1.0,
        )
        l2_regime_crisis_gross_cap = cls._validate_range(
            "l2_regime_crisis_gross_cap",
            cls._as_float(
                params.get("l2_regime_crisis_gross_cap", _dc.l2_regime_crisis_gross_cap),
                _dc.l2_regime_crisis_gross_cap,
            ),
            0.0,
            1.0,
        )
        if l2_regime_bull_gross_cap <= 0.0:
            raise ValueError("l2_regime_bull_gross_cap must be in range (0.0, 1.0]")
        if l2_regime_bear_gross_cap <= 0.0:
            raise ValueError("l2_regime_bear_gross_cap must be in range (0.0, 1.0]")
        if l2_regime_crisis_gross_cap <= 0.0:
            raise ValueError("l2_regime_crisis_gross_cap must be in range (0.0, 1.0]")
        _reliability_env = os.environ.get("L2_REGIME_RELIABILITY", "")
        l2_regime_reliability_enabled = (
            _reliability_env not in ("", "0", "false", "False")
            if _reliability_env != ""
            else bool(
                params.get("l2_regime_reliability_enabled", _dc.l2_regime_reliability_enabled)
            )
        )
        l2_regime_reliability_window = max(
            1,
            int(
                cls._as_int(
                    params.get("l2_regime_reliability_window", _dc.l2_regime_reliability_window),
                    _dc.l2_regime_reliability_window,
                )
            ),
        )
        l2_regime_reliability_floor = cls._as_float(
            params.get("l2_regime_reliability_floor", _dc.l2_regime_reliability_floor), _dc.l2_regime_reliability_floor,
        )
        if l2_regime_reliability_floor <= 0.0 or l2_regime_reliability_floor > 1.0:
            raise ValueError("l2_regime_reliability_floor must be in range (0.0, 1.0]")
        return cls(
            k_rank=cls._as_int(params.get("K_RANK", 3), 3),
            rebalance_bars=cls._as_int(params.get("REBALANCE_BARS", 3), 3),
            kelly_fraction=cls._as_float(params.get("kelly_fraction", 0.25), 0.25),
            min_abs_rank_z=cls._as_float(min_abs_rank_z, 0.0),
            rank_buffer=cls._as_int(params.get("rank_buffer", 1), 1),
            no_trade_band=cls._as_float(params.get("no_trade_band", 0.01), 0.01),
            max_ann_vol=vol_target,
            l2_min_cagr=cls._as_float(params.get("l2_min_cagr", _dc.l2_min_cagr), _dc.l2_min_cagr),
            l2_min_mar=cls._as_float(params.get("l2_min_mar", _dc.l2_min_mar), _dc.l2_min_mar),
            l2_min_sortino=cls._as_float(
                params.get("l2_min_sortino", params.get("l2_min_sortino_abs", _dc.l2_min_sortino)),
                _dc.l2_min_sortino,
            ),
            l2_min_sharpe_abs=cls._as_float(
                params.get("l2_min_sharpe_abs", _dc.l2_min_sharpe_abs), _dc.l2_min_sharpe_abs,
            ),
            l2_min_calmar=cls._as_float(params.get("l2_min_calmar", _dc.l2_min_calmar), _dc.l2_min_calmar),
            l2_max_mdd_abs=cls._as_float(params.get("l2_max_mdd_abs", _dc.l2_max_mdd_abs), _dc.l2_max_mdd_abs),
            l2_mdd_material_floor=cls._as_float(
                params.get("l2_mdd_material_floor", _dc.l2_mdd_material_floor), _dc.l2_mdd_material_floor,
            ),
            l2_mdd_rel_tol=cls._as_float(params.get("l2_mdd_rel_tol", _dc.l2_mdd_rel_tol), _dc.l2_mdd_rel_tol),
            l2_min_fold_pass_ratio=cls._as_float(
                params.get("l2_min_fold_pass_ratio", _dc.l2_min_fold_pass_ratio), _dc.l2_min_fold_pass_ratio,
            ),
            l2_min_sharpe_uplift=cls._as_float(
                params.get("l2_min_sharpe_uplift", _dc.l2_min_sharpe_uplift), _dc.l2_min_sharpe_uplift,
            ),
            l2_min_growth_uplift=cls._as_float(
                params.get("l2_min_growth_uplift", _dc.l2_min_growth_uplift), _dc.l2_min_growth_uplift,
            ),
            l2_min_psr=cls._as_float(params.get("l2_min_psr", _dc.l2_min_psr), _dc.l2_min_psr),
            l2_min_friction_pass=cls._as_float(
                params.get("l2_min_friction_pass", _dc.l2_min_friction_pass), _dc.l2_min_friction_pass,
            ),
            fixed_cost_safety_mult=fixed_cost_safety_mult,
            l2_min_dsr=cls._as_float(params.get("l2_min_dsr", _dc.l2_min_dsr), _dc.l2_min_dsr),
            l2_max_cvar_95=cls._as_float(params.get("l2_max_cvar_95", _dc.l2_max_cvar_95), _dc.l2_max_cvar_95),
            l2_min_active_blocks=cls._as_int(
                params.get("l2_min_active_blocks", _dc.l2_min_active_blocks), _dc.l2_min_active_blocks,
            ),
            l2_min_sortino_abs=cls._as_float(
                params.get("l2_min_sortino_abs", _dc.l2_min_sortino_abs), _dc.l2_min_sortino_abs,
            ),
            l2_min_trades=cls._as_int(params.get("l2_min_trades", _dc.l2_min_trades), _dc.l2_min_trades),
            l2_growth_lcb_z=cls._as_float(params.get("l2_growth_lcb_z", _dc.l2_growth_lcb_z), _dc.l2_growth_lcb_z),
            edge_throttle_enabled=bool(params.get("edge_throttle_enabled", True)),
            edge_floor_bps=cls._as_float(params.get("edge_floor_bps", 0.0), 0.0),
            edge_ref_bps=cls._as_float(params.get("edge_ref_bps", 5.0), 5.0),
            edge_throttle_gamma=cls._as_float(params.get("edge_throttle_gamma", 1.0), 1.0),
            deploy_cost_safety_mult=deploy_cost_safety_mult,
            edge_throttle_min_active_mult=edge_throttle_min_active_mult,
            risk_budget_floor_ratio=risk_budget_floor_ratio,
            risk_budget_max_scale=risk_budget_max_scale,
            adaptive_breadth_enabled=bool(params.get("adaptive_breadth_enabled", False)),
            adaptive_k_extra=adaptive_k_extra,
            adaptive_expand_below_vol_ratio=adaptive_expand_below_vol_ratio,
            l2_objective_risk_util_target=l2_objective_risk_util_target,
            l2_objective_risk_util_weight=l2_objective_risk_util_weight,
            l2_objective_trade_target=l2_objective_trade_target,
            l2_objective_trade_weight=l2_objective_trade_weight,
            l2_replay_max_fallbacks=l2_replay_max_fallbacks,
            l2_min_worst_fold_cagr=l2_min_worst_fold_cagr,
            l2_min_positive_block_delta_ratio=l2_min_positive_block_delta_ratio,
            l2_worst_fold_cagr_penalty_weight=l2_worst_fold_cagr_penalty_weight,
            l2_block_delta_penalty_weight=l2_block_delta_penalty_weight,
            l2_worst_fold_penalty_threshold=cls._as_float(
                params.get("l2_worst_fold_penalty_threshold", -0.30), -0.30
            ),
            l2_worst_fold_penalty_weight=cls._as_float(
                params.get("l2_worst_fold_penalty_weight", 0.005), 0.005
            ),
            l2_deploy_enabled=bool(params.get("l2_deploy_enabled", True)),
            l2_deploy_mdd_margin=cls._as_float(params.get("l2_deploy_mdd_margin", 0.30), 0.30),
            l2_deploy_cvar_margin=cls._as_float(params.get("l2_deploy_cvar_margin", 0.20), 0.20),
            l2_deploy_l_hard_cap=cls._as_float(params.get("l2_deploy_l_hard_cap", 20.0), 20.0),
            l2_deploy_fit_mdd_crisis_gate=l2_deploy_fit_mdd_crisis_gate,
            l2_max_exchange_leverage=l2_max_exchange_leverage,
            l2_require_recent_fold_pass=bool(params.get("l2_require_recent_fold_pass", True)),
            l2_min_recent_fold_sharpe=cls._as_float(
                params.get("l2_min_recent_fold_sharpe", 0.0),
                0.0,
            ),
            l2_is_expansion_bars=cls._as_int(params.get("l2_is_expansion_bars", 0), 0),
            l2_sleeve_combine_method=combine_method,
            l2_sleeve_conviction_cap_mult=conviction_cap_mult,
            l2_diag_attribution_enabled=bool(params.get("l2_diag_attribution_enabled", False))
            or os.environ.get("L2_DIAG_ATTR", "") not in ("", "0", "false", "False"),
            l2_diag_sleeve_top_k=cls._as_int(params.get("l2_diag_sleeve_top_k", 15), 15),
            l2_diag_sleeve_sample_every=cls._as_int(params.get("l2_diag_sleeve_sample_every", 0), 0),
            l2_max_cost_drag_ratio=cls._validate_range(
                "l2_max_cost_drag_ratio",
                cls._as_float(params.get("l2_max_cost_drag_ratio", 0.60), 0.60),
                0.0,
            ),
            l2_turnover_penalty_weight=cls._validate_range(
                "l2_turnover_penalty_weight",
                cls._as_float(params.get("l2_turnover_penalty_weight", 0.0), 0.0),
                0.0,
            ),
            l2_tf_inclusion_enabled=bool(params.get("l2_tf_inclusion_enabled", True)),
            l2_tf_inclusion_min_edge=cls._as_float(
                params.get("l2_tf_inclusion_min_edge", 0.0), 0.0
            ),
            l2_routing_mode=(
                "bucket"
                if str(
                    os.environ.get(
                        "L2_ROUTING_MODE",
                        params.get("l2_routing_mode", params.get("L2_ROUTING_MODE", "bucket")),
                    )
                )
                == "bucket"
                else "pool"
            ),
            l2_bucket_cost_bps=cls._as_float(params.get("l2_bucket_cost_bps", 6.0), 6.0),
            l2_bucket_min_n=cls._as_int(params.get("l2_bucket_min_n", 15), 15),
            l2_bucket_shrinkage=cls._validate_range(
                "l2_bucket_shrinkage",
                cls._as_float(params.get("l2_bucket_shrinkage", 0.3), 0.3),
                0.0,
                1.0,
            ),
            l2_bucket_edge_floor_bps=cls._as_float(
                os.environ.get("L2_BUCKET_EDGE_FLOOR_BPS", params.get("l2_bucket_edge_floor_bps", 0.0)), 0.0
            ),
            l2_bucket_min_reliability=cls._validate_range(
                "l2_bucket_min_reliability",
                cls._as_float(params.get("l2_bucket_min_reliability", 0.55), 0.55),
                0.0,
                1.0,
            ),
            l2_portfolio_cov_mode=(
                "correlated"
                if str(
                    os.environ.get("L2_PORTFOLIO_COV_MODE")
                    or params.get("l2_portfolio_cov_mode", "diagonal"),
                ).strip().lower() == "correlated"
                else "diagonal"
            ),
            l2_portfolio_cov_lookback_bars=cls._as_int(
                params.get("l2_portfolio_cov_lookback_bars", 180), 180,
            ),
            l2_portfolio_cov_min_obs=cls._as_int(
                params.get("l2_portfolio_cov_min_obs", 20), 20,
            ),
            l2_regime_compression_enabled=bool(params.get("l2_regime_compression_enabled", True)),
            l2_regime_proof_enabled=bool(params.get("l2_regime_proof_enabled", True)),
            l2_regime_proof_nw_tstat=cls._validate_range(
                "l2_regime_proof_nw_tstat",
                cls._as_float(params.get("l2_regime_proof_nw_tstat", 1.5), 1.5),
                0.0,
            ),
            l2_regime_proof_fold_pass_ratio=cls._validate_range(
                "l2_regime_proof_fold_pass_ratio",
                cls._as_float(params.get("l2_regime_proof_fold_pass_ratio", 0.60), 0.60),
                0.0,
                1.0,
            ),
            l2_regime_fallback_mode=fallback_mode,
            l2_regime_policy_mode=policy_mode,
            l2_regime_cal_min_n=l2_regime_cal_min_n,
            l2_regime_min_cal_lift_bps=cls._as_float(
                params.get("l2_regime_min_cal_lift_bps", 8.0), 8.0,
            ),
            l2_regime_block_lift_bps=cls._as_float(
                params.get("l2_regime_block_lift_bps", -12.0), -12.0,
            ),
            l2_regime_soft_downweight_min=l2_regime_soft_downweight_min,
            l2_regime_soft_downweight_max=l2_regime_soft_downweight_max,
            l2_regime_min_policy_confidence=cls._validate_range(
                "l2_regime_min_policy_confidence",
                cls._as_float(params.get("l2_regime_min_policy_confidence", 0.55), 0.55),
                0.0,
                1.0,
            ),
            l2_regime_hard_block_enabled=bool(
                params.get("l2_regime_hard_block_enabled", _dc.l2_regime_hard_block_enabled)
            ),
            l2_regime_block_min_confidence=l2_regime_block_min_confidence,
            l2_regime_require_sign_consistency=bool(
                params.get("l2_regime_require_sign_consistency", _dc.l2_regime_require_sign_consistency)
            ),
            l2_regime_scale_signal_mu=bool(
                params.get("l2_regime_scale_signal_mu", _dc.l2_regime_scale_signal_mu)
            ),
            l2_regime_scale_quality_weight=bool(
                params.get("l2_regime_scale_quality_weight", _dc.l2_regime_scale_quality_weight)
            ),
            l2_regime_max_pooled_ratio_for_effective=cls._validate_range(
                "l2_regime_max_pooled_ratio_for_effective",
                cls._as_float(params.get("l2_regime_max_pooled_ratio_for_effective", 0.80), 0.80),
                0.0,
                1.0,
            ),
            l2_regime_min_action_ratio_for_effective=cls._validate_range(
                "l2_regime_min_action_ratio_for_effective",
                cls._as_float(params.get("l2_regime_min_action_ratio_for_effective", 0.10), 0.10),
                0.0,
                1.0,
            ),
            l2_regime_min_mu_abs_change=cls._validate_range(
                "l2_regime_min_mu_abs_change",
                cls._as_float(params.get("l2_regime_min_mu_abs_change", 0.03), 0.03),
                0.0,
            ),
            l2_regime_risk_cap_enabled=bool(
                params.get("l2_regime_risk_cap_enabled", _dc.l2_regime_risk_cap_enabled)
            ),
            l2_regime_bull_gross_cap=l2_regime_bull_gross_cap,
            l2_regime_bear_gross_cap=l2_regime_bear_gross_cap,
            l2_regime_crisis_gross_cap=l2_regime_crisis_gross_cap,
            l2_regime_reliability_enabled=l2_regime_reliability_enabled,
            l2_regime_reliability_window=l2_regime_reliability_window,
            l2_regime_reliability_floor=l2_regime_reliability_floor,
            l2_entry_cooldown_bars=cls._as_int(params.get("l2_entry_cooldown_bars", 12), 12),
            l2_entry_spike_penalty_weight=cls._validate_range(
                "l2_entry_spike_penalty_weight",
                cls._as_float(params.get("l2_entry_spike_penalty_weight", 0.05), 0.05),
                0.0,
            ),
            l2_entry_spike_warn_threshold=cls._validate_range(
                "l2_entry_spike_warn_threshold",
                cls._as_float(params.get("l2_entry_spike_warn_threshold", 0.20), 0.20),
                0.0,
                1.0,
            ),
            l2_regime_pooled_is_passthrough=bool(
                params.get("l2_regime_pooled_is_passthrough", _dc.l2_regime_pooled_is_passthrough)
            ),
            l2_regime_min_fit_n_floor=int(
                cls._validate_range(
                    "l2_regime_min_fit_n_floor",
                    cls._as_int(
                        params.get("l2_regime_min_fit_n_floor", _dc.l2_regime_min_fit_n_floor),
                        _dc.l2_regime_min_fit_n_floor,
                    ),
                    0,
                )
            ),
            l2_regime_require_fit_n_for_downweight=bool(
                params.get("l2_regime_require_fit_n_for_downweight", _dc.l2_regime_require_fit_n_for_downweight)
            ),
            l2_crowding_persistence_bars=cls._as_int(
                params.get("l2_crowding_persistence_bars", _dc.l2_crowding_persistence_bars),
                _dc.l2_crowding_persistence_bars,
            ),
            l2_crowding_recovery_cooldown_bars=cls._as_int(
                params.get("l2_crowding_recovery_cooldown_bars", _dc.l2_crowding_recovery_cooldown_bars),
                _dc.l2_crowding_recovery_cooldown_bars,
            ),
            l2_crowding_floor_mult=cls._as_float(
                params.get("l2_crowding_floor_mult", _dc.l2_crowding_floor_mult),
                _dc.l2_crowding_floor_mult if _dc.l2_crowding_floor_mult is not None else 0.0,
            ) or None,
            # L2 regime directional veto
            l2_regime_directional_veto_enabled=bool(
                params.get("l2_regime_directional_veto_enabled", _dc.l2_regime_directional_veto_enabled)
            ),
            l2_regime_directional_veto_mode=_validate_directional_veto_mode(
                str(params.get("l2_regime_directional_veto_mode", _dc.l2_regime_directional_veto_mode))
            ),
            l2_regime_directional_veto_persistence_bars=int(
                cls._validate_range(
                    "l2_regime_directional_veto_persistence_bars",
                    cls._as_int(params.get("l2_regime_directional_veto_persistence_bars",
                                           _dc.l2_regime_directional_veto_persistence_bars),
                                _dc.l2_regime_directional_veto_persistence_bars),
                    1,
                )
            ),
            l2_regime_directional_veto_loss_lookback_bars=int(
                cls._validate_range(
                    "l2_regime_directional_veto_loss_lookback_bars",
                    cls._as_int(params.get("l2_regime_directional_veto_loss_lookback_bars",
                                           _dc.l2_regime_directional_veto_loss_lookback_bars),
                                _dc.l2_regime_directional_veto_loss_lookback_bars),
                    1,
                )
            ),
            l2_regime_directional_veto_loss_trigger_bps=cls._validate_range(
                "l2_regime_directional_veto_loss_trigger_bps",
                cls._as_float(params.get("l2_regime_directional_veto_loss_trigger_bps",
                                         _dc.l2_regime_directional_veto_loss_trigger_bps),
                              _dc.l2_regime_directional_veto_loss_trigger_bps),
                0.0,
            ),
            l2_regime_directional_veto_cap_mu_bps=cls._validate_range(
                "l2_regime_directional_veto_cap_mu_bps",
                cls._as_float(params.get("l2_regime_directional_veto_cap_mu_bps",
                                         _dc.l2_regime_directional_veto_cap_mu_bps),
                              _dc.l2_regime_directional_veto_cap_mu_bps),
                0.0,
            ),
            l2_regime_directional_veto_release_raw_mu_nonpos=bool(
                params.get("l2_regime_directional_veto_release_raw_mu_nonpos",
                           _dc.l2_regime_directional_veto_release_raw_mu_nonpos)
            ),
            l2_regime_directional_veto_release_regime_bull_bars=int(
                cls._validate_range(
                    "l2_regime_directional_veto_release_regime_bull_bars",
                    cls._as_int(params.get("l2_regime_directional_veto_release_regime_bull_bars",
                                           _dc.l2_regime_directional_veto_release_regime_bull_bars),
                                _dc.l2_regime_directional_veto_release_regime_bull_bars),
                    1,
                )
            ),
            l2_regime_directional_veto_cooldown_bars=int(
                cls._validate_range(
                    "l2_regime_directional_veto_cooldown_bars",
                    cls._as_int(params.get("l2_regime_directional_veto_cooldown_bars",
                                           _dc.l2_regime_directional_veto_cooldown_bars),
                                _dc.l2_regime_directional_veto_cooldown_bars),
                    0,
                )
            ),
            l2_regime_directional_veto_max_fit_net_value_loss=cls._validate_range(
                "l2_regime_directional_veto_max_fit_net_value_loss",
                cls._as_float(params.get("l2_regime_directional_veto_max_fit_net_value_loss",
                                         _dc.l2_regime_directional_veto_max_fit_net_value_loss),
                              _dc.l2_regime_directional_veto_max_fit_net_value_loss),
                0.0,
            ),
            l2_regime_directional_veto_min_l3_total_return_delta=cls._validate_range(
                "l2_regime_directional_veto_min_l3_total_return_delta",
                cls._as_float(params.get("l2_regime_directional_veto_min_l3_total_return_delta",
                                         _dc.l2_regime_directional_veto_min_l3_total_return_delta),
                              _dc.l2_regime_directional_veto_min_l3_total_return_delta),
                0.0,
            ),
            l2_regime_directional_veto_max_l2_cagr_delta_loss=cls._validate_range(
                "l2_regime_directional_veto_max_l2_cagr_delta_loss",
                cls._as_float(params.get("l2_regime_directional_veto_max_l2_cagr_delta_loss",
                                         _dc.l2_regime_directional_veto_max_l2_cagr_delta_loss),
                              _dc.l2_regime_directional_veto_max_l2_cagr_delta_loss),
                0.0,
            ),
            l2_regime_directional_veto_symbols=_validate_directional_veto_symbols(
                params.get("l2_regime_directional_veto_symbols", _dc.l2_regime_directional_veto_symbols)
            ),
            l2_regime_directional_veto_adverse_codes=_validate_directional_veto_adverse_codes(
                params.get("l2_regime_directional_veto_adverse_codes", _dc.l2_regime_directional_veto_adverse_codes)
            ),
            l2_regime_directional_veto_long_eps_bps=cls._validate_range(
                "l2_regime_directional_veto_long_eps_bps",
                cls._as_float(
                    params.get("l2_regime_directional_veto_long_eps_bps", _dc.l2_regime_directional_veto_long_eps_bps),
                    _dc.l2_regime_directional_veto_long_eps_bps,
                ),
                0.0,
            ),
            l2_regime_directional_veto_action=_validate_directional_veto_action(
                str(params.get("l2_regime_directional_veto_action", _dc.l2_regime_directional_veto_action))
            ),
            l2_regime_directional_veto_max_fit_false_positive_rate=cls._validate_range(
                "l2_regime_directional_veto_max_fit_false_positive_rate",
                cls._as_float(
                    params.get(
                        "l2_regime_directional_veto_max_fit_false_positive_rate",
                        _dc.l2_regime_directional_veto_max_fit_false_positive_rate,
                    ),
                    _dc.l2_regime_directional_veto_max_fit_false_positive_rate,
                ),
                0.0,
                1.0,
            ),
            l2_regime_directional_veto_max_turnover_delta=cls._validate_range(
                "l2_regime_directional_veto_max_turnover_delta",
                cls._as_float(
                    params.get(
                        "l2_regime_directional_veto_max_turnover_delta",
                        _dc.l2_regime_directional_veto_max_turnover_delta,
                    ),
                    _dc.l2_regime_directional_veto_max_turnover_delta,
                ),
                0.0,
            ),
            l2_regime_directional_veto_min_gross_ratio=cls._validate_range(
                "l2_regime_directional_veto_min_gross_ratio",
                cls._as_float(
                    params.get(
                        "l2_regime_directional_veto_min_gross_ratio",
                        _dc.l2_regime_directional_veto_min_gross_ratio,
                    ),
                    _dc.l2_regime_directional_veto_min_gross_ratio,
                ),
                0.0,
                1.0,
            ),
            # L1 Intra-Symbol Divergence Dampener
            l2_intra_symbol_divergence_enabled=bool(
                params.get("l2_intra_symbol_divergence_enabled", _dc.l2_intra_symbol_divergence_enabled)
            ),
            l2_intra_symbol_divergence_symbols=tuple(
                str(s) for s in cast(
                    "tuple[str, ...]",
                    params.get(
                        "l2_intra_symbol_divergence_symbols",
                        _dc.l2_intra_symbol_divergence_symbols,
                    ),
                )
            ),
            l2_intra_symbol_divergence_dominant_families=tuple(
                str(f) for f in cast(
                    "tuple[str, ...]",
                    params.get(
                        "l2_intra_symbol_divergence_dominant_families",
                        _dc.l2_intra_symbol_divergence_dominant_families,
                    ),
                )
            ),
            l2_intra_symbol_divergence_persistence_bars=int(
                cls._validate_range(
                    "l2_intra_symbol_divergence_persistence_bars",
                    cls._as_int(params.get("l2_intra_symbol_divergence_persistence_bars",
                                           _dc.l2_intra_symbol_divergence_persistence_bars),
                               _dc.l2_intra_symbol_divergence_persistence_bars),
                    1,
                )
            ),
            l2_intra_symbol_divergence_release_bars=int(
                cls._validate_range(
                    "l2_intra_symbol_divergence_release_bars",
                    cls._as_int(params.get("l2_intra_symbol_divergence_release_bars",
                                           _dc.l2_intra_symbol_divergence_release_bars),
                               _dc.l2_intra_symbol_divergence_release_bars),
                    1,
                )
            ),
            l2_intra_symbol_divergence_cooldown_bars=int(
                cls._validate_range(
                    "l2_intra_symbol_divergence_cooldown_bars",
                    cls._as_int(params.get("l2_intra_symbol_divergence_cooldown_bars",
                                           _dc.l2_intra_symbol_divergence_cooldown_bars),
                               _dc.l2_intra_symbol_divergence_cooldown_bars),
                    0,
                )
            ),
            l2_intra_symbol_divergence_dominant_damp_mult=cls._validate_range(
                "l2_intra_symbol_divergence_dominant_damp_mult",
                cls._as_float(params.get("l2_intra_symbol_divergence_dominant_damp_mult",
                                         _dc.l2_intra_symbol_divergence_dominant_damp_mult),
                             _dc.l2_intra_symbol_divergence_dominant_damp_mult),
                0.0,
            ),
            l2_intra_symbol_divergence_dissent_boost_mult=cls._validate_range(
                "l2_intra_symbol_divergence_dissent_boost_mult",
                cls._as_float(params.get("l2_intra_symbol_divergence_dissent_boost_mult",
                                         _dc.l2_intra_symbol_divergence_dissent_boost_mult),
                             _dc.l2_intra_symbol_divergence_dissent_boost_mult),
                0.0,
            ),
        )


# SSOT: from_mapping에서 사용할 dataclass 기본값 인스턴스
_L2_DEFAULT_CONFIG: Layer2AllocationConfig = Layer2AllocationConfig()


@dataclass(slots=True, frozen=True)
class Layer2SignalSchedule:
    """Causal Layer2 event schedule materialized per bar."""

    events: tuple[ValidatedSignalEvent, ...]
    start_idx: int
    end_idx: int
    _events_by_bar: tuple[dict[str, ValidatedSignalEvent], ...] = field(
        default=(),
        repr=False,
    )


@dataclass(frozen=True)
class L2SimulationCache:
    """Pre-computed matrices for L2 simulation.

    Attributes:
        vol_matrix_2d: 변동성 행렬 [T, N].
        tradeable_mask_2d: 거래가능 마스크 [T, N].
        hurdle_2d: hurdle bps [T, N].
        funding_2d: funding bps [T, N].
        beta_1d: BTC 베타 [N].
        expected_gross_bps_2d: sleeve 단위 gross edge [T, S].
        expected_net_bps_2d: sleeve 단위 net edge [T, S].
        holding_bars_2d: sleeve 단위 holding bars [T, S].
        side_2d: sleeve 단위 방향 [T, S].
        quality_weight_2d: sleeve 단위 quality weight [T, S].
        signal_mask_2d: sleeve 단위 활성 마스크 [T, S].
        sleeve_to_sym: sleeve j → symbol column idx 매핑 [S].
        sleeve_ids: (symbol, strategy_id) 결정적 정렬 튜플 [S].
    """

    vol_matrix_2d: NDArray[np.float64]
    tradeable_mask_2d: NDArray[np.bool_]
    hurdle_2d: NDArray[np.float64]
    funding_2d: NDArray[np.float64]
    beta_1d: NDArray[np.float64]

    # Vectorized Signal Matrices (Shape: [T, S] where S = n_sleeves)
    expected_gross_bps_2d: NDArray[np.float64]
    expected_net_bps_2d: NDArray[np.float64]
    holding_bars_2d: NDArray[np.float64]
    side_2d: NDArray[np.float64]
    quality_weight_2d: NDArray[np.float64]
    signal_mask_2d: NDArray[np.bool_]

    # Sleeve→symbol mapping (신규, multi-TF 핵심)
    sleeve_to_sym: NDArray[np.int64]  # [S]
    sleeve_ids: tuple[tuple[str, str], ...]  # [S] (symbol, strategy_id)
    sleeve_to_tf: tuple[str, ...]  # [S] each sleeve's native TF (from strategy_id suffix)

    # Pre-computed bucket realized edges (trial-param independent → cached once)
    bucket_edges_by_fold: tuple[dict[tuple[int, str, str], float], ...] = ()
    pooled_edges_by_fold: tuple[dict[tuple[str, str], float], ...] = ()
    # Pre-computed regime code 1d (trial-param independent → cached once)
    regime_code_1d: NDArray[np.int8] | None = None
    regime_routing_diagnostics: RegimeRoutingDiagnostics | None = None
    regime_policy_by_fold: tuple[dict[tuple[int, str, str], RegimeCellPolicy], ...] = ()


@dataclass(frozen=True, slots=True)
class RegimeCellPolicy:
    state: int
    state_name: str
    family: str
    tf: str
    action: RegimePolicyAction
    reason: RegimePolicyReason
    edge_multiplier: float
    confidence: float
    fit_edge_bps: float
    pooled_fit_edge_bps: float
    cal_edge_bps: float
    pooled_cal_edge_bps: float
    fit_lift_bps: float
    cal_lift_bps: float
    sign_consistent: bool
    hard_block_eligible: bool
    n_fit: int
    n_cal: int
    reliability: float = 0.0


@dataclass(frozen=True, slots=True)
class RegimePolicyDiagnostics:
    mode: RegimePolicyMode
    enabled: bool
    global_reliable: bool
    reason: str
    n_cells_total: int
    n_allow: int
    n_downweight: int
    n_block: int
    n_pooled: int
    n_unstable: int
    n_hard_block_eligible: int
    mean_fit_lift_bps: float
    mean_cal_lift_bps: float
    min_cal_lift_bps: float
    max_cal_lift_bps: float
    mean_confidence: float
    sign_consistency_ratio: float
    hard_block_enabled: bool


@dataclass(frozen=True, slots=True)
class RegimePolicyApplication:
    sleeve_sigs: dict[tuple[str, str], SymbolSignal]
    sleeve_edges: dict[tuple[str, str], float]
    n_input: int
    n_allow: int
    n_downweight: int
    n_block: int
    n_pooled: int
    gross_edge_before_bps: float = 0.0
    gross_edge_after_bps: float = 0.0
    abs_mu_before_bps: float = 0.0
    abs_mu_after_bps: float = 0.0
    quality_weight_before: float = 0.0
    quality_weight_after: float = 0.0


@dataclass(frozen=True, slots=True)
class RegimeRoutingDiagnostics:
    active_state_count: int
    active_state_names: tuple[str, ...]
    compression_enabled: bool
    proof_passed: bool
    conditioning_path: Literal["regime_conditioned", "pooled_fallback"]
    mean_lift_bps: float
    n_eff: float
    nw_tstat: float
    deflated_sharpe: float
    fold_pass_ratio: float
    n_folds_evaluated: int
    bucket_hit_pct_by_fold: tuple[float, ...]
    js_divergence_by_fold: tuple[float, ...]
    policy_diagnostics: RegimePolicyDiagnostics | None = None
    debug_diagnostics: RegimeDebugDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class RegimeCellDebugStat:
    fold_idx: int
    state: int
    state_name: str
    family: str
    tf: str
    n_fit: int
    n_oos: int
    fit_edge_bps: float
    pooled_fit_edge_bps: float
    oos_realized_edge_bps: float
    edge_gap_bps: float
    sign_hit_rate: float
    selected_hit_pct: float


@dataclass(frozen=True, slots=True)
class RegimeGranularityDebugStat:
    label: Literal["pooled", "effective_3", "raw_6"]
    state_count: int
    proof_passed: bool
    conditioning_path: Literal["regime_conditioned", "pooled_fallback"]
    mean_lift_bps: float
    nw_tstat: float
    fold_pass_ratio: float
    n_folds_evaluated: int
    bucket_hit_pct_mean: float
    oos_cell_ic: float
    oos_cell_rmse_bps: float
    oos_cell_bias_bps: float


@dataclass(frozen=True, slots=True)
class RegimeDebugDiagnostics:
    granularity_stats: tuple[RegimeGranularityDebugStat, ...]
    top_positive_cells: tuple[RegimeCellDebugStat, ...]
    top_negative_cells: tuple[RegimeCellDebugStat, ...]
    worst_error_cells: tuple[RegimeCellDebugStat, ...]
    compression_loss_bps: float
    selected_regime_return_bps: tuple[float, ...]
    selected_regime_bar_count: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RegimeRoutingPlan:
    effective_bucket_edges_by_fold: tuple[dict[tuple[int, str, str], float], ...]
    raw_bucket_edges_by_fold: tuple[dict[tuple[int, str, str], float], ...]
    pooled_edges_by_fold: tuple[dict[tuple[str, str], float], ...]
    effective_regime_code_1d: NDArray[np.int8]
    diagnostics: RegimeRoutingDiagnostics
    policy_by_fold: tuple[dict[tuple[int, str, str], RegimeCellPolicy], ...] = ()


@dataclass(slots=True, frozen=True)
class Layer2SimulationDiagnostics:
    """Layer2 simulation observability payload."""

    signal_event_count: int
    active_signal_bar_ratio: float
    mean_active_positions: float
    mean_gross_exposure: float
    mean_net_exposure: float
    mean_long_exposure: float
    mean_short_exposure: float
    support_leak_count: int
    cap_saturation_ratio: float
    execution_cost_return: float
    funding_return: float


@dataclass(slots=True, frozen=True)
class Layer3Result:
    """Layer3 Holdout 최종 검증 결과. [ADR_20260704_L2_DIRECTIONAL_VETO]

    [ADR_20260705_TF_VALIDATION_ROOT_CAUSE_CAPTURE]

    Attributes:
        cagr: 전략 연평균 복리 수익률.
        mdd: 전략 최대 낙폭 (양수).
        sharpe: 전략 Sharpe.
        mar: MAR ratio (CAGR / MDD).
        cagr_baseline: 기준 전략 CAGR.
        mdd_baseline: 기준 전략 MDD.
        sharpe_baseline: 기준 전략 Sharpe.
        mar_baseline: 기준 전략 MAR.
        gate_passed: L3 통과 여부.
        blocker_reason: 실패 원인 키. 통과 시 빈 문자열.
        total_return: 홀드아웃 종료 시 누적수익률 (terminal_multiple - 1).
        equity_multiple: 누적 복리 배수 (terminal_multiple).
        sortino: 연율화 Sortino (하방위험 조정, 진단용).
        sortino_baseline: 기준 전략 Sortino (진단용).
        n_trades: 홀드아웃 체결 수 (통계적 유의성 sanity).
        cvar95: per-bar 95% CVaR loss (꼬리위험, 양수, 진단용).
        avg_gross_exposure: 평균 총노출 (실제 배치 여부 진단용).
        deploy_leverage: L2 champion deployment scalar applied to hybrid holdout returns.
        risk_off_bars: Reversal-kill risk-off bar count from fold attribution.
        risk_off_realized_price: Realized price impact during risk-off bars.
        risk_on_realized_price: Realized price impact during risk-on bars.
        reversal_kill_active: Whether L2_REVERSAL_KILL env was active for this run.
        regime_bull_pct: [ADR_20260704_L3_REGIME] OOS holdout window bull regime % (diagnostics-only).
        regime_bear_pct: OOS holdout window bear regime % (diagnostics-only).
        regime_crisis_pct: OOS holdout window crisis regime % (diagnostics-only).
        mean_trend_efficiency: Mean Kaufman Efficiency Ratio over OOS (diagnostics-only).
        trend_efficiency_corr: Trend efficiency correlation with returns over OOS (diagnostics-only).
        realized_price_long: [ADR_20260704_L2L3_LONGSHORT] Long-leg realized price
            P&L over the OOS holdout (diagnostics-only, always-on).
        realized_price_short: Short-leg realized price P&L over the OOS holdout
            (diagnostics-only, always-on).
        bars_long: Count of OOS bars with any nonzero long exposure.
        bars_short: Count of OOS bars with any nonzero short exposure.
        realized_price_long_by_symbol: [ADR_20260704_L2L3_PERSYMBOL] Per-symbol
            long-leg realized price P&L over the OOS holdout (diagnostics-only).
        realized_price_short_by_symbol: Per-symbol short-leg realized price P&L
            over the OOS holdout (diagnostics-only).
        major_symbol_diag: [ADR_20260704_L3_MAJORDIAG] Per-symbol signal-vs-sizing
            mismatch ratios for MAJOR_DIAG_SYMBOLS over the OOS holdout
            (diagnostics-only, always-on).
    """

    cagr: float
    mdd: float
    sharpe: float
    mar: float
    cagr_baseline: float
    mdd_baseline: float
    sharpe_baseline: float
    mar_baseline: float
    gate_passed: bool
    blocker_reason: str = ""
    # ── 신규: 단일 OOS 복리/배치 건전성 (lean) ──
    total_return: float = 0.0
    equity_multiple: float = 1.0
    sortino: float = 0.0
    sortino_baseline: float = 0.0
    n_trades: int = 0
    cvar95: float = 0.0
    avg_gross_exposure: float = 0.0
    deploy_leverage: float = 1.0
    min_trades: int = 10
    max_mdd_abs: float = 0.35
    min_sharpe: float = 0.0
    min_sortino: float = 0.0
    max_cvar95: float = 0.06
    risk_off_bars: int = 0
    risk_off_realized_price: float = 0.0
    risk_on_realized_price: float = 0.0
    reversal_kill_active: bool = False
    risk_off_episodes: tuple[ReversalEpisode, ...] = ()
    regime_bull_pct: float = 0.0
    regime_bear_pct: float = 0.0
    regime_crisis_pct: float = 0.0
    mean_trend_efficiency: float = 0.0
    trend_efficiency_corr: float = 0.0
    realized_price_long: float = 0.0
    realized_price_short: float = 0.0
    realized_price_long_by_symbol: tuple[tuple[str, float], ...] = ()
    realized_price_short_by_symbol: tuple[tuple[str, float], ...] = ()
    bars_long: int = 0
    bars_short: int = 0
    major_symbol_diag: tuple[MajorSymbolSignalSizingSummary, ...] = ()
    major_symbol_sleeve_diag: tuple[MajorSymbolSleeveContributionSummary, ...] = ()
    major_symbol_incoherence: tuple[MajorSymbolIncoherenceSummary, ...] = ()
    directional_veto_summary: tuple[DirectionalVetoSummary, ...] = ()
    validation_parity_report: ValidationParityReport | None = None


@dataclass(slots=True, frozen=True)
class FoldDiagnostic:
    """CPCV fold별 진단 데이터 (IC·breadth·n_valid·n_events 동일 fold 출처 보장).

    Attributes:
        fold: 1-based fold 번호.
        ic: fold 내 pooled time-series Spearman rank IC. None = 유효 이벤트 부족(<4).
        breadth: valid 심볼 비율.
        n_valid: valid 심볼 수.
        n_events: OOS 이벤트 수.
        passed: ic is not None and ic > 0.
    """

    fold: int
    ic: float | None
    breadth: float
    n_valid: int
    n_eligible: int
    n_events: int
    n_fit: int
    fit_status: FoldFitStatus
    passed: bool
