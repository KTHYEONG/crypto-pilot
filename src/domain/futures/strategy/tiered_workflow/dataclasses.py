# src/domain/futures/strategy/tiered_workflow/dataclasses.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.futures.strategy.candidate_contracts import (
        FoldFitStatus,
        Layer1FoldReadiness,
        Layer1GateReport,
        Layer1InferenceArtifact,
        QualifiedSignalRegistry,
        SymbolStrategyEvidence,
    )
    from src.domain.futures.strategy.cs_rank import SymbolSignal


@dataclass(slots=True, frozen=True)
class Layer1Result:
    """Layer1 SWF-K 검증 결과.

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
    inference_artifact: Layer1InferenceArtifact | None = None


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
class Layer2Result:
    """Layer2 AWF 포트폴리오 검증 결과.

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
        blocker_reason: 실패 원인 키. "" = 통과. 값: no_deployment/cagr/mar/sharpe_abs/mdd_rel/mdd_abs/fold/uplift.
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


@dataclass(slots=True, frozen=True)
class Layer3Result:
    """Layer3 Holdout 최종 검증 결과.

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
