"""MHS report schema: dataclasses and key rename registry.

``RENAME_REGISTRY`` maps old persisted JSON keys to their new names.
The golden comparison applies it before diffing, so any unregistered
key change fails the identity gate.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.application.research.mhs.contracts import (
        MhsBookReport,
        MhsFoldReport,
        MhsResearchGoResult,
        MhsResourceMeasurement,
    )
    from src.application.research.mhs.resources import ProcessTreeMemoryStats
    from src.mhs.discovery import DiscoveryQualificationResult
    from src.mhs.evidence import (
        AnchoredPurgedFold,
        DeploymentReadinessResult,
        DsrDecomposition,
    )


@dataclass(frozen=True, slots=True)
class MhsHorizonDiagnosticReport:
    """Top-level MHS Phase 1 diagnostic report.

    Canonical definition (P1): moved out of
    ``src.application.research.mhs.contracts``, which now re-exports this
    class so every existing ``from ... import MhsHorizonDiagnosticReport``
    import path keeps working unchanged.
    """

    feature: str
    status: str
    start: str
    end: str
    resolved_end: str
    partition: str
    execution_tiers_bps: tuple[float, ...]
    books: dict[str, MhsBookReport]
    blend: MhsBookReport | None
    blend_target_gross: float
    blend_cash_fraction: float
    eligible_symbols: int
    trials_attempted: int
    deflated_sharpe_ratio: float | None
    xs_rank_ic: dict[str, float]
    date_clustered_regression: dict[str, float]
    horizon_diagnostics: dict[str, float]
    bootstrap_ci: tuple[float, float] | None
    placebo_sharpe_percentile: float | None
    deployment_readiness: DeploymentReadinessResult
    synthetic_stress: dict[str, dict[str, Any]]
    participation_warnings: dict[str, float]
    termination_counts: dict[str, int]
    unsupported_assumptions: tuple[str, ...]
    anchored_folds: tuple[AnchoredPurgedFold, ...]
    folds: tuple[MhsFoldReport, ...]
    research_go: MhsResearchGoResult
    fill_source: str
    mark_source: str
    execution_timeframe: str
    execution_universe_size: int
    execution_symbols: tuple[str, ...]
    run_elapsed_seconds: float
    resource_measurements: tuple[MhsResourceMeasurement, ...] = ()
    discovery_qualification: dict[str, DiscoveryQualificationResult] | None = None
    realized_execution_roster_size: float | None = None
    full_history_yearly_net_t: dict[str, dict[int, float]] | None = None
    funding_carry_worst_year_corr: float | None = None
    trend_sleeve_diagnostic: dict[str, Any] | None = None
    multi_feature_diagnostic: dict[str, Any] | None = None
    committee_diagnostic: dict[str, Any] | None = None
    funding_dropped_symbols: dict[str, str] | None = None
    fold_blend_parity: dict[str, Any] | None = None
    fold_growth_concentration: dict[str, Any] | None = None
    fold_realized_risk_parity: dict[str, Any] | None = None
    # 선언형 alpha 증거 게이트의 런별 보정 결과(null_alpha/임계값/pooled LCB).
    evidence_calibration: dict[str, Any] | None = None
    fill_mark_parity: dict[str, Any] | None = None
    growth_envelope: dict[str, Any] | None = None
    committee_member_attribution: dict[str, Any] | None = None
    worker_plan: dict[str, int] = field(default_factory=dict)
    tree_memory: ProcessTreeMemoryStats | None = None
    # 선택창 겹침 공시(I1): 보고 구간이 기본값 선택창과 겹치면 > 0 이며
    # research_go.reason_codes 의 SELECTION_WINDOW_OVERLAP 와 짝을 이룬다.
    selection_overlap_fraction: float | None = None
    # DSR 분모(trials_attempted)의 출처: 'history' | 'constant' | 'constant_fallback'.
    trials_attempted_source: str | None = None
    # DSR 분해 계측: 어느 항(SR 마진/분산/n_eff/N)이 통계를 끌어내렸는지 노출.
    dsr_decomposition: DsrDecomposition | None = None
    # 폴드 Sharpe 분산 sqrt(V): 사전등록 후보 탐색의 1차 목적함수.
    fold_sharpe_dispersion: float | None = None
    # 폴드 분산 기반 DSR 프록시(관측 전용): 게이트에 절대 재진입하지 않는다.
    deflated_sharpe_ratio_fold_proxy: float | None = None
    # 폴드별 committee 가중치 적합 샘플(< COMMITTEE_OOS_START) 누출 비율(관측 전용).
    fold_committee_weight_leak: dict[str, float] | None = None
    # 인과 레짐(trailing high / trailing vol tercile, 1-bar shift) 조건부 Sharpe(관측 전용).
    regime_conditional_sharpe: dict[str, dict[str, float]] | None = None
    # 봉인 경계(HOLDOUT_CUTOFF) 이후 hold-out 꼬리 구간 자체의 성과 요약
    # (holdout_tail_evidence 결과); 보고 구간이 봉인을 넘지 않으면 None.
    holdout_tail: dict[str, Any] | None = None
    # parameter-fit 경계 기준 인샘플/OOS 성과 분할(parameter_oos_split_evidence 결과); 어느 한쪽 표본이 부족하면 None.
    parameter_oos_split: dict[str, Any] | None = None
    # 시행 풀 계측 공시(I-DISCLOSURE): 배제 사유별 건수·distinct 키·원장 크기 등
    # 관측 페이로드. GO reason code를 발생시키지 않는다.
    trial_pool: dict[str, Any] | None = None

    def to_payload(self) -> Any:
        from src.mhs.report.artifacts import _jsonable
        return _jsonable(dataclasses.asdict(self))


# Frozen old-key → new-key map for persisted report JSON.
# Only keys explicitly listed here may change name between versions.
# The golden comparison applies renames to the golden before diffing.
RENAME_REGISTRY: dict[str, str] = {
    # P3: MHS_ prefix stripped from params constants
    "MHS_COMMITTEE_TARGET_GROSS": "COMMITTEE_TARGET_GROSS",
    "MHS_COMMITTEE_TARGET_VOL": "COMMITTEE_TARGET_VOL",
    # P3: PHASE_1_ prefix stripped
    "PHASE_1_BOOK_SPECS": "BOOK_SPECS",
    "PHASE_1_BOOK_BLEND_WEIGHTS": "BOOK_BLEND_WEIGHTS",
}
