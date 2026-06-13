"""3-Layer 티어드 파이프라인 오케스트레이터.

Layer1 (SWF-K Signal Validation) → Layer2 (AWF Portfolio) → Layer3 (Holdout) 순서로
게이트 기반 단계적 검증을 수행한다.

Time Complexity: O(F * T * N) — F=folds, T=bars, N=symbols
Space Complexity: O(F * N) — fold별 signal 집계
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm, spearmanr

from src.domain.futures.portfolio.portfolio_constructor import (
    PortfolioCaps,
    diagonal_kelly_weights,
)
from src.domain.futures.portfolio.signal_composer import (
    compose_symbol_signals,
    composer_sigma_lookback_bars,
    rolling_per_bar_return_std,
)
from src.domain.futures.strategy.candidate_contracts import (
    CandidateModelOutput,
    FoldFitStatus,
    Layer1FoldReadiness,
    Layer1GateCheck,
    Layer1GateReport,
    Layer1InferenceArtifact,
    MatchedBaselineKey,
    QualifiedSignalRegistry,
    SignalSourceKey,
    SymbolStrategyEvidence,
    ValidatedSignalBatch,
    ValidatedSignalEvent,
)
from src.domain.futures.strategy.candidate_dataset import (
    build_candidate_dataset,
    fit_candidate_feature_schema,
)
from src.domain.futures.strategy.candidate_ensemble import (
    fit_regime_conditional_ensemble,
    predict_regime_conditional_ensemble,
)
from src.domain.futures.strategy.candidate_workflow import _fit_and_predict_single_fold
from src.domain.futures.strategy.cs_rank import (
    VOL_FLOOR,
    SymbolSignal,
    rank_and_select,
)
from src.domain.futures.strategy.tiered_logging import (
    format_layer1_deployment_registry_table,
    format_layer1_gate_table,
    format_layer1_outer_fold_table,
    format_layer1_table,
    format_layer2_table,
    format_layer3_table,
    format_system_status,
)
from src.domain.futures.strategy.walk_forward import (
    WFFold,
    build_l1_nested_swf_folds,
    build_l1_swf_folds,
    build_walk_forward_folds,
)

if TYPE_CHECKING:
    from src.domain.futures.optimization.opt_config import LayeredWindow
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.config import CandidateStrategyConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 결과 데이터클래스
# ---------------------------------------------------------------------------

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
        turnover: 평균 단방향 회전율.
        friction_pass_pct: 마찰 허들 통과 심볼 비율.
        gate_passed: L2 통과 여부.
    """

    selected_last: frozenset[str]
    weights_last: dict[str, float]
    sharpe_hybrid: float
    sharpe_baseline: float
    mdd_hybrid: float
    mdd_baseline: float
    turnover: float
    friction_pass_pct: float
    gate_passed: bool


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


# ---------------------------------------------------------------------------
# fold 진단 데이터 구조 (single source of truth — 인덱스 분리 버그 방지)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

_BARS_PER_YEAR: float = 2190.0  # 4h 기준
_VALID_COVERAGE_FLAG_THRESHOLD: float = 0.80  # per-fold valid 비율 임계 (L1 게이트와 일치)
_TRAINED_FOLD_COVERAGE_THRESHOLD: float = 0.80


def _sharpe(rets: list[float], bars_per_year: float = _BARS_PER_YEAR) -> float:
    """연율화 Sharpe 계산.

    Args:
        rets: per-bar 수익률 리스트.
        bars_per_year: 연율화 팩터.

    Returns:
        Sharpe Ratio (float). 데이터 부족 시 0.0.

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if len(rets) < 2:
        return 0.0
    arr = np.asarray(rets, dtype=np.float64)
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    return float(mu * bars_per_year / (sd * np.sqrt(bars_per_year) + 1e-9))


def _mdd(rets: list[float]) -> float:
    """최대 낙폭 계산 (양수 반환).

    Args:
        rets: per-bar 수익률 리스트.

    Returns:
        최대 낙폭 절대값. 데이터 없으면 0.0.

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not rets:
        return 0.0
    cum = np.cumsum(np.asarray(rets, dtype=np.float64))
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    return float(np.max(drawdown))


def _cagr(rets: list[float], bars_per_year: float = _BARS_PER_YEAR) -> float:
    """연율화 CAGR 계산.

    Args:
        rets: per-bar 수익률 리스트.
        bars_per_year: 연율화 팩터.

    Returns:
        CAGR. 빈 리스트면 0.0, total loss(합산 pnl <= -1.0)면 -1.0.

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not rets:
        return 0.0
    total_pnl = float(np.sum(np.asarray(rets, dtype=np.float64)))
    n = len(rets)
    base = 1.0 + total_pnl
    if base <= 0.0:
        return -1.0
    return float(base ** (bars_per_year / n) - 1.0)


@dataclass(slots=True)
class _AwfSimResult:
    """run_awf 내부 시뮬레이션 결과 (private)."""

    rets_hybrid: list[float]
    rets_baseline: list[float]
    last_selected: frozenset[str]
    last_w: NDArray[np.float64]
    all_turnovers: list[float]
    friction_pass_total: int
    signal_total: int


def _run_awf_simulation(
    *,
    l1_oos: dict[str, SymbolSignal],
    aligned: AlignedMarketData,
    awf_folds: tuple[WFFold, ...],
    l2_params: dict[str, Any],
    caps: PortfolioCaps,
    tf: str = "4h",
) -> _AwfSimResult:
    """AWF 시뮬레이션 핵심 루프 (L2/L3 공용).

    Args:
        l1_oos: L1 합본 symbol→SymbolSignal.
        aligned: AlignedMarketData.
        awf_folds: WFFold 튜플.
        l2_params: Layer2 하이퍼파라미터 dict.
        caps: PortfolioCaps.
        tf: 타임프레임 문자열.

    Returns:
        _AwfSimResult.

    Time Complexity: O(F * T * N)
    Space Complexity: O(T)
    """
    k_rank = int(l2_params.get("K_RANK", 3))
    rank_buffer = int(l2_params.get("rank_buffer", 1))
    kelly_fraction = float(l2_params.get("kelly_fraction", 0.25))
    vol_target: float | None = l2_params.get("vol_target")
    if not isinstance(vol_target, float):
        vol_target = None
    no_trade_band = float(l2_params.get("no_trade_band", 0.01))
    rebalance_bars = int(l2_params.get("REBALANCE_BARS", 3))

    symbols = aligned.symbols
    n_sym = len(symbols)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    lookback = composer_sigma_lookback_bars(tf)

    # 1. 시뮬레이션 진입 전 모든 심볼의 롤링 변동성을 사전 일괄 계산
    vol_matrix = np.full_like(aligned.close_2d, VOL_FLOOR)
    for i in range(n_sym):
        close_col = aligned.close_2d[:, i]
        v_std = rolling_per_bar_return_std(close_col, lookback)
        vol_matrix[:, i] = np.maximum(v_std, VOL_FLOOR)

    all_rets_hybrid: list[float] = []
    all_rets_baseline: list[float] = []
    all_turnovers: list[float] = []
    friction_pass_total = 0
    signal_total = 0

    prev_selection: frozenset[str] = frozenset()
    prev_w: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
    last_selected: frozenset[str] = frozenset()
    last_w: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)

    for fold in awf_folds:
        for t in range(fold.oos_start, fold.oos_end - 1, rebalance_bars):
            t_end = min(t + rebalance_bars, fold.oos_end - 1)

            # 2. 현재 시점 vol 추정 (std 연산을 제거하고 사전 계산 행렬을 단순 룩업)
            valid_signals: dict[str, SymbolSignal] = {}
            for sym, sig in l1_oos.items():
                if sym not in sym_to_idx:
                    continue
                i = sym_to_idx[sym]
                vol = float(vol_matrix[t, i])
                valid_signals[sym] = SymbolSignal(
                    raw_mu=sig.raw_mu,
                    volatility=vol,
                    n_obs=sig.n_obs,
                    t_stat=sig.t_stat,
                    valid=sig.valid,
                    beta_btc=sig.beta_btc,
                )

            # 2. rank_and_select
            selected, _z_scores = rank_and_select(
                valid_signals,
                k_rank=k_rank,
                sector_cap=n_sym,
                prev_selection=prev_selection,
                rank_buffer=rank_buffer,
            )
            last_selected = selected

            # 3. mu/sigma 배열 구성
            mu_arr: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
            sig_arr: NDArray[np.float64] = np.full(n_sym, VOL_FLOOR, dtype=np.float64)
            for sym, ss in valid_signals.items():
                if sym in sym_to_idx:
                    i = sym_to_idx[sym]
                    if sym in selected:
                        mu_arr[i] = ss.raw_mu
                    sig_arr[i] = ss.volatility

            # 4. friction hurdle
            if (
                aligned.execution_cost_bps_2d is not None
                and t < aligned.execution_cost_bps_2d.shape[0]
            ):
                hurdle = aligned.execution_cost_bps_2d[t].astype(np.float64)
            else:
                hurdle = np.full(n_sym, 3.8, dtype=np.float64)

            btc_beta: NDArray[np.float64] | None = None
            if aligned.beta_vs_market_1d is not None:
                btc_beta = aligned.beta_vs_market_1d.astype(np.float64)

            # 5. diagonal_kelly_weights
            w = diagonal_kelly_weights(
                mu_bps=mu_arr,
                sigma=sig_arr,
                kelly_fraction=kelly_fraction,
                vol_target=vol_target,
                friction_hurdle_bps=hurdle,
                caps=caps,
                prev_w=prev_w,
                no_trade_band=no_trade_band,
                btc_beta=btc_beta,
            )
            last_w = w

            # 6. turnover & friction stats (선택 심볼 기준)
            turnover = float(np.sum(np.abs(w - prev_w))) / 2.0
            all_turnovers.append(turnover)
            selected_idxs = [sym_to_idx[s] for s in selected if s in sym_to_idx]
            if selected_idxs:
                sel_idx_arr = np.array(selected_idxs, dtype=np.intp)
                friction_pass = int(np.sum(np.abs(mu_arr[sel_idx_arr]) >= hurdle[sel_idx_arr]))
            else:
                friction_pass = 0
            friction_pass_total += friction_pass
            signal_total += max(1, len(selected))

            # 7. PnL 계산 (bar별)
            n_valid_sym = max(1, sum(1 for ss in valid_signals.values() if ss.valid))
            w_base = np.array(
                [
                    1.0 / n_valid_sym
                    if s in valid_signals and valid_signals[s].valid
                    else 0.0
                    for s in symbols
                ],
                dtype=np.float64,
            )

            for t2 in range(t, t_end):
                if t2 + 1 >= aligned.close_2d.shape[0]:
                    break
                c_cur = aligned.close_2d[t2]
                c_nxt = aligned.close_2d[t2 + 1]
                bar_ret = np.where(c_cur > 0, (c_nxt - c_cur) / c_cur, 0.0)
                all_rets_hybrid.append(float(np.dot(w, bar_ret)))
                all_rets_baseline.append(float(np.dot(w_base, bar_ret)))

            prev_selection = selected
            prev_w = w

    return _AwfSimResult(
        rets_hybrid=all_rets_hybrid,
        rets_baseline=all_rets_baseline,
        last_selected=last_selected,
        last_w=last_w,
        all_turnovers=all_turnovers,
        friction_pass_total=friction_pass_total,
        signal_total=signal_total,
    )


def _stack_oos_signals(
    signals_per_fold: tuple[dict[str, SymbolSignal], ...],
    realized_stats: dict[str, SymbolRealizedStat] | None = None,
) -> dict[str, SymbolSignal]:
    """fold별 SymbolSignal을 per-symbol로 집계 (raw_mu 평균).

    raw_mu/vol/beta는 예측 기반 fold 평균(사이징용).
    t_stat/valid/n_obs는 realized_stats에서 주입(BUG-A 제거: 마지막 fold 편향 제거).

    Args:
        signals_per_fold: fold별 symbol→SymbolSignal 매핑 튜플.
        realized_stats: SymbolRealizedStat 매핑. None이면 보수적 valid=False 처리.

    Returns:
        합본 symbol→SymbolSignal 매핑.

    Time Complexity: O(F * N)
    Space Complexity: O(N)
    """
    sym_mu_lists: dict[str, list[float]] = defaultdict(list)
    sym_vol_lists: dict[str, list[float]] = defaultdict(list)
    sym_beta_lists: dict[str, list[float | None]] = defaultdict(list)

    for fold_sigs in signals_per_fold:
        for sym, sig in fold_sigs.items():
            sym_mu_lists[sym].append(sig.raw_mu)
            sym_vol_lists[sym].append(sig.volatility)
            sym_beta_lists[sym].append(sig.beta_btc)

    oos_stacked: dict[str, SymbolSignal] = {}
    for sym, mus in sym_mu_lists.items():
        real = realized_stats.get(sym) if realized_stats else None
        betas = [b for b in sym_beta_lists[sym] if b is not None]
        avg_beta: float | None = float(np.mean(betas)) if betas else None
        avg_vol = float(np.mean(sym_vol_lists[sym])) if sym_vol_lists[sym] else VOL_FLOOR
        oos_stacked[sym] = SymbolSignal(
            raw_mu=float(np.mean(mus)),
            volatility=avg_vol,
            n_obs=real.n_obs if real is not None else 0,
            t_stat=real.t_stat if real is not None else 0.0,
            valid=real.valid if real is not None else False,
            beta_btc=avg_beta,
        )
    return oos_stacked


def _date_to_idx(datetimes: NDArray[np.datetime64], target_date: Any) -> int:
    """target_date에 해당하는 bar 인덱스 검색.

    Args:
        datetimes: datetime64 배열 (정렬됨).
        target_date: 검색 대상 날짜 (date, str, datetime64 호환).

    Returns:
        bar 인덱스 (0-based). 범위 초과 시 마지막 인덱스.
    """
    target = np.datetime64(target_date, "D")
    idx = int(np.searchsorted(datetimes.astype("datetime64[D]"), target))
    return min(idx, len(datetimes) - 1)


def _is_non_constant_finite_array(values: NDArray[np.float64]) -> bool:
    """Return whether a finite-valued vector has positive dispersion."""
    if values.size < 1:
        return False
    finite = values[np.isfinite(values)]
    if finite.size < 1:
        return False
    return float(np.nanstd(finite)) > 0.0


def _is_trained_fold_output(fold_out: Any) -> bool:
    """Return whether the fold completed training with non-degenerate predictions."""
    return getattr(fold_out, "fit_status", "trained") == "trained"


def _fold_eligible_symbol_mask(
    *,
    aligned: AlignedMarketData,
    fold: WFFold,
    min_bar_coverage: float = 0.80,
) -> NDArray[np.bool_]:
    """Compute fold-local PIT eligible symbols from OOS universe and warm/kill masks."""
    if fold.oos_end <= fold.oos_start:
        return np.zeros(len(aligned.symbols), dtype=bool)

    active_mask = getattr(aligned, "inference_active_mask", None)
    if not isinstance(active_mask, np.ndarray):
        active_mask = getattr(aligned, "active_mask", None)
    if not isinstance(active_mask, np.ndarray):
        active_mask = np.ones((len(aligned.datetimes), len(aligned.symbols)), dtype=bool)

    warm_mask = getattr(aligned, "inference_entry_warm_mask", None)
    if not isinstance(warm_mask, np.ndarray):
        warm_mask = getattr(aligned, "warm_mask", None)
    if not isinstance(warm_mask, np.ndarray):
        warm_mask = np.ones((len(aligned.datetimes), len(aligned.symbols)), dtype=bool)

    entry_block_mask = getattr(aligned, "entry_block_mask", None)
    if not isinstance(entry_block_mask, np.ndarray):
        entry_block_mask = np.zeros((len(aligned.datetimes), len(aligned.symbols)), dtype=bool)

    kill_mask = getattr(aligned, "kill_mask", None)
    if not isinstance(kill_mask, np.ndarray):
        kill_mask = np.zeros((len(aligned.datetimes), len(aligned.symbols)), dtype=bool)

    oos_slice = slice(fold.oos_start, fold.oos_end)
    eligible_2d = (
        active_mask[oos_slice]
        & warm_mask[oos_slice]
        & ~entry_block_mask[oos_slice]
        & ~kill_mask[oos_slice]
    )
    coverage = np.mean(eligible_2d.astype(np.float64), axis=0)
    return np.asarray(coverage >= float(min_bar_coverage), dtype=bool)


# ---------------------------------------------------------------------------
# Layer1: 심볼별 time-series rank IC
# ---------------------------------------------------------------------------


def compute_per_symbol_ic(
    *,
    fold_tuples: list[tuple[int, Any, Any]],
) -> dict[str, float]:
    """심볼별 time-series Spearman rank IC (expected_net_bps vs oos_set.y_return_bps).

    oos_set.y_return_bps는 방향·barrier·비용 반영 정준 실현 수익 라벨이다.

    Args:
        fold_tuples: (fold_idx, wf_fold, fold_out) 리스트.
            fold_out.oos_set.y_return_bps와 oos_set.event_index가 필요하다.

    Returns:
        symbol → fold 평균 rank IC dict.

    Time Complexity: O(F * E) — E = events per fold
    Space Complexity: O(S * F) — S = symbols, F = folds
    """
    sym_ic_lists: dict[str, list[float]] = defaultdict(list)

    for _, _, fold_out in fold_tuples:
        if not _is_trained_fold_output(fold_out):
            continue
        oos_set = getattr(fold_out, "oos_set", None)
        if oos_set is None:
            continue

        y_realized = getattr(oos_set, "y_return_bps", None)
        if y_realized is None:
            y_realized = getattr(oos_set, "y_edge_bps", None)
        if y_realized is None:
            continue

        events_df: pd.DataFrame = getattr(oos_set, "event_index", pd.DataFrame())
        if events_df.empty or "symbol" not in events_df.columns:
            continue

        pred: NDArray[np.float64] = np.asarray(
            fold_out.model_output.expected_net_bps, dtype=np.float64
        )
        realized: NDArray[np.float64] = np.asarray(y_realized, dtype=np.float64)

        if len(pred) != len(realized) or len(pred) != len(events_df):
            continue

        symbols_arr = events_df["symbol"].to_numpy()

        for sym in np.unique(symbols_arr):
            sym_mask = symbols_arr == sym
            p = pred[sym_mask]
            r = realized[sym_mask]
            valid_mask = np.isfinite(p) & np.isfinite(r)
            if valid_mask.sum() < 4:
                continue
            if not _is_non_constant_finite_array(p[valid_mask]):
                continue
            if not _is_non_constant_finite_array(r[valid_mask]):
                continue
            ic_val, _ = spearmanr(p[valid_mask], r[valid_mask])
            if not np.isnan(ic_val):
                sym_ic_lists[str(sym)].append(float(ic_val))

    return {sym: float(np.mean(ics)) for sym, ics in sym_ic_lists.items() if ics}


def _nw_tstat_realized(r_sym: NDArray[np.float64]) -> float:
    """Bartlett NW HAC t-stat on a realized return series.

    Uses lag m = clip(n//20, 1, n-1) (5-percentile bandwidth).
    Returns 0.0 for n<4 or degenerate (std<1e-9) series.

    Args:
        r_sym: 1-D realized return array (bps), shape [N].

    Returns:
        NW HAC t-statistic.

    Time Complexity: O(N * m)
    Space Complexity: O(N)
    """
    n = len(r_sym)
    if n < 4:
        return 0.0
    if float(np.std(r_sym)) < 1e-9:
        return 0.0
    mu = float(np.mean(r_sym))
    demeaned = r_sym - mu
    m = min(n - 1, max(1, n // 20))
    gamma0 = float(np.dot(demeaned, demeaned)) / n
    gamma_sum = gamma0
    for j in range(1, m + 1):
        w = 1.0 - j / (m + 1)
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j])) / n
        gamma_sum += 2.0 * w * gamma_j
    se_hac = float(np.sqrt(max(gamma_sum, 1e-20) / n))
    return mu / se_hac if se_hac > 1e-20 else 0.0


def _compute_fold_realized_valid_set(
    fold_out: Any,
    *,
    min_obs: int = 20,
    t_stat_floor: float = 1.96,
) -> frozenset[str]:
    """Per-fold: symbols passing realized NW t-stat QC (BUG-B 방어).

    예측값 분산이 아닌 실현 y_return_bps 기반으로 심볼 유효성을 판정한다.
    → 상수 예측 심볼의 t-stat 폭발이 breadth 측정을 오염하지 않는다.

    Args:
        fold_out: fold 출력 (oos_set.y_return_bps, oos_set.event_index 필요).
        min_obs: 최소 이벤트 수.
        t_stat_floor: 최소 |t-stat|.

    Returns:
        유효 심볼 frozenset.

    Time Complexity: O(S * E/S) = O(E)  — S=symbols, E=events
    Space Complexity: O(S)
    """
    if not _is_trained_fold_output(fold_out):
        return frozenset()
    oos_set = getattr(fold_out, "oos_set", None)
    if oos_set is None:
        return frozenset()
    y_realized = getattr(oos_set, "y_return_bps", None)
    if y_realized is None:
        y_realized = getattr(oos_set, "y_edge_bps", None)
    if y_realized is None:
        return frozenset()
    events_df: pd.DataFrame = getattr(oos_set, "event_index", pd.DataFrame())
    if events_df.empty or "symbol" not in events_df.columns:
        return frozenset()

    realized = np.asarray(y_realized, dtype=np.float64)
    symbols_arr = events_df["symbol"].to_numpy()
    if len(realized) != len(symbols_arr):
        return frozenset()

    valid_syms: set[str] = set()
    for sym in np.unique(symbols_arr):
        mask = symbols_arr == sym
        r_sym = realized[mask]
        r_sym = r_sym[np.isfinite(r_sym)]
        if len(r_sym) < min_obs:
            continue
        t = _nw_tstat_realized(r_sym)
        if abs(t) >= t_stat_floor:
            valid_syms.add(str(sym))
    return frozenset(valid_syms)


def compute_per_symbol_realized_stats(
    *,
    fold_tuples: list[tuple[int, Any, Any]],
    min_obs: int,
    t_stat_floor: float,
    per_symbol_ic: dict[str, float],
) -> dict[str, SymbolRealizedStat]:
    """fold-pooled 실현 수익 기반 per-symbol QC (BUG-A+B 교정).

    QC 기준: 실현 엣지 NW t-stat (예측값 독립) + IC 부호 정합성.
    예측이 상수여도 실현 라벨이 유의 양이면 valid=True (BUG-B 무해화).

    Args:
        fold_tuples: (fold_idx, wf_fold, fold_out) 리스트.
        min_obs: 최소 이벤트 수 (fold 합산).
        t_stat_floor: 최소 |t-stat|.
        per_symbol_ic: compute_per_symbol_ic 결과 (IC 부호 정합 검증용).

    Returns:
        symbol → SymbolRealizedStat 매핑.

    Time Complexity: O(F * E)
    Space Complexity: O(S * E/S) = O(E)
    """
    sym_returns: dict[str, list[float]] = defaultdict(list)

    for _, _, fold_out in fold_tuples:
        if not _is_trained_fold_output(fold_out):
            continue
        oos_set = getattr(fold_out, "oos_set", None)
        if oos_set is None:
            continue
        y_realized = getattr(oos_set, "y_return_bps", None)
        if y_realized is None:
            y_realized = getattr(oos_set, "y_edge_bps", None)
        if y_realized is None:
            continue
        events_df: pd.DataFrame = getattr(oos_set, "event_index", pd.DataFrame())
        if events_df.empty or "symbol" not in events_df.columns:
            continue

        realized = np.asarray(y_realized, dtype=np.float64)
        symbols_arr = events_df["symbol"].to_numpy()
        if len(realized) != len(symbols_arr):
            continue

        for sym in np.unique(symbols_arr):
            mask = symbols_arr == sym
            r_sym = realized[mask]
            r_valid = r_sym[np.isfinite(r_sym)]
            sym_returns[str(sym)].extend(r_valid.tolist())

    result: dict[str, SymbolRealizedStat] = {}
    for sym, returns_list in sym_returns.items():
        r_arr = np.asarray(returns_list, dtype=np.float64)
        n = len(r_arr)
        mu = float(np.mean(r_arr)) if n > 0 else 0.0
        t = _nw_tstat_realized(r_arr) if n >= 4 else 0.0
        ic = per_symbol_ic.get(sym, 0.0)
        valid = (
            n >= min_obs
            and abs(t) >= t_stat_floor
            and bool(np.isfinite(mu))
            and bool(np.isfinite(t))
            and ic > 0.0
        )
        result[sym] = SymbolRealizedStat(
            realized_mu_bps=mu,
            t_stat=t,
            n_obs=n,
            ic=ic,
            valid=valid,
        )
    return result


def compute_per_strategy_oos_validation(
    *,
    fold_tuples: list[tuple[int, Any, Any]],
    min_obs: int = 30,
    t_stat_floor: float = 1.5,
    consistency_floor: float = 0.60,
) -> tuple[StrategySignal, ...]:
    """rule-family:variant별 OOS 독립검증."""
    per_strategy_realized: dict[str, list[float]] = defaultdict(list)
    per_strategy_fold_edge: dict[str, dict[int, float]] = defaultdict(dict)

    for fold_idx, _, fold_out in fold_tuples:
        if not _is_trained_fold_output(fold_out):
            continue
        oos_set = getattr(fold_out, "oos_set", None)
        if oos_set is None:
            continue

        y_realized = getattr(oos_set, "y_return_bps", None)
        if y_realized is None:
            y_realized = getattr(oos_set, "y_edge_bps", None)
        events_df: pd.DataFrame = getattr(oos_set, "event_index", pd.DataFrame())
        if y_realized is None or events_df.empty:
            continue

        realized = np.asarray(y_realized, dtype=np.float64)
        if len(realized) != len(events_df):
            continue

        if "family" in events_df.columns:
            family_col = events_df["family"].astype(str)
        elif "archetype" in events_df.columns:
            family_col = events_df["archetype"].astype(str)
        else:
            family_col = pd.Series(["_unknown"] * len(events_df), index=events_df.index, dtype="object")
        if "variant" in events_df.columns:
            variant_col = events_df["variant"].astype(str)
        else:
            variant_col = pd.Series(["_unknown"] * len(events_df), index=events_df.index, dtype="object")

        fold_bucket: dict[str, list[float]] = defaultdict(list)
        for idx, value in enumerate(realized):
            if not np.isfinite(value):
                continue
            strategy_id = f"{family_col.iat[idx]}:{variant_col.iat[idx]}"
            per_strategy_realized[strategy_id].append(float(value))
            fold_bucket[strategy_id].append(float(value))

        for strategy_id, values in fold_bucket.items():
            if values:
                per_strategy_fold_edge[strategy_id][fold_idx] = float(np.mean(values))

    panel: list[StrategySignal] = []
    for strategy_id in sorted(per_strategy_realized):
        realized_clean = np.asarray(per_strategy_realized[strategy_id], dtype=np.float64)
        if len(realized_clean) == 0:
            continue
        fold_edges = tuple(
            sorted((int(fold_id), float(edge)) for fold_id, edge in per_strategy_fold_edge.get(strategy_id, {}).items())
        )
        n_folds = len(fold_edges)
        fold_consistency = (
            float(sum(1 for _, edge in fold_edges if edge > 0.0) / n_folds)
            if n_folds > 0 else 0.0
        )
        nw_tstat = _nw_tstat_realized(realized_clean)
        panel.append(
            StrategySignal(
                strategy_id=strategy_id,
                oos_edge_bps=float(np.mean(realized_clean)),
                oos_nw_tstat=nw_tstat,
                hit_rate=float(np.mean(realized_clean > 0.0)),
                fold_sign_consistency=fold_consistency,
                n_obs=len(realized_clean),
                n_folds=n_folds,
                valid=bool(
                    len(realized_clean) >= min_obs
                    and nw_tstat >= t_stat_floor
                    and fold_consistency >= consistency_floor
                ),
                _fold_edges=fold_edges,
            )
        )
    return tuple(panel)


def compute_panel_diversity(panel: tuple[StrategySignal, ...]) -> float:
    """유효 전략 fold-edge 상관 기반 다양성."""
    valid_panel = [sig for sig in panel if sig.valid]
    if len(valid_panel) < 2:
        return 0.0

    pairwise_abs_corr: list[float] = []
    for idx, left in enumerate(valid_panel[:-1]):
        left_map = dict(left._fold_edges)
        for right in valid_panel[idx + 1:]:
            right_map = dict(right._fold_edges)
            common_folds = sorted(set(left_map) & set(right_map))
            if len(common_folds) < 2:
                pairwise_abs_corr.append(1.0)
                continue
            left_vec = np.asarray([left_map[k] for k in common_folds], dtype=np.float64)
            right_vec = np.asarray([right_map[k] for k in common_folds], dtype=np.float64)
            if not _is_non_constant_finite_array(left_vec) or not _is_non_constant_finite_array(right_vec):
                pairwise_abs_corr.append(1.0)
                continue
            corr = float(np.corrcoef(left_vec, right_vec)[0, 1])
            pairwise_abs_corr.append(abs(corr) if np.isfinite(corr) else 1.0)

    if not pairwise_abs_corr:
        return 0.0
    return float(np.clip(1.0 - float(np.mean(pairwise_abs_corr)), 0.0, 1.0))


def compute_breadth_weighted_ic(
    per_symbol_ic: dict[str, float],
    per_symbol_n: dict[str, int],
) -> tuple[float, float]:
    """이벤트 가중 평균 per-symbol IC + cross-symbol IC IR t-stat (BUG-C 교정).

    글로벌 풀 Spearman 대비: 이종 심볼 vol/레벨 rank 오염 없이 타이밍 알파를 측정.

    ic_mean_weighted = Σ(n_i · IC_i) / Σn_i
    ic_ir_tstat      = mean(IC_i) / (std(IC_i) / sqrt(S))   [cross-symbol IC IR]

    Args:
        per_symbol_ic: symbol → TS rank IC.
        per_symbol_n: symbol → 이벤트 수 (가중치).

    Returns:
        (ic_mean_weighted, ic_ir_tstat) 튜플.

    Time Complexity: O(S)
    Space Complexity: O(S)
    """
    if not per_symbol_ic:
        return 0.0, 0.0

    syms = list(per_symbol_ic.keys())
    ic_arr = np.array([per_symbol_ic[s] for s in syms], dtype=np.float64)
    n_arr = np.array([float(max(per_symbol_n.get(s, 1), 1)) for s in syms], dtype=np.float64)

    total_n = float(n_arr.sum())
    ic_weighted = float(np.dot(ic_arr, n_arr) / total_n) if total_n > 0 else 0.0

    s = len(syms)
    if s < 2:
        return ic_weighted, 0.0

    ic_mean = float(ic_arr.mean())
    ic_std = float(ic_arr.std(ddof=1))
    ic_ir = ic_mean / (ic_std / np.sqrt(s) + 1e-12)

    return ic_weighted, float(ic_ir)


def _newey_west_ic_tstat(
    pred: NDArray[np.float64],
    realized: NDArray[np.float64],
    max_lag: int | None = None,
) -> float:
    """Newey-West HAC t-stat for Spearman rank IC.

    Algorithm:
        1. rank-transform both series to rp, rr in [0,1]
        2. u_t = (rp_t - 0.5) * (rr_t - 0.5)
        3. ic_est = 12.0 * mean(u_t)  (approximation normalized to ~[-1,1])
        4. HAC: S_NW = (1/N) * sum_{|l|<=L} w_l * gamma_l
           L = int(4*(N/100)^(2/9)) [Andrews 1991 automatic]
           w_l = 1 - |l|/(L+1)  (Bartlett kernel)
           gamma_l = autocovariance(u, lag=l)
        5. t_stat = ic_est / sqrt(max(S_NW, 1e-12) / N)

    Returns 0.0 if N < 4 or all values identical.

    Args:
        pred: Predicted values array, shape [N].
        realized: Realized values array, shape [N].
        max_lag: NW bandwidth. None → Andrews automatic.

    Returns:
        NW HAC t-statistic (float).

    Time Complexity: O(N * L)
    Space Complexity: O(N)
    """
    n_obs = len(pred)
    if n_obs < 4:
        return 0.0

    from scipy.stats import rankdata

    # rank transform → [0, 1]
    rp = rankdata(pred).astype(np.float64) / n_obs
    rr = rankdata(realized).astype(np.float64) / n_obs

    # cross-product series
    u = (rp - 0.5) * (rr - 0.5)
    ic_est = 12.0 * float(np.mean(u))

    # HAC bandwidth (Andrews 1991)
    nw_lag = max_lag if max_lag is not None else int(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))
    nw_lag = max(1, min(nw_lag, n_obs - 1))

    # autocovariance gamma_l
    u_dm = u - np.mean(u)
    gamma_0 = float(np.dot(u_dm, u_dm)) / n_obs

    s_nw = gamma_0
    for lag in range(1, nw_lag + 1):
        gamma_l = float(np.dot(u_dm[lag:], u_dm[:-lag])) / n_obs
        w_l = 1.0 - lag / (nw_lag + 1.0)  # Bartlett kernel
        s_nw += 2.0 * w_l * gamma_l

    s_nw = max(s_nw, 1e-12)
    # SE(IC_hat) = 12 * sqrt(S_NW / N) since IC_hat = 12 * mean(u).
    # 분자(ic_est)와 동일 스케일 적용 — 누락 시 |t| 12배 과대평가.
    se_ic = 12.0 * np.sqrt(s_nw / n_obs)
    t_stat = ic_est / (se_ic + 1e-12)
    return float(t_stat)


def _compute_fold_ts_ic(*, fold_out: Any) -> float | None:
    """fold OOS pooled time-series Spearman rank IC (expected_net_bps vs y_return_bps).

    oos_set.y_return_bps는 방향·barrier·비용 반영 정준 실현 수익 라벨이다.
    Returns None if oos_set unavailable or fewer than 4 valid events.
    """
    if not _is_trained_fold_output(fold_out):
        return None
    oos_set = getattr(fold_out, "oos_set", None)
    if oos_set is None:
        return None

    y_realized = getattr(oos_set, "y_return_bps", None)
    if y_realized is None:
        y_realized = getattr(oos_set, "y_edge_bps", None)
    if y_realized is None:
        return None

    pred: NDArray[np.float64] = np.asarray(
        fold_out.model_output.expected_net_bps, dtype=np.float64
    )
    realized: NDArray[np.float64] = np.asarray(y_realized, dtype=np.float64)

    if len(pred) != len(realized) or len(pred) < 4:
        return None

    mask = np.isfinite(pred) & np.isfinite(realized)
    if mask.sum() < 4:
        return None
    if not _is_non_constant_finite_array(pred[mask]):
        return None
    if not _is_non_constant_finite_array(realized[mask]):
        return None

    ic_val, _ = spearmanr(pred[mask], realized[mask])
    return float(ic_val) if not np.isnan(ic_val) else None


# ---------------------------------------------------------------------------
# C0 Diagnostic: Prediction Decomposition
# ---------------------------------------------------------------------------

_N_REGIMES_DEFAULT: int = 6  # Binance regime codes 1-6


def compute_prediction_decomposition_diag(
    *,
    fold_tuples: list[tuple[int, Any, Any]],
) -> PredictionDecompositionDiag:
    """OOS 이벤트에서 예측의 정적/동적 분산 분해 + archetype 엣지 + decile lift (진단 전용).

    어떤 게이트 값도 변경하지 않는다. 순수 관측.

    Args:
        fold_tuples: (fold_idx, wf_fold, fold_out) 리스트.

    Returns:
        PredictionDecompositionDiag — 분석 결과.

    Time Complexity: O(F * E)
    Space Complexity: O(E) — 전 fold 이벤트 합산
    """
    all_pred: list[NDArray[np.float64]] = []
    all_real: list[NDArray[np.float64]] = []
    all_archetype: list[list[str]] = []
    all_regime: list[list[int]] = []
    all_variant: list[list[str]] = []
    score_cal_ratios: list[float] = []

    for _, _, fold_out in fold_tuples:
        if not _is_trained_fold_output(fold_out):
            continue
        oos_set = getattr(fold_out, "oos_set", None)
        if oos_set is None:
            continue

        y_realized = getattr(oos_set, "y_return_bps", None)
        if y_realized is None:
            y_realized = getattr(oos_set, "y_edge_bps", None)
        if y_realized is None:
            continue

        events_df: pd.DataFrame = getattr(oos_set, "event_index", pd.DataFrame())
        if events_df.empty:
            continue

        pred: NDArray[np.float64] = np.asarray(
            fold_out.model_output.expected_net_bps, dtype=np.float64
        )
        real: NDArray[np.float64] = np.asarray(y_realized, dtype=np.float64)

        if len(pred) != len(real) or len(pred) != len(events_df):
            continue

        mask = np.isfinite(pred) & np.isfinite(real)
        if mask.sum() < 4:
            continue

        pred_m = pred[mask]
        real_m = real[mask]
        ev_m = events_df.loc[mask.tolist() if not isinstance(mask, np.ndarray) else events_df.index[mask]]

        n_m = int(mask.sum())
        arch_col = (
            ev_m["archetype"].astype(str).tolist() if "archetype" in ev_m.columns
            else ["_unknown"] * n_m
        )
        regime_col: list[int] = []
        if "entry_regime_code" in ev_m.columns:
            regime_col = [int(v) if pd.notna(v) else -1 for v in ev_m["entry_regime_code"]]
        else:
            regime_col = [-1] * n_m
        variant_col = (
            ev_m["variant"].astype(str).tolist() if "variant" in ev_m.columns
            else ["_unknown"] * n_m
        )

        all_pred.append(pred_m)
        all_real.append(real_m)
        all_archetype.append(arch_col)
        all_regime.append(regime_col)
        all_variant.append(variant_col)

        # score_cal_valid_ratio 집계
        val_diag = getattr(fold_out.model_output, "validation_diagnostics", {})
        ens_diag = val_diag.get("ensemble_diagnostics", {}) if isinstance(val_diag, dict) else {}
        n_valid_reg = int(ens_diag.get("num_valid_regimes", 0)) if isinstance(ens_diag, dict) else 0
        n_unique_reg = max(len(set(regime_col) - {-1}), 1)
        score_cal_ratios.append(float(n_valid_reg) / float(n_unique_reg))

    # ── 빈 케이스 ────────────────────────────────────────────────────────────
    if not all_pred:
        return PredictionDecompositionDiag(
            static_variance_share=0.0,
            dynamic_variance_share=0.0,
            score_cal_valid_ratio=0.0,
            per_archetype_oos_edge={},
            decile_lift_bps=0.0,
        )

    pred_arr = np.concatenate(all_pred, axis=0)   # [E]
    real_arr = np.concatenate(all_real, axis=0)   # [E]
    arch_arr = np.array([a for sub in all_archetype for a in sub])
    regime_arr = np.array([r for sub in all_regime for r in sub], dtype=np.int32)
    variant_arr = np.array([v for sub in all_variant for v in sub])

    n_total = len(pred_arr)

    # ── 정적 분산 비율 ─────────────────────────────────────────────────────
    total_var = float(np.var(pred_arr)) if n_total > 1 else 0.0
    static_var = 0.0
    if total_var > 1e-20:
        group_keys = [f"{a}|{r}|{v}" for a, r, v in zip(arch_arr, regime_arr, variant_arr, strict=True)]
        group_key_arr = np.array(group_keys)
        group_mean_arr: NDArray[np.float64] = np.zeros(n_total, dtype=np.float64)
        for gk in np.unique(group_key_arr):
            gm = group_key_arr == gk
            n_g = int(gm.sum())
            if n_g < 2:
                group_mean_arr[gm] = pred_arr[gm]
                continue
            group_mean_arr[gm] = float(np.mean(pred_arr[gm]))
        static_var = float(np.var(group_mean_arr)) if n_total > 1 else 0.0

    static_share = float(np.clip(static_var / (total_var + 1e-20), 0.0, 1.0))
    dynamic_share = float(max(0.0, 1.0 - static_share))

    # ── archetype 실현엣지 ─────────────────────────────────────────────────
    per_archetype_oos_edge: dict[str, tuple[float, float]] = {}
    for arch in np.unique(arch_arr):
        a_mask = arch_arr == arch
        r_sub: NDArray[np.float64] = real_arr[a_mask]
        if len(r_sub) < 4:
            continue
        mu_arch = float(np.mean(r_sub))
        t_arch = _nw_tstat_realized(r_sub)
        per_archetype_oos_edge[str(arch)] = (mu_arch, t_arch)

    # ── decile lift ────────────────────────────────────────────────────────
    decile_lift_bps = 0.0
    if n_total >= 20:
        n10 = max(1, n_total // 10)
        order = np.argsort(pred_arr)
        top_real = real_arr[order[-n10:]]
        bot_real = real_arr[order[:n10]]
        decile_lift_bps = float(np.mean(top_real) - np.mean(bot_real))

    score_cal_valid_ratio = float(np.mean(score_cal_ratios)) if score_cal_ratios else 0.0

    return PredictionDecompositionDiag(
        static_variance_share=static_share,
        dynamic_variance_share=dynamic_share,
        score_cal_valid_ratio=score_cal_valid_ratio,
        per_archetype_oos_edge=per_archetype_oos_edge,
        decile_lift_bps=decile_lift_bps,
    )


# ---------------------------------------------------------------------------
# C5 Diagnostic: Fold-level Regime & Archetype Analysis
# ---------------------------------------------------------------------------


def _log_fold_regime_analysis(
    *,
    fold_tuples: list[tuple[int, Any, Any]],
    datetimes: NDArray[np.datetime64],
) -> None:
    """각 fold OOS 날짜 범위·regime 분포·archetype별 실현mu를 [SWF-DIAG-FOLDn] 로깅.

    게이트 불변. 순수 진단. fold 5 비정상성 원인 파악용.

    Args:
        fold_tuples: (fold_idx, wf_fold, fold_out) 리스트.
        datetimes: aligned.datetimes — bar 인덱스 → 날짜 변환에 사용.
    """
    n_bars = len(datetimes)

    for fold_idx, wf_fold, fold_out in fold_tuples:
        # 날짜 범위
        oos_s = int(getattr(wf_fold, "oos_start", 0))
        oos_e = int(getattr(wf_fold, "oos_end", 0))
        oos_s_clamp = max(0, min(oos_s, n_bars - 1))
        oos_e_clamp = max(0, min(oos_e - 1, n_bars - 1))
        date_start = str(datetimes[oos_s_clamp])[:10]
        date_end = str(datetimes[oos_e_clamp])[:10]

        if not _is_trained_fold_output(fold_out):
            logger.info(
                "[SWF-DIAG-FOLD%d] %s~%s fit_status=%s SKIP",
                fold_idx + 1, date_start, date_end,
                getattr(fold_out, "fit_status", "unknown"),
            )
            continue

        oos_set = getattr(fold_out, "oos_set", None)
        if oos_set is None:
            logger.info("[SWF-DIAG-FOLD%d] %s~%s oos_set=None SKIP", fold_idx + 1, date_start, date_end)
            continue

        events_df: pd.DataFrame = getattr(oos_set, "event_index", pd.DataFrame())
        y_raw = getattr(oos_set, "y_return_bps", None)
        if events_df.empty or y_raw is None:
            logger.info("[SWF-DIAG-FOLD%d] %s~%s no_events SKIP", fold_idx + 1, date_start, date_end)
            continue

        y_arr: NDArray[np.float64] = np.asarray(y_raw, dtype=np.float64)
        n_ev = len(y_arr)

        # regime 분포
        regime_dist = ""
        if "entry_regime_code" in events_df.columns and n_ev > 0:
            rc = events_df["entry_regime_code"].dropna().astype(int)
            counts = rc.value_counts().sort_index()
            regime_dist = " ".join(f"r{k}:{v}" for k, v in counts.items())

        # archetype별 realized mu
        arch_mu_str = ""
        if "archetype" in events_df.columns and n_ev == len(events_df):
            arch_col = events_df["archetype"].astype(str).to_numpy()
            arch_parts = []
            for arch in np.unique(arch_col):
                mask = arch_col == arch
                r_sub = y_arr[mask]
                finite = r_sub[np.isfinite(r_sub)]
                if len(finite) >= 4:
                    mu = float(np.mean(finite))
                    label = arch.replace("_reversion", "").replace("_continuation", "").replace("time_series_", "ts_")
                    arch_parts.append(f"{label}:{mu:.1f}(n={len(finite)})")
            arch_mu_str = " | ".join(arch_parts)

        # CS IC (fold_ic)
        pred_arr: NDArray[np.float64] = np.asarray(
            fold_out.model_output.expected_net_bps, dtype=np.float64
        )
        cs_ic_str = "ic=N/A"
        if len(pred_arr) == n_ev and n_ev >= 4:
            mask_f = np.isfinite(pred_arr) & np.isfinite(y_arr)
            if mask_f.sum() >= 4:
                from scipy.stats import spearmanr as _sr
                _ic, _ = _sr(pred_arr[mask_f], y_arr[mask_f])
                cs_ic_str = f"ic={float(_ic):.4f}"

        logger.info(
            "[SWF-DIAG-FOLD%d] %s~%s n=%d %s | regime[%s] | arch[%s]",
            fold_idx + 1, date_start, date_end, n_ev, cs_ic_str, regime_dist, arch_mu_str,
        )


# ---------------------------------------------------------------------------
# Layer1: SWF-K Signal Validation
# ---------------------------------------------------------------------------


def run_l1_swf(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    folds: tuple[WFFold, ...],
    l1_params: dict[str, Any],
    min_obs: int = 20,
    t_stat_floor: float = 1.96,
    tf: str = "4h",
) -> Layer1Result:
    """Layer1 SWF-K 신호 검증.

    각 WFFold에서 모델 학습/예측 후 SymbolSignal 집계, Pooled IC + NW HAC 게이트 적용.

    Args:
        labeled_events: 레이블링된 이벤트 DataFrame.
        aligned: AlignedMarketData (close_2d, symbols, datetimes 포함).
        cfg: CandidateStrategyConfig.
        folds: WFFold 튜플 (build_l1_swf_folds 출력).
        l1_params: Layer1 하이퍼파라미터 dict.
        min_obs: QC 최소 관측 수.
        t_stat_floor: QC 최소 |t-stat|.
        tf: 타임프레임 문자열.

    Returns:
        Layer1Result.

    Time Complexity: O(F * T * N)
    Space Complexity: O(F * N)
    """
    from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars

    purge_bars, _embargo_bars = resolve_purge_and_embargo_bars(cfg)

    import multiprocessing
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor

    import src.domain.futures.strategy.candidate_workflow as cw

    planned_workers = max(1, (os.cpu_count() or 4) // 2)
    max_workers = min(len(folds), planned_workers)

    t_start = time.perf_counter()
    logger.debug(
        "[SWF-START] Starting SWF-K L1 signal validation with %d folds (max_workers=%d)",
        len(folds),
        max_workers,
    )

    signals_per_fold: list[dict[str, SymbolSignal]] = []
    fold_diags: list[FoldDiagnostic] = []
    symbols = aligned.symbols
    n_total = len(symbols)

    cw._GLOBAL_LABELED_EVENTS = labeled_events
    cw._GLOBAL_ALIGNED = aligned
    cw._GLOBAL_CFG = cfg
    cw._GLOBAL_PURGE_BARS = purge_bars
    mp_ctx = multiprocessing.get_context("fork")

    futures: list[tuple[int, WFFold, Any]] = []
    try:
        if max_workers <= 1 or len(folds) <= 1:
            for fold_idx, wf_fold in enumerate(folds):
                try:
                    fold_out = _fit_and_predict_single_fold(
                        fold_idx, wf_fold, labeled_events, aligned, cfg, purge_bars
                    )
                    futures.append((fold_idx, wf_fold, fold_out))
                except Exception:
                    logger.warning("run_l1_swf: fold %d 학습 실패, 스킵", fold_idx, exc_info=True)
        else:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
                submits: list[tuple[int, WFFold, Any]] = []
                for fold_idx, wf_fold in enumerate(folds):
                    submits.append(
                        (
                            fold_idx,
                            wf_fold,
                            executor.submit(
                                cw._fit_and_predict_single_fold_from_globals, fold_idx, wf_fold
                            ),
                        )
                    )
                for fold_idx, wf_fold, fut in submits:
                    try:
                        fold_out = fut.result()
                        futures.append((fold_idx, wf_fold, fold_out))
                    except Exception:
                        logger.warning("run_l1_swf: fold %d 학습 실패, 스킵", fold_idx, exc_info=True)
    finally:
        cw._GLOBAL_LABELED_EVENTS = None
        cw._GLOBAL_ALIGNED = None
        cw._GLOBAL_CFG = None
        cw._GLOBAL_PURGE_BARS = None

    for fold_loop_idx, (_fold_idx, _wf_fold, fold_out) in enumerate(futures):
        beta_f32 = aligned.beta_vs_market_1d
        beta_f64: NDArray[np.float64] | None = (
            beta_f32.astype(np.float64) if beta_f32 is not None else None
        )
        fold_sigs: dict[str, SymbolSignal] = {}
        if _is_trained_fold_output(fold_out):
            fold_sigs = compose_symbol_signals(
                model_output=fold_out.model_output,
                close_2d=aligned.close_2d,
                symbols=symbols,
                tf=tf,
                min_obs=min_obs,
                t_stat_floor=t_stat_floor,
                beta_vs_market_1d=beta_f64,
                opt_cfg=None,
            )
            signals_per_fold.append(fold_sigs)

        # time-series pooled rank IC (expected_net_bps vs oos_set.y_return_bps)
        fold_ic: float | None = _compute_fold_ts_ic(fold_out=fold_out)

        eligible_mask = _fold_eligible_symbol_mask(aligned=aligned, fold=_wf_fold)
        f_n_eligible = int(np.count_nonzero(eligible_mask))
        # BUG-B fix: realized 기반 per-fold valid (예측 분산 오염 제거)
        fold_realized_valid = _compute_fold_realized_valid_set(
            fold_out, min_obs=min_obs, t_stat_floor=t_stat_floor
        )
        eligible_syms = {s for s, e in zip(symbols, eligible_mask, strict=True) if e}
        f_n_valid = len(fold_realized_valid & eligible_syms)
        f_breadth = f_n_valid / max(1, f_n_eligible)
        f_n_events = len(fold_out.model_output.expected_net_bps)
        fold_diags.append(FoldDiagnostic(
            fold=fold_loop_idx + 1,
            ic=fold_ic,
            breadth=f_breadth,
            n_valid=f_n_valid,
            n_eligible=f_n_eligible,
            n_events=f_n_events,
            n_fit=int(getattr(fold_out, "n_fit", 0)),
            fit_status=getattr(fold_out, "fit_status", "failed"),
            passed=fold_ic is not None and fold_ic > 0,
        ))

    # timing_profile 집계
    total_folds = len(futures)
    if total_folds > 0:
        avg_profile = dict.fromkeys(
            (
                "schema",
                "dataset_fit",
                "dataset_early_stop",
                "dataset_calibration_fit",
                "dataset_calibration_eval",
                "dataset_oos",
                "edge_fit",
                "inference",
                "selection",
            ),
            0.0,
        )
        for _, _, fold_out in futures:
            prof = getattr(fold_out, "timing_profile", {})
            for k in avg_profile:
                avg_profile[k] += prof.get(k, 0.0)

        for k in avg_profile:
            avg_profile[k] /= total_folds

        logger.debug(
            "[SWF-PROFILE] Average sub-fold execution breakdown: "
            "schema=%.3fs, ds_fit=%.3fs, ds_es=%.3fs, ds_cal_fit=%.3fs, ds_cal_eval=%.3fs, "
            "ds_oos=%.3fs, edge_fit=%.3fs, inference=%.3fs, selection=%.3fs",
            avg_profile["schema"],
            avg_profile["dataset_fit"],
            avg_profile["dataset_early_stop"],
            avg_profile["dataset_calibration_fit"],
            avg_profile["dataset_calibration_eval"],
            avg_profile["dataset_oos"],
            avg_profile["edge_fit"],
            avg_profile["inference"],
            avg_profile["selection"],
        )

    logger.debug(
        "[SWF-END] SWF-K L1 signal validation completed in %.2fs",
        time.perf_counter() - t_start,
    )

    # ── Per-symbol IC (정준: 예측 vs 실현, BUG-C 기반) ──────────────────
    per_sym_ic = compute_per_symbol_ic(fold_tuples=futures)

    # ── Realized stats: fold-pooled 실현 엣지 QC (BUG-A+B 교정) ─────────
    per_sym_realized = compute_per_symbol_realized_stats(
        fold_tuples=futures,
        min_obs=min_obs,
        t_stat_floor=t_stat_floor,
        per_symbol_ic=per_sym_ic,
    )

    # ── OOS stacking (realized stats 주입: BUG-A 제거) ───────────────────
    sigs_tuple = tuple(signals_per_fold)
    oos_stacked = _stack_oos_signals(sigs_tuple, realized_stats=per_sym_realized)
    strategy_panel = compute_per_strategy_oos_validation(fold_tuples=futures)
    n_valid_strategies = sum(1 for sig in strategy_panel if sig.valid)
    panel_diversity = compute_panel_diversity(strategy_panel)

    # --- IC 통계 (fold_diags 기반) ---
    fold_perf_details: list[dict[str, Any]] = [
        {
            "fold": d.fold,
            "ic": d.ic,
            "breadth": d.breadth,
            "n_valid": d.n_valid,
            "n_eligible": d.n_eligible,
            "n_events": d.n_events,
            "n_fit": d.n_fit,
            "fit_status": d.fit_status,
            "pass": d.passed,
        }
        for d in fold_diags
    ]

    valid_fold_ics = [float(d.ic) for d in fold_diags if d.ic is not None]
    if valid_fold_ics:
        cs_ic_mean = float(np.mean(valid_fold_ics))
        cs_ic_fold_pass_ratio = float(sum(1 for ic in valid_fold_ics if ic > 0.0) / len(valid_fold_ics))
        if len(valid_fold_ics) >= 2:
            cs_ic_std = float(np.std(valid_fold_ics, ddof=1))
            cs_ic_tstat = float(
                cs_ic_mean / (cs_ic_std / np.sqrt(len(valid_fold_ics)) + 1e-12)
            )
        else:
            cs_ic_tstat = 0.0
    else:
        cs_ic_mean = 0.0
        cs_ic_tstat = 0.0
        cs_ic_fold_pass_ratio = 0.0

    # ── Diagnostic: 글로벌 풀 Spearman IC (강등, 게이트 미사용) ─────────
    _pred_parts: list[NDArray[np.float64]] = []
    _real_parts: list[NDArray[np.float64]] = []
    for _, _wf_fold, fold_out in futures:
        if not _is_trained_fold_output(fold_out):
            continue
        oos = getattr(fold_out, "oos_set", None)
        if oos is None:
            continue
        _y_ret = getattr(oos, "y_return_bps", None)
        _y_edg = getattr(oos, "y_edge_bps", None)
        y_lab = _y_ret if _y_ret is not None else _y_edg
        if y_lab is None:
            continue
        p_arr = np.asarray(fold_out.model_output.expected_net_bps, dtype=np.float64)
        r_arr = np.asarray(y_lab, dtype=np.float64)
        if len(p_arr) != len(r_arr) or len(p_arr) < 4:
            continue
        _mask = np.isfinite(p_arr) & np.isfinite(r_arr)
        if _mask.sum() < 4:
            continue
        if not _is_non_constant_finite_array(p_arr[_mask]):
            continue
        if not _is_non_constant_finite_array(r_arr[_mask]):
            continue
        _pred_parts.append(p_arr[_mask])
        _real_parts.append(r_arr[_mask])

    if _pred_parts:
        _p_all = np.concatenate(_pred_parts)
        _r_all = np.concatenate(_real_parts)
        _global_ic_raw, _ = spearmanr(_p_all, _r_all)
        _global_ic = float(_global_ic_raw) if not np.isnan(_global_ic_raw) else 0.0
        _global_tstat = _newey_west_ic_tstat(_p_all, _r_all)
    else:
        _global_ic = 0.0
        _global_tstat = 0.0
    logger.debug(
        "[SWF-IC-DIAG] global_pooled_ic=%.4f global_tstat=%.2f (diagnostic, not gate)",
        _global_ic,
        _global_tstat,
    )

    # ── Primary gate metrics: breadth-weighted IC + IC IR (BUG-C 교정) ──
    per_sym_n: dict[str, int] = {sym: s.n_obs for sym, s in per_sym_realized.items()}
    pooled_ic_val, pooled_tstat_val = compute_breadth_weighted_ic(per_sym_ic, per_sym_n)

    # ── Fold-wise event-weighted (diagnostic only) ─────────────────────
    _valid_pairs = [(d.ic, d.n_events) for d in fold_diags if d.ic is not None]
    if _valid_pairs:
        _w_total = sum(n for _, n in _valid_pairs)
        fold_pass_ratio = (
            sum(n for ic, n in _valid_pairs if ic > 0) / _w_total
            if _w_total > 0 else 0.0
        )
    else:
        fold_pass_ratio = 0.0

    # --- breadth / valid_coverage (FoldDiagnostic 기반, realized breadth) ---
    breadth = float(np.mean([d.breadth for d in fold_diags])) if fold_diags else 0.0
    valid_coverage = (
        float(sum(1 for d in fold_diags if d.breadth >= _VALID_COVERAGE_FLAG_THRESHOLD) / len(fold_diags))
        if fold_diags else 0.0
    )
    trained_fold_coverage = (
        float(sum(1 for d in fold_diags if d.fit_status == "trained") / len(fold_diags))
        if fold_diags else 0.0
    )

    # n_valid: realized stats 기준 (BUG-A 제거)
    n_valid = sum(1 for s in per_sym_realized.values() if s.valid)

    # Final Per-symbol aggregate (realized t_stat/valid 사용, BUG-A 제거)
    sym_details: list[dict[str, Any]] = []
    for sym, sig in sorted(oos_stacked.items()):
        real = per_sym_realized.get(sym)
        sym_details.append({
            "symbol": sym,
            "raw_mu": sig.raw_mu,
            "vol": sig.volatility,
            "t_stat": real.t_stat if real is not None else 0.0,
            "ic": per_sym_ic.get(sym, 0.0),
            "valid": real.valid if real is not None else False,
        })

    _diag = compute_prediction_decomposition_diag(fold_tuples=futures)
    gate_passed: bool = bool(
        (trained_fold_coverage >= _TRAINED_FOLD_COVERAGE_THRESHOLD)
        and (n_valid_strategies >= cfg.l1_min_valid_strategies)
        and (panel_diversity >= cfg.l1_min_panel_diversity)
        and (cs_ic_fold_pass_ratio >= cfg.l1_min_cs_fold_pass_ratio)
    )

    result = Layer1Result(
        signals_per_fold=sigs_tuple,
        oos_stacked=oos_stacked,
        pooled_ic=pooled_ic_val,
        pooled_tstat=pooled_tstat_val,
        breadth=breadth,
        valid_coverage=valid_coverage,
        fold_pass_ratio=fold_pass_ratio,
        gate_passed=gate_passed,
        n_valid=n_valid,
        n_total=n_total,
        n_trade_scope=n_total,
        cs_ic_mean=cs_ic_mean,
        cs_ic_tstat=cs_ic_tstat,
        cs_ic_fold_pass_ratio=cs_ic_fold_pass_ratio,
        decile_lift_bps=_diag.decile_lift_bps,
        strategy_panel=strategy_panel,
        n_valid_strategies=n_valid_strategies,
        panel_diversity=panel_diversity,
    )
    logger.info(format_layer1_table(result, fold_details=fold_perf_details, per_symbol_top10=sym_details))
    if strategy_panel:
        top_panel = sorted(
            strategy_panel,
            key=lambda item: (item.valid, item.oos_edge_bps, item.oos_nw_tstat),
            reverse=True,
        )[: min(10, len(strategy_panel))]
        panel_str = ", ".join(
            (
                f"{sig.strategy_id}:edge={sig.oos_edge_bps:.1f}"
                f"/t={sig.oos_nw_tstat:.2f}"
                f"/cons={sig.fold_sign_consistency:.2f}"
                f"/valid={'Y' if sig.valid else 'N'}"
            )
            for sig in top_panel
        )
        logger.info("[STRATEGY-PANEL] valid=%d diversity=%.3f | %s", n_valid_strategies, panel_diversity, panel_str)
    logger.info(
        "[SWF-LEGACY-IC] pooled_ic=%.4f pooled_tstat=%.2f breadth=%.3f valid_coverage=%.3f",
        pooled_ic_val,
        pooled_tstat_val,
        breadth,
        valid_coverage,
    )

    # ── C0 Diagnostic (게이트 불변, 관측 전용) ────────────────────────────
    logger.info(
        "[SWF-DIAG] static_share=%.3f dynamic_share=%.3f score_cal_ratio=%.3f decile_lift=%.2fbps",
        _diag.static_variance_share,
        _diag.dynamic_variance_share,
        _diag.score_cal_valid_ratio,
        _diag.decile_lift_bps,
    )
    if _diag.per_archetype_oos_edge:
        arch_lines = ", ".join(
            f"{a}: mu={m:.2f} t={t:.2f}" for a, (m, t) in sorted(_diag.per_archetype_oos_edge.items())
        )
        logger.info("[SWF-DIAG-ARCH] %s", arch_lines)

    # ── C5 Diagnostic: fold-level regime/archetype 분석 ────────────────────
    _log_fold_regime_analysis(fold_tuples=futures, datetimes=aligned.datetimes)

    return result


def _holding_bucket(holding_bars: int) -> int:
    if holding_bars <= 4:
        return 4
    if holding_bars <= 8:
        return 8
    if holding_bars <= 12:
        return 12
    if holding_bars <= 24:
        return 24
    return int(max(holding_bars, 1))


def _series_tstat(values: NDArray[np.float64]) -> float:
    if values.size < 2:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 0.0
    sigma = float(np.std(finite, ddof=1))
    if sigma <= 0.0:
        return 0.0
    return float(np.mean(finite) / (sigma / np.sqrt(finite.size)))


def _one_sided_p_value(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return 1.0
    return float(norm.sf(t_stat))


def _expected_gross_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.expected_gross_bps, dtype=np.float64)


def _q10_gross_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.q10_gross_bps, dtype=np.float64)


def _q90_gross_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.q90_gross_bps, dtype=np.float64)


def _signal_source_key_from_row(row: pd.Series) -> SignalSourceKey:
    strategy_id = str(
        row.get(
            "strategy_id",
            f"{row.get('family', '')}:{row.get('variant', '')}",
        )
    )
    activation_context = str(
        row.get(
            "activation_context",
            row.get("signal_cell", row.get("entry_regime", "all")),
        )
    )
    return SignalSourceKey(
        symbol=str(row.get("symbol", "")),
        strategy_id=strategy_id,
        activation_context=activation_context or "all",
    )


def _batch_to_frame(batch: ValidatedSignalBatch) -> pd.DataFrame:
    if not batch.events:
        return pd.DataFrame(
            columns=[
                "decision_idx",
                "decision_time",
                "symbol",
                "strategy_id",
                "activation_context",
                "side",
                "expected_gross_bps",
                "q10_gross_bps",
                "q90_gross_bps",
                "expected_holding_bars",
                "reliability",
                "registry_version",
                "model_version",
            ]
        )
    return pd.DataFrame(
        [
            {
                "decision_idx": event.decision_idx,
                "decision_time": event.decision_time,
                "symbol": event.symbol,
                "strategy_id": event.strategy_id,
                "activation_context": event.activation_context,
                "side": event.side,
                "expected_gross_bps": event.expected_gross_bps,
                "q10_gross_bps": event.q10_gross_bps,
                "q90_gross_bps": event.q90_gross_bps,
                "expected_holding_bars": event.expected_holding_bars,
                "reliability": event.reliability,
                "registry_version": event.registry_version,
                "model_version": event.model_version,
            }
            for event in batch.events
        ]
    )


def _by_q_values(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    if p_values.size == 0:
        return np.zeros((0,), dtype=np.float64)
    order = np.argsort(p_values)
    ordered = p_values[order]
    m = float(p_values.size)
    harmonic = float(np.sum(1.0 / np.arange(1, p_values.size + 1, dtype=np.float64)))
    adjusted = np.empty_like(ordered)
    running = 1.0
    for idx in range(ordered.size - 1, -1, -1):
        rank = float(idx + 1)
        candidate = min(1.0, ordered[idx] * m * harmonic / rank)
        running = min(running, candidate)
        adjusted[idx] = running
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _event_results_from_fold_output(
    *,
    fold_id: int,
    fold_out: Any,
) -> pd.DataFrame:
    event_frame = getattr(fold_out.model_output, "events", pd.DataFrame()).copy()
    if event_frame.empty:
        return event_frame
    gross_pred = _expected_gross_bps(fold_out.model_output)
    q10_pred = _q10_gross_bps(fold_out.model_output)
    q90_pred = _q90_gross_bps(fold_out.model_output)
    size = min(len(event_frame), gross_pred.size)
    event_frame = event_frame.iloc[:size].reset_index(drop=True)
    event_frame["expected_gross_bps"] = gross_pred[:size]
    event_frame["q10_gross_bps"] = q10_pred[:size]
    event_frame["q90_gross_bps"] = q90_pred[:size]
    event_frame["fold_id"] = int(fold_id)
    event_frame["decision_idx"] = (
        pd.to_numeric(
            event_frame.get("entry_idx", pd.Series(0, index=event_frame.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        - 1
    )
    if "strategy_id" not in event_frame.columns:
        event_frame["strategy_id"] = (
            event_frame.get("family", pd.Series("", index=event_frame.index)).astype(str)
            + ":"
            + event_frame.get("variant", pd.Series("", index=event_frame.index)).astype(str)
        )
    if "activation_context" not in event_frame.columns:
        event_frame["activation_context"] = event_frame.get(
            "signal_cell",
            event_frame.get("entry_regime", pd.Series("all", index=event_frame.index)),
        ).astype(str)
    if "uniqueness_weight" not in event_frame.columns:
        oos_set = getattr(fold_out, "oos_set", None)
        weights = getattr(oos_set, "edge_weight", None) if oos_set is not None else None
        if weights is not None and len(weights) >= size:
            event_frame["uniqueness_weight"] = np.asarray(weights[:size], dtype=np.float64)
        else:
            event_frame["uniqueness_weight"] = np.ones(size, dtype=np.float64)
    if "gross_event_bps" not in event_frame.columns:
        oos_set = getattr(fold_out, "oos_set", None)
        y_return = getattr(oos_set, "y_return_bps", None) if oos_set is not None else None
        if y_return is not None and len(y_return) >= size:
            event_frame["gross_event_bps"] = np.asarray(y_return[:size], dtype=np.float64)
        else:
            event_frame["gross_event_bps"] = np.zeros(size, dtype=np.float64)
    event_frame["realized_side_adjusted_gross_bps"] = pd.to_numeric(
        event_frame["gross_event_bps"],
        errors="coerce",
    ).fillna(0.0)
    return event_frame


def compute_symbol_strategy_evidence(
    *,
    event_results: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    seed: int,
) -> tuple[SymbolStrategyEvidence, ...]:
    """Compute per-source signal evidence from event-level OOS results."""
    del seed
    if event_results.empty:
        return ()
    frame = event_results.copy()
    if "strategy_id" not in frame.columns:
        frame["strategy_id"] = (
            frame.get("family", pd.Series("", index=frame.index)).astype(str)
            + ":"
            + frame.get("variant", pd.Series("", index=frame.index)).astype(str)
        )
    if "activation_context" not in frame.columns:
        frame["activation_context"] = frame.get(
            "signal_cell",
            frame.get("entry_regime", pd.Series("all", index=frame.index)),
        ).astype(str)
    if "uniqueness_weight" not in frame.columns:
        frame["uniqueness_weight"] = 1.0
    frame["gross_event_bps"] = pd.to_numeric(
        frame.get("gross_event_bps", frame.get("realized_side_adjusted_gross_bps", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    frame["side"] = (
        pd.to_numeric(frame.get("side", pd.Series(1, index=frame.index)), errors="coerce")
        .fillna(1.0)
        .astype(int)
    )
    frame["expected_holding_bars"] = (
        pd.to_numeric(
            frame.get("expected_holding_bars", pd.Series(1, index=frame.index)),
            errors="coerce",
        )
        .fillna(1)
        .clip(lower=1)
        .astype(int)
    )
    frame["fold_id"] = (
        pd.to_numeric(frame.get("fold_id", pd.Series(0, index=frame.index)), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    frame["holding_bucket"] = frame["expected_holding_bars"].map(_holding_bucket)
    if "baseline_gross_bps" not in frame.columns:
        baseline_map = (
            frame.groupby(["symbol", "side", "holding_bucket"], sort=False)["gross_event_bps"]
            .mean()
            .to_dict()
        )
        frame["baseline_gross_bps"] = [
            baseline_map.get((str(symbol), int(side), int(bucket)), 0.0)
            for symbol, side, bucket in zip(
                frame["symbol"],
                frame["side"],
                frame["holding_bucket"],
                strict=True,
            )
        ]
    frame["incremental_bps"] = frame["gross_event_bps"] - pd.to_numeric(
        frame["baseline_gross_bps"],
        errors="coerce",
    ).fillna(0.0)
    grouped = frame.groupby(["symbol", "strategy_id", "activation_context"], sort=True)
    evidence_list: list[SymbolStrategyEvidence] = []
    raw_p_values: list[float] = []
    for (symbol, strategy_id, activation_context), group in grouped:
        weights = pd.to_numeric(group["uniqueness_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        gross = group["gross_event_bps"].to_numpy(dtype=np.float64, copy=False)
        incremental = group["incremental_bps"].to_numpy(dtype=np.float64, copy=False)
        n_obs = int(group.shape[0])
        weight_sum = float(np.sum(weights))
        effective_n = 0.0
        if weight_sum > 0.0:
            denom = float(np.sum(np.square(weights)))
            if denom > 0.0:
                effective_n = (weight_sum * weight_sum) / denom
        mean_gross = float(np.average(gross, weights=weights)) if weight_sum > 0.0 else 0.0
        mean_incremental = float(np.average(incremental, weights=weights)) if weight_sum > 0.0 else 0.0
        fold_means = [
            float(group_fold["incremental_bps"].mean())
            for _, group_fold in group.groupby("fold_id", sort=True)
            if not group_fold.empty
        ]
        positive_fold_ratio = (
            float(sum(1 for value in fold_means if value > 0.0) / len(fold_means))
            if fold_means
            else 0.0
        )
        t_stat = _series_tstat(incremental)
        p_value = _one_sided_p_value(t_stat)
        reliability = float(
            np.clip(
                max(mean_incremental, 0.0)
                * max(t_stat, 0.0)
                / max(float(getattr(cfg, "l1_pair_min_incremental_tstat", 1.0)), 1.0)
                / max(abs(float(getattr(cfg, "l1_pair_min_mean_gross_bps", 1.0))) + 1.0, 1.0),
                0.0,
                1.0,
            )
        )
        rejection_reasons: list[str] = []
        if effective_n < float(cfg.l1_pair_min_effective_obs):
            rejection_reasons.append("insufficient_effective_obs")
        if len(fold_means) < int(cfg.l1_pair_min_folds):
            rejection_reasons.append("insufficient_folds")
        if mean_gross <= float(cfg.l1_pair_min_mean_gross_bps):
            rejection_reasons.append("negative_gross_edge")
        if mean_incremental <= float(cfg.l1_pair_min_incremental_bps):
            rejection_reasons.append("no_incremental_edge")
        if t_stat < float(cfg.l1_pair_min_incremental_tstat):
            rejection_reasons.append("weak_tstat")
        if positive_fold_ratio < float(cfg.l1_pair_min_positive_fold_ratio):
            rejection_reasons.append("unstable_folds")
        evidence_list.append(
            SymbolStrategyEvidence(
                key=SignalSourceKey(
                    symbol=str(symbol),
                    strategy_id=str(strategy_id),
                    activation_context=str(activation_context or "all"),
                ),
                mean_gross_bps=mean_gross,
                mean_incremental_bps=mean_incremental,
                bootstrap_tstat_incremental=t_stat,
                p_value=p_value,
                q_value=1.0,
                positive_fold_ratio=positive_fold_ratio,
                n_obs=n_obs,
                effective_n=effective_n,
                n_folds=len(fold_means),
                reliability=reliability,
                qualified=False,
                rejection_reasons=tuple(rejection_reasons),
            )
        )
        raw_p_values.append(p_value)
    q_values = _by_q_values(np.asarray(raw_p_values, dtype=np.float64))
    final_evidence: list[SymbolStrategyEvidence] = []
    for idx, evidence in enumerate(evidence_list):
        reasons = list(evidence.rejection_reasons)
        q_value = float(q_values[idx])
        if q_value > float(cfg.l1_pair_fdr_alpha):
            reasons.append("fdr_reject")
        final_evidence.append(
            SymbolStrategyEvidence(
                key=evidence.key,
                mean_gross_bps=evidence.mean_gross_bps,
                mean_incremental_bps=evidence.mean_incremental_bps,
                bootstrap_tstat_incremental=evidence.bootstrap_tstat_incremental,
                p_value=evidence.p_value,
                q_value=q_value,
                positive_fold_ratio=evidence.positive_fold_ratio,
                n_obs=evidence.n_obs,
                effective_n=evidence.effective_n,
                n_folds=evidence.n_folds,
                reliability=evidence.reliability,
                qualified=not reasons,
                rejection_reasons=tuple(reasons),
            )
        )
    return tuple(final_evidence)


def build_qualified_signal_registry(
    *,
    evidence: tuple[SymbolStrategyEvidence, ...],
    symbols: tuple[str, ...],
    min_signals_per_symbol: int,
    registry_version: str,
) -> QualifiedSignalRegistry:
    grouped: dict[str, list[SymbolStrategyEvidence]] = defaultdict(list)
    for item in evidence:
        if item.qualified:
            grouped[item.key.symbol].append(item)
    by_symbol: dict[str, tuple[SymbolStrategyEvidence, ...]] = {}
    ready_symbols: list[str] = []
    for symbol in symbols:
        items = tuple(
            sorted(
                grouped.get(symbol, ()),
                key=lambda candidate: (
                    candidate.reliability,
                    candidate.mean_incremental_bps,
                    candidate.bootstrap_tstat_incremental,
                ),
                reverse=True,
            )
        )
        if len(items) >= min_signals_per_symbol:
            by_symbol[symbol] = items
            ready_symbols.append(symbol)
    return QualifiedSignalRegistry(
        by_symbol=by_symbol,
        ready_symbols=tuple(ready_symbols),
        trade_scope_count=len(symbols),
        registry_version=registry_version,
    )


def _registry_to_symbol_signals(
    registry: QualifiedSignalRegistry,
) -> dict[str, SymbolSignal]:
    """Compatibility adapter until Layer2 consumes ValidatedSignalBatch directly."""
    adapted: dict[str, SymbolSignal] = {}
    for symbol, evidence_items in registry.by_symbol.items():
        if not evidence_items:
            continue
        best = evidence_items[0]
        adapted[symbol] = SymbolSignal(
            raw_mu=float(best.mean_gross_bps),
            volatility=VOL_FLOOR,
            n_obs=max(round(best.effective_n), 0),
            t_stat=float(best.bootstrap_tstat_incremental),
            valid=True,
            beta_btc=None,
        )
    return adapted


def _candidate_output_to_signal_batch(
    *,
    model_output: CandidateModelOutput,
    registry: QualifiedSignalRegistry,
    datetimes: NDArray[np.datetime64],
    symbols: tuple[str, ...],
    model_version: str,
    activation_floor_bps: float,
) -> ValidatedSignalBatch:
    frame = model_output.events.reset_index(drop=True).copy()
    if frame.empty:
        return ValidatedSignalBatch(
            events=(),
            start_idx=0,
            end_idx=0,
            symbols=symbols,
            registry_version=registry.registry_version,
            model_version=model_version,
        )
    gross = _expected_gross_bps(model_output)
    q10 = _q10_gross_bps(model_output)
    q90 = _q90_gross_bps(model_output)
    source_keys = {
        (item.key.symbol, item.key.strategy_id, item.key.activation_context)
        for items in registry.by_symbol.values()
        for item in items
    }
    events: list[ValidatedSignalEvent] = []
    start_idx = int(frame["entry_idx"].min()) if "entry_idx" in frame.columns and not frame.empty else 0
    end_idx = int(frame["entry_idx"].max()) + 1 if "entry_idx" in frame.columns and not frame.empty else 0
    for idx, row in frame.iterrows():
        key = _signal_source_key_from_row(row)
        if (key.symbol, key.strategy_id, key.activation_context) not in source_keys:
            continue
        pred = float(gross[idx]) if idx < gross.size else 0.0
        if pred <= activation_floor_bps:
            continue
        entry_idx = int(pd.to_numeric(row.get("entry_idx", 0), errors="coerce"))
        decision_idx = entry_idx - 1
        if decision_idx < 0 or decision_idx >= datetimes.shape[0]:
            continue
        side_val = int(pd.to_numeric(row.get("side", 1), errors="coerce"))
        side: int = 1 if side_val >= 0 else -1
        holding = max(int(pd.to_numeric(row.get("expected_holding_bars", 1), errors="coerce")), 1)
        reliability = 0.0
        for evidence in registry.by_symbol.get(key.symbol, ()):
            if evidence.key == key:
                reliability = evidence.reliability
                break
        events.append(
            ValidatedSignalEvent(
                decision_idx=decision_idx,
                decision_time=datetimes[decision_idx],
                symbol=key.symbol,
                strategy_id=key.strategy_id,
                activation_context=key.activation_context,
                side=1 if side >= 0 else -1,
                expected_gross_bps=pred,
                q10_gross_bps=float(q10[idx]) if idx < q10.size else pred,
                q90_gross_bps=float(q90[idx]) if idx < q90.size else pred,
                expected_holding_bars=holding,
                reliability=reliability,
                registry_version=registry.registry_version,
                model_version=model_version,
            )
        )
    events.sort(key=lambda item: (item.decision_idx, item.symbol, item.strategy_id, item.activation_context))
    return ValidatedSignalBatch(
        events=tuple(events),
        start_idx=start_idx,
        end_idx=end_idx,
        symbols=symbols,
        registry_version=registry.registry_version,
        model_version=model_version,
    )


def select_outer_symbol_opportunities(
    *,
    predictions: ValidatedSignalBatch,
    registry: QualifiedSignalRegistry,
) -> ValidatedSignalBatch:
    del registry
    best_by_slot: dict[tuple[int, str], ValidatedSignalEvent] = {}
    for event in predictions.events:
        slot = (event.decision_idx, event.symbol)
        candidate = best_by_slot.get(slot)
        if candidate is None:
            best_by_slot[slot] = event
            continue
        current_score = (
            event.expected_gross_bps
            / max(event.expected_holding_bars, 1)
            * max(event.reliability, 0.0)
        )
        best_score = (
            candidate.expected_gross_bps
            / max(candidate.expected_holding_bars, 1)
            * max(candidate.reliability, 0.0)
        )
        if current_score > best_score or (
            np.isclose(current_score, best_score)
            and (event.strategy_id, event.activation_context) < (candidate.strategy_id, candidate.activation_context)
        ):
            best_by_slot[slot] = event
    selected = tuple(
        sorted(
            best_by_slot.values(),
            key=lambda item: (item.decision_idx, item.symbol, item.strategy_id, item.activation_context),
        )
    )
    return ValidatedSignalBatch(
        events=selected,
        start_idx=predictions.start_idx,
        end_idx=predictions.end_idx,
        symbols=tuple(dict.fromkeys(event.symbol for event in selected)),
        registry_version=predictions.registry_version,
        model_version=predictions.model_version,
    )


def evaluate_outer_signal_opportunities(
    *,
    opportunities: ValidatedSignalBatch,
    realized_event_results: pd.DataFrame,
    volatility_2d: NDArray[np.float64],
    fold: WFFold,
    cfg: CandidateStrategyConfig,
    seed: int,
) -> Layer1FoldReadiness:
    del seed
    opp_frame = _batch_to_frame(opportunities)
    if opp_frame.empty:
        return Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=fold.fit_end,
            outer_oos_start_idx=fold.oos_start,
            outer_oos_end_idx=fold.oos_end,
            ready_symbols=(),
            valid_opportunity_timestamp_count=0,
            opportunity_ic=0.0,
            opportunity_ic_series=(),
            probe_gross_edge_bps=0.0,
            probe_gross_edge_series_bps=(),
            passed=False,
            blockers=("empty_opportunities",),
        )
    realized = realized_event_results.copy()
    if "strategy_id" not in realized.columns:
        realized["strategy_id"] = (
            realized.get("family", pd.Series("", index=realized.index)).astype(str)
            + ":"
            + realized.get("variant", pd.Series("", index=realized.index)).astype(str)
        )
    if "activation_context" not in realized.columns:
        realized["activation_context"] = realized.get(
            "signal_cell",
            realized.get("entry_regime", pd.Series("all", index=realized.index)),
        ).astype(str)
    realized["decision_idx"] = (
        pd.to_numeric(realized.get("entry_idx", pd.Series(0, index=realized.index)), errors="coerce")
        .fillna(0)
        .astype(int)
        - 1
    )
    if "realized_side_adjusted_gross_bps" not in realized.columns:
        realized["realized_side_adjusted_gross_bps"] = pd.to_numeric(
            realized.get("gross_event_bps", pd.Series(0.0, index=realized.index)),
            errors="coerce",
        ).fillna(0.0)
    merge_cols = [
        "decision_idx",
        "symbol",
        "strategy_id",
        "activation_context",
        "realized_side_adjusted_gross_bps",
    ]
    if "exit_idx" in realized.columns:
        merge_cols.append("exit_idx")
    merged = opp_frame.merge(
        realized[merge_cols],
        on=["decision_idx", "symbol", "strategy_id", "activation_context"],
        how="left",
    )
    if "exit_idx" in merged.columns:
        merged = merged.loc[
            pd.to_numeric(merged["exit_idx"], errors="coerce").fillna(fold.oos_end - 1).astype(int) < fold.oos_end
        ].copy()
    if merged.empty:
        return Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=fold.fit_end,
            outer_oos_start_idx=fold.oos_start,
            outer_oos_end_idx=fold.oos_end,
            ready_symbols=(),
            valid_opportunity_timestamp_count=0,
            opportunity_ic=0.0,
            opportunity_ic_series=(),
            probe_gross_edge_bps=0.0,
            probe_gross_edge_series_bps=(),
            passed=False,
            blockers=("empty_realized_merge",),
        )
    symbol_to_idx = {
        symbol: idx for idx, symbol in enumerate(opportunities.symbols)
    }
    ic_series: list[float] = []
    probe_series: list[float] = []
    for decision_idx, group in merged.groupby("decision_idx", sort=True):
        group = group.drop_duplicates(subset=["symbol"], keep="first")
        if group.shape[0] < int(cfg.l1_min_cross_section):
            continue
        pred = (
            group["side"].to_numpy(dtype=np.float64, copy=False)
            * group["expected_gross_bps"].to_numpy(dtype=np.float64, copy=False)
            / np.maximum(group["expected_holding_bars"].to_numpy(dtype=np.float64, copy=False), 1.0)
        )
        real = (
            group["side"].to_numpy(dtype=np.float64, copy=False)
            * group["realized_side_adjusted_gross_bps"].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            / np.maximum(group["expected_holding_bars"].to_numpy(dtype=np.float64, copy=False), 1.0)
        )
        ic_val, _ = spearmanr(pred, real)
        if np.isfinite(ic_val):
            ic_series.append(float(ic_val))
        risk_scores: list[tuple[float, int]] = []
        for row_idx, row in enumerate(group.itertuples(index=False)):
            symbol_idx = symbol_to_idx.get(str(row.symbol))
            if symbol_idx is None or decision_idx < 0 or decision_idx >= volatility_2d.shape[0]:
                continue
            vol = float(volatility_2d[int(decision_idx), symbol_idx])
            denom = max(vol, VOL_FLOOR)
            risk_scores.append((abs(float(row.expected_gross_bps)) / denom, row_idx))
        if risk_scores:
            risk_scores.sort(reverse=True)
            selected_idx = [row_idx for _, row_idx in risk_scores[: int(cfg.l1_probe_top_k)]]
            selected_real = real[np.asarray(selected_idx, dtype=np.int64)]
            if selected_real.size > 0:
                probe_series.append(float(np.mean(selected_real)))
    ready_symbols = tuple(sorted(str(symbol) for symbol in merged["symbol"].dropna().unique()))
    opportunity_ic = float(np.mean(ic_series)) if ic_series else 0.0
    probe_gross_edge = float(np.mean(probe_series)) if probe_series else 0.0
    blockers: list[str] = []
    fold_min_ready_symbols = max(1, min(int(cfg.l1_min_ready_symbols), int(cfg.l1_min_cross_section)))
    if len(ready_symbols) < fold_min_ready_symbols:
        blockers.append("insufficient_ready_symbols")
    if len(ic_series) < int(cfg.l1_min_opportunity_timestamps):
        blockers.append("insufficient_opportunity_timestamps")
    if not np.isfinite(opportunity_ic):
        blockers.append("non_finite_ic")
    if probe_gross_edge <= 0.0:
        blockers.append("non_positive_probe")
    return Layer1FoldReadiness(
        fold_id=fold.oos_start,
        registry_source_end_idx=fold.fit_end,
        outer_oos_start_idx=fold.oos_start,
        outer_oos_end_idx=fold.oos_end,
        ready_symbols=ready_symbols,
        valid_opportunity_timestamp_count=len(ic_series),
        opportunity_ic=opportunity_ic,
        opportunity_ic_series=tuple(ic_series),
        probe_gross_edge_bps=probe_gross_edge,
        probe_gross_edge_series_bps=tuple(probe_series),
        passed=not blockers,
        blockers=tuple(blockers),
    )


def evaluate_layer1_readiness(
    *,
    fold_reports: tuple[Layer1FoldReadiness, ...],
    trained_outer_fold_coverage: float,
    trade_scope_count: int,
    cfg: CandidateStrategyConfig,
) -> Layer1GateReport:
    symbol_counter: Counter[str] = Counter()
    ic_series: list[float] = []
    probe_series: list[float] = []
    ready_fold_count = 0
    for report in fold_reports:
        if report.passed:
            ready_fold_count += 1
        symbol_counter.update(report.ready_symbols)
        ic_series.extend([value for value in report.opportunity_ic_series if np.isfinite(value)])
        probe_series.extend([value for value in report.probe_gross_edge_series_bps if np.isfinite(value)])
    stable_ready_symbols = [
        symbol
        for symbol, count in symbol_counter.items()
        if count >= int(cfg.l1_min_ready_outer_folds)
    ]
    ready_outer_fold_ratio = float(ready_fold_count / len(fold_reports)) if fold_reports else 0.0
    opportunity_ic_mean = float(np.mean(ic_series)) if ic_series else 0.0
    opportunity_ic_tstat = _series_tstat(np.asarray(ic_series, dtype=np.float64))
    probe_gross_edge_bps = float(np.mean(probe_series)) if probe_series else 0.0
    probe_gross_edge_tstat = _series_tstat(np.asarray(probe_series, dtype=np.float64))
    stable_ready_symbol_ratio = float(len(stable_ready_symbols) / max(1, trade_scope_count))
    check_specs = (
        (
            "trained_outer_fold_coverage",
            trained_outer_fold_coverage,
            float(getattr(cfg, "l1_min_trained_outer_fold_coverage", 0.8)),
            "ge",
        ),
        ("stable_ready_symbol_count", float(len(stable_ready_symbols)), float(cfg.l1_min_ready_symbols), "ge"),
        ("stable_ready_symbol_ratio", stable_ready_symbol_ratio, float(cfg.l1_min_ready_symbol_ratio), "ge"),
        ("ready_outer_fold_ratio", ready_outer_fold_ratio, float(cfg.l1_min_ready_outer_fold_ratio), "ge"),
        ("opportunity_ic_mean", opportunity_ic_mean, float(cfg.l1_min_opportunity_ic), "ge"),
        ("opportunity_ic_tstat", opportunity_ic_tstat, float(cfg.l1_min_opportunity_ic_tstat), "ge"),
        ("probe_gross_edge_bps", probe_gross_edge_bps, float(cfg.l1_min_probe_gross_edge_bps), "gt"),
        ("probe_gross_edge_tstat", probe_gross_edge_tstat, float(cfg.l1_min_probe_gross_edge_tstat), "ge"),
    )
    checks: list[Layer1GateCheck] = []
    blockers: list[str] = []
    for key, value, threshold, comparator in check_specs:
        finite_value = np.isfinite(value)
        passed = finite_value and (value >= threshold if comparator == "ge" else value > threshold)
        blocker = None if passed else f"{value:.3f}"
        comparator_literal = cast(Literal["ge", "gt"], comparator)
        if blocker is not None:
            blockers.append(f"{key}:{blocker}")
        checks.append(
            Layer1GateCheck(
                key=key,
                value=float(value),
                threshold=float(threshold),
                comparator=comparator_literal,
                passed=passed,
                blocker=blocker,
            )
        )
    return Layer1GateReport(
        checks=tuple(checks),
        passed=all(check.passed for check in checks),
        blockers=tuple(blockers),
    )


def fit_layer1_inference_artifact(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    deployment_registry: QualifiedSignalRegistry,
    fit_start_idx: int,
    fit_end_idx: int,
    cfg: CandidateStrategyConfig,
    seed: int,
) -> Layer1InferenceArtifact:
    del seed
    schema = fit_candidate_feature_schema(
        labeled_events=labeled_events,
        cfg=cfg,
        split_start=fit_start_idx,
        split_end=fit_end_idx,
    )
    fit_set = build_candidate_dataset(
        labeled_events=labeled_events,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=fit_start_idx,
        split_end=fit_end_idx,
        is_fit_split=True,
    )
    train_events = fit_set.event_index.copy()
    gross_targets = getattr(fit_set, "y_gross_return_bps", None)
    if gross_targets is None:
        gross_targets = fit_set.y_return_bps
    train_events["gross_return_bps"] = np.asarray(gross_targets, dtype=np.float64)
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)
    baseline_by_key: dict[MatchedBaselineKey, float] = {}
    baseline_frame = fit_set.event_index.copy()
    if "gross_event_bps" not in baseline_frame.columns and gross_targets is not None:
        baseline_frame["gross_event_bps"] = np.asarray(gross_targets, dtype=np.float64)
    if "side" in baseline_frame.columns and "expected_holding_bars" in baseline_frame.columns:
        baseline_frame["holding_bucket"] = pd.to_numeric(
            baseline_frame["expected_holding_bars"],
            errors="coerce",
        ).fillna(1).astype(int).map(_holding_bucket)
        grouped = baseline_frame.groupby(["symbol", "side", "holding_bucket"], sort=False)
        for (symbol, side, holding_bucket), group in grouped:
            side_literal: Literal[-1, 1] = 1 if int(side) >= 0 else -1
            baseline_by_key[MatchedBaselineKey(str(symbol), side_literal, int(holding_bucket))] = float(
                pd.to_numeric(group["gross_event_bps"], errors="coerce").fillna(0.0).mean()
            )
    config_hash = sha256(str(cfg).encode("utf-8")).hexdigest()[:12]
    return Layer1InferenceArtifact(
        feature_schema=schema,
        model=model,
        deployment_registry=deployment_registry,
        baseline_by_key=baseline_by_key,
        l1_fit_end_idx=fit_end_idx,
        model_version=schema.version,
        config_hash=config_hash,
    )


def predict_layer1_signals(
    *,
    artifact: Layer1InferenceArtifact,
    candidate_events: pd.DataFrame,
    aligned: AlignedMarketData,
    start_idx: int,
    end_idx: int,
    cfg: CandidateStrategyConfig,
) -> ValidatedSignalBatch:
    inference_set = build_candidate_dataset(
        labeled_events=candidate_events,
        aligned=aligned,
        cfg=cfg,
        schema=artifact.feature_schema,
        split_start=start_idx,
        split_end=end_idx,
        require_label_within_split=False,
    )
    prediction = predict_regime_conditional_ensemble(
        model=artifact.model,
        oos_events=inference_set.event_index,
        cfg=cfg,
    )
    return _candidate_output_to_signal_batch(
        model_output=prediction,
        registry=artifact.deployment_registry,
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        model_version=artifact.model_version,
        activation_floor_bps=float(cfg.l1_signal_activation_floor_bps),
    )


def run_l1_nested_swf(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    outer_folds: tuple[WFFold, ...],
    cfg: CandidateStrategyConfig,
    seed: int,
) -> Layer1Result:
    """Run nested Layer1 validation using inner selection and outer evaluation."""
    from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars

    purge_bars, embargo_bars = resolve_purge_and_embargo_bars(cfg)
    volatility_2d = rolling_per_bar_return_std(
        aligned.close_2d,
        composer_sigma_lookback_bars("4h"),
    )
    outer_reports: list[Layer1FoldReadiness] = []
    outer_event_frames: list[pd.DataFrame] = []
    trained_count = 0
    for outer_idx, outer_fold in enumerate(outer_folds):
        inner_train_end = outer_fold.oos_start
        try:
            inner_folds = build_l1_swf_folds(
                n_bars=inner_train_end,
                n_folds=max(1, min(int(getattr(cfg, "wf_n_folds", 1)), 3)),
                l1_start_bars=outer_fold.fit_start,
                l1_end_bars=inner_train_end,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
            )
        except ValueError:
            inner_folds = ()
        inner_frames: list[pd.DataFrame] = []
        for inner_idx, inner_fold in enumerate(inner_folds):
            inner_out = _fit_and_predict_single_fold(
                inner_idx,
                inner_fold,
                labeled_events,
                aligned,
                cfg,
                purge_bars,
            )
            if _is_trained_fold_output(inner_out):
                inner_frames.append(
                    _event_results_from_fold_output(
                        fold_id=inner_idx,
                        fold_out=inner_out,
                    )
                )
        inner_event_results = (
            pd.concat(inner_frames, ignore_index=True)
            if inner_frames
            else pd.DataFrame()
        )
        evidence = compute_symbol_strategy_evidence(
            event_results=inner_event_results,
            cfg=cfg,
            seed=seed + outer_idx,
        )
        registry = build_qualified_signal_registry(
            evidence=evidence,
            symbols=aligned.symbols,
            min_signals_per_symbol=int(cfg.l1_min_signals_per_symbol),
            registry_version=f"outer-{outer_idx}",
        )
        outer_out = _fit_and_predict_single_fold(
            outer_idx,
            outer_fold,
            labeled_events,
            aligned,
            cfg,
            purge_bars,
        )
        if _is_trained_fold_output(outer_out):
            trained_count += 1
        outer_events = _event_results_from_fold_output(
            fold_id=outer_idx,
            fold_out=outer_out,
        )
        outer_event_frames.append(outer_events)
        prediction_batch = _candidate_output_to_signal_batch(
            model_output=outer_out.model_output,
            registry=registry,
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            model_version=f"outer-{outer_idx}",
            activation_floor_bps=float(cfg.l1_signal_activation_floor_bps),
        )
        opportunities = select_outer_symbol_opportunities(
            predictions=prediction_batch,
            registry=registry,
        )
        outer_reports.append(
            evaluate_outer_signal_opportunities(
                opportunities=opportunities,
                realized_event_results=outer_events,
                volatility_2d=volatility_2d,
                fold=outer_fold,
                cfg=cfg,
                seed=seed + outer_idx,
            )
        )
    trained_outer_fold_coverage = (
        float(trained_count / len(outer_folds))
        if outer_folds
        else 0.0
    )
    deployment_event_results = (
        pd.concat(outer_event_frames, ignore_index=True)
        if outer_event_frames
        else pd.DataFrame()
    )
    deployment_evidence = compute_symbol_strategy_evidence(
        event_results=deployment_event_results,
        cfg=cfg,
        seed=seed,
    )
    gate_report = evaluate_layer1_readiness(
        fold_reports=tuple(outer_reports),
        trained_outer_fold_coverage=trained_outer_fold_coverage,
        trade_scope_count=len(aligned.symbols),
        cfg=cfg,
    )
    logger.info(format_layer1_gate_table(gate_report))
    logger.info(format_layer1_outer_fold_table(tuple(outer_reports)))
    deployment_registry: QualifiedSignalRegistry | None = None
    inference_artifact: Layer1InferenceArtifact | None = None
    oos_stacked: dict[str, SymbolSignal] = {}
    if gate_report.passed:
        deployment_registry = build_qualified_signal_registry(
            evidence=deployment_evidence,
            symbols=aligned.symbols,
            min_signals_per_symbol=int(cfg.l1_min_signals_per_symbol),
            registry_version="deployment",
        )
        oos_stacked = _registry_to_symbol_signals(deployment_registry)
        fit_start_idx = min((fold.fit_start for fold in outer_folds), default=0)
        fit_end_idx = max((fold.oos_end for fold in outer_folds), default=0)
        inference_artifact = fit_layer1_inference_artifact(
            labeled_events=labeled_events,
            aligned=aligned,
            deployment_registry=deployment_registry,
            fit_start_idx=fit_start_idx,
            fit_end_idx=fit_end_idx,
            cfg=cfg,
            seed=seed,
        )
        logger.info(format_layer1_deployment_registry_table(deployment_registry))
    return Layer1Result(
        signals_per_fold=(),
        oos_stacked=oos_stacked,
        pooled_ic=0.0,
        pooled_tstat=0.0,
        breadth=0.0,
        valid_coverage=0.0,
        fold_pass_ratio=0.0,
        gate_passed=gate_report.passed,
        n_valid=len(deployment_registry.ready_symbols) if deployment_registry is not None else 0,
        n_total=len(aligned.symbols),
        n_trade_scope=len(aligned.symbols),
        outer_fold_reports=tuple(outer_reports),
        deployment_evidence=deployment_evidence,
        gate_report=gate_report,
        deployment_registry=deployment_registry,
        inference_artifact=inference_artifact,
    )


# ---------------------------------------------------------------------------
# Layer2: AWF Portfolio Validation
# ---------------------------------------------------------------------------


def run_l2_awf(
    *,
    l1_oos: dict[str, SymbolSignal],
    aligned: AlignedMarketData,
    awf_folds: tuple[WFFold, ...],
    l2_params: dict[str, Any],
    caps: PortfolioCaps,
    tf: str = "4h",
) -> Layer2Result:
    """Layer2 AWF 포트폴리오 시뮬레이션.

    L1 OOS 신호를 AWF fold에서 Diagonal Kelly로 리밸런스하며 PnL 집계.

    Args:
        l1_oos: L1 합본 symbol→SymbolSignal.
        aligned: AlignedMarketData.
        awf_folds: WFFold 튜플 (build_walk_forward_folds 출력).
        l2_params: Layer2 하이퍼파라미터 dict.
        caps: PortfolioCaps.
        tf: 타임프레임 문자열.

    Returns:
        Layer2Result.

    Time Complexity: O(F * T * N)
    Space Complexity: O(T) — 수익률 리스트 누적
    """
    sim = _run_awf_simulation(
        l1_oos=l1_oos,
        aligned=aligned,
        awf_folds=awf_folds,
        l2_params=l2_params,
        caps=caps,
        tf=tf,
    )
    symbols = aligned.symbols
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    sharpe_hybrid = _sharpe(sim.rets_hybrid)
    sharpe_baseline = _sharpe(sim.rets_baseline)
    mdd_hybrid = _mdd(sim.rets_hybrid)
    mdd_baseline = _mdd(sim.rets_baseline)
    avg_turnover = float(np.mean(sim.all_turnovers)) if sim.all_turnovers else 0.0
    friction_pass_pct = (
        sim.friction_pass_total / sim.signal_total if sim.signal_total > 0 else 0.0
    )

    gate_passed: bool = bool(
        (sharpe_hybrid >= sharpe_baseline * 1.20) and (mdd_hybrid <= mdd_baseline)
    )

    result = Layer2Result(
        selected_last=sim.last_selected,
        weights_last={
            s: float(sim.last_w[sym_to_idx[s]])
            for s in sim.last_selected
            if s in sym_to_idx
        },
        sharpe_hybrid=sharpe_hybrid,
        sharpe_baseline=sharpe_baseline,
        mdd_hybrid=mdd_hybrid,
        mdd_baseline=mdd_baseline,
        turnover=avg_turnover,
        friction_pass_pct=friction_pass_pct,
        gate_passed=gate_passed,
    )
    logger.info(format_layer2_table(result))
    return result


# ---------------------------------------------------------------------------
# Layer3: Holdout Final Validation
# ---------------------------------------------------------------------------


def run_l3_holdout(
    *,
    l1_oos: dict[str, SymbolSignal],
    aligned: AlignedMarketData,
    holdout_span: tuple[int, int],
    frozen_params: dict[str, Any],
    caps: PortfolioCaps,
    tf: str = "4h",
) -> Layer3Result:
    """Layer3 Holdout 최종 검증.

    단일 holdout 구간에서 L2 로직을 재실행하여 CAGR/MDD/Sharpe/MAR 계산.

    Args:
        l1_oos: L1 합본 symbol→SymbolSignal.
        aligned: AlignedMarketData.
        holdout_span: (oos_start_idx, oos_end_idx) bar 인덱스 튜플.
        frozen_params: L2 파라미터 (run_l2_awf l2_params 형식).
        caps: PortfolioCaps.
        tf: 타임프레임 문자열.

    Returns:
        Layer3Result.

    Time Complexity: O(T * N) — holdout 구간 단일 패스
    Space Complexity: O(T)
    """
    ho_start, ho_end = holdout_span
    dummy_fold = WFFold(
        fit_start=0,
        fit_end=ho_start,
        cal_start=max(0, ho_start // 2),
        cal_end=ho_start,
        oos_start=ho_start,
        oos_end=ho_end,
    )

    sim = _run_awf_simulation(
        l1_oos=l1_oos,
        aligned=aligned,
        awf_folds=(dummy_fold,),
        l2_params=frozen_params,
        caps=caps,
        tf=tf,
    )

    sharpe = _sharpe(sim.rets_hybrid)
    sharpe_baseline = _sharpe(sim.rets_baseline)
    mdd = _mdd(sim.rets_hybrid)
    mdd_baseline = _mdd(sim.rets_baseline)
    cagr = _cagr(sim.rets_hybrid)
    cagr_baseline = _cagr(sim.rets_baseline)
    mar = cagr / (mdd + 1e-9)
    mar_baseline = cagr_baseline / (mdd_baseline + 1e-9)

    gate_passed: bool = bool(
        (sharpe >= sharpe_baseline) and (mdd <= mdd_baseline)
    )

    result = Layer3Result(
        cagr=cagr,
        mdd=mdd,
        sharpe=sharpe,
        mar=mar,
        cagr_baseline=cagr_baseline,
        mdd_baseline=mdd_baseline,
        sharpe_baseline=sharpe_baseline,
        mar_baseline=mar_baseline,
        gate_passed=gate_passed,
    )
    logger.info(format_layer3_table(result, ho_start=str(ho_start), ho_end=str(ho_end)))
    return result


# ---------------------------------------------------------------------------
# 최상위 오케스트레이터
# ---------------------------------------------------------------------------


def run_tiered_pipeline(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    window: LayeredWindow,
    l1_params: dict[str, Any],
    l2_params: dict[str, Any],
    caps: PortfolioCaps | None = None,
    tf: str = "4h",
) -> tuple[Layer1Result, Layer2Result | None, Layer3Result | None]:
    """3-Layer 티어드 파이프라인 실행.

    L1 CPCV → L2 AWF → L3 Holdout 순서로 게이트 기반 단계적 검증.
    각 게이트 실패 시 즉시 (result, None, None) 반환.

    Args:
        labeled_events: 레이블링된 이벤트 DataFrame.
        aligned: AlignedMarketData.
        cfg: CandidateStrategyConfig.
        window: LayeredWindow (L1/L2/holdout 날짜 범위).
        l1_params: Layer1 하이퍼파라미터 dict.
        l2_params: Layer2 하이퍼파라미터 dict.
        caps: PortfolioCaps (None이면 기본값 사용).
        tf: 타임프레임 문자열.

    Returns:
        (Layer1Result, Layer2Result | None, Layer3Result | None) 튜플.

    Time Complexity: O(F * T * N)
    Space Complexity: O(F * N)
    """
    from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars

    if caps is None:
        caps = PortfolioCaps(
            gross=3.0,
            per_symbol=0.15,
            net=0.5,
            beta=1.0,
            target_ann_vol=0.20,
        )

    purge_bars, embargo_bars = resolve_purge_and_embargo_bars(cfg)
    n_bars = len(aligned.datetimes)

    # Layer1: nested SWF readiness
    _is_ts = pd.Timestamp(window.l1_start, tz="UTC")
    _oos_ts = pd.Timestamp(window.l2_start, tz="UTC")
    l1_start_bars = int(np.searchsorted(aligned.datetimes, np.datetime64(_is_ts.replace(tzinfo=None), "ns")))
    l1_end_bars = int(np.searchsorted(aligned.datetimes, np.datetime64(_oos_ts.replace(tzinfo=None), "ns")))

    outer_folds = build_l1_nested_swf_folds(
        n_bars=n_bars,
        l1_start_idx=l1_start_bars,
        l1_end_idx=l1_end_bars,
        max_label_horizon_bars=max(int(getattr(cfg, "max_holding_bars", 1)), purge_bars + embargo_bars),
        cfg=cfg,
    )
    l1 = run_l1_nested_swf(
        labeled_events=labeled_events,
        aligned=aligned,
        outer_folds=outer_folds,
        cfg=cfg,
        seed=int(getattr(cfg, "seed", 42)),
    )

    if not l1.gate_passed:
        logger.info(format_system_status(l1, None, None))
        return (l1, None, None)

    # Layer2: AWF
    awf_folds = build_walk_forward_folds(n_bars=n_bars, cfg=cfg)
    l2 = run_l2_awf(
        l1_oos=l1.oos_stacked,
        aligned=aligned,
        awf_folds=awf_folds,
        l2_params=l2_params,
        caps=caps,
        tf=tf,
    )

    if not l2.gate_passed:
        logger.info(format_system_status(l1, l2, None))
        return (l1, l2, None)

    # Layer3: Holdout
    ho_start_idx = _date_to_idx(aligned.datetimes, window.holdout_start)
    ho_end_idx = _date_to_idx(aligned.datetimes, window.holdout_end)
    l3 = run_l3_holdout(
        l1_oos=l1.oos_stacked,
        aligned=aligned,
        holdout_span=(ho_start_idx, ho_end_idx),
        frozen_params=l2_params,
        caps=caps,
        tf=tf,
    )

    logger.info(format_system_status(l1, l2, l3))
    return (l1, l2, l3)
