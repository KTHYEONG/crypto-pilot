"""3-Layer 티어드 파이프라인 오케스트레이터.

Layer1 (CPCV Signal Validation) → Layer2 (AWF Portfolio) → Layer3 (Holdout) 순서로
게이트 기반 단계적 검증을 수행한다.

Time Complexity: O(F * T * N) — F=folds, T=bars, N=symbols
Space Complexity: O(F * N) — fold별 signal 집계
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr

from src.domain.futures.portfolio.portfolio_constructor import (
    PortfolioCaps,
    diagonal_kelly_weights,
)
from src.domain.futures.portfolio.signal_composer import (
    compose_symbol_signals,
    composer_sigma_lookback_bars,
    rolling_per_bar_return_std,
)
from src.domain.futures.strategy.candidate_workflow import _fit_and_predict_single_fold
from src.domain.futures.strategy.cs_rank import (
    VOL_FLOOR,
    SymbolSignal,
    rank_and_select,
)
from src.domain.futures.strategy.tiered_logging import (
    format_layer1_table,
    format_layer2_table,
    format_layer3_table,
    format_system_status,
)
from src.domain.futures.strategy.walk_forward import (
    CPCVFold,
    WFFold,
    build_cpcv_folds,
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
    """Layer1 CPCV 검증 결과.

    Attributes:
        signals_per_fold: fold별 symbol→SymbolSignal 매핑 튜플.
        oos_stacked: fold 횡단 합본 (L2 입력용, look-ahead 없음).
        mean_ic: fold IC 평균 (HAC 단순화: 독립 fold 가정).
        ic_tstat: HAC IC t-통계량.
        breadth: 평균 valid 심볼 비율 (per fold).
        valid_coverage: valid 심볼 비율 ≥ 0.5인 fold 비율.
        fold_pass_ratio: IC > 0인 fold 비율.
        gate_passed: L1 통과 여부.
        n_valid: 마지막 fold 기준 valid 심볼 수.
        n_total: 전체 심볼 수 (aligned width).
        n_trade_scope: tiered aligned scope 크기 (bridge와 동일한 Stage6 OOS ∩ data-valid).
    """

    signals_per_fold: tuple[dict[str, SymbolSignal], ...]
    oos_stacked: dict[str, SymbolSignal]
    mean_ic: float
    ic_tstat: float
    breadth: float
    valid_coverage: float
    fold_pass_ratio: float
    gate_passed: bool
    n_valid: int
    n_total: int
    n_trade_scope: int = 0


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
    n_events: int
    passed: bool


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

_BARS_PER_YEAR: float = 2190.0  # 4h 기준
_VALID_COVERAGE_FLAG_THRESHOLD: float = 0.80  # per-fold valid 비율 임계 (L1 게이트와 일치)


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
    prev_w = np.zeros(n_sym, dtype=np.float64)
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
            mu_arr = np.zeros(n_sym, dtype=np.float64)
            sig_arr = np.full(n_sym, VOL_FLOOR, dtype=np.float64)
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


def _cpcv_to_wf_fold(cpcv: CPCVFold) -> WFFold | None:
    """CPCVFold → WFFold 변환.

    test_spans 없는 fallback fold는 None 반환.

    Args:
        cpcv: CPCVFold 인스턴스.

    Returns:
        WFFold 또는 None (변환 불가).
    """
    if not cpcv.fit_spans or not cpcv.test_spans:
        return None
    fit_s = min(s for s, _e in cpcv.fit_spans)
    fit_e = max(e for _s, e in cpcv.fit_spans)
    oos_s = min(s for s, _e in cpcv.test_spans)
    oos_e = max(e for _s, e in cpcv.test_spans)
    cal_len = max(1, (fit_e - fit_s) // 5)
    cal_s = fit_e - cal_len
    return WFFold(
        fit_start=fit_s,
        fit_end=cal_s,
        cal_start=cal_s,
        cal_end=fit_e,
        oos_start=oos_s,
        oos_end=oos_e,
    )


def _stack_oos_signals(
    signals_per_fold: tuple[dict[str, SymbolSignal], ...],
) -> dict[str, SymbolSignal]:
    """fold별 SymbolSignal을 per-symbol로 집계 (raw_mu 평균).

    Args:
        signals_per_fold: fold별 symbol→SymbolSignal 매핑 튜플.

    Returns:
        합본 symbol→SymbolSignal 매핑.

    Time Complexity: O(F * N)
    Space Complexity: O(N)
    """
    sym_mu_lists: dict[str, list[float]] = defaultdict(list)
    sym_sig_ref: dict[str, SymbolSignal] = {}
    for fold_sigs in signals_per_fold:
        for sym, sig in fold_sigs.items():
            sym_mu_lists[sym].append(sig.raw_mu)
            sym_sig_ref[sym] = sig  # 마지막 fold 기준 메타 사용

    oos_stacked: dict[str, SymbolSignal] = {}
    for sym, mus in sym_mu_lists.items():
        ref = sym_sig_ref[sym]
        oos_stacked[sym] = SymbolSignal(
            raw_mu=float(np.mean(mus)),
            volatility=ref.volatility,
            n_obs=ref.n_obs * len(mus),
            t_stat=ref.t_stat,
            valid=ref.valid,
            beta_btc=ref.beta_btc,
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
            ic_val, _ = spearmanr(p[valid_mask], r[valid_mask])
            if not np.isnan(ic_val):
                sym_ic_lists[str(sym)].append(float(ic_val))

    return {sym: float(np.mean(ics)) for sym, ics in sym_ic_lists.items() if ics}


def _compute_fold_ts_ic(*, fold_out: Any) -> float | None:
    """fold OOS pooled time-series Spearman rank IC (expected_net_bps vs y_return_bps).

    oos_set.y_return_bps는 방향·barrier·비용 반영 정준 실현 수익 라벨이다.
    Returns None if oos_set unavailable or fewer than 4 valid events.
    """
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

    ic_val, _ = spearmanr(pred[mask], realized[mask])
    return float(ic_val) if not np.isnan(ic_val) else None


# ---------------------------------------------------------------------------
# Layer1: CPCV Signal Validation
# ---------------------------------------------------------------------------


def run_l1_cpcv(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    folds: tuple[CPCVFold, ...],
    l1_params: dict[str, Any],
    min_obs: int = 20,
    t_stat_floor: float = 1.96,
    tf: str = "4h",
) -> Layer1Result:
    """Layer1 CPCV 신호 검증.

    각 CPCV fold에서 모델 학습/예측 후 SymbolSignal 집계, HAC IC 게이트 적용.

    Args:
        labeled_events: 레이블링된 이벤트 DataFrame.
        aligned: AlignedMarketData (close_2d, symbols, datetimes 포함).
        cfg: CandidateStrategyConfig.
        folds: CPCVFold 튜플.
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
        "[CPCV-START] Starting CPCV L1 signal validation parallelization with %d folds (max_workers=%d)",
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

    futures = []
    try:
        if max_workers <= 1 or len(folds) <= 1:
            for fold_idx, cpcv_fold in enumerate(folds):
                wf_fold = _cpcv_to_wf_fold(cpcv_fold)
                if wf_fold is None:
                    continue
                try:
                    fold_out = _fit_and_predict_single_fold(
                        fold_idx, wf_fold, labeled_events, aligned, cfg, purge_bars
                    )
                    futures.append((fold_idx, wf_fold, fold_out))
                except Exception:
                    logger.warning("run_l1_cpcv: fold %d 학습 실패, 스킵", fold_idx, exc_info=True)
        else:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
                submits = []
                for fold_idx, cpcv_fold in enumerate(folds):
                    wf_fold = _cpcv_to_wf_fold(cpcv_fold)
                    if wf_fold is None:
                        continue
                    submits.append(
                        (
                            fold_idx,
                            wf_fold,
                            executor.submit(cw._fit_and_predict_single_fold_from_globals, fold_idx, wf_fold)
                        )
                    )
                for fold_idx, wf_fold, fut in submits:
                    try:
                        fold_out = fut.result()
                        futures.append((fold_idx, wf_fold, fold_out))
                    except Exception:
                        logger.warning("run_l1_cpcv: fold %d 학습 실패, 스킵", fold_idx, exc_info=True)
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

        f_n_valid = sum(1 for s in fold_sigs.values() if s.valid)
        f_breadth = f_n_valid / max(1, n_total)
        f_n_events = len(fold_out.model_output.expected_net_bps)
        fold_diags.append(FoldDiagnostic(
            fold=fold_loop_idx + 1,
            ic=fold_ic,
            breadth=f_breadth,
            n_valid=f_n_valid,
            n_events=f_n_events,
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
            "[CPCV-PROFILE] Average sub-fold execution breakdown: "
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
        "[CPCV-END] CPCV L1 signal validation parallel execution completed in %.2fs",
        time.perf_counter() - t_start,
    )

    # --- OOS stacking ---
    sigs_tuple = tuple(signals_per_fold)
    oos_stacked = _stack_oos_signals(sigs_tuple)

    # --- IC 통계 (fold_diags 기반 — 인덱스 분리 없음) ---
    fold_perf_details: list[dict[str, Any]] = [
        {
            "fold": d.fold,
            "ic": d.ic,
            "breadth": d.breadth,
            "n_valid": d.n_valid,
            "n_events": d.n_events,
            "pass": d.passed,
        }
        for d in fold_diags
    ]

    valid_ics: list[float] = [d.ic for d in fold_diags if d.ic is not None]
    if valid_ics:
        mean_ic = float(np.mean(valid_ics))
        std_ic = float(np.std(valid_ics, ddof=1)) if len(valid_ics) > 1 else 0.0
        n_f = len(valid_ics)
        # CPCV 중첩 보정: 인접 fold IC 자기상관 rho_hat으로 유효 표본 수 축소
        if n_f >= 3:
            ic_arr = np.asarray(valid_ics, dtype=np.float64)
            demeaned = ic_arr - mean_ic
            rho_hat = float(np.dot(demeaned[:-1], demeaned[1:]) / (np.dot(demeaned, demeaned) + 1e-20))
            rho_hat = float(np.clip(rho_hat, 0.0, 0.99))
            n_eff = max(1.0, n_f / (1.0 + 2.0 * rho_hat))
        else:
            n_eff = float(n_f)
        ic_tstat = mean_ic * np.sqrt(n_eff) / (std_ic + 1e-9)
        fold_pass_ratio = float(sum(1 for ic in valid_ics if ic > 0) / n_f)
    else:
        mean_ic = 0.0
        ic_tstat = 0.0
        fold_pass_ratio = 0.0

    # --- breadth / valid_coverage (FoldDiagnostic 기반) ---
    breadth = float(np.mean([d.breadth for d in fold_diags])) if fold_diags else 0.0
    valid_coverage = (
        float(sum(1 for d in fold_diags if d.breadth >= _VALID_COVERAGE_FLAG_THRESHOLD) / len(fold_diags))
        if fold_diags else 0.0
    )

    # n_valid: oos_stacked 기준
    n_valid = sum(1 for s in oos_stacked.values() if s.valid)

    # Per-symbol time-series rank IC (oos_set.y_return_bps 기반 정준 계산)
    per_sym_ic = compute_per_symbol_ic(fold_tuples=futures)

    # Final Per-symbol aggregate collection
    sym_details: list[dict[str, Any]] = []
    for sym, sig in sorted(oos_stacked.items()):
        sym_details.append({
            "symbol": sym,
            "raw_mu": sig.raw_mu,
            "vol": sig.volatility,
            "t_stat": sig.t_stat,
            "ic": per_sym_ic.get(sym, 0.0),
            "valid": sig.valid,
        })

    gate_passed: bool = bool(
        (mean_ic >= 0.030)
        and (ic_tstat >= 1.96)
        and (breadth >= 0.30)
        and (valid_coverage >= 0.80)
        and (fold_pass_ratio >= 0.60)
    )

    result = Layer1Result(
        signals_per_fold=sigs_tuple,
        oos_stacked=oos_stacked,
        mean_ic=mean_ic,
        ic_tstat=ic_tstat,
        breadth=breadth,
        valid_coverage=valid_coverage,
        fold_pass_ratio=fold_pass_ratio,
        gate_passed=gate_passed,
        n_valid=n_valid,
        n_total=n_total,
        n_trade_scope=n_total,
    )
    logger.info(format_layer1_table(result, fold_details=fold_perf_details, per_symbol_top10=sym_details))
    return result


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

    # Layer1: CPCV
    cpcv_folds = build_cpcv_folds(
        n_bars=n_bars,
        n_groups=6,
        n_test_groups=2,
        embargo_bars=embargo_bars,
        purge_bars=purge_bars,
    )
    l1 = run_l1_cpcv(
        labeled_events=labeled_events,
        aligned=aligned,
        cfg=cfg,
        folds=cpcv_folds,
        l1_params=l1_params,
        tf=tf,
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
