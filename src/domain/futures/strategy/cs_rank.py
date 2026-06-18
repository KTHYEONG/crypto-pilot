"""Cross-sectional ranking and signal neutralization for futures strategy.

Provides SymbolSignal dataclass, BTC-beta neutralization, and Top-K selection
with hysteresis buffer for live 24/7 deployment.

Time Complexity: O(N log N) for rank_and_select (argsort dominates).
Space Complexity: O(N) — no intermediate NxN structures.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

# Trace level for high-volume debug logs
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

# VOL_FLOOR: 무한 레버리지 방지용 하한 (~1% daily sigma 기준의 4h bar 환산)
VOL_FLOOR: float = 1e-4  # per-bar sigma 최소값


@dataclass(frozen=True, slots=True)
class SymbolSignal:
    """단일 심볼의 신호 요약.

    Attributes:
        raw_mu: 절대 기대수익 bps (사이징용, magnitude 보존).
        volatility: per-bar sigma, >= VOL_FLOOR 보장 필요 (호출자 책임).
        n_obs: QC 관측 수.
        t_stat: HAC(Newey-West) t-stat.
        valid: Reliability QC 통과 여부.
        beta_btc: BTC 베타 (BTC-β neutralize용, None이면 demean 폴백).
    """

    raw_mu: float
    volatility: float
    n_obs: int
    t_stat: float
    valid: bool
    beta_btc: float | None = None
    quality_weight: float = 1.0


def neutralize_cross_section(
    mu: NDArray[np.float64],
    beta_btc: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """BTC-beta single-factor CS beta-neutralize.

    beta_btc=None이면 단순 CS demean(평균 차감).
    beta_btc 제공 시: μ_neutral_i = μ_i - β_i * μ_mkt (μ_mkt = CS 단순 평균).
    출력: CS 평균 ≈ 0, 순위 보존.

    Args:
        mu: 심볼별 기대수익 배열. Shape: [N], float64.
        beta_btc: BTC 베타 배열. Shape: [N], float64. None이면 단순 demean.

    Returns:
        CS-neutralized μ 배열. Shape: [N], float64. NaN → 0.0으로 대체.

    Time Complexity: O(N).
    Space Complexity: O(N).
    """
    # mu가 비어있거나 len<2이면 그대로 반환
    if mu.size < 2:
        return mu.copy()

    if beta_btc is None:
        mu_neutral = mu - mu.mean()
    else:
        # 단순 평균 사용 (가중평균은 과적합 유발)
        mu_mkt: float = float(mu.mean())
        mu_neutral = mu - beta_btc * mu_mkt

    # epsilon guard: non-finite → 0.0
    return np.where(np.isfinite(mu_neutral), mu_neutral, 0.0)


def rank_and_select(
    signals: Mapping[str, SymbolSignal],
    *,
    k_rank: int,
    sector_cap: int,
    prev_selection: frozenset[str],
    rank_buffer: int,
    min_abs_z: float = 0.0,
    selection_mode: Literal["signed", "absolute"] = "signed",
) -> tuple[frozenset[str], dict[str, float]]:
    """횡단면 Z-score 랭킹으로 Top-K 선택. hysteresis buffer 적용.

    처리 순서:
    1. valid=False 심볼 제외.
    2. BTC-beta neutralize 후 Sharpe Z-score (mu_neutral / vol).
    3. prev_selection 내 심볼은 rank ≤ k_rank + rank_buffer까지 유지(hysteresis).
    4. sector_cap: 동일 sector 최대 선택 수 (sector 미구현 시 무제한).

    Args:
        signals: 심볼 → SymbolSignal 매핑.
        k_rank: 최종 선택 심볼 수.
        sector_cap: 동일 sector 최대 선택 수 (현재 미사용, 확장 예약).
        prev_selection: 이전 bar 선택 집합 (hysteresis 기준).
        rank_buffer: prev_selection 유지를 위한 랭크 여유 폭.
        min_abs_z: 최소 절대 Z-score. 미만 후보는 선택 제외.
        selection_mode: "signed"는 legacy top-k, "absolute"는 futures 대칭 선택.

    Returns:
        Tuple of:
          - frozenset[str]: 선택된 심볼 집합.
          - dict[str, float]: 유효 심볼별 Z-score (선택 여부 무관).

    Time Complexity: O(N log N) — argsort 지배.
    Space Complexity: O(N).
    """
    # valid 심볼만 추출 (삽입 순서 보존: Python 3.7+)
    valid_sigs: dict[str, SymbolSignal] = {
        k: v for k, v in signals.items() if v.valid
    }

    if not valid_sigs:
        _logger.debug("rank_and_select: no valid signals, returning empty selection")
        return frozenset(), {}


    syms: list[str] = list(valid_sigs.keys())
    n: int = len(syms)

    # mu 배열 구성 — float64 강제
    mu_arr: NDArray[np.float64] = np.array(
        [valid_sigs[s].raw_mu * max(valid_sigs[s].quality_weight, 0.0) for s in syms],
        dtype=np.float64,
    )
    vol_arr: NDArray[np.float64] = np.array(
        [max(valid_sigs[s].volatility, VOL_FLOOR) for s in syms], dtype=np.float64
    )

    # beta_btc 배열: 하나라도 non-None이면 배열 구성, 모두 None이면 None 전달 → CS demean 폴백
    beta_btc_arr: NDArray[np.float64] | None = None
    if any(v.beta_btc is not None for v in valid_sigs.values()):
        beta_btc_arr = np.array(
            [v.beta_btc if v.beta_btc is not None else 0.0 for v in (valid_sigs[s] for s in syms)],
            dtype=np.float64,
        )

    # CS beta-neutralize
    mu_neutral: NDArray[np.float64] = neutralize_cross_section(mu_arr, beta_btc_arr)

    sharpe_neutral: NDArray[np.float64] = mu_neutral / vol_arr
    if selection_mode == "signed":
        rank_metric = sharpe_neutral
        rank_order = [syms[i] for i in np.argsort(rank_metric)[::-1].tolist()]
        metric_std = float(rank_metric.std())
        z_metric: NDArray[np.float64] = (rank_metric - rank_metric.mean()) / (metric_std + 1e-12)
    elif selection_mode == "absolute":
        rank_metric = np.abs(sharpe_neutral)
        rank_order = [syms[i] for i in np.argsort(rank_metric)[::-1].tolist()]
        metric_std = float(rank_metric.std())
        rank_z = (rank_metric - rank_metric.mean()) / (metric_std + 1e-12)
        z_metric = np.sign(sharpe_neutral) * rank_z
    else:
        raise ValueError(f"unsupported selection_mode: {selection_mode}")

    # 심볼 → z_score 매핑
    z_scores_dict: dict[str, float] = {syms[i]: float(z_metric[i]) for i in range(n)}
    eligible_symbols: set[str] = {syms[i] for i in range(n) if float(abs(z_metric[i])) >= min_abs_z}

    rank_of: dict[str, int] = {s: idx + 1 for idx, s in enumerate(rank_order)}

    # Hysteresis: prev_selection 중 rank ≤ k_rank + rank_buffer 유지
    sticky: set[str] = {
        s
        for s in prev_selection
        if s in eligible_symbols and rank_of[s] <= k_rank + rank_buffer
    }

    # 새로 진입할 Top candidates (sticky 제외, 여유 슬롯만큼)
    fresh_slots: int = max(0, k_rank - len(sticky))
    fresh: list[str] = [
        s for s in rank_order if s not in sticky and s in eligible_symbols
    ][:fresh_slots]

    selected: set[str] = sticky | set(fresh)

    # k_rank 초과 시 기존 랭킹 우선순위로 trimming
    if len(selected) > k_rank:
        selected = set([s for s in rank_order if s in selected][:k_rank])

    _logger.log(
        TRACE,
        "rank_and_select: n_valid=%d, k_rank=%d, sticky=%d, fresh=%d, selected=%d",
        n,
        k_rank,
        len(sticky),
        len(fresh),
        len(selected),
    )

    return frozenset(selected), {sym: float(z_val) for sym, z_val in z_scores_dict.items()}
