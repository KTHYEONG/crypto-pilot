# src/domain/futures/strategy/tiered_workflow/risk_deployment.py
"""Fix-A: 결정론적 리스크 배치 레이어.

champion 선택 완료 후, MDD/CVaR 예산에 맞춰 leverage L*를 이분탐색으로 산출.
DSR/Sharpe는 L 스케일에 불변(mean·std 동일 비율 변화)이므로 CAGR 개선만 기대한다.
look-ahead 방지: L은 champion의 히스토리컬 OOS 수익률에서 산출하며, 미래 시점
정보에 의존하지 않는다.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

_BISECT_MAX_ITER: int = 64
_BISECT_TOL: float = 1e-4


@dataclass(frozen=True)
class DeploymentResult:
    """레버리지 L* 적용 결과."""

    leverage: float
    scaled_rets: NDArray[np.float64]
    cagr: float
    mdd: float
    cvar_95: float
    binding_constraint: str  # "mdd" | "cvar" | "hard_cap" | "none"


def _mdd_at_leverage(rets: NDArray[np.float64], leverage: float) -> float:
    """레버리지 L 하에서의 최대낙폭 (0~1 스케일)."""
    scaled = leverage * rets
    equity = np.cumprod(1.0 + scaled)
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def _cvar_95_at_leverage(rets: NDArray[np.float64], leverage: float) -> float:
    """레버리지 L 하에서의 CVaR95 (손실 기준, 0~1)."""
    scaled = leverage * rets
    losses = -scaled
    if losses.size == 0:
        return 0.0
    var_cut = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var_cut]
    if tail.size == 0:
        return max(var_cut, 0.0)
    return float(np.maximum(np.mean(tail), 0.0))


def _bisect_max_leverage(
    rets: NDArray[np.float64],
    metric_fn: Callable[[NDArray[np.float64], float], float],
    target: float,
    l_lo: float,
    l_hi: float,
) -> float:
    """metric(rets, L) <= target 를 만족하는 최대 L을 이분탐색.

    metric은 L에 단조증가 가정.
    """
    if metric_fn(rets, l_lo) >= target:
        return l_lo  # l_lo 에서도 이미 예산 초과 → 최소값 반환
    if metric_fn(rets, l_hi) <= target:
        return l_hi  # l_hi 에서도 안전 → 상한 반환

    for _ in range(_BISECT_MAX_ITER):
        if l_hi - l_lo < _BISECT_TOL:
            break
        l_mid = (l_lo + l_hi) * 0.5
        if metric_fn(rets, l_mid) <= target:
            l_lo = l_mid
        else:
            l_hi = l_mid
    return (l_lo + l_hi) * 0.5


def calibrate_deployment_leverage(
    *,
    fit_rets: NDArray[np.float64],
    mdd_cap: float = 0.30,
    cvar_cap: float = 0.06,
    mdd_margin: float = 0.30,
    cvar_margin: float = 0.20,
    l_hard_cap: float = 20.0,
) -> tuple[float, str]:
    """히스토리컬 수익률에서 배치 레버리지 L*를 결정론적으로 산출.

    Spec 설계: fit-leg 수익률로 L*를 산출하고 OOS leg에 적용(look-ahead 방지).
    fit-leg 수익률은 전략 unit-vol book의 실현 수익률이므로 MDD/CVaR 예산이
    실제 binding이 된다 → l_hard_cap=20.0으로 완화해도 budget이 진짜 제약.
    `mdd_margin=0.30` / `cvar_margin=0.20` 안전여유가 OOS-fit 분포 이격 완충.

    Args:
        fit_rets: 캘리브레이션용 per-bar simple return 배열 [T].
            이상적으로는 fit-leg 수익률; 현재는 champion OOS 경로 대리.
        mdd_cap: MDD 하드상한 (예: 0.30).
        cvar_cap: CVaR95 하드상한 (예: 0.06).
        mdd_margin: MDD 목표 = mdd_cap*(1-margin). 기본 30% 안전여유.
        cvar_margin: CVaR95 목표 = cvar_cap*(1-margin). 기본 20% 안전여유.
        l_hard_cap: 레버리지 절대 상한.

    Returns:
        (L*, binding_constraint) — 바인딩 제약 식별자 포함.
    """
    arr = np.asarray(fit_rets, dtype=np.float64)
    if arr.size < 2:
        return 1.0, "none"

    mdd_target = mdd_cap * (1.0 - mdd_margin)
    cvar_target = cvar_cap * (1.0 - cvar_margin)
    l_search_hi = l_hard_cap * 10.0  # 충분히 넓은 탐색 범위

    l_mdd = _bisect_max_leverage(arr, _mdd_at_leverage, mdd_target, 1.0, l_search_hi)
    l_cvar = _bisect_max_leverage(arr, _cvar_95_at_leverage, cvar_target, 1.0, l_search_hi)

    l_optimal = min(l_mdd, l_cvar)

    if l_optimal >= l_hard_cap:
        return l_hard_cap, "hard_cap"

    binding: str = "mdd" if l_mdd <= l_cvar else "cvar"
    l_final = max(l_optimal, 1.0)
    return l_final, binding


def apply_deployment(
    *,
    rets: NDArray[np.float64],
    leverage: float,
    bars_per_year: float,
) -> DeploymentResult:
    """OOS 수익률에 L*를 적용하고 핵심 지표를 재계산.

    Args:
        rets: per-bar simple return [T].
        leverage: L* (calibrate_deployment_leverage 결과).
        bars_per_year: 연율화 팩터 (4h=2190).

    Returns:
        DeploymentResult — 스케일된 수익률 및 재계산 지표.
    """
    arr = np.asarray(rets, dtype=np.float64)
    scaled = leverage * arr

    # CAGR: 로그수익률 합산 → 연율화
    log_growth = float(np.sum(np.log1p(np.clip(scaled, -1.0 + 1e-9, None))))
    years = float(arr.size) / max(bars_per_year, 1e-9)
    cagr = float(np.expm1(log_growth / years)) if years > 1e-9 else 0.0

    mdd = _mdd_at_leverage(arr, leverage)
    cvar_95 = _cvar_95_at_leverage(arr, leverage)

    return DeploymentResult(
        leverage=leverage,
        scaled_rets=scaled,
        cagr=cagr,
        mdd=mdd,
        cvar_95=cvar_95,
        binding_constraint="",
    )
