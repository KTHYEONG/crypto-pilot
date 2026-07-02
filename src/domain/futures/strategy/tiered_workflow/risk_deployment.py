# src/domain/futures/strategy/tiered_workflow/risk_deployment.py
"""Fix-A: 결정론적 리스크 배치 레이어.

champion 선택 완료 후, MDD/CVaR 예산에 맞춰 leverage L*를 이분탐색으로 산출.
DSR/Sharpe는 L 스케일에 불변(mean·std 동일 비율 변화)이므로 CAGR 개선만 기대한다.
look-ahead 방지: L은 champion의 히스토리컬 OOS 수익률에서 산출하며, 미래 시점
정보에 의존하지 않는다.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2FoldDiagnostics

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


def trend_efficiency_gross_mult(
    trailing_er: float,
    *,
    target: float,
    floor_mult: float,
) -> float:
    if not np.isfinite(trailing_er):
        return floor_mult
    if trailing_er >= target:
        return 1.0
    return float(floor_mult + (1.0 - floor_mult) * (trailing_er / max(target, 1e-12)))


def _annualized_cagr_from_returns(
    returns: NDArray[np.float64],
    *,
    bars_per_year: float,
) -> float:
    if returns.size == 0:
        return 0.0
    clipped = np.clip(returns, -1.0 + 1e-9, None)
    log_growth = float(np.sum(np.log1p(clipped)))
    years = float(returns.size) / max(float(bars_per_year), 1e-9)
    if years <= 1e-9:
        return 0.0
    return float(np.expm1(log_growth / years))


def _mdd_from_returns(returns: NDArray[np.float64]) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.maximum(peak, 1e-12)
    return float(np.max(drawdown)) if drawdown.size else 0.0


def _sharpe_from_returns(
    returns: NDArray[np.float64],
    *,
    bars_per_year: float,
) -> float:
    if returns.size < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float((mean / std) * np.sqrt(max(float(bars_per_year), 1e-9)))


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
    oos_rets: NDArray[np.float64] | None = None,
    mdd_cap: float = 0.30,
    cvar_cap: float = 0.06,
    mdd_margin: float = 0.30,
    cvar_margin: float = 0.20,
    l_hard_cap: float = 20.0,
    exchange_leverage_cap: float | None = None,
    l_floor: float = 1.0,
    oos_budget_blend: float = 0.5,
    oos_floor_cap: float = 4.0,
    fit_mdd_crisis_gate: float | None = None,
) -> tuple[float, str, float]:
    """히스토리컬 수익률에서 배치 레버리지 L*를 결정론적으로 산출.

    Spec 설계: fit-leg 수익률로 L*를 산출하고 OOS leg에 적용(look-ahead 방지).
    fit-leg 수익률은 전략 unit-vol book의 실현 수익률이므로 MDD/CVaR 예산이
    실제 binding이 된다 → l_hard_cap=20.0으로 완화해도 budget이 진짜 제약.
    `mdd_margin=0.30` / `cvar_margin=0.20` 안전여유가 OOS-fit 분포 이격 완충.
    `exchange_leverage_cap`으로 거래소 실행가능 notional 상한 제한(trading_bot.md §4).

    RC-2: fit/OOS 역전 시 OOS 실현 리스크 예산을 직접 사용하도록 blended budget L*
    도입. `oos_budget_blend`로 fit-OOS 혼합비 조절, `oos_floor_cap`으로 OOS-floor
    상한 파라미터화(기존 하드코딩 2.0 대체).

    Args:
        fit_rets: 캘리브레이션용 per-bar simple return 배열 [T].
            이상적으로는 fit-leg 수익률; 현재는 champion OOS 경로 대리.
        oos_rets: OOS per-bar simple return 배열 [T] (선택). 제공 시 L*의 OOS
            크로스 검증 MDD를 계산하여 세 번째 반환값으로 전달.
        mdd_cap: MDD 하드상한 (예: 0.30).
        cvar_cap: CVaR95 하드상한 (예: 0.06).
        mdd_margin: MDD 목표 = mdd_cap*(1-margin). 기본 30% 안전여유.
        cvar_margin: CVaR95 목표 = cvar_cap*(1-margin). 기본 20% 안전여유.
        l_hard_cap: 레버리지 절대 상한.
        exchange_leverage_cap: 거래소 실행가능 notional 레버리지 상한. None=무제한.
            Binance perp 기본 10x. L* > cap 이면 "exchange_cap" binding으로 차단.
        l_floor: L* 하한. 기본 1.0(기존 동작 보존). <1.0 허용 시 100%-vol book을
            MDD 예산까지 de-lever 가능(RC-5 수정 동반 필수).
        oos_budget_blend: fit-OOS blended budget ratio. 0=pure fit, 1=pure OOS.
            기본 0.5. RC-2 look-ahead 완충.
        oos_floor_cap: OOS-floor L* 상한. 기본 4.0 (기존 하드코딩 2.0 대체).
        fit_mdd_crisis_gate: fit-leg unit-vol MDD 임계값(0~1). None(기본)=비활성(기존 동작).
            지정 시 fit_MDD_vol1 >= 임계값이면 RC-2 oos_blend 분기 자체를 건너뛰고
            fit-only calibration 결과(binding∈{mdd,cvar,hard_cap,exchange_cap})를 유지.

    Returns:
        (L*, binding_constraint, cross_valid_MDD_at_L) — cross_valid_MDD_at_L는
        oos_rets가 제공된 경우에만 실제 계산값. 미제공 시 0.0 반환.
        binding ∈ {"mdd","cvar","hard_cap","exchange_cap","oos_blend","none"}.
    """
    arr = np.asarray(fit_rets, dtype=np.float64)
    if arr.size < 2:
        _logger.debug("[L2-CALIB] fit_rets size<2, returning L*=1.0 (none)")
        return 1.0, "none", 0.0

    mdd_target = mdd_cap * (1.0 - mdd_margin)
    cvar_target = cvar_cap * (1.0 - cvar_margin)
    l_search_hi = l_hard_cap * 10.0  # 충분히 넓은 탐색 범위

    l_mdd = _bisect_max_leverage(arr, _mdd_at_leverage, mdd_target, float(l_floor), l_search_hi)
    l_cvar = _bisect_max_leverage(arr, _cvar_95_at_leverage, cvar_target, float(l_floor), l_search_hi)

    # 모든 제약 후보 수집 → argmin으로 binding 결정 (realism: trading_bot.md §4)
    candidates: list[tuple[float, str]] = [
        (l_mdd, "mdd"),
        (l_cvar, "cvar"),
        (l_hard_cap, "hard_cap"),
    ]
    if exchange_leverage_cap is not None and exchange_leverage_cap > 0.0:
        candidates.append((exchange_leverage_cap, "exchange_cap"))

    l_optimal, binding = min(candidates, key=lambda x: x[0])
    l_final = max(l_optimal, float(l_floor))

    # OOS 크로스 검증 + RC-2 blended budget (fit/OOS 역전 시 OOS 예산 직접 사용)
    cross_valid_mdd: float = 0.0
    if oos_rets is not None:
        oos_arr = np.asarray(oos_rets, dtype=np.float64)
        if oos_arr.size >= 2:
            cross_valid_mdd = _mdd_at_leverage(oos_arr, l_final)
            _oos_mdd_v1 = _mdd_at_leverage(oos_arr, 1.0)
            _fit_mdd_v1 = _mdd_at_leverage(arr, 1.0)
            _mdd_ratio = _oos_mdd_v1 / _fit_mdd_v1 if _fit_mdd_v1 > 1e-12 else 1.0
            _fit_cagr_v1 = _annualized_cagr_from_returns(arr, bars_per_year=2190)
            _oos_cagr_v1 = _annualized_cagr_from_returns(oos_arr, bars_per_year=2190)
            _fit_sharpe_v1 = _sharpe_from_returns(arr, bars_per_year=2190)
            _oos_sharpe_v1 = _sharpe_from_returns(oos_arr, bars_per_year=2190)
            _logger.debug(
                "[L2-CALIB-CV] L*=%.4f(%s) | fit_MDD_vol1=%.6f OOS_MDD_vol1=%.6f "
                "MDD_ratio=%.2f | OOS_deployed_MDD=%.6f (cap=%.4f) | "
                "fit_CAGR_v1=%.4f fit_sharpe_v1=%.4f OOS_CAGR_v1=%.4f OOS_sharpe_v1=%.4f",
                l_final, binding,
                _fit_mdd_v1, _oos_mdd_v1, _mdd_ratio,
                cross_valid_mdd, mdd_cap,
                _fit_cagr_v1, _fit_sharpe_v1, _oos_cagr_v1, _oos_sharpe_v1,
            )
            # Crisis gate: fit-leg 자체가 재앙적 MDD(>=threshold)면 OOS가 "안전해 보인다"는
            # 이유만으로 레버리지를 올리지 않음 — fit-leg의 보수적 경고를 무시하지 않도록 조기 차단.
            if fit_mdd_crisis_gate is not None and _fit_mdd_v1 >= fit_mdd_crisis_gate:
                _logger.debug(
                    "[L2-OOS-BLEND-SUPPRESSED] fit_MDD_vol1=%.4f >= crisis_gate=%.4f "
                    "-> oos_blend skipped, L* stays %.4f (%s)",
                    _fit_mdd_v1, fit_mdd_crisis_gate, l_final, binding,
                )
            # RC-2: fit/OOS 역전 시 blended budget (기존 매직캡 min(2.0, ...) 대체)
            elif _mdd_ratio < 1.0:
                _mdd_target_oos = mdd_cap * (1.0 - mdd_margin * 0.5)
                l_oos = _bisect_max_leverage(oos_arr, _mdd_at_leverage, _mdd_target_oos, float(l_floor), l_search_hi)
                l_blend = l_final * (1.0 - oos_budget_blend) + l_oos * oos_budget_blend
                _l_candidate = min(l_blend, oos_floor_cap)
                if _l_candidate > l_final:
                    _prev_final = l_final
                    l_final = _l_candidate
                    binding = "oos_blend"
                    # 불변식: OOS deployed MDD ≤ mdd_cap*(1-mdd_margin*0.5)
                    _deployed_mdd = _mdd_at_leverage(oos_arr, l_final)
                    _oos_invariant = mdd_cap * (1.0 - mdd_margin * 0.5)
                    if _deployed_mdd > _oos_invariant:
                        l_final = _prev_final
                        binding = "oos_blend"
                        _logger.debug(
                            "[L2-OOS-BLEND] L_candidate=%.4f exceeds OOS invariant %.4f "
                            "→ reverted to L*=%.4f",
                            _l_candidate, _oos_invariant, l_final,
                        )
                    else:
                        cross_valid_mdd = _deployed_mdd
                        _logger.debug(
                            "[L2-OOS-BLEND] Raised L* from %.4f to %.4f (blend=%.2f, "
                            "oos_mdd_v1=%.4f, L_oos=%.4f, OOS_deployed_MDD=%.4f ≤ %.4f)",
                            _prev_final, l_final, oos_budget_blend,
                            _oos_mdd_v1, l_oos, cross_valid_mdd, _oos_invariant,
                        )
        else:
            _logger.debug("[L2-CALIB-CV] oos_rets size<2, skipping cross-validation")

    # 최종: clip(L*, l_floor, min(l_hard_cap, exchange_cap)) — spec Algorithm §6
    if exchange_leverage_cap is not None and exchange_leverage_cap > 0.0 and l_final > exchange_leverage_cap:
        l_final = exchange_leverage_cap
        binding = "exchange_cap"
    elif l_final > l_hard_cap:
        l_final = l_hard_cap
        binding = "hard_cap"

    return l_final, binding, cross_valid_mdd


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


def compute_layer2_fold_diagnostics(
    *,
    fold_rets_hybrid: Sequence[Sequence[float]],
    fold_selected_symbols: Sequence[Sequence[str]],
    leverage: float,
    bars_per_year: float,
) -> Layer2FoldDiagnostics:
    """Compute fold-level deployment diagnostics for Layer2."""
    fold_compound_pass: list[bool | None] = []
    fold_unit_sharpes: list[float] = []
    fold_deployed_cagrs: list[float | None] = []
    fold_deployed_mdds: list[float | None] = []
    selected_symbols: list[tuple[str, ...]] = []
    deployed_cagrs_nonempty: list[float] = []
    recent_fold_passed: bool | None = None
    recent_fold_sharpe: float | None = None
    recent_fold_cagr = 0.0
    recent_fold_mdd = 0.0

    for fold_idx, fold_rets in enumerate(fold_rets_hybrid):
        fold_arr = np.asarray(fold_rets, dtype=np.float64)
        if fold_idx < len(fold_selected_symbols):
            selected = tuple(sorted(str(sym) for sym in fold_selected_symbols[fold_idx]))
        else:
            selected = ()
        selected_symbols.append(selected)

        if fold_arr.size == 0:
            fold_compound_pass.append(None)
            fold_unit_sharpes.append(0.0)
            fold_deployed_cagrs.append(None)
            fold_deployed_mdds.append(None)
            continue

        unit_sharpe = _sharpe_from_returns(fold_arr, bars_per_year=bars_per_year)
        deployed = apply_deployment(
            rets=fold_arr,
            leverage=float(leverage),
            bars_per_year=bars_per_year,
        )
        fold_unit_sharpes.append(unit_sharpe)
        fold_compound_pass.append(bool(deployed.cagr > 0.0))
        fold_deployed_cagrs.append(float(deployed.cagr))
        fold_deployed_mdds.append(float(deployed.mdd))
        deployed_cagrs_nonempty.append(float(deployed.cagr))
        recent_fold_passed = bool(deployed.cagr > 0.0)
        recent_fold_sharpe = unit_sharpe
        recent_fold_cagr = float(deployed.cagr)
        recent_fold_mdd = float(deployed.mdd)

    nonempty_pass = [value for value in fold_compound_pass if value is not None]
    fold_pass_ratio = (
        sum(1 for value in nonempty_pass if value) / len(nonempty_pass)
        if nonempty_pass
        else 0.0
    )
    latest_to_median_cagr = 0.0
    if deployed_cagrs_nonempty:
        median_cagr = float(np.median(np.asarray(deployed_cagrs_nonempty, dtype=np.float64)))
        if abs(median_cagr) > 1e-12:
            latest_to_median_cagr = float(recent_fold_cagr / median_cagr)

    return Layer2FoldDiagnostics(
        fold_pass_ratio=float(fold_pass_ratio),
        fold_compound_pass=tuple(fold_compound_pass),
        fold_unit_sharpes=tuple(fold_unit_sharpes),
        fold_deployed_cagrs=tuple(fold_deployed_cagrs),
        fold_deployed_mdds=tuple(fold_deployed_mdds),
        fold_selected_symbols=tuple(selected_symbols),
        recent_fold_passed=recent_fold_passed,
        recent_fold_sharpe=recent_fold_sharpe,
        recent_fold_cagr=float(recent_fold_cagr),
        recent_fold_mdd=float(recent_fold_mdd),
        latest_to_median_cagr=float(latest_to_median_cagr),
    )
