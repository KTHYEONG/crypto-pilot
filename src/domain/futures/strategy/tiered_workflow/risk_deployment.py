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


def select_worst_fold_returns(
    fit_rets_by_fold: tuple[tuple[float, ...], ...],
) -> NDArray[np.float64]:
    """[ADR_20260717_L2_DEPLOY_LEVERAGE_KELLY_WORST_FOLD_SAFETY] 챔피언 자신의
    walk-forward fold 중 단위 레버리지 MDD가 가장 큰 fold의 fit-leg 수익률을
    반환한다.

    Crisis window를 전혀 참조하지 않는다 — 입력은 이미 champion selection이
    끝난 학습 horizon 내부의 fold들뿐이다(look-ahead 없음).

    Args:
        fit_rets_by_fold: fold별 fit-leg per-bar simple return 튜플들.

    Returns:
        최악(unit-leverage MDD 최대) fold의 수익률 배열. 입력이 비어있거나
        모든 fold의 크기가 2 미만이면 빈 배열(size=0)을 반환한다.
    """
    if len(fit_rets_by_fold) < 2:
        return np.array([], dtype=np.float64)

    worst_idx = -1
    worst_mdd = -1.0
    for i, fold in enumerate(fit_rets_by_fold):
        if len(fold) < 2:
            continue
        arr = np.asarray(fold, dtype=np.float64)
        mdd = _mdd_from_returns(arr)
        if mdd > worst_mdd:
            worst_mdd = mdd
            worst_idx = i

    if worst_idx < 0:
        return np.array([], dtype=np.float64)

    return np.asarray(fit_rets_by_fold[worst_idx], dtype=np.float64)


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


def _resolve_safety_ceiling(
    fit_rets: NDArray[np.float64],
    *,
    mdd_target: float,
    cvar_target: float,
    l_floor: float,
    l_hard_cap: float,
    l_search_hi: float,
    exchange_leverage_cap: float | None,
    worst_fold_rets: NDArray[np.float64] | None,
    kelly_safety_fraction: float | None,
) -> tuple[float, str, float, str]:
    """[ADR_20260717_L2_LEVERAGE_CEILING_REFACTOR] Returns (l_full, full_binding,
    l_hard, hard_binding). l_full includes mdd/cvar (RC-2 OOS-blend may still
    override these); l_hard excludes them and is the absolute ceiling no
    adaptive logic downstream may exceed."""
    l_mdd = _bisect_max_leverage(fit_rets, _mdd_at_leverage, mdd_target, float(l_floor), l_search_hi)
    l_cvar = _bisect_max_leverage(fit_rets, _cvar_95_at_leverage, cvar_target, float(l_floor), l_search_hi)

    candidates: list[tuple[float, str]] = [
        (l_mdd, "mdd"),
        (l_cvar, "cvar"),
        (l_hard_cap, "hard_cap"),
    ]
    if exchange_leverage_cap is not None and exchange_leverage_cap > 0.0:
        candidates.append((exchange_leverage_cap, "exchange_cap"))
    if worst_fold_rets is not None:
        wf_arr = np.asarray(worst_fold_rets, dtype=np.float64)
        if wf_arr.size >= 2:
            l_worst_fold = _bisect_max_leverage(wf_arr, _mdd_at_leverage, mdd_target, float(l_floor), l_search_hi)
            candidates.append((l_worst_fold, "worst_fold"))
    if kelly_safety_fraction is not None:
        mu = float(np.mean(fit_rets))
        sigma = float(np.std(fit_rets, ddof=1))
        if mu > 0.0 and sigma > 1e-12:
            l_kelly = kelly_safety_fraction * mu / (sigma * sigma)
            if l_kelly > 0.0 and np.isfinite(l_kelly):
                candidates.append((l_kelly, "kelly_theoretical"))

    l_full, full_binding = min(candidates, key=lambda x: x[0])
    hard_candidates = [(v, l) for v, l in candidates if l not in ("mdd", "cvar")]
    l_hard, hard_binding = min(hard_candidates, key=lambda x: x[0]) if hard_candidates else (l_full, full_binding)
    return l_full, full_binding, l_hard, hard_binding


def _resolve_oos_adaptive_leverage(
    fit_rets: NDArray[np.float64],
    oos_rets: NDArray[np.float64] | None,
    *,
    l_ceiling: float,
    l_hard_ceiling: float,
    ceiling_binding: str,
    mdd_cap: float,
    mdd_margin: float,
    oos_budget_blend: float,
    oos_floor_cap: float,
    fit_mdd_crisis_gate: float | None,
    l_floor: float,
    l_search_hi: float,
) -> tuple[float, str, float]:
    """[ADR_20260717_L2_LEVERAGE_CEILING_REFACTOR] RC-2 blend raise is clamped
    to l_hard_ceiling — the fix for the OOS-blend bypass bug."""
    l_final = max(l_ceiling, float(l_floor))
    binding = ceiling_binding
    cross_valid_mdd: float = 0.0

    if oos_rets is not None:
        oos_arr = np.asarray(oos_rets, dtype=np.float64)
        if oos_arr.size >= 2:
            cross_valid_mdd = _mdd_at_leverage(oos_arr, l_final)
            _oos_mdd_v1 = _mdd_at_leverage(oos_arr, 1.0)
            _fit_mdd_v1 = _mdd_at_leverage(fit_rets, 1.0)
            _mdd_ratio = _oos_mdd_v1 / _fit_mdd_v1 if _fit_mdd_v1 > 1e-12 else 1.0
            _fit_cagr_v1 = _annualized_cagr_from_returns(fit_rets, bars_per_year=2190)
            _oos_cagr_v1 = _annualized_cagr_from_returns(oos_arr, bars_per_year=2190)
            _fit_sharpe_v1 = _sharpe_from_returns(fit_rets, bars_per_year=2190)
            _oos_sharpe_v1 = _sharpe_from_returns(oos_arr, bars_per_year=2190)
            _logger.debug(
                "[L2-CALIB-CV] L*=%.4f(%s) | fit_MDD_vol1=%.6f OOS_MDD_vol1=%.6f "
                "MDD_ratio=%.2f | OOS_deployed_MDD=%.6f (cap=%.4f) | "
                "fit_CAGR_v1=%.4f fit_sharpe_v1=%.4f OOS_CAGR_v1=%.4f OOS_sharpe_v1=%.4f",
                l_final,
                binding,
                _fit_mdd_v1,
                _oos_mdd_v1,
                _mdd_ratio,
                cross_valid_mdd,
                mdd_cap,
                _fit_cagr_v1,
                _fit_sharpe_v1,
                _oos_cagr_v1,
                _oos_sharpe_v1,
            )
            if fit_mdd_crisis_gate is not None and _fit_mdd_v1 >= fit_mdd_crisis_gate:
                _logger.debug(
                    "[L2-OOS-BLEND-SUPPRESSED] fit_MDD_vol1=%.4f >= crisis_gate=%.4f "
                    "-> oos_blend skipped, L* stays %.4f (%s)",
                    _fit_mdd_v1, fit_mdd_crisis_gate, l_final, binding,
                )
            elif _mdd_ratio < 1.0:
                _mdd_target_oos = mdd_cap * (1.0 - mdd_margin * 0.5)
                l_oos = _bisect_max_leverage(oos_arr, _mdd_at_leverage, _mdd_target_oos, float(l_floor), l_search_hi)
                l_blend = l_final * (1.0 - oos_budget_blend) + l_oos * oos_budget_blend
                _l_candidate = min(l_blend, oos_floor_cap, l_hard_ceiling)
                if _l_candidate > l_final:
                    _prev_final = l_final
                    l_final = _l_candidate
                    binding = "oos_blend"
                    _deployed_mdd = _mdd_at_leverage(oos_arr, l_final)
                    _oos_invariant = mdd_cap * (1.0 - mdd_margin * 0.5)
                    if _deployed_mdd > _oos_invariant:
                        l_final = _prev_final
                        binding = "oos_blend"
                        _logger.debug(
                            "[L2-OOS-BLEND] L_candidate=%.4f exceeds OOS invariant %.4f → reverted to L*=%.4f",
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

    return l_final, binding, cross_valid_mdd


def _apply_concentration_haircut(
    l_final: float,
    binding: str,
    *,
    diversification_ratio_fit: NDArray[np.float64] | None,
    diversification_gate_enabled: bool,
    concentration_recent_window_bars: int,
    concentration_floor: float | None,
) -> tuple[float, str]:
    if diversification_gate_enabled:
        if concentration_floor is None:
            raise ValueError("concentration_floor must be explicitly set when diversification_gate_enabled=True")
        if diversification_ratio_fit is not None:
            dr_arr = np.asarray(diversification_ratio_fit, dtype=np.float64)
            if dr_arr.size >= concentration_recent_window_bars:
                dr_fit_median = float(np.median(dr_arr))
                dr_recent = float(np.median(dr_arr[-concentration_recent_window_bars:]))
                if dr_fit_median > 1e-9:
                    concentration_ratio = float(np.clip(dr_recent / dr_fit_median, concentration_floor, 1.0))
                    if concentration_ratio < 1.0:
                        l_final = l_final * concentration_ratio
                        binding = "concentration_gate"
    return l_final, binding


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
    diversification_ratio_fit: NDArray[np.float64] | None = None,
    diversification_gate_enabled: bool = False,
    concentration_recent_window_bars: int = 60,
    concentration_floor: float | None = None,
    worst_fold_rets: NDArray[np.float64] | None = None,
    kelly_safety_fraction: float | None = None,
) -> tuple[float, str, float]:
    arr = np.asarray(fit_rets, dtype=np.float64)
    if arr.size < 2:
        _logger.debug("[L2-CALIB] fit_rets size<2, returning L*=1.0 (none)")
        return 1.0, "none", 0.0

    mdd_target = mdd_cap * (1.0 - mdd_margin)
    cvar_target = cvar_cap * (1.0 - cvar_margin)
    l_search_hi = l_hard_cap * 10.0

    if kelly_safety_fraction is not None and (kelly_safety_fraction <= 0.0 or kelly_safety_fraction > 1.0):
        raise ValueError(f"kelly_safety_fraction must be in (0, 1], got {kelly_safety_fraction}")

    # Stage 1: Safety Ceiling
    l_ceiling, ceiling_binding, l_hard_ceiling, _ = _resolve_safety_ceiling(
        arr,
        mdd_target=mdd_target,
        cvar_target=cvar_target,
        l_floor=float(l_floor),
        l_hard_cap=float(l_hard_cap),
        l_search_hi=l_search_hi,
        exchange_leverage_cap=exchange_leverage_cap,
        worst_fold_rets=worst_fold_rets,
        kelly_safety_fraction=kelly_safety_fraction,
    )

    # Stage 2: OOS Adaptive
    l_final, binding, cross_valid_mdd = _resolve_oos_adaptive_leverage(
        arr,
        oos_rets,
        l_ceiling=l_ceiling,
        l_hard_ceiling=l_hard_ceiling,
        ceiling_binding=ceiling_binding,
        mdd_cap=mdd_cap,
        mdd_margin=mdd_margin,
        oos_budget_blend=oos_budget_blend,
        oos_floor_cap=oos_floor_cap,
        fit_mdd_crisis_gate=fit_mdd_crisis_gate,
        l_floor=float(l_floor),
        l_search_hi=l_search_hi,
    )

    # Stage 3: Concentration Haircut
    l_final, binding = _apply_concentration_haircut(
        l_final,
        binding,
        diversification_ratio_fit=diversification_ratio_fit,
        diversification_gate_enabled=diversification_gate_enabled,
        concentration_recent_window_bars=concentration_recent_window_bars,
        concentration_floor=concentration_floor,
    )

    # Stage 4: Final invariant clip (defense-in-depth)
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
    fold_pass_ratio = sum(1 for value in nonempty_pass if value) / len(nonempty_pass) if nonempty_pass else 0.0
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
