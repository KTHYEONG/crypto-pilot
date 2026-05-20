from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DualDecayConfig:
    """Configuration for Dual Decay evaluation.

    Attributes:
        percent_decay_floor: Floor threshold for percent decay.
        absolute_decay_floor_bps: Floor threshold for absolute decay in basis points.

    """

    percent_decay_floor: float = -0.15          # -15% (coarse_CAGR > 0일 때만)
    absolute_decay_floor_bps: float = -500.0    # -500bps (항상 적용)


@dataclass
class DualDecayResult:
    """Result of Dual Decay evaluation.

    Attributes:
        passed: Whether the strategy passed the decay checks.
        percent_decay: The computed percent decay or None.
        absolute_decay_bps: The computed absolute decay in basis points.
        failures: List of failure reasons.

    """

    passed: bool
    percent_decay: float | None                 # coarse_CAGR <= 0 이면 None
    absolute_decay_bps: float
    failures: list[str]


def evaluate_dual_decay(
    intrabar_cagr: float,
    coarse_cagr: float,
    cfg: DualDecayConfig | None = None,
) -> DualDecayResult:
    """Evaluate dual decay of candidate strategy against coarse CAGR.

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if cfg is None:
        cfg = DualDecayConfig()

    failures: list[str] = []
    
    # 1. Percent Decay 계산 (coarse_cagr > 0 일 때만)
    percent_decay: float | None = None
    if coarse_cagr > 0.0:
        percent_decay = (intrabar_cagr - coarse_cagr) / coarse_cagr
        if percent_decay < cfg.percent_decay_floor:
            failures.append("DUAL_DECAY_PERCENT")
            
    # 2. Absolute Decay 계산 (bps)
    absolute_decay_bps = (intrabar_cagr - coarse_cagr) * 10000.0
    if absolute_decay_bps < cfg.absolute_decay_floor_bps:
        failures.append("DUAL_DECAY_ABSOLUTE")
        
    passed = len(failures) == 0
    return DualDecayResult(
        passed=passed,
        percent_decay=percent_decay,
        absolute_decay_bps=absolute_decay_bps,
        failures=failures,
    )


# Drawdown Tiers & Scale
DD_TIER_1_LOSS: float = -0.10     # rolling 30d loss > 10%
DD_TIER_2_LOSS: float = -0.15     # rolling 30d loss > 15%
DD_RECOVERY_LOSS: float = -0.05   # recovery threshold: rolling 30d loss < 5%
DD_TIER_1_SCALE: float = 0.70
DD_TIER_2_SCALE: float = 0.40


def compute_drawdown_gross_scale(
    rolling_30d_return: float,
    current_scale: float = 1.0,
) -> float:
    """Compute gross scaling factor based on rolling 30-day drawdown tiers.

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    # rolling_30d_return 은 0.05 (= +5%) 또는 -0.12 (= -12%) 형식의 float 임
    if rolling_30d_return < DD_TIER_2_LOSS:
        return DD_TIER_2_SCALE
    elif rolling_30d_return < DD_TIER_1_LOSS:
        # TIER 1 혹은 2에 있던 상태에서 TIER 2 미만은 아니지만 TIER 1 이하인 경우
        # 현재 스케일이 TIER 2(0.40)보다 작거나 같으면 0.40 유지
        if current_scale <= DD_TIER_2_SCALE:
            return DD_TIER_2_SCALE
        return DD_TIER_1_SCALE
    elif rolling_30d_return > DD_RECOVERY_LOSS:
        # 복귀 조건 충족 시 단계적 복귀
        if current_scale == DD_TIER_2_SCALE:
            return DD_TIER_1_SCALE
        elif current_scale == DD_TIER_1_SCALE:
            return 1.0
        return 1.0
    else:
        # -10% ~ -5% 사이 구간: 기존 스케일 유지
        return current_scale


NO_TRADE_THRESHOLD_BPS: float = 2.0


def apply_no_trade_buffer(
    target_weights: np.ndarray,
    current_weights: np.ndarray,
    cost_bps_per_symbol: np.ndarray,
    threshold_multiplier: float = 2.0,
) -> np.ndarray:
    """Apply a no-trade buffer to omit trivial adjustments.

    If change in weight is less than threshold_multiplier * cost_bps, keep current weight.

    Time Complexity: O(N) where N is the number of symbols.
    Space Complexity: O(N) for returned target weight array.
    """
    tw = np.asarray(target_weights, dtype=np.float64).copy()
    cw = np.asarray(current_weights, dtype=np.float64)
    cost = np.asarray(cost_bps_per_symbol, dtype=np.float64)
    
    # delta_w(i) in bps = abs(target - current) * 10000.0
    delta_w_bps = np.abs(tw - cw) * 10000.0
    threshold = threshold_multiplier * cost
    
    # threshold_multiplier 가 0 이면 버퍼가 비활성화됨
    if threshold_multiplier <= 0.0:
        return tw
        
    mask = delta_w_bps < threshold
    tw[mask] = cw[mask]
    return tw
