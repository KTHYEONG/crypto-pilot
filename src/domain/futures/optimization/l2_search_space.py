from __future__ import annotations

from typing import Any

L2_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "K_RANK": {"type": "int", "low": 1, "high": 8, "step": 1},
    "REBALANCE_BARS": {"type": "categorical", "choices": (1, 2, 3, 6)},
    "CS_Z_SCORE_THRESHOLD": {"type": "float", "low": 0.0, "high": 1.2, "step": 0.1},
    "deploy_cost_safety_mult": {"type": "float", "low": 1.00, "high": 1.25, "step": 0.05},
    "edge_throttle_min_active_mult": {"type": "float", "low": 0.00, "high": 0.60, "step": 0.05},
    "edge_ref_bps": {"type": "float", "low": 2.0, "high": 12.0, "step": 0.5},
    "edge_throttle_gamma": {"type": "float", "low": 0.50, "high": 2.50, "step": 0.25},
    "risk_budget_floor_ratio": {"type": "float", "low": 0.00, "high": 1.00, "step": 0.05},
    "risk_budget_max_scale": {"type": "float", "low": 1.00, "high": 6.00, "step": 0.25},
    "l2_regime_long_short_asymmetry_enabled": {"type": "categorical", "choices": (False, True)},
    "l2_regime_bear_long_extra_mult": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
    "l2_regime_crisis_long_extra_mult": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
    "l2_regime_cap_release_cooldown_bars": {"type": "int", "low": 0, "high": 36, "step": 1},
    "l2_regime_crisis_gross_cap": {"type": "float", "low": 0.10, "high": 0.25, "step": 0.01},
    # [ADR_20260718_L2_DEPLOYED_SCALE_GROWTH_OBJECTIVE] 배치-스케일 성장 블렌드 가중치.
    # 200-trial 실측(scratch/spec_l2_growth_objective_full_validation.py): champion이
    # 0.8을 선택, 정상장 CAGR 게이트 + 위기 crisis MDD 제약 동시 통과 최초 확인.
    "l2_objective_growth_lcb_weight": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
    # [ADR_20260718_L2_REGIME_SEVERITY_SIGNAL_REDESIGN] 방향-변동성 분리 cap-gating.
    "l2_regime_severity_gating_enabled": {"type": "categorical", "choices": (False, True)},
    # [SPEC_L2_DEPLOYMENT_MARGIN_CAGR_GATE] 정상장 leverage 캘리브레이션 안전마진.
    # crisis 예산(l2_deploy_crisis_mdd_margin)은 별도 고정 필드로 분리되어 있어 탐색 영향 없음.
    "l2_deploy_mdd_margin": {"type": "float", "low": 0.05, "high": 0.30, "step": 0.05},
}
