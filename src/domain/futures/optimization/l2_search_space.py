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
}
