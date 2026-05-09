from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PortfolioPolicyConfig:
    target_ann_vol: float
    gross_exposure_cap: float
    per_symbol_cap: float
    top_k_long: int
    top_k_short: int
    entry_edge_threshold: float
    rebalance_bars: int
    min_long_pf: float
    min_short_pf: float
    min_is_net_alpha_pct: float


def load_portfolio_policy_config(cfg: dict[str, Any]) -> PortfolioPolicyConfig:
    block = cfg.get("FUTURES_PORTFOLIO_POLICY", {})
    return PortfolioPolicyConfig(
        target_ann_vol=float(block.get("target_ann_vol", 0.45)),
        gross_exposure_cap=float(block.get("gross_exposure_cap", 1.2)),
        per_symbol_cap=float(block.get("per_symbol_cap", 0.25)),
        top_k_long=int(block.get("top_k_long", 3)),
        top_k_short=int(block.get("top_k_short", 3)),
        entry_edge_threshold=float(block.get("entry_edge_threshold", 0.15)),
        rebalance_bars=int(block.get("rebalance_bars", 3)),
        min_long_pf=float(block.get("min_long_pf", 1.05)),
        min_short_pf=float(block.get("min_short_pf", 1.05)),
        min_is_net_alpha_pct=float(block.get("min_is_net_alpha_pct", 0.0)),
    )


def apply_hmm_regime_exposure(base_exposure: float, hmm_state: int, crisis_prob: float) -> float:
    """Scale base exposure based on HMM regime and crisis probability.
    
    Returns:
        0.0 if crisis_prob > 0.5 or hmm_state == 4 (CRISIS)
        0.5 * base_exposure if hmm_state == 2 (BEAR)
        0.7 * base_exposure if hmm_state == 3 (CHOP)
        base_exposure otherwise (BULL_CALM=0, BULL_VOL_UP=1)
    """
    if crisis_prob > 0.5 or int(hmm_state) == 4:
        return 0.0
    if int(hmm_state) == 2:
        return 0.5 * base_exposure
    if int(hmm_state) == 3:
        return 0.7 * base_exposure
    return base_exposure


def apply_policy_constraints(params: dict[str, Any], policy: PortfolioPolicyConfig) -> dict[str, Any]:
    out = dict(params)
    out["K_LONG"] = int(max(1, min(int(out.get("K_LONG", policy.top_k_long)), policy.top_k_long)))
    out["K_SHORT"] = int(max(1, min(int(out.get("K_SHORT", policy.top_k_short)), policy.top_k_short)))
    out["REBALANCE_BARS"] = int(max(1, out.get("REBALANCE_BARS", policy.rebalance_bars)))
    out["MAX_EXPOSURE_PER_COIN"] = float(
        min(max(float(out.get("MAX_EXPOSURE_PER_COIN", policy.per_symbol_cap)), 0.05), policy.per_symbol_cap)
    )
    out["MAX_EXPOSURE"] = float(
        min(max(float(out.get("MAX_EXPOSURE", policy.gross_exposure_cap)), 0.20), policy.gross_exposure_cap)
    )
    out["TARGET_ANN_VOL"] = float(
        min(max(float(out.get("TARGET_ANN_VOL", policy.target_ann_vol)), 0.05), policy.target_ann_vol)
    )
    return out

