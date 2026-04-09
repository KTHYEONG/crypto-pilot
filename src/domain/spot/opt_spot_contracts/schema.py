"""Typed contracts for spot strategy archetype and deployment reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FeatureFamily(str, Enum):
    BREAKOUT = "Breakout"
    TREND_STRENGTH = "TrendStrength"
    VOLATILITY_EXPANSION = "VolatilityExpansion"
    EFFICIENCY = "Efficiency"
    RELATIVE_STRENGTH = "RelativeStrength"
    LIQUIDITY_STRESS = "LiquidityStress"


class RegimeLabel(str, Enum):
    TREND = "trend"
    CHOP = "chop"
    VOLATILITY_EXPANSION = "volatility_expansion"
    STRESS = "stress"


@dataclass(frozen=True)
class SpotArchetypeContract:
    """Versioned label for the implemented spot trend-following archetype."""

    archetype_id: str = "spot_v2_pullback_fractal_rsi2"
    signal_template: str = "RegimeAwareFractalPullbackRSI2EvR"
    execution_template: str = "VolTargetedAtrChandelierTrailScaleOut"
    schema_version: str = "2.0.0"


@dataclass
class DeployableUniverseReport:
    """Structured output after discovery + generalization gates."""

    archetype: SpotArchetypeContract = field(default_factory=SpotArchetypeContract)
    deployable_symbols: List[str] = field(default_factory=list)
    forbidden_regime_hints: List[str] = field(default_factory=list)
    risk_band: Dict[str, float] = field(default_factory=dict)
    loso_scores: Dict[str, float] = field(default_factory=dict)
    stress_symbol_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dsr: Optional[float] = None
    psr_paths: Optional[float] = None
    regime_coverage_ratio: Optional[float] = None
    stability_ok: Optional[bool] = None
    # Portfolio / deployment contract (shared-cash optimization)
    allocation_rule: str = "shared_cash_ranked_entries"
    ranking_rule: str = "slot_rank_score_cluster_weight"
    cash_state_rule: str = "idle_when_no_slot_or_no_signal"
    max_concurrent_positions: Optional[int] = None
    discovery_veto_passed: Optional[bool] = None
    mean_path_terminal_wealth_ratio: Optional[float] = None
    min_path_terminal_wealth_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archetype": {
                "archetype_id": self.archetype.archetype_id,
                "signal_template": self.archetype.signal_template,
                "execution_template": self.archetype.execution_template,
                "schema_version": self.archetype.schema_version,
            },
            "deployable_symbols": self.deployable_symbols,
            "forbidden_regime_hints": self.forbidden_regime_hints,
            "risk_band": self.risk_band,
            "loso_scores": self.loso_scores,
            "stress_symbol_metrics": self.stress_symbol_metrics,
            "dsr": self.dsr,
            "psr_paths": self.psr_paths,
            "regime_coverage_ratio": self.regime_coverage_ratio,
            "stability_ok": self.stability_ok,
            "allocation_rule": self.allocation_rule,
            "ranking_rule": self.ranking_rule,
            "cash_state_rule": self.cash_state_rule,
            "max_concurrent_positions": self.max_concurrent_positions,
            "discovery_veto_passed": self.discovery_veto_passed,
            "mean_path_terminal_wealth_ratio": self.mean_path_terminal_wealth_ratio,
            "min_path_terminal_wealth_ratio": self.min_path_terminal_wealth_ratio,
        }
