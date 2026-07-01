from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ReversalRiskConfig:
    enabled: bool = True
    dd_threshold: float = 0.12
    persistence_bars: int = 3
    recovery_cooldown_bars: int = 8


def default_reversal_risk_config() -> ReversalRiskConfig:
    return ReversalRiskConfig()
