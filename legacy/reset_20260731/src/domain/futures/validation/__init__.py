"""Validation helpers for futures optimization refactor."""

from .gates import (
    AtomicBlockConfig,
    AtomicBlockResult,
    ChampionGateConfig,
    FuturesResearchGateInput,
    GateResult,
    ModulePurgeBarsMeta,
    PurgeBarsRegistry,
    build_atomic_blocks,
    evaluate_atomic_blocks,
    evaluate_champion_gates,
    evaluate_research_gates,
)

__all__ = [
    "AtomicBlockConfig",
    "AtomicBlockResult",
    "ChampionGateConfig",
    "FuturesResearchGateInput",
    "GateResult",
    "ModulePurgeBarsMeta",
    "PurgeBarsRegistry",
    "build_atomic_blocks",
    "evaluate_atomic_blocks",
    "evaluate_champion_gates",
    "evaluate_research_gates",
]
