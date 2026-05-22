"""Validation helpers for futures optimization refactor."""

from .gates import (
    AtomicBlockConfig,
    AtomicBlockResult,
    FuturesResearchGateInput,
    GateResult,
    ModulePurgeBarsMeta,
    PurgeBarsRegistry,
    V3HardGates,
    build_atomic_blocks,
    evaluate_atomic_blocks,
    evaluate_research_gates,
    evaluate_v3_hard_gates,
)

__all__ = [
    "AtomicBlockConfig",
    "AtomicBlockResult",
    "FuturesResearchGateInput",
    "GateResult",
    "ModulePurgeBarsMeta",
    "PurgeBarsRegistry",
    "V3HardGates",
    "build_atomic_blocks",
    "evaluate_atomic_blocks",
    "evaluate_research_gates",
    "evaluate_v3_hard_gates",
]
