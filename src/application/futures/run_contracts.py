from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.domain.futures.alpha_foundry.contracts import AlphaFoundryRuntimeConfig

ActivePhase = Literal["l0", "l1", "l2", "l3"]
SyncMode = Literal["auto", "skip"]


@dataclass(slots=True, frozen=True)
class FuturesRunConfig:
    """[ADR_20260715_L0_L1_NATIVE_CONTRACT] Canonical phase/run contract."""
    timeframe: str
    date: str | None
    trials: int
    phase: ActivePhase
    sync: SyncMode
    refresh_universe: bool
    sync_metrics: bool
    seed: int = 42
    l0_runtime: AlphaFoundryRuntimeConfig = field(default_factory=AlphaFoundryRuntimeConfig)

    @property
    def alpha_foundry(self) -> AlphaFoundryRuntimeConfig:
        return self.l0_runtime
