from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.domain.futures.alpha_foundry.contracts import AlphaFoundryRuntimeConfig

ActivePhase = Literal["l0", "l1", "l2", "l3"]
SyncMode = Literal["auto", "skip"]


class RunPolicyError(ValueError):
    """Raised when effective runner policy is invalid or ambiguous."""


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    heavy_process_workers: Literal[1] = 1
    ltf_io_workers: Literal[1, 2] = 1
    max_rss_mb: int = 12_000


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
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    policy_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.phase in {"l0", "l1"} and self.l0_runtime.mode != "gate":
            raise RunPolicyError(
                f"phase={self.phase} requires l0_runtime.mode='gate', got {self.l0_runtime.mode!r}"
            )

    @property
    def alpha_foundry(self) -> AlphaFoundryRuntimeConfig:
        return self.l0_runtime
