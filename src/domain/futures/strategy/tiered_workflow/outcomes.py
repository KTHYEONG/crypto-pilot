from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.domain.futures.strategy.config import PerTfL1Result

TieredRunStatus = Literal["completed", "failed"]


@dataclass(frozen=True, slots=True)
class TieredRunFailure:
    """[ADR_20260715_L0_L1_RUNTIME_TERMINAL_OBSERVABILITY] Explicit non-measurement failure."""

    code: Literal["native_event_contract", "missing_native_frame", "runtime_policy", "unexpected"]
    timeframe: str | None
    message: str


@dataclass(frozen=True, slots=True)
class TieredRunOutcome:
    """[ADR_20260715_L0_L1_RUNTIME_TERMINAL_OBSERVABILITY] Completed or failed tiered outcome."""

    status: TieredRunStatus
    l1_result: object | None
    l2_result: object | None
    l3_result: object | None
    per_tf_l1: tuple[PerTfL1Result, ...]
    failure: TieredRunFailure | None
    policy_fingerprint: str
    diagnostic_complete: bool
