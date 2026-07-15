"""Active futures runner phase config with L0 runtime mapping. [ADR_20260710_L0_TERMINAL_DEBUG_OBSERVABILITY]"""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Any

from src.application.futures.run_contracts import ActivePhase
from src.application.futures.run_contracts import FuturesRunConfig as FuturesRunConfig
from src.application.futures.run_policy import build_effective_run_config

_ACTIVE_PHASES: frozenset[str] = frozenset({"l0", "l1", "l2", "l3"})
_LEGACY_PHASES: frozenset[str] = frozenset({"strategy-smoke", "quick-backtest"})
_LEGACY_FLAGS: tuple[str, ...] = (
    "alpha_only",
    "skip_universe",
    "skip_data_sync",
    "bypass_champion_guard",
    "seed",
    "resume",
)
_LEGACY_ARG_KEYS: tuple[str, ...] = (
    "tf",
    "reference_date",
    "rebuild_universe",
    "force_universe_rebuild",
    "sync_mode",
)


def parse_active_phase(phase: str) -> ActivePhase:
    """Parse and validate active phase name."""
    if phase in _LEGACY_PHASES:
        raise ValueError(f"legacy phase is not allowed in active runner: {phase}")
    if phase not in _ACTIVE_PHASES:
        raise ValueError(f"invalid active phase: {phase}")
    return phase  # type: ignore[return-value]


def validate_run_config(config: FuturesRunConfig) -> FuturesRunConfig:
    """Validate cross-field contracts for active runner config."""
    if config.trials < 1:
        raise ValueError("trials must be >= 1")
    return config


def build_run_config_from_args(args: Namespace | dict[str, Any]) -> FuturesRunConfig:
    """Build validated active run config via canonical RunPolicyFactory."""
    raw = vars(args) if isinstance(args, Namespace) else dict(args)

    for legacy_flag in _LEGACY_FLAGS:
        if bool(raw.get(legacy_flag, False)):
            raise ValueError(f"legacy flag is not allowed in active runner: {legacy_flag}")
    for legacy_key in _LEGACY_ARG_KEYS:
        if legacy_key in raw:
            raise ValueError(f"legacy argument key is not allowed in active runner: {legacy_key}")

    config = build_effective_run_config(raw, environ=os.environ)
    return validate_run_config(config)
