"""Futures runner runtime config with Alpha Foundry mode.

[ADR_20260706_ALPHA_FOUNDRY_MAIN_WIRING][ADR_20260710_L0_TERMINAL_DEBUG_OBSERVABILITY]
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Mapping
from typing import Any

from src.application.futures.run_contracts import ActivePhase
from src.application.futures.run_contracts import FuturesRunConfig as FuturesRunConfig
from src.application.futures.run_policy import build_effective_run_config

_ACTIVE_PHASES: frozenset[str] = frozenset({"l0", "l1", "l2", "l3"})
_REMOVED_PHASES: frozenset[str] = frozenset({"strategy-smoke", "quick-backtest"})
_REMOVED_ALPHA_FOUNDRY_ARG_KEYS: tuple[str, ...] = (
    "alpha_foundry",
    "alpha_foundry_total_l1_budget",
    "alpha_foundry_min_conviction_lcb_bps",
    "alpha_foundry_enable_fast_tf",
)
_REMOVED_ARG_KEYS: tuple[str, ...] = (
    "alpha_only",
    "skip_universe",
    "skip_data_sync",
    "bypass_champion_guard",
    "symbols",
    "quick_backtest",
    "mode",
    "tf",
    "reference_date",
)


# FuturesRunConfig is now canonical at src.application.futures.run_contracts


def parse_active_phase(phase: str) -> ActivePhase:
    if phase in _REMOVED_PHASES:
        raise ValueError(f"removed phase: {phase}")
    if phase not in _ACTIVE_PHASES:
        raise ValueError(f"unknown phase: {phase}, expected one of {sorted(_ACTIVE_PHASES)}")
    return phase  # type: ignore[return-value]


def validate_run_config(config: FuturesRunConfig) -> FuturesRunConfig:
    if config.trials < 1:
        raise ValueError(f"trials must be >= 1, got {config.trials}")
    return config


def build_run_config_from_args(args: Namespace | Mapping[str, Any]) -> FuturesRunConfig:
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC] Build runner config via canonical RunPolicyFactory."""
    if isinstance(args, Namespace):
        args = vars(args)
    parse_active_phase(args.get("phase", "l3"))
    for key in _REMOVED_ALPHA_FOUNDRY_ARG_KEYS:
        value = args.get(key)
        if value is not None and value is not False:
            raise ValueError(f"removed argument: --{key.replace('_', '-')}")
    for key in _REMOVED_ARG_KEYS:
        value = args.get(key)
        if value is not None and value is not False:
            raise ValueError(f"removed argument: --{key.replace('_', '-')}")
    sync = str(args.get("sync", "auto"))
    if sync not in {"auto", "skip"}:
        raise ValueError(f"invalid sync mode: {sync!r}, expected 'auto' or 'skip'")

    config = build_effective_run_config(args, environ=os.environ)
    return validate_run_config(config)
