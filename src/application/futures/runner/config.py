from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

ActivePhase = Literal["l1", "l2", "l3"]
SyncMode = Literal["auto", "skip"]

_ACTIVE_PHASES: frozenset[str] = frozenset({"l1", "l2", "l3"})
_REMOVED_PHASES: frozenset[str] = frozenset({"strategy-smoke", "quick-backtest"})
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


@dataclass(slots=True, frozen=True)
class FuturesRunConfig:
    timeframe: str
    date: str | None
    trials: int
    phase: ActivePhase
    sync: SyncMode
    refresh_universe: bool
    sync_metrics: bool


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
    if isinstance(args, Namespace):
        args = vars(args)
    phase = parse_active_phase(args.get("phase", "l3"))
    for key in _REMOVED_ARG_KEYS:
        value = args.get(key)
        if value is not None and value is not False:
            raise ValueError(f"removed argument: --{key.replace('_', '-')}")
    sync = str(args.get("sync", "auto"))
    if sync not in {"auto", "skip"}:
        raise ValueError(f"invalid sync mode: {sync!r}, expected 'auto' or 'skip'")
    config = FuturesRunConfig(
        timeframe=str(args.get("timeframe", "4h")),
        date=args.get("date"),
        trials=int(args.get("trials", 42)),
        phase=phase,
        sync=sync,  # type: ignore[arg-type]
        refresh_universe=bool(args.get("refresh_universe", False)),
        sync_metrics=bool(args.get("sync_metrics", False)),
    )
    return validate_run_config(config)
