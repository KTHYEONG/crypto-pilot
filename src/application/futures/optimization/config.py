from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Literal

ActivePhase = Literal["l3", "l2", "l1"]
SyncMode = Literal["full", "fast", "skip"]

_ACTIVE_PHASES: frozenset[str] = frozenset({"l3", "l2", "l1"})
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


@dataclass(slots=True, frozen=True)
class FuturesRunConfig:
    """Active futures optimization runner configuration."""

    timeframe: str
    date: str | None
    trials: int
    phase: ActivePhase
    sync: SyncMode
    refresh_universe: bool


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
    """Build validated active run config from argparse namespace or mapping."""
    raw = vars(args) if isinstance(args, Namespace) else dict(args)

    for legacy_flag in _LEGACY_FLAGS:
        if bool(raw.get(legacy_flag, False)):
            raise ValueError(f"legacy flag is not allowed in active runner: {legacy_flag}")
    for legacy_key in _LEGACY_ARG_KEYS:
        if legacy_key in raw:
            raise ValueError(f"legacy argument key is not allowed in active runner: {legacy_key}")

    phase_raw = str(raw.get("phase", "l3"))
    phase = parse_active_phase(phase_raw)
    
    sync = str(raw.get("sync", "full"))
    if sync not in {"full", "fast", "skip"}:
        raise ValueError(f"invalid sync mode: {sync}")

    config = FuturesRunConfig(
        timeframe=str(raw.get("timeframe", "4h")),
        date=raw.get("date"),
        trials=int(raw.get("trials", 100)),
        phase=phase,
        sync=sync,  # type: ignore[arg-type]
        refresh_universe=bool(raw.get("refresh_universe", False)),
    )
    return validate_run_config(config)
