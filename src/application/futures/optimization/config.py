from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Literal

ActiveMode = Literal["quick-backtest", "strategy", "alpha"]
SyncMode = Literal["full", "fast", "skip"]

_ACTIVE_MODES: frozenset[str] = frozenset({"quick-backtest", "strategy", "alpha"})
_LEGACY_MODES: frozenset[str] = frozenset({"full", "strategy-smoke"})
_LEGACY_FLAGS: tuple[str, ...] = (
    "alpha_only",
    "skip_universe",
    "skip_data_sync",
    "bypass_champion_guard",
    "seed",
    "resume",
)


@dataclass(slots=True, frozen=True)
class FuturesRunConfig:
    """Active futures optimization runner configuration."""

    timeframe: str
    reference_date: str | None
    trials: int
    mode: ActiveMode
    sync: SyncMode
    rebuild_universe: bool


def parse_active_mode(mode: str) -> ActiveMode:
    """Parse and validate active mode name."""
    if mode in _LEGACY_MODES:
        raise ValueError(f"legacy mode is not allowed in active runner: {mode}")
    if mode not in _ACTIVE_MODES:
        raise ValueError(f"invalid active mode: {mode}")
    return mode  # type: ignore[return-value]


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

    mode = parse_active_mode(str(raw.get("mode", "strategy")))
    
    # Map simplified sync values or handle legacy if passed through dict
    sync_raw = str(raw.get("sync", raw.get("sync_mode", "full")))
    sync_map = {
        "full_history_master": "full",
        "elite_fast": "fast",
        "full": "full",
        "fast": "fast",
        "skip": "skip",
    }
    sync = sync_map.get(sync_raw, "full")

    config = FuturesRunConfig(
        timeframe=str(raw.get("timeframe", raw.get("tf", "4h"))),
        reference_date=raw.get("reference_date"),
        trials=int(raw.get("trials", 100)),
        mode=mode,
        sync=sync,  # type: ignore[arg-type]
        rebuild_universe=bool(raw.get("rebuild_universe", raw.get("force_universe_rebuild", False))),
    )
    return validate_run_config(config)

