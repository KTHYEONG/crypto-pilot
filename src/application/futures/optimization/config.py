from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Literal

ActiveMode = Literal["quick-backtest", "strategy", "alpha"]
SyncMode = Literal["full_history_master", "elite_fast"]

_ACTIVE_MODES: frozenset[str] = frozenset({"quick-backtest", "strategy", "alpha"})
_LEGACY_MODES: frozenset[str] = frozenset({"full", "strategy-smoke"})
_LEGACY_FLAGS: tuple[str, ...] = (
    "alpha_only",
    "skip_universe",
    "skip_data_sync",
    "bypass_champion_guard",
)


@dataclass(slots=True, frozen=True)
class FuturesRunConfig:
    """Active futures optimization runner configuration."""

    timeframe: str
    reference_date: str | None
    trials: int
    mode: ActiveMode
    sync_mode: SyncMode
    force_universe_rebuild: bool


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
    raw: dict[str, Any]
    if isinstance(args, Namespace):
        raw = vars(args)
    else:
        raw = dict(args)

    for legacy_flag in _LEGACY_FLAGS:
        if bool(raw.get(legacy_flag, False)):
            raise ValueError(f"legacy flag is not allowed in active runner: {legacy_flag}")

    mode = parse_active_mode(str(raw.get("mode", "strategy")))
    sync_mode_raw = str(raw.get("sync_mode", "full_history_master"))
    if sync_mode_raw not in {"full_history_master", "elite_fast"}:
        raise ValueError(f"invalid sync_mode: {sync_mode_raw}")
    config = FuturesRunConfig(
        timeframe=str(raw.get("timeframe", raw.get("tf", "4h"))),
        reference_date=raw.get("reference_date"),
        trials=int(raw.get("trials", 100)),
        mode=mode,
        sync_mode=sync_mode_raw,  # type: ignore[arg-type]
        force_universe_rebuild=bool(raw.get("force_universe_rebuild", False)),
    )
    return validate_run_config(config)

