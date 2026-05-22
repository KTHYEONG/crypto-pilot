from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Literal, cast

ActiveMode = Literal["quick-backtest", "strategy", "strategy-smoke"]
ActiveStrategyName = Literal["momentum_v0", "eh_st_v1"]
SyncMode = Literal["full_history_master", "elite_fast"]

_ACTIVE_MODES: frozenset[str] = frozenset({"quick-backtest", "strategy", "strategy-smoke"})
_LEGACY_MODES: frozenset[str] = frozenset({"full"})
_LEGACY_FLAGS: tuple[str, ...] = ("alpha_only", "hmm_only")


@dataclass(slots=True, frozen=True)
class FuturesRunConfig:
    """Active futures optimization runner configuration."""

    tf: str
    reference_date: str | None
    symbols: tuple[str, ...] | None
    trials: int
    mode: ActiveMode
    strategy: ActiveStrategyName | None
    sync_mode: SyncMode
    skip_universe: bool
    skip_data_sync: bool
    force_universe_rebuild: bool


def parse_active_mode(mode: str) -> ActiveMode:
    """Parse and validate active mode name."""
    if mode in _LEGACY_MODES:
        raise ValueError(f"legacy mode is not allowed in active runner: {mode}")
    if mode not in _ACTIVE_MODES:
        raise ValueError(f"invalid active mode: {mode}")
    return mode  # type: ignore[return-value]


def _parse_strategy_name(strategy: str | None) -> ActiveStrategyName | None:
    if strategy is None:
        return None
    if strategy not in ("momentum_v0", "eh_st_v1"):
        raise ValueError(f"unsupported strategy: {strategy}")
    return cast(ActiveStrategyName, strategy)


def _normalize_symbols(symbols: tuple[str, ...] | list[str] | None) -> tuple[str, ...] | None:
    if symbols is None:
        return None
    out: list[str] = []
    for sym in symbols:
        value = str(sym).strip()
        if value:
            out.append(value)
    return tuple(out) or None


def validate_run_config(config: FuturesRunConfig) -> FuturesRunConfig:
    """Validate cross-field contracts for active runner config."""
    if config.mode in {"strategy", "strategy-smoke"} and config.strategy is None:
        raise ValueError(f"{config.mode} mode requires strategy")
    if config.mode == "quick-backtest" and config.strategy is not None:
        raise ValueError("quick-backtest mode cannot set strategy")
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

    mode = parse_active_mode(str(raw.get("mode", "quick-backtest")))
    strategy = _parse_strategy_name(raw.get("strategy"))
    sync_mode_raw = str(raw.get("sync_mode", "full_history_master"))
    if sync_mode_raw not in {"full_history_master", "elite_fast"}:
        raise ValueError(f"invalid sync_mode: {sync_mode_raw}")
    config = FuturesRunConfig(
        tf=str(raw.get("tf", "4h")),
        reference_date=raw.get("reference_date"),
        symbols=_normalize_symbols(raw.get("symbols")),
        trials=int(raw.get("trials", 1)),
        mode=mode,
        strategy=strategy,
        sync_mode=sync_mode_raw,  # type: ignore[arg-type]
        skip_universe=bool(raw.get("skip_universe", False)),
        skip_data_sync=bool(raw.get("skip_data_sync", False)),
        force_universe_rebuild=bool(raw.get("force_universe_rebuild", False)),
    )
    return validate_run_config(config)
