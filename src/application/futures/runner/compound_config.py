from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.application.futures.runner.models import RunnerResult
from src.domain.futures.data_lake.contracts import DataLakeConfig
from src.domain.futures.universe.config import PITUniverseConfig


@dataclass(slots=True, frozen=True)
class UniverseConfig:
    max_symbols: int = 60
    min_listing_days: int = 90
    min_volume_usdt: float = 20_000_000.0


@dataclass(slots=True, frozen=True)
class CompoundRunConfig:
    reference_date: str | None
    sync: Literal["auto", "skip"]
    refresh_universe: bool
    seed: int = 42
    base_timeframe: Literal["1h"] = "1h"
    max_rss_mb: int = 12_000
    history_days: int = 730
    portfolio_nav_usdt: float = 100_000.0
    max_daily_symbols: int = 45
    max_axis_symbols: int = 240
    data_lake: DataLakeConfig = field(default_factory=lambda: DataLakeConfig(root=Path("data/futures/lake")))
    universe: PITUniverseConfig = field(default_factory=PITUniverseConfig)
    allow_network_sync: bool = False


@dataclass(slots=True, frozen=True)
class CompoundRunArtifacts:
    result_path: str
    target_weights_path: str
    manifest_path: str


def _to_mapping(args: object) -> Mapping[str, Any]:
    if isinstance(args, Mapping):
        return args
    if hasattr(args, "__dict__"):
        return vars(args)
    raise ValueError(f"cannot convert {type(args).__name__} to mapping")


def build_compound_run_config(args: object) -> CompoundRunConfig:
    args_dict = _to_mapping(args)
    reference_date: str | None = args_dict.get("date")
    sync_raw = args_dict.get("sync", "auto")
    sync: Literal["auto", "skip"]
    if sync_raw == "auto":
        sync = "auto"
    elif sync_raw == "skip":
        sync = "skip"
    else:
        raise ValueError(f"invalid sync mode: {sync_raw!r}, expected 'auto' or 'skip'")
    refresh_universe = bool(args_dict.get("refresh_universe", False))
    seed = int(args_dict.get("seed", 42))
    max_rss_mb = int(args_dict.get("max_rss_mb", 12_000))
    history_days = int(args_dict.get("history_days", 730))
    portfolio_nav_usdt = float(args_dict.get("portfolio_nav_usdt", 100_000.0))
    max_daily_symbols = int(args_dict.get("max_daily_symbols", 120))
    max_axis_symbols = int(args_dict.get("max_axis_symbols", 240))
    allow_network_sync = bool(args_dict.get("allow_network_sync", False))
    if max_rss_mb <= 0:
        raise ValueError(f"max_rss_mb must be positive, got {max_rss_mb}")
    if not (0 <= seed <= 2**31 - 1):
        raise ValueError(f"seed out of valid range: {seed}")
    if history_days < 1:
        raise ValueError(f"history_days must be >= 1, got {history_days}")
    if portfolio_nav_usdt <= 0:
        raise ValueError(f"portfolio_nav_usdt must be positive, got {portfolio_nav_usdt}")
    if max_daily_symbols < 1:
        raise ValueError(f"max_daily_symbols must be >= 1, got {max_daily_symbols}")
    if max_axis_symbols < max_daily_symbols:
        raise ValueError(f"max_axis_symbols {max_axis_symbols} < max_daily_symbols {max_daily_symbols}")
    return CompoundRunConfig(
        reference_date=reference_date,
        sync=sync,
        refresh_universe=refresh_universe,
        seed=seed,
        max_rss_mb=max_rss_mb,
        history_days=history_days,
        portfolio_nav_usdt=portfolio_nav_usdt,
        max_daily_symbols=max_daily_symbols,
        max_axis_symbols=max_axis_symbols,
        allow_network_sync=allow_network_sync,
    )


__all__ = [
    "CompoundRunArtifacts",
    "CompoundRunConfig",
    "RunnerResult",
    "UniverseConfig",
    "build_compound_run_config",
]
