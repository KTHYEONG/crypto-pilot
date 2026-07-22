from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from src.application.futures.runner.models import RunnerResult


@dataclass(slots=True, frozen=True)
class CompoundRunConfig:
    reference_date: str | None
    sync: Literal["auto", "skip"]
    refresh_universe: bool
    seed: int = 42
    base_timeframe: Literal["1h"] = "1h"
    max_rss_mb: int = 12_000


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
    if max_rss_mb <= 0:
        raise ValueError(f"max_rss_mb must be positive, got {max_rss_mb}")
    if not (0 <= seed <= 2**31 - 1):
        raise ValueError(f"seed out of valid range: {seed}")
    return CompoundRunConfig(
        reference_date=reference_date,
        sync=sync,
        refresh_universe=refresh_universe,
        seed=seed,
        max_rss_mb=max_rss_mb,
    )


__all__ = [
    "CompoundRunArtifacts",
    "CompoundRunConfig",
    "RunnerResult",
    "build_compound_run_config",
]
