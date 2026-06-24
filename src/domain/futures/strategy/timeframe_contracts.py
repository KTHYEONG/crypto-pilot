"""Shared timeframe contract helpers for futures strategy modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

HOURS_PER_BAR: dict[str, float] = {
    "1m": 1.0 / 60.0,
    "5m": 5.0 / 60.0,
    "15m": 0.25,
    "30m": 0.5,
    "1h": 1.0,
    "2h": 2.0,
    "4h": 4.0,
    "6h": 6.0,
    "8h": 8.0,
    "12h": 12.0,
    "1d": 24.0,
}

RESAMPLE_ALIAS: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
}

PROBE_SOURCE_TFS: tuple[str, ...] = ("1h", "4h")

_FLOAT_TOL = 1e-9

RESAMPLE_METADATA_BOOL_COLS: tuple[str, ...] = (
    "universe_active_mask",
    "universe_entry_warm_mask",
    "membership_kill_signal",
    "entry_block_mask",
)

RESAMPLE_METADATA_FLOAT_COLS: tuple[str, ...] = (
    "cluster_id",
    "beta_vs_market",
    "cluster_size",
    "anchor_cluster_member",
    "vol_30d",
    "friction_score",
    "alpha_capacity_score",
    "diversification_score",
    "tradeable_score",
)


def hours_per_bar(tf: str) -> float:
    """Return the canonical hours per bar for a timeframe."""
    return HOURS_PER_BAR.get(tf, 4.0)


def scale_bar_count(bars_base: int, tf: str, base_tf: str = "4h", *, minimum: int = 1) -> int:
    """Scale a bar count to preserve the same wall-clock horizon across timeframes."""
    hours_target = float(bars_base) * hours_per_bar(base_tf)
    scaled = round(hours_target / hours_per_bar(tf))
    return max(int(minimum), int(scaled))


def is_resample_compatible(source_tf: str, target_tf: str) -> bool:
    """Return True when source_tf can be aggregated cleanly into target_tf."""
    source_hpb = hours_per_bar(source_tf)
    target_hpb = hours_per_bar(target_tf)
    if source_hpb > target_hpb:
        return False
    ratio = target_hpb / source_hpb
    return abs(ratio - round(ratio)) <= _FLOAT_TOL


def resample_alias(tf: str) -> str:
    """Return the pandas resample alias for a timeframe."""
    return RESAMPLE_ALIAS.get(tf, tf)


def select_probe_source_tf(sym_maps: Mapping[str, Any], target_tf: str) -> str | None:
    """Select the finest cached source TF that can produce target_tf."""
    if target_tf in sym_maps:
        return target_tf

    compatible: list[str] = []
    for candidate_tf in PROBE_SOURCE_TFS:
        if candidate_tf not in sym_maps:
            continue
        if is_resample_compatible(candidate_tf, target_tf):
            compatible.append(candidate_tf)
    if not compatible:
        return None
    return min(compatible, key=hours_per_bar)

