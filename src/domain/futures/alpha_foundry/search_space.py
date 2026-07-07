"""Alpha Foundry search space construction and lifecycle.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

from src.domain.futures.alpha_foundry.contracts import (
    AlphaSignalBlueprint,
    L0SearchCell,
)

DEFAULT_ALPHA_TIMEFRAME_GRID: tuple[str, ...] = ("30m", "1h", "2h", "3h", "4h", "6h", "8h", "12h", "1d")

_TIMEFRAME_RE = re.compile(r"^(\d+)([mhd])$")


def timeframe_to_minutes(tf: str) -> int:
    m = _TIMEFRAME_RE.match(tf)
    if m is None:
        raise ValueError(f"unsupported timeframe: {tf!r}")
    value = int(m.group(1))
    unit = m.group(2)
    if value <= 0:
        raise ValueError("timeframe value must be positive")
    if unit == "m":
        return value
    elif unit == "h":
        return value * 60
    elif unit == "d":
        return value * 1440
    raise ValueError(f"unsupported timeframe: {tf!r}")


def resolve_alpha_timeframe_grid(
    *,
    enable_fast_timeframes: bool,
    include_daily: bool = True,
) -> tuple[str, ...]:
    if enable_fast_timeframes:
        base = list(DEFAULT_ALPHA_TIMEFRAME_GRID)
    else:
        base = [tf for tf in DEFAULT_ALPHA_TIMEFRAME_GRID if tf not in ("30m", "1h", "2h")]
    if not include_daily:
        base = [tf for tf in base if tf != "1d"]
    return tuple(base)


def make_alpha_blueprint_id(
    *,
    family: str,
    variant: str,
    timeframe: str,
    params: Mapping[str, float | int | str],
) -> str:
    raw = f"{family}:{variant}:{timeframe}:{sorted(params.items())}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{family}:{variant}:{timeframe}:{digest}"


def build_l0_search_cells(
    *,
    blueprints: Sequence[AlphaSignalBlueprint],
    family_prior_scores: Mapping[str, float],
    cost_floor_bps_by_tf: Mapping[str, float],
) -> tuple[L0SearchCell, ...]:
    cells: list[L0SearchCell] = []
    seen: set[str] = set()
    for bp in blueprints:
        cell_id = make_alpha_blueprint_id(
            family=bp.family,
            variant=bp.variant,
            timeframe=bp.timeframe,
            params={
                "lookback": ",".join(str(lb) for lb in bp.lookback_bars),
                "holding": bp.holding_bars,
            },
        )
        if cell_id in seen:
            continue
        seen.add(cell_id)
        tf_minutes = max(timeframe_to_minutes(bp.timeframe), 1)
        cost_floor = cost_floor_bps_by_tf.get(bp.timeframe, 0.0)
        prior = family_prior_scores.get(bp.family, 0.0)
        cells.append(
            L0SearchCell(
                blueprint_id=cell_id,
                family=bp.family,
                variant=bp.variant,
                timeframe=bp.timeframe,
                tf_minutes=tf_minutes,
                symbol_scope="global",
                cost_floor_bps=cost_floor,
                expected_event_rate=1.0 / max(bp.holding_bars, 1),
                family_prior_score=prior,
                status="pending",
            )
        )
    return tuple(cells)


def mark_retired_search_cells(
    *,
    cells: Sequence[L0SearchCell],
    failed_keys: set[tuple[str, str, str]],
) -> tuple[L0SearchCell, ...]:
    from dataclasses import replace

    updated: list[L0SearchCell] = []
    for cell in cells:
        key = (cell.family, cell.timeframe, cell.variant)
        if key in failed_keys:
            updated.append(replace(cell, status="retired"))
        else:
            updated.append(cell)
    return tuple(updated)
