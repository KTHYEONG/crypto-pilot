"""Alpha Foundry search space construction and lifecycle.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260707_ALPHA_FOUNDRY_RESULT_SYNC]
[ADR_20260707_ALPHA_FOUNDRY_CANONICAL_GATE_WIRING]
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace

from src.domain.futures.alpha_foundry.contracts import (
    AlphaFeatureBlueprint,
    AlphaFoundryRuntimeConfig,
    AlphaHypothesis,
    AlphaSignalBlueprint,
    CandidateFeatureFamily,
    L0SearchCell,
    L0SignalCandidate,
)

DEFAULT_ALPHA_TIMEFRAME_GRID: tuple[str, ...] = ("30m", "1h", "2h", "3h", "4h", "6h", "8h", "12h", "1d")

_TIMEFRAME_RE = re.compile(r"^(\d+)([mhd])$")

_BARS_PER_YEAR: dict[str, float] = {
    "30m": 17520.0,
    "1h": 8760.0,
    "2h": 4380.0,
    "3h": 2920.0,
    "4h": 2190.0,
    "6h": 1460.0,
    "8h": 1095.0,
    "12h": 730.0,
    "1d": 365.0,
}


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


def build_alpha_hypotheses(
    *,
    blueprints: Sequence[AlphaSignalBlueprint],
    family_prior_scores: Mapping[str, float],
    timeframe_cost_floor_bps: Mapping[str, float],
) -> tuple[AlphaHypothesis, ...]:
    hypotheses: list[AlphaHypothesis] = []
    seen: set[str] = set()
    for bp in blueprints:
        hid = make_alpha_blueprint_id(
            family=bp.family,
            variant=bp.variant,
            timeframe=bp.timeframe,
            params={"lookback": max(bp.lookback_bars) if bp.lookback_bars else 1, "holding": bp.holding_bars},
        )
        if hid in seen:
            continue
        seen.add(hid)
        prior = family_prior_scores.get(bp.family, 0.0)
        hypotheses.append(
            AlphaHypothesis(
                hypothesis_id=hid,
                family=bp.family,
                variant=bp.variant,
                archetype=bp.archetype,
                timeframe=bp.timeframe,  # type: ignore[arg-type]
                data_scope=("global",),
                entry_mode=bp.entry_mode,
                causal_lag_bars=bp.causal_lag_bars,
                holding_bars=bp.holding_bars,
                turnover_budget_per_year=bp.max_turnover_per_year,
                prior_score=prior,
            )
        )
    return tuple(hypotheses)


def build_feature_blueprints(
    *,
    hypotheses: Sequence[AlphaHypothesis],
    feature_family_by_family: Mapping[str, CandidateFeatureFamily],
    threshold_templates: Mapping[str, Mapping[str, float]],
    compute_cost_by_family: Mapping[str, float],
) -> tuple[AlphaFeatureBlueprint, ...]:
    blueprints: list[AlphaFeatureBlueprint] = []
    for hyp in hypotheses:
        ff = feature_family_by_family.get(hyp.family, "price_structure")
        thresholds = threshold_templates.get(hyp.family, {"z_entry": 1.5})
        cost = compute_cost_by_family.get(hyp.family, 1.0)
        blueprints.append(
            AlphaFeatureBlueprint(
                blueprint_id=f"fb_{hyp.hypothesis_id}",
                hypothesis_id=hyp.hypothesis_id,
                feature_family=ff,
                lookback_bars=(hyp.holding_bars,),
                thresholds=thresholds,
                direction_rule="trend_follow",
                required_fields=("close",),
                validity_mask_name="active",
                max_compute_cost_score=cost,
            )
        )
    return tuple(blueprints)


def apply_cost_prior_screen(
    *,
    cells: Sequence[L0SearchCell],
    runtime_config: AlphaFoundryRuntimeConfig,
) -> tuple[L0SearchCell, ...]:
    updated: list[L0SearchCell] = []
    for cell in cells:
        cost_floor = runtime_config.cost_prior_floor_by_tf.get(cell.timeframe, cell.cost_floor_bps)
        bpy = _BARS_PER_YEAR.get(cell.timeframe, 365.0)
        if cost_floor > bpy / max(cell.turnover_budget_per_year, 1.0):
            updated.append(replace(cell, status="retired", retire_reason="cost_prior_failed"))
        else:
            updated.append(cell)
    return tuple(updated)


def update_search_policy_state(
    *,
    cells: Sequence[L0SearchCell],
    candidates: Sequence[L0SignalCandidate],
    min_trials: int = 3,
) -> tuple[L0SearchCell, ...]:
    from collections import Counter

    candidate_failed: Counter[tuple[str, str, str]] = Counter()
    candidate_cost_drag: dict[tuple[str, str, str], list[float]] = {}

    for c in candidates:
        key = (c.family, c.timeframe, c.variant)
        if c.discovery_tier == "blocked":
            candidate_failed[key] += 1
        candidate_cost_drag.setdefault(key, []).append(c.cost_drag_ratio)

    updated: list[L0SearchCell] = []
    for cell in cells:
        key = (cell.family, cell.timeframe, cell.variant)
        failed_count = candidate_failed.get(key, 0)
        cost_drags = candidate_cost_drag.get(key, [])

        tested = cell.tested_count + len(candidate_cost_drag.get(key, []))
        survivor = cell.survivor_count + sum(
            1 for c in candidates if (c.family, c.timeframe, c.variant) == key and c.discovery_tier != "blocked"
        )

        retire_reason = cell.retire_reason

        if failed_count >= min_trials and len(cost_drags) > 0:
            median_drag = sorted(cost_drags)[len(cost_drags) // 2]
            if median_drag > 1.0:
                retire_reason = "repeated_hard_reject"

        updated.append(
            replace(
                cell,
                tested_count=tested,
                survivor_count=survivor,
                retire_reason=retire_reason,
                status="retired" if retire_reason is not None and retire_reason != cell.retire_reason else cell.status,
            )
        )
    return tuple(updated)


def build_l0_search_cells(
    *,
    blueprints: Sequence[AlphaSignalBlueprint],
    family_prior_scores: Mapping[str, float],
    cost_floor_bps_by_tf: Mapping[str, float],
    generator_exists_by_family: Mapping[str, bool] | None = None,
    feature_family_by_family: Mapping[str, CandidateFeatureFamily] | None = None,
    max_compute_cost_by_family: Mapping[str, float] | None = None,
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
        ff = (feature_family_by_family or {}).get(bp.family, "price_structure")
        max_cost = (max_compute_cost_by_family or {}).get(bp.family, 1.0)

        if generator_exists_by_family is not None:
            if bp.family not in generator_exists_by_family:
                raise ValueError(f"family {bp.family!r} missing from generator_exists_by_family")
            has_generator = generator_exists_by_family[bp.family]
            if not has_generator:
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
                        status="retired",
                        retire_reason="no_generator",
                        feature_family=ff,
                        turnover_budget_per_year=bp.max_turnover_per_year,
                        max_compute_cost_score=max_cost,
                    )
                )
                continue

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
                feature_family=ff,
                turnover_budget_per_year=bp.max_turnover_per_year,
                max_compute_cost_score=max_cost,
            )
        )
    return tuple(cells)


def mark_retired_search_cells(
    *,
    cells: Sequence[L0SearchCell],
    failed_keys: set[tuple[str, str, str]],
) -> tuple[L0SearchCell, ...]:
    updated: list[L0SearchCell] = []
    for cell in cells:
        key = (cell.family, cell.timeframe, cell.variant)
        if key in failed_keys:
            updated.append(replace(cell, status="retired"))
        else:
            updated.append(cell)
    return tuple(updated)
