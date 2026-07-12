"""Futures runner runtime config with Alpha Foundry mode.

[ADR_20260706_ALPHA_FOUNDRY_MAIN_WIRING][ADR_20260710_L0_TERMINAL_DEBUG_OBSERVABILITY]
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from src.domain.futures.alpha_foundry.contracts import AlphaFoundryRuntimeConfig


def _l0_cross_tf_diversity_audit_enabled() -> bool:
    """[ADR_20260711_L0_STRATEGY_DELIVERY_HARDENING] measurement-run opt-in env gate."""
    return os.environ.get("L0_CROSS_TF_DIVERSITY_AUDIT", "") not in ("", "0", "false", "False")


def _l0_cross_tf_pruning_enabled() -> bool:
    """[ADR_20260711_L0_CROSS_TF_PRUNING_ADMISSION] measurement-run opt-in env gate."""
    return os.environ.get("L0_CROSS_TF_PRUNING", "") not in ("", "0", "false", "False")


def _l0_parallel_max_workers() -> int:
    """[ADR_20260711_L0_L1_PIPELINE_LATENCY_PROFILING] measurement-run opt-in env gate."""
    raw = os.environ.get("L0_PARALLEL_MAX_WORKERS", "")
    return int(raw) if raw.strip() else 1


def _l0_ltf_pool_widened() -> bool:
    """[ADR_20260712_L0_EVIDENCE_CONDITIONED_CROSS_TF_ADMISSION] opt-in env gate."""
    return os.environ.get("L0_LTF_POOL_WIDENED", "") not in ("", "0", "false", "False")


ActivePhase = Literal["l0", "l1", "l2", "l3"]
SyncMode = Literal["auto", "skip"]

_ACTIVE_PHASES: frozenset[str] = frozenset({"l0", "l1", "l2", "l3"})
_REMOVED_PHASES: frozenset[str] = frozenset({"strategy-smoke", "quick-backtest"})
_REMOVED_ALPHA_FOUNDRY_ARG_KEYS: tuple[str, ...] = (
    "alpha_foundry",
    "alpha_foundry_total_l1_budget",
    "alpha_foundry_min_conviction_lcb_bps",
    "alpha_foundry_enable_fast_tf",
)
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
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC] Runtime config for futures runner orchestration."""

    timeframe: str
    date: str | None
    trials: int
    phase: ActivePhase
    sync: SyncMode
    refresh_universe: bool
    sync_metrics: bool
    seed: int = 42
    l0_runtime: AlphaFoundryRuntimeConfig = field(default_factory=AlphaFoundryRuntimeConfig)

    @property
    def alpha_foundry(self) -> AlphaFoundryRuntimeConfig:
        return self.l0_runtime


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


def build_alpha_foundry_runtime_config(
    args: Namespace | Mapping[str, Any],
) -> AlphaFoundryRuntimeConfig:
    if isinstance(args, Namespace):
        args = vars(args)
    alpha_foundry_mode = str(args.get("alpha_foundry", "off"))
    total_l1_budget = int(args.get("alpha_foundry_total_l1_budget", 30))
    min_conviction = float(args.get("alpha_foundry_min_conviction_lcb_bps", 5.0))
    enable_fast_tf = bool(args.get("alpha_foundry_enable_fast_tf", False))
    config = AlphaFoundryRuntimeConfig(
        mode=alpha_foundry_mode,  # type: ignore[arg-type]
        total_l1_verification_budget=max(1, total_l1_budget),
        min_conviction_lcb_bps=min_conviction,
        enable_fast_discovery_timeframes=enable_fast_tf,
        artifact_write_enabled=alpha_foundry_mode != "off",
    )
    return validate_alpha_foundry_runtime_config(config)


def validate_alpha_foundry_runtime_config(
    config: AlphaFoundryRuntimeConfig,
) -> AlphaFoundryRuntimeConfig:
    if config.mode not in {"off", "audit", "gate"}:
        raise ValueError(f"invalid alpha_foundry mode: {config.mode!r}")
    if config.max_recipes_per_family < 1:
        raise ValueError(f"max_recipes_per_family must be >= 1, got {config.max_recipes_per_family}")
    if config.top_k_per_family_tf < 1:
        raise ValueError(f"top_k_per_family_tf must be >= 1, got {config.top_k_per_family_tf}")
    if config.initial_fold_budget < 1:
        raise ValueError(f"initial_fold_budget must be >= 1, got {config.initial_fold_budget}")
    if config.total_l1_verification_budget < 1:
        raise ValueError(f"total_l1_verification_budget must be >= 1, got {config.total_l1_verification_budget}")
    if config.min_conviction_lcb_bps < 0.0:
        raise ValueError(f"min_conviction_lcb_bps must be >= 0.0, got {config.min_conviction_lcb_bps}")
    if config.enable_fast_discovery_timeframes:
        for tf in config.fast_discovery_timeframes:
            if not tf.endswith("h") or not tf[:-1].isdigit():
                raise ValueError(f"invalid fast discovery timeframe: {tf!r}")
    # Validate cheap_gate constraints
    cg = config.cheap_gate
    if cg.min_seed_slots_per_archetype < 1:
        raise ValueError(f"min_seed_slots_per_archetype must be >= 1, got {cg.min_seed_slots_per_archetype}")
    if cg.min_seed_slots_per_timeframe < 1:
        raise ValueError(f"min_seed_slots_per_timeframe must be >= 1, got {cg.min_seed_slots_per_timeframe}")
    for archetype, floor in cg.archetype_event_floors.items():
        if floor < 0:
            raise ValueError(f"negative event floor for archetype {archetype}: {floor}")
    for family, floor in cg.family_event_floors.items():
        if floor < 0:
            raise ValueError(f"negative event floor for family {family}: {floor}")
    return config


def build_l0_runtime_config(
    *,
    phase: ActivePhase,
    settings: Mapping[str, Any],
) -> AlphaFoundryRuntimeConfig:
    """Build internal L0 runtime config from phase + static settings, not CLI flags."""
    raw_budget = settings.get("alpha_foundry_total_l1_budget")
    total_l1_budget = int(raw_budget) if raw_budget is not None else 30
    raw_conviction = settings.get("alpha_foundry_min_conviction_lcb_bps")
    min_conviction = float(raw_conviction) if raw_conviction is not None else 5.0
    raw_fast_tf = settings.get("alpha_foundry_enable_fast_tf")
    enable_fast_tf = bool(raw_fast_tf) if raw_fast_tf is not None else False
    mode: Literal["audit", "gate"] = "gate"
    config = AlphaFoundryRuntimeConfig(
        mode=mode,
        total_l1_verification_budget=max(1, total_l1_budget),
        min_conviction_lcb_bps=min_conviction,
        enable_fast_discovery_timeframes=enable_fast_tf,
        artifact_write_enabled=False,
        observability_mode="debug_log",
        enable_cross_tf_diversity_audit=_l0_cross_tf_diversity_audit_enabled(),
        enable_cross_tf_pruning=_l0_cross_tf_pruning_enabled(),
        l0_parallel_max_workers=_l0_parallel_max_workers(),
        enable_ltf_family_pool_experiment=_l0_ltf_pool_widened(),
    )
    return validate_alpha_foundry_runtime_config(config)


def build_run_config_from_args(args: Namespace | Mapping[str, Any]) -> FuturesRunConfig:
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC] Build runner config from CLI args or mapping."""
    if isinstance(args, Namespace):
        args = vars(args)
    phase = parse_active_phase(args.get("phase", "l3"))
    for key in _REMOVED_ALPHA_FOUNDRY_ARG_KEYS:
        value = args.get(key)
        if value is not None and value is not False:
            raise ValueError(f"removed argument: --{key.replace('_', '-')}")
    for key in _REMOVED_ARG_KEYS:
        value = args.get(key)
        if value is not None and value is not False:
            raise ValueError(f"removed argument: --{key.replace('_', '-')}")
    sync = str(args.get("sync", "auto"))
    if sync not in {"auto", "skip"}:
        raise ValueError(f"invalid sync mode: {sync!r}, expected 'auto' or 'skip'")
    l0_runtime_cfg = build_l0_runtime_config(
        phase=phase,
        settings=args,
    )
    config = FuturesRunConfig(
        timeframe=str(args.get("timeframe", "4h")),
        date=args.get("date"),
        trials=int(args.get("trials", 42)),
        phase=phase,
        sync=sync,  # type: ignore[arg-type]
        refresh_universe=bool(args.get("refresh_universe", False)),
        sync_metrics=bool(args.get("sync_metrics", False)),
        seed=int(args.get("seed", 42)),
        l0_runtime=l0_runtime_cfg,
    )
    return validate_run_config(config)
