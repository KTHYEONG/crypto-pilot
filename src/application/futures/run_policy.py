from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from src.application.futures.run_contracts import (
    ExecutionPolicy,
    FuturesRunConfig,
    RunPolicyError,
)
from src.domain.futures.alpha_foundry.contracts import AlphaFoundryRuntimeConfig

_ALLOWED_ENV_OVERRIDES: Mapping[str, type] = {
    "L0_LTF_EXEC_1M_MAX_WORKERS": int,
    "L0_PARALLEL_MAX_WORKERS": int,
    "L0_CROSS_TF_DIVERSITY_AUDIT": bool,
    "L0_CROSS_TF_PRUNING": bool,
    "L0_LTF_POOL_WIDENED": bool,
}

_BOOL_TRUE_VALUES = frozenset({"1", "true", "yes"})


def _parse_env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, typ in _ALLOWED_ENV_OVERRIDES.items():
        raw = environ.get(key)
        if raw is None or raw == "":
            continue
        try:
            if typ is bool:
                overrides[key] = raw.strip().lower() in _BOOL_TRUE_VALUES
            else:
                overrides[key] = typ(raw.strip())
        except (ValueError, TypeError) as exc:
            raise RunPolicyError(f"invalid value for {key}={raw!r}, expected {typ.__name__}") from exc
    return overrides


def _build_policy_fingerprint(
    args: Mapping[str, Any],
    env_overrides: Mapping[str, Any],
    effective_runtime: AlphaFoundryRuntimeConfig,
    execution_policy: ExecutionPolicy,
) -> str:
    canonical: dict[str, Any] = {
        "phase": str(args.get("phase", "l1")),
        "timeframe": str(args.get("timeframe", "4h")),
        "trials": int(args.get("trials", 100)),
        "sync": str(args.get("sync", "auto")),
        "refresh_universe": bool(args.get("refresh_universe", False)),
        "l0_runtime_mode": effective_runtime.mode,
        "l0_runtime_budget": effective_runtime.total_l1_verification_budget,
        "l0_runtime_min_conviction": effective_runtime.min_conviction_lcb_bps,
        "l0_runtime_enable_fast_tf": effective_runtime.enable_fast_discovery_timeframes,
        "execution_policy": {
            "heavy_process_workers": execution_policy.heavy_process_workers,
            "ltf_io_workers": execution_policy.ltf_io_workers,
            "max_rss_mb": execution_policy.max_rss_mb,
        },
        "env_overrides": dict(sorted(env_overrides.items())),
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def build_effective_run_config(
    args: Mapping[str, Any],
    *,
    environ: Mapping[str, str],
) -> FuturesRunConfig:
    """[ADR_20260715_L0_L1_RUNTIME_TERMINAL_OBSERVABILITY] Construct the effective active-run policy."""
    env_overrides = _parse_env_overrides(environ)

    phase_raw = str(args.get("phase", "l1"))
    if phase_raw not in {"l0", "l1", "l2", "l3"}:
        raise RunPolicyError(f"invalid phase: {phase_raw!r}")

    sync = str(args.get("sync", "auto"))
    if sync not in {"auto", "skip"}:
        raise RunPolicyError(f"invalid sync mode: {sync!r}")

    lt_f_workers = env_overrides.get("L0_LTF_EXEC_1M_MAX_WORKERS", 1)
    if lt_f_workers not in (1, 2):
        raise RunPolicyError(f"L0_LTF_EXEC_1M_MAX_WORKERS must be 1 or 2, got {lt_f_workers}")

    execution_policy = ExecutionPolicy(
        heavy_process_workers=1,
        ltf_io_workers=lt_f_workers,
        max_rss_mb=int(args.get("max_rss_mb", 12_000)),
    )

    raw_mode = args.get("alpha_foundry")
    mode_raw = "off" if raw_mode is None else str(raw_mode)
    if mode_raw not in {"off", "audit", "gate"}:
        raise RunPolicyError(f"invalid alpha_foundry mode: {mode_raw!r}")
    is_active_run = phase_raw in {"l0", "l1"}
    l0_runtime_mode = "gate" if is_active_run else mode_raw

    total_l1_budget = int(args.get("alpha_foundry_total_l1_budget", 30))
    min_conviction = float(args.get("alpha_foundry_min_conviction_lcb_bps", 5.0))
    enable_fast_tf = bool(args.get("alpha_foundry_enable_fast_tf", False))

    effective_runtime = AlphaFoundryRuntimeConfig(
        mode=l0_runtime_mode,  # type: ignore[arg-type]
        total_l1_verification_budget=max(1, total_l1_budget),
        min_conviction_lcb_bps=min_conviction,
        enable_fast_discovery_timeframes=enable_fast_tf,
        artifact_write_enabled=l0_runtime_mode != "off",
        enable_cross_tf_diversity_audit=bool(env_overrides.get("L0_CROSS_TF_DIVERSITY_AUDIT", False)),
        enable_cross_tf_pruning=bool(env_overrides.get("L0_CROSS_TF_PRUNING", True)),
        l0_parallel_max_workers=env_overrides.get("L0_PARALLEL_MAX_WORKERS", 1),
        enable_ltf_family_pool_experiment=bool(env_overrides.get("L0_LTF_POOL_WIDENED", False)),
        ltf_exec_1m_max_workers=lt_f_workers,
    )

    fingerprint = _build_policy_fingerprint(
        args, env_overrides, effective_runtime, execution_policy
    )

    config = FuturesRunConfig(
        timeframe=str(args.get("timeframe", "4h")),
        date=args.get("date"),
        trials=int(args.get("trials", 100)),
        phase=phase_raw,  # type: ignore[arg-type]
        sync=sync,  # type: ignore[arg-type]
        refresh_universe=bool(args.get("refresh_universe", False)),
        sync_metrics=bool(args.get("sync_metrics", False)),
        seed=int(args.get("seed", 42)),
        l0_runtime=effective_runtime,
        execution_policy=execution_policy,
        policy_fingerprint=fingerprint,
    )
    return config
