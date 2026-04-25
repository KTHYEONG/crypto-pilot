"""Trial-budget PBO/DSR gate helpers (futures-opt Phase 3, Holm-style trial scaling)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def trial_adjusted_pbo_ceiling(
    base: float,
    n_trials: int,
    *,
    step: float = 0.01,
    bucket: int = 100,
    clamp_min: float = 0.38,
) -> float:
    """Lower the PBO acceptance ceiling as trial count grows (stricter with more searches).

    Formula: ``min(base, max(clamp_min, base - step * (n_trials // bucket)))`` so the ceiling
    never exceeds ``base`` (misconfig-safe) and never tightens below ``clamp_min``.
    """
    b = max(1, int(bucket))
    k = max(0, int(n_trials)) // b
    adj = float(base) - float(step) * float(k)
    # Never loosen vs `base`; clamp only limits tightening (Holm-style floor on the ceiling).
    return float(min(float(base), max(float(clamp_min), adj)))


def trial_adjusted_dsr_floor(
    base: float,
    n_trials: int,
    *,
    step: float = 0.02,
    bucket: int = 100,
    clamp_max: float = 0.95,
) -> float:
    """Raise minimum DSR when trial count grows (optional; use a high ``base`` from config)."""
    b = max(1, int(bucket))
    k = max(0, int(n_trials)) // b
    adj = float(base) + float(step) * float(k)
    return float(min(float(clamp_max), adj))


def wf_path_ergodicity_deviation_pct(leg_tw: Sequence[float]) -> float:
    """Max abs(leg_tw - mean) / mean as percent; futures-opt P4 diagnostic (15%% guideline)."""
    arr = np.asarray(list(leg_tw), dtype=np.float64)
    if arr.size < 2:
        return 0.0
    m = float(np.mean(arr))
    if m < 1e-12:
        return 0.0
    return float(np.max(np.abs(arr - m)) / m * 100.0)


def resolve_adjusted_gates(cfg: dict[str, Any], n_trials: int) -> tuple[float, float, float]:
    """Return (pbo_max_hard, dsr_min_hard, pbo_champion_max) after optional trial adjustment."""
    raw_pbo_max = float(cfg.get("FUTURES_PBO_MAX", 0.45))
    raw_dsr_min = float(cfg.get("FUTURES_ML_GATE1_DSR_MIN", 0.20))
    raw_champ = float(cfg.get("FUTURES_CHAMPION_PBO_STRICT_MAX", 0.40))
    if not bool(cfg.get("FUTURES_MC_GATE_TRIAL_ADJUST_ENABLED", False)):
        return raw_pbo_max, raw_dsr_min, raw_champ

    bucket = int(cfg.get("FUTURES_MC_GATE_BUCKET_TRIALS", 100))
    pbo_step = float(cfg.get("FUTURES_MC_PBO_STEP_PER_BUCKET", 0.01))
    pbo_clamp = float(cfg.get("FUTURES_MC_PBO_CEILING_CLAMP_MIN", 0.38))

    pbo_max = trial_adjusted_pbo_ceiling(
        raw_pbo_max, n_trials, step=pbo_step, bucket=bucket, clamp_min=pbo_clamp
    )
    champ = trial_adjusted_pbo_ceiling(
        raw_champ, n_trials, step=pbo_step, bucket=bucket, clamp_min=pbo_clamp
    )
    dsr_min = raw_dsr_min
    if bool(cfg.get("FUTURES_MC_DSR_TRIAL_ADJUST_ENABLED", False)):
        dsr_step = float(cfg.get("FUTURES_MC_DSR_STEP_PER_BUCKET", 0.02))
        dsr_cap = float(cfg.get("FUTURES_MC_DSR_FLOOR_CAP", 0.95))
        dsr_min = trial_adjusted_dsr_floor(
            raw_dsr_min, n_trials, step=dsr_step, bucket=bucket, clamp_max=dsr_cap
        )
    return pbo_max, dsr_min, champ
