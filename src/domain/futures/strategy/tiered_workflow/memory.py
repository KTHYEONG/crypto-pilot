from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.domain.futures.strategy.walk_forward import WFFold

MIB: int = 1024 * 1024
GIB: int = 1024 * MIB


@dataclass(slots=True, frozen=True)
class ProcessTreeMemory:
    parent_rss_bytes: int
    tree_pss_bytes: int | None
    tree_uss_bytes: int | None
    available_bytes: int


@dataclass(slots=True, frozen=True)
class L1MemoryPlan:
    workers: int
    estimated_worker_private_bytes: int
    projected_tree_bytes: int
    reason: str
    binding_constraint: str


def estimate_unique_array_bytes(value: object) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, pd.DataFrame):
        total = 0
        seen = set()
        for col in value.columns:
            arr = value[col].values
            if hasattr(arr, "base") and arr.base is not None:
                ptr = id(arr.base)
                if ptr in seen:
                    continue
                seen.add(ptr)
                total += int(arr.base.nbytes)
            else:
                total += int(value[col].nbytes)
        return total
    if isinstance(value, pd.Series):
        return estimate_unique_array_bytes(pd.DataFrame({"_": value}))
    return 0


def measure_worker_private_bytes(
    before: ProcessTreeMemory,
    after: ProcessTreeMemory,
    workers: int,
) -> int | None:
    """Empirically derive per-worker private growth from tree snapshots taken
    immediately before fork-pool submission and immediately after result collection.

    [ADR pending] Validates (or refutes) the fixed `worker_private = max(GIB, ...)`
    assumption in `resolve_l1_memory_plan` against actual fork COW-defeat growth.
    Returns None when PSS/USS metrics are unavailable on either snapshot (fail-open,
    diagnostic only — never affects worker count).
    """
    before_bytes = before.tree_pss_bytes if before.tree_pss_bytes is not None else before.tree_uss_bytes
    after_bytes = after.tree_pss_bytes if after.tree_pss_bytes is not None else after.tree_uss_bytes
    if before_bytes is None or after_bytes is None or workers <= 0:
        return None
    return max(0, after_bytes - before_bytes) // workers


def snapshot_process_tree_memory(root_pid: int) -> ProcessTreeMemory:
    """Capture parent and descendant memory for fork planning.

    [ADR_20260714_L1_MEMORY_EXECUTION]
    """
    import psutil

    try:
        proc = psutil.Process(root_pid)
        parent_rss = proc.memory_info().rss
        available = psutil.virtual_memory().available
        parent_full = proc.memory_full_info()
        parent_pss = getattr(parent_full, "pss", None)
        parent_uss = getattr(parent_full, "uss", None)
        tree_pss: int | None = int(parent_pss) if parent_pss is not None else None
        tree_uss: int | None = int(parent_uss) if parent_uss is not None else None
        children = proc.children(recursive=True)
        for child in children:
            try:
                full_info = child.memory_full_info()
                child_pss = getattr(full_info, "pss", None)
                child_uss = getattr(full_info, "uss", None)
                if child_pss is not None:
                    tree_pss = (tree_pss or 0) + int(child_pss)
                if child_uss is not None:
                    tree_uss = (tree_uss or 0) + int(child_uss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        parent_rss = 0
        tree_pss = None
        tree_uss = None
        try:
            available = psutil.virtual_memory().available
        except Exception:
            available = 0

    return ProcessTreeMemory(
        parent_rss_bytes=parent_rss,
        tree_pss_bytes=tree_pss,
        tree_uss_bytes=tree_uss,
        available_bytes=available,
    )


def resolve_l1_memory_plan(
    *,
    n_tasks: int,
    shared_input_bytes: int,
    result_soft_cap_bytes: int,
    snapshot: ProcessTreeMemory,
    stage_cap: int,
    cpu_cap: int,
    pinned: int | None = None,
    tree_pss_cap_bytes: int = 10 * GIB,
    reserve_bytes: int = 1 * GIB,
    worker_private_floor_bytes: int = GIB,
) -> L1MemoryPlan:
    """Resolve bounded nested-worker concurrency from process-tree memory.

    [ADR_20260714_L1_MEMORY_EXECUTION]

    worker_private_floor_bytes: minimum assumed per-worker private growth (default 1GiB,
    a conservative fork-COW-defeat estimate). Overridable via
    OPT_FUTURES_CONFIG["L1_WORKER_PRIVATE_FLOOR_MB"] for empirical calibration runs —
    see [SYS] stage=worker_private_measured for the actually observed value.
    """
    pss = snapshot.tree_pss_bytes
    uss = snapshot.tree_uss_bytes
    available = snapshot.available_bytes

    if pss is None and uss is None:
        return L1MemoryPlan(
            workers=1,
            estimated_worker_private_bytes=0,
            projected_tree_bytes=0,
            reason="memory_metrics_unavailable",
            binding_constraint="metrics_unavailable",
        )

    tree_bytes = pss if pss is not None else (uss or 0)

    worker_private = max(worker_private_floor_bytes, result_soft_cap_bytes + int(0.25 * shared_input_bytes))
    headroom = min(
        tree_pss_cap_bytes - tree_bytes - reserve_bytes,
        available - reserve_bytes,
    )

    tree_available = tree_pss_cap_bytes - tree_bytes - reserve_bytes
    system_available = available - reserve_bytes
    if headroom <= 0:
        binding = "tree_pss_cap" if tree_available <= system_available else "system_available"
        return L1MemoryPlan(
            workers=1,
            estimated_worker_private_bytes=worker_private,
            projected_tree_bytes=tree_bytes + worker_private,
            reason="memory_floor_serial",
            binding_constraint=binding,
        )

    memory_workers = max(1, int(headroom / worker_private))
    candidates: list[tuple[str, int]] = [
        ("n_tasks", n_tasks),
        ("pinned", pinned if pinned is not None else n_tasks),
        ("stage_cap", stage_cap),
        ("cpu_cap", cpu_cap),
        ("memory_workers", memory_workers),
    ]
    workers = min(v for _, v in candidates)
    workers = max(1, workers)

    binding = next(name for name, v in candidates if v == workers)

    projected = tree_bytes + workers * worker_private
    return L1MemoryPlan(
        workers=workers,
        estimated_worker_private_bytes=worker_private,
        projected_tree_bytes=projected,
        reason="ok" if workers > 1 else "memory_floor_serial",
        binding_constraint=binding,
    )


# ── Worker-private online calibration store (Phase 1 adaptive) ────────────
# [ADR_20260720_L1_MEMORY_FLOOR_ADAPTIVE_CALIBRATION]

_WORKER_PRIVATE_OBSERVATIONS: dict[str, list[tuple[float, float]]] = {}


def reset_worker_private_calibration() -> None:
    """[LIMIT-05] Clear all accumulated (shared_mb, measured_mb) observations."""
    _WORKER_PRIVATE_OBSERVATIONS.clear()


def record_worker_private_observation(stage: str, shared_mb: float, measured_mb: float) -> None:
    """Append one (shared_mb, measured_mb) sample for `stage`."""
    _WORKER_PRIVATE_OBSERVATIONS.setdefault(stage, []).append((shared_mb, measured_mb))


def get_worker_private_observations(stage: str) -> list[tuple[float, float]]:
    """Return a copy of accumulated observations for `stage` (empty list if none)."""
    return list(_WORKER_PRIVATE_OBSERVATIONS.get(stage, []))


def fit_worker_private_linear_model(
    observations: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """OLS fit of measured_mb = intercept + slope * shared_mb.

    Returns None if fewer than 2 observations or if all shared_mb values
    are identical (zero variance, undefined slope).
    """
    if len(observations) < 2:
        return None

    n = len(observations)
    x_vals = [x for x, _ in observations]
    y_vals = [y for _, y in observations]

    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n

    num = sum((x - x_mean) * (y - y_mean) for x, y in observations)
    den = sum((x - x_mean) ** 2 for x in x_vals)

    if abs(den) < 1e-12:
        return None

    slope = num / den
    intercept = y_mean - slope * x_mean
    return (intercept, slope)


MIN_PILOT_TASKS: int = 6
MIN_WORKER_BYTES: int = 512 * MIB


@dataclass(frozen=True, slots=True)
class L1PilotMeasurement:
    stage: str
    fold_id: int
    shared_input_bytes: int
    worker_private_bytes: int
    result_bytes: int
    elapsed_seconds: float


def resolve_post_pilot_memory_plan(
    *,
    n_remaining: int,
    pilot: L1PilotMeasurement,
    snapshot: ProcessTreeMemory,
    stage_cap: int,
    cpu_cap: int,
    pinned: int | None = None,
    tree_pss_cap_bytes: int = 10 * GIB,
    reserve_bytes: int = GIB,
    safety_margin: float = 1.3,
) -> L1MemoryPlan:
    usable = min(
        snapshot.available_bytes,
        tree_pss_cap_bytes - (snapshot.tree_pss_bytes or 0),
    ) - reserve_bytes

    per_worker = max(
        pilot.worker_private_bytes + pilot.result_bytes,
        MIN_WORKER_BYTES,
    )
    if usable <= 0:
        return L1MemoryPlan(
            workers=1,
            estimated_worker_private_bytes=per_worker,
            projected_tree_bytes=0,
            reason="memory_floor_serial",
            binding_constraint="usable_zero",
        )

    safe_workers = max(1, int(usable / (per_worker * safety_margin)))
    candidates: list[tuple[str, int]] = [
        ("n_remaining", n_remaining),
        ("stage_cap", stage_cap),
        ("cpu_cap", cpu_cap),
    ]
    if pinned is not None:
        candidates.append(("pinned", pinned))
    resolved = min(v for _, v in candidates)
    resolved = max(1, min(safe_workers, resolved))
    binding = "memory_floor_serial"
    if resolved > 1:
        for name, val in candidates:
            if resolved == val:
                binding = name
                break

    projected = (snapshot.tree_pss_bytes or 0) + resolved * per_worker
    return L1MemoryPlan(
        workers=resolved,
        estimated_worker_private_bytes=per_worker,
        projected_tree_bytes=projected,
        reason="ok" if resolved > 1 else "memory_floor_serial",
        binding_constraint=binding,
    )


def select_l1_pilot_fold_index(folds: tuple[WFFold, ...]) -> int:
    """Select the fold with largest estimated input (bar span x event count).

    Tie-break: smallest fold_id.
    """
    best_idx = 0
    best_size = -1
    for i, f in enumerate(folds):
        span = f.fit_end - f.fit_start
        cal_span = f.cal_end - f.cal_start
        size = span + cal_span
        if size > best_size:
            best_size = size
            best_idx = i
    return best_idx


def predict_calibrated_worker_private_mb(
    observations: list[tuple[float, float]],
    shared_mb: float,
    default_mb: float,
    margin: float = 1.3,
) -> float:
    """[LIMIT-01][LIMIT-02][LIMIT-03] Predict worker_private for the next TF.

    - len(observations) < 2 -> return default_mb unchanged (cold start).
    - Else: fit linear model, predict at `shared_mb`, clamp to
      max(predicted, max(m for _, m in observations)), multiply by `margin`.
    """
    if len(observations) < 2:
        return default_mb

    model = fit_worker_private_linear_model(observations)
    if model is None:
        return default_mb

    intercept, slope = model
    predicted = intercept + slope * shared_mb
    observed_max = max(m for _, m in observations)
    clamped = max(predicted, observed_max)
    return clamped * margin
