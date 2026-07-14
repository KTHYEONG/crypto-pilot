from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
) -> L1MemoryPlan:
    """Resolve bounded nested-worker concurrency from process-tree memory.

    [ADR_20260714_L1_MEMORY_EXECUTION]
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
        )

    tree_bytes = pss if pss is not None else (uss or 0)

    worker_private = max(GIB, result_soft_cap_bytes + int(0.25 * shared_input_bytes))
    headroom = min(
        tree_pss_cap_bytes - tree_bytes - reserve_bytes,
        available - reserve_bytes,
    )

    if headroom <= 0:
        return L1MemoryPlan(
            workers=1,
            estimated_worker_private_bytes=worker_private,
            projected_tree_bytes=tree_bytes + worker_private,
            reason="memory_floor_serial",
        )

    memory_workers = max(1, int(headroom / worker_private))
    workers = min(
        n_tasks,
        pinned if pinned is not None else n_tasks,
        stage_cap,
        cpu_cap,
        memory_workers,
    )
    workers = max(1, workers)

    projected = tree_bytes + workers * worker_private
    return L1MemoryPlan(
        workers=workers,
        estimated_worker_private_bytes=worker_private,
        projected_tree_bytes=projected,
        reason="ok" if workers > 1 else "memory_floor_serial",
    )
