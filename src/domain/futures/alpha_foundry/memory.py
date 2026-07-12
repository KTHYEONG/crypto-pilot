"""L0 memory budget detection and stage admission.

[ADR_20260712_L0_MEMORY_BOUND_DATAFLOW]
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class L0MemoryBudget:
    limit_mb: int
    safety_margin_mb: int


@dataclass(frozen=True, slots=True)
class LtfExec1mPlan:
    symbols: tuple[str, ...]
    max_workers: int
    skip_reason: str | None


def _read_vm_rss_mb() -> int:
    """Read VmRSS from /proc/self/status (no psutil dependency)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError) as exc:
        _logger.debug("[MEM] VmRSS read failed: %s", exc)
    return 0


def _detect_cgroup_limit_mb() -> int | None:
    """Detect cgroup memory limit (finite only)."""
    try:
        for path in ("/sys/fs/cgroup/memory/memory.limit_in_bytes",):
            p = Path(path)
            if p.exists():
                raw = int(p.read_text().strip())
                if raw > 0 and raw < 2**63 - 1:
                    return int(raw // (1024 * 1024))
    except (OSError, ValueError, OverflowError) as exc:
        _logger.debug("[MEM] cgroup limit read failed: %s", exc)
    return None


def _detect_physical_limit_mb() -> int | None:
    """Detect physical memory via sysconf."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        pages = libc.sysconf(200)  # _SC_PHYS_PAGES
        page_size = libc.sysconf(209)  # _SC_PAGE_SIZE
        if pages > 0 and page_size > 0:
            return int(pages * page_size // (1024 * 1024))
    except (OSError, TypeError, ValueError) as exc:
        _logger.debug("[MEM] physical memory detection failed: %s", exc)
    # fallback: /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError) as exc:
        _logger.debug("[MEM] /proc/meminfo fallback failed: %s", exc)
    return None


def current_process_rss_mb() -> int:
    """Return current process RSS in megabytes."""
    return _read_vm_rss_mb()


def resolve_effective_memory_budget(
    *,
    hard_limit_mb: int = 10_240,
    fraction_cap: float = 0.60,
) -> L0MemoryBudget:
    """Resolve effective memory budget from cgroup, physical limit, and config."""
    detected_mb: int | None = _detect_cgroup_limit_mb()
    if detected_mb is None:
        detected_mb = _detect_physical_limit_mb()

    if detected_mb is not None and detected_mb > 0:
        fraction_limit_mb = int(detected_mb * fraction_cap)
        limit_mb = min(hard_limit_mb, fraction_limit_mb)
    else:
        limit_mb = hard_limit_mb

    safety_margin_mb = max(1, int(limit_mb * 0.05))
    return L0MemoryBudget(limit_mb=limit_mb, safety_margin_mb=safety_margin_mb)


def admit_memory_stage(
    *,
    budget: L0MemoryBudget,
    stage: str,
    estimated_increment_mb: int,
    current_rss_mb: int | None = None,
) -> bool:
    """Check whether a stage fits within the memory budget."""
    rss = current_rss_mb if current_rss_mb is not None else current_process_rss_mb()
    projected = rss + estimated_increment_mb + budget.safety_margin_mb
    admitted = projected <= budget.limit_mb
    _logger.debug(
        "[MEM] stage=%s rss_mb=%d estimated_increment_mb=%d safety_margin_mb=%d "
        "budget_mb=%d admitted=%s",
        stage, rss, estimated_increment_mb, budget.safety_margin_mb,
        budget.limit_mb, admitted,
    )
    return admitted


def resolve_ltf_exec_1m_plan(
    *,
    covered_symbols: frozenset[str],
    valid_symbols: frozenset[str],
    max_symbols: int = 64,
    max_workers: int = 1,
    budget: L0MemoryBudget | None = None,
    skip_reason: str | None = None,
) -> LtfExec1mPlan:
    """Resolve the LTF exec_1m plan from coverage and budget."""
    if skip_reason:
        return LtfExec1mPlan(symbols=(), max_workers=max_workers, skip_reason=skip_reason)

    eligible = sorted(covered_symbols & valid_symbols)
    selected = tuple(eligible[:max_symbols])
    if not selected:
        return LtfExec1mPlan(symbols=(), max_workers=max_workers, skip_reason="no_covered_symbols")

    if budget is not None and not admit_memory_stage(
        budget=budget, stage="ltf_stream",
        estimated_increment_mb=512,
    ):
        return LtfExec1mPlan(symbols=(), max_workers=max_workers, skip_reason="budget")

    return LtfExec1mPlan(symbols=selected, max_workers=min(max_workers, 1), skip_reason=None)
