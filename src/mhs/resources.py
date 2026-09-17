"""RAM-budget guard primitives (I4 seam: current_rss_bytes).

The OOM-safety metric is COW-correct: parent-only RSS misses fork-child
private allocation, and sum-of-RSS double-counts COW-shared pages, so
``_TreeMemorySampler`` records process-tree PSS/USS peaks plus the system
``available`` floor instead.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
from collections.abc import Callable

import psutil

from src.common.errors import DataIntegrityError
from src.mhs.contracts import MhsResourceMeasurement
from src.mhs.types import (
    RAM_BUDGET_FRACTION,
    RAM_RESERVE_FLOOR_BYTES,
    RAM_RESERVE_FRACTION,
)

_logger = logging.getLogger("MhsHorizonDiagnostic")

MHS_TREE_PSS_BUDGET_BYTES: int = int(2.5 * 2**30)
MHS_REPLAY_BUDGET_BYTES: int = int(1.5 * 2**30)
MHS_AVAILABLE_FLOOR_BYTES: int = 2 * 2**30


def _current_rss_bytes() -> int:
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        return -1


def _read_cgroup_limit_bytes() -> int | None:
    try:
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as handle:
            raw = handle.read().strip()
        if raw == "" or raw == "max":
            return None
        limit = int(raw)
        return limit if limit > 0 else None
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", encoding="utf-8") as handle:
            limit = int(handle.read().strip())
        return limit if limit > 0 else None
    except (OSError, ValueError):
        return None


def _read_cgroup_remaining_bytes() -> int | None:
    try:
        with open("/sys/fs/cgroup/memory.current", encoding="utf-8") as handle:
            current = int(handle.read().strip())
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as handle:
            raw = handle.read().strip()
        if raw == "" or raw == "max":
            return None
        remaining = int(raw) - current
        return remaining if remaining > 0 else 0
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes", encoding="utf-8") as handle:
            usage = int(handle.read().strip())
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", encoding="utf-8") as handle:
            limit = int(handle.read().strip())
        remaining = limit - usage
        return remaining if remaining > 0 else 0
    except (OSError, ValueError):
        return None


def _host_total_bytes() -> int:
    try:
        total = int(psutil.virtual_memory().total)
    except Exception as exc:  # noqa: BLE001
        raise DataIntegrityError(f"physical-memory telemetry unavailable: cannot read host total: {exc}") from exc
    if total <= 0:
        raise DataIntegrityError(f"physical-memory telemetry unavailable: non-positive host total {total}")
    return total


def _current_available_bytes() -> int:
    try:
        available = int(psutil.virtual_memory().available)
    except Exception as exc:  # noqa: BLE001
        raise DataIntegrityError(f"physical-memory telemetry unavailable: cannot read available memory: {exc}") from exc
    if available < 0:
        raise DataIntegrityError(f"physical-memory telemetry unavailable: negative available {available}")
    return available


def _current_tree_pss_bytes() -> int:
    try:
        me = psutil.Process(os.getpid())
        procs = [me, *me.children(recursive=True)]
    except Exception as exc:  # noqa: BLE001
        raise DataIntegrityError(f"physical-memory telemetry unavailable: cannot enumerate process tree: {exc}") from exc
    total = 0
    observed = 0
    for proc in procs:
        try:
            total += int(getattr(proc.memory_full_info(), "pss", 0))
            observed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:  # noqa: BLE001, S112 - single unreadable process never fails admission
            continue
    if observed == 0:
        raise DataIntegrityError("physical-memory telemetry unavailable: no process-tree PSS observation")
    return total


def _current_tree_swap_bytes() -> int | None:
    try:
        me = psutil.Process(os.getpid())
        procs = [me, *me.children(recursive=True)]
    except Exception:  # noqa: BLE001 - optional observation
        return None
    total = 0
    observed = False
    for proc in procs:
        try:
            total += int(getattr(proc.memory_full_info(), "swap", 0))
            observed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:  # noqa: BLE001, S112 - optional observation
            continue
    return total if observed else None


def _resolve_ram_budget(
    max_rss_bytes: int | None,
    ram_guard: bool,
) -> tuple[int | None, int | None]:
    """Resolve safe physical-memory admission limits.

    Args:
        max_rss_bytes: Optional positive explicit working-set ceiling.
        ram_guard: Whether physical-memory safeguards are required.

    Returns:
        Effective budget and available-memory reserve in bytes.

    Raises:
        ValueError: The explicit budget is invalid.
        DataIntegrityError: Required physical-memory telemetry is unavailable.
    """
    if not ram_guard:
        return (None, None)
    if max_rss_bytes is not None and max_rss_bytes <= 0:
        raise ValueError(f"max_rss_bytes must be positive, got {max_rss_bytes}")
    total = _host_total_bytes()
    cgroup_limit = _read_cgroup_limit_bytes()
    effective = min(total, cgroup_limit) if cgroup_limit is not None and cgroup_limit > 0 else total
    if max_rss_bytes is not None:
        budget = min(max_rss_bytes, MHS_TREE_PSS_BUDGET_BYTES)
    else:
        budget = min(int(effective * RAM_BUDGET_FRACTION), MHS_TREE_PSS_BUDGET_BYTES)
    remaining = _read_cgroup_remaining_bytes()
    if remaining is not None and remaining > 0:
        budget = min(budget, remaining)
    if budget <= 0:
        raise DataIntegrityError(
            f"minimum safe work is impossible: effective total {effective} leaves no usable budget"
        )
    reserve = max(int(effective * RAM_RESERVE_FRACTION), RAM_RESERVE_FLOOR_BYTES, MHS_AVAILABLE_FLOOR_BYTES)
    return (budget, reserve)


def assert_mhs_allocation_budget(*, estimated_bytes: int, budget_bytes: int | None, reserve_bytes: int | None) -> None:
    """Reject unsafe allocation before decoding or constructing execution planes.

    Args:
        estimated_bytes: Conservative additional working-set estimate.
        budget_bytes: Effective process-tree budget.
        reserve_bytes: Required available physical-memory floor.

    Returns:
        None when both budget and reserve admit allocation.

    Raises:
        DataIntegrityError: Admission is unsafe or cannot be measured.
    """
    if budget_bytes is None and reserve_bytes is None:
        return
    if estimated_bytes < 0:
        raise DataIntegrityError(f"mhs allocation estimate is invalid: estimated_bytes={estimated_bytes}")
    if budget_bytes is not None:
        if budget_bytes <= 0:
            raise DataIntegrityError(f"mhs allocation budget is invalid: budget_bytes={budget_bytes}")
        current = _current_tree_pss_bytes()
        if current + estimated_bytes > budget_bytes:
            raise DataIntegrityError(
                f"mhs allocation budget exceeded: tree_pss={current} estimated={estimated_bytes} "
                f"budget={budget_bytes}; no decoder or plane allocation begins"
            )
    if reserve_bytes is not None:
        if reserve_bytes <= 0:
            raise DataIntegrityError(f"mhs allocation reserve is invalid: reserve_bytes={reserve_bytes}")
        available = _current_available_bytes()
        if available < reserve_bytes or available - estimated_bytes < reserve_bytes:
            raise DataIntegrityError(
                f"mhs allocation reserve breached: available={available} estimated={estimated_bytes} "
                f"reserve={reserve_bytes}"
            )
    swap = _current_tree_swap_bytes()
    if swap is not None and swap > 0:
        raise DataIntegrityError(
            f"mhs allocation swap growth detected: tree_swap={swap}; replay aborts safely with diagnostics"
        )


def _assert_stage_rss_budget(
    stage: str,
    budget_bytes: int | None,
    reserve_bytes: int | None,
) -> None:
    """Deterministic fail-closed RAM barrier at a named stage boundary.

    (a) A positive ``budget_bytes`` exceeded by the current process RSS raises
    ``DataIntegrityError`` naming the stage. (b) When ``reserve_bytes`` is set
    and the system's available memory drops below it, ``DataIntegrityError`` is
    raised BEFORE the OS OOM killer can fire (WSL kills the whole VM, so the
    process must abort while headroom remains). psutil exceptions inside the
    reserve probe are swallowed (observational). Both ``None`` makes it a no-op.
    The guard never alters computed values.
    """
    if budget_bytes is not None:
        observed = _current_rss_bytes()
        if observed > budget_bytes:
            raise DataIntegrityError(
                f"RAM budget exceeded at stage '{stage}': "
                f"rss={observed} > budget={budget_bytes}"
            )
    if reserve_bytes is not None:
        try:
            available = int(psutil.virtual_memory().available)
        except Exception:  # noqa: BLE001
            return
        if available < reserve_bytes:
            raise DataIntegrityError(
                f"system RAM reserve breached at stage '{stage}': "
                f"available={available} < reserve={reserve_bytes}"
            )


def _assert_execution_rss_budget(
    stage: str,
    budget: int | None,
    completed_windows: int,
    reserve_bytes: int | None = None,
) -> None:
    """Deterministic fail-closed provenance for a configured RSS budget.

    A positive ``budget`` exceeded at a window boundary raises
    ``DataIntegrityError`` carrying the stage, observed RSS, configured budget,
    and completed window count; the default ``None`` applies no artificial cap.
    When ``reserve_bytes`` is set and the system's available memory drops below
    it, the same stable ``rss budget``-prefixed ``DataIntegrityError`` is raised
    so ``_classify_execution_failure`` keeps mapping it to
    ``GO_REASON_RESOURCE_BREACH`` -- the fork-worker OOM guard (only the
    system reserve applies to workers; the auto 85% budget is parent-only
    because fork-child RSS double-counts COW-shared pages).
    """
    if budget is None and reserve_bytes is None:
        return
    observed = _current_rss_bytes()
    if budget is not None and observed > budget:
        raise DataIntegrityError(
            "execution RSS budget exceeded at window boundary: "
            f"stage={stage} observed_rss={observed} "
            f"budget={budget} completed_windows={completed_windows}"
        )
    if reserve_bytes is not None:
        try:
            available = int(psutil.virtual_memory().available)
        except Exception:  # noqa: BLE001
            return
        if available < reserve_bytes:
            raise DataIntegrityError(
                "execution RSS budget (system reserve) breached at window boundary: "
                f"stage={stage} available={available} "
                f"reserve={reserve_bytes} completed_windows={completed_windows}"
            )


class _StageRecorder:
    """Collects ordered ``MhsResourceMeasurement`` records and emits ``[SYS]`` logs."""

    def __init__(self, log_run: bool) -> None:
        self._records: list[MhsResourceMeasurement] = []
        self._log_run = log_run
        self._last = time.perf_counter()
        self._peak_rss = -1
        self._worker_plan: dict[str, int] = {}

    @property
    def records(self) -> tuple[MhsResourceMeasurement, ...]:
        return tuple(self._records)

    @property
    def worker_plan(self) -> dict[str, int]:
        """Fork-point planner decisions: '<stage>' -> granted workers and
        '<stage>_per_worker_bytes' -> the per-worker budget used."""
        return dict(self._worker_plan)

    def record_worker_plan(
        self,
        stage: str,
        requested: int,
        granted: int,
        available_bytes: int,
        reserve_bytes: int,
        *,
        per_worker_bytes: int | None = None,
    ) -> None:
        """Record one ``plan_worker_count`` decision (observational, never raises)."""
        self._worker_plan[stage] = int(granted)
        if per_worker_bytes is not None:
            self._worker_plan[f"{stage}_per_worker_bytes"] = int(per_worker_bytes)
        if self._log_run:
            _logger.info(
                "[SYS] worker_plan stage=%s requested=%d granted=%d available=%d reserve=%d",
                stage, requested, granted, available_bytes, reserve_bytes,
            )

    def record(
        self,
        stage: str,
        grid_bars: int | None = None,
        n_symbols: int | None = None,
        fill_count: int | None = None,
        window_start: str | None = None,
        window_end: str | None = None,
        active_symbols: int | None = None,
    ) -> None:
        now = time.perf_counter()
        elapsed_ms = int((now - self._last) * 1000)
        self._last = now
        rss = _current_rss_bytes()
        self._peak_rss = max(self._peak_rss, rss)
        self._records.append(
            MhsResourceMeasurement(
                stage=stage,
                elapsed_ms=elapsed_ms,
                rss_bytes=rss,
                grid_bars=grid_bars,
                n_symbols=n_symbols,
                fill_count=fill_count,
                window_start=window_start,
                window_end=window_end,
                active_symbols=active_symbols,
                peak_rss_bytes=self._peak_rss,
            )
        )
        if self._log_run:
            _logger.info(
                "[SYS] stage=%s rss=%d elapsed_ms=%d",
                stage, rss, elapsed_ms,
            )

    def absorb(self, records: tuple[MhsResourceMeasurement, ...]) -> None:
        """Merge frozen records (e.g. from a book subprocess) into this recorder.

        Appends in arrival order, folds the peak-RSS tracking, and resets the
        elapsed baseline so the next ``record`` measures from the absorption
        point rather than from the last absorbed stage.
        """
        if not records:
            return
        self._records.extend(records)
        self._peak_rss = max(self._peak_rss, max(r.peak_rss_bytes or 0 for r in records))
        self._last = time.perf_counter()


@dataclasses.dataclass(frozen=True, slots=True)
class ProcessTreeMemoryStats:
    """COW-correct memory footprint of one run's whole process tree.

    Sum-of-RSS is deliberately NOT a field: it double-counts COW-shared parent
    pages (measured 16.61 GB vs a true 11.88 GB PSS). OOM safety is judged on
    ``min_system_available_bytes`` only. Sampling peaks are sampled observations,
    not exact instantaneous guarantees. Missing optional observations are null,
    not zero; ``MhsResourceMeasurement`` stays per-stage evidence and is never
    used as an aggregate PSS record.
    """

    tree_pss_peak_bytes: int
    tree_uss_peak_bytes: int
    min_system_available_bytes: int
    max_concurrent_procs: int
    samples_taken: int
    parent_rss_peak_bytes: int | None = None
    child_rss_peak_bytes: int | None = None
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    process_swap_growth_bytes: int | None = None


class _TreeMemorySampler:
    """Background daemon sampler over self + children(recursive=True).

    Records per-sample sums of ``memory_full_info().pss``/``.uss`` and
    ``psutil.virtual_memory().available``; ``stop()`` folds them into a
    ``ProcessTreeMemoryStats``. Every psutil call is wrapped so an
    observational failure can never raise into the run;
    ``NoSuchProcess``/``AccessDenied`` are skipped per process.
    """

    def __init__(self, interval_seconds: float = 1.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._interval = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._tree_pss_peak_bytes = -1
        self._tree_uss_peak_bytes = -1
        self._min_system_available_bytes = -1
        self._max_concurrent_procs = 0
        self._samples_taken = 0
        self._parent_rss_peak_bytes = -1
        self._child_rss_peak_bytes = -1
        self._swap_peak_bytes = -1
        self._swap_start_bytes: int | None = None
        self._start_wall: float | None = None
        self._start_cpu: float | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            self._start_wall = time.perf_counter()
            cpu = psutil.Process(os.getpid()).cpu_times()
            self._start_cpu = float(cpu.user + cpu.system)
        except Exception:  # noqa: BLE001 - observational
            self._start_wall = time.perf_counter()
            self._start_cpu = None
        self._swap_start_bytes = _current_tree_swap_bytes()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_loop, name="mhs-tree-memory-sampler", daemon=True,
        )
        self._thread.start()

    def stop(self) -> ProcessTreeMemoryStats:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(2.0 * self._interval, 2.0))
        end_wall = time.perf_counter()
        wall = end_wall - self._start_wall if self._start_wall is not None else 0.0
        cpu: float = 0.0
        try:
            now_cpu = psutil.Process(os.getpid()).cpu_times()
            now_total = float(now_cpu.user + now_cpu.system)
            if self._start_cpu is not None:
                cpu = max(now_total - self._start_cpu, 0.0)
        except Exception:  # noqa: BLE001 - observational
            cpu = 0.0
        with self._lock:
            parent_peak = self._parent_rss_peak_bytes if self._parent_rss_peak_bytes >= 0 else None
            child_peak = self._child_rss_peak_bytes if self._child_rss_peak_bytes >= 0 else None
            if self._swap_peak_bytes >= 0 and self._swap_start_bytes is not None:
                swap_growth: int | None = max(self._swap_peak_bytes - self._swap_start_bytes, 0)
            else:
                swap_growth = None
            return ProcessTreeMemoryStats(
                tree_pss_peak_bytes=max(self._tree_pss_peak_bytes, 0),
                tree_uss_peak_bytes=max(self._tree_uss_peak_bytes, 0),
                min_system_available_bytes=max(self._min_system_available_bytes, 0),
                max_concurrent_procs=self._max_concurrent_procs,
                samples_taken=self._samples_taken,
                parent_rss_peak_bytes=parent_peak,
                child_rss_peak_bytes=child_peak,
                wall_seconds=float(wall),
                cpu_seconds=float(cpu),
                process_swap_growth_bytes=swap_growth,
            )

    def _sample_once(self) -> None:
        pss_sum = 0
        uss_sum = 0
        swap_sum = 0
        n_procs = 0
        parent_rss = -1
        child_peak = -1
        try:
            me = psutil.Process(os.getpid())
            procs = [me, *me.children(recursive=True)]
        except Exception:  # noqa: BLE001 - observational
            procs = []
        for index, proc in enumerate(procs):
            try:
                info = proc.memory_full_info()
                pss_sum += int(getattr(info, "pss", 0))
                uss_sum += int(getattr(info, "uss", 0))
                swap_sum += int(getattr(info, "swap", 0))
                rss = int(info.rss)
                if index == 0:
                    parent_rss = rss
                else:
                    child_peak = max(child_peak, rss)
                n_procs += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception:  # noqa: BLE001, S112 - observational sampler never raises into the run
                continue
        try:
            available = int(psutil.virtual_memory().available)
        except Exception:  # noqa: BLE001 - observational
            available = -1
        with self._lock:
            self._samples_taken += 1
            if n_procs:
                self._max_concurrent_procs = max(self._max_concurrent_procs, n_procs)
                self._tree_pss_peak_bytes = max(self._tree_pss_peak_bytes, pss_sum)
                self._tree_uss_peak_bytes = max(self._tree_uss_peak_bytes, uss_sum)
                self._swap_peak_bytes = max(self._swap_peak_bytes, swap_sum)
                if parent_rss >= 0:
                    self._parent_rss_peak_bytes = max(self._parent_rss_peak_bytes, parent_rss)
                if child_peak >= 0:
                    self._child_rss_peak_bytes = max(self._child_rss_peak_bytes, child_peak)
            if available >= 0:
                if self._min_system_available_bytes < 0:
                    self._min_system_available_bytes = available
                else:
                    self._min_system_available_bytes = min(
                        self._min_system_available_bytes, available,
                    )

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self._interval)


def _worker_plan_observer(
    recorder: _StageRecorder | None,
    stage: str,
    per_worker_bytes: int | None = None,
) -> Callable[[str, int, int, int, int], None] | None:
    """Bind a fork-point stage label onto the recorder's worker-plan log.

    ``plan_worker_count`` invokes its observer with a neutral stage label, so
    each call site re-binds the true fork-point name ('books',
    'anchored_folds', 'post_book_folds', 'fold_safe_discovery') plus the
    per-worker budget it requested.
    """
    if recorder is None:
        return None

    def _observe(
        _stage: str, requested: int, granted: int, available_bytes: int, reserve_bytes: int,
    ) -> None:
        recorder.record_worker_plan(
            stage, requested, granted, available_bytes, reserve_bytes,
            per_worker_bytes=per_worker_bytes,
        )

    return _observe


def _peak_rss_bytes(
    resource_measurements: tuple[MhsResourceMeasurement, ...],
) -> int | None:
    return max((m.rss_bytes for m in resource_measurements), default=None)
