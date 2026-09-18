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
from typing import Literal

import psutil

from src.common.errors import DataIntegrityError
from src.mhs.contracts import MhsResourceMeasurement
from src.mhs.types import (
    RAM_BUDGET_FRACTION,
    RAM_RESERVE_FLOOR_BYTES,
    RAM_RESERVE_FRACTION,
)

_logger = logging.getLogger("MhsHorizonDiagnostic")

MHS_TREE_PSS_BUDGET_BYTES: int = 6 * 2**30
MHS_REPLAY_BUDGET_BYTES: int = 4 * 2**30
MHS_AVAILABLE_FLOOR_BYTES: int = 2 * 2**30


@dataclasses.dataclass(frozen=True, slots=True)
class MhsMemoryBudget:
    """Stage-specific process-tree memory limits in bytes. Total and replay limits constrain resident state plus admitted allocations; physical-memory reserve also accounts for cgroup headroom. Limits never alter financial decisions or source coverage. All fields are positive integers and the replay limit cannot exceed the total limit."""

    total_tree_pss_bytes: int = MHS_TREE_PSS_BUDGET_BYTES
    replay_tree_pss_bytes: int = MHS_REPLAY_BUDGET_BYTES
    min_available_bytes: int = MHS_AVAILABLE_FLOOR_BYTES

    def __post_init__(self) -> None:
        for name in ("total_tree_pss_bytes", "replay_tree_pss_bytes", "min_available_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.replay_tree_pss_bytes > self.total_tree_pss_bytes:
            raise ValueError(
                f"replay_tree_pss_bytes={self.replay_tree_pss_bytes} cannot exceed "
                f"total_tree_pss_bytes={self.total_tree_pss_bytes}"
            )


class MhsResourceAdmissionError(DataIntegrityError):
    """Typed resource rejection carrying measured stage and cause. The diagnostic message preserves observed and requested bytes; classification does not infer OOM from text, RSS peaks or a nonzero exit status."""

    def __init__(
        self, *, stage: str, error_code: Literal["MEMORY_BUDGET", "MEMORY_RESERVE", "SWAP_GROWTH", "RESOURCE_TELEMETRY"], message: str,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.error_code = error_code


def _resolve_memory_budget(budget: MhsMemoryBudget | None) -> MhsMemoryBudget:
    resolved = budget if budget is not None else MhsMemoryBudget()
    if not isinstance(resolved, MhsMemoryBudget):
        raise ValueError(f"budget must be MhsMemoryBudget or None, got {resolved!r}")
    return resolved


def resolve_mhs_memory_budget(budget: MhsMemoryBudget | None) -> MhsMemoryBudget:
    """Resolve one run's measured-profile ceilings against physical capacity.

    Args:
        budget: Explicit stage ceilings and reserve, or the full-period profile.
    Returns:
        Positive stage ceilings bounded by host/cgroup capacity while preserving
        the requested physical reserve. A ceiling is not a memory reservation.
    Raises:
        ValueError: The budget contract is invalid.
        MhsResourceAdmissionError: Required telemetry is unavailable or capacity
            cannot preserve the requested physical reserve.
    """
    if budget is None:
        requested = MhsMemoryBudget()
    elif isinstance(budget, MhsMemoryBudget):
        requested = budget
    else:
        raise ValueError(f"budget must be MhsMemoryBudget or None, got {budget!r}")
    try:
        total = _host_total_bytes()
    except DataIntegrityError as exc:
        raise MhsResourceAdmissionError(
            stage="process_prepare_panel", error_code="RESOURCE_TELEMETRY",
            message=f"resource telemetry unavailable while resolving memory budget: {exc}",
        ) from exc
    cgroup_limit = _read_cgroup_limit_bytes()
    effective = total if cgroup_limit is None else min(total, cgroup_limit)
    reserve = requested.min_available_bytes
    if effective <= reserve:
        raise MhsResourceAdmissionError(
            stage="process_prepare_panel", error_code="MEMORY_RESERVE",
            message=f"mhs memory budget cannot preserve reserve: effective={effective} reserve={reserve}",
        )
    cap = effective - reserve
    total_ceiling = min(requested.total_tree_pss_bytes, cap)
    replay_ceiling = min(requested.replay_tree_pss_bytes, cap, total_ceiling)
    return MhsMemoryBudget(
        total_tree_pss_bytes=total_ceiling,
        replay_tree_pss_bytes=replay_ceiling,
        min_available_bytes=reserve,
    )


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
            info = proc.memory_full_info()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise DataIntegrityError(f"physical-memory telemetry unavailable: unreadable live process: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise DataIntegrityError(f"physical-memory telemetry unavailable: cannot read process memory: {exc}") from exc
        pss = getattr(info, "pss", None)
        if pss is None:
            raise DataIntegrityError("physical-memory telemetry unavailable: process-tree PSS observation missing")
        try:
            total += int(pss)
        except (TypeError, ValueError) as exc:
            raise DataIntegrityError(f"physical-memory telemetry unavailable: invalid PSS observation: {exc}") from exc
        observed += 1
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


def assert_mhs_allocation_budget(*, estimated_bytes: int, budget_bytes: int | None, reserve_bytes: int | None, stage: str = "execution_allocation", initial_swap_bytes: int | None = None) -> None:
    """Reject unsafe additional working memory before execution decoding.

    Args:
        estimated_bytes: Additional simultaneously live allocation estimate.
        budget_bytes: Total admitted resident process-tree PSS ceiling.
        reserve_bytes: Minimum remaining effective physical headroom.
        stage: Stable failure provenance for the allocation boundary.
        initial_swap_bytes: Run-entry tree swap, if observable.
    Returns:
        None when tree PSS, physical headroom and swap-growth checks pass.
    Raises:
        ValueError: A supplied size, limit or stage is invalid.
        MhsResourceAdmissionError: Admission or required telemetry fails.
    """
    if budget_bytes is None and reserve_bytes is None:
        return
    if isinstance(estimated_bytes, bool) or not isinstance(estimated_bytes, int) or estimated_bytes < 0:
        raise ValueError(f"estimated_bytes must be a non-negative integer, got {estimated_bytes!r}")
    if not isinstance(stage, str) or not stage:
        raise ValueError(f"stage must be a non-empty string, got {stage!r}")
    for limit_name, limit_value in (("budget_bytes", budget_bytes), ("reserve_bytes", reserve_bytes)):
        if limit_value is not None and (isinstance(limit_value, bool) or not isinstance(limit_value, int) or limit_value <= 0):
            raise ValueError(f"{limit_name} must be a positive integer or None, got {limit_value!r}")
    if initial_swap_bytes is not None and (
        isinstance(initial_swap_bytes, bool) or not isinstance(initial_swap_bytes, int) or initial_swap_bytes < 0
    ):
        raise ValueError(f"initial_swap_bytes must be a non-negative integer or None, got {initial_swap_bytes!r}")
    if budget_bytes is not None:
        try:
            current = _current_tree_pss_bytes()
        except DataIntegrityError as exc:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="RESOURCE_TELEMETRY",
                message=f"resource telemetry unavailable at stage '{stage}': {exc}",
            ) from exc
        if current + estimated_bytes > budget_bytes:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="MEMORY_BUDGET",
                message=f"mhs allocation budget exceeded at '{stage}': tree_pss={current} estimated={estimated_bytes} budget={budget_bytes}; no decoder or plane allocation begins",
            )
    if reserve_bytes is not None:
        try:
            headroom = _tree_headroom_bytes()
        except DataIntegrityError as exc:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="RESOURCE_TELEMETRY",
                message=f"resource telemetry unavailable at stage '{stage}': {exc}",
            ) from exc
        if headroom - estimated_bytes < reserve_bytes:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="MEMORY_RESERVE",
                message=f"mhs allocation reserve breached at '{stage}': headroom={headroom} estimated={estimated_bytes} reserve={reserve_bytes}",
            )
    if initial_swap_bytes is not None:
        current_swap = _current_tree_swap_bytes()
        if current_swap is not None and current_swap > initial_swap_bytes:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="SWAP_GROWTH",
                message=f"mhs allocation swap growth at '{stage}': initial_swap={initial_swap_bytes} current_swap={current_swap}",
            )


def _tree_headroom_bytes() -> int:
    """Minimum of host available and known cgroup remaining bytes."""
    available = _current_available_bytes()
    remaining = _read_cgroup_remaining_bytes()
    if remaining is None:
        return available
    return min(available, remaining)


def current_mhs_headroom_bytes() -> int:
    """Observe effective physical headroom for worker and supervisor admission.

    Returns:
        Minimum host available and known nonnegative cgroup remaining bytes.
    Raises:
        DataIntegrityError: Required host telemetry is unavailable or invalid.
    """
    return _tree_headroom_bytes()


def assert_mhs_stage_allocation(
    *, stage: str, estimated_bytes: int, budget: MhsMemoryBudget,
    replay: bool, initial_swap_bytes: int | None,
) -> None:
    """Admit a named allocation before its decoder or financial buffers start.

    Args:
        stage: Stable diagnostic stage identifier.
        estimated_bytes: Conservative additional simultaneously live bytes.
        budget: Validated process-tree and available-memory limits.
        replay: Whether the stricter replay limit applies.
        initial_swap_bytes: Run-entry tree swap baseline, if measurable.

    Returns:
        None when measured tree PSS and physical headroom admit allocation.

    Raises:
        DataIntegrityError: Admission is unsafe or required telemetry is unknown.
        ValueError: Budget fields, stage or allocation size are invalid.
    """
    if not isinstance(stage, str) or not stage:
        raise ValueError(f"stage must be a non-empty string, got {stage!r}")
    if isinstance(estimated_bytes, bool) or not isinstance(estimated_bytes, int) or estimated_bytes < 0:
        raise ValueError(f"estimated_bytes must be a non-negative integer, got {estimated_bytes!r}")
    resolved = _resolve_memory_budget(budget)
    if not isinstance(replay, bool):
        raise ValueError(f"replay must be bool, got {replay!r}")
    if initial_swap_bytes is not None and (
        isinstance(initial_swap_bytes, bool) or not isinstance(initial_swap_bytes, int) or initial_swap_bytes < 0
    ):
        raise ValueError(f"initial_swap_bytes must be a non-negative integer or None, got {initial_swap_bytes!r}")
    limit = resolved.replay_tree_pss_bytes if replay else resolved.total_tree_pss_bytes
    try:
        current = _current_tree_pss_bytes()
    except DataIntegrityError as exc:
        raise MhsResourceAdmissionError(stage=stage, error_code="RESOURCE_TELEMETRY", message=f"resource telemetry unavailable at stage '{stage}': {exc}") from exc
    if current + estimated_bytes > limit:
        raise MhsResourceAdmissionError(
            stage=stage, error_code="MEMORY_BUDGET",
            message=f"mhs stage allocation rejected at '{stage}': tree_pss={current} estimated={estimated_bytes} budget={limit}",
        )
    try:
        headroom = _tree_headroom_bytes()
    except DataIntegrityError as exc:
        raise MhsResourceAdmissionError(stage=stage, error_code="RESOURCE_TELEMETRY", message=f"resource telemetry unavailable at stage '{stage}': {exc}") from exc
    if headroom - estimated_bytes < resolved.min_available_bytes:
        raise MhsResourceAdmissionError(
            stage=stage, error_code="MEMORY_RESERVE",
            message=f"mhs stage reserve breached at '{stage}': headroom={headroom} estimated={estimated_bytes} reserve={resolved.min_available_bytes}",
        )
    if initial_swap_bytes is not None:
        current_swap = _current_tree_swap_bytes()
        if current_swap is not None and current_swap > initial_swap_bytes:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="SWAP_GROWTH",
                message=f"mhs stage swap growth at '{stage}': initial_swap={initial_swap_bytes} current_swap={current_swap}",
            )


def _assert_stage_rss_budget(
    stage: str,
    budget_bytes: int | None,
    reserve_bytes: int | None,
) -> None:
    """Deterministic fail-closed RAM barrier at a named stage boundary.

    Budget compares complete live process-tree PSS and reserve compares the
    minimum of host available and known cgroup remaining bytes; required
    telemetry failures raise instead of passing silently. Both ``None`` makes
    it a no-op. The guard never alters computed values.
    """
    if budget_bytes is None and reserve_bytes is None:
        return
    if budget_bytes is not None:
        try:
            observed = _current_tree_pss_bytes()
        except DataIntegrityError as exc:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="RESOURCE_TELEMETRY",
                message=f"resource telemetry unavailable at stage '{stage}': {exc}",
            ) from exc
        if observed > budget_bytes:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="MEMORY_BUDGET",
                message=f"RAM budget exceeded at stage '{stage}': tree_pss={observed} > budget={budget_bytes}",
            )
    if reserve_bytes is not None:
        try:
            headroom = _tree_headroom_bytes()
        except DataIntegrityError as exc:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="RESOURCE_TELEMETRY",
                message=f"resource telemetry unavailable at stage '{stage}': {exc}",
            ) from exc
        if headroom < reserve_bytes:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="MEMORY_RESERVE",
                message=f"system RAM reserve breached at stage '{stage}': headroom={headroom} < reserve={reserve_bytes}",
            )


def _assert_execution_rss_budget(
    stage: str,
    budget: int | None,
    completed_windows: int,
    reserve_bytes: int | None = None,
) -> None:
    """Deterministic fail-closed provenance for a configured RSS budget.

    Budget compares complete live process-tree PSS and reserve compares the
    minimum of host available and known cgroup remaining bytes; required
    telemetry failures raise instead of passing silently. The default ``None``
    applies no artificial cap. The stable ``rss budget``-prefixed messages keep
    mapping to ``GO_REASON_RESOURCE_BREACH``.
    """
    if budget is None and reserve_bytes is None:
        return
    if budget is not None:
        try:
            observed = _current_tree_pss_bytes()
        except DataIntegrityError as exc:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="RESOURCE_TELEMETRY",
                message=f"resource telemetry unavailable at stage '{stage}': {exc}",
            ) from exc
        if observed > budget:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="MEMORY_BUDGET",
                message="execution RSS budget exceeded at window boundary: "
                f"stage={stage} observed_tree_pss={observed} "
                f"budget={budget} completed_windows={completed_windows}",
            )
    if reserve_bytes is not None:
        try:
            headroom = _tree_headroom_bytes()
        except DataIntegrityError as exc:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="RESOURCE_TELEMETRY",
                message=f"resource telemetry unavailable at stage '{stage}': {exc}",
            ) from exc
        if headroom < reserve_bytes:
            raise MhsResourceAdmissionError(
                stage=stage, error_code="MEMORY_RESERVE",
                message="execution RSS budget (system reserve) breached at window boundary: "
                f"stage={stage} headroom={headroom} "
                f"reserve={reserve_bytes} completed_windows={completed_windows}",
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

    Global sampled tree PSS/USS peaks cover every sample; stage-specific
    sampled PSS peaks isolate preparation versus replay residency; optional
    individual RSS peaks and headroom describe single-process evidence.
    Admission uses dual PSS/headroom limits: tree PSS plus admitted bytes must
    stay within the stage budget and post-allocation headroom must keep the
    reserve. Sampling peaks are sampled observations, not exact instantaneous
    guarantees. Missing optional observations are null, not zero;
    ``MhsResourceMeasurement`` stays per-stage evidence and is never used as
    an aggregate PSS record.
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
    preparation_tree_pss_peak_bytes: int | None = None
    replay_tree_pss_peak_bytes: int | None = None


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
        self._stage: str | None = None
        self._preparation_pss_peak_bytes = -1
        self._replay_pss_peak_bytes = -1

    def set_stage(self, stage: Literal["preparation", "replay"]) -> None:
        """Attribute subsequent resource samples to the current physical phase.

        Args:
            stage: Preparation or replay, without changing financial engine state.

        Returns:
            None; existing global peaks and swap baseline remain continuous.

        Raises:
            ValueError: The phase is unsupported.
        """
        if stage not in ("preparation", "replay"):
            raise ValueError(f"unsupported stage '{stage}'")
        with self._lock:
            if self._stage == stage:
                return
            self._stage = stage
            try:
                me = psutil.Process(os.getpid())
                procs = [me, *me.children(recursive=True)]
            except Exception:  # noqa: BLE001 - observational boundary
                return
            boundary = 0
            seen = False
            for proc in procs:
                try:
                    boundary += int(getattr(proc.memory_full_info(), "pss", 0))
                    seen = True
                except Exception:  # noqa: BLE001, S112 - observational boundary
                    continue
            if not seen:
                return
            self._tree_pss_peak_bytes = max(self._tree_pss_peak_bytes, boundary)
            if stage == "preparation":
                self._preparation_pss_peak_bytes = max(self._preparation_pss_peak_bytes, boundary)
            else:
                self._replay_pss_peak_bytes = max(self._replay_pss_peak_bytes, boundary)

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
            preparation_peak = self._preparation_pss_peak_bytes if self._preparation_pss_peak_bytes >= 0 else None
            replay_peak = self._replay_pss_peak_bytes if self._replay_pss_peak_bytes >= 0 else None
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
                preparation_tree_pss_peak_bytes=preparation_peak,
                replay_tree_pss_peak_bytes=replay_peak,
            )

    def _sample_once(self) -> None:
        with self._lock:
            active = self._stage
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
                if active == "preparation":
                    self._preparation_pss_peak_bytes = max(self._preparation_pss_peak_bytes, pss_sum)
                elif active == "replay":
                    self._replay_pss_peak_bytes = max(self._replay_pss_peak_bytes, pss_sum)
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

@dataclasses.dataclass(frozen=True, slots=True)
class MhsExecutionAllocation:
    """Additional simultaneously live execution bytes. Fixed bytes include per-piece metadata and symbol/state staging; per-bar bytes include all decoded/aligned planes and every bound's temporary arrays; decoder bytes bound projected row-group and conversion transients. Resident history and retained results are measured separately as tree PSS. Components are nonnegative integers and bytes_per_bar is positive."""

    fixed_bytes: int
    bytes_per_bar: int
    decoder_bytes: int

    def __post_init__(self) -> None:
        for name in ("fixed_bytes", "decoder_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        value = self.bytes_per_bar
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"bytes_per_bar must be a positive integer, got {value!r}")


def plan_mhs_execution_bars(
    *,
    requested_bars: int,
    minimum_bars: int,
    allocation: MhsExecutionAllocation,
    budget_bytes: int | None,
    reserve_bytes: int | None,
) -> int:
    """Plan an admissible physical grid without changing logical decisions.

    Args:
        requested_bars: Preferred physical grid size for the logical partition.
        minimum_bars: Smallest grid preserving the next decision and timeout.
        allocation: Conservative additional working-set model.
        budget_bytes: Effective replay tree-PSS limit.
        reserve_bytes: Minimum host and cgroup physical headroom.

    Returns:
        Largest admissible grid size between minimum and requested bars.

    Raises:
        ValueError: Planning dimensions or allocation components are invalid.
        MhsResourceAdmissionError: Required measurements fail or the minimum cannot fit.
    """
    for name, value in (("requested_bars", requested_bars), ("minimum_bars", minimum_bars)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if minimum_bars < 2:
        raise ValueError(f"minimum_bars must preserve at least two completed bars, got {minimum_bars!r}")
    if requested_bars < minimum_bars:
        raise ValueError(
            f"requested_bars={requested_bars} cannot be smaller than minimum_bars={minimum_bars}"
        )
    if not isinstance(allocation, MhsExecutionAllocation):
        raise ValueError(f"allocation must be MhsExecutionAllocation, got {allocation!r}")
    for limit_name, limit_value in (("budget_bytes", budget_bytes), ("reserve_bytes", reserve_bytes)):
        if limit_value is not None and (isinstance(limit_value, bool) or not isinstance(limit_value, int) or limit_value <= 0):
            raise ValueError(f"{limit_name} must be a positive integer or None, got {limit_value!r}")
    if budget_bytes is None and reserve_bytes is None:
        return int(requested_bars)
    current = _current_tree_pss_bytes()
    headroom = _tree_headroom_bytes()
    for bars in range(int(requested_bars), int(minimum_bars) - 1, -1):
        estimated = int(allocation.fixed_bytes) + bars * int(allocation.bytes_per_bar) + int(allocation.decoder_bytes)
        if budget_bytes is not None and current + estimated > budget_bytes:
            continue
        if reserve_bytes is not None and headroom - estimated < reserve_bytes:
            continue
        return int(bars)
    minimum_estimated = int(allocation.fixed_bytes) + int(minimum_bars) * int(allocation.bytes_per_bar) + int(allocation.decoder_bytes)
    if budget_bytes is not None and current + minimum_estimated > budget_bytes:
        error_code: Literal["MEMORY_BUDGET", "MEMORY_RESERVE", "SWAP_GROWTH", "RESOURCE_TELEMETRY"] = "MEMORY_BUDGET"
    else:
        error_code = "MEMORY_RESERVE"
    raise MhsResourceAdmissionError(
        stage="process_execution_plan", error_code=error_code,
        message=f"mhs execution plan rejected: minimum {minimum_bars} bars require {minimum_estimated} bytes "
        f"but tree_pss={current} headroom={headroom} budget={budget_bytes} reserve={reserve_bytes}; "
        "no decoder or plane allocation begins",
    )
