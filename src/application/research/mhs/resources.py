"""RAM-budget guard primitives (I4 seam: current_rss_bytes)."""

from __future__ import annotations

import logging
import time

import psutil

from src.application.research.mhs.contracts import MhsResourceMeasurement
from src.common.errors import DataIntegrityError
from src.mhs.contracts import (
    MHS_RAM_BUDGET_FRACTION,
    MHS_RAM_RESERVE_FLOOR_BYTES,
    MHS_RAM_RESERVE_FRACTION,
)

_logger = logging.getLogger("MhsHorizonDiagnostic")

def _current_rss_bytes() -> int:
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        return -1


def _resolve_ram_budget(
    max_rss_bytes: int | None,
    ram_guard: bool,
) -> tuple[int | None, int | None]:
    """Resolve the automatic RAM-guard budget and reserve from the environment.

    Returns ``(budget_bytes, reserve_bytes)``. With ``ram_guard=False`` both are
    ``None`` (the legacy unlimited semantics). With the guard on, the budget is
    ``max_rss_bytes`` when explicitly set, otherwise ``int(total *
    MHS_RAM_BUDGET_FRACTION)``, and the reserve is
    ``max(int(total * MHS_RAM_RESERVE_FRACTION), MHS_RAM_RESERVE_FLOOR_BYTES)``.
    A psutil failure or a non-positive total yields ``(None, None)`` -- an
    observational failure disables the guard and never alters computed values.
    """
    if not ram_guard:
        return (None, None)
    try:
        total = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        return (None, None)
    if total <= 0:
        return (None, None)
    budget = (
        max_rss_bytes
        if max_rss_bytes is not None
        else int(total * MHS_RAM_BUDGET_FRACTION)
    )
    reserve = max(int(total * MHS_RAM_RESERVE_FRACTION), MHS_RAM_RESERVE_FLOOR_BYTES)
    return (budget, reserve)


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
    ``MHS_GO_REASON_RESOURCE_BREACH`` -- the fork-worker OOM guard (only the
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

    @property
    def records(self) -> tuple[MhsResourceMeasurement, ...]:
        return tuple(self._records)

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


def _peak_rss_bytes(
    resource_measurements: tuple[MhsResourceMeasurement, ...],
) -> int | None:
    return max((m.rss_bytes for m in resource_measurements), default=None)
