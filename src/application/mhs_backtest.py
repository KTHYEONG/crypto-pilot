"""Source-owned continuous-process backtest service."""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.mhs.backtest.contracts import ProcessInventoryBacktestError, ProcessInventoryReport
from src.mhs.backtest.inventory import evaluate_process_inventory_backtest
from src.mhs.params import PROCESS_EVALUATION_CEILING
from src.mhs.process import ProcessExecutionPolicy
from src.mhs.reporting.inventory import persist_process_inventory_failure, persist_process_inventory_report
from src.mhs.reporting.process import persist_process_targets
from src.mhs.resources import MhsMemoryBudget, resolve_mhs_memory_budget

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MhsBacktestRequest:
    """Inputs and one private fresh domain-result destination for a 3-minute evaluation."""

    start: pd.Timestamp
    end: pd.Timestamp
    data_root: str | None
    result_output: Path
    targets_output: Path | None = None
    tracking_error_threshold: float | None = None
    memory_budget: MhsMemoryBudget | None = None
    evidence_root: Path | None = None
    registry_path: Path | None = None
    run_id: str | None = None


def _validate_managed_run(registry_path: Path, run_id: str) -> None:
    """Require managed publication context to match an already registered run."""
    if (
        not isinstance(run_id, str)
        or len(run_id) != 32
        or any(c not in "0123456789abcdefABCDEF" for c in run_id)
    ):
        raise ValueError(f"run_id must be UUID hex, got {run_id!r}")
    if not registry_path.is_file():
        raise ValueError(f"managed registry is unavailable: {registry_path}")
    conn = sqlite3.connect(str(registry_path), timeout=5.0)
    try:
        registered = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    except sqlite3.Error as exc:
        raise ValueError(f"managed registry is unavailable: {registry_path}") from exc
    finally:
        conn.close()
    if registered is None:
        raise ValueError(f"managed run is not registered: {run_id!r}")


def validate_mhs_backtest_request(request: MhsBacktestRequest) -> None:
    """Validate evaluation controls and distinct evidence destinations.

    Args:
        request: Typed UTC evaluation controls and evidence paths.
    Returns:
        None when dates, controls and paths satisfy the run contract.
    Raises:
        ValueError: A control or destination is invalid, conflicting or occupied.
    """
    if not isinstance(request, MhsBacktestRequest):
        raise ValueError(f"request must be MhsBacktestRequest, got {request!r}")
    for label, value in (("start", request.start), ("end", request.end)):
        if not isinstance(value, pd.Timestamp) or value.tzinfo is None:
            raise ValueError(f"{label} must be a timezone-aware Timestamp, got {value!r}")
    start = request.start.tz_convert("UTC")
    end = request.end.tz_convert("UTC")
    if start >= end:
        raise ValueError("start must precede end")
    if end > PROCESS_EVALUATION_CEILING:
        raise ValueError(f"end {end} exceeds PROCESS_EVALUATION_CEILING")
    ProcessExecutionPolicy(tracking_error_threshold=request.tracking_error_threshold)
    for label, candidate, suffix in (("result_output", request.result_output, ".json"),):
        if not isinstance(candidate, Path) or candidate.suffix != suffix:
            raise ValueError(f"{label} must be a {suffix} path, got {candidate!r}")
    if request.targets_output is not None and (
        not isinstance(request.targets_output, Path) or request.targets_output.suffix != ".parquet"
    ):
        raise ValueError(f"targets_output must be a parquet path, got {request.targets_output!r}")
    managed = (request.evidence_root, request.registry_path, request.run_id)
    if any(item is None for item in managed) and not all(item is None for item in managed):
        raise ValueError("managed publication context must be fully provided or all None")
    if all(item is not None for item in managed):
        assert request.evidence_root is not None
        assert request.registry_path is not None
        assert request.run_id is not None
        if not isinstance(request.evidence_root, Path):
            raise ValueError(f"evidence_root must be a Path or None, got {request.evidence_root!r}")
        if not isinstance(request.registry_path, Path):
            raise ValueError(f"registry_path must be a Path or None, got {request.registry_path!r}")
        _validate_managed_run(request.registry_path, request.run_id)
    candidates: list[tuple[str, Path]] = [("result_output", request.result_output)]
    if request.targets_output is not None:
        candidates.append(("targets_output", request.targets_output))
    seen: set[Path] = set()
    for label, candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            raise ValueError(f"{label} must be distinct from other destinations")
        seen.add(resolved)
        if os.path.lexists(candidate):
            raise ValueError(f"{label} must be fresh: {candidate} already exists")


def execute_mhs_backtest(request: MhsBacktestRequest) -> ProcessInventoryReport:
    """Persist the production three-minute inventory outcome without masking failure.

    Args:
        request: Explicit dates, strategy control, budget and fresh evidence paths.
    Returns:
        Completed base/stress inventory results after primary persistence.
    Raises:
        ValueError: Request validation fails before execution.
        ProcessInventoryBacktestError: Evaluation fails; dedicated diagnostics are
            persisted when possible and the original chained cause is preserved.
        OSError: Result or target persistence fails.
    """
    validate_mhs_backtest_request(request)
    start = request.start.tz_convert("UTC")
    end = request.end.tz_convert("UTC")
    policy = ProcessExecutionPolicy(tracking_error_threshold=request.tracking_error_threshold)
    try:
        report = evaluate_process_inventory_backtest(
            start,
            end,
            data_root=request.data_root,
            execution_policy=policy,
            memory_budget=resolve_mhs_memory_budget(request.memory_budget),
        )
    except ProcessInventoryBacktestError as exc:
        try:
            persist_process_inventory_failure(exc.report, request.result_output)
        except Exception as persist_exc:  # noqa: BLE001
            _logger.exception(
                "[EVAL] status=failed stage=%s error_code=%s failure_path=%s persist_error=%s",
                exc.report.stage,
                exc.report.error_code,
                request.result_output,
                persist_exc,
            )
            raise exc from persist_exc
        raise
    persist_process_inventory_report(
        report, request.result_output,
        evidence_root=request.evidence_root,
        registry_path=request.registry_path,
        run_id=request.run_id,
    )
    if request.targets_output is not None:
        persist_process_targets(report.proxy.base, request.targets_output)
    return report
