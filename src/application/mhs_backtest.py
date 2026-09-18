"""Source-owned continuous-process backtest service."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.mhs.params import PROCESS_EVALUATION_CEILING
from src.mhs.process import ProcessExecutionPolicy
from src.mhs.process_backtest import (
    PROCESS_POLICY_REPORT_PATH,
    PROCESS_REPORT_PATH,
    ProcessInventoryBacktestError,
    ProcessInventoryReport,
    evaluate_process_inventory_backtest,
    persist_process_inventory_failure,
    persist_process_inventory_report,
    persist_process_targets,
)
from src.mhs.resources import MhsMemoryBudget, resolve_mhs_memory_budget

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MhsBacktestRequest:
    """Explicit continuous-process evaluation and evidence destinations.

    Dates bound source preparation and preserve the registered evaluation ceiling.
    Three-minute inventory accounting is primary evidence; hourly proxy state is
    comparative and does not confer deployment eligibility.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    data_root: str | None
    output: Path
    failure_output: Path
    targets_output: Path | None = None
    tracking_error_threshold: float | None = None
    memory_budget: MhsMemoryBudget | None = None


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
    for label, candidate, suffix in (
        ("output", request.output, ".json"),
        ("failure_output", request.failure_output, ".json"),
    ):
        if not isinstance(candidate, Path) or candidate.suffix != suffix:
            raise ValueError(f"{label} must be a {suffix} path, got {candidate!r}")
    if request.targets_output is not None and (
        not isinstance(request.targets_output, Path) or request.targets_output.suffix != ".parquet"
    ):
        raise ValueError(f"targets_output must be a parquet path, got {request.targets_output!r}")
    reserved = {PROCESS_REPORT_PATH.resolve(), PROCESS_POLICY_REPORT_PATH.resolve()}
    candidates: list[tuple[str, Path]] = [("output", request.output), ("failure_output", request.failure_output)]
    if request.targets_output is not None:
        candidates.append(("targets_output", request.targets_output))
    seen: set[Path] = set()
    for label, candidate in candidates:
        resolved = candidate.resolve()
        if resolved in reserved:
            raise ValueError(f"{label} conflicts with reserved evidence")
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
            persist_process_inventory_failure(exc.report, request.failure_output)
        except Exception as persist_exc:  # noqa: BLE001
            _logger.exception(
                "[EVAL] status=failed stage=%s error_code=%s failure_path=%s persist_error=%s",
                exc.report.stage,
                exc.report.error_code,
                request.failure_output,
                persist_exc,
            )
            raise exc
        raise
    persist_process_inventory_report(report, request.output)
    if request.targets_output is not None:
        persist_process_targets(report.proxy.base, request.targets_output)
    return report
