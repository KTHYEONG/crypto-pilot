"""Process backtest domain contracts shared by maintained evaluation stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.contracts import MhsResourceMeasurement
from src.mhs.deploy_gate import DeployGateResult
from src.mhs.execution.contracts import ExecutionDataGap, FundingCoverageGap, StrategyExecutionReplayResult
from src.mhs.process import ProcessExecutionPolicy, ProcessRiskSizingSpec, RefitPoint
from src.mhs.resources import ProcessTreeMemoryStats

PROCESS_CERTIFICATION_LEVEL: str = "process_proxy_1h_ledger"

@dataclass(frozen=True, slots=True)
class ProcessMarketData:
    """Causally aligned inputs shared by every cost tier of one process run."""

    grid_1h: pd.DatetimeIndex
    decision_grid: pd.DatetimeIndex
    opens_1h: pd.DataFrame
    bar_funding_1h: pd.DataFrame
    log_close_step: pd.DataFrame
    funding_step: pd.DataFrame
    member_books: dict[str, pd.DataFrame]
    execution_mask: pd.DataFrame


@dataclass(frozen=True, slots=True)
class RefitRecord:
    """Audit record of one refit's decisions."""

    point: RefitPoint
    member_weights: dict[str, float]
    smoothing_halflife_days: float


@dataclass(frozen=True, slots=True)
class ProcessPath:
    """One cost tier's continuous proxy path and its exact execution targets.

    Unit targets precede volatility sizing; target weights are the sized
    decision rows actually evaluated. Keeping them prevents a later
    inventory replay from rebuilding a different strategy from summary
    statistics. Policy and targets are identical across cost tiers.
    """

    one_way_bps: float
    daily_returns: pd.Series
    unit_daily_returns: pd.Series
    exposure: pd.Series
    refits: tuple[RefitRecord, ...]
    leverage_cap: float
    execution_policy: ProcessExecutionPolicy
    unit_target_weights: pd.DataFrame
    target_weights: pd.DataFrame
    turnover_1h: pd.Series
    risk_sizing: ProcessRiskSizingSpec | None = None


@dataclass(frozen=True, slots=True)
class ProcessBacktestReport:
    """Proxy evidence for the continuous process; ``gate`` is never a deploy verdict."""

    start: pd.Timestamp
    end: pd.Timestamp
    certification_level: str
    n_candidates: int
    base: ProcessPath
    stress: ProcessPath
    gate: DeployGateResult


@dataclass(frozen=True, slots=True)
class ProcessInventoryReport:
    """Production 3m inventory evidence for identical process decisions."""

    proxy: ProcessBacktestReport
    base: StrategyExecutionReplayResult
    stress: StrategyExecutionReplayResult
    gate: DeployGateResult
    resource_measurements: tuple[MhsResourceMeasurement, ...]
    memory_stats: ProcessTreeMemoryStats
    funding_coverage_gaps: tuple[FundingCoverageGap, ...] = ()


@dataclass(frozen=True, slots=True)
class ProcessInventoryFailureReport:
    """Failed production three-minute evaluation evidence.
Completed coverage means decisions consumed by every required bound, not merely
validated inputs. Unknown totals or resource observations remain null. Partial
coverage and observed source gaps are diagnostics, never completed performance,
deployment certification or proof of operating-system OOM."""

    status: Literal["failed"]
    start: pd.Timestamp
    end: pd.Timestamp
    data_root: str | None
    execution_policy: ProcessExecutionPolicy
    stage: str
    error_code: Literal[
        "MEMORY_BUDGET", "MEMORY_RESERVE", "SWAP_GROWTH",
        "RESOURCE_TELEMETRY", "DATA_INTEGRITY", "UNEXPECTED_ERROR",
    ]
    error_type: str
    error_message: str
    total_decisions: int | None
    validated_decisions: int
    completed_decisions: int
    completed_windows: int
    completed_decision_start: pd.Timestamp | None
    completed_decision_end: pd.Timestamp | None
    source_gaps: tuple[ExecutionDataGap, ...]
    source_gap_excluded_symbols: tuple[str, ...]
    resource_measurements: tuple[MhsResourceMeasurement, ...]
    memory_stats: ProcessTreeMemoryStats | None
    funding_coverage_gaps: tuple[FundingCoverageGap, ...] = ()


class ProcessInventoryBacktestError(DataIntegrityError):
    """Carry typed failure evidence while preserving fail-closed callers."""

    report: ProcessInventoryFailureReport

    def __init__(self, report: ProcessInventoryFailureReport) -> None:
        """Carry typed failure evidence while preserving fail-closed callers.

        Args:
            report: Observed failed-run coverage, provenance and resources.

        Returns:
            None; the original evaluation exception remains the chained cause.
        """
        super().__init__(report.error_message)
        self.report = report
