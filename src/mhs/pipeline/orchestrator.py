"""MHS pipeline: orchestration layer.

``run_mhs_diagnostic`` composes the six stage functions plus report assembly.
Each stage is a function with signature ``(ctx: PipelineContext, telemetry:
StageTelemetry) -> None`` (``assemble_report`` returns the report) that reads and
writes a shared ``PipelineContext`` -- the single communication channel for
long-lived state. The orchestrator handles setup and wiring only (<=150 lines).
"""

from __future__ import annotations

import time

import pandas as pd

from src.application.research.mhs.evaluation import (
    DISCOVERY_START,
    HOLDOUT_CUTOFF,
    _get_symbol_mark_frame,
    resolve_evaluation_end,
)
from src.mhs.pipeline.config import MhsRunConfig
from src.mhs.pipeline.context import PipelineContext
from src.mhs.pipeline.runner import run_stages
from src.mhs.report.schema import MhsHorizonDiagnosticReport
from src.mhs.telemetry import StageTelemetry


def run_mhs_diagnostic(config: MhsRunConfig) -> MhsHorizonDiagnosticReport:
    """Compose the dev-only MHS diagnostic: six stages + report assembly.

    Constructs a ``PipelineContext`` from the run config and drives the stage
    functions in the original computation order (I-IDENTITY). The previous
    delegation to the monolithic ``run_mhs_horizon_diagnostic`` has been
    removed -- this function is now the sole composition point.
    """
    _get_symbol_mark_frame.cache_clear()
    resolved_end = resolve_evaluation_end(config.end, unseal_holdout=False)
    _run_start = time.perf_counter()
    if config.partition != "dev":
        raise RuntimeError(
            "MHS Phase 1 is dev-only; the holdout partition requires an "
            "architecture-freeze final-OOS command"
        )
    if config.start is not None:
        start = pd.Timestamp(config.start)
        start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    else:
        start = DISCOVERY_START

    if resolved_end is not None:
        end = pd.Timestamp(resolved_end)
        end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
    else:
        end = HOLDOUT_CUTOFF
    if end > HOLDOUT_CUTOFF:
        raise RuntimeError(f"Holdout sealed: requested end {end} past {HOLDOUT_CUTOFF}")

    ctx = PipelineContext(
        config=config,
        resolved_end=resolved_end,
        start=start,
        end=end,
        rss_budget_bytes=None,
        rss_reserve_bytes=None,
        root="",
        grid_1h=pd.DatetimeIndex([]),
        close=pd.DataFrame(),
        opens=pd.DataFrame(),
        quote_vol=pd.DataFrame(),
        taker_buy_quote=None,
        symbols=[],
    )
    ctx.run_start = _run_start
    telemetry = StageTelemetry(log_run=config.log_run)
    return run_stages(ctx, telemetry)
