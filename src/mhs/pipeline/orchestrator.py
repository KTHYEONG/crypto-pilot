"""MHS pipeline: orchestration layer.

``run_mhs_diagnostic`` composes the six stage functions plus report assembly.
Each stage is a function with signature ``(ctx: PipelineContext, telemetry:
StageTelemetry) -> None`` (``assemble_report`` returns the report) that reads and
writes a shared ``PipelineContext`` -- the single communication channel for
long-lived state. The orchestrator handles setup and wiring only (<=150 lines).
"""

from __future__ import annotations

import dataclasses
import time

import pandas as pd

from src.application.research.mhs.evaluation import (
    DISCOVERY_START,
    HOLDOUT_CUTOFF,
    _get_symbol_mark_frame,
    resolve_evaluation_end,
)
from src.mhs.params import MHS_FINAL_OOS_CUTOFF_2026H1
from src.application.research.mhs.resources import _TreeMemorySampler
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

    A ``_TreeMemorySampler`` observes the whole process tree for the duration
    of the run; its COW-correct PSS/USS/available-floor stats are attached to
    the report as ``tree_memory`` (observational, never raises into the run).
    """
    _get_symbol_mark_frame.cache_clear()
    _evaluation_ceiling = (
        MHS_FINAL_OOS_CUTOFF_2026H1 if config.final_oos_2026h1 else HOLDOUT_CUTOFF
    )
    resolved_end = resolve_evaluation_end(config.end, unseal_holdout=config.final_oos_2026h1, ceiling=_evaluation_ceiling)
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

    end = resolved_end
    if end > _evaluation_ceiling:
        raise RuntimeError(f"Holdout sealed: requested end {end} past {_evaluation_ceiling}")

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
    tree_sampler = _TreeMemorySampler()
    try:
        tree_sampler.start()
        report = run_stages(ctx, telemetry)
    finally:
        tree_stats = tree_sampler.stop()
    return dataclasses.replace(report, tree_memory=tree_stats)
