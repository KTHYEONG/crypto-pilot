"""Layer C composition root. Sits outside the evaluation package."""

from __future__ import annotations

import time

import pandas as pd

from src.application.research.mhs.contracts import MhsDiagnosticRequest, MhsHorizonDiagnosticReport
from src.application.research.mhs.marks import _get_symbol_mark_frame
from src.mhs.pipeline.context import PipelineContext
from src.mhs.pipeline.runner import run_stages
from src.mhs.telemetry import StageTelemetry
from src.mhs.types import DISCOVERY_START
from src.research.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end


def run_mhs_horizon_diagnostic(request: MhsDiagnosticRequest) -> MhsHorizonDiagnosticReport:
    """Compose the dev-only Phase 1 diagnostic: pre-screen + strict-proxy evidence.

    Forces ``partition='dev'`` and resolves the sealed evaluation end; a holdout
    partition or an end past ``HOLDOUT_CUTOFF`` raises ``RuntimeError``.

    Decomposed into the six pipeline stages (``src/mhs/pipeline/stages``) driven
    by ``run_stages``; this wrapper builds the shared ``PipelineContext`` from the
    request and returns the assembled report (byte-identical to the monolith).
    """
    _get_symbol_mark_frame.cache_clear()
    resolved_end = resolve_evaluation_end(request.end, unseal_holdout=False)
    _run_start = time.perf_counter()
    if request.partition != "dev":
        raise RuntimeError(
            "MHS Phase 1 is dev-only; the holdout partition requires an "
            "architecture-freeze final-OOS command"
        )
    if request.start is not None:
        start = pd.Timestamp(request.start)
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
        config=request,
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
    telemetry = StageTelemetry(log_run=request.log_run)
    return run_stages(ctx, telemetry)
