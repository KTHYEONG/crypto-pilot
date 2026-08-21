"""Shared stage runner for the MHS horizon diagnostic.

Both ``run_mhs_horizon_diagnostic`` (``MhsDiagnosticRequest`` entry) and
``run_mhs_diagnostic`` (``MhsRunConfig`` entry) build a ``PipelineContext`` and
drive the same six stage functions in the original computation order. Stage
functions are imported lazily so there is no import-time cycle with
``evaluation.py`` (which the stage modules import from).
"""

from __future__ import annotations

from typing import cast

from src.mhs.pipeline.context import PipelineContext
from src.mhs.report.schema import MhsHorizonDiagnosticReport
from src.mhs.telemetry import StageTelemetry


def run_stages(ctx: PipelineContext, telemetry: StageTelemetry) -> MhsHorizonDiagnosticReport:
    """Run the six stages in order; return the assembled report or a terminal report.

    Each stage reads/writes ``ctx`` and sets ``ctx._terminal_report`` (and returns)
    when a RAM-budget guard breaches, allowing the caller to short-circuit.
    """
    from src.mhs.pipeline.stages.assemble import assemble_report
    from src.mhs.pipeline.stages.book import build_books
    from src.mhs.pipeline.stages.committee import build_committee
    from src.mhs.pipeline.stages.fold import run_folds
    from src.mhs.pipeline.stages.panel import load_panel
    from src.mhs.pipeline.stages.replay import run_replays
    from src.mhs.pipeline.stages.selection import select_horizons

    load_panel(ctx, telemetry)
    if ctx._terminal_report is not None:
        return cast(MhsHorizonDiagnosticReport, ctx._terminal_report)
    select_horizons(ctx, telemetry)
    if ctx._terminal_report is not None:
        return ctx._terminal_report
    build_books(ctx, telemetry)
    if ctx._terminal_report is not None:
        return ctx._terminal_report
    build_committee(ctx, telemetry)
    if ctx._terminal_report is not None:
        return ctx._terminal_report
    run_replays(ctx, telemetry)
    if ctx._terminal_report is not None:
        return ctx._terminal_report
    run_folds(ctx, telemetry)
    if ctx._terminal_report is not None:
        return cast(MhsHorizonDiagnosticReport, ctx._terminal_report)
    return assemble_report(ctx, telemetry)
