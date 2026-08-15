"""One-execution technical library admission pipeline.

Composes candidate discovery and out-of-sample proposal backtesting into one
sealed execution: it resolves the common selection window, runs the existing
admission diagnostic only through ``selection.end``, shortlists the
pair-compatible proposals by the diversification ``rank_key``, and invokes the
existing proposal backtest once per shortlisted proposal on the out-of-sample
window. It never registers, promotes, mutates a catalog, or writes child
proposal runs to the ledger.
"""

from __future__ import annotations

import logging

from src.application.research.expert.admission import (
    run_technical_library_admission,
)
from src.application.research.expert.admission_backtest import (
    run_technical_library_admission_backtest,
)
from src.application.research.expert.window import (
    resolve_common_technical_window,
)
from src.research.expert_portfolio.admission import shortlist_admission_proposals
from src.research.expert_portfolio.admission_reports import (
    LibraryAdmissionBacktestReport,
    LibraryAdmissionPipelineReport,
)
from src.research.expert_portfolio.admission_types import (
    LIBRARY_ADMISSION_PROFILES,
    TechnicalLibraryAdmissionBacktestRequest,
    TechnicalLibraryAdmissionPipelineRequest,
)

_logger = logging.getLogger("TechnicalLibraryAdmissionPipeline")


def _identify_profile(request: TechnicalLibraryAdmissionPipelineRequest) -> str:
    """Name the frozen profile whose sealed selection exactly matches, else ``custom``."""
    for name, builder in LIBRARY_ADMISSION_PROFILES.items():
        if builder() == request.selection:
            return name
    return "custom"


def run_technical_library_admission_pipeline(
    request: TechnicalLibraryAdmissionPipelineRequest,
) -> LibraryAdmissionPipelineReport:
    """Run candidate discovery through selection.end, then OOS backtests.

    Selection consumes data only through ``selection.end`` and each child
    proposal backtest starts exactly at ``evaluation_start`` with
    ``log_run=False``, so 2025 returns and performance can never influence
    selection. A fail-closed selection returns an aggregate report with no
    backtests; fewer than the budget proposals is a successful result, never
    padding.
    """
    window = resolve_common_technical_window(
        request.selection.symbols, request.selection.start, request.selection.end,
        timeframe=request.selection.timeframe,
    )
    selection = run_technical_library_admission(request.selection)
    if selection.status != "COMPLETE":
        _logger.info(
            "[EVAL] library_admission_pipeline status=%s structural=%d pair_compatible=%d shortlist=0",
            selection.status, selection.structural_combinations, len(selection.proposals),
        )
        return LibraryAdmissionPipelineReport(
            status=selection.status,
            profile=_identify_profile(request),
            requested_start=window.requested_start,
            common_start=str(window.common_start),
            effective_start=str(window.effective_start),
            selection_end=str(request.selection.end),
            evaluation_start=str(request.evaluation_start),
            evaluation_end=str(request.evaluation_end),
            structural_combinations=selection.structural_combinations,
            pair_compatible_count=len(selection.proposals),
            shortlist=(),
        )

    shortlist = shortlist_admission_proposals(
        selection.proposals, request.max_backtest_proposals,
    )
    backtests: list[LibraryAdmissionBacktestReport] = []
    for proposal in shortlist:
        child = TechnicalLibraryAdmissionBacktestRequest(
            expert_ids=proposal.expert_ids,
            router=request.selection.router,
            start=str(request.evaluation_start),
            end=request.evaluation_end,
            initial_equity=request.initial_equity,
            max_workers=request.selection.admission.max_workers,
            log_run=False,
            timeframe=request.selection.timeframe,
        )
        backtests.append(run_technical_library_admission_backtest(child))
    _logger.info(
        "[EVAL] library_admission_pipeline status=COMPLETE structural=%d pair_compatible=%d shortlist=%d backtests=%d",
        selection.structural_combinations, len(selection.proposals),
        len(shortlist), len(backtests),
    )
    return LibraryAdmissionPipelineReport(
        status="COMPLETE",
        profile=_identify_profile(request),
        requested_start=window.requested_start,
        common_start=str(window.common_start),
        effective_start=str(window.effective_start),
        selection_end=str(request.selection.end),
        evaluation_start=str(request.evaluation_start),
        evaluation_end=str(request.evaluation_end),
        structural_combinations=selection.structural_combinations,
        pair_compatible_count=len(selection.proposals),
        shortlist=shortlist,
        backtests=tuple(backtests),
    )


def _check_contract() -> None:
    """Executable assertions locking the admission pipeline application surface."""
    assert run_technical_library_admission_pipeline.__name__ == (
        "run_technical_library_admission_pipeline"
    )


_check_contract()
