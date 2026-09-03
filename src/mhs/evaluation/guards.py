from __future__ import annotations

import dataclasses

import pandas as pd

from src.application.research.mhs import statistics as _statistics
from src.application.research.mhs.contracts import (
    MhsBookFailure,
    MhsBookReport,
    MhsDiagnosticRequest,
    MhsHorizonDiagnosticReport,
    MhsResearchGoResult,
)
from src.application.research.mhs.research_go import GO_REASON_RESOURCE_BREACH
from src.application.research.mhs.resources import _assert_stage_rss_budget, _StageRecorder
from src.common.errors import DataIntegrityError
from src.mhs.evidence import (
    PhaseDiagnosticResult,
    TailSensitivityResult,
    compute_deployment_readiness,
    required_cost_tiers,
)
from src.mhs.params import FEATURE_NAME
from src.mhs.params import PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H


def _terminal_resource_breach_report(
    request: MhsDiagnosticRequest,
    exc: DataIntegrityError,
    telemetry: _StageRecorder,
    resolved_end: str,
    start: str,
    end: str,
) -> MhsHorizonDiagnosticReport:
    """A serializable terminal rejection for a top-level RSS/RAM-budget breach.

    The MHS-28 terminal-report contract (a resource breach yields a persisted
    terminal ``COMPLETE`` report rather than an uncaught process error) applies
    to the top-level stage barriers too, not just the book replays. When a
    stage-guard ``DataIntegrityError`` carrying an RSS/RAM message escapes the
    body, both top-level books are reported failed with
    ``RESOURCE_BUDGET_BREACH`` and the Research-GO gate carries the same stable
    code. Every heavy replay object is absent (``primary=None``), so persistence
    stays lossless and never fabricates evidence.
    """
    failure = MhsBookFailure(
        stage="resource_budget_guard",
        error_class=type(exc).__name__,
        reason=GO_REASON_RESOURCE_BREACH,
        message=str(exc),
    )
    phase = PhaseDiagnosticResult(
        n_phases=0,
        ensemble_ann=float("nan"),
        ensemble_sharpe=float("nan"),
        mean_phase_ann=float("nan"),
        min_phase_ann=float("nan"),
        max_phase_ann=float("nan"),
        phase_spread_ann=float("nan"),
        degenerate=False,
    )
    tail = TailSensitivityResult(
        base_net_ann=float("nan"),
        base_sharpe=float("nan"),
        winsor_curve={},
        event_window_bars=0,
        event_count=0,
        top1_event_share=0.0,
        top5_event_share=0.0,
        top1pct_events_share=0.0,
        leave_worst_event_out_sharpe=float("nan"),
    )
    base_book = MhsBookReport(
        name="",
        band="",
        horizon_hours=0,
        step_hours=0,
        tranche_count=0,
        n_symbols=0,
        phase=phase,
        prescreen={},
        tail=tail,
        primary=None,
        stress=None,
        primary_autocorr_sharpe=None,
        primary_naive_sharpe=None,
        primary_net_ann=None,
        primary_geometric_cagr=None,
        primary_max_drawdown=None,
        primary_annualized_turnover=None,
        stress_naive_sharpe=None,
        failure=failure,
    )
    books: dict[str, MhsBookReport] = {}
    for bname in ("fast_reversal", "slow_momentum"):
        books[bname] = dataclasses.replace(base_book, name=bname)
    deployment = compute_deployment_readiness(
        pd.Series(
            [1.0, 1.0],
            index=pd.DatetimeIndex([pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(hours=1)]),
        ),
        _PERIODS_PER_YEAR_1H,
        research_go_eligible=False,
        n_bootstrap=_statistics._BOOTSTRAP_REPLICATES,
    )
    research_go = MhsResearchGoResult(
        eligible=False,
        reason_codes=(GO_REASON_RESOURCE_BREACH,),
        evaluated_folds=0,
        folds_passed=0,
        data_integrity_reason_codes=(GO_REASON_RESOURCE_BREACH,),
    )
    return MhsHorizonDiagnosticReport(
        feature=FEATURE_NAME,
        status="COMPLETE",
        start=start,
        end=end,
        resolved_end=resolved_end,
        partition="dev",
        execution_tiers_bps=required_cost_tiers(),
        books=books,
        blend=None,
        blend_target_gross=0.0,
        blend_cash_fraction=1.0,
        eligible_symbols=0,
        trials_attempted=0,
        deflated_sharpe_ratio=None,
        xs_rank_ic={},
        date_clustered_regression={},
        horizon_diagnostics={},
        bootstrap_ci=None,
        placebo_sharpe_percentile=None,
        deployment_readiness=deployment,
        synthetic_stress={},
        participation_warnings={},
        termination_counts={},
        unsupported_assumptions=(),
        anchored_folds=(),
        folds=(),
        research_go=research_go,
        fill_source="NOT_RUN_NO_EXECUTION_DATA",
        mark_source="NOT_RUN_NO_EXECUTION_DATA",
        execution_timeframe=request.execution_timeframe,
        execution_universe_size=request.execution_universe_size,
        execution_symbols=(),
        run_elapsed_seconds=0.0,
        resource_measurements=telemetry.records,
        realized_execution_roster_size=None,
    )


def _guard_stage_or_breach(
    stage: str,
    budget_bytes: int | None,
    reserve_bytes: int | None,
    request: MhsDiagnosticRequest,
    telemetry: _StageRecorder,
    resolved_end: str,
    start: str,
    end: str,
) -> MhsHorizonDiagnosticReport | None:
    """Run a top-level stage RSS barrier, converting a resource breach to a terminal report.

    Returns the terminal rejection report when the barrier detects an
    RSS/RAM-budget breach (MHS-28 fail-closed contract); ``None`` when the
    barrier passes. Any other ``DataIntegrityError`` is re-raised unchanged.
    """
    try:
        _assert_stage_rss_budget(stage, budget_bytes, reserve_bytes)
    except DataIntegrityError as exc:
        message = str(exc).lower()
        if "rss budget" in message or "ram budget" in message or "reserve" in message:
            return _terminal_resource_breach_report(
                request, exc, telemetry, resolved_end, start, end,
            )
        raise
    return None



