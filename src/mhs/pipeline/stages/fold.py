"""S7: Anchored purged fold pool + post-book concurrent diagnostics.

Extracted verbatim from ``evaluation.py`` lines 4061-4172 (``_run_post_book_concurrently``
call, post-fold committee/multi-feature diagnostic opt-ins, deflated-sharpe
evidence, fold blend parity / growth concentration, the research-GO gate, and
the deployment-readiness patch). Calls the existing isolated functions unchanged;
only variable threading via ``ctx`` moves. The ``del eligible`` / ``del opens,
bar_funding`` / ``del funding_window, minute_grid`` + ``gc.collect()`` at 4169-4172
are preserved at the end of this function.
"""

from __future__ import annotations

import dataclasses
import gc

import pandas as pd

from src.application.research.mhs.evaluation import (
    _PERIODS_PER_YEAR_1H,
    COMMITTEE_MEMBERS,
    FEATURE_REGISTRY,
    SEARCH_TRIALS_ATTEMPTED,
    DataIntegrityError,
    _assert_stage_rss_budget,
    _committee_diagnostic,
    _fold_blend_parity,
    _fold_growth_concentration,
    _get_symbol_mark_frame,
    _guard_stage_or_breach,
    _load_feature_panels,
    _multi_feature_diagnostic,
    _research_go,
    _run_post_book_concurrently,
    _statistics,
    compute_deployment_readiness,
    feature_registry_panel_columns,
)
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry


def run_folds(ctx: PipelineContext, telemetry: StageTelemetry) -> None:
    """Run the fold pool and all post-book statistical diagnostics."""
    ctx.trials_attempted = SEARCH_TRIALS_ATTEMPTED
    ctx.deflated_sharpe_ratio = None

    ctx.bootstrap_ci = None
    ctx.placebo_percentile = None
    ctx.participation = {}
    ctx.termination_counts = {}
    ctx.unsupported = (
        "partial_fill", "queue_position", "post_only_rejection",
        "cancel_replace_latency", "order_size_impact",
    )

    # Folds, statistical diagnostics, and deployment readiness are independent
    # post-book streams: the fold pool runs in fork workers while a background
    # thread computes the diagnostics + deployment tail (spec Phase 3, P14).
    # The top-level feature matrices stay alive through that thread and are
    # released after it joins so the wide multi-year panels never coexist with
    # the final assembly.
    (
        ctx.bootstrap_ci, ctx.placebo_percentile, ctx.participation, ctx.termination_counts,
        fold_reports, ctx.deployment,
    ) = _run_post_book_concurrently(
        ctx.blend_report, ctx.root, ctx.config, ctx.execution_symbols, ctx.minute_grid,
        ctx.signal_48h, ctx.eligible, ctx.opens, ctx.bar_funding, ctx.grid_1h, ctx.fast,
        ctx.fold_funding, ctx.initial_equity, ctx.recorder, ctx.fold_slow_horizons, ctx.fold_fast_horizons,
        ctx.fold_funding_carry, ctx._fold_committee_weights,
    )
    ctx.folds = tuple(fold_reports)
    # Free mark frame cache so opt-in diagnostics run with minimal parent memory.
    _get_symbol_mark_frame.cache_clear()
    gc.collect()
    _terminal = _guard_stage_or_breach(
        "post_folds", ctx.rss_budget_bytes, ctx.rss_reserve_bytes,
        ctx.config, ctx.recorder, str(ctx.resolved_end), str(ctx.start), str(ctx.end),
    )
    if _terminal is not None:
        ctx._terminal_report = _terminal
        return
    if ctx.config.multi_feature_book or ctx.config.committee_book:
        if ctx.config.multi_feature_book:
            _diag_panel_columns = feature_registry_panel_columns(FEATURE_REGISTRY)
        else:
            _diag_panel_columns = feature_registry_panel_columns(
                [
                    spec for spec in FEATURE_REGISTRY
                    if spec.name in set(COMMITTEE_MEMBERS)
                ],
            )
        _diag_panels = _load_feature_panels(
            ctx.root, ctx.start, ctx.end, ctx.grid_1h, ctx.aligned_symbols, columns=_diag_panel_columns,
        )
        ctx.recorder.record("diagnostic_feature_panels")
        _assert_stage_rss_budget("diagnostic_feature_panels", ctx.rss_budget_bytes, ctx.rss_reserve_bytes)
        if ctx.config.committee_book:
            ctx.committee_diagnostic = _committee_diagnostic(
                ctx.root, ctx.start, ctx.end, ctx.grid_1h, ctx.aligned_symbols, ctx.execution_mask, ctx.opens,
                ctx.bar_funding, panels=_diag_panels,
                rss_budget_bytes=ctx.rss_budget_bytes,
                rss_reserve_bytes=ctx.rss_reserve_bytes,
                telemetry=ctx.recorder,
                sizing_mode="kelly_blend" if ctx.config.committee_kelly_sizing else "vol_target",
                growth_diagnostic=ctx.config.committee_growth_diagnostic,
            )
            ctx.recorder.record("committee_diagnostic")
        if ctx.config.multi_feature_book:
            ctx.multi_feature_diagnostic = _multi_feature_diagnostic(
                ctx.root, ctx.start, ctx.end, ctx.grid_1h, ctx.aligned_symbols, ctx.execution_mask, ctx.opens,
                ctx.bar_funding, panels=_diag_panels,
                rss_budget_bytes=ctx.rss_budget_bytes,
                rss_reserve_bytes=ctx.rss_reserve_bytes,
                telemetry=ctx.recorder,
            )
            ctx.recorder.record("multi_feature_diagnostic")
        del _diag_panels
        gc.collect()
    ctx.deflated_sharpe_ratio = _statistics._deflated_sharpe_evidence(
        ctx.blend_report, ctx.folds, ctx.trials_attempted,
    )
    ctx.fold_blend_parity, parity_reasons = _fold_blend_parity(ctx.blend_traces, ctx.folds)
    ctx.fold_growth_concentration, concentration_reasons = _fold_growth_concentration(ctx.folds)
    ctx.research_go = _research_go._mhs_research_go(
        ctx.folds, ctx.book_reasons, parity_reasons + concentration_reasons,
        blend_primary_max_drawdown=(
            ctx.blend_report.primary_max_drawdown if ctx.blend_report is not None else None
        ),
    )

    if ctx.blend_report is not None and ctx.blend_report.primary is not None:
        if ctx.minute_grid is None:
            raise DataIntegrityError("blend report requires a minute replay grid")
        # The deployment tail was computed with ``research_go_eligible=None``;
        # patch in the fold-derived gate decision now that it is resolved.
        assert ctx.deployment is not None
        ctx.deployment = dataclasses.replace(
            ctx.deployment, research_go_eligible=ctx.research_go.eligible,
        )
        ctx.recorder.record(
            "blend_participation",
            fill_count=len(ctx.blend_report.primary.simulated_fills),
        )
        ctx.recorder.record("statistical_diagnostics")
    else:
        ctx.deployment = compute_deployment_readiness(
            pd.Series(
                [1.0, 1.0],
                index=pd.DatetimeIndex([ctx.start, ctx.start + pd.Timedelta(hours=1)]),
            ),
            _PERIODS_PER_YEAR_1H,
            research_go_eligible=ctx.research_go.eligible,
            n_bootstrap=_statistics._BOOTSTRAP_REPLICATES,
        )

    del ctx.eligible
    del ctx.opens, ctx.bar_funding
    del ctx.funding_window, ctx.minute_grid
    gc.collect()
