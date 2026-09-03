"""S3: Book weights + execution roster + beta/regime/sleeve + blend.

Extracted verbatim from ``evaluation.py`` lines 3656-3755 (fast/slow book
weights, PIT execution roster, execution_coverage_gate dynamic exclusion,
horizon-ensemble execution weights, beta neutralization, and the
``del quote_vol`` release for the non-committee path). Highest-risk step in the
parent spec but here it is pure variable threading: 10 interacting flags are
preserved exactly as in the source.

The ``del quote_vol`` at 3751-3752 is preserved at the equivalent point.
``w_fast_tilted`` is a local (single_horizon mode) and is released locally.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.mhs.evaluation import (
    CAUSAL_BETA_LOOKBACK_BARS,
    CAUSAL_BETA_MIN_PERIODS,
    DataIntegrityError,
    apply_dynamic_gap_exclusion,
    apply_dynamic_mark_gap_exclusion,
    assert_relevant_execution_data_coverage,
    assert_relevant_mark_price_coverage,
    beta_neutralize_weights,
    books,
    causal_market_beta,
    inverse_realized_vol_tilt,
    realized_vol,
    renormalize_within_mask,
    specs,
)
from src.mhs.marks import _pit_execution_mask
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry

_logger = logging.getLogger("MhsHorizonDiagnostic")


def build_books(ctx: PipelineContext, telemetry: StageTelemetry) -> None:
    """Construct fast/slow book weights, execution roster, and execution weights."""
    ctx.fast_grid = pd.date_range(ctx.start, ctx.end, freq="6h", tz="UTC")
    ctx.slow_grid = pd.date_range(ctx.start, ctx.end, freq="24h", tz="UTC")

    ctx.fast_ema = specs._signal_ema_span(ctx.fast.band.sign, ctx.fast.horizon_hours, ctx.fast.step_hours)
    ctx.slow_ema = specs._signal_ema_span(ctx.slow.band.sign, ctx.slow.horizon_hours, ctx.slow.step_hours)
    ctx.w_fast = books._book_weights(ctx.log_close, ctx.eligible, ctx.fast, ctx.fast_grid, ema_span=ctx.fast_ema)
    ctx.w_slow = books._book_weights(ctx.log_close, ctx.eligible, ctx.slow, ctx.slow_grid, ema_span=ctx.slow_ema)
    ctx.w_fast_1h = ctx.w_fast.reindex(ctx.grid_1h).ffill().fillna(0.0)
    ctx.w_slow_1h = ctx.w_slow.reindex(ctx.grid_1h).ffill().fillna(0.0)
    ctx.execution_mask = _pit_execution_mask(
        ctx.quote_vol, ctx.eligible, ctx.config.execution_universe_size
    )
    if ctx.config.execution_coverage_gate:
        # Relevance-scoped data-integrity handling (spec
        # mhs_data_integrity_relevance_scoping.md §3), opt-in via the same
        # flag as the pre-existing strict gates below (default False keeps
        # every other call byte-identical, matching this file's established
        # opt-in-flag convention). Dynamic large-gap exclusion replaces a
        # static per-symbol exclusion list with a live computation over the
        # current cache and the current roster mask, so a symbol whose gap is
        # later backfilled is automatically re-admitted and one whose cache
        # degrades is automatically excluded. Gaps below the threshold are
        # left untouched -- the per-event
        # MISSING_DATA/RELEVANT_EXECUTION_DATA_GAP fold reporting already
        # handles those correctly.
        _had_any_roster_member = bool(ctx.execution_mask.to_numpy().any())
        ctx.execution_mask, _execution_gap_excluded = apply_dynamic_gap_exclusion(
            ctx.execution_mask, ctx.config.execution_timeframe, root=ctx.config.data_root,
        )
        ctx.execution_mask, _mark_gap_excluded = apply_dynamic_mark_gap_exclusion(ctx.execution_mask)
        if _execution_gap_excluded or _mark_gap_excluded:
            _logger.info(
                "[DATA] stage=dynamic_gap_exclusion execution_symbols=%d mark_symbols=%d",
                len(_execution_gap_excluded), len(_mark_gap_excluded),
            )
        if _had_any_roster_member and not bool(ctx.execution_mask.to_numpy().any()):
            # Dynamic exclusion is meant to drop individual symbols/periods
            # with a structurally unusable gap, never the entire roster.
            # Every member being excluded is a systemic misconfiguration
            # (wrong data_root, execution_timeframe never collected at all)
            # rather than ordinary per-symbol data noise, and must fail
            # closed loudly instead of silently producing a report over zero
            # executed symbols.
            raise DataIntegrityError(
                "dynamic gap exclusion removed every roster member -- "
                f"execution_timeframe={ctx.config.execution_timeframe!r} data_root="
                f"{ctx.config.data_root!r} likely has no coverage at all for this window"
            )
        # Relevance-scoped pre-flight gates: the full-universe
        # Cartesian-product gate is replaced here by per-roster-membership
        # scope -- gaps outside a symbol's membership interval are ignored, and
        # mark-price coverage is validated with the exact causal availability
        # semantics the replay applies, so a pass cannot die mid-replay. Runs
        # after dynamic exclusion, so this now only ever fires on sub-threshold
        # gaps for users who want zero-tolerance instead of the default
        # auto-exclusion.
        assert_relevant_execution_data_coverage(
            ctx.execution_mask, ctx.config.execution_timeframe, root=ctx.config.data_root,
        )
    ctx.realized_execution_roster_size = float(ctx.execution_mask.sum(axis=1).mean())
    if ctx.config.execution_coverage_gate:
        assert_relevant_mark_price_coverage(
            ctx.execution_mask,
            "1h",
            stale_hours=24 if ctx.config.mark_mode == "cache_required_stale_carry" else 0,
        )
    if ctx.config.fast_book_mode == "horizon_ensemble":
        ctx.w_fast_execution = books._horizon_ensemble_execution_weights(
            ctx.log_close, ctx.eligible, ctx.execution_mask, ctx.fast, ctx.fast_grid,
            "horizon_ensemble", "raw", ctx.fast_ema,
        )
    else:
        w_fast_tilted = inverse_realized_vol_tilt(
            ctx.w_fast, realized_vol(ctx.log_close, ctx.fast.horizon_hours).reindex(ctx.fast_grid),
        )
        ctx.w_fast_execution = renormalize_within_mask(
            w_fast_tilted, ctx.execution_mask.reindex(ctx.w_fast.index).fillna(False), ctx.fast.min_symbols,
        )
    ctx.w_slow_execution = books._horizon_ensemble_execution_weights(
        ctx.log_close, ctx.eligible, ctx.execution_mask, ctx.slow, ctx.slow_grid,
        ctx.config.slow_book_mode, ctx.config.ensemble_signal, ctx.slow_ema,
    )
    if ctx.config.beta_neutralize:
        ctx.w_slow_execution = beta_neutralize_weights(
            ctx.w_slow_execution,
            causal_market_beta(
                ctx.log_close, ctx.eligible,
                CAUSAL_BETA_LOOKBACK_BARS, CAUSAL_BETA_MIN_PERIODS,
            ).reindex(ctx.w_slow_execution.index),
            ctx.execution_mask.reindex(ctx.w_slow_execution.index).fillna(False),
            ctx.slow.min_symbols,
        )
    if ctx.config.fast_book_mode == "single_horizon":
        del w_fast_tilted
    # Eligibility and the execution roster are now materialized.  The raw
    # volume matrix otherwise stays alive while phase diagnostics create their
    # temporary target-weight matrices.
    if not ctx.config.committee_capital:
        del ctx.quote_vol
