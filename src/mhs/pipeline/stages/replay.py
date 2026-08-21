"""S6: Top-level execution replay (fast/slow/blend books).

Extracted verbatim from ``evaluation.py`` lines 3987-4059 (execution-symbol
resolution, minute-grid construction, ``has_minute_data`` check,
``_prewarm_mark_frames``, the ``_run_books_concurrent`` call, and the four
``del`` statements at 4034-4037 released verbatim at the end of this function).

The ``del w_fast, w_fast_execution, phase_fast`` / ``del w_slow,
w_slow_execution, phase_slow`` / ``del blend_1h, phase_blend, regime_scale,
committee_execution_book`` + ``gc.collect()`` are preserved at the stage
boundary so the measured peak-RSS release (ADR_20260817) is retained.
"""

from __future__ import annotations

import gc
import os

import pandas as pd

from src.application.research.mhs.evaluation import (
    _guard_stage_or_breach,
    _prewarm_mark_frames,
    _run_books_concurrent,
)
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry


def run_replays(ctx: PipelineContext, telemetry: StageTelemetry) -> None:
    """Run the top-level fast/slow/blend book replays against the minute market."""
    ctx.execution_symbols = sorted(
        set(ctx.w_fast_execution.columns[ctx.w_fast_execution.ne(0.0).any(axis=0)])
        | set(ctx.w_slow_execution.columns[ctx.w_slow_execution.ne(0.0).any(axis=0)])
        | (
            set(ctx.blend_1h.columns[ctx.blend_1h.ne(0.0).any(axis=0)])
            if ctx.config.committee_capital
            else set()
        )
    )
    ctx.initial_equity = 1.0
    ctx.minute_grid = pd.date_range(
        ctx.start, ctx.end,
        freq={"1m": "1min", "3m": "3min", "5m": "5min"}[ctx.config.execution_timeframe],
        tz="UTC",
    )
    ctx.has_minute_data = any(
        os.path.exists(os.path.join(ctx.root, ctx.config.execution_timeframe, f"{s}.parquet"))
        for s in ctx.execution_symbols
    )
    if ctx.has_minute_data and ctx.execution_symbols:
        ctx.recorder.record(
            "minute_market_mark_funding",
            grid_bars=len(ctx.minute_grid),
            n_symbols=len(ctx.execution_symbols),
        )
        _terminal = _guard_stage_or_breach(
            "pre_books", ctx.rss_budget_bytes, ctx.rss_reserve_bytes,
            ctx.config, ctx.recorder, str(ctx.resolved_end), str(ctx.start), str(ctx.end),
        )
        if _terminal is not None:
            ctx._terminal_report = _terminal
            return
        # Each book worker now loads only its own windows' roster slices from
        # Parquet (window-keyed reads, page-cache backed) and inherits the
        # execution roster's mark frames warmed here copy-on-write, so no
        # full-period minute-frame preload is needed before forking -- the three
        # books run concurrently in fork children (spec Phase 3, P10) with a
        # fraction of the former resident set.
        _prewarm_mark_frames(ctx.execution_symbols)
        book_report_fast, book_report_slow, book_report_blend, ctx.blend_traces = _run_books_concurrent(
            ctx.root, ctx.config, len(ctx.funded), ctx.grid_1h, ctx.fast, ctx.slow, ctx.fast_grid, ctx.slow_grid,
            ctx.w_fast, ctx.w_slow, ctx.w_fast_execution, ctx.w_slow_execution, ctx.opens, ctx.bar_funding,
            ctx.phase_fast, ctx.phase_slow, ctx.phase_blend, ctx.start, ctx.end, ctx.funding_by_symbol,
            ctx.blend_1h, ctx.execution_mask, ctx.initial_equity, ctx.recorder, ctx.regime_scale,
            committee_execution_book=ctx.committee_execution_book,
        )
        # All three books have completed; the single-use step-weight inputs are
        # released together (spec §3.1, ``memory_opt``).
        del ctx.w_fast, ctx.w_fast_execution, ctx.phase_fast
        del ctx.w_slow, ctx.w_slow_execution, ctx.phase_slow
        del ctx.blend_1h, ctx.phase_blend, ctx.regime_scale, ctx.committee_execution_book
        gc.collect()
        _terminal = _guard_stage_or_breach(
            "post_books", ctx.rss_budget_bytes, ctx.rss_reserve_bytes,
            ctx.config, ctx.recorder, str(ctx.resolved_end), str(ctx.start), str(ctx.end),
        )
        if _terminal is not None:
            ctx._terminal_report = _terminal
            return
        # execution_mask stays alive: the post-fold opt-in diagnostics consume
        # it (a bool panel, ~20 MB).
        ctx.books = {"fast_reversal": book_report_fast, "slow_momentum": book_report_slow}
        ctx.blend_report = book_report_blend
    else:
        ctx.books = {}
        ctx.blend_report = None
        ctx.blend_traces = {}

    ctx.book_reasons = tuple(
        sorted(
            b.failure.reason
            for b in [*ctx.books.values(), ctx.blend_report]
            if b is not None and b.failure is not None
        )
    )
