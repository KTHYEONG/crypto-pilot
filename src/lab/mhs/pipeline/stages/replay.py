"""S6: Top-level execution replay: the blend book always, the standalone fast/slow reference books on request.

Extracted verbatim from ``evaluation.py`` lines 3987-4059 (execution-symbol
resolution, minute-grid construction, ``has_minute_data`` check,
the ``concurrency._run_books_concurrent`` call, and the four
``del`` statements at 4034-4037 released verbatim at the end of this function).

The ``del w_fast, w_fast_execution, phase_fast`` / ``del w_slow,
w_slow_execution, phase_slow`` / ``del blend_1h, phase_blend, regime_scale,
committee_execution_book`` + ``gc.collect()`` are preserved at the stage
boundary so the measured peak-RSS release (ADR_20260817) is retained.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Sequence
from typing import Final, Literal

import pandas as pd

import src.lab.mhs.evaluation.committee as committee
import src.lab.mhs.evaluation.concurrency as concurrency
import src.lab.mhs.evaluation.guards as guards
from src.core.marks import _missing_execution_sources
from src.lab.mhs.pipeline.context import PipelineContext
from src.lab.mhs.telemetry import StageTelemetry, Tag

EXECUTION_SOURCE_MISSING_UNTARGETED: Final[str] = "EXECUTION_SOURCE_MISSING_UNTARGETED"


def _untargeted_missing_execution_disclosure(
    execution_mask: pd.DataFrame,
    execution_symbols: Sequence[str],
    root: str,
    timeframe: Literal["3m"],
) -> tuple[str, ...]:
    """Disclose roster members that lack an execution source but were never targeted.

    Such a symbol cannot change the replay (it carries no target), so the run proceeds
    unchanged; the disclosure exists so a reader can see that the executable universe
    was narrower than the point-in-time roster rather than infer it from absent fills.
    Targeted symbols without a source are not listed here: the execution stream fails
    closed on them.

    Args:
        execution_mask: Post-gate point-in-time roster (decision x symbol, bool).
        execution_symbols: Symbols with a non-zero target in any replayed book.
        root: OHLCV root.
        timeframe: Execution timeframe.
    Returns:
        ``()`` when no untargeted roster member lacks a source; otherwise exactly one
        token ``"EXECUTION_SOURCE_MISSING_UNTARGETED:n=<count>:<SYM1>,<SYM2>,..."`` with
        symbols sorted.
    """
    targeted = set(execution_symbols)
    roster = [c for c in execution_mask.columns if bool(execution_mask[c].any())]
    candidates = [s for s in roster if s not in targeted]
    missing = _missing_execution_sources(root, candidates, timeframe)
    if not missing:
        return ()
    return (f"{EXECUTION_SOURCE_MISSING_UNTARGETED}:n={len(missing)}:{','.join(missing)}",)


def run_replays(ctx: PipelineContext, telemetry: StageTelemetry) -> None:
    """Replay the top-level books against the minute market; only the blend's failure enters ``book_reasons``."""
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
    ctx.execution_source_disclosure = _untargeted_missing_execution_disclosure(
        ctx.execution_mask, ctx.execution_symbols, ctx.root, ctx.config.execution_timeframe,
    )
    if ctx.execution_source_disclosure:
        token = ctx.execution_source_disclosure[0]
        rest = token.removeprefix(f"{EXECUTION_SOURCE_MISSING_UNTARGETED}:")
        count_str, _, syms = rest.partition(":")
        telemetry.log(
            Tag.DATA, "execution_source_audit",
            missing_untargeted=int(count_str.removeprefix("n=")),
            symbols=syms.split(",") if syms else [],
        )
    ctx.minute_grid = pd.date_range(
        ctx.start, ctx.end,
        freq="3min",
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
        _terminal = guards._guard_stage_or_breach(
            "pre_books", ctx.rss_budget_bytes, ctx.rss_reserve_bytes,
            ctx.config, ctx.recorder, str(ctx.resolved_end), str(ctx.start), str(ctx.end),
        )
        if _terminal is not None:
            ctx._terminal_report = _terminal
            return
        # Each book worker now loads only its own windows' roster slices from
        # Parquet (window-keyed reads, page-cache backed), so no full-period
        # preload is needed before forking -- the three books run concurrently
        # in fork children (spec Phase 3, P10) with a fraction of the former
        # resident set.
        book_report_fast, book_report_slow, book_report_blend, ctx.blend_traces, member_reports = concurrency._run_books_concurrent(
            ctx.root, ctx.config, len(ctx.funded), ctx.grid_1h, ctx.fast, ctx.slow, ctx.fast_grid, ctx.slow_grid,
            ctx.w_fast, ctx.w_slow, ctx.w_fast_execution, ctx.w_slow_execution, ctx.opens, ctx.bar_funding,
            ctx.phase_fast, ctx.phase_slow, ctx.phase_blend, ctx.start, ctx.end, ctx.funding_by_symbol,
            ctx.blend_1h, ctx.execution_mask, ctx.initial_equity, ctx.recorder, ctx.regime_scale,
            committee_execution_book=ctx.committee_execution_book,
            committee_member_books=ctx.committee_member_books,
        )
        # All three books have completed; the single-use step-weight inputs are
        # released together (spec §3.1, ``memory_opt``).
        del ctx.w_fast, ctx.w_fast_execution, ctx.phase_fast
        del ctx.w_slow, ctx.w_slow_execution, ctx.phase_slow
        del ctx.blend_1h, ctx.phase_blend, ctx.regime_scale, ctx.committee_execution_book
        gc.collect()

        # Compute member attribution from individual member book replays (I5:
        # observational only). member_proxy_sharpe comes from the 1h
        # prescreen ledger computed in the committee stage (D6), never from
        # member_reports itself -- both sides must be independent sources or
        # proxy_vs_ledger_rank_spearman compares the 3m ledger to itself.
        if member_reports and ctx.config.committee_member_attribution:
            ctx.committee_member_attribution = committee._committee_member_attribution(
                member_reports, ctx.committee_member_proxy_sharpe or {},
            )
        else:
            ctx.committee_member_attribution = None
        _terminal = guards._guard_stage_or_breach(
            "post_books", ctx.rss_budget_bytes, ctx.rss_reserve_bytes,
            ctx.config, ctx.recorder, str(ctx.resolved_end), str(ctx.start), str(ctx.end),
        )
        if _terminal is not None:
            ctx._terminal_report = _terminal
            return
        # execution_mask stays alive: the post-fold opt-in diagnostics consume
        # it (a bool panel, ~20 MB).
        ctx.books = {}
        if book_report_fast is not None:
            ctx.books["fast_reversal"] = book_report_fast
        if book_report_slow is not None:
            ctx.books["slow_momentum"] = book_report_slow
        ctx.blend_report = book_report_blend
    else:
        ctx.books = {}
        ctx.blend_report = None
        ctx.blend_traces = {}

    ctx.book_reasons = (
        (ctx.blend_report.failure.reason,)
        if ctx.blend_report is not None and ctx.blend_report.failure is not None
        else ()
    )
