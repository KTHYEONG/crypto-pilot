"""S2: Eligibility mask, fold-safe horizon selection, candidate weight books.

Extracted verbatim from ``evaluation.py`` lines 3612-3655 (eligibility mask,
fill-mark-parity eligibility, ``log_close``, candidate weight books, the
fold-safe horizon selection ``_run_fold_safe_discovery_parallel`` scan, and the
top-level horizon override via ``dataclasses.replace``).

The ``del close`` at the original 3618-3619 is preserved here at the same
relative point: the raw close panel is released before phase/weight
construction unless the committee path still needs it (gated by
``ctx.config.committee_capital``).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.application.research.mhs.evaluation import (
    BOOK_SPECS,
    liquid_half_eligibility,
)
from src.application.research.mhs.marks import _fill_mark_parity_eligibility
from src.application.research.mhs.stage_services import (
    _candidate_weight_books,
    _run_fold_safe_discovery_parallel,
)
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry


def select_horizons(ctx: PipelineContext, telemetry: StageTelemetry) -> None:
    """Compute eligibility, candidate books, and fold-safe horizon selection."""
    ctx.eligible = liquid_half_eligibility(ctx.quote_vol, lookback_bars=720, min_history_bars=720)
    ctx.eligible, ctx._fill_mark_parity_census = _fill_mark_parity_eligibility(
        ctx.close, ctx.eligible, ctx.config.fill_mark_parity_gate
    )
    ctx.log_close = np.log(ctx.close)
    # The raw close panel is not used after its log transform.  Releasing it
    # before phase/weight construction avoids retaining two full multi-year
    # price matrices at once.
    if not ctx.config.committee_capital:
        del ctx.close

    ctx.specs = BOOK_SPECS
    ctx.fast = ctx.specs["fast_reversal"]
    ctx.slow = ctx.specs["slow_momentum"]
    # Fold-safe horizon selection (spec §1.5, ``wiring``): computed once in the
    # parent before either the top-level books or the fold pool are forked,
    # reusing the already-loaded full-period panel. Only the resolved plain
    # horizon ``int`` (or None) is passed down to fold workers, so no worker
    # ever reloads a wide ``[train_start, train_end]`` panel.
    ctx.fold_slow_horizons = {}
    ctx.fold_fast_horizons = {}
    ctx.fold_funding_carry = {}
    # Candidate weight books are built exactly once in the parent and shared by
    # both the fold-safe discovery scan and the top-level discovery gate (the
    # byte-identical duplicate build is eliminated: -5.23 GB peak, -70 s wall).
    ctx.candidate_books = None
    if ctx.config.fold_safe_horizon_selection or ctx.config.discovery_gate:
        ctx.candidate_books = _candidate_weight_books(
            ctx.log_close, ctx.eligible, ctx.bar_funding, ctx.specs
        )
    if ctx.config.fold_safe_horizon_selection:
        # Fold-safe horizon selection (spec §1.5, ``wiring``): the three folds'
        # slow/fast/funding-carry gates run in fork workers (candidate weight
        # books built once in the parent and inherited COW), replacing the
        # sequential per-fold loop.
        (
            ctx.fold_slow_horizons, ctx.fold_fast_horizons, ctx.fold_funding_carry,
        ) = _run_fold_safe_discovery_parallel(
            ctx.specs, ctx.log_close, ctx.eligible, ctx.opens, ctx.bar_funding, ctx.grid_1h,
            precomputed=ctx.candidate_books,
            telemetry=ctx.recorder,
        )
        # The top-level report uses fold index 2's selection (train=2021-2024,
        # the widest leak-free window that still excludes 2025), making the
        # full-period report's horizon choice walk-forward-safe relative to
        # 2025 without a second, redundant discovery scan.
        ctx.top_level_horizon = ctx.fold_slow_horizons.get(2)
        if ctx.top_level_horizon is not None:
            ctx.slow = dataclasses.replace(ctx.slow, horizon_hours=ctx.top_level_horizon)
