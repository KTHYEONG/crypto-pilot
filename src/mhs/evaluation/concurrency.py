from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import pandas as pd

import src.mhs.evaluation.participation as participation_mod
from src.common.errors import DataIntegrityError
from src.mhs import statistics as _statistics
from src.mhs.contracts import MhsBookReport, MhsDiagnosticRequest, MhsFoldReport
from src.mhs.evidence import (
    DeploymentReadinessResult,
    PhaseDiagnosticResult,
    compute_deployment_readiness,
    resolved_anchored_folds,
)
from src.mhs.parallel import (
    FORK_CONTEXT,
    assert_fork_admission,
    fork_shared_payload,
    frozen_gc_heap,
    plan_worker_count,
)
from src.mhs.params import PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H
from src.mhs.resources import (
    _resolve_ram_budget,
    _StageRecorder,
    _worker_plan_observer,
)
from src.mhs.types import BOOK_BLEND_WEIGHTS, WORKER_PEAK_RSS_BYTES, BookSpec

from . import books, folds, windows


def _run_books_concurrent(
    root: str,
    request: MhsDiagnosticRequest,
    n_symbols: int,
    grid_1h: pd.DatetimeIndex,
    fast: BookSpec,
    slow: BookSpec,
    fast_grid: pd.DatetimeIndex,
    slow_grid: pd.DatetimeIndex,
    w_fast: pd.DataFrame,
    w_slow: pd.DataFrame,
    w_fast_execution: pd.DataFrame,
    w_slow_execution: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    phase_fast: PhaseDiagnosticResult | None,
    phase_slow: PhaseDiagnosticResult | None,
    phase_blend: PhaseDiagnosticResult | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    funding_by_symbol: dict[str, pd.Series],
    blend_1h: pd.DataFrame,
    execution_mask: pd.DataFrame,
    initial_equity: float,
    telemetry: _StageRecorder | None = None,
    regime_scale: pd.Series | None = None,
    committee_execution_book: pd.DataFrame | None = None,
    committee_member_books: dict[str, pd.DataFrame] | None = None,
) -> tuple[
    MhsBookReport | None,
    MhsBookReport | None,
    MhsBookReport,
    dict[int, dict[str, float]],
    dict[str, MhsBookReport] | None,
]:
    """Run the top-level book replays concurrently in fork children.

    The blend book always runs; the standalone fast_reversal / slow_momentum reference books run
    only when ``request.reference_books_diagnostic`` is set and are otherwise returned as
    ``None``. Member-attribution books run when ``committee_member_books`` is given. The books
    share zero mutable state and only read the immutable 1h panels and the minute-frame cache,
    so they are embarrassingly parallel: ``ProcessPoolExecutor`` (fork) is used instead of
    threads because the replay loops are a CPU-bound Python/numpy mix and fork shares the
    read-only panels copy-on-write. Per-book telemetry is merged into the parent recorder in
    declared book order (members, then fast, slow, blend).

    ``blend_replay`` (the blend book's actual execution-replay weights) is built independently
    from ``blend_1h``: it must stay restricted to the execution-roster symbols tradable at
    minute granularity, while ``blend_1h`` covers the full eligible universe and is
    prescreen/tail-diagnostic only. ``regime_scale`` is applied to ``blend_replay`` here so the
    blend's primary/stress replay reflects it.

    Returns:
        ``(fast_report, slow_report, blend_report, blend_traces, member_reports)``; the first two
        are ``None`` unless the reference books were requested, ``member_reports`` is ``None``
        when no member book ran.
    """
    active_spec, active_grid = books._active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    blend_step = blend_1h.reindex(active_grid)
    blend_replay = (
        committee_execution_book.reindex(grid_1h).ffill().fillna(0.0)
        if committee_execution_book is not None
        else (
            BOOK_BLEND_WEIGHTS["fast_reversal"] * w_fast_execution.reindex(grid_1h).ffill().fillna(0.0)
            + BOOK_BLEND_WEIGHTS["slow_momentum"] * w_slow_execution.reindex(grid_1h).ffill().fillna(0.0)
        )
    ).reindex(active_grid)
    if regime_scale is not None:
        blend_replay = blend_replay.mul(regime_scale.reindex(active_grid).fillna(1.0), axis=0)

    # The three book workers share the immutable 1h panels and per-book weights
    # through ``fork_shared_payload`` (inherited copy-on-write by the fork
    # children), so only a short token crosses the submit boundary -- the
    # pickled-argument copies measured at ~1 GB per book are eliminated.
    books_dict: dict[str, tuple[Any, ...]] = {}
    if request.reference_books_diagnostic:
        books_dict["fast_reversal"] = (fast, fast_grid, w_fast, phase_fast, fast.horizon_hours, w_fast_execution)
        books_dict["slow_momentum"] = (slow, slow_grid, w_slow, phase_slow, slow.horizon_hours, w_slow_execution)
    books_dict["blend"] = (active_spec, active_grid, blend_step, phase_blend, 168, blend_replay)
    member_names: list[str] = []
    if committee_member_books:
        for m_name, m_book in committee_member_books.items():
            m_step = m_book.reindex(slow_grid).ffill().fillna(0.0)
            member_names.append(m_name)
            books_dict[f"member_{m_name}"] = (
                slow, slow_grid, m_step, phase_slow, slow.horizon_hours, m_step,
            )
    n_total_workers = 1 + len(member_names) + (2 if request.reference_books_diagnostic else 0)
    _books_reserve = _resolve_ram_budget(request.max_rss_bytes, request.ram_guard)[1]
    _books_workers = plan_worker_count(
        n_total_workers, WORKER_PEAK_RSS_BYTES, request.ram_guard,
        observer=_worker_plan_observer(telemetry, "books", WORKER_PEAK_RSS_BYTES),
        reserve_bytes=_books_reserve,
    )
    assert_fork_admission("books", _books_workers, WORKER_PEAK_RSS_BYTES, _books_reserve)
    with (
        fork_shared_payload({
            "grid_1h": grid_1h,
            "opens": opens,
            "bar_funding": bar_funding,
            "funding_by_symbol": funding_by_symbol,
            "books": books_dict,
        }) as token,
        frozen_gc_heap(),
        ProcessPoolExecutor(max_workers=_books_workers, mp_context=FORK_CONTEXT) as pool,
    ):
        f_fast = (
            pool.submit(
                windows._book_outcome_worker,
                "fast_reversal", token, n_symbols, root, request, start, end, initial_equity,
            )
            if request.reference_books_diagnostic
            else None
        )
        f_slow = (
            pool.submit(
                windows._book_outcome_worker,
                "slow_momentum", token, n_symbols, root, request, start, end, initial_equity,
            )
            if request.reference_books_diagnostic
            else None
        )
        f_blend = pool.submit(
            windows._book_outcome_worker,
            "blend", token, n_symbols, root, request, start, end, initial_equity,
        )
        f_members = {
            m_name: pool.submit(
                windows._book_outcome_worker,
                f"member_{m_name}", token, n_symbols, root, request, start, end, initial_equity,
            )
            for m_name in member_names
        }
        fast_out = f_fast.result() if f_fast is not None else None
        slow_out = f_slow.result() if f_slow is not None else None
        fast_report, fast_records = (fast_out[0], fast_out[1]) if fast_out is not None else (None, ())
        slow_report, slow_records = (slow_out[0], slow_out[1]) if slow_out is not None else (None, ())
        blend_report, blend_records, blend_traces = f_blend.result()
        member_reports: dict[str, MhsBookReport] = {}
        for m_name, f_member in f_members.items():
            m_report, m_records, _ = f_member.result()
            member_reports[m_name] = m_report
            if telemetry is not None:
                telemetry.absorb(m_records)

    if telemetry is not None:
        if fast_records:
            telemetry.absorb(fast_records)
        if slow_records:
            telemetry.absorb(slow_records)
        telemetry.absorb(blend_records)
    return fast_report, slow_report, blend_report, blend_traces, member_reports or None


def _run_post_diag_deploy(
    blend_report: MhsBookReport,
    root: str,
    request: MhsDiagnosticRequest,
    execution_symbols: list[str],
    minute_grid: pd.DatetimeIndex,
    signal_48h: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    fast: BookSpec,
) -> tuple[
    tuple[float, float] | None,
    float | None,
    dict[str, float],
    dict[str, int],
    DeploymentReadinessResult,
]:
    """Report-only diagnostics + deployment readiness, one background-thread unit.

    ``compute_deployment_readiness`` is invoked with ``research_go_eligible=None``: the only
    value it needs from the anchored folds is the final Research-GO boolean, which the caller
    patches in after the folds resolve, so this tail overlaps the fold pool. The bootstrap CI
    (``bootstrap_ci_diagnostic``) and the 48h placebo percentile (``placebo_diagnostic``) are
    opt-in report-only values and are ``None`` when not requested; participation warnings,
    termination counts and deployment readiness are always computed.

    Raises:
        DataIntegrityError: the blend has no primary replay, or the placebo is requested and
            the blend has no naive Sharpe.
    """
    bootstrap_ci: tuple[float, float] | None = None
    if blend_report.primary is None:
        raise DataIntegrityError("post-book tail requires a blend primary replay")
    equity_1h = blend_report.primary.ledger.equity.resample("1h").last().dropna()
    net_1h = equity_1h.pct_change().dropna()
    if request.bootstrap_ci_diagnostic and len(net_1h) >= 2:
        bootstrap_ci = _statistics._bootstrap_ci(
            net_1h, _statistics._BOOTSTRAP_REPLICATES, _statistics._BOOTSTRAP_MEAN_BLOCK, _statistics._BOOTSTRAP_SEED,
        )
    participation = participation_mod._participation_warnings(
        blend_report.primary, root, request.execution_timeframe,
        execution_symbols, minute_grid,
    )
    termination_counts = dict(blend_report.primary.termination_counts)
    placebo_percentile: float | None = None
    if request.placebo_diagnostic:
        if blend_report.primary_naive_sharpe is None:
            raise DataIntegrityError("blend report requires a naive Sharpe for the placebo")
        placebo_percentile = _statistics._placebo_sharpe_percentile(
            signal_48h, eligible, opens, bar_funding, grid_1h,
            fast, blend_report.primary_naive_sharpe, 500, _statistics._BOOTSTRAP_SEED,
        )
    deployment = compute_deployment_readiness(
        equity_1h,
        _PERIODS_PER_YEAR_1H,
        participation_warnings=participation,
        primary_valid=blend_report.primary.ledger.primary_valid,
        research_go_eligible=None,
        n_bootstrap=_statistics._BOOTSTRAP_REPLICATES,
        mean_block_bars=_statistics._BOOTSTRAP_MEAN_BLOCK,
        seed=_statistics._BOOTSTRAP_SEED,
    )
    return bootstrap_ci, placebo_percentile, participation, termination_counts, deployment


def _run_post_book_concurrently(
    blend_report: MhsBookReport | None,
    root: str,
    request: MhsDiagnosticRequest,
    execution_symbols: list[str],
    minute_grid: pd.DatetimeIndex | None,
    signal_48h: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    fast: BookSpec,
    fold_funding: dict[str, pd.Series],
    initial_equity: float,
    telemetry: _StageRecorder | None = None,
    fold_slow_horizons: dict[int, int | None] | None = None,
    fold_fast_horizons: dict[int, tuple[int, str]] | None = None,
    fold_funding_carry: dict[int, tuple[int | None, int | None, str, float | None]] | None = None,
    fold_committee_weights: dict[int, dict[str, float]] | None = None,
    base_panel: dict[str, pd.DataFrame] | None = None,
) -> tuple[
    tuple[float, float] | None,
    float | None,
    dict[str, float],
    dict[str, int],
    tuple[MhsFoldReport, ...],
    DeploymentReadinessResult | None,
]:
    """Run anchored folds, diagnostics, and deployment readiness concurrently.

    The fold pool forks at the first phase-V submit while the main process is quiescent (the
    book workers have joined and no diagnostic thread exists yet), then a single background
    thread runs the diagnostics + deployment-readiness tail overlapping all three fold phases
    (validation, shared reference, execution). The fold result telemetry is recorded in
    fold order; ``blend_participation``/``statistical_diagnostics`` telemetry is
    left to the caller so the ordered-stage contract is preserved deterministically.
    """
    fold_list = resolved_anchored_folds(request)
    has_primary = blend_report is not None and blend_report.primary is not None
    worker_request = request

    bootstrap_ci: tuple[float, float] | None = None
    placebo_percentile: float | None = None
    participation: dict[str, float] = {}
    termination_counts: dict[str, int] = {}
    fold_reports: tuple[MhsFoldReport, ...] = ()
    deployment: DeploymentReadinessResult | None = None

    if not fold_list:
        if blend_report is not None and blend_report.primary is not None:
            (
                bootstrap_ci, placebo_percentile, participation,
                termination_counts, deployment,
            ) = _run_post_diag_deploy(
                blend_report, root, request, execution_symbols, minute_grid,
                signal_48h, eligible, opens, bar_funding, grid_1h, fast,
            )
        return (
            bootstrap_ci, placebo_percentile, participation,
            termination_counts, fold_reports, deployment,
        )

    _post_book_reserve = _resolve_ram_budget(request.max_rss_bytes, request.ram_guard)[1]
    max_workers = plan_worker_count(
        min(3, len(fold_list)), WORKER_PEAK_RSS_BYTES, request.ram_guard,
        observer=_worker_plan_observer(telemetry, "post_book_folds", WORKER_PEAK_RSS_BYTES),
        reserve_bytes=_post_book_reserve,
    )
    assert_fork_admission("post_book_folds", max_workers, WORKER_PEAK_RSS_BYTES, _post_book_reserve)
    with (
        fork_shared_payload({"base_panel": base_panel, "fold_funding": fold_funding}) as token,
        frozen_gc_heap(),
        ProcessPoolExecutor(max_workers=max_workers, mp_context=FORK_CONTEXT) as pool,
    ):
        validation_futures = folds._submit_fold_validation_phase(
            pool, token, root, fold_list, worker_request, fold_slow_horizons, fold_committee_weights,
        )
        # The fold pool is now forked; start the diagnostics/deployment thread.
        with ThreadPoolExecutor(max_workers=1) as tpool:
            post_future = None
            if has_primary:
                assert blend_report is not None
                post_future = tpool.submit(
                    _run_post_diag_deploy,
                    blend_report, root, request, execution_symbols, minute_grid,
                    signal_48h, eligible, opens, bar_funding, grid_1h, fast,
                )
            fold_reports = folds._complete_anchored_folds(
                pool, token, validation_futures, root, fold_list, worker_request, initial_equity,
                fold_slow_horizons, fold_fast_horizons, fold_funding_carry, fold_committee_weights,
            )
            if post_future is not None:
                (
                    bootstrap_ci, placebo_percentile, participation,
                    termination_counts, deployment,
                ) = post_future.result()
    if telemetry is not None:
        for fold_report in fold_reports:
            fill_count = (
                len(fold_report.strict.simulated_fills) + len(fold_report.stress.simulated_fills)
                if fold_report.strict is not None and fold_report.stress is not None
                else 0
            )
            telemetry.record(f"anchored_fold_{fold_report.fold_index}", fill_count=fill_count)
    return (
        bootstrap_ci, placebo_percentile, participation,
        termination_counts, fold_reports, deployment,
    )
