"""Fold-safe (train-only) horizon and funding-carry sign discovery per anchored fold."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from src.mhs.discovery import (
    DiscoveryQualificationResult,
    fold_train_only_discovery_qualification,
)
from src.mhs.evidence import AnchoredPurgedFold, phase_1_anchored_purged_folds
from src.mhs.execution import mhs_ledger_pnl
from src.mhs.parallel import (
    FORK_CONTEXT,
    assert_fork_admission,
    fork_shared_payload,
    frozen_gc_heap,
    plan_worker_count,
    resolve_fork_shared,
)
from src.mhs.params import (
    DISCOVERY_GATE_TRANCHE_COUNT,
    FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
    MEASURED_EXECUTION_COST_TIERS_BPS,
)
from src.mhs.resources import _resolve_ram_budget, _StageRecorder, _worker_plan_observer
from src.mhs.types import WORKER_PEAK_RSS_BYTES, BookSpec

from . import books


def _fold_safe_slow_book_spec(
    selection: DiscoveryQualificationResult,
    default: BookSpec,
) -> tuple[BookSpec, int, str]:
    """Resolve one fold's ``slow_momentum`` spec from its fold-scoped selection.

    Returns ``(spec, horizon_hours, source)``. ``source`` is
    ``"fold_train_only_discovery"`` only when the fold-scoped gate admitted a
    candidate (spec is ``default`` with ``horizon_hours`` replaced by the
    selected horizon, keeping band/step_hours/min_symbols identical to the
    frozen default); otherwise ``"frozen_default"`` with ``spec is default``
    unchanged.
    """
    if selection.admitted and selection.selected_horizon is not None:
        return (
            BookSpec(
                band=default.band,
                horizon_hours=selection.selected_horizon,
                step_hours=default.step_hours,
                min_symbols=default.min_symbols,
            ),
            selection.selected_horizon,
            "fold_train_only_discovery",
        )
    return default, default.horizon_hours, "frozen_default"


def _fold_safe_fast_horizon(
    selection: DiscoveryQualificationResult,
    default_horizon: int,
) -> tuple[int, str]:
    """Resolve one fold's ``fast_reversal`` horizon from its fold-scoped selection.

    Diagnostic-only: returns ``(horizon_hours, source)`` instead of a
    ``BookSpec`` because fast_reversal's book construction and
    ``BOOK_BLEND_WEIGHTS`` stay frozen at 0.0 capital (the result is
    evidence for a separate governance decision, never a weight change).
    ``source`` is ``"fold_train_only_discovery"`` only when the fold-scoped
    gate admitted a candidate (``admitted`` and ``selected_horizon`` both
    truthy); otherwise ``"frozen_default"`` with ``default_horizon`` unchanged.
    """
    if selection.admitted and selection.selected_horizon is not None:
        return selection.selected_horizon, "fold_train_only_discovery"
    return default_horizon, "frozen_default"

def _prefer_funding_carry_selection(
    long_result: DiscoveryQualificationResult,
    short_result: DiscoveryQualificationResult,
) -> tuple[int, int] | None:
    """Pick the funding-carry sign family with the strongest admitted evidence.

    Unlike the fast/slow bands -- each with one pre-registered sign -- the
    funding-carry SIGN is itself the object being discovered, so the two
    families' fold-scoped gate results are compared directly: an admitted
    family is preferred over a non-admitted one, and when both admit the
    family with the larger ``|qualification_net_t|`` wins (ties break toward
    sign=+1, the first family in iteration order). Returns
    ``(lookback_hours, sign)`` or None when neither family admits.
    """
    candidates: list[tuple[int, float, int]] = []
    for sign, result in ((1, long_result), (-1, short_result)):
        if (
            result.admitted
            and result.selected_horizon is not None
            and result.qualification_net_t is not None
        ):
            candidates.append((result.selected_horizon, abs(result.qualification_net_t), sign))
    if not candidates:
        return None
    lookback, _, sign = max(candidates, key=lambda candidate: candidate[1])
    return lookback, sign


def _fold_safe_discovery_worker(
    fold: AnchoredPurgedFold,
    fold_index: int,
    token: str,
) -> tuple[int | None, tuple[int, str], tuple[int | None, int | None, str, float | None]]:
    """One anchored fold's leak-free slow/fast/funding-carry selection.

    The exact per-fold body of the fold-safe discovery loop: slow-momentum and
    fast-reversal use their fold-train-only gate with the precomputed candidate
    books, and funding-carry picks the stronger admitted sign family, scoring
    its train-window orthogonality correlation against the fold's own
    slow-momentum book. Returns
    ``(slow_horizon_or_None, (fast_horizon, source), (fc_lookback, fc_sign,
    fc_source, fc_corr))``.

    The panels and candidate books are resolved from the fork-shared payload by
    ``token`` (registered via ``fork_shared_payload`` in the parent before the
    pool forks) so no ``pd.DataFrame`` crosses the ``ProcessPoolExecutor.submit``
    pickle boundary.
    """
    shared = resolve_fork_shared(token)
    specs: dict[str, BookSpec] = shared["specs"]
    log_close: pd.DataFrame = shared["log_close"]
    eligible: pd.DataFrame = shared["eligible"]
    opens: pd.DataFrame = shared["opens"]
    bar_funding: pd.DataFrame = shared["bar_funding"]
    grid_1h: pd.DatetimeIndex = shared["grid_1h"]
    precomputed: dict[str, dict[int, pd.DataFrame]] = shared["precomputed"]
    slow_weights = precomputed["slow"]
    fast_weights = precomputed["fast"]
    funding_long = precomputed["funding_long"]
    funding_short = precomputed["funding_short"]
    _spec, _horizon, _source = _fold_safe_slow_book_spec(
        fold_train_only_discovery_qualification(
            sign=1,
            horizon_candidates=specs["slow_momentum"].band.horizons_hours,
            log_close=log_close, eligible=eligible, opens=opens,
            bar_funding=bar_funding, grid_1h=grid_1h, fold=fold,
            tranche_count=DISCOVERY_GATE_TRANCHE_COUNT,
            precomputed_candidate_weights=slow_weights,
        ),
        specs["slow_momentum"],
    )
    slow_horizon = _horizon if _source == "fold_train_only_discovery" else None
    fast_tuple = _fold_safe_fast_horizon(
        fold_train_only_discovery_qualification(
            sign=-1,
            horizon_candidates=specs["fast_reversal"].band.horizons_hours,
            log_close=log_close, eligible=eligible, opens=opens,
            bar_funding=bar_funding, grid_1h=grid_1h, fold=fold,
            tranche_count=DISCOVERY_GATE_TRANCHE_COUNT,
            precomputed_candidate_weights=fast_weights,
        ),
        specs["fast_reversal"].horizon_hours,
    )
    _fc_long = fold_train_only_discovery_qualification(
        sign=1, horizon_candidates=FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
        log_close=log_close, eligible=eligible, opens=opens,
        bar_funding=bar_funding, grid_1h=grid_1h, fold=fold,
        tranche_count=DISCOVERY_GATE_TRANCHE_COUNT,
        precomputed_candidate_weights=funding_long,
    )
    _fc_short = fold_train_only_discovery_qualification(
        sign=-1, horizon_candidates=FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
        log_close=log_close, eligible=eligible, opens=opens,
        bar_funding=bar_funding, grid_1h=grid_1h, fold=fold,
        tranche_count=DISCOVERY_GATE_TRANCHE_COUNT,
        precomputed_candidate_weights=funding_short,
    )
    _fc_pick = _prefer_funding_carry_selection(_fc_long, _fc_short)
    _fc_lookback: int | None = None
    _fc_sign: int | None = None
    _fc_source = "frozen_default"
    _fc_corr: float | None = None
    if _fc_pick is not None:
        _fc_lookback, _fc_sign = _fc_pick
        _fc_source = "fold_train_only_discovery"
        _fc_weights = funding_long if _fc_sign == 1 else funding_short
        _train_mask = (grid_1h >= fold.train_start) & (grid_1h <= fold.train_end)
        _fc_net, _ = mhs_ledger_pnl(
            _fc_weights[_fc_lookback].loc[_train_mask],
            opens.loc[_train_mask], bar_funding.loc[_train_mask],
            MEASURED_EXECUTION_COST_TIERS_BPS["base"],
        )
        _fc_daily = (1.0 + _fc_net).resample("1D").apply(lambda s: s.prod() - 1.0)
        _mom_horizon = slow_horizon or specs["slow_momentum"].horizon_hours
        _mom_net, _ = mhs_ledger_pnl(
            slow_weights[_mom_horizon].loc[_train_mask],
            opens.loc[_train_mask], bar_funding.loc[_train_mask],
            MEASURED_EXECUTION_COST_TIERS_BPS["base"],
        )
        _mom_daily = (1.0 + _mom_net).resample("1D").apply(lambda s: s.prod() - 1.0)
        _fc_corr = float(
            pd.concat([_fc_daily, _mom_daily], axis=1).corr().iloc[0, 1]
        )
    return slow_horizon, fast_tuple, (_fc_lookback, _fc_sign, _fc_source, _fc_corr)


def _run_fold_safe_discovery_parallel(
    specs: dict[str, BookSpec],
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    precomputed: dict[str, dict[int, pd.DataFrame]] | None = None,
    telemetry: _StageRecorder | None = None,
) -> tuple[
    dict[int, int | None],
    dict[int, tuple[int, str]],
    dict[int, tuple[int | None, int | None, str, float | None]],
]:
    """Fold-safe horizon selection for all anchored folds in fork workers.

    The three folds' slow/fast/funding-carry gates are embarrassingly
    independent; forking them (``ProcessPoolExecutor``, the same pattern as
    ``concurrency._run_books_concurrent``/``_run_folds_parallel``) replaces the sequential
    parent loop and collapses the fold-safe discovery wall clock ~3x. The
    candidate weight books are built once in the parent and inherited by the
    fork children copy-on-write via ``fork_shared_payload``: only a short token
    crosses the ``submit`` boundary (zero pickle bytes), and the worker resolves
    ``specs/log_close/eligible/opens/bar_funding/grid_1h/precomputed`` from the
    shared registry. Results are keyed by fold index.

    ``precomputed`` lets the caller pass the ``books._candidate_weight_books`` result
    shared with the top-level discovery gate; when omitted it is built here once.
    """
    if precomputed is None:
        precomputed = books._candidate_weight_books(log_close, eligible, bar_funding, specs)
    folds = phase_1_anchored_purged_folds()
    _fold_safe_reserve = _resolve_ram_budget(None, True)[1]
    max_workers = plan_worker_count(
        min(3, len(folds)), WORKER_PEAK_RSS_BYTES, ram_guard=True,
        observer=_worker_plan_observer(telemetry, "fold_safe_discovery", WORKER_PEAK_RSS_BYTES),
        reserve_bytes=_fold_safe_reserve,
    )
    assert_fork_admission(
        "fold_safe_discovery", max_workers, WORKER_PEAK_RSS_BYTES, _fold_safe_reserve,
    )
    slow: dict[int, int | None] = {}
    fast: dict[int, tuple[int, str]] = {}
    funding_carry: dict[int, tuple[int | None, int | None, str, float | None]] = {}
    with (
        fork_shared_payload({
            "specs": specs, "log_close": log_close, "eligible": eligible,
            "opens": opens, "bar_funding": bar_funding, "grid_1h": grid_1h,
            "precomputed": precomputed,
        }) as token,
        frozen_gc_heap(),
        ProcessPoolExecutor(max_workers=max_workers, mp_context=FORK_CONTEXT) as pool,
    ):
        futures = {
            pool.submit(_fold_safe_discovery_worker, fold, idx, token): idx
            for idx, fold in enumerate(folds)
        }
        for future in as_completed(futures):
            idx = futures[future]
            slow[idx], fast[idx], funding_carry[idx] = future.result()
    return slow, fast, funding_carry
