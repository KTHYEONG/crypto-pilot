from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs import scaling as _scaling
from src.mhs import statistics as _statistics
from src.mhs.contracts import MhsDiagnosticRequest, MhsFoldReport
from src.mhs.discovery import (
    DiscoveryQualificationResult,
    fold_train_only_discovery_qualification,
)
from src.mhs.evidence import AnchoredPurgedFold, phase_1_anchored_purged_folds, resolved_anchored_folds
from src.mhs.execution import (
    mhs_ledger_pnl,
    replay_execution_window_batch,
    replay_execution_windows,
)
from src.mhs.execution.window_stream import MhsExecutionWindow
from src.mhs.features import FeatureAdmission
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
    FOLD_PANEL_WARMUP_HOURS,
    FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
    MEASURED_EXECUTION_COST_TIERS_BPS,
    TRAIN_REFERENCE_PREFIX_TARGET_ATOL,
)
from src.mhs.params import (
    PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H,
)
from src.mhs.research_go import (
    GO_REASON_EXECUTION_GAP,
    GO_REASON_INCOMPLETE_FOLD,
    GO_REASON_INVALID_PRIMARY,
    GO_REASON_NONFINITE_EQUITY,
)
from src.mhs.resources import (
    _assert_execution_rss_budget,
    _resolve_ram_budget,
    _StageRecorder,
    _worker_plan_observer,
)
from src.mhs.trend_sleeve import market_basket_log_price, time_series_trend_position, trend_sleeve_weights
from src.mhs.types import BOOK_SPECS, TREND_SLEEVE_HORIZONS_HOURS, WORKER_PEAK_RSS_BYTES, BookSpec, ExecutionSpec

from . import books, fold_weights, integrity, regime, specs, windows

_logger = logging.getLogger("MhsHorizonDiagnostic")


def _incomplete_fold_report(
    fold: AnchoredPurgedFold, fold_index: int, failures: tuple[str, ...],
) -> MhsFoldReport:
    """A fold that could not be replayed, failed closed with its reason codes."""
    return MhsFoldReport(
        fold_index=fold_index,
        validation_start=str(fold.validation_start),
        validation_end=str(fold.validation_end),
        strict=None,
        stress=None,
        primary_valid=False,
        primary_autocorr_sharpe=float("nan"),
        primary_naive_sharpe=float("nan"),
        primary_net_ann=float("nan"),
        primary_geometric_cagr=float("nan"),
        primary_max_drawdown=float("nan"),
        stress_naive_sharpe=float("nan"),
        decision_intents=0,
        termination_counts={},
        failures=tuple(sorted(set(failures))),
        strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
    )


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



def _trend_sleeve_position(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
) -> pd.Series:
    """Ensemble trend position on the eligible market basket, held to 1h bars.

    Thin wrapper reusing the frozen ``market_basket_log_price`` and
    ``time_series_trend_position`` primitives verbatim -- no new math.
    """
    basket = market_basket_log_price(log_close, eligible)
    return time_series_trend_position(basket, TREND_SLEEVE_HORIZONS_HOURS, decision_grid)


def _apply_trend_sleeve(
    blend_1h: pd.DataFrame,
    position: pd.Series,
    execution_mask: pd.DataFrame,
    gross_budget: float,
) -> pd.DataFrame:
    """Add the gross-budget sleeve weights to the book blend, purely.

    Returns a new frame (``blend_1h`` is never mutated in place). The sleeve is
    deliberately not dollar-neutral, so row sums of the result may be nonzero.
    """
    sleeve = trend_sleeve_weights(position, execution_mask, gross_budget)
    return blend_1h.add(sleeve.reindex(blend_1h.index).fillna(0.0), fill_value=0.0)


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


def _fold_exposure_warmup(
    exposure_warmup_returns: pd.Series | None,
    validation_start: pd.Timestamp,
) -> pd.Series | None:
    """Pure slicer keeping only warmup rows strictly before a fold's start.

    Defense in depth against leak (I-WARM): even though the fold worker's
    scale primitive fail-closes on overlap, the run-level warmup reference is
    sliced here so no row at/after ``validation_start`` ever reaches it.
    """
    if exposure_warmup_returns is None:
        return None
    return exposure_warmup_returns.loc[
        exposure_warmup_returns.index < validation_start
    ]


_TrainReferenceGroupKey = tuple[
    pd.Timestamp, int | None, tuple[tuple[str, float], ...] | None, tuple[str, ...] | None,
]


def _train_reference_group_key(
    fold: AnchoredPurgedFold,
    slow_horizon_override: int | None,
    committee_member_weights: Mapping[str, float] | None,
    *,
    committee_admission: FeatureAdmission | None = None,
) -> _TrainReferenceGroupKey:
    """Key under which folds share one train-reference replay.

    The train reference depends on run constants plus ``train_start``, ``train_end``,
    ``slow_horizon_override`` and ``committee_member_weights``; folds with equal keys differ only
    in ``train_end`` and can read prefixes of one replay. Fast-horizon and funding-carry
    overrides are report-only for the reference and are deliberately excluded.

    Returns:
        ``(train_start, slow_horizon_override, sorted (name, float) weight items or None,
        admitted tuple or None)``.
    """
    if committee_member_weights is None:
        weights_key: tuple[tuple[str, float], ...] | None = None
    else:
        weights_key = tuple(sorted((name, float(weight)) for name, weight in committee_member_weights.items()))
    admitted_key = tuple(committee_admission.admitted) if committee_admission is not None else None
    return (fold.train_start, slow_horizon_override, weights_key, admitted_key)


@dataclass(frozen=True, slots=True)
class _SharedTrainReference:
    """One train-reference replay to a group horizon, reusable as per-fold prefixes.

    Attributes:
        group_key: Key shared by every fold of the group.
        reference_start: ``train_start + FOLD_PANEL_WARMUP_HOURS``.
        horizon_end: ``train_end`` of the group's latest usable fold.
        target_weights: Train-window targets on ``[reference_start, horizon_end]`` exactly as
            built for the horizon fold, restricted to columns with any NaN or nonzero entry.
        signal_available_at: Signal times aligned row-for-row with ``target_weights``.
        daily_returns: The horizon fold's certified daily returns (rows ``< horizon_end``).
    """

    group_key: _TrainReferenceGroupKey
    reference_start: pd.Timestamp
    horizon_end: pd.Timestamp
    target_weights: pd.DataFrame
    signal_available_at: pd.DatetimeIndex
    daily_returns: pd.Series


def _fold_reference_targets(
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    fold_index: int,
    slow_horizon_override: int | None,
    committee_member_weights: dict[str, float] | None,
    *,
    committee_admission: FeatureAdmission | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    """Train-window decision path of one fold, without replay.

    Returns:
        ``(target_weights, signal_available_at, minute_roster)`` from
        ``fold_weights._build_fold_target_weights`` with ``decision_start = train_start +
        FOLD_PANEL_WARMUP_HOURS`` and ``decision_end = train_end``.
    Raises:
        DataIntegrityError: Empty reference window (message unchanged:
            ``"fold <i>: train reference window is empty; do not borrow pre-DISCOVERY data"``).
        ValueError, RuntimeError, DataIntegrityError: Propagated from the target build.
    """
    reference_start = fold.train_start + pd.Timedelta(hours=FOLD_PANEL_WARMUP_HOURS)
    reference_end = fold.train_end
    if not reference_start < reference_end:
        raise DataIntegrityError(f"fold {fold_index}: train reference window is empty; do not borrow pre-DISCOVERY data")
    target_weights, signal_available_at, minute_roster, _grid_1h = fold_weights._build_fold_target_weights(
        root, fold, request, funding_by_symbol, slow_horizon_override, committee_member_weights,
        decision_start=reference_start, decision_end=reference_end,
        committee_admission=committee_admission,
    )
    return target_weights, signal_available_at, minute_roster


def _replay_fold_train_reference(
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    initial_equity: float,
    fold_index: int,
    target_weights: pd.DataFrame,
    signal_available_at: pd.DatetimeIndex,
    minute_roster: list[str],
) -> pd.Series:
    """Replay a fold's train decision path and return certified daily returns before ``train_end``.

    Raises:
        DataIntegrityError: perf_01a ledger certification, returns validity
            (``_assert_train_reference_returns_valid``) or replay integrity errors, unchanged.
    """
    reference_start = fold.train_start + pd.Timedelta(hours=FOLD_PANEL_WARMUP_HOURS)
    reference_end = fold.train_end
    target_replay = target_weights[minute_roster]

    def _ref_windows(target_frame: pd.DataFrame, signals: pd.DatetimeIndex) -> Iterator[MhsExecutionWindow]:
        base_spec = specs._resolved_base_execution_spec(request)
        execution_grid = pd.date_range(reference_start, reference_end, freq="3min", tz="UTC")
        truncated, truncated_signals, _censored = integrity._truncate_replayable_decisions(target_frame, signals, execution_grid, base_spec)
        yield from windows._iter_mhs_execution_windows(truncated, truncated_signals, root, request.execution_timeframe, reference_start, reference_end, funding_by_symbol, base_spec)

    _ref_iter = _ref_windows(target_replay, signal_available_at)
    base_spec = specs._resolved_base_execution_spec(request)
    ref_replay = replay_execution_windows(_ref_iter, initial_equity, "OHLCV_IMMEDIATE_TAKER", base_spec, retain_event_snapshots=False)
    integrity._assert_train_reference_ledger_certified(ref_replay, fold_index)
    daily = ref_replay.ledger.equity.resample("1D").last().dropna().pct_change().dropna().astype("float64")
    daily = pd.Series(daily.to_numpy(dtype="float64"), index=daily.index, dtype="float64")
    daily = daily.loc[daily.index < fold.train_end]
    integrity._assert_train_reference_returns_valid(daily, fold.train_end, fold_index)
    del target_weights, target_replay
    return daily


def _fold_train_reference_returns(
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    initial_equity: float,
    fold_index: int,
    slow_horizon_override: int | None,
    committee_member_weights: dict[str, float] | None,
    *,
    committee_admission: FeatureAdmission | None = None,
) -> pd.Series:
    target_weights, signal_available_at, minute_roster = _fold_reference_targets(
        root, fold, request, funding_by_symbol, fold_index, slow_horizon_override, committee_member_weights,
        committee_admission=committee_admission,
    )
    return _replay_fold_train_reference(
        root, fold, request, funding_by_symbol, initial_equity, fold_index,
        target_weights, signal_available_at, minute_roster,
    )


def _active_reference_columns(frame: pd.DataFrame) -> set[str]:
    """Columns carrying any NaN or nonzero entry (all-zero columns are absent = 0.0)."""
    active: set[str] = set()
    for column in frame.columns:
        series = frame[column]
        if bool(series.isna().any()) or bool((series.fillna(0.0) != 0.0).any()):
            active.add(str(column))
    return active


def _build_shared_train_reference(
    root: str,
    group_folds: tuple[tuple[int, AnchoredPurgedFold], ...],
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    initial_equity: float,
    slow_horizon_override: int | None,
    committee_member_weights: dict[str, float] | None,
    *,
    committee_admissions: Mapping[int, FeatureAdmission] | None = None,
) -> _SharedTrainReference | None:
    """Replay the train reference once, to the latest ``train_end`` of a fold group.

    The horizon fold builds its targets with its OWN
    admission ``committee_admissions[horizon_index]`` (cutoff = its train_end); a
    missing admission fails inside the guarded build and returns None.

    The result is an optimization artifact, never evidence on its own: any failure of the
    shared build, replay or perf_01a certification returns None so each fold recomputes its
    independent reference and reports its own verdict (a shared failure may stem from data
    after a shorter fold's ``train_end`` and must not leak into that fold).

    Args:
        group_folds: ``(fold_index, fold)`` pairs with one common group key, at least two.
    Returns:
        The shared reference, or None when the shared computation failed.
    Raises:
        ValueError: Precondition violation (fewer than two folds, mixed ``train_start``),
            raised before any guarded work and never converted to None.
    """
    if len(group_folds) < 2:
        raise ValueError(f"shared train reference requires at least two folds, got {len(group_folds)}")
    first_start = group_folds[0][1].train_start
    if any(fold.train_start != first_start for _, fold in group_folds):
        raise ValueError("shared train reference requires one common train_start")
    if committee_admissions is not None:
        _adm_tuples = set()
        for idx, _fold in group_folds:
            _adm = (committee_admissions or {}).get(idx)
            _adm_tuples.add(tuple(_adm.admitted) if _adm is not None else None)
        if len(_adm_tuples) > 1:
            raise ValueError("shared train reference requires one common admitted tuple")
    horizon_index, horizon_fold = group_folds[0]
    for candidate_index, candidate_fold in group_folds[1:]:
        if candidate_fold.train_end > horizon_fold.train_end or (
            candidate_fold.train_end == horizon_fold.train_end and candidate_index < horizon_index
        ):
            horizon_index, horizon_fold = candidate_index, candidate_fold
    ordered_indices = sorted(idx for idx, _ in group_folds)
    try:
        target_weights, signal_available_at, minute_roster = _fold_reference_targets(
            root, horizon_fold, request, funding_by_symbol, horizon_index,
            slow_horizon_override, committee_member_weights,
            committee_admission=(committee_admissions or {}).get(horizon_index),
        )
        daily_returns = _replay_fold_train_reference(
            root, horizon_fold, request, funding_by_symbol, initial_equity, horizon_index,
            target_weights, signal_available_at, minute_roster,
        )
    except (DataIntegrityError, RuntimeError, ValueError) as exc:
        _logger.info(
            "[RISK] shared_train_reference folds=%s horizon_end=%s outcome=unavailable cause=%s",
            ",".join(str(idx) for idx in ordered_indices), horizon_fold.train_end, type(exc).__name__,
        )
        return None
    active = _active_reference_columns(target_weights)
    restricted = target_weights[[c for c in target_weights.columns if str(c) in active]] if active else target_weights.iloc[:, 0:0]
    shared = _SharedTrainReference(
        group_key=_train_reference_group_key(
            horizon_fold, slow_horizon_override, committee_member_weights,
            committee_admission=(committee_admissions or {}).get(horizon_index),
        ),
        reference_start=horizon_fold.train_start + pd.Timedelta(hours=FOLD_PANEL_WARMUP_HOURS),
        horizon_end=horizon_fold.train_end,
        target_weights=restricted,
        signal_available_at=signal_available_at,
        daily_returns=daily_returns,
    )
    _logger.info(
        "[RISK] shared_train_reference folds=%s horizon_end=%s outcome=built cause=none",
        ",".join(str(idx) for idx in ordered_indices), horizon_fold.train_end,
    )
    return shared


def _shared_train_reference_slice(
    shared: _SharedTrainReference,
    fold: AnchoredPurgedFold,
    fold_index: int,
    own_target_weights: pd.DataFrame,
    own_signal_available_at: pd.DatetimeIndex,
    spec: ExecutionSpec,
) -> pd.Series | None:
    """Return fold k's train reference as a prefix of the shared replay, or None to fall back.

    Reuse is allowed only when the shared replay provably consumed fold k's own decision path
    before ``train_end`` (the only channel through which data after ``train_end`` could reach
    the prefix) and the prefix boundary cannot read past ``train_end``. Otherwise the caller
    replays fold k independently, which is always correct.

    Args:
        own_target_weights, own_signal_available_at: Fold k's own train-window targets and
            signals (``_fold_reference_targets``), compared against the shared prefix.
        spec: Resolved base execution spec (censoring timeout).
    Returns:
        Daily returns with index ``< fold.train_end`` from ``shared.daily_returns``, or None.
    """
    expected_start = fold.train_start + pd.Timedelta(hours=FOLD_PANEL_WARMUP_HOURS)
    if not (expected_start == shared.reference_start and fold.train_end <= shared.horizon_end):
        _logger.debug("[RISK] train_reference fold=%s source=independent reason=horizon", fold_index)
        return None
    if fold.train_end != fold.train_end.normalize():
        _logger.debug("[RISK] train_reference fold=%s source=independent reason=train_end_not_midnight", fold_index)
        return None
    mask = np.asarray(shared.target_weights.index <= fold.train_end)
    prefix = shared.target_weights.loc[mask] if len(shared.target_weights) else shared.target_weights
    try:
        prefix_signals = shared.signal_available_at[mask]
    except (IndexError, ValueError):
        _logger.debug("[RISK] train_reference fold=%s source=independent reason=target_prefix_mismatch", fold_index)
        return None
    if not own_target_weights.index.equals(prefix.index):
        _logger.debug("[RISK] train_reference fold=%s source=independent reason=target_prefix_mismatch", fold_index)
        return None
    if not own_signal_available_at.equals(prefix_signals):
        _logger.debug("[RISK] train_reference fold=%s source=independent reason=target_prefix_mismatch", fold_index)
        return None
    own_active = _active_reference_columns(own_target_weights)
    prefix_active = _active_reference_columns(prefix)
    if own_active != prefix_active:
        _logger.debug("[RISK] train_reference fold=%s source=independent reason=target_prefix_mismatch", fold_index)
        return None
    for column in own_active:
        own_series = own_target_weights[column]
        prefix_series = prefix[column]
        own_nan = own_series.isna().to_numpy()
        prefix_nan = prefix_series.isna().to_numpy()
        if not bool((own_nan == prefix_nan).all()):
            _logger.debug("[RISK] train_reference fold=%s source=independent reason=target_prefix_mismatch", fold_index)
            return None
        finite = ~own_nan
        if finite.any():
            delta = float(np.max(np.abs(
                own_series.to_numpy(dtype="float64")[finite] - prefix_series.to_numpy(dtype="float64")[finite]
            )))
            if not delta <= TRAIN_REFERENCE_PREFIX_TARGET_ATOL:
                _logger.debug("[RISK] train_reference fold=%s source=independent reason=target_prefix_mismatch", fold_index)
                return None
    execution_grid = pd.date_range(shared.reference_start, fold.train_end, freq="3min", tz="UTC")
    truncated, _truncated_signals, _censored = integrity._truncate_replayable_decisions(
        own_target_weights, own_signal_available_at, execution_grid, spec,
    )
    retained = set(truncated.index)
    step = execution_grid[1] - execution_grid[0] if len(execution_grid) >= 2 else pd.Timedelta(minutes=3)
    cutoff = fold.train_end - step
    for position, decision_time in enumerate(own_target_weights.index):
        if decision_time in retained:
            continue
        if not own_signal_available_at[position] >= cutoff:
            _logger.debug("[RISK] train_reference fold=%s source=independent reason=censor_boundary", fold_index)
            return None
    sliced = shared.daily_returns.loc[shared.daily_returns.index < fold.train_end]
    try:
        integrity._assert_train_reference_returns_valid(sliced, fold.train_end, fold_index)
    except DataIntegrityError:
        _logger.debug("[RISK] train_reference fold=%s source=independent reason=slice_invalid", fold_index)
        return None
    _logger.debug("[RISK] train_reference fold=%s source=shared", fold_index)
    return pd.Series(sliced.to_numpy(dtype="float64"), index=sliced.index, dtype="float64")


def _reference_from_shared(
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    initial_equity: float,
    fold_index: int,
    slow_horizon_override: int | None,
    committee_member_weights: dict[str, float] | None,
    shared: _SharedTrainReference,
    *,
    committee_admission: FeatureAdmission | None = None,
) -> pd.Series:
    """Fold k's train reference: the shared prefix when every reuse guard holds, else independent.

    Raises:
        Exactly what ``_fold_train_reference_returns`` raises for this fold (the own target
        build always runs; on fallback the own replay runs on those same targets).
    """
    own_target_weights, own_signal_available_at, minute_roster = _fold_reference_targets(
        root, fold, request, funding_by_symbol, fold_index, slow_horizon_override, committee_member_weights,
        committee_admission=committee_admission,
    )
    if _train_reference_group_key(
        fold, slow_horizon_override, committee_member_weights,
        committee_admission=committee_admission,
    ) != shared.group_key:
        _logger.debug("[RISK] train_reference fold=%s source=independent reason=group_key", fold_index)
        return _replay_fold_train_reference(
            root, fold, request, funding_by_symbol, initial_equity, fold_index,
            own_target_weights, own_signal_available_at, minute_roster,
        )
    sliced = _shared_train_reference_slice(
        shared, fold, fold_index, own_target_weights, own_signal_available_at,
        specs._resolved_base_execution_spec(request),
    )
    if sliced is not None:
        return sliced
    return _replay_fold_train_reference(
        root, fold, request, funding_by_symbol, initial_equity, fold_index,
        own_target_weights, own_signal_available_at, minute_roster,
    )


def _fold_failure_report(
    fold: AnchoredPurgedFold, fold_index: int, exc: Exception,
) -> MhsFoldReport:
    """Incomplete-fold report for an expected fold error, with the stable reason code.

    ``DataIntegrityError`` (a ``ValueError`` subclass, so checked first) maps through
    ``integrity._classify_execution_failure``; any other ``RuntimeError``/``ValueError`` maps to
    ``INCOMPLETE_ANCHORED_FOLD``. Shared by ``_run_anchored_fold`` and the validation phase so
    both report identical codes.
    """
    if isinstance(exc, DataIntegrityError):
        return _incomplete_fold_report(fold, fold_index, (integrity._classify_execution_failure(exc),))
    return _incomplete_fold_report(fold, fold_index, (GO_REASON_INCOMPLETE_FOLD,))


@dataclass(frozen=True, slots=True)
class _FoldValidationPlan:
    """Validation-window decision inputs of one anchored fold, built once and replayed as-is.

    Built before the train-reference replay so a fold whose validation window cannot be
    evaluated exits without paying for that replay, and consumed unchanged by the
    validation replays so the targets are never rebuilt. Picklable (module-level, pandas
    fields only) so a later phase may transport it across a fork-pool boundary.

    Attributes:
        target_weights: Decision-grid targets exactly as returned by
            ``fold_weights._build_fold_target_weights`` for the validation window (all
            aligned columns); feeds ``books._book_structure_trace``.
        target_replay: ``target_weights[minute_roster]`` after terminal censoring on the
            validation 3m execution grid; the replayed decision path.
        signal_available_at: Signal times aligned row-for-row with ``target_replay``.
        terminal_censored: Decisions censored by ``_truncate_replayable_decisions``.
        decision_intents: Count of finite cells in ``target_replay``.
    """

    target_weights: pd.DataFrame
    target_replay: pd.DataFrame
    signal_available_at: pd.DatetimeIndex
    terminal_censored: int
    decision_intents: int


def _build_fold_validation_plan(
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    slow_horizon_override: int | None,
    committee_member_weights: dict[str, float] | None,
    *,
    committee_admission: FeatureAdmission | None = None,
) -> _FoldValidationPlan:
    """Build one fold's validation decision path and prove it is replayable.

    Uses only data the validation window itself reads (panel from
    ``max(train_start, validation_start - FOLD_PANEL_WARMUP_HOURS)`` to
    ``validation_end``); never touches the train reference. The execution grid is the
    3-minute grid ``[validation_start, validation_end]`` with the request's resolved base
    execution spec, identical to the grid the validation replays use.

    Raises:
        ValueError, RuntimeError: Validation window unusable (no panel survivor, no
            funded/aligned symbol, empty decision grid, no minute roster), propagated
            unchanged from ``fold_weights._build_fold_target_weights``.
        DataIntegrityError: Propagated unchanged from the target build or the
            terminal-censoring helper.
    """
    target_weights, signal_available_at, minute_roster, _grid_1h = fold_weights._build_fold_target_weights(
        root, fold, request, funding_by_symbol, slow_horizon_override, committee_member_weights,
        committee_admission=committee_admission,
    )
    execution_grid = pd.date_range(
        fold.validation_start, fold.validation_end,
        freq="3min",
        tz="UTC",
    )
    target_replay, signal_available_at, terminal_censored = integrity._truncate_replayable_decisions(
        target_weights[minute_roster], signal_available_at, execution_grid,
        specs._resolved_base_execution_spec(request),
    )
    return _FoldValidationPlan(
        target_weights=target_weights,
        target_replay=target_replay,
        signal_available_at=signal_available_at,
        terminal_censored=terminal_censored,
        decision_intents=int(np.isfinite(target_replay.to_numpy()).sum()),
    )


def _run_anchored_fold(
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    initial_equity: float,
    fold_index: int,
    telemetry: _StageRecorder | None = None,
    slow_horizon_override: int | None = None,
    fast_horizon_override: tuple[int, str] | None = None,
    funding_carry_override: tuple[int | None, int | None, str, float | None] | None = None,
    committee_member_weights: dict[str, float] | None = None,
    *,
    committee_admission: FeatureAdmission | None = None,
    validation_plan: _FoldValidationPlan | None = None,
    shared_reference: _SharedTrainReference | None = None,
) -> MhsFoldReport:
    """Replay one anchored fold: validation plan first, then the train-only sizing reference.

    The validation decision path is built before the train-reference replay so a fold whose
    validation window is provably unusable fails closed immediately with the incomplete-fold
    report (``INCOMPLETE_ANCHORED_FOLD`` for ``RuntimeError``/``ValueError``, the classified
    code for ``DataIntegrityError``) without spending the reference replay. When both the
    validation build and the reference would fail, the validation failure is reported. The
    plan is built exactly once and replayed unchanged; train, validation and replay slices
    stay chronological, and the same 3m OHLCV economics drive every replay.

    ``validation_plan`` injects a plan already built by the validation phase for this exact
    fold and request (never rebuilt); ``shared_reference`` offers a group replay whose prefix
    replaces this fold's train-reference replay only when every reuse guard of
    ``_shared_train_reference_slice`` holds, else the independent reference is computed as
    without it.
    """
    try:
        vs = fold.validation_start
        ve = fold.validation_end
        if validation_plan is None:
            plan = _build_fold_validation_plan(
                root, fold, request, funding_by_symbol, slow_horizon_override, committee_member_weights,
                committee_admission=committee_admission,
            )
        else:
            plan = validation_plan
        if shared_reference is None:
            train_reference = _fold_train_reference_returns(root, fold, request, funding_by_symbol, initial_equity, fold_index, slow_horizon_override, committee_member_weights, committee_admission=committee_admission)
        else:
            train_reference = _reference_from_shared(root, fold, request, funding_by_symbol, initial_equity, fold_index, slow_horizon_override, committee_member_weights, shared_reference, committee_admission=committee_admission)
        if telemetry is not None:
            telemetry.record(f"anchored_fold_{fold_index}_sizing_reference", grid_bars=len(train_reference), window_start=str(train_reference.index[0]), window_end=str(train_reference.index[-1]))
        from src.mhs.research_go import _resolved_growth_envelope as _resolve_envelope
        _envelope = _resolve_envelope(request)
        if str(request.pnl_vol_target_mode) in ("growth_budget", "constant_risk"):
            _local_target_vol: float | None = _scaling._growth_budget_target_vol_by_boundary(train_reference, _envelope, {f"fold_{fold_index}": fold.train_end})[f"fold_{fold_index}"]
        else:
            _local_target_vol = None

        # Fork workers get the SYSTEM reserve check (not the auto 85% budget,
        # whose fork-child RSS would double-count COW-shared parent pages).
        _window_rss_reserve = _resolve_ram_budget(None, request.ram_guard)[1]

        def _windows() -> Iterator[MhsExecutionWindow]:
            return windows._iter_mhs_execution_windows(
                plan.target_replay, plan.signal_available_at, root, request.execution_timeframe,
                vs, ve, funding_by_symbol, specs._resolved_base_execution_spec(request),
            )

        def _window_telemetry(
            gen: Iterator[MhsExecutionWindow], prefix: str,
        ) -> Iterator[MhsExecutionWindow]:
            for idx, w in enumerate(gen):
                if telemetry is not None:
                    telemetry.record(
                        f"{prefix}_{idx}",
                        grid_bars=len(w.minute_grid),
                        active_symbols=len(w.symbols),
                        window_start=str(w.window_start),
                        window_end=str(w.window_end),
                    )
                yield w
                _assert_execution_rss_budget(
                    prefix, request.max_rss_bytes, idx + 1,
                    reserve_bytes=_window_rss_reserve,
                )

        window_prefix = f"anchored_fold_{fold_index}_window"
        # Streaming replay: reference pass streams directly; the rescaled
        # primary/stress pair reuses one regenerated window stream.
        cached_windows = list(_windows())
        primary = replay_execution_windows(
            _window_telemetry(iter(cached_windows), window_prefix),
            initial_equity, "OHLCV_IMMEDIATE_TAKER", specs._resolved_base_execution_spec(request),
            retain_event_snapshots=False,
        )
        # Two-pass primary (reference -> P&L-vol-target rescale -> reported):
        # fold-local train reference로 적합한 단일 target을 모든 모드에 그대로 쓴다.
        reference_daily_returns = primary.ledger.equity.resample("1D").last().pct_change()
        pnl_vol_target_scale = _scaling._replay_exposure_scale(
            reference_daily_returns, request, _local_target_vol,
            warmup_returns=train_reference,
        )
        primary, stress = replay_execution_window_batch(
            _window_telemetry(
                windows._rescaled_windows(iter(cached_windows), pnl_vol_target_scale),
                f"{window_prefix}_rescaled",
            ),
            initial_equity,
            [
                ("OHLCV_IMMEDIATE_TAKER", specs._resolved_base_execution_spec(request)),
                ("OHLCV_IMMEDIATE_TAKER", specs._stress_cost_execution_spec(specs._resolved_base_execution_spec(request))),
            ],
            retain_event_snapshots=False,
        )

        failures: list[str] = []
        equity = primary.ledger.equity
        # 관측 전용: 스케일이 실제 적용된 배치 원장의 실현 연변동성
        # (fold_realized_risk_parity 입력) -- 스케일 이전 참조 패스가 아니다.
        _deployed_daily = equity.resample("1D").last().pct_change().dropna()
        realized_annualized_vol = (
            float(_deployed_daily.std(ddof=1) * np.sqrt(365.0))
            if len(_deployed_daily) >= 2
            else None
        )
        if not np.isfinite(equity.to_numpy()).all() or not (equity > 0).all():
            failures.append(GO_REASON_NONFINITE_EQUITY)
        # Disclosed terminal inventory is evidence, not data loss: the shared
        # integrity.replay_ledger_certified verdict keeps the alpha verdict clean
        # (deployment is still blocked downstream by backtest reliability).
        # 단일 인증 헬퍼로 합산 판정한다(인라인 중복 규칙 금지).
        certified = integrity.replay_ledger_certified(primary)
        if not certified:
            failures.append(GO_REASON_INVALID_PRIMARY)
        if (
            primary.termination_counts.get("MISSING_DATA", 0) > 0
            or (primary.termination_counts.get("UNKNOWN_TERMINATION", 0) > 0 and not certified)
        ):
            failures.append(GO_REASON_EXECUTION_GAP)
        _fold_debug_mode = ("adaptive" if request.committee_regime_adaptive_tranche else str(request.committee_tranche_count) if request.committee_tranche_smoothing else "1")
        _fold_debug_tag = (
            f"fold{fold_index}_tranche{_fold_debug_mode}"
            if request.committee_capital else None
        )
        # (제거) fold별 level 코드 3개를 failures에 append하지 않는다 -- level은
        # pooled 하한 게이트(research_go)의 단일 소유다. 아래 값들은 MhsFoldReport
        # 관측 기록용으로 유지된다.
        primary_autocorr = _statistics._daily_autocorr_sharpe(primary.ledger, debug_tag=_fold_debug_tag)
        stress_sharpe = _statistics._naive_sharpe(stress.ledger)

        equity_1h, net_returns_1h, _turnover_1h = _statistics._hourly_ledger_series(
            equity, primary.ledger.fill_turnover,
        )
        primary_net_ann = _statistics._mean_ann(net_returns_1h, _PERIODS_PER_YEAR_1H)
        if _fold_debug_tag is not None and _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "[EVAL] tag=%s ann_turnover=%.3f ann_net_ret=%.4f mdd=%.4f",
                _fold_debug_tag,
                _statistics._mean_ann(_turnover_1h, _PERIODS_PER_YEAR_1H),
                _statistics._mean_ann(net_returns_1h, _PERIODS_PER_YEAR_1H),
                _statistics._mdd(equity),
            )
        return MhsFoldReport(
            fold_index=fold_index,
            validation_start=str(vs),
            validation_end=str(ve),
            strict=primary,
            stress=stress,
            primary_valid=primary.ledger.primary_valid,
            primary_autocorr_sharpe=primary_autocorr,
            primary_naive_sharpe=_statistics._naive_sharpe(primary.ledger),
            primary_net_ann=primary_net_ann,
            primary_geometric_cagr=_statistics._geometric_cagr(equity_1h),
            primary_max_drawdown=_statistics._mdd(equity),
            stress_naive_sharpe=stress_sharpe,
            decision_intents=plan.decision_intents,
            termination_counts=dict(primary.termination_counts),
            failures=tuple(sorted(set(failures))),
            strict_elapsed_seconds=primary.elapsed_seconds,
            stress_elapsed_seconds=stress.elapsed_seconds,
            terminal_censored_decisions=plan.terminal_censored,
            slow_horizon_hours=(
                slow_horizon_override
                if slow_horizon_override is not None
                else BOOK_SPECS["slow_momentum"].horizon_hours
            ),
            slow_horizon_source=(
                "fold_train_only_discovery" if slow_horizon_override is not None else "frozen_default"
            ),
            fast_horizon_hours=(
                fast_horizon_override[0]
                if fast_horizon_override is not None
                else BOOK_SPECS["fast_reversal"].horizon_hours
            ),
            fast_horizon_source=(
                fast_horizon_override[1] if fast_horizon_override is not None else "frozen_default"
            ),
            funding_carry_lookback_hours=(
                funding_carry_override[0] if funding_carry_override is not None else None
            ),
            funding_carry_sign=(
                funding_carry_override[1] if funding_carry_override is not None else None
            ),
            funding_carry_source=(
                funding_carry_override[2]
                if funding_carry_override is not None
                else "frozen_default"
            ),
            funding_carry_vs_slow_momentum_daily_corr=(
                funding_carry_override[3] if funding_carry_override is not None else None
            ),
            book_structure={
                **books._book_structure_trace(plan.target_weights),
                # Deployed-gross observability: the parity guard must see the
                # exposure scale actually applied to this fold, not just the
                # pre-scale decision book.
                "exposure_scale_mean": float(pnl_vol_target_scale.mean()),
                "exposure_scale_cap_binding_fraction": float(
                    (pnl_vol_target_scale >= _scaling.resolved_exposure_cap(request) - 1e-12).mean(),
                ),
                "sizing_reference_start": str(train_reference.index[0]),
                "sizing_reference_end": str(train_reference.index[-1]),
                "sizing_reference_daily_rows": float(len(train_reference)),
            },
            regime_characterization=regime._fold_regime_characterization(root, fold),
            realized_annualized_vol=realized_annualized_vol,
        )
    except DataIntegrityError as exc:
        return _fold_failure_report(fold, fold_index, exc)
    except (RuntimeError, ValueError) as exc:
        return _fold_failure_report(fold, fold_index, exc)

def _fold_validation_task(
    token: str,
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    fold_index: int,
    slow_horizon_override: int | None,
    committee_member_weights: dict[str, float] | None,
    *,
    committee_admission: FeatureAdmission | None = None,
) -> _FoldValidationPlan | MhsFoldReport:
    """Phase V: build one fold's validation plan in a fork worker.

    ``fold_funding`` is resolved from the fork-shared payload ``token``. An expected failure
    returns the fold's final incomplete report (``_fold_failure_report``), exactly the report
    ``_run_anchored_fold`` would produce for that failure.
    """
    funding_by_symbol: dict[str, pd.Series] = resolve_fork_shared(token)["fold_funding"]
    try:
        return _build_fold_validation_plan(
            root, fold, request, funding_by_symbol, slow_horizon_override, committee_member_weights,
            committee_admission=committee_admission,
        )
    except (DataIntegrityError, RuntimeError, ValueError) as exc:
        return _fold_failure_report(fold, fold_index, exc)


def _shared_train_reference_task(
    token: str,
    root: str,
    group_folds: tuple[tuple[int, AnchoredPurgedFold], ...],
    request: MhsDiagnosticRequest,
    initial_equity: float,
    slow_horizon_override: int | None,
    committee_member_weights: dict[str, float] | None,
    *,
    committee_admissions: Mapping[int, FeatureAdmission] | None = None,
) -> _SharedTrainReference | None:
    """Phase R: ``_build_shared_train_reference`` in a fork worker (funding via ``token``)."""
    funding_by_symbol: dict[str, pd.Series] = resolve_fork_shared(token)["fold_funding"]
    return _build_shared_train_reference(
        root, group_folds, request, funding_by_symbol, initial_equity,
        slow_horizon_override, committee_member_weights,
        committee_admissions=committee_admissions,
    )


def _anchored_fold_task(
    token: str,
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    initial_equity: float,
    fold_index: int,
    slow_horizon_override: int | None,
    fast_horizon_override: tuple[int, str] | None,
    funding_carry_override: tuple[int | None, int | None, str, float | None] | None,
    committee_member_weights: dict[str, float] | None,
    validation_plan: _FoldValidationPlan,
    shared_reference: _SharedTrainReference | None,
    *,
    committee_admission: FeatureAdmission | None = None,
) -> MhsFoldReport:
    """Phase E: the module-global ``_run_anchored_fold`` (telemetry None) with the injected plan
    and optional shared reference, funding resolved from ``token``."""
    funding_by_symbol: dict[str, pd.Series] = resolve_fork_shared(token)["fold_funding"]
    return _run_anchored_fold(
        root, fold, request, funding_by_symbol, initial_equity, fold_index, None,
        slow_horizon_override, fast_horizon_override, funding_carry_override,
        committee_member_weights, committee_admission=committee_admission,
        validation_plan=validation_plan, shared_reference=shared_reference,
    )


def _submit_fold_validation_phase(
    pool: Executor,
    token: str,
    root: str,
    fold_list: Sequence[AnchoredPurgedFold],
    request: MhsDiagnosticRequest,
    fold_slow_horizons: Mapping[int, int | None] | None,
    fold_committee_weights: Mapping[int, dict[str, float]] | None,
    fold_committee_admission: Mapping[int, FeatureAdmission] | None = None,
) -> dict[Future[_FoldValidationPlan | MhsFoldReport], int]:
    """Submit phase V for every fold; returns futures keyed to fold indices.

    Submits at least one task whenever ``fold_list`` is non-empty, so a fork pool has forked
    all workers when this returns and the caller may start threads afterwards.
    """
    validation_futures: dict[Future[_FoldValidationPlan | MhsFoldReport], int] = {}
    for idx, fold in enumerate(fold_list):
        future = pool.submit(
            _fold_validation_task,
            token, root, fold, request, idx,
            (fold_slow_horizons or {}).get(idx),
            (fold_committee_weights or {}).get(idx),
            committee_admission=(fold_committee_admission or {}).get(idx),
        )
        validation_futures[future] = idx
    return validation_futures


def _complete_anchored_folds(
    pool: Executor,
    token: str,
    validation_futures: Mapping[Future[_FoldValidationPlan | MhsFoldReport], int],
    root: str,
    fold_list: Sequence[AnchoredPurgedFold],
    request: MhsDiagnosticRequest,
    initial_equity: float,
    fold_slow_horizons: Mapping[int, int | None] | None,
    fold_fast_horizons: Mapping[int, tuple[int, str]] | None,
    fold_funding_carry: Mapping[int, tuple[int | None, int | None, str, float | None]] | None,
    fold_committee_weights: Mapping[int, dict[str, float]] | None,
    fold_committee_admission: Mapping[int, FeatureAdmission] | None = None,
) -> tuple[MhsFoldReport, ...]:
    """Finish phases V → R → E and return one report per fold in fold-index order.

    Validation-unusable folds keep their phase-V incomplete report and never contribute to a
    group horizon. Usable folds whose reference window is non-empty
    (``train_start + FOLD_PANEL_WARMUP_HOURS < train_end``) are grouped by
    ``_train_reference_group_key``; each group of two or more gets one phase-R task, and its
    folds' phase-E tasks receive the result. Every other usable fold runs phase E immediately
    with no shared reference.
    """
    plans: dict[int, _FoldValidationPlan] = {}
    reports: dict[int, MhsFoldReport] = {}
    for future, idx in validation_futures.items():
        result = future.result()
        if isinstance(result, MhsFoldReport):
            reports[idx] = result
        else:
            plans[idx] = result
    usable = sorted(plans.keys())
    groups: dict[_TrainReferenceGroupKey, list[int]] = {}
    for idx in usable:
        fold = fold_list[idx]
        if not fold.train_start + pd.Timedelta(hours=FOLD_PANEL_WARMUP_HOURS) < fold.train_end:
            continue
        key = _train_reference_group_key(
            fold, (fold_slow_horizons or {}).get(idx), (fold_committee_weights or {}).get(idx),
            committee_admission=(fold_committee_admission or {}).get(idx),
        )
        groups.setdefault(key, []).append(idx)
    shared_groups = sorted(
        (sorted(indices) for indices in groups.values() if len(indices) >= 2),
        key=lambda indices: indices[0],
    )
    shared_indices = {idx for group in shared_groups for idx in group}
    _logger.info(
        "[RISK] train_reference_plan folds=%s usable=%s shared_groups=%s shared_folds=%s independent_folds=%s",
        len(fold_list), len(usable), len(shared_groups), len(shared_indices), len(usable) - len(shared_indices),
    )
    fold_e_futures: dict[Future[MhsFoldReport], int] = {}
    for idx in usable:
        if idx in shared_indices:
            continue
        fold_e_futures[pool.submit(
            _anchored_fold_task,
            token, root, fold_list[idx], request, initial_equity, idx,
            (fold_slow_horizons or {}).get(idx),
            (fold_fast_horizons or {}).get(idx),
            (fold_funding_carry or {}).get(idx),
            (fold_committee_weights or {}).get(idx),
            plans[idx], None,
            committee_admission=(fold_committee_admission or {}).get(idx),
        )] = idx
    reference_futures: dict[Future[_SharedTrainReference | None], list[int]] = {}
    for group in shared_groups:
        group_folds = tuple((idx, fold_list[idx]) for idx in group)
        reference_futures[pool.submit(
            _shared_train_reference_task,
            token, root, group_folds, request, initial_equity,
            (fold_slow_horizons or {}).get(group[0]),
            (fold_committee_weights or {}).get(group[0]),
            committee_admissions={idx: (fold_committee_admission or {})[idx] for idx in group if idx in (fold_committee_admission or {})} or None,
        )] = group
    for reference_future in as_completed(reference_futures):
        group = reference_futures[reference_future]
        shared = reference_future.result()
        for idx in group:
            fold_e_futures[pool.submit(
                _anchored_fold_task,
                token, root, fold_list[idx], request, initial_equity, idx,
                (fold_slow_horizons or {}).get(idx),
                (fold_fast_horizons or {}).get(idx),
                (fold_funding_carry or {}).get(idx),
                (fold_committee_weights or {}).get(idx),
                plans[idx], shared,
                committee_admission=(fold_committee_admission or {}).get(idx),
            )] = idx
    for fold_future, idx in fold_e_futures.items():
        reports[idx] = fold_future.result()
    return tuple(reports[i] for i in range(len(fold_list)))


def _run_folds_parallel(
    root: str,
    request: MhsDiagnosticRequest,
    fold_funding: dict[str, pd.Series],
    initial_equity: float,
    telemetry: _StageRecorder | None = None,
    fold_slow_horizons: dict[int, int | None] | None = None,
    fold_fast_horizons: dict[int, tuple[int, str]] | None = None,
    fold_funding_carry: dict[int, tuple[int | None, int | None, str, float | None]] | None = None,
    fold_committee_weights: dict[int, dict[str, float]] | None = None,
    fold_committee_admission: Mapping[int, FeatureAdmission] | None = None,
) -> tuple[MhsFoldReport, ...]:
    """Run all resolved anchored folds in phased V/R/E over a fork pool.

    Each fold builds its own 1h panel and executes an independent strict/stress
    replay pair, so the folds are embarrassingly parallel.  ``ProcessPoolExecutor``
    (fork) keeps each worker's RSS independent and bounded: three workers at a
    measured peak of ~2.6GB each stay well inside the 8GB soft budget.  The
    ``MhsFoldReport`` returned by every worker is picklable (frozen+slots,
    holding only pd.Series/pd.DataFrame/numpy/native types), and per-worker
    telemetry is recorded by the parent after each fold completes.  A fold that
    cannot be replayed is reported (not raised) with machine-readable failure
    codes, matching the sequential path.

    ``fork`` (not ``spawn``) is required: spawn workers re-import the module and
    lose the caller's monkeypatched ``funding_path`` (used
    by the synthetic-market test suite and reproducible diagnostic fixtures),
    and the Phase-1 11.4GiB RSS regression was traced to the main process's own
    top-level matrices and minute-frame retention, not to fork-COW sharing, so
    spawn would not reduce it.
    """
    folds = resolved_anchored_folds(request)
    if not folds:
        return ()
    _folds_reserve = _resolve_ram_budget(request.max_rss_bytes, request.ram_guard)[1]
    max_workers = plan_worker_count(
        min(3, len(folds)), WORKER_PEAK_RSS_BYTES, request.ram_guard,
        observer=_worker_plan_observer(telemetry, "anchored_folds", WORKER_PEAK_RSS_BYTES),
        reserve_bytes=_folds_reserve,
    )
    assert_fork_admission("anchored_folds", max_workers, WORKER_PEAK_RSS_BYTES, _folds_reserve)
    with (
        fork_shared_payload({"fold_funding": fold_funding}) as token,
        frozen_gc_heap(),
        ProcessPoolExecutor(max_workers=max_workers, mp_context=FORK_CONTEXT) as pool,
    ):
        validation_futures = _submit_fold_validation_phase(
            pool, token, root, folds, request, fold_slow_horizons, fold_committee_weights,
            fold_committee_admission,
        )
        ordered = _complete_anchored_folds(
            pool, token, validation_futures, root, folds, request, initial_equity,
            fold_slow_horizons, fold_fast_horizons, fold_funding_carry, fold_committee_weights,
            fold_committee_admission,
        )
    if telemetry is not None:
        for fold_report in ordered:
            fill_count = (
                len(fold_report.strict.simulated_fills) + len(fold_report.stress.simulated_fills)
                if fold_report.strict is not None and fold_report.stress is not None
                else 0
            )
            telemetry.record(f"anchored_fold_{fold_report.fold_index}", fill_count=fill_count)
    return ordered
