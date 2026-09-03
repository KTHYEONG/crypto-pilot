# mypy: ignore-errors
# ruff: noqa: F401, F821, I001, E402
from __future__ import annotations  # mypy: ignore-errors

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from src.application.research.mhs import scaling as _scaling
from src.application.research.mhs import statistics as _statistics
from src.application.research.mhs.contracts import MhsDiagnosticRequest, MhsFoldReport
from src.application.research.mhs.research_go import (
    GO_REASON_EXECUTION_GAP,
    GO_REASON_INCOMPLETE_FOLD,
    GO_REASON_INVALID_PRIMARY,
    GO_REASON_NONFINITE_EQUITY,
)
from src.application.research.mhs.resources import (
    _assert_execution_rss_budget,
    _assert_stage_rss_budget,
    _resolve_ram_budget,
    _StageRecorder,
    _worker_plan_observer,
)
from src.common.errors import DataIntegrityError
from src.mhs.discovery import (
    DiscoveryQualificationResult,
    fold_train_only_discovery_qualification,
)
from src.mhs.evidence import AnchoredPurgedFold, phase_1_anchored_purged_folds
from src.mhs.execution import (
    mhs_ledger_pnl,
    replay_execution_window_batch,
    replay_execution_windows,
)
from src.mhs.parallel import (
    FORK_CONTEXT,
    assert_fork_admission,
    fork_shared_payload,
    plan_worker_count,
    resolve_fork_shared,
)
from src.mhs.params import (
    DISCOVERY_GATE_TRANCHE_COUNT,
    FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
    MEASURED_EXECUTION_COST_TIERS_BPS,
)
from src.mhs.params import (
    PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H,
)
from src.mhs.trend_sleeve import market_basket_log_price, time_series_trend_position, trend_sleeve_weights
from src.mhs.types import BOOK_SPECS, TREND_SLEEVE_HORIZONS_HOURS, WORKER_PEAK_RSS_BYTES, BookSpec

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
    max_workers = plan_worker_count(
        min(3, len(folds)), WORKER_PEAK_RSS_BYTES, ram_guard=True,
        observer=_worker_plan_observer(telemetry, "fold_safe_discovery", WORKER_PEAK_RSS_BYTES),
    )
    _fold_safe_reserve = _resolve_ram_budget(None, True)[1]
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
    growth_budget_target_vol: float | None = None,
    exposure_warmup_returns: pd.Series | None = None,
    blend_exposure_scale: pd.Series | None = None,
) -> MhsFoldReport:
    """One independently flat strict/immediate-taker blend replay per fold.

    The 1h panel spans ``[train_start, validation_end]`` so warm-up history
    feeds features only; the replay decisions and the fresh flat ledger cover
    only the validation window. The fold uses the same at-most-31-day windowed
    execution engine as the top-level books (``windows._iter_mhs_execution_windows`` +
    ``replay_execution_windows``, immediate-taker primary and cost-stressed
    stress) so dense event snapshots stay disabled
    and per-window resource telemetry/RSS budgets are applied inside the fold,
    not only at the top level. A fold that cannot be replayed is reported (not
    raised) with machine-readable failure codes.
    """
    try:
        vs = fold.validation_start
        ve = fold.validation_end
        target_weights, signal_available_at, minute_roster, _grid_1h = fold_weights._build_fold_target_weights(
            root, fold, request, funding_by_symbol, slow_horizon_override, committee_member_weights,
        )
        target_replay = target_weights[minute_roster]
        execution_grid = pd.date_range(
            vs, ve,
            freq={"1m": "1min", "3m": "3min", "5m": "5min"}[request.execution_timeframe],
            tz="UTC",
        )
        target_replay, signal_available_at, terminal_censored = integrity._truncate_replayable_decisions(
            target_replay, signal_available_at, execution_grid, specs._resolved_base_execution_spec(request),
        )
        decision_intents = int(np.isfinite(target_replay.to_numpy()).sum())

        # Fork workers get the SYSTEM reserve check (not the auto 85% budget,
        # whose fork-child RSS would double-count COW-shared parent pages).
        _window_rss_reserve = _resolve_ram_budget(None, request.ram_guard)[1]

        def _windows() -> Iterator[MhsExecutionWindow]:
            return windows._iter_mhs_execution_windows(
                target_replay, signal_available_at, root, request.execution_timeframe,
                vs, ve, funding_by_symbol, request.mark_mode, specs._resolved_base_execution_spec(request),
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
        primary = replay_execution_windows(
            _window_telemetry(_windows(), window_prefix),
            initial_equity, "OHLCV_IMMEDIATE_TAKER", specs._resolved_base_execution_spec(request),
            retain_event_snapshots=False,
        )
        # Two-pass primary (reference -> P&L-vol-target rescale -> reported):
        # constant_risk는 blend가 배치 확정한 exposure_scale을 그대로 슬라이스
        # 재사용한다(I-SCALE-IS-DEPLOYED-OVERLAY, fold-local EWMA 재적합 금지).
        reference_daily_returns = primary.ledger.equity.resample("1D").last().pct_change()
        if request.pnl_vol_target_mode == "constant_risk":
            if blend_exposure_scale is None:
                raise DataIntegrityError(f"fold {fold_index}: constant_risk requires blend_exposure_scale")
            pnl_vol_target_scale = blend_exposure_scale.reindex(reference_daily_returns.index)
            if pnl_vol_target_scale.isna().any():
                raise DataIntegrityError(
                    f"fold {fold_index}: blend exposure_scale missing for "
                    f"{int(pnl_vol_target_scale.isna().sum())} validation dates"
                )
        else:
            pnl_vol_target_scale = _scaling._replay_exposure_scale(
                reference_daily_returns, request, growth_budget_target_vol,
                warmup_returns=_fold_exposure_warmup(exposure_warmup_returns, vs),
            )
        primary, stress = replay_execution_window_batch(
            _window_telemetry(
                windows._rescaled_windows(_windows(), pnl_vol_target_scale),
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
        if not primary.ledger.primary_valid:
            failures.append(GO_REASON_INVALID_PRIMARY)
        if (
            primary.termination_counts.get("MISSING_DATA", 0) > 0
            or primary.termination_counts.get("UNKNOWN_TERMINATION", 0) > 0
        ):
            failures.append(GO_REASON_EXECUTION_GAP)
        _fold_debug_mode = (
            "adaptive" if request.committee_regime_adaptive_tranche
            else "3" if request.committee_tranche_smoothing
            else "1"
        )
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
            decision_intents=decision_intents,
            termination_counts=dict(primary.termination_counts),
            failures=tuple(sorted(set(failures))),
            strict_elapsed_seconds=primary.elapsed_seconds,
            stress_elapsed_seconds=stress.elapsed_seconds,
            terminal_censored_decisions=terminal_censored,
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
                **books._book_structure_trace(target_weights),
                # Deployed-gross observability: the parity guard must see the
                # exposure scale actually applied to this fold, not just the
                # pre-scale decision book.
                "exposure_scale_mean": float(pnl_vol_target_scale.mean()),
                "exposure_scale_cap_binding_fraction": float(
                    (pnl_vol_target_scale >= _scaling.resolved_exposure_cap(request) - 1e-12).mean(),
                ),
            },
            regime_characterization=regime._fold_regime_characterization(root, fold),
            realized_annualized_vol=realized_annualized_vol,
        )
    except DataIntegrityError as exc:
        return _incomplete_fold_report(fold, fold_index, (integrity._classify_execution_failure(exc),))
    except (RuntimeError, ValueError):
        return _incomplete_fold_report(fold, fold_index, (GO_REASON_INCOMPLETE_FOLD,))

def _run_folds_parallel(
    root: str,
    request: MhsDiagnosticRequest,
    fold_funding: dict[str, pd.Series],
    initial_equity: float,
    telemetry: _StageRecorder | None = None,
    fold_slow_horizons: dict[int, int | None] | None = None,
    fold_fast_horizons: dict[int, tuple[int, str]] | None = None,
    fold_funding_carry: dict[int, tuple[int | None, int | None, str, float | None]] | None = None,
    exposure_warmup_returns: pd.Series | None = None,
) -> tuple[MhsFoldReport, ...]:
    """Run the three anchored folds concurrently, one process each.

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
    lose the caller's monkeypatched ``funding_path``/``_mark_price_path`` (used
    by the synthetic-market test suite and reproducible diagnostic fixtures),
    and the Phase-1 11.4GiB RSS regression was traced to the main process's own
    top-level matrices and minute-frame retention, not to fork-COW sharing, so
    spawn would not reduce it.
    """
    folds = phase_1_anchored_purged_folds()
    if not folds:
        return ()
    reports: dict[int, MhsFoldReport] = {}
    max_workers = plan_worker_count(
        min(3, len(folds)), WORKER_PEAK_RSS_BYTES, request.ram_guard,
        observer=_worker_plan_observer(telemetry, "anchored_folds", WORKER_PEAK_RSS_BYTES),
    )
    _folds_reserve = _resolve_ram_budget(request.max_rss_bytes, request.ram_guard)[1]
    assert_fork_admission("anchored_folds", max_workers, WORKER_PEAK_RSS_BYTES, _folds_reserve)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=FORK_CONTEXT) as pool:
        futures = {
            pool.submit(
                _run_anchored_fold,
                root, fold, request, fold_funding, initial_equity, idx, None,
                (fold_slow_horizons or {}).get(idx),
                (fold_fast_horizons or {}).get(idx),
                (fold_funding_carry or {}).get(idx),
                exposure_warmup_returns=exposure_warmup_returns,
            ): idx
            for idx, fold in enumerate(folds)
        }
        for future in as_completed(futures):
            idx = futures[future]
            reports[idx] = future.result()
    ordered = tuple(reports[i] for i in range(len(folds)))
    if telemetry is not None:
        for fold_report in ordered:
            fill_count = (
                len(fold_report.strict.simulated_fills) + len(fold_report.stress.simulated_fills)
                if fold_report.strict is not None and fold_report.stress is not None
                else 0
            )
            telemetry.record(f"anchored_fold_{fold_report.fold_index}", fill_count=fill_count)
    return ordered