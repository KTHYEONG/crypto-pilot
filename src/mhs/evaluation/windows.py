# mypy: ignore-errors
# ruff: noqa: F401, F821, I001, E402
from __future__ import annotations  # mypy: ignore-errors

import dataclasses
import gc
import os
from collections.abc import Iterable, Iterator
from dataclasses import replace as dataclass_replace
from typing import Literal

import numpy as np
import pandas as pd

from src.application.research.mhs import research_go as _research_go
from src.application.research.mhs import scaling as _scaling
from src.application.research.mhs import statistics as _statistics
from src.application.research.mhs.contracts import MhsBookFailure, MhsBookReport, MhsDiagnosticRequest
from src.application.research.mhs.marks import _build_window_frames, _cached_mark_panel, _load_window_minute_frames
from src.application.research.mhs.resources import _assert_execution_rss_budget, _resolve_ram_budget, _StageRecorder
from src.common.errors import DataIntegrityError
from src.mhs.books import portfolio_rebalance_trigger
from src.mhs.evidence import CostResponsePoint, PhaseDiagnosticResult, TailSensitivityResult, book_evidence, phase_1_anchored_purged_folds, required_cost_tiers
from src.mhs.execution import ExecutionReplayWindow, StrategyExecutionReplayResult, bar_funding_panel, replay_execution_window_batch_isolated, replay_execution_windows, replay_execution_windows_coupled
from src.mhs.parallel import resolve_fork_shared
from src.mhs.params import MEASURED_EXECUTION_COST_TIERS_BPS, REBALANCE_TRACKING_ERROR_THRESHOLD, REFERENCE_PASS_EQUITY_FLOOR
from src.mhs.params import PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H
from src.mhs.types import BookSpec, ExecutionSpec

from . import books, integrity, specs


def _resolve_ns_vectorized(
    spos_all: np.ndarray,
    full_grid_ns: np.ndarray,
    n_grid: int,
    timeout_ns_delta: int,
) -> np.ndarray:
    """Vectorized ``resolve_ns`` computation for the window generator.

    Bit-identical to the scalar per-decision loop: ``resolve_ns[i]`` is the
    exact timeout bar ``full_grid_ns[spos_all[i]] + timeout_ns_delta`` when it
    lies on the grid, else ``-1``.  ``searchsorted`` (``side="left"``) keeps the
    same semantics; the ``np.minimum`` guards keep out-of-range positions from
    raising instead of silently skipping (matching the scalar ``continue``).
    """
    resolve_ns = np.full(len(spos_all), -1, dtype="int64")
    s = np.minimum(spos_all, n_grid - 1)
    timeout_ns = full_grid_ns[s] + timeout_ns_delta
    tpos = np.searchsorted(full_grid_ns, timeout_ns, side="left")
    valid = (
        (spos_all < n_grid)
        & (tpos < n_grid)
        & (full_grid_ns[np.minimum(tpos, n_grid - 1)] == timeout_ns)
    )
    resolve_ns[valid] = timeout_ns[valid]
    return resolve_ns

def _iter_mhs_execution_windows(
    target_weights: pd.DataFrame,
    signal_available_at: pd.DatetimeIndex,
    root: str,
    timeframe: Literal["1m", "3m", "5m"],
    start: pd.Timestamp,
    end: pd.Timestamp,
    funding_by_symbol: dict[str, pd.Series],
    mark_mode: Literal["cache_required", "cache_required_stale_carry", "ohlcv_close_fallback"],
    spec: ExecutionSpec,
) -> Iterator[MhsExecutionWindow]:
    """Yield at-most-31-day execution windows with only the active roster read.

    Each window's minute grid starts at the previous window's last decision
    (the decision-time funding/MTM lead) and ends at the final order's strict
    timeout bar; the last window covers the full evaluation grid so a forced
    exit can always resolve. Only symbols with a non-zero target in the window
    or carried inventory from the previous window are read; the canonical
    column order is preserved on every window for artifact-shape equivalence.
    In ``cache_required`` mode each window's decision marks are asserted
    fail-closed before the window is yielded.
    """
    if len(target_weights) != len(signal_available_at):
        raise DataIntegrityError("signal_available_at must align with target_weights")
    if start >= end:
        raise DataIntegrityError("start must precede end")
    columns = tuple(target_weights.columns)
    freq = {"1m": "1min", "3m": "3min", "5m": "5min"}[timeframe]
    full_grid = pd.date_range(start, end, freq=freq, tz="UTC")
    full_grid_ns = np.asarray(full_grid, dtype="datetime64[ns]").astype("int64")
    n_grid = len(full_grid_ns)
    timeout_ns_delta = int(spec.passive_timeout_minutes) * 60_000_000_000
    signal_ns = np.asarray(signal_available_at, dtype="datetime64[ns]").astype("int64")
    spos_all = np.searchsorted(full_grid_ns, signal_ns, side="right")
    resolve_ns = _resolve_ns_vectorized(spos_all, full_grid_ns, n_grid, timeout_ns_delta)

    if target_weights.empty:
        empty_marks = (
            pd.DataFrame(index=full_grid) if mark_mode in ("cache_required", "cache_required_stale_carry") else None
        )
        yield ExecutionReplayWindow(
            window_start=start,
            window_end=end,
            columns=columns,
            symbols=(),
            minute_grid=full_grid,
            highs=pd.DataFrame(index=full_grid),
            lows=pd.DataFrame(index=full_grid),
            closes=pd.DataFrame(index=full_grid),
            marks=empty_marks,
            bar_funding=pd.DataFrame(index=full_grid),
            target_weights=target_weights,
            signal_available_at=signal_available_at,
        )
        return

    decision_times = pd.DatetimeIndex(target_weights.index)
    max_window = pd.Timedelta(days=31)
    bounds: list[tuple[int, int]] = []
    i0 = 0
    while i0 < len(decision_times):
        i1 = i0 + 1
        while i1 < len(decision_times) and decision_times[i1] - decision_times[i0] <= max_window:
            i1 += 1
        bounds.append((i0, i1))
        i0 = i1

    prev_active: set[str] = set()
    for wi, (i0, i1) in enumerate(bounds):
        w_weights = target_weights.iloc[i0:i1]
        w_signals = signal_available_at[i0:i1]
        is_last = wi == len(bounds) - 1
        grid_start = start if wi == 0 else decision_times[i0 - 1]
        if is_last:
            grid_end = end
        else:
            max_resolve = int(resolve_ns[i0:i1].max())
            if max_resolve < 0:
                max_resolve = int(
                    np.asarray(decision_times[i1 - 1] + pd.Timedelta(hours=2), dtype="datetime64[ns]").astype("int64")
                )
            grid_end = pd.Timestamp(max_resolve, unit="ns", tz="UTC")
        if grid_end > end:
            grid_end = end
        minute_grid = pd.date_range(grid_start, grid_end, freq=freq, tz="UTC")
        non_zero = w_weights.notna() & w_weights.ne(0.0)
        active = set(w_weights.columns[non_zero.any(axis=0)])
        roster_set = active | prev_active
        prev_active = active
        roster = [
            s for s in columns
            if s in roster_set and os.path.exists(os.path.join(root, timeframe, f"{s}.parquet"))
        ]

        symbol_frames = _load_window_minute_frames(
            root, roster, grid_start, grid_end, timeframe,
        )
        aligned = _build_window_frames(
            symbol_frames, roster, grid_start, grid_end, minute_grid, timeframe,
        )
        if aligned is None:
            highs = pd.DataFrame(index=minute_grid)
            lows = pd.DataFrame(index=minute_grid)
            closes = pd.DataFrame(index=minute_grid)
        else:
            highs, lows, closes = aligned
        for s in roster:
            if s not in highs.columns:
                highs[s] = np.nan
                lows[s] = np.nan
                closes[s] = np.nan
        highs = highs.reindex(columns=roster)
        lows = lows.reindex(columns=roster)
        closes = closes.reindex(columns=roster)

        minute_marks: pd.DataFrame | None = None
        if mark_mode in ("cache_required", "cache_required_stale_carry"):
            if roster:
                stale_hours = 24 if mark_mode == "cache_required_stale_carry" else 0
                minute_marks = _cached_mark_panel(
                    roster, "1h", minute_grid, stale_hours,
                )
                if mark_mode == "cache_required":
                    integrity._assert_cache_required_marks("window", w_weights[roster], w_signals, minute_marks)
            else:
                minute_marks = pd.DataFrame(index=minute_grid)

        minute_period = minute_grid[1] - minute_grid[0] if len(minute_grid) > 1 else pd.Timedelta(minutes=1)
        funding_window = {
            s: funding_by_symbol[s].loc[
                (funding_by_symbol[s].index >= grid_start)
                & (funding_by_symbol[s].index < grid_end + minute_period)
            ]
            for s in roster
            if s in funding_by_symbol
        }
        minute_funding = (
            bar_funding_panel(funding_window, minute_grid)
            .reindex(columns=roster)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .fillna(0.0)
        )

        yield ExecutionReplayWindow(
            window_start=grid_start,
            window_end=grid_end,
            columns=columns,
            symbols=tuple(roster),
            minute_grid=minute_grid,
            highs=highs,
            lows=lows,
            closes=closes,
            marks=minute_marks,
            bar_funding=minute_funding,
            target_weights=w_weights[roster],
            signal_available_at=w_signals,
        )



def _rescaled_windows(
    windows: Iterable[MhsExecutionWindow],
    scale: pd.Series | None,
) -> Iterator[MhsExecutionWindow]:
    """Yield the frozen windows with ``target_weights`` rescaled by ``scale``.

    ``scale=None`` yields the windows unchanged (zero-copy). Otherwise each
    window's target weights are multiplied by ``scale`` reindexed to the
    window's decision index (ffill + fillna(1.0)), reproducing the production
    ``target_replay.mul(scale.reindex(...).fillna(1.0), axis=0)`` slicing. The
    invariant is that the scaling must preserve each window's active-roster
    zero pattern; a scale that zeroes a held position fails closed with
    ``DataIntegrityError`` because the materialized window's roster would then
    diverge from a freshly regenerated window.
    """
    if scale is None:
        for w in windows:
            yield w
        return
    for w in windows:
        scaled = w.target_weights.mul(
            scale.reindex(w.target_weights.index, method="ffill").fillna(1.0),
            axis=0,
        )
        original_active = (
            w.target_weights.notna() & w.target_weights.ne(0.0)
        ).any(axis=0)
        scaled_active = (scaled.notna() & scaled.ne(0.0)).any(axis=0)
        if (
            list(scaled.columns) != list(w.target_weights.columns)
            or list(scaled.columns) != list(w.symbols)
            or not bool((original_active == scaled_active).all())
        ):
            raise DataIntegrityError(
                "pnl-vol-target scaling changed a window's active roster; "
                "the scale must preserve the zero pattern across replay passes"
            )
        yield dataclasses.replace(w, target_weights=scaled)


def _book_outcome(
    name: str,
    spec: BookSpec,
    n_symbols: int,
    step_grid: pd.DatetimeIndex,
    weights_step: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    phase: PhaseDiagnosticResult,
    root: str,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    event_window_bars: int,
    initial_equity: float,
    replay_weights_step: pd.DataFrame | None = None,
    telemetry: _StageRecorder | None = None,
) -> tuple[MhsBookReport, dict[int, dict[str, float]]]:
    import src.application.research.mhs.evaluation as ev
    weights_1h = weights_step.reindex(grid_1h).ffill().fillna(0.0)
    cost_grid = tuple(dict.fromkeys((0.0, 2.0, 4.0, 8.0, *required_cost_tiers())))
    reference_evidence = book_evidence(
        weights_1h, opens, bar_funding, cost_grid, _PERIODS_PER_YEAR_1H, event_window_bars,
    )
    prescreen = reference_evidence.prescreen
    tail = reference_evidence.tail
    # The pre-screen matrices are consumed by ``book_evidence`` above and hold
    # no references from those results.  Releasing them before the minute
    # replay keeps three full multi-year price/weight matrices out of the replay
    # baseline (spec §3.1, ``memory_opt``).
    del weights_1h, reference_evidence
    gc.collect()

    # RC-1: the same significance instruments, pointed at the book that
    # actually carries capital (roster + ensemble + tilt + regime scale). The
    # reference (``weights_step``) and executed (``replay_weights_step``) books
    # are now measured side by side under distinct labels.
    executed_prescreen: dict[float, CostResponsePoint] | None = None
    executed_tail: TailSensitivityResult | None = None
    executed_prescreen_net_t: float | None = None
    if replay_weights_step is not None:
        replay_weights_1h = replay_weights_step.reindex(grid_1h).ffill().fillna(0.0)
        executed_evidence = book_evidence(
            replay_weights_1h, opens, bar_funding, cost_grid, _PERIODS_PER_YEAR_1H, event_window_bars,
        )
        executed_prescreen = executed_evidence.prescreen
        executed_tail = executed_evidence.tail
        executed_prescreen_net_t = executed_evidence.prescreen[
            MEASURED_EXECUTION_COST_TIERS_BPS["base"]
        ].net_t
        del replay_weights_1h, executed_evidence
        gc.collect()

    target_weights = (replay_weights_step if replay_weights_step is not None else weights_step).reindex(step_grid)
    if request.rebalance_filter == "portfolio_trigger":
        target_weights = portfolio_rebalance_trigger(
            target_weights, REBALANCE_TRACKING_ERROR_THRESHOLD,
        )
    else:
        target_weights = _scaling._apply_rebalance_deadband(target_weights)
    blend_traces: dict[int, dict[str, float]] = {}
    if name == "blend":
        blend_traces = {
            idx: books._book_structure_trace(
                target_weights.loc[
                    (target_weights.index >= fold.validation_start)
                    & (target_weights.index <= fold.validation_end)
                ]
            )
            for idx, fold in enumerate(phase_1_anchored_purged_folds())
        }
    signal_available_at = step_grid + pd.Timedelta(hours=1)
    execution_grid = pd.date_range(
        start, end,
        freq={"1m": "1min", "3m": "3min", "5m": "5min"}[request.execution_timeframe],
        tz="UTC",
    )
    target_replay, signal_replay, censored = integrity._truncate_replayable_decisions(
        target_weights, signal_available_at, execution_grid, specs._resolved_base_execution_spec(request),
    )
    replay_symbols = list(target_replay.columns)

    # Fork workers get the SYSTEM reserve check (not the auto 85% budget, whose
    # fork-child RSS would double-count COW-shared parent pages).
    _window_rss_reserve = _resolve_ram_budget(None, request.ram_guard)[1]

    def _windows() -> Iterator[MhsExecutionWindow]:
        return ev._iter_mhs_execution_windows(
            target_replay, signal_replay, root, request.execution_timeframe,
            start, end, funding_by_symbol, request.mark_mode, specs._resolved_base_execution_spec(request),
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

    touch = None
    touch_naive_sharpe = None
    ladder = None
    ladder_naive_sharpe = None
    peg_chase = None
    peg_chase_naive_sharpe = None
    peg_chase_fill_rate = None
    peg_chase_maker_share = None
    patient_reference = None
    patient_reference_naive_sharpe = None
    pre_vol_target_reference = None
    pre_vol_target_reference_naive_sharpe = None
    # Two-pass 경로에서만 채워진다(coupled 스트리밍은 per-prefix 재계산이라
    # 단일 Series가 없다). constant_risk는 항상 two-pass다.
    pnl_vol_target_scale: pd.Series | None = None
    try:
        # One cost model for EVERY bound in the batch (primary, stress, strict,
        # and each diagnostic): the bounds must compete on identical taker
        # crossing costs, never a single bound overridden in isolation.
        replay_base_spec = (
            dataclass_replace(specs._resolved_base_execution_spec(request), liquidity_cost_model="corwin_schultz")
            if request.liquidity_cost_model == "corwin_schultz"
            else specs._resolved_base_execution_spec(request)
        )
        batch_bounds: list[
            tuple[
                Literal[
                    "OHLCV_STRICT_PROXY",
                    "OHLCV_TOUCH_PROXY",
                    "OHLCV_IMMEDIATE_TAKER",
                    "OHLCV_LADDERED_PROXY",
                    "OHLCV_PEG_CHASE_PROXY",
                ],
                ExecutionSpec,
            ]
        ] = [
            ("OHLCV_IMMEDIATE_TAKER", replay_base_spec),
            (
                "OHLCV_IMMEDIATE_TAKER",
                dataclass_replace(
                    specs._stress_cost_execution_spec(replay_base_spec),
                    liquidity_cost_model=replay_base_spec.liquidity_cost_model,
                ),
            ),
            ("OHLCV_STRICT_PROXY", replay_base_spec),
        ]
        # Explicit result indices for the optional diagnostic bounds: a negative
        # index silently misbinds once another bound is appended.
        optional_bound_indices: dict[str, int] = {}
        if request.touch_diagnostic:
            batch_bounds.append(("OHLCV_TOUCH_PROXY", replay_base_spec))
            optional_bound_indices["touch"] = len(batch_bounds) - 1
        if request.ladder_diagnostic:
            integrity._validate_ladder_schedule_contract()
            batch_bounds.append(("OHLCV_LADDERED_PROXY", replay_base_spec))
            optional_bound_indices["ladder"] = len(batch_bounds) - 1
        if request.peg_chase_diagnostic:
            batch_bounds.append(
                ("OHLCV_PEG_CHASE_PROXY", dataclass_replace(replay_base_spec, decision_anchor="submit_bar"))
            )
            optional_bound_indices["peg_chase"] = len(batch_bounds) - 1
        isolated_indices = frozenset(
            i for i, (bound, _spec) in enumerate(batch_bounds)
            if bound in specs.REFERENCE_ONLY_EXECUTION_BOUNDS
        )
        # D1 (gated): when the resolved exposure scale is strictly causal and
        # prefix-deterministic, stream the windows ONCE -- the reference
        # consumes each loaded window, the prefix scale is recomputed, and the
        # scaled bounds consume the same already-loaded window. Any fail-closed
        # condition (incomplete preceding day, roster drift) falls back to the
        # exact two-pass path below.
        coupled: tuple[StrategyExecutionReplayResult, Any] | None = None
        if request.pnl_vol_target and _scaling.is_streaming_scale_mode(request):
            try:
                coupled = replay_execution_windows_coupled(
                    _window_telemetry(_windows(), "execution_window"),
                    initial_equity,
                    ("OHLCV_IMMEDIATE_TAKER", specs._resolved_base_execution_spec(request)),
                    batch_bounds,
                    lambda daily_returns: _scaling._replay_exposure_scale(daily_returns, request),
                    retain_event_snapshots=False,
                    min_equity_fraction=REFERENCE_PASS_EQUITY_FLOOR,
                    isolated_bound_indices=isolated_indices,
                )
            except DataIntegrityError:
                coupled = None
        if coupled is not None:
            pre_vol_target_reference, batch = coupled
            pre_vol_target_reference_naive_sharpe = _statistics._naive_sharpe(
                pre_vol_target_reference.ledger
            )
        else:
            # Exact two-pass path: Phase A (reference, unscaled), then the
            # P&L-vol-target scale, then Phase B (rescaled batch).
            primary_two_pass = replay_execution_windows(
                _window_telemetry(_windows(), "execution_window"),
                initial_equity, "OHLCV_IMMEDIATE_TAKER", specs._resolved_base_execution_spec(request),
                retain_event_snapshots=False,
                min_equity_fraction=REFERENCE_PASS_EQUITY_FLOOR,
            )
            reference_daily_returns = primary_two_pass.ledger.equity.resample("1D").last().pct_change()
            if name == "blend" and request.exposure_scale_two_sided:
                # The audit verifies the registered ceiling against the ONE
                # book that actually deploys capital (I3: once per run). The
                # fast/slow standalone reference books stay diagnostic-only
                # under committee_capital=True and can be genuinely losing in
                # a given train window -- their bootstrap frontier is
                # legitimately infeasible and must never crash the run.
                _scaling._assert_envelope_leverage_ceiling_verified(
                    _research_go._resolved_growth_envelope(request),
                    reference_daily_returns,
                )
            pnl_vol_target_scale = _scaling._replay_exposure_scale(reference_daily_returns, request)
            replay_scale = pnl_vol_target_scale if request.pnl_vol_target else None
            if name == "blend":
                # I5: the parity guard must see the SAME deployed-gross scale
                # this replay actually applies -- an unscaled run (pnl_vol_target
                # off) deploys at scale 1.0, never at the diagnostic-only
                # pnl_vol_target_scale value that was computed but not applied.
                _deployed_scale = (
                    replay_scale if replay_scale is not None
                    else pd.Series(1.0, index=pnl_vol_target_scale.index)
                )
                # I4 observability: resolve the cap once per run -- it is a
                # data-independent policy constant, never per-fold state.
                _resolved_cap = _scaling.resolved_exposure_cap(request)
                for _idx, _fold in enumerate(phase_1_anchored_purged_folds()):
                    _fold_scale = _deployed_scale.loc[
                        (_deployed_scale.index >= _fold.validation_start)
                        & (_deployed_scale.index <= _fold.validation_end)
                    ].dropna()
                    if _idx in blend_traces and len(_fold_scale) > 0:
                        blend_traces[_idx]["exposure_scale_mean"] = float(_fold_scale.mean())
                        blend_traces[_idx]["exposure_scale_cap_binding_fraction"] = float(
                            (_fold_scale >= _resolved_cap - 1e-12).mean(),
                        )
            pre_vol_target_reference = primary_two_pass
            pre_vol_target_reference_naive_sharpe = _statistics._naive_sharpe(primary_two_pass.ledger)
            batch = ev.replay_execution_window_batch_isolated(
                _window_telemetry(
                    _rescaled_windows(_windows(), replay_scale),
                    "execution_window_rescaled",
                ),
                initial_equity, batch_bounds,
                retain_event_snapshots=False,
                min_equity_fraction=REFERENCE_PASS_EQUITY_FLOOR,
                isolated_bound_indices=isolated_indices,
            )
        primary = batch.results[0]  # non-isolated index cannot be None
        stress = batch.results[1]
        patient_reference = batch.results[2]
        assert primary is not None
        assert stress is not None
        patient_reference_naive_sharpe = (
            _statistics._naive_sharpe(patient_reference.ledger) if patient_reference is not None else None
        )
        if request.touch_diagnostic and "touch" in optional_bound_indices:
            touch = batch.results[optional_bound_indices["touch"]]
            touch_naive_sharpe = _statistics._naive_sharpe(touch.ledger) if touch is not None else None
        if request.ladder_diagnostic and "ladder" in optional_bound_indices:
            ladder = batch.results[optional_bound_indices["ladder"]]
            ladder_naive_sharpe = _statistics._naive_sharpe(ladder.ledger) if ladder is not None else None
        if request.peg_chase_diagnostic and "peg_chase" in optional_bound_indices:
            peg_chase = batch.results[optional_bound_indices["peg_chase"]]
            if peg_chase is not None:
                peg_chase_naive_sharpe = _statistics._naive_sharpe(peg_chase.ledger)
                peg_chase_fill_rate = specs._peg_chase_fill_rate(peg_chase)
                peg_chase_maker_share = specs._peg_chase_maker_share(peg_chase)
        reference_bound_failures = tuple(
            MhsBookFailure(
                stage=f"replay_{name}_{f.execution_bound}",
                error_class=f.error_class,
                reason=integrity._classify_execution_failure(DataIntegrityError(f.message)),
                message=f.message,
            )
            for f in batch.isolated_failures
        )
        if telemetry is not None:
            telemetry.record(
                f"replay_{name}_strict",
                n_symbols=len(replay_symbols),
                fill_count=len(primary.simulated_fills),
            )
            telemetry.record(
                f"replay_{name}_stress",
                n_symbols=len(replay_symbols),
                fill_count=len(stress.simulated_fills),
            )
        if request.mark_mode == "cache_required":
            integrity._assert_cache_required_ledger_valid(name, primary)
    except DataIntegrityError as exc:
        failure = MhsBookFailure(
            stage=f"replay_{name}",
            error_class=type(exc).__name__,
            reason=integrity._classify_execution_failure(exc),
            message=str(exc),
        )
        if telemetry is not None:
            telemetry.record(
                f"replay_{name}_failed",
                n_symbols=len(replay_symbols),
                fill_count=0,
            )
        return MhsBookReport(
            name=name,
            band=spec.band.name,
            horizon_hours=spec.horizon_hours,
            step_hours=spec.step_hours,
            tranche_count=spec.tranche_count(),
            n_symbols=n_symbols,
            phase=phase,
            prescreen=prescreen,
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
            terminal_censored_decisions=censored,
            failure=failure,
            touch=touch,
            touch_naive_sharpe=touch_naive_sharpe,
            ladder=ladder,
            ladder_naive_sharpe=ladder_naive_sharpe,
            peg_chase=peg_chase,
            peg_chase_naive_sharpe=peg_chase_naive_sharpe,
            peg_chase_fill_rate=peg_chase_fill_rate,
            peg_chase_maker_share=peg_chase_maker_share,
            patient_reference=patient_reference,
            patient_reference_naive_sharpe=patient_reference_naive_sharpe,
            pre_vol_target_reference=pre_vol_target_reference,
            pre_vol_target_reference_naive_sharpe=pre_vol_target_reference_naive_sharpe,
            executed_prescreen=executed_prescreen,
            executed_tail=executed_tail,
            executed_prescreen_net_t=executed_prescreen_net_t,
            target_weights=target_weights if name == "blend" else None,
        ), blend_traces
    equity_1h, net_returns_1h, turnover_1h = _statistics._hourly_ledger_series(
        primary.ledger.equity, primary.ledger.fill_turnover,
    )
    return MhsBookReport(
        name=name,
        band=spec.band.name,
        horizon_hours=spec.horizon_hours,
        step_hours=spec.step_hours,
        tranche_count=spec.tranche_count(),
        n_symbols=n_symbols,
        phase=phase,
        prescreen=prescreen,
        tail=tail,
        primary=primary,
        stress=stress,
        primary_autocorr_sharpe=_statistics._daily_autocorr_sharpe(primary.ledger),
        primary_naive_sharpe=_statistics._naive_sharpe(primary.ledger),
        primary_net_ann=_statistics._mean_ann(net_returns_1h, _PERIODS_PER_YEAR_1H),
        primary_geometric_cagr=_statistics._geometric_cagr(equity_1h),
        primary_max_drawdown=_statistics._mdd(primary.ledger.equity),
        primary_annualized_turnover=_statistics._mean_ann(turnover_1h, _PERIODS_PER_YEAR_1H),
        stress_naive_sharpe=_statistics._naive_sharpe(stress.ledger),
        terminal_censored_decisions=censored,
        touch=touch,
        touch_naive_sharpe=touch_naive_sharpe,
        ladder=ladder,
        ladder_naive_sharpe=ladder_naive_sharpe,
        peg_chase=peg_chase,
        peg_chase_naive_sharpe=peg_chase_naive_sharpe,
        peg_chase_fill_rate=peg_chase_fill_rate,
        peg_chase_maker_share=peg_chase_maker_share,
        patient_reference=patient_reference,
        patient_reference_naive_sharpe=patient_reference_naive_sharpe,
        pre_vol_target_reference=pre_vol_target_reference,
        pre_vol_target_reference_naive_sharpe=pre_vol_target_reference_naive_sharpe,
        executed_prescreen=executed_prescreen,
        executed_tail=executed_tail,
        executed_prescreen_net_t=executed_prescreen_net_t,
        reference_bound_failures=reference_bound_failures,
        primary_realized_shortfall_bps=primary.all_intent_shortfall_bps,
        primary_notional_weighted_shortfall_bps=primary.notional_weighted_shortfall_bps,
        stress_realized_shortfall_bps=stress.all_intent_shortfall_bps,
        stress_notional_weighted_shortfall_bps=stress.notional_weighted_shortfall_bps,
        primary_fill_count=primary.fill_count,
        primary_unfilled_count=primary.unfilled_count,
        primary_forced_exit_notional=primary.forced_exit_notional,
        # I-SCALE-IS-DEPLOYED-OVERLAY: fold가 재적합하지 않고 읽어가는
        # blend의 배치 확정 스케일. name=="blend"일 때만 노출한다.
        exposure_scale=pnl_vol_target_scale if name == "blend" else None,
        target_weights=target_weights if name == "blend" else None,
    ), blend_traces


def _book_outcome_worker(
    name: str,
    token: str,
    n_symbols: int,
    root: str,
    request: MhsDiagnosticRequest,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_equity: float,
) -> tuple[MhsBookReport, tuple[MhsResourceMeasurement, ...], dict[int, dict[str, float]]]:
    """Run one ``_book_outcome`` in a fork child with its own telemetry recorder.

    The typed failure conversion inside ``_book_outcome`` is preserved; a book
    that fails its replay is still returned (with ``failure`` set) so the other
    two books' results are never lost.  The per-window telemetry and the blend
    book's post-deadband structure trace are returned so the parent can merge
    them in declared order.

    The book's spec/grids/weights/phase and the shared 1h panels and funding
    series are resolved from the fork-shared payload by ``token`` (registered via
    ``fork_shared_payload`` in the parent before the pool forks) so no
    ``pd.DataFrame``/``pd.Series`` crosses the ``submit`` pickle boundary.
    """
    shared = resolve_fork_shared(token)
    spec, step_grid, weights_step, phase, event_window_bars, replay_weights_step = shared["books"][name]
    recorder = _StageRecorder(log_run=False)
    report, blend_traces = _book_outcome(
        name, spec, n_symbols, step_grid, weights_step, shared["grid_1h"],
        shared["opens"], shared["bar_funding"], phase, root, request,
        shared["funding_by_symbol"], start, end, event_window_bars, initial_equity,
        replay_weights_step, telemetry=recorder,
    )
    return report, recorder.records, blend_traces