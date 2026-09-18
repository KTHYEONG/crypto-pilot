# mypy: ignore-errors
# ruff: noqa: F401, F821, I001, E402
from __future__ import annotations  # mypy: ignore-errors

import dataclasses
import gc
import json
import os
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import replace as dataclass_replace
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from src.common.paths import BASE_DIR
from src.mhs import research_go as _research_go
from src.mhs import scaling as _scaling
from src.mhs import statistics as _statistics
from src.mhs.contracts import MhsBookFailure, MhsBookReport, MhsDiagnosticRequest
from src.mhs.marks import _build_window_frames, _cached_mark_panel, _load_window_minute_frames
from src.mhs.resources import MhsExecutionAllocation, assert_mhs_allocation_budget, plan_mhs_execution_bars, _assert_execution_rss_budget, _resolve_ram_budget, _StageRecorder
from src.common.errors import DataIntegrityError
from src.mhs.books import portfolio_rebalance_trigger
from src.mhs.evidence import CostResponsePoint, PhaseDiagnosticResult, TailSensitivityResult, book_evidence, required_cost_tiers, resolved_anchored_folds
from src.mhs.execution import ExecutionReplayWindow, StrategyExecutionReplayResult, align_funding_with_knowledge, bar_funding_panel, replay_execution_window_batch_isolated, replay_execution_windows, replay_execution_windows_coupled
from src.mhs.parallel import resolve_fork_shared
from src.mhs.params import MEASURED_EXECUTION_COST_TIERS_BPS, REBALANCE_TRACKING_ERROR_THRESHOLD, REFERENCE_PASS_EQUITY_FLOOR
from src.mhs.params import PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H
from src.mhs.types import BookSpec, ExecutionSpec

from . import books, integrity, specs


def _window_spill_root() -> str:
    """Disk-backed spill root for window IPC scratch files."""
    root = os.environ.get("MHS_SPILL_DIR") or str(BASE_DIR / "tmp" / "mhs_spill")
    os.makedirs(root, exist_ok=True)
    return root


from src.mhs.execution.window_stream import MhsExecutionWindow as MhsExecutionWindow  # noqa: E402
from src.mhs.execution.window_stream import _estimate_mhs_execution_allocation as _estimate_mhs_execution_allocation  # noqa: E402
from src.mhs.execution.window_stream import _iter_mhs_execution_windows as _iter_mhs_execution_windows  # noqa: E402
from src.mhs.execution.window_stream import _materialize_execution_piece as _materialize_execution_piece  # noqa: E402
from src.mhs.execution.window_stream import _minimum_mhs_execution_bars as _minimum_mhs_execution_bars  # noqa: E402
from src.mhs.execution.window_stream import _resolve_ns_vectorized as _resolve_ns_vectorized  # noqa: E402



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


def _spill_window_to_ipc(window: ExecutionReplayWindow, target_path: str) -> None:
    """Spill one execution window to an Arrow IPC scratch file (fail-closed).

    Every frame's float64 values travel as an Arrow IPC stream inside a zip
    container with a JSON header carrying the window metadata and per-frame
    indexes; a load of the file reproduces the window bit-identically.
    """
    try:
        minute_ns = np.asarray(window.minute_grid, dtype="datetime64[ns]").astype("int64")
        signal_ns = np.asarray(window.signal_available_at, dtype="datetime64[ns]").astype("int64")
        frames: dict[str, pd.DataFrame | None] = {
            "highs": window.highs,
            "lows": window.lows,
            "closes": window.closes,
            "marks": window.marks,
            "bar_funding": window.bar_funding,
            "target_weights": window.target_weights,
            "quote_volumes": window.quote_volumes,
            "funding_known": window.funding_known.astype("float64") if window.funding_known is not None else None,
        }
        meta_frames: dict[str, Any] = {}
        buffers: dict[str, bytes] = {}
        for name, frame in frames.items():
            if frame is None:
                meta_frames[name] = None
                continue
            idx_ns = np.asarray(frame.index, dtype="datetime64[ns]").astype("int64").tolist()
            meta_frames[name] = {"columns": list(frame.columns), "index_ns": idx_ns}
            arrays = [
                pa.array(frame[c].to_numpy(dtype="float64", copy=False), type=pa.float64())
                for c in frame.columns
            ]
            schema = pa.schema([pa.field(c, pa.float64()) for c in frame.columns])
            batch = pa.record_batch(arrays, schema=schema) if arrays else None
            sink = pa.BufferOutputStream()
            writer = pa_ipc.new_stream(sink, schema)
            if batch is not None:
                writer.write_batch(batch)
            writer.close()
            buffers[name] = bytes(sink.getvalue())
        meta = {
            "window_start_ns": int(window.window_start.value),
            "window_end_ns": int(window.window_end.value),
            "columns": list(window.columns),
            "symbols": list(window.symbols),
            "minute_grid_ns": minute_ns.tolist(),
            "signal_ns": signal_ns.tolist(),
            "bar_available_ns": np.asarray(window.bar_available_at, dtype="datetime64[ns]").astype("int64").tolist() if window.bar_available_at is not None else None,
            "logical_partition": list(window.logical_partition) if window.logical_partition is not None else None,
            "frames": meta_frames,
            "funding_coverage_gaps": [
                {"symbol": g.symbol, "start_ns": int(g.start.value), "end_ns": int(g.end.value), "reason": g.reason}
                for g in window.funding_coverage_gaps
            ],
        }
        with zipfile.ZipFile(target_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("meta.json", json.dumps(meta))
            for name, buf in buffers.items():
                zf.writestr(f"{name}.arrow", buf)
    except Exception as exc:
        raise DataIntegrityError(f"window IPC spill failed for {target_path}: {exc}") from exc


def _load_window_from_ipc(target_path: str) -> ExecutionReplayWindow:
    """Load one spilled execution window bit-identically (fail-closed)."""
    try:
        with zipfile.ZipFile(target_path, "r") as zf:
            meta = json.loads(zf.read("meta.json"))
            buffers = {
                name: zf.read(f"{name}.arrow")
                for name, spec in meta["frames"].items()
                if spec is not None
            }
        minute_grid = pd.DatetimeIndex(
            pd.to_datetime(np.asarray(meta["minute_grid_ns"], dtype="int64"), unit="ns", utc=True)
        )
        signal_available_at = pd.DatetimeIndex(
            pd.to_datetime(np.asarray(meta["signal_ns"], dtype="int64"), unit="ns", utc=True)
        )
        frames: dict[str, pd.DataFrame | None] = {}
        for name, spec in meta["frames"].items():
            if spec is None:
                frames[name] = None
                continue
            reader = pa_ipc.open_stream(pa.py_buffer(buffers[name]))
            table = reader.read_all()
            idx = pd.DatetimeIndex(
                pd.to_datetime(np.asarray(spec["index_ns"], dtype="int64"), unit="ns", utc=True)
            )
            data = {
                c: np.asarray(table.column(c).to_numpy(zero_copy_only=False), dtype="float64")
                for c in spec["columns"]
            }
            frames[name] = pd.DataFrame(data, index=idx, columns=spec["columns"])
            frames[name] = frames[name].astype("float64")
        known_frame = frames["funding_known"].astype(bool) if frames["funding_known"] is not None else None
        available = pd.DatetimeIndex(pd.to_datetime(np.asarray(meta.get("bar_available_ns"), dtype="int64"), unit="ns", utc=True)) if meta.get("bar_available_ns") is not None else None
        logical_partition = meta.get("logical_partition")
        from src.mhs.execution.contracts import FundingCoverageGap as _FundingCoverageGap

        coverage = tuple(
            _FundingCoverageGap(
                symbol=str(entry["symbol"]),
                start=pd.Timestamp(int(entry["start_ns"]), unit="ns", tz="UTC"),
                end=pd.Timestamp(int(entry["end_ns"]), unit="ns", tz="UTC"),
                reason=str(entry["reason"]),
            )
            for entry in meta.get("funding_coverage_gaps", [])
        )
        return ExecutionReplayWindow(
            window_start=pd.Timestamp(meta["window_start_ns"], unit="ns", tz="UTC"),
            window_end=pd.Timestamp(meta["window_end_ns"], unit="ns", tz="UTC"),
            columns=tuple(meta["columns"]),
            symbols=tuple(meta["symbols"]),
            minute_grid=minute_grid,
            highs=frames["highs"],
            lows=frames["lows"],
            closes=frames["closes"],
            marks=frames["marks"],
            bar_funding=frames["bar_funding"],
            target_weights=frames["target_weights"],
            signal_available_at=signal_available_at,
            quote_volumes=frames["quote_volumes"],
            funding_known=known_frame,
            bar_available_at=available,
            logical_partition=tuple(logical_partition) if logical_partition is not None else None,
            funding_coverage_gaps=coverage,
        )
    except Exception as exc:
        raise DataIntegrityError(f"window IPC load failed for {target_path}: {exc}") from exc


def _spill_and_stream_windows(
    windows: Iterable[ExecutionReplayWindow], spill_dir: str
) -> Iterator[ExecutionReplayWindow]:
    """Pass-1 stream: yield each window while spilling it to ``spill_dir``.

    Single-window spill buffer over the 31-day window stream; the caller owns
    scratch-directory lifecycle (tempfile.TemporaryDirectory + try/finally).
    """
    os.makedirs(spill_dir, exist_ok=True)
    for idx, window in enumerate(windows):
        _spill_window_to_ipc(window, os.path.join(spill_dir, f"window_{idx:05d}.arrow"))
        yield window


def _iter_spilled_windows(spill_dir: str) -> Iterator[ExecutionReplayWindow]:
    """Pass-2 stream: replay spilled windows from disk in filename order."""
    try:
        names = sorted(
            n for n in os.listdir(spill_dir) if n.startswith("window_") and n.endswith(".arrow")
        )
    except Exception as exc:
        raise DataIntegrityError(f"window IPC spill directory unreadable: {spill_dir}: {exc}") from exc
    for name in names:
        yield _load_window_from_ipc(os.path.join(spill_dir, name))


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
            for idx, fold in enumerate(resolved_anchored_folds(request))
        }
    signal_available_at = step_grid + pd.Timedelta(hours=1)
    execution_grid = pd.date_range(
        start, end,
        freq="3min",
        tz="UTC",
    )
    target_replay, signal_replay, censored = integrity._truncate_replayable_decisions(
        target_weights, signal_available_at, execution_grid, specs._resolved_base_execution_spec(request),
    )
    replay_symbols = list(target_replay.columns)

    # Fork workers get the SYSTEM reserve check (not the auto 85% budget, whose
    # fork-child RSS would double-count COW-shared parent pages).
    _window_budget, _window_rss_reserve = _resolve_ram_budget(request.max_rss_bytes, request.ram_guard)
    execution_bound_count = 3 + int(bool(request.touch_diagnostic)) + int(bool(request.ladder_diagnostic)) + int(bool(request.peg_chase_diagnostic))

    def _windows() -> Iterator[MhsExecutionWindow]:
        return _iter_mhs_execution_windows(
            target_replay, signal_replay, root, request.execution_timeframe,
            start, end, funding_by_symbol, request.mark_mode, specs._resolved_base_execution_spec(request),
            execution_bound_count=execution_bound_count,
            budget_bytes=_window_budget, reserve_bytes=_window_rss_reserve)

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
            # Exact two-pass path: Phase A (reference, unscaled) spills each
            # window to Arrow IPC scratch while streaming, then the
            # P&L-vol-target scale, then Phase B (rescaled batch) streams the
            # identical windows back from disk (0MB RAM amplification).
            # 북당 ~4.3GB IPC 스필 x 3북 동시 — RAM 기반 /tmp tmpfs(8.3GB)에서 ENOSPC·메모리 압박.
            spill_temp = tempfile.TemporaryDirectory(prefix="mhs_windows_", dir=_window_spill_root())
            primary_two_pass = replay_execution_windows(
                _window_telemetry(_spill_and_stream_windows(_windows(), spill_temp.name), "execution_window"),
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
                for _idx, _fold in enumerate(resolved_anchored_folds(request)):
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
            try:
                batch = replay_execution_window_batch_isolated(
                    _window_telemetry(
                        _rescaled_windows(_iter_spilled_windows(spill_temp.name), replay_scale),
                        "execution_window_rescaled",
                    ),
                    initial_equity, batch_bounds,
                    retain_event_snapshots=False,
                    min_equity_fraction=REFERENCE_PASS_EQUITY_FLOOR,
                    isolated_bound_indices=isolated_indices,
                )
            finally:
                spill_temp.cleanup()
                gc.collect()
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
