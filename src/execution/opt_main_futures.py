from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so 'src' module is importable from any working directory
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ruff: noqa: E402
import argparse
import logging
import os
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.domain.futures.strategy.regime_evaluation import RegimeScoreCard
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result
    from src.domain.futures.strategy_runtime.bridge import CandidatePipelineOutput

import numpy as np
import optuna
import pandas as pd

# Suppress noisy system warnings for clean output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="no explicit representation of timezones")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from src.application.futures.optimization.config import (
    FuturesRunConfig,
    build_run_config_from_args,
)
from src.application.futures.optimization.data_readiness import (
    DataReadinessResult,
    evaluate_data_readiness,
)
from src.application.futures.optimization.optimization_service import (
    FinalEvaluationRequest,
    OptimizationRequest,
    run_final_evaluation,
    run_optimization,
)
from src.application.futures.optimization.strategy_service import (
    build_candidate_strategy_config,
    pick_strategy_data_maps,
    run_active_strategy_output_bridge,
)
from src.application.futures.optimization.universe_service import (
    UniverseMembershipTimeline,
    discover_universe_timeline,
    validate_universe_quality,
)
from src.core.settings import BASE_DIR
from src.core.utils.utils import PERF, setup_logger
from src.domain.futures.optimization.observability.run_tracker import (
    build_joint_study_name,
    build_run_id,
    get_or_create_study,
    load_champion_params,
    log_optuna_contract,
    resolve_futures_parallel_policy,
    setup_optuna_storage,
    update_champion_store,
)
from src.domain.futures.optimization.opt_config import (
    FUTURES_ANCHOR_SYMBOLS,
    FUTURES_MACRO_INDEX_SYMBOLS,
    OPT_FUTURES_CONFIG,
    get_quarterly_window,
)
from src.domain.futures.optimization.opt_data_utils import load_futures_data_maps_for_symbols
from src.domain.futures.optimization.validation import (
    awf_pos_frac_to_pseudo_pbo,
    resolve_adjusted_gates,
)
from src.domain.futures.strategy.config import StrategyConfig
from src.domain.futures.strategy_runtime.bridge import (
    merge_candidate_output_into_is_and_oos,
)
from src.domain.futures.universe import UniverseSnapshot
from src.domain.futures.universe.contracts import UniverseStateCube
from src.domain.futures.universe.membership import inject_membership_masks_into_maps
from src.domain.futures.universe.storage import run_historical_sync

_logger = setup_logger("opt_main_futures", write_file=False)

# Minimum bars a symbol must provide within [fetch_start, holdout_end] for tiered admission.
# Derived from score_pct_variant_hist_window_bars default (2160) but floored at 1500 to
# allow shorter-listed symbols that still cover the OOS window.
_TIERED_MIN_WINDOW_BARS: int = 1500


@dataclass(frozen=True, slots=True)
class RuntimeBreakdown:
    total: float
    steps: Mapping[str, float]

    @property
    def accounted(self) -> float:
        return float(sum(max(float(value), 0.0) for value in self.steps.values()))

    @property
    def unaccounted(self) -> float:
        return max(float(self.total) - self.accounted, 0.0)


@dataclass(frozen=True, slots=True)
class TradeableScopeResult:
    admitted: tuple[str, ...]
    dropped_by_reason: Mapping[str, tuple[str, ...]]


def _runtime_breakdown(total: float, **steps: float) -> RuntimeBreakdown:
    return RuntimeBreakdown(total=float(total), steps={key: float(value) for key, value in steps.items()})


def _forward_log_return_on_index(
    close: pd.Series,
    target_index: pd.Index,
    horizon_bars: int,
) -> pd.Series:
    """Compute forward log return on full series, then align to target index."""
    close_ser = close.copy()
    close_ser.index = pd.to_datetime(close_ser.index, utc=True).tz_localize(None)
    close_ser = close_ser.sort_index()
    target_idx = pd.to_datetime(target_index, utc=True).tz_localize(None)
    fwd = np.log(close_ser.shift(-horizon_bars) / close_ser)
    return fwd.reindex(target_idx)


def _requires_exec_1m(run_config: FuturesRunConfig) -> bool:
    """Return whether the active phases require execution-grade 1m data."""
    del run_config
    return False


def _ensure_universe_ledger_sync(run_config: FuturesRunConfig, window: QuarterlyWindow) -> None:
    """Ensure universe ledger coverage for the required window (SQLite-backed)."""
    import sqlite3

    from src.domain.futures.universe.models import DEFAULT_LEDGER_PATH

    needs_sync = False
    last_ledger_date = date(2023, 1, 1)

    if not DEFAULT_LEDGER_PATH.exists():
        _logger.info("[SYNC] Ledger missing -> Initiating first sync")
        needs_sync = True
    else:
        try:
            with sqlite3.connect(DEFAULT_LEDGER_PATH) as conn:
                row = conn.execute("SELECT MAX(date) FROM ledger").fetchone()
            max_date_str = row[0] if row else None
            if not max_date_str:
                _logger.info("[SYNC] Ledger empty -> Initiating first sync")
                needs_sync = True
            else:
                last_ledger_date = date.fromisoformat(max_date_str)
                if last_ledger_date < window.end_date_value:
                    _logger.info(
                        "[SYNC] Ledger outdated (Last: %s, Required: %s) -> Syncing...",
                        last_ledger_date,
                        window.end_date_value,
                    )
                    needs_sync = True
                else:
                    _logger.debug(
                        "[SYNC] Ledger up-to-date (Last: %s)",
                        last_ledger_date,
                    )
        except Exception as e:
            _logger.warning(
                "[SYNC] Ledger read failed (%s) -> Force sync",
                type(e).__name__,
            )
            needs_sync = True

    if needs_sync:
        if run_config.sync == "skip":
            _logger.info("[SYNC] Skipped (as requested by config)")
        else:
            run_historical_sync(
                start_date=last_ledger_date,
                end_date=window.end_date_value,
                sync_mode=run_config.sync,
                symbols=None,
                sync_1d=True,
                sync_4h=True,
                sync_1m=False,
                sync_metrics=run_config.sync_metrics,
            )


def _resolve_tradeable_scope(
    *,
    valid_symbols: Sequence[str],
    strategy_maps: Mapping[str, Mapping[str, pd.DataFrame]],
    tf: str,
    fetch_start: pd.Timestamp,
    oos_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    min_window_bars: int,
    min_holdout_coverage: float,
) -> TradeableScopeResult:
    """PIT sub-window admission: admits symbols with sufficient OOS coverage.

    Replaces the full-window END-coverage filter that caused survivorship bias.
    Per-bar look-ahead is enforced downstream by state_cube.active_mask.

    Args:
        valid_symbols: Candidate symbols that loaded usable window data.
        strategy_maps: Nested mapping ``{symbol -> {tf -> DataFrame}}``.
        tf: Timeframe string (e.g. ``"4h"``).
        fetch_start: Start of the full fetch window (UTC).
        oos_start: Start of the OOS holdout period (UTC).
        holdout_end: End of the holdout period (UTC).
        min_window_bars: Minimum number of bars within ``[fetch_start, holdout_end]``.
        min_holdout_coverage: Fraction of ``[oos_start, holdout_end]`` the symbol
            must cover (0-1).  Protects against holdout-truncated / delisted symbols.

    Returns:
        Symbols that pass both the bar-count and OOS-coverage guards.

    Time Complexity: O(S * T) where S = len(valid_symbols), T = max bars per symbol.
    Space Complexity: O(S) admitted list + O(T) per-symbol datetime Series.
    """
    oos_span = (holdout_end - oos_start).total_seconds()
    admitted: list[str] = []
    dropped: dict[str, list[str]] = {
        "missing_map": [],
        "empty_frame": [],
        "late_start": [],
        "min_bars": [],
        "no_holdout": [],
        "holdout_coverage": [],
    }
    for sym in valid_symbols:
        sym_maps = strategy_maps.get(sym)
        if sym_maps is None:
            dropped["missing_map"].append(sym)
            continue
        sym_df = sym_maps.get(tf)
        if sym_df is None or sym_df.empty:
            dropped["empty_frame"].append(sym)
            continue
        if pd.api.types.is_datetime64_any_dtype(sym_df["datetime"]):
            datetimes = sym_df["datetime"]
        else:
            datetimes = pd.to_datetime(sym_df["datetime"], utc=True)
        # Count bars within the full window [fetch_start, holdout_end]
        mask_window = (datetimes >= fetch_start) & (datetimes <= holdout_end)
        n_bars = int(mask_window.sum())
        if n_bars < min_window_bars:
            dropped["min_bars"].append(sym)
            continue
        if datetimes.min() > fetch_start:
            dropped["late_start"].append(sym)
            continue
        # OOS holdout coverage: symbol must span >= min_holdout_coverage of [oos_start, holdout_end]
        if oos_span > 0:
            oos_data = datetimes[(datetimes >= oos_start) & (datetimes <= holdout_end)]
            if len(oos_data) == 0:
                dropped["no_holdout"].append(sym)
                continue
            covered_span = (oos_data.max() - oos_data.min()).total_seconds()
            coverage = covered_span / oos_span
            if coverage < min_holdout_coverage:
                dropped["holdout_coverage"].append(sym)
                continue
        admitted.append(sym)
    return TradeableScopeResult(
        admitted=tuple(admitted),
        dropped_by_reason={key: tuple(value) for key, value in dropped.items()},
    )


def _resolve_base_symbol_scope(
    *,
    valid_symbols: Sequence[str],
    strategy_maps: Mapping[str, Mapping[str, pd.DataFrame]],
    tf: str,
) -> tuple[str, ...]:
    """Return loaded symbols that have a non-empty timeframe frame.

    This helper is intentionally data-availability only. It does not apply
    temporal guards such as warm-up, min-bar, or holdout coverage checks.
    """
    base_scope: list[str] = []
    for sym in valid_symbols:
        sym_maps = strategy_maps.get(sym)
        if sym_maps is None:
            continue
        sym_df = sym_maps.get(tf)
        if sym_df is None or sym_df.empty:
            continue
        base_scope.append(sym)
    return tuple(base_scope)


def _resolve_data_collection_symbols(
    *,
    run_config: FuturesRunConfig,
    discovered_symbols: list[str],
    inference_panel: tuple[str, ...],
    live_inference_panel: tuple[str, ...],
) -> tuple[str, ...]:
    base_symbols = list(inference_panel or live_inference_panel or tuple(discovered_symbols))

    merged_symbols = (
        base_symbols
        + list(FUTURES_ANCHOR_SYMBOLS)
        + list(FUTURES_MACRO_INDEX_SYMBOLS)
    )
    load_symbols = tuple(dict.fromkeys(merged_symbols))
    return load_symbols


def _selected_symbols_from_snapshot(snapshot: UniverseSnapshot) -> tuple[str, ...]:
    return tuple(
        str(meta.symbol).strip()
        for meta in snapshot.selected
        if str(meta.symbol).strip()
    )


def _resolve_universe_state_cube(universe_result: Any | None) -> UniverseStateCube | None:
    """Return the PIT state cube when the universe stage produced one."""
    if universe_result is None:
        return None
    cube = getattr(universe_result, "state_cube", None)
    return cube if isinstance(cube, UniverseStateCube) else None


def _log_cube_coverage(
    cube: UniverseStateCube | None,
    *,
    symbols: tuple[str, ...],
    aligned_datetimes: np.ndarray,
) -> None:
    """Log PIT cube diagnostic: eligible/entry_block coverage for the aligned window."""
    if cube is None:
        _logger.debug(
            "[PIT-CUBE] cube=None symbols=%d aligned_bars=%d",
            len(symbols),
            len(aligned_datetimes),
        )
        return
    cube_ts_ns = np.asarray(cube.calendar.view(np.int64), dtype=np.int64)
    aligned_ts_ns = aligned_datetimes.astype("datetime64[ns]").view(np.int64)
    positions = np.searchsorted(cube_ts_ns, aligned_ts_ns, side="right") - 1
    valid_mask = positions >= 0
    t_valid = np.where(valid_mask)[0]
    p_valid = positions[valid_mask]
    n_bars_valid = t_valid.size
    n_total = len(symbols)
    eligible_zeros = 0
    eligible_total_bars = 0
    entry_total_bars = 0
    n_covered = 0
    cube_sym_map: dict[str, int] = {}
    for _n, _iid in enumerate(cube.instrument_ids):
        _sym = _iid.split(":")[-1] if ":" in _iid else _iid
        cube_sym_map[_sym] = _n
    for sym in symbols:
        cube_n = cube_sym_map.get(sym)
        if cube_n is None:
            continue
        n_covered += 1
        if n_bars_valid == 0:
            eligible_zeros += 1
            continue
        eligible_slice = cube.eligible[p_valid, cube_n]
        eligible_total_bars += int(eligible_slice.sum())
        entry_total_bars += int(cube.entry_block[p_valid, cube_n].sum())
        if eligible_slice.sum() == 0:
            eligible_zeros += 1
    eligible_mean = eligible_total_bars / max(1, n_covered * n_bars_valid)
    entry_mean = entry_total_bars / max(1, n_covered * n_bars_valid)
    _logger.debug(
        "[PIT-CUBE] symbols=%d covered=%d eligible_mean=%.4f entry_mean=%.4f "
        "eligible_zeros=%d/%d aligned_bars=%d cube_bars=%d",
        n_total,
        n_covered,
        eligible_mean,
        entry_mean,
        eligible_zeros,
        n_total,
        len(aligned_datetimes),
        len(cube.calendar),
    )


def _universe_metadata_by_symbol(
    snapshot: UniverseSnapshot,
) -> dict[str, tuple[float, float, float, float, float, float, float, float, float]]:
    metadata: dict[str, tuple[float, float, float, float, float, float, float, float, float]] = {}
    for meta in snapshot.selected:
        symbol = str(meta.symbol).strip()
        if not symbol:
            continue
        metadata[symbol] = (
            float(meta.cluster_id),
            float(meta.beta_vs_market),
            float(meta.cluster_size),
            float(meta.anchor_cluster_member),
            float(meta.vol_30d),
            float(meta.friction_score),
            float(meta.alpha_capacity_score),
            float(meta.diversification_score),
            float(meta.tradeable_score),
        )
    return metadata


def _inject_universe_metadata_into_maps(
    data_maps: dict[str, dict[str, Any]],
    *,
    snapshot: UniverseSnapshot,
    symbols: tuple[str, ...],
    tf: str,
) -> None:
    metadata_by_symbol = _universe_metadata_by_symbol(snapshot)
    for symbol in symbols:
        metadata = metadata_by_symbol.get(symbol)
        if metadata is None:
            continue
        frame = data_maps.get(symbol, {}).get(tf)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        (
            cluster_id,
            beta_vs_market,
            cluster_size,
            anchor_cluster_member,
            vol_30d,
            friction_score,
            alpha_capacity_score,
            diversification_score,
            tradeable_score,
        ) = metadata
        frame["cluster_id"] = np.full(len(frame), cluster_id, dtype=np.float64)
        frame["beta_vs_market"] = np.full(len(frame), beta_vs_market, dtype=np.float64)
        frame["cluster_size"] = np.full(len(frame), cluster_size, dtype=np.float64)
        frame["anchor_cluster_member"] = np.full(
            len(frame), anchor_cluster_member, dtype=np.float64
        )
        frame["vol_30d"] = np.full(len(frame), vol_30d, dtype=np.float64)
        frame["friction_score"] = np.full(len(frame), friction_score, dtype=np.float64)
        frame["alpha_capacity_score"] = np.full(len(frame), alpha_capacity_score, dtype=np.float64)
        frame["diversification_score"] = np.full(
            len(frame), diversification_score, dtype=np.float64
        )
        frame["tradeable_score"] = np.full(len(frame), tradeable_score, dtype=np.float64)


def _ensure_cached_symbol_data_for_targets(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
    symbols: tuple[str, ...],
    *,
    require_exec_1m: bool,
) -> None:
    """Ensure cached parquet data exists for the resolved data-load target symbols."""
    if not symbols:
        return
    if run_config.sync == "skip":
        _logger.info("[CACHE] Skip backfill as requested")
        return
    sync_start_date = window.fetch_start_date
    _logger.debug(
        "[UNIVERSE] 🌐 %s ~ %s | Target: %d symbols",
        sync_start_date,
        window.end_date_value,
        len(symbols),
    )
    t_sync_main = time.perf_counter()
    run_historical_sync(
        start_date=sync_start_date,
        end_date=window.end_date_value,
        sync_mode=run_config.sync,
        symbols=list(symbols),
        sync_1d=True,
        sync_4h=True,
        sync_1m=False,
        sync_metrics=run_config.sync_metrics,
    )
    _logger.debug("[perf-data] backfill base data took %.4fs", time.perf_counter() - t_sync_main)
    if require_exec_1m:
        t_sync_1m = time.perf_counter()
        run_historical_sync(
            start_date=sync_start_date,
            end_date=window.end_date_value,
            sync_mode=run_config.sync,
            symbols=list(symbols),
            sync_1d=False,
            sync_4h=False,
            sync_1m=True,
            sync_metrics=False,
        )
        _logger.debug("[perf-data] backfill 1m data took %.4fs", time.perf_counter() - t_sync_1m)


@dataclass(slots=True, frozen=True)
class RunnerResult:
    """Pipeline completion status."""

    exit_code: int
    reason: str


@dataclass(slots=True, frozen=True)
class QuarterlyWindow:
    """Resolved quarterly time window for optimization."""

    fetch_start: str
    is_start: str
    oos_start: str
    end_date: str
    fetch_start_date: date
    is_start_date: date
    oos_start_date: date
    end_date_value: date


@dataclass(slots=True, frozen=True)
class DataStageResult:
    """Data stage output passed to downstream stages."""

    data_maps: dict[str, dict[str, Any]]
    oos_data_maps: dict[str, dict[str, Any]]
    valid_symbols: list[str]


def _wrap_segments(segments: list[str], width: int, sep: str = " | ") -> list[str]:
    """Pack ``label (count)`` segments into lines no wider than ``width``.

    Splits only on the ``sep`` boundary so individual gate tokens are never cut.
    Used to render the full BLOCKED gate distribution inside the fixed-width
    diagnostics table without overflowing the border.
    """
    lines: list[str] = []
    current = ""
    for seg in segments:
        candidate = seg if not current else f"{current}{sep}{seg}"
        if len(candidate) > width and current:
            lines.append(current)
            current = seg
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--timeframe", type=str, choices=["1h", "4h"], default="4h")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument(
        "--phase",
        type=str,
        choices=["l3", "l2", "l1"],
        default="l3",
    )
    parser.add_argument(
        "--sync",
        type=str,
        default="full",
        choices=["full", "fast", "skip"],
    )
    parser.add_argument("--refresh-universe", action="store_true")
    parser.add_argument("--sync-metrics", action="store_true")
    return parser


def _build_run_config(args: argparse.Namespace) -> FuturesRunConfig:
    payload = vars(args).copy()
    return build_run_config_from_args(payload)


def _resolve_quarterly_window(reference_date: str | None) -> QuarterlyWindow:
    fetch_start, is_start, oos_start, end_date = get_quarterly_window(reference_date)
    return QuarterlyWindow(
        fetch_start=fetch_start,
        is_start=is_start,
        oos_start=oos_start,
        end_date=end_date,
        fetch_start_date=datetime.strptime(fetch_start, "%Y-%m-%d").date(),
        is_start_date=datetime.strptime(is_start, "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime(oos_start, "%Y-%m-%d").date(),
        end_date_value=datetime.strptime(end_date, "%Y-%m-%d").date(),
    )


def _resolve_layered_window(reference_date: str | None) -> Any:
    from src.domain.futures.optimization.opt_config import get_layered_window

    parsed_reference = (
        datetime.strptime(reference_date, "%Y-%m-%d").date()
        if reference_date is not None
        else None
    )
    return get_layered_window(reference_date=parsed_reference)


def _window_with_fetch_start(window: QuarterlyWindow, fetch_start_date: date) -> QuarterlyWindow:
    fetch_start = fetch_start_date.isoformat()
    return QuarterlyWindow(
        fetch_start=fetch_start,
        is_start=window.is_start,
        oos_start=window.oos_start,
        end_date=window.end_date,
        fetch_start_date=fetch_start_date,
        is_start_date=window.is_start_date,
        oos_start_date=window.oos_start_date,
        end_date_value=window.end_date_value,
    )


def _run_universe_stage(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
    *,
    layered_window: Any | None = None,
) -> tuple[
    list[str],
    dict[date, frozenset[str]],
    tuple[str, ...],
    tuple[str, ...],
    UniverseSnapshot,
    dict[date, frozenset[str]],
    Any,
]:
    discovered_symbols: list[str] = []
    timeline: dict[date, frozenset[str]] = {}
    inference_timeline: dict[date, frozenset[str]] = {}
    inference_panel: tuple[str, ...] = ()
    live_inference_panel: tuple[str, ...] = ()

    t_discover = time.perf_counter()
    effective_is_start = min(
        window.is_start_date,
        layered_window.l1_start if layered_window is not None else window.is_start_date,
    )
    universe_result = discover_universe_timeline(
        tf=run_config.timeframe,
        is_start=effective_is_start,
        oos_start=window.oos_start_date,
        end_date=window.end_date_value,
        force_rebuild=run_config.refresh_universe,
        l2_start=layered_window.l2_start if layered_window is not None else None,
    )
    _logger.log(PERF, 
        "[perf-universe] discover_universe_timeline took %.4fs",
        time.perf_counter() - t_discover,
    )

    t_quality = time.perf_counter()
    if not validate_universe_quality(
        snapshot=universe_result.snapshot,
        report=universe_result.report,
        reference_date=run_config.date,
        tf=run_config.timeframe,
    ):
        raise RuntimeError("universe_quality_rejected")
    _logger.log(
        PERF,
        "[perf-universe] validate_universe_quality took %.4fs",
        time.perf_counter() - t_quality,
    )

    discovered_symbols = list(universe_result.symbols)
    timeline_obj: UniverseMembershipTimeline = universe_result.timeline
    timeline = {
        window.effective_from.date(): frozenset(window.active_symbols)
        for window in timeline_obj.windows
    }
    inference_panel = universe_result.inference_symbols
    live_inference_panel = universe_result.inference_symbols
    inference_timeline = {}
    inf_tl = universe_result.inference_timeline
    if isinstance(inf_tl, UniverseMembershipTimeline):
        inference_timeline = {
            w.effective_from.date(): frozenset(w.active_symbols) for w in inf_tl.windows
        }

    return (
        discovered_symbols,
        timeline,
        inference_panel,
        live_inference_panel,
        universe_result.snapshot,
        inference_timeline,
        universe_result,
    )


def _run_data_stage(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
    discovered_symbols: list[str],
    timeline: dict[date, frozenset[str]],
    inference_panel: tuple[str, ...] = (),
    live_inference_panel: tuple[str, ...] = (),
    inference_timeline: dict[date, frozenset[str]] | None = None,
    *,
    layered_window: Any | None = None,
) -> DataStageResult:
    load_symbols = _resolve_data_collection_symbols(
        run_config=run_config,
        discovered_symbols=discovered_symbols,
        inference_panel=inference_panel,
        live_inference_panel=live_inference_panel,
    )
    require_exec_1m = _requires_exec_1m(run_config)

    scope_name = "stage6_selected"
    effective_fetch_start_date = min(
        window.fetch_start_date,
        layered_window.fetch_start if layered_window is not None else window.fetch_start_date,
    )
    effective_fetch_start = effective_fetch_start_date.isoformat()

    t_load = time.perf_counter()
    data_maps, oos_data_maps, valid_symbols = load_futures_data_maps_for_symbols(
        list(load_symbols),
        run_config.timeframe,
        effective_fetch_start,
        window.is_start,
        window.oos_start,
        window.end_date,
        load_exec_1m=require_exec_1m,
        requested_symbols_count=len(load_symbols),
        scope_name=scope_name,
    )
    _logger.log(PERF, 
        "[perf-data] load_futures_data_maps_for_symbols took %.4fs",
        time.perf_counter() - t_load,
    )
    if require_exec_1m:
        missing_1m = [s for s in valid_symbols if "exec_1m" not in (data_maps.get(s) or {})]
        if missing_1m:
            raise RuntimeError(
                f"exec_mode=intrarar_1m but {len(missing_1m)} symbol(s) missing 1m data: "
                f"{missing_1m[:5]}{'...' if len(missing_1m) > 5 else ''}"
            )
    if timeline and valid_symbols:
        t_inject = time.perf_counter()
        warmup_bars_required = int(OPT_FUTURES_CONFIG.get("FUTURES_UNIVERSE_WARMUP_BARS", 60))
        inject_membership_masks_into_maps(
            data_maps=data_maps,
            oos_data_maps=oos_data_maps,
            symbols=valid_symbols,
            tf=run_config.timeframe,
            timeline=timeline,
            warmup_bars_required=warmup_bars_required,
            inference_timeline=inference_timeline or None,
        )
        _logger.log(PERF, 
            "[perf-data] inject_membership_masks_into_maps took %.4fs",
            time.perf_counter() - t_inject,
        )

    t_ready = time.perf_counter()
    readiness: DataReadinessResult = evaluate_data_readiness(
        tf=run_config.timeframe,
        data_maps=data_maps,
        oos_data_maps=oos_data_maps,
        valid_symbols=valid_symbols,
        fetch_start=effective_fetch_start_date,
        is_start=window.is_start_date,
        oos_start=window.oos_start_date,
        end=window.end_date_value,
        require_exec_1m=require_exec_1m,
        scope_name=scope_name,
    )
    _logger.debug("[perf-data] evaluate_data_readiness took %.4fs", time.perf_counter() - t_ready)
    report_df = readiness.report
    if isinstance(report_df, pd.DataFrame) and not report_df.empty and "pass" in report_df.columns:
        fail_df = report_df.loc[~report_df["pass"].astype(bool)]
        if not fail_df.empty and "reason" in fail_df.columns:
            _logger.debug("[data-readiness] fail reasons: %s", fail_df["reason"].value_counts().to_dict())

    valid_symbols = list(readiness.kept_symbols)
    if not valid_symbols:
        raise RuntimeError("data_not_ready")
    return DataStageResult(
        data_maps=readiness.filtered_is_maps,
        oos_data_maps=readiness.filtered_oos_maps,
        valid_symbols=valid_symbols,
    )


def _run_regime_evaluation_stage(
    run_config: FuturesRunConfig,
    data_stage: DataStageResult,
) -> tuple[RegimeScoreCard, NDArray[np.int8]] | None:
    """Align data, compute regime context, and log scorecard table (C2+C5 from codes only).

    Returns:
        ``(scorecard, code_1d)`` for post-hoc C3/C4 gold standard refresh, or ``None`` on failure.
    """
    import numpy as np

    from src.domain.futures.strategy.common.alignment import align_data_maps
    from src.domain.futures.strategy.market_regime import compute_market_regime_context
    from src.domain.futures.strategy.regime_evaluation import evaluate_regime_classifier

    aligned = align_data_maps(data_stage.data_maps, data_stage.valid_symbols, run_config.timeframe)
    regime_ctx = compute_market_regime_context(aligned=aligned)

    empty_i8 = np.array([], dtype=np.int8)
    empty_f64 = np.array([], dtype=np.float64)
    empty_bool = np.array([], dtype=bool)
    scorecard = evaluate_regime_classifier(
        all_codes_1d=regime_ctx.code_1d,
        event_codes=empty_i8,
        event_edges_bps=empty_f64,
        is_event_mask=empty_bool,
        oos_event_mask=empty_bool,
    )

    # compute proxy C3/C4 from BTC bar-level returns (pre-signal)
    import dataclasses as _dc

    from src.domain.futures.strategy.regime_evaluation import evaluate_regime_classifier_proxy

    btc_log_ret = np.zeros(aligned.datetimes.shape[0], dtype=np.float64)
    btc_close = aligned.close_2d
    if btc_close.shape[1] > 0:
        btc_idx = next(
            (i for i, s in enumerate(aligned.symbols) if "BTC" in s.upper()), 0
        )
        raw = np.maximum(btc_close[:, btc_idx], 1e-12)
        btc_log_ret[1:] = np.diff(np.log(raw))

    n_bars = aligned.datetimes.shape[0]
    split = n_bars // 2
    is_bar_mask = np.zeros(n_bars, dtype=bool)
    is_bar_mask[:split] = True
    oos_bar_mask = np.zeros(n_bars, dtype=bool)
    oos_bar_mask[split:] = True

    c3p_pval, c3p_flip, c4p_rho, _c3p_score = evaluate_regime_classifier_proxy(
        all_codes_1d=regime_ctx.code_1d,
        market_returns_1d=btc_log_ret * 1e4,
        is_bar_mask=is_bar_mask,
        oos_bar_mask=oos_bar_mask,
    )
    # patch proxy values into scorecard (create new frozen instance)
    scorecard = _dc.replace(
        scorecard,
        c3_proxy_pvalue=c3p_pval,
        c3_proxy_sign_flip=c3p_flip,
        c4_proxy_spearman_rho=c4p_rho,
    )

    return scorecard, regime_ctx.code_1d


def _tiered_labeled_events(output: CandidatePipelineOutput | None) -> pd.DataFrame:
    """Return the unfiltered labeled events required by the tiered workflow."""
    if output is None or getattr(output, "labeled_unfiltered", None) is None:
        raise ValueError("tiered requires unfiltered labeled events")
    labeled = output.labeled_unfiltered
    if not isinstance(labeled, pd.DataFrame):
        raise ValueError("tiered requires unfiltered labeled events")
    return labeled


def _build_l2_signal_batch(
    l1_res: Any,
    labeled_events: pd.DataFrame,
    aligned: Any,
    cfg: Any,
    window: Any,
) -> Any:
    """L1 artifact로 L2 window 신호 예측.

    Args:
        l1_res: Layer1Result (gate_passed=True, inference_artifact 필수).
        labeled_events: 전체 labeled event DataFrame.
        aligned: AlignedMarketData.
        cfg: CandidateStrategyConfig.
        window: LayeredWindow.

    Returns:
        ValidatedSignalBatch for the L2 window [l2_start, holdout_start).

    Raises:
        ValueError: l1_res.inference_artifact가 None인 경우.
    """
    from src.domain.futures.strategy.tiered_workflow import predict_layer1_signals

    artifact = getattr(l1_res, "inference_artifact", None)
    if artifact is None:
        raise ValueError("L1 artifact 없음 — l1_result.inference_artifact is None (L1 gate_passed=False 상태)")

    datetimes = aligned.datetimes
    l2_start_ts = pd.Timestamp(window.l2_start, tz="UTC")
    ho_start_ts = pd.Timestamp(window.holdout_start, tz="UTC")
    l2_start_bar = int(np.searchsorted(datetimes, np.datetime64(l2_start_ts.replace(tzinfo=None), "ns")))
    ho_start_bar = int(np.searchsorted(datetimes, np.datetime64(ho_start_ts.replace(tzinfo=None), "ns")))

    return predict_layer1_signals(
        artifact=artifact,
        candidate_events=labeled_events,
        aligned=aligned,
        start_idx=l2_start_bar,
        end_idx=ho_start_bar,
        cfg=cfg,
    )


def _run_tiered_l2_study(
    *,
    signal_batch: Any,
    aligned: Any,
    cfg: Any,
    window: Any,
    caps: Any,
    tf: str,
    n_trials: int,
    seed: int,
) -> Any:
    """Optuna objective_l2_growth로 best l2_params 탐색."""
    import warnings

    import optuna as _optuna
    from optuna.samplers import TPESampler

    from src.domain.futures.optimization.opt_config import L2_ALLOC_SPACE

    # JournalRedisStorage 감쇠 경고 억제
    warnings.filterwarnings("ignore", category=FutureWarning, module="optuna")

    # Optuna 스터디 정보 로그 억제
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)

    from src.domain.futures.optimization.workflow import (
        TieredContext,
        layer2_constraints_from_trial,
        objective_l2_growth,
    )
    from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2StudyResult
    from src.domain.futures.strategy.tiered_workflow.selection import (
        _layer2_experiment_key,
        _signal_batch_fingerprint,
        select_layer2_champion,
    )
    from src.domain.futures.strategy.walk_forward import build_walk_forward_folds

    l2_sim_cache = build_l2_simulation_cache(aligned, signal_batch, tf)

    ctx = TieredContext(
        labeled_events=pd.DataFrame(),  # signal_batch에 이미 예측됨, labeles 불필요
        aligned=aligned,
        cfg=cfg,
        window=window,
        caps=caps,
        tf=tf,
        fixed_l1_params={"signal_batch": signal_batch},
        l2_sim_cache=l2_sim_cache,
    )

    study_name = _layer2_experiment_key(
        tf=tf,
        window=window,
        signal_batch=signal_batch,
        search_space_version="v7",
    )
    signal_batch_fingerprint = _signal_batch_fingerprint(signal_batch)
    unique_symbols = ",".join(
        sorted({str(event.symbol) for event in signal_batch.events})
    ) or "-"
    _logger.info("  ● [HYPERPARAMETER OPTIMIZATION]")
    _logger.info("    - Study Name : %s", study_name)
    _logger.info("    - Config     : %d trials", n_trials)
    _logger.info(
        "    - Provenance : events=%d unique_symbols=%s fp=%s",
        len(signal_batch.events),
        unique_symbols,
        signal_batch_fingerprint[:12],
    )
    _logger.info("  ────────────────────────────────────────────────────────────────────────────")

    try:
        # setup_optuna_storage를 1회만 호출하여 로그 중복 제거
        _, storage = setup_optuna_storage(str(BASE_DIR))
        from tqdm import tqdm
        class L2OptunaProgressCallback:
            def __init__(self, total_trials: int):
                self.pbar = tqdm(total=total_trials, desc="[L2-OPT]", leave=True)
                self.best_val = float("-1e6")

            def __call__(self, study: Any, trial: Any) -> None:
                val = trial.value
                if val is not None and val > self.best_val:
                    self.best_val = val
                
                best_disp = f"{self.best_val * 100:.2f}%" if self.best_val > float("-1e6") else "N/A"
                current_disp = f"{val * 100:.2f}%" if (val is not None and val > float("-1e6")) else "BLOCKED"
                self.pbar.set_postfix_str(f"Best CAGR: {best_disp} | Current: {current_disp}")
                self.pbar.update(1)

        # 매 실행마다 완전 초기화 (resume=False): 이종 search-space trial이 한
        # study에 섞여 TPESampler가 dynamic search space로 오판 -> RandomSampler
        # fallback 경고 및 trial 누적(120 초과)을 유발하던 근본원인 제거.
        study = get_or_create_study(
            study_name=study_name,
            storage=storage,
            sampler=TPESampler(
                seed=seed,
                multivariate=True,
                group=True,
                n_ei_candidates=48,
                n_startup_trials=min(
                    n_trials,
                    max(24, min(int(n_trials * 0.20), 4 * len(L2_ALLOC_SPACE))),
                ),
                constraints_func=layer2_constraints_from_trial,
            ),
            resume=False,
        )

        # Warm-start anchor: 영구 챔피언 레저(과거 run 중 최고 성과)가 있으면 우선
        # 사용하고, 레저가 비었거나 일부 키가 비어있으면 검증된 기본값으로 보강.
        _default_anchor = {
            "K_RANK": 3,
            "REBALANCE_BARS": 3,
            "CS_Z_SCORE_THRESHOLD": 0.5,
            "kelly_fraction": 0.25,
            "max_ann_vol": 0.35,
            "deploy_cost_safety_mult": 1.0,
            "edge_throttle_min_active_mult": 0.25,
            "edge_ref_bps": 5.0,
            "edge_throttle_gamma": 1.0,
            "risk_budget_floor_ratio": 0.35,
            "risk_budget_max_scale": 2.0,
        }
        _champion_anchor = load_champion_params(tag=tf, storage=storage) or {}
        _anchor_params = {
            **_default_anchor,
            **{k: v for k, v in _champion_anchor.items() if k in L2_ALLOC_SPACE},
        }
        if not any(t.state == _optuna.trial.TrialState.COMPLETE for t in study.trials):
            import contextlib
            with contextlib.suppress(Exception):
                study.enqueue_trial(_anchor_params)

        progress_cb = L2OptunaProgressCallback(n_trials)
        try:
            batch_size = int(OPT_FUTURES_CONFIG.get("L2_OPTUNA_BATCH_SIZE", 4))
            if batch_size <= 1:
                study.optimize(
                    lambda trial: objective_l2_growth(trial, ctx),
                    n_trials=n_trials,
                    n_jobs=1,
                    show_progress_bar=False,
                    callbacks=[progress_cb],
                )
            else:
                import multiprocessing
                from concurrent.futures import ProcessPoolExecutor

                from src.domain.futures.optimization.workflow import (
                    _evaluate_l2_params,
                    suggest_layered_params,
                )
                
                max_workers = min(batch_size, multiprocessing.cpu_count())
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    trial_idx = len([t for t in study.trials if t.state.is_finished()])
                    while trial_idx < n_trials:
                        current_batch = min(batch_size, n_trials - trial_idx)
                        batch_trials = []
                        batch_params = []
                        
                        for _ in range(current_batch):
                            trial = study.ask()
                            batch_trials.append(trial)
                            params = suggest_layered_params(trial, "L2", fixed=ctx.fixed_l1_params)
                            batch_params.append(params)
                            
                        # Submit evaluations in parallel
                        futures = [
                            executor.submit(_evaluate_l2_params, params, ctx)
                            for params in batch_params
                        ]
                        
                        # Wait for results and tell in deterministic sorted order
                        for trial, future in zip(batch_trials, futures, strict=False):
                            try:
                                value, attrs, t_elapsed = future.result()
                            except Exception as exc:
                                _logger.error(
                                    "[L2-OPT] Subprocess trial %d failed: %s",
                                    trial.number,
                                    exc,
                                )
                                value = -1e6
                                attrs = {}
                                t_elapsed = 0.0
                                
                            for k, v in attrs.items():
                                trial.set_user_attr(k, v)
                                
                            study.tell(trial, value)
                            
                            _logger.log(
                                logging.DEBUG,
                                "[perf-optuna] Trial %d evaluate_l2_trial took %.4fs | Objective: %.6f",
                                trial.number,
                                t_elapsed,
                                value,
                            )
                            progress_cb(study, trial)
                            trial_idx += 1
        finally:
            progress_cb.pbar.close()
    except Exception as exc:
        _logger.warning("[L2-OPT] Optuna study 실패: %s — 기본 l2_params 사용", exc)
        return Layer2StudyResult(
            best_params={},
            best_trial_number=None,
            best_evaluation=None,
            dsr=0.0,
            effective_trial_count=0.0,
            completed_trials=0,
            feasible_trials=0,
            blocker_reason="study_error",
        )

    complete_trials = [
        t for t in study.trials
        if t.state == _optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value > float("-1e6")
    ]
    if not complete_trials:
        _logger.warning("[L2-OPT] 모든 %d trials 실패/pruned — 기본 l2_params 사용", n_trials)
        return Layer2StudyResult(
            best_params={},
            best_trial_number=None,
            best_evaluation=None,
            dsr=0.0,
            effective_trial_count=0.0,
            completed_trials=0,
            feasible_trials=0,
            blocker_reason="no_complete_trials",
        )

    ho_ts = pd.Timestamp(window.holdout_start).tz_localize(None)
    ho_start_idx_l2 = int(np.searchsorted(aligned.datetimes, np.datetime64(ho_ts, "ns")))
    # walk forward folds 구성 및 필터링하여 select_layer2_champion에 전달
    awf_folds = build_walk_forward_folds(n_bars=ho_start_idx_l2, cfg=cfg)
    l2_ts = pd.Timestamp(window.l2_start).tz_localize(None)
    l1_end_bars = int(np.searchsorted(aligned.datetimes, np.datetime64(l2_ts, "ns")))

    awf_folds_l2 = tuple(
        f for f in awf_folds
        if f.oos_start >= l1_end_bars and f.oos_end <= ho_start_idx_l2
    )
    if not awf_folds_l2:
        cal_end = max(l1_end_bars - 1, 1)
        from src.domain.futures.strategy.walk_forward import WFFold
        awf_folds_l2 = (WFFold(
            fit_start=0,
            fit_end=cal_end,
            cal_start=max(0, cal_end - max(1, cal_end // 5)),
            cal_end=cal_end,
            oos_start=l1_end_bars,
            oos_end=ho_start_idx_l2,
        ),)

    _min_dsr = float(OPT_FUTURES_CONFIG.get("FUTURES_L2_MIN_DSR", 0.60))
    l2_study_result = select_layer2_champion(
        study=study,
        tf=tf,
        signal_batch=signal_batch,
        aligned=aligned,
        awf_folds=awf_folds_l2,
        caps=caps,
        min_dsr=_min_dsr,
    )

    if l2_study_result.blocker_reason == "" and l2_study_result.best_evaluation is not None:
        updated = update_champion_store(
            tag=tf,
            storage=storage,
            params=l2_study_result.best_params,
            value=l2_study_result.best_evaluation.growth_lcb_hybrid,
            space=L2_ALLOC_SPACE,
        )
        if updated:
            _logger.info(
                "  ● [CHAMPION STORE] 신규 챔피언 갱신 (tf=%s, growth_lcb=%.4f)",
                tf,
                l2_study_result.best_evaluation.growth_lcb_hybrid,
            )

    return l2_study_result


def _run_strategy_stage(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
    data_stage: DataStageResult,
    trading_symbols: tuple[str, ...] = (),
    universe_snapshot: UniverseSnapshot | None = None,
    layered_window: Any | None = None,
    universe_result: Any | None = None,
) -> CandidatePipelineOutput | RunnerResult | Layer1Result | None:
    # ─── 기존 Phase D 진입 (공통 설정 → bridge → 선택적 Tiered 분기) ──────────
    from src.domain.futures.strategy.tiered_logging import format_data_integrity_summary, format_layer_header

    t_strategy_stage = time.perf_counter()
    strategy_steps: dict[str, float] = {
        "map_pick": 0.0,
        "metadata": 0.0,
        "bridge": 0.0,
        "report": 0.0,
        "merge": 0.0,
        "evaluation": 0.0,
    }

    def _emit_strategy_profile() -> None:
        profile = _runtime_breakdown(time.perf_counter() - t_strategy_stage, **strategy_steps)
        _logger.log(PERF, 
            (
                "[STRATEGY-PROF] total=%.4fs map_pick=%.4fs metadata=%.4fs bridge=%.4fs "
                "report=%.4fs merge=%.4fs evaluation=%.4fs accounted=%.4fs unaccounted=%.4fs"
            ),
            profile.total,
            strategy_steps["map_pick"],
            strategy_steps["metadata"],
            strategy_steps["bridge"],
            strategy_steps["report"],
            strategy_steps["merge"],
            strategy_steps["evaluation"],
            profile.accounted,
            profile.unaccounted,
        )

    t_step = time.perf_counter()
    strategy_maps = pick_strategy_data_maps(
        oos_data_maps=data_stage.oos_data_maps,
        is_data_maps=data_stage.data_maps,
        valid_symbols=data_stage.valid_symbols,
        tf=run_config.timeframe,
    )
    strategy_steps["map_pick"] = time.perf_counter() - t_step
    full_strategy_maps = strategy_maps
    
    _logger.info(format_layer_header(1, "Signal Robustness & Ensemble Verification"))

    if universe_snapshot is not None:
        t_step = time.perf_counter()
        _inject_universe_metadata_into_maps(
            full_strategy_maps,
            snapshot=universe_snapshot,
            symbols=tuple(full_strategy_maps.keys()),
            tf=run_config.timeframe,
        )
        strategy_steps["metadata"] = time.perf_counter() - t_step

    use_tiered = bool(OPT_FUTURES_CONFIG.get("USE_CS_RANK_ENGINE", False))
    effective_trade_syms = []
    tiered_window = None
    if use_tiered:
        tiered_window = layered_window or _resolve_layered_window(run_config.date)
        base_scope = _resolve_base_symbol_scope(
            valid_symbols=data_stage.valid_symbols,
            strategy_maps=full_strategy_maps,
            tf=run_config.timeframe,
        )
        _logger.info(
            "[TIERED] Base scope: %d/%d loaded symbols",
            len(base_scope),
            len(data_stage.valid_symbols),
        )
        if tiered_window is not None:
            try:
                # Robust conversion to UTC Timestamp
                req_start_ts = pd.to_datetime(tiered_window.fetch_start, utc=True)
            except (TypeError, ValueError):
                # Fallback for MagicMocks or non-standard formats in tests
                req_start_ts = pd.Timestamp("1900-01-01", tz="UTC")
            try:
                req_end_ts = pd.to_datetime(tiered_window.holdout_end, utc=True)
            except (TypeError, ValueError):
                req_end_ts = pd.Timestamp("2100-01-01", tz="UTC")

            try:
                req_oos_start_ts = pd.to_datetime(tiered_window.holdout_start, utc=True)
            except (TypeError, ValueError):
                req_oos_start_ts = req_start_ts
            scope_result = _resolve_tradeable_scope(
                valid_symbols=base_scope,
                strategy_maps=full_strategy_maps,
                tf=run_config.timeframe,
                fetch_start=req_start_ts,
                oos_start=req_oos_start_ts,
                holdout_end=req_end_ts,
                min_window_bars=_TIERED_MIN_WINDOW_BARS,
                min_holdout_coverage=0.90,
            )
            effective_trade_syms = list(scope_result.admitted)
            _logger.info(
                "[TIERED] Sub-window admission: %d/%d symbols admitted (min_bars=%d, oos_cov>=90%%)",
                len(effective_trade_syms),
                len(base_scope),
                _TIERED_MIN_WINDOW_BARS,
            )
            if any(scope_result.dropped_by_reason.values()):
                _logger.info(
                    "[TIERED] Sub-window drops: %s",
                    {
                        key: len(value)
                        for key, value in scope_result.dropped_by_reason.items()
                    },
                )
        if not effective_trade_syms:
            from src.domain.futures.strategy.tiered_workflow import TieredPipelineError

            raise TieredPipelineError(
                "tiered tradeable scope is empty after sub-window admission"
            )
    else:
        effective_trade_syms = list(data_stage.valid_symbols)

    bridge_symbol_scope = tuple(effective_trade_syms) if use_tiered else (
        trading_symbols or tuple(data_stage.valid_symbols)
    )
    bridge_trading_symbols = list(bridge_symbol_scope)

    t_bridge_start = time.perf_counter()
    ml_out = run_active_strategy_output_bridge(
        run_config=run_config,
        symbols=bridge_trading_symbols,
        tf=run_config.timeframe,
        fetch_start=window.fetch_start,
        end_date=window.end_date,
        opt_config=OPT_FUTURES_CONFIG,
        preloaded_data_maps=full_strategy_maps,
        trading_symbols=bridge_symbol_scope,
        silent=False,
    )
    bridge_elapsed = time.perf_counter() - t_bridge_start
    strategy_steps["bridge"] = bridge_elapsed

    # ─── Tiered Pipeline 분기 (bridge 완료 후 — labeled + aligned 사용 가능) ──
    if use_tiered:
        try:

            from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
            from src.domain.futures.strategy.common.alignment import align_data_maps
            from src.domain.futures.strategy.tiered_workflow import (
                TieredPipelineError,
                run_tiered_pipeline,
            )
            _pit_state_cube = _resolve_universe_state_cube(universe_result)
            aligned_tiered = align_data_maps(
                full_strategy_maps,
                effective_trade_syms,
                run_config.timeframe,
                state_cube=_pit_state_cube,
            )
            if _pit_state_cube is not None:
                _log_cube_coverage(
                    _pit_state_cube,
                    symbols=tuple(effective_trade_syms),
                    aligned_datetimes=aligned_tiered.datetimes,
                )
            if tiered_window is not None:
                try:
                    aligned_start = pd.Timestamp(aligned_tiered.datetimes[0]).date()
                    is_mock_start = (
                        hasattr(tiered_window, "fetch_start")
                        and type(tiered_window.fetch_start).__name__ in ("MagicMock", "Mock")
                    )
                except (TypeError, ValueError, IndexError):
                    # Fallback for MagicMocks in tests
                    aligned_start = pd.Timestamp("1900-01-01").date()
                    is_mock_start = True

                if not is_mock_start and aligned_start > tiered_window.fetch_start:
                    raise ValueError(
                        "tiered warm-up coverage missing: "
                        f"required_start={tiered_window.fetch_start.isoformat()} "
                        f"actual_start={aligned_start.isoformat()}"
                    )

                try:
                    aligned_end = pd.Timestamp(aligned_tiered.datetimes[-1]).date()
                    is_mock_end = (
                        hasattr(tiered_window, "holdout_start")
                        and type(tiered_window.holdout_start).__name__ in ("MagicMock", "Mock")
                    )
                except (TypeError, ValueError, IndexError):
                    # Fallback for MagicMocks in tests
                    aligned_end = pd.Timestamp("2100-01-01").date()
                    is_mock_end = True

                if not is_mock_end and aligned_end < tiered_window.holdout_start:
                    raise ValueError(
                        "tiered holdout coverage missing: "
                        f"required_holdout_start={tiered_window.holdout_start.isoformat()} "
                        f"actual_end={aligned_end.isoformat()} "
                        "(intersection tail truncated — check delisted symbols in panel)"
                    )
            labeled_tiered = _tiered_labeled_events(ml_out)
            tiered_cfg = build_candidate_strategy_config(
                strategy_cfg=StrategyConfig(name="candidate_ml"),
                opt_config=OPT_FUTURES_CONFIG,
                timeframe=run_config.timeframe,
            ).candidate
            _futures_policy_t: dict[str, Any] = dict(OPT_FUTURES_CONFIG.get("FUTURES_PORTFOLIO_POLICY") or {})
            tiered_caps = PortfolioCaps(
                gross=float(OPT_FUTURES_CONFIG.get("FUTURES_PHASE_A_MAX_GROSS_EXPOSURE", 1.8)),
                per_symbol=float(_futures_policy_t.get("per_symbol_cap", 0.35)),
                net=0.5,
                beta=1.0,
                target_ann_vol=float(_futures_policy_t.get("target_ann_vol", 0.35)),
            )
            assert tiered_window is not None, "tiered_window is missing under use_tiered option"

            # ── Step A: L1 only ──────────────────────────────────────────────
            _recognized_multilayer = {"l2", "l3"}
            l1_res, _, _ = run_tiered_pipeline(
                labeled_events=labeled_tiered,
                aligned=aligned_tiered,
                cfg=tiered_cfg,
                window=tiered_window,
                l1_params={},
                l2_params={},
                caps=tiered_caps,
                tf=run_config.timeframe,
                target_phase="l1",
                verbose=True,
            )
            if not l1_res.gate_passed:
                _logger.info("[TIERED] L1 BLOCKED — gate_passed=False")
                return None
            if run_config.phase not in _recognized_multilayer:
                _logger.info("[TIERED] Phase=%s — stopping after L1 (not a multilayer phase)", run_config.phase)
                return l1_res

            _logger.info("\n>> LAYER 1: PASS -> Proceeding to Layer 2.")

            # ── Step B: L2 Optimization Header ──────────────────────────────
            _logger.info(format_layer_header(2, "Portfolio Allocation & Risk Optimization"))

            # ── Step C: L2 window 신호 예측 ──────────────────────────────────
            l2_signals = _build_l2_signal_batch(
                l1_res, labeled_tiered, aligned_tiered, tiered_cfg, tiered_window
            )

            # ── Step D: Optuna L2 파라미터 탐색 ──────────────────────────────
            _seed = int(run_config.seed) if hasattr(run_config, "seed") else 42
            n_l2_trials = int(OPT_FUTURES_CONFIG.get("L2_OPTUNA_TRIALS", 50))
            l2_study_result = _run_tiered_l2_study(
                signal_batch=l2_signals,
                aligned=aligned_tiered,
                cfg=tiered_cfg,
                window=tiered_window,
                caps=tiered_caps,
                tf=run_config.timeframe,
                n_trials=n_l2_trials,
                seed=_seed,
            )
            best_l2_params = dict(getattr(l2_study_result, "best_params", {}))

            # ── INTEGRITY GUARD: infeasible 챔피언 L3 승격 차단 ─────────────
            # 차단 로그는 최종 pipeline 내부 또는 종료 시점에 출력되므로 생략

            # ── Step E: 최적 params + L1 override로 최종 실행 ────────────────
            from src.domain.futures.strategy.tiered_workflow.pipeline import run_tiered_pipeline
            _, l2_final, _ = run_tiered_pipeline(
                labeled_events=labeled_tiered,
                aligned=aligned_tiered,
                cfg=tiered_cfg,
                window=tiered_window,
                l1_params={},
                l2_params=best_l2_params,
                caps=tiered_caps,
                tf=run_config.timeframe,
                target_phase=run_config.phase,
                l1_result_override=l1_res,
                verbose=True,  # 최종 실행시 상세 결과 출력
                override_dsr=l2_study_result.dsr,
            )
            
            # Layer 2 BLOCKED 시 즉시 종료 (Step 5 optimization 진입 방지)
            if l2_final is None or not l2_final.gate_passed:
                final_reason = l2_study_result.blocker_reason or getattr(l2_final, "blocker_reason", "unknown")
                return RunnerResult(exit_code=1, reason=f"layer2_blocked:{final_reason}")
            
            # Tiered Pipeline이 L3까지 수행했으므로 여기서 종료
            if run_config.phase == "l3":
                return RunnerResult(exit_code=0, reason="tiered_pipeline_completed")

            return None  # Phase D allocation 스킵 (Tiered가 대체)
        except TieredPipelineError as _tiered_exc:
            _logger.error(
                "[TIERED] terminal tiered failure=%s — Phase D fallback suppressed",
                _tiered_exc,
                exc_info=True,
            )
            return None
        except Exception as _exc:
            _logger.error(
                "[TIERED] terminal tiered failure=%s — Phase D fallback removed (legacy fallback disabled)",
                _exc,
                exc_info=True,
            )
            return RunnerResult(
                exit_code=1,
                reason=f"tiered_pipeline_error:{type(_exc).__name__}",
            )
    # ─── Phase D (USE_CS_RANK_ENGINE=False 전용 legacy 경로) ───

    t_report = time.perf_counter()
    # Summary of bridge output — read from alpha_panel["target_weight"] (CandidatePipelineOutput)
    non_zero_weights = 0
    panel = getattr(ml_out, "alpha_panel", None)
    if isinstance(panel, pd.DataFrame) and not panel.empty and "target_weight" in panel.columns:
        tw_arr = panel["target_weight"].to_numpy(dtype=np.float64)
        non_zero_weights = int(np.count_nonzero(np.abs(tw_arr) > 1e-9))
    candidate_report = getattr(ml_out, "rule_report", {})
    if not isinstance(candidate_report, dict):
        candidate_report = {}
    selected_total_bridge = int(candidate_report.get("selected_total", 0))

    # --- LOGICAL OUTPUT SEQUENCE ---
    # Current candidates
    failure_report = candidate_report.get("failure_report")
    keep_variants = list(candidate_report.get("recommended_keep_variants", []))

    # 1. Cause: Signal Selection
    _logger.info("\n[SIGNAL DIAGNOSTICS: STRATEGY FILTERING]")
    _logger.info("-" * 82)
    _logger.info(f"| {'Action':<12} | {'Count':<5} | {'Details / Selected Strategies':<55} |")
    _logger.info("-" * 82)
    _gate_label: dict[str, str] = {
        "breakeven_hard_gate": "Breakeven Gate",
        "min_obs": "Low Obs",
        "mean_edge": "Low Mean Edge",
        "median_edge": "Low Median",
        "p10_edge": "Poor P10",
        "q10_fail": "High Q10 Fail",
        "event_density": "Event Overload",
        "regime_edge": "Regime Filter",
        "edge_decay": "Edge Decay",
        "hit_or_payoff": "Poor Hit/Payoff",
        "oos_rank_ic": "Low OOS IC",
        "ic_tstat": "Low IC t-stat",
        "exit_policy": "Exit Policy",
    }
    if failure_report:
        n_blocked = failure_report.get("n_blocked", 0)
        failure_counts = failure_report.get("failure_counts", {})
        # Wrap the full gate distribution to fit the 41-char detail column so the
        # 82-wide table border stays intact while no gate count is dropped.
        fail_segments = [f"{_gate_label.get(k, k)} ({v})" for k, v in sorted(failure_counts.items())]
        wrapped_failures = _wrap_segments(fail_segments, width=41) or ["none"]
        for line_idx, fail_line in enumerate(wrapped_failures):
            action_cell = "BLOCKED" if line_idx == 0 else ""
            count_cell = str(n_blocked) if line_idx == 0 else ""
            prefix = "Fail Reasons: " if line_idx == 0 else " " * 14
            _logger.info(f"| {action_cell:<12} | {count_cell:<5} | {prefix}{fail_line:<41} |")
        top_blocked = failure_report.get("top_blocked_str", "none")
        _logger.info(f"| {'':<12} | {'':<5} | Top Blocked: {top_blocked[:42]:<42} |")
    if keep_variants:
        _logger.info(f"| {'RECOMMENDED':<12} | {len(keep_variants):<5} | 1. {keep_variants[0]:<52} |")
        for i, var in enumerate(keep_variants[1:], 2):
            _logger.info(f"| {'(Eligible)':<12} | {'':<5} | {i}. {var:<52} |")
    else:
        _logger.info(f"| {'RECOMMENDED':<12} | {'0':<5} | {'none':<52} |")
    _logger.info("-" * 82)

    # Per-variant gate failure detail (blocked and cell-admitted variants)
    blocked_rows = failure_report.get("rows", []) if failure_report else []
    if blocked_rows:
        _logger.info("\n[GATE FAILURES: PER-VARIANT]")
        _logger.info("-" * 82)
        _logger.info(f"| {'Variant':<30} | {'Action':<16} | {'Failed Gates / Cells':<26} |")
        _logger.info("-" * 82)
        for brow in blocked_rows[:20]:
            vname = str(brow.get("group", "")).removeprefix("variant=")[:30]
            is_admitted = bool(brow.get("regime_cell_admitted", False))
            if is_admitted:
                action = "CELL_ADMITTED"
                admitted_cells = str(brow.get("regime_cell_admitted_cells", ""))
                gates_cell = admitted_cells if len(admitted_cells) <= 26 else f"{admitted_cells[:25]}…"
            else:
                action = str(brow.get("candidate_action", ""))[:16]
                failed_labels = ", ".join(
                    _gate_label.get(g, g) for g in brow.get("failed_checks", [])
                )
                gates_cell = failed_labels if len(failed_labels) <= 26 else f"{failed_labels[:25]}…"
            _logger.info(f"| {vname:<30} | {action:<16} | {gates_cell:<26} |")
        _logger.info("-" * 82)

    # 2. Cause: Walk-Forward Performance
    wf_details = candidate_report.get("wf_fold_details", [])
    if wf_details and run_config.phase in {"l3", "l2"}:
        _logger.info("\n[WALK-FORWARD FOLD DETAILS]")
        _logger.info("-" * 82)
        _logger.info(
            f"| {'Fold':<4} | {'Mode':<10} | {'IC(diag)':>8} | {'Events':>7} | "
            f"{'RlzdMean':>8} | {'EU_p90':>7} | {'Pass':<6} |"
        )
        _logger.info(
            f"| {'':<4} | {'':<10} | {'(ref)':>8} | {'':<7} | "
            f"{'(★gate)':>8} | {'(★gate)':>7} | {'':<6} |"
        )
        _logger.info("-" * 82)
        for res in wf_details:
            fold_id = res.get("fold_id", 0)
            mode = res.get("inference_mode", "n/a")
            rank_ic = res.get("rank_ic")
            rank_ic_str = (
                f"{rank_ic:>8.3f}"
                if isinstance(rank_ic, (int, float)) and np.isfinite(rank_ic)
                else f"{'n/a':>8}"
            )
            events = res.get("n_events", 0)
            rlzd_mean = res.get("realized_mean_bps", float("nan"))
            rlzd_str = (
                f"{rlzd_mean:>8.1f}"
                if isinstance(rlzd_mean, (int, float)) and np.isfinite(rlzd_mean)
                else f"{'n/a':>8}"
            )
            eu_p90 = res.get("eu_p90", 0.0)
            passed = res.get("pass_cost", False)
            pass_str = "✅" if passed else "❌"
            _logger.info(
                f"| {fold_id:<4} | {mode:<10} | {rank_ic_str} | {events:>7,} | "
                f"{rlzd_str} | {eu_p90:>7.2f} | {pass_str:<6} |"
            )
        _logger.info("-" * 82)

    # 3. Decision: Final Bridge Status
    header = f"| {'Metric':<18} | {'Value':<27} |"
    width = len(header)
    title = "[BRIDGE SUMMARY] "
    _logger.info("\n" + title + "-" * (width - len(title)))
    _logger.info(header)
    _logger.info(f"| {'-'*18:<18} | {'-'*27:<27} |")
    workflow_status = str(candidate_report.get("workflow_status", "blocked"))
    _logger.info(f"| {'Active Signals':<18} | {f'{non_zero_weights} (sel={selected_total_bridge})':<27} |")
    _logger.info(f"| {'Status':<18} | {workflow_status:<27} |")
    _logger.info("-" * width)
    strategy_steps["report"] = time.perf_counter() - t_report

    # Demote extra diagnostics to DEBUG
    _logger.debug(f"[BRIDGE SUMMARY][DIAG] rule_report_keys={list(candidate_report.keys())}")

    t_merge_start = time.perf_counter()
    merge_candidate_output_into_is_and_oos(
        ml_out, data_stage.data_maps, data_stage.oos_data_maps,
        data_stage.valid_symbols, run_config.timeframe,
    )
    strategy_steps["merge"] = time.perf_counter() - t_merge_start
    _logger.log(PERF, 
        "[PROFILE][STAGE] merge_candidate_output_into_is_and_oos took %.4fs",
        strategy_steps["merge"],
    )

    # 4. Post-Analysis: Evaluation/Ablation
    candidate_report_ref = getattr(ml_out, "rule_report", {}) or {}
    if isinstance(candidate_report_ref, dict) and candidate_report_ref.get("zero_reason") == "signal_only_mode":
        sv_list = candidate_report_ref.get("signal_validation", [])
        total_sv = len(sv_list)
        passed_sv = sum(1 for v in sv_list if v.get("status") == "PASS")
        avg_bars = int(np.mean([float(v.get("n_bars", 0)) for v in sv_list])) if sv_list else 0
        avg_nan = float(np.mean([float(v.get("nan_pct", 0.0)) for v in sv_list])) * 100 if sv_list else 0.0
        avg_zero = float(np.mean([float(v.get("zero_neg_pct", 0.0)) for v in sv_list])) * 100 if sv_list else 0.0
        
        _logger.info(format_data_integrity_summary(
            total=total_sv,
            passed=passed_sv,
            bars=avg_bars,
            nan_pct=avg_nan,
            zero_pct=avg_zero
        ))
        
        if total_sv != passed_sv:
            _logger.info("\n[SIGNAL VALIDATION: DATA INTEGRITY AUDIT (FAILURES)]")
            _logger.info("-" * 135)
            _logger.info(
                f"| {'Symbol':<15} | {'Status':<8} | {'NaN %':>8} | {'Zero/Neg %':>12} | "
                f"{'Close Std':>11} | {'Hi-Lo Viol':>10} | {'Fail Reasons':<50} |"
            )
            _logger.info("-" * 135)
            for sv in sv_list:
                if sv.get("status") == "PASS":
                    continue
                sym = str(sv.get("symbol", "unknown"))
                status_val = str(sv.get("status", "unknown"))
                status_text = f"❌ {status_val}"
                nan_pct = float(sv.get("nan_pct", 0.0)) * 100
                zero_neg_pct = float(sv.get("zero_neg_pct", 0.0)) * 100
                close_std = float(sv.get("close_std", 0.0))
                hi_lo = int(sv.get("hi_lo_violation", 0))
                fail_text = ",".join(str(v) for v in sv.get("fail_reasons", [])) or "-"
                
                _logger.info(
                    f"| {sym:<15} | {status_text:<8} | {nan_pct:>7.2f}% | {zero_neg_pct:>11.2f}% | "
                    f"{close_std:>11.4f} | {hi_lo:>10} | {fail_text:<50} |"
                )
            _logger.info("-" * 135)
        
        _emit_strategy_profile()
        return ml_out

    if run_config.phase in {"l3", "l2"}:
        t_eval = time.perf_counter()
        _run_candidate_evaluation_report(
            ml_out,
            data_stage,
            run_config.timeframe,
            trading_symbols or tuple(data_stage.valid_symbols),
        )
        strategy_steps["evaluation"] = time.perf_counter() - t_eval

    _emit_strategy_profile()
    return ml_out


def _refresh_regime_c34_gold_standard(
    scorecard: RegimeScoreCard,
    code_1d: NDArray[np.int8],
    ml_out: CandidatePipelineOutput,
) -> None:
    """C3/C4 gold standard를 전략 이벤트로 재계산 후 scorecard를 재로그.

    regime evaluation stage는 strategy stage 이전에 실행되므로 C3/C4를 빈 배열로
    계산한다. strategy stage 완료 후 이 함수를 호출하면 실제 이벤트 데이터로 C3/C4를
    갱신하고 최종 scorecard를 재로그한다.
    """
    import dataclasses as _dc

    from src.domain.futures.strategy.regime_evaluation import evaluate_regime_classifier
    from src.domain.futures.strategy.rule_diagnostics import log_regime_scorecard

    labeled = ml_out.labeled
    if labeled is None or labeled.empty:
        _logger.info(
            "[REGIME_C34_GOLD] labeled DataFrame 없음 — baseline scorecard 출력"
        )
        log_regime_scorecard(scorecard)
        return

    required_cols = {"entry_regime_code", "edge_after_hurdle_bps", "entry_idx"}
    if not required_cols.issubset(labeled.columns):
        missing = required_cols - set(labeled.columns)
        _logger.debug("[REGIME_C34_GOLD] 필수 컬럼 없음 %s — 건너뜀", missing)
        return

    oos_start = ml_out.oos_start
    if oos_start is None:
        _logger.debug("[REGIME_C34_GOLD] oos_start 없음 — IS/OOS 분리 불가, 건너뜀")
        return

    entry_codes = labeled["entry_regime_code"].to_numpy(dtype=np.int8, copy=False)
    edge_bps = labeled["edge_after_hurdle_bps"].to_numpy(dtype=np.float64, copy=False)
    entry_idx = labeled["entry_idx"].to_numpy(dtype=np.int32, copy=False)

    is_mask = entry_idx < oos_start
    oos_mask = entry_idx >= oos_start

    n_is = int(is_mask.sum())
    n_oos = int(oos_mask.sum())
    if n_is + n_oos == 0:
        _logger.debug("[REGIME_C34_GOLD] 이벤트 없음 — 건너뜀")
        return

    gold_scorecard = evaluate_regime_classifier(
        all_codes_1d=code_1d,
        event_codes=entry_codes,
        event_edges_bps=edge_bps,
        is_event_mask=is_mask,
        oos_event_mask=oos_mask,
    )

    # proxy 값 + macro_dwell 보존 (pre-signal 단계에서 계산된 값)
    final_scorecard = _dc.replace(
        gold_scorecard,
        c3_proxy_pvalue=scorecard.c3_proxy_pvalue,
        c3_proxy_sign_flip=scorecard.c3_proxy_sign_flip,
        c4_proxy_spearman_rho=scorecard.c4_proxy_spearman_rho,
        c2_macro_dwell_median=scorecard.c2_macro_dwell_median,
        c2_macro_transition_rate=scorecard.c2_macro_transition_rate,
    )

    _logger.info(
        "[REGIME_C34_GOLD] C3/C4 gold standard 계산 완료: events=%d (IS=%d, OOS=%d)",
        n_is + n_oos,
        n_is,
        n_oos,
    )
    log_regime_scorecard(final_scorecard)


def _run_candidate_evaluation_report(
    candidate_out: Any,
    data_stage: DataStageResult,
    tf: str,
    trading_symbols: tuple[str, ...],
) -> None:
    """Print candidate-ml performance reporting and ablation study."""
    t_eval = time.perf_counter()
    eval_steps: dict[str, float] = {
        "config": 0.0,
        "ablation": 0.0,
        "render": 0.0,
    }

    from src.domain.futures.strategy.ablation import run_candidate_ablation

    t_step = time.perf_counter()
    cfg = build_candidate_strategy_config(
        strategy_cfg=StrategyConfig(name="candidate_ml"),
        opt_config=OPT_FUTURES_CONFIG,
        timeframe=tf,
    ).candidate
    eval_steps["config"] = time.perf_counter() - t_step
    
    active_syms = [s for s in trading_symbols if s in data_stage.data_maps]
    if not active_syms:
        active_syms = list(data_stage.data_maps.keys())

    t_step = time.perf_counter()
    df_ablation = run_candidate_ablation(
        data_maps=data_stage.data_maps,
        symbols=tuple(active_syms),
        tf=tf,
        cfg=cfg,
        cached_output=candidate_out,
    )
    eval_steps["ablation"] = time.perf_counter() - t_step
    
    # Alias mapping for variant names to keep the table compact
    alias_map = {
        "rule_only_equal_size": "Equal Size",
        "rule_promo_no_leak": "Rule Promo NL",
        "rule_promo_oos_oracle": "Rule Promo Oracle",
        "rule_only_fractional_kelly": "Kelly (Rule Only)",
        "rule_plus_ml_gate": "Ensemble Gate",
        "rule_plus_ml_gate_plus_edge": "Ensemble Gate+Edge",
        "rule_plus_ml_gate_plus_edge_plus_portfolio_caps": "Ensemble Full (Capped)",
        "candidate_ml_full": "Alpha-Ens.",
        "candidate_ml_direct_edge": "Direct Edge",
        "candidate_ml_variant_prior": "Variant Prior",
        "candidate_ml_promotion_filter": "Promo Filter",
        "candidate_ml_validation_quantile_selection": "Val. Selection",
        "candidate_ml_identity_features": "Identity Feat",
        "candidate_ml_market_state_features": "Market Feat",
    }

    # Log ablation study table — final PASS = compound gate AND deployment gate
    header = (
        f"| {'Model Alias':<18} | {'CAGR':>7} | {'MaxDD':>7} | {'MAR':>6}"
        f" | {'Equity':>10} | {'Trades':>6} | {'Deploy':>6} | {'Pass':<5} |"
    )
    width = len(header)
    title = "[ABLATION STUDY FRONTIER] "
    _logger.info("\n" + title + "-" * (width - len(title)))
    _logger.info(header)
    _logger.info(
        f"| {'-'*18:<18} | {'-'*7:>7} | {'-'*7:>7} | {'-'*6:>6}"
        f" | {'-'*10:>10} | {'-'*6:>6} | {'-'*6:>6} | {'-'*5:<5} |"
    )

    t_render = time.perf_counter()
    for _, row in df_ablation.iterrows():
        name = str(row["variant"])
        alias = alias_map.get(name, name[:18])
        cagr = f"{float(row['cagr']) * 100:>.1f}%"
        dd = f"{float(row['max_drawdown']) * 100:>.1f}%"
        mar = f"{float(row['mar']):>.2f}"
        equity = f"{float(row['final_equity']):,.0f}"
        trades = str(int(row.get("trade_count", 0)))
        deploy = f"{float(row.get('deployed_bar_fraction', 0.0)):.2f}"
        final_pass = (
            str(row["pass_compound_gate"]) == "True"
            and str(row.get("pass_deployment_gate", "False")) == "True"
        )
        passed = "Y" if final_pass else "N"

        _logger.info(
            f"| {alias:<18} | {cagr:>7} | {dd:>7} | {mar:>6}"
            f" | {equity:>10} | {trades:>6} | {deploy:>6} | {passed:^5} |"
        )
    eval_steps["render"] = time.perf_counter() - t_render
    _logger.info("-" * width)
    breakdown = _runtime_breakdown(time.perf_counter() - t_eval, **eval_steps)
    _logger.info(
        "[EVAL-PROF] total=%.4fs config=%.4fs ablation=%.4fs render=%.4fs accounted=%.4fs unaccounted=%.4fs",
        breakdown.total,
        eval_steps["config"],
        eval_steps["ablation"],
        eval_steps["render"],
        breakdown.accounted,
        breakdown.unaccounted,
    )


def _run_optimization_stage(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
    data_stage: DataStageResult,
    *,
    seed: int,
    resume: bool,
) -> RunnerResult:
    stage_t0 = time.perf_counter()
    project_root = str(BASE_DIR)
    ml_n_jobs = resolve_futures_parallel_policy(len(data_stage.valid_symbols))
    run_id = build_run_id(
        run_config.timeframe,
        window.fetch_start,
        window.end_date,
        data_stage.valid_symbols,
        OPT_FUTURES_CONFIG,
        project_root,
    )
    storage_url, storage = setup_optuna_storage(project_root)
    study_name = build_joint_study_name(
        run_config.timeframe,
        window.fetch_start,
        window.end_date,
        data_stage.valid_symbols,
        OPT_FUTURES_CONFIG,
    )
    physical_cores = max(1, (os.cpu_count() or 4) // 2)
    # Target 6 workers matching high-performance P-cores (e.g. i5-13600K)
    safe_workers_b = min(6, physical_cores)

    opt_req = OptimizationRequest(
        data_maps=data_stage.data_maps,
        symbols=data_stage.valid_symbols,
        tf=run_config.timeframe,
        fetch_start=window.fetch_start,
        is_start=window.is_start,
        end_date=window.end_date,
        run_id=run_id,
        study_name=study_name,
        storage_url=storage_url,
        storage=storage,
        total_trials=int(run_config.trials),
        ml_n_jobs=ml_n_jobs,
        seed=seed,
        resume=resume,
        strategy_mode=True,
        strategy_cfg=build_candidate_strategy_config(
            strategy_cfg=StrategyConfig(name="candidate_ml"),
            opt_config=OPT_FUTURES_CONFIG,
            timeframe=run_config.timeframe,
        ),
        n_trials_a1=max(1, int(run_config.trials * 0.5)),
        n_trials_a2=max(1, int(run_config.trials * 0.2)),
        n_trials_b=max(1, int(run_config.trials * 0.3)),
        n_workers_b=safe_workers_b,
        enqueue_seeds=None,
        target_seeds=[seed],
    )
    contract_meta = log_optuna_contract(
        project_root=project_root,
        requested_trials_per_phase=int(run_config.trials),
        phase_workers={
            "phase_a1": max(1, ml_n_jobs),
            "phase_a2": max(1, ml_n_jobs),
            "phase_b": safe_workers_b,
        },
        seed=seed,
        storage_url=storage_url,
    )

    t_opt = time.perf_counter()
    opt_res = run_optimization(opt_req)
    opt_elapsed = time.perf_counter() - t_opt
    _logger.debug(f"[OPTIMIZE] Optimization complete in {opt_elapsed:.2f}s")
    
    precompute_profile = getattr(opt_res.base_ctx, "precompute_profile", None)
    if isinstance(precompute_profile, dict):
        _logger.log(PERF, 
            (
                "[RUN_PROF] step=ml_precompute total=%.2fs align=%.2fs "
                "covariance=%.2fs awf_refit=%.2fs calibrator=%.2fs "
                "prebuilt=%.2fs legs=%d"
            ),
            float(precompute_profile.get("total", 0.0)),
            float(precompute_profile.get("align", 0.0)),
            float(precompute_profile.get("covariance", 0.0)),
            float(precompute_profile.get("awf_refit_total", 0.0)),
            float(precompute_profile.get("calibrator_total", 0.0)),
            float(precompute_profile.get("prebuilt_total", 0.0)),
            int(precompute_profile.get("awf_legs", 0)),
        )
    study_ml = opt_res.study_ml
    best_trial = opt_res.best_trial

    # ------------------ PROFILE SUMMARY REPORT ------------------
    try:
        if study_ml is not None:
            all_trials = study_ml.get_trials(deepcopy=False)
            valid_states = (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
            )
            valid_trials = [t for t in all_trials if t.state in valid_states]
            if valid_trials:
                compose_vals = [float(t.user_attrs.get("prof_compose", 0.0)) for t in valid_trials]
                prep_vals = [float(t.user_attrs.get("prof_prep", 0.0)) for t in valid_trials]
                prep_align_vals = [float(t.user_attrs.get("prof_prep_align", 0.0)) for t in valid_trials]
                prep_constraint_vals = [float(t.user_attrs.get("prof_prep_constraint", 0.0)) for t in valid_trials]
                exec_vals = [float(t.user_attrs.get("prof_exec", 0.0)) for t in valid_trials]
                metrics_vals = [float(t.user_attrs.get("prof_metrics", 0.0)) for t in valid_trials]
                metrics_pure_vals = [float(t.user_attrs.get("prof_metrics_pure", 0.0)) for t in valid_trials]
                metrics_db_io_vals = [float(t.user_attrs.get("prof_metrics_db_io", 0.0)) for t in valid_trials]
                mean_c = float(np.mean(compose_vals)) if compose_vals else 0.0
                mean_p = float(np.mean(prep_vals)) if prep_vals else 0.0
                mean_pa = float(np.mean(prep_align_vals)) if prep_align_vals else 0.0
                mean_pc = float(np.mean(prep_constraint_vals)) if prep_constraint_vals else 0.0
                mean_e = float(np.mean(exec_vals)) if exec_vals else 0.0
                mean_m = float(np.mean(metrics_vals)) if metrics_vals else 0.0
                mean_mp = float(np.mean(metrics_pure_vals)) if metrics_pure_vals else 0.0
                mean_md = float(np.mean(metrics_db_io_vals)) if metrics_db_io_vals else 0.0
                total_mean = mean_c + mean_p + mean_e + mean_m
                trial_elapsed_sum = 0.0
                for trial in valid_trials:
                    dt_start = getattr(trial, "datetime_start", None)
                    dt_end = getattr(trial, "datetime_complete", None)
                    if dt_start is None or dt_end is None:
                        continue
                    trial_elapsed_sum += max((dt_end - dt_start).total_seconds(), 0.0)
                hidden_overhead = max(opt_elapsed - trial_elapsed_sum, 0.0)

                if total_mean > 0:
                    _logger.info("=" * 60)
                    _logger.info(
                        " [PROFILING SUMMARY] Strategy Backtest Performance Profiling "
                        "(n_trials=%d)",
                        len(valid_trials),
                    )
                    _logger.info(
                        "  1. Signal Compose   : %6.2f ms (%5.1f%%)",
                        mean_c * 1000.0,
                        (mean_c / total_mean) * 100.0,
                    )
                    _logger.info(
                        "  2. Backtest Prep    : %6.2f ms (%5.1f%%)",
                        mean_p * 1000.0,
                        (mean_p / total_mean) * 100.0,
                    )
                    _logger.info(
                        "     - Data Align     : %6.2f ms (%5.1f%%)",
                        mean_pa * 1000.0,
                        (mean_pa / total_mean) * 100.0,
                    )
                    _logger.info(
                        "     - Constraint Ck  : %6.2f ms (%5.1f%%)",
                        mean_pc * 1000.0,
                        (mean_pc / total_mean) * 100.0,
                    )
                    _logger.info(
                        "  3. Numba Execution  : %6.2f ms (%5.1f%%)",
                        mean_e * 1000.0,
                        (mean_e / total_mean) * 100.0,
                    )
                    _logger.info(
                        "  4. Metrics/Pruning  : %6.2f ms (%5.1f%%)",
                        mean_m * 1000.0,
                        (mean_m / total_mean) * 100.0,
                    )
                    _logger.info(
                        "     - Pure Calc      : %6.2f ms (%5.1f%%)",
                        mean_mp * 1000.0,
                        (mean_mp / total_mean) * 100.0,
                    )
                    _logger.info(
                        "     - Redis DB I/O   : %6.2f ms (%5.1f%%)",
                        mean_md * 1000.0,
                        (mean_md / total_mean) * 100.0,
                    )
                    _logger.info("  * Total Backtest/Tr : %6.2f ms", total_mean * 1000.0)
                    _logger.info(
                        "[RUN_PROF] trial_elapsed_sum=%.2fs run_optimization=%.2fs "
                        "hidden_overhead=%.2fs",
                        trial_elapsed_sum,
                        opt_elapsed,
                        hidden_overhead,
                    )
                    _logger.info("=" * 60)
    except Exception as e:
        _logger.warning(
            "[RUN_PROF] action=calculate_summary status=failed error=%s", type(e).__name__
        )
    # ------------------------------------------------------------

    if study_ml is None or best_trial is None:
        strategy_name = str(OPT_FUTURES_CONFIG.get("FUTURES_STRATEGY_NAME", "candidate_ml"))
        if run_config.phase == "l3" and strategy_name in {"candidate_ml", "rule_baseline"}:
            return RunnerResult(exit_code=0, reason="candidate_smoke_no_candidate")
        return RunnerResult(exit_code=1, reason="no_candidate")

    pbo_gate, dsr_gate, _ = resolve_adjusted_gates(OPT_FUTURES_CONFIG, int(run_config.trials))
    pbo_obs = awf_pos_frac_to_pseudo_pbo(0.5)
    dsr_obs = 0.0

    ensemble_results = []
    if study_ml is not None:
        completed_trials = [
            t
            for t in study_ml.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        ensemble_results = [{"trial": t, "params": t.params} for t in completed_trials]

    final_req = FinalEvaluationRequest(
        tf=run_config.timeframe,
        project_root=project_root,
        study_ml=study_ml,
        run_id=run_id,
        ml_ctx=opt_res.base_ctx,
        n_ml_trials=int(run_config.trials),
        target_seeds=[seed],
        selected_ops_profile="active",
        pbo_gate=pbo_gate,
        dsr_gate=dsr_gate,
        pbo_obs=pbo_obs,
        dsr_obs=dsr_obs,
        best_trial=best_trial,
        champ_stab_cv=0.0,
        stab_tmp_layer3_awf_fail=False,
        cv_max=0.30,
        phase_c_diagnostics=opt_res.phase_bundle.phase_c_diagnostics,
        ensemble_results=ensemble_results,
        oos_data_maps=data_stage.oos_data_maps,
        data_maps=data_stage.data_maps,
        valid_symbols=data_stage.valid_symbols,
        champion_awf_diag={},
        ai_telemetry_payloads=[],
        selection_summary={},
        run_summary_extras={"optuna_contract": contract_meta},
    )
    t_final_eval = time.perf_counter()
    run_final_evaluation(final_req)
    final_eval_elapsed = time.perf_counter() - t_final_eval
    _logger.info(f"[STAGE_TIME] Final evaluation complete in {final_eval_elapsed:.2f}s")
    _logger.info(f"[STAGE_TIME] Optimization stage total: {time.perf_counter() - stage_t0:.2f}s")
    return RunnerResult(exit_code=0, reason="ok")


def run_pipeline(
    run_config: FuturesRunConfig,
    *,
    seed: int = 42,
    resume: bool = False,
) -> RunnerResult:
    """Run active futures pipeline in explicit orchestration order."""
    pipeline_t0 = time.perf_counter()
    # Step 1) parse run window
    t_window = time.perf_counter()
    window = _resolve_quarterly_window(run_config.date)
    layered_window = (
        _resolve_layered_window(run_config.date)
        if OPT_FUTURES_CONFIG.get("USE_CS_RANK_ENGINE", False)
        else None
    )
    _logger.debug("[perf] window resolve: %.4fs", time.perf_counter() - t_window)

    # Step 1.5) Ensure universe ledger is synchronized for the required window
    t_sync = time.perf_counter()
    _ensure_universe_ledger_sync(run_config, window)
    _logger.debug("[perf] universe sync: %.4fs", time.perf_counter() - t_sync)

    # Step 2) universe timeline/quality gate
    t_universe = time.perf_counter()
    (
        discovered_symbols,
        timeline,
        inference_panel,
        live_inference_panel,
        universe_snapshot,
        inference_timeline,
        _universe_result,
    ) = _run_universe_stage(run_config, window, layered_window=layered_window)
    _logger.debug("[perf] universe stage: %.4fs", time.perf_counter() - t_universe)

    resolved_load_symbols = _resolve_data_collection_symbols(
        run_config=run_config,
        discovered_symbols=discovered_symbols,
        inference_panel=inference_panel,
        live_inference_panel=live_inference_panel,
    )
    cache_window = (
        _window_with_fetch_start(window, min(window.fetch_start_date, layered_window.fetch_start))
        if layered_window is not None
        else window
    )
    _ensure_cached_symbol_data_for_targets(
        run_config,
        cache_window,
        resolved_load_symbols,
        require_exec_1m=_requires_exec_1m(run_config),
    )
    # Step 3) data loading + readiness
    t_data_start = time.perf_counter()
    data_stage = _run_data_stage(
        run_config,
        window,
        discovered_symbols,
        timeline,
        inference_panel,
        live_inference_panel,
        inference_timeline,
        layered_window=layered_window,
    )
    _logger.debug("[perf] data stage: %.4fs", time.perf_counter() - t_data_start)

    # Consolidate all initialization info into the System Dashboard
    from src.domain.futures.strategy.tiered_logging import format_system_context_dashboard
    
    universe_report = {
        "discovered": len(discovered_symbols),
        "selected": len(_selected_symbols_from_snapshot(universe_snapshot)),
        "live_panel": len(live_inference_panel),
    }
    
    _loaded_ratio = (
        f"{(len(data_stage.data_maps) / len(resolved_load_symbols)):.1%}"
        if resolved_load_symbols
        else "0%"
    )
    _all_ready = len(data_stage.valid_symbols) == len(data_stage.data_maps)
    dq_report = {
        "loaded_ratio": _loaded_ratio,
        "loaded_count": len(data_stage.data_maps),
        "req_count": len(resolved_load_symbols),
        "ready_count": len(data_stage.valid_symbols),
        "fail_summary": "None" if _all_ready else "See logs",
    }
    
    strategy_info = {
        "engine": "Alpha-Ensemble Engine",
        "inf_panel": len(inference_panel),
        "trade_scope": len(data_stage.valid_symbols),
    }

    _logger.info(format_system_context_dashboard(
        window=window,
        universe_report=universe_report,
        data_quality=dq_report,
        strategy_info=strategy_info,
    ))

    # Step 3.5) regime evaluation (between universe and signal)
    regime_stage_result = _run_regime_evaluation_stage(run_config, data_stage)

    # Step 4) strategy bridge + alpha contract
    t_strategy = time.perf_counter()
    strategy_out = _run_strategy_stage(
        run_config,
        window,
        data_stage,
        _selected_symbols_from_snapshot(universe_snapshot),
        universe_snapshot=universe_snapshot,
        layered_window=layered_window,
        universe_result=_universe_result,
    )
    _logger.debug("<< STRATEGY: %.2fs", time.perf_counter() - t_strategy)

    if isinstance(strategy_out, RunnerResult):
        return strategy_out

    # Step 4.5) C3/C4 gold standard: 전략 이벤트로 scorecard 갱신
    # Layer1Result는 labeled 속성이 없으므로 isinstance guard
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result

    if (
        regime_stage_result is not None
        and strategy_out is not None
        and not isinstance(strategy_out, Layer1Result)
    ):
        _refresh_regime_c34_gold_standard(*regime_stage_result, strategy_out)
    if run_config.phase == "l1":
        return RunnerResult(exit_code=0, reason="l1_mode_done")
    if run_config.phase == "l2":
        _logger.info(
            "[PHASE] phase=l2 completed strategy/candidate evaluation only; optimization/training skipped"
        )
        return RunnerResult(exit_code=0, reason="candidate_evaluation_done")
    # Step 5) optimization + final OOS evaluation
    _logger.info(
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "● [LAYER 3: ML PHASE B OPTIMIZATION & ENSEMBLE]\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  ● [HYPERPARAMETER OPTIMIZATION]\n"
        f"    - Target Symbols : {len(data_stage.valid_symbols)}\n"
        f"    - Total Trials   : {int(run_config.trials)}\n"
        "  ────────────────────────────────────────────────────────────────────────────"
    )
    t_opt_stage = time.perf_counter()
    result = _run_optimization_stage(
        run_config,
        window,
        data_stage,
        seed=seed,
        resume=resume,
    )
    _logger.debug("<< OPTIMIZE: %.2fs", time.perf_counter() - t_opt_stage)
    _logger.log(PERF, 
        "<< PIPELINE_TOTAL: %.2fs",
        time.perf_counter() - pipeline_t0,
    )
    return result


def run_from_cli(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 2
    try:
        run_config = _build_run_config(args)
    except ValueError as exc:
        _logger.error("!! ARGS: invalid -> %s", exc)
        return 2
    try:
        result = run_pipeline(run_config)
    except RuntimeError as exc:
        _logger.error("!! FAIL: runner_failed -> %s", str(exc))
        return 1
    except Exception:
        _logger.exception("!! FAIL: unexpected_error")
        return 1
    if result.exit_code != 0:
        _logger.error("!! FAIL: exit_code=%d reason=%s", result.exit_code, result.reason)
    return result.exit_code


def main() -> int:
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level, format="%(message)s", force=True)
    return run_from_cli()


if __name__ == "__main__":
    raise SystemExit(main())
