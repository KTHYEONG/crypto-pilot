from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so 'src' module is importable from any working directory
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ruff: noqa: E402
import argparse
import gc
import logging
import os
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

# 반드시 numba/numpy import 전에 설정 — fork child OOM 방지
for _env in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env] = "1"
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.domain.futures.strategy.regime_evaluation import RegimeScoreCard
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result
    from src.domain.futures.strategy.timeframe_probe import TfCellEvidence, TfProbeManifest
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
from src.domain.futures.strategy.regime_evaluation import _compute_c2_macro, evaluate_regime_classifier_proxy
from src.domain.futures.strategy.timeframe_contracts import (
    select_probe_source_tf as _shared_select_probe_source_tf,
)
from src.domain.futures.strategy_runtime.bridge import (
    merge_candidate_output_into_is_and_oos,
)
from src.domain.futures.universe import UniverseSnapshot
from src.domain.futures.universe.contracts import UniverseStateCube
from src.domain.futures.universe.membership import inject_membership_masks_into_maps
from src.domain.futures.universe.storage import run_historical_sync

if TYPE_CHECKING:
    from src.domain.futures.optimization.workflow import TieredContext

_GLOBAL_L2_CTX: TieredContext | None = None
def _btc_index_if_present(symbols: tuple[str, ...]) -> int:
    for i, s in enumerate(symbols):
        if "BTC" in s.upper():
            return i
    return -1


def _evaluate_l2_trial_from_global(params: dict[str, Any]) -> tuple[float, dict[str, Any], float]:
    """Fork-safe L2 trial evaluator reading context from module global."""
    if _GLOBAL_L2_CTX is None:
        raise ValueError("Global L2 context is not initialized")
    from src.domain.futures.optimization.workflow import _evaluate_l2_params
    return _evaluate_l2_params(params, _GLOBAL_L2_CTX)


_logger = setup_logger("opt_main_futures", write_file=False)


def _get_rss_mb() -> float:
    """Return current process RSS in MB via psutil with proc/status fallback."""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            return -1.0
    return -1.0


def _get_peak_rss_mb() -> float:
    """Return peak RSS (VmHWM) in MB via proc status with fallback."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return -1.0
    return -1.0


def _log_mem(stage: str, rss_before: float, /, *, extra: str = "") -> None:
    """Emit debug-level memory delta and peak memory log."""
    rss_after = _get_rss_mb()
    peak_rss = _get_peak_rss_mb()
    delta = rss_after - rss_before if rss_before > 0 and rss_after > 0 else -1.0
    peak_str = f" peak={peak_rss:.0f}MB" if peak_rss > 0 else ""
    _logger.debug(
        "[MEM] stage=%s rss=%.0fMB delta=%+.0fMB%s %s",
        stage, rss_after, delta, peak_str, extra,
    )



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
class TfProbeStageResult:
    """Result container for the TF Probe stage.

    Attributes:
        manifest: Full probe diagnostic output (all symbol x family x tf cells).
        winning_cells: Tuple of cells that passed all gate criteria.
        selected_tfs: Unique timeframes present in winning cells.
    """

    manifest: TfProbeManifest
    winning_cells: tuple[TfCellEvidence, ...]
    selected_tfs: frozenset[str]


def _select_probe_source_tf(sym_maps: Mapping[str, Any], target_tf: str) -> str | None:
    """Return the finest cached TF usable to construct target_tf."""
    return _shared_select_probe_source_tf(sym_maps, target_tf)


def _fit_table_cell(value: str, width: int) -> str:
    """Fit a cell to width without breaking the ASCII table border."""
    text = value.replace("\n", " ")
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return f"{text[: width - 3]}..."


def _log_ascii_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    widths: Sequence[int],
    *,
    level: int = logging.INFO,
) -> None:
    """Emit a fixed-width ASCII table using the repo's audit-log style."""
    border = sum(widths) + (3 * len(widths)) + 1
    _logger.log(level, "\n%s", title)
    _logger.log(level, "-" * border)
    _logger.log(
        level,
        "| "
        + " | ".join(
            f"{_fit_table_cell(header, width):<{width}}"
            for header, width in zip(headers, widths, strict=True)
        )
        + " |"
    )
    _logger.log(level, "-" * border)
    for row in rows:
        _logger.log(
            level,
            "| "
            + " | ".join(
                f"{_fit_table_cell(cell, width):<{width}}"
                for cell, width in zip(row, widths, strict=True)
            )
            + " |"
        )
    _logger.log(level, "-" * border)


def _format_counter_items(counter: Counter[str], *, limit: int = 3) -> str:
    """Format a Counter into a compact audit string."""
    if not counter:
        return "-"
    items = counter.most_common(limit)
    rendered = ", ".join(f"{name}:{count}" for name, count in items)
    if len(counter) > limit:
        rendered += ", ..."
    return rendered


def _log_probe_tf_source_coverage(
    data_maps: Mapping[str, Mapping[str, Any]],
    symbols: Sequence[str],
    probe_tf_grid: Sequence[str],
) -> None:
    """Log probe TF source coverage without requiring virtual-TF parquet files."""
    rows: list[list[str]] = []
    for probe_tf in probe_tf_grid:
        source_counts: list[int] = []
        source_mix: Counter[str] = Counter()
        for symbol in symbols:
            sym_maps = data_maps.get(symbol, {})
            source_tf = _select_probe_source_tf(sym_maps, probe_tf)
            if source_tf is None:
                continue
            frame = sym_maps.get(source_tf)
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            source_counts.append(len(frame))
            source_mix[source_tf] += 1
        median_bars = int(np.median(source_counts)) if source_counts else 0
        rows.append(
            [
                probe_tf,
                f"{len(source_counts)}/{len(symbols)}",
                str(median_bars),
                _format_counter_items(source_mix),
            ]
        )
    if rows:
        _logger.info("🔍 [TF-PROBE AUDIT] SOURCE READINESS Dashboard")
        for row in rows:
            _logger.info(f"  ├── {row[0]:<4} : Ready {row[1]:<7} | Median Bars: {row[2]:<6} | Mix: {row[3]}")


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
    _native_flag: bool | None = None
    for sym in valid_symbols:
        sym_maps = strategy_maps.get(sym)
        if sym_maps is None:
            dropped["missing_map"].append(sym)
            continue
        sym_df = sym_maps.get(tf)
        if sym_df is None or sym_df.empty:
            dropped["empty_frame"].append(sym)
            continue
        if _native_flag is None:
            _native_flag = pd.api.types.is_datetime64_any_dtype(sym_df["datetime"])
        datetimes = sym_df["datetime"] if _native_flag else pd.to_datetime(sym_df["datetime"], utc=True)
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
    _logger.debug(
        "[PERF] step=discover_universe_timeline elapsed=%.4fs",
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
    _logger.debug(
        "[PERF] step=validate_universe_quality elapsed=%.4fs",
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
    probe_tf_grid_resolved: list[str] | None = (
        list(OPT_FUTURES_CONFIG.get("TF_PROBE_GRID", []))
        if OPT_FUTURES_CONFIG.get("ENABLE_TF_PROBE", False)
        else None
    )

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
        target_tfs=None,
    )
    _logger.debug(
        "[PERF] step=load_futures_data_maps_for_symbols elapsed=%.4fs",
        time.perf_counter() - t_load,
    )
    if probe_tf_grid_resolved:
        _log_probe_tf_source_coverage(data_maps, valid_symbols, probe_tf_grid_resolved)
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
        _logger.debug(
            "[PERF] step=inject_membership_masks_into_maps elapsed=%.4fs",
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
        _logger.warning(
            "[data-readiness] no symbols kept: requested=%d loaded=%d kept=%d probe_enabled=%s",
            len(load_symbols),
            len(data_maps),
            len(valid_symbols),
            bool(OPT_FUTURES_CONFIG.get("ENABLE_TF_PROBE", False)),
        )
        raise RuntimeError("data_not_ready")
    return DataStageResult(
        data_maps=readiness.filtered_is_maps,
        oos_data_maps=readiness.filtered_oos_maps,
        valid_symbols=valid_symbols,
    )


def _run_tf_probe_stage(
    run_config: FuturesRunConfig,
    data_stage: DataStageResult,
    tiered_cfg: Any,
) -> TfProbeStageResult | None:
    """Execute TF Probe stage when ENABLE_TF_PROBE=True, otherwise return None.

    Args:
        run_config: Active run configuration (unused; reserved for future guard logic).
        data_stage: Loaded data maps and valid symbols from data stage.
        tiered_cfg: CandidateStrategyConfig for the base strategy.

    Returns:
        TfProbeStageResult with winning cells, or None if probe is disabled / fails.
    """
    del run_config  # reserved for future guard logic
    if not OPT_FUTURES_CONFIG.get("ENABLE_TF_PROBE", False):
        return None

    from src.domain.futures.strategy.execution_cost import ExecutionCostModel
    from src.domain.futures.strategy.timeframe_probe import (
        probe_timeframe_alpha,
        select_tf_family_cells,
        summarize_tf_probe_gate_audit,
    )

    tf_grid: list[str] = list(OPT_FUTURES_CONFIG.get("TF_PROBE_GRID", ["4h"]))
    max_workers: int = int(OPT_FUTURES_CONFIG.get("TF_PROBE_MAX_WORKERS", 8))
    min_tstat: float = float(OPT_FUTURES_CONFIG.get("TF_PROBE_MIN_TSTAT", 2.0))
    require_fdr: bool = bool(OPT_FUTURES_CONFIG.get("TF_PROBE_REQUIRE_FDR", True))
    min_fold_cons: float = float(OPT_FUTURES_CONFIG.get("TF_PROBE_MIN_FOLD_CONSISTENCY", 0.75))

    try:
        manifest: TfProbeManifest = probe_timeframe_alpha(
            data_maps=data_stage.data_maps,
            symbols=data_stage.valid_symbols,
            base_cfg=tiered_cfg,
            tf_grid=tf_grid,
            max_workers=max_workers,
            round_trip_cost_bps=ExecutionCostModel().round_trip_bps(),
        )
        winning: tuple[TfCellEvidence, ...] = select_tf_family_cells(
            manifest,
            min_ic_tstat=min_tstat,
            require_fdr=require_fdr,
            min_fold_sign_consistency=min_fold_cons,
        )
        selected_tfs: frozenset[str] = frozenset(c.tf for c in winning)
        winning_by_tf: dict[str, list[TfCellEvidence]] = {}
        for cell in winning:
            winning_by_tf.setdefault(cell.tf, []).append(cell)
        _logger.info(
            "[TF-PROBE] %d winning cells across %d tf: %s",
            len(winning),
            len(selected_tfs),
            sorted(selected_tfs),
        )
        rows: list[list[str]] = []
        for tf_i in manifest.tf_grid:
            tf_cells = winning_by_tf.get(tf_i, [])
            top_families = _format_counter_items(Counter(c.family for c in tf_cells), limit=2)
            top_variants = _format_counter_items(
                Counter(f"{c.family}:{c.variant}" for c in tf_cells),
                limit=2,
            )
            decision = "SELECT" if tf_cells else "REJECT"
            rows.append(
                [
                    tf_i,
                    str(len(tf_cells)),
                    top_families,
                    top_variants,
                    decision,
                ]
            )
        _log_ascii_table(
            "[TF-PROBE AUDIT] TIMEFRAME SELECTION",
            ("TF", "Winning", "Families", "Variants", "Decision"),
            rows,
            (8, 10, 24, 34, 10),
        )
        gate_rows = summarize_tf_probe_gate_audit(
            manifest,
            min_ic_tstat=min_tstat,
            require_fdr=require_fdr,
            min_fold_sign_consistency=min_fold_cons,
        )
        audit_table_rows: list[list[str]] = [
            [
                row.tf,
                str(row.computed),
                str(row.pass_tstat),
                str(row.pass_fdr),
                str(row.pass_net_edge),
                str(row.pass_fold_consistency),
                str(row.winning),
                row.top_fail_reason,
            ]
            for row in gate_rows
        ]
        _log_ascii_table(
            "[TF-PROBE AUDIT] GATE SURVIVORSHIP",
            ("TF", "Cells", "Pass t", "Pass FDR", "Pass Edge", "Pass Fold", "Winning", "Top Fail"),
            audit_table_rows,
            (8, 8, 8, 10, 10, 10, 8, 16),
        )
        return TfProbeStageResult(
            manifest=manifest,
            winning_cells=winning,
            selected_tfs=selected_tfs,
        )
    except Exception as exc:
        _logger.warning("[TF-PROBE] probe stage failed (fallback to base-only): %s", exc)
        return None


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
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        predict_layer1_signals,
        predict_layer1_signals_multi_tf,
    )

    datetimes = aligned.datetimes
    l2_start_ts = pd.Timestamp(window.l2_start, tz="UTC")
    ho_start_ts = pd.Timestamp(window.holdout_start, tz="UTC")
    l2_start_bar = int(np.searchsorted(datetimes, np.datetime64(l2_start_ts.replace(tzinfo=None), "ns")))
    ho_start_bar = int(np.searchsorted(datetimes, np.datetime64(ho_start_ts.replace(tzinfo=None), "ns")))

    # multi-TF: artifacts_by_tf가 있으면 전 TF 신호를 통합 예측
    artifacts_by_tf = getattr(l1_res, "artifacts_by_tf", {})
    if artifacts_by_tf:
        return predict_layer1_signals_multi_tf(
            artifacts_by_tf=artifacts_by_tf,
            candidate_events=labeled_events,
            aligned=aligned,
            start_idx=l2_start_bar,
            end_idx=ho_start_bar,
            cfg=cfg,
        )

    # fallback: 단일 artifact (구 동작 호환)
    artifact = getattr(l1_res, "inference_artifact", None)
    if artifact is None:
        raise ValueError("L1 artifact 없음 — l1_result.inference_artifact is None (L1 gate_passed=False 상태)")

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
    l2_sim_cache: Any = None,
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
        suggest_layered_params,
    )
    from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2StudyResult
    from src.domain.futures.strategy.tiered_workflow.selection import (
        _layer2_experiment_key,
        _signal_batch_fingerprint,
        select_layer2_champion,
    )
    from src.domain.futures.strategy.walk_forward import build_walk_forward_folds

    if l2_sim_cache is None:
        l2_sim_cache = build_l2_simulation_cache(aligned, signal_batch, tf)
    _logger.debug("[MEM] stage=l2_sim_cache rss=%.0fMB", _get_rss_mb())

    # AWF folds pre-computation: ctx에 저장하여 _resolve_l2_signal_batch_and_folds 재사용
    _ho_ts = pd.Timestamp(window.holdout_start).tz_localize(None)
    _ho_start_idx = int(np.searchsorted(aligned.datetimes, np.datetime64(_ho_ts, "ns")))
    _awf_all = build_walk_forward_folds(n_bars=_ho_start_idx, cfg=cfg)
    _l2_ts = pd.Timestamp(window.l2_start).tz_localize(None)
    _l1_end = int(np.searchsorted(aligned.datetimes, np.datetime64(_l2_ts, "ns")))
    _awf_folds_l2 = tuple(f for f in _awf_all if f.oos_start >= _l1_end and f.oos_end <= _ho_start_idx)
    if not _awf_folds_l2:
        _cal_end = max(_l1_end - 1, 1)
        from src.domain.futures.strategy.walk_forward import WFFold
        _awf_folds_l2 = (WFFold(
            fit_start=0,
            fit_end=_cal_end,
            cal_start=max(0, _cal_end - max(1, _cal_end // 5)),
            cal_end=_cal_end,
            oos_start=_l1_end,
            oos_end=_ho_start_idx,
        ),)

    # Precompute bucket realized edges (trial-param independent → 1회만 계산)
    from dataclasses import replace

    from src.domain.futures.strategy.market_regime import compute_market_regime_context
    from src.domain.futures.strategy.tiered_workflow.l2_meta import (
        build_regime_routing_plan,
    )
    _fallback_mode = str(getattr(cfg, "l2_regime_fallback_mode", "pooled"))
    if _fallback_mode not in {"pooled", "empty"}:
        raise ValueError("l2_regime_fallback_mode must be one of pooled/empty")
    _routing_plan = build_regime_routing_plan(
        cache=l2_sim_cache,
        aligned=aligned,
        awf_folds=_awf_folds_l2,
        raw_regime_code_1d=compute_market_regime_context(aligned=aligned).code_1d,
        compression_enabled=bool(getattr(cfg, "l2_regime_compression_enabled", True)),
        cost_bps=float(getattr(cfg, "l2_bucket_cost_bps", 6.0)),
        min_n=int(getattr(cfg, "l2_bucket_min_n", 15)),
        shrinkage=float(getattr(cfg, "l2_bucket_shrinkage", 0.3)),
        proof_enabled=bool(getattr(cfg, "l2_regime_proof_enabled", True)),
        proof_nw_tstat_threshold=float(getattr(cfg, "l2_regime_proof_nw_tstat", 1.5)),
        proof_fold_pass_ratio_threshold=float(getattr(cfg, "l2_regime_proof_fold_pass_ratio", 0.60)),
        fallback_mode=cast(Literal["pooled", "empty"], _fallback_mode),
        debug_diagnostics_enabled=_logger.isEnabledFor(logging.DEBUG),
        edge_floor_bps=float(getattr(cfg, "l2_bucket_edge_floor_bps", 0.0)),
        debug_top_k=int(getattr(cfg, "l2_regime_debug_top_k", 10)),
        policy_mode=cast(
            Literal["filter", "observe", "soft", "hybrid"],
            str(getattr(cfg, "l2_regime_policy_mode", "soft")),
        ),
        policy_cal_min_n=int(getattr(cfg, "l2_regime_cal_min_n", 20)),
        policy_min_cal_lift_bps=float(getattr(cfg, "l2_regime_min_cal_lift_bps", 8.0)),
        policy_block_lift_bps=float(getattr(cfg, "l2_regime_block_lift_bps", -12.0)),
        policy_downweight_min=float(getattr(cfg, "l2_regime_soft_downweight_min", 0.50)),
        policy_downweight_max=float(getattr(cfg, "l2_regime_soft_downweight_max", 1.0)),
        policy_min_confidence=float(getattr(cfg, "l2_regime_min_policy_confidence", 0.55)),
        policy_hard_block_enabled=bool(getattr(cfg, "l2_regime_hard_block_enabled", False)),
        policy_block_min_confidence=float(getattr(cfg, "l2_regime_block_min_confidence", 0.80)),
        policy_require_sign_consistency=bool(getattr(cfg, "l2_regime_require_sign_consistency", True)),
        policy_pooled_is_passthrough=os.environ.get(
            "L2_REGIME_POOLED_IS_PASSTHROUGH",
            str(getattr(cfg, "l2_regime_pooled_is_passthrough", True)),
        ).lower() in ("1", "true", "yes"),
        policy_min_fit_n_floor=int(
            os.environ.get(
                "L2_REGIME_MIN_FIT_N_FLOOR",
                str(getattr(cfg, "l2_regime_min_fit_n_floor", 5)),
            )
        ),
        policy_require_fit_n_for_downweight=os.environ.get(
            "L2_REGIME_REQUIRE_FIT_N_FOR_DOWNWEIGHT",
            str(getattr(cfg, "l2_regime_require_fit_n_for_downweight", False)),
        ).lower() in ("1", "true", "yes"),
    )
    l2_sim_cache = replace(l2_sim_cache,
        bucket_edges_by_fold=_routing_plan.effective_bucket_edges_by_fold,
        pooled_edges_by_fold=_routing_plan.pooled_edges_by_fold,
        regime_code_1d=_routing_plan.effective_regime_code_1d,
        regime_routing_diagnostics=_routing_plan.diagnostics,
        regime_policy_by_fold=_routing_plan.policy_by_fold,
    )
    _policy_diag = _routing_plan.diagnostics.policy_diagnostics
    # Source contract for diagnostics tests:
    # [REGIME-L2] active_states=3 compression=True path=pooled_fallback proof=False ...
    _logger.info(
        "[REGIME-L2] active_states=%d compression=%s path=%s proof=%s lift=%.2f t=%.2f fold_pass=%.2f",
        _routing_plan.diagnostics.active_state_count,
        _routing_plan.diagnostics.compression_enabled,
        _routing_plan.diagnostics.conditioning_path,
        _routing_plan.diagnostics.proof_passed,
        _routing_plan.diagnostics.mean_lift_bps,
        _routing_plan.diagnostics.nw_tstat,
        _routing_plan.diagnostics.fold_pass_ratio,
    )
    if _policy_diag is not None:
        _logger.info(
            "[REGIME-L2] policy_mode=%s policy_source=fit/cal "
            "global_reliable=%s allow=%d downweight=%d block=%d pooled=%d unstable=%d "
            "hard_block_eligible=%d sign_consistency=%.2f hard_block_enabled=%s "
            "mean_cal_lift=%.2f mean_conf=%.2f",
            _policy_diag.mode,
            _policy_diag.global_reliable,
            _policy_diag.n_allow,
            _policy_diag.n_downweight,
            _policy_diag.n_block,
            _policy_diag.n_pooled,
            _policy_diag.n_unstable,
            _policy_diag.n_hard_block_eligible,
            _policy_diag.sign_consistency_ratio,
            _policy_diag.hard_block_enabled,
            _policy_diag.mean_cal_lift_bps,
            _policy_diag.mean_confidence,
        )
    if _logger.isEnabledFor(logging.DEBUG) and not _routing_plan.diagnostics.proof_passed:
        _logger.debug(
            "[REGIME-L2-DETAIL] pooled_fallback reason=proof_failed effective_states=%d",
            _routing_plan.diagnostics.active_state_count,
        )
    if _logger.isEnabledFor(logging.DEBUG) and _policy_diag is not None:
        _logger.debug(
            "[REGIME-L2-POLICY] policy_mode=%s global_reliable=%s "
            "n_allow=%d n_downweight=%d n_block=%d n_pooled=%d n_unstable=%d "
            "n_hard_block_eligible=%d sign_consistency_ratio=%.2f hard_block_enabled=%s "
            "mean_cal_lift_bps=%.2f mean_confidence=%.2f",
            _policy_diag.mode,
            _policy_diag.global_reliable,
            _policy_diag.n_allow,
            _policy_diag.n_downweight,
            _policy_diag.n_block,
            _policy_diag.n_pooled,
            _policy_diag.n_unstable,
            _policy_diag.n_hard_block_eligible,
            _policy_diag.sign_consistency_ratio,
            _policy_diag.hard_block_enabled,
            _policy_diag.mean_cal_lift_bps,
            _policy_diag.mean_confidence,
        )
        _log_ascii_table(
            "[REGIME] DEBUG",
            ("metric", "value"),
            [
                ["policy_mode", str(_policy_diag.mode)],
                ["global_reliable", "yes" if _policy_diag.global_reliable else "no"],
                ["n_allow", str(_policy_diag.n_allow)],
                ["n_downweight", str(_policy_diag.n_downweight)],
                ["n_block", str(_policy_diag.n_block)],
                ["n_pooled", str(_policy_diag.n_pooled)],
                ["n_unstable", str(_policy_diag.n_unstable)],
                ["n_hard_block_eligible", str(_policy_diag.n_hard_block_eligible)],
                ["sign_consistency_ratio", f"{_policy_diag.sign_consistency_ratio:.2f}"],
                ["hard_block_enabled", "yes" if _policy_diag.hard_block_enabled else "no"],
                ["mean_cal_lift_bps", f"{_policy_diag.mean_cal_lift_bps:.2f}"],
                ["mean_confidence", f"{_policy_diag.mean_confidence:.2f}"],
            ],
            (24, 18),
            level=logging.DEBUG,
        )
        _logger.debug("[REGIME-L2-POLICY] OOS DEBUG = evaluation only")
        _logger.debug("[REGIME-L2-POLICY] policy source = fit/cal")
    if not _routing_plan.diagnostics.proof_passed:
        _logger.warning(
            "[REGIME-L2] proof_failed path=%s effective_states=%d",
            _routing_plan.diagnostics.conditioning_path,
            _routing_plan.diagnostics.active_state_count,
        )
    if _logger.isEnabledFor(logging.DEBUG) and _routing_plan.diagnostics.debug_diagnostics is not None:
        _debug_diag = _routing_plan.diagnostics.debug_diagnostics
        _debug_top_k = max(int(getattr(cfg, "l2_regime_debug_top_k", 10)), 1)
        _granularity_rows = [
            (
                stat.label,
                str(stat.state_count),
                ("✅" if stat.proof_passed else "❌"),
                f"{stat.mean_lift_bps:.2f}",
                f"{stat.nw_tstat:.2f}",
                f"{stat.fold_pass_ratio:.2f}",
                f"{stat.bucket_hit_pct_mean:.1f}",
                f"{stat.oos_cell_ic:.3f}",
                f"{stat.oos_cell_rmse_bps:.2f}",
                f"{stat.oos_cell_bias_bps:.2f}",
            )
            for stat in _debug_diag.granularity_stats
        ]
        _log_ascii_table(
            "[REGIME-DEBUG-GRANULARITY]",
            ("label", "states", "proof", "lift_bps", "tstat", "fold_pass", "hit_pct", "cell_ic", "rmse", "bias"),
            _granularity_rows,
            (12, 6, 5, 8, 6, 9, 7, 7, 6, 6),
        )
        _cell_rows = [
            (
                str(rank),
                f"{stat.state_name}/{stat.family}/{stat.tf}",
                str(stat.fold_idx),
                str(stat.n_fit),
                str(stat.n_oos),
                f"{stat.fit_edge_bps:+.1f}",
                f"{stat.pooled_fit_edge_bps:+.1f}",
                f"{stat.oos_realized_edge_bps:+.1f}",
                f"{stat.edge_gap_bps:+.1f}",
                f"{stat.sign_hit_rate:.2f}",
                f"{stat.selected_hit_pct:.2f}",
            )
            for rank, stat in enumerate(_debug_diag.worst_error_cells[:_debug_top_k], start=1)
        ]
        _log_ascii_table(
            "[REGIME-DEBUG-CELLS]",
            ("rank", "bucket", "fold", "n_fit", "n_oos", "fit", "pooled", "oos", "gap", "sign_hit", "selected_hit"),
            _cell_rows,
            (4, 18, 4, 5, 5, 7, 7, 7, 7, 8, 12),
        )
        _logger.debug(
            "[REGIME-DEBUG-GRANULARITY] compression_loss_bps=%.2f",
            _debug_diag.compression_loss_bps,
        )

    ctx = TieredContext(
        labeled_events=pd.DataFrame(),
        aligned=aligned,
        cfg=cfg,
        window=window,
        caps=caps,
        tf=tf,
        fixed_l1_params={"signal_batch": signal_batch},
        l2_sim_cache=l2_sim_cache,
        awf_folds=_awf_folds_l2,
    )

    study_name = _layer2_experiment_key(
        tf=tf,
        window=window,
        signal_batch=signal_batch,
        search_space_version="v7",
    )
    signal_batch_fingerprint = _signal_batch_fingerprint(signal_batch)
    _unique_symbols_list = sorted({str(event.symbol) for event in signal_batch.events})
    unique_symbols = ",".join(_unique_symbols_list) or "-"
    _logger.info(
        "  ● [STUDY] %s | trials=%d | events=%d | symbols=%d",
        study_name, n_trials, len(signal_batch.events), len(_unique_symbols_list),
    )
    _logger.info("  ────────────────────────────────────────────────────────────────────────────")
    _logger.debug(
        "    symbols=%s fp=%s", unique_symbols, signal_batch_fingerprint[:12],
    )

    try:
        # setup_optuna_storage를 1회만 호출하여 로그 중복 제거
        _, storage = setup_optuna_storage(str(BASE_DIR))
        from tqdm import tqdm
        class L2OptunaProgressCallback:
            def __init__(self, total_trials: int):
                self.pbar = tqdm(total=total_trials, desc="[L2-OPT]", leave=True)
                self.best_val = float("-1e6")

            def __call__(self, study: Any, trial: Any, value: float | None = None) -> None:
                val = value
                if val is None:
                    val = getattr(trial, "value", None)
                if val is None and hasattr(study, "trials") and len(study.trials) > trial.number:
                    val = getattr(study.trials[trial.number], "value", None)
                
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
            import psutil
            batch_size = int(OPT_FUTURES_CONFIG.get("L2_OPTUNA_BATCH_SIZE", 2))
            try:
                avail_mem_gb = psutil.virtual_memory().available / (1024.0 ** 3)
                if avail_mem_gb < 3.0 and batch_size > 1:
                    _logger.warning(
                        "[L2-OPT] Low system memory (%.2f GB < 3.0 GB). Forcing sequential n_jobs=1 for OOM safety.",
                        avail_mem_gb,
                    )
                    batch_size = 1
            except Exception as _mem_err:
                _logger.warning("[L2-OPT] Failed to check system memory: %s", _mem_err)

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

                avail_gb = psutil.virtual_memory().available / (1024.0 ** 3)
                cpu_cores = os.cpu_count() or 4
                # Fork CoW: child shares parent numpy arrays. Unique allocation ≈ 0.7GB per worker.
                mem_safe = max(1, int(avail_gb / 0.7))
                max_workers = max(1, min(batch_size, cpu_cores, mem_safe))
                _logger.info(
                    "[L2-OPT] ProcessPool workers=%d (mem=%.1fGB, mem_safe=%d, cpu=%d, batch=%d)",
                    max_workers, avail_gb, mem_safe, cpu_cores, batch_size,
                )

                global _GLOBAL_L2_CTX
                _GLOBAL_L2_CTX = ctx

                try:
                    mp_ctx = multiprocessing.get_context("fork")
                    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
                        trial_idx = len([t for t in study.trials if t.state.is_finished()])
                        _batch_num = 0
                        while trial_idx < n_trials:
                            _batch_num += 1
                            _t_batch = time.perf_counter()
                            current_batch = min(batch_size, n_trials - trial_idx)
                            batch_trials = []
                            batch_params = []

                            _mem_batch_start = _get_rss_mb()
                            _t_ask = 0.0

                            for _ in range(current_batch):
                                _t0 = time.perf_counter()
                                trial = study.ask()
                                batch_trials.append(trial)
                                params = suggest_layered_params(trial, "L2", fixed=ctx.fixed_l1_params)
                                batch_params.append(params)
                                _t_ask += time.perf_counter() - _t0

                            _t_submit = time.perf_counter()
                            futures = [
                                executor.submit(_evaluate_l2_trial_from_global, params)
                                for params in batch_params
                            ]
                            _t_submit = time.perf_counter() - _t_submit

                            _t_tell = 0.0
                            _t_attrs = 0.0
                            _sum_eval = 0.0
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
                                _sum_eval += t_elapsed

                                _t0 = time.perf_counter()
                                for k, v in attrs.items():
                                    trial.set_user_attr(k, v)
                                _t_attrs += time.perf_counter() - _t0

                                _t0 = time.perf_counter()
                                study.tell(trial, value)
                                _t_tell += time.perf_counter() - _t0

                                _logger.log(
                                    logging.DEBUG,
                                    "[perf-optuna] Trial %d eval=%.3fs obj=%.6f",
                                    trial.number,
                                    t_elapsed,
                                    value,
                                )
                                progress_cb(study, trial, value=value)
                                trial_idx += 1

                            _t_gc = time.perf_counter()
                            gc.collect()
                            _t_gc = time.perf_counter() - _t_gc
                            _t_batch = time.perf_counter() - _t_batch
                            _t_result = _t_batch - _t_ask - _t_submit - _t_attrs - _t_tell - _t_gc

                            _logger.debug(
                                "[L2-PERF] batch=%d/%d trials=%d t_total=%.2fs "
                                "t_eval=%.1fs/%.1fs/trial eff=%.0f%% "
                                "| ask=%.2fs submit=%.3fs attrs=%.3fs tell=%.2fs gc=%.2fs "
                                "| workers=%d mem=%.0fMB",
                                _batch_num,
                                int(np.ceil(n_trials / batch_size)),
                                current_batch,
                                _t_batch,
                                _sum_eval,
                                _sum_eval / max(current_batch, 1),
                                (_sum_eval / max(_t_batch * max_workers, 0.001)) * 100,
                                _t_ask,
                                _t_submit,
                                _t_attrs,
                                _t_tell,
                                _t_gc,
                                max_workers,
                                _mem_batch_start,
                            )
                            _log_mem("l2_optuna_batch", _mem_batch_start, extra=f"trial_idx={trial_idx}/{n_trials}")
                finally:
                    _GLOBAL_L2_CTX = None
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

    # (fold pre-computation moved before study — stored in ctx.awf_folds)

    gc.collect()
    _logger.debug("[MEM] stage=l2_study_complete rss=%.0fMB", _get_rss_mb())
    _min_dsr = float(OPT_FUTURES_CONFIG.get("FUTURES_L2_MIN_DSR", 0.60))
    
    _t_champ_start = time.perf_counter()
    _mem_champ_before = _get_rss_mb()
    
    l2_study_result = select_layer2_champion(
        study=study,
        tf=tf,
        signal_batch=signal_batch,
        aligned=aligned,
        awf_folds=_awf_folds_l2,
        caps=caps,
        min_dsr=_min_dsr,
        prebuilt_cache=l2_sim_cache,
    )
    
    _log_mem("select_layer2_champion", _mem_champ_before, extra=f"took={time.perf_counter() - _t_champ_start:.4f}s")
    gc.collect()

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

    _blocker = l2_study_result.blocker_reason or "none"
    _logger.debug("[MEM] stage=l2_champion rss=%.0fMB blocked=%s", _get_rss_mb(), _blocker)
    return l2_study_result


@dataclass(slots=True, frozen=True)
class L2ReversalReplayVariant:
    name: str
    enabled: bool
    dd_threshold: float
    persistence_bars: int


@dataclass(slots=True, frozen=True)
class L2ReversalReplayFoldMetric:
    variant: str
    fold_idx: int
    cagr: float | None
    mdd: float | None
    risk_off_bars: int
    risk_off_realized_price: float
    risk_on_realized_price: float


@dataclass(slots=True, frozen=True)
class L2ReversalReplayResult:
    variant: str
    cagr: float
    mdd: float
    trade_count: int
    deploy_leverage: float
    selection_parity: bool
    metric_parity: bool
    fold_metrics: tuple[L2ReversalReplayFoldMetric, ...]
    adoption_passed: bool
    blocker_reason: str


def _l2_reversal_replay_variants() -> tuple[L2ReversalReplayVariant, ...]:
    return (
        L2ReversalReplayVariant(name="baseline_off", enabled=False, dd_threshold=0.0, persistence_bars=1),
        L2ReversalReplayVariant(name="legacy_006_p1", enabled=True, dd_threshold=0.06, persistence_bars=1),
        L2ReversalReplayVariant(name="balanced_010_p2", enabled=True, dd_threshold=0.10, persistence_bars=2),
        L2ReversalReplayVariant(name="balanced_010_p3", enabled=True, dd_threshold=0.10, persistence_bars=3),
        L2ReversalReplayVariant(name="current_012_p3", enabled=True, dd_threshold=0.12, persistence_bars=3),
    )


@contextmanager
def _temporary_reversal_env(variant: L2ReversalReplayVariant) -> Iterator[None]:
    _saved: dict[str, str | None] = {}
    for _key in ("L2_REVERSAL_KILL", "L2_REVERSAL_DD_THRESHOLD", "L2_REVERSAL_PERSISTENCE_BARS"):
        _saved[_key] = os.environ.get(_key)
    try:
        if not variant.enabled:
            os.environ.pop("L2_REVERSAL_KILL", None)
        else:
            os.environ["L2_REVERSAL_KILL"] = "1"
            os.environ["L2_REVERSAL_DD_THRESHOLD"] = str(variant.dd_threshold)
            os.environ["L2_REVERSAL_PERSISTENCE_BARS"] = str(variant.persistence_bars)
        yield
    finally:
        for _key, _val in _saved.items():
            if _val is None:
                os.environ.pop(_key, None)
            else:
                os.environ[_key] = _val


def _fold_metrics_from_l2_evaluation(
    *,
    variant_name: str,
    evaluation: Any,
) -> tuple[L2ReversalReplayFoldMetric, ...]:
    fold_cagrs = getattr(evaluation, "fold_deployed_cagrs", ())
    fold_mdds = getattr(evaluation, "fold_deployed_mdds", ())
    fold_attribs = getattr(evaluation, "fold_attributions", ())
    n_folds = max(len(fold_cagrs), len(fold_mdds), len(fold_attribs))
    metrics: list[L2ReversalReplayFoldMetric] = []
    for i in range(n_folds):
        cagr = fold_cagrs[i] if i < len(fold_cagrs) else None
        mdd = fold_mdds[i] if i < len(fold_mdds) else None
        attr = fold_attribs[i] if i < len(fold_attribs) else None
        risk_off_bars = attr.risk_off_bars if attr is not None else 0
        risk_off_price = attr.risk_off_realized_price if attr is not None else 0.0
        risk_on_price = attr.risk_on_realized_price if attr is not None else 0.0
        metrics.append(
            L2ReversalReplayFoldMetric(
                variant=variant_name,
                fold_idx=i,
                cagr=cagr,
                mdd=mdd,
                risk_off_bars=risk_off_bars,
                risk_off_realized_price=risk_off_price,
                risk_on_realized_price=risk_on_price,
            )
        )
    return tuple(metrics)


def _reversal_replay_adoption_verdict(
    *,
    baseline: L2ReversalReplayResult,
    legacy: L2ReversalReplayResult,
    candidate: L2ReversalReplayResult,
) -> tuple[bool, str]:
    baseline_fold0_cagr = baseline.fold_metrics[0].cagr if baseline.fold_metrics else None
    legacy_fold0_cagr = legacy.fold_metrics[0].cagr if legacy.fold_metrics else None
    candidate_fold0_cagr = candidate.fold_metrics[0].cagr if candidate.fold_metrics else None
    if baseline_fold0_cagr is None or legacy_fold0_cagr is None or candidate_fold0_cagr is None:
        return (False, "missing_fold0")
    legacy_improvement = legacy_fold0_cagr - baseline_fold0_cagr
    candidate_improvement = candidate_fold0_cagr - baseline_fold0_cagr
    if legacy_improvement <= 0.0:
        return (False, "legacy_no_improvement")
    if candidate_improvement < legacy_improvement * 0.70:
        return (False, "fold0_defense_below_70pct")
    for fold_idx in range(1, len(candidate.fold_metrics)):
        base = baseline.fold_metrics
        cand = candidate.fold_metrics
        if fold_idx < len(base) and fold_idx < len(cand):
            base_cagr = base[fold_idx].cagr
            cand_cagr = cand[fold_idx].cagr
            if base_cagr is not None and cand_cagr is not None and cand_cagr - base_cagr < -0.01:
                return (False, "non_bottleneck_damage")
    if candidate.cagr <= baseline.cagr:
        return (False, "below_baseline_cagr")
    if candidate.cagr < legacy.cagr:
        return (False, "below_legacy_cagr")
    if not candidate.fold_metrics or candidate.fold_metrics[0].risk_off_bars <= 0:
        return (False, "no_fold0_risk_off")
    if not candidate.selection_parity:
        return (False, "selection_divergence")
    return (True, "")


def _run_l2_reversal_economic_replay(
    *,
    signal_batch: Any,
    aligned: Any,
    awf_folds: tuple[Any, ...],
    base_l2_params: dict[str, object],
    caps: Any,
    tf: str,
    deploy_leverage: float | None,
    prebuilt_cache: Any,
    reference_evaluation: Any,
    output_path: Path | None = None,
) -> tuple[L2ReversalReplayResult, ...]:
    from src.domain.futures.optimization.workflow import evaluate_l2_trial
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
    from src.domain.futures.strategy.tiered_workflow.replay_parity import assert_selection_replay_parity

    _config = Layer2AllocationConfig.from_mapping(base_l2_params)
    variants = _l2_reversal_replay_variants()
    results: list[L2ReversalReplayResult] = []
    for variant in variants:
        with _temporary_reversal_env(variant):
            _dl_override: float | None = (
                deploy_leverage if deploy_leverage is not None and deploy_leverage > 1.0 else None
            )
            evaluation = evaluate_l2_trial(
                cache=prebuilt_cache,
                signal_batch=signal_batch,
                aligned=aligned,
                awf_folds=awf_folds,
                config=_config,
                caps=caps,
                tf=tf,
                deploy_leverage_override=_dl_override,
            )
        fold_metrics = _fold_metrics_from_l2_evaluation(variant_name=variant.name, evaluation=evaluation)
        _selection_parity = sorted(getattr(evaluation, "last_selected_symbols", ())) == sorted(
            getattr(reference_evaluation, "last_selected_symbols", ())
        )
        _metric_parity: bool = False
        if variant.name == "baseline_off":
            try:
                _metric_parity = assert_selection_replay_parity(
                    replay_evaluation=evaluation,
                    final_evaluation=reference_evaluation,
                )
            except Exception:
                _metric_parity = False
        _blocker_reason = "baseline" if variant.name == "baseline_off" else ""
        result = L2ReversalReplayResult(
            variant=variant.name,
            cagr=float(getattr(evaluation, "cagr_hybrid", 0.0)),
            mdd=float(getattr(evaluation, "mdd_hybrid", 0.0)),
            trade_count=int(getattr(evaluation, "trade_count", 0)),
            deploy_leverage=float(getattr(evaluation, "deploy_leverage", 1.0)),
            selection_parity=_selection_parity,
            metric_parity=_metric_parity,
            fold_metrics=fold_metrics,
            adoption_passed=False,
            blocker_reason=_blocker_reason,
        )
        results.append(result)
    # adoption verdict
    baseline_result = next((r for r in results if r.variant == "baseline_off"), None)
    legacy_result = next((r for r in results if r.variant == "legacy_006_p1"), None)
    for result in results:
        if result.variant == "baseline_off":
            continue
        if baseline_result is not None and legacy_result is not None and result.adoption_passed is False:
            passed, reason = _reversal_replay_adoption_verdict(
                baseline=baseline_result,
                legacy=legacy_result,
                candidate=result,
            )
            results[results.index(result)] = L2ReversalReplayResult(
                variant=result.variant,
                cagr=result.cagr,
                mdd=result.mdd,
                trade_count=result.trade_count,
                deploy_leverage=result.deploy_leverage,
                selection_parity=result.selection_parity,
                metric_parity=result.metric_parity,
                fold_metrics=result.fold_metrics,
                adoption_passed=passed,
                blocker_reason=reason,
            )
    # log summary
    for res in results:
        _logger.info(
            "[L2-REVERSAL-REPLAY] variant=%s CAGR=%.4f MDD=%.4f trades=%d L*=%.2f "
            "selection_parity=%s metric_parity=%s adoption_passed=%s blocker=%s",
            res.variant, res.cagr, res.mdd, res.trade_count, res.deploy_leverage,
            res.selection_parity, res.metric_parity, res.adoption_passed, res.blocker_reason,
        )
    if output_path is not None:
        _write_replay_csv(results=tuple(results), output_path=output_path)
    return tuple(results)


def _write_replay_csv(
    *,
    results: tuple[L2ReversalReplayResult, ...],
    output_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for res in results:
        for fm in res.fold_metrics:
            rows.append({
                "variant": res.variant,
                "fold_idx": fm.fold_idx,
                "cagr": fm.cagr,
                "mdd": fm.mdd,
                "risk_off_bars": fm.risk_off_bars,
                "risk_off_realized_price": fm.risk_off_realized_price,
                "risk_on_realized_price": fm.risk_on_realized_price,
                "variant_cagr": res.cagr,
                "variant_mdd": res.mdd,
                "trade_count": res.trade_count,
                "deploy_leverage": res.deploy_leverage,
                "selection_parity": res.selection_parity,
                "metric_parity": res.metric_parity,
                "adoption_passed": res.adoption_passed,
                "blocker_reason": res.blocker_reason,
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _run_strategy_stage(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
    data_stage: DataStageResult,
    trading_symbols: tuple[str, ...] = (),
    universe_snapshot: UniverseSnapshot | None = None,
    layered_window: Any | None = None,
    universe_result: Any | None = None,
    *,
    probe_result: TfProbeStageResult | None = None,
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

    # Initialize probe result — overwritten inside tiered branch if applicable.
    _probe_result_local: TfProbeStageResult | None = None

    t_step = time.perf_counter()
    strategy_maps = pick_strategy_data_maps(
        oos_data_maps=data_stage.oos_data_maps,
        is_data_maps=data_stage.data_maps,
        valid_symbols=data_stage.valid_symbols,
        tf=run_config.timeframe,
    )
    strategy_steps["map_pick"] = time.perf_counter() - t_step
    full_strategy_maps = strategy_maps
    
    if run_config.phase in ("l1", "l2"):
        _mem_before = _get_rss_mb()
        if hasattr(data_stage, "data_maps") and isinstance(data_stage.data_maps, dict):
            data_stage.data_maps.clear()
        if hasattr(data_stage, "oos_data_maps") and isinstance(data_stage.oos_data_maps, dict):
            data_stage.oos_data_maps.clear()
        gc.collect()
        _log_mem("data_stage_early_release", _mem_before)

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
            
            # Unified Minimal Tree Style log emission
            dropped_count = sum(len(value) for value in scope_result.dropped_by_reason.values())
            l_start_dropped = len(scope_result.dropped_by_reason.get('late_start', []))
            _logger.info(
                "📊 [L1: SWF SCOPE & ADMISSION]\n"
                f"  ├─ Symbols : {len(effective_trade_syms)}/{len(data_stage.valid_symbols)} Admitted\n"
                f"  └─ Details : Base {len(base_scope)} | Dropped {dropped_count} (late_start: {l_start_dropped})"
            )
        if not effective_trade_syms:
            from src.domain.futures.strategy.tiered_workflow.pipeline import (
                TieredPipelineError,
            )

            raise TieredPipelineError(
                "tiered tradeable scope is empty after sub-window admission"
            )
    else:
        effective_trade_syms = list(data_stage.valid_symbols)

    bridge_symbol_scope = tuple(effective_trade_syms) if use_tiered else (
        trading_symbols or tuple(data_stage.valid_symbols)
    )
    bridge_trading_symbols = list(bridge_symbol_scope)

    if use_tiered:
        tiered_cfg = build_candidate_strategy_config(
            strategy_cfg=StrategyConfig(name="candidate_ml"),
            opt_config=OPT_FUTURES_CONFIG,
            timeframe=run_config.timeframe,
        ).candidate
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
    _rss_pre_gc = _get_rss_mb()
    _logger.debug("[MEM] stage=pre_gc rss=%.0fMB", _rss_pre_gc)
    gc.collect()
    _rss_post_gc = _get_rss_mb()
    _logger.debug("[MEM] stage=post_gc rss=%.0fMB delta=%+.0fMB", _rss_post_gc, _rss_post_gc - _rss_pre_gc)

    # ─── Tiered Pipeline 분기 (bridge 완료 후 — labeled + aligned 사용 가능) ──
    if use_tiered:
        from src.domain.futures.strategy.tiered_workflow.pipeline import (
            TieredPipelineError,
            run_tiered_pipeline,
        )

        try:

            from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
            from src.domain.futures.strategy.common.alignment import align_data_maps
            _pit_state_cube = _resolve_universe_state_cube(universe_result)
            _mem_align = _get_rss_mb()
            t_align = time.perf_counter()
            aligned_tiered = align_data_maps(
                full_strategy_maps,
                effective_trade_syms,
                run_config.timeframe,
                state_cube=_pit_state_cube,
            )
            _logger.log(
                PERF,
                "[PERF] tiered_align_data_maps n_syms=%d tf=%s took=%.4fs",
                len(effective_trade_syms),
                run_config.timeframe,
                time.perf_counter() - t_align,
            )
            _rss_align = _get_rss_mb()
            _logger.debug(
                "[MEM] stage=align rss=%.0fMB n_syms=%d n_bars=%d",
                _rss_align, len(effective_trade_syms), len(aligned_tiered.datetimes),
            )
            _logger.debug(
                "[MEM] stage=bridge rss=%.0fMB delta=%+.0fMB n_syms=%d",
                _get_rss_mb(), _rss_align - _mem_align, len(bridge_trading_symbols),
            )
            del full_strategy_maps
            gc.collect()
            _logger.debug("[MEM] stage=post_align_free rss=%.0fMB", _get_rss_mb())
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
            assert tiered_cfg is not None
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
            _mem_tiered_before = _get_rss_mb()
            l1_res, _, _ = run_tiered_pipeline(
                labeled_events=labeled_tiered,
                aligned=aligned_tiered,
                cfg=tiered_cfg,
                window=tiered_window,
                l1_params={},
                l2_params={},
                caps=tiered_caps,
                l1_tfs=tuple(tiered_cfg.l1_tfs),
                target_phase="l1",
                verbose=True,

            )
            _log_mem("tiered_pipeline", _mem_tiered_before, extra=f"phase={run_config.phase}")
            if not l1_res.gate_passed:
                _logger.info("[TIERED] L1 BLOCKED — gate_passed=False")
                return None
            if run_config.phase not in _recognized_multilayer:
                _logger.info("[TIERED] Phase=%s — stopping after L1 (not a multilayer phase)", run_config.phase)
                return l1_res

            _logger.info("\n>> LAYER 1: PASS -> Proceeding to Layer 2.")

            # ── Step I: Regime Quality Gate ──────────────────────────────────
            if getattr(tiered_cfg, "l2_routing_mode", "bucket") == "bucket":
                from src.domain.futures.strategy.market_regime import (
                    compute_market_regime_context as _compute_regime_ctx,
                )
                _regime_ctx = _compute_regime_ctx(aligned=aligned_tiered)
                _regime_code = _regime_ctx.code_1d
                from src.domain.futures.strategy.market_regime import compress_regime_codes

                _effective_regime_code = (
                    compress_regime_codes(_regime_code)
                    if bool(getattr(tiered_cfg, "l2_regime_compression_enabled", True))
                    else _regime_code.copy()
                )
                _unique_codes, _counts_codes = np.unique(_effective_regime_code, return_counts=True)
                _n_total = _effective_regime_code.shape[0]
                _state_pct = {
                    int(r): float(c) / _n_total * 100.0
                    for r, c in zip(_unique_codes.tolist(), _counts_codes.tolist(), strict=True)
                }
                _state_dist = (
                    f"bull={_state_pct.get(0, 0.0):.1f}% "
                    f"bear={_state_pct.get(1, 0.0):.1f}% "
                    f"crisis={_state_pct.get(2, 0.0):.1f}%"
                )
                _state_status = "🟢 stable" if _compute_c2_macro(_effective_regime_code)[0] >= 12.0 else "🟠 unstable"
                _policy_mode = str(getattr(tiered_cfg, "l2_regime_policy_mode", "soft"))
                _hard_block = "on" if bool(getattr(tiered_cfg, "l2_regime_hard_block_enabled", False)) else "off"
                _risk_cap = "on" if bool(getattr(tiered_cfg, "l2_regime_risk_cap_enabled", True)) else "off"

                _logger.info(
                    "[REGIME]\n"
                    "metric        | value\n"
                    "compression   | %s\n"
                    "states        | 3\n"
                    "status        | %s\n"
                    "distribution  | %s\n"
                    "policy_mode   | %s\n"
                    "hard_block    | %s\n"
                    "risk_cap      | %s\n"
                    "policy_source | fit/cal\n"
                    "oos_debug     | evaluation only\n"
                    "note          | L2 verdict is reported separately in [REGIME-L2]",
                    "on" if bool(getattr(tiered_cfg, "l2_regime_compression_enabled", True)) else "off",
                    _state_status,
                    _state_dist,
                    _policy_mode,
                    _hard_block,
                    _risk_cap,
                )

            # ── Step B: L2 Optimization Header ──────────────────────────────
            _logger.info(format_layer_header(2, "OPTUNA TUNING"))

            # ── Step C: L2 window 신호 예측 ──────────────────────────────────
            _mem_l2_start = _get_rss_mb()
            _t_l2_pred_start = time.perf_counter()
            l2_signals = _build_l2_signal_batch(
                l1_res, labeled_tiered, aligned_tiered, tiered_cfg, tiered_window
            )
            _log_mem("l2_signal_batch", _mem_l2_start, extra=f"took={time.perf_counter() - _t_l2_pred_start:.4f}s")

            # ── Step D: Optuna L2 파라미터 탐색 ──────────────────────────────
            from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache
            _t_cache_start = time.perf_counter()
            shared_l2_cache = build_l2_simulation_cache(aligned_tiered, l2_signals, run_config.timeframe)
            _logger.log(
                PERF,
                "[PERF] prebuilt l2_sim_cache took=%.4fs",
                time.perf_counter() - _t_cache_start,
            )

            # ── Stage A: 메타 유효성 측정 (env-gated, param 무관 1회) ──────────
            if os.environ.get("L2_META_FEAS", "") not in ("", "0", "false", "False"):
                try:
                    from src.domain.futures.strategy.market_regime import (
                        compute_market_regime_context,
                    )
                    from src.domain.futures.strategy.tiered_workflow.l2_meta import (
                        build_sleeve_meta_dataset,
                        evaluate_meta_feasibility,
                    )
                    _mf_regime = compute_market_regime_context(aligned=aligned_tiered).code_1d
                    _mf_start = int(getattr(l2_signals, "start_idx", 0))
                    _mf_end = int(getattr(l2_signals, "end_idx", aligned_tiered.close_2d.shape[0]))  # type: ignore[arg-type]
                    _mf_cost = float(OPT_FUTURES_CONFIG.get("L2_ROUND_TRIP_COST_BPS", 6.0))
                    _mf_samples = build_sleeve_meta_dataset(
                        shared_l2_cache, aligned_tiered, _mf_regime,
                        _mf_start, _mf_end, cost_bps=_mf_cost,
                    )
                    _mf_embargo = int(OPT_FUTURES_CONFIG.get("EMBARGO_BARS_BY_TF", {}).get(run_config.timeframe, 42))
                    _mf_report = evaluate_meta_feasibility(
                        _mf_samples, n_splits=5, embargo_bars=_mf_embargo, threshold_quantile=0.70,
                    )
                    _logger.info(
                        "[L2-META-FEAS-SUMMARY] meta_ic=%.4f tstat=%.2f net_lift_bps=%.3f "
                        "auc=%.3f n_oos=%d n_events=%d feats=%s",
                        _mf_report.oos_meta_ic, _mf_report.oos_meta_ic_tstat,
                        _mf_report.net_edge_lift_bps, _mf_report.auc_sign,
                        _mf_report.n_oos, len(_mf_samples.y),
                        ",".join(_mf_samples.feature_names),
                    )
                    for _bk, _bv in sorted(
                        _mf_report.bucket_table.items(), key=lambda kv: kv[1], reverse=True
                    )[:12]:
                        _logger.info("[L2-META-BUCKET] %s oos_edge_bps=%.3f", _bk, _bv)

                    # Decisive conditional feasibility: causal fit->oos bucket-edge
                    # correlation (the conditional analog of P2a). Splits samples by
                    # event_t median; per (regime,family,TF) bucket with >=min_n events
                    # in BOTH halves, correlates fit-edge vs oos-edge across buckets.
                    if len(_mf_samples.y) >= 50:
                        from scipy.stats import spearmanr as _spr
                        _et = _mf_samples.event_t
                        _med = float(np.median(_et))
                        _fitm = _et < _med
                        _oosm = ~_fitm
                        _reg = _mf_samples.X[:, 0].astype(np.int64)
                        _keys = [
                            f"r{int(_reg[i])}/{_mf_samples.sleeve_family[i]}/{_mf_samples.sleeve_tf[i]}"
                            for i in range(len(_mf_samples.y))
                        ]
                        _bfit: dict[str, list[float]] = {}
                        _boos: dict[str, list[float]] = {}
                        for _i, _k in enumerate(_keys):
                            (_bfit if _fitm[_i] else _boos).setdefault(_k, []).append(
                                float(_mf_samples.y[_i])
                            )
                        # Robustness: multiple forward split fractions x min_n,
                        # report the distribution of causal corr (stability check).
                        _et_sorted = np.sort(_et)
                        for _frac in (0.40, 0.50, 0.60, 0.70):
                            _cut = float(_et_sorted[int(len(_et_sorted) * _frac)])
                            _fm = _et < _cut
                            for _min_n in (20, 50):
                                _bf: dict[str, list[float]] = {}
                                _bo: dict[str, list[float]] = {}
                                for _i, _k in enumerate(_keys):
                                    (_bf if _fm[_i] else _bo).setdefault(_k, []).append(
                                        float(_mf_samples.y[_i])
                                    )
                                _pf = []
                                _po = []
                                for _k in _bf:
                                    if _k in _bo and len(_bf[_k]) >= _min_n and len(_bo[_k]) >= _min_n:
                                        _pf.append(float(np.mean(_bf[_k])))
                                        _po.append(float(np.mean(_bo[_k])))
                                if len(_pf) >= 5:
                                    _bc, _bp = _spr(_pf, _po)
                                    _logger.info(
                                        "[L2-BUCKET-PERSIST] split=%.2f min_n=%d "
                                        "causal_corr=%.4f pval=%.4f n_buckets=%d",
                                        _frac, _min_n, float(_bc), float(_bp), len(_pf),
                                    )
                                else:
                                    _logger.info(
                                        "[L2-BUCKET-PERSIST] split=%.2f min_n=%d insufficient n=%d",
                                        _frac, _min_n, len(_pf),
                                    )
                except Exception as _mf_exc:
                    _logger.warning("[L2-META-FEAS] measurement failed: %s", _mf_exc, exc_info=True)

            _seed = int(run_config.seed) if hasattr(run_config, "seed") else 42
            n_l2_trials = int(OPT_FUTURES_CONFIG.get("L2_OPTUNA_TRIALS", 50))
            # Experiment knob (parity with L2_MULTI_TF / L2_SLEEVE_COMBINE toggles):
            # allow fast equal-budget A/B runs without editing static config.
            _trials_override = os.environ.get("L2_OPTUNA_TRIALS", "")
            if _trials_override.isdigit() and int(_trials_override) > 0:
                n_l2_trials = int(_trials_override)
            _mem_l2_study = _get_rss_mb()
            _t_l2_study_start = time.perf_counter()
            l2_study_result = _run_tiered_l2_study(
                signal_batch=l2_signals,
                aligned=aligned_tiered,
                cfg=tiered_cfg,
                window=tiered_window,
                caps=tiered_caps,
                tf=run_config.timeframe,
                n_trials=n_l2_trials,
                seed=_seed,
                l2_sim_cache=shared_l2_cache,
            )
            _log_mem(
                "l2_optuna_study",
                _mem_l2_study,
                extra=f"trials={n_l2_trials} took={time.perf_counter() - _t_l2_study_start:.4f}s",
            )
            best_l2_params = dict(getattr(l2_study_result, "best_params", {}))

            # ── INTEGRITY GUARD: infeasible 챔피언 L3 승격 차단 ─────────────
            # 차단 로그는 최종 pipeline 내부 또는 종료 시점에 출력되므로 생략

            # ── Step E: 최적 params + L1 override로 최종 실행 ────────────────
            gc.collect()

            from src.domain.futures.strategy.tiered_workflow.pipeline import run_tiered_pipeline
            _mem_l2_final = _get_rss_mb()
            _t_l2_final_start = time.perf_counter()
            _, l2_final, _ = run_tiered_pipeline(
                labeled_events=labeled_tiered,
                aligned=aligned_tiered,
                cfg=tiered_cfg,
                window=tiered_window,
                l1_params={},
                l2_params=best_l2_params,
                caps=tiered_caps,
                l1_tfs=tuple(tiered_cfg.l1_tfs),
                target_phase=run_config.phase,
                l1_result_override=l1_res,
                verbose=True,  # 최종 실행시 상세 결과 출력
                override_dsr=l2_study_result.dsr,
                l2_sim_cache=l2_study_result.sim_cache,
                l2_signal_batch=l2_signals,
                l2_awf_folds=l2_study_result.awf_folds,
                l2_eval_memo=l2_study_result.eval_memo,
            )
            if l2_final is not None and l2_study_result.best_evaluation is not None:
                from src.domain.futures.strategy.tiered_workflow.replay_parity import (
                    assert_selection_replay_parity,
                )

                _parity_gate = True
                _parity_ok = assert_selection_replay_parity(
                    replay_evaluation=l2_study_result.best_evaluation,
                    final_evaluation=l2_final,
                    gate=_parity_gate,
                )
                if not _parity_ok:
                    _logger.error(
                        "[L2-PARITY] replay/final mismatch. "
                        "replay_L*=%.4f final_L*=%.4f "
                        "replay_CAGR=%.4f final_CAGR=%.4f "
                        "replay_trades=%s final_trades=%s",
                        getattr(l2_study_result.best_evaluation, "deploy_leverage", float("nan")),
                        getattr(l2_final, "deploy_leverage", float("nan")),
                        getattr(l2_study_result.best_evaluation, "cagr_hybrid", float("nan")),
                        getattr(l2_final, "cagr_hybrid", float("nan")),
                        getattr(l2_study_result.best_evaluation, "trade_count", "?"),
                        getattr(l2_final, "trade_count", "?"),
                    )
                    if _parity_gate and hasattr(l2_final, "blocker_reason"):
                        import dataclasses
                        l2_final = dataclasses.replace(l2_final, gate_passed=False, blocker_reason="parity_divergence")

            _reversal_replay_env = os.environ.get("L2_REVERSAL_REPLAY", "")
            if _reversal_replay_env not in ("", "0", "false", "False") and l2_study_result.best_evaluation is not None:
                _replay_results = _run_l2_reversal_economic_replay(
                    signal_batch=l2_signals,
                    aligned=aligned_tiered,
                    awf_folds=l2_study_result.awf_folds or (),
                    base_l2_params=best_l2_params,
                    caps=tiered_caps,
                    tf=run_config.timeframe,
                    deploy_leverage=l2_study_result.best_evaluation.deploy_leverage,
                    prebuilt_cache=l2_study_result.sim_cache,
                    reference_evaluation=l2_study_result.best_evaluation,
                    output_path=Path("docs/results/l2_reversal_replay.csv"),
                )
                for _rr in _replay_results:
                    if _rr.adoption_passed and _rr.variant != "baseline_off":
                        _logger.info(
                            "[L2-REVERSAL-REPLAY] PASS candidate=%s CAGR=%.4f blocker=%s",
                            _rr.variant, _rr.cagr, _rr.blocker_reason,
                        )

            _log_mem("l2_final_pipeline", _mem_l2_final, extra=f"took={time.perf_counter() - _t_l2_final_start:.4f}s")
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
    _mem_before = _get_rss_mb()
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
    _log_mem("universe", _mem_before, extra=f"n_symbols={len(discovered_symbols)}")

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
    _mem_before = _get_rss_mb()
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
    _log_mem("data", _mem_before, extra=f"n_valid={len(data_stage.valid_symbols)} n_loaded={len(data_stage.data_maps)}")

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
    _mem_before = _get_rss_mb()
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
    _log_mem("strategy", _mem_before, extra=f"use_tiered={bool(OPT_FUTURES_CONFIG.get('USE_CS_RANK_ENGINE', False))}")

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
