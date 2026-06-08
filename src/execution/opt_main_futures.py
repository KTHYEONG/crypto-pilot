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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.domain.futures.strategy.regime_evaluation import RegimeScoreCard
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
from src.core.settings import BASE_DIR, FUTURES_DATA_DIR
from src.domain.futures.optimization.observability.run_tracker import (
    build_joint_study_name,
    build_run_id,
    log_optuna_contract,
    resolve_futures_parallel_policy,
    setup_optuna_storage,
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
from src.domain.futures.universe.membership import inject_membership_masks_into_maps
from src.domain.futures.universe.storage import run_historical_sync

_logger = logging.getLogger("opt_main_futures")


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
    """Ensure universe ledger coverage for the required window."""
    ledger_path = FUTURES_DATA_DIR / "universe_ledger.parquet"
    needs_sync = False
    last_ledger_date = date(2023, 1, 1)

    if not ledger_path.exists():
        _logger.info("[SYNC] Ledger missing -> Initiating first sync")
        needs_sync = True
    else:
        try:
            # We only need the 'date' column to check the last coverage
            df_ledger = pd.read_parquet(ledger_path, columns=["date"])
            if df_ledger.empty:
                needs_sync = True
            else:
                last_ledger_date = pd.to_datetime(df_ledger["date"]).max().date()
                # If the ledger doesn't cover up to the required OOS end date, we need more data
                if last_ledger_date < window.end_date_value:
                    _logger.info(
                        "[SYNC] Ledger outdated (Last: %s, Required: %s) -> Syncing...",
                        last_ledger_date,
                        window.end_date_value,
                    )
                    needs_sync = True
        except Exception as e:
            _logger.warning(
                "[SYNC] Verification failed (%s) -> Force sync",
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
            )


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
    ledger_path = FUTURES_DATA_DIR / "universe_ledger.parquet"
    last_ledger_date = date(2023, 1, 1)
    if ledger_path.exists():
        try:
            df_ledger = pd.read_parquet(ledger_path, columns=["date"])
            if not df_ledger.empty:
                last_ledger_date = pd.to_datetime(df_ledger["date"]).max().date()
        except Exception as e:
            _logger.warning(
                "!! SYNC: check_ledger_date failed (%s)", type(e).__name__
            )
    sync_start_date = window.fetch_start_date
    _logger.info(
        "[CACHE] Backfill: %s ~ %s | Symbols: %d | Last: %s",
        sync_start_date,
        window.end_date_value,
        len(symbols),
        last_ledger_date,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--timeframe", type=str, choices=["1h", "4h"], default="4h")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument(
        "--phase",
        type=str,
        choices=["full", "alo", "signal"],
        default="full",
    )
    parser.add_argument(
        "--sync",
        type=str,
        default="full",
        choices=["full", "fast", "skip"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "signal"],
        default="full",
        help=argparse.SUPPRESS,
        dest="mode",
    )
    parser.add_argument("--refresh-universe", action="store_true")
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


def _run_universe_stage(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
) -> tuple[
    list[str],
    dict[date, frozenset[str]],
    tuple[str, ...],
    tuple[str, ...],
    UniverseSnapshot,
    dict[date, frozenset[str]],
]:
    discovered_symbols: list[str] = []
    timeline: dict[date, frozenset[str]] = {}
    inference_timeline: dict[date, frozenset[str]] = {}
    inference_panel: tuple[str, ...] = ()
    live_inference_panel: tuple[str, ...] = ()

    t_discover = time.perf_counter()
    universe_result = discover_universe_timeline(
        tf=run_config.timeframe,
        is_start=window.is_start_date,
        oos_start=window.oos_start_date,
        end_date=window.end_date_value,
        force_rebuild=run_config.refresh_universe,
    )
    _logger.debug(
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
    _logger.debug(
        "[perf-universe] validate_universe_quality took %.4fs",
        time.perf_counter() - t_quality,
    )
    discovered_symbols = list(universe_result.symbols)
    timeline_obj: UniverseMembershipTimeline = universe_result.timeline
    timeline = {
        window.effective_from.date(): frozenset(window.active_symbols)
        for window in timeline_obj.windows
    }
    inference_panel = universe_result.snapshot.inference_panel
    live_inference_panel = universe_result.snapshot.live_inference_panel
    # Stage6 quarterly membership → inference_timeline for dual mask injection
    inf_tl = universe_result.inference_timeline
    if isinstance(inf_tl, UniverseMembershipTimeline):
        inference_timeline = {
            w.effective_from.date(): frozenset(w.active_symbols) for w in inf_tl.windows
        }

    header = f"| {'Metric':<18} | {'Value':<27} |"
    width = len(header)
    title = "[UNIVERSE REPORT] "
    _logger.info("\n" + title + "-" * (width - len(title)))
    _logger.info(header)
    _logger.info(f"| {'-'*18:<18} | {'-'*27:<27} |")
    _logger.info(f"| {'Selected (Stg6)':<18} | {len(universe_result.snapshot.selected):<27} |")
    _logger.info(f"| {'Panels (Inf/Live)':<18} | {f'{len(inference_panel)} / {len(live_inference_panel)}':<27} |")
    _logger.info(f"| {'Windows (Inf)':<18} | {len(inference_timeline):<27} |")
    _logger.info("-" * width)

    return (
        discovered_symbols,
        timeline,
        inference_panel,
        live_inference_panel,
        universe_result.snapshot,
        inference_timeline,
    )


def _run_data_stage(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
    discovered_symbols: list[str],
    timeline: dict[date, frozenset[str]],
    inference_panel: tuple[str, ...] = (),
    live_inference_panel: tuple[str, ...] = (),
    inference_timeline: dict[date, frozenset[str]] | None = None,
) -> DataStageResult:
    load_symbols = _resolve_data_collection_symbols(
        run_config=run_config,
        discovered_symbols=discovered_symbols,
        inference_panel=inference_panel,
        live_inference_panel=live_inference_panel,
    )
    require_exec_1m = _requires_exec_1m(run_config)

    scope_name = "stage6_selected"

    t_load = time.perf_counter()
    data_maps, oos_data_maps, valid_symbols = load_futures_data_maps_for_symbols(
        list(load_symbols),
        run_config.timeframe,
        window.fetch_start,
        window.is_start,
        window.oos_start,
        window.end_date,
        load_exec_1m=require_exec_1m,
        requested_symbols_count=len(load_symbols),
        scope_name=scope_name,
    )
    _logger.debug(
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
        _logger.debug(
            "[perf-data] inject_membership_masks_into_maps took %.4fs",
            time.perf_counter() - t_inject,
        )

    t_ready = time.perf_counter()
    readiness: DataReadinessResult = evaluate_data_readiness(
        tf=run_config.timeframe,
        data_maps=data_maps,
        oos_data_maps=oos_data_maps,
        valid_symbols=valid_symbols,
        fetch_start=window.fetch_start_date,
        is_start=window.is_start_date,
        oos_start=window.oos_start_date,
        end=window.end_date_value,
        require_exec_1m=require_exec_1m,
        scope_name=scope_name,
    )
    _logger.debug("[perf-data] evaluate_data_readiness took %.4fs", time.perf_counter() - t_ready)
    report_df = readiness.report
    fail_reasons: dict[str, int] = {}
    if isinstance(report_df, pd.DataFrame) and not report_df.empty and "pass" in report_df.columns:
        fail_df = report_df.loc[~report_df["pass"].astype(bool)]
        if not fail_df.empty and "reason" in fail_df.columns:
            fail_reasons = {
                str(k): int(v)
                for k, v in fail_df["reason"].value_counts(dropna=False).to_dict().items()
            }
    # We can infer audit metrics from the readiness report
    req_count = len(load_symbols)
    actual_load = len(valid_symbols)
    coverage = actual_load / req_count if req_count > 0 else 0.0

    header = f"| {'Metric':<18} | {'Value':<27} |"
    width = len(header)
    title = "[DATA QUALITY] "
    _logger.info("\n" + title + "-" * (width - len(title)))
    _logger.info(header)
    _logger.info(f"| {'-'*18:<18} | {'-'*27:<27} |")
    _logger.info(f"| {'Symbols (Req/Load)':<18} | {f'{req_count} / {actual_load} ({coverage:.1%})':<27} |")
    _logger.info(f"| {'Kept (Ready)':<18} | {len(readiness.kept_symbols):<27} |")
    
    if fail_reasons:
        reason_str = ", ".join([f"{k}:{v}" for k, v in list(fail_reasons.items())[:2]])
        if len(fail_reasons) > 2:
            reason_str += "..."
        _logger.info(f"| {'Fail Reasons':<18} | {reason_str:<27} |")
    
    _logger.info("-" * width)

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


def _run_strategy_stage(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
    data_stage: DataStageResult,
    inference_panel: tuple[str, ...] = (),
    live_inference_panel: tuple[str, ...] = (),
    trading_symbols: tuple[str, ...] = (),
    universe_snapshot: UniverseSnapshot | None = None,
) -> CandidatePipelineOutput | None:
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
        _logger.debug(
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

    if universe_snapshot is not None:
        t_step = time.perf_counter()
        _inject_universe_metadata_into_maps(
            full_strategy_maps,
            snapshot=universe_snapshot,
            symbols=tuple(full_strategy_maps.keys()),
            tf=run_config.timeframe,
        )
        strategy_steps["metadata"] = time.perf_counter() - t_step

    # inference_panel은 데이터가 실제 로드된 심볼만 필터링하여 전달
    loaded_sym_set = set(data_stage.data_maps.keys())
    effective_inference = tuple(s for s in inference_panel if s in loaded_sym_set) or None
    effective_live = tuple(s for s in live_inference_panel if s in loaded_sym_set) or None
    
    strategy_name = str(OPT_FUTURES_CONFIG.get("FUTURES_STRATEGY_NAME", "candidate_ml"))
    header = f"| {'Component':<18} | {'Status/Value':<27} |"
    width = len(header)
    title = f"[STRATEGY: {strategy_name}] "
    _logger.info("\n" + title + "-" * (width - len(title)))
    _logger.info(header)
    _logger.info(f"| {'-'*18:<18} | {'-'*27:<27} |")
    _logger.info(f"| {'Inf Panel':<18} | {f'{len(effective_inference or ())} symbols':<27} |")
    _logger.info(f"| {'Live Panel':<18} | {f'{len(effective_live or ())} symbols':<27} |")
    _logger.info(f"| {'Trade Symbols':<18} | {len(trading_symbols or tuple(data_stage.valid_symbols)):<27} |")
    _logger.info("-" * width)

    bridge_trading_symbols = list(trading_symbols or data_stage.valid_symbols)

    t_bridge_start = time.perf_counter()
    ml_out = run_active_strategy_output_bridge(
        run_config=run_config,
        symbols=bridge_trading_symbols,
        tf=run_config.timeframe,
        fetch_start=window.fetch_start,
        end_date=window.end_date,
        opt_config=OPT_FUTURES_CONFIG,
        preloaded_data_maps=full_strategy_maps,
        training_panel=trading_symbols or tuple(data_stage.valid_symbols),
        inference_panel=effective_inference,
        live_inference_panel=effective_live,
        trading_symbols=trading_symbols or tuple(data_stage.valid_symbols),
        silent=False,
    )
    bridge_elapsed = time.perf_counter() - t_bridge_start
    strategy_steps["bridge"] = bridge_elapsed
    
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
    if failure_report:
        n_blocked = failure_report.get("n_blocked", 0)
        failure_counts = failure_report.get("failure_counts", {})
        mapping = {
            "min_obs": "Low Obs", "mean_edge": "Low Mean Edge", "median_edge": "Low Median Edge",
            "p10_edge": "Poor P10 Edge", "q10_fail": "High Q10 Fail", "event_density": "Event Overload",
            "regime_edge": "Regime Filter", "edge_decay": "Edge Decay", "hit_or_payoff": "Poor Hit/Payoff",
            "exit_policy": "Exit Policy",
        }
        readable_failures = " | ".join([f"{mapping.get(k, k)} ({v})" for k, v in sorted(failure_counts.items())])
        top_blocked = failure_report.get("top_blocked_str", "none")
        _logger.info(f"| {'BLOCKED':<12} | {n_blocked:<5} | Fail Reasons: {readable_failures[:40]:<40} |")
        _logger.info(f"| {'':<12} | {'':<5} | Top Blocked: {top_blocked[:41]:<41} |")
    if keep_variants:
        _logger.info(f"| {'RECOMMENDED':<12} | {len(keep_variants):<5} | 1. {keep_variants[0]:<52} |")
        for i, var in enumerate(keep_variants[1:], 2):
            _logger.info(f"| {'(Eligible)':<12} | {'':<5} | {i}. {var:<52} |")
    else:
        _logger.info(f"| {'RECOMMENDED':<12} | {'0':<5} | {'none':<52} |")
    _logger.info("-" * 82)

    # 2. Cause: Walk-Forward Performance
    wf_details = candidate_report.get("wf_fold_details", [])
    if wf_details and run_config.phase in {"full", "alo"}:
        _logger.info("\n[WALK-FORWARD FOLD DETAILS]")
        _logger.info("-" * 82)
        _logger.info(
            f"| {'Fold':<4} | {'Mode':<10} | {'Rank IC':>8} | {'Events':>7} | "
            f"{'PriorP90':>8} | {'EU_p90':>7} | {'Pass':<6} |"
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
            prior = res.get("prior_bps", 0.0)
            eu_p90 = res.get("eu_p90", 0.0)
            passed = res.get("pass_cost", False)
            pass_str = "✅" if passed else "❌"
            _logger.info(
                f"| {fold_id:<4} | {mode:<10} | {rank_ic_str} | {events:>7,} | "
                f"{prior:>8.2f} | {eu_p90:>7.2f} | {pass_str:<6} |"
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
    _logger.info(f"| {'Execution Time':<18} | {f'{bridge_elapsed:.2f}s':<27} |")
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
    _logger.debug(
        "[PROFILE][STAGE] merge_candidate_output_into_is_and_oos took %.4fs",
        strategy_steps["merge"],
    )

    # 4. Post-Analysis: Evaluation/Ablation
    candidate_report_ref = getattr(ml_out, "rule_report", {}) or {}
    if isinstance(candidate_report_ref, dict) and candidate_report_ref.get("zero_reason") == "signal_only_mode":
        sv_list = candidate_report_ref.get("signal_validation", [])
        _logger.info("\n[SIGNAL VALIDATION: FILTERING IMPACT]")
        _logger.info("-" * 122)
        _logger.info(
            f"| {'Variant':<22} | {'Events':>8} | {'Hit Rate':>10} | "
            f"{'Mean Edge':>10} | {'Status':<15} | {'Fail Reasons':<44} |"
        )
        _logger.info("-" * 122)
        mean_unfiltered = mean_filtered = 0.0
        for sv in sv_list:
            v_name = str(sv.get("variant", "unknown"))
            if v_name == "rule_only_equal_size":
                display_name, mean_unfiltered = "rule_only (Raw)", float(sv.get("net_edge_bps_mean", 0.0))
            elif v_name == "rule_promo_no_leak":
                display_name, mean_filtered = "rule_promo (Filter)", float(sv.get("net_edge_bps_mean", 0.0))
            else:
                display_name = v_name[:22]
            status_text = "✅ PASS" if sv.get("survives_cost") else "❌ FAIL"
            fail_text = ",".join(str(v) for v in sv.get("fail_reasons", [])) or "-"
            _logger.info(
                f"| {display_name:<22} | {sv.get('n_events', 0):>8,} | "
                f"{float(sv.get('hit_rate', 0.0))*100:>9.1f}% | "
                f"{float(sv.get('net_edge_bps_mean', 0.0)):>6.1f} bps | "
                f"{status_text:<14} | {fail_text:<44} |"
            )
        _logger.info("-" * 122)
        _logger.info(
            ">> Conclusion: Filtering improved Mean Edge by %+0.1f bps. "
            "Proceeding to Ensemble phase.\n",
            mean_filtered - mean_unfiltered,
        )
        _emit_strategy_profile()
        return ml_out

    if run_config.phase in {"full", "alo"}:
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
        "candidate_ml_full": "Cand. Ensemble",
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
    header = f"| {'Parameter':<18} | {'Value':<27} |"
    width = len(header)
    title = "[OPTIMIZATION] "
    _logger.info("\n" + title + "-" * (width - len(title)))
    _logger.info(header)
    _logger.info(f"| {'-'*18:<18} | {'-'*27:<27} |")
    _logger.info(f"| {'Target Symbols':<18} | {len(data_stage.valid_symbols):<27} |")
    _logger.info(f"| {'Total Trials':<18} | {int(run_config.trials):<27} |")
    _logger.info(f"| {'Parallel Workers':<18} | {safe_workers_b:<27} |")
    _logger.info("-" * width)

    t_opt = time.perf_counter()
    opt_res = run_optimization(opt_req)
    opt_elapsed = time.perf_counter() - t_opt
    _logger.info(f"[OPTIMIZE] Optimization complete in {opt_elapsed:.2f}s")
    
    precompute_profile = getattr(opt_res.base_ctx, "precompute_profile", None)
    if isinstance(precompute_profile, dict):
        _logger.info(
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
        if run_config.phase == "full" and strategy_name in {"candidate_ml", "rule_baseline"}:
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
    elapsed_window = time.perf_counter() - t_window
    header = f"| {'Property':<18} | {'Value':<27} |"
    width = len(header)
    title = "[WINDOW] "
    _logger.info(title + "-" * (width - len(title)))
    _logger.info(header)
    _logger.info(f"| {'-'*18:<18} | {'-'*27:<27} |")
    _logger.info(f"| {'Range':<18} | {f'{window.fetch_start} ~ {window.end_date}':<27} |")
    _logger.info(f"| {'IS Start':<18} | {window.is_start:<27} |")
    _logger.info(f"| {'OOS Start':<18} | {window.oos_start:<27} |")
    _logger.info(f"| {'Elapsed':<18} | {f'{elapsed_window:.2f}s':<27} |")
    _logger.info("-" * width)

    # Step 1.5) Ensure universe ledger is synchronized for the required window
    t_sync = time.perf_counter()
    _ensure_universe_ledger_sync(run_config, window)
    _logger.debug("[SYNC] Completed in %.2fs", time.perf_counter() - t_sync)

    # Step 2) universe timeline/quality gate
    t_universe = time.perf_counter()
    (
        discovered_symbols,
        timeline,
        inference_panel,
        live_inference_panel,
        universe_snapshot,
        inference_timeline,
    ) = _run_universe_stage(run_config, window)
    elapsed_universe = time.perf_counter() - t_universe
    _logger.info(f"[UNIVERSE] Discovery complete: {len(discovered_symbols)} symbols ({elapsed_universe:.2f}s)")

    # Log selected symbols in a clean grid
    selected_sorted = sorted(discovered_symbols)
    chunks = [selected_sorted[i : i + 6] for i in range(0, len(selected_sorted), 6)]
    grid_width = 52
    title = "[SELECTED SYMBOLS] "
    _logger.info("\n" + title + "-" * (grid_width - len(title)))
    for chunk in chunks:
        _logger.info(f"| {', '.join(f'{s:<7}' for s in chunk):<48} |")
    _logger.info("-" * grid_width)

    resolved_load_symbols = _resolve_data_collection_symbols(
        run_config=run_config,
        discovered_symbols=discovered_symbols,
        inference_panel=inference_panel,
        live_inference_panel=live_inference_panel,
    )
    _ensure_cached_symbol_data_for_targets(
        run_config,
        window,
        resolved_load_symbols,
        require_exec_1m=_requires_exec_1m(run_config),
    )
    # Step 3) data loading + readiness
    _t_data = time.perf_counter()
    data_stage = _run_data_stage(
        run_config,
        window,
        discovered_symbols,
        timeline,
        inference_panel,
        live_inference_panel,
        inference_timeline,
    )
    # Step 3.5) regime evaluation (between universe and signal)
    regime_stage_result = _run_regime_evaluation_stage(run_config, data_stage)

    # Step 4) strategy bridge + alpha contract
    t_strategy = time.perf_counter()
    _strategy_name = str(OPT_FUTURES_CONFIG.get("FUTURES_STRATEGY_NAME", "candidate_ml"))
    strategy_out = _run_strategy_stage(
        run_config,
        window,
        data_stage,
        inference_panel,
        live_inference_panel,
        _selected_symbols_from_snapshot(universe_snapshot),
        universe_snapshot=universe_snapshot,
    )
    _logger.info("<< STRATEGY: %.2fs", time.perf_counter() - t_strategy)

    # Step 4.5) C3/C4 gold standard: 전략 이벤트로 scorecard 갱신
    if regime_stage_result is not None and strategy_out is not None:
        _refresh_regime_c34_gold_standard(*regime_stage_result, strategy_out)
    if run_config.phase == "signal":
        return RunnerResult(exit_code=0, reason="signal_mode_done")
    if run_config.phase == "alo":
        _logger.info(
            "[PHASE] phase=alo completed strategy/candidate evaluation only; optimization/training skipped"
        )
        return RunnerResult(exit_code=0, reason="candidate_evaluation_done")
    # Step 5) optimization + final OOS evaluation
    _logger.info(
        ">> OPTIMIZE: n=%d trials=%d",
        len(data_stage.valid_symbols),
        int(run_config.trials),
    )
    t_opt_stage = time.perf_counter()
    result = _run_optimization_stage(
        run_config,
        window,
        data_stage,
        seed=seed,
        resume=resume,
    )
    _logger.info("<< OPTIMIZE: %.2fs", time.perf_counter() - t_opt_stage)
    _logger.info(
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
