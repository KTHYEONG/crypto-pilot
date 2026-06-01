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
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from datetime import date, datetime
from typing import Any, TypedDict

import numpy as np
import optuna
import pandas as pd

# Suppress noisy system warnings for clean output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", message="no explicit representation of timezones")

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
    assert_strategy_alpha_ready,
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
    MLPipelineOutput,
    merge_ml_output_into_is_and_oos,
    run_ml_pipeline_for_universe,
)
from src.domain.futures.universe.membership import inject_membership_masks_into_maps
from src.domain.futures.universe.storage import run_historical_sync

_logger = logging.getLogger("opt_main_futures")


class AlphaPhase1Verdict(TypedDict):
    alpha_pass: bool
    fail_reasons: list[str]
    blocker_categories: dict[str, list[str]]
    policy_no_trade: bool
    resid_ic: float
    resid_t_stat_nw: float
    be_eff: float
    gap_eff: float
    bear_ic: float
    bear_pass: bool
    dsr: float
    port_ic: float
    be_raw: float
    gap_raw: float
    clip_preservation_ratio: float
    basket_net_bps: float
    basket_ir_t: float
    sweep_pass_count: int
    bear_basket_net_bps: float
    gate_results: dict[str, bool]


class ExecDiagVerdict(TypedDict):
    status: str
    fail_reasons: list[str]
    port_ic: float
    be_raw: float
    gap_raw: float
    basket_net_bps: float


def _summarize_alpha_phase1_verdict(
    report: Any,
    *,
    basket_net_bps: float,
    basket_ir_t: float,
    sweep_pass_count: int,
    bear_basket_net_bps: float = float("nan"),
) -> AlphaPhase1Verdict:
    """Unified alpha acceptance verdict: G0+G1 + G2 (economic) + G3 (robustness)."""
    bear_ic = float(report.per_regime_ic.get("bear", float("nan")))
    be_eff = float(report.breakeven_ic_eff)
    resid_ic = float(report.resid_ic)
    resid_t = float(getattr(report, "resid_t_stat_nw", float("nan")))
    gap_eff = resid_ic - be_eff if np.isfinite(resid_ic) and np.isfinite(be_eff) else float("nan")
    port_ic = float(report.net_ic)
    be_raw = float(report.breakeven_ic_eff)  # N_eff 기준으로 통일 (G1과 동일 기준)
    gap_raw = port_ic - be_raw if np.isfinite(port_ic) and np.isfinite(be_raw) else float("nan")
    clip_pres = float(getattr(report, "clip_preservation_ratio", float("nan")))

    # G1: 신호 스킬 (evaluate_alpha 내부에서 판정)
    # sweep≥2 AND DSR≥0.95 조합이 t-stat 단독 실패를 override: 복수 horizon 통과+DSR은
    # 단일 NW t-stat보다 더 robust한 signal 존재 증거 (alpha5.md §2.2 근거).
    _dsr = float(getattr(report, "dsr", float("nan")))
    _t_nw = float(getattr(report, "ic_t_stat_nw", float("nan")))
    _sweep_override = sweep_pass_count >= 2 and np.isfinite(_dsr) and _dsr >= 0.95
    _t_stat_fail_overridden = (
        "signal_t_stat_too_low" in report.fail_reasons
        and _sweep_override
        and np.isfinite(_t_nw) and _t_nw >= 2.0
    )
    g1_pass = bool(report.passes) or _t_stat_fail_overridden

    # G2: 경제 거래성
    portfolio_ic_above_breakeven = bool(np.isfinite(gap_raw) and gap_raw > 0.0)
    basket_evaluated = not bool(getattr(report, "policy_no_trade", False))
    _policy_val_lcb = float(getattr(report, "policy_validation_net_lcb_bps", float("nan")))
    _policy_val_ir = float(getattr(report, "policy_validation_ir_t", float("nan")))
    _basket_direct_ok = (
        np.isfinite(basket_net_bps) and basket_net_bps > 0.0
        and np.isfinite(basket_ir_t) and basket_ir_t >= 2.0
    )
    # Policy validation uses turnover-weighted cost — more accurate than basket spread ir_t.
    _policy_basket_ok = (
        np.isfinite(_policy_val_lcb) and _policy_val_lcb > 0.0
        and np.isfinite(_policy_val_ir) and _policy_val_ir >= 2.0
    )
    basket_net_positive = bool(True if not basket_evaluated else (_basket_direct_ok or _policy_basket_ok))
    signal_preserved_after_selection = bool(np.isfinite(clip_pres) and clip_pres >= 0.7)
    multi_horizon_sweep_passes = bool(sweep_pass_count >= 1)

    # G3: 강건성
    bear_market_basket_safe = bool(
        not np.isfinite(bear_basket_net_bps) or bear_basket_net_bps >= 0.0
    )

    gate_results: dict[str, bool] = {
        "signal_skill_passes":              g1_pass,
        "portfolio_ic_above_breakeven":     portfolio_ic_above_breakeven,
        "basket_net_positive":              basket_net_positive,
        "signal_preserved_after_selection": signal_preserved_after_selection,
        "multi_horizon_sweep_passes":       multi_horizon_sweep_passes,
        "bear_market_basket_safe":          bear_market_basket_safe,
    }

    fail_reasons: list[str] = list(report.fail_reasons)
    if not portfolio_ic_above_breakeven:
        fail_reasons.append("portfolio_ic_below_raw_breakeven")
    if basket_evaluated and not basket_net_positive:
        fail_reasons.append("basket_net_not_profitable")
    if not signal_preserved_after_selection:
        fail_reasons.append("signal_lost_after_selection")
    if not multi_horizon_sweep_passes:
        fail_reasons.append("no_profitable_horizon_found")
    if not bear_market_basket_safe:
        fail_reasons.append("bear_market_basket_negative")

    # Categorize blockers for ALPHA_PASS=false diagnostics.
    reason_category_map: dict[str, str] = {
        "signal_below_effective_breakeven": "rank_skill",
        "signal_t_stat_too_low": "statistical_robustness",
        "policy_economics.validation_net_lcb_non_positive": "policy_economics",
        "portfolio_ic_below_raw_breakeven": "policy_economics",
        "signal_lost_after_selection": "policy_economics",
        "no_profitable_horizon_found": "policy_economics",
        "basket_net_lcb_non_positive": "execution_realism",
        "basket_net_not_profitable": "execution_realism",
        "bear_regime_ic_negative": "execution_realism",
        "bear_market_basket_negative": "execution_realism",
        "deflated_sharpe_too_low": "statistical_robustness",
        "quantile_coverage_out_of_range": "statistical_robustness",
        "long_nz_below_threshold": "mechanical_integrity",
        "short_nz_below_threshold": "mechanical_integrity",
        "xs_long_preservation_below_threshold": "rank_skill",
        "xs_short_preservation_below_threshold": "rank_skill",
        "tradable_long_nz_below_threshold": "execution_realism",
        "tradable_short_nz_below_threshold": "execution_realism",
        "alpha_p95_below_cost_wall": "policy_economics",
    }
    blocker_categories: dict[str, list[str]] = {
        "mechanical_integrity": [],
        "rank_skill": [],
        "policy_economics": [],
        "execution_realism": [],
        "statistical_robustness": [],
    }
    for reason in fail_reasons:
        category = reason_category_map.get(reason)
        if category is not None and reason not in blocker_categories[category]:
            blocker_categories[category].append(reason)

    alpha_pass = (
        g1_pass
        and portfolio_ic_above_breakeven
        and basket_net_positive
        and signal_preserved_after_selection
        and multi_horizon_sweep_passes
        and bear_market_basket_safe
    )

    return {
        "alpha_pass": alpha_pass,
        "fail_reasons": fail_reasons,
        "blocker_categories": blocker_categories,
        "policy_no_trade": bool(getattr(report, "policy_no_trade", False)),
        "resid_ic": resid_ic,
        "resid_t_stat_nw": resid_t,
        "be_eff": be_eff,
        "gap_eff": gap_eff,
        "bear_ic": bear_ic,
        "bear_pass": not (np.isfinite(bear_ic) and bear_ic < 0.0),
        "dsr": float(report.deflated_sharpe),
        "port_ic": port_ic,
        "be_raw": be_raw,
        "gap_raw": gap_raw if np.isfinite(gap_raw) else float("nan"),
        "clip_preservation_ratio": clip_pres,
        "basket_net_bps": basket_net_bps,
        "basket_ir_t": basket_ir_t,
        "sweep_pass_count": sweep_pass_count,
        "bear_basket_net_bps": bear_basket_net_bps,
        "gate_results": gate_results,
    }


def _summarize_exec_diag_verdict(
    *,
    report: Any,
    basket_net_bps: float,
) -> ExecDiagVerdict:
    """Execution realism diagnostic separated from robust alpha extraction verdict."""
    port_ic = float(report.net_ic)
    be_raw = float(report.breakeven_ic)
    gap_raw = port_ic - be_raw if np.isfinite(port_ic) and np.isfinite(be_raw) else float("nan")
    fail_reasons: list[str] = []
    if not np.isfinite(port_ic):
        fail_reasons.append("portfolio_ic_missing")
    elif port_ic <= 0.0:
        fail_reasons.append("portfolio_ic_not_positive")
    if np.isfinite(gap_raw) and gap_raw <= 0.0:
        fail_reasons.append("portfolio_ic_below_raw_breakeven")
    if np.isfinite(basket_net_bps) and basket_net_bps <= 0.0:
        fail_reasons.append("basket_net_returns_negative")
    status = "PASS" if not fail_reasons else "FAIL"
    if not np.isfinite(port_ic) and not np.isfinite(basket_net_bps):
        status = "UNKNOWN"
    return {
        "status": status,
        "fail_reasons": fail_reasons,
        "port_ic": port_ic,
        "be_raw": be_raw,
        "gap_raw": gap_raw,
        "basket_net_bps": basket_net_bps,
    }


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
    """Return whether this run mode requires execution-grade 1m data."""
    if run_config.mode in {"alpha", "strategy"}:
        return False
    exec_mode_cfg = str(OPT_FUTURES_CONFIG.get("FUTURES_EXECUTION_MODE", "coarse"))
    return exec_mode_cfg == "intrabar_1m"


def _ensure_universe_ledger_sync(run_config: FuturesRunConfig, window: QuarterlyWindow) -> None:
    """Ensure universe ledger coverage for the required window."""
    ledger_path = FUTURES_DATA_DIR / "universe_ledger.parquet"
    needs_sync = False
    last_ledger_date = date(2023, 1, 1)

    if not ledger_path.exists():
        _logger.info("🔄 SYNC: missing -> init_sync")
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
                        "🔄 SYNC: outdated (last=%s) required=%s -> sync",
                        last_ledger_date,
                        window.end_date_value,
                    )
                    needs_sync = True
        except Exception as e:
            _logger.warning(
                "⚠️ SYNC: verify failed (%s) -> force_sync",
                type(e).__name__,
            )
            needs_sync = True

    if needs_sync:
        if run_config.sync_mode == "skip":
            _logger.info("🔄 SYNC: skip -> skip synchronization as requested")
        else:
            run_historical_sync(
                start_date=last_ledger_date,
                end_date=window.end_date_value,
                sync_mode=run_config.sync_mode,
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
    scope = StrategyConfig(name="lambdamart").ml.training_universe_scope
    if scope == "historical_stage5_union" and inference_panel:
        base_symbols = list(inference_panel)
    elif scope == "stage5_passed" and live_inference_panel:
        base_symbols = list(live_inference_panel)
    else:
        base_symbols = list(discovered_symbols)

    merged_symbols = (
        base_symbols
        + list(FUTURES_ANCHOR_SYMBOLS)
        + list(FUTURES_MACRO_INDEX_SYMBOLS)
    )
    load_symbols = tuple(dict.fromkeys(merged_symbols))
    return load_symbols


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
    if run_config.sync_mode == "skip":
        _logger.info(".. CACHE: skip backfill as requested")
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
        ".. CACHE: backfill %s ~ %s (targets=%d, last=%s)",
        sync_start_date,
        window.end_date_value,
        len(symbols),
        last_ledger_date,
    )
    t_sync_main = time.perf_counter()
    run_historical_sync(
        start_date=sync_start_date,
        end_date=window.end_date_value,
        sync_mode=run_config.sync_mode,
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
            sync_mode=run_config.sync_mode,
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
    parser.add_argument("--tf", type=str, choices=["1h", "4h"], default="4h")
    parser.add_argument("--reference-date", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick-backtest", "strategy", "alpha", "full"],
        default="strategy",
    )
    parser.add_argument(
        "--sync-mode",
        type=str,
        default="full_history_master",
        choices=["full_history_master", "elite_fast", "skip"],
    )
    parser.add_argument("--force-universe-rebuild", action="store_true")
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
    tuple[str, ...],
    dict[date, frozenset[str]],
]:
    discovered_symbols: list[str] = []
    timeline: dict[date, frozenset[str]] = {}
    inference_timeline: dict[date, frozenset[str]] = {}
    inference_panel: tuple[str, ...] = ()
    live_inference_panel: tuple[str, ...] = ()
    selected_symbols: tuple[str, ...] = ()

    t_discover = time.perf_counter()
    universe_result = discover_universe_timeline(
        tf=run_config.timeframe,
        is_start=window.is_start_date,
        oos_start=window.oos_start_date,
        end_date=window.end_date_value,
        force_rebuild=run_config.force_universe_rebuild,
    )
    _logger.debug(
        "[perf-universe] discover_universe_timeline took %.4fs",
        time.perf_counter() - t_discover,
    )

    t_quality = time.perf_counter()
    if not validate_universe_quality(
        snapshot=universe_result.snapshot,
        report=universe_result.report,
        reference_date=run_config.reference_date,
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
    # Stage5 quarterly membership → inference_timeline for dual mask injection
    inf_tl = universe_result.inference_timeline
    if isinstance(inf_tl, UniverseMembershipTimeline):
        inference_timeline = {
            w.effective_from.date(): frozenset(w.active_symbols) for w in inf_tl.windows
        }
    _logger.info(
        ".. UNIVERSE: panels(inf=%d, live=%d) selected=%d windows=%d",
        len(inference_panel),
        len(live_inference_panel),
        len(universe_result.snapshot.selected),
        len(inference_timeline),
    )
    selected_symbols = tuple(
        str(meta.symbol).strip()
        for meta in universe_result.snapshot.selected
        if str(meta.symbol).strip()
    )
    return (
        discovered_symbols,
        timeline,
        inference_panel,
        live_inference_panel,
        selected_symbols,
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

    if inference_panel:
        scope_name = "historical_stage5_union"
    elif live_inference_panel:
        scope_name = "stage5_passed"
    else:
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
    _logger.info(
        ".. DATA: ok=%d keep=%d fail=%s",
        len(valid_symbols),
        len(readiness.kept_symbols),
        fail_reasons,
    )
    valid_symbols = list(readiness.kept_symbols)
    if not valid_symbols:
        raise RuntimeError("data_not_ready")
    return DataStageResult(
        data_maps=readiness.filtered_is_maps,
        oos_data_maps=readiness.filtered_oos_maps,
        valid_symbols=valid_symbols,
    )


def _run_strategy_stage(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
    data_stage: DataStageResult,
    inference_panel: tuple[str, ...] = (),
    live_inference_panel: tuple[str, ...] = (),
    trading_symbols: tuple[str, ...] = (),
) -> None:
    strategy_maps = pick_strategy_data_maps(
        oos_data_maps=data_stage.oos_data_maps,
        is_data_maps=data_stage.data_maps,
        valid_symbols=data_stage.valid_symbols,
        tf=run_config.timeframe,
    )
    # C1/C2 inference panel이 있으면 해당 심볼 데이터도 strategy_maps에 포함
    all_inference_syms = list(
        dict.fromkeys(
            list(inference_panel or live_inference_panel) + list(data_stage.valid_symbols)
        )
    )
    if run_config.mode in {"strategy", "alpha"} and (
        inference_panel or live_inference_panel
    ):
        full_strategy_maps = pick_strategy_data_maps(
            oos_data_maps=data_stage.oos_data_maps,
            is_data_maps=data_stage.data_maps,
            valid_symbols=all_inference_syms,
            tf=run_config.timeframe,
        )
    else:
        full_strategy_maps = strategy_maps

    # inference_panel은 데이터가 실제 로드된 심볼만 필터링하여 전달
    loaded_sym_set = set(data_stage.data_maps.keys())
    effective_inference = tuple(s for s in inference_panel if s in loaded_sym_set) or None
    effective_live = tuple(s for s in live_inference_panel if s in loaded_sym_set) or None
    _logger.info(
        ".. STRATEGY: panels(inf=%d, live=%d) trade=%d",
        len(effective_inference or ()),
        len(effective_live or ()),
        len(trading_symbols or tuple(data_stage.valid_symbols)),
    )
    bridge_trading_symbols = list(trading_symbols or data_stage.valid_symbols)

    # alpha 모드에서는 gate 실패 시에도 평가 리포트를 출력하기 위해 예외를 캡처
    _alpha_gate_error: Exception | None = None
    t_bridge_start = time.perf_counter()
    try:
        _logger.debug(
            "[latency] Starting run_active_strategy_output_bridge for alpha/strategy mode"
        )
        ml_out = run_active_strategy_output_bridge(
            run_config=run_config,
            symbols=bridge_trading_symbols,
            tf=run_config.timeframe,
            fetch_start=window.fetch_start,
            end_date=window.end_date,
            opt_config=OPT_FUTURES_CONFIG,
            preloaded_data_maps=(
                full_strategy_maps
                if run_config.mode in {"strategy", "alpha"}
                else None
            ),
            inference_panel=effective_inference,
            live_inference_panel=effective_live,
            trading_symbols=trading_symbols or tuple(data_stage.valid_symbols),
        )
        _logger.debug(
            "[latency] Completed run_active_strategy_output_bridge: %.4fs",
            time.perf_counter() - t_bridge_start,
        )
    except RuntimeError as _e:
        _logger.debug(
            "[latency] Exception in active_strategy_output_bridge after %.4fs: %s",
            time.perf_counter() - t_bridge_start,
            str(_e),
        )
        if run_config.mode == "alpha" and ("alpha gate" in str(_e) or "quality gate" in str(_e)):
            _alpha_gate_error = _e
            _logger.warning("⚠️ ALPHA: gate failed (evaluation continues) -> %s", _e)
            # gate 실패 시 ml_pipeline에서 이미 panel.attrs에 quality_report가 저장됨
            # bridge에서 예외 전 ml_out을 얻을 수 없으므로 gate 없이 재실행

            _s_cfg = StrategyConfig(name="lambdamart")
            if effective_inference:
                _eff_syms = list(effective_inference)
            elif effective_live:
                _eff_syms = list(effective_live)
            else:
                _eff_syms = bridge_trading_symbols
            _updated_ml = _dc_replace(
                _s_cfg.ml,
                trading_symbols=tuple(bridge_trading_symbols),
                alpha_gate_min_long_nz=0.0,
                alpha_gate_min_short_nz=0.0,
                alpha_gate_min_xs_preservation=0.0,
                alpha_gate_min_tradable_long_nz=0.0,
                alpha_gate_min_tradable_short_nz=0.0,
                alpha_gate_cost_wall_tolerance_bps=9999.0,
            )
            _s_cfg = _dc_replace(_s_cfg, ml=_updated_ml)
            t_bypass_start = time.perf_counter()
            try:
                _logger.debug(
                    "[latency] Starting fallback run_ml_pipeline_for_universe (gate bypass)"
                )
                ml_out = run_ml_pipeline_for_universe(
                    symbols=_eff_syms,
                    tf=run_config.timeframe,
                    fetch_start=window.fetch_start,
                    end_date=window.end_date,
                    opt_config=OPT_FUTURES_CONFIG,
                    strategy_cfg=_s_cfg,
                    preloaded_data_maps=full_strategy_maps,
                )
                _logger.debug(
                    "[latency] Completed fallback run_ml_pipeline_for_universe: %.4fs",
                    time.perf_counter() - t_bypass_start,
                )
            except RuntimeError as _e2:
                # alpha 모드에서 gate-bypass 재실행도 quality gate 실패 시
                # 평가 리포트 출력을 위해 MLPipelineOutput 빈 객체 사용
                _logger.warning(
                    "⚠️ ALPHA: gate-bypass rerun also failed (%s) -> evaluation with empty ml_out",
                    _e2,
                )
                ml_out = MLPipelineOutput()
        else:
            raise
    
    t_merge_start = time.perf_counter()
    merge_ml_output_into_is_and_oos(
        ml_out,
        data_stage.data_maps,
        data_stage.oos_data_maps,
        data_stage.valid_symbols,
        run_config.timeframe,
    )
    _logger.debug(
        "[latency] merge_ml_output_into_is_and_oos: %.4fs",
        time.perf_counter() - t_merge_start,
    )

    if run_config.mode == "strategy":
        assert_strategy_alpha_ready(
            ml_out=ml_out,
            oos_data_maps=data_stage.oos_data_maps,
            valid_symbols=data_stage.valid_symbols,
            tf=run_config.timeframe,
        )
    elif run_config.mode == "alpha":
        t_report_start = time.perf_counter()
        _logger.debug("[latency] Starting _run_alpha_evaluation_report")
        _run_alpha_evaluation_report(
            ml_out, data_stage, run_config.timeframe,
            trading_symbols or tuple(data_stage.valid_symbols),
            trials=int(run_config.trials),
        )
        _logger.debug(
            "[latency] Completed _run_alpha_evaluation_report: %.4fs",
            time.perf_counter() - t_report_start,
        )


def _run_alpha_evaluation_report(
    ml_out: Any,
    data_stage: DataStageResult,
    tf: str,
    trading_symbols: tuple[str, ...] = (),
    trials: int = 1,
) -> None:
    """Run and print alpha quality metrics and horizon sweep like phase0_alpha_eval.py."""
    from src.domain.futures.strategy.alpha_evaluation import (
        derive_signed_rank_signal,
        evaluate_alpha,
        sweep_horizon_breakeven,
    )

    alpha_panel = getattr(ml_out, "alpha_panel", None)
    if alpha_panel is None or alpha_panel.empty:
        _logger.error("!! FAIL: alpha_panel is empty — no OOS fold")
        return

    # realized returns matching
    panel_reset = alpha_panel.reset_index()
    panel_reset["datetime"] = pd.to_datetime(panel_reset["datetime"], utc=True).dt.tz_convert(None)
    pivot_long = panel_reset.pivot(index="datetime", columns="symbol", values="alpha_long")
    pivot_short = panel_reset.pivot(index="datetime", columns="symbol", values="alpha_short")
    # C3 trading_symbols만 평가 — non-trading 심볼의 alpha=0이 IC를 왜곡하는 것을 방지
    trading_set = set(trading_symbols) if trading_symbols else set()
    common_syms = [
        s for s in pivot_long.columns
        if s in data_stage.data_maps and (not trading_set or s in trading_set)
    ]
    pivot_long = pivot_long[common_syms].fillna(0.0)
    pivot_short = pivot_short[common_syms].fillna(0.0)

    # L1: C1 전체 심볼 dense rank score (unclipped, raw — 랭커 순수 예측력 측정)
    _pivot_rank_l_c1: pd.DataFrame | None = None
    _pivot_rank_s_c1: pd.DataFrame | None = None
    _all_syms_with_data = [s for s in panel_reset["symbol"].unique() if s in data_stage.data_maps]
    if "rank_score_long" in panel_reset.columns and "rank_score_short" in panel_reset.columns:
        _pivot_rank_l_c1 = (
            panel_reset.pivot(index="datetime", columns="symbol", values="rank_score_long")
            .reindex(columns=_all_syms_with_data)
            .fillna(0.0)
        )
        _pivot_rank_s_c1 = (
            panel_reset.pivot(index="datetime", columns="symbol", values="rank_score_short")
            .reindex(columns=_all_syms_with_data)
            .fillna(0.0)
        )

    # L2: C3 필터링된 rank score (cost-adjusted evaluation용)
    _pivot_rank_l_c3: pd.DataFrame | None = None
    _pivot_rank_s_c3: pd.DataFrame | None = None
    if "rank_score_long" in panel_reset.columns and "rank_score_short" in panel_reset.columns:
        _pivot_rank_l_c3 = (
            panel_reset.pivot(index="datetime", columns="symbol", values="rank_score_long")
            .reindex(columns=common_syms)
            .fillna(0.0)
        )
        _pivot_rank_s_c3 = (
            panel_reset.pivot(index="datetime", columns="symbol", values="rank_score_short")
            .reindex(columns=common_syms)
            .fillna(0.0)
        )

    # rank_cs_neutral: apply rank-based selection to SCOREBOARD evaluation
    # if rank_score columns are present in panel, apply quantile selection
    if "rank_score_long" in panel_reset.columns and "rank_score_short" in panel_reset.columns:
        from src.domain.futures.forecast.compose import _cs_zscore
        from src.domain.futures.strategy.config import StrategyMLConfig
        from src.domain.futures.strategy.rank_selection import apply_rank_selection_policy, policy_from_dict
        _ml_cfg = StrategyMLConfig()
        if _ml_cfg.post_cost_admission_mode == "rank_cs_neutral":
            _q = float(_ml_cfg.rank_select_quantile)
            _pivot_rs_long_raw = panel_reset.pivot(
                index="datetime", columns="symbol", values="rank_score_long"
            ).reindex(columns=common_syms)
            _pivot_rs_short_raw = panel_reset.pivot(
                index="datetime", columns="symbol", values="rank_score_short"
            ).reindex(columns=common_syms)
            _finite_mask = np.isfinite(_pivot_rs_long_raw.to_numpy()) & np.isfinite(_pivot_rs_short_raw.to_numpy())

            _rs_l = _pivot_rs_long_raw.fillna(0.0).to_numpy(dtype=np.float64)
            _rs_s = _pivot_rs_short_raw.fillna(0.0).to_numpy(dtype=np.float64)
            # Phase 1b: NET 신호 기반 단일 선택
            # Phase 0-diag: 별도 L/S 선택이 NET +35.5bps 엣지를 파괴(ew=-12.94bps).
            # NET = rank_score_long - rank_score_short 기준으로 long/short 동시 선택.
            _net_signal = derive_signed_rank_signal(_rs_l, _rs_s)
            _net_signal = np.where(_finite_mask, _net_signal, np.nan)
            _policy_payload = getattr(alpha_panel, "attrs", {}).get("rank_selection_policy")
            if isinstance(_policy_payload, dict):
                from src.domain.futures.strategy.rank_selection import (
                    build_simulation_beta_context,
                    merge_metadata_with_runtime_config,
                )
                _policy_payload = merge_metadata_with_runtime_config(_policy_payload, _ml_cfg)
                _policy = policy_from_dict(_policy_payload)
                _beta_2d = build_simulation_beta_context(
                    common_syms, data_stage.data_maps, tf, pivot_long.index
                )

                _alpha_l, _alpha_s = apply_rank_selection_policy(
                    signed_score_2d=_net_signal,
                    eligible_2d=_finite_mask,
                    policy=_policy,
                    beta_2d=_beta_2d,
                )
                _long_mask = _alpha_l > 0.0
                _short_mask = _alpha_s > 0.0
                pivot_long = pd.DataFrame(_alpha_l, index=pivot_long.index, columns=common_syms)
                pivot_short = pd.DataFrame(_alpha_s, index=pivot_short.index, columns=common_syms)
            else:
                _z_net = _cs_zscore(_net_signal)  # cross-sectional z-score
                _long_mask = np.zeros_like(_z_net, dtype=bool)
                _short_mask = np.zeros_like(_z_net, dtype=bool)
                for _t in range(_z_net.shape[0]):
                    _n = _z_net.shape[1]
                    _k = max(1, int(np.ceil(_n * _q)))
                    _row = _z_net[_t]
                    _fin = np.isfinite(_row)
                    if _fin.sum() > 0:
                        _fin_idx = np.flatnonzero(_fin)
                        _sorted = _fin_idx[np.argsort(_row[_fin_idx])]
                        _long_mask[_t, _sorted[-_k:]] = True
                        _short_mask[_t, _sorted[:_k]] = True
                pivot_long = pd.DataFrame(
                    np.where(_long_mask, _z_net, 0.0),
                    index=pivot_long.index, columns=common_syms,
                )
                pivot_short = pd.DataFrame(
                    np.where(_short_mask, -_z_net, 0.0),
                    index=pivot_short.index, columns=common_syms,
                )
            _logger.info(
                "[RANK-SCOREBOARD] net_signal applied: q=%.2f long_nz=%.3f short_nz=%.3f",
                _q,
                float(np.count_nonzero(_long_mask) / max(_long_mask.size, 1)),
                float(np.count_nonzero(_short_mask) / max(_short_mask.size, 1)),
            )

    # get horizon from config
    horizon = int(OPT_FUTURES_CONFIG.get("label_horizon_bars", 12))

    realized_rows: dict[str, pd.Series] = {}
    for sym in common_syms:
        df = data_stage.data_maps[sym][tf].set_index("datetime")
        fwd_ret = _forward_log_return_on_index(
            df["close"],
            pivot_long.index,
            horizon,
        )
        realized_rows[sym] = fwd_ret
    realized_df = pd.DataFrame(realized_rows, index=pivot_long.index)

    # rank_score_long이 finite인 날짜(OOS fold 예측 구간)만 추출
    # C3 심볼 교집합 전에 전체 심볼 기준으로 OOS 날짜를 확정한다.
    _raw_rs_all = (
        panel_reset.pivot(index="datetime", columns="symbol", values="rank_score_long")
        .reindex(columns=_all_syms_with_data)
    )
    _oos_dt_mask = np.any(np.isfinite(_raw_rs_all.to_numpy()), axis=1)
    _oos_idx = _raw_rs_all.index[_oos_dt_mask]

    common_idx = _oos_idx.intersection(realized_df.index)
    _logger.info(
        "[OOS-DIAG] rank_cols=%d finite_rows=%d oos_idx=%d common_idx=%d | "
        "common_syms[:3]=%s rank_cols[:3]=%s",
        len(_raw_rs_all.columns),
        int(_oos_dt_mask.sum()),
        len(_oos_idx),
        len(common_idx),
        list(common_syms)[:3],
        list(_raw_rs_all.columns)[:3],
    )
    al = pivot_long.loc[common_idx].to_numpy(dtype=np.float64)
    as_ = pivot_short.loc[common_idx].to_numpy(dtype=np.float64)
    real = realized_df.loc[common_idx].to_numpy(dtype=np.float64)

    al = al[:-horizon]
    as_ = as_[:-horizon]
    real = real[:-horizon]

    # L2 C3 rank signal [T-h, N_c3] — dense, unclipped
    _rank_pred_c3: np.ndarray | None = None
    if _pivot_rank_l_c3 is not None and _pivot_rank_s_c3 is not None:
        _rl_c3 = _pivot_rank_l_c3.reindex(common_idx).iloc[:-horizon].to_numpy(dtype=np.float64)
        _rs_c3 = _pivot_rank_s_c3.reindex(common_idx).iloc[:-horizon].to_numpy(dtype=np.float64)
        _rank_pred_c3 = derive_signed_rank_signal(_rl_c3, _rs_c3)
        _logger.info(
            "[RANK-CONTRACT] c3_signed_nz=%.3f c3_signed_std=%.6f c3_long_short_absdiff_p50=%.6f",
            float(
                np.count_nonzero(np.isfinite(_rank_pred_c3) & (_rank_pred_c3 != 0.0))
                / max(_rank_pred_c3.size, 1)
            ),
            float(np.nanstd(_rank_pred_c3)),
            float(np.nanmedian(np.abs(_rl_c3 - _rs_c3))),
        )

    # L1 C1 rank signal [T-h, N_c1] — for separate RANK-QUALITY log
    _rank_pred_c1: np.ndarray | None = None
    _real_c1: np.ndarray | None = None
    if _pivot_rank_l_c1 is not None and _pivot_rank_s_c1 is not None:
        _c1_real_rows_rs: dict[str, pd.Series] = {}
        for _sym_c1 in _all_syms_with_data:
            _df_c1 = data_stage.data_maps[_sym_c1][tf].set_index("datetime")
            _c1_real_rows_rs[_sym_c1] = _forward_log_return_on_index(
                _df_c1["close"],
                common_idx,
                horizon,
            )
        _c1_real_df = pd.DataFrame(_c1_real_rows_rs, index=common_idx)
        _rank_pred_c1 = derive_signed_rank_signal(
            _pivot_rank_l_c1.reindex(common_idx).iloc[:-horizon].to_numpy(dtype=np.float64),
            _pivot_rank_s_c1.reindex(common_idx).iloc[:-horizon].to_numpy(dtype=np.float64),
        )
        _real_c1 = _c1_real_df.iloc[:-horizon].to_numpy(dtype=np.float64)

    eth_close = data_stage.data_maps.get("ETHUSDT", {}).get(tf, pd.DataFrame())
    if eth_close is not None and not eth_close.empty:
        eth_close_ser = eth_close.set_index("datetime")["close"].reindex(common_idx).ffill()
        btc_close_1d = eth_close_ser.iloc[:-horizon].to_numpy(dtype=np.float64)
    else:
        btc_close_1d = None

    # Phase 0: IC Decomposition Diagnostic (dense C1 전체 심볼, C3 마스크 포함)
    from src.domain.futures.strategy.alpha_evaluation import diagnose_alpha_ic_decomposition
    from src.domain.futures.strategy.labels import _compute_trailing_beta

    # C1 전체 심볼 dense alpha — C3 필터·regime gate 미적용
    all_syms = [s for s in panel_reset["symbol"].unique() if s in data_stage.data_maps]
    dense_long_df = panel_reset.pivot(
        index="datetime", columns="symbol", values="alpha_long"
    ).reindex(columns=all_syms).fillna(0.0)
    dense_short_df = panel_reset.pivot(
        index="datetime", columns="symbol", values="alpha_short"
    ).reindex(columns=all_syms).fillna(0.0)

    dense_long_df.index = pd.to_datetime(dense_long_df.index, utc=True).tz_localize(None)
    dense_short_df.index = pd.to_datetime(dense_short_df.index, utc=True).tz_localize(None)

    c1_real_rows: dict[str, pd.Series] = {}
    for sym in all_syms:
        df_sym = data_stage.data_maps[sym][tf].set_index("datetime")
        c1_real_rows[sym] = _forward_log_return_on_index(
            df_sym["close"],
            dense_long_df.index,
            horizon,
        )
    c1_real_df = pd.DataFrame(c1_real_rows, index=dense_long_df.index)

    diag_common_idx = dense_long_df.index.intersection(c1_real_df.index)
    dense_long_arr = dense_long_df.loc[diag_common_idx].iloc[:-horizon].to_numpy(dtype=np.float64)
    dense_short_arr = dense_short_df.loc[diag_common_idx].iloc[:-horizon].to_numpy(dtype=np.float64)
    dense_pred = dense_long_arr - dense_short_arr
    c1_real_arr = c1_real_df.loc[diag_common_idx].iloc[:-horizon].to_numpy(dtype=np.float64)

    # C1 전체 심볼 beta + market_fwd 계산 (resid IC 분해용)
    # C3(common_syms) beta와 별도로 계산해야 shape가 [T_diag, N_c1] 정합
    _c1_close_rows: dict[str, pd.Series] = {}
    for _sym in all_syms:
        _df_c1 = data_stage.data_maps[_sym][tf].set_index("datetime")
        _c1_close_rows[_sym] = _df_c1["close"].reindex(diag_common_idx)
    _c1_close_2d = pd.DataFrame(_c1_close_rows, index=diag_common_idx).to_numpy(dtype=np.float64)
    _c1_beta_full = _compute_trailing_beta(_c1_close_2d)           # [T_diag, N_c1]
    _c1_beta_2d = _c1_beta_full[:-horizon]                         # [T_diag-h, N_c1]
    with np.errstate(all="ignore"):
        _c1_mkt_fwd_raw = np.nanmean(c1_real_arr, axis=1)          # [T_diag-h]
    _c1_market_fwd = np.where(np.isfinite(_c1_mkt_fwd_raw), _c1_mkt_fwd_raw, 0.0)

    # C3 mask: trading_set 기반
    c3_mask = np.array([s in trading_set for s in all_syms], dtype=np.bool_)

    _ic_decomp = diagnose_alpha_ic_decomposition(
        pred_dense_2d=dense_pred,
        realized_raw_2d=c1_real_arr,
        beta_2d=_c1_beta_2d,
        market_fwd_1d=_c1_market_fwd,
        trading_mask_1d=c3_mask if np.any(c3_mask) else None,
        horizon_bars=horizon,
    )
    _logger.info(
        "🔬 [IC-DECOMP] dense_c1_raw=%.4f(hit=%.3f br=%.1f)"
        " dense_c1_resid=%.4f dense_c3_raw=%.4f dense_c3_resid=%.4f",
        _ic_decomp.get("dense_c1_raw_ic", float("nan")),
        _ic_decomp.get("dense_c1_raw_hit", float("nan")),
        _ic_decomp.get("dense_c1_raw_breadth", float("nan")),
        _ic_decomp.get("dense_c1_resid_ic", float("nan")),
        _ic_decomp.get("dense_c3_raw_ic", float("nan")),
        _ic_decomp.get("dense_c3_resid_ic", float("nan")),
    )

    # Phase 1B: SCOREBOARD realized return beta-residualize (타깃 정합)
    # 모델 학습 타깃(beta-residualized)과 측정 타깃 정합 → raw return의 시장 성분 제거
    _close_rows: dict[str, pd.Series] = {}
    for sym in common_syms:
        _df_sym = data_stage.data_maps[sym][tf].set_index("datetime")
        _close_rows[sym] = _df_sym["close"].reindex(common_idx)
    _close_2d = pd.DataFrame(_close_rows, index=common_idx).to_numpy(dtype=np.float64)
    _beta_2d_full = _compute_trailing_beta(_close_2d)           # [T_full, N_c3]
    _beta_2d = _beta_2d_full[:-horizon]                         # [T-h, N_c3]
    with np.errstate(all="ignore"):
        _market_fwd = np.nanmean(real, axis=1)                  # [T-h]
    _market_fwd = np.where(np.isfinite(_market_fwd), _market_fwd, 0.0)
    real_resid = real - _beta_2d * _market_fwd[:, np.newaxis]  # [T-h, N_c3]
    _logger.info(
        "🔧 [1B-RESID] market_fwd_std=%.4f beta_mean=%.3f real_std=%.4f resid_std=%.4f",
        float(np.nanstd(_market_fwd)),
        float(np.nanmean(_beta_2d)),
        float(np.nanstd(real)),
        float(np.nanstd(real_resid)),
    )

    # L1: Dense rank IC (C1 전체, unclipped raw rank score)
    if _rank_pred_c1 is not None and _real_c1 is not None:
        from src.domain.futures.strategy.alpha_evaluation import (
            compute_effective_breadth,
            compute_net_ic,
        )
        from src.domain.futures.strategy.rank_selection import _ema_2d
        _smoothed_c1 = _ema_2d(_rank_pred_c1, span=horizon)
        _r_c1_ic = compute_net_ic(_smoothed_c1, _real_c1, horizon_bars=horizon)
        _r_c1_br = compute_effective_breadth(_smoothed_c1, np.zeros_like(_smoothed_c1))
        _logger.info(
            "🏅 [RANK-QUALITY L1] ic=%.4f t=%.2f hit=%.3f breadth=%.1f"
            " | signed rank score vs forward_gross_ret (C1 dense, unclipped)",
            _r_c1_ic["mean_ic"],
            _r_c1_ic["t_stat_nw"],
            _r_c1_ic["hit_ratio"],
            _r_c1_br,
        )
        _gen_report = getattr(ml_out, "alpha_panel", pd.DataFrame()).attrs.get(
            "generalization_report", {}
        )
        if _gen_report:
            _logger.info(
                "🏅 [RANK-GENERALIZE] oos_rank_ic=%.4f is_rank_ic=%.4f retention=%.2f decision=%s",
                float(_gen_report.get("oos_rank_ic", float("nan"))),
                float(_gen_report.get("is_rank_ic", float("nan"))),
                float(_gen_report.get("retention_ratio", float("nan"))),
                str(_gen_report.get("decision", "unknown")),
            )

    # L2 C3 rank score IC — evaluate_alpha inference_signed_2d로 전달
    # rank_pred_c3는 C3 기준 dense signal이므로 al과 shape 일치
    _dense_pred_for_eval: np.ndarray | None = _rank_pred_c3
    # DSR 정직화: 실제 탐색한 하이퍼파라미터 trials 반영
    _n_folds = int(getattr(ml_out, "n_folds", 2))
    _n_trials_dsr = max(1, trials)

    _opt_q = 0.35
    _policy_payload = getattr(alpha_panel, "attrs", {}).get("rank_selection_policy")
    if isinstance(_policy_payload, dict):
        _opt_q = float(_policy_payload.get("quantile", 0.35))

    report = evaluate_alpha(
        alpha_long_2d=al,
        alpha_short_2d=as_,
        realized_fwd_ret_2d=real_resid,
        inference_signed_2d=_dense_pred_for_eval,
        btc_close_1d=btc_close_1d,
        n_trials=_n_trials_dsr,
        horizon_bars=horizon,
        cost_floor_bps=24.0,  # 24bps 고정 물리 비용 장벽 보존
        basket_quantile=_opt_q,
        policy_validation_net_lcb_bps=float(
            getattr(alpha_panel, "attrs", {}).get("rank_selection_policy", {}).get(
                "validation_net_lcb_bps", float("nan")
            )
        ),
        policy_validation_gross_bps=float(
            getattr(alpha_panel, "attrs", {}).get("rank_selection_policy", {}).get(
                "validation_gross_bps", float("nan")
            )
        ),
        policy_validation_ir_t=float(
            getattr(alpha_panel, "attrs", {}).get("rank_selection_policy", {}).get(
                "validation_ir_t", float("nan")
            )
        ),
        policy_validation_monotonicity=float(
            getattr(alpha_panel, "attrs", {}).get("rank_selection_policy", {}).get(
                "validation_monotonicity", float("nan")
            )
        ),
        policy_validation_turnover=float(
            getattr(alpha_panel, "attrs", {}).get("rank_selection_policy", {}).get(
                "validation_turnover", float("nan")
            )
        ),
        policy_validation_cost_bps=float(
            getattr(alpha_panel, "attrs", {}).get("rank_selection_policy", {}).get(
                "validation_cost_bps", float("nan")
            )
        ),
        policy_validation_breadth=float(
            getattr(alpha_panel, "attrs", {}).get("rank_selection_policy", {}).get(
                "validation_breadth", float("nan")
            )
        ),
        policy_selection_mode=str(
            getattr(alpha_panel, "attrs", {}).get(
                "rank_policy_selection_mode",
                getattr(alpha_panel, "attrs", {}).get("rank_selection_policy", {}).get(
                    "selection_mode", ""
                ),
            )
        ),
        policy_no_trade=bool(getattr(alpha_panel, "attrs", {}).get("rank_policy_no_trade", False)),
        policy_min_breadth=float(getattr(_ml_cfg, "rank_policy_target_breadth_min", 8)),
        policy_max_turnover=float(getattr(_ml_cfg, "rank_policy_max_turnover", 1.25)),
        promotion_min_oos_folds=int(getattr(_ml_cfg, "alpha_promotion_min_oos_folds", 2)),
        observed_oos_folds=int(_n_folds),
    )

    _policy_payload = getattr(alpha_panel, "attrs", {}).get("rank_selection_policy", {})
    _policy_no_trade = bool(getattr(alpha_panel, "attrs", {}).get("rank_policy_no_trade", False))
    _policy_reason = str(
        getattr(alpha_panel, "attrs", {}).get(
            "rank_policy_failure_reason",
            "validation_net_lcb_non_positive" if _policy_no_trade else "none",
        )
    )
    _logger.info(
        "[ALPHA-GATE] alpha_output_unit=%s alpha_cost_wall_required=%s policy_no_trade=%s",
        getattr(_ml_cfg, "alpha_output_unit", "rank_weight"),
        bool(getattr(_ml_cfg, "require_alpha_cost_wall", False)),
        _policy_no_trade,
    )
    _logger.info(
        (
            "[ALPHA-POLICY] policy_no_trade=%s reason=%s val_lcb=%.2f val_ir=%.2f mono=%.2f "
            "pre_ic=%.4f post_ic=%.4f pres=%.4f soft_beta=%s soft_beta_w=%.2f"
        ),
        _policy_no_trade,
        _policy_reason,
        float(_policy_payload.get("validation_net_lcb_bps", float("nan"))),
        float(_policy_payload.get("validation_ir_t", float("nan"))),
        float(_policy_payload.get("validation_monotonicity", float("nan"))),
        float(_policy_payload.get("validation_pre_ic", float("nan"))),
        float(_policy_payload.get("validation_post_ic", float("nan"))),
        float(_policy_payload.get("validation_clip_preservation", float("nan"))),
        bool(_policy_payload.get("soft_beta_neutralize", False)),
        float(_policy_payload.get("soft_beta_neutralize_weight", float("nan"))),
    )
    _logger.info(
        "[ALPHA-POLICY-PORT] mode=%s hold=%s breadth=%.2f turnover=%.2f cost=%.2f net_lcb=%.2f beta=%.4f net=%.4f",
        str(_policy_payload.get("selection_mode", "")),
        str(_policy_payload.get("holding_bars", "")),
        float(_policy_payload.get("validation_breadth", float("nan"))),
        float(_policy_payload.get("validation_turnover", float("nan"))),
        float(_policy_payload.get("validation_cost_bps", float("nan"))),
        float(_policy_payload.get("validation_net_lcb_bps", float("nan"))),
        float(_policy_payload.get("validation_abs_beta_exposure", float("nan"))),
        float(_policy_payload.get("validation_abs_net_exposure", float("nan"))),
    )
    _logger.info(
        "[ALPHA-POLICY-BASKET] gross=%.2f net=%.2f ir=%.2f hit=%.3f strict_cost=%.2f",
        float(_policy_payload.get("validation_basket_gross_bps", float("nan"))),
        float(_policy_payload.get("validation_basket_net_bps", float("nan"))),
        float(_policy_payload.get("validation_basket_ir_t", float("nan"))),
        float(_policy_payload.get("validation_basket_hit", float("nan"))),
        float(getattr(_ml_cfg, "rank_policy_strict_cost_floor_bps", 24.0)),
    )

    # Compact Alpha Scoreboard (Phase 1: resid_ic / N_eff-based breakeven)
    def pass_emoji(condition: bool) -> str:
        return "✅" if condition else "❌"

    _logger.info("\n📊 [ALPHA SCOREBOARD]")
    _logger.info(
        "Metric | RESID_IC |  T-STAT  |  N_EFF   |   DSR    | BE_EFF(%dh) | BEAR_IC",
        horizon
    )
    _logger.info(
        "Value  | %7.4f  | %7.2f  | %7.1f  | %7.4f  |  %7.4f  | %7.4f",
        report.resid_ic,
        report.resid_t_stat_nw,
        report.n_eff,
        report.deflated_sharpe,
        report.breakeven_ic_eff,
        report.per_regime_ic.get("bear", float("nan")),
    )
    _logger.info(
        "Result |    %s    |    %s    |  N_eff   |    %s    |  (gap=%+5.1fbps)  |    %s",
        pass_emoji(report.resid_ic >= report.breakeven_ic_eff),
        pass_emoji(report.resid_t_stat_nw >= 3.0),
        pass_emoji(report.deflated_sharpe >= 0.95),
        (report.resid_ic - report.breakeven_ic_eff) * 10000.0,
        pass_emoji(
            not (
                bool(np.isfinite(report.per_regime_ic.get("bear", float("nan"))))
                and report.per_regime_ic.get("bear", 0.0) < 0.0
            )
        ),
    )
    _logger.info(
        "📊 [PASS=%s] fail=%s | net_ic=%.4f be_raw=%.4f gap_raw=%+.1fbps",
        "✅" if report.passes else "❌",
        report.fail_reasons,
        report.net_ic,
        report.breakeven_ic,
        (report.net_ic - report.breakeven_ic) * 10000.0,
    )
    _infer_stat = report.metrics_by_panel.get("inference_stat", {})
    _rank_ic_c3 = float(_infer_stat.get("net_ic", float("nan")))
    _rank_t_c3 = float(_infer_stat.get("ic_t_stat_nw", float("nan")))
    _rank_br_c3 = float(_infer_stat.get("effective_breadth", float("nan")))
    if _infer_stat:
        _logger.info(
            "📊 [RANK-IC C3] ic=%7.4f  t=%7.2f  lcb=%7.4f  breadth=%7.2f"
            " | signed rank score vs beta-resid (dense, unclipped, C3)",
            _rank_ic_c3, _rank_t_c3, float(report.rank_ic_lcb), _rank_br_c3,
        )

    # Phase 0-diag: Tail-Basket Monotonicity + Beta-Tilt 진단
    # 목적: full-CS IC(+) vs tail-basket(-) 괴리의 selection 원인 정량 확정
    if _dense_pred_for_eval is not None:
        from src.domain.futures.strategy.alpha_evaluation import (
            diagnose_selection_monotonicity,
        )
        _mono = diagnose_selection_monotonicity(
            _dense_pred_for_eval,
            real_resid,
            _beta_2d,
            n_deciles=5,
            horizon_bars=horizon,
        )
        _decile_strs = " | ".join(
            f"Q{d}={_mono.get(f'decile_mean_ret_bps_{d}', float('nan')):+.1f}"
            for d in range(5)
        )
        _logger.info(
            "🔬 [MONOTONICITY] top-bot=%+.1fbps mono_rho=%.2f"
            " beta_tilt=%+.3f (L=%.2f S=%.2f) n=%d",
            _mono["top_minus_bottom_bps"],
            _mono["monotonicity_spearman"],
            _mono["beta_tilt"],
            _mono["long_decile_beta_mean"],
            _mono["short_decile_beta_mean"],
            int(_mono["n_obs"]),
        )
        _logger.info("🔬 [DECILE-RET] %s", _decile_strs)

    _policy_turnover = float(_policy_payload.get("validation_turnover", float("nan")))
    _basket_cost_per_bar = (
        _policy_turnover * 24.0 if np.isfinite(_policy_turnover) and _policy_turnover > 0.0 else 24.0
    )
    from src.domain.futures.strategy.rank_selection import _equal_weight_basket_metrics
    _weights_l3 = al.astype(np.float64) - as_.astype(np.float64)
    _elig_l3 = np.isfinite(_weights_l3) & np.isfinite(real_resid)
    _basket_eq = _equal_weight_basket_metrics(
        weights=_weights_l3,
        realized=real_resid,
        eligible=_elig_l3,
        cost_per_bar_bps=_basket_cost_per_bar,
    )
    _basket = {
        "mean_bps": float(_basket_eq.get("validation_basket_gross_bps", float("nan"))),
        "net_bps": float(_basket_eq.get("validation_basket_net_bps", float("nan"))),
        "ir_t": float(_basket_eq.get("validation_basket_ir_t", float("nan"))),
        "hit": float(_basket_eq.get("validation_basket_hit", float("nan"))),
        "n": float(np.count_nonzero(np.isfinite(_weights_l3).any(axis=1))),
        "mean_bps_zw": float("nan"),
    }
    _logger.info(
        "🧺 [L3-BASKET] ew_bps=%.2f net_bps=%.2f ir_t=%.2f hit=%.3f n=%d"
        " | zw_bps=%.2f(confound) | RANK-IC C3=%.4f",
        _basket.get("mean_bps", float("nan")),
        _basket.get("net_bps", float("nan")),
        _basket.get("ir_t", float("nan")),
        _basket.get("hit", float("nan")),
        int(_basket.get("n", 0.0)),
        _basket.get("mean_bps_zw", float("nan")),
        _rank_ic_c3,
    )

    _logger.info(
        "📊 [C3-EXEC]  NET_IC=%7.4f  T-STAT=%7.2f  BRDTH=%7.2f  BE_IC(%dh)=%7.4f  gap=%+5.1fbps",
        report.net_ic,
        report.ic_t_stat_nw,
        report.effective_breadth,
        horizon,
        report.breakeven_ic,
        (report.net_ic - report.breakeven_ic) * 10000.0,
    )
    
    reg_ics = [f"{r.capitalize()}: {ic:5.3f}" for r, ic in report.per_regime_ic.items()]
    _logger.info(f"🌐 [REGIME IC] {' | '.join(reg_ics)}")

    # sweep horizons
    realized_map: dict[int, np.ndarray] = {}
    alpha_long_map: dict[int, np.ndarray] = {}
    alpha_short_map: dict[int, np.ndarray] = {}
    for h in [6, 12, 18]:
        r_rows: dict[str, pd.Series] = {}
        for sym in common_syms:
            df = data_stage.data_maps[sym][tf].set_index("datetime")
            fwd = _forward_log_return_on_index(df["close"], common_idx, h)
            r_rows[sym] = fwd
        r_df = pd.DataFrame(r_rows, index=common_idx)
        r_arr = r_df.iloc[:-h].to_numpy(dtype=np.float64)
        clip = min(len(al), len(r_arr))
        realized_map[h] = r_arr[:clip]
        alpha_long_map[h] = al[:clip]
        alpha_short_map[h] = as_[:clip]

    # Phase 2: 엄격한 24bps 비용 검증 (상각 없음, 물리적 진실 보존)
    _cost_map = dict.fromkeys([6, 12, 18], 24.0)
    sweep = sweep_horizon_breakeven(
        realized_map, alpha_long_map, alpha_short_map, cost_map=_cost_map
    )
    h_sweep_strs = []
    for h in sorted(sweep.keys()):
        h_res = sweep[h]
        pass_s = "✅" if h_res["ic_exceeds_breakeven"] else "❌"
        h_sweep_strs.append(f"[{h}h: ic={h_res['net_ic']:5.3f} {pass_s}]")
    
    _logger.info(f"📈 SWEEP: {' '.join(h_sweep_strs)}")
    _sweep_pass_n = int(sum(1 for v in sweep.values() if v.get("ic_exceeds_breakeven", 0.0) > 0.0))

    # Bear-only basket 실측 계산
    bear_basket_net_bps = float("nan")
    if btc_close_1d is not None:
        from src.domain.futures.strategy.alpha_evaluation import _compute_regime_labels
        _labels = _compute_regime_labels(btc_close_1d)
        _spread_arr = np.full(len(_labels), np.nan, dtype=np.float64)
        for _t in range(min(len(_labels), _weights_l3.shape[0])):
            _w_row = _weights_l3[_t]
            _r_row = real_resid[_t]
            _mask = np.isfinite(_w_row) & np.isfinite(_r_row)
            _lmask = _mask & (_w_row > 0.0)
            _smask = _mask & (_w_row < 0.0)
            if int(np.count_nonzero(_lmask)) < 1 or int(np.count_nonzero(_smask)) < 1:
                continue
            _spread_arr[_t] = float(np.mean(_r_row[_lmask]) - np.mean(_r_row[_smask]))
        _clip_len = min(len(_labels), len(_spread_arr))
        _bear_indices = [
            t for t in range(_clip_len)
            if _labels[t] == "bear" and np.isfinite(_spread_arr[t])
        ]
        if len(_bear_indices) >= 5:
            _bear_spreads = _spread_arr[_bear_indices]
            bear_basket_net_bps = float(np.mean(_bear_spreads)) * 1e4 - 24.0

    _alpha_verdict = _summarize_alpha_phase1_verdict(
        report,
        basket_net_bps=float(_basket.get("net_bps", float("nan"))),
        basket_ir_t=float(_basket.get("ir_t", float("nan"))),
        sweep_pass_count=_sweep_pass_n,
        bear_basket_net_bps=bear_basket_net_bps,
    )
    _exec_verdict = _summarize_exec_diag_verdict(
        report=report,
        basket_net_bps=float(_basket.get("net_bps", float("nan"))),
    )
    _gate_str = " ".join(
        f"{k}={'OK' if v else 'FAIL'}" for k, v in _alpha_verdict["gate_results"].items()
    )
    _logger.info(
        ">> ALPHA_PASS: %s [%s]"
        " [IC_SKILL: resid_ic=%.4f be_eff=%.4f gap=%+.4f t=%.2f bear_ic=%.4f dsr=%.3f]"
        " [BASKET: gap_raw=%+.4f net_bps=%.1f ir_t=%.2f presv=%.2f sweep=%d/3%s]"
        " [PROMOTION: stage=%s mode=%s breadth=%.2f turnover=%.2f cost=%.2f]"
        " [fail=%s blockers=%s]",
        str(bool(_alpha_verdict["alpha_pass"])).upper(),
        _gate_str,
        float(_alpha_verdict["resid_ic"]),
        float(_alpha_verdict["be_eff"]),
        float(_alpha_verdict["gap_eff"]),
        float(_alpha_verdict["resid_t_stat_nw"]),
        float(_alpha_verdict["bear_ic"]),
        float(_alpha_verdict["dsr"]),
        float(_alpha_verdict["gap_raw"]),
        float(_alpha_verdict["basket_net_bps"]),
        float(_alpha_verdict["basket_ir_t"]),
        float(_alpha_verdict["clip_preservation_ratio"]),
        int(_alpha_verdict["sweep_pass_count"]),
        "" if not _alpha_verdict["policy_no_trade"] else " skipped=no-trade",
        str(report.promotion_stage),
        str(report.policy_selection_mode),
        float(report.policy_validation_breadth),
        float(report.policy_validation_turnover),
        float(report.policy_validation_cost_bps),
        _alpha_verdict["fail_reasons"],
        _alpha_verdict["blocker_categories"],
    )
    _logger.info(
        ">> EXEC_DIAG: %s [port_ic=%.4f be_raw=%.4f gap_raw=%+.4f basket_net_bps=%.2f fail=%s]",
        _exec_verdict["status"],
        _exec_verdict["port_ic"],
        _exec_verdict["be_raw"],
        _exec_verdict["gap_raw"],
        _exec_verdict["basket_net_bps"],
        _exec_verdict["fail_reasons"],
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
        strategy_cfg=StrategyConfig(name="lambdamart"),
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
    _logger.info("[STAGE_TIME] step=run_optimization elapsed_s=%.2f", opt_elapsed)
    precompute_profile = getattr(opt_res.base_ctx, "precompute_profile", None)
    if isinstance(precompute_profile, dict):
        _logger.info(
            (
                "[RUN_PROF] step=ml_precompute total_s=%.2f align_s=%.2f "
                "covariance_s=%.2f awf_refit_s=%.2f calibrator_s=%.2f "
                "prebuilt_s=%.2f legs=%d"
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
                c_times = [
                    float(t.user_attrs.get("prof_compose", 0.0))
                    for t in valid_trials
                    if "prof_compose" in t.user_attrs
                ]
                p_times = [
                    float(t.user_attrs.get("prof_prep", 0.0))
                    for t in valid_trials
                    if "prof_prep" in t.user_attrs
                ]
                pa_times = [
                    float(t.user_attrs.get("prof_prep_align", 0.0))
                    for t in valid_trials
                    if "prof_prep_align" in t.user_attrs
                ]
                pc_times = [
                    float(t.user_attrs.get("prof_prep_constraint", 0.0))
                    for t in valid_trials
                    if "prof_prep_constraint" in t.user_attrs
                ]
                e_times = [
                    float(t.user_attrs.get("prof_exec", 0.0))
                    for t in valid_trials
                    if "prof_exec" in t.user_attrs
                ]
                m_times = [
                    float(t.user_attrs.get("prof_metrics", 0.0))
                    for t in valid_trials
                    if "prof_metrics" in t.user_attrs
                ]
                mp_times = [
                    float(t.user_attrs.get("prof_metrics_pure", 0.0))
                    for t in valid_trials
                    if "prof_metrics_pure" in t.user_attrs
                ]
                md_times = [
                    float(t.user_attrs.get("prof_metrics_db_io", 0.0))
                    for t in valid_trials
                    if "prof_metrics_db_io" in t.user_attrs
                ]

                mean_c = float(np.mean(c_times)) if c_times else 0.0
                mean_p = float(np.mean(p_times)) if p_times else 0.0
                mean_pa = float(np.mean(pa_times)) if pa_times else 0.0
                mean_pc = float(np.mean(pc_times)) if pc_times else 0.0
                mean_e = float(np.mean(e_times)) if e_times else 0.0
                mean_m = float(np.mean(m_times)) if m_times else 0.0
                mean_mp = float(np.mean(mp_times)) if mp_times else 0.0
                mean_md = float(np.mean(md_times)) if md_times else 0.0
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
        # quick-backtest has no active signal source by design, so every trial
        # prunes on zero trades; a clean prune-to-completion is a passing smoke.
        if run_config.mode == "quick-backtest":
            return RunnerResult(exit_code=0, reason="quick_backtest_smoke_no_candidate")
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
    _logger.info("[STAGE_TIME] step=final_evaluation elapsed_s=%.2f", final_eval_elapsed)
    _logger.info(
        "[STAGE_TIME] step=optimization_stage_total elapsed_s=%.2f",
        time.perf_counter() - stage_t0,
    )
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
    window = _resolve_quarterly_window(run_config.reference_date)
    _logger.info(
        ">> WINDOW: %s ~ %s [IS: %s | OOS: %s]",
        window.fetch_start,
        window.end_date,
        window.is_start,
        window.oos_start,
    )
    _logger.info("<< WINDOW: %.2fs", time.perf_counter() - t_window)
    # Step 1.5) Ensure universe ledger is synchronized for the required window
    t_sync = time.perf_counter()
    _ensure_universe_ledger_sync(run_config, window)
    _logger.info("<< SYNC: %.2fs", time.perf_counter() - t_sync)
    # Step 2) universe timeline/quality gate
    t_universe = time.perf_counter()
    (
        discovered_symbols,
        timeline,
        inference_panel,
        live_inference_panel,
        selected_symbols,
        inference_timeline,
    ) = _run_universe_stage(run_config, window)
    _logger.info(
        ">> UNIVERSE: n=%d windows=%d",
        len(discovered_symbols),
        len(timeline),
    )
    _logger.info("<< UNIVERSE: %.2fs", time.perf_counter() - t_universe)
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
    t_data = time.perf_counter()
    data_stage = _run_data_stage(
        run_config,
        window,
        discovered_symbols,
        timeline,
        inference_panel,
        live_inference_panel,
        inference_timeline,
    )
    _logger.info("<< DATA: %.2fs (ok=%d)", time.perf_counter() - t_data, len(data_stage.valid_symbols))
    # Step 4) strategy bridge + alpha contract
    t_strategy = time.perf_counter()
    _logger.info(
        ">> STRATEGY: %s | %s",
        run_config.mode,
        "lambdamart",
    )
    _run_strategy_stage(
        run_config,
        window,
        data_stage,
        inference_panel,
        live_inference_panel,
        selected_symbols,
    )
    _logger.info("<< STRATEGY: %.2fs", time.perf_counter() - t_strategy)
    if run_config.mode == "alpha":
        return RunnerResult(exit_code=0, reason="alpha_evaluation_done")
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
        result = run_pipeline(
            run_config,
            seed=int(args.seed),
            resume=bool(args.resume),
        )
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
