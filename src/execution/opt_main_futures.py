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
from typing import Any

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
    run_historical_sync(
        start_date=sync_start_date,
        end_date=window.end_date_value,
        sync_mode=run_config.sync_mode,
        symbols=list(symbols),
        sync_1d=True,
        sync_4h=True,
        sync_1m=False,
    )
    if require_exec_1m:
        run_historical_sync(
            start_date=sync_start_date,
            end_date=window.end_date_value,
            sync_mode=run_config.sync_mode,
            symbols=list(symbols),
            sync_1d=False,
            sync_4h=False,
            sync_1m=True,
        )


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

    universe_result = discover_universe_timeline(
        tf=run_config.timeframe,
        is_start=window.is_start_date,
        oos_start=window.oos_start_date,
        end_date=window.end_date_value,
        force_rebuild=run_config.force_universe_rebuild,
    )
    if not validate_universe_quality(
        snapshot=universe_result.snapshot,
        report=universe_result.report,
        reference_date=run_config.reference_date,
        tf=run_config.timeframe,
    ):
        raise RuntimeError("universe_quality_rejected")
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
    if require_exec_1m:
        missing_1m = [s for s in valid_symbols if "exec_1m" not in (data_maps.get(s) or {})]
        if missing_1m:
            raise RuntimeError(
                f"exec_mode=intrarar_1m but {len(missing_1m)} symbol(s) missing 1m data: "
                f"{missing_1m[:5]}{'...' if len(missing_1m) > 5 else ''}"
            )
    if timeline and valid_symbols:
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
) -> None:
    """Run and print alpha quality metrics and horizon sweep like phase0_alpha_eval.py."""
    from src.domain.futures.strategy.alpha_evaluation import evaluate_alpha, sweep_horizon_breakeven

    alpha_panel = getattr(ml_out, "alpha_panel", None)
    if alpha_panel is None or alpha_panel.empty:
        _logger.error("!! FAIL: alpha_panel is empty — no OOS fold")
        return

    # realized returns matching
    panel_reset = alpha_panel.reset_index()
    panel_reset["datetime"] = pd.to_datetime(panel_reset["datetime"], utc=True)
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

    # rank_cs_neutral: apply rank-based selection to SCOREBOARD evaluation
    # if rank_score columns are present in panel, apply quantile selection
    if "rank_score_long" in panel_reset.columns and "rank_score_short" in panel_reset.columns:
        from src.domain.futures.forecast.compose import _cs_zscore
        from src.domain.futures.strategy.config import StrategyMLConfig
        _ml_cfg = StrategyMLConfig()
        if _ml_cfg.post_cost_admission_mode == "rank_cs_neutral":
            _q = float(_ml_cfg.rank_select_quantile)
            _pivot_rs_long = panel_reset.pivot(
                index="datetime", columns="symbol", values="rank_score_long"
            ).reindex(columns=common_syms).fillna(0.0)
            _pivot_rs_short = panel_reset.pivot(
                index="datetime", columns="symbol", values="rank_score_short"
            ).reindex(columns=common_syms).fillna(0.0)
            _rs_l = _pivot_rs_long.to_numpy(dtype=np.float64)
            _rs_s = _pivot_rs_short.to_numpy(dtype=np.float64)
            _z_l = _cs_zscore(_rs_l)
            _z_s = _cs_zscore(-_rs_s)  # short: negate so higher z = better short (matches compose.py)  # noqa: E501
            _long_mask = np.zeros_like(_z_l, dtype=bool)
            _short_mask = np.zeros_like(_z_s, dtype=bool)
            for _t in range(_z_l.shape[0]):
                _n = _z_l.shape[1]
                _k = max(1, int(np.ceil(_n * _q)))
                _lrow = _z_l[_t]
                _srow = _z_s[_t]
                _lf = np.isfinite(_lrow)
                _sf = np.isfinite(_srow)
                if _lf.sum() > 0:
                    _idx_l = np.flatnonzero(_lf)
                    _top_l = _idx_l[np.argsort(_lrow[_idx_l])[::-1][:_k]]
                    _long_mask[_t, _top_l] = True
                if _sf.sum() > 0:
                    _idx_s = np.flatnonzero(_sf)
                    _top_s = _idx_s[np.argsort(_srow[_idx_s])[::-1][:_k]]
                    _short_mask[_t, _top_s] = True
            # Use z-score as signal (not raw EV) — EV is near-zero OOS and kills breadth
            pivot_long = pd.DataFrame(
                np.where(_long_mask, _z_l, 0.0),
                index=pivot_long.index, columns=common_syms,
            )
            pivot_short = pd.DataFrame(
                np.where(_short_mask, _z_s, 0.0),
                index=pivot_short.index, columns=common_syms,
            )
            _logger.info(
                "[RANK-SCOREBOARD] rank_cs_neutral applied: q=%.2f long_nz=%.3f short_nz=%.3f",
                _q,
                float(np.count_nonzero(_long_mask) / max(_long_mask.size, 1)),
                float(np.count_nonzero(_short_mask) / max(_short_mask.size, 1)),
            )

    # get horizon from config
    horizon = int(OPT_FUTURES_CONFIG.get("label_horizon_bars", 12))

    realized_rows: dict[str, pd.Series] = {}
    for sym in common_syms:
        df = data_stage.data_maps[sym][tf].set_index("datetime")
        close = df["close"].reindex(pivot_long.index)
        fwd_ret = np.log(close.shift(-horizon) / close)
        realized_rows[sym] = fwd_ret
    realized_df = pd.DataFrame(realized_rows, index=pivot_long.index)

    common_idx = pivot_long.index.intersection(realized_df.index)
    al = pivot_long.loc[common_idx].to_numpy(dtype=np.float64)
    as_ = pivot_short.loc[common_idx].to_numpy(dtype=np.float64)
    real = realized_df.loc[common_idx].to_numpy(dtype=np.float64)

    al = al[:-horizon]
    as_ = as_[:-horizon]
    real = real[:-horizon]

    eth_close = data_stage.data_maps.get("ETHUSDT", {}).get(tf, pd.DataFrame())
    if eth_close is not None and not eth_close.empty:
        eth_close_ser = eth_close.set_index("datetime")["close"].reindex(common_idx).ffill()
        btc_close_1d = eth_close_ser.iloc[:-horizon].to_numpy(dtype=np.float64)
    else:
        btc_close_1d = None

    # Phase 0: IC Decomposition Diagnostic (dense C1 전체 심볼, C3 마스크 포함)
    from src.domain.futures.strategy.alpha_evaluation import diagnose_alpha_ic_decomposition

    # C1 전체 심볼 dense alpha — C3 필터·regime gate 미적용
    all_syms = [s for s in panel_reset["symbol"].unique() if s in data_stage.data_maps]
    dense_long_df = panel_reset.pivot(
        index="datetime", columns="symbol", values="alpha_long"
    ).reindex(columns=all_syms).fillna(0.0)
    dense_short_df = panel_reset.pivot(
        index="datetime", columns="symbol", values="alpha_short"
    ).reindex(columns=all_syms).fillna(0.0)

    dense_long_df.index = pd.to_datetime(dense_long_df.index, utc=True)
    dense_short_df.index = pd.to_datetime(dense_short_df.index, utc=True)

    c1_real_rows: dict[str, pd.Series] = {}
    for sym in all_syms:
        df_sym = data_stage.data_maps[sym][tf].set_index("datetime")
        close_sym = df_sym["close"].reindex(dense_long_df.index)
        c1_real_rows[sym] = np.log(close_sym.shift(-horizon) / close_sym)
    c1_real_df = pd.DataFrame(c1_real_rows, index=dense_long_df.index)

    diag_common_idx = dense_long_df.index.intersection(c1_real_df.index)
    dense_long_arr = dense_long_df.loc[diag_common_idx].iloc[:-horizon].to_numpy(dtype=np.float64)
    dense_short_arr = dense_short_df.loc[diag_common_idx].iloc[:-horizon].to_numpy(dtype=np.float64)
    dense_pred = dense_long_arr - dense_short_arr
    c1_real_arr = c1_real_df.loc[diag_common_idx].iloc[:-horizon].to_numpy(dtype=np.float64)

    # C3 mask: trading_set 기반
    c3_mask = np.array([s in trading_set for s in all_syms], dtype=np.bool_)

    _ic_decomp = diagnose_alpha_ic_decomposition(
        pred_dense_2d=dense_pred,
        realized_raw_2d=c1_real_arr,
        beta_2d=None,
        market_fwd_1d=None,
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
    from src.domain.futures.strategy.labels import _compute_trailing_beta
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

    # Pass dense pre-clip signed signal (C1) when shapes align with C3 panel
    _dense_pred_for_eval: np.ndarray | None = (
        dense_pred if dense_pred.shape == al.shape else None
    )
    report = evaluate_alpha(
        alpha_long_2d=al,
        alpha_short_2d=as_,
        realized_fwd_ret_2d=real_resid,
        inference_signed_2d=_dense_pred_for_eval,
        btc_close_1d=btc_close_1d,
        n_trials=1,
        horizon_bars=horizon,
    )

    # Compact Alpha Scoreboard
    def pass_emoji(condition: bool) -> str:
        return "✅" if condition else "❌"

    _logger.info("\n📊 [ALPHA SCOREBOARD]")
    _logger.info(
        "Metric |  NET_IC  |  T-STAT  |  BRDTH   |   DSR    |  BE_IC(%dh)",
        horizon
    )
    _logger.info(
        "Value  | %7.4f  | %7.2f  | %7.2f  | %7.4f  |  %7.4f",
        report.net_ic,
        report.ic_t_stat_nw,
        report.effective_breadth,
        report.deflated_sharpe,
        report.breakeven_ic
    )
    _logger.info(
        "Result |    %s    |    %s    |    %s    |    %s    |  (gap=%+5.1f)",
        pass_emoji(report.net_ic >= 0.03),
        pass_emoji(report.ic_t_stat_nw >= 2.0),
        pass_emoji(report.effective_breadth >= 3.0),
        pass_emoji(report.deflated_sharpe >= 0.95),
        (report.net_ic - report.breakeven_ic) * 10000.0
    )
    _infer_stat = report.metrics_by_panel.get("inference_stat", {})
    if _infer_stat:
        _logger.info(
            "📊 [C1-STAT]  NET_IC=%7.4f  T-STAT=%7.2f  BRDTH=%7.2f  DSR=%7.4f",
            _infer_stat.get("net_ic", float("nan")),
            _infer_stat.get("ic_t_stat_nw", float("nan")),
            _infer_stat.get("effective_breadth", float("nan")),
            report.deflated_sharpe,
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
            close = df["close"].reindex(common_idx)
            fwd = np.log(close.shift(-h) / close)
            r_rows[sym] = fwd
        r_df = pd.DataFrame(r_rows, index=common_idx)
        r_arr = r_df.iloc[:-h].to_numpy(dtype=np.float64)
        clip = min(len(al), len(r_arr))
        realized_map[h] = r_arr[:clip]
        alpha_long_map[h] = al[:clip]
        alpha_short_map[h] = as_[:clip]

    sweep = sweep_horizon_breakeven(realized_map, alpha_long_map, alpha_short_map)
    h_sweep_strs = []
    for h in sorted(sweep.keys()):
        h_res = sweep[h]
        pass_s = "✅" if h_res["ic_exceeds_breakeven"] else "❌"
        h_sweep_strs.append(f"[{h}h: ic={h_res['net_ic']:5.3f} {pass_s}]")
    
    _logger.info(f"📈 SWEEP: {' '.join(h_sweep_strs)}")
    _logger.info(f">> ALPHA_PASS: {str(report.passes).upper()}\n")


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
    _logger.info("<< DATA: %.2fs (ok=%d)", time.perf_counter() - t_data, len(data_stage.valid_symbols))  # noqa: E501
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
