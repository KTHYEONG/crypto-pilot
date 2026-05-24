from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so 'src' module is importable from any working directory
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

import src.domain.futures.optimization.opt_config
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
from src.domain.futures.strategy_runtime.bridge import merge_ml_output_into_is_and_oos
from src.domain.futures.universe.membership import inject_membership_masks_into_maps
from src.domain.futures.universe.storage import run_historical_sync

_logger = logging.getLogger("opt_main_futures")


def _ensure_data_sync_for_window(run_config: FuturesRunConfig, window: QuarterlyWindow) -> None:
    """Check if ledger covers the required window and sync if needed."""
    if run_config.skip_data_sync:
        _logger.info("Data sync skipped by config.")
        return

    ledger_path = FUTURES_DATA_DIR / "universe_ledger.parquet"
    needs_sync = False
    last_ledger_date = date(2023, 1, 1)

    if not ledger_path.exists():
        _logger.info("Universe ledger missing. Initializing first-time sync...")
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
                        "Outdated ledger (last=%s < req=%s). Syncing...",
                        last_ledger_date,
                        window.end_date_value,
                    )
                    needs_sync = True
        except Exception as e:
            _logger.warning("Failed to verify ledger readiness (%s). Forcing sync for safety.", e)
            needs_sync = True

    if needs_sync:
        # Sync from the last known date (or default start) up to the end of the required window
        target_symbols: list[str] | None = None
        if run_config.symbols:
            # Include targets, anchors, and macros for seamless readiness
            target_symbols = list(set(
                list(run_config.symbols)
                + FUTURES_ANCHOR_SYMBOLS
                + FUTURES_MACRO_INDEX_SYMBOLS
            ))
        
        # 1m data is massive and sync duration is long.
        # We explicitly set sync_1m=False here to skip 1m data fetch
        # for the entire candidate population.
        # 1m data for the filtered backtest symbols will be targetedly
        # pre-fetched during _run_data_stage.
        run_historical_sync(
            start_date=last_ledger_date,
            end_date=window.end_date_value,
            sync_mode=run_config.sync_mode,
            symbols=target_symbols,
            sync_1d=True,
            sync_4h=True,
            sync_1m=False,
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
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--trials", type=int, default=OPT_FUTURES_CONFIG["total_trials"])
    parser.add_argument("--tf", type=str, choices=["1h", "4h"], default="4h")
    parser.add_argument("--reference-date", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick-backtest", "strategy", "strategy-smoke", "full"],
        default="strategy",
    )
    parser.add_argument(
        "--quick-backtest",
        action="store_true",
        help="Alias for --mode quick-backtest.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ml_lambdamart_v1",
        choices=["momentum_v0", "eh_st_v1", "ml_lambdamart_v1", "xs_reversal"],
    )
    parser.add_argument("--skip-universe", action="store_true")
    parser.add_argument("--skip-data-sync", action="store_true")
    parser.add_argument(
        "--sync-mode",
        type=str,
        default="full_history_master",
        choices=["full_history_master", "elite_fast"],
    )
    parser.add_argument("--force-universe-rebuild", action="store_true")
    parser.add_argument("--bypass-champion-guard", action="store_true")
    parser.add_argument("--alpha-only", action="store_true")
    return parser


def _build_run_config(args: argparse.Namespace) -> FuturesRunConfig:
    payload = vars(args).copy()
    if args.quick_backtest:
        payload["mode"] = "quick-backtest"
    if payload.get("symbols"):
        payload["symbols"] = tuple(str(payload["symbols"]).split(","))
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
) -> tuple[list[str], dict[date, frozenset[str]]]:
    discovered_symbols: list[str] = []
    timeline: dict[date, frozenset[str]] = {}
    if run_config.skip_universe:
        return discovered_symbols, timeline

    universe_result = discover_universe_timeline(
        tf=run_config.tf,
        is_start=window.is_start_date,
        oos_start=window.oos_start_date,
        end_date=window.end_date_value,
        force_rebuild=run_config.force_universe_rebuild,
    )
    if not validate_universe_quality(
        snapshot=universe_result.snapshot,
        report=universe_result.report,
        reference_date=run_config.reference_date,
        tf=run_config.tf,
    ):
        raise RuntimeError("universe_quality_rejected")
    discovered_symbols = list(universe_result.symbols)
    timeline_obj: UniverseMembershipTimeline = universe_result.timeline
    timeline = {
        window.effective_from.date(): frozenset(window.active_symbols)
        for window in timeline_obj.windows
    }
    return discovered_symbols, timeline


def _run_data_stage(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
    discovered_symbols: list[str],
    timeline: dict[date, frozenset[str]],
) -> DataStageResult:
    configured = tuple(
        discovered_symbols or src.domain.futures.optimization.opt_config.FUTURES_SYMBOLS
    )
    target_symbols = list(run_config.symbols or configured)
    load_symbols = list(set(target_symbols + FUTURES_ANCHOR_SYMBOLS + FUTURES_MACRO_INDEX_SYMBOLS))
    require_exec_1m = OPT_FUTURES_CONFIG.get("FUTURES_EXECUTION_MODE") == "intrabar_1m"

    # Targeted Pre-fetch for 1m data:
    # If 1m execution is required and data sync is not skipped,
    # run targeted sync only for the load_symbols which have
    # successfully passed the universe filtering stages.
    if require_exec_1m and not run_config.skip_data_sync:
        _logger.info(
            "Targeted Pre-fetch: Syncing 1m data only for %d universe-filtered symbols.",
            len(load_symbols),
        )
        ledger_path = FUTURES_DATA_DIR / "universe_ledger.parquet"
        last_ledger_date = date(2023, 1, 1)
        if ledger_path.exists():
            try:
                df_ledger = pd.read_parquet(ledger_path, columns=["date"])
                if not df_ledger.empty:
                    last_ledger_date = pd.to_datetime(df_ledger["date"]).max().date()
            except Exception as e:
                _logger.warning("Failed to check ledger date in data stage (%s).", e)
        
        run_historical_sync(
            start_date=last_ledger_date,
            end_date=window.end_date_value,
            sync_mode=run_config.sync_mode,
            symbols=load_symbols,
            sync_1d=False,
            sync_4h=False,
            sync_1m=True,
        )

    data_maps, oos_data_maps, valid_symbols = load_futures_data_maps_for_symbols(
        load_symbols,
        run_config.tf,
        window.fetch_start,
        window.is_start,
        window.oos_start,
        window.end_date,
        load_exec_1m=require_exec_1m,
    )
    if timeline and valid_symbols:
        warmup_bars_required = int(OPT_FUTURES_CONFIG.get("FUTURES_UNIVERSE_WARMUP_BARS", 60))
        inject_membership_masks_into_maps(
            data_maps=data_maps,
            oos_data_maps=oos_data_maps,
            symbols=valid_symbols,
            tf=run_config.tf,
            timeline=timeline,
            warmup_bars_required=warmup_bars_required,
        )

    readiness: DataReadinessResult = evaluate_data_readiness(
        tf=run_config.tf,
        data_maps=data_maps,
        oos_data_maps=oos_data_maps,
        valid_symbols=valid_symbols,
        fetch_start=window.fetch_start_date,
        is_start=window.is_start_date,
        oos_start=window.oos_start_date,
        end=window.end_date_value,
        require_exec_1m=require_exec_1m,
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
) -> None:
    strategy_maps = pick_strategy_data_maps(
        oos_data_maps=data_stage.oos_data_maps,
        is_data_maps=data_stage.data_maps,
        valid_symbols=data_stage.valid_symbols,
        tf=run_config.tf,
    )
    ml_out = run_active_strategy_output_bridge(
        run_config=run_config,
        symbols=data_stage.valid_symbols,
        tf=run_config.tf,
        fetch_start=window.fetch_start,
        end_date=window.end_date,
        opt_config=OPT_FUTURES_CONFIG,
        preloaded_data_maps=(
            strategy_maps if run_config.mode in {"strategy", "strategy-smoke"} else None
        ),
    )
    merge_ml_output_into_is_and_oos(
        ml_out,
        data_stage.data_maps,
        data_stage.oos_data_maps,
        data_stage.valid_symbols,
        run_config.tf,
    )
    if run_config.mode in {"strategy", "strategy-smoke"}:
        assert_strategy_alpha_ready(
            ml_out=ml_out,
            oos_data_maps=data_stage.oos_data_maps,
            valid_symbols=data_stage.valid_symbols,
            tf=run_config.tf,
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
        run_config.tf,
        window.fetch_start,
        window.end_date,
        data_stage.valid_symbols,
        OPT_FUTURES_CONFIG,
        project_root,
    )
    storage_url, storage = setup_optuna_storage(project_root)
    study_name = build_joint_study_name(
        run_config.tf,
        window.fetch_start,
        window.end_date,
        data_stage.valid_symbols,
        OPT_FUTURES_CONFIG,
    )
    import os
    physical_cores = max(1, (os.cpu_count() or 4) // 2)
    # Target 6 workers matching high-performance P-cores (e.g. i5-13600K)
    safe_workers_b = min(6, physical_cores)

    opt_req = OptimizationRequest(
        data_maps=data_stage.data_maps,
        symbols=data_stage.valid_symbols,
        tf=run_config.tf,
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
        strategy_mode=(run_config.strategy is not None),
        strategy_cfg=(
            StrategyConfig(name=run_config.strategy)
            if run_config.strategy is not None
            else None
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
    _logger.info("[STAGE-TIME] step=run_optimization elapsed=%.2fs", opt_elapsed)
    study_ml = opt_res.study_ml
    best_trial = opt_res.best_trial

    # ------------------ PROFILE SUMMARY REPORT ------------------
    try:
        if study_ml is not None:
            import numpy as np
            import optuna
            all_trials = study_ml.get_trials(deepcopy=False)
            valid_trials = [t for t in all_trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
            if valid_trials:
                c_times = [float(t.user_attrs.get("prof_compose", 0.0)) for t in valid_trials if "prof_compose" in t.user_attrs]
                p_times = [float(t.user_attrs.get("prof_prep", 0.0)) for t in valid_trials if "prof_prep" in t.user_attrs]
                pa_times = [float(t.user_attrs.get("prof_prep_align", 0.0)) for t in valid_trials if "prof_prep_align" in t.user_attrs]
                pc_times = [float(t.user_attrs.get("prof_prep_constraint", 0.0)) for t in valid_trials if "prof_prep_constraint" in t.user_attrs]
                e_times = [float(t.user_attrs.get("prof_exec", 0.0)) for t in valid_trials if "prof_exec" in t.user_attrs]
                m_times = [float(t.user_attrs.get("prof_metrics", 0.0)) for t in valid_trials if "prof_metrics" in t.user_attrs]
                mp_times = [float(t.user_attrs.get("prof_metrics_pure", 0.0)) for t in valid_trials if "prof_metrics_pure" in t.user_attrs]
                md_times = [float(t.user_attrs.get("prof_metrics_db_io", 0.0)) for t in valid_trials if "prof_metrics_db_io" in t.user_attrs]
                
                mean_c = float(np.mean(c_times)) if c_times else 0.0
                mean_p = float(np.mean(p_times)) if p_times else 0.0
                mean_pa = float(np.mean(pa_times)) if pa_times else 0.0
                mean_pc = float(np.mean(pc_times)) if pc_times else 0.0
                mean_e = float(np.mean(e_times)) if e_times else 0.0
                mean_m = float(np.mean(m_times)) if m_times else 0.0
                mean_mp = float(np.mean(mp_times)) if mp_times else 0.0
                mean_md = float(np.mean(md_times)) if md_times else 0.0
                total_mean = mean_c + mean_p + mean_e + mean_m
                
                if total_mean > 0:
                    _logger.info("=" * 60)
                    _logger.info(" [PROFILING SUMMARY] Strategy Backtest Performance Profiling (n_trials=%d)", len(valid_trials))
                    _logger.info("  1. Signal Compose   : %6.2f ms (%5.1f%%)", mean_c * 1000.0, (mean_c / total_mean) * 100.0)
                    _logger.info("  2. Backtest Prep    : %6.2f ms (%5.1f%%)", mean_p * 1000.0, (mean_p / total_mean) * 100.0)
                    _logger.info("     - Data Align     : %6.2f ms (%5.1f%%)", mean_pa * 1000.0, (mean_pa / total_mean) * 100.0)
                    _logger.info("     - Constraint Ck  : %6.2f ms (%5.1f%%)", mean_pc * 1000.0, (mean_pc / total_mean) * 100.0)
                    _logger.info("  3. Numba Execution  : %6.2f ms (%5.1f%%)", mean_e * 1000.0, (mean_e / total_mean) * 100.0)
                    _logger.info("  4. Metrics/Pruning  : %6.2f ms (%5.1f%%)", mean_m * 1000.0, (mean_m / total_mean) * 100.0)
                    _logger.info("     - Pure Calc      : %6.2f ms (%5.1f%%)", mean_mp * 1000.0, (mean_mp / total_mean) * 100.0)
                    _logger.info("     - Redis DB I/O   : %6.2f ms (%5.1f%%)", mean_md * 1000.0, (mean_md / total_mean) * 100.0)
                    _logger.info("  * Total Backtest/Tr : %6.2f ms", total_mean * 1000.0)
                    _logger.info("=" * 60)
    except Exception as e:
        _logger.warning("Failed to calculate profile summary: %s", e)
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
            t for t in study_ml.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        ensemble_results = [
            {"trial": t, "params": t.params} for t in completed_trials
        ]

    final_req = FinalEvaluationRequest(
        tf=run_config.tf,
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
    _logger.info("[STAGE-TIME] step=final_evaluation elapsed=%.2fs", final_eval_elapsed)
    _logger.info(
        "[STAGE-TIME] step=optimization_stage_total elapsed=%.2fs",
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
        "[STAGE] step=window fetch=%s is=%s oos=%s end=%s",
        window.fetch_start,
        window.is_start,
        window.oos_start,
        window.end_date,
    )
    _logger.info("[STAGE-TIME] step=window elapsed=%.2fs", time.perf_counter() - t_window)
    # Step 1.5) Ensure data is synchronized for the required window
    t_sync = time.perf_counter()
    _ensure_data_sync_for_window(run_config, window)
    _logger.info("[STAGE-TIME] step=data_sync elapsed=%.2fs", time.perf_counter() - t_sync)
    # Step 2) universe timeline/quality gate
    t_universe = time.perf_counter()
    discovered_symbols, timeline = _run_universe_stage(run_config, window)
    _logger.info(
        "[STAGE] step=universe skip=%s discovered=%d timeline_windows=%d",
        run_config.skip_universe,
        len(discovered_symbols),
        len(timeline),
    )
    _logger.info("[STAGE-TIME] step=universe elapsed=%.2fs", time.perf_counter() - t_universe)
    # Step 3) data loading + readiness
    t_data = time.perf_counter()
    data_stage = _run_data_stage(run_config, window, discovered_symbols, timeline)
    _logger.info(
        "[STAGE] step=data valid=%d require_exec_1m=%s",
        len(data_stage.valid_symbols),
        OPT_FUTURES_CONFIG.get("FUTURES_EXECUTION_MODE") == "intrabar_1m",
    )
    _logger.info("[STAGE-TIME] step=data elapsed=%.2fs", time.perf_counter() - t_data)
    # Step 4) strategy bridge + alpha contract
    t_strategy = time.perf_counter()
    _logger.info(
        "[STAGE] step=strategy mode=%s strategy=%s strategy_mode=%s",
        run_config.mode,
        run_config.strategy,
        run_config.strategy is not None,
    )
    _run_strategy_stage(run_config, window, data_stage)
    _logger.info("[STAGE-TIME] step=strategy elapsed=%.2fs", time.perf_counter() - t_strategy)
    if run_config.mode == "strategy-smoke":
        return RunnerResult(exit_code=0, reason="strategy_smoke_done")
    # Step 5) optimization + final OOS evaluation
    _logger.info(
        "[STAGE] step=optimize symbols=%d trials=%d",
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
    _logger.info("[STAGE-TIME] step=optimize elapsed=%.2fs", time.perf_counter() - t_opt_stage)
    _logger.info(
        "[STAGE-TIME] step=pipeline_total elapsed=%.2fs",
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
        _logger.error("invalid_args: %s", exc)
        return 2
    try:
        result = run_pipeline(
            run_config,
            seed=int(args.seed),
            resume=bool(args.resume),
        )
    except RuntimeError as exc:
        _logger.error("runner_failed: reason=%s", str(exc))
        return 1
    except Exception:
        _logger.exception("runner_failed: unexpected_error")
        return 1
    if result.exit_code != 0:
        _logger.error("runner_failed: reason=%s", result.reason)
    return result.exit_code


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    return run_from_cli()


if __name__ == "__main__":
    raise SystemExit(main())
