from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

# Project Root Setup
project_root: str = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings  # noqa: E402

import config.opt_config  # noqa: E402
from config.ops_profiles import resolve_ops_profile  # noqa: E402
from config.opt_config import (  # noqa: E402
    FUTURES_ANCHOR_SYMBOLS,
    FUTURES_MACRO_INDEX_SYMBOLS,
    OPT_FUTURES_CONFIG,
    get_quarterly_window,
)
from src.core.utils.utils import setup_logger  # noqa: E402
from src.domain.futures.ml_pipeline import run_ml_pipeline_for_universe  # noqa: E402
from src.domain.futures.ml_pipeline.pipeline_runner import (  # noqa: E402
    merge_ml_output_into_is_and_oos,
)
from src.domain.futures.optimization.candidate_selector import (  # noqa: E402
    check_stability_layer3,
    select_and_rank_candidates,
    select_v43_phase_b_top_candidates,
)
from src.domain.futures.optimization.dashboard import (  # noqa: E402
    log_alpha_component_summary,
    log_hmm_report_summary,
    log_ml_merge_feature_stats,
)
from src.domain.futures.optimization.final_evaluator import (  # noqa: E402
    run_final_oos_evaluation,
)
from src.domain.futures.optimization.opt_data_utils import (  # noqa: E402
    load_futures_data_maps_for_symbols,
)
from src.domain.futures.optimization.optimizer import (  # noqa: E402
    MLPhaseDContext,
    _base_engine_params,
    _cached_kill_fund_lev,
    _run_portfolio_numba_block,
    build_ml_phase_d_params,
    precompute_ml_optimization_context,
)
from src.domain.futures.optimization.phase_runner import (  # noqa: E402
    run_v43_phase_optimization_skeleton,
)
from src.domain.futures.optimization.run_tracker import (  # noqa: E402
    apply_ops_profile_overrides,
    build_joint_study_name,
    build_p7_ops_summary,
    build_run_id,
    resolve_futures_parallel_policy,
    setup_optuna_storage,
)
from src.domain.futures.optimization.trial_observability import (  # noqa: E402
    classify_no_valid_candidates,
)
from src.domain.futures.optimization.validation import (  # noqa: E402
    awf_pos_frac_to_pseudo_pbo,
    resolve_adjusted_gates,
)
from src.domain.futures.universe import load_or_build_universe_snapshot  # noqa: E402

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*Overfitting detector is active.*")

# Force Linux 'fork' method for memory efficiency (CoW)
if sys.platform != "win32":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
_logger: logging.Logger = logging.getLogger("opt_futures")

setup_logger("DataCollector", write_file=False)
setup_logger("BinanceClient", write_file=False)
setup_logger("src.domain.futures.ml_pipeline", write_file=False)
logging.getLogger("DataCollector").setLevel(logging.WARNING)
logging.getLogger("BinanceClient").setLevel(logging.WARNING)
logging.getLogger("src.domain.futures.ml_pipeline").setLevel(logging.INFO)


def _collect_p7_ops_summary(
    *,
    mode: str,
    ml_out: Any,
    study_ml: optuna.Study | None,
    selection_summary: dict[str, Any],
) -> dict[str, Any]:
    alpha_panel = getattr(ml_out, "alpha_panel", None)
    alpha_attrs = getattr(alpha_panel, "attrs", {}) if alpha_panel is not None else {}
    hmm_report = getattr(ml_out, "hmm_report", {}) or {}
    study_attrs = dict(getattr(study_ml, "user_attrs", {}) or {}) if study_ml is not None else {}
    return build_p7_ops_summary(
        mode=mode,
        ml_integrity_report=getattr(ml_out, "integrity_report", {}) or {},
        alpha_filter_meta=dict(alpha_attrs.get("alpha_component_filter", {}) or {}),
        alpha_goal_meta=dict(alpha_attrs.get("alpha_goal_eval_meta", {}) or {}),
        hmm_goal_meta=dict(hmm_report.get("hmm_goal_eval_meta", {}) or {}),
        alpha_cache_meta=dict(alpha_attrs.get("alpha_cache", {}) or {}),
        study_user_attrs=study_attrs,
        selection_summary=selection_summary,
    )


def _log_p7_ops_summary(summary: dict[str, Any]) -> None:
    integrity = summary.get("integrity", {}) if isinstance(summary.get("integrity"), dict) else {}
    alpha = summary.get("alpha", {}) if isinstance(summary.get("alpha"), dict) else {}
    obs = (
        summary.get("optuna_observability", {})
        if isinstance(summary.get("optuna_observability"), dict)
        else {}
    )
    _logger.info(
        " [P7] health=%s mode=%s panel_nan=%.4f prefill_nan=%.4f alpha_survive=%d/%d "
        "elite_zero=%s no_candidate=%s reasons=%s",
        str(summary.get("health_status", "unknown")),
        str(summary.get("mode", "unknown")),
        float(integrity.get("panel_nan_pct", 0.0) or 0.0),
        float(integrity.get("panel_prefill_nan_pct", 0.0) or 0.0),
        int(alpha.get("n_surviving", 0) or 0),
        int(alpha.get("n_components", 0) or 0),
        str(bool(alpha.get("elite_zero_after_survival", False))).lower(),
        str(obs.get("no_candidate_reason", "") or "-"),
        ",".join(str(x) for x in (summary.get("reason_codes", []) or [])) or "-",
    )








def _discover_symbols_via_universe(
    *,
    tf: str,
    reference_date: str | None,
    force_rebuild: bool = False,
) -> tuple[list[str], Any, pd.DataFrame]:
    """Build/load point-in-time universe snapshot and return selected symbols."""
    _fetch_start, _start, as_of_date, _end = get_quarterly_window(reference_date)

    if force_rebuild:
        from src.domain.futures.universe import build_universe
        snapshot, selected_frame, report = build_universe(
            as_of=as_of_date,
            tf=tf,
        )
    else:
        snapshot, selected_frame, report = load_or_build_universe_snapshot(
            as_of=as_of_date,
            tf=tf,
        )

    selected_symbols: list[str] = []
    if selected_frame is not None and not selected_frame.empty and "symbol" in selected_frame.columns:
        selected_symbols = [
            str(symbol).strip() for symbol in selected_frame["symbol"].astype(str).tolist() if str(symbol).strip()
        ]
    if not selected_symbols:
        selected_symbols = [
            str(meta.symbol).strip() for meta in snapshot.selected if str(meta.symbol).strip()
        ]
    return list(dict.fromkeys(selected_symbols)), snapshot, report


def validate_universe_quality(
    snapshot: Any,
    report: pd.DataFrame,
    reference_date: str | None,
    tf: str,
) -> bool:
    """Evaluate universe quality against objective metrics and enforce hard-stop."""
    from src.domain.futures.universe import load_universe_snapshot
    from src.domain.futures.universe.contracts import RejectCode

    _logger.info("\n [QUALITY] UNIVERSE QUALITY GATE CHECK")
    _logger.info(" ─────────────────────────────────────────────────────────────────────────────────────")

    if not snapshot.selected:
        _logger.error(" [!] FAIL: No symbols selected in universe snapshot.")
        return False

    # 1. Cost & Capacity Base
    costs = [m.execution_cost_bps for m in snapshot.selected]
    advs = [m.adv_usdt for m in snapshot.selected]
    
    median_cost = float(np.median(costs))
    median_adv = float(np.median(advs))
    
    _logger.info(" [1] Cost & Capacity Base:")
    _logger.info("     - Median Execution Cost: %.2f bps (Gate: <= 50.0)", median_cost)
    _logger.info("     - Median ADV: %s USDT (Gate: >= 25M)", f"{median_adv:,.0f}")

    cost_pass = median_cost <= 50.0
    adv_pass = median_adv >= 25_000_000.0

    # 2. Unexpected Forced Dropout Rate
    # Find previous quarter's snapshot to calculate dropout rate
    ref_dt = datetime.now().date()
    if reference_date:
        ref_dt = datetime.strptime(reference_date, "%Y-%m-%d").date()
    
    from dateutil.relativedelta import relativedelta
    prev_quarter_dt = ref_dt - relativedelta(months=3)
    _, _, prev_as_of, _ = get_quarterly_window(prev_quarter_dt.isoformat())
    
    previous_snapshot_frame = load_universe_snapshot(as_of=prev_as_of, tf=tf)
    dropout_pass = True
    dropout_rate = 0.0

    if previous_snapshot_frame is not None and not previous_snapshot_frame.empty:
        prev_symbols = set(previous_snapshot_frame["symbol"].tolist())
        curr_symbols = set(m.symbol for m in snapshot.selected)
        
        dropped_symbols = prev_symbols - curr_symbols
        forced_dropouts = 0
        
        for sym in dropped_symbols:
            filt_report = snapshot.rejected.get(sym)
            if filt_report:
                # Forced if rejected in Stage 1-5, or Stage 6 reason is not RANKED_OUT
                is_forced = any([
                    filt_report.stage1_reason, filt_report.stage2_reason,
                    filt_report.stage3_reason, filt_report.stage4_reason,
                    filt_report.stage5_reason
                ])
                if not is_forced and filt_report.stage6_reason and filt_report.stage6_reason != RejectCode.RANKED_OUT:
                    is_forced = True
                
                if is_forced:
                    forced_dropouts += 1
        
        dropout_rate = (forced_dropouts / len(prev_symbols)) if prev_symbols else 0.0
        _logger.info(" [2] Unexpected Forced Dropout Rate:")
        _logger.info("     - Rate: %.2f%% (%d/%d) (Gate: <= 10.0%%)", dropout_rate * 100, forced_dropouts, len(prev_symbols))
        dropout_pass = dropout_rate <= 0.10
    else:
        _logger.info(" [2] Unexpected Forced Dropout Rate: SKIPPED (No previous snapshot found for %s)", prev_as_of)

    # Final Verdict
    if cost_pass and adv_pass and dropout_pass:
        _logger.info(" ✅ UNIVERSE QUALITY PASS")
        return True
    else:
        if not cost_pass:
            _logger.error(" [!] FAIL: Median execution cost (%.2f bps) exceeds 50.0 bps gate.", median_cost)
        if not adv_pass:
            _logger.error(" [!] FAIL: Median ADV (%s) is below 25M USDT gate.", f"{median_adv:,.0f}")
        if not dropout_pass:
            _logger.error(" [!] FAIL: Forced dropout rate (%.2f%%) exceeds 10.0%% gate.", dropout_rate * 100)
        
        _logger.error(" [!] Universe quality gates failed. Aborting optimization to prevent suboptimal execution.")
        return False


def main() -> None:
    ai_telemetry_payloads: list[dict[str, Any]] = []
    run_id: str | None = None
    run_summary_written = False
    discovered_symbols: list[str] = []
    snapshot: Any = None
    report: pd.DataFrame = pd.DataFrame()
    selection_summary: dict[str, Any] = {
        "selected_by": None,
        "selected_trial_number": None,
        "deploy_score": None,
        "selection_reject_reason_count": {},
    }
    run_summary_extras: dict[str, Any] = {}

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--skip-universe", action="store_true")
    pre_parser.add_argument("--skip-data-sync", action="store_true")
    pre_parser.add_argument("--sync-limit", type=int, default=None)
    pre_parser.add_argument("--reference-date", type=str, default=None)
    pre_parser.add_argument("--tf", type=str, default="4h")
    pre_args, remaining_args = pre_parser.parse_known_args()

    fetch_start_date_str, start_date_str, is_end_date_str, end_date_str = get_quarterly_window(pre_args.reference_date)
    fetch_start_date = datetime.strptime(fetch_start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # [STEP 0/4] DATA AUTO-SYNC
    if not pre_args.skip_universe and not pre_args.skip_data_sync:
        _logger.info("\n [STEP 0/4] DATA AUTO-SYNC")
        _logger.info(" ─────────────────────────────────────────────────────────────────────────────────────")
        try:
            from src.domain.futures.universe.sync_utils import run_historical_sync
            run_historical_sync(fetch_start_date, end_date, limit=pre_args.sync_limit)
        except Exception as exc:
            _logger.error("Data auto-sync failed: %s", exc)

    # [STEP 1/4] UNIVERSE DISCOVERY & DATA LOADING
    _logger.info("\n [STEP 1/4] UNIVERSE DISCOVERY & DATA LOADING")
    _logger.info(" ─────────────────────────────────────────────────────────────────────────────────────")
    if not pre_args.skip_universe:
        try:
            # Check for force-rebuild in remaining_args
            force_rebuild = "--force-universe-rebuild" in remaining_args
            discovered_symbols, snapshot, report = _discover_symbols_via_universe(
                tf=pre_args.tf,
                reference_date=pre_args.reference_date,
                force_rebuild=force_rebuild,
            )
        except Exception as exc:
            _logger.error("Universe discovery via snapshot failed: %s", exc)
            return
        if not discovered_symbols:
            _logger.error("Universe discovery via snapshot returned no symbols.")
            return
        
        # Universe Quality Gate Check
        if not validate_universe_quality(snapshot, report, pre_args.reference_date, pre_args.tf):
            sys.exit(1)

        _logger.info(" ✅ Universe discovery complete. %d symbols selected:", len(discovered_symbols))
        _logger.info("    > %s", ", ".join(discovered_symbols))

    parser_default_symbols = discovered_symbols or list(config.opt_config.FUTURES_SYMBOLS)

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(parser_default_symbols))
    parser.add_argument("--trials", type=int, default=OPT_FUTURES_CONFIG["total_trials"])
    parser.add_argument("--tf", type=str, choices=["1h", "4h"], default=pre_args.tf)
    parser.add_argument("--reference-date", type=str, default=pre_args.reference_date)
    parser.add_argument("--alpha-only", action="store_true")
    parser.add_argument("--hmm-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bypass-champion-guard", action="store_true")
    parser.add_argument("--ops-profile", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-data-sync", action="store_true", help="Skip Step 0 data sync.")
    parser.add_argument("--sync-limit", type=int, default=None, help="Limit symbols for Step 0 sync.")
    parser.add_argument("--force-universe-rebuild", action="store_true", help="Force rebuild universe snapshot.")
    args = parser.parse_args(remaining_args)

    resolved_ops_profile = resolve_ops_profile(args.ops_profile)
    selected_ops_profile = (
        resolved_ops_profile.get("id") if resolved_ops_profile else (args.ops_profile or "custom")
    )
    apply_ops_profile_overrides(OPT_FUTURES_CONFIG, resolved_ops_profile)

    if resolved_ops_profile:
        _logger.info(" [OPS] profile=%s trials=%s seeds=%s — %s",
                     selected_ops_profile, resolved_ops_profile.get("trials"),
                     resolved_ops_profile.get("seeds"), resolved_ops_profile.get("description", ""))

    # Reuse dates from pre_parser step
    # fetch_start_date_str, start_date_str, is_end_date_str, end_date_str already calculated
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    load_symbols = list(set(symbols + FUTURES_ANCHOR_SYMBOLS + FUTURES_MACRO_INDEX_SYMBOLS))

    data_maps, oos_data_maps, valid_symbols = load_futures_data_maps_for_symbols(
        load_symbols, args.tf, fetch_start_date_str, start_date_str, is_end_date_str, end_date_str
    )

    if not valid_symbols:
        _logger.error("No valid symbols loaded. Aborting.")
        return

    # [STEP 2/4] ML PIPELINE
    _logger.info("\n" + "═" * 85)
    _logger.info(" [STEP 2/4] ML PIPELINE: Alpha & HMM Goal Audit")
    _logger.info("═" * 85)

    ml_n_jobs = resolve_futures_parallel_policy(len(valid_symbols))
    ml_cfg = dict(OPT_FUTURES_CONFIG)
    # Force AlphaFactory backend for futures execution so runs are not tied to legacy miner path.
    ml_cfg["FUTURES_ML_ALPHA_BACKEND"] = "factory_v1"
    ml_out = run_ml_pipeline_for_universe(
        valid_symbols, args.tf, fetch_start_date, end_date, ml_cfg,
        workers=ml_n_jobs, n_jobs=ml_n_jobs, is_end_date=is_end_date_str, is_start_date=start_date_str,
        gp_only=args.alpha_only, hmm_only=args.hmm_only,
        preloaded_data_maps=oos_data_maps if not pre_args.skip_universe else None,
    )

    if hasattr(ml_out, "hmm_report") and ml_out.hmm_report:
        log_hmm_report_summary(ml_out.hmm_report)

    if hasattr(ml_out, "alpha_panel") and not ml_out.alpha_panel.empty:
        log_alpha_component_summary(ml_out.alpha_panel, is_end_date=is_end_date_str)

    # G-ALPHA v8.0: Hard-Kill Switch
    # 최적화 진입 전 최소 1개 이상의 정예 알파가 생존해야 함을 보장.
    alpha_panel = getattr(ml_out, "alpha_panel", None)
    filt_meta = getattr(alpha_panel, "attrs", {}).get("alpha_component_filter", {}) if alpha_panel is not None else {}
    n_surv = int(filt_meta.get("n_surviving", 0))
    # Final aggregated count check (strictly what will be used in optimization)
    n_post = int(filt_meta.get("post_agg_selected_long_count", 0))
    
    if not args.hmm_only:  # HMM-only 모드가 아닐 때만 알파 생존 여부 체크
        if n_surv <= 0 or n_post <= 0:
            _logger.error("\n [!] G-ALPHA v8.0 CRITICAL: No alpha components survived the strict gates (survive=%d, post=%d).", n_surv, n_post)
            _logger.error(" [!] Optimization (Step 3/4) aborted to prevent noise overfitting.")
            return

    if args.alpha_only or args.hmm_only:
        hmm_report = getattr(ml_out, "hmm_report", {}) or {}
        alpha_panel = getattr(ml_out, "alpha_panel", None)
        alpha_non_empty = bool(alpha_panel is not None and not alpha_panel.empty)
        
        # Improved component count discovery
        if alpha_non_empty and alpha_panel is not None:
            if "component" in alpha_panel.index.names:
                alpha_component_count = int(alpha_panel.index.get_level_values("component").nunique())
            else:
                alpha_component_count = len([c for c in alpha_panel.columns if c.startswith("alpha_long_") or c == "alpha_long"])
        else:
            alpha_component_count = 0
            
        hmm_report_present = bool(hmm_report)

        if args.hmm_only and not hmm_report_present:
            _logger.error(" [ML-ONLY] hmm-only requested but hmm_report is empty.")
            return

        _logger.info(
            " [ML-ONLY] mode=%s hmm_report_present=%s alpha_panel_non_empty=%s alpha_component_count=%d",
            "hmm-only" if args.hmm_only else "alpha-only",
            hmm_report_present,
            alpha_non_empty,
            alpha_component_count,
        )

        # alpha_panel.attrs에 저장된 filter 통계 출력
        filt_meta = getattr(alpha_panel, "attrs", {}).get("alpha_component_filter", {}) if alpha_panel is not None else {}
        if filt_meta:
            n_surv = int(filt_meta.get("n_surviving", 0))
            n_comp = int(filt_meta.get("n_components", 0))
            is_mu = float(filt_meta.get("primary_is_mu", 0.0))
            oos_mu = float(filt_meta.get("primary_oos_mu", 0.0))
            _logger.info(
                " [FILTER] FDR/DSR/OOS gate: %d / %d slots survive | primary IS-IC=%.4f OOS-IC=%.4f | "
                "fail_fdr=%d fail_dsr=%d fail_oos=%d fail_hl=%d",
                n_surv, n_comp,
                is_mu, oos_mu,
                int(filt_meta.get("fail_fdr", 0)), int(filt_meta.get("fail_dsr", 0)),
                int(filt_meta.get("fail_oos", 0)), int(filt_meta.get("fail_half_life", 0)),
            )

        p7_ml_only_summary = _collect_p7_ops_summary(
            mode="hmm-only" if args.hmm_only else "alpha-only",
            ml_out=ml_out,
            study_ml=None,
            selection_summary=selection_summary,
        )
        run_summary_extras["p7_ops_summary"] = p7_ml_only_summary
        _log_p7_ops_summary(p7_ml_only_summary)
        _logger.info(" [ML-ONLY] optimization skipped by mode flag.")
        return

    # FEATURE INTEGRATION
    merge_ml_output_into_is_and_oos(ml_out, data_maps, oos_data_maps, valid_symbols, args.tf)
    log_ml_merge_feature_stats(oos_data_maps, valid_symbols, args.tf)

    # [STEP 3/4] OPTIMIZATION
    _logger.info("\n" + "═" * 85)
    _logger.info(" [STEP 3/4] Optimization: Multi-Phase Strategy Tuning")
    _logger.info("═" * 85)

    n_ml_trials = (
        int(resolved_ops_profile["trials"]) if resolved_ops_profile else int(args.trials)
    )
    target_seeds = (
        [int(s) for s in (resolved_ops_profile.get("seeds") or [42])]
        if resolved_ops_profile else ([int(args.seed)] if args.seed else [42])
    )
    seed_learn = target_seeds[0]

    run_id = build_run_id(
        args.tf, fetch_start_date, end_date, valid_symbols, OPT_FUTURES_CONFIG, project_root
    )
    _logger.info(" [RUN] run_id=%s", run_id)

    base_ctx = MLPhaseDContext(
        data_maps=data_maps, symbols=valid_symbols, tf=args.tf, seed=seed_learn,
        effective_total_trials=n_ml_trials, ml_pipeline_fetch_start=fetch_start_date,
        ml_pipeline_end=end_date, ml_pipeline_is_start=start_date,
        ml_pipeline_workers=ml_n_jobs, run_id=run_id,
    )

    precompute_ml_optimization_context(base_ctx)
    if base_ctx.awf_leg_slices:
        test_leg = base_ctx.awf_leg_slices[0]["data"]
        zkill, zfund, lev_leg = _cached_kill_fund_lev(test_leg, _base_engine_params({}, args.tf))
        _run_portfolio_numba_block(
            _base_engine_params({}, args.tf), test_leg, zkill, zfund, lev_leg
        )

    storage_url, storage = setup_optuna_storage(project_root)
    study_name = build_joint_study_name(
        args.tf, fetch_start_date, end_date, valid_symbols, OPT_FUTURES_CONFIG
    )

    def _persist_run_summary(
        status: str, force: bool = False, best_cand: dict[str, Any] | None = None
    ) -> None:
        """Collect and log run summary without writing to disk."""
        nonlocal run_summary_written
        if (run_summary_written and not force) or run_id is None:
            return
        # Removed disk write (write_run_summary_snapshot) as requested.
        run_summary_written = True

    opt_workers = int(OPT_FUTURES_CONFIG.get("FUTURES_OPT_MAX_WORKERS", ml_n_jobs))
    opt_workers = max(1, min(opt_workers, ml_n_jobs))
    phase_a1_trials = int(OPT_FUTURES_CONFIG.get("FUTURES_V43_PHASE_A1_TRIALS", 150))
    phase_a2_trials = int(OPT_FUTURES_CONFIG.get("FUTURES_V43_PHASE_A2_TRIALS", 100))
    phase_b_trials = int(OPT_FUTURES_CONFIG.get("FUTURES_V43_PHASE_B_TRIALS", 300))
    phase_bundle = run_v43_phase_optimization_skeleton(
        base_ctx=base_ctx,
        base_study_name=study_name,
        storage_url=storage_url,
        storage=storage,
        n_trials=n_ml_trials,
        n_trials_a1=phase_a1_trials,
        n_trials_a2=phase_a2_trials,
        n_trials_b=phase_b_trials,
        seed=seed_learn,
        resume=args.resume,
        n_workers=opt_workers,
        enqueue_seeds=None,
        target_seeds=target_seeds,
    )
    study_ml = phase_bundle.study_b
    if phase_bundle.phase_c_diagnostics:
        run_summary_extras["phase_c_diagnostics"] = dict(phase_bundle.phase_c_diagnostics)

    _persist_run_summary("optimized")

    ensemble_top_candidates: list[dict[str, Any]] = []
    is_v43_phase_b_study = str(getattr(study_ml, "study_name", "")).endswith("_phase_b")
    if is_v43_phase_b_study:
        ensemble_top_candidates, sel_sum = select_v43_phase_b_top_candidates(
            study_ml,
            base_ctx,
            OPT_FUTURES_CONFIG,
            top_k=5,
        )
        best_cand = ensemble_top_candidates[0] if ensemble_top_candidates else {}
    else:
        best_cand, sel_sum = select_and_rank_candidates(study_ml, base_ctx, OPT_FUTURES_CONFIG)
        if best_cand:
            ensemble_top_candidates = [best_cand]
    if not best_cand:
        completed_trials = [
            t for t in study_ml.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        no_candidate_reason = classify_no_valid_candidates(
            selection_summary=sel_sum,
            completed_trials=completed_trials,
        )
        try:
            study_ml.set_user_attr("obs_no_valid_candidates_reason", no_candidate_reason)
        except Exception:
            pass
        p7_no_candidate_summary = _collect_p7_ops_summary(
            mode="optimization",
            ml_out=ml_out,
            study_ml=study_ml,
            selection_summary=sel_sum,
        )
        run_summary_extras["p7_ops_summary"] = p7_no_candidate_summary
        _log_p7_ops_summary(p7_no_candidate_summary)
        _logger.error("No valid candidates found. reason=%s", no_candidate_reason)
        return
    selection_summary.update(sel_sum)

    champion_raw_params = dict(best_cand["params"])
    champion_awf_diag = best_cand["awf_diag"]
    best_trial_coord = best_cand["trial"]

    # [STABILITY] Layer 3
    champ_stab_cv, champ_l3_fail = check_stability_layer3(
        best_cand, base_ctx, OPT_FUTURES_CONFIG
    )

    pbo_gate, dsr_gate, _ = resolve_adjusted_gates(OPT_FUTURES_CONFIG, n_ml_trials)
    pbo_obs = awf_pos_frac_to_pseudo_pbo(float(champion_awf_diag.get("awf_pos_frac", 0.0)))
    dsr_obs = float(champion_awf_diag.get("dsr_awf", 0.0))

    # [STEP 4/4] FINAL EVALUATION
    run_final_oos_evaluation(
        ensemble_results=ensemble_top_candidates[:5], oos_data_maps=oos_data_maps, data_maps=data_maps,
        valid_symbols=valid_symbols, champion_awf_diag=champion_awf_diag, args=args,
        project_root=project_root, study_ml=study_ml, run_id=run_id,
        ai_telemetry_payloads=ai_telemetry_payloads, selection_summary=selection_summary,
        run_summary_extras=run_summary_extras, ml_ctx=base_ctx, n_ml_trials=n_ml_trials,
        target_seeds=target_seeds, selected_ops_profile=str(selected_ops_profile or "custom"),
        pbo_gate=pbo_gate, dsr_gate=dsr_gate, pbo_obs=pbo_obs, dsr_obs=dsr_obs,
        best_trial=best_trial_coord, params=build_ml_phase_d_params(champion_raw_params, args.tf),
        champ_stab_cv=champ_stab_cv, stab_tmp_layer3_awf_fail=champ_l3_fail,
        cv_max=float(OPT_FUTURES_CONFIG.get("FUTURES_CHAMP_STABILITY_CV_MAX", 0.30)),
        phase_c_diagnostics=phase_bundle.phase_c_diagnostics,
    )
    p7_done_summary = _collect_p7_ops_summary(
        mode="optimization",
        ml_out=ml_out,
        study_ml=study_ml,
        selection_summary=selection_summary,
    )
    run_summary_extras["p7_ops_summary"] = p7_done_summary
    _log_p7_ops_summary(p7_done_summary)
    _persist_run_summary("done", force=True, best_cand=best_cand)


if __name__ == "__main__":
    main()
