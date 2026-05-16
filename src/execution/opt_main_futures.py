from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna

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
from config.settings import (  # noqa: E402
    FUTURES_DATA_DIR,
)
from src.core.utils.utils import setup_logger  # noqa: E402
from src.domain.futures.data_loader import (  # noqa: E402
    DataCollector,
)
from src.domain.futures.ml_pipeline import run_ml_pipeline_for_universe  # noqa: E402
from src.domain.futures.ml_pipeline.pipeline_runner import (  # noqa: E402
    merge_ml_output_into_is_and_oos,
)
from src.domain.futures.optimization.candidate_selector import (  # noqa: E402
    check_stability_layer3,
    select_v43_phase_b_top_candidates,
    select_and_rank_candidates,
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
    build_run_id,
    collect_run_summary_from_study,
    resolve_futures_parallel_policy,
    setup_optuna_storage,
    write_run_summary_snapshot,
)
from src.domain.futures.optimization.screener import (  # noqa: E402
    orchestrate_universe_discovery,
)
from src.domain.futures.optimization.validation import (  # noqa: E402
    awf_pos_frac_to_pseudo_pbo,
    resolve_adjusted_gates,
)

warnings.filterwarnings("ignore")

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


def main() -> None:
    ai_telemetry_payloads: list[dict[str, Any]] = []
    run_id: str | None = None
    run_summary_written = False
    selection_summary: dict[str, Any] = {
        "selected_by": None,
        "selected_trial_number": None,
        "deploy_score": None,
        "selection_reject_reason_count": {},
    }
    run_summary_extras: dict[str, Any] = {}

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--skip-universe", action="store_true")
    pre_parser.add_argument("--reference-date", type=str, default=None)
    pre_parser.add_argument("--tf", type=str, default="4h")
    pre_args, remaining_args = pre_parser.parse_known_args()

    # [STEP 1/4] UNIVERSE DISCOVERY & DATA LOADING
    if not pre_args.skip_universe:
        collector = DataCollector()
        success = orchestrate_universe_discovery(
            collector, pre_args.tf, pre_args.reference_date,
            FUTURES_DATA_DIR, FUTURES_ANCHOR_SYMBOLS
        )
        if not success:
            return

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(config.opt_config.FUTURES_SYMBOLS))
    parser.add_argument("--trials", type=int, default=OPT_FUTURES_CONFIG["total_trials"])
    parser.add_argument("--tf", type=str, choices=["1h", "4h"], default=pre_args.tf)
    parser.add_argument("--reference-date", type=str, default=pre_args.reference_date)
    parser.add_argument("--alpha-only", action="store_true")
    parser.add_argument("--hmm-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force-retrain-alpha", action="store_true")
    parser.add_argument("--bypass-champion-guard", action="store_true")
    parser.add_argument("--ops-profile", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
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

    if args.force_retrain_alpha:
        OPT_FUTURES_CONFIG["FUTURES_ML_FORCE_RETRAIN_ALPHA"] = True

    fetch_start_date, start_date, is_end_date, end_date = get_quarterly_window(args.reference_date)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    load_symbols = list(set(symbols + FUTURES_ANCHOR_SYMBOLS + FUTURES_MACRO_INDEX_SYMBOLS))

    data_maps, oos_data_maps, valid_symbols = load_futures_data_maps_for_symbols(
        load_symbols, args.tf, fetch_start_date, start_date, is_end_date, end_date
    )

    if not valid_symbols:
        _logger.error("No valid symbols loaded. Aborting.")
        return

    # [STEP 2/4] ML PIPELINE
    _logger.info("\n" + "═" * 85)
    _logger.info(" [STEP 2/4] ML PIPELINE: Universal Cross-Sectional Alpha")
    _logger.info("═" * 85)

    ml_n_jobs = resolve_futures_parallel_policy(len(valid_symbols))
    ml_out = run_ml_pipeline_for_universe(
        valid_symbols, args.tf, fetch_start_date, end_date, dict(OPT_FUTURES_CONFIG),
        workers=ml_n_jobs, n_jobs=ml_n_jobs, is_end_date=is_end_date, is_start_date=start_date,
        gp_only=args.alpha_only, hmm_only=args.hmm_only,
        preloaded_data_maps=oos_data_maps if not pre_args.skip_universe else None,
    )

    if hasattr(ml_out, "hmm_report") and ml_out.hmm_report:
        log_hmm_report_summary(ml_out.hmm_report)

    if hasattr(ml_out, "alpha_panel") and not ml_out.alpha_panel.empty:
        log_alpha_component_summary(ml_out.alpha_panel)

    if args.alpha_only or args.hmm_only:
        hmm_report = getattr(ml_out, "hmm_report", {}) or {}
        alpha_panel = getattr(ml_out, "alpha_panel", None)
        alpha_non_empty = bool(alpha_panel is not None and not alpha_panel.empty)
        alpha_component_count = (
            int(alpha_panel.index.get_level_values("component").nunique())
            if alpha_non_empty and "component" in alpha_panel.index.names
            else 0
        )
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
        _logger.info(" [ML-ONLY] optimization skipped by mode flag.")
        return

    # FEATURE INTEGRATION
    merge_ml_output_into_is_and_oos(ml_out, data_maps, oos_data_maps, valid_symbols, args.tf)
    log_ml_merge_feature_stats(oos_data_maps, valid_symbols, args.tf)

    # [STEP 3/4] OPTIMIZATION
    _logger.info("\n" + "═" * 85)
    _logger.info(" [STEP 3/4] Optimization: Joint Multi-Objective TPE")
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
        status: str, force: bool = False, best_cand: dict | None = None
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
        _logger.error("No valid candidates found.")
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
        target_seeds=target_seeds, selected_ops_profile=selected_ops_profile,
        pbo_gate=pbo_gate, dsr_gate=dsr_gate, pbo_obs=pbo_obs, dsr_obs=dsr_obs,
        best_trial=best_trial_coord, params=build_ml_phase_d_params(champion_raw_params, args.tf),
        champ_stab_cv=champ_stab_cv, stab_tmp_layer3_awf_fail=champ_l3_fail,
        cv_max=float(OPT_FUTURES_CONFIG.get("FUTURES_CHAMP_STABILITY_CV_MAX", 0.30)),
        phase_c_diagnostics=phase_bundle.phase_c_diagnostics,
    )
    _persist_run_summary("done", force=True, best_cand=best_cand)


if __name__ == "__main__":
    main()
