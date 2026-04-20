from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import TrialState

# Project Root Setup
project_root: str = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings  # noqa: E402

import config.opt_config  # noqa: E402
from config.opt_config import (  # noqa: E402
    FUTURES_ANCHOR_SYMBOLS,
    FUTURES_SCREENER_CONFIG,
    OPT_FUTURES_CONFIG,
    get_quarterly_window,
)
from config.settings import (  # noqa: E402
    FUTURES_CACHE_DIR,
    FUTURES_DATA_DIR,
)
from src.core.optimization.opt_utils import compute_segment_merge_index  # noqa: E402
from src.domain.futures.data_collector import DataCollector  # noqa: E402
from src.domain.futures.funding_utils import merge_funding_into_ohlcv  # noqa: E402
from src.domain.futures.metrics_utils import merge_metrics_into_ohlcv  # noqa: E402
from src.domain.futures.ml_pipeline import (  # noqa: E402
    copy_data_maps_tf_clone,
    merge_ml_output_into_data_maps,
    merge_ml_output_into_is_and_oos,
    run_hmm_fusion_for_is_end,
    run_ml_pipeline_for_universe,
)
from src.domain.futures.opt_futures_utils.objective import (  # noqa: E402
    inject_cs_momentum_ranks,
)
from src.domain.futures.opt_futures_utils.objective_ml import (  # noqa: E402
    MLPhaseDContext,
    build_ml_phase_d_params,
    check_hard_gates_ml,
    objective_ml_phase_d,
    select_best_trial_by_holdout_log_ret,
)
from src.domain.futures.opt_futures_utils.oos_evaluator import (  # noqa: E402
    run_cpcv_complement_evaluation,
    run_oos_margin_shared_portfolio,
)

warnings.filterwarnings("ignore")

# Force Linux 'fork' method for memory efficiency (CoW)
if sys.platform != "win32":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from src.core.utils.utils import setup_logger  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
_logger: logging.Logger = logging.getLogger("opt_futures")

setup_logger("DataCollector")
setup_logger("BinanceClient")
logging.getLogger("DataCollector").setLevel(logging.WARNING)
logging.getLogger("BinanceClient").setLevel(logging.WARNING)

SEP_WIDTH: int = 60
PROGRESS_MIN_INTERVAL: float = 0.2
MODE_MULTI: str = "multi"
BEST_PARAMS_FUTURES_JSON_STEM: str = "best_futures_1h"

def _ml_phase_d_tpe_sampler(seed: int) -> TPESampler:
    return TPESampler(
        seed=seed,
        n_startup_trials=200,
        multivariate=True,
        group=True,
        constant_liar=True,
        n_ei_candidates=24,
    )


def _ml_trial_passes_hard_gates(
    trial: optuna.trial.FrozenTrial, pbo_obs: float = 0.0, check_pbo: bool = True
) -> bool:
    cfg = OPT_FUTURES_CONFIG
    if check_pbo and float(pbo_obs) >= float(cfg.get("FUTURES_PBO_MAX", 0.45)):
        return False
    dsr = float(trial.user_attrs.get("gate1_dsr", -9.0))
    if dsr < float(cfg.get("FUTURES_ML_GATE1_DSR_MIN", 0.20)):
        return False
    if float(trial.user_attrs.get("ml_worst_mdd_cpcv", 999.0)) >= 25.0:
        return False
    if float(trial.user_attrs.get("avg_trades", 0.0)) < 10.0:
        return False
    return True


def _resolve_futures_parallel_policy(symbol_count: int) -> int:
    logical_cpus = max(1, os.cpu_count() or 1)
    return max(1, min(8, logical_cpus))


def _load_single_symbol_data(
    sym: str,
    tf: str,
    fetch_start: str,
    start: str,
    is_end: str,
    end: str,
    skip_metrics: bool = False,
) -> tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    try:
        temp_is: Dict[str, Any] = {}
        temp_oos: Dict[str, Any] = {}
        insufficient = False
        collector = DataCollector()
        collector.ensure_funding_data(sym, fetch_start, end)
        if not skip_metrics:
            collector.ensure_metrics_data(sym, fetch_start, end)
            
        for tf_l in [tf, "1d"]:
            raw_df = collector.collect_and_save(sym, tf_l, fetch_start, end)
            df = merge_funding_into_ohlcv(sym, raw_df, Path(FUTURES_DATA_DIR))
            df = merge_metrics_into_ohlcv(sym, df, Path(FUTURES_DATA_DIR))

            if df is None or df.empty or "datetime" not in df.columns:
                insufficient = True
                break
            
            df.reset_index(drop=True, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

            is_start_dt = pd.Timestamp(start, tz="UTC")
            is_end_dt = pd.Timestamp(is_end, tz="UTC")

            is_mask = df["datetime"] < is_end_dt
            is_end_idx = int(is_mask.to_numpy().sum())

            # [Dynamic Quality Gate] 1h requires more bars than 4h/1d
            min_bars_map = {"1h": 2000, "4h": 500, "1d": 300}
            min_bars_threshold = min_bars_map.get(tf_l, 300)
            
            if is_end_idx < min_bars_threshold:
                insufficient = True
                break

            temp_is[tf_l] = df.iloc[:is_end_idx].copy()
            mask = temp_is[tf_l]["datetime"] >= is_start_dt
            temp_is[f"is_start_idx_{tf_l}"] = int(mask.to_numpy().argmax()) if mask.any() else 0
            temp_oos[tf_l] = df
            mask_oos = df["datetime"] >= is_end_dt
            idx_oos = int(mask_oos.to_numpy().argmax()) if mask_oos.any() else len(df)
            temp_oos[f"oos_start_idx_{tf_l}"] = idx_oos

        if insufficient:
            return sym, None, None, True

        temp_is[f"merge_idx_{tf}"] = compute_segment_merge_index(temp_is[tf], temp_is["1d"])
        temp_oos[f"merge_idx_{tf}"] = compute_segment_merge_index(temp_oos[tf], temp_oos["1d"])
        return sym, temp_is, temp_oos, False
    except Exception as e:
        _logger.warning("Failed to load symbol %s: %s", sym, e)
        return sym, None, None, True


def _load_futures_data_maps_for_symbols(
    symbols: List[str],
    tf: str,
    fetch_start: str,
    start: str,
    is_end: str,
    end: str,
    skip_metrics: bool = False,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    data_maps: Dict[str, Dict[str, Any]] = {}
    oos_data_maps: Dict[str, Dict[str, Any]] = {}
    valid_symbols: List[str] = []
    
    # [Fix] Filter out non-ASCII symbols before processing
    symbols = [s for s in symbols if all(ord(c) < 128 for c in s)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                _load_single_symbol_data, sym, tf, fetch_start, start, is_end, end, skip_metrics
            )
            for sym in symbols
        ]
        for f in concurrent.futures.as_completed(futures):
            sym, t_is, t_oos, insufficient = f.result()
            if not insufficient and t_is and t_oos:
                data_maps[sym], oos_data_maps[sym] = t_is, t_oos
                valid_symbols.append(sym)

    if len(valid_symbols) > 1:
        inject_cs_momentum_ranks(data_maps, valid_symbols, tf)
        inject_cs_momentum_ranks(oos_data_maps, valid_symbols, tf)

    _logger.info("[STEP 1/5] Data loading complete: %d symbols valid.", len(valid_symbols))
    return data_maps, oos_data_maps, valid_symbols


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--skip-universe", action="store_true")
    pre_parser.add_argument("--reference-date", type=str, default=None)
    pre_parser.add_argument("--tf", type=str, default="1h")
    pre_args, remaining_args = pre_parser.parse_known_args()

    if not pre_args.skip_universe:
        from src.domain.futures.opt_futures_utils.universe_screener_futures import (
            screen_futures_universe,
            screen_symbol_refinement_futures,
        )

        res = get_quarterly_window(pre_args.reference_date)
        fetch_start_date, start_date, is_end_date, end_date = res
        collector = DataCollector()
        
        broad_candidates, _ = screen_futures_universe(
            collector,
            [],
            pre_args.tf,
            FUTURES_SCREENER_CONFIG,
            fetch_start_date,
            is_end_date,
            data_dir=FUTURES_DATA_DIR,
        )

        if not broad_candidates:
            _logger.error("No broad candidates. Aborting.")
            return

        data_maps_broad, _, valid_broad = _load_futures_data_maps_for_symbols(
            broad_candidates,
            pre_args.tf,
            fetch_start_date,
            start_date,
            is_end_date,
            end_date,
            skip_metrics=True,
        )

        success = screen_symbol_refinement_futures(
            broad_candidates=list(broad_candidates),
            winning_signal_type="CS_RANK",
            is_end_date=is_end_date,
            tf=pre_args.tf,
            symbol_dfs_4h={s: data_maps_broad[s][pre_args.tf] for s in valid_broad},
            daily_dfs={s: data_maps_broad[s]["1d"] for s in valid_broad},
            phase_b_params=None,
            anchor_symbols=FUTURES_ANCHOR_SYMBOLS,
        )
        if not success:
            return
        importlib.reload(config.opt_config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(config.opt_config.FUTURES_SYMBOLS))
    parser.add_argument("--trials", type=int, default=OPT_FUTURES_CONFIG["total_trials"])
    parser.add_argument("--tf", type=str, choices=["1h", "4h"], default="1h")
    parser.add_argument("--reference-date", type=str, default=None)
    parser.add_argument("--gp-only", action="store_true", help="Stop after GP IC calculation")
    parser.add_argument("--hmm-only", action="store_true", help="Stop after HMM regime inference")
    args = parser.parse_args(remaining_args)

    fetch_start_date, start_date, is_end_date, end_date = get_quarterly_window(args.reference_date)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    data_maps, oos_data_maps, valid_symbols = _load_futures_data_maps_for_symbols(
        symbols, args.tf, fetch_start_date, start_date, is_end_date, end_date
    )

    if not valid_symbols:
        return

    ml_n_jobs = _resolve_futures_parallel_policy(len(valid_symbols))

    # [Institutional Quant] Universal Cross-Sectional ML Pipeline
    _logger.info("[STEP 2/5] ML pipeline: Universal Cross-Sectional GP & Regime Inference (1h).")
    ml_out = run_ml_pipeline_for_universe(
        valid_symbols,
        args.tf,
        fetch_start_date,
        end_date,
        dict(OPT_FUTURES_CONFIG),
        workers=ml_n_jobs,
        n_jobs=ml_n_jobs,
        is_end_date=is_end_date,
        is_start_date=start_date,
        gp_only=args.gp_only,
        hmm_only=args.hmm_only,
    )

    if args.gp_only:
        _logger.info(" [GP-ONLY] Analysis complete. Exiting as requested.")
        return

    if args.hmm_only:
        _logger.info(" [HMM-ONLY] Analysis complete. Exiting as requested.")
        return

    _logger.info("[STEP 3/5] Integrating ML Features (GP/HMM) into Data Maps...")
    merge_ml_output_into_is_and_oos(ml_out, data_maps, oos_data_maps, valid_symbols, args.tf)

    # Safe Cache Cleanup (Avoid rmtree hang)
    from src.domain.futures.opt_futures_utils.signal_cache import _MEM_CACHE, DISK_CACHE_ROOT
    _MEM_CACHE.clear()
    if DISK_CACHE_ROOT.exists():
        for cache_file in DISK_CACHE_ROOT.glob("*.parquet"):
            try:
                cache_file.unlink()
            except Exception:  # noqa: S110
                pass

    # [PHASE 5] Optuna Portfolio Optimization Starting
    n_ml_trials = int(args.trials)
    seed = int(OPT_FUTURES_CONFIG.get("seeds", [42])[0])
    ml_ctx = MLPhaseDContext(data_maps=data_maps, symbols=valid_symbols, tf=args.tf, seed=seed)
    
    # [VISIBILITY] Silence individual trial logs for a clean progress bar
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Custom Progress Callback for visibility in non-TTY environments
    _trial_counter = {"n": 0}
    def progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            _trial_counter["n"] += 1
        n_completed = _trial_counter["n"]
        batch_size = max(1, n_ml_trials // 10)
        if n_completed > 0 and (n_completed % batch_size == 0 or n_completed == n_ml_trials):
            _logger.info(f" [STEP 4/5] Progress: {n_completed}/{n_ml_trials} trials complete...")

    # Use TqdmCallback with explicit n_trials
    try:
        from optuna.integration import TqdmCallback
    except ImportError:
        try:
            from optuna_integration import TqdmCallback
        except ImportError:
            tqdm_callback_cls = None
        else:
            tqdm_callback_cls = TqdmCallback
    else:
        tqdm_callback_cls = TqdmCallback

    tqdm_cb = None
    if tqdm_callback_cls is not None:
        tqdm_cb = tqdm_callback_cls(n_trials=n_ml_trials)
    
    callbacks = [progress_callback]
    if tqdm_cb:
        callbacks.append(tqdm_cb)

    # [PERFORMANCE] Use multi-processing (n_jobs)
    n_jobs = min(4, _resolve_futures_parallel_policy(len(valid_symbols)))
    _logger.info("\n" + "=" * 85)
    _logger.info(f" [STEP 4/5] Portfolio Optimization: {n_ml_trials} trials | {n_jobs} cores")
    _logger.info("=" * 85 + "\n")

    study_ml = optuna.create_study(
        direction="minimize",
        sampler=_ml_phase_d_tpe_sampler(seed),
    )

    study_ml.optimize(
        lambda tr: objective_ml_phase_d(tr, ml_ctx),
        n_trials=n_ml_trials,
        callbacks=callbacks,
        n_jobs=n_jobs,
    )

    completed = [t for t in study_ml.trials if t.state == TrialState.COMPLETE]
    completed.sort(key=lambda tr: tr.value if tr.value is not None else 1e18)
    mai = ml_ctx.multi_alignment_info or {}
    paths = mai.get("cpcv_paths")
    br = mai.get("cpcv_all_block_ranges")
    best_trial: optuna.trial.FrozenTrial | None = None
    _logger.info(" [STEP 4.5/5] Filtering top trials with Hard Gates (Lazy PBO)...")
    iter_limit = min(300, len(completed))
    for i in range(iter_limit):
        t = completed[i]
        # 1. Cheap checks first (DSR, MDD, Trades)
        if not _ml_trial_passes_hard_gates(t, check_pbo=False):
            continue

        # 2. Expensive PBO check only if cheap gates pass
        p_params = build_ml_phase_d_params(dict(t.params), args.tf)
        pbo_obs = 1.0
        if paths and br:
            oos_tw = t.user_attrs.get("cpcv_path_oos_log_tw") or []
            if len(oos_tw) == len(paths):
                _logger.info(f"  Evaluating Top Trial {i+1}/{iter_limit}: Checking PBO...")
                pbo_obs, _rho = run_cpcv_complement_evaluation(
                    p_params,
                    valid_symbols,
                    args.tf,
                    data_maps,
                    paths,
                    br,
                    oos_path_scores=oos_tw,
                )
        
        if _ml_trial_passes_hard_gates(t, float(pbo_obs), check_pbo=True):
            _logger.info(f"  --> Best Trial found at rank {i+1} (PBO={pbo_obs:.4f})")
            best_trial = t
            break
    if best_trial is None and completed:
        best_trial = select_best_trial_by_holdout_log_ret(completed)
        _logger.warning(
            "All completed trials fail PBO/DSR/MDD/trade gates; falling back to best holdout."
        )
    elif best_trial is None:
        cand = [t for t in study_ml.trials if t.params]
        best_trial = cand[0] if cand else None
    if best_trial is None:
        raise RuntimeError("Optuna study produced no usable trials for best-params export.")

    params = build_ml_phase_d_params(dict(best_trial.params), args.tf)
    
    _logger.info("\n[STEP 5/5] Finalizing OOS Evaluation with Best Parameters...")
    oos_port = run_oos_margin_shared_portfolio(
        valid_symbols, args.tf, params, oos_data_maps, cache_root=FUTURES_CACHE_DIR
    )

    if bool(OPT_FUTURES_CONFIG.get("FUTURES_PHASE3_HARD_GATE", True)):
        mai = ml_ctx.multi_alignment_info or {}
        oos_tw = best_trial.user_attrs.get("cpcv_path_oos_log_tw") or []
        paths = mai.get("cpcv_paths")
        br = mai.get("cpcv_all_block_ranges")
        if oos_tw and paths and br:
            pbo_obs, rho_obs = run_cpcv_complement_evaluation(
                params,
                valid_symbols,
                args.tf,
                data_maps,
                paths,
                br,
                oos_path_scores=oos_tw,
            )
        else:
            pbo_obs, rho_obs = (0.5, 0.0)
        dsr_obs = float(best_trial.user_attrs.get("gate1_dsr", 0.0))
        gate_ok = check_hard_gates_ml(oos_port, float(pbo_obs), dsr_obs, 0.55)
        _logger.info(
            " [Phase3 HARD GATE] PBO=%.4f (rho=%.4f) gate1_dsr=%.4f -> %s",
            float(pbo_obs),
            float(rho_obs),
            dsr_obs,
            "PASS" if gate_ok else "FAIL",
        )

    n_wf = int(OPT_FUTURES_CONFIG.get("FUTURES_WF_OOS_LEGS", 1))
    wf_hmm_refit = bool(OPT_FUTURES_CONFIG.get("FUTURES_WF_HMM_LEG_REFIT", True))
    wf_tw_floor = float(OPT_FUTURES_CONFIG.get("FUTURES_WF_LEG_TW_MIN_ALL", 1.0))
    wf_tw_mean_min = float(OPT_FUTURES_CONFIG.get("FUTURES_WF_LEG_TW_MEAN_MIN", 1.05))
    if n_wf > 1 and valid_symbols:
        ref_sym = valid_symbols[0]
        ref_df = oos_data_maps[ref_sym][args.tf]
        o0 = int(oos_data_maps[ref_sym][f"oos_start_idx_{args.tf}"])
        span = max(0, len(ref_df) - o0)
        leg_w = max(1, span // n_wf)
        wf_tw_sum = 0.0
        tw_legs: List[float] = []
        for leg in range(n_wf):
            ls = o0 + leg * leg_w
            le = o0 + (leg + 1) * leg_w if leg < n_wf - 1 else len(ref_df)
            oos_maps_leg = oos_data_maps
            if wf_hmm_refit and ml_out.alpha_panel is not None and not ml_out.alpha_panel.empty:
                leg_anchor = pd.to_datetime(ref_df["datetime"].iloc[ls], utc=True)
                pref_1h = {
                    s: oos_data_maps[s]["1h"].copy()
                    for s in valid_symbols
                    if s in oos_data_maps and "1h" in oos_data_maps[s]
                }
                coll = DataCollector()
                ml_leg = run_hmm_fusion_for_is_end(
                    valid_symbols,
                    args.tf,
                    fetch_start_date,
                    end_date,
                    dict(OPT_FUTURES_CONFIG),
                    oos_data_maps,
                    pref_1h,
                    None,
                    ml_out.alpha_panel,
                    leg_anchor,
                    coll,
                    workers=ml_n_jobs,
                    n_jobs=ml_n_jobs,
                    include_fusion=True,
                    summary_mode_label=f" (WF leg {leg + 1}/{n_wf})",
                    prefetch_label_start=start_date,
                )
                ml_leg.alpha_panel = ml_out.alpha_panel
                oos_maps_leg = copy_data_maps_tf_clone(oos_data_maps, valid_symbols, args.tf)
                merge_ml_output_into_data_maps(
                    ml_leg, oos_maps_leg, valid_symbols, args.tf, log_tag=f" WF{leg + 1}"
                )
            leg_port = run_oos_margin_shared_portfolio(
                valid_symbols,
                args.tf,
                params,
                oos_maps_leg,
                cache_root=FUTURES_CACHE_DIR,
                oos_start_idx=ls,
                oos_end_idx=le,
            )
            tw = float(leg_port.get("terminal_wealth_ratio", 1.0))
            wf_tw_sum += tw
            tw_legs.append(tw)
            _suffix = " [HMM reanchored]" if wf_hmm_refit else ""
            _logger.info(
                " [WF] OOS leg %d/%d idx [%d,%d) terminal_wealth_ratio=%.4f%s",
                leg + 1,
                n_wf,
                ls,
                le,
                tw,
                _suffix,
            )
        _logger.info(" [WF] sum terminal_wealth_ratio (all legs)=%.4f", wf_tw_sum)
        if wf_hmm_refit and tw_legs:
            all_ok = all(t >= wf_tw_floor for t in tw_legs)
            mean_ok = (sum(tw_legs) / len(tw_legs)) >= wf_tw_mean_min
            _logger.info(
                " [WF HARD GATE] all legs >= %.2f: %s | mean >= %.2f: %s -> %s",
                wf_tw_floor,
                all_ok,
                wf_tw_mean_min,
                mean_ok,
                "PASS" if (all_ok and mean_ok) else "FAIL",
            )
    
    _logger.info("\n" + "=" * 85)
    _logger.info(" [STEP 5/5] FINAL OOS PERFORMANCE REPORT")
    _logger.info("-" * 85)
    _logger.info(f" CAGR:        {oos_port['cagr_pct']:>8.2f}%")
    _logger.info(f" MDD:         {oos_port['mdd_pct']:>8.2f}%")
    _logger.info(f" Profit Factor: {oos_port['profit_factor']:>8.2f}")
    _logger.info(f" Win Rate:     {oos_port['win_rate_pct']:>8.2f}%")
    _logger.info(f" Total Trades: {oos_port['total_trades']:>8d}")
    _logger.info(f" Calmar Ratio: {oos_port['calmar_ratio']:>8.2f}")
    _logger.info(f" Ulcer Index:  {oos_port['ulcer_index']:>8.2f}")
    _logger.info(f" DSR (Target): {best_trial.user_attrs.get('gate1_dsr', 0.0):>8.4f}")
    _logger.info(f" L/S Minority: {oos_port['oos_long_short_minority_pct']:>8.2f}%")
    _logger.info("-" * 85)
    _logger.info(f" Net PnL / Trading Cost: {oos_port.get('ev_cost_ratio', 0.0):.2f}")
    _logger.info("=" * 85 + "\n")

    res_dir = Path(project_root) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / f"{BEST_PARAMS_FUTURES_JSON_STEM}.json", "w") as f:
        json.dump(params, f, indent=4)
    _logger.info(f"Best parameters saved to results/{BEST_PARAMS_FUTURES_JSON_STEM}.json")

if __name__ == "__main__":
    main()
