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
from optuna.samplers import NSGAIISampler, TPESampler

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
    run_ml_pipeline_for_universe,
)
from src.domain.futures.opt_futures_utils.objective import (  # noqa: E402
    inject_cs_momentum_ranks,
)
from src.domain.futures.opt_futures_utils.objective_ml import (  # noqa: E402
    MLPhaseDContext,
    build_ml_phase_d_params,
    objective_ml_phase_d,
    topsis_select_best,
)
from src.domain.futures.opt_futures_utils.oos_evaluator import (  # noqa: E402
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
BEST_PARAMS_FUTURES_JSON_STEM: str = "best_futures_4h"

_MDD_CONSTRAINT_LIMIT: float = float(OPT_FUTURES_CONFIG.get("FUTURES_MAX_AVG_CPCV_MDD", 25.0))


def futures_frozen_trial_constraints(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
    pf = float(trial.user_attrs.get("avg_pf", 0.0) or 0.0)
    trades = float(trial.user_attrs.get("avg_trades", 0.0) or 0.0)
    avg_mdd = float(trial.user_attrs.get("avg_mdd", 100.0) or 100.0)
    ls_ratio = float(trial.user_attrs.get("long_short_ratio", 0.0) or 0.0)
    ev_cost = float(trial.user_attrs.get("ev_cost_ratio", 0.0) or 0.0)

    return (
        1.35 - pf,
        25.0 - trades,
        avg_mdd - _MDD_CONSTRAINT_LIMIT,
        0.15 - ls_ratio,
        3.0 - ev_cost,
    )


def _futures_tpe_sampler(seed: int) -> TPESampler:
    return TPESampler(
        seed=seed,
        n_startup_trials=0,
        multivariate=True,
        group=True,
        constant_liar=True,
        constraints_func=futures_frozen_trial_constraints,
    )


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
) -> tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    try:
        temp_is: Dict[str, Any] = {}
        temp_oos: Dict[str, Any] = {}
        insufficient = False
        collector = DataCollector()
        collector.ensure_funding_data(sym, fetch_start, end)
        collector.ensure_metrics_data(sym, fetch_start, end)
        for tf_l in [tf, "1d"]:
            raw_df = collector.collect_and_save(sym, tf_l, fetch_start, end)
            df = merge_funding_into_ohlcv(sym, raw_df, Path(FUTURES_DATA_DIR))
            df = merge_metrics_into_ohlcv(sym, df, Path(FUTURES_DATA_DIR))

            if df is None or df.empty:
                insufficient = True
                break
            
            df.reset_index(drop=True, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

            is_start_dt = pd.Timestamp(start, tz="UTC")
            is_end_dt = pd.Timestamp(is_end, tz="UTC")

            is_mask = df["datetime"] < is_end_dt
            is_end_idx = int(is_mask.to_numpy().sum())

            if is_end_idx < 300:
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
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    data_maps: Dict[str, Dict[str, Any]] = {}
    oos_data_maps: Dict[str, Dict[str, Any]] = {}
    valid_symbols: List[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_load_single_symbol_data, sym, tf, fetch_start, start, is_end, end)
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

    _logger.info("Data loading complete: %d symbols valid.", len(valid_symbols))
    return data_maps, oos_data_maps, valid_symbols


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--skip-universe", action="store_true")
    pre_parser.add_argument("--reference-date", type=str, default=None)
    pre_parser.add_argument("--tf", type=str, default="4h")
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
            broad_candidates, pre_args.tf, fetch_start_date, start_date, is_end_date, end_date
        )

        success = screen_symbol_refinement_futures(
            broad_candidates=list(broad_candidates),
            winning_signal_type="CS_RANK",
            is_end_date=is_end_date,
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
    parser.add_argument("--tf", type=str, choices=["1h", "4h"], default="4h")
    parser.add_argument("--reference-date", type=str, default=None)
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
    _logger.info("ML pipeline: Universal Cross-Sectional GP & Regime Inference (1h).")
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
    )

    # Update data_maps with ranked ML features
    for sym in valid_symbols:
        if sym in ml_out.meta_feature_frame_by_symbol:
            mff = ml_out.meta_feature_frame_by_symbol[sym].copy()
            # Ensure mff['datetime'] is tz-aware UTC for comparison
            mff["datetime"] = pd.to_datetime(mff["datetime"], utc=True)
            
            cutoff_dt = pd.to_datetime(is_end_date or end_date)
            if cutoff_dt.tzinfo is None:
                cutoff_dt = cutoff_dt.tz_localize("UTC")
            else:
                cutoff_dt = cutoff_dt.tz_convert("UTC")
            
            # Extract only the newly added ML features to merge into the existing map
            hmm_dyn = sorted(
                (c for c in mff.columns if str(c).startswith("hmm_prob_")),
                key=lambda x: int(str(x).split("_")[-1]),
            )
            ml_cols = [
                "datetime",
                "gp_alpha_00",
                "hmm_modulator",
                "slot_rank_score",
                "ml_calib_prob",
                "ml_calib_prob_long",
                "ml_calib_prob_short",
                *hmm_dyn,
            ]
            ml_cols = [c for c in ml_cols if c in mff.columns]
            ml_features = mff[ml_cols].copy()
            
            # [FIX] Avoid column name collisions by dropping existing ML columns before merge
            drop_cols = [c for c in ml_cols if c != "datetime"]
            
            # Update IS Data
            original_is_df = data_maps[sym][args.tf]
            is_cols_to_drop = [c for c in drop_cols if c in original_is_df.columns]
            if is_cols_to_drop:
                original_is_df = original_is_df.drop(columns=is_cols_to_drop)
            
            merged_is = pd.merge(
                original_is_df, ml_features, on="datetime", how="left"
            ).sort_values("datetime")
            
            if "gp_alpha_00" in merged_is.columns:
                nan_pct = float(merged_is["gp_alpha_00"].isna().mean() * 100.0)
                _logger.info(f" [MERGE] IS {sym} gp_alpha_00 NaN ratio: {nan_pct:.4f}%")
            data_maps[sym][args.tf] = merged_is.fillna(0.0)

            # Update OOS Data
            original_oos_df = oos_data_maps[sym][args.tf]
            oos_cols_to_drop = [c for c in drop_cols if c in original_oos_df.columns]
            if oos_cols_to_drop:
                original_oos_df = original_oos_df.drop(columns=oos_cols_to_drop)
            
            merged_oos = pd.merge(
                original_oos_df, ml_features, on="datetime", how="left"
            ).sort_values("datetime")
            
            if "gp_alpha_00" in merged_oos.columns:
                nan_pct_oos = float(merged_oos["gp_alpha_00"].isna().mean() * 100.0)
                _logger.info(f" [MERGE] OOS {sym} gp_alpha_00 NaN ratio: {nan_pct_oos:.4f}%")
            oos_data_maps[sym][args.tf] = merged_oos.fillna(0.0)

    # Safe Cache Cleanup (Avoid rmtree hang)
    from src.domain.futures.opt_futures_utils.signal_cache import _MEM_CACHE, DISK_CACHE_ROOT
    _MEM_CACHE.clear()
    if DISK_CACHE_ROOT.exists():
        for f in DISK_CACHE_ROOT.glob("*.parquet"):
            try:
                f.unlink()
            except Exception:  # noqa: S110
                pass

    # [PHASE 5] Optuna Portfolio Optimization Starting
    n_ml_trials = int(args.trials)
    seed = int(OPT_FUTURES_CONFIG.get("seeds", [42])[0])
    ml_ctx = MLPhaseDContext(data_maps=data_maps, symbols=valid_symbols, tf=args.tf, seed=seed)
    
    # [VISIBILITY] Silence individual trial logs for a clean progress bar
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Custom Progress Callback for visibility in non-TTY environments
    def progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        batch_size = max(1, n_ml_trials // 10)
        if n_completed > 0 and (n_completed % batch_size == 0 or n_completed == n_ml_trials):
            _logger.info(f" [PHASE 5] Progress: {n_completed}/{n_ml_trials} trials complete...")

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
    n_jobs = _resolve_futures_parallel_policy(len(valid_symbols))
    _logger.info("\n" + "=" * 85)
    _logger.info(f" [PHASE 5] Optuna Optimization: {n_ml_trials} trials | {n_jobs} cores")
    _logger.info("=" * 85 + "\n")

    study_ml = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=NSGAIISampler(seed=seed),
    )
    
    # If using n_jobs > 1, TqdmCallback might be tricky. 
    study_ml.optimize(
        lambda tr: objective_ml_phase_d(tr, ml_ctx), 
        n_trials=n_ml_trials, 
        callbacks=callbacks,
        n_jobs=n_jobs
    )

    # Final Report & Saving
    try:
        valid_trials = [
            t
            for t in study_ml.best_trials
            if all(c <= 0.0 for c in futures_frozen_trial_constraints(t))
        ]
        best_trial = topsis_select_best(valid_trials if valid_trials else study_ml.best_trials)
    except Exception:
        best_trial = study_ml.trials[0]

    params = build_ml_phase_d_params(dict(best_trial.params), args.tf)
    
    _logger.info("\nFinalizing OOS Evaluation with Best Parameters...")
    oos_port = run_oos_margin_shared_portfolio(
        valid_symbols, args.tf, params, oos_data_maps, cache_root=FUTURES_CACHE_DIR
    )

    n_wf = int(OPT_FUTURES_CONFIG.get("FUTURES_WF_OOS_LEGS", 1))
    if n_wf > 1 and valid_symbols:
        ref_sym = valid_symbols[0]
        ref_df = oos_data_maps[ref_sym][args.tf]
        o0 = int(oos_data_maps[ref_sym][f"oos_start_idx_{args.tf}"])
        span = max(0, len(ref_df) - o0)
        leg_w = max(1, span // n_wf)
        for leg in range(n_wf):
            ls = o0 + leg * leg_w
            le = o0 + (leg + 1) * leg_w if leg < n_wf - 1 else len(ref_df)
            leg_port = run_oos_margin_shared_portfolio(
                valid_symbols,
                args.tf,
                params,
                oos_data_maps,
                cache_root=FUTURES_CACHE_DIR,
                oos_start_idx=ls,
                oos_end_idx=le,
            )
            tw = float(leg_port.get("terminal_wealth_ratio", 1.0))
            _logger.info(
                " [WF] OOS leg %d/%d idx [%d,%d) terminal_wealth_ratio=%.4f",
                leg + 1,
                n_wf,
                ls,
                le,
                tw,
            )
    
    _logger.info("\n" + "=" * 85)
    _logger.info(" [FINAL OOS PERFORMANCE REPORT]")
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
