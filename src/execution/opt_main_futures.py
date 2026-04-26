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
from typing import Any

import numpy as np
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
    FUTURES_INITIAL_BALANCE,
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
from src.domain.futures.opt_futures_utils.mc_gate_adjust import (  # noqa: E402
    resolve_adjusted_gates,
    wf_path_ergodicity_deviation_pct,
)
from src.domain.futures.opt_futures_utils.metrics import (  # noqa: E402
    calc_net_alpha_with_friction,
    calc_time_to_target_wealth,
)
from src.domain.futures.opt_futures_utils.objective import (  # noqa: E402
    inject_cs_momentum_ranks,
)
from src.domain.futures.opt_futures_utils.objective_ml import (  # noqa: E402
    MLPhaseDContext,
    build_ml_phase_d_params,
    build_phase_d_enqueue_params_from_deploy_json,
    check_hard_gates_ml,
    objective_ml_phase_d,
    precompute_ml_optimization_context,
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

def _ml_phase_d_sampler(seed: int, n_trials: int = 200) -> optuna.samplers.BaseSampler:
    # NSGA-II: 2-obj Pareto (Growth | Stability). population_size from config.
    # Required: n_trials ≥ population_size * 10 (≥10 generations) for convergence.
    if OPT_FUTURES_CONFIG.get("FUTURES_ML_GP_NSGA2_ENABLED", False):
        pop = int(OPT_FUTURES_CONFIG.get("FUTURES_NSGA2_POPULATION_SIZE", 30))
        return optuna.samplers.NSGAIISampler(
            seed=seed,
            population_size=pop,
            crossover_prob=0.9,
            mutation_prob=0.1,
        )
    # Session 36/41: n_startup_trials=n_trials ⇒ pure RandomSearch (TPE dead).
    # Honor tpe_n_startup_trials; cap below n_trials so ≥1 post-startup trial when n_trials>1.
    cfg_startup = int(OPT_FUTURES_CONFIG.get("tpe_n_startup_trials", 50))
    frac = float(OPT_FUTURES_CONFIG.get("FUTURES_ML_PHASE_D_TPE_STARTUP_FRAC", 1.0))
    frac = max(0.01, min(1.0, frac))
    from_frac = max(1, int(float(n_trials) * frac))
    n_startup = max(1, min(cfg_startup, from_frac, max(1, n_trials - 1)))
    return TPESampler(
        seed=seed,
        n_startup_trials=n_startup,
        multivariate=True,
        group=True,
        constant_liar=True,
        n_ei_candidates=24,
    )


def _print_performance_report(
    title: str,
    port: dict[str, Any],
    dsr: float | None = None,
    pbo: float | None = None,
    tf: str = "1h",
) -> None:
    # --- Extra Metrics Calculation ---
    eq = port.get("equity_curve", np.array([FUTURES_INITIAL_BALANCE]))
    trades = port.get("trades_df", pd.DataFrame())

    # 1. Volatility & Sharpe/Sortino
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
    hrs = int(tf.replace("h", "")) if tf.endswith("h") else 4
    ann_factor = (365 * 24) / hrs
    
    ann_vol = np.std(rets) * np.sqrt(ann_factor) * 100.0 if rets.size > 0 else 0.0
    sharpe = 0.0
    if rets.size > 0 and np.std(rets) > 1e-9:
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(ann_factor)
    
    downside = rets[rets < 0]
    sortino = (np.mean(rets) / np.std(downside)) * np.sqrt(ann_factor) if downside.size > 0 else 0.0
    
    # 2. t-stat of Avg Trade
    t_stat = 0.0
    if not trades.empty:
        pnl_arr = trades["pnl"].to_numpy()
        mu_pnl = np.mean(pnl_arr)
        std_pnl = np.std(pnl_arr, ddof=1)
        if std_pnl > 1e-9:
            t_stat = mu_pnl / (std_pnl / np.sqrt(len(pnl_arr)))

    # 3. Market Exposure (%)
    exposure = 0.0
    n_syms = max(1, len(port.get("symbol_names", [])))
    if not trades.empty and len(eq) > 1:
        exposure = (trades["exit_idx"] - trades["entry_idx"]).sum() / (len(eq) * n_syms)

    _logger.info("\n" + "╔" + "═" * 83 + "╗")
    _logger.info(f"║ {title:<81} ║")
    _logger.info("╠" + "═" * 83 + "╣")
    
    # Section A: COMPOUNDING (Wealth Expansion)
    _logger.info(f"║ [A] COMPOUNDING (Wealth Expansion) {' ':<46} ║")
    _logger.info(
        f"║   CAGR:        {port['cagr_pct']:>8.2f}% | MDD:         {port['mdd_pct']:>8.2f}% | "
        f"Win Rate:     {port['win_rate_pct']:>8.2f}% ║"
    )
    _logger.info(
        f"║   Terminal TW: {port.get('terminal_wealth_ratio', 1.0):>8.2f}x | "
        f"Profit Factor:{port['profit_factor']:>8.2f}  | Avg PnL %:    {port.get('avg_trade_pnl_pct', 0.0):>8.2f}% ║"  # noqa: E501
    )
    _logger.info("╟" + "─" * 83 + "╢")
    
    # Section B: ROBUSTNESS (Risk & Stability)
    _logger.info(f"║ [B] ROBUSTNESS (Risk & Stability) {' ':<47} ║")
    _logger.info(
        f"║   Sharpe:      {sharpe:>8.2f}  | Sortino:     {sortino:>8.2f}  | Ann. Vol:    {ann_vol:>8.2f}% ║"  # noqa: E501
    )
    _logger.info(
        f"║   Calmar:      {port['calmar_ratio']:>8.2f}  | Ulcer Index: {port['ulcer_index']:>8.2f}  | t-stat (Tr): {t_stat:>8.2f} ║"  # noqa: E501
    )
    _logger.info(
        f"║   Exposure:    {exposure * 100.0:>8.2f}% {' ':<56} ║"
    )

    
    if dsr is not None or pbo is not None:
        dsr_str = f"{dsr:>8.4f}" if dsr is not None else "   N/A  "
        pbo_str = f"{pbo:>8.4f}" if pbo is not None else "   N/A  "
        _logger.info(f"║   DSR (IS Ref): {dsr_str} | PBO (IS Ref): {pbo_str} {' ':<28} ║")

    _logger.info("╟" + "─" * 83 + "╢")
    _logger.info(
        f"║   Total Trades: {port['total_trades']:>8d} | L/S Minority: {port['oos_long_short_minority_pct']:>8.2f}% | "  # noqa: E501
        f"PnL/Cost:     {port.get('ev_cost_ratio', 0.0):>8.2f}  ║"
    )

    _logger.info("╚" + "═" * 83 + "╝\n")




def _print_dual_audit_dashboard(
    new_m: dict[str, Any],
    champ_m: dict[str, Any],
    gate_status: str,
) -> None:
    _logger.info("\n" + "╔" + "═" * 93 + "╗")
    _logger.info(f"║ [FINAL STRATEGY AUDIT] Candidate vs Current Champion (OOS METRICS ONLY) {' ':<14} ║")  # noqa: E501
    _logger.info("╠" + "═" * 93 + "╣")
    _logger.info(
        "║ CATEGORY          | METRIC (OOS)          | CHAMPION    | CANDIDATE   | DELTA (Δ)   ║"
    )
    _logger.info("╟" + "─" * 19 + "┼" + "─" * 23 + "┼" + "─" * 13 + "┼" + "─" * 13 + "┼" + "─" * 13 + "╢")  # noqa: E501
    
    # 1. Reliability (Statistical)
    pbo_c, pbo_n = champ_m.get("pbo", 0.5), new_m.get("pbo", 0.5)
    dsr_c, dsr_n = champ_m.get("dsr", 0.0), new_m.get("dsr", 0.0)
    
    _logger.info(
        f"║ RELIABILITY       | PBO (Lower is Better) | {pbo_c:>11.4f} | {pbo_n:>11.4f} | {pbo_n - pbo_c:>+11.4f} ║"  # noqa: E501
    )
    _logger.info(
        f"║ (Statistical)     | DSR (Stability)       | {dsr_c:>11.4f} | {dsr_n:>11.4f} | {dsr_n - dsr_c:>+11.4f} ║"  # noqa: E501
    )
    _logger.info("╟" + "─" * 19 + "┼" + "─" * 23 + "┼" + "─" * 13 + "┼" + "─" * 13 + "┼" + "─" * 13 + "╢")  # noqa: E501
    
    # 2. Compounding (Wealth & Risk)
    cagr_c, cagr_n = champ_m.get("cagr", 0.0), new_m.get("cagr", 0.0)
    mdd_c, mdd_n = champ_m.get("mdd", 0.0), new_m.get("mdd", 0.0)
    cvar_c, cvar_n = champ_m.get("cvar", 0.0), new_m.get("cvar", 0.0)
    
    _logger.info(
        f"║ COMPOUNDING       | CAGR (%)              | {cagr_c:>11.2f}% | {cagr_n:>11.2f}% | {cagr_n - cagr_c:>+11.2f}%p ║"  # noqa: E501
    )
    _logger.info(
        f"║ (Wealth & Risk)   | Max Drawdown (%)      | {mdd_c:>11.2f}% | {mdd_n:>11.2f}% | {mdd_n - mdd_c:>+11.2f}%p ║"  # noqa: E501
    )
    _logger.info(
        f"║                   | CVaR 5% (Tail Risk)   | {cvar_c:>11.2f}% | {cvar_n:>11.2f}% | {cvar_n - cvar_c:>+11.2f}%p ║"  # noqa: E501
    )
    _logger.info("╟" + "─" * 19 + "┼" + "─" * 23 + "┼" + "─" * 13 + "┼" + "─" * 13 + "┼" + "─" * 13 + "╢")  # noqa: E501

    # 3. Microstructure (Friction Proof)
    apnl_c, apnl_n = champ_m.get("avg_pnl", 0.0), new_m.get("avg_pnl", 0.0)
    pf_c, pf_n = champ_m.get("pf", 1.0), new_m.get("pf", 1.0)
    
    _logger.info(
        f"║ MICROSTRUCTURE    | Avg Trade PnL (%)     | {apnl_c:>11.2f}% | {apnl_n:>11.2f}% | {apnl_n - apnl_c:>+11.2f}%p ║"  # noqa: E501
    )
    _logger.info(
        f"║ (Friction Proof)  | Profit Factor         | {pf_c:>11.2f}  | {pf_n:>11.2f}  | {pf_n - pf_c:>+11.2f}  ║"  # noqa: E501
    )
    _logger.info("╠" + "═" * 93 + "╣")
    
    # 4. Sanity & Degradation Check
    is_cagr = new_m.get("is_cagr", 0.0)
    ho_cagr = new_m.get("ho_cagr", 0.0)
    retention = (new_m.get("cagr", 0.0) / is_cagr * 100.0) if abs(is_cagr) > 1e-6 else 0.0
    
    is_status = "PASS" if is_cagr > 80.0 else "FAIL"
    ho_status = "PASS" if ho_cagr > 60.0 else "FAIL"
    ret_label = "HEALTHY" if retention > 60.0 else "WARNING"
    
    _logger.info(f"║ [SANITY & DEGRADATION CHECK] (IS / Hold-out Reference) {' ':<39} ║")
    _logger.info("╟" + "─" * 93 + "╢")
    _logger.info(f"║ IS Path Survival : {is_status:<4} (IS CAGR: {is_cagr:>6.1f}% > 80.0%) {' ':<36} ║")  # noqa: E501
    _logger.info(f"║ Recent Regime    : {ho_status:<4} (Hold-out CAGR: {ho_cagr:>6.1f}% > 60.0%) {' ':<31} ║")  # noqa: E501
    _logger.info(f"║ OOS Degradation  : {ret_label:<7} (OOS CAGR {new_m['cagr']:>5.1f}% / IS CAGR {is_cagr:>5.1f}% = {retention:>3.0f}% Retention) {' ':<14} ║")  # noqa: E501
    _logger.info("╠" + "═" * 93 + "╣")

    _logger.info(f"║ FINAL VERDICT: {gate_status:<78} ║")
    _logger.info("╚" + "═" * 93 + "╝\n")





def _feature_slice_stats(series: pd.Series) -> tuple[float, float, float]:
    arr = pd.to_numeric(series, errors="coerce")
    n = max(int(arr.shape[0]), 1)
    nan_pct = float(arr.isna().mean()) * 100.0
    zero_pct = float((arr.notna() & (arr == 0.0)).mean()) * 100.0
    std_v = float(arr.std(ddof=0)) if n > 0 else 0.0
    return std_v, nan_pct, zero_pct


def _log_ml_merge_feature_stats(
    oos_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> None:
    cols = ("gp_alpha_00", "xs_score_long", "hmm_modulator_long")
    for col in cols:
        for sym in valid_symbols[: min(8, len(valid_symbols))]:
            df = oos_data_maps[sym][tf]
            if col not in df.columns:
                _logger.warning("[ML_MERGE] %s missing column %s", sym, col)
                continue
            o0 = int(oos_data_maps[sym][f"oos_start_idx_{tf}"])
            is_ser, oos_ser = df[col].iloc[:o0], df[col].iloc[o0:]
            is_std, is_nan, is_z = _feature_slice_stats(is_ser)
            oos_std, oos_nan, oos_z = _feature_slice_stats(oos_ser)
            _logger.debug(
                "[ML_MERGE] %s %s IS std=%.6f nan%%=%.2f zero%%=%.2f | "
                "OOS std=%.6f nan%%=%.2f zero%%=%.2f",
                sym,
                col,
                is_std,
                is_nan,
                is_z,
                oos_std,
                oos_nan,
                oos_z,
            )


def _assert_oos_gp_signal_alive(
    oos_data_maps: dict[str, dict[str, Any]], valid_symbols: list[str], tf: str
) -> None:
    for sym in valid_symbols[: min(5, len(valid_symbols))]:
        df = oos_data_maps[sym][tf]
        if "gp_alpha_00" not in df.columns:
            raise RuntimeError(f"Pre-OOS: {sym} missing gp_alpha_00.")
        gp = df["gp_alpha_00"]
        if not pd.api.types.is_numeric_dtype(gp):
            raise RuntimeError(f"Pre-OOS: {sym} gp_alpha_00 non-numeric dtype={gp.dtype}")
        o0 = int(oos_data_maps[sym][f"oos_start_idx_{tf}"])
        oos_std = float(pd.to_numeric(gp.iloc[o0:], errors="coerce").std(ddof=0) or 0.0)
        if oos_std < 1e-6:
            raise RuntimeError(f"Pre-OOS: {sym} OOS gp_alpha_00 std={oos_std:.2e} (dead signal).")


def _ml_trial_passes_hard_gates(
    trial: optuna.trial.FrozenTrial,
    pbo_obs: float = 0.0,
    check_pbo: bool = True,
    *,
    pbo_max: float | None = None,
    dsr_min: float | None = None,
) -> bool:
    cfg = OPT_FUTURES_CONFIG
    pbo_lim = float(pbo_max if pbo_max is not None else cfg.get("FUTURES_PBO_MAX", 0.40))
    if check_pbo and float(pbo_obs) >= pbo_lim:
        return False
    dsr = float(trial.user_attrs.get("gate1_dsr", -9.0))
    dsr_floor = float(dsr_min if dsr_min is not None else cfg.get("FUTURES_ML_GATE1_DSR_MIN", 0.80))
    if dsr < dsr_floor:
        return False
    p10_floor = float(cfg.get("FUTURES_CPCV_P10_LOG_TW_MIN", 0.05))
    p10_cpcv = float(trial.user_attrs.get("ml_p10_log_growth_cpcv", -999.0))
    if p10_cpcv <= p10_floor:
        return False
    mdd_limit = float(cfg.get("FUTURES_MAX_MDD", 22.0))
    if float(trial.user_attrs.get("ml_worst_mdd_cpcv", 999.0)) >= mdd_limit:
        return False
    if float(trial.user_attrs.get("avg_trades", 0.0)) < 12.0:
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
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, bool]:
    try:
        temp_is: dict[str, Any] = {}
        temp_oos: dict[str, Any] = {}
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
    symbols: list[str],
    tf: str,
    fetch_start: str,
    start: str,
    is_end: str,
    end: str,
    skip_metrics: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    data_maps: dict[str, dict[str, Any]] = {}
    oos_data_maps: dict[str, dict[str, Any]] = {}
    valid_symbols: list[str] = []
    
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

    return data_maps, oos_data_maps, valid_symbols





def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--skip-universe", action="store_true")
    pre_parser.add_argument("--reference-date", type=str, default=None)
    pre_parser.add_argument("--tf", type=str, default="1h")
    pre_args, remaining_args = pre_parser.parse_known_args()

    if not pre_args.skip_universe:
        _logger.info("\n" + "═" * 85)
        _logger.info(" [STEP 1/5] UNIVERSE DISCOVERY & DATA LOADING")
        _logger.info("═" * 85)
        
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
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args(remaining_args)

    fetch_start_date, start_date, is_end_date, end_date = get_quarterly_window(args.reference_date)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if pre_args.skip_universe:
        _logger.info("\n" + "═" * 85)
        _logger.info(" [STEP 1/5] DATA LOADING & INTEGRITY CHECK")
        _logger.info("═" * 85)

    data_maps, oos_data_maps, valid_symbols = _load_futures_data_maps_for_symbols(
        symbols, args.tf, fetch_start_date, start_date, is_end_date, end_date
    )

    if not valid_symbols:
        _logger.error(" [FAIL] No valid symbols loaded. Aborting.")
        return
    
    _logger.info(f" [SUCCESS] Data integrity check complete ({len(valid_symbols)} symbols).")



    ml_n_jobs = _resolve_futures_parallel_policy(len(valid_symbols))

    # [Institutional Quant] Universal Cross-Sectional ML Pipeline
    _logger.info("\n" + "═" * 85)
    _logger.info(" [STEP 2/5] ML PIPELINE: Universal Cross-Sectional GP & Regime Inference")
    _logger.info("═" * 85)

    
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

    _logger.info("\n" + "═" * 85)
    _logger.info(" [STEP 3/5] FEATURE INTEGRATION & SIGNAL QUALITY AUDIT")
    _logger.info("═" * 85)
    
    _logger.info("  --> Merging ML features into panel data maps...")
    merge_ml_output_into_is_and_oos(ml_out, data_maps, oos_data_maps, valid_symbols, args.tf)

    if args.tf != "1h":
        _logger.debug("  --> Syncing ML features to 1h base...")
        merge_ml_output_into_is_and_oos(ml_out, data_maps, oos_data_maps, valid_symbols, "1h")

    _logger.info("  --> Running Signal Quality Audit (IS vs OOS stability)...")
    _log_ml_merge_feature_stats(oos_data_maps, valid_symbols, args.tf)
    
    _logger.info("  [SUCCESS] Signal integration and quality audit complete.")


    for sym in valid_symbols[:3]:
        df = oos_data_maps[sym][args.tf]
        if "gp_alpha_00" not in df.columns:
            _logger.error("[SIG CHECK] %s: no gp_alpha_00 column.", sym)
            raise RuntimeError(f"OOS merge missing gp_alpha_00 for {sym}.")
        o0 = int(oos_data_maps[sym][f"oos_start_idx_{args.tf}"])
        gp = pd.to_numeric(df["gp_alpha_00"], errors="coerce")
        is_std = float(gp.iloc[:o0].std(ddof=0) or 0.0)
        oos_std = float(gp.iloc[o0:].std(ddof=0) or 0.0)
        _logger.debug("[SIG CHECK] %s IS gp_std=%.6f OOS gp_std=%.6f", sym, is_std, oos_std)
        if oos_std < 1e-4:
            _logger.error("[ABORT] %s OOS gp_alpha_00 std < 1e-4. Check merge/tz.", sym)
            raise RuntimeError(f"OOS signal dead for {sym}.")

    # [PHASE 5] Optuna Portfolio Optimization Starting
    n_ml_trials = int(args.trials)
    if OPT_FUTURES_CONFIG.get("FUTURES_ML_GP_NSGA2_ENABLED", False):
        pop = int(OPT_FUTURES_CONFIG.get("FUTURES_NSGA2_POPULATION_SIZE", 30))
        min_nsga2_trials = pop * 10
        if n_ml_trials < min_nsga2_trials:
            _logger.warning(
                "[NSGA-II] trials=%d < pop*10=%d → auto-bumped to %d for ≥10 generations.",
                n_ml_trials, min_nsga2_trials, min_nsga2_trials,
            )
            n_ml_trials = min_nsga2_trials
    pbo_max_eff, dsr_min_eff, pbo_champ_eff = resolve_adjusted_gates(
        OPT_FUTURES_CONFIG, n_ml_trials
    )
    if bool(OPT_FUTURES_CONFIG.get("FUTURES_MC_GATE_TRIAL_ADJUST_ENABLED", False)):
        _logger.info(
            "[MC-GATE] trial_budget=%d PBO_max=%.4f DSR_min=%.4f champion_PBO_max=%.4f",
            n_ml_trials,
            pbo_max_eff,
            dsr_min_eff,
            pbo_champ_eff,
        )
    cfg_seed = int(OPT_FUTURES_CONFIG.get("seeds", [42])[0])
    seed = int(args.seed) if args.seed is not None else cfg_seed
    ml_ctx = MLPhaseDContext(data_maps=data_maps, symbols=valid_symbols, tf=args.tf, seed=seed)
    
    # [PERFORMANCE] Precompute context upfront to avoid race conditions and per-trial overhead
    precompute_ml_optimization_context(ml_ctx)
    
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
        from optuna.integration import TqdmCallback  # type: ignore[attr-defined]
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

    is_nsga2 = OPT_FUTURES_CONFIG.get("FUTURES_ML_GP_NSGA2_ENABLED", False)
    # NSGA-II: n_jobs=1 (genetic evolution requires sequential population updates)
    n_jobs = 1 if is_nsga2 else min(4, _resolve_futures_parallel_policy(len(valid_symbols)))
    _logger.info("\n" + "═" * 85)
    _logger.info(f" [STEP 4/5] PORTFOLIO OPTIMIZATION: {n_ml_trials} Trials | {n_jobs} Cores")
    _logger.info("═" * 85 + "\n")


    study_ml = optuna.create_study(
        directions=["minimize", "minimize"] if is_nsga2 else ["minimize"],
        sampler=_ml_phase_d_sampler(seed, n_ml_trials),
    )

    if not is_nsga2 and bool(
        OPT_FUTURES_CONFIG.get("FUTURES_ML_PHASE_D_ENQUEUE_DEPLOY_JSON", False)
    ):
        rel = str(
            OPT_FUTURES_CONFIG.get(
                "FUTURES_ML_PHASE_D_DEPLOY_JSON_REL", "results/best_futures_1h.json"
            )
        )
        deploy_path = Path(project_root) / rel
        if deploy_path.is_file():
            try:
                with open(deploy_path, encoding="utf-8") as bf:
                    deploy_data = json.load(bf)
                enq = build_phase_d_enqueue_params_from_deploy_json(deploy_data)
                if enq is not None:
                    study_ml.enqueue_trial(enq)
                    _logger.info(
                        "[PHASE-D] Enqueued deploy baseline from %s (warm-start; in n_trials).",
                        rel,
                    )
                else:
                    _logger.warning(
                        "[PHASE-D] Deploy enqueue skipped: map %s → Phase-D params failed.",
                        rel,
                    )
            except (OSError, json.JSONDecodeError, TypeError) as _enq_e:
                _logger.warning("[PHASE-D] Deploy enqueue read failed: %s", _enq_e)
        else:
            _logger.info("[PHASE-D] Deploy enqueue skipped: missing file %s", rel)

    study_ml.optimize(
        lambda tr: objective_ml_phase_d(tr, ml_ctx),
        n_trials=n_ml_trials,
        callbacks=callbacks,
        n_jobs=n_jobs,
    )

    completed = [t for t in study_ml.trials if t.state == TrialState.COMPLETE]
    if OPT_FUTURES_CONFIG.get("FUTURES_ML_GP_NSGA2_ENABLED", False):
        from src.domain.futures.opt_futures_utils.objective_ml import topsis_select_best
        # Sort by topsis distance to ideal point for multi-objective
        try:
            best_nsga = topsis_select_best(completed)
            # Put the best NSGA-II trial at the front so the loop considers it first
            completed.sort(key=lambda tr: 0 if tr.number == best_nsga.number else 1)
        except Exception as e:
            _logger.warning("TOPSIS sorting failed: %s. Falling back to obj1 sort.", e)
            completed.sort(key=lambda tr: tr.values[0] if tr.values else 1e18)
    else:
        completed.sort(key=lambda tr: tr.value if tr.value is not None else 1e18)
    pruned_n = sum(1 for t in study_ml.trials if t.state == TrialState.PRUNED)
    n_trials_all = len(study_ml.trials)
    _logger.info(
        "[STEP 4.5/5] completed=%d pruned=%d (zero-trade proxy: %.1f%%)",
        len(completed),
        pruned_n,
        100.0 * pruned_n / max(1, n_trials_all),
    )
    if pruned_n >= 0.8 * max(1, n_trials_all):
        _logger.error(
            "[ABORT] Pruned ratio >= 80%%. Suspect signal path. "
            "Review xs_score, gp_alpha_00, computed_dir logs.",
        )
        raise RuntimeError("Optimization signal path broken: mass-pruning detected.")

    mai = ml_ctx.multi_alignment_info or {}
    paths = mai.get("cpcv_paths")
    br = mai.get("cpcv_all_block_ranges")
    best_trial: optuna.trial.FrozenTrial | None = None
    pbo_obs = 0.5 # Default IS-PBO reference
    _logger.info("\n" + "═" * 85)
    _logger.info(" [STEP 4.5/5] ROBUSTNESS AUDIT (Hard Gates & CPCV PBO)")
    _logger.info("═" * 85)
    
    iter_limit = min(300, len(completed))
    for i in range(iter_limit):
        t = completed[i]
        # 1. Cheap checks first (DSR, MDD, Trades)
        if not _ml_trial_passes_hard_gates(
            t, check_pbo=False, pbo_max=pbo_max_eff, dsr_min=dsr_min_eff
        ):
            continue

        # 2. Expensive PBO check only if cheap gates pass
        p_params = build_ml_phase_d_params(dict(t.params), args.tf)
        pbo_obs = 1.0
        if paths and br:
            oos_tw = t.user_attrs.get("cpcv_path_oos_log_tw") or []
            if len(oos_tw) == len(paths):
                _logger.info(f"  Evaluating Rank {i+1}: Checking PBO stability...")
                pbo_obs, _rho = run_cpcv_complement_evaluation(
                    p_params,
                    valid_symbols,
                    args.tf,
                    data_maps,
                    paths,
                    br,
                    oos_path_scores=oos_tw,
                )
        
        if _ml_trial_passes_hard_gates(
            t, float(pbo_obs), check_pbo=True, pbo_max=pbo_max_eff, dsr_min=dsr_min_eff
        ):
            _logger.info(f"  [PASS] Best Trial found at rank {i+1} (PBO={pbo_obs:.4f})")
            best_trial = t
            break
    
    if best_trial is None:
        trade_ok = [
            t
            for t in completed
            if float(t.user_attrs.get("avg_trades", 0.0)) >= 10.0
        ]
        if trade_ok:
            best_trial = select_best_trial_by_holdout_log_ret(trade_ok)
            _logger.warning(" [WARNING] PBO/DSR gates failed. Falling back to trade-volume filter.")
        else:
            _logger.error(" [ABORT] Signal path broken: no tradeable trials.")
            raise RuntimeError("No trial produced tradeable output.")

    params = build_ml_phase_d_params(dict(best_trial.params), args.tf)
    _assert_oos_gp_signal_alive(oos_data_maps, valid_symbols, args.tf)

    _logger.info("\n" + "═" * 85)
    _logger.info(" [STEP 5/5] FINAL OOS EVALUATION & WF ADAPTATION")
    _logger.info("═" * 85)
    
    oos_port = run_oos_margin_shared_portfolio(
        valid_symbols, args.tf, params, oos_data_maps, cache_root=FUTURES_CACHE_DIR
    )

    gate_ok = True
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
        gate_ok = check_hard_gates_ml(
            oos_port,
            float(pbo_obs),
            dsr_obs,
            0.55,
            pbo_max_override=pbo_max_eff,
            dsr_min_override=dsr_min_eff,
        )
        _logger.info(
            " [PHASE 3 AUDIT] PBO=%.4f | DSR=%.4f | RESULT: %s",
            float(pbo_obs),
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
        tw_legs: list[float] = []
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
            # CRISIS% diagnostic per WF leg — uses original OOS HMM (pre-refit) for comparison.
            # ref_df already has hmm_prob_crisis from main pipeline merge.
            if "hmm_prob_crisis" in ref_df.columns:
                _crisis_thr = float(OPT_FUTURES_CONFIG.get("FUTURES_HMM_CRISIS_THRESHOLD", 0.6))
                _leg_pc = ref_df["hmm_prob_crisis"].iloc[ls:le].to_numpy(dtype=np.float64)
                _crisis_hard_pct = float(np.mean(_leg_pc > _crisis_thr)) * 100.0
                _crisis_avg_pct = float(np.mean(_leg_pc)) * 100.0
                _logger.info(
                    " [WF CRISIS] leg %d/%d [%d,%d): bars_above_thr(%.2f)=%.1f%% avg_prob=%.1f%%",
                    leg + 1,
                    n_wf,
                    ls,
                    le,
                    _crisis_thr,
                    _crisis_hard_pct,
                    _crisis_avg_pct,
                )
        _logger.info(" [WF] sum terminal_wealth_ratio (all legs)=%.4f", wf_tw_sum)
        if len(tw_legs) >= 2:
            _erg = wf_path_ergodicity_deviation_pct(tw_legs)
            _eguide = float(OPT_FUTURES_CONFIG.get("FUTURES_ERGODICITY_GUIDELINE_PCT", 15.0))
            _logger.info(
                " [ERGODICITY] wf_leg_tw max_deviation_from_mean=%.2f%% "
                "(guideline %.1f%%)",
                _erg,
                _eguide,
            )
            if bool(OPT_FUTURES_CONFIG.get("FUTURES_ERGODICITY_SOFT_WARN_ENABLED", True)):
                if _erg > _eguide:
                    _logger.warning(
                        " [ERGODICITY HARD GATE] max_deviation %.2f%% "
                        "exceeds guideline %.1f%%. Failing gate.",
                        _erg,
                        _eguide,
                    )
                    gate_ok = False
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
            if not (all_ok and mean_ok):
                gate_ok = False
                _logger.warning(
                    " [WF HARD GATE] Persist blocked (all legs >= %.2f and mean >= %.2f required).",
                    wf_tw_floor,
                    wf_tw_mean_min,
                )

    # [STEP 5.1/5] IS & Hold-out Evaluation
    _logger.info("[STEP 5.1/5] Evaluating IS and Hold-out Performance...")
    is_data_maps: dict[str, dict[str, Any]] = {}
    ho_data_maps: dict[str, dict[str, Any]] = {}

    mai = ml_ctx.multi_alignment_info or {}
    alignment_offsets = mai.get("alignment_offsets", {})
    eff_len = mai.get("eff_ref_len", 0)
    ho_ratio = 0.20
    cpcv_zone_len = max(200, int(eff_len * (1.0 - ho_ratio)))

    for sym in valid_symbols:
        # Get perfectly aligned IS start used during model training
        sym_is_start = data_maps[sym].get(f"is_start_idx_{args.tf}", 0)
        aligned_is_start = alignment_offsets.get(sym, sym_is_start)

        # IS: Evaluated on the same aligned range as model training
        is_dm = data_maps[sym].copy()
        is_dm[f"oos_start_idx_{args.tf}"] = aligned_is_start
        is_data_maps[sym] = is_dm

        # Hold-out: Final 20% of the aligned IS period (consistent datetime across symbols)
        ho_dm = data_maps[sym].copy()
        # Safety: Ensure hold-out start doesn't exceed data length
        ho_start = min(aligned_is_start + cpcv_zone_len, len(ho_dm[args.tf]) - 2)
        ho_dm[f"oos_start_idx_{args.tf}"] = max(aligned_is_start, ho_start)
        ho_data_maps[sym] = ho_dm

    is_port = run_oos_margin_shared_portfolio(
        valid_symbols, args.tf, params, is_data_maps, cache_root=FUTURES_CACHE_DIR
    )
    ho_port = run_oos_margin_shared_portfolio(
        valid_symbols, args.tf, params, ho_data_maps, cache_root=FUTURES_CACHE_DIR
    )

    # [STEP 5.2/5] Final Performance Reports
    dsr_obs = float(best_trial.user_attrs.get("gate1_dsr", 0.0))
    # pbo_obs might have been calculated in the Hard Gate check block, let's ensure it's available
    pbo_val = locals().get("pbo_obs", None)
    
    is_port["symbol_names"] = valid_symbols
    ho_port["symbol_names"] = valid_symbols
    oos_port["symbol_names"] = valid_symbols

    _print_performance_report(
        "IS (In-Sample) PERFORMANCE REPORT", is_port, dsr=dsr_obs, pbo=pbo_val, tf=args.tf
    )
    _print_performance_report(
        "Hold-out PERFORMANCE REPORT", ho_port, dsr=dsr_obs, pbo=pbo_val, tf=args.tf
    )
    _print_performance_report(
        "FINAL OOS PERFORMANCE REPORT", oos_port, dsr=dsr_obs, pbo=pbo_val, tf=args.tf
    )

    # [IS Structural Balance Gate] Added 2026-04-22 after lgbm-200-v2 wrongly saved IS=-3.4%.
    # Root cause: CPCV path mean (used in objective) can be positive while full-IS CAGR is
    # negative — partial CPCV test folds cover only favorable sub-periods. This gate catches
    # the divergence. Validated: v5 repro (IS=-0.75%) correctly blocked, v4 champion preserved.
    if gate_ok:
        is_cagr_v = float(is_port.get("cagr_pct", is_port.get("cagr", 0.0)))
        rets_is = np.diff(is_port.get("equity_curve", np.array([FUTURES_INITIAL_BALANCE])))
        hrs_is = int(args.tf.replace("h", "")) if args.tf.endswith("h") else 4
        ann_f = (365 * 24) / hrs_is
        is_sharpe_v = 0.0
        if rets_is.size > 0 and np.std(rets_is) > 1e-9:
            is_sharpe_v = float(np.mean(rets_is) / np.std(rets_is)) * np.sqrt(ann_f)
            
        oos_eq = oos_port.get("equity_curve", np.array([FUTURES_INITIAL_BALANCE]))
        oos_rets = np.diff(oos_eq) / np.maximum(oos_eq[:-1], 1e-9)
        oos_sharpe_v = 0.0
        if oos_rets.size > 0 and np.std(oos_rets) > 1e-9:
            oos_sharpe_v = float(np.mean(oos_rets) / np.std(oos_rets)) * np.sqrt(ann_f)

        # P4: Gate on CPCV growth_signal, not sequential IS CAGR. Use IS CAGR as sanity check only.
        cpcv_growth = float(best_trial.user_attrs.get("ml_mean_log_growth_cpcv", 0.0))
        _pbo_cur = float(pbo_val) if pbo_val is not None else 0.5
        
        # IS sanity: CPCV log-TW growth > 0.05 AND sequential IS CAGR > 30%
        cpcv_pass = (cpcv_growth > 0.05) and (is_cagr_v > 30.0)

        if not cpcv_pass:
            gate_ok = False
            _logger.warning(
                " [IS STRUCTURAL GATE] CPCV Growth=%.4f IS CAGR=%.2f%% "
                "IS Sharpe=%.2f OOS Sharpe=%.2f "
                "PBO=%.4f FAIL. CPCV Growth must be > 0.05 and IS CAGR > 30.0%%. Blocked.",
                cpcv_growth,
                is_cagr_v,
                is_sharpe_v,
                oos_sharpe_v,
                _pbo_cur,
            )
        else:
            _logger.info(
                " [IS STRUCTURAL GATE] PASS. CPCV Growth=%.4f IS CAGR=%.2f%% "
                "IS Sharpe=%.2f OOS Sharpe=%.2f "
                "PBO=%.4f",
                 cpcv_growth, is_cagr_v, is_sharpe_v, oos_sharpe_v, _pbo_cur,
            )

        p10_cpcv = float(best_trial.user_attrs.get("ml_p10_log_growth_cpcv", -10.0))
        cvar10_cpcv = float(best_trial.user_attrs.get("ml_cvar10_log_growth_cpcv", -10.0))
        worst_path_cpcv = float(
            best_trial.user_attrs.get("ml_worst_path_log_growth_cpcv", -10.0)
        )
        p10_tw = float(np.exp(p10_cpcv))
        p10_floor = float(OPT_FUTURES_CONFIG.get("FUTURES_CPCV_P10_LOG_TW_MIN", 0.0))
        dist_ok = p10_cpcv > p10_floor
        _logger.info(
            " [CPCV HARDENING] p10_log_tw=%.4f tw=%.4f | cvar10=%.4f | worst=%.4f | "
            "floor=%.4f -> %s",
            p10_cpcv,
            p10_tw,
            cvar10_cpcv,
            worst_path_cpcv,
            p10_floor,
            "PASS" if dist_ok else "FAIL",
        )
        if not dist_ok:
            gate_ok = False
            _logger.warning(
                " [CPCV HARDENING] Persist blocked. 10th percentile CPCV path must satisfy "
                "log(TW) > %.4f (TW > %.4f).",
                p10_floor,
                float(np.exp(p10_floor)),
            )

    # [Champion Comparison Guard] Only overwrite if new run improves OOS CAGR or PBO.
    # Prevents regression from gate-passing runs that are still worse than the current champion.
    # Reads metrics from logs/experiments/champion.json (updated by hardening workflow).
    if gate_ok:
        champion_json_path = Path(project_root) / "logs" / "champion.json"
        if champion_json_path.exists():
            try:
                with open(champion_json_path) as _cf:
                    _champ = json.load(_cf)
                _champ_oos = float(_champ.get("metrics", {}).get("oos_cagr_pct", -999.0))
                _met_g = _champ.get("metrics", {})
                _champ_pbo = float(_met_g.get("pbo_paired", _met_g.get("pbo", 1.0)))
                _new_oos = float(oos_port.get("cagr_pct", oos_port.get("cagr", 0.0)))
                _new_ho = float(ho_port.get("cagr_pct", ho_port.get("cagr", 0.0)))
                _champ_ho = float(_champ.get("metrics", {}).get("holdout_cagr_pct", -999.0))
                
                _oos_improved = _new_oos > _champ_oos
                _pbo_improved = float(pbo_obs) < _champ_pbo
                _oos_acceptable = _new_oos > (0.75 * _champ_oos)

                _pbo_champ_max = float(pbo_champ_eff)
                _robustness_upgrade = (_champ_ho < 0) and (_new_ho > 0) and (
                    float(pbo_obs) < (_pbo_champ_max - 0.05)
                )

                _pbo_strict = float(pbo_obs) <= _pbo_champ_max
                # Reject if hold-out regresses below max(50% of champion, 8% floor).
                # Absolute 15% gate blocks champion-level candidates; relative avoids Catch-22.
                _holdout_fail = (_champ_ho > 0.0) and (_new_ho < max(0.5 * _champ_ho, 8.0))
                _pbo_regression = float(pbo_obs) > (_champ_pbo + 0.05)
                _oos_large_jump = _new_oos >= (_champ_oos + 10.0)

                _base_condition = (
                    _oos_improved or (_pbo_improved and _oos_acceptable) or _robustness_upgrade
                )
                if _pbo_regression and not _oos_large_jump:
                    _base_condition = False
                if _holdout_fail:
                    _base_condition = False
                if not (_base_condition and _pbo_strict):
                    gate_ok = False
                    _logger.warning(
                        " [CHAMPION GUARD] No improvement over current champion "
                        "(OOS %.2f%% vs %.2f%% | PBO %.4f vs %.4f | HO %.2f%% vs %.2f%%). "
                        "Champion preserved.",
                        _new_oos,
                        _champ_oos,
                        float(pbo_obs),
                        _champ_pbo,
                        _new_ho,
                        _champ_ho,
                    )
                else:
                    _label = "Robustness Upgrade" if _robustness_upgrade else "Improvement"
                    _logger.info(
                        " [CHAMPION GUARD] %s: OOS %.2f%%->%.2f%% | PBO %.4f->%.4f | "
                        "HO %.2f%%->%.2f%%.",
                        _label,
                        _champ_oos,
                        _new_oos,
                        _champ_pbo,
                        float(pbo_obs),
                        _champ_ho,
                        _new_ho,
                    )
            except Exception as _ce:
                _logger.warning(
                    " [CHAMPION GUARD] champion.json read failed (%s). Guard skipped.", _ce
                )

    # [Dual-Audit Dashboard] Integrated Performance & Reliability side-by-side
    champion_json_path = Path(project_root) / "logs" / "champion.json"
    champ_m: dict[str, Any] = {
        "pbo": 0.5, "p10": 0.0, "dsr": 0.0, "tw": 1.0, "cagr": 0.0, "mdd": 0.0,
        "time_2x": 999.0, "cvar": 0.0, "net_alpha": 0.0, "avg_pnl": 0.0, "pf": 1.0
    }
    if champion_json_path.exists():
        try:
            with open(champion_json_path) as _cf:
                _c = json.load(_cf)
            _met = _c.get("metrics", {})
            champ_m = {
                "pbo": float(_met.get("pbo_paired", _met.get("pbo", 0.5))),
                "p10": float(_met.get("cpcv_p10_log_tw", 0.0)),
                "dsr": float(_met.get("dsr", 0.0)),
                "tw": float(_met.get("oos_terminal_wealth", 1.0)),
                "cagr": float(_met.get("oos_cagr_pct", 0.0)),
                "mdd": float(_met.get("oos_mdd_pct", 0.0)),
                "time_2x": float(_met.get("oos_time_to_2x", 999.0)),
                "cvar": float(_met.get("oos_cvar_pct", 0.0)),
                "net_alpha": float(_met.get("oos_net_alpha_pct", 0.0)),
                "avg_pnl": float(_met.get("oos_avg_trade_pnl_pct", 0.0)),
                "pf": float(_met.get("oos_profit_factor", 1.0)),
            }
        except Exception as _ce:
            _logger.debug("Champion metrics parse failed: %s", _ce)

    # SOTA WEALTH (futures-opt) calculation for Candidate
    eq_arr = np.asarray(oos_port.get("equity_curve", []), dtype=np.float64)
    hrs = int(args.tf.replace("h", "")) if args.tf.endswith("h") else 4
    bpy = (24.0 / hrs) * 365.0
    if eq_arr.size > 1:
        step_log = np.log(np.clip(eq_arr[1:] / eq_arr[:-1], 1e-9, None))
        t2x_n, _ = calc_time_to_target_wealth(step_log, 2.0, bpy)
        nalpha_n = calc_net_alpha_with_friction(eq_arr, 0.0, bpy)
    else:
        t2x_n, nalpha_n = 999.0, 0.0

    new_m = {
        "pbo": float(pbo_obs) if 'pbo_obs' in locals() else 0.5,
        "p10": float(best_trial.user_attrs.get("ml_p10_log_growth_cpcv", 0.0)),
        "dsr": float(best_trial.user_attrs.get("gate1_dsr", 0.0)),
        "tw": float(oos_port.get("terminal_wealth_ratio", 1.0)),
        "cagr": float(oos_port.get("cagr_pct", 0.0)),
        "mdd": float(oos_port.get("mdd_pct", 0.0)),
        "time_2x": float(t2x_n),
        "cvar": float(oos_port.get("cvar_pct", 0.0)),
        "net_alpha": float(nalpha_n * 100.0),
        "avg_pnl": float(oos_port.get("avg_trade_pnl_pct", 0.0)),
        "pf": float(oos_port.get("profit_factor", 1.0)),
        "is_cagr": float(is_port.get("cagr_pct", 0.0)),
        "ho_cagr": float(ho_port.get("cagr_pct", 0.0)),
    }
    
    _verdict = "PROMOTE ✅" if gate_ok else "HOLD ❌"
    _print_dual_audit_dashboard(new_m, champ_m, _verdict)

    res_dir = Path(project_root) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = res_dir / f"{BEST_PARAMS_FUTURES_JSON_STEM}.json"
    if gate_ok:
        with open(best_params_path, "w") as f:
            json.dump(params, f, indent=4)
        _logger.info(f"Best parameters saved to results/{BEST_PARAMS_FUTURES_JSON_STEM}.json")
    else:
        _logger.warning(
            "Best parameters NOT persisted (Phase3, WF legs, IS structural, or champion guard). "
            "Preserving existing results/%s.json artifact.",
            BEST_PARAMS_FUTURES_JSON_STEM,
        )


if __name__ == "__main__":
    main()
