from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import importlib
import json
import logging
import math
import gc
import multiprocessing
import os
import re
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from joblib import Parallel, delayed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

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
from config.ops_profiles import check_run_summary_against_profile, resolve_ops_profile  # noqa: E402
from config.opt_config import (  # noqa: E402
    FUTURES_ANCHOR_SYMBOLS,
    FUTURES_MACRO_INDEX_SYMBOLS,
    FUTURES_SCREENER_CONFIG,
    OPT_FUTURES_CONFIG,
    get_quarterly_window,
)
from config.settings import (  # noqa: E402
    FUTURES_CACHE_DIR,
    FUTURES_DATA_DIR,
    FUTURES_INITIAL_BALANCE,
)
from src.core.indicators.numpy_ops_futures import compute_atr_numpy  # noqa: E402
from src.core.optimization.opt_utils import compute_segment_merge_index  # noqa: E402
from src.domain.futures.data_loader import (  # noqa: E402
    DataCollector,
    merge_funding_into_ohlcv,
    merge_metrics_into_ohlcv,
)
from src.domain.futures.ml_pipeline import run_ml_pipeline_for_universe  # noqa: E402
from src.domain.futures.ml_pipeline.pipeline_runner import (  # noqa: E402
    merge_ml_output_into_is_and_oos,
)
from src.domain.futures.optimization.evaluator import (  # noqa: E402
    calc_cvar5_loss_pct_from_equity,
    calc_mdd_from_equity,
    calc_net_alpha_with_friction,
    calc_time_to_target_wealth,
    calc_ulcer_index_from_equity,
    perform_online_capital_allocation,
    run_oos_margin_shared_portfolio,
    stationary_bootstrap_spa,
)
from src.domain.futures.optimization.optimizer import (  # noqa: E402
    EMBARGO_BARS,
    MLPhaseDContext,
    _base_engine_params,
    _cached_kill_fund_lev,
    _run_portfolio_numba_block,
    build_ml_phase_d_params,
    build_phase_d_enqueue_params_from_deploy_json,
    inject_cs_momentum_ranks,
    objective_ml_phase_d,
    precompute_ml_optimization_context,
    replay_robust_awf_for_trial_params,
    rerun_precompute_for_ctx,
    topsis_select_best,
)
from src.domain.futures.optimization.screener import (  # noqa: E402
    screen_futures_universe,
    screen_symbol_refinement_futures,
)
from src.domain.futures.optimization.validation import (  # noqa: E402
    awf_pos_frac_to_pseudo_pbo,
    resolve_adjusted_gates,
)
from src.domain.futures.portfolio.portfolio_optimizer import (  # noqa: E402
    finalize_strategy_portfolio_params,
    load_portfolio_policy_config,
)
from src.domain.futures.validation.candidate_selector import (  # noqa: E402
    CandidateSelectionResult,
)
from src.domain.futures.validation.champion_registry import (  # noqa: E402
    ChampionMetrics,
    append_champion_history,
    load_champion_metrics_for_guard,
    resolve_champion_record_path,
    run_champion_promotion_guard,
    write_champion_record,
)
from src.domain.futures.validation.tmp_md_champion import (  # noqa: E402
    collect_tmp_md_champion_gate_failures,
    tmp_md_layer1_failures_from_awf_diag,
)
from src.domain.futures.validation.unified_gates import (  # noqa: E402
    GATE_CODE_DESCRIPTIONS,
    FuturesResearchGateInput,
    evaluate_research_gates,
)
from src.domain.futures.validation.walk_forward import (  # noqa: E402
    WalkForwardConfig,
    mirror_walk_forward_result_from_awf_user_attrs,
)

warnings.filterwarnings("ignore")

# Force Linux 'fork' method for memory efficiency (CoW)
if sys.platform != "win32":
    try:
        # We use 'fork' for maximum startup speed.
        # CoW preservation is handled via gc.freeze() later.
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

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


def _safe_float(val: Any, default: float = 0.0, clip: float | None = None) -> float:
    try:
        out = float(val)
    except (TypeError, ValueError):
        out = default
    if not np.isfinite(out):
        out = default
    if clip is not None:
        out = float(np.clip(out, -abs(float(clip)), abs(float(clip))))
    return out


def _sanitize_metric_map(m: dict[str, Any]) -> dict[str, float]:
    limits = {
        "pbo": 1e3,
        "p10": 100.0,
        "dsr": 1e3,
        "tw": 1e6,
        "cagr": 1e5,
        "mdd": 1e3,
        "time_2x": 1e6,
        "cvar": 1e3,
        "net_alpha": 1e5,
        "avg_pnl": 1e5,
        "pf": 1e3,
        "is_cagr": 1e5,
        "ho_cagr": 1e5,
        "awf_pos_frac": 10.0,
        "mu_awf": 100.0,
        "sig_awf": 100.0,
        "plgd": 100.0,
        "erg_dev": 1e4,
        "oos_long_pf": 1e3,
        "oos_short_pf": 1e3,
        "oos_retention_pct": 1e5,
        "is_alpha": 1e5,
    }
    out: dict[str, float] = {}
    for k, v in m.items():
        out[k] = _safe_float(v, default=0.0, clip=limits.get(k, 1e6))
    return out

def _ml_phase_d_sampler(seed: int, n_trials: int = 200) -> optuna.samplers.BaseSampler:
    # NSGA-II: 2-obj Pareto (Growth | Stability). population_size from config.
    # Required: n_trials ≥ population_size * 10 (≥10 generations) for convergence.
    if OPT_FUTURES_CONFIG.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False):
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
        n_ei_candidates=48,
    )


def _ml_phase_d_sampler_coordinate(
    seed: int, n_trials: int, phase: str
) -> optuna.samplers.BaseSampler:
    """Phase B: more startup trials + multivariate TPE; phases A/C use 20 startup."""
    if OPT_FUTURES_CONFIG.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False):
        return _ml_phase_d_sampler(seed, n_trials)
    nt = max(2, int(n_trials))
    if phase == "B":
        n_startup = min(40, max(1, nt - 1))
        return TPESampler(
            seed=seed,
            n_startup_trials=n_startup,
            multivariate=True,
            group=True,
            constant_liar=True,
            n_ei_candidates=48,
        )
    n_startup = min(20, max(1, nt - 1))
    return TPESampler(
        seed=seed,
        n_startup_trials=n_startup,
        multivariate=False,
        group=False,
        constant_liar=True,
        n_ei_candidates=24,
    )


def _get_vis_width(s: str) -> int:
    """Calculate visual width of string, stripping ANSI and handling common emojis/symbols."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    plain = ansi_escape.sub('', s)
    w = 0
    i = 0
    while i < len(plain):
        c = plain[i]
        if c in ('\u200d', '\ufe0f'):
            i += 1
            continue
        # Heuristic for wide characters (emojis, CJK)
        if ord(c) > 0x1100 and (
            0x2e80 <= ord(c) <= 0x1f9ff  # Broad range for CJK and Emojis
        ):
            w += 2
        else:
            w += 1
        i += 1
    return w

def _fmt_row(text: str, width: int = 90, align: str = "left", char: str = " ") -> str:
    """Format a row with borders and correct visual padding."""
    v = _get_vis_width(text)
    pad = max(0, width - v)
    if align == "left":
        return f"║ {text}{char * pad} ║"
    elif align == "right":
        return f"║ {char * pad}{text} ║"
    else: # center
        l_pad = pad // 2
        r_pad = pad - l_pad
        return f"║ {char * l_pad}{text}{char * r_pad} ║"

def _print_performance_report(
    title: str,
    port: dict[str, Any],
    dsr: float | None = None,
    pbo: float | None = None,
    tf: str = "1h",
    benchmark_cagr: float | None = None,
    meta_port: dict[str, Any] | None = None,
) -> None:
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

    # 2. PSR (Probabilistic Sharpe Ratio) — López de Prado 2012
    psr = 0.5
    if rets.size >= 4:
        _sr_hat = float(np.mean(rets)) / (float(np.std(rets, ddof=1)) + 1e-12)
        _sk = float(np.nan_to_num(pd.Series(rets).skew()))
        _ex_k = float(np.nan_to_num(pd.Series(rets).kurt()))
        _denom = max(1e-12, 1.0 - _sk * _sr_hat + ((_ex_k + 2.0) / 4.0) * _sr_hat**2)
        _se_sr = math.sqrt(_denom / max(int(rets.size) - 1, 1))
        _z_psr = _sr_hat / (_se_sr + 1e-12)
        psr = float(0.5 * (1.0 + math.erf(_z_psr / math.sqrt(2.0))))

    # 3. t-stat of Avg Trade
    t_stat = 0.0
    if not trades.empty:
        pnl_arr = trades["pnl"].to_numpy()
        mu_pnl = np.mean(pnl_arr)
        std_pnl = np.std(pnl_arr, ddof=1)
        if std_pnl > 1e-9:
            t_stat = mu_pnl / (std_pnl / np.sqrt(len(pnl_arr)))

    # 4. Market Exposure (%)
    exposure = 0.0
    n_syms = max(1, len(port.get("symbol_names", [])))
    if not trades.empty and len(eq) > 1:
        exposure = (trades["exit_idx"] - trades["entry_idx"]).sum() / (len(eq) * n_syms)

    w = 90
    _logger.info("\n" + "╔" + "═" * (w + 2) + "╗")
    _logger.info(_fmt_row(title, w))
    _logger.info("╠" + "═" * (w + 2) + "╣")

    # Section A: COMPOUNDING
    _logger.info(_fmt_row("[A] COMPOUNDING (Wealth Expansion)", w))
    cagr_str = f"{port['cagr_pct']:>8.2f}%"
    mdd_str = f"{port['mdd_pct']:>8.2f}%"
    if meta_port:
        cagr_str = f"{port['cagr_pct']:>6.1f} -> {meta_port['cagr_pct']:>6.1f}%"
        mdd_str = f"{port['mdd_pct']:>6.1f} -> {meta_port['mdd_pct']:>6.1f}%"

    row1 = (
        f"  CAGR:        {cagr_str} | MDD:         {mdd_str} | "
        f"Win Rate:     {port['win_rate_pct']:>8.2f}%"
    )
    _logger.info(_fmt_row(row1, w))

    tw_str = f"{port.get('terminal_wealth_ratio', 1.0):>8.2f}x"
    if meta_port:
        tw_str = f"{port.get('terminal_wealth_ratio', 1.0):>6.1f} -> {meta_port.get('terminal_wealth_ratio', 1.0):>6.1f}x"

    row2 = (
        f"  Terminal TW: {tw_str} | "
        f"Profit Factor:{port['profit_factor']:>8.2f}  | "
        f"Avg PnL %:    {port.get('avg_trade_pnl_pct', 0.0):>8.2f}%"
    )
    _logger.info(_fmt_row(row2, w))
    _logger.info("╟" + "─" * (w + 2) + "╢")

    # Section B: ROBUSTNESS
    _logger.info(_fmt_row("[B] ROBUSTNESS (Risk & Stability)", w))

    sharpe_str = f"{sharpe:>8.2f}"
    if meta_port:
        m_rets = np.diff(meta_port["equity_curve"]) / np.maximum(meta_port["equity_curve"][:-1], 1e-9)
        m_sharpe = (np.mean(m_rets) / np.std(m_rets)) * np.sqrt(ann_factor) if np.std(m_rets) > 1e-9 else 0.0
        sharpe_str = f"{sharpe:>6.2f} -> {m_sharpe:>6.2f}"

    row3 = (
        f"  Sharpe:      {sharpe_str}  | "
        f"Sortino:     {sortino:>8.2f}  | "
        f"Ann. Vol:    {ann_vol:>8.2f}%"
    )
    _logger.info(_fmt_row(row3, w))

    calmar_str = f"{port['calmar_ratio']:>8.2f}"
    if meta_port:
        calmar_str = f"{port['calmar_ratio']:>6.2f} -> {meta_port['calmar_ratio']:>6.2f}"

    row4 = (
        f"  Calmar:      {calmar_str}  | "
        f"Ulcer Index: {port['ulcer_index']:>8.2f}  | "
        f"t-stat (Tr): {t_stat:>8.2f}"
    )
    _logger.info(_fmt_row(row4, w))
    _logger.info(_fmt_row(f"  Exposure:    {exposure * 100.0:>8.2f}%", w))
    # Section C: MARKET-RELATIVE ALPHA
    if benchmark_cagr is not None:
        _net_alpha = float(port.get("cagr_pct", 0.0)) - benchmark_cagr
        alpha_val = _net_alpha
        if meta_port:
            meta_alpha = float(meta_port.get("cagr_pct", 0.0)) - benchmark_cagr
            alpha_str = f"{_net_alpha:>+6.1f} -> {meta_alpha:>+6.1f}%"
        else:
            alpha_str = f"{_net_alpha:>+8.2f}%"

        na_txt = (
            f"  PSR (BLP):   {psr:>8.4f}  | "
            f"Benchmark:  {benchmark_cagr:>7.1f}%  | "
            f"Net Alpha:  {alpha_str}"
        )
        _logger.info(_fmt_row(na_txt, w))
    else:
        _logger.info(_fmt_row(f"  PSR (BLP):   {psr:>8.4f}", w))

    _logger.info("╟" + "─" * (w + 2) + "╢")
    row5 = (
        f"  Total Trades: {port['total_trades']:>8d} | "
        f"L/S Minority: {port['oos_long_short_minority_pct']:>8.2f}% | "
        f"PnL/Cost:     {port.get('ev_cost_ratio', 0.0):>8.2f}"
    )
    _logger.info(_fmt_row(row5, w))

    _logger.info("╚" + "═" * (w + 2) + "╝\n")


def _print_human_dashboard(
    is_port: dict[str, Any],
    ho_port: dict[str, Any],
    oos_port: dict[str, Any],
    gate_status: str,
    benchmark_is: float = 0.0,
    benchmark_oos: float = 0.0,
    meta_port: dict[str, Any] | None = None,
) -> None:
    """Unified Human Dashboard for strategy performance summary."""
    c_grn = "\033[92m"
    c_red = "\033[91m"
    c_ylw = "\033[93m"
    c_rst = "\033[0m"
    c_bld = "\033[1m"

    w = 90
    _logger.info("\n" + "╔" + "═" * (w + 2) + "╗")
    _logger.info(_fmt_row(f"{c_bld}🧑‍💻 [HUMAN DASHBOARD] STRATEGY PERFORMANCE SUMMARY{c_rst}", w))
    _logger.info("╠" + "═" * (w + 2) + "╣")

    def get_val(port: dict[str, Any], key: str, default: float = 0.0) -> float:
        return float(port.get(key, default))

    # Header Row
    h_txt = f"{'[COMPOUNDING]':<20} {'IS':>12} {'Hold-Out':>18} {'OOS (Forward)':>24}"
    _logger.info(_fmt_row(h_txt, w))

    is_cagr = get_val(is_port, "cagr_pct")
    oos_cagr = get_val(oos_port, "cagr_pct")
    is_alpha = is_cagr - benchmark_is
    oos_alpha = oos_cagr - benchmark_oos

    if meta_port:
        meta_cagr = get_val(meta_port, "cagr_pct")
        meta_alpha = meta_cagr - benchmark_oos

    metrics = [
        ("CAGR (%)", "cagr_pct", True),
        ("Net Alpha (%)", "net_alpha_pct", True),
        ("Max Drawdown (%)", "mdd_pct", True),
        ("Profit Factor", "profit_factor", False),
    ]

    for label, key, is_pct in metrics:
        suffix = "%" if is_pct else ""
        if key == "net_alpha_pct":
            oos_val_str = f"{oos_alpha:>10.2f}{suffix}"
            if meta_port:
                oos_val_str = f"{oos_alpha:>5.1f}->{meta_alpha:>4.1f}{suffix}"
            row = (
                f"  {label:<18} : {is_alpha:>10.2f}{suffix} "
                f"{'N/A':>17} {oos_val_str:>23}"
            )
            _logger.info(_fmt_row(row, w))
            continue

        is_val = get_val(is_port, key)
        ho_val = get_val(ho_port, key)
        oos_val = get_val(oos_port, key)

        oos_val_str = f"{oos_val:>10.2f}{suffix}"
        if meta_port:
            m_val = get_val(meta_port, key)
            oos_val_str = f"{oos_val:>5.1f}->{m_val:>4.1f}{suffix}"

        row = (
            f"  {label:<18} : {is_val:>10.2f}{suffix} "
            f"{ho_val:>17.2f}{suffix} {oos_val_str:>23}"
        )
        _logger.info(_fmt_row(row, w))

    _logger.info("╟" + "─" * (w + 2) + "╢")
    _logger.info(_fmt_row("[SANITY CHECK & VERDICT]", w))

    retention = (oos_cagr / is_cagr * 100.0) if is_cagr > 1e-6 else 0.0
    if meta_port:
        meta_retention = (meta_cagr / is_cagr * 100.0) if is_cagr > 1e-6 else 0.0
        ret_val = meta_retention
    else:
        ret_val = retention

    ret_color = c_grn if ret_val > 60.0 else c_ylw if ret_val > 40.0 else c_red
    ret_text = f"{ret_color}{ret_val:>5.1f}%{c_rst} of IS Performance"
    if meta_port:
        ret_text += f" (Ensemble Improvement: {meta_retention - retention:>+4.1f}%)"

    _logger.info(_fmt_row(f"  OOS Retention      : {ret_text}", w))
    _logger.info(_fmt_row("", w))

    v_color = c_grn if "PROMOTE" in gate_status else c_red
    v_msg = f"FINAL VERDICT        : {v_color}{c_bld}{gate_status}{c_rst}"
    persisted = " (Ensemble saved)" if "PROMOTE" in gate_status else " (Parameters NOT persisted)"
    _logger.info(_fmt_row(f"  {v_msg}{persisted}", w))
    _logger.info("╚" + "═" * (w + 2) + "╝\n")



def _print_dual_audit_dashboard(
    new_m: dict[str, Any],
    champ_m: dict[str, Any],
    gate_status: str,
) -> None:
    """SOTA Dashboard for Strategy Promotion Audit.

    Uses ANSI colors and calculated alignment for professional CLI output.
    """
    c_grn = "\033[92m"
    c_red = "\033[91m"
    c_ylw = "\033[93m"
    c_rst = "\033[0m"
    c_bld = "\033[1m"

    # Total content width: borders and separators included

    def get_delta_str(val: float, is_pct: bool = False, lower_is_better: bool = False) -> str:
        val = _safe_float(val, 0.0, clip=1e6)
        suffix = "%p" if is_pct else ""
        precision = ".2f" if is_pct else ".4f"

        # 1. Create the raw numeric string with sign
        raw_val = f"{val:>{precision}}"
        if val > 1e-9:
            raw_val = "+" + raw_val

        # 2. Add suffix and pad to w_val BEFORE applying color
        full_str = raw_val + suffix
        padded_val = f"{full_str:>12}"
        if abs(val) < 1e-7:
            return padded_val
        is_good = (val < 0) if lower_is_better else (val > 0)
        color = c_grn if is_good else c_red
        return f"{color}{padded_val}{c_rst}"

    w = 90
    _logger.info("\n" + "╔" + "═" * (w + 2) + "╗")
    h_title = f"{c_bld}[FINAL STRATEGY AUDIT] Candidate vs Champion (OOS ONLY){c_rst}"
    _logger.info(_fmt_row(h_title, w))
    _logger.info("╠" + "═" * (w + 2) + "╣")

    # Column widths for table internal alignment
    tw_cat = 18
    tw_met = 22
    tw_val = 12
    l_fmt_int = "{:<" + str(tw_cat) + "} │ {:<" + str(tw_met) + "} │ {:>12} │ {:>12} │ {:>12}"

    # Table Header
    h_row = l_fmt_int.format('CATEGORY', 'METRIC (OOS)', 'CHAMPION', 'CANDIDATE', 'DELTA (Δ)')
    _logger.info(_fmt_row(h_row, w))
    _logger.info("╟" + "─" * (tw_cat + 1) + "┼" + "─" * (tw_met + 2) + "┼" + "─" * (tw_val + 2) +
                 "┼" + "─" * (tw_val + 2) + "┼" + "─" * (tw_val + 2) + "╢")

    # 1. Reliability
    pbo_c, pbo_n = champ_m.get("pbo", 0.5), new_m.get("pbo", 0.5)
    dsr_c, dsr_n = champ_m.get("dsr", 0.0), new_m.get("dsr", 0.0)

    _logger.info(_fmt_row(l_fmt_int.format(
        "RELIABILITY", "PBO (Lower Better)", f"{pbo_c:.4f}", f"{pbo_n:.4f}",
        get_delta_str(pbo_n - pbo_c, lower_is_better=True).strip()
    ), w))
    _logger.info(_fmt_row(l_fmt_int.format(
        "(Statistical)", "DSR (Stability)", f"{dsr_c:.4f}", f"{dsr_n:.4f}",
        get_delta_str(dsr_n - dsr_c).strip()
    ), w))
    _logger.info("╟" + "─" * (tw_cat + 1) + "┼" + "─" * (tw_met + 2) + "┼" + "─" * (tw_val + 2) +
                 "┼" + "─" * (tw_val + 2) + "┼" + "─" * (tw_val + 2) + "╢")

    # 2. Compounding
    cagr_c, cagr_n = champ_m.get("cagr", 0.0), new_m.get("cagr", 0.0)
    mdd_c, mdd_n = champ_m.get("mdd", 0.0), new_m.get("mdd", 0.0)
    pf_c, pf_n = champ_m.get("pf", 1.0), new_m.get("pf", 1.0)
    nalpha_c, nalpha_n = champ_m.get("net_alpha", 0.0), new_m.get("net_alpha", 0.0)
    pnl_c, pnl_n = champ_m.get("avg_pnl", 0.0), new_m.get("avg_pnl", 0.0)

    _logger.info(_fmt_row(l_fmt_int.format(
        "COMPOUNDING", "CAGR (%)", f"{cagr_c:.2f}%", f"{cagr_n:.2f}%",
        get_delta_str(cagr_n - cagr_c, is_pct=True).strip()
    ), w))
    _logger.info(_fmt_row(l_fmt_int.format(
        "(Wealth & Risk)", "Max Drawdown (%)", f"{mdd_c:.2f}%", f"{mdd_n:.2f}%",
        get_delta_str(mdd_n - mdd_c, is_pct=True, lower_is_better=True).strip()
    ), w))
    _logger.info(_fmt_row(l_fmt_int.format(
        "", "Profit Factor", f"{pf_c:.2f}", f"{pf_n:.2f}",
        get_delta_str(pf_n - pf_c).strip()
    ), w))
    _logger.info(_fmt_row(l_fmt_int.format(
        "", "Net Alpha (%)", f"{nalpha_c:.2f}%", f"{nalpha_n:.2f}%",
        get_delta_str(nalpha_n - nalpha_c, is_pct=True).strip()
    ), w))
    _logger.info(_fmt_row(l_fmt_int.format(
        "", "Avg Trade PnL (%)", f"{pnl_c:.2f}%", f"{pnl_n:.2f}%",
        get_delta_str(pnl_n - pnl_c, is_pct=True).strip()
    ), w))
    _logger.info("╠" + "═" * (w + 2) + "╣")

    # 3. Sanity
    is_cagr = new_m.get("is_cagr", 0.0)
    ho_cagr = new_m.get("ho_cagr", 0.0)
    retention = (new_m.get("cagr", 0.0) / is_cagr * 100.0) if abs(is_cagr) > 1e-6 else 0.0

    def get_status_tag(val: float, threshold: float) -> str:
        color = c_grn if val >= threshold else c_red
        status = "PASS" if val >= threshold else "FAIL"
        return f"{color}{status}{c_rst}"

    _logger.info(_fmt_row("[SANITY & DEGRADATION CHECK]", w))
    is_alpha = new_m.get("is_alpha", 0.0)
    s1 = f" - IS Net Alpha     : {get_status_tag(is_alpha, 0.0)} (IS Alpha: {is_alpha:>5.1f}%)"
    _logger.info(_fmt_row(s1, w))
    s2 = f" - Recent Regime    : DIAG (HO CAGR: {ho_cagr:>5.1f}%)"
    _logger.info(_fmt_row(s2, w))

    ret_color = c_grn if retention > 60.0 else c_ylw if retention > 40.0 else c_red
    ret_txt = f"{ret_color}{retention:>5.1f}%{c_rst}"
    _logger.info(_fmt_row(f" - OOS Retention    : {ret_txt} of IS Performance", w))
    _logger.info("╠" + "═" * (w + 2) + "╣")

    # Verdict
    v_color = c_grn if "PROMOTE" in gate_status else c_red
    v_msg = f"FINAL VERDICT: {v_color}{c_bld}{gate_status}{c_rst}"
    _logger.info(_fmt_row(v_msg, w))
    _logger.info("╚" + "═" * (w + 2) + "╝\n")











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
    cols = ("ml_alpha_00", "xs_score_long", "hmm_modulator_long")
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
        if "ml_alpha_00" not in df.columns:
            raise RuntimeError(f"Pre-OOS: {sym} missing ml_alpha_00.")
        gp = df["ml_alpha_00"]
        if not pd.api.types.is_numeric_dtype(gp):
            raise RuntimeError(f"Pre-OOS: {sym} ml_alpha_00 non-numeric dtype={gp.dtype}")
        o0 = int(oos_data_maps[sym][f"oos_start_idx_{tf}"])
        oos_std = float(pd.to_numeric(gp.iloc[o0:], errors="coerce").std(ddof=0) or 0.0)
        if oos_std < 1e-6:
            raise RuntimeError(f"Pre-OOS: {sym} OOS ml_alpha_00 std={oos_std:.2e} (dead signal).")


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
    p10_floor = float(cfg.get("FUTURES_AWF_P10_LOG_TW_MIN", -0.10))
    p10_cpcv = float(
        trial.user_attrs.get(
            "awf_worst_leg_log_tw", trial.user_attrs.get("ml_p10_log_growth_cpcv", -999.0)
        )
    )
    if p10_cpcv <= p10_floor:
        return False
    mdd_limit = float(cfg.get("FUTURES_MAX_MDD", 22.0))
    if float(
        trial.user_attrs.get("awf_worst_mdd_pct", trial.user_attrs.get("ml_worst_mdd_cpcv", 999.0))
    ) >= mdd_limit:
        return False
    # Dynamic trade density: scale minimum trades with IS span to avoid regime-size bias.
    # FUTURES_BARS_PER_TRADE_EST = expected bars between trades (default 200).
    # gate1_eff_ref_len is stored by objective_ml per trial.
    span_bars = int(trial.user_attrs.get("gate1_eff_ref_len", 0))
    bars_per_trade_est = float(cfg.get("FUTURES_BARS_PER_TRADE_EST", 200))
    min_trades_dynamic = max(12.0, float(span_bars) / bars_per_trade_est) if span_bars > 0 else 12.0
    if float(trial.user_attrs.get("avg_trades", 0.0)) < min_trades_dynamic:
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
    target_tfs: list[str] | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, bool]:
    try:
        temp_is: dict[str, Any] = {}
        temp_oos: dict[str, Any] = {}
        insufficient = False
        collector = DataCollector()
        
        # Load raw dependencies once per symbol
        from src.domain.futures.ml_pipeline.pipeline_runner import (
            merge_funding_into_ohlcv, 
            merge_metrics_into_ohlcv
        )

        tfs_to_load = set(target_tfs) if target_tfs else {tf, "1d", "1h", "4h"}
        
        # [Optimization] Pre-load funding and metrics data once to avoid redundant I/O per TF
        funding_df = None
        metrics_df = None
        if not skip_metrics:
            f_path = Path(FUTURES_DATA_DIR) / f"{sym.replace('/', '_')}_funding.parquet"
            if f_path.exists():
                try:
                    funding_df = pd.read_parquet(f_path)
                except Exception: pass
            
            m_path = Path(FUTURES_DATA_DIR) / f"{sym.replace('/', '_')}_metrics.parquet"
            if m_path.exists():
                try:
                    metrics_df = pd.read_parquet(m_path)
                except Exception: pass

        for tf_l in tfs_to_load:
            raw_df = collector.collect_and_save(sym, tf_l, fetch_start, end)
            if raw_df is None or raw_df.empty:
                insufficient = True
                break
            
            if "datetime" not in raw_df.columns:
                raw_df = raw_df.reset_index()
                if "datetime" not in raw_df.columns and len(raw_df.columns) > 0:
                    raw_df = raw_df.rename(columns={str(raw_df.columns[0]): "datetime"})

            try:
                # [Optimization] Use localized merge logic to benefit from pre-loaded data
                df = raw_df.copy()
                if funding_df is not None and not funding_df.empty:
                    df["timestamp"] = pd.to_datetime(df["datetime"]).astype("int64") // 10**6
                    f_tmp = funding_df.copy()
                    f_tmp["timestamp"] = pd.to_datetime(f_tmp["timestamp"], unit="ms").astype("int64") // 10**6
                    exclude_fr = ["datetime", "symbol"]
                    cols_fr = [c for c in f_tmp.columns if c not in exclude_fr]
                    df = pd.merge_asof(df.sort_values("timestamp"), f_tmp[cols_fr].sort_values("timestamp"), on="timestamp", direction="backward")
                
                if metrics_df is not None and not metrics_df.empty:
                    if "timestamp" not in df.columns:
                        df["timestamp"] = pd.to_datetime(df["datetime"]).astype("int64") // 10**6
                    m_tmp = metrics_df.copy()
                    m_tmp["timestamp"] = pd.to_datetime(m_tmp["datetime"]).astype("int64") // 10**6
                    exclude_m = ["timestamp", "datetime", "create_time", "symbol"]
                    cols_m = [c for c in m_tmp.columns if c not in exclude_m]
                    df = pd.merge_asof(df.sort_values("timestamp"), m_tmp[["timestamp"] + cols_m].sort_values("timestamp"), on="timestamp", direction="backward")

                # Enrich with GP features
                if not skip_metrics:
                    from src.domain.futures.ml_pipeline.pipeline_runner import _enrich_with_gp_features
                    df = _enrich_with_gp_features(df, tf=tf_l)
            except Exception as e:
                _logger.error("[%s] Merge/Enrich failed: %s", sym, e)
                insufficient = True
                break

            if df is None or df.empty or "datetime" not in df.columns:
                insufficient = True
                break

            df.reset_index(drop=True, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

            is_start_dt = pd.Timestamp(start, tz="UTC")
            is_end_dt = pd.Timestamp(is_end, tz="UTC")

            is_mask = df["datetime"] < is_end_dt
            is_end_idx = int(is_mask.to_numpy().sum())

            # [Dynamic Quality Gate]
            min_bars_map = {"1h": 2000, "4h": 500, "1d": 300}
            min_bars_threshold = min_bars_map.get(tf_l, 300)

            if is_end_idx < min_bars_threshold:
                _logger.debug(
                    "[%s] %s history too short (%d < %d)",
                    sym,
                    tf_l,
                    is_end_idx,
                    min_bars_threshold,
                )
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
        _logger.debug("[%s] Critical load failure: %s", sym, e)
        return sym, None, None, True


def _load_futures_data_maps_for_symbols(
    symbols: list[str],
    tf: str,
    fetch_start: str,
    start: str,
    is_end: str,
    end: str,
    skip_metrics: bool = False,
    target_tfs: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    data_maps: dict[str, dict[str, Any]] = {}
    oos_data_maps: dict[str, dict[str, Any]] = {}
    valid_symbols: list[str] = []

    # [Fix] Filter out non-ASCII symbols before processing
    symbols = [s for s in symbols if all(ord(c) < 128 for c in s)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                _load_single_symbol_data, sym, tf, fetch_start, start, is_end, end, skip_metrics, target_tfs
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





def optimize_worker(s_name: str, s_url: str, trials: int, ctx: MLPhaseDContext):
    # Each process loads the study and runs its portion of trials
    # This completely bypasses the GIL.
    # Note: Use a local storage object to avoid sharing it across processes
    inner_storage = optuna.storages.RDBStorage(s_url, engine_kwargs={"connect_args": {"timeout": 60, "check_same_thread": False}})
    study = optuna.load_study(study_name=s_name, storage=inner_storage)
    study.optimize(
        lambda tr: objective_ml_phase_d(tr, ctx),
        n_trials=trials,
        n_jobs=1,  # 1 job per process
        catch=(ValueError, RuntimeError),
    )

def main() -> None:
    ai_telemetry_payloads: list[dict[str, Any]] = []
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--skip-universe", action="store_true")
    pre_parser.add_argument("--reference-date", type=str, default=None)
    pre_parser.add_argument("--tf", type=str, default="4h")
    pre_args, remaining_args = pre_parser.parse_known_args()

    if not pre_args.skip_universe:
        _logger.info("\n" + "═" * 85)
        _logger.info(" [STEP 1/5] UNIVERSE DISCOVERY & DATA LOADING")
        _logger.info("═" * 85)

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
            target_tfs=[pre_args.tf, "1d"],
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
    parser.add_argument("--tf", type=str, choices=["1h", "4h"], default=pre_args.tf)
    parser.add_argument("--reference-date", type=str, default=pre_args.reference_date)
    parser.add_argument("--alpha-only", action="store_true", help="Stop after ALPHA IC calculation")
    parser.add_argument("--hmm-only", action="store_true", help="Stop after HMM regime inference")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--force-retrain-alpha",
        action="store_true",
        help="Bypass Alpha raw cache and alpha retraining.",
    )
    parser.add_argument(
        "--bypass-champion-guard",
        action="store_true",
        help="Force promotion regardless of champion comparison.",
    )
    parser.add_argument(
        "--ops-profile",
        type=str,
        default=None,
        help="Preset trials/seeds: smoke | candidate | promotion (see config/ops_profiles.py).",
    )
    args = parser.parse_args(remaining_args)

    if args.force_retrain_alpha:
        OPT_FUTURES_CONFIG["FUTURES_ML_FORCE_RETRAIN_ALPHA"] = True
        _logger.info("[ML] FORCE_RETRAIN_ALPHA enabled via CLI; raw Alpha cache will be bypassed.")

    ai_telemetry_payloads.append({
        "stage": "execution_context",
        "tf": args.tf,
        "trials": args.trials,
        "seed": args.seed,
    })

    fetch_start_date, start_date, is_end_date, end_date = get_quarterly_window(args.reference_date)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if pre_args.skip_universe:
        _logger.info("\n" + "═" * 85)
        _logger.info(" [STEP 1/5] DATA LOADING & INTEGRITY CHECK")
        _logger.info("═" * 85)

    # [3-Tier Universe] Ensure Anchors and Macro Index symbols are always loaded for systemic HMM
    load_symbols = list(set(symbols + FUTURES_ANCHOR_SYMBOLS + FUTURES_MACRO_INDEX_SYMBOLS))

    data_maps, oos_data_maps, valid_symbols = _load_futures_data_maps_for_symbols(
        load_symbols, args.tf, fetch_start_date, start_date, is_end_date, end_date
    )

    if not valid_symbols:
        _logger.error(" [FAIL] No valid symbols loaded. Aborting.")
        return

    _logger.info(f" [SUCCESS] Data integrity check complete ({len(valid_symbols)} symbols).")



    ml_n_jobs = _resolve_futures_parallel_policy(len(valid_symbols))

    # [Institutional Quant] Universal Cross-Sectional ML Pipeline
    _logger.info("\n" + "═" * 85)
    _logger.info(" [STEP 2/5] ML PIPELINE: Universal Cross-Sectional Alpha & Regime Inference")
    _logger.info("═" * 85)


    # [Institutional Quant] Alpha Dual-TF: Train on 4h even if execution is 1h to avoid noise/fees decay.
    ml_train_tf = args.tf
    if args.tf == "1h":
        _logger.info("  --> [ML] ALPHA DUAL-TF: Training on 4h (Noise Reduction) -> Merging to 1h")
        ml_train_tf = "4h"

    # [Institutional Quant] Alpha Dual-TF: Train on 4h even if execution is 1h to avoid noise/fees decay.
    ml_train_tf = args.tf
    if args.tf == "1h":
        _logger.info("  --> [ML] ALPHA DUAL-TF: Training on 4h (Noise Reduction) -> Merging to 1h")
        ml_train_tf = "4h"

    # [Optimization #1] Pass preloaded broad screening data to pipeline
    ml_pipeline_cfg = dict(OPT_FUTURES_CONFIG)
    if args.ops_profile == "smoke":
        ml_pipeline_cfg["FUTURES_USE_META_LABELER"] = False
        _logger.info("  --> [OPS] SMOKE profile: Meta-Labeler disabled for speed.")

    ml_out = run_ml_pipeline_for_universe(
        valid_symbols,
        ml_train_tf,
        fetch_start_date,
        end_date,
        ml_pipeline_cfg,
        workers=ml_n_jobs,
        n_jobs=ml_n_jobs,
        is_end_date=is_end_date,
        is_start_date=start_date,
        gp_only=args.alpha_only,
        hmm_only=args.hmm_only,
        preloaded_data_maps=oos_data_maps if not pre_args.skip_universe else None,
        preloaded_1h_maps={s: oos_data_maps[s]["1h"] for s in valid_symbols if s in oos_data_maps and "1h" in oos_data_maps[s]} if not pre_args.skip_universe else None,
    )

    # [TELEMETRY] ML Pipeline Audit
    if ml_out.alpha_panel is not None and hasattr(ml_out.alpha_panel, "attrs"):
        best_fitness = ml_out.alpha_panel.attrs.get("best_fitness", 0.0)
        rep = ml_out.alpha_panel.attrs.get("alpha_component_filter", {})
        if rep:
            n_tried = int(rep.get("n_components", 0))
            n_survived = int(rep.get("n_surviving", 0))
            is_ic = float(rep.get("primary_is_mu", 0.0))
            oos_ic = float(rep.get("primary_oos_mu", 0.0))
            hl = float(rep.get("primary_half_life", 0.0))
            fail_fdr = int(rep.get("fail_fdr", 0))
            fail_dsr = int(rep.get("fail_dsr", 0))
            fail_oos = int(rep.get("fail_oos", 0))
            fail_hl = int(rep.get("fail_half_life", 0))
            fail_sym = int(rep.get("fail_sym_bal", 0))
            fail_reg = int(rep.get("fail_regime", 0))
            ic_ok = "✅" if oos_ic > 0.02 else ("⚠️" if oos_ic > 0.0 else "❌")
            _logger.info(
                "┌─ 🤖 Alpha Performance Report ────────────────────────────────\n"
                "│  Components : %d tried → %d survived  (fitness=%.4f)\n"
                "│  IC         : IS=%.4f  OOS=%.4f %s  half-life=%.1f bars\n"
                "│  Filter     : fdr=%d dsr=%d oos=%d hl=%d sym=%d regime=%d\n"
                "└──────────────────────────────────────────────────────────────",
                n_tried, n_survived, float(best_fitness),
                is_ic, oos_ic, ic_ok, hl,
                fail_fdr, fail_dsr, fail_oos, fail_hl, fail_sym, fail_reg,
            )
            ai_telemetry_payloads.append({
                "stage": "alpha_audit_ml",
                "is_best_fitness": float(best_fitness),
                "n_tried": n_tried,
                "n_survived": n_survived,
                "is_mean_ic": is_ic,
                "oos_mean_ic": oos_ic,
                "ic_half_life": hl,
                "fail_fdr": fail_fdr,
                "fail_dsr": fail_dsr,
                "fail_oos": fail_oos,
                "fail_half_life": fail_hl,
                "fail_sym_bal": fail_sym,
                "fail_regime": fail_reg,
            })
    if hasattr(ml_out, "hmm_report") and ml_out.hmm_report:
        h_rep = ml_out.hmm_report
        ai_telemetry_payloads.append({
            "stage": "hmm_audit",
            "bull_prob": float(h_rep.get("hmm_prob_bull_calm", 0)) + float(h_rep.get("hmm_prob_bull_vol_up", 0)),
            "bear_prob": float(h_rep.get("hmm_prob_bear_trend", 0)),
            "chop_prob": float(h_rep.get("hmm_prob_chop", 0)),
            "crisis_prob": float(h_rep.get("hmm_prob_crisis", 0)),
            "bull_g_log": float(h_rep.get("hmm_bull_g_log", 0)),
            "crisis_g_log": float(h_rep.get("hmm_crisis_g_log", 0)),
            "tail_capture": float(h_rep.get("hmm_tail_capture", 0)),
            "avg_duration": float(h_rep.get("hmm_avg_duration", 0)),
        })


    if args.alpha_only:
        _logger.info(" [ALPHA-ONLY] Analysis complete. Exiting as requested.")
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

    # [ATR Injection] Ensure ATR column is populated in all data maps.
    # backtest_target_weights_numba skips every entry when atr_prev <= 0 → zero trades.
    _atr_period = int(OPT_FUTURES_CONFIG.get("FUTURES_ATR_PERIOD_FIXED", 30))
    _tfs_to_patch = list({args.tf, "1h"})
    for _maps in (data_maps, oos_data_maps):
        for _sym in valid_symbols:
            if _sym not in _maps:
                continue
            for _tf in _tfs_to_patch:
                if _tf not in _maps[_sym]:
                    continue
                _df = _maps[_sym][_tf]
                _needs_atr = (
                    "atr" not in _df.columns
                    or _df["atr"].isna().all()
                    or (_df["atr"].fillna(0) == 0).all()
                )
                if _needs_atr:
                    _atr_arr = compute_atr_numpy(
                        _df["high"].to_numpy(dtype=np.float64),
                        _df["low"].to_numpy(dtype=np.float64),
                        _df["close"].to_numpy(dtype=np.float64),
                        _atr_period,
                    )
                    _df = _df.copy()
                    _df["atr"] = pd.Series(_atr_arr, index=_df.index).ffill().fillna(
                        _df["close"] * 0.01
                    )
                    _maps[_sym][_tf] = _df
                    _logger.info(
                        "[ATR] %s/%s: computed ATR(period=%d) — was missing/zero",
                        _sym, _tf, _atr_period,
                    )

    for sym in valid_symbols:
        df = oos_data_maps[sym][args.tf]
        if "ml_alpha_00" not in df.columns:
            _logger.error("[SIG CHECK] %s: no ml_alpha_00 column.", sym)
            raise RuntimeError(f"OOS merge missing ml_alpha_00 for {sym}.")
            
        o0 = int(oos_data_maps[sym][f"oos_start_idx_{args.tf}"])
        gp = pd.to_numeric(df["ml_alpha_00"], errors="coerce")
        is_std = float(gp.iloc[:o0].std(ddof=0) or 0.0)
        oos_std = float(gp.iloc[o0:].std(ddof=0) or 0.0)
        
        # [v15.2 Diagnostic] Specific logging for problematic symbol or dead signal
        if sym == "1000SHIB/USDT" or oos_std < 1e-4:
            oos_slice = gp.iloc[o0:]
            nz_count = int((oos_slice != 0).sum())
            nz_count_neutral = int((oos_slice != 0.5).sum())
            _logger.info(
                "[SIG CHECK] %s OOS ml_alpha_00 status: STD=%.6f, NonZero=%d, NonNeutral=%d, Mean=%.4f",
                sym, oos_std, nz_count, nz_count_neutral, float(oos_slice.mean())
            )
            
            if oos_std < 1e-4:
                if args.ops_profile == "smoke":
                    _logger.warning("[SIG CHECK] %s OOS ml_alpha_00 std < 1e-4 but continuing due to smoke profile.", sym)
                else:
                    _logger.error("[ABORT] %s OOS ml_alpha_00 std < 1e-4. Check merge/tz or symbol discovery.", sym)
                    raise RuntimeError(f"OOS signal dead for {sym}.")
        else:
            _logger.debug("[SIG CHECK] %s IS gp_std=%.6f OOS gp_std=%.6f", sym, is_std, oos_std)

    # [PHASE 5] Optuna Portfolio Optimization Starting
    _prof = resolve_ops_profile(args.ops_profile)
    if _prof is not None:
        n_ml_trials = int(_prof["trials"])
        target_seeds = [int(s) for s in (_prof.get("seeds") or [42])]
        _logger.info(
            " [OPS] profile=%s trials=%d seeds=%s — %s",
            _prof.get("id"),
            n_ml_trials,
            target_seeds,
            _prof.get("description", ""),
        )
    else:
        n_ml_trials = int(args.trials)
        learning = OPT_FUTURES_CONFIG.get("FUTURES_LEARNING_SEEDS", [42])
        target_seeds = [int(s) for s in learning] if args.seed is None else [int(args.seed)]

    _logger.info("\n" + "═" * 85)
    _logger.info(" [STEP 4/5] OPTIMIZATION (Joint NSGA-II)")
    _logger.info("═" * 85 + "\n")

    seed_learn = int(OPT_FUTURES_CONFIG.get("FUTURES_LEARNING_SEEDS", [42])[0])
    base_ctx = MLPhaseDContext(
        data_maps=data_maps,
        symbols=valid_symbols,
        tf=args.tf,
        seed=seed_learn,
        effective_total_trials=n_ml_trials,
        ml_pipeline_fetch_start=fetch_start_date,
        ml_pipeline_end=end_date,
        ml_pipeline_is_start=start_date,
        ml_pipeline_workers=ml_n_jobs,
    )
    ml_ctx = base_ctx


    # [Speed Optimization] Numba Warm-up
    _logger.info("  --> [WARM-UP] JIT-compiling Numba kernels...")
    # Force enable NSGA2 for Joint optimization
    OPT_FUTURES_CONFIG["FUTURES_ML_ALPHA_NSGA2_ENABLED"] = True
    precompute_ml_optimization_context(base_ctx)
    if base_ctx.awf_leg_slices:
        test_leg = base_ctx.awf_leg_slices[0]["data"]
        zkill, zfund, lev_leg = _cached_kill_fund_lev(test_leg, _base_engine_params({}, args.tf))
        _run_portfolio_numba_block(_base_engine_params({}, args.tf), test_leg, zkill, zfund, lev_leg)

    # [Speed Optimization] High-Performance Optuna Storage (SQLite WAL)
    storage_path = Path(project_root) / "logs" / "optuna_futures.db"
    if storage_path.exists():
        storage_path.unlink()
    
    storage_url = f"sqlite:///{storage_path}"
    
    # 1. SQLAlchemy Engine with WAL mode & Connection Pooling
    # check_same_thread=False is required for multi-threaded access.
    engine = create_engine(
        storage_url,
        connect_args={"check_same_thread": False, "timeout": 60},
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20
    )
    
    with engine.begin() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        conn.execute(text("PRAGMA synchronous=NORMAL;"))
        conn.execute(text("PRAGMA cache_size=-64000;"))  # 64MB Cache

    storage = optuna.storages.RDBStorage(storage_url, engine_kwargs={"connect_args": {"timeout": 60, "check_same_thread": False}})
    study_name = "futures_joint_nsga2"

    # [Memory Optimization] gc.freeze() to preserve CoW before parallel branching
    # This prevents memory explosion in WSL by stopping refcount updates on data_maps.
    _logger.info("  --> [MEMORY] Freezing GC to preserve CoW...")
    gc.collect()
    gc.freeze()

    study_ml = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["minimize", "minimize"],  # Obj1: -Mean Log TW, Obj2: -Min Log TW
        sampler=optuna.samplers.NSGAIISampler(
            population_size=int(OPT_FUTURES_CONFIG.get("FUTURES_NSGA2_POPULATION_SIZE", 30)),
            seed=seed_learn,
        ),
        load_if_exists=True,
    )

    _logger.info("  --> [EXEC] Running Joint NSGA-II optimization (6 Parallel Processes)...")
    
    # Distribute trials across 6 processes
    n_workers = 6
    trials_per_worker = n_ml_trials // n_workers
    remainder = n_ml_trials % n_workers
    
    worker_tasks = []
    for i in range(n_workers):
        t_count = trials_per_worker + (1 if i < remainder else 0)
        if t_count > 0:
            worker_tasks.append(delayed(optimize_worker)(study_name, storage_url, t_count, base_ctx))

    Parallel(n_jobs=n_workers, backend="multiprocessing")(worker_tasks)

    all_trials = study_ml.get_trials()
    best_trials = study_ml.best_trials

    if not best_trials:
        _logger.error("  [FAIL] No trials completed successfully. Stopping optimization.")
        return

    # [TOPSIS] Selection from Pareto front
    best_trial_coord = topsis_select_best(best_trials)
    champion_raw_params = dict(best_trial_coord.params)
    _logger.info("  [SUCCESS] Optimization complete. Best Trial: %d", best_trial_coord.number)

    # [STABILITY] Runner-up logic for Joint optimization
    runner_up_merged = None
    if len(best_trials) >= 2:
        # Simple runner-up: next closest to ideal point (this is a placeholder for more complex logic)
        runner_up_merged = dict(best_trials[1].params)


    champ_stab_cv: float | None = None
    stab_tmp_layer3_awf_fail = False
    stab_seeds = [
        int(x) for x in (OPT_FUTURES_CONFIG.get("FUTURES_STABILITY_SEEDS") or [])
    ]
    cv_max = float(OPT_FUTURES_CONFIG.get("FUTURES_CHAMP_STABILITY_CV_MAX", 0.30))
    stab_hard = bool(OPT_FUTURES_CONFIG.get("FUTURES_CHAMP_STABILITY_HARD_GATE", False))
    l3_hard = bool(OPT_FUTURES_CONFIG.get("FUTURES_TMP_LAYER3_HARD_GATE", False))

    def _run_champ_stability(raw_prms: dict[str, Any]) -> tuple[float | None, bool]:
        if len(stab_seeds) < 1 or ml_ctx is None or not raw_prms:
            return None, False
        _l3_fail = False
        _objs: list[float] = []
        for _sx in stab_seeds:
            _logger.info("  --> [STABILITY] Replaying seed=%s", _sx)
            _sctx = dataclasses.replace(ml_ctx, seed=int(_sx))
            # [Performance] Force fresh precompute and clear memory for each seed
            rerun_precompute_for_ctx(_sctx)
            gc.collect()
            
            val, diag_r = replay_robust_awf_for_trial_params(_sctx, raw_prms)
            if isinstance(val, tuple):
                _objs.append(float(val[0]))
            else:
                _objs.append(float(val))
            if bool(OPT_FUTURES_CONFIG.get("FUTURES_TMP_LAYER3_ALL_SEEDS_LAYER1", True)):
                lf = tmp_md_layer1_failures_from_awf_diag(diag_r, OPT_FUTURES_CONFIG)
                if lf:
                    _l3_fail = True
                    _logger.warning(
                        " [TMP LAYER3] AWF replay seed=%s Layer-1 fail codes=%s",
                        _sx,
                        lf,
                    )
            # Clear leg caches to prevent OOM during stability run
            _sctx.awf_leg_slices = None
            gc.collect()

        _mu = float(np.mean(_objs))
        _sig = float(np.std(_objs, ddof=1)) if len(_objs) > 1 else 0.0
        _cv = float(_sig / max(abs(_mu), 1e-12))
        _logger.info(
            " [CHAMP STABILITY] seeds=%s obj_mean=%.6f std=%.6f cv=%.4f (max=%.4f)",
            stab_seeds,
            _mu,
            _sig,
            _cv,
            cv_max,
        )
        if _cv > cv_max:
            _logger.warning(
                " [CHAMP STABILITY] cv %.4f exceeds max %.4f", _cv, cv_max
            )
        if _l3_fail:
            _logger.warning(
                " [TMP LAYER3] One or more stability seeds failed Layer-1 on AWF replay.",
            )
        return _cv, _l3_fail

    champ_stab_cv, stab_tmp_layer3_awf_fail = _run_champ_stability(champion_raw_params)

    primary_failed = (
        (champ_stab_cv is not None and champ_stab_cv > cv_max and stab_hard)
        or (stab_tmp_layer3_awf_fail and l3_hard)
    )
    if primary_failed and runner_up_merged is not None:
        _logger.warning(
            " [STABILITY] primary candidate failed hard gates — trying runner-up (phase C)",
        )
        cv_ru, l3_ru = _run_champ_stability(runner_up_merged)
        ru_ok = (cv_ru is None or cv_ru <= cv_max or not stab_hard) and (
            not l3_ru or not l3_hard
        )
        if ru_ok:
            champion_raw_params = runner_up_merged
            champ_stab_cv = cv_ru
            stab_tmp_layer3_awf_fail = l3_ru
            best_trial_coord = c_complete_ranked[1]
            _logger.info(" [STABILITY] runner-up accepted for deploy path.")

    champ_params = build_ml_phase_d_params(champion_raw_params, args.tf)
    champ_params["ESTIMATED_B"] = float(base_ctx.estimated_b)
    ensemble_results = [
        CandidateSelectionResult(
            params=champ_params,
            representative_trial=best_trial_coord,
            n_completed=len(
                [t for t in study_ml.trials if t.state == TrialState.COMPLETE]
            ),
            n_passing=0,
            n_basin=1,
            method="joint_nsga2",
        )
    ]
    _logger.info(
        "  [SELECTOR] method=joint_nsga2 members=1 trials_total=%d",
        len(all_trials),
    )

    n_trials_for_gates = int(n_ml_trials)
    pbo_gate, dsr_gate, pbo_champ_eff = resolve_adjusted_gates(
        OPT_FUTURES_CONFIG, n_trials_for_gates
    )

    def _is_passing_trial(t: optuna.trial.FrozenTrial) -> bool:
        _awf_pos = float(t.user_attrs.get("awf_pos_frac", 0.0))
        pbo_obs_t = awf_pos_frac_to_pseudo_pbo(_awf_pos)
        return _ml_trial_passes_hard_gates(
            t,
            pbo_obs_t,
            check_pbo=True,
            pbo_max=pbo_gate,
            dsr_min=dsr_gate,
        )

    if not ensemble_results:
        _logger.error(" [FAIL] No champion candidate after optimization.")
        return

    _logger.info("\n" + "═" * 85)
    _logger.info(
        " [CHAMPION] %s candidate(s) ready",
        len(ensemble_results),
    )
    winner_res = ensemble_results[0]
    params = winner_res.params
    best_trial = winner_res.representative_trial
    _seed_w = args.seed if args.seed is not None else 0
    winner = {"seed": _seed_w, "params": params, "trial": best_trial}
    passing_trials = [
        t for t in all_trials if t.state == TrialState.COMPLETE and _is_passing_trial(t)
    ]

    # Proceed to Final Evaluation and Persistence
    gate_ok = True
    gate_failures: list[str] = []

    # [PLGD Breakdown + AWF Leg Matrix] — AI diagnostic + user sanity check
    _leg_tws = (
        best_trial.user_attrs.get("awf_path_leg_log_tw")
        or best_trial.user_attrs.get("awf_leg_log_tw")
        or best_trial.user_attrs.get("cpcv_path_oos_log_tw")
        or []
    )
    _mu_awf  = float(best_trial.user_attrs.get("awf_mu_log", 0.0))
    _sig_awf = float(best_trial.user_attrs.get("awf_sigma_log", 0.0))
    _plgd_v = float(best_trial.user_attrs.get("awf_plgd", 0.0))
    _reward_v = float(best_trial.user_attrs.get("awf_contract_reward", _plgd_v))
    _n_tr_cfg = float(
        best_trial.user_attrs.get("awf_plgd_n_trials", OPT_FUTURES_CONFIG.get("total_trials", 1000))
    )
    # Legacy PLGD diagnostic breakdown (post-opt log only; not configurable).
    _ldef = 0.5
    _ltail = 2.0
    _sr_b = math.sqrt(2.0 * math.log(max(_n_tr_cfg, 2.0)))
    _vd   = 0.5 * _sig_awf ** 2
    _def  = _ldef * _sr_b * _sig_awf / math.sqrt(max(float(len(_leg_tws)), 1.0))
    _wl   = min(_leg_tws) if _leg_tws else 0.0
    _tp   = _ltail * max(0.0, -_wl)
    _logger.info(
        "  [AWF OBJ] mu=%.4f  var_drag=%.4f  deflation=%.4f  tail=%.4f  plgd_legacy=%.4f  "
        "contract_reward=%.4f",
        _mu_awf, _vd, _def, _tp, _plgd_v, _reward_v,
    )
    if _leg_tws:
        _leg_l_pf = best_trial.user_attrs.get("leg_l_pf", [])
        _leg_s_pf = best_trial.user_attrs.get("leg_s_pf", [])
        _leg_l_cnt = best_trial.user_attrs.get("leg_long_counts", [])
        _leg_s_cnt = best_trial.user_attrs.get("leg_short_counts", [])
        _leg_crisis = best_trial.user_attrs.get("leg_crisis_mean", [])
        _logger.info(
            "  [AWF LEGS]  leg   log_tw     TW%     Long_PF  Short_PF  L_n  S_n  CrisisP"
        )
        for _li, _ltw in enumerate(_leg_tws):
            _tw_pct = (math.exp(float(_ltw)) - 1.0) * 100.0
            _flag = "✓" if _ltw > 0.0 else "✗"
            _lpf_s = f"{_leg_l_pf[_li]:.2f}" if _li < len(_leg_l_pf) else "  —"
            _spf_s = f"{_leg_s_pf[_li]:.2f}" if _li < len(_leg_s_pf) else "  —"
            _lcnt_s = str(_leg_l_cnt[_li]) if _li < len(_leg_l_cnt) else "—"
            _scnt_s = str(_leg_s_cnt[_li]) if _li < len(_leg_s_cnt) else "—"
            _crisis_s = f"{_leg_crisis[_li]:.3f}" if _li < len(_leg_crisis) else "  —"
            _logger.info(
                "    %d/%d  %+.4f  %+.2f%%  %s  %s  %s  %4s  %4s  %s",
                _li + 1, len(_leg_tws), _ltw, _tw_pct, _flag,
                _lpf_s, _spf_s, _lcnt_s, _scnt_s, _crisis_s,
            )

    # [TELEMETRY] Optimization & AWF Audit
    ai_telemetry_payloads.append({
        "stage": "awf_audit",
        "plgd": float(_plgd_v),
        "mu_awf": float(_mu_awf),
        "sig_awf": float(_sig_awf),
        "leg_tws": [float(t) for t in _leg_tws],
        "awf_pos_frac": float(best_trial.user_attrs.get("awf_pos_frac", 0.0)),
    })

    _assert_oos_gp_signal_alive(oos_data_maps, valid_symbols, args.tf)

    _logger.info("\n" + "═" * 85)
    _logger.info(" [STEP 5/5] FINAL OOS EVALUATION & WF ADAPTATION")
    _logger.info("═" * 85)

    policy_cfg = load_portfolio_policy_config(OPT_FUTURES_CONFIG)

    _logger.info("  [ENSEMBLE EVALUATION]")
    ensemble_curves = []
    ensemble_ports = []
    for i, res in enumerate(ensemble_results):
        m_params = finalize_strategy_portfolio_params(res.params, policy_cfg)
        m_port = run_oos_margin_shared_portfolio(
            valid_symbols, args.tf, m_params, oos_data_maps, cache_root=FUTURES_CACHE_DIR
        )
        ensemble_ports.append(m_port)
        ensemble_curves.append(m_port["equity_curve"])
        m_cagr = m_port.get("cagr_pct", 0.0)
        m_mdd = m_port.get("mdd_pct", 0.0)
        _logger.info(
            f"    Member {i+1}/{len(ensemble_results)}: CAGR={m_cagr:7.2f}% | "
            f"MDD={m_mdd:6.2f}% | Trial={res.representative_trial.number}"
        )

    # Online Capital Allocation (Meta-Strategy)
    meta_window = int(OPT_FUTURES_CONFIG.get("FUTURES_META_ALLOC_WINDOW", 24))
    meta_eta = float(OPT_FUTURES_CONFIG.get("FUTURES_META_ALLOC_ETA", 0.1))
    meta_equity, weight_history = perform_online_capital_allocation(
        ensemble_curves, float(FUTURES_INITIAL_BALANCE), window_size=meta_window, eta=meta_eta
    )
    
    # Log final weights
    final_weights = weight_history[-1]
    _logger.info("  [META-ALLOCATION] EG Update Complete.")
    for i, w_val in enumerate(final_weights):
        _logger.info(f"    Member {i+1} Final Weight: {w_val:.4f}")

    # Calculate Meta-Strategy Metrics
    meta_final_bal = meta_equity[-1]
    meta_moic = meta_final_bal / float(FUTURES_INITIAL_BALANCE)
    meta_mdd = calc_mdd_from_equity(meta_equity)
    
    hours_per_bar = int(args.tf.replace("h", "")) if args.tf.endswith("h") else 4
    n_days = (len(meta_equity) * hours_per_bar) / 24.0
    
    try:
        exponent = 365.0 / max(n_days, 1e-3)
        log_meta_moic = math.log(max(meta_moic, 1e-9))
        meta_cagr = (math.exp(exponent * log_meta_moic) - 1.0) * 100.0
    except (OverflowError, ValueError):
        meta_cagr = 1e8 if meta_moic > 1.0 else -100.0

    meta_port = {
        "cagr_pct": meta_cagr,
        "mdd_pct": meta_mdd,
        "equity_curve": meta_equity,
        "moic": meta_moic,
        "terminal_wealth_ratio": meta_moic,
        "profit_factor": ensemble_ports[0]["profit_factor"], # Best member as proxy
        "total_trades": ensemble_ports[0]["total_trades"],
        "win_rate_pct": ensemble_ports[0]["win_rate_pct"],
        "oos_long_short_minority_pct": ensemble_ports[0]["oos_long_short_minority_pct"],
        "calmar_ratio": meta_cagr / abs(meta_mdd) if abs(meta_mdd) > 1e-6 else 0.0,
        "ulcer_index": calc_ulcer_index_from_equity(meta_equity),
        "cvar_pct": calc_cvar5_loss_pct_from_equity(meta_equity),
        "long_pf": ensemble_ports[0]["long_pf"],
        "short_pf": ensemble_ports[0]["short_pf"],
    }

    # [STEP 5.1/5] Detailed Ensemble Audit
    _print_performance_report(
        "ENSEMBLE (META-STRATEGY) OOS PERFORMANCE",
        ensemble_ports[0],
        dsr=float(best_trial.user_attrs.get("gate1_dsr", 0.0)),
        pbo=awf_pos_frac_to_pseudo_pbo(float(best_trial.user_attrs.get("awf_pos_frac", 0.0))),
        tf=args.tf,
        meta_port=meta_port
    )

    # Continue with the 'winner' (best member) for standard evaluation
    params = finalize_strategy_portfolio_params(params, policy_cfg)
    oos_port = ensemble_ports[0]

    _awf_pos_best = float(best_trial.user_attrs.get("awf_pos_frac", 0.0))
    pbo_obs = awf_pos_frac_to_pseudo_pbo(_awf_pos_best)
    dsr_obs = float(best_trial.user_attrs.get("gate1_dsr", 0.0))

    if bool(OPT_FUTURES_CONFIG.get("FUTURES_PHASE3_HARD_GATE", True)):
        _awf_leg_log_tw = (
            best_trial.user_attrs.get("awf_leg_log_tw")
            or best_trial.user_attrs.get("cpcv_path_oos_log_tw")
            or []
        )
        if len(_awf_leg_log_tw) >= 3:
            _spa_p = stationary_bootstrap_spa(np.asarray(_awf_leg_log_tw, dtype=np.float64))
            _spa_max = float(OPT_FUTURES_CONFIG.get("FUTURES_SPA_P_VALUE_MAX", 0.10))
            _logger.info(
                " [SPA] H0(zero alpha) p-value=%.4f (threshold=%.2f) -> %s",
                _spa_p,
                _spa_max,
                "REJECT H0" if _spa_p <= _spa_max else "FAIL-TO-REJECT",
            )
        _logger.info(
            " [PHASE 3 AUDIT] awf_pos_frac=%.4f pseudo_pbo=%.4f | DSR=%.4f",
            _awf_pos_best,
            float(pbo_obs),
            dsr_obs,
        )

    vcfg_block = OPT_FUTURES_CONFIG.get("FUTURES_VALIDATION_CONFIG", {})
    _emb_cfg = int(vcfg_block.get("wf_anchored_embargo_bars", -1))
    _emb = _emb_cfg if _emb_cfg >= 0 else int(EMBARGO_BARS.get(args.tf, 12))
    wf_cfg = WalkForwardConfig(
        n_legs=int(vcfg_block.get("wf_n_legs", 10)),
        purge_bars=int(vcfg_block.get("wf_purge_bars", 24)),
        min_positive_leg_ratio=float(vcfg_block.get("wf_min_positive_leg_ratio", 0.70)),
        worst_leg_tw_floor=float(vcfg_block.get("wf_worst_leg_tw_floor", 0.95)),
        mean_leg_tw_floor=float(vcfg_block.get("wf_mean_leg_tw_floor", 1.00)),
        ergodicity_guideline_pct=float(vcfg_block.get("wf_ergodicity_guideline_pct", 15.0)),
        ergodicity_hard_gate_enabled=bool(vcfg_block.get("wf_ergodicity_hard_gate_enabled", True)),
        use_anchored_awf_geometry=bool(
            vcfg_block.get("use_anchored_awf_geometry", False)
        ),
        anchored_is_pool_frac=float(vcfg_block.get("wf_anchored_is_pool_frac", 0.70)),
        anchored_embargo_bars=_emb,
    )
    _erg_dev_val = 0.0
    wf_result = None
    if valid_symbols and wf_cfg.n_legs > 1:
        wf_result = mirror_walk_forward_result_from_awf_user_attrs(
            dict(best_trial.user_attrs),
            wf_cfg,
        )
        _logger.info(
            " [WF] mode=awf_mirror legs=%d erg_dev=%.2f%% "
            "(single AWF geometry; mirrored from trial AWF attrs)",
            len(wf_result.tw_legs),
            wf_result.ergodicity_dev_pct,
        )
        _erg_dev_val = float(wf_result.ergodicity_dev_pct)
        _logger.info(
            " [WF] legs=%d pos_ratio=%.2f worst_tw=%.4f mean_tw=%.4f erg_dev=%.2f%%",
            len(wf_result.tw_legs),
            wf_result.positive_leg_ratio,
            wf_result.worst_leg_tw,
            wf_result.mean_leg_tw,
            wf_result.ergodicity_dev_pct,
        )
        if wf_result.leg_adaptation_logs:
            _logger.info(" [WF/DRIFT] per-leg feature drift (alpha/calib/crisis vs prev leg)")
            for row in wf_result.leg_adaptation_logs[:20]:
                _logger.info(
                    "   leg=%s tw=%.4f d_alpha=%s",
                    row.get("leg"),
                    row.get("tw"),
                    row.get("ml_alpha_00_delta_vs_prev"),
                )
        if not wf_result.passed:
            _logger.warning(
                " [WF PRECHECK] leg thresholds not met %s",
                ",".join(wf_result.failures),
            )
        ai_telemetry_payloads.append({
            "stage": "wf_ergodicity",
            "erg_dev": float(wf_result.ergodicity_dev_pct),
            "guideline": float(wf_cfg.ergodicity_guideline_pct),
            "tw_legs": [float(t) for t in wf_result.tw_legs],
            "positive_leg_ratio": float(wf_result.positive_leg_ratio),
            "worst_leg_tw": float(wf_result.worst_leg_tw),
            "leg_adaptation_logs": [dict(r) for r in wf_result.leg_adaptation_logs],
        })

    # IS & Hold-out Evaluation
    is_data_maps: dict[str, dict[str, Any]] = {}
    ho_data_maps: dict[str, dict[str, Any]] = {}

    mai = ml_ctx.multi_alignment_info or {}
    alignment_offsets = mai.get("alignment_offsets", {})
    eff_len = mai.get("eff_ref_len", 0)
    ho_ratio = 0.20
    aligned_main_is_bars = max(200, int(eff_len * (1.0 - ho_ratio)))

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
        ho_start = min(aligned_is_start + aligned_main_is_bars, len(ho_dm[args.tf]) - 2)
        ho_dm[f"oos_start_idx_{args.tf}"] = max(aligned_is_start, ho_start)
        ho_data_maps[sym] = ho_dm

    is_port = run_oos_margin_shared_portfolio(
        valid_symbols, args.tf, params, is_data_maps, cache_root=FUTURES_CACHE_DIR
    )
    ho_port = run_oos_margin_shared_portfolio(
        valid_symbols, args.tf, params, ho_data_maps, cache_root=FUTURES_CACHE_DIR
    )

    # [STEP 5.2/5] Final Performance Reports
    # BTC buy-and-hold CAGR over the OOS period as market benchmark.
    # Used for Net Alpha display only — not a gate. IS-boundary safe (no look-ahead).
    _btc_benchmark_oos: float | None = None
    _btc_benchmark_is: float | None = None
    _btc_sym = next((s for s in valid_symbols if "BTC" in s.upper()), None)
    if _btc_sym and _btc_sym in oos_data_maps:
        _hrs_tf = int(args.tf.replace("h", "")) if args.tf.endswith("h") else 4
        _btc_tf_df = oos_data_maps[_btc_sym][args.tf]
        _btc_o0 = int(oos_data_maps[_btc_sym][f"oos_start_idx_{args.tf}"])
        _slices: list[tuple[str, np.ndarray, str]] = [
            ("OOS", _btc_tf_df["close"].iloc[_btc_o0:].to_numpy(dtype=np.float64), "_btc_benchmark_oos"),  # noqa: E501
            ("IS", _btc_tf_df["close"].iloc[:_btc_o0].to_numpy(dtype=np.float64), "_btc_benchmark_is"),  # noqa: E501
        ]
        for _label, _arr_slice, _out_var in _slices:
            if _arr_slice.size > 1 and _arr_slice[0] > 0:
                _years = _arr_slice.size * _hrs_tf / (365.0 * 24.0)
                if _years > 0.01:
                    _cagr = (float(_arr_slice[-1]) / float(_arr_slice[0])) ** (1.0 / _years) - 1.0
                    if _out_var == "_btc_benchmark_oos":
                        _btc_benchmark_oos = _cagr * 100.0
                    else:
                        _btc_benchmark_is = _cagr * 100.0

    is_port["symbol_names"] = valid_symbols
    ho_port["symbol_names"] = valid_symbols
    oos_port["symbol_names"] = valid_symbols

    # OOS performance vs IS performance
    is_cagr_v = float(is_port.get("cagr_pct", is_port.get("cagr", 0.0)))
    oos_cagr_v = float(oos_port.get("cagr_pct", oos_port.get("cagr", 0.0)))
    oos_retention = (oos_cagr_v / is_cagr_v * 100.0) if abs(is_cagr_v) > 1e-6 else 0.0
    
    # Pre-calculate IS Net Alpha for telemetry even if gate fails
    is_net_alpha_v = is_cagr_v - (_btc_benchmark_is if _btc_benchmark_is is not None else 0.0)

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

    _lpf_oos = float(oos_port.get("long_pf", oos_port.get("long_profit_factor", 1.0)))
    _spf_oos = float(oos_port.get("short_pf", oos_port.get("short_profit_factor", 1.0)))
    worst_leg = float(
        best_trial.user_attrs.get(
            "awf_worst_leg_log_tw", best_trial.user_attrs.get("ml_p10_log_growth_cpcv", -10.0)
        )
    )
    worst_tw = float(np.exp(worst_leg))
    p10_floor = float(OPT_FUTURES_CONFIG.get("FUTURES_AWF_P10_LOG_TW_MIN", -0.10))

    wf_fail_t = tuple(wf_result.failures) if wf_result is not None else ()

    _gate_inp = FuturesResearchGateInput(
        phase3_enabled=bool(OPT_FUTURES_CONFIG.get("FUTURES_PHASE3_HARD_GATE", True)),
        pbo_max=float(pbo_gate),
        dsr_min=float(dsr_gate),
        is_precision=0.55,
        oos_port=oos_port,
        pbo_obs=float(pbo_obs),
        dsr_obs=float(dsr_obs),
        wf_failures=wf_fail_t,
        min_is_net_alpha_pct=float(policy_cfg.min_is_net_alpha_pct),
        is_net_alpha_pct=float(is_net_alpha_v),
        min_long_pf=float(policy_cfg.min_long_pf),
        min_short_pf=float(policy_cfg.min_short_pf),
        oos_long_pf=float(_lpf_oos),
        oos_short_pf=float(_spf_oos),
        is_cagr_pct=float(is_cagr_v),
        is_sharpe=float(is_sharpe_v),
        is_survival_min_cagr=float(
            OPT_FUTURES_CONFIG.get("FUTURES_IS_SURVIVAL_MIN_CAGR_PCT", 15.0)
        ),
        is_survival_min_sharpe=float(
            OPT_FUTURES_CONFIG.get("FUTURES_IS_SURVIVAL_MIN_SHARPE", 1.5)
        ),
        worst_leg_log_tw=float(worst_leg),
        awf_p10_log_tw_floor=float(p10_floor),
    )
    gate_ok, _gf_codes = evaluate_research_gates(_gate_inp)
    gate_failures = list(_gf_codes)
    if bool(
        OPT_FUTURES_CONFIG.get("FUTURES_TMP_MD_CHAMPION_GATES_ENABLED", True)
    ):
        tmp_gf = collect_tmp_md_champion_gate_failures(
            dict(best_trial.user_attrs),
            oos_bar_rets=oos_rets,
            ann_factor=float(ann_f),
            cfg=OPT_FUTURES_CONFIG,
        )
        if tmp_gf:
            gate_failures.extend(tmp_gf)
            gate_ok = False
    if (
        bool(OPT_FUTURES_CONFIG.get("FUTURES_TMP_LAYER3_HARD_GATE", False))
        and stab_tmp_layer3_awf_fail
    ):
        gate_failures.append("TMP_LAYER3_STABILITY_LAYER1")
        gate_ok = False
    if (
        champ_stab_cv is not None
        and champ_stab_cv > cv_max
        and bool(OPT_FUTURES_CONFIG.get("FUTURES_CHAMP_STABILITY_HARD_GATE", False))
    ):
        gate_failures.append("CHAMP_STAB_CV")
        gate_ok = False
    for _code in gate_failures:
        _logger.warning(
            " [GATE] %s — %s",
            _code,
            GATE_CODE_DESCRIPTIONS.get(_code, ""),
        )
    if gate_ok:
        _logger.info(
            " [GATES] unified PASS | IS Sharpe=%.2f OOS Sharpe=%.2f | "
            "worst_leg_log_tw=%.4f (tw=%.4f floor=%.4f)",
            is_sharpe_v,
            oos_sharpe_v,
            worst_leg,
            worst_tw,
            p10_floor,
        )
        _pbo_cur = float(pbo_obs)
        _logger.info(
            " [IS SURVIVAL] IS CAGR=%.2f%% IS Sharpe=%.2f OOS Sharpe=%.2f pseudo_pbo=%.4f",
            is_cagr_v,
            is_sharpe_v,
            oos_sharpe_v,
            _pbo_cur,
        )
    else:
        _logger.warning(" [GATES] unified FAIL codes=%s", ",".join(gate_failures))

    pbo_val = float(pbo_obs)


    # [Dual-Audit Dashboard] Integrated Performance & Reliability side-by-side
    logs_dir = Path(project_root) / "logs"
    champ_path = resolve_champion_record_path(logs_dir)
    champ_m: dict[str, Any] = {
        "pbo": 0.5, "p10": 0.0, "dsr": 0.0, "tw": 1.0, "cagr": 0.0, "mdd": 0.0,
        "time_2x": 999.0, "cvar": 0.0, "net_alpha": 0.0, "avg_pnl": 0.0, "pf": 1.0
    }
    if champ_path and champ_path.exists():
        try:
            with open(champ_path, encoding="utf-8") as _cf:
                _c = json.load(_cf)
            _met = _c.get("metrics", {})
            champ_m = {
                "pbo": _safe_float(_met.get("pbo_paired", _met.get("pbo", 0.5)), 0.5, 1e3),
                "p10": _safe_float(
                    _met.get(
                        "awf_worst_leg_log_tw",
                        _met.get("cpcv_p10_log_tw", 0.0),
                    ),
                    0.0,
                    100.0,
                ),
                "dsr": _safe_float(_met.get("dsr", 0.0), 0.0, 1e3),
                "tw": _safe_float(_met.get("oos_terminal_wealth", 1.0), 1.0, 1e6),
                "cagr": _safe_float(_met.get("oos_cagr_pct", 0.0), 0.0, 1e5),
                "mdd": _safe_float(_met.get("oos_mdd_pct", 0.0), 0.0, 1e3),
                "time_2x": _safe_float(_met.get("oos_time_to_2x", 999.0), 999.0, 1e6),
                "cvar": _safe_float(_met.get("oos_cvar_pct", 0.0), 0.0, 1e3),
                "net_alpha": _safe_float(_met.get("oos_net_alpha_pct", 0.0), 0.0, 1e5),
                "avg_pnl": _safe_float(_met.get("oos_avg_trade_pnl_pct", 0.0), 0.0, 1e5),
                "pf": _safe_float(_met.get("oos_profit_factor", 1.0), 1.0, 1e3),
            }
        except Exception as _ce:
            _logger.debug("Champion metrics parse failed: %s", _ce)

    # SOTA WEALTH (futures-opt) calculation for Candidate
    eq_arr = np.asarray(meta_port.get("equity_curve", []), dtype=np.float64)
    hrs = int(args.tf.replace("h", "")) if args.tf.endswith("h") else 4
    bpy = (24.0 / hrs) * 365.0
    if eq_arr.size > 1:
        step_log = np.log(np.clip(eq_arr[1:] / eq_arr[:-1], 1e-9, None))
        t2x_n, _ = calc_time_to_target_wealth(step_log, 2.0, bpy)
        nalpha_n = calc_net_alpha_with_friction(eq_arr, 0.0, bpy)
    else:
        t2x_n, nalpha_n = 999.0, 0.0

    new_m = _sanitize_metric_map({
        "pbo": float(pbo_obs) if 'pbo_obs' in locals() else 0.5,
        "p10": float(
            best_trial.user_attrs.get(
                "awf_worst_leg_log_tw", best_trial.user_attrs.get("ml_p10_log_growth_cpcv", 0.0)
            )
        ),
        "dsr": float(best_trial.user_attrs.get("gate1_dsr", 0.0)),
        "tw": float(meta_port.get("terminal_wealth_ratio", 1.0)),
        "cagr": float(meta_port.get("cagr_pct", 0.0)),
        "mdd": float(meta_port.get("mdd_pct", 0.0)),
        "time_2x": float(t2x_n),
        "cvar": float(meta_port.get("cvar_pct", 0.0)),
        "net_alpha": float(nalpha_n * 100.0),
        "avg_pnl": float(oos_port.get("avg_trade_pnl_pct", 0.0)),
        "pf": float(meta_port.get("profit_factor", 1.0)),
        "is_cagr": float(is_port.get("cagr_pct", 0.0)),
        "ho_cagr": float(ho_port.get("cagr_pct", 0.0)),
        "awf_pos_frac": float(best_trial.user_attrs.get("awf_pos_frac", 0.0)),
        "mu_awf": float(best_trial.user_attrs.get("awf_mu_log", 0.0)),
        "sig_awf": float(best_trial.user_attrs.get("awf_sigma_log", 0.0)),
        "plgd": float(
            best_trial.user_attrs.get(
                "awf_contract_reward", best_trial.user_attrs.get("awf_plgd", 0.0)
            )
        ),
        "erg_dev": float(locals().get("_erg_dev_val", 0.0)),
        "oos_long_pf": float(meta_port.get("long_pf", 1.0)),
        "oos_short_pf": float(meta_port.get("short_pf", 1.0)),
        "oos_retention_pct": float(oos_retention),
        "is_alpha": float(is_net_alpha_v) if 'is_net_alpha_v' in locals() else 0.0,
    })

    # Update params to include ensemble info for persistence
    params["ensemble_members"] = [res.params for res in ensemble_results]
    params["meta_allocation"] = {
        "window_size": meta_window,
        "eta": meta_eta,
        "final_weights": final_weights.tolist()
    }
    params["is_ensemble"] = True

    gate_ok_before_champ = gate_ok
    if gate_ok:
        cand_metrics = ChampionMetrics(
            cagr=float(new_m.get("cagr", 0.0)),
            mdd=abs(float(new_m.get("mdd", 100.0))),
            net_alpha=float(new_m.get("net_alpha", 0.0)),
            sharpe=float(oos_sharpe_v),
            pbo=float(new_m.get("pbo", 1.0)),
        )
        allow, reason = run_champion_promotion_guard(
            Path(project_root) / "logs",
            Path(project_root),
            cand_metrics,
            float(pbo_champ_eff),
            bool(args.bypass_champion_guard),
        )
        champ_metrics = load_champion_metrics_for_guard(
            Path(project_root) / "logs",
            Path(project_root),
        )
        if not allow:
            gate_ok = False
            _logger.warning(" [CHAMPION GUARD] HOLD reason=%s", reason)
        else:
            _logger.info(
                " [CHAMPION GUARD] PASS reason=%s | sharpe %.2f->%.2f | "
                "cagr %.2f->%.2f | pbo %.4f->%.4f",
                reason,
                champ_metrics.sharpe,
                cand_metrics.sharpe,
                champ_metrics.cagr,
                cand_metrics.cagr,
                champ_metrics.pbo,
                cand_metrics.pbo,
            )

    # [SMART VERDICT] Distinguish between strategy quality and champion competition
    if not gate_ok:
        if gate_ok_before_champ:
            _verdict = "HOLD (CHAMPION_BLOCKED) 🛡️"
        else:
            _verdict = "HOLD (GATE_FAIL) ⚠️"
    else:
        _verdict = "PROMOTE ✅"

    eval_payload = {
        "stage": "eval_audit",
        "winning_seed": int(winner["seed"]),
        "gate_ok": bool(gate_ok),
        "gate_failures": gate_failures,
        "total_trades": int(oos_port.get("total_trades", 0)),
        "ops_profile": args.ops_profile,
        "n_seeds": len(target_seeds),
        "trials_per_seed": int(n_ml_trials),
    }
    if args.ops_profile:
        _prof_ok, _prof_issues = check_run_summary_against_profile(
            args.ops_profile,
            {
                "gate_ok": bool(gate_ok),
                "n_seeds": len(target_seeds),
                "trials_per_seed": int(n_ml_trials),
                "revalidation_ok": True,
            },
        )
        if not _prof_ok:
            _logger.warning(" [OPS PROFILE] %s issues=%s", args.ops_profile, _prof_issues)
    # Merge all numerical metrics from new_m (which tracks Candidate stats)
    eval_payload.update(new_m)
    # Ensure is_alpha uses the value calculated regardless of gate status
    eval_payload["is_alpha"] = float(is_net_alpha_v) if 'is_net_alpha_v' in locals() else 0.0
    ai_telemetry_payloads.append(eval_payload)

    # [AI TELEMETRY DUMP] Structured JSON Lines for AI parsing
    _logger.info("\n--- 🤖 [AI_TELEMETRY_START] ---")
    for payload in ai_telemetry_payloads:
        _logger.info(json.dumps(payload))
    _logger.info("--- [AI_TELEMETRY_END] ---\n")

    # [HUMAN DASHBOARD] Unified Performance View
    _print_human_dashboard(
        is_port, ho_port, oos_port, _verdict,
        benchmark_is=(_btc_benchmark_is if _btc_benchmark_is is not None else 0.0),
        benchmark_oos=(_btc_benchmark_oos if _btc_benchmark_oos is not None else 0.0),
        meta_port=meta_port
    )

    # [CHAMPION AUDIT] Side-by-side with current champion
    _print_dual_audit_dashboard(new_m, champ_m, _verdict)

    # [PERSISTENCE STRATEGY] 
    # 1. Archive: All gate-passing candidates (research assets) -> results/futures/archive/
    # 2. Production: The absolute champion (trading bot config)
    #    -> results/futures/best_futures_{tf}.json/enc
    
    res_dir = Path(project_root) / "results" / "futures"
    archive_dir = res_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    prod_json = res_dir / f"best_futures_{args.tf}.json"
    prod_enc = res_dir / f"best_futures_{args.tf}.enc"
    
    # Save gate-passing candidates to ARCHIVE (Timestamped for research)
    if gate_ok_before_champ:
        try:
            from src.core.utils.secure_config import encrypt_config, get_strategy_secret
            ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
            arch_json = archive_dir / f"cand_{args.tf}_{ts}.json"
            arch_enc = archive_dir / f"cand_{args.tf}_{ts}.enc"
            
            with open(arch_json, "w") as f_json:
                json.dump(params, f_json, indent=4)
            
            secret = get_strategy_secret() or "default_futures_secret_2026"
            enc_data = encrypt_config(params, secret)
            with open(arch_enc, "wb") as f_enc:
                f_enc.write(enc_data)
                
            _logger.info(f" [ARCHIVE] Candidate preserved to {archive_dir}")
        except Exception as _se:
            _logger.warning(" [ARCHIVE] Failed: %s", _se)

    # Save absolute CHAMPION to PRODUCTION (Fixed name for Bot)
    if gate_ok:
        try:
            from src.core.utils.secure_config import encrypt_config, get_strategy_secret
            
            # Save JSON
            with open(prod_json, "w") as f_prod_json:
                json.dump(params, f_prod_json, indent=4)
                
            # Save Encrypted ENC
            secret = get_strategy_secret() or "default_futures_secret_2026"
            enc_data = encrypt_config(params, secret)
            with open(prod_enc, "wb") as f_prod_enc:
                f_prod_enc.write(enc_data)
                
            _logger.info(f" [PRODUCTION] New Champion deployed to {prod_json} and {prod_enc}")

            # Persist canonical champion record (see config/champion.schema.json).
            champion_data = {
                "schema_version": 1,
                "id": f"cawf-r-{args.tf}-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}",
                "promoted_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "note": f"Promoted via opt_futures.py with {args.trials} trials.",
                "architecture": (
                    "CAWF-R (K=5 AWF legs + contract objective; PLGD legacy diagnostics)"
                ),
                "parameters": params,
                "metrics": _sanitize_metric_map({
                    "pbo_paired": new_m["pbo"],
                    "dsr": new_m["dsr"],
                    "awf_pos_frac": new_m["awf_pos_frac"],
                    "awf_mu_log": new_m["mu_awf"],
                    "awf_sig_log": new_m["sig_awf"],
                    "awf_contract_reward": float(
                        best_trial.user_attrs.get("awf_contract_reward", 0.0)
                    ),
                    "awf_plgd_legacy": float(best_trial.user_attrs.get("awf_plgd", 0.0)),
                    "awf_worst_leg_log_tw": new_m["p10"],
                    "wf_mean_tw": new_m["tw"],
                    "wf_erg_dev": new_m["erg_dev"],
                    "holdout_cagr_pct": new_m["ho_cagr"],
                    "oos_cagr_pct": new_m["cagr"],
                    "oos_mdd_pct": new_m["mdd"],
                    "oos_net_alpha_pct": new_m["net_alpha"],
                    "oos_sharpe_ratio": _safe_float(oos_sharpe_v, 0.0, 1e3),
                    "oos_avg_trade_pnl_pct": new_m["avg_pnl"],
                    "oos_profit_factor": new_m["pf"],
                    "oos_time_to_2x": new_m["time_2x"],
                    "oos_cvar_pct": new_m["cvar"],
                    "is_cagr_pct": new_m["is_cagr"],
                }),
                "gates": {
                    "pbo_strict_guard": "PASS",
                    "awf_pos_frac_gate": "PASS",
                    "awf_worst_leg_hardening": "PASS",
                    "wf_stability_gate": "PASS",
                },
            }
            out_champ = write_champion_record(Path(project_root), champion_data)
            append_champion_history(
                Path(project_root),
                {
                    "id": champion_data.get("id"),
                    "promoted_at": champion_data.get("promoted_at"),
                    "metrics": champion_data.get("metrics", {}),
                    "gate_ok": True,
                    "path": str(out_champ),
                },
            )
            _logger.info(" [CHAMPION] registry updated at %s", out_champ)
        except Exception as _pe:
            _logger.error(" [PRODUCTION] Save failed: %s", _pe)
    else:
        _logger.warning(
            " [PRODUCTION] Absolute champion preserved. "
            "New candidate did not exceed performance benchmarks."
        )


if __name__ == "__main__":
    main()
