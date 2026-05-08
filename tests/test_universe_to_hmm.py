import importlib
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

# Project Root Setup
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config.opt_config
from config.opt_config import (
    FUTURES_ANCHOR_SYMBOLS,
    FUTURES_SCREENER_CONFIG,
    OPT_FUTURES_CONFIG,
    get_quarterly_window,
)
from config.settings import FUTURES_DATA_DIR
from src.domain.futures.data_collector import DataCollector
from src.domain.futures.ml_pipeline.ml_pipeline_runner import run_ml_pipeline_for_universe
from src.execution.opt_main_futures import _load_futures_data_maps_for_symbols
from src.domain.futures.opt_futures_utils.universe_screener_futures import (
    screen_futures_universe,
    screen_symbol_refinement_futures,
)

warnings.filterwarnings("ignore")

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("test_universe_to_hmm")

def audit_hmm_logic_changes(ml_out, symbols, data_maps, tf):
    """Deep audit of HMM regime classification with Log-Wealth & Tail Capture metrics."""
    _logger.info("\n" + "╔" + "═" * 83 + "╗")
    _logger.info(f"║ [INSTITUTIONAL HMM AUDIT] Log-Wealth & Ergodicity Analysis (v15) {' ':<14} ║")
    _logger.info("╠" + "═" * 83 + "╣")

    # sym = symbols[0] if symbols else None
    sym = None
    for s in symbols:
        if s in ml_out.meta_feature_frame_by_symbol and s in data_maps:
            sym = s
            break

    if not sym:
        _logger.error(f"║ No common data for HMM audit among {symbols}. {' ':<30} ║")
        _logger.info("╚" + "═" * 83 + "╝")
        return

    mff = ml_out.meta_feature_frame_by_symbol[sym]
    df = data_maps[sym][tf]
    
    # Ensure returns are available
    if "datetime" not in mff.columns:
        mff = mff.reset_index()
    
    mff["datetime"] = pd.to_datetime(mff["datetime"], utc=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    
    merged = pd.merge(mff, df[["datetime", "close"]], on="datetime", how="left")
    merged["ret"] = merged["close"].pct_change().fillna(0.0)

    semantic_cols = [c for c in ["hmm_prob_bull_trend", "hmm_prob_bear_trend", "hmm_prob_chop", "hmm_prob_crisis"] if c in merged.columns]
    if not semantic_cols:
        _logger.error("║ Missing HMM probability columns. {' ':<46} ║")
        return

    merged["regime"] = merged[semantic_cols].idxmax(axis=1)
    
    # v15 Label Mapping (Hierarchical Decoupling)
    label_map = {
        "BULL_TREND": "CALM",
        "BEAR_TREND": "HIGH_VOL",
        "CHOP": "BLEEDING",
        "CRISIS": "CRISIS"
    }

    # 1. Log-Wealth Dispersion (g = mu - 0.5 * sigma^2)
    _logger.info("║ [A] LOG-WEALTH DISPERSION (Regime Purity) {' ':<42} ║")
    _logger.info("╟──────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────────╢")
    _logger.info("║ REGIME       │ TIME %   │ MU (%)   │ SIG (%)  │ G_log(%) │ BEHAVIOR             ║")
    _logger.info("╟──────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────────╢")
    
    for col in semantic_cols:
        raw_name = col.replace("hmm_prob_", "").upper()
        reg_name = label_map.get(raw_name, raw_name)
        mask = merged["regime"] == col
        if mask.any():
            r = merged.loc[mask, "ret"]
            mu = float(r.mean() * 100.0)
            sig = float(r.std() * 100.0)
            # Geometric Growth Approximation in % units: mu - 0.5 * (sig^2 / 100)
            g = mu - 0.5 * (sig**2 / 100.0)
            time_pct = float(mask.mean() * 100.0)
            
            behavior = "UNSTABLE"
            if g > 0.05: behavior = "WEALTH_EXP"
            elif g < -0.10: behavior = "TAIL_DEFENSE"
            elif abs(g) < 0.05: behavior = "NOISE_LOCKED"
            
            _logger.info(f"║ {reg_name:<12} │ {time_pct:>8.1f}% │ {mu:>8.3f} │ {sig:>8.3f} │ {g:>8.3f} │ {behavior:<20} ║")
        else:
            _logger.info(f"║ {reg_name:<12} │ {0.0:>8.1f}% │ {'-':>8} │ {'-':>8} │ {'-':>8} │ {'-':<20} ║")
    
    # 2. Left-Tail Capture Ratio (Worst 5% Isolation)
    _logger.info("╟──────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────────╢")
    _logger.info("║ [B] LEFT-TAIL CAPTURE (Tail-Risk Isolation) {' ':<39} ║")
    
    q05_thr = float(merged["ret"].quantile(0.05))
    worst_bars = merged[merged["ret"] <= q05_thr]
    if not worst_bars.empty:
        tail_dist = worst_bars["regime"].value_counts(normalize=True) * 100
        # In v15 mapping, CRISIS and HIGH_VOL are risk states
        crisis_capture = float(tail_dist.get("hmm_prob_crisis", 0.0) + tail_dist.get("hmm_prob_bear_trend", 0.0))
        
        status = "FAIL" if crisis_capture < 60 else "PASS" if crisis_capture > 80 else "ACCEPTABLE"
        _logger.info(f"║ Worst 5% Events: {len(worst_bars):<4} | CRISIS/HIGH_VOL Capture: {crisis_capture:>5.1f}% | Verdict: {status:<10} ║")
    
    # 3. Switching Friction & Stability (uses sticky Viterbi hard state to avoid posterior noise)
    _logger.info("╟─────────────────────────────────────────────────────────────────────────────────╢")
    _logger.info("║ [C] REGIME STABILITY & FRICTION {' ':<46} ║")

    _HARD_STATE_NAMES = ["hmm_prob_bull_trend", "hmm_prob_bear_trend", "hmm_prob_chop", "hmm_prob_crisis"]
    if "hmm_hard_state" in merged.columns:
        hard_regime = merged["hmm_hard_state"].astype(int).map(
            {i: _HARD_STATE_NAMES[i] for i in range(4)}
        ).fillna(merged["regime"])
    else:
        hard_regime = merged["regime"]

    transitions = int((hard_regime != hard_regime.shift(1)).sum())
    avg_duration = float(len(merged) / max(1, transitions))
    friction_est = transitions * 0.00025 * 100.0  # 2.5bps per switch estimate

    _logger.info(f"║ Total Switches: {transitions:<4} | Avg Duration: {avg_duration:>6.1f} bars | Friction Est: {friction_est:>5.2f}% IS ║")
    
    _logger.info("╚" + "═" * 83 + "╝")

def audit_oos_and_ic(ml_out, symbols, is_data_maps, oos_data_maps, tf):
    """OOS Holdout Audit + IC vs Forward Returns."""
    _logger.info("\n" + "╔" + "═" * 83 + "╗")
    _logger.info(f"║ [OOS + IC AUDIT] Predictive Power & Generalization Check              {' ':<10} ║")
    _logger.info("╠" + "═" * 83 + "╣")

    sym = None
    for s in symbols:
        if s in ml_out.meta_feature_frame_by_symbol and s in is_data_maps and s in oos_data_maps:
            sym = s
            break
    if not sym:
        _logger.info("║ No symbol with full IS+OOS data for IC audit.                              ║")
        _logger.info("╚" + "═" * 83 + "╝")
        return

    mff = ml_out.meta_feature_frame_by_symbol[sym]
    if "datetime" not in mff.columns:
        mff = mff.reset_index()
    mff["datetime"] = pd.to_datetime(mff["datetime"], utc=True)

    semantic_cols = [c for c in ["hmm_prob_bull_trend", "hmm_prob_bear_trend", "hmm_prob_chop", "hmm_prob_crisis"] if c in mff.columns]
    if not semantic_cols:
        _logger.info("║ Missing HMM columns for IC audit.                                          ║")
        _logger.info("╚" + "═" * 83 + "╝")
        return

    # IS data for IC computation
    is_df = is_data_maps[sym][tf].copy()
    is_df["datetime"] = pd.to_datetime(is_df["datetime"], utc=True)
    merged_is = pd.merge(mff, is_df[["datetime", "close"]], on="datetime", how="inner")
    merged_is = merged_is.sort_values("datetime").reset_index(drop=True)
    merged_is["ret"] = merged_is["close"].pct_change().fillna(0.0)

    # IC: Spearman correlation between hmm_prob_crisis and forward returns
    _logger.info("║ [A] INFORMATION COEFFICIENT (IC) vs Forward Returns                         ║")
    _logger.info("╟──────────────┬──────────┬──────────┬──────────┬──────────────────────────────╢")
    _logger.info("║ Predictor    │ IC(t+1)  │ IC(t+12) │ IC(t+24) │ Interpretation               ║")
    _logger.info("╟──────────────┼──────────┼──────────┼──────────┼──────────────────────────────╢")

    from scipy.stats import spearmanr
    for col in ["hmm_prob_crisis", "hmm_prob_bull_trend"]:
        if col not in merged_is.columns:
            continue
        tag = "CRISIS_P" if "crisis" in col else "BULL_P"
        sign = -1 if "crisis" in col else 1  # crisis should predict negative return
        ics = []
        for horizon in [1, 12, 24]:
            fwd = merged_is["ret"].shift(-horizon)
            valid = merged_is[col].notna() & fwd.notna()
            if valid.sum() > 100:
                ic, _ = spearmanr(merged_is.loc[valid, col], fwd[valid])
                ics.append(float(ic * sign))  # sign-adjusted: expect positive IC
            else:
                ics.append(float("nan"))
        interp = "PREDICTIVE" if len([x for x in ics if not np.isnan(x) and x > 0.02]) >= 2 else "WEAK"
        _logger.info(f"║ {tag:<12} │ {ics[0]:>8.4f} │ {ics[1]:>8.4f} │ {ics[2]:>8.4f} │ {interp:<28} ║")

    # OOS Tail Capture
    oos_df = oos_data_maps[sym][tf].copy()
    oos_df["datetime"] = pd.to_datetime(oos_df["datetime"], utc=True)
    merged_oos = pd.merge(mff, oos_df[["datetime", "close"]], on="datetime", how="inner")

    if len(merged_oos) > 100:
        merged_oos = merged_oos.sort_values("datetime").reset_index(drop=True)
        merged_oos["ret"] = merged_oos["close"].pct_change().fillna(0.0)
        merged_oos["regime"] = merged_oos[semantic_cols].idxmax(axis=1)
        q05 = float(merged_oos["ret"].quantile(0.05))
        worst_oos = merged_oos[merged_oos["ret"] <= q05]
        if not worst_oos.empty:
            tail_dist = worst_oos["regime"].value_counts(normalize=True) * 100
            capture = float(tail_dist.get("hmm_prob_crisis", 0.0) + tail_dist.get("hmm_prob_bear_trend", 0.0))
            verdict = "PASS" if capture > 70 else "ACCEPTABLE" if capture > 50 else "FAIL"
            _logger.info(f"╟─────────────────────────────────────────────────────────────────────────────╢")
            _logger.info(f"║ [B] OOS Tail Capture ({len(oos_df)} bars): CRISIS+BEAR = {capture:5.1f}% | Verdict: {verdict:<10}       ║")
        else:
            _logger.info("║ [B] OOS: No tail events found.                                              ║")
    else:
        _logger.info("║ [B] OOS: Insufficient data overlap for holdout audit.                       ║")

    _logger.info("╚" + "═" * 83 + "╝")


def test_universe_gp_hmm_flow(tf="1h"):
    _logger.info("=" * 85)
    _logger.info(f" [TEST] Universe -> GP Alpha -> HMM Regime Flow (Integrated) TF: {tf}")
    _logger.info("=" * 85)
    
    # 1. Window Setup (Aligned with opt_main_futures.py)
    fetch_start_date, start_date, is_end_date, end_date = get_quarterly_window(None)
    collector = DataCollector()
    
    _logger.info(f"Window: {fetch_start_date} ~ {end_date} (IS End: {is_end_date})")

    # 2. Universe Discovery & Filtering (Aligned with opt_main_futures.py)
    _logger.info("\n[STEP] Universe Discovery & Filtering...")
    broad_candidates, _ = screen_futures_universe(
        collector,
        [],
        tf,
        FUTURES_SCREENER_CONFIG,
        fetch_start_date,
        is_end_date,
        data_dir=FUTURES_DATA_DIR,
    )
    
    if not broad_candidates:
        _logger.error("No broad candidates found. Aborting.")
        return

    data_maps_broad, _, valid_broad = _load_futures_data_maps_for_symbols(
        broad_candidates,
        tf,
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
        tf=tf,
        symbol_dfs_4h={s: data_maps_broad[s][tf] for s in valid_broad},
        daily_dfs={s: data_maps_broad[s]["1d"] for s in valid_broad},
        phase_b_params=None,
        anchor_symbols=FUTURES_ANCHOR_SYMBOLS,
    )
    if not success:
        _logger.error("Universe refinement failed.")
        return
    
    importlib.reload(config.opt_config)
    final_symbols = config.opt_config.FUTURES_SYMBOLS[:3] # Limit to top 3 for speed
    _logger.info(f"Final Symbols (Top 3 for Audit): {final_symbols}")

    # 3. Clear HMM Cache to force fresh training
    from config.settings import FUTURES_CACHE_DIR
    _logger.info(f"Clearing HMM cache in {FUTURES_CACHE_DIR}...")
    import os
    if FUTURES_CACHE_DIR.exists():
        for f in os.listdir(FUTURES_CACHE_DIR):
            if "HMM" in f and f.endswith(".parquet"):
                try:
                    os.remove(FUTURES_CACHE_DIR / f)
                except Exception:
                    pass

    # 4. Data Loading for ML (Only for selected symbols)
    data_maps, oos_data_maps, valid_ml_symbols = _load_futures_data_maps_for_symbols(
        final_symbols, tf, fetch_start_date, start_date, is_end_date, end_date
    )
    
    _logger.info(f"ML Ready Universe: {valid_ml_symbols}")

    # 5. ML Pipeline (HMM Focused)
    cfg = dict(OPT_FUTURES_CONFIG)
    cfg["FUTURES_ML_ALPHA_USE_TBM_WEIGHT"] = False
    # Minimal GP for speed
    cfg["FUTURES_ML_GP_GENERATIONS"] = 1
    cfg["FUTURES_ML_GP_POPULATION"] = 50
    # Balanced HMM settings for CPU execution over long window
    cfg["FUTURES_HMM_N_ITER"] = 100
    cfg["FUTURES_HMM_FIT_STEP"] = 1000
    
    import time
    start_time = time.time()
    
    _logger.info("\nExecuting ML Pipeline (Integrated HMM Audit Mode)...")
    ml_out = run_ml_pipeline_for_universe(
        valid_ml_symbols,
        tf,
        fetch_start_date, 
        end_date,
        cfg,
        workers=4,
        n_jobs=4,
        is_end_date=is_end_date,
        is_start_date=start_date,
        gp_only=False,
        hmm_only=False
    )
    
    elapsed_time = time.time() - start_time
    _logger.info(f"\n[TIME] ML Pipeline (GP + JAX HMM) took: {elapsed_time:.2f} seconds")
    
    # 6. Verification & Deep Audit
    # We use data_maps (IS only) for audit consistency
    audit_hmm_logic_changes(ml_out, valid_ml_symbols, data_maps, tf)

    # 7. OOS + IC Audit
    audit_oos_and_ic(ml_out, valid_ml_symbols, data_maps, oos_data_maps, tf)

    _logger.info("=" * 85)
    _logger.info(" [RESULT] Integrated Universe -> HMM test completed.")
    _logger.info("=" * 85)

if __name__ == "__main__":
    test_universe_gp_hmm_flow("1h")
