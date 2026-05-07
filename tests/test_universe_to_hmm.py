import importlib
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
            # Geometric Growth Approximation: mu - 0.5 * var
            g = mu - 0.5 * (float(r.var()) * 100.0)
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
    
    # 3. Switching Friction & Stability
    _logger.info("╟─────────────────────────────────────────────────────────────────────────────────╢")
    _logger.info("║ [C] REGIME STABILITY & FRICTION {' ':<46} ║")
    
    transitions = int((merged["regime"] != merged["regime"].shift(1)).sum())
    avg_duration = float(len(merged) / max(1, transitions))
    friction_est = transitions * 0.00025 * 100.0 # 2.5bps per switch estimate
    
    _logger.info(f"║ Total Switches: {transitions:<4} | Avg Duration: {avg_duration:>6.1f} bars | Friction Est: {friction_est:>5.2f}% IS ║")
    
    _logger.info("╚" + "═" * 83 + "╝")

def test_universe_gp_hmm_flow(tf="1h"):
    _logger.info("=" * 85)
    _logger.info(f" [TEST] Universe -> GP Alpha -> HMM Regime Flow (Integrated) TF: {tf}")
    _logger.info("=" * 85)
    
    # 1. Window Setup (Realistic Audit Window)
    fetch_start = "2023-10-02"
    start = "2024-01-01"
    is_end = "2025-01-01"
    end = "2025-01-01"
    collector = DataCollector()
    
    _logger.info(f"Window: {fetch_start} ~ {is_end} (Audit Focus: 1 Year IS window)")

    # 2. Clear HMM Cache to force fresh training
    from config.settings import FUTURES_CACHE_DIR
    _logger.info(f"Clearing HMM cache in {FUTURES_CACHE_DIR}...")
    import os
    if FUTURES_CACHE_DIR.exists():
        for f in os.listdir(FUTURES_CACHE_DIR):
            if "HMM" in f and f.endswith(".parquet"):
                os.remove(FUTURES_CACHE_DIR / f)

    # 3. Universe Filtering (Bypassed for Speed Audit)
    # broad_candidates, _ = screen_futures_universe(...)
    
    # 4. Data Loading
    final_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    data_maps_broad, _, valid_broad = _load_futures_data_maps_for_symbols(
        final_symbols, tf, fetch_start, start, is_end, end, skip_metrics=True
    )
    
    _logger.info(f"Final Selected Universe (Direct Load): {valid_broad}")

    # 6. ML Pipeline (HMM Focused)
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
        valid_broad,
        tf,
        fetch_start, 
        end,
        cfg,
        workers=4,
        n_jobs=4,
        is_end_date=is_end,
        is_start_date=start,
        gp_only=False,
        hmm_only=False
    )
    
    elapsed_time = time.time() - start_time
    _logger.info(f"\n[TIME] ML Pipeline (GP + JAX HMM) took: {elapsed_time:.2f} seconds")
    
    # 7. Verification & Deep Audit
    audit_hmm_logic_changes(ml_out, valid_broad, data_maps_broad, tf)
        
    _logger.info("=" * 85)
    _logger.info(" [RESULT] Integrated Universe -> HMM test completed.")
    _logger.info("=" * 85)

if __name__ == "__main__":
    test_universe_gp_hmm_flow("1h")
    # test_universe_gp_hmm_flow("4h") # Disable 4h for faster execution
