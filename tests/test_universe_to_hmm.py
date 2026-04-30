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

def audit_hmm_logic_changes(ml_out, symbols):
    """Deep audit of HMM regime classification after architectural changes."""
    _logger.info("\n" + "=" * 85)
    _logger.info(" [HMM DEEP AUDIT] Evaluating New Regime Logic (v6)")
    _logger.info("=" * 85)

    # 1. Systemic HMM Consistency Check
    sym = symbols[0] if symbols else None
    if not sym or sym not in ml_out.meta_feature_frame_by_symbol:
        _logger.error("No data for HMM audit.")
        return

    mff = ml_out.meta_feature_frame_by_symbol[sym]
    
    # Check for new semantic columns
    semantic_cols = ["hmm_prob_bull_trend", "hmm_prob_bear_trend", "hmm_prob_chop", "hmm_prob_crisis"]
    missing = [c for c in semantic_cols if c not in mff.columns]
    if missing:
        _logger.warning(f" Missing semantic columns: {missing}")
    else:
        _logger.info(" [OK] All v6 semantic columns found.")

    # 2. Modulator Audit
    mod_cols = ["hmm_modulator_long", "hmm_modulator_short", "btc_trend_vol_adj_24h"]
    for c in mod_cols:
        if c in mff.columns:
            mean_val = mff[c].mean()
            std_val = mff[c].std()
            _logger.info(f" [MODULATOR] {c:<25}: Mean={mean_val:.4f}, Std={std_val:.4f}")
        else:
            _logger.warning(f" [MISSING] {c} not found in meta feature frame.")

    # 3. Regime Distribution Audit
    if not missing:
        _logger.info("\n [REGIME DISTRIBUTION]")
        regimes = mff[semantic_cols].idxmax(axis=1)
        dist = regimes.value_counts(normalize=True) * 100
        for regime, pct in dist.items():
            st_name = regime.replace("hmm_prob_", "").upper()
            _logger.info(f"   - {st_name:<15}: {pct:>6.2f}%")

    # 4. Asymmetric Response Audit
    if "hmm_prob_crisis" in mff.columns and "hmm_modulator_long" in mff.columns:
        crisis_period = mff[mff["hmm_prob_crisis"] > 0.7]
        if not crisis_period.empty:
            avg_mod_l = crisis_period["hmm_modulator_long"].mean()
            _logger.info(f"\n [STRESS TEST] Crisis Regime (p>0.7) detected in {len(crisis_period)} bars.")
            _logger.info(f"   - Avg Long Modulator during Crisis: {avg_mod_l:.4f} (Expect < 0.5)")
        else:
            _logger.info("\n [STRESS TEST] No extreme Crisis regime (p>0.7) in this window.")

    # 5. Bull Boost Audit
    if "hmm_prob_bull_trend" in mff.columns and "hmm_modulator_long" in mff.columns:
        bull_period = mff[mff["hmm_prob_bull_trend"] > 0.5]
        if not bull_period.empty:
            avg_mod_l = bull_period["hmm_modulator_long"].mean()
            _logger.info(f"\n [ALPHA TEST] Bull Regime (p>0.5) detected in {len(bull_period)} bars.")
            _logger.info(f"   - Avg Long Modulator during Bull:   {avg_mod_l:.4f} (Expect > 1.2)")

    _logger.info("=" * 85)

def test_universe_gp_hmm_flow():
    _logger.info("=" * 85)
    _logger.info(" [TEST] Universe -> GP Alpha -> HMM Regime Flow (Integrated)")
    _logger.info("=" * 85)
    
    # 1. Window Setup (Reduced for speed)
    res = get_quarterly_window()
    fetch_start, start, is_end, end = res
    tf = "1h"
    collector = DataCollector()
    
    # Adjust window to be smaller for testing (last 500 bars of IS)
    is_end_dt = pd.to_datetime(is_end)
    start_dt = is_end_dt - pd.Timedelta(hours=500)
    start = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    _logger.info(f"Window: {fetch_start} ~ {is_end} (Audit Focus: last 500 IS bars)")

    # 2. Universe Filtering (Broad)
    from src.domain.futures.opt_futures_utils.universe_screener_futures import (
        screen_futures_universe,
        screen_symbol_refinement_futures,
    )
    
    broad_candidates, _ = screen_futures_universe(
        collector, [], tf, FUTURES_SCREENER_CONFIG, fetch_start, is_end, data_dir=FUTURES_DATA_DIR
    )
    
    if not broad_candidates:
        _logger.error("No broad candidates found. Aborting.")
        return

    # 3. Data Loading for Refinement
    test_symbols = list(broad_candidates)[:5] # More restricted for speed
    data_maps_broad, _, valid_broad = _load_futures_data_maps_for_symbols(
        test_symbols, tf, fetch_start, start, is_end, end, skip_metrics=True
    )
    
    # 4. Refinement
    success = screen_symbol_refinement_futures(
        broad_candidates=valid_broad,
        winning_signal_type="CS_RANK",
        is_end_date=is_end,
        tf=tf,
        symbol_dfs_4h={s: data_maps_broad[s][tf] for s in valid_broad},
        daily_dfs={s: data_maps_broad[s]["1d"] for s in valid_broad},
        phase_b_params=None,
        anchor_symbols=FUTURES_ANCHOR_SYMBOLS,
    )
    
    importlib.reload(config.opt_config)
    final_symbols = config.opt_config.FUTURES_SYMBOLS
    _logger.info(f"Final Selected Universe: {final_symbols}")

    # 5. ML Pipeline (HMM Focused)
    cfg = dict(OPT_FUTURES_CONFIG)
    # Minimal GP for speed, focusing on HMM
    cfg["FUTURES_ML_GP_GENERATIONS"] = 1
    cfg["FUTURES_ML_GP_POPULATION"] = 100
    
    _logger.info("\nExecuting ML Pipeline (Integrated HMM Audit Mode)...")
    ml_out = run_ml_pipeline_for_universe(
        final_symbols,
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
    
    # 6. Verification & Deep Audit
    audit_hmm_logic_changes(ml_out, final_symbols)
        
    _logger.info("=" * 85)
    _logger.info(" [RESULT] Integrated Universe -> HMM test completed.")
    _logger.info("=" * 85)

if __name__ == "__main__":
    test_universe_gp_hmm_flow()
