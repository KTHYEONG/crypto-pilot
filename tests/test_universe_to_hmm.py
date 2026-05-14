import importlib
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr

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
from src.domain.futures.ml_pipeline.features.engineering import HMM_SEMANTIC_PROB_COLUMNS
from src.domain.futures.data_loader import DataCollector
from src.domain.futures.ml_pipeline.pipeline_runner import run_ml_pipeline_for_universe
from src.domain.futures.optimization.opt_data_utils import load_futures_data_maps_for_symbols
from src.domain.futures.optimization.screener import (
    screen_futures_universe,
    screen_symbol_refinement_futures,
)

warnings.filterwarnings("ignore")

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("test_universe_to_hmm")

def audit_hmm_logic_changes(ml_out, symbols, data_maps, tf):
    """Deep audit of HMM regime classification with Log-Wealth & Tail Capture metrics (Compact V2)."""
    _logger.info("\n [HMM INTEGRATED AUDIT] TF: %s", tf)
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")

    # Try to get data from symbols first, then fallback to market_probs
    mff = None
    sym = None
    for s in symbols:
        if s in ml_out.meta_feature_frame_by_symbol and s in data_maps:
            sym = s
            mff = ml_out.meta_feature_frame_by_symbol[s]
            break
    
    if mff is None and not ml_out.market_probs.empty:
        mff = ml_out.market_probs
        for s in ["BTC/USDT", "BTCUSDT"] + symbols:
            if s in data_maps:
                sym = s
                break

    if mff is None or sym is None:
        _logger.error(" [FAIL] No common data for HMM audit.")
        return

    df = data_maps[sym][tf].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")
    df["ret"] = df["close"].pct_change().fillna(0.0)
    
    if "datetime" not in mff.columns:
        mff = mff.reset_index()
    mff["datetime"] = pd.to_datetime(mff["datetime"], utc=True)
    merged = pd.merge(mff, df[["datetime", "close", "ret"]], on="datetime", how="inner")
    
    if merged.empty:
        _logger.error(" [FAIL] Merge produced empty frame. Check datetime alignment.")
        return

    regime_cols = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in merged.columns]
    if not regime_cols:
        _logger.error(" [FAIL] Missing HMM probability columns.")
        return

    merged["regime"] = merged[regime_cols].idxmax(axis=1)
    
    regime_display: dict[str, str] = {
        "hmm_prob_bull_calm": "🐂 BULL-CALM ",
        "hmm_prob_bull_vol_up": "🚀 BULL-VOL  ",
        "hmm_prob_bear_trend": "🐻 BEAR-TREND",
        "hmm_prob_chop": "🎢 CHOP-ZONE ",
        "hmm_prob_crisis": "💀 CRISIS    ",
    }

    # 1. Log-Wealth Dispersion
    for col in regime_cols:
        label = regime_display.get(col, col.replace("hmm_prob_", "").upper()[:12])
        mask = merged["regime"] == col
        if mask.any():
            r = merged.loc[mask, "ret"]
            mu = float(r.mean() * 100.0)
            sig = float(r.std() * 100.0)
            g = mu - 0.5 * (sig**2 / 100.0)
            time_pct = float(mask.mean() * 100.0)

            behavior = "NOISE"
            if g > 0.02: behavior = "GROWTH"
            elif g < -0.10: behavior = "DEFENSE"
            
            v_tag = ""
            if "BEAR" in label:
                v_tag = " [PASS]" if mu < -0.2 else " [FAIL]" if mu >= 0 else " [WARN]"
            elif "CRISIS" in label:
                v_tag = " [PREVENTIVE]" if mu > 0.5 else " [LAGGING]"

            _logger.info(f"  {label} : {time_pct:>5.1f}% | G: {g:+.3f}% | {behavior:<8}{v_tag}")
        else:
            _logger.info(f"  {label} :   0.0% | G:  ----   | [None]")
    
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")

    # 2. Tail Capture
    q05_thr = float(merged["ret"].quantile(0.05))
    worst_bars = merged[merged["ret"] <= q05_thr]
    if not worst_bars.empty:
        tail_dist = worst_bars["regime"].value_counts(normalize=True) * 100
        crisis_capture = float(tail_dist.get("hmm_prob_crisis", 0.0) + tail_dist.get("hmm_prob_bear_trend", 0.0))
        status = "PASS" if crisis_capture >= 40 else "FAIL"
        _logger.info(f"  > Tail-Capture: {crisis_capture:>5.1f}% ({status}) | Worst 5%%: {len(worst_bars)} bars")
    
    # 3. Stability
    transitions = int((merged["regime"] != merged["regime"].shift(1)).sum())
    avg_duration = float(len(merged) / max(1, transitions))
    stab_status = "PASS" if avg_duration >= 24 else "FAIL"
    _logger.info(f"  > Stability   : {avg_duration:>5.1f} bars/regime ({stab_status}) | Switches: {transitions}")

    # 4. Lead-Lag & IC
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    is_crisis_arr = (merged["regime"] == "hmm_prob_crisis").to_numpy()
    is_tail_arr   = (merged["ret"] <= q05_thr).to_numpy()
    crisis_enters = is_crisis_arr & np.concatenate([[False], ~is_crisis_arr[:-1]])
    fwd_tail = np.array([bool(np.any(is_tail_arr[i : min(i + 9, len(merged))])) for i in range(len(merged))])
    
    n_entries = int(crisis_enters.sum())
    if n_entries > 0:
        ll_capture = float(fwd_tail[crisis_enters].mean() * 100.0)
        _logger.info(f"  > Lead-Lag (8b): {ll_capture:>5.1f}% (N={n_entries} entries)")

    crisis_col = next((c for c in regime_cols if "crisis" in c), None)
    if crisis_col:
        fwd_ret = merged["ret"].shift(-4)
        v_mask = fwd_ret.notna() & merged[crisis_col].notna()
        if v_mask.sum() > 50:
            ic_val, _ = spearmanr(merged.loc[v_mask, crisis_col], fwd_ret[v_mask])
            _logger.info(f"  > Regime IC     : {ic_val:>+.4f} (Spearman p_crisis vs fwd_4b_ret)")

    _logger.info(" ────────────────────────────────────────────────────────────────────────────\n")

def audit_oos_and_ic(ml_out, symbols, is_data_maps, oos_data_maps, tf):
    """OOS Holdout Audit for HMM Tail Capture (Compact V2)."""
    _logger.info(" [OOS HMM AUDIT] Generalization Check | TF: %s", tf)
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")

    mff = None
    sym = None
    for s in symbols:
        if s in ml_out.meta_feature_frame_by_symbol and s in is_data_maps and s in oos_data_maps:
            sym = s
            mff = ml_out.meta_feature_frame_by_symbol[s]
            break
    
    if mff is None and not ml_out.market_probs.empty:
        mff = ml_out.market_probs
        for s in ["BTC/USDT", "BTCUSDT"] + symbols:
            if s in is_data_maps and s in oos_data_maps:
                sym = s
                break

    if mff is None or sym is None:
        _logger.info("  [SKIP] No IS+OOS data for audit.")
        return

    if "datetime" not in mff.columns:
        mff = mff.reset_index()
    mff["datetime"] = pd.to_datetime(mff["datetime"], utc=True)
    regime_cols = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in mff.columns]

    oos_df = oos_data_maps[sym][tf].copy()
    oos_df["datetime"] = pd.to_datetime(oos_df["datetime"], utc=True)
    merged_oos = pd.merge(mff, oos_df[["datetime", "close"]], on="datetime", how="inner")

    if len(merged_oos) > 100:
        merged_oos = merged_oos.sort_values("datetime").reset_index(drop=True)
        merged_oos["ret"] = merged_oos["close"].pct_change().fillna(0.0)
        merged_oos["regime"] = merged_oos[regime_cols].idxmax(axis=1)
        q05 = float(merged_oos["ret"].quantile(0.05))
        worst_oos = merged_oos[merged_oos["ret"] <= q05]
        
        if not worst_oos.empty:
            tail_dist = worst_oos["regime"].value_counts(normalize=True) * 100
            capture = float(tail_dist.get("hmm_prob_crisis", 0.0) + tail_dist.get("hmm_prob_bear_trend", 0.0))
            verdict = "PASS" if capture >= 50 else "WARN"
            _logger.info(f"  > OOS Tail Capture : {capture:5.1f}% ({verdict})")

        crisis_col = next((c for c in regime_cols if "crisis" in c), None)
        if crisis_col:
            fwd_ret = merged_oos["ret"].shift(-4)
            v_mask = fwd_ret.notna() & merged_oos[crisis_col].notna()
            if v_mask.sum() > 50:
                ic_val, _ = spearmanr(merged_oos.loc[v_mask, crisis_col], fwd_ret[v_mask])
                _logger.info(f"  > OOS Regime IC    : {ic_val:>+.4f}")

    _logger.info(" ────────────────────────────────────────────────────────────────────────────\n")



def test_universe_gp_hmm_flow(tf="4h"):
    _logger.info("=" * 85)
    _logger.info(f" [TEST] Universe -> HMM Regime Flow (Integrated) TF: {tf}")
    _logger.info("=" * 85)
    
    # 1. Window Setup (Aligned with opt_futures.py)
    fetch_start_date, start_date, is_end_date, end_date = get_quarterly_window(None)
    collector = DataCollector()
    
    _logger.info(f"Window: {fetch_start_date} ~ {end_date} (IS End: {is_end_date})")

    # 2. Universe Discovery & Filtering (Aligned with opt_futures.py)
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

    data_maps_broad, _, valid_broad = load_futures_data_maps_for_symbols(
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
    
    # Reload config to get the updated FUTURES_SYMBOLS
    importlib.reload(config.opt_config)
    from config.opt_config import FUTURES_MACRO_INDEX_SYMBOLS, FUTURES_SYMBOLS
    final_symbols = list(set(FUTURES_SYMBOLS + FUTURES_ANCHOR_SYMBOLS + FUTURES_MACRO_INDEX_SYMBOLS))
    _logger.info(f"Verified ML Universe: {len(final_symbols)} symbols (Anchors + Macro Index included)")

    # 3. Clear HMM Cache to force fresh training
    from config.settings import FUTURES_CACHE_DIR
    import os
    if FUTURES_CACHE_DIR.exists():
        for f in os.listdir(FUTURES_CACHE_DIR):
            if "HMM" in f and f.endswith(".parquet"):
                try:
                    os.remove(FUTURES_CACHE_DIR / f)
                except Exception:
                    pass

    # 4. Data Loading for ML (Ensuring all symbols including anchors are loaded)
    data_maps, oos_data_maps, valid_ml_symbols = load_futures_data_maps_for_symbols(
        final_symbols, tf, fetch_start_date, start_date, is_end_date, end_date
    )

    # 5. ML Pipeline (HMM Focused)
    cfg = dict(OPT_FUTURES_CONFIG)
    # Ensure consistency with opt_main_futures.py defaults
    # cfg["FUTURES_ML_ALPHA_USE_TBM_WEIGHT"] = True (from opt_config)
    
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
        hmm_only=True
    )
    
    elapsed_time = time.time() - start_time
    _logger.info(f"\n[TIME] ML Pipeline (JAX HMM) took: {elapsed_time:.2f} seconds")
    
    # 6. Verification & Deep Audit
    # We use data_maps (IS only) for audit consistency
    audit_hmm_logic_changes(ml_out, valid_ml_symbols, data_maps, tf)

    # 7. OOS + IC Audit
    audit_oos_and_ic(ml_out, valid_ml_symbols, data_maps, oos_data_maps, tf)

    _logger.info("=" * 85)
    _logger.info(" [RESULT] Integrated Universe -> HMM test completed.")
    _logger.info("=" * 85)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str, default="4h", choices=["1h", "4h"])
    args = parser.parse_known_args()[0]
    test_universe_gp_hmm_flow(args.tf)

