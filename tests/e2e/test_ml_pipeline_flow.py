import importlib
import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

# Project Root Setup
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config.opt_config
from config.opt_config import (
    FUTURES_ANCHOR_SYMBOLS,
    FUTURES_MACRO_INDEX_SYMBOLS,
    OPT_FUTURES_CONFIG,
    get_quarterly_window,
)
from config.settings import FUTURES_CACHE_DIR
from src.domain.futures.strategy_runtime.bridge import (
    HMM_SEMANTIC_PROB_COLUMNS,
    run_ml_pipeline_for_universe,
)
from src.domain.futures.optimization.opt_data_utils import load_futures_data_maps_for_symbols
from src.domain.futures.universe import load_or_build_universe_snapshot

warnings.filterwarnings("ignore")

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("test_ml_pipeline_flow")

def audit_hmm_logic_changes(ml_out, symbols, data_maps, tf):
    """Deep audit of HMM regime classification with Log-Wealth & Tail Capture metrics."""
    _logger.info("\n [HMM INTEGRATED AUDIT] TF: %s", tf)
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")

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

    for col in regime_cols:
        label = regime_display.get(col, col.replace("hmm_prob_", "").upper()[:12])
        mask = merged["regime"] == col
        if mask.any():
            r = merged.loc[mask, "ret"]
            mu = float(r.mean() * 100.0)
            sig = float(r.std() * 100.0)
            g = mu - 0.5 * (sig**2 / 100.0)
            time_pct = float(mask.mean() * 100.0)
            _logger.info(f"  {label} : {time_pct:>5.1f}% | G: {g:+.3f}%")
        else:
            _logger.info(f"  {label} :   0.0% | G:  ----")
    
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")

def test_full_pipeline_flow():
    """E2E test covering Universe Screening -> Refinement -> ML Pipeline -> HMM Audit.
    """
    tf = "4h"
    _logger.info("=" * 85)
    _logger.info(f" [E2E TEST] Universe -> ML Pipeline Flow | TF: {tf}")
    _logger.info("=" * 85)
    
    # 1. Window Setup
    fetch_start, start, is_end, end = get_quarterly_window(None)
    # 2. Universe Snapshot Build/Load (spec path)
    try:
        snapshot, selected_frame, _report = load_or_build_universe_snapshot(as_of=is_end, tf=tf)
        if selected_frame is not None and not selected_frame.empty and "symbol" in selected_frame.columns:
            selected_symbols = [
                str(symbol).strip()
                for symbol in selected_frame["symbol"].astype(str).tolist()
                if str(symbol).strip()
            ]
        else:
            selected_symbols = [
                str(meta.symbol).strip() for meta in snapshot.selected if str(meta.symbol).strip()
            ]
    except FileNotFoundError:
        selected_symbols = list(config.opt_config.FUTURES_SYMBOLS)
    assert len(selected_symbols) > 0, "Universe snapshot/config returned no symbols"

    importlib.reload(config.opt_config)
    final_symbols = list(set(selected_symbols + FUTURES_ANCHOR_SYMBOLS + FUTURES_MACRO_INDEX_SYMBOLS))
    
    # 5. Clear HMM Cache
    if FUTURES_CACHE_DIR.exists():
        for f in os.listdir(FUTURES_CACHE_DIR):
            if "HMM" in f and f.endswith(".parquet"):
                try: os.remove(FUTURES_CACHE_DIR / f)
                except Exception: pass

    # 6. Data Loading for ML
    data_maps, oos_data_maps, valid_ml_symbols = load_futures_data_maps_for_symbols(
        final_symbols, tf, fetch_start, start, is_end, end
    )
    
    # 7. ML Pipeline
    cfg = dict(OPT_FUTURES_CONFIG)
    ml_out = run_ml_pipeline_for_universe(
        valid_ml_symbols, tf, fetch_start, end, cfg,
        workers=4, n_jobs=4, is_end_date=is_end, is_start_date=start,
        preloaded_data_maps=data_maps, hmm_only=True
    )
    
    # 8. HMM Audit
    audit_hmm_logic_changes(ml_out, valid_ml_symbols, data_maps, tf)
    _logger.info(" [RESULT] E2E Pipeline Flow Test Completed Successfully.")

if __name__ == "__main__":
    test_full_pipeline_flow()
