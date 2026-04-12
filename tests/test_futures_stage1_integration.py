import time
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Project Root Setup
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Stage 1 and dependencies
from src.execution.opt_main_futures import _run_stage1_futures_tournament
from src.domain.futures.signals import FUTURES_SIGNAL_REGISTRY
from src.domain.futures.regimes import FUTURES_REGIME_REGISTRY
from config.opt_config import OPT_FUTURES_CONFIG

# Ensure results are returned even with random data
OPT_FUTURES_CONFIG["FUTURES_MIN_TRADES_PER_CPCV_SEGMENT"] = 0

def generate_synthetic_data(symbol: str, length: int = 2000) -> Dict[str, Any]:
    # 4h data
    dates = pd.date_range("2022-01-01", periods=length, freq="4h")
    # Add significant volatility to trigger signals
    base_price = 1000.0
    vols = np.cumsum(np.random.randn(length) * 50)
    close = base_price + vols
    open_p = close + np.random.randn(length) * 10
    high = np.maximum(open_p, close) + np.random.rand(length) * 30
    low = np.minimum(open_p, close) - np.random.rand(length) * 30
    
    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.rand(length) * 10000,
        "taker_buy_volume": np.random.rand(length) * 5000
    })
    
    # Funding rate is expected in separate files or merged. 
    # In opt_main_futures, it merges it. Here we simulate it's already there.
    df["funding_rate"] = np.random.randn(length) * 0.0001
    df["datetime"] = dates
    
    # 1d data
    def to_1d(df_in):
        df_1d = df_in.set_index("datetime").resample("D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).reset_index()
        return df_1d

    df_1d = to_1d(df)
    
    return {
        "4h": df,
        "1d": df_1d,
        "is_start_idx_4h": 0,
        "merge_idx_4h": 0 
    }

if __name__ == "__main__":
    print("=== Stage 1 Performance & Bottleneck Test ===")
    print(f"Detected {len(FUTURES_SIGNAL_REGISTRY)} signals and {len(FUTURES_REGIME_REGISTRY)} regimes.")
    
    # Combination count (same as in opt_main_futures)
    combos_count = len(FUTURES_SIGNAL_REGISTRY) * len(FUTURES_REGIME_REGISTRY) * 2
    print(f"Total combinations to evaluate: {combos_count}")
    
    symbols = ["SYM_1", "SYM_2"]
    print(f"Generating synthetic data for {symbols}...")
    data_maps = {s: generate_synthetic_data(s) for s in symbols}
    
    print("\nStarting Stage 1 Tournament...")
    print("NOTE: If parallelization is working, you should see multiple processes active.")
    
    # Debug single run
    sig_choices = list(FUTURES_SIGNAL_REGISTRY.keys())
    reg_choices = list(FUTURES_REGIME_REGISTRY.keys())
    print(f"\n[DEBUG] Running single combo for verification: {sig_choices[0]} x {reg_choices[0]}")
    try:
        debug_res = _eval_combo_task(
            sig_choices[0], reg_choices[0], "inv_vol_parity",
            data_maps, symbols, "4h", project_root, None
        )
        print(f"[DEBUG] Single combo result: {debug_res}")
    except Exception as e:
        print(f"[DEBUG] Single combo FAILED: {e}")
        import traceback
        traceback.print_exc()

    start_time = time.time()
    
    try:
        # Run Stage 1
        results = _run_stage1_futures_tournament(
            data_maps=data_maps,
            symbols=symbols,
            tf="4h",
            project_root=project_root
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*50)
        print(f"Tournament result count: {len(results)}")
        print(f"Total time taken: {duration:.2f} seconds")
        print(f"Average time per combo: {duration/combos_count:.4f}s (including overhead)")
        print("="*50)
        
        if len(results) > 0:
            print(f"Top Score: {results[0].p10_gmgr:.4f} ({results[0].signal} x {results[0].regime} x {results[0].sizing})")
            print("\n[JUDGEMENT] Stage 1 is functioning correctly.")
            if duration < (combos_count * 1.5): # Rough heuristic for 8 cores vs sequential
                print("[JUDGEMENT] Parallelization confirmed (Execution time is low).")
            else:
                print("[JUDGEMENT] Check if parallelization efficiency can be improved.")
        else:
            print("\n[JUDGEMENT] FAILED: No results returned.")
            
    except Exception as e:
        print(f"\nERROR during Stage 1: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
