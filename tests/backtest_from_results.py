import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Project Root Setup
project_root: str = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.data_collector import DataCollector
from src.domain.futures.strategies_futures import UltimateStrategy
from src.domain.futures.funding_utils import merge_funding_into_ohlcv
from src.domain.futures.opt_futures_utils.evaluator import evaluate_symbol_fold as evaluate_symbol_fold_futures

# Spot Imports
from src.domain.spot.data_collector_spot import DataCollectorSpot
from src.domain.spot.strategies_spot import UltimateSpotStrategy
from src.domain.spot.opt_spot_utils.evaluator import evaluate_symbol_fold as evaluate_symbol_fold_spot

from src.core.optimization.opt_utils import compute_segment_merge_index
from config.settings import DATA_DIR
from config.opt_config import get_quarterly_window

def main():
    parser = argparse.ArgumentParser(description="Run futures/spot backtests using results JSON files.")
    parser.add_argument("--file", type=str, help="Comma-separated paths to JSON parameter files (e.g., results/file1.json,results/file2.json)")
    parser.add_argument("--type", type=str, choices=["k", "u"], help="Type of strategies to backtest: 'k' for KRW/Spot (Upbit), 'u' for USDT/Futures (Binance)")
    parser.add_argument("--reference-date", type=str, default=None, help="Reference date for quarterly window (YYYY-MM-DD)")
    args = parser.parse_args()

    # 1. Resolve File Paths
    json_files = []
    if args.file:
        raw_paths = [p.strip() for p in args.file.split(',')]
        for rp in raw_paths:
            if rp:
                p = Path(rp)
                if p.exists():
                    json_files.append(p)
                else:
                    print(f"Warning: File not found - {p}")
    else:
        # Default scan based on type
        results_dir = Path(project_root) / "results"
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            return
            
        pattern = "*USDT*.json"
        if args.type == "k":
            pattern = "*KRW*.json"
        elif args.type == "u":
            pattern = "*USDT*.json"
            
        json_files = sorted(list(results_dir.glob(pattern)))
        if json_files:
            type_label = "USDT/Futures" if "USDT" in pattern else "KRW/Spot"
            print(f"No files specified. Auto-detecting {len(json_files)} {type_label} result files...\n")
        else:
            print(f"No matching result files found in {results_dir} for pattern {pattern}")
            return

    if not json_files:
        print("No valid parameter files to process.")
        return

    # 2. Setup Context
    FETCH_START_DATE, START_DATE, IS_END_DATE, END_DATE = get_quarterly_window(args.reference_date)
    
    # Collectors
    collector_futures = DataCollector()
    collector_spot = DataCollectorSpot()
    
    # Cache for data
    data_cache = {}

    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            params = json.load(f)

        filename = json_path.stem
        
        # Determine if it's Spot or Futures based on filename
        is_spot = "KRW" in filename
        current_type = "SPOT" if is_spot else "FUTURES"
        
        # Extract symbol
        parts = filename.split('_')
        symbol = "UNKNOWN"
        for p in parts:
            if "USDT" in p or "KRW" in p:
                symbol = p
                break
        
        tf = params.get("TIMEFRAME", "4h")
        
        # Clean symbol for Collector
        clean_symbol = symbol
        if "/" not in symbol:
            if symbol.endswith("USDT"):
                clean_symbol = f"{symbol[:-4]}/USDT"
            elif symbol.startswith("KRW-"):
                clean_symbol = symbol
            elif symbol.startswith("KRW"):
                clean_symbol = f"KRW-{symbol[3:]}"
        
        print(f"────────────────────────────────────────────────────────────────────────────────")
        print(f"[{current_type}] {clean_symbol} ({tf}) | File: {json_path.name}")
        
        # 3. Load Data
        cache_key = (clean_symbol, tf, current_type)
        if cache_key not in data_cache:
            try:
                if is_spot:
                    df_tf = collector_spot.collect_and_save(clean_symbol, tf, FETCH_START_DATE, END_DATE)
                    df_1d = collector_spot.collect_and_save(clean_symbol, "1d", FETCH_START_DATE, END_DATE)
                else:
                    df_tf = collector_futures.collect_and_save(clean_symbol, tf, FETCH_START_DATE, END_DATE)
                    df_1d = collector_futures.collect_and_save(clean_symbol, "1d", FETCH_START_DATE, END_DATE)
                    df_tf = merge_funding_into_ohlcv(clean_symbol, df_tf, DATA_DIR)
                data_cache[cache_key] = (df_tf, df_1d)
            except Exception as e:
                print(f"Error loading data for {clean_symbol}: {e}")
                continue
        
        full_df_main, full_df_daily = data_cache[cache_key]
        if full_df_main is None or full_df_main.empty:
            print(f"Skipping {clean_symbol}: No data found.")
            continue

        # Setup Indices
        tz = full_df_main["datetime"].dt.tz
        is_start_dt = pd.to_datetime(START_DATE).tz_localize(tz) if tz else pd.to_datetime(START_DATE)
        is_end_dt = pd.to_datetime(IS_END_DATE).tz_localize(tz) if tz else pd.to_datetime(IS_END_DATE)
        
        # IS Data
        is_df_tf = full_df_main[full_df_main["datetime"] < is_end_dt].reset_index(drop=True)
        is_df_1d = full_df_daily[full_df_daily["datetime"] < is_end_dt].reset_index(drop=True)
        m = is_df_tf["datetime"] >= is_start_dt
        is_start_idx = int(m.to_numpy().argmax()) if m.any() else 0
        merge_idx_is = compute_segment_merge_index(is_df_tf, is_df_1d)
        
        # OOS Data
        oos_start_m = full_df_main["datetime"] >= is_end_dt
        oos_start_idx = int(oos_start_m.to_numpy().argmax()) if oos_start_m.any() else len(full_df_main)
        merge_idx_oos = compute_segment_merge_index(full_df_main, full_df_daily)

        # 4. Run Backtests
        if is_spot:
            # Spot logic
            strat_is = UltimateSpotStrategy(name=f"IS_{clean_symbol}", params=params)
            res_is = evaluate_symbol_fold_spot(strat_is, params, clean_symbol, tf, is_df_tf, is_df_1d, merge_idx_is, None, is_start_idx, len(is_df_tf))
            
            strat_oos = UltimateSpotStrategy(name=f"OOS_{clean_symbol}", params=params)
            res_oos = evaluate_symbol_fold_spot(strat_oos, params, clean_symbol, tf, full_df_main, full_df_daily, merge_idx_oos, None, oos_start_idx, len(full_df_main))
        else:
            # Futures logic
            strat_is = UltimateStrategy(name=f"IS_{clean_symbol}", params=params)
            res_is = evaluate_symbol_fold_futures(strat_is, params, clean_symbol, tf, is_df_tf, is_df_1d, merge_idx_is, None, is_start_idx, len(is_df_tf))
            
            strat_oos = UltimateStrategy(name=f"OOS_{clean_symbol}", params=params)
            res_oos = evaluate_symbol_fold_futures(strat_oos, params, clean_symbol, tf, full_df_main, full_df_daily, merge_idx_oos, None, oos_start_idx, len(full_df_main))

        # 5. Output Results
        def format_res(res, spot=False):
            # res: (cagr, ret_pct, mdd_pct, trd, wr, pf, lc, sc?, eq)
            cagr, ret, mdd, trd, wr, pf = res[0], res[1], res[2], res[3], res[4], res[5]
            return f"CAGR: {cagr:>7.2f}% | Ret: {ret:>5.1f}% | MDD: {mdd:>5.1f}% | Trd: {int(trd):>4} | PF: {pf:>4.2f} | Win: {wr:>4.1f}%"

        print(f"  > IS : {format_res(res_is, is_spot)}")
        print(f"  > OOS: {format_res(res_oos, is_spot)}")

    print(f"────────────────────────────────────────────────────────────────────────────────\n")

if __name__ == "__main__":
    main()
