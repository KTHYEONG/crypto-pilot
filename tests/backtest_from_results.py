import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project Root Setup
project_root: str = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.secure_config import decrypt_config, get_strategy_secret
from config.settings import DATA_DIR, SPOT_INITIAL_BALANCE
from config.opt_config import (
    get_quarterly_window, 
    SPOT_SYMBOLS, 
    FUTURES_SYMBOLS,
    SPOT_ANCHOR_SYMBOLS,
    FUTURES_ANCHOR_SYMBOLS
)

# Spot Imports
from src.domain.spot.data_collector_spot import DataCollectorSpot
from src.domain.spot.strategies_spot import UltimateSpotStrategy
from src.domain.spot.opt_spot_utils.oos_evaluator import (
    run_holdout_shared_cash_portfolio,
    evaluate_symbol_fold as evaluate_symbol_fold_spot
)
from src.domain.spot.opt_spot_utils.data_utils import compute_segment_merge_index

# Futures Imports
from src.domain.futures.data_collector import DataCollector
from src.domain.futures.strategies_futures import UltimateStrategy as UltimateFuturesStrategy
from src.domain.futures.funding_utils import merge_funding_into_ohlcv
from src.domain.futures.opt_futures_utils.oos_evaluator import (
    run_oos_margin_shared_portfolio,
    evaluate_symbol_fold as evaluate_symbol_fold_futures
)

def load_params(file_path: Path) -> Dict[str, Any]:
    """Load parameters from JSON or ENC file."""
    if file_path.suffix == ".enc":
        secret = get_strategy_secret()
        if not secret:
            raise ValueError(f"STRATEGY_SECRET_KEY not set. Cannot decrypt {file_path}")
        return decrypt_config(file_path.read_bytes(), secret)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def run_spot_portfolio(params: Dict[str, Any], reference_date: Optional[str] = None):
    """Run portfolio backtest for Spot."""
    FETCH_START, START, IS_END, END = get_quarterly_window(reference_date)
    collector = DataCollectorSpot()
    symbols = SPOT_SYMBOLS
    tf = params.get("TIMEFRAME", "4h")
    
    data_maps = {}
    print(f"Loading data for {len(symbols)} Spot symbols...")
    
    # 1. Collect all symbol data
    for sym in symbols:
        df_tf = collector.collect_and_save(sym, tf, FETCH_START, END)
        df_1d = collector.collect_and_save(sym, "1d", FETCH_START, END)
        if df_tf is None or df_tf.empty:
            continue
            
        data_maps[sym] = {
            tf: df_tf,
            "1d": df_1d,
        }

    valid_symbols = list(data_maps.keys())
    if not valid_symbols:
        print("No valid data for symbols.")
        return

    # 2. Add Benchmark (BTC/ETH) if available for regime/mb
    for b_sym in ["KRW-BTC", "KRW-ETH"]:
        if b_sym in valid_symbols:
            b_data = data_maps[b_sym][tf][["datetime", "close"]].copy()
            suffix = "btc_close" if "BTC" in b_sym else "eth_close"
            b_data = b_data.rename(columns={"close": suffix})
            for sym in valid_symbols:
                if sym != b_sym:
                    data_maps[sym][tf] = data_maps[sym][tf].merge(b_data, on="datetime", how="left")

    # 3. Setup Indices for OOS Evaluator
    for sym in valid_symbols:
        df = data_maps[sym][tf]
        tz = df["datetime"].dt.tz
        is_end_dt = pd.to_datetime(IS_END).tz_localize(tz) if tz else pd.to_datetime(IS_END)
        m_oos = df["datetime"] >= is_end_dt
        data_maps[sym][f"oos_start_idx_{tf}"] = int(m_oos.to_numpy().argmax()) if m_oos.any() else len(df)
        data_maps[sym][f"merge_idx_{tf}"] = compute_segment_merge_index(df, data_maps[sym]["1d"])

    # 4. Run Portfolio Evaluator
    print(f"Running Spot Portfolio Backtest (OOS from {IS_END} to {END})...")
    res = run_holdout_shared_cash_portfolio(params, valid_symbols, tf, data_maps)
    
    print_portfolio_report(res, "SPOT")

def run_futures_portfolio(params: Dict[str, Any], reference_date: Optional[str] = None):
    """Run portfolio backtest for Futures."""
    FETCH_START, START, IS_END, END = get_quarterly_window(reference_date)
    collector = DataCollector()
    symbols = FUTURES_SYMBOLS
    tf = params.get("TIMEFRAME", "4h")
    
    oos_data_maps = {}
    print(f"Loading data for {len(symbols)} Futures symbols...")
    
    for sym in symbols:
        df_tf = collector.collect_and_save(sym, tf, FETCH_START, END)
        if df_tf is None or df_tf.empty:
            continue
        
        # Merge funding
        df_tf = merge_funding_into_ohlcv(sym, df_tf, DATA_DIR)
        
        tz = df_tf["datetime"].dt.tz
        is_end_dt = pd.to_datetime(IS_END).tz_localize(tz) if tz else pd.to_datetime(IS_END)
        m_oos = df_tf["datetime"] >= is_end_dt
        
        oos_data_maps[sym] = {
            tf: df_tf,
            f"oos_start_idx_{tf}": int(m_oos.to_numpy().argmax()) if m_oos.any() else len(df_tf)
        }

    valid_symbols = list(oos_data_maps.keys())
    if not valid_symbols:
        print("No valid data for symbols.")
        return

    print(f"Running Futures Portfolio Backtest (OOS from {IS_END} to {END})...")
    res = run_oos_margin_shared_portfolio(valid_symbols, tf, params, oos_data_maps)
    
    if res.get("ok"):
        print_portfolio_report(res, "FUTURES")
    else:
        print("Futures portfolio backtest failed.")

def print_portfolio_report(res: Dict[str, Any], mode: str):
    """Print a clean report for portfolio backtest results."""
    print(f"\n============================================================")
    print(f" [{mode} PORTFOLIO BACKTEST REPORT]")
    print(f"============================================================")
    
    if mode == "SPOT":
        cagr = res.get("portfolio_cagr_pct", 0.0)
        mdd = res.get("mdd_pct", 0.0)
        pf = res.get("profit_factor", 0.0)
        wr = res.get("win_rate_pct", 0.0)
        trds = res.get("long_trades", 0)
        moic = res.get("moic", 1.0)
        tail = res.get("tail_ratio", 0.0)
        calmar = res.get("calmar_ratio", 0.0)
    else:
        cagr = res.get("cagr_pct", 0.0)
        mdd = res.get("mdd_pct", 0.0)
        pf = res.get("profit_factor", 0.0)
        wr = res.get("win_rate_pct", 0.0)
        trds = res.get("total_trades", 0)
        moic = res.get("moic", 1.0)
        tail = res.get("tail_ratio", 0.0)
        calmar = res.get("calmar_ratio", 0.0)

    print(f" CAGR          : {cagr:>8.2f}%")
    print(f" MDD           : {mdd:>8.2f}%")
    print(f" Profit Factor : {pf:>8.2f}")
    print(f" Win Rate      : {wr:>8.2f}%")
    print(f" Total Trades  : {int(trds):>8}")
    print(f" MOIC          : {moic:>8.2f}x")
    print(f" Tail Ratio    : {tail:>8.2f}")
    print(f" Calmar Ratio  : {calmar:>8.2f}")
    print(f"============================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Clean backtest runner from optimization results.")
    parser.add_argument("--file", type=str, help="Path to result .json or .enc file")
    parser.add_argument("--reference-date", type=str, default=None, help="Reference date (YYYY-MM-DD)")
    args = parser.parse_args()

    if not args.file:
        print("Usage: python backtest_from_results.py --file results/best_spot_4h.json")
        return

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    try:
        params = load_params(file_path)
    except Exception as e:
        print(f"Error loading params: {e}")
        return

    filename = file_path.name.lower()
    
    # Logic to determine if it's Spot or Futures and if it's Portfolio or Single
    is_spot = "spot" in filename or "krw" in filename
    is_portfolio = "best_spot" in filename or "best_futures" in filename or "multi" in filename

    if is_portfolio:
        if is_spot:
            run_spot_portfolio(params, args.reference_date)
        else:
            run_futures_portfolio(params, args.reference_date)
    else:
        # Fallback to single symbol if possible (not requested but good to have)
        print("Single symbol backtest detected (based on filename). Running simplified backtest...")
        # For now, recommend using the portfolio mode for best_spot results.
        if is_spot:
            run_spot_portfolio(params, args.reference_date)
        else:
            run_futures_portfolio(params, args.reference_date)

if __name__ == "__main__":
    main()
