"""
Independent Test: 4-Phase Advanced Universe Screening for Futures.
Runs Phase 1-4 and prints the selected orthogonal portfolio.
"""

import sys
import logging
from pathlib import Path

# Project Root Setup
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from src.domain.futures.data_collector import DataCollector
from src.domain.futures.opt_futures_utils.universe_screener_futures import screen_futures_universe
from config.opt_config import FUTURES_ANCHOR_SYMBOLS, FUTURES_DYNAMIC_CANDIDATE_POOL, FUTURES_SCREENER_CONFIG
from config.settings import FUTURES_DATA_DIR

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("test_universe")

def main():
    _logger.info("=" * 70)
    _logger.info("🚀 RUNNING ADVANCED 4-PHASE UNIVERSE SCREENING TEST")
    _logger.info("=" * 70)

    collector = DataCollector()
    
    # 1. Prepare Candidate Pool (Anchors + Dynamic Pool)
    # We use a broad pool to see how the screener filters and clusters them.
    candidate_pool = [] # Empty list triggers full Binance USDT perpetual scan
    
    # 2. Time Window Setup (Recent 2 years for robust stats)
    # Using fixed dates for the test to ensure consistency
    fetch_start = "2023-01-01"
    end_date = "2026-04-11" # Current date in context
    tf = "4h"

    _logger.info(f"Target Pool Size: FULL BINANCE USDT MARKET")
    _logger.info(f"Time Range: {fetch_start} ~ {end_date}")
    _logger.info(f"Timeframe: {tf}")
    _logger.info("-" * 70)

    # 3. Run the Advanced Screener
    # This will execute:
    # Phase 1: Amihud Liquidity & Zombie Check
    # Phase 2: Hurst Exponent & EMA 200 Trend
    # Phase 3: Downside Correlation Clustering (K-Means)
    # Phase 4: Squeeze-Aware Funding Bonus
    selected_symbols, n_passed_gates = screen_futures_universe(
        collector=collector,
        candidate_pool=candidate_pool,
        tf=tf,
        cfg=FUTURES_SCREENER_CONFIG,
        fetch_start=fetch_start,
        end_date=end_date,
        data_dir=FUTURES_DATA_DIR
    )

    _logger.info("\n" + "=" * 70)
    _logger.info("✅ FINAL SELECTED UNIVERSE (ORTHOGONAL PORTFOLIO)")
    _logger.info("=" * 70)
    _logger.info(f"Total Symbols Passed Phase 1 & 2 Gates: {n_passed_gates}")
    _logger.info(f"Final Selected (Clustered & Rank-Optimized): {len(selected_symbols)}")
    _logger.info("-" * 70)
    
    for i, sym in enumerate(selected_symbols):
        _logger.info(f"  [{i+1}] {sym}")
    
    _logger.info("=" * 70)
    _logger.info("Note: These symbols are now persisted in config/opt_config.py")

if __name__ == "__main__":
    main()
