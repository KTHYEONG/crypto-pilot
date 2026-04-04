import sys
import os
import logging
import pandas as pd
import numpy as np
import re
import random
import time
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, List, Tuple
from tqdm import tqdm

# Project Root Setup
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.spot_strategy.upbit_client import UpbitClient
from src.spot_strategy.data_collector_spot import DataCollectorSpot
from config.settings import SPOT_BACKTEST_START_DATE, SPOT_BACKTEST_END_DATE

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("universe_screener")

def calculate_hurst_exponent(ts: np.ndarray) -> float:
    """Returns the Hurst Exponent of the time series vector ts (R/S simplified)."""
    if len(ts) < 100:
        return 0.5
    lags = range(2, 60)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    try:
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(poly[0] * 2.0)
    except Exception:
        return 0.5

def calculate_efficiency_ratio(close: pd.Series, n: int = 168) -> float:
    """Kaufman's Efficiency Ratio (4H based: 168 bars ~ 1 month context)"""
    if len(close) < n:
        return 0.0
    direction = abs(float(close.iloc[-1]) - float(close.iloc[-n]))
    volatility = close.diff().abs().rolling(n).sum().iloc[-1]
    if volatility == 0:
        return 0.0
    return float(direction / volatility)

def screen_universe():
    client = UpbitClient()
    collector = DataCollectorSpot()
    
    _logger.info("Fetching all KRW markets from Upbit...")
    all_markets = client.exchange.load_markets()
    krw_symbols = [m['id'] for m in all_markets.values() if m['id'].startswith('KRW-')]
    
    _logger.info(f"Found {len(krw_symbols)} KRW symbols. Screening...")
    
    from config.opt_config import get_quarterly_window
    try:
        FETCH_START_DATE, START_DATE, _, _ = get_quarterly_window()
    except Exception:
        FETCH_START_DATE = (pd.to_datetime(SPOT_BACKTEST_END_DATE) - timedelta(days=365*3)).strftime("%Y-%m-%d")
        START_DATE = (pd.to_datetime(SPOT_BACKTEST_END_DATE) - timedelta(days=365*2)).strftime("%Y-%m-%d")
        
    end_date = SPOT_BACKTEST_END_DATE
    start_date = (pd.to_datetime(end_date) - timedelta(days=365)).strftime("%Y-%m-%d")
    
    stats = []
    symbol_returns: Dict[str, pd.Series] = {}

    for sym in tqdm(krw_symbols, desc="Screening Symbols"):
        try:
            # Prevent 429 Too Many Requests
            time.sleep(0.12)
            
            # 0. Fast Data Period Validation (Direct API Check)
            # Fetch just the first candle starting from FETCH_START_DATE to see when it was listed
            ccxt_sym = client._normalize_symbol(sym)
            since_ms = client.exchange.parse8601(f"{FETCH_START_DATE}T00:00:00Z")
            
            # Use small limit with retry to check oldest available data without full download
            first_candle = None
            for retry in range(3):
                try:
                    first_candle = client.exchange.fetch_ohlcv(ccxt_sym, "1d", since=since_ms, limit=1)
                    break
                except Exception as e:
                    if "too_many_requests" in str(e).lower() and retry < 2:
                        time.sleep(1.0 + random.random())
                        continue
                    raise e
            
            if not first_candle:
                continue
            
            oldest_ts = first_candle[0][0]
            oldest_dt = pd.to_datetime(oldest_ts, unit='ms')
            
            # Allow 5-day grace period for listing dates
            if oldest_dt > pd.to_datetime(FETCH_START_DATE) + timedelta(days=5):
                # Coin was listed after FETCH_START_DATE, skip it.
                continue
            
            # 1. Fetch 4h data only for the RECENT 1 year for Trend/ER analysis
            # Full history download is deferred until final selection or opt_spot main loop
            df = collector.collect_and_save(sym, "4h", start_date, end_date)
            if df is None or len(df) < 500: # Need enough history for analysis (1 year context)
                continue
                
            close = df['close']
            returns = np.log(close / close.shift(1)).fillna(0)
            
            # 1. ADV (Average Daily Volume in KRW) - Last 30 days
            recent_df = df.tail(180) # ~30 days
            adv = (recent_df['close'] * recent_df['volume']).mean() * 6 # 4H * 6 = Daily proxy
            
            # 2. Hurst Exponent (Long term memory)
            hurst = calculate_hurst_exponent(close.to_numpy())
            
            # 3. Efficiency Ratio (Trend purity)
            er = calculate_efficiency_ratio(close, n=168)
            
            stats.append({
                'symbol': sym,
                'adv': adv,
                'hurst': hurst,
                'er': er
            })
            symbol_returns[sym] = returns
            
        except Exception as e:
            _logger.warning(f"Failed to process {sym}: {e}")

    stats_df = pd.DataFrame(stats)
    if stats_df.empty:
        _logger.error("No symbols passed initial data check.")
        return

    # Phase A: Liquidity & Trend Filter
    # ADV top 60%
    adv_threshold = stats_df['adv'].quantile(0.4)
    filtered_df = stats_df[stats_df['adv'] >= adv_threshold].copy()
    
    # Hurst > 0.48 and ER ranking
    filtered_df = filtered_df[filtered_df['hurst'] >= 0.48] # Conservative threshold
    filtered_df['score'] = filtered_df['hurst'] + filtered_df['er'] * 0.5
    top_candidates = filtered_df.sort_values('score', ascending=False).head(15)
    
    _logger.info(f"Top 15 candidates based on Trendiness:\n{top_candidates[['symbol', 'hurst', 'er', 'score']]}")

    # Phase B: Correlation Clustering
    selected_returns = pd.DataFrame({s: symbol_returns[s] for s in top_candidates['symbol']})
    corr_matrix = selected_returns.corr()
    dist_matrix = 1 - corr_matrix.fillna(0)
    
    # 5~6 clusters
    n_clusters = 6
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
    labels = clusterer.fit_predict(dist_matrix)
    
    top_candidates['cluster'] = labels
    
    final_symbols = []
    cluster_names = {}
    
    for i in range(n_clusters):
        cluster_members = top_candidates[top_candidates['cluster'] == i]
        if cluster_members.empty:
            continue
        # Pick best from cluster
        best = cluster_members.sort_values('score', ascending=False).iloc[0]
        final_symbols.append(best['symbol'])
        cluster_names[best['symbol']] = f"cluster_{i}"

    _logger.info(f"Final Scientifically Selected Symbols: {final_symbols}")
    
    # Update config/opt_config.py
    update_config_file(final_symbols, cluster_names)

def update_config_file(symbols: List[str], clusters: Dict[str, str]):
    config_path = Path("config/opt_config.py")
    if not config_path.exists():
        _logger.error("config/opt_config.py not found.")
        return
    
    content = config_path.read_text(encoding="utf-8")

    # 1. Update SPOT_SYMBOLS block (Top level list)
    pattern_sym = r"SPOT_SYMBOLS(?:: List\[str\])?\s*=\s*\[.*?\]"
    new_sym_block = "SPOT_SYMBOLS: List[str] = [\n"
    for s in symbols:
        new_sym_block += f"    \"{s}\",\n"
    new_sym_block += "]"
    content = re.sub(pattern_sym, new_sym_block, content, count=1, flags=re.DOTALL)

    # 2. Update SPOT_SYMBOL_CLUSTER block (Inside Dict)
    pattern_cls = r"\"SPOT_SYMBOL_CLUSTER\":\s*\{.*?\}(?=,)"
    new_cls_block = "\"SPOT_SYMBOL_CLUSTER\": {\n"
    for s, c in clusters.items():
        new_cls_block += f"            \"{s}\": \"{c}\",\n"
    new_cls_block += "        }"
    content = re.sub(pattern_cls, new_cls_block, content, count=1, flags=re.DOTALL)

    config_path.write_text(content, encoding="utf-8")
    _logger.info("Successfully updated config/opt_config.py with new symbols and clusters.")

if __name__ == "__main__":
    screen_universe()
