"""
Advanced 4-Phase Universe Screener for Binance Futures.
Focus: Geometric wealth compounding, slippage minimization, and orthogonal diversification.
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

# Project Root Setup
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import re
from config.opt_config import (
    FUTURES_ANCHOR_SYMBOLS,
    FUTURES_DYNAMIC_CANDIDATE_POOL,
    OPT_FUTURES_CONFIG,
)
from config.settings import FUTURES_DATA_DIR, SLIPPAGE_RATE
from src.domain.futures.data_collector import DataCollector
from src.domain.futures.funding_utils import merge_funding_into_ohlcv

_logger: logging.Logger = logging.getLogger("universe_screener_futures")

# --- Phase 2: Structural Trendiness Utilities ---

def _calculate_hurst_exponent(ts: np.ndarray) -> float:
    """
    Simplified R/S analysis for Hurst Exponent.
    H > 0.5: Trending, H < 0.5: Mean-reverting.
    """
    if ts.size < 100:
        return 0.5
    
    # Calculate log returns
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    
    # Linear fit to log-log plot
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(reg[0] * 2.0)
    except:
        return 0.5

# --- Phase 1: Capacity & Slippage Utilities ---

def _calculate_amihud_illiquidity(df: pd.DataFrame) -> float:
    """
    Amihud = mean(|return| / USDT_volume).
    Lower is better (more liquid/thicker order book).
    """
    if df.empty or len(df) < 20:
        return 999.0
    
    returns = df["close"].pct_change().abs()
    notional_vol = df["volume"] * df["close"]
    
    # Avoid zero division
    illiq = returns / notional_vol.replace(0, 1e-9)
    return float(illiq.tail(180).mean() * 1e6)  # Scale for easier comparison

# --- Phase 3: Orthogonal Diversification Utilities ---

def _get_downside_beta(symbol_returns: pd.Series, btc_returns: pd.Series) -> float:
    """
    Calculates Beta specifically during BTC down days.
    Measures 'Cascade Risk' susceptibility.
    """
    common_idx = symbol_returns.index.intersection(btc_returns.index)
    s_ret = symbol_returns.loc[common_idx]
    b_ret = btc_returns.loc[common_idx]
    
    # Filter for BTC negative returns
    down_mask = b_ret < 0
    if down_mask.sum() < 10:
        return 1.0
    
    s_down = s_ret[down_mask]
    b_down = b_ret[down_mask]
    
    try:
        slope, _, _, _, _ = stats.linregress(b_down, s_down)
        return float(slope)
    except:
        return 1.0

# --- Core Screening Worker ---

def _screen_worker_v2(
    sym: str,
    tf: str,
    fetch_start: str,
    end_date: str,
    cfg: Dict[str, Any],
    data_dir: Path,
) -> Dict[str, Any] | None:
    """Advanced Worker implementing Phase 1 & 2 Gates."""
    from src.domain.futures.data_collector import DataCollector
    collector = DataCollector()
    
    try:
        collector.ensure_funding_data(sym, fetch_start, end_date)
        df = collector.collect_and_save(sym, tf, fetch_start, end_date)
        df = merge_funding_into_ohlcv(sym, df, data_dir)
    except:
        return None

    min_bars = int(cfg.get("MIN_HISTORY_BARS", 2000))
    if df is None or len(df) < min_bars:
        return None

    # --- Phase 1: Dynamic Liquidity Gate ---
    notional_vol = (df["volume"] * df["close"]).to_numpy()
    adv_90d = np.mean(notional_vol[-540:]) if len(notional_vol) >= 540 else np.mean(notional_vol)
    adv_7d = np.mean(notional_vol[-42:]) if len(notional_vol) >= 42 else adv_90d
    
    # Prune 'Zombie' coins (Liquidity decaying)
    if adv_7d < adv_90d * 0.4:
        return None
    
    amihud = _calculate_amihud_illiquidity(df)
    
    # --- Phase 2: structural Trendiness & Quality ---
    close_arr = df["close"].to_numpy()
    hurst = _calculate_hurst_exponent(close_arr)
    
    # EMA 200 Filter: Anti-Top / Momentum alignment
    ema200 = df["close"].ewm(span=200, adjust=False).mean().iloc[-1]
    last_price = close_arr[-1]
    
    # Robust ATR% check
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close_arr, 1)), np.abs(low - np.roll(close_arr, 1))))
    atr_pct = (np.mean(tr[-14:]) / last_price) * 100.0

    # --- Phase 4: Squeeze-Aware Funding Stats ---
    funding = df["funding_rate"].to_numpy()
    mean_funding = np.mean(funding[-180:]) if len(funding) >= 180 else 0.0
    
    # Score for Phase 3 ranking
    # Higher hurst, lower amihud, squeeze potential (neg funding + price > ema)
    is_squeeze_potential = 1.2 if (mean_funding < 0 and last_price > ema200) else 1.0
    quality_score = (hurst * 10.0) * is_squeeze_potential / (np.log1p(amihud) + 1.0)

    return {
        "symbol": sym,
        "adv": adv_7d,
        "amihud": amihud,
        "hurst": hurst,
        "price_above_ema": last_price > ema200,
        "atr_pct": atr_pct,
        "mean_funding": mean_funding,
        "quality_score": quality_score,
        "returns": df["close"].pct_change().tail(500) # For clustering
    }

def screen_futures_universe(
    collector: DataCollector,
    candidate_pool: List[str],
    tf: str,
    cfg: Dict[str, Any],
    fetch_start: str,
    end_date: str,
    *,
    data_dir: Path | None = None,
) -> Tuple[List[str], int]:
    """
    The Ultimate 4-Phase Screener.
    Replaces old grid-based screening with Orthogonal Diversification.
    """
    dd = data_dir if data_dir is not None else FUTURES_DATA_DIR
    anchors = set(FUTURES_ANCHOR_SYMBOLS)
    
    # 1. Ticker Pre-Filter (Normalization fixed)
    _logger.info("Starting Advanced Universe Screening (4-Phase Architecture)...")
    try:
        tickers = collector.client.exchange.fetch_tickers()
        valid_tickers = []
        min_adv_ticker = float(cfg["ADV_MIN_USDT_DAY"]) * 0.3 # Relaxed pre-filter
        
        pool_set = set(candidate_pool) if candidate_pool else None
        for sym, t in tickers.items():
            norm_sym = sym.split(":")[0]
            if pool_set and norm_sym not in pool_set: continue
            if not (sym.endswith("/USDT") or sym.endswith("/USDT:USDT")): continue
            
            if norm_sym in anchors or float(t.get("quoteVolume") or 0.0) >= min_adv_ticker:
                valid_tickers.append(norm_sym)
        candidate_pool = list(dict.fromkeys(valid_tickers))
    except Exception as e:
        _logger.warning(f"Ticker pre-filter failed: {e}")

    # 2. Parallel History Screening (Phases 1, 2, 4)
    n_workers = max(1, min(int(os.cpu_count() or 4), 8))
    worker_fn = partial(_screen_worker_v2, tf=tf, fetch_start=fetch_start, end_date=end_date, cfg=cfg, data_dir=dd)
    
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        raw_results = list(tqdm(pool.map(worker_fn, candidate_pool), total=len(candidate_pool), desc="[Universe Gates]"))
    
    passed_rows = [r for r in raw_results if r is not None]
    
    # Hard Gates Pruning (Phase 2 & 1)
    final_candidates = []
    atr_min, atr_max = 2.5, 12.0 # Futures-optimized range
    
    for r in passed_rows:
        sym = r["symbol"]
        if sym in anchors:
            final_candidates.append(r)
            continue
            
        if r["hurst"] < 0.52: continue # Must be trending
        if not r["price_above_ema"]: continue # Must have momentum
        if not (atr_min <= r["atr_pct"] <= atr_max): continue # Reasonable volatility
        if r["amihud"] > np.percentile([x["amihud"] for x in passed_rows], 85): continue # Prune illiquid tail
        
        final_candidates.append(r)

    _logger.info(f"Phase 1 & 2 complete: {len(final_candidates)} symbols passed structural gates.")
    if not final_candidates: return list(anchors), 0

    # 3. Phase 3: Downside-Clustered Orthogonal Diversification
    # We use K-Means to group symbols by downside behavior
    try:
        from sklearn.cluster import KMeans
        
        # Build return matrix
        returns_df = pd.DataFrame({r["symbol"]: r["returns"] for r in final_candidates}).fillna(0)
        
        # Identify BTC downside periods (Macro stress)
        btc_sym = "BTC/USDT"
        if btc_sym in returns_df.columns:
            btc_rets = returns_df[btc_sym]
            stress_mask = btc_rets < btc_rets.quantile(0.20) # Worst 20% days
            stress_returns = returns_df[stress_mask]
        else:
            stress_returns = returns_df
            
        corr_matrix = stress_returns.corr().fillna(0)
        dist_matrix = 1.0 - corr_matrix # Distance metric
        
        n_clusters = min(len(final_candidates), int(cfg.get("MP_MIN_SYMBOLS", 5)) + 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(dist_matrix)
        
        # Map symbols to clusters
        sym_to_cluster = {final_candidates[i]["symbol"]: clusters[i] for i in range(len(final_candidates))}
        
        # Pick 1 'Alpha Leader' from each cluster based on quality_score
        cluster_buckets: Dict[int, List[Dict]] = {}
        for r in final_candidates:
            cid = sym_to_cluster[r["symbol"]]
            if cid not in cluster_buckets: cluster_buckets[cid] = []
            cluster_buckets[cid].append(r)
            
        final_list = []
        # Ensure Anchors are always prioritized
        for a in FUTURES_ANCHOR_SYMBOLS:
            if any(r["symbol"] == a for r in final_candidates):
                final_list.append(a)
        
        # Fill remaining slots from leaders of other clusters
        cluster_leaders = []
        for cid, members in cluster_buckets.items():
            # Sort by quality_score descending
            members.sort(key=lambda x: x["quality_score"], reverse=True)
            leader = members[0]
            if leader["symbol"] not in final_list:
                cluster_leaders.append(leader)
        
        # Sort cluster leaders by global quality score and add
        cluster_leaders.sort(key=lambda x: x["quality_score"], reverse=True)
        for leader in cluster_leaders:
            final_list.append(leader["symbol"])
            if len(final_list) >= int(cfg.get("MP_MAX_SYMBOLS", 8)): break
            
    except Exception as e:
        _logger.warning(f"Phase 3 Clustering failed ({e}). Falling back to quality-score ranking.")
        final_candidates.sort(key=lambda x: x["quality_score"], reverse=True)
        final_list = list(dict.fromkeys(list(anchors) + [r["symbol"] for r in final_candidates]))[:int(cfg.get("MP_MAX_SYMBOLS", 8))]

    _logger.info(f"Refinement complete: {len(final_list)} orthogonal symbols selected: {final_list}")
    
    # persist to config
    from .universe_screener_futures import update_futures_config_file
    update_futures_config_file(final_list)
    
    return final_list, len(final_candidates)

def update_futures_config_file(symbols: List[str]) -> None:
    config_path = Path("config/opt_config.py")
    if not config_path.exists(): return
    content = config_path.read_text(encoding="utf-8")
    pattern = r"FUTURES_SYMBOLS(?::\s*List\[str\])?\s*=\s*\[.*?\]"
    new_block = "FUTURES_SYMBOLS: List[str] = [\n"
    for s in symbols: new_block += f'    "{s}",\n'
    new_block += "]"
    new_content = re.sub(pattern, new_block, content, count=1, flags=re.DOTALL)
    config_path.write_text(new_content, encoding="utf-8")
