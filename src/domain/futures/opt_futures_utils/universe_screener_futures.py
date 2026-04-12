"""
Advanced 4-Phase Universe Screener for Binance Futures.
Focus: Conditional Anchor logic, geometric growth, and orthogonal diversification.
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
    OPT_FUTURES_CONFIG,
)
from config.settings import FUTURES_DATA_DIR, SLIPPAGE_RATE
from src.domain.futures.data_collector import DataCollector
from src.domain.futures.funding_utils import merge_funding_into_ohlcv

_logger: logging.Logger = logging.getLogger("universe_screener_futures")

# --- Phase 2: Structural Trendiness Utilities ---

def _calculate_hurst_exponent(ts: np.ndarray) -> float:
    if ts.size < 100:
        return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(reg[0] * 2.0)
    except:
        return 0.5

# --- Phase 1: Capacity & Slippage Utilities ---

def _calculate_amihud_illiquidity(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 20:
        return 999.0
    returns = df["close"].pct_change().abs()
    notional_vol = df["volume"] * df["close"]
    illiq = returns / notional_vol.replace(0, 1e-9)
    return float(illiq.tail(180).mean() * 1e6)

# --- Core Screening Worker ---

def _screen_worker_v2(
    sym: str,
    tf: str,
    fetch_start: str,
    end_date: str,
    cfg: Dict[str, Any],
    data_dir: Path,
) -> Dict[str, Any] | None:
    from src.domain.futures.data_collector import DataCollector
    collector = DataCollector()
    
    min_bars = int(cfg.get("MIN_HISTORY_BARS", 2000))
    
    # [FIX] History Pruning: Check listed period before download
    meta = collector._load_meta()
    mk = collector._meta_key(sym, tf)
    if mk in meta and isinstance(meta[mk], dict):
        earliest = meta[mk].get("earliest_available")
        if earliest:
            try:
                delta = pd.to_datetime(end_date) - pd.to_datetime(earliest)
                bars_per_day = {"1h": 24, "4h": 6, "1d": 1}.get(tf, 6)
                if delta.days * bars_per_day < min_bars:
                    return None
            except: pass

    try:
        collector.ensure_funding_data(sym, fetch_start, end_date)
        df = collector.collect_and_save(sym, tf, fetch_start, end_date)
        df = merge_funding_into_ohlcv(sym, df, data_dir)
    except:
        return None

    if df is None or len(df) < min_bars:
        return None

    # --- Phase 1: Dynamic Liquidity ---
    notional_vol = (df["volume"] * df["close"]).to_numpy()
    adv_90d = np.mean(notional_vol[-540:]) if len(notional_vol) >= 540 else np.mean(notional_vol)
    adv_7d = np.mean(notional_vol[-42:]) if len(notional_vol) >= 42 else adv_90d
    if adv_7d < adv_90d * 0.4: return None # Prune zombie coins
    
    amihud = _calculate_amihud_illiquidity(df)
    
    # --- Phase 2: Structural Trendiness ---
    close_arr = df["close"].to_numpy()
    hurst = _calculate_hurst_exponent(close_arr)
    last_price = close_arr[-1]
    
    # [EXPLOSIVE GROWTH] Add directional momentum factor (180d) to ensure we aren't picking structural downtrends
    lookback_180 = min(len(close_arr), 1080) # 180 days in 4h tf
    mom_180d = (close_arr[-1] / close_arr[-lookback_180]) - 1.0 if len(close_arr) >= lookback_180 else 0.0
    
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close_arr, 1)), np.abs(low - np.roll(close_arr, 1))))
    atr_pct = (np.mean(tr[-14:]) / last_price) * 100.0

    # --- Phase 4: Funding Stats ---
    funding = df["funding_rate"].to_numpy()
    mean_funding = np.mean(funding[-180:]) if len(funding) >= 180 else 0.0
    
    # [EXPLOSIVE GROWTH] Quality Score now rewards trendiness + positive bias + liquidity
    # We penalize negative momentum to avoid "smooth downtrend" traps.
    bias_factor = np.clip(1.0 + mom_180d, 0.5, 2.0)
    quality_score = (hurst * 10.0 * bias_factor) / (np.log1p(amihud) + 1.0)

    return {
        "symbol": sym,
        "adv": adv_7d,
        "amihud": amihud,
        "hurst": hurst,
        "atr_pct": atr_pct,
        "mom_180d": mom_180d,
        "quality_score": quality_score,
        "returns": df["close"].pct_change().tail(500)
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
    n_workers_override: int | None = None,
) -> Tuple[List[str], int]:
    """
    Ultimate 4-Phase Screener with 'Conditional Anchor' System.
    Logical bias-free version (No Point-in-time filters).
    """
    dd = data_dir if data_dir is not None else FUTURES_DATA_DIR
    anchors = set(FUTURES_ANCHOR_SYMBOLS)
    
    # 1. Full Market Scan
    _logger.info("Starting Advanced Universe Screening (Bias-Free 10/10 Architecture)...")
    try:
        tickers = collector.client.exchange.fetch_tickers()
        valid_tickers = []
        min_adv_ticker = float(cfg["ADV_MIN_USDT_DAY"]) * 0.3
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

    # 2. Parallel History Screening
    if n_workers_override is None:
        n_workers = max(1, min(int(os.cpu_count() or 4), 8))
    else:
        n_workers = max(1, min(int(n_workers_override), int(os.cpu_count() or 4), 8))
    worker_fn = partial(_screen_worker_v2, tf=tf, fetch_start=fetch_start, end_date=end_date, cfg=cfg, data_dir=dd)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        raw_results = list(tqdm(pool.map(worker_fn, candidate_pool), total=len(candidate_pool), desc="[Universe Gates]"))
    
    passed_rows = [r for r in raw_results if r is not None]
    
    # --- PHASE 1 & 2: NO FREE PASS GATING ---
    # Everyone (including Anchors) must qualify on Hurst and Volatility.
    final_candidates = []
    # [EXPLOSIVE GROWTH] Remove ATR_MAX cap. Volatility is an opportunity, not a risk to be filtered out at screener level.
    atr_min = 2.0
    # [EXPLOSIVE GROWTH] Slightly lower Hurst floor to include budding trends (0.50)
    hurst_floor = 0.50
    amihud_limit = float(cfg.get("MAX_AMIHUD_ILLIQUIDITY", 10.0))

    for r in passed_rows:
        if r["hurst"] < hurst_floor: continue 
        if r["atr_pct"] < atr_min: continue
        if r["amihud"] > amihud_limit: continue
        final_candidates.append(r)

    _logger.info(f"Phase 1 & 2 complete: {len(final_candidates)} symbols passed.")
    if not final_candidates: return list(anchors), 0

    # --- PHASE 3: SEEDED PLAYER PRIORITY & CLUSTERING ---
    final_list: List[str] = []
    
    # 1. VIP Allocation: Qualified Anchors take their seats first.
    qualified_anchors = [r for r in final_candidates if r["symbol"] in anchors]
    for qa in qualified_anchors:
        final_list.append(qa["symbol"])
        _logger.info(f"  [ANCHOR OK] {qa['symbol']} qualified and prioritized.")

    # 2. Alpha Filling: Fill remaining slots with orthogonal Alts.
    others = [r for r in final_candidates if r["symbol"] not in anchors]
    max_slots = int(cfg.get("MP_MAX_SYMBOLS", 8))
    
    if len(final_list) < max_slots and others:
        try:
            from sklearn.cluster import KMeans
            remaining_slots = max_slots - len(final_list)
            
            # Cluster the non-anchor candidates
            returns_df = pd.DataFrame({r["symbol"]: r["returns"] for r in others}).fillna(0)
            
            # Correlation on stress periods if possible
            if "BTC/USDT" in [r["symbol"] for r in final_candidates]:
                btc_rets = [r for r in final_candidates if r["symbol"] == "BTC/USDT"][0]["returns"]
                stress_returns = returns_df.loc[btc_rets < btc_rets.quantile(0.20)]
            else:
                stress_returns = returns_df
            
            n_clusters = min(len(others), remaining_slots + 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(1.0 - stress_returns.corr().fillna(0))
            
            # Group others by clusters
            buckets: Dict[int, List[Dict]] = {}
            for i, r in enumerate(others):
                cid = clusters[i]
                if cid not in buckets: buckets[cid] = []
                buckets[cid].append(r)
            
            # Pick best from each cluster
            leaders = []
            for members in buckets.values():
                members.sort(key=lambda x: x["quality_score"], reverse=True)
                leaders.append(members[0])
            
            # Add leaders by quality until slots full
            leaders.sort(key=lambda x: x["quality_score"], reverse=True)
            for leader in leaders:
                if len(final_list) >= max_slots: break
                final_list.append(leader["symbol"])
                
        except Exception as e:
            _logger.warning(f"Clustering failed ({e}), falling back to quality sort.")
            others.sort(key=lambda x: x["quality_score"], reverse=True)
            for r in others:
                if len(final_list) >= max_slots: break
                final_list.append(r["symbol"])

    _logger.info(f"Refinement complete: {len(final_list)} symbols selected: {final_list}")
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
