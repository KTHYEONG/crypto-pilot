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
    funding = df["funding_rate"].to_numpy() if "funding_rate" in df.columns else np.zeros(len(df))
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

import json
import random
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.domain.futures.opt_futures_utils.metrics import calc_profit_factor_from_pnl
from src.domain.futures.opt_futures_utils.oos_evaluator import evaluate_symbol_fold
from src.domain.futures.strategies_futures import FuturesPipelineStrategy
from src.core.optimization.opt_utils import compute_segment_merge_index


def calculate_microstructure_vector(df: pd.DataFrame) -> np.ndarray:
    """
    Computes a 3D microstructure property vector: [Hurst, Kurtosis, Mean_ATR_Pct].
    """
    if df.empty or len(df) < 100:
        return np.array([0.5, 0.0, 3.0])
    
    close = df["close"].to_numpy()
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    
    # 1. Hurst Exponent
    hurst = _calculate_hurst_exponent(close)
    
    # 2. Kurtosis (Fat-tailedness)
    rets = np.diff(np.log(close))
    kurt = stats.kurtosis(rets, fisher=True)
    
    # 3. Mean ATR Pct (Volatility Scale)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    # Valid index excluding the roll wrap-around
    atr_pct = (tr[1:] / close[1:]) * 100.0
    mean_atr_pct = np.mean(atr_pct[-500:]) if len(atr_pct) >= 500 else np.mean(atr_pct)
    
    return np.array([hurst, kurt, mean_atr_pct])


def _calculate_stress_correlation(
    base_rets: pd.Series, 
    alt_rets: pd.Series, 
    stress_threshold: float
) -> float:
    """
    Computes Pearson correlation specifically on dates where base_rets < stress_threshold.
    """
    stress_mask = base_rets < stress_threshold
    s_base = base_rets[stress_mask]
    s_alt = alt_rets[stress_mask]
    
    if len(s_base) < 10:
        return float(base_rets.corr(alt_rets))  # Fallback
    
    return float(s_base.corr(s_alt))


def _calculate_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def screen_symbol_refinement_futures(
    broad_candidates: List[str],
    winning_signal_type: str,
    is_end_date: str,
    *,
    symbol_dfs_4h: Dict[str, pd.DataFrame],
    daily_dfs: Dict[str, pd.DataFrame],
    phase_b_params: Optional[Dict[str, Any]] = None,
    anchor_symbols: Optional[List[str]] = None,
) -> bool:
    """
    MHRH (Microstructure-Homogeneous, Returns-Heterogeneous) Refinement.
    1. Homogeneity: Microstructure vector similarity to BTC.
    2. Heterogeneity: Stress-period correlation with BTC (Non-parametric P15).
    """
    from config.opt_config import FUTURES_ANCHOR_SYMBOLS, FUTURES_SCREENER_CONFIG

    cfg = FUTURES_SCREENER_CONFIG
    mp_max = int(cfg.get("MP_MAX_SYMBOLS", 5))
    
    anchors = list(anchor_symbols) if anchor_symbols is not None else list(FUTURES_ANCHOR_SYMBOLS)
    anchor_set = set(anchors)
    
    if "BTC/USDT" not in symbol_dfs_4h:
        _logger.error("Phase C: BTC/USDT data missing. Cannot calculate MHRH.")
        return False

    # 1. Base Reference (BTC)
    btc_df = _slice_df_to_is(symbol_dfs_4h["BTC/USDT"], "1970-01-01", is_end_date)
    if btc_df.empty or len(btc_df) < 100:
        raw_btc_len = len(symbol_dfs_4h.get("BTC/USDT", []))
        _logger.error(
            f"Phase C: BTC/USDT df is empty or too small (len={len(btc_df)}). "
            f"Raw df len={raw_btc_len}. Available symbols: {list(symbol_dfs_4h.keys())}"
        )
        return False
        
    btc_vec = calculate_microstructure_vector(btc_df)
    btc_rets_full = btc_df["close"].pct_change().dropna()
    if btc_rets_full.empty:
        _logger.error("Phase C: BTC/USDT returns calculation failed (empty).")
        return False
        
    stress_threshold = btc_rets_full.quantile(0.15)
    _logger.info("MHRH Reference [BTC]: Vec=%s, P15_Stress_Thr=%.4f", btc_vec, stress_threshold)

    # 2. Extract Vectors & Returns for all valid candidates
    stats_map: Dict[str, Dict[str, Any]] = {}
    all_targets = list(dict.fromkeys(broad_candidates + anchors))
    
    for sym in all_targets:
        df4 = symbol_dfs_4h.get(sym)
        if df4 is None or df4.empty: continue
        
        is_df = _slice_df_to_is(df4, "1970-01-01", is_end_date)
        if len(is_df) < 300: continue
        
        vec = calculate_microstructure_vector(is_df)
        rets = is_df["close"].pct_change()
        
        # ADV calculation for capacity gating
        tail_vol = is_df.tail(180)
        # 6 * 4h = 1 day (approximation)
        adv = float((tail_vol["close"] * tail_vol["volume"]).median() * 6)
        
        stats_map[sym] = {
            "vector": vec,
            "rets": rets,
            "adv": adv,
            "cosine": _calculate_cosine_similarity(btc_vec, vec)
        }

    # 3. MHRH Scoring
    # [Homogeneity] Filter candidates similar to BTC
    adv_min = float(cfg.get("ADV_MIN_USDT_DAY", 50_000_000.0))
    filtered_dyn = []
    for sym, res in stats_map.items():
        if sym in anchor_set: continue
        if res["cosine"] > 0.80 and res["adv"] >= adv_min:
            filtered_dyn.append(sym)

    _logger.info("Homogeneity Filter: %d dynamic symbols pass (Cosine > 0.80)", len(filtered_dyn))

    # 4. [Heterogeneity] Minimal Tail correlation with BTC
    final_pool_stats = []
    for sym in filtered_dyn:
        res = stats_map[sym]
        merged = pd.concat([btc_rets_full, res["rets"]], axis=1, join="inner").dropna()
        if len(merged) < 50: continue
        
        s_corr = _calculate_stress_correlation(merged.iloc[:, 0], merged.iloc[:, 1], stress_threshold)
        # Logic: High similarity (Cosine) + Low stress correlation (1 - s_corr)
        score = res["cosine"] + (1.0 - s_corr)
        final_pool_stats.append({
            "symbol": sym, "score": score, "s_corr": s_corr, "cosine": res["cosine"]
        })

    final_pool_stats.sort(key=lambda x: x["score"], reverse=True)
    
    # 5. Assemble final list (Anchors + Top-K Dynamics)
    final_symbols = [a for a in anchors if a in stats_map]
    for item in final_pool_stats:
        if len(final_symbols) >= mp_max: break
        final_symbols.append(item["symbol"])

    if not final_symbols:
        _logger.error("Phase C MHRH: no symbols selected.")
        return False

    _logger.info("Final MHRH Symbols (%d): %s", len(final_symbols), final_symbols)
    update_futures_config_file(final_symbols)
    return True



def _slice_df_to_is(df: pd.DataFrame, is_start: str, is_end: str) -> pd.DataFrame:
    if df.empty or "datetime" not in df.columns:
        return df.iloc[0:0].copy()
    is_s = pd.to_datetime(is_start)
    is_e = pd.to_datetime(is_end)
    dt = df["datetime"]
    if dt.dt.tz is not None:
        if is_s.tzinfo is None: is_s = is_s.tz_localize(dt.dt.tz)
        if is_e.tzinfo is None: is_e = is_e.tz_localize(dt.dt.tz)
    mask = (dt >= is_s) & (dt < is_e)
    return df.loc[mask].reset_index(drop=True)


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
    Phase A: Lightweight Market-Wide Scan.
    In V5 MHRH Architecture, we only perform an ADV (Volume) check here. 
    Strict filtering, structural evaluation, and parallel data gathering are deferred to Phase C (MHRH).
    """
    anchors = set(FUTURES_ANCHOR_SYMBOLS)
    _logger.info("Phase A: Lightweight Market Scan (ADV Filter Only)...")
    
    try:
        tickers = collector.client.exchange.fetch_tickers()
        valid_tickers = []
        # Allow a slight buffer for ADV check
        min_adv_ticker = float(cfg.get("ADV_MIN_USDT_DAY", 50000000.0)) * 0.3
        
        pool_set = set(candidate_pool) if candidate_pool else None
        
        for sym, t in tickers.items():
            norm_sym = sym.split(":")[0]
            if pool_set and norm_sym not in pool_set: continue
            if not (sym.endswith("/USDT") or sym.endswith("/USDT:USDT")): continue
            
            vol = float(t.get("quoteVolume") or 0.0)
            if norm_sym in anchors or vol >= min_adv_ticker:
                valid_tickers.append({"symbol": norm_sym, "vol": vol})
                
    except Exception as e:
        _logger.warning(f"Ticker pre-filter failed: {e}")
        return list(anchors), 0

    # Sort by recent volume
    valid_tickers.sort(key=lambda x: x["vol"], reverse=True)
    
    # Extract unique symbols
    final_list = []
    seen = set()
    for item in valid_tickers:
        if item["symbol"] not in seen:
            seen.add(item["symbol"])
            final_list.append(item["symbol"])
            
    pool_k = int(cfg.get("BROAD_POOL_K", 60))
    if len(final_list) > pool_k:
        final_list = final_list[:pool_k]

    _logger.info(f"Phase A Scan complete: {len(final_list)} broad candidates sent to Phase C (e.g. {final_list[:5]}...)")
    return final_list, len(final_list)

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
