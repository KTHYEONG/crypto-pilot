"""
Advanced 4-Phase Universe Screener for Binance Futures.
Focus: Conditional Anchor logic, geometric growth, and orthogonal diversification.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy import stats

from config.opt_config import FUTURES_ANCHOR_SYMBOLS
from src.domain.futures.data_collector import DataCollector
from src.domain.futures.funding_utils import merge_funding_into_ohlcv

# Project Root Setup
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


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
    except Exception:
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
            except Exception:
                _logger.debug("History pruning check failed for %s", sym)

    try:
        collector.ensure_funding_data(sym, fetch_start, end_date)
        df = collector.collect_and_save(sym, tf, fetch_start, end_date)
        df = merge_funding_into_ohlcv(sym, df, data_dir)
    except Exception:
        return None

    if df is None or len(df) < min_bars:
        return None

    # --- Phase 1: Dynamic Liquidity ---
    notional_vol = (df["volume"] * df["close"]).to_numpy()
    adv_90d = np.mean(notional_vol[-540:]) if len(notional_vol) >= 540 else np.mean(notional_vol)
    adv_7d = np.mean(notional_vol[-42:]) if len(notional_vol) >= 42 else adv_90d
    if adv_7d < adv_90d * 0.4:
        return None  # Prune zombie coins
    
    amihud = _calculate_amihud_illiquidity(df)
    
    # --- Phase 2: Structural Trendiness ---
    close_arr = df["close"].to_numpy()
    hurst = _calculate_hurst_exponent(close_arr)
    last_price = close_arr[-1]
    
    # [EXPLOSIVE GROWTH] Add directional momentum factor (180d)
    # Ensure we aren't picking structural downtrends
    lookback_180 = min(len(close_arr), 1080)  # 180 days in 4h tf
    if len(close_arr) >= lookback_180:
        mom_180d = (close_arr[-1] / close_arr[-lookback_180]) - 1.0
    else:
        mom_180d = 0.0
    
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    side_diff = np.abs(high - np.roll(close_arr, 1))
    side_diff_low = np.abs(low - np.roll(close_arr, 1))
    tr = np.maximum(high - low, np.maximum(side_diff, side_diff_low))
    atr_pct = (np.mean(tr[-14:]) / last_price) * 100.0

    # --- Phase 4: Funding Stats ---
    funding = df["funding_rate"].to_numpy() if "funding_rate" in df.columns else np.zeros(len(df))
    np.mean(funding[-180:]) if len(funding) >= 180 else 0.0
    
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
    side_diff = np.abs(high - np.roll(close, 1))
    side_diff_low = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high - low, np.maximum(side_diff, side_diff_low))
    # Valid index excluding the roll wrap-around
    atr_pct = (tr[1:] / close[1:]) * 100.0
    mean_atr_pct = np.mean(atr_pct[-500:]) if len(atr_pct) >= 500 else np.mean(atr_pct)
    
    # 4. Cleanup and return
    vec = np.array([hurst, kurt, mean_atr_pct])
    return cast(np.ndarray, np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64))


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


def _calculate_standardized_affinity(
    pop_vectors: np.ndarray, 
    ref_vec: np.ndarray, 
    target_vec: np.ndarray
) -> float:
    """
    Computes affinity based on Standardized Euclidean Distance.
    Affinity = 1 / (1 + distance)
    """
    # 1. Calculate population stats for standardization
    # Use nan-aware functions in case some vectors contain nans
    means = np.nanmean(pop_vectors, axis=0)
    stds = np.nanstd(pop_vectors, axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)
    
    # Fill any remaining nans in means/stds with defaults
    means = np.nan_to_num(means)
    stds = np.nan_to_num(stds, nan=1.0)
    
    # 2. Standardize reference and target
    z_ref = (ref_vec - means) / stds
    z_target = (target_vec - means) / stds
    
    # 3. Euclidean Distance in standardized space
    # z_target might still have nans if target_vec had a nan
    diff = np.nan_to_num(z_ref - z_target)
    dist = float(np.linalg.norm(diff))
    
    return 1.0 / (1.0 + dist)


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
    1. Homogeneity: Standardized Affinity to BTC.
    2. Heterogeneity: Stress-period correlation with BTC.
    3. Growth: Directional momentum check.
    """
    from config.opt_config import FUTURES_ANCHOR_SYMBOLS, FUTURES_SCREENER_CONFIG

    cfg = FUTURES_SCREENER_CONFIG
    
    anchors = list(anchor_symbols) if anchor_symbols is not None else list(FUTURES_ANCHOR_SYMBOLS)
    anchor_set = set(anchors)
    
    if "BTC/USDT" not in symbol_dfs_4h:
        _logger.error("Phase B: BTC/USDT data missing. Cannot calculate MHRH.")
        return False

    # 1. Base Reference (BTC)
    btc_df = _slice_df_to_is(symbol_dfs_4h["BTC/USDT"], "1970-01-01", is_end_date)
    if btc_df.empty or len(btc_df) < 100:
        _logger.error("Phase B: BTC/USDT df is empty or too small.")
        return False
        
    btc_vec = calculate_microstructure_vector(btc_df)
    btc_rets_full = btc_df["close"].pct_change().dropna()
    if btc_rets_full.empty:
        _logger.error("Phase B: BTC/USDT returns calculation failed.")
        return False
        
    stress_threshold = btc_rets_full.quantile(0.15)
    
    _logger.info("=" * 70)
    _logger.info("[PHASE B] MHRH Statistical Refinement (Ref: BTC/USDT)")
    _logger.info("-" * 70)
    _logger.info("  - Microstructure Vector : [Hurst: %.4f, Kurt: %.4f, ATR%%: %.4f]", 
                 btc_vec[0], btc_vec[1], btc_vec[2])
    _logger.info("  - P15 Stress Threshold  : %.4f", stress_threshold)
    _logger.info("-" * 70)

    # 2. Process all candidates
    stats_map: Dict[str, Dict[str, Any]] = {}
    all_targets = list(dict.fromkeys(broad_candidates + anchors))
    all_vectors: List[np.ndarray] = []
    
    pruned_count = 0
    for sym in all_targets:
        df4 = symbol_dfs_4h.get(sym)
        if df4 is None or df4.empty:
            continue
        
        is_df = _slice_df_to_is(df4, "1970-01-01", is_end_date)
        if len(is_df) < 300:
            continue
        
        # [HARD CUTOFF] 180d Momentum Check
        close_arr = is_df["close"].to_numpy()
        lookback_180 = min(len(close_arr), 1080)
        mom_180d = 0.0
        if len(close_arr) >= 100:
            mom_180d = float((close_arr[-1] / close_arr[-lookback_180]) - 1.0)
        
        if sym not in anchor_set and mom_180d < -0.50:
            pruned_count += 1
            continue
            
        vec = calculate_microstructure_vector(is_df)
        rets = is_df["close"].pct_change()
        tail_vol = is_df.tail(180)
        adv = float((tail_vol["close"] * tail_vol["volume"]).median() * 6)
        
        stats_map[sym] = {
            "vector": vec, "rets": rets, "adv": adv, "mom_180d": mom_180d
        }
        all_vectors.append(vec)

    if not all_vectors or "BTC/USDT" not in stats_map:
        _logger.error("Phase B: No valid candidates or BTC missing.")
        return False

    pop_vectors = np.stack(all_vectors)
    adv_min = float(cfg.get("ADV_MIN_USDT_DAY", 50_000_000.0))
    
    # 3. Homogeneity & Heterogeneity Scoring
    final_pool_stats = []
    _logger.info("[1/2] Calculating Standardized Affinity & Stress Correlation...")
    
    for sym, res in stats_map.items():
        if sym in anchor_set:
            continue
        if res["adv"] < adv_min:
            continue
        
        affinity = _calculate_standardized_affinity(pop_vectors, btc_vec, res["vector"])
        if affinity < 0.35:
            continue # Relaxed from 0.40 slightly for initial testing
        
        merged = pd.concat([btc_rets_full, res["rets"]], axis=1, join="inner").dropna()
        if len(merged) < 50:
            continue
        s_corr = _calculate_stress_correlation(
            merged.iloc[:, 0], merged.iloc[:, 1], stress_threshold
        )
        
        # [MULTIPLICATIVE SCORE]
        # Score = Affinity * (1 - s_corr) * max(1.0, 1 + mom)
        score = affinity * (1.0 - s_corr) * max(1.0, 1.0 + res["mom_180d"])
        
        final_pool_stats.append({
            "symbol": sym, "score": score, "s_corr": s_corr, "affinity": affinity
        })

    final_pool_stats.sort(key=lambda x: x["score"], reverse=True)
    
    # 5. [METHOD B] Universe Assembly & Greedy Marginal Growth Test
    # Start with anchors as the baseline portfolio
    final_symbols = [a for a in anchors if a in stats_map]
    
    if not final_symbols:
        _logger.error("Phase B MHRH: no symbols selected (even anchors missing).")
        return False

    # Calculate Baseline Geometric Growth (G = E[R] - 0.5*Var[R])
    port_rets_df = pd.concat([stats_map[s]["rets"] for s in final_symbols], axis=1).dropna()
    if port_rets_df.empty:
        _logger.error("Phase B: Anchor returns data insufficient for alignment.")
        return False
        
    avg_rets = port_rets_df.mean(axis=1)
    current_g = float(avg_rets.mean() - 0.5 * avg_rets.var())
    
    _logger.info(f"- Pruning: {pruned_count} symbols removed (Momentum < -50%)")
    _logger.info(f"- Screening: {len(final_pool_stats)} valid candidates analyzed.")
    _logger.info(f"- Baseline G: {current_g:.8f} (Anchors: {final_symbols})")
    _logger.info("- Greedy Search Result:")

    # Greedy Selection: Add candidate if it increases Portfolio G
    abs_max_limit = 12
    rejected_set = []
    
    for item in final_pool_stats:
        if len(final_symbols) >= abs_max_limit:
            break
            
        sym = item["symbol"]
        test_df = pd.concat([port_rets_df, stats_map[sym]["rets"]], axis=1, join="inner").dropna()
        if len(test_df) < 200:
            continue
            
        test_avg_rets = test_df.mean(axis=1)
        new_g = float(test_avg_rets.mean() - 0.5 * test_avg_rets.var())
        
        if new_g > current_g:
            improvement = new_g - current_g
            _logger.info(f"  [+] {sym}: G improved to {new_g:.8f} (+{improvement:.8f})")
            current_g = new_g
            final_symbols.append(sym)
            port_rets_df = test_df # Update baseline
        else:
            rejected_set.append(sym)

    if rejected_set:
        _logger.info(f"- Rejected: {len(rejected_set)} symbols failed to improve G")

    _logger.info(
        f"- Result: {len(final_symbols)} symbols selected: {final_symbols} "
        f"(Final G: {current_g:.8f})"
    )
    _logger.info("=" * 70)
    
    update_futures_config_file(final_symbols)
    return True



def _slice_df_to_is(df: pd.DataFrame, is_start: str, is_end: str) -> pd.DataFrame:
    if df.empty or "datetime" not in df.columns:
        return df.iloc[0:0].copy()
    is_s = pd.to_datetime(is_start)
    is_e = pd.to_datetime(is_end)
    dt = df["datetime"]
    if dt.dt.tz is not None:
        if is_s.tzinfo is None:
            is_s = is_s.tz_localize(dt.dt.tz)
        if is_e.tzinfo is None:
            is_e = is_e.tz_localize(dt.dt.tz)
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
    Strict filtering, structural evaluation and gathering are deferred to Phase B.
    """
    anchors = set(FUTURES_ANCHOR_SYMBOLS)
    _logger.info("Phase A: Market Scan...")
    
    try:
        tickers = collector.client.exchange.fetch_tickers()
        valid_tickers = []
        # Allow a slight buffer for ADV check
        min_adv_ticker = float(cfg.get("ADV_MIN_USDT_DAY", 50000000.0)) * 0.3
        
        pool_set = set(candidate_pool) if candidate_pool else None
        
        for sym, t in tickers.items():
            norm_sym = sym.split(":")[0]
            if pool_set and norm_sym not in pool_set:
                continue
            if not (sym.endswith("/USDT") or sym.endswith("/USDT:USDT")):
                continue
            
            vol = float(t.get("quoteVolume") or 0.0)
            if norm_sym in anchors or vol >= min_adv_ticker:
                valid_tickers.append({"symbol": norm_sym, "vol": vol})
                
    except Exception as e:
        import traceback
        _logger.debug(f"Ticker pre-filter failed: {e}")
        _logger.debug(traceback.format_exc())
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

    _logger.info(f"Phase A Scan complete: SUCCESS ({len(final_list)} candidates sent to Phase B)")
    return final_list, len(final_list)

def update_futures_config_file(symbols: List[str]) -> None:
    config_path = Path("config/opt_config.py")
    if not config_path.exists():
        return
    content = config_path.read_text(encoding="utf-8")
    pattern = r"FUTURES_SYMBOLS(?::\s*List\[str\])?\s*=\s*\[.*?\]"
    new_block = "FUTURES_SYMBOLS: List[str] = [\n"
    for s in symbols:
        new_block += f'    "{s}",\n'
    new_block += "]"
    new_content = re.sub(pattern, new_block, content, count=1, flags=re.DOTALL)
    config_path.write_text(new_content, encoding="utf-8")
