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

import json
import random
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.domain.futures.opt_futures_utils.metrics import calc_profit_factor_from_pnl
from src.domain.futures.opt_futures_utils.oos_evaluator import evaluate_symbol_fold
from src.domain.futures.strategies_futures import FuturesPipelineStrategy
from src.core.optimization.opt_utils import compute_segment_merge_index


def _indices_is_last_year_futures(df: pd.DataFrame, is_end_date: pd.Timestamp | str) -> tuple[int, int]:
    """Last ~365 days of in-sample rows strictly before is_end_date (OOS boundary)."""
    if df.empty or "datetime" not in df.columns:
        return 0, 0
    dt = df["datetime"]
    is_end = pd.to_datetime(is_end_date)
    if dt.dt.tz is not None and is_end.tzinfo is None:
        is_end = is_end.tz_localize(dt.dt.tz)
    elif dt.dt.tz is None and is_end.tzinfo is not None:
        is_end = is_end.tz_localize(None)

    mask_before_oos = dt < is_end
    if not mask_before_oos.any():
        return 0, 0
    last_is_pos = int(mask_before_oos.to_numpy().nonzero()[0][-1]) + 1
    end_ts = dt.iloc[last_is_pos - 1]
    start_ts = end_ts - pd.Timedelta(days=365)
    mask_win = (dt >= start_ts) & (dt < is_end)
    if not mask_win.any():
        return 0, last_is_pos
    is_start_idx = int(mask_win.to_numpy().argmax())
    return is_start_idx, last_is_pos


def _equity_simple_returns_futures(eq: np.ndarray, dt: pd.Series) -> pd.Series:
    eq = np.asarray(eq, dtype=np.float64)
    if len(eq) < 3 or len(dt) != len(eq):
        return pd.Series(dtype=np.float64)
    r = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    idx = pd.to_datetime(dt.iloc[1:].values)
    return pd.Series(r, index=idx, dtype=np.float64)


def _run_mini_backtest_window_futures(
    sym: str,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    params: dict[str, Any],
    test_start: int,
    test_end: int,
) -> tuple[float, int, float, float, pd.Series]:
    """Single-symbol mini backtest for screening."""
    strat = FuturesPipelineStrategy(name=f"Screener_{sym}", params=params)
    merge_idx = compute_segment_merge_index(df_4h, df_1d)
    
    # evaluate_symbol_fold returns:
    # cagr, ret_pct, mdd, trades, wins, pf, long_c, short_c, eq_curve, fpaid, gross_ret
    res = evaluate_symbol_fold(
        strat, params, sym, "4h", df_4h, df_1d, merge_idx, 
        None, test_start, test_end
    )
    cagr, _, mdd, n_trades, _, pf, _, _, eq_curve, _, _ = res
    
    # Simple return series from equity curve for mRMR
    if eq_curve is not None and len(eq_curve) > 1:
        rets = np.diff(eq_curve) / np.maximum(eq_curve[:-1], 1e-12)
        # Use dummy dates for now as we just need correlation
        ret_ser = pd.Series(rets)
    else:
        ret_ser = pd.Series(dtype=np.float64)

    rel = pf * float(np.log1p(float(n_trades)))
    return pf, n_trades, cagr, rel, ret_ser


def screen_by_strategy_fit_futures(
    vol_df: pd.DataFrame,
    symbol_dfs: Dict[str, pd.DataFrame],
    daily_dfs: Dict[str, pd.DataFrame],
    fixed_params: dict[str, Any],
    *,
    is_end_date: pd.Timestamp | str,
    min_trades: int,
    min_pf: float,
    min_cagr_pct: float,
    signal_type: str | None = None,
) -> tuple[pd.DataFrame, Dict[str, pd.Series]]:
    rows: list[dict[str, float | str]] = []
    ret_map: Dict[str, pd.Series] = {}
    fp_base = dict(fixed_params)
    if signal_type is not None:
        fp_base["SIGNAL_TYPE"] = str(signal_type)

    for _, row in tqdm(vol_df.iterrows(), total=len(vol_df), desc="Strategy mini-BT (Futures)"):
        sym = str(row["symbol"])
        df4 = symbol_dfs.get(sym)
        d1 = daily_dfs.get(sym)
        if df4 is None or d1 is None:
            continue
        ts, te = _indices_is_last_year_futures(df4, is_end_date)
        if te <= ts + 10:
            continue

        pf, n_tr, cagr, rel, ret_ser = _run_mini_backtest_window_futures(sym, df4, d1, fp_base, ts, te)
        if n_tr < int(min_trades) or pf < float(min_pf) or cagr <= float(min_cagr_pct):
            continue
            
        rows.append(
            {
                "symbol": sym,
                "adv": float(row["adv"]),
                "pf": pf,
                "n_trades": float(n_tr),
                "cagr_pct": cagr,
                "relevance": rel,
            }
        )
        ret_map[sym] = ret_ser

    return pd.DataFrame(rows), ret_map


def marchenko_pastur_n_factors(
    returns_aligned: pd.DataFrame,
    *,
    min_n: int,
    max_n: int,
) -> int:
    if returns_aligned.empty or returns_aligned.shape[1] < 2:
        return int(min_n)

    rets = returns_aligned.dropna(how="any")
    t_obs = len(rets)
    n_dim = int(returns_aligned.shape[1])
    if t_obs < 2 or n_dim < 1:
        return int(min_n)
    if rets.shape[0] < n_dim + 2:
        return int(min_n)

    corr = rets.corr().to_numpy()
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.sort(eigvals)[::-1]

    gamma = n_dim / float(t_obs)
    lambda_plus = (1.0 + np.sqrt(gamma)) ** 2
    tol = 1e-9
    signal_count = int(np.sum(eigvals > lambda_plus * (1.0 + tol)))

    if signal_count <= 0:
        return int(min_n)

    return int(max(min_n, min(max_n, signal_count)))


def select_by_mrmr(
    candidates_df: pd.DataFrame,
    strategy_returns: Dict[str, pd.Series],
    n_select: int,
) -> List[str]:
    """Greedy mRMR on strategy equity returns: score = relevance - mean(|corr| to selected)."""
    if candidates_df.empty:
        return []

    work = candidates_df.copy()
    if "relevance" not in work.columns:
        return []

    symbols = [str(s) for s in work["symbol"].tolist()]
    work = work.set_index("symbol", drop=False)

    if len(symbols) <= n_select:
        return work.sort_values("relevance", ascending=False)["symbol"].tolist()

    # Align all return series
    valid_rets = {s: strategy_returns[s] for s in symbols if not strategy_returns[s].empty}
    if not valid_rets:
        return work.sort_values("relevance", ascending=False)["symbol"].tolist()
        
    rets = pd.concat(valid_rets, axis=1).dropna(how='any')
    if rets.shape[0] < 10:
        return work.sort_values("relevance", ascending=False)["symbol"].tolist()

    corr = rets.corr().abs()
    relevance = {s: float(work.loc[s, "relevance"]) for s in symbols if s in work.index}

    selected: List[str] = []
    remaining = [s for s in symbols if s in corr.index]
    if not remaining:
        return work.sort_values("relevance", ascending=False)["symbol"].tolist()
        
    seed = max(remaining, key=lambda s: relevance.get(s, 0.0))
    selected.append(seed)
    remaining.remove(seed)

    while len(selected) < n_select and remaining:
        best_s: str | None = None
        best_score = -np.inf
        for s in remaining:
            reds = [
                float(corr.loc[s, s2]) for s2 in selected if s in corr.index and s2 in corr.columns
            ]
            red = float(np.mean(reds)) if reds else 0.0
            score = relevance.get(s, 0.0) - red
            if score > best_score:
                best_score = score
                best_s = s
        if best_s is None:
            break
        selected.append(best_s)
        remaining.remove(best_s)

    return selected


def load_screener_fixed_params_futures(
    project_root: Path,
    winning_signal_type: str | None = None,
) -> dict[str, Any]:
    path_json = project_root / "results" / "best_futures_4h.json"
    if path_json.is_file():
        try:
            raw = json.loads(path_json.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                out = dict(raw)
            else:
                out = _default_screener_params_from_space_futures()
        except:
            out = _default_screener_params_from_space_futures()
    else:
        out = _default_screener_params_from_space_futures()
    if winning_signal_type is not None:
        out["SIGNAL_TYPE"] = str(winning_signal_type)
    return out


def _default_screener_params_from_space_futures() -> dict[str, Any]:
    from src.domain.futures.opt_futures_utils.opt_params import build_full_discovery_space_futures
    space = build_full_discovery_space_futures()
    params: dict[str, Any] = {"TIMEFRAME": "4h", "LEVERAGE": 5, "USE_COMPOUNDING": True}
    for name, spec in space.items():
        t = spec.get("type")
        if t == "categorical":
            ch = tuple(spec.get("choices", ()))
            params[name] = ch[0] if ch else None
        elif t == "float":
            params[name] = (float(spec["low"]) + float(spec["high"])) / 2.0
        elif t == "int":
            params[name] = (int(spec["low"]) + int(spec["high"])) // 2
    return params


def screen_symbol_refinement_futures(
    broad_candidates: List[str],
    winning_signal_type: str,
    is_end_date: str,
    *,
    symbol_dfs_4h: Dict[str, pd.DataFrame],
    daily_dfs: Dict[str, pd.DataFrame],
    phase_b_params: Optional[Dict[str, Any]] = None,
    anchor_symbols: Optional[List[str]] = None,
) -> None:
    from config.opt_config import FUTURES_ANCHOR_SYMBOLS, FUTURES_SCREENER_CONFIG

    cfg = FUTURES_SCREENER_CONFIG
    min_tr = int(cfg.get("SCREENER_MIN_TRADES_DYNAMIC", 12))
    min_pf = float(cfg.get("SCREENER_MIN_PF", 1.15)) # Relaxed threshold
    mp_min = int(cfg.get("MP_MIN_SYMBOLS", 1))
    mp_max = int(cfg.get("MP_MAX_SYMBOLS", 5))
    top_k = int(cfg.get("CANDIDATES_TOP_K", 12))

    anchor_syms = list(anchor_symbols) if anchor_symbols is not None else list(FUTURES_ANCHOR_SYMBOLS)
    anchor_set = set(anchor_syms)
    
    if phase_b_params is not None:
        fixed_params = dict(phase_b_params)
        _logger.info("Phase C: using Phase-B probe params.")
    else:
        fixed_params = _default_screener_params_from_space_futures()
        _logger.info("Phase C: using default space midpoints.")
    
    fixed_params["SIGNAL_TYPE"] = winning_signal_type
    fixed_params.setdefault("TIMEFRAME", "4h")

    # Calculate ADV
    rows_adv: list[dict[str, float | str]] = []
    # Make sure we evaluate everything, not just intersection
    all_targets = list(dict.fromkeys(broad_candidates + anchor_syms))
    for sym in all_targets:
        df4 = symbol_dfs_4h.get(sym)
        if df4 is None or df4.empty: continue
        tail = _slice_df_to_is(df4, "1970-01-01", is_end_date).tail(180)
        if tail.empty: continue
        adv = float((tail["close"] * tail["volume"]).median() * 6)
        rows_adv.append({"symbol": sym, "adv": adv})

    vol_df = pd.DataFrame(rows_adv)
    if len(vol_df) < mp_min:
        _logger.error("Phase C Futures: insufficient symbols with data.")
        return

    sym_in_vol = set(str(s) for s in vol_df["symbol"].tolist())
    valid_anchors_ordered = [s for s in anchor_syms if s in sym_in_vol]
    
    _logger.info("Phase C anchor: %s (%d valid anchors)", ", ".join(valid_anchors_ordered), len(valid_anchors_ordered))

    anchor_vol = vol_df[vol_df["symbol"].isin(valid_anchors_ordered)].copy()
    dynamic_sym_list = [s for s in all_targets if s not in anchor_set and s in sym_in_vol]
    dyn_vol = vol_df[vol_df["symbol"].isin(dynamic_sym_list)].copy()

    _logger.info("Phase C anchor vol count: %d, dyn vol count: %d", len(anchor_vol), len(dyn_vol))

    # 1. Screen Anchors (Loose Criteria)
    fit_anchor, ret_anchor = pd.DataFrame(), {}
    if not anchor_vol.empty:
        fit_anchor, ret_anchor = screen_by_strategy_fit_futures(
            anchor_vol, symbol_dfs_4h, daily_dfs, fixed_params,
            is_end_date=is_end_date, min_trades=1, min_pf=0.0, min_cagr_pct=-1e9
        )

    # 2. Screen Dynamics (Strict Criteria)
    fit_dyn, ret_dyn = pd.DataFrame(), {}
    if not dyn_vol.empty:
        fit_dyn, ret_dyn = screen_by_strategy_fit_futures(
            dyn_vol, symbol_dfs_4h, daily_dfs, fixed_params,
            is_end_date=is_end_date, min_trades=min_tr, min_pf=min_pf, min_cagr_pct=0.0
        )

    # Fallback for dynamics if too few pass
    if not dyn_vol.empty and (fit_dyn.empty or len(fit_dyn) < mp_min):
        if phase_b_params is not None:
            _logger.warning("Phase C dynamic tier: too few symbols passed. Retrying with default midpoints.")
            fallback_params = _default_screener_params_from_space_futures()
            fallback_params["SIGNAL_TYPE"] = winning_signal_type
            fallback_params.setdefault("TIMEFRAME", "4h")
            fit_dyn, ret_dyn = screen_by_strategy_fit_futures(
                dyn_vol, symbol_dfs_4h, daily_dfs, fallback_params,
                is_end_date=is_end_date, min_trades=min_tr, min_pf=min_pf, min_cagr_pct=0.0
            )

    n_dyn_pass = len(fit_dyn) if not fit_dyn.empty else 0
    _logger.info("Phase C dynamic: %d symbols passed mini-BT", n_dyn_pass)

    # Combine returns for MP
    mp_cols: List[str] = []
    for s in valid_anchors_ordered:
        if s in ret_anchor and len(ret_anchor[s]) >= 30:
            mp_cols.append(s)
    if not fit_dyn.empty:
        for s in fit_dyn["symbol"].tolist():
            sym = str(s)
            if sym in ret_dyn and sym not in mp_cols:
                mp_cols.append(sym)

    if len(mp_cols) < 2:
        _logger.warning("Phase C: fewer than 2 return series for MP. Falling back to anchors.")
        final_symbols = valid_anchors_ordered[:mp_max]
    else:
        strat_combined: Dict[str, pd.Series] = {**ret_anchor, **ret_dyn}
        returns_for_mp = pd.concat({s: strat_combined[s] for s in mp_cols}, axis=1, join="inner")
        n_select = marchenko_pastur_n_factors(returns_for_mp, min_n=mp_min, max_n=mp_max)
        n_select = min(n_select, mp_max)
        n_anchor = len(valid_anchors_ordered)
        
        if fit_dyn.empty:
            n_select = min(n_select, n_anchor)
            final_symbols = valid_anchors_ordered[:n_select]
            _logger.info("Phase C: dynamic tier empty; anchor-only after MP clamp.")
        else:
            n_dynamic_slots = max(1, int(n_select) - n_anchor)
            pool_dyn = fit_dyn.sort_values("relevance", ascending=False).head(top_k).copy()
            dyn_picked = select_by_mrmr(pool_dyn, ret_dyn, min(n_dynamic_slots, len(pool_dyn)))
            ordered_dyn = [s for s in dyn_picked if s not in anchor_set]
            final_symbols = list(dict.fromkeys([*valid_anchors_ordered, *ordered_dyn]))[:mp_max]

    if not final_symbols:
        _logger.error("Phase C Futures: empty final symbol list.")
        return

    _logger.info("Final Futures symbols (%d): %s", len(final_symbols), final_symbols)
    update_futures_config_file(final_symbols)


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
