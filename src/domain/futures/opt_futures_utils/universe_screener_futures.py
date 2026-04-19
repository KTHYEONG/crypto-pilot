"""
Cross-Sectional Universe Screener for Binance Futures.
Focus: Liquidity, BTC Alignment (Beta/Corr), and Amihud Illiquidity.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.opt_config import FUTURES_ANCHOR_SYMBOLS, FUTURES_SCREENER_CONFIG
from src.domain.futures.data_collector import DataCollector

_logger = logging.getLogger(__name__)


def _default_universe_history_path() -> Path:
    return Path(__file__).resolve().parents[4] / "config" / "universe_history.json"


def _quarter_key_end_utc(key: str) -> pd.Timestamp:
    y_str, q_str = key.split("-Q")
    y, q = int(y_str), int(q_str)
    start_month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
    start = pd.Timestamp(year=y, month=start_month, day=1, tz="UTC")
    return start + pd.offsets.QuarterEnd(0)


def _point_in_time_symbols_from_history(
    as_of: str,
    history_path: Path | None,
) -> list[str]:
    """P3.A: listed universe at quarter end <= as_of (survivorship mitigation)."""
    path = history_path or _default_universe_history_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        _logger.debug("universe_history read failed: %s", e)
        return []
    quarters: dict[str, Any] = raw.get("quarters", {})
    if not quarters:
        return []
    as_ts = pd.Timestamp(as_of)
    if as_ts.tzinfo is None:
        as_ts = as_ts.tz_localize("UTC")
    else:
        as_ts = as_ts.tz_convert("UTC")
    best_end = pd.Timestamp.min.tz_localize("UTC")
    best_syms: list[str] = []
    for key, syms in quarters.items():
        try:
            q_end = _quarter_key_end_utc(str(key))
        except (ValueError, KeyError, IndexError):
            continue
        if q_end <= as_ts and q_end >= best_end and isinstance(syms, list):
            best_end = q_end
            best_syms = [str(s) for s in syms]
    return best_syms


def _calculate_amihud_illiquidity(df: pd.DataFrame, tail_bars: int = 180) -> float:
    if df.empty or len(df) < 20:
        return 999.0
    returns = df["close"].pct_change().abs()
    notional_vol = df["volume"] * df["close"]
    illiq = returns / notional_vol.replace(0, 1e-9)
    return float(illiq.tail(tail_bars).mean() * 1e6)


def _slice_df_to_is(df: pd.DataFrame, is_start: str, is_end: str) -> pd.DataFrame:
    if df.empty or "datetime" not in df.columns:
        return df.iloc[0:0].copy()
    is_s = pd.to_datetime(is_start).tz_localize("UTC") if pd.to_datetime(is_start).tzinfo is None else pd.to_datetime(is_start)
    is_e = pd.to_datetime(is_end).tz_localize("UTC") if pd.to_datetime(is_end).tzinfo is None else pd.to_datetime(is_end)
    dt = pd.to_datetime(df["datetime"], utc=True)
    mask = (dt >= is_s) & (dt < is_e)
    return df.loc[mask].reset_index(drop=True)


def _mean_abs_funding(daily_df: pd.DataFrame) -> float:
    """Institutional Gate: Detect crowded positions via funding rates (last 90d)."""
    if daily_df.empty or "funding_rate" not in daily_df.columns:
        return 0.0
    return float(daily_df["funding_rate"].abs().tail(90).mean())


def _has_regime_diversity(df: pd.DataFrame, tf: str = "4h", min_vol_cv: float = 0.3) -> bool:
    """ML Gate: Ensure symbol has multiple regimes for HMM/GP to learn effectively."""
    if df.empty or "close" not in df.columns:
        return True
    rets = np.log1p(df["close"].pct_change().dropna())
    # Use 4-day rolling window for volatility variability check
    rolling_window = 96 if tf == "1h" else 24
    rolling_vol = rets.rolling(rolling_window).std().dropna()
    if len(rolling_vol) < 50:
        return True
    cv = rolling_vol.std() / (rolling_vol.mean() + 1e-9)
    return float(cv) >= min_vol_cv


def screen_futures_universe(
    collector: DataCollector,
    candidate_pool: List[str],
    tf: str,
    cfg: Dict[str, Any],
    fetch_start: str,
    end_date: str,
    *,
    data_dir: Path | None = None,
    universe_history_path: Path | None = None,
) -> Tuple[List[str], int]:
    """Phase A: Market-Wide Liquidity Scan + PIT history + cached delisted symbols."""
    _ = candidate_pool
    _ = fetch_start
    _ = data_dir
    _logger.info("Phase A: Market Scan (ADV check) + PIT/cache universe merge...")
    try:
        tickers = collector.client.exchange.fetch_tickers()
        valid_tickers = []
        # Use MIN_ADV_USDT from new config structure
        min_adv = float(cfg.get("MIN_ADV_USDT", 25_000_000))
        for sym, t in tickers.items():
            if not (sym.endswith("/USDT") or sym.endswith("/USDT:USDT")):
                continue
            vol = float(t.get("quoteVolume") or 0.0)
            if vol >= min_adv:
                valid_tickers.append({"symbol": sym.split(":")[0], "vol": vol})
    except Exception as e:
        _logger.error("Ticker scan failed: %s", e)
        return list(FUTURES_ANCHOR_SYMBOLS), 0

    valid_tickers.sort(key=lambda x: x["vol"], reverse=True)
    by_vol = [item["symbol"] for item in valid_tickers]
    pit_syms = _point_in_time_symbols_from_history(end_date, universe_history_path)
    if pit_syms:
        _logger.info("PIT universe_history: merged %d symbols (<= %s).", len(pit_syms), end_date)
    try:
        cached_syms = collector.list_cached_parquet_symbols(tf)
    except Exception:
        cached_syms = []
    if cached_syms:
        _logger.info("Cached parquet symbols on %s: %d", tf, len(cached_syms))

    merged = list(
        dict.fromkeys(
            list(FUTURES_ANCHOR_SYMBOLS) + pit_syms + cached_syms + by_vol
        )
    )
    k = int(cfg.get("BROAD_POOL_K", 80))
    final_list = merged[:k]
    return final_list, len(final_list)


def screen_symbol_refinement_futures(
    broad_candidates: List[str],
    winning_signal_type: str,
    is_end_date: str,
    tf: str = "1h",
    *,
    symbol_dfs_4h: Dict[str, pd.DataFrame],
    daily_dfs: Dict[str, pd.DataFrame],
    phase_b_params: Optional[Dict[str, Any]] = None,
    anchor_symbols: Optional[List[str]] = None,
) -> bool:
    """Phase B: Institutional Risk Filtering (Beta, Corr, Funding, Diversity)."""
    anchors = list(anchor_symbols) if anchor_symbols else FUTURES_ANCHOR_SYMBOLS
    if "BTC/USDT" not in symbol_dfs_4h:
        _logger.error("BTC/USDT missing from symbol_dfs. Cannot refine universe.")
        return False
    
    cfg = FUTURES_SCREENER_CONFIG
    
    # [Dynamic Thresholds] 1h vs 4h
    min_total_bars = 4000 if tf == "1h" else 1000
    min_is_bars = 2000 if tf == "1h" else 500
    min_corr_pairs = 800 if tf == "1h" else 200
    tail_adv_bars = 2160 if tf == "1h" else 540
    tail_amihud_bars = 720 if tf == "1h" else 180
    adv_multiplier = 24 if tf == "1h" else 6
    
    btc_df = _slice_df_to_is(symbol_dfs_4h["BTC/USDT"], "1970-01-01", is_end_date)
    btc_rets = btc_df["close"].pct_change().fillna(0.0)

    candidate_stats = []
    for sym in list(dict.fromkeys(broad_candidates + anchors)):
        df = symbol_dfs_4h.get(sym)
        if df is None or len(df) < min_total_bars:
            continue
        is_df = _slice_df_to_is(df, "1970-01-01", is_end_date)
        if len(is_df) < min_is_bars:
            continue
        
        # [ML Gate] Regime Diversity
        if not _has_regime_diversity(is_df, tf=tf, min_vol_cv=cfg.get("MIN_VOL_CV", 0.3)):
            continue
            
        # [Institutional Gate] Funding Rate Crowding
        if _mean_abs_funding(daily_dfs.get(sym, pd.DataFrame())) > cfg.get("FUNDING_RATE_MAX_ABS", 0.0008):
            if sym not in anchors:
                continue
        
        rets = is_df["close"].pct_change().fillna(0.0)
        merged = pd.concat([btc_rets, rets], axis=1, join="inner").dropna()
        if len(merged) < min_corr_pairs:
            continue
        
        # [Risk Gates] Beta & Correlation vs BTC
        beta = np.cov(merged.iloc[:, 0], merged.iloc[:, 1])[0, 1] / (np.var(merged.iloc[:, 0]) + 1e-12)
        corr = merged.iloc[:, 0].corr(merged.iloc[:, 1])
        
        adv = (is_df["close"] * is_df["volume"]).tail(tail_adv_bars).mean() * adv_multiplier
        amihud = _calculate_amihud_illiquidity(is_df, tail_bars=tail_amihud_bars)
        candidate_stats.append({"symbol": sym, "beta": beta, "corr": corr, "adv": adv, "amihud": amihud})

    # Liquidity Sort (Top 60)
    candidate_stats.sort(key=lambda x: x["adv"], reverse=True)
    pool = candidate_stats[:60]
    
    # Amihud Pruning (Keep Top 75%)
    pool.sort(key=lambda x: x["amihud"])
    prune_ratio = float(cfg.get("AMIHUD_PRUNE_RATIO", 0.75))
    pool = pool[:int(len(pool) * prune_ratio)]
    
    # Final Strict Thresholds
    min_corr = float(cfg.get("MIN_CORR_BTC", 0.50))
    max_beta = float(cfg.get("MAX_BETA_BTC", 1.40))
    min_beta = float(cfg.get("MIN_BETA_BTC", 0.60))
    
    final = [
        c["symbol"] for c in pool 
        if (min_beta <= c["beta"] <= max_beta and c["corr"] > min_corr) or c["symbol"] in anchors
    ]
    
    final_symbols = list(dict.fromkeys(final))[:int(cfg.get("FINAL_POOL_K", 40))]
    if "BTC/USDT" not in final_symbols:
        final_symbols.insert(0, "BTC/USDT")
    
    _logger.info(f"Final Refined Universe ({tf}): {final_symbols}")
    update_futures_config_file(final_symbols)
    return True


def update_futures_config_file(symbols: List[str]) -> None:
    path = Path("config/opt_config.py")
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    pattern = r"FUTURES_SYMBOLS(?::\s*List\[str\])?\s*=\s*\[.*?\]"
    new_block = "FUTURES_SYMBOLS: List[str] = [\n" + "".join([f'    "{s}",\n' for s in symbols]) + "]"
    path.write_text(re.sub(pattern, new_block, content, count=1, flags=re.DOTALL), encoding="utf-8")
