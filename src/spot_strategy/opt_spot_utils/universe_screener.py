from __future__ import annotations

import json
import logging
import random
import re
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Project Root Setup
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import SPOT_BACKTEST_END_DATE, SPOT_INITIAL_BALANCE
from src.spot_strategy.data_collector_spot import DataCollectorSpot
from src.spot_strategy.engine_spot import BacktestEngineFastSpot
from src.spot_strategy.opt_spot_utils.metrics import calc_profit_factor_from_pnl
from src.spot_strategy.strategies_spot import UltimateSpotStrategy
from src.spot_strategy.upbit_client import UpbitClient

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("universe_screener")


def filter_by_adv_floor(stats_df: pd.DataFrame, min_adv_krw_day: float) -> pd.DataFrame:
    if stats_df.empty:
        return stats_df
    return stats_df[stats_df["adv"] >= float(min_adv_krw_day)].copy()


def filter_by_p25_bar_liquidity(
    stats_df: pd.DataFrame, min_p25_bar_krw: float
) -> pd.DataFrame:
    """Reject coins where p25 4H-bar KRW volume < min_p25_bar_krw.
    Prevents spike-dominant volume from passing ADV filter."""
    if stats_df.empty or "p25_bar_vol" not in stats_df.columns:
        return stats_df
    return stats_df[stats_df["p25_bar_vol"] >= float(min_p25_bar_krw)].copy()


def _median_atr_percent(
    df: pd.DataFrame,
    atr_period: int,
    tail_bars: int = 42,
) -> float:
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    ap = int(max(2, atr_period))
    atr = tr.rolling(ap, min_periods=ap).mean()
    atr_pct = (atr / close) * 100.0
    recent = atr_pct.tail(int(tail_bars)).dropna()
    if recent.empty:
        return float("nan")
    return float(recent.median())


def screen_by_volatility_fit(
    stats_df: pd.DataFrame,
    symbol_dfs: Dict[str, pd.DataFrame],
    *,
    atr_period: int,
    atr_pct_min: float,
    atr_pct_max: float,
    atr_tail_bars: int = 42,
) -> pd.DataFrame:
    """Keep symbols whose median ATR% (4H) lies in [atr_pct_min, atr_pct_max]."""
    rows: list[dict[str, float | str]] = []
    for _, row in stats_df.iterrows():
        sym = str(row["symbol"])
        df = symbol_dfs.get(sym)
        if df is None or df.empty or len(df) < atr_period + 5:
            continue
        atr_pct = _median_atr_percent(df, atr_period, tail_bars=int(atr_tail_bars))
        if not np.isfinite(atr_pct):
            continue
        if atr_pct < float(atr_pct_min) or atr_pct > float(atr_pct_max):
            continue
        rows.append({"symbol": sym, "adv": float(row["adv"]), "atr_pct_median": atr_pct})
    return pd.DataFrame(rows)


def _segment_with_context(
    full_signal_df: pd.DataFrame,
    exec_start_idx: int,
    exec_end_idx: int,
) -> tuple[pd.DataFrame, int]:
    slice_start = max(0, int(exec_start_idx) - 1)
    slice_end = max(slice_start, int(exec_end_idx))
    segment = full_signal_df.iloc[slice_start:slice_end].copy()
    execution_start_idx = int(exec_start_idx) - slice_start
    if execution_start_idx == 0 and len(segment) > 1:
        execution_start_idx = 1
    return segment, execution_start_idx


def _align_is_end_to_series_tz(is_end: pd.Timestamp, dt_series: pd.Series) -> pd.Timestamp:
    """Align IS end timestamp to df datetime timezone for strict IS-only windows."""
    ts = pd.to_datetime(is_end)
    ref_tz = getattr(dt_series.dt, "tz", None)
    if ref_tz is not None:
        if ts.tzinfo is None:
            return ts.tz_localize(ref_tz)
        return ts.tz_convert(ref_tz)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def _indices_is_last_year(df: pd.DataFrame, is_end_date: pd.Timestamp | str) -> tuple[int, int]:
    """
    Last ~365 days of in-sample rows strictly before is_end_date (OOS boundary).
    Avoids look-ahead: mini-BT window uses only IS bars, never OOS.
    """
    if df.empty or "datetime" not in df.columns:
        return 0, 0
    dt = df["datetime"]
    is_end = _align_is_end_to_series_tz(pd.to_datetime(is_end_date), dt)
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


def _equity_simple_returns(eq: np.ndarray, dt: pd.Series) -> pd.Series:
    eq = np.asarray(eq, dtype=np.float64)
    if len(eq) < 3 or len(dt) != len(eq):
        return pd.Series(dtype=np.float64)
    r = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    idx = pd.to_datetime(dt.iloc[1:].values)
    return pd.Series(r, index=idx, dtype=np.float64)


def _run_mini_backtest_window(
    sym: str,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    params: dict[str, Any],
    test_start: int,
    test_end: int,
) -> tuple[float, int, float, float, pd.Series]:
    """
    Single-symbol mini backtest on [test_start:test_end).
    Returns pf, n_trades, cagr_pct, relevance_score, equity_simple_returns (indexed).
    """
    strategy = UltimateSpotStrategy(name=f"Screener_{sym}", params=dict(params))
    p = {**dict(params), "USE_COMPOUNDING": True}
    strategy.params = p
    full_signal = strategy.generate_signals(df_4h.copy(deep=True))
    sig_oos, exec_idx = _segment_with_context(full_signal, test_start, test_end)
    sig_oos.attrs = {"warmup_bars": 0}

    engine = BacktestEngineFastSpot(
        hourly_df=sig_oos,
        daily_df=df_1d,
        strategy=strategy,
        initial_balance=float(SPOT_INITIAL_BALANCE),
        merge_index_map=None,
        precomputed_daily_df=None,
        warmup_bars=0,
        execution_start_idx=exec_idx,
    )
    engine.strategy.params = p

    try:
        result = engine.run()
    except Exception as e:
        _logger.warning("Mini backtest failed %s: %s", sym, e)
        return 0.0, 0, -100.0, 0.0, pd.Series(dtype=np.float64)

    trades_df = result.get("trades_df", pd.DataFrame())
    if trades_df is None or trades_df.empty:
        return 0.0, 0, -100.0, 0.0, pd.Series(dtype=np.float64)

    n_trades = int(len(trades_df))
    pf = float(calc_profit_factor_from_pnl(trades_df["pnl"]))
    ret_pct = float(result.get("total_return_pct", 0.0))

    dt_exec = sig_oos["datetime"].iloc[min(exec_idx, len(sig_oos) - 1)]
    span_days = max(
        1.0,
        float((sig_oos["datetime"].iloc[-1] - dt_exec).total_seconds() / 86400.0),
    )
    total_ret_ratio = 1.0 + (ret_pct / 100.0)
    if total_ret_ratio > 0:
        cagr = ((total_ret_ratio ** (365.0 / span_days)) - 1.0) * 100.0
    else:
        cagr = -100.0

    eq = getattr(engine, "_equity_curve", None)
    if eq is None:
        ret_ser = pd.Series(dtype=np.float64)
    else:
        ret_ser = _equity_simple_returns(np.asarray(eq), sig_oos["datetime"])

    rel = pf * float(np.log1p(float(n_trades)))
    return pf, n_trades, float(cagr), rel, ret_ser


def screen_by_strategy_fit(
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
    """
    Run fixed-parameter mini backtest (last ~1y of IS, before is_end_date) per symbol.
    Pass criteria: trades >= min_trades, PF >= min_pf, CAGR > min_cagr_pct.
    """
    rows: list[dict[str, float | str]] = []
    ret_map: Dict[str, pd.Series] = {}
    fp_base = dict(fixed_params)
    if signal_type is not None:
        fp_base["SIGNAL_TYPE"] = str(signal_type)

    for _, row in tqdm(vol_df.iterrows(), total=len(vol_df), desc="Strategy mini-BT"):
        sym = str(row["symbol"])
        df4 = symbol_dfs.get(sym)
        d1 = daily_dfs.get(sym)
        if df4 is None or d1 is None:
            continue
        ts, te = _indices_is_last_year(df4, is_end_date)
        if te <= ts + 10:
            continue

        pf, n_tr, cagr, rel, ret_ser = _run_mini_backtest_window(sym, df4, d1, fp_base, ts, te)
        if n_tr < int(min_trades) or pf < float(min_pf) or cagr <= float(min_cagr_pct):
            continue
        if ret_ser is None or len(ret_ser) < 30:
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
    t_obs = int(len(rets))
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

    rets = pd.concat({s: strategy_returns[s] for s in symbols}, axis=1, join="inner")
    if rets.shape[0] < 30:
        return work.sort_values("relevance", ascending=False)["symbol"].tolist()

    corr = rets.corr().abs()
    relevance = {s: float(work.loc[s, "relevance"]) for s in symbols if s in work.index}

    selected: List[str] = []
    remaining = set(symbols)
    seed = max(remaining, key=lambda s: relevance.get(s, 0.0))
    selected.append(seed)
    remaining.remove(seed)

    while len(selected) < n_select and remaining:
        best_s: str | None = None
        best_score = -np.inf
        for s in remaining:
            reds = [
                float(corr.loc[s, s2])
                for s2 in selected
                if s in corr.index and s2 in corr.columns
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


def load_screener_fixed_params(
    project_root: Path,
    winning_signal_type: str | None = None,
) -> dict[str, Any]:
    path_json = project_root / "results" / "best_params_4h.json"
    out: dict[str, Any]
    if path_json.is_file():
        raw = json.loads(path_json.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            out = dict(raw)
        else:
            out = _default_screener_params_from_space()
    else:
        out = _default_screener_params_from_space()
    if winning_signal_type is not None:
        out["SIGNAL_TYPE"] = str(winning_signal_type)
    return out


def _default_screener_params_from_space() -> dict[str, Any]:
    from src.spot_strategy.opt_spot_utils.opt_params import build_full_discovery_space

    space = build_full_discovery_space()
    params: dict[str, Any] = {"TIMEFRAME": "4h", "LEVERAGE": 1}
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


def _slice_df_to_is(df: pd.DataFrame, is_start: str, is_end: str) -> pd.DataFrame:
    if df.empty or "datetime" not in df.columns:
        return df.iloc[0:0].copy()
    is_s = pd.to_datetime(is_start)
    is_e = pd.to_datetime(is_end)
    dt = df["datetime"]
    ref_tz = getattr(dt.dt, "tz", None)
    if ref_tz is not None:
        if is_s.tzinfo is None:
            is_s = is_s.tz_localize(ref_tz)
        else:
            is_s = is_s.tz_convert(ref_tz)
        if is_e.tzinfo is None:
            is_e = is_e.tz_localize(ref_tz)
        else:
            is_e = is_e.tz_convert(ref_tz)
    mask = (dt >= is_s) & (dt < is_e)
    return df.loc[mask].reset_index(drop=True)


def update_broad_candidates_in_config(symbols: List[str]) -> None:
    config_path = Path("config/opt_config.py")
    if not config_path.exists():
        _logger.error("config/opt_config.py not found.")
        return
    content = config_path.read_text(encoding="utf-8")
    pattern = r"SPOT_BROAD_CANDIDATES(?:: List\[str\])?\s*=\s*\[.*?\]"
    new_block = "SPOT_BROAD_CANDIDATES: List[str] = [\n"
    for s in symbols:
        new_block += f'    "{s}",\n'
    new_block += "]"
    content_new = re.sub(pattern, new_block, content, count=1, flags=re.DOTALL)
    if content_new == content:
        _logger.warning("SPOT_BROAD_CANDIDATES block not updated (regex miss).")
    else:
        config_path.write_text(content_new, encoding="utf-8")
        _logger.info("Updated SPOT_BROAD_CANDIDATES (%d symbols).", len(symbols))


def screen_broad_universe(
    *,
    is_start: str,
    is_end: str,
    fetch_end: str | None = None,
) -> List[str]:
    """
    Phase A: ADV + ATR% only (signal-agnostic). Uses in-sample bars only for gates; OOS excluded.
    """
    from config.opt_config import SPOT_SCREENER_CONFIG, get_quarterly_window

    cfg = SPOT_SCREENER_CONFIG
    adv_min = float(cfg["ADV_MIN_KRW_DAY"])
    atr_period = int(cfg["SCREENER_ATR_PERIOD"])
    atr_min = float(cfg["SCREENER_ATR_PCT_MIN"])
    atr_max = float(cfg["SCREENER_ATR_PCT_MAX"])
    mp_min = int(cfg["MP_MIN_SYMBOLS"])
    min_is_bars = 500
    atr_tail_bars = 540

    end_date = fetch_end if fetch_end is not None else SPOT_BACKTEST_END_DATE
    start_date = (pd.to_datetime(end_date) - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    _logger.info(
        "Phase A: structural gate is_start=%s is_end=%s fetch_end=%s (mini-BT uses IS last year only)",
        is_start,
        is_end,
        end_date,
    )

    client = UpbitClient()
    collector = DataCollectorSpot()

    try:
        fetch_start_date, _, _, _ = get_quarterly_window()
    except Exception:
        fetch_start_date = (pd.to_datetime(SPOT_BACKTEST_END_DATE) - timedelta(days=365 * 3)).strftime(
            "%Y-%m-%d"
        )

    stats: list[dict[str, float | str]] = []
    symbol_dfs: Dict[str, pd.DataFrame] = {}

    _logger.info("Fetching all KRW markets from Upbit (Phase A)...")
    all_markets = client.exchange.load_markets()
    krw_symbols = [m["id"] for m in all_markets.values() if m["id"].startswith("KRW-")]

    for sym in tqdm(krw_symbols, desc="Phase A 4H"):
        try:
            time.sleep(0.12)
            ccxt_sym = client._normalize_symbol(sym)
            since_ms = client.exchange.parse8601(f"{fetch_start_date}T00:00:00Z")
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
            oldest_dt = pd.to_datetime(first_candle[0][0], unit="ms")
            if oldest_dt > pd.to_datetime(fetch_start_date) + timedelta(days=5):
                continue

            df = collector.collect_and_save(sym, "4h", start_date, end_date)
            if df is None or df.empty:
                continue
            df_is = _slice_df_to_is(df, is_start, is_end)
            if len(df_is) < min_is_bars:
                continue

            recent_df = df_is.tail(180)
            krw_vol_4h = recent_df["close"] * recent_df["volume"]
            adv = float(krw_vol_4h.median() * 6)  # median × 6 bars/day
            p25_bar_vol = float(krw_vol_4h.quantile(0.25))  # quiet-bar floor
            stats.append({"symbol": sym, "adv": adv, "p25_bar_vol": p25_bar_vol})
            symbol_dfs[sym] = df_is
        except Exception as e:
            _logger.warning("Failed to process %s: %s", sym, e)

    stats_df = pd.DataFrame(stats)
    if stats_df.empty:
        _logger.error("Phase A: no symbols passed initial data check.")
        return []

    adv_filtered_df = filter_by_adv_floor(stats_df, adv_min)
    min_p25 = float(cfg.get("SCREENER_MIN_P25_BAR_KRW", 0.0))
    if min_p25 > 0:
        adv_filtered_df = filter_by_p25_bar_liquidity(adv_filtered_df, min_p25)
    if len(adv_filtered_df) < mp_min:
        _logger.error(
            "Phase A ADV floor: insufficient symbols (%s).",
            len(adv_filtered_df),
        )
        return []

    vol_df = screen_by_volatility_fit(
        adv_filtered_df,
        symbol_dfs,
        atr_period=atr_period,
        atr_pct_min=atr_min,
        atr_pct_max=atr_max,
        atr_tail_bars=atr_tail_bars,
    )
    if len(vol_df) < mp_min:
        _logger.error("Phase A volatility gate: insufficient symbols.")
        return []

    vol_df = vol_df.sort_values("adv", ascending=False).reset_index(drop=True)
    out = [str(s) for s in vol_df["symbol"].tolist()]
    update_broad_candidates_in_config(out)
    _logger.info("Phase A: %d broad candidates (ADV+ATR, sorted by ADV desc).", len(out))
    return out


def screen_symbol_refinement(
    broad_candidates: List[str],
    winning_signal_type: str,
    is_end_date: str,
    *,
    symbol_dfs_4h: Dict[str, pd.DataFrame],
    daily_dfs: Dict[str, pd.DataFrame],
    adv_by_symbol: Optional[Dict[str, float]] = None,
    phase_b_params: Optional[Dict[str, Any]] = None,
    phase_a_broad: Optional[List[str]] = None,
    anchor_symbols: Optional[List[str]] = None,
) -> None:
    """
    Phase C+D: mini-BT on IS last year with winning SIGNAL_TYPE, then MP + mRMR → SPOT_SYMBOLS.

    phase_b_params: midpoint params derived from the Phase-B winning combo (via build_probe_params).
    When supplied, avoids loading stale best_params_4h.json and breaks the circular bias where
    the universe is pre-filtered by the previous optimization cycle's best parameters.
    When None, falls back to search-space midpoints (_default_screener_params_from_space).

    phase_a_broad: Phase A symbol list (no anchor merge). Used to build the dynamic tier as
    (phase_a_broad \\ anchor_symbols). If None, uses broad_candidates for subtraction.

    anchor_symbols: Tier-1 anchors (always included when data exists); mini-BT uses loose gates
    only to collect return series for MP. When None, loads SPOT_ANCHOR_SYMBOLS from config.
    If no anchor has data, falls back to legacy single-stream Phase C.
    """
    from config.opt_config import SPOT_ANCHOR_SYMBOLS, SPOT_SCREENER_CONFIG

    cfg = SPOT_SCREENER_CONFIG
    min_tr = int(cfg["SCREENER_MIN_TRADES"])
    min_pf = float(cfg["SCREENER_MIN_PF"])
    min_tr_dyn = int(cfg.get("SCREENER_MIN_TRADES_DYNAMIC", 3))
    mp_min = int(cfg["MP_MIN_SYMBOLS"])
    mp_max = int(cfg["MP_MAX_SYMBOLS"])
    top_k = int(cfg["CANDIDATES_TOP_K"])
    adaptive_adv = bool(cfg["ADAPTIVE_SLIPPAGE_REF_ADV"])

    anchor_syms: List[str] = list(anchor_symbols) if anchor_symbols is not None else list(SPOT_ANCHOR_SYMBOLS)
    anchor_set = set(anchor_syms)
    phase_a_list: List[str] = list(phase_a_broad) if phase_a_broad is not None else list(broad_candidates)

    if phase_b_params is not None:
        fixed_params = dict(phase_b_params)
        _logger.info(
            "Phase C: using Phase-B probe params (signal-agnostic; no stale best_params bias)."
        )
    else:
        fixed_params = _default_screener_params_from_space()
        _logger.info(
            "Phase C: no Phase-B params supplied; using search-space midpoints as fixed_params."
        )
    fixed_params["SIGNAL_TYPE"] = winning_signal_type
    fixed_params.setdefault("TIMEFRAME", "4h")

    _logger.info(
        'Phase C+D: mini-BT SIGNAL_TYPE=%s (Stage1 winning), is_end="%s"',
        winning_signal_type,
        is_end_date,
    )

    rows_adv: list[dict[str, float | str]] = []
    for sym in broad_candidates:
        df4 = symbol_dfs_4h.get(sym)
        if df4 is None or df4.empty:
            continue
        if adv_by_symbol is not None and sym in adv_by_symbol:
            adv = float(adv_by_symbol[sym])
        else:
            tail = _slice_df_to_is(df4, "1970-01-01", is_end_date).tail(180)
            if tail.empty or len(tail) < 10:
                continue
            krw_4h = tail["close"] * tail["volume"]
            adv = float(krw_4h.median() * 6)
        rows_adv.append({"symbol": sym, "adv": adv})

    vol_df = pd.DataFrame(rows_adv)
    if len(vol_df) < mp_min:
        _logger.error("Phase C: insufficient symbols with data. Leaving opt_config.py unchanged.")
        return

    sym_in_vol = set(str(s) for s in vol_df["symbol"].tolist())
    valid_anchors_ordered = [s for s in anchor_syms if s in sym_in_vol]
    use_anchor_arch = bool(anchor_syms) and bool(valid_anchors_ordered)

    if not use_anchor_arch:
        _logger.info(
            "Phase C: anchor tier disabled or no anchor data in vol_df; legacy single-stream mini-BT."
        )
        fit_df, strat_returns = screen_by_strategy_fit(
            vol_df,
            symbol_dfs_4h,
            daily_dfs,
            fixed_params,
            is_end_date=is_end_date,
            min_trades=min_tr,
            min_pf=min_pf,
            min_cagr_pct=0.0,
            signal_type=winning_signal_type,
        )
        if fit_df.empty or len(fit_df) < mp_min:
            if phase_b_params is not None:
                _logger.warning(
                    "Phase C: Phase-B params yielded %d symbols (need %d); retrying with default space midpoints.",
                    len(fit_df) if not fit_df.empty else 0,
                    mp_min,
                )
                fallback_params = _default_screener_params_from_space()
                fallback_params["SIGNAL_TYPE"] = winning_signal_type
                fallback_params.setdefault("TIMEFRAME", "4h")
                fit_df, strat_returns = screen_by_strategy_fit(
                    vol_df,
                    symbol_dfs_4h,
                    daily_dfs,
                    fallback_params,
                    is_end_date=is_end_date,
                    min_trades=min_tr,
                    min_pf=min_pf,
                    min_cagr_pct=0.0,
                    signal_type=winning_signal_type,
                )
            if fit_df.empty or len(fit_df) < mp_min:
                _logger.error(
                    "Phase C strategy screen: insufficient symbols after fallback. Leaving opt_config.py unchanged."
                )
                return

        pool = fit_df.sort_values("relevance", ascending=False).head(top_k).copy()
        cand_syms = [str(s) for s in pool["symbol"].tolist()]

        if len(cand_syms) < 2:
            _logger.error("Too few symbols with equity return series for MP/mRMR.")
            return

        returns_for_mp = pd.concat({s: strat_returns[s] for s in cand_syms}, axis=1, join="inner")
        n_select = marchenko_pastur_n_factors(returns_for_mp, min_n=mp_min, max_n=mp_max)

        final_symbols = select_by_mrmr(pool, strat_returns, n_select)
        if not final_symbols:
            _logger.error("mRMR produced no symbols. Leaving opt_config.py unchanged.")
            return

        cluster_names = {s: f"mrmr_{i}" for i, s in enumerate(final_symbols)}
        adv_series = vol_df.set_index("symbol")["adv"]
        median_adv = float(adv_series.reindex(final_symbols).dropna().median())
        if not np.isfinite(median_adv) or median_adv <= 0:
            median_adv = float(adv_series.median())

        _logger.info("Final symbols (%s): %s", len(final_symbols), final_symbols)
        _logger.info("Marchenko-Pastur n_select=%s, median ADV=%s KRW/day", n_select, f"{median_adv:,.0f}")

        update_config_file(
            final_symbols,
            cluster_names,
            median_adv_krw=median_adv,
            adaptive_slippage=adaptive_adv,
        )
        return

    _logger.info(
        "Phase C anchor: %s (%d valid anchors)",
        ", ".join(valid_anchors_ordered),
        len(valid_anchors_ordered),
    )

    anchor_vol = vol_df[vol_df["symbol"].isin(valid_anchors_ordered)].copy()
    dynamic_sym_list = [s for s in phase_a_list if s not in anchor_set and s in sym_in_vol]
    dyn_vol = vol_df[vol_df["symbol"].isin(dynamic_sym_list)].copy()

    fit_anchor, ret_anchor = screen_by_strategy_fit(
        anchor_vol,
        symbol_dfs_4h,
        daily_dfs,
        fixed_params,
        is_end_date=is_end_date,
        min_trades=1,
        min_pf=0.0,
        min_cagr_pct=-1e9,
        signal_type=winning_signal_type,
    )
    if (fit_anchor.empty or len(fit_anchor) < len(valid_anchors_ordered)) and phase_b_params is not None:
        _logger.warning(
            "Phase C anchor tier: retrying mini-BT with default space midpoints (partial anchor pass)."
        )
        fallback_params = _default_screener_params_from_space()
        fallback_params["SIGNAL_TYPE"] = winning_signal_type
        fallback_params.setdefault("TIMEFRAME", "4h")
        fit_anchor, ret_anchor = screen_by_strategy_fit(
            anchor_vol,
            symbol_dfs_4h,
            daily_dfs,
            fallback_params,
            is_end_date=is_end_date,
            min_trades=1,
            min_pf=0.0,
            min_cagr_pct=-1e9,
            signal_type=winning_signal_type,
        )

    fit_dyn, ret_dyn = screen_by_strategy_fit(
        dyn_vol,
        symbol_dfs_4h,
        daily_dfs,
        fixed_params,
        is_end_date=is_end_date,
        min_trades=min_tr_dyn,
        min_pf=min_pf,
        min_cagr_pct=0.0,
        signal_type=winning_signal_type,
    )
    if fit_dyn.empty or len(fit_dyn) < mp_min:
        if phase_b_params is not None:
            _logger.warning(
                "Phase C dynamic tier: Phase-B params yielded %d symbols (threshold=%d trades); "
                "retrying with default space midpoints.",
                len(fit_dyn) if not fit_dyn.empty else 0,
                min_tr_dyn,
            )
            fallback_params = _default_screener_params_from_space()
            fallback_params["SIGNAL_TYPE"] = winning_signal_type
            fallback_params.setdefault("TIMEFRAME", "4h")
            fit_dyn, ret_dyn = screen_by_strategy_fit(
                dyn_vol,
                symbol_dfs_4h,
                daily_dfs,
                fallback_params,
                is_end_date=is_end_date,
                min_trades=min_tr_dyn,
                min_pf=min_pf,
                min_cagr_pct=0.0,
                signal_type=winning_signal_type,
            )

    n_dyn_pass = len(fit_dyn) if not fit_dyn.empty else 0
    _logger.info(
        "Phase C dynamic: %d symbols passed mini-BT (threshold=%d trades)",
        n_dyn_pass,
        min_tr_dyn,
    )

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
        _logger.warning(
            "Phase C: fewer than 2 return series for MP (anchors+dynamic); using legacy single-stream."
        )
        fit_df, strat_returns = screen_by_strategy_fit(
            vol_df,
            symbol_dfs_4h,
            daily_dfs,
            fixed_params,
            is_end_date=is_end_date,
            min_trades=min_tr,
            min_pf=min_pf,
            min_cagr_pct=0.0,
            signal_type=winning_signal_type,
        )
        if fit_df.empty or len(fit_df) < mp_min:
            if phase_b_params is not None:
                fallback_params = _default_screener_params_from_space()
                fallback_params["SIGNAL_TYPE"] = winning_signal_type
                fallback_params.setdefault("TIMEFRAME", "4h")
                fit_df, strat_returns = screen_by_strategy_fit(
                    vol_df,
                    symbol_dfs_4h,
                    daily_dfs,
                    fallback_params,
                    is_end_date=is_end_date,
                    min_trades=min_tr,
                    min_pf=min_pf,
                    min_cagr_pct=0.0,
                    signal_type=winning_signal_type,
                )
        if fit_df.empty or len(fit_df) < mp_min:
            final_symbols = valid_anchors_ordered[:mp_max]
            n_select = len(final_symbols)
            _logger.info("Phase C fallback: anchor-only symbols: %s", final_symbols)
        else:
            pool = fit_df.sort_values("relevance", ascending=False).head(top_k).copy()
            cand_syms = [str(s) for s in pool["symbol"].tolist()]
            if len(cand_syms) < 2:
                final_symbols = valid_anchors_ordered[:mp_max]
                n_select = len(final_symbols)
            else:
                returns_for_mp = pd.concat({s: strat_returns[s] for s in cand_syms}, axis=1, join="inner")
                n_select = marchenko_pastur_n_factors(returns_for_mp, min_n=mp_min, max_n=mp_max)
                picked = select_by_mrmr(pool, strat_returns, n_select)
                final_symbols = picked if picked else valid_anchors_ordered[:mp_max]
    else:
        strat_combined: Dict[str, pd.Series] = {**ret_anchor, **ret_dyn}
        returns_for_mp = pd.concat({s: strat_combined[s] for s in mp_cols}, axis=1, join="inner")
        n_select = marchenko_pastur_n_factors(returns_for_mp, min_n=mp_min, max_n=mp_max)
        n_select = min(n_select, mp_max)
        n_anchor = len(valid_anchors_ordered)
        if fit_dyn.empty:
            n_select = min(n_select, n_anchor)
            final_symbols = valid_anchors_ordered[:n_select]
            _logger.info("Phase C: dynamic tier empty; anchor-only after MP clamp: %s", final_symbols)
        else:
            n_dynamic_slots = max(1, int(n_select) - n_anchor)
            pool_dyn = fit_dyn.sort_values("relevance", ascending=False).head(top_k).copy()
            dyn_picked = select_by_mrmr(pool_dyn, ret_dyn, min(n_dynamic_slots, len(pool_dyn)))
            ordered_dyn = [s for s in dyn_picked if s not in anchor_set]
            final_symbols = list(dict.fromkeys([*valid_anchors_ordered, *ordered_dyn]))[:mp_max]

    if not final_symbols:
        _logger.error("Phase C: empty final symbol list. Leaving opt_config.py unchanged.")
        return

    cluster_names = {s: f"mrmr_{i}" for i, s in enumerate(final_symbols)}
    adv_series = vol_df.set_index("symbol")["adv"]
    median_adv = float(adv_series.reindex(final_symbols).dropna().median())
    if not np.isfinite(median_adv) or median_adv <= 0:
        median_adv = float(adv_series.median())

    _logger.info("Final symbols (%s): %s", len(final_symbols), final_symbols)
    _logger.info("Marchenko-Pastur n_select=%s, median ADV=%s KRW/day", n_select, f"{median_adv:,.0f}")

    update_config_file(
        final_symbols,
        cluster_names,
        median_adv_krw=median_adv,
        adaptive_slippage=adaptive_adv,
    )


def screen_universe() -> None:
    """CLI entry: Phase A then C+D using SIGNAL_TYPE from best_params (no Stage1). Prefer opt_spot.py Phase 0."""
    from config.opt_config import SPOT_ANCHOR_SYMBOLS, SPOT_SCREENER_CONFIG, get_quarterly_window

    mp_min = int(SPOT_SCREENER_CONFIG["MP_MIN_SYMBOLS"])
    _, is_start, is_end, end_date = get_quarterly_window(None)
    broad = screen_broad_universe(is_start=is_start, is_end=is_end, fetch_end=end_date)
    if not broad:
        return

    proj = Path(project_root)
    fixed = load_screener_fixed_params(proj)
    win_sig = str(fixed.get("SIGNAL_TYPE", "ADX_BREAKOUT"))
    _logger.warning(
        "universe_screener CLI: Phase C uses SIGNAL_TYPE=%s from best_params (no Stage1). "
        "Use opt_spot.py for full Phase B winning signal.",
        win_sig,
    )

    start_date = (pd.to_datetime(end_date) - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    collector = DataCollectorSpot()
    symbol_dfs_4h: Dict[str, pd.DataFrame] = {}
    daily_dfs: Dict[str, pd.DataFrame] = {}

    symbols_to_load = list(dict.fromkeys(list(broad) + [s for s in SPOT_ANCHOR_SYMBOLS if s not in broad]))
    for sym in tqdm(symbols_to_load, desc="Load 4H/1D for refinement"):
        try:
            time.sleep(0.08)
            df4 = collector.collect_and_save(sym, "4h", start_date, end_date)
            d1 = collector.collect_and_save(sym, "1d", start_date, end_date)
            if df4 is None or d1 is None or d1.empty:
                continue
            df4_is = _slice_df_to_is(df4, is_start, is_end)
            if len(df4_is) < 100:
                continue
            symbol_dfs_4h[sym] = df4_is
            daily_dfs[sym] = d1
        except Exception as e:
            _logger.warning("Load failed %s: %s", sym, e)

    loaded_ok = [s for s in symbols_to_load if s in symbol_dfs_4h and s in daily_dfs]
    refinement_symbols = list(
        dict.fromkeys([s for s in broad if s in loaded_ok] + [s for s in SPOT_ANCHOR_SYMBOLS if s in loaded_ok])
    )
    if len(loaded_ok) < mp_min:
        _logger.error("After reload: insufficient symbols for refinement (%s).", len(loaded_ok))
        return

    screen_symbol_refinement(
        refinement_symbols,
        win_sig,
        is_end,
        symbol_dfs_4h={k: symbol_dfs_4h[k] for k in loaded_ok},
        daily_dfs={k: daily_dfs[k] for k in loaded_ok},
        phase_a_broad=list(broad),
        anchor_symbols=list(SPOT_ANCHOR_SYMBOLS),
    )


def update_config_file(
    symbols: List[str],
    clusters: Dict[str, str],
    *,
    median_adv_krw: float | None = None,
    adaptive_slippage: bool = False,
) -> None:
    config_path = Path("config/opt_config.py")
    if not config_path.exists():
        _logger.error("config/opt_config.py not found.")
        return

    content = config_path.read_text(encoding="utf-8")

    pattern_sym = r"SPOT_SYMBOLS(?:: List\[str\])?\s*=\s*\[.*?\]"
    new_sym_block = "SPOT_SYMBOLS: List[str] = [\n"
    for s in symbols:
        new_sym_block += f'    "{s}",\n'
    new_sym_block += "]"
    content = re.sub(pattern_sym, new_sym_block, content, count=1, flags=re.DOTALL)

    pattern_cls = r'"SPOT_SYMBOL_CLUSTER":\s*\{.*?\}(?=,)'
    new_cls_block = '"SPOT_SYMBOL_CLUSTER": {\n'
    for s, c in clusters.items():
        new_cls_block += f'            "{s}": "{c}",\n'
    new_cls_block += "        }"
    content = re.sub(pattern_cls, new_cls_block, content, count=1, flags=re.DOTALL)

    if adaptive_slippage and median_adv_krw is not None and np.isfinite(median_adv_krw):
        slip_pat = r"SLIPPAGE_REFERENCE_ADV_KRW:\s*float\s*=\s*[\d.eE+-]+"
        slip_rep = f"SLIPPAGE_REFERENCE_ADV_KRW: float = {median_adv_krw}"
        content, n_slip = re.subn(slip_pat, slip_rep, content, count=1)
        if n_slip != 1:
            _logger.warning("SLIPPAGE_REFERENCE_ADV_KRW pattern not updated (regex miss).")
        else:
            _logger.info("Updated SLIPPAGE_REFERENCE_ADV_KRW to %s", f"{median_adv_krw:,.0f}")

    config_path.write_text(content, encoding="utf-8")
    _logger.info("Successfully updated config/opt_config.py with new symbols and clusters.")


if __name__ == "__main__":
    screen_universe()
