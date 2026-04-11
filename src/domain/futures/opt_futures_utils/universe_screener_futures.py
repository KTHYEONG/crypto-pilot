"""
Binance USDT perpetual universe screen: ADV, ATR%, history depth, funding stability.
Optimized version: Ticker-based pre-filtering + parallel discovery + mini-BT refinement.
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
from src.domain.futures.engine_single_futures import BacktestEngineFast
from src.domain.futures.funding_utils import merge_funding_into_ohlcv
from src.domain.futures.opt_futures_utils.metrics import calc_profit_factor_from_pnl
from src.domain.futures.strategies_futures import UltimateStrategy

_logger: logging.Logger = logging.getLogger("universe_screener_futures")


def update_futures_config_file(symbols: List[str]) -> None:
    """Updates FUTURES_SYMBOLS in config/opt_config.py to persist selected universe."""
    config_path = Path("config/opt_config.py")
    if not config_path.exists():
        _logger.error("config/opt_config.py not found.")
        return

    content = config_path.read_text(encoding="utf-8")
    pattern = r"FUTURES_SYMBOLS(?::\s*List\[str\])?\s*=\s*\[.*?\]"
    new_block = "FUTURES_SYMBOLS: List[str] = [\n"
    for s in symbols:
        new_block += f'    "{s}",\n'
    new_block += "]"

    new_content = re.sub(pattern, new_block, content, count=1, flags=re.DOTALL)
    if new_content == content:
        if re.search(pattern, content, flags=re.DOTALL):
            _logger.info("FUTURES_SYMBOLS unchanged; no update needed.")
        else:
            _logger.warning("FUTURES_SYMBOLS pattern not found in opt_config.py; skipping update.")
        return

    config_path.write_text(new_content, encoding="utf-8")
    _logger.info("Successfully updated config/opt_config.py with %d FUTURES_SYMBOLS.", len(symbols))


def _median_atr_percent(
    df: pd.DataFrame,
    atr_period: int,
    tail_bars: int = 42,
) -> float:
    if df.empty or len(df) < atr_period:
        return float("nan")
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


def _calculate_adv_metrics(df: pd.DataFrame, tail_bars: int = 180) -> tuple[float, float]:
    """
    Robust liquidity proxy: median-based ADV and p25 quiet-bar floor.
    Uses quote volume from recent 4H bars (6 bars/day if 4h).
    """
    if df is None or df.empty or len(df) < 20:
        return 0.0, 0.0
    t = df.tail(int(tail_bars)).copy()
    # Calculate bar volume in USDT: close * volume
    bar_vol = (t["volume"].astype(np.float64) * t["close"].astype(np.float64)).to_numpy()
    if bar_vol.size == 0:
        return 0.0, 0.0

    med_bar = float(np.nanmedian(bar_vol))
    p25_bar = float(np.nanquantile(bar_vol, 0.25))

    # Scale median bar to daily ADV: median × (bars per day)
    # Estimate bars per day: 1440 min / (mean diff between bars)
    try:
        diff_min = float(t["datetime"].diff().median().total_seconds() / 60.0)
        bars_per_day = 1440.0 / max(1.0, diff_min)
    except Exception:
        bars_per_day = 6.0  # Fallback for 4h

    return med_bar * bars_per_day, p25_bar


def _funding_mean_abs(symbol: str, data_dir: Path, tail: int = 500) -> float:
    safe = symbol.replace("/", "_")
    path = data_dir / f"{safe}_funding.parquet"
    if not path.exists():
        return 0.0
    try:
        fr_df = pd.read_parquet(path)
    except Exception:
        return 0.0
    if fr_df.empty or "funding_rate" not in fr_df.columns:
        return 0.0
    s = fr_df["funding_rate"].astype(np.float64).tail(int(tail)).abs()
    if s.empty:
        return 0.0
    return float(s.mean())


def _screen_worker(
    sym: str,
    tf: str,
    fetch_start: str,
    end_date: str,
    cfg: Dict[str, Any],
    data_dir: Path,
) -> Dict[str, Any] | None:
    """Worker for parallel universe screening."""
    # Isolated import to avoid circular dependencies in workers
    from src.domain.futures.data_collector import DataCollector

    collector = DataCollector()

    # [[FIX]] 데이터 수집 전 메타데이터 확인하여 히스토리가 너무 짧으면 즉시 제외
    min_bars = int(cfg.get("MIN_HISTORY_BARS", 2000))
    meta = collector._load_meta()
    mk = collector._meta_key(sym, tf)
    if mk in meta and isinstance(meta[mk], dict):
        earliest = meta[mk].get("earliest_available")
        if earliest:
            try:
                # 대략적인 기간 계산 (상장일부터 종료일까지의 바 개수 추정)
                e_dt = pd.to_datetime(earliest)
                end_dt = pd.to_datetime(end_date)
                delta = end_dt - e_dt

                # 타임프레임별 바 개수 근사치
                bars_per_day = {"1h": 24, "4h": 6, "1d": 1}.get(tf, 6)
                est_bars = delta.days * bars_per_day
                if est_bars < min_bars:
                    _logger.debug(
                        f"Skipping {sym}: Estimated bits {est_bars} < {min_bars} (Listed: {earliest})"
                    )
                    return None
            except Exception:
                ...

    try:
        collector.ensure_funding_data(sym, fetch_start, end_date)
        df = collector.collect_and_save(sym, tf, fetch_start, end_date)
        df = merge_funding_into_ohlcv(sym, df, data_dir)
    except Exception:
        return None

    if df is None or df.empty or len(df) < min_bars:
        return None

    adv, p25_vol = _calculate_adv_metrics(df)
    atr_p = int(cfg["SCREENER_ATR_PERIOD"])
    atr_pct = _median_atr_percent(df, atr_p)
    fund_m = _funding_mean_abs(sym, data_dir)

    return {
        "symbol": sym,
        "adv": adv,
        "p25_vol": p25_vol,
        "atr_pct": atr_pct,
        "fund_m": fund_m,
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
    Phase A: Optimized ADV + ATR% band + min bars + funding stability.
    Uses ticker-based pre-filtering to avoid heavy downloads for illiquid coins.
    """
    dd = data_dir if data_dir is not None else FUTURES_DATA_DIR
    min_adv = float(cfg["ADV_MIN_USDT_DAY"])
    atr_min = float(cfg["SCREENER_ATR_PCT_MIN"])
    atr_max = float(cfg["SCREENER_ATR_PCT_MAX"])
    fund_thr = float(cfg["FUNDING_EXTREME_THRESHOLD"])
    top_k = int(cfg.get("BROAD_POOL_K", cfg["CANDIDATES_TOP_K"]))
    anchors = set(FUTURES_ANCHOR_SYMBOLS)

    # 1. Dynamic Discovery + Ticker-based Pre-filter
    _logger.info("Discovering symbols and applying ticker pre-filter...")
    try:
        tickers = collector.client.exchange.fetch_tickers()

        # Fast prune: must have at least 20% of min_adv in last 24h to even be considered
        pre_prune_threshold = min_adv * 0.2
        valid_candidates = []

        pool_set = set(candidate_pool) if candidate_pool else None

        for sym, t in tickers.items():
            if not t or t.get("active") is False:
                continue

            is_usdt_perp = sym.endswith("/USDT") or sym.endswith("/USDT:USDT")
            if not is_usdt_perp:
                continue

            norm_sym = sym.split(":")[0]
            if pool_set and norm_sym not in pool_set:
                continue

            if norm_sym in anchors:
                valid_candidates.append(norm_sym)
                continue

            vol_24h = float(t.get("quoteVolume") or 0.0)
            if vol_24h >= pre_prune_threshold:
                valid_candidates.append(norm_sym)

        valid_candidates = list(dict.fromkeys(valid_candidates))
        _logger.info(
            "Ticker filter: Found %d symbols meeting %s USDT/day threshold",
            len(valid_candidates),
            f"{pre_prune_threshold:,.0f}",
        )
        candidate_pool = valid_candidates
    except Exception as exc:
        _logger.warning("Ticker-based pre-filter failed: %s. Using provided pool.", exc)
        if not candidate_pool:
            candidate_pool = list(FUTURES_ANCHOR_SYMBOLS) + list(FUTURES_DYNAMIC_CANDIDATE_POOL)

    # 2. Parallel History Screening
    def get_mp_ctx():

        if sys.platform == "win32":
            return mp.get_context("spawn")
        try:
            return mp.get_context("fork")
        except ValueError:
            return mp.get_context("spawn")

    n_workers = max(1, min(int(os.cpu_count() or 4), 8))
    _logger.info(
        "Phase A: Screening %d symbols in parallel (%d workers)...",
        len(candidate_pool),
        n_workers,
    )

    worker_fn = partial(
        _screen_worker,
        tf=tf,
        fetch_start=fetch_start,
        end_date=end_date,
        cfg=cfg,
        data_dir=dd,
    )

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_mp_ctx()) as pool:
        results = list(
            tqdm(
                pool.map(worker_fn, candidate_pool),
                total=len(candidate_pool),
                desc="[Universe Screen]",
            )
        )

    rows: List[Dict[str, Any]] = []
    min_p25 = float(cfg.get("SCREENER_MIN_P25_BAR_USDT", 0.0))

    for r in results:
        if r is None:
            continue
        sym = r["symbol"]
        adv = r["adv"]
        p25 = r["p25_vol"]
        atr_pct = r["atr_pct"]
        fund_m = r["fund_m"]

        if sym in anchors:
            rows.append({"symbol": sym, "adv": adv, "atr_pct": atr_pct, "anchor": True})
            continue

        if adv < min_adv:
            continue
        if min_p25 > 0 and p25 < min_p25:
            continue
        if not np.isfinite(atr_pct) or atr_pct < atr_min or atr_pct > atr_max:
            continue
        if fund_m > fund_thr:
            continue

        rows.append({"symbol": sym, "adv": adv, "atr_pct": atr_pct, "anchor": False})

    non_anchors = [r for r in rows if not r.get("anchor")]
    non_anchors.sort(key=lambda x: -float(x["adv"]))
    picked_non = [str(r["symbol"]) for r in non_anchors[:top_k]]

    out: List[str] = []
    for s in FUTURES_ANCHOR_SYMBOLS:
        # Check if anchor exists in results
        if any(r["symbol"] == s for r in rows) and s not in out:
            out.append(s)
    for s in picked_non:
        if s not in out:
            out.append(s)

    _logger.info("Phase A complete: %d symbols passed structural gates.", len(rows))
    return out, len(rows)


def _correlation_gate(
    final_list: List[str],
    data_maps: Dict[str, Dict[str, Any]],
    tf: str,
    threshold: float = 0.85,
) -> List[str]:
    """
    고상관 심볼 쌍 감지 후 경고 로그 출력.
    앵커 심볼(BTC/ETH/SOL)은 제거하지 않고 non-anchor만 교체 후보 마킹.
    실제 제거는 호출자가 판단 — 현재는 경고 only.
    """
    rets: Dict[str, pd.Series] = {}
    for sym in final_list:
        df = data_maps.get(sym, {}).get(tf)
        if df is not None and not df.empty and len(df) >= 60:
            rets[sym] = df["close"].pct_change().tail(180).dropna()

    warned_pairs: List[Tuple[str, str]] = []
    for a, b in itertools.combinations(final_list, 2):
        if a not in rets or b not in rets:
            continue
        common = rets[a].index.intersection(rets[b].index)
        if len(common) < 30:
            continue
        rho = float(rets[a].loc[common].corr(rets[b].loc[common]))
        if rho > threshold:
            warned_pairs.append((a, b))
            _logger.warning(
                "  [CORR WARN] %s vs %s: rho=%.2f >= %.2f threshold. "
                "Portfolio diversification may be illusory.",
                a,
                b,
                rho,
                threshold,
            )

    if not warned_pairs:
        _logger.info("  [CORR OK] All symbol pairs below rho=%.2f threshold.", threshold)
    return final_list  # 경고만 발행, 리스트 변경 없음


def _walk_forward_passes(
    sym: str,
    df_tf: pd.DataFrame,
    df_1d: pd.DataFrame,
    params: Dict[str, Any],
    slippage_mult: float,
    n_folds: int = 3,
    min_pf: float = 1.1,
    min_trades: int = 4,
) -> tuple[int, int, float, float]:
    """
    n_folds 분할 walk-forward mini-BT 수행.
    Returns: (pass_count, total_folds, mean_pf, mean_ret)

    각 fold: train = 앞 2/3, test = 뒤 1/3 (expanding window 아닌 순수 3분할).
    fold 통과 기준: pf >= min_pf AND trades >= min_trades AND ret >= -10%.
    """
    n = len(df_tf)
    if n < n_folds * 30:
        return 0, n_folds, 0.0, -100.0

    fold_size = n // n_folds
    pass_count = 0
    pf_list: List[float] = []
    ret_list: List[float] = []

    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n

        fold_tf = df_tf.iloc[start:end].reset_index(drop=True)
        # 1D 대응 구간: datetime 범위 기준으로 슬라이스
        if df_1d is not None and not df_1d.empty and "datetime" in fold_tf.columns:
            fold_start_dt = fold_tf["datetime"].iloc[0]
            fold_end_dt = fold_tf["datetime"].iloc[-1]
            fold_1d = df_1d[
                (df_1d["datetime"] >= fold_start_dt) & (df_1d["datetime"] <= fold_end_dt)
            ].reset_index(drop=True)
            if len(fold_1d) < 10:
                fold_1d = df_1d.reset_index(drop=True)
        else:
            fold_1d = df_1d

        pf, n_tr, ret = _run_mini_backtest_futures(
            sym, fold_tf, fold_1d, params, slippage_mult=slippage_mult
        )
        pf_list.append(pf)
        ret_list.append(ret)
        if pf >= min_pf and n_tr >= min_trades and ret >= -10.0:
            pass_count += 1

    mean_pf = float(np.mean(pf_list)) if pf_list else 0.0
    mean_ret = float(np.mean(ret_list)) if ret_list else -100.0
    return pass_count, n_folds, mean_pf, mean_ret


def _run_mini_backtest_futures(
    sym: str,
    df_tf: pd.DataFrame,
    df_1d: pd.DataFrame,
    params: Dict[str, Any],
    *,
    slippage_mult: float = 1.0,
) -> tuple[float, int, float]:
    """Quick IS backtest for symbol refinement."""
    strategy = UltimateStrategy(name=f"Screener_{sym}", params=params)
    try:
        # 1. Generate signals (indicators required by Fast engine)
        df_sig = strategy.generate_signals(df_tf.copy())

        # 2. Run engine
        engine = BacktestEngineFast(
            hourly_df=df_sig,
            daily_df=df_1d,
            strategy=strategy,
            initial_balance=10000.0,
        )
        engine.slippage_rate = float(SLIPPAGE_RATE) * float(slippage_mult)
        result = engine.run()
        trades = result.get("trades_df", pd.DataFrame())
        if trades is None or trades.empty:
            return 0.0, 0, -100.0
        pf = calc_profit_factor_from_pnl(trades["pnl"])
        ret = float(result.get("total_return_pct", -100.0))
        return float(pf), len(trades), ret
    except Exception as exc:
        _logger.debug("Mini-backtest failed for %s: %s", sym, exc)
        return 0.0, 0, -100.0


def screen_futures_symbol_refinement(
    broad_candidates: List[str],
    anchors: List[str],
    cfg: Dict[str, Any],
    *,
    data_maps: Dict[str, Dict[str, Any]] | None = None,
    winning_params: Dict[str, Any] | None = None,
) -> List[str]:
    """
    Phase C: Refinement via mini-backtest (like Spot).
    Uses winning combo from Phase B to filter broad candidates.
    """
    mp_min = int(cfg["MP_MIN_SYMBOLS"])
    mp_max = int(cfg["MP_MAX_SYMBOLS"])

    if data_maps is None or winning_params is None:
        _logger.warning("Refinement skipped (no data/params); using ADV-based fallback.")
        out = []
        for a in anchors:
            if a not in out:
                out.append(a)
        for s in broad_candidates:
            if s not in out:
                out.append(s)
            if len(out) >= mp_max:
                break
        return out[:mp_max]

    _logger.info(
        "Phase C: Refining %d symbols via 3-fold walk-forward mini-BT...",
        len(broad_candidates),
    )

    # Walk-forward 설정: n_folds 분할, 과반수(ceil(n_folds/2)) fold 통과 시 선정
    n_folds = 3
    required_passes = 2  # n_folds=3에서 다수결

    scored: List[Dict[str, Any]] = []
    marginal_fallback: List[Dict[str, Any]] = []
    tf = winning_params.get("TIMEFRAME", "4h")
    pf_prem = float(OPT_FUTURES_CONFIG.get("FUTURES_NON_ANCHOR_MIN_PF_PREMIUM", 0.0))
    slip_mult = float(OPT_FUTURES_CONFIG.get("FUTURES_NON_ANCHOR_SLIPPAGE_MULT", 1.0))
    max_non_anchor = int(OPT_FUTURES_CONFIG.get("FUTURES_NON_ANCHOR_MAX_COUNT", 1))
    anchor_set = {str(a) for a in anchors}

    base_min_pf = float(cfg.get("SCREENER_MIN_PF", 1.1))
    min_trades_total = int(cfg.get("SCREENER_MIN_TRADES_DYNAMIC", 12))
    # fold당 최소 거래 수: 전체 최소의 1/n_folds (최소 4)
    min_trades_per_fold = max(4, min_trades_total // n_folds)

    for sym in broad_candidates:
        symbol_data = data_maps.get(sym, {})
        df = symbol_data.get(tf)
        d1 = symbol_data.get("1d")
        if df is None or df.empty or d1 is None or d1.empty:
            continue

        is_anchor = sym in anchor_set
        sm = 1.0 if is_anchor else slip_mult
        fold_min_pf = 1.0 if is_anchor else max(1.1, 1.0 + pf_prem)

        pass_count, total_folds, mean_pf, mean_ret = _walk_forward_passes(
            sym,
            df,
            d1,
            winning_params,
            sm,
            n_folds=n_folds,
            min_pf=fold_min_pf,
            min_trades=min_trades_per_fold,
        )

        overall_pf_gate = 1.0 if is_anchor else max(base_min_pf, 1.1 + pf_prem)

        if pass_count < required_passes or mean_pf < overall_pf_gate or mean_ret < -10.0:
            _logger.debug(
                "Refinement rejected %s: WF passes=%d/%d, mean_PF=%.2f, mean_Ret=%.1f%%",
                sym,
                pass_count,
                total_folds,
                mean_pf,
                mean_ret,
            )
            # [개선 5] fallback 기준 강화: pf>=1.15, ret>=-2.0, trades >= max(8, min-2)
            fallback_min_tr = max(8, min_trades_total - 4)
            if (
                not is_anchor
                and pass_count >= 1  # 최소 1 fold 통과
                and mean_pf >= 1.15  # 기존 1.0 → 1.15
                and mean_ret >= -2.0  # 기존 -5.0 → -2.0
                and min_trades_per_fold * n_folds >= fallback_min_tr
            ):
                marginal_fallback.append(
                    {
                        "symbol": sym,
                        "pf": mean_pf,
                        "n_trades": min_trades_per_fold * n_folds,
                        "ret": mean_ret,
                        "passes": pass_count,
                    }
                )
            continue

        ret_mult = max(0.1, 1.0 + mean_ret / 100.0)
        # WF 통과 횟수도 score에 반영 (pass_count/total_folds 보너스)
        wf_bonus = pass_count / total_folds
        scored.append(
            {
                "symbol": sym,
                "pf": mean_pf,
                "ret": mean_ret,
                "passes": pass_count,
                "score": mean_pf * np.log1p(min_trades_per_fold * n_folds) * ret_mult * wf_bonus,
            }
        )
        _logger.info(
            "  [WF OK] %s: passes=%d/%d, mean_PF=%.2f, mean_Ret=%.1f%%",
            sym,
            pass_count,
            total_folds,
            mean_pf,
            mean_ret,
        )

    scored.sort(key=lambda x: x["score"], reverse=True)

    final_list: List[str] = []
    # 1. Anchors first (항상 포함)
    for a in anchors:
        if a not in final_list:
            final_list.append(a)

    # 2. Top performers from broad pool
    non_anchor_added = 0
    for item in scored:
        s = str(item["symbol"])
        if s in final_list:
            continue
        if s not in anchor_set and non_anchor_added >= max_non_anchor:
            continue
        final_list.append(s)
        if s not in anchor_set:
            non_anchor_added += 1
        if len(final_list) >= mp_max:
            break

    # 3. [개선 3] Correlation gate (경고 발행)
    _correlation_gate(final_list, data_maps, tf, threshold=0.85)

    if len(final_list) < mp_min:
        marginal_fallback.sort(key=lambda x: x["ret"], reverse=True)
        for item in marginal_fallback:
            s = str(item["symbol"])
            if s not in final_list:
                _logger.warning(
                    "[MARGINAL] Fallback adding %s (mean_pf=%.2f, WF passes=%d/%d, ret=%.1f%%); "
                    "insufficient qualified symbols.",
                    s,
                    item["pf"],
                    item["passes"],
                    n_folds,
                    item["ret"],
                )
                final_list.append(s)
            if len(final_list) >= mp_min:
                break
        if len(final_list) < mp_min:
            _logger.warning(
                "Refinement: only %d symbols qualified (< mp_min=%d); "
                "proceeding with reduced symbol set.",
                len(final_list),
                mp_min,
            )

    _logger.info("Refinement complete: %d symbols selected: %s", len(final_list), final_list)
    update_futures_config_file(final_list[:mp_max])
    return final_list[:mp_max]
