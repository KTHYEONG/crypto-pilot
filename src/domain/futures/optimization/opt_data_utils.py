from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUTURES_DATA_DIR
from src.core.optimization.opt_utils import compute_segment_merge_index
from src.domain.futures.data_loader import DataCollector, summarize_dataframe_integrity
from src.domain.futures.strategy_runtime.bridge import _enrich_with_gp_features
from src.domain.futures.optimization.dashboard import REGIME_NAMES
from src.domain.futures.optimization.optimizer import inject_cs_momentum_ranks

_logger: logging.Logger = logging.getLogger("opt_data_utils")
_SUFFICIENCY_LOG_DIR = Path("logs/futures/data")


def _tf_delta(tf: str) -> pd.Timedelta:
    tf_l = str(tf).lower()
    if tf_l == "1h":
        return pd.Timedelta(hours=1)
    if tf_l == "4h":
        return pd.Timedelta(hours=4)
    if tf_l == "1d":
        return pd.Timedelta(days=1)
    if tf_l == "1m":
        return pd.Timedelta(minutes=1)
    return pd.Timedelta(hours=4)


def _bars_between(start: str, end: str, tf: str) -> int:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    if e <= s:
        return 0
    return int((e - s) / _tf_delta(tf))


def _resolve_warmup_bars(tf: str) -> int:
    bars_per_day = {"1h": 24, "4h": 6, "1d": 1}.get(str(tf).lower(), 6)
    lookback = int(OPT_FUTURES_CONFIG.get("FUTURES_MOMENTUM_LOOKBACK", 252))
    cov = int(OPT_FUTURES_CONFIG.get("FUTURES_COV_LOOKBACK", 252))
    sigma = int(OPT_FUTURES_CONFIG.get("FUTURES_COMPOSER_SIGMA_LOOKBACK", 252))
    atr = int(OPT_FUTURES_CONFIG.get("FUTURES_ATR_PERIOD", 14))
    embargo = int(OPT_FUTURES_CONFIG.get("FUTURES_EMBARGO_BARS", 0))
    platt = int(OPT_FUTURES_CONFIG.get("FUTURES_PLATT_MIN_TRAIN_BARS", 120))
    min_membership_warm = int(OPT_FUTURES_CONFIG.get("FUTURES_MEMBERSHIP_WARM_DAYS", 42)) * bars_per_day
    return max(lookback, cov, sigma, atr, embargo, platt, min_membership_warm)


def evaluate_symbol_data_sufficiency(
    *,
    symbol: str,
    tf: str,
    symbol_map: dict[str, Any],
    fetch_start: str,
    is_start: str,
    oos_start: str,
    oos_end: str,
    require_exec_1m: bool,
    warmup_bars_required: int,
) -> dict[str, Any]:
    frame = symbol_map.get(tf)
    if not isinstance(frame, pd.DataFrame) or frame.empty or "datetime" not in frame.columns:
        return {"symbol": symbol, "tf": tf, "pass": False, "reason": "missing_tf_frame"}
    dt = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dropna().sort_values()
    if dt.empty:
        return {"symbol": symbol, "tf": tf, "pass": False, "reason": "invalid_datetime"}

    first_dt = dt.iloc[0]
    last_dt = dt.iloc[-1]
    fetch_ok = first_dt <= pd.Timestamp(fetch_start, tz="UTC") and last_dt >= pd.Timestamp(
        oos_end, tz="UTC"
    )

    is_start_ts = pd.Timestamp(is_start, tz="UTC")
    oos_start_ts = pd.Timestamp(oos_start, tz="UTC")
    oos_end_ts = pd.Timestamp(oos_end, tz="UTC")
    bars_before_is = int((dt < is_start_ts).sum())
    actual_is_bars = int(((dt >= is_start_ts) & (dt < oos_start_ts)).sum())
    actual_oos_bars = int(((dt >= oos_start_ts) & (dt <= oos_end_ts)).sum())
    required_is_bars = _bars_between(is_start, oos_start, tf)
    required_oos_bars = _bars_between(oos_start, oos_end, tf)
    min_is_bars = int(required_is_bars * 0.95)
    min_oos_bars = int(required_oos_bars * 0.95)
    warmup_ok = bars_before_is >= warmup_bars_required

    exec_1m_ok = True
    exec_1m_cov = 1.0
    if require_exec_1m:
        exec_1m = symbol_map.get("exec_1m")
        if not isinstance(exec_1m, pd.DataFrame) or exec_1m.empty or "datetime" not in exec_1m.columns:
            exec_1m_ok = False
            exec_1m_cov = 0.0
        else:
            exec_dt = pd.to_datetime(exec_1m["datetime"], utc=True, errors="coerce").dropna()
            actual_1m = int(
                ((exec_dt >= pd.Timestamp(fetch_start, tz="UTC")) & (exec_dt <= oos_end_ts)).sum()
            )
            required_1m = max(1, _bars_between(fetch_start, oos_end, "1m"))
            exec_1m_cov = float(actual_1m / required_1m)
            exec_1m_ok = exec_1m_cov >= 0.95

    pass_flag = bool(fetch_ok and warmup_ok and actual_is_bars >= min_is_bars and actual_oos_bars >= min_oos_bars and exec_1m_ok)
    reason = "ok"
    if not fetch_ok:
        reason = "fetch_window_short"
    elif not warmup_ok:
        reason = "warmup_insufficient"
    elif actual_is_bars < min_is_bars:
        reason = "is_coverage_short"
    elif actual_oos_bars < min_oos_bars:
        reason = "oos_coverage_short"
    elif not exec_1m_ok:
        reason = "missing_exec_1m"

    return {
        "symbol": symbol,
        "tf": tf,
        "pass": pass_flag,
        "reason": reason,
        "fetch_ok": fetch_ok,
        "warmup_bars": warmup_bars_required,
        "bars_before_is": bars_before_is,
        "required_is_bars": required_is_bars,
        "actual_is_bars": actual_is_bars,
        "required_oos_bars": required_oos_bars,
        "actual_oos_bars": actual_oos_bars,
        "exec_1m_coverage": exec_1m_cov,
        "first_dt": first_dt.isoformat(),
        "last_dt": last_dt.isoformat(),
    }


def filter_symbols_by_data_sufficiency(
    *,
    tf: str,
    data_maps: dict[str, dict[str, Any]],
    oos_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    fetch_start: str,
    is_start: str,
    oos_start: str,
    oos_end: str,
    require_exec_1m: bool,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]], pd.DataFrame, int]:
    warmup_bars = _resolve_warmup_bars(tf)
    rows: list[dict[str, Any]] = []
    kept: list[str] = []
    for symbol in valid_symbols:
        rec = evaluate_symbol_data_sufficiency(
            symbol=symbol,
            tf=tf,
            symbol_map=oos_data_maps.get(symbol, {}),
            fetch_start=fetch_start,
            is_start=is_start,
            oos_start=oos_start,
            oos_end=oos_end,
            require_exec_1m=require_exec_1m,
            warmup_bars_required=warmup_bars,
        )
        rows.append(rec)
        _logger.info(
            "[DATA-SUFFICIENCY] symbol=%s tf=%s fetch_ok=%s warmup_bars=%d required_is_bars=%d actual_is_bars=%d required_oos_bars=%d actual_oos_bars=%d pass=%s reason=%s exec_1m_coverage=%.3f",
            symbol,
            tf,
            str(bool(rec.get("fetch_ok", False))).lower(),
            int(rec.get("warmup_bars", 0)),
            int(rec.get("required_is_bars", 0)),
            int(rec.get("actual_is_bars", 0)),
            int(rec.get("required_oos_bars", 0)),
            int(rec.get("actual_oos_bars", 0)),
            str(bool(rec.get("pass", False))).lower(),
            str(rec.get("reason", "unknown")),
            float(rec.get("exec_1m_coverage", 0.0)),
        )
        if bool(rec.get("pass", False)):
            kept.append(symbol)
    report_df = pd.DataFrame(rows)
    _SUFFICIENCY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not report_df.empty:
        report_df.to_parquet(_SUFFICIENCY_LOG_DIR / "data_sufficiency_report.parquet", index=False)

    kept_set = set(kept)
    filtered_is = {s: m for s, m in data_maps.items() if s in kept_set}
    filtered_oos = {s: m for s, m in oos_data_maps.items() if s in kept_set}
    return kept, filtered_is, filtered_oos, report_df, warmup_bars


def _to_unix_ms(dt: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Datetime series → Unix milliseconds int64, safe across datetime64[ms/us/ns] dtypes.

    Pandas 2.x preserves storage resolution (ms/us/ns), so astype('int64') returns
    values in the native unit. Upcasting to ns first normalizes all variants.
    """
    return pd.to_datetime(dt, utc=True).astype("datetime64[ns, UTC]").astype("int64") // 10**6


def _should_load_exec_1m(load_exec_1m: bool | None) -> bool:
    """Resolve whether execution 1m data should be loaded."""
    if load_exec_1m is not None:
        return bool(load_exec_1m)
    mode = os.getenv("FUTURES_EXECUTION_MODE", "coarse").strip().lower()
    return mode == "intrabar_1m"


def _safe_read_funding_parquet(symbol: str) -> pd.DataFrame | None:
    """Read funding parquet for symbol with defensive fallback."""
    f_path = Path(FUTURES_DATA_DIR) / f"{symbol.replace('/', '_')}_funding.parquet"
    if not f_path.exists():
        return None
    try:
        df = pd.read_parquet(f_path)
        return df if not df.empty else None
    except Exception:
        _logger.warning("Failed to load funding parquet for %s", symbol)
        return None


def _build_funding_event_arrays_1m(
    exec_df_1m: pd.DataFrame,
    funding_df: pd.DataFrame | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build 1m funding event mask/rate arrays aligned to exec 1m rows.

    Events are stamped only on exact funding timestamps (typically 8h cadence).
    Non-event bars are zero.
    """
    if exec_df_1m is None or exec_df_1m.empty or funding_df is None or funding_df.empty:
        return None, None
    if "datetime" not in exec_df_1m.columns:
        return None, None
    if "timestamp" not in funding_df.columns or "funding_rate" not in funding_df.columns:
        return None, None

    exec_ms = _to_unix_ms(exec_df_1m["datetime"]).to_numpy(dtype=np.int64, copy=False)
    event_ms = _to_unix_ms(pd.to_datetime(funding_df["timestamp"], unit="ms", utc=True)).to_numpy(
        dtype=np.int64, copy=False
    )
    event_rate = pd.to_numeric(funding_df["funding_rate"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )

    n = exec_ms.size
    if n == 0 or event_ms.size == 0:
        return None, None
    mask = np.zeros(n, dtype=np.float64)
    rate = np.zeros(n, dtype=np.float64)
    pos = np.searchsorted(exec_ms, event_ms, side="left")
    valid = (pos >= 0) & (pos < n) & (exec_ms[pos] == event_ms) & np.isfinite(event_rate)
    if not np.any(valid):
        return None, None
    pos_v = pos[valid]
    rate_v = event_rate[valid]
    np.add.at(mask, pos_v, 1.0)
    np.add.at(rate, pos_v, rate_v)
    mask = np.where(mask > 0.0, 1.0, 0.0)
    return mask, rate


_FEATURE_GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
    "price": ("open", "high", "low", "close", "ret_", "vol_", "realized_vol", "atr", "mom", "beta"),
    "funding": ("funding",),
    "oi": ("open_interest", "oi", "sum_open_interest"),
    "lsr": ("lsr", "long_short", "top_trader", "global_lsr"),
    "taker_orderflow": ("taker", "buy_sell", "imbalance", "orderflow"),
    "macro": ("macro_", "btc_", "market_", "cs_dispersion"),
    "hmm_derived": ("hmm_", "regime_", "tail_risk"),
}


def _feature_group_coverage(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if df is None or df.empty:
        for g in _FEATURE_GROUP_PATTERNS:
            out[g] = {"col_count": 0.0, "non_null_coverage": 0.0, "non_zero_coverage": 0.0}
        return out
    n_rows = max(len(df), 1)
    for group, pats in _FEATURE_GROUP_PATTERNS.items():
        cols = [c for c in df.columns if any(p in str(c).lower() for p in pats)]
        if not cols:
            out[group] = {"col_count": 0.0, "non_null_coverage": 0.0, "non_zero_coverage": 0.0}
            continue
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        out[group] = {
            "col_count": float(len(cols)),
            "non_null_coverage": float(1.0 - (sub.isna().sum().sum() / max(sub.size, 1))),
            "non_zero_coverage": float((sub.fillna(0.0) != 0.0).sum().sum() / max(sub.size, 1)),
        }
    return out


def _append_stage_integrity(
    audit: list[dict[str, Any]],
    *,
    symbol: str,
    timeframe: str,
    stage: str,
    df: pd.DataFrame,
    fillna_cols: list[str] | None = None,
) -> None:
    rec: dict[str, Any] = {"symbol": symbol, "timeframe": timeframe, "stage": stage}
    rec.update(summarize_dataframe_integrity(df, timeframe=timeframe))
    if fillna_cols:
        pre_na = df[fillna_cols].isna().sum().sum() if fillna_cols else 0
        denom = max(int(len(df) * len(fillna_cols)), 1)
        rec["pre_fillna_nan_pct"] = float(pre_na / denom)
    audit.append(rec)


def infer_regime_codes(df: pd.DataFrame) -> np.ndarray:
    """Infer HMM regime codes from probability columns."""
    n = len(df)

    def _float_col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.zeros(n, dtype=np.float64)
        # Force a writable array; some pandas-backed arrays can be read-only views.
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64, copy=True)

    bull = _float_col("hmm_prob_bull_calm")
    bull += _float_col("hmm_prob_bull_vol_up")
    bear = _float_col("hmm_prob_bear_trend")
    chop = _float_col("hmm_prob_chop")
    crisis = _float_col("hmm_prob_crisis")
    probs = np.column_stack(
        [
            np.nan_to_num(bull, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(bear, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(chop, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(crisis, nan=0.0, posinf=0.0, neginf=0.0),
        ]
    )
    return np.argmax(probs, axis=1).astype(np.int64, copy=False)


def compute_oos_regime_attribution(
    oos_port: dict[str, Any],
    oos_data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> dict[str, Any]:
    """Compute performance attribution by market regime."""
    time_counts = np.zeros(len(REGIME_NAMES), dtype=np.int64)
    total_symbol_bars = 0
    for sym in symbols:
        smap = oos_data_maps.get(sym, {})
        df = smap.get(tf)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        o0 = int(smap.get(f"oos_start_idx_{tf}", 0))
        o0 = max(0, min(o0, len(df)))
        oos_df = df.iloc[o0:]
        if oos_df.empty:
            continue
        rc = infer_regime_codes(oos_df)
        time_counts += np.bincount(rc, minlength=len(REGIME_NAMES))
        total_symbol_bars += int(rc.size)

    trades_df = oos_port.get("trades_df", pd.DataFrame())
    if not isinstance(trades_df, pd.DataFrame):
        trades_df = pd.DataFrame()
    n_trades = len(trades_df)
    trade_codes = np.full(n_trades, -1, dtype=np.int64)

    aligned_master = oos_port.get("aligned_master_index")
    full_signal_dfs = oos_port.get("full_signal_dfs", {})
    if n_trades > 0 and isinstance(full_signal_dfs, dict):
        aligned_ns = np.array([], dtype=np.int64)
        if isinstance(aligned_master, pd.Series):
            aligned_ns = (
                pd.to_datetime(aligned_master, errors="coerce")
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64, copy=False)
            )
        elif aligned_master is not None:
            aligned_idx = pd.Index(np.asarray(aligned_master).ravel())
            aligned_ns = (
                pd.to_datetime(aligned_idx, errors="coerce")
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64, copy=False)
            )
        if aligned_ns.size > 0:
            nat_i64 = np.iinfo(np.int64).min
            sym_map: dict[str, pd.Series] = {}
            for sym in np.unique(trades_df["symbol"].astype(str).to_numpy()):
                sdf = full_signal_dfs.get(sym)
                if not isinstance(sdf, pd.DataFrame) or sdf.empty or "datetime" not in sdf.columns:
                    continue
                dt_ns = (
                    pd.to_datetime(sdf["datetime"], errors="coerce")
                    .to_numpy(dtype="datetime64[ns]")
                    .astype(np.int64, copy=False)
                )
                valid_dt = dt_ns != nat_i64
                if not valid_dt.any():
                    continue
                rc = infer_regime_codes(sdf)
                sr = pd.Series(rc[valid_dt], index=dt_ns[valid_dt], dtype=np.int64)
                sym_map[sym] = sr.groupby(level=0).last()

            entry_idx = pd.to_numeric(trades_df["entry_idx"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            sym_arr = trades_df["symbol"].astype(str).to_numpy()
            for sym, sr in sym_map.items():
                pos = np.where(sym_arr == sym)[0]
                if pos.size == 0:
                    continue
                e = entry_idx[pos]
                finite = np.isfinite(e)
                if not finite.any():
                    continue
                loc = e[finite].astype(np.int64)
                inb = (loc >= 0) & (loc < aligned_ns.size)
                if not inb.any():
                    continue
                valid_rows = pos[finite][inb]
                keys = aligned_ns[loc[inb]]
                mapped = sr.reindex(keys).to_numpy()
                ok = ~pd.isna(mapped)
                if ok.any():
                    trade_codes[valid_rows[ok]] = mapped[ok].astype(np.int64, copy=False)

    pnl = (
        pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if n_trades > 0
        else np.array([], dtype=np.float64)
    )
    side_arr = (
        trades_df["side"].astype(str).to_numpy() if n_trades > 0 else np.array([], dtype=object)
    )
    entry_idx = (
        pd.to_numeric(trades_df["entry_idx"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        if n_trades > 0
        else np.array([], dtype=np.int64)
    )
    sym_arr = (
        trades_df["symbol"].astype(str).to_numpy() if n_trades > 0 else np.array([], dtype=object)
    )

    regime_metrics: dict[str, dict[str, float | int]] = {}
    for ridx, rname in enumerate(REGIME_NAMES):
        r_mask = trade_codes == ridx
        r_count = int(r_mask.sum())
        if r_count > 0:
            r_pnl = pnl[r_mask]
            gains = float(r_pnl[r_pnl > 0.0].sum())
            losses = abs(float(r_pnl[r_pnl < 0.0].sum()))
            if losses == 0.0:
                pf = 5.0 if gains > 0.0 else 1.0
            else:
                pf = gains / losses
            win_rate = float(np.mean(r_pnl > 0.0) * 100.0)
            avg_pnl = float(np.mean(r_pnl))
        else:
            win_rate = 0.0
            pf = 1.0
            avg_pnl = 0.0
        time_pct = float(100.0 * time_counts[ridx] / max(total_symbol_bars, 1))
        regime_metrics[rname] = {
            "time_pct": time_pct,
            "trade_count": r_count,
            "win_rate": win_rate,
            "profit_factor": float(pf),
            "avg_pnl": avg_pnl,
        }

    chop_idx = REGIME_NAMES.index("chop")
    chop_mask = trade_codes == chop_idx
    chop_trade_count = int(chop_mask.sum())
    chop_losses = abs(float(pnl[chop_mask & (pnl < 0.0)].sum())) if n_trades > 0 else 0.0
    total_losses = abs(float(pnl[pnl < 0.0].sum())) if n_trades > 0 else 0.0
    chop_loss_share = float(chop_losses / max(total_losses, 1e-12))
    chop_trade_share = float(chop_trade_count / max(n_trades, 1))

    flip_count = 0
    flip_pairs = 0
    if n_trades > 1 and chop_trade_count > 1:
        for sym in np.unique(sym_arr[chop_mask]):
            s_mask = (sym_arr == sym) & chop_mask
            if int(s_mask.sum()) < 2:
                continue
            ord_idx = np.argsort(entry_idx[s_mask], kind="mergesort")
            s_side = side_arr[s_mask][ord_idx]
            flip_count += int(np.sum(s_side[1:] != s_side[:-1]))
            flip_pairs += int(max(s_side.size - 1, 0))
    chop_flip_proxy = float(flip_count / max(flip_pairs, 1))

    return {
        "regime_metrics": regime_metrics,
        "chop_loss_share": chop_loss_share,
        "chop_trade_share": chop_trade_share,
        "chop_flip_proxy": chop_flip_proxy,
        "chop_flip_proxy_label": "side_switch_rate_within_chop_trades_by_symbol (proxy)",
        "trade_regime_coverage_pct": float(
            100.0 * np.mean(trade_codes >= 0) if n_trades > 0 else 0.0
        ),
    }


def assert_oos_gp_signal_alive(
    oos_data_maps: dict[str, dict[str, Any]], valid_symbols: list[str], tf: str
) -> None:
    """Verify that ML signals are not dead in the OOS period."""
    for sym in valid_symbols[: min(5, len(valid_symbols))]:
        df = oos_data_maps[sym][tf]
        if "alpha_long_00" not in df.columns:
            raise RuntimeError(f"Pre-OOS: {sym} missing alpha_long_00.")
        gp = df["alpha_long_00"]
        if not pd.api.types.is_numeric_dtype(gp):
            raise RuntimeError(f"Pre-OOS: {sym} alpha_long_00 non-numeric dtype={gp.dtype}")
        o0 = int(oos_data_maps[sym][f"oos_start_idx_{tf}"])
        oos_std = float(pd.to_numeric(gp.iloc[o0:], errors="coerce").std(ddof=0) or 0.0)
        if oos_std < 1e-6:
            raise RuntimeError(f"Pre-OOS: {sym} OOS alpha_long_00 std={oos_std:.2e} (dead signal).")


def load_single_symbol_data(
    sym: str,
    tf: str,
    fetch_start: str,
    start: str,
    is_end: str,
    end: str,
    skip_metrics: bool = False,
    target_tfs: list[str] | None = None,
    load_exec_1m: bool | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, bool]:
    """Load and enrich data for a single symbol across multiple timeframes."""
    try:
        temp_is: dict[str, Any] = {}
        temp_oos: dict[str, Any] = {}
        integrity_audit: list[dict[str, Any]] = []
        insufficient = False
        collector = DataCollector()

        tfs_to_load = set(target_tfs) if target_tfs else {tf, "1d", "1h", "4h"}
        use_exec_1m = _should_load_exec_1m(load_exec_1m)

        # [Optimization] Pre-load funding and metrics data once to avoid redundant I/O per TF
        funding_df = None
        metrics_df = None
        if not skip_metrics:
            funding_df = _safe_read_funding_parquet(sym)

            m_path = Path(FUTURES_DATA_DIR) / f"{sym.replace('/', '_')}_metrics.parquet"
            if m_path.exists():
                try:
                    metrics_df = pd.read_parquet(m_path)
                except Exception:
                    _logger.warning("Failed to load metrics data for %s", sym)

        for tf_l in tfs_to_load:
            raw_df = collector.collect_and_save(sym, tf_l, fetch_start, end)
            if raw_df is None or raw_df.empty:
                insufficient = True
                break
            _append_stage_integrity(
                integrity_audit,
                symbol=sym,
                timeframe=tf_l,
                stage="raw",
                df=raw_df,
            )

            if "datetime" not in raw_df.columns:
                raw_df = raw_df.reset_index()
                if "datetime" not in raw_df.columns and len(raw_df.columns) > 0:
                    raw_df = raw_df.rename(columns={str(raw_df.columns[0]): "datetime"})

            try:
                # [Optimization] Use localized merge logic to benefit from pre-loaded data
                df = raw_df.copy()
                if funding_df is not None and not funding_df.empty:
                    df["timestamp"] = _to_unix_ms(df["datetime"])
                    f_tmp = funding_df.copy()
                    f_tmp["timestamp"] = _to_unix_ms(
                        pd.to_datetime(f_tmp["timestamp"], unit="ms", utc=True)
                    )
                    exclude_fr = ["datetime", "symbol"]
                    cols_fr = [c for c in f_tmp.columns if c not in exclude_fr]
                    df = pd.merge_asof(
                        df.sort_values("timestamp"),
                        f_tmp[cols_fr].sort_values("timestamp"),
                        on="timestamp",
                        direction="backward",
                    )

                if metrics_df is not None and not metrics_df.empty:
                    if "timestamp" not in df.columns:
                        df["timestamp"] = _to_unix_ms(df["datetime"])
                    m_tmp = metrics_df.copy()
                    m_tmp["timestamp"] = _to_unix_ms(m_tmp["datetime"])
                    exclude_m = ["timestamp", "datetime", "create_time", "symbol"]
                    cols_m = [c for c in m_tmp.columns if c not in exclude_m]
                    df = pd.merge_asof(
                        df.sort_values("timestamp"),
                        m_tmp[["timestamp", *cols_m]].sort_values("timestamp"),
                        on="timestamp",
                        direction="backward",
                    )
                _append_stage_integrity(
                    integrity_audit,
                    symbol=sym,
                    timeframe=tf_l,
                    stage="merged",
                    df=df,
                )

                # Enrich with GP features
                if not skip_metrics:
                    df = _enrich_with_gp_features(df, tf=tf_l)
                    _append_stage_integrity(
                        integrity_audit,
                        symbol=sym,
                        timeframe=tf_l,
                        stage="engineered",
                        df=df,
                    )
            except Exception as e:
                _logger.error("[%s] Merge/Enrich failed: %s", sym, e)
                insufficient = True
                break

            if df is None or df.empty or "datetime" not in df.columns:
                insufficient = True
                break

            df.reset_index(drop=True, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

            is_start_dt = pd.Timestamp(start, tz="UTC")
            is_end_dt = pd.Timestamp(is_end, tz="UTC")

            is_mask = df["datetime"] < is_end_dt
            is_end_idx = int(is_mask.to_numpy().sum())

            # [Dynamic Quality Gate]
            min_bars_map = {"1h": 2000, "4h": 500, "1d": 300}
            min_bars_threshold = min_bars_map.get(tf_l, 300)

            if is_end_idx < min_bars_threshold:
                _logger.debug(
                    "[%s] %s history too short (%d < %d)",
                    sym,
                    tf_l,
                    is_end_idx,
                    min_bars_threshold,
                )
                insufficient = True
                break

            temp_is[tf_l] = df.iloc[:is_end_idx].copy()
            mask = temp_is[tf_l]["datetime"] >= is_start_dt
            temp_is[f"is_start_idx_{tf_l}"] = int(mask.to_numpy().argmax()) if mask.any() else 0
            temp_oos[tf_l] = df
            mask_oos = df["datetime"] >= is_end_dt
            idx_oos = int(mask_oos.to_numpy().argmax()) if mask_oos.any() else len(df)
            temp_oos[f"oos_start_idx_{tf_l}"] = idx_oos

        if use_exec_1m:
            try:
                exec_1m = collector.collect_1m_ohlcv(sym, fetch_start, end)
            except Exception as e:
                _logger.warning("[%s] collect_1m_ohlcv failed: %s", sym, e)
                exec_1m = pd.DataFrame()
            if isinstance(exec_1m, pd.DataFrame) and not exec_1m.empty:
                if "datetime" not in exec_1m.columns:
                    exec_1m = exec_1m.reset_index()
                    if "datetime" not in exec_1m.columns and len(exec_1m.columns) > 0:
                        exec_1m = exec_1m.rename(columns={str(exec_1m.columns[0]): "datetime"})
                if "datetime" in exec_1m.columns:
                    exec_1m = exec_1m.copy()
                    exec_1m["datetime"] = pd.to_datetime(exec_1m["datetime"], utc=True)
                    exec_1m.sort_values("datetime", inplace=True)
                    exec_1m.reset_index(drop=True, inplace=True)
                    temp_is["exec_1m"] = exec_1m[exec_1m["datetime"] < is_end_dt].copy()
                    mask_exec_is = temp_is["exec_1m"]["datetime"] >= is_start_dt
                    temp_is["is_start_idx_exec_1m"] = (
                        int(mask_exec_is.to_numpy().argmax()) if mask_exec_is.any() else 0
                    )
                    temp_oos["exec_1m"] = exec_1m
                    mask_exec_oos = exec_1m["datetime"] >= is_end_dt
                    temp_oos["oos_start_idx_exec_1m"] = (
                        int(mask_exec_oos.to_numpy().argmax())
                        if mask_exec_oos.any()
                        else len(exec_1m)
                    )

                    if funding_df is None and not skip_metrics:
                        funding_df = _safe_read_funding_parquet(sym)
                    mask_1m, rate_1m = _build_funding_event_arrays_1m(exec_1m, funding_df)
                    if mask_1m is not None and rate_1m is not None:
                        is_exec_len = len(temp_is["exec_1m"])
                        temp_is["funding_event_mask_1m"] = mask_1m[:is_exec_len].copy()
                        temp_is["funding_rate_event_1m"] = rate_1m[:is_exec_len].copy()
                        temp_oos["funding_event_mask_1m"] = mask_1m.copy()
                        temp_oos["funding_rate_event_1m"] = rate_1m.copy()

        if insufficient:
            return sym, None, None, True

        temp_is[f"merge_idx_{tf}"] = compute_segment_merge_index(temp_is[tf], temp_is["1d"])
        temp_oos[f"merge_idx_{tf}"] = compute_segment_merge_index(temp_oos[tf], temp_oos["1d"])
        temp_is["integrity_audit"] = integrity_audit
        temp_oos["integrity_audit"] = integrity_audit
        tf_df = temp_oos.get(tf)
        if isinstance(tf_df, pd.DataFrame) and not tf_df.empty:
            temp_oos["feature_group_coverage"] = _feature_group_coverage(tf_df)
        return sym, temp_is, temp_oos, False
    except Exception as e:
        _logger.debug("[%s] Critical load failure: %s", sym, e)
        return sym, None, None, True


def load_futures_data_maps_for_symbols(
    symbols: list[str],
    tf: str,
    fetch_start: str,
    start: str,
    is_end: str,
    end: str,
    skip_metrics: bool = False,
    target_tfs: list[str] | None = None,
    load_exec_1m: bool | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    """Load data for multiple symbols in parallel and inject momentum ranks."""
    data_maps: dict[str, dict[str, Any]] = {}
    oos_data_maps: dict[str, dict[str, Any]] = {}
    valid_symbols: list[str] = []

    # [Fix] Filter out non-ASCII symbols before processing
    symbols = [s for s in symbols if all(ord(c) < 128 for c in s)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                load_single_symbol_data,
                sym,
                tf,
                fetch_start,
                start,
                is_end,
                end,
                skip_metrics,
                target_tfs,
                load_exec_1m,
            )
            for sym in symbols
        ]
        for f in concurrent.futures.as_completed(futures):
            sym, t_is, t_oos, insufficient = f.result()
            if not insufficient and t_is and t_oos:
                data_maps[sym], oos_data_maps[sym] = t_is, t_oos
                valid_symbols.append(sym)

    if len(valid_symbols) > 1:
        inject_cs_momentum_ranks(data_maps, valid_symbols, tf)
        inject_cs_momentum_ranks(oos_data_maps, valid_symbols, tf)

    audit_rows: list[dict[str, Any]] = []
    for sym in valid_symbols:
        recs = data_maps.get(sym, {}).get("integrity_audit", [])
        if isinstance(recs, list):
            audit_rows.extend([r for r in recs if isinstance(r, dict)])
    if audit_rows:
        audit_df = pd.DataFrame(audit_rows)
        grp = (
            audit_df.groupby(["stage", "timeframe"], dropna=False)[
                ["nan_pct", "inf_count", "zero_ratio", "duplicate_dt", "gap_count", "nonpositive_price_count"]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        _logger.info(" 📊 [DATA INTEGRITY AUDIT] symbols=%d rows=%d", len(valid_symbols), len(audit_rows))
        _logger.info(" ──────────────────────────────────────────────────────────────")
        _logger.info("  STAGE    TF      NAN-PCT    ZERO-RAT    GAPS/DUP   VERDICT")
        _logger.info(" ──────────────────────────────────────────────────────────────")

        # Smart consolidation: If all metrics are same across timeframes for a stage, collapse them.
        for stage_name in sorted(grp["stage"].unique()):
            stage_df = grp[grp["stage"] == stage_name]
            
            # Check if metrics are identical across all TFs in this stage
            # We check nan_pct, zero_ratio, gap_count, duplicate_dt
            unique_metrics = stage_df[["nan_pct", "zero_ratio", "gap_count", "duplicate_dt"]].round(6).drop_duplicates()
            
            rows_to_print = []
            if len(unique_metrics) == 1 and len(stage_df) > 1:
                # All TFs are identical, collapse to 'All'
                row = stage_df.iloc[0].to_dict()
                row["timeframe"] = "All"
                rows_to_print.append(row)
            else:
                # Different metrics or only one TF, print all
                rows_to_print = stage_df.to_dict(orient="records")

            for row in rows_to_print:
                stg = str(row.get("stage", "??")).upper()[:7]
                tf = str(row.get("timeframe", "??"))
                nan = float(row.get("nan_pct", 0.0)) * 100
                zero = float(row.get("zero_ratio", 0.0)) * 100
                gaps = float(row.get("gap_count", 0.0))
                dups = float(row.get("duplicate_dt", 0.0))
                
                verdict = "[PASS]"
                if nan > 25.0: verdict = "[FAIL: NaN]"
                elif nan > 10.0: verdict = "[WARN: NaN]"
                elif gaps > 0 or dups > 0: verdict = "[FAIL: Gaps]"
                
                # Robust whitespace alignment (No vertical bars)
                _logger.info(f"  {stg:<8} {tf:<5} {nan:>8.2f}% {zero:>10.2f}% {gaps:>5.0f}/{dups:<3.0f}   {verdict}")

        _logger.info(" ──────────────────────────────────────────────────────────────")

    return data_maps, oos_data_maps, valid_symbols


def compute_regime_drift(
    data_maps: dict[str, dict[str, Any]],
    oos_data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> dict[str, Any]:
    """S3: Compute IS vs OOS regime distribution drift (KL divergence).

    Detects when OOS regime distribution diverges from IS training distribution.
    KL > 0.5 nats → significant drift warning; KL > 1.0 → severe drift.
    """
    def _regime_dist(maps: dict[str, dict[str, Any]], oos_split: bool) -> np.ndarray:
        counts = np.zeros(len(REGIME_NAMES), dtype=np.int64)
        for sym in symbols:
            smap = maps.get(sym, {})
            df = smap.get(tf)
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if oos_split:
                o0 = int(smap.get(f"oos_start_idx_{tf}", 0))
                slice_df = df.iloc[o0:]
            else:
                o0 = int(smap.get(f"is_start_idx_{tf}", 0))
                slice_df = df.iloc[o0:]
            if slice_df.empty:
                continue
            rc = infer_regime_codes(slice_df)
            counts += np.bincount(rc, minlength=len(REGIME_NAMES))
        total = float(counts.sum())
        if total < 1:
            return np.ones(len(REGIME_NAMES), dtype=np.float64) / len(REGIME_NAMES)
        return counts.astype(np.float64) / total

    is_dist = _regime_dist(data_maps, oos_split=False)
    oos_dist = _regime_dist(oos_data_maps, oos_split=True)

    eps = 1e-9
    kl_is_to_oos = float(np.sum(is_dist * np.log((is_dist + eps) / (oos_dist + eps))))
    kl_oos_to_is = float(np.sum(oos_dist * np.log((oos_dist + eps) / (is_dist + eps))))
    kl_sym = (kl_is_to_oos + kl_oos_to_is) / 2.0

    drift_label = "OK"
    if kl_sym > 1.0:
        drift_label = "SEVERE"
    elif kl_sym > 0.5:
        drift_label = "MODERATE"
    elif kl_sym > 0.2:
        drift_label = "MILD"

    regime_shift = {}
    for ridx, rname in enumerate(REGIME_NAMES):
        regime_shift[rname] = {
            "is_pct": float(is_dist[ridx] * 100.0),
            "oos_pct": float(oos_dist[ridx] * 100.0),
            "ratio": float((oos_dist[ridx] + eps) / (is_dist[ridx] + eps)),
        }
    is_oos_crisis_ratio = float(regime_shift.get("CRISIS", {}).get("ratio", 1.0))

    status_map = {
        "SEVERE": "🔴 SEVERE",
        "MODERATE": "🟡 MODERATE",
        "MILD": "🔵 MILD",
        "OK": "🟢 OK"
    }
    status_text = status_map.get(drift_label, "🟢 OK")

    _logger.info("\n 🔍 [STRATEGY DRIFT AUDIT]")
    _logger.info(" ──────────────────────────────────────────────────────────────")
    _logger.info(f"  Overall Status : {status_text} (Score: {kl_sym:.3f})")
    _logger.info(" ──────────────────────────────────────────────────────────────")
    _logger.info("  REGIME        IS-DIST      OOS-DIST     SHIFT-RATIO")
    _logger.info(" ──────────────────────────────────────────────────────────────")
    
    for rname, v in regime_shift.items():
        ratio_str = f"x{v['ratio']:.2f}x" if v['ratio'] < 100 else ">x100x"
        _logger.info(
            f"  {rname:<10}   {v['is_pct']:>6.1f}%  ➔  {v['oos_pct']:>6.1f}%    {ratio_str:>9}"
        )
        
    _logger.info(" ──────────────────────────────────────────────────────────────")
    if drift_label in ("MODERATE", "SEVERE"):
        _logger.warning("  ⚠️  CRITICAL DRIFT: OOS environment differs significantly from IS.")
        _logger.warning("  建议: Walk-forward refit or parameter re-tuning recommended.")
    else:
        _logger.info("  ✅ STABLE: OOS regime distribution matches IS training.")
    _logger.info(" ──────────────────────────────────────────────────────────────")

    return {
        "kl_is_to_oos": kl_is_to_oos,
        "kl_oos_to_is": kl_oos_to_is,
        "kl_sym": kl_sym,
        "drift_label": drift_label,
        "is_oos_crisis_ratio": is_oos_crisis_ratio,
        "regime_shift": regime_shift,
    }
