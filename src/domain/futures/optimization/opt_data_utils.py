from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.optimization.opt_utils import compute_segment_merge_index
from src.core.settings import FUTURES_DATA_DIR, LOG_DIR
from src.domain.futures.backtest.data_loader import (
    DataCollector,
    summarize_dataframe_integrity,
    summarize_ohlcv_collection_integrity,
)
from src.domain.futures.optimization.common import inject_cs_momentum_ranks
from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

_logger: logging.Logger = logging.getLogger("opt_data_utils")
_SUFFICIENCY_LOG_DIR = LOG_DIR / "futures/data"


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
    min_membership_warm_days = int(OPT_FUTURES_CONFIG.get("FUTURES_MEMBERSHIP_WARM_DAYS", 42))
    min_membership_warm = min_membership_warm_days * bars_per_day
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
    scope_name: str = "unknown",
    onboard_date: str | None = None,
) -> dict[str, Any]:
    frame = symbol_map.get(tf)
    if not isinstance(frame, pd.DataFrame) or frame.empty or "datetime" not in frame.columns:
        return {"symbol": symbol, "tf": tf, "pass": False, "reason": "missing_tf_frame"}
    dt_col = frame["datetime"]
    if not pd.api.types.is_datetime64_any_dtype(dt_col):
        dt = pd.to_datetime(dt_col, utc=True, errors="coerce").dropna().sort_values()
    else:
        dt = dt_col.dropna()

    if dt.empty:
        return {"symbol": symbol, "tf": tf, "pass": False, "reason": "invalid_datetime"}

    first_dt = dt.iloc[0]
    last_dt = dt.iloc[-1]
    
    effective_fetch_start = pd.Timestamp(fetch_start, tz="UTC")
    if onboard_date is not None:
        onboard_ts = pd.Timestamp(onboard_date, tz="UTC")
        effective_fetch_start = max(effective_fetch_start, onboard_ts)

    fetch_ok = first_dt <= effective_fetch_start and last_dt >= pd.Timestamp(
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
        if (
            not isinstance(exec_1m, pd.DataFrame)
            or exec_1m.empty
            or "datetime" not in exec_1m.columns
        ):
            exec_1m_ok = False
            exec_1m_cov = 0.0
        else:
            exec_dt_col = exec_1m["datetime"]
            if not pd.api.types.is_datetime64_any_dtype(exec_dt_col):
                exec_dt = pd.to_datetime(exec_dt_col, utc=True, errors="coerce").dropna()
            else:
                exec_dt = exec_dt_col.dropna()
            actual_1m = int(
                ((exec_dt >= pd.Timestamp(fetch_start, tz="UTC")) & (exec_dt <= oos_end_ts)).sum()
            )
            required_1m = max(1, _bars_between(fetch_start, oos_end, "1m"))
            exec_1m_cov = float(actual_1m / required_1m)
            exec_1m_ok = exec_1m_cov >= 0.95

    is_historical_stage5 = str(scope_name).strip().lower() == "historical_stage5_union"

    # C1 inference panel: 신규 상장 심볼이 공유 time axis를 끌어내려
    # adj_train + folds=1 붕괴를 유발하지 않도록 최소 패널 이력 체크.
    # 기준: first_dt ≤ oos_start - FUTURES_INFERENCE_MIN_HISTORY_MONTHS
    panel_history_ok = True
    if is_historical_stage5:
        _min_hist_months = int(OPT_FUTURES_CONFIG.get("FUTURES_INFERENCE_MIN_HISTORY_MONTHS", 33))
        _panel_cutoff = pd.Timestamp(oos_start, tz="UTC") - pd.DateOffset(months=_min_hist_months)
        panel_history_ok = first_dt <= _panel_cutoff

    if is_historical_stage5:
        # C1 학습 패널은 delisted 포함 historical union이므로
        # 전체 fetch/OOS 종단 커버리지를 강제하지 않는다.
        pass_flag = bool(
            warmup_ok
            and actual_is_bars >= min_is_bars
            and exec_1m_ok
            and panel_history_ok
        )
    else:
        pass_flag = bool(
            fetch_ok
            and warmup_ok
            and actual_is_bars >= min_is_bars
            and actual_oos_bars >= min_oos_bars
            and exec_1m_ok
        )
    reason = "ok"
    if not fetch_ok and not is_historical_stage5:
        reason = "fetch_window_short"
    elif not warmup_ok:
        reason = "warmup_insufficient"
    elif actual_is_bars < min_is_bars:
        reason = "is_coverage_short"
    elif not panel_history_ok and is_historical_stage5:
        reason = "panel_history_insufficient"
    elif actual_oos_bars < min_oos_bars and not is_historical_stage5:
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
    scope_name: str = "unknown",
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]], pd.DataFrame, int]:
    warmup_bars = _resolve_warmup_bars(tf)
    
    # Load symbol sync profiles to get onboard_date
    onboard_dates: dict[str, str] = {}
    try:
        import json
        profiles_path = Path(FUTURES_DATA_DIR) / "symbol_sync_profiles.json"
        if not profiles_path.exists():
            try:
                from src.domain.futures.universe.storage import _load_symbol_sync_profiles
                _load_symbol_sync_profiles()
            except Exception as e:
                _logger.debug("Failed to populate symbol sync profiles: %s", e)
            
        if profiles_path.exists():
            with open(profiles_path, encoding="utf-8") as f:
                p_data = json.load(f)
            for s, item in p_data.items():
                ob = item.get("onboard_date")
                if ob:
                    onboard_dates[s] = ob
    except Exception as e:
        _logger.warning("Failed to load symbol onboard dates for sufficiency evaluation: %s", e)

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
            scope_name=scope_name,
            onboard_date=onboard_dates.get(symbol),
        )
        rows.append(rec)
        _logger.debug(
            ".. DATA_S: sym=%s tf=%s pass=%s reason=%s",
            symbol,
            tf,
            str(bool(rec.get("pass", False))).lower(),
            str(rec.get("reason", "unknown")),
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
    if isinstance(dt, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(dt):
            return dt.astype("datetime64[ns]").astype("int64") // 10**6
    elif isinstance(dt, pd.DatetimeIndex):
        return dt.astype("datetime64[ns]").astype("int64") // 10**6
    return pd.to_datetime(dt, utc=True).astype("datetime64[ns]").astype("int64") // 10**6


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
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "calc_time" in df.columns:
            df = df.rename(columns={"calc_time": "timestamp"})
        if "fundingRate" in df.columns:
            df = df.rename(columns={"fundingRate": "funding_rate"})
        if "timestamp" not in df.columns and len(df.columns) > 0:
            df = df.rename(columns={df.columns[0]: "timestamp"})
        if "funding_rate" not in df.columns and len(df.columns) > 2:
            df = df.rename(columns={df.columns[2]: "funding_rate"})
        if "timestamp" not in df.columns or "funding_rate" not in df.columns:
            return None
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
        df = df.dropna(subset=["timestamp", "funding_rate"])
        if df.empty:
            return None
        df["timestamp"] = df["timestamp"].astype("int64")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df[["timestamp", "funding_rate", "datetime"]].drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
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
    pos_safe = np.clip(pos, 0, n - 1)  # guard before indexing; out-of-bounds filtered by pos < n
    valid = (pos >= 0) & (pos < n) & (exec_ms[pos_safe] == event_ms) & np.isfinite(event_rate)
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
}


def _feature_group_coverage(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if df is None or df.empty:
        for g in _FEATURE_GROUP_PATTERNS:
            out[g] = {"col_count": 0.0, "non_null_coverage": 0.0, "non_zero_coverage": 0.0}
        return out
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
    # [Optimization] Skip expensive integrity check for "raw" stage to reduce CPU/GIL bottleneck.
    # We only care about the "merged" (final) state for the audit report.
    if stage == "merged":
        rec.update(summarize_ohlcv_collection_integrity(df, timeframe=timeframe))
    elif stage == "raw":
        # Minimal metrics for raw stage
        rec.update({"rows": float(len(df)), "cols": float(len(df.columns))})
    else:
        rec.update(summarize_dataframe_integrity(df, timeframe=timeframe))
    if fillna_cols:
        pre_na = df[fillna_cols].isna().sum().sum() if fillna_cols else 0
        denom = max(int(len(df) * len(fillna_cols)), 1)
        rec["pre_fillna_nan_pct"] = float(pre_na / denom)
    audit.append(rec)


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

        # [Optimization] Pre-load and prepare funding and metrics data once
        # to avoid redundant I/O and processing per TF.
        funding_df_prepared = None
        metrics_df_prepared = None
        if not skip_metrics:
            funding_df = _safe_read_funding_parquet(sym)
            if funding_df is not None and not funding_df.empty:
                exclude_fr = ["datetime", "symbol"]
                cols_fr = [c for c in funding_df.columns if c not in exclude_fr]
                funding_df_prepared = (
                    funding_df[cols_fr].sort_values("timestamp").reset_index(drop=True)
                )

            m_path = Path(FUTURES_DATA_DIR) / f"{sym.replace('/', '_')}_metrics.parquet"
            if m_path.exists():
                try:
                    m_df = pd.read_parquet(m_path)
                    if m_df is not None and not m_df.empty:
                        m_df = m_df.loc[:, ~m_df.columns.duplicated(keep="first")]
                        m_df["timestamp"] = _to_unix_ms(m_df["datetime"])
                        exclude_m = ["datetime", "create_time", "symbol"]
                        cols_m = [c for c in m_df.columns if c not in exclude_m]
                        metrics_df_prepared = (
                            m_df[cols_m].sort_values("timestamp").reset_index(drop=True)
                        )
                except Exception:
                    _logger.warning("Failed to load metrics data for %s", sym)

        for tf_l in tfs_to_load:
            raw_df = collector.collect_and_save(
                sym, tf_l, fetch_start, end, fetch_network=False
            )
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
                # [Optimization] Use pre-prepared data outside the loop for high-speed merge
                df = raw_df.copy()
                if funding_df_prepared is not None and not funding_df_prepared.empty:
                    if "timestamp" not in df.columns:
                        df["timestamp"] = _to_unix_ms(df["datetime"])
                    df = pd.merge_asof(
                        df.sort_values("timestamp"),
                        funding_df_prepared,
                        on="timestamp",
                        direction="backward",
                    )

                if metrics_df_prepared is not None and not metrics_df_prepared.empty:
                    if "timestamp" not in df.columns:
                        df["timestamp"] = _to_unix_ms(df["datetime"])
                    df = pd.merge_asof(
                        df.sort_values("timestamp"),
                        metrics_df_prepared,
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

            except Exception as e:
                _logger.error("[%s] Merge/Enrich failed: %s", sym, e)
                insufficient = True
                break

            if df is None or df.empty or "datetime" not in df.columns:
                insufficient = True
                break

            df.reset_index(drop=True, inplace=True)
            # [Optimization] Avoid redundant to_datetime if already normalized by collector
            if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
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
                exec_1m = collector.collect_1m_ohlcv(
                    sym, fetch_start, end, fetch_network=False
                )
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
    requested_symbols_count: int | None = None,
    scope_name: str = "unknown",
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    """Load data for multiple symbols in parallel and inject momentum ranks."""
    data_maps: dict[str, dict[str, Any]] = {}
    oos_data_maps: dict[str, dict[str, Any]] = {}
    valid_symbols: list[str] = []

    # [Fix] Filter out non-ASCII symbols before processing
    symbols = [s for s in symbols if all(ord(c) < 128 for c in s)]

    # [Optimization] Use ProcessPoolExecutor to bypass GIL for CPU-bound pandas operations (merge, integrity check).
    # Since we are loading 90+ symbols, the CPU overhead of thread serialization is significant.
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 2)) as executor:
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
    requested_count = int(requested_symbols_count or len(symbols))
    loaded_count = len(valid_symbols)
    if audit_rows:
        audit_df = pd.DataFrame(audit_rows)
        grp = (
            audit_df.groupby(["stage", "timeframe"], dropna=False)[
                ["nan_pct", "inf_count", "gap_count", "duplicate_dt", "nonpositive_price_count"]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        # Condensed single-line audit summary
        failed_tfs = []
        for stage_name in sorted(grp["stage"].unique()):
            stage_df = grp[grp["stage"] == stage_name]
            for _, row in stage_df.iterrows():
                nan = float(row.get("nan_pct", 0.0)) * 100
                gaps = float(row.get("gap_count", 0.0))
                dups = float(row.get("duplicate_dt", 0.0))
                nonpos = float(row.get("nonpositive_price_count", 0.0))
                if nan > 10.0 or gaps > 0 or dups > 0 or nonpos > 0:
                    failed_tfs.append(f"{row['timeframe']}({row['stage']}: nan={nan:.1f}%)")
        
        coverage_val = float(loaded_count / max(requested_count, 1))
        audit_msg = (
            f".. AUDIT: req={requested_count} load={loaded_count} "
            f"coverage={coverage_val:.2f}"
        )
        if failed_tfs:
            audit_msg += f" | !! FAIL: {', '.join(failed_tfs[:3])}"
        else:
            audit_msg += " | ok"
        _logger.debug(audit_msg)
    else:
        _logger.debug(".. AUDIT: req=%d load=%d coverage=%.2f | ok (0 rows)", 
                     requested_count, loaded_count, float(loaded_count / max(requested_count, 1)))

    return data_maps, oos_data_maps, valid_symbols

    return data_maps, oos_data_maps, valid_symbols


