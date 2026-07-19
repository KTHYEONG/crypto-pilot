from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from src.core.optimization.opt_utils import compute_segment_merge_index
from src.core.settings import LOG_DIR, FuturesStorageLayout
from src.domain.futures.backtest.data_loader import (
    DataCollector,
    summarize_dataframe_integrity,
)
from src.domain.futures.optimization.common import inject_cs_momentum_ranks
from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

_logger: logging.Logger = logging.getLogger("opt_data_utils")
_SUFFICIENCY_LOG_DIR = LOG_DIR / "futures/data"


def _bars_per_day(tf: str) -> float:
    """Replaces ad hoc {'1h':24,'4h':6,'1d':1}.get(tf,6) with canonical hours_per_bar(). [LIMIT-09]"""
    from src.domain.futures.strategy.timeframe_contracts import hours_per_bar

    return 24.0 / hours_per_bar(tf)


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
    bpd = _bars_per_day(tf)
    lookback = int(OPT_FUTURES_CONFIG.get("FUTURES_MOMENTUM_LOOKBACK", 252))
    cov = int(OPT_FUTURES_CONFIG.get("FUTURES_COV_LOOKBACK", 252))
    sigma = int(OPT_FUTURES_CONFIG.get("FUTURES_COMPOSER_SIGMA_LOOKBACK", 252))
    atr = int(OPT_FUTURES_CONFIG.get("FUTURES_ATR_PERIOD", 14))
    embargo = int(OPT_FUTURES_CONFIG.get("FUTURES_EMBARGO_BARS", 0))
    platt = int(OPT_FUTURES_CONFIG.get("FUTURES_PLATT_MIN_TRAIN_BARS", 120))
    min_membership_warm_days = int(OPT_FUTURES_CONFIG.get("FUTURES_MEMBERSHIP_WARM_DAYS", 42))
    min_membership_warm = round(min_membership_warm_days * bpd)
    return max(lookback, cov, sigma, atr, embargo, platt, min_membership_warm)


def resolve_warmup_days_for_tf(tf: str, *, safety_margin_days: int = 20) -> int:
    """지표 워밍업 요구치(bar)를 캘린더 일수로 환산. [ADR_20260706_DATA_WINDOW_FLOOR_CONSISTENCY]"""
    return math.ceil(_resolve_warmup_bars(tf) / _bars_per_day(tf)) + safety_margin_days


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
    """[ADR_20260710_L0_SIGNAL_BREADTH_DIVERSITY_REDESIGN] exec_1m no longer gates admission `pass`/`reason`."""
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

    # Continuity gap check: detect largest consecutive missing-bar stretch
    _sorted_dt = dt.sort_values() if not dt.is_monotonic_increasing else dt
    if len(_sorted_dt) > 1:
        _bar_delta = _tf_delta(tf)
        _diffs = _sorted_dt.diff().dropna()
        _max_diff = _diffs.max()
        max_gap_bars = max(0, round(float(_max_diff / _bar_delta)) - 1)
    else:
        max_gap_bars = 0

    # Align boundary with universe G6 gate (eligibility.py: `_max_gap > _max_gap_bars`):
    # max_gap == threshold (24h @4h) is admitted by universe, so it must pass here too.
    _max_gap_threshold = int(OPT_FUTURES_CONFIG.get("FUTURES_BACKTEST_MAX_GAP_BARS", 6))
    gap_ok = max_gap_bars <= _max_gap_threshold

    effective_fetch_start = pd.Timestamp(fetch_start, tz="UTC")
    if onboard_date is not None:
        onboard_ts = pd.Timestamp(onboard_date, tz="UTC")
        effective_fetch_start = max(effective_fetch_start, onboard_ts)

    fetch_ok = first_dt <= effective_fetch_start and last_dt >= pd.Timestamp(oos_end, tz="UTC")

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
            exec_dt_col = exec_1m["datetime"]
            if not pd.api.types.is_datetime64_any_dtype(exec_dt_col):
                exec_dt = pd.to_datetime(exec_dt_col, utc=True, errors="coerce").dropna()
            else:
                exec_dt = exec_dt_col.dropna()
            actual_1m = int(((exec_dt >= pd.Timestamp(fetch_start, tz="UTC")) & (exec_dt <= oos_end_ts)).sum())
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
        pass_flag = bool(warmup_ok and actual_is_bars >= min_is_bars and panel_history_ok and gap_ok)
    else:
        pass_flag = bool(
            fetch_ok and warmup_ok and actual_is_bars >= min_is_bars and actual_oos_bars >= min_oos_bars and gap_ok
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
    elif not gap_ok:
        reason = "gap_too_large"
    # exec_1m_ok intentionally excluded from pass_flag/reason [LIMIT-01].

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
        "exec_1m_ok": exec_1m_ok,
        "exec_1m_coverage": exec_1m_cov,
        "max_gap_bars": max_gap_bars,
        "first_dt": first_dt.isoformat(),
        "last_dt": last_dt.isoformat(),
    }


def data_observable(
    *,
    symbol: str,
    tf: str,
    symbol_map: dict[str, Any],
    onboard_date: str | None = None,
) -> dict[str, Any]:
    """Lifecycle-aware data observability check (PIT-safe).

    Unlike ``evaluate_symbol_data_sufficiency``, this function does NOT require:
    - Data coverage until OOS end (allows symbols that delist mid-period).
    - Full IS/OOS bar counts.
    - exec_1m coverage.

    Only checks: frame exists, has datetime column, non-empty, basic reachability.
    Strategy lookback readiness is delegated to StrategyReadinessCube.

    Args:
        symbol: Instrument symbol key.
        tf: Timeframe key (e.g. "4h").
        symbol_map: Dict mapping timeframe -> DataFrame for one symbol.
        onboard_date: ISO date string of exchange onboarding; adjusts effective
            start if later than first observed bar.

    Returns:
        Dict with keys: symbol, tf, pass (bool), reason, and optional
        first_dt, last_dt, effective_start, n_bars, is_historical_stage5.

    Time Complexity: O(n) where n = len(frame).
    Space Complexity: O(1) auxiliary (no copy of the frame).
    """
    frame = symbol_map.get(tf)
    if not isinstance(frame, pd.DataFrame) or frame.empty or "datetime" not in frame.columns:
        return {"symbol": symbol, "tf": tf, "pass": False, "reason": "missing_tf_frame"}

    dt_col = frame["datetime"]
    if not pd.api.types.is_datetime64_any_dtype(dt_col):
        dt: pd.Series = pd.to_datetime(dt_col, utc=True, errors="coerce").dropna().sort_values()
    else:
        dt = dt_col.dropna().sort_values()

    if dt.empty:
        return {"symbol": symbol, "tf": tf, "pass": False, "reason": "empty_datetime"}

    first_dt: pd.Timestamp = dt.iloc[0]
    last_dt: pd.Timestamp = dt.iloc[-1]

    if onboard_date is not None:
        onboard_ts = pd.Timestamp(onboard_date, tz="UTC")
        effective_start: pd.Timestamp = max(first_dt, onboard_ts)
    else:
        effective_start = first_dt

    n_bars = len(dt)
    return {
        "symbol": symbol,
        "tf": tf,
        "pass": True,
        "reason": "data_observable",
        "first_dt": first_dt.isoformat(),
        "last_dt": last_dt.isoformat(),
        "effective_start": effective_start.isoformat(),
        "n_bars": n_bars,
        "is_historical_stage5": False,
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

        from src.core.settings import FuturesStorageLayout

        profiles_path = FuturesStorageLayout.get_metadata_path("symbol_sync_profiles.json")
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

    Pandas 3.x raises TypeError when .astype('datetime64[ns]') is used on
    tz-aware data. Strip timezone first (values are in UTC, so no shift).
    """
    if isinstance(dt, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(dt):
            if dt.dt.tz is not None:
                dt = dt.dt.tz_localize(None)
            return dt.astype("datetime64[ns]").astype("int64") // 10**6
    elif isinstance(dt, pd.DatetimeIndex):
        if dt.tz is not None:
            dt = dt.tz_localize(None)
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
    from src.core.settings import FuturesStorageLayout

    f_path = FuturesStorageLayout.get_funding_path(symbol)
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
    event_rate = pd.to_numeric(funding_df["funding_rate"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

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


_COL_GROUP_CACHE: dict[tuple[str | frozenset[str], ...], dict[str, str | None]] = {}

_FEATURE_GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
    "price": ("open", "high", "low", "close", "ret_", "vol_", "realized_vol", "atr", "mom", "beta"),
    "funding": ("funding",),
    "oi": ("open_interest", "oi", "sum_open_interest"),
    "lsr": ("lsr", "long_short", "top_trader", "global_lsr"),
    "taker_orderflow": ("taker", "buy_sell", "imbalance", "orderflow"),
    "macro": ("macro_", "btc_", "market_", "cs_dispersion"),
}


def _feature_group_coverage(df: pd.DataFrame, *, tf_label: str = "") -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if df is None or df.empty:
        for g in _FEATURE_GROUP_PATTERNS:
            out[g] = {"col_count": 0.0, "non_null_coverage": 0.0, "non_zero_coverage": 0.0}
        return out

    # OPT-4: Cache column→group mapping across symbols (columns deterministic per TF)
    cache_key = (tf_label, frozenset(str(c).lower() for c in df.columns))
    mapping = _COL_GROUP_CACHE.get(cache_key)
    if mapping is None:
        mapping = {}
        for group, pats in _FEATURE_GROUP_PATTERNS.items():
            for col in df.columns:
                col_lower = str(col).lower()
                if any(p in col_lower for p in pats):
                    mapping[col_lower] = group
        _COL_GROUP_CACHE[cache_key] = mapping

    for group in _FEATURE_GROUP_PATTERNS:
        cols = [c for c in df.columns if mapping.get(str(c).lower()) == group]
        if not cols:
            out[group] = {"col_count": 0.0, "non_null_coverage": 0.0, "non_zero_coverage": 0.0}
            continue
        sub_df = df[cols]
        numeric_cols = sub_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == len(cols):
            sub = sub_df
        else:
            non_numeric_cols = [c for c in cols if c not in numeric_cols]
            sub_numeric = sub_df[numeric_cols]
            sub_non_numeric = sub_df[non_numeric_cols].apply(pd.to_numeric, errors="coerce")
            sub = pd.concat([sub_numeric, sub_non_numeric], axis=1)[cols]
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
    # [Optimization] Lightweight audit for merged/raw stages.
    # Heavy summarize_ohlcv_collection_integrity scanned entire DataFrame
    # (NaN, inf, gap, OHLCV violations) for every TF per symbol — deferred.
    if stage in ("merged", "raw"):
        rec.update({"rows": float(len(df)), "cols": float(len(df.columns))})
    else:
        rec.update(summarize_dataframe_integrity(df, timeframe=timeframe))
    if fillna_cols:
        pre_na = df[fillna_cols].isna().sum().sum() if fillna_cols else 0
        denom = max(int(len(df) * len(fillna_cols)), 1)
        rec["pre_fillna_nan_pct"] = float(pre_na / denom)
    audit.append(rec)


def _scan_enriched_dataset(
    valid_paths: list[Path],
    start_ms: int,
    end_ms: int,
    columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Scan multiple enriched parquet files via Arrow Dataset (Pass-1, P2).

    Uses pyarrow.dataset per-file to leverage C++ multithreaded row-group
    predicate pushdown on the ``timestamp`` (int64 unix-ms) column.
    GIL is released during Arrow I/O + decode → superior to pandas full-read.

    Args:
        valid_paths: Enriched parquet files that passed mtime cache validation.
            Filename convention: ``{safe_sym}_{tf_l}_enriched.parquet``.
        start_ms: Window start (unix-ms, int64).
        end_ms: Window end (unix-ms, int64).
        columns: Optional projection. None means all columns.

    Returns:
        Dict mapping ``"{safe_sym}_{tf_l}"`` key to a window-filtered DataFrame.
        Files that lack a ``timestamp`` column or fail to scan are silently
        skipped (per-file fallback handled by caller).

    Time Complexity: O(W) where W = rows in window (Arrow skips non-overlapping
        row-groups entirely via statistics). Space: O(W x C).
    """
    filt = (pc.field("timestamp") >= start_ms) & (pc.field("timestamp") <= end_ms)  # type: ignore[no-untyped-call]
    result: dict[str, pd.DataFrame] = {}

    for path in valid_paths:
        # Key: strip "_enriched.parquet" suffix → "{safe_sym}_{tf_l}"
        key = path.stem.removesuffix("_enriched")
        try:
            dataset = ds.dataset(str(path), format="parquet")  # type: ignore[no-untyped-call]
            # Validate timestamp column present in schema
            schema_names = dataset.schema.names
            if "timestamp" not in schema_names:
                _logger.debug("[scan] %s missing timestamp col — skipping pushdown", path.name)
                continue
            proj = columns if columns is not None else schema_names
            # Intersect requested columns with available columns
            proj = [c for c in proj if c in schema_names]
            table = dataset.to_table(filter=filt, columns=proj, use_threads=True)
            df = table.to_pandas()
            # Dynamically reconstruct datetime from timestamp if missing
            if "timestamp" in df.columns and "datetime" not in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            # Ensure datetime column is tz-aware UTC (Arrow may return tz-naive)
            if "datetime" in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                elif df["datetime"].dt.tz is None:
                    df["datetime"] = df["datetime"].dt.tz_localize("UTC")
            result[key] = df
        except Exception as _e:
            _logger.debug("[scan] %s Arrow scan failed (%s) — caller will fallback", path.name, _e)

    return result


def _prepare_funding_metrics(
    sym: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Load and prepare funding and metrics data for a symbol (P1-A lazy helper).

    Extracted so that cache-hit paths with no exec_1m requirement can skip
    expensive I/O entirely — caller uses ``_ensure_fm_loaded`` guard.

    Args:
        sym: Instrument symbol (e.g. "BTCUSDT").

    Returns:
        Tuple of (funding_df, funding_df_prepared, metrics_df_prepared).
        Any element may be None if the corresponding data is absent.
    """
    funding_df = _safe_read_funding_parquet(sym)
    funding_df_prepared: pd.DataFrame | None = None
    metrics_df_prepared: pd.DataFrame | None = None

    if funding_df is not None and not funding_df.empty:
        exclude_fr = ["datetime", "symbol"]
        cols_fr = [c for c in funding_df.columns if c not in exclude_fr]
        funding_df_prepared = funding_df[cols_fr].sort_values("timestamp").reset_index(drop=True)

    from src.core.settings import FuturesStorageLayout

    m_path = FuturesStorageLayout.get_metrics_path(sym)
    if m_path.exists():
        try:
            m_df = pd.read_parquet(m_path)
            if m_df is not None and not m_df.empty:
                m_df = m_df.loc[:, ~m_df.columns.duplicated(keep="first")]
                release_col = "available_at" if "available_at" in m_df.columns else "datetime"
                m_df[release_col] = pd.to_datetime(m_df[release_col], utc=True, errors="coerce")
                m_df = m_df.dropna(subset=[release_col]).sort_values(release_col)
                m_df["metrics_release_ts"] = _to_unix_ms(m_df[release_col])
                exclude_m = ["datetime", "create_time", "symbol", "available_at"]
                cols_m = [c for c in m_df.columns if c not in exclude_m]
                metrics_df_prepared = m_df[cols_m].sort_values("metrics_release_ts").reset_index(drop=True)
        except Exception:
            _logger.warning("Failed to load metrics data for %s", sym)

    return funding_df, funding_df_prepared, metrics_df_prepared


def _resolve_timestamp_column(df: pd.DataFrame) -> str | None:
    """[ADR_20260717_L2_CRISIS_BTC_REGIME_DATA_INTEGRITY_FIX] merge-suffixed schema fallback."""
    if "timestamp" in df.columns:
        return "timestamp"
    if "timestamp_x" in df.columns:
        return "timestamp_x"
    return None




@dataclass(slots=True, frozen=True)
class _DepFileSignature:
    mtime_ns: int
    size_bytes: int


def _capture_dep_signatures(dep_paths: Sequence[Path]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for dep in dep_paths:
        if dep.exists():
            st = dep.stat()
            result[str(dep)] = [st.st_mtime_ns, st.st_size]
    return result


def _enriched_signature_sidecar_path(enriched_path: Path) -> Path:
    return enriched_path.with_suffix(".sig.json")


def _write_enriched_cache_signature(enriched_path: Path, dep_paths: Sequence[Path]) -> None:
    sig_path = _enriched_signature_sidecar_path(enriched_path)
    sig_path.write_text(json.dumps(_capture_dep_signatures(dep_paths)))


def _is_enriched_cache_fresh(enriched_path: Path, dep_paths: Sequence[Path]) -> bool:
    if not enriched_path.exists():
        return False
    sig_path = _enriched_signature_sidecar_path(enriched_path)
    if not sig_path.exists():
        enriched_mtime = enriched_path.stat().st_mtime
        return all(not d.exists() or d.stat().st_mtime <= enriched_mtime for d in dep_paths)
    try:
        stored: object = json.loads(sig_path.read_text())
    except Exception:
        return False
    return _capture_dep_signatures(dep_paths) == stored

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

        # [P1-A] Lazy funding/metrics: initialize as None; load only on first
        # cache-miss merge entry OR when exec_1m funding arrays are needed.
        # cache-hit + no exec_1m → funding/metrics I/O skipped entirely.
        funding_df: pd.DataFrame | None = None
        funding_df_prepared: pd.DataFrame | None = None
        metrics_df_prepared: pd.DataFrame | None = None
        _fm_loaded = False  # guard: ensure _prepare_funding_metrics called at most once

        def _ensure_fm_loaded() -> None:
            nonlocal funding_df, funding_df_prepared, metrics_df_prepared, _fm_loaded
            if _fm_loaded or skip_metrics:
                return
            funding_df, funding_df_prepared, metrics_df_prepared = _prepare_funding_metrics(sym)
            _fm_loaded = True

        for tf_l in tfs_to_load:
            req_start_dt = pd.Timestamp(fetch_start, tz="UTC")
            req_end_dt = pd.Timestamp(end, tz="UTC")
            from_cache = False
            from src.core.settings import FuturesStorageLayout

            enriched_path = FuturesStorageLayout.get_enriched_path(sym, tf_l)
            deps = [
                FuturesStorageLayout.get_ohlcv_path(sym, tf_l),
                FuturesStorageLayout.get_funding_path(sym),
                FuturesStorageLayout.get_metrics_path(sym),
            ]
            if enriched_path.exists() and _is_enriched_cache_fresh(enriched_path, deps):
                # [P1-B] predicate pushdown via row-group statistics.
                # timestamp column is int64 unix-ms, sorted at write time (wide_df.to_parquet).
                    # filters skip row-groups outside [start_ms, end_ms] → reduced decode.
                    # Fallback: full-read + mask when filters raises (e.g. legacy pyarrow engine).
                    start_ms = int(req_start_dt.value // 1_000_000)
                    end_ms = int(req_end_dt.value // 1_000_000)
                    try:
                        df = pd.read_parquet(
                            enriched_path,
                            filters=[
                                ("timestamp", ">=", start_ms),
                                ("timestamp", "<=", end_ms),
                            ],
                        )
                        if not df.empty and "datetime" not in df.columns:
                            _ts_col = _resolve_timestamp_column(df)
                            if _ts_col is not None:
                                df["datetime"] = pd.to_datetime(df[_ts_col], unit="ms", utc=True)
                        # Row-group boundary precision: trim any residual rows outside window.
                        boundary_mask = (df["datetime"] >= req_start_dt) & (df["datetime"] <= req_end_dt)
                        df = df.loc[boundary_mask]
                    except Exception as _e:
                        _logger.debug(
                            "[%s] %s pushdown failed (%s), falling back to full-read",
                            sym,
                            tf_l,
                            _e,
                        )
                        df_full = pd.read_parquet(enriched_path)
                        if not df_full.empty and "datetime" not in df_full.columns:
                            _ts_col = _resolve_timestamp_column(df_full)
                            if _ts_col is not None:
                                df_full["datetime"] = pd.to_datetime(df_full[_ts_col], unit="ms", utc=True)
                        fallback_mask = (df_full["datetime"] >= req_start_dt) & (df_full["datetime"] <= req_end_dt)
                        df = df_full.loc[fallback_mask].copy()
                    if df.empty:
                        insufficient = True
                        break
                    _append_stage_integrity(integrity_audit, symbol=sym, timeframe=tf_l, stage="raw", df=df)
                    _append_stage_integrity(integrity_audit, symbol=sym, timeframe=tf_l, stage="merged", df=df)
                    from_cache = True

            if not from_cache:
                # cache-miss: funding/metrics needed for merge → load now (lazy)
                _ensure_fm_loaded()
                raw_df = collector.collect_and_save(sym, tf_l, fetch_start, end, fetch_network=False)
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
                    needs_funding = funding_df_prepared is not None and not funding_df_prepared.empty
                    needs_metrics = metrics_df_prepared is not None and not metrics_df_prepared.empty
                    needs_merge = needs_funding or needs_metrics

                    df = raw_df.copy() if needs_merge else raw_df
                    if needs_merge:
                        df["timestamp"] = _to_unix_ms(df["datetime"])
                        if not df["timestamp"].is_monotonic_increasing:
                            df = df.sort_values("timestamp")
                        if needs_funding:
                            df = pd.merge_asof(df, funding_df_prepared, on="timestamp", direction="backward")
                        if needs_metrics:
                            df = pd.merge_asof(
                                df,
                                metrics_df_prepared,
                                left_on="timestamp",
                                right_on="metrics_release_ts",
                                direction="backward",
                                tolerance=6 * 60 * 60 * 1000,
                                allow_exact_matches=True,
                            )
                    _append_stage_integrity(integrity_audit, symbol=sym, timeframe=tf_l, stage="merged", df=df)
                except Exception as e:
                    _logger.error("[%s] Merge/Enrich failed: %s", sym, e)
                    insufficient = True
                    break

                if df is None or df.empty or "datetime" not in df.columns:
                    insufficient = True
                    break

                if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

                for _mc in (
                    "coverage_60d",
                    "last_60d_coverage",
                    "vol_30d",
                    "friction_score",
                    "alpha_capacity_score",
                    "diversification_score",
                    "tradeable_score",
                    "cluster_id",
                    "beta_vs_market",
                    "cluster_size",
                    "anchor_cluster_member",
                ):
                    if _mc in df.columns:
                        df[_mc] = pd.to_numeric(df[_mc], errors="coerce")

                # Save enriched cache (full date range) for future runs
                enriched_stale = not _is_enriched_cache_fresh(enriched_path, deps)
                if enriched_stale:
                    try:
                        wide_df = collector.collect_and_save(sym, tf_l, "1970-01-01", "2099-12-31", fetch_network=False)
                        if wide_df is not None and not wide_df.empty:
                            if "datetime" not in wide_df.columns:
                                wide_df = wide_df.reset_index()
                            if not pd.api.types.is_datetime64_any_dtype(wide_df["datetime"]):
                                wide_df["datetime"] = pd.to_datetime(wide_df["datetime"], utc=True)
                            wide_df["timestamp"] = _to_unix_ms(wide_df["datetime"])
                            if needs_funding:
                                wide_df = pd.merge_asof(
                                    wide_df.sort_values("timestamp"),
                                    funding_df_prepared,
                                    on="timestamp",
                                    direction="backward",
                                )
                            if needs_metrics:
                                wide_df = pd.merge_asof(
                                    wide_df.sort_values("timestamp"),
                                    metrics_df_prepared,
                                    left_on="timestamp",
                                    right_on="metrics_release_ts",
                                    direction="backward",
                                    tolerance=6 * 60 * 60 * 1000,
                                    allow_exact_matches=True,
                                )
                            for _mc in (
                                "coverage_60d",
                                "last_60d_coverage",
                                "vol_30d",
                                "friction_score",
                                "alpha_capacity_score",
                                "diversification_score",
                                "tradeable_score",
                                "cluster_id",
                                "beta_vs_market",
                                "cluster_size",
                                "anchor_cluster_member",
                            ):
                                if _mc in wide_df.columns:
                                    wide_df[_mc] = pd.to_numeric(wide_df[_mc], errors="coerce")
                            # Optimize storage: drop datetime and cast price columns to float32
                            wide_df_to_save = wide_df.copy()
                            if "datetime" in wide_df_to_save.columns:
                                wide_df_to_save = wide_df_to_save.drop(columns=["datetime"])
                            for col in ["open", "high", "low", "close"]:
                                if col in wide_df_to_save.columns:
                                    wide_df_to_save[col] = wide_df_to_save[col].astype("float32")
                            wide_df_to_save.to_parquet(enriched_path, index=False, compression="zstd")
                            _write_enriched_cache_signature(enriched_path, deps)
                    except Exception as _ec:
                        _logger.debug("[%s] Failed to save enriched cache: %s", sym, _ec)

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
                exec_1m = collector.collect_1m_ohlcv(sym, fetch_start, end, fetch_network=False)
            except Exception as e:
                _logger.warning("[%s] collect_1m_ohlcv failed: %s", sym, e)
                exec_1m = pd.DataFrame()
            if isinstance(exec_1m, pd.DataFrame) and not exec_1m.empty:
                if "datetime" not in exec_1m.columns:
                    exec_1m = exec_1m.reset_index()
                    if "datetime" not in exec_1m.columns and len(exec_1m.columns) > 0:
                        exec_1m = exec_1m.rename(columns={str(exec_1m.columns[0]): "datetime"})
                if "datetime" in exec_1m.columns:
                    dt_ser = exec_1m["datetime"]
                    needs_convert = not pd.api.types.is_datetime64_any_dtype(dt_ser)
                    needs_sort = not dt_ser.is_monotonic_increasing
                    if needs_convert or needs_sort:
                        exec_1m = exec_1m.copy()
                        if needs_convert:
                            exec_1m["datetime"] = pd.to_datetime(exec_1m["datetime"], utc=True)
                        if needs_sort or not exec_1m["datetime"].is_monotonic_increasing:
                            exec_1m.sort_values("datetime", inplace=True)
                            exec_1m.reset_index(drop=True, inplace=True)
                    temp_is["exec_1m"] = exec_1m[exec_1m["datetime"] < is_end_dt].copy()
                    mask_exec_is = temp_is["exec_1m"]["datetime"] >= is_start_dt
                    temp_is["is_start_idx_exec_1m"] = int(mask_exec_is.to_numpy().argmax()) if mask_exec_is.any() else 0
                    temp_oos["exec_1m"] = exec_1m
                    mask_exec_oos = exec_1m["datetime"] >= is_end_dt
                    temp_oos["oos_start_idx_exec_1m"] = (
                        int(mask_exec_oos.to_numpy().argmax()) if mask_exec_oos.any() else len(exec_1m)
                    )

                    # exec_1m funding arrays need funding_df → lazy load if not yet loaded
                    _ensure_fm_loaded()
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
            temp_oos["feature_group_coverage"] = _feature_group_coverage(tf_df, tf_label=tf)
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
    """Load data for multiple symbols with 2-pass Arrow scan (P2).

    Pass-1 (C++ parallel I/O, GIL-free):
        Collect mtime-valid enriched paths → ``_scan_enriched_dataset`` →
        Arrow row-group predicate pushdown on ``timestamp`` column.

    Pass-2 (per-symbol Python post-processing):
        Arrow DataFrame → IS/OOS split → merge_idx → coverage.
        cache-miss symbols fall back to ``load_single_symbol_data`` via ThreadPool.
    """
    data_maps: dict[str, dict[str, Any]] = {}
    oos_data_maps: dict[str, dict[str, Any]] = {}
    valid_symbols: list[str] = []

    # [Fix] Filter out non-ASCII symbols before processing
    symbols = [s for s in symbols if all(ord(c) < 128 for c in s)]

    tfs_to_load: set[str] = set(target_tfs) if target_tfs else {tf, "1d", "1h", "4h"}
    req_start_dt = pd.Timestamp(fetch_start, tz="UTC")
    req_end_dt = pd.Timestamp(end, tz="UTC")
    start_ms = int(req_start_dt.value // 1_000_000)
    end_ms = int(req_end_dt.value // 1_000_000)
    use_exec_1m = _should_load_exec_1m(load_exec_1m)

    # ── Pass-1: classify cache-valid vs. cache-miss ──────────────────────────
    # valid_enriched: (sym, tf_l) → enriched_path (all TFs must be valid for sym)
    valid_enriched: dict[str, dict[str, Path]] = {}  # sym → {tf_l: path}
    cache_miss_syms: list[str] = []

    for sym in symbols:
        safe_sym = sym.replace("/", "_")
        sym_paths: dict[str, Path] = {}
        all_cache_hit = True
        for tf_l in tfs_to_load:
            from src.core.settings import FuturesStorageLayout

            enriched_path = FuturesStorageLayout.get_enriched_path(sym, tf_l)
            if enriched_path.exists():
                deps = [
                    FuturesStorageLayout.get_ohlcv_path(sym, tf_l),
                    FuturesStorageLayout.get_funding_path(sym),
                    FuturesStorageLayout.get_metrics_path(sym),
                ]
                if _is_enriched_cache_fresh(enriched_path, deps):
                    sym_paths[tf_l] = enriched_path
                    continue
            all_cache_hit = False
            break
        if all_cache_hit and len(sym_paths) == len(tfs_to_load):
            valid_enriched[sym] = sym_paths
        else:
            cache_miss_syms.append(sym)

    # exec_1m opt-out: Arrow fast-path does not handle exec_1m / funding arrays.
    # Route all valid_enriched symbols to ThreadPool fallback when exec_1m is required.
    if use_exec_1m and valid_enriched:
        _logger.debug(
            "exec_1m required — routing all %d symbols to ThreadPool fallback",
            len(valid_enriched),
        )
        cache_miss_syms.extend(list(valid_enriched.keys()))
        valid_enriched.clear()

    # Pass-1: Arrow scan per (sym, tf_l) — C++ multithreaded row-group decode
    # dict key: "{safe_sym}_{tf_l}" (matches _scan_enriched_dataset convention)
    arrow_frames: dict[str, pd.DataFrame] = {}
    if valid_enriched:
        all_paths = [path for sym_paths in valid_enriched.values() for path in sym_paths.values()]
        arrow_frames = _scan_enriched_dataset(all_paths, start_ms, end_ms)

    # ── Pass-2: per-symbol post-processing (Python-bound) ────────────────────
    is_start_dt = pd.Timestamp(start, tz="UTC")
    is_end_dt = pd.Timestamp(is_end, tz="UTC")
    min_bars_map = {"1h": 2000, "4h": 500, "1d": 300}

    for sym, sym_paths in valid_enriched.items():
        try:
            temp_is: dict[str, Any] = {}
            temp_oos: dict[str, Any] = {}
            integrity_audit: list[dict[str, Any]] = []
            insufficient = False
            safe_sym = sym.replace("/", "_")

            for tf_l in sym_paths:
                key = f"{safe_sym}_{tf_l}"
                raw_df = arrow_frames.get(key)
                if raw_df is None or raw_df.empty:
                    # Arrow scan skipped (missing timestamp col or error) → fallback
                    _logger.debug("[%s] %s not in arrow_frames — routing to fallback", sym, tf_l)
                    insufficient = True
                    break

                df = raw_df
                # Ensure datetime is UTC-aware after Arrow→pandas conversion
                if "datetime" not in df.columns:
                    insufficient = True
                    break
                if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                    df = df.copy()
                    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                elif df["datetime"].dt.tz is None:
                    df = df.copy()
                    df["datetime"] = df["datetime"].dt.tz_localize("UTC")

                _append_stage_integrity(integrity_audit, symbol=sym, timeframe=tf_l, stage="raw", df=df)
                _append_stage_integrity(integrity_audit, symbol=sym, timeframe=tf_l, stage="merged", df=df)

                dt_ns = df["datetime"].values.view("i8")
                is_end_dt_ns = is_end_dt.value
                is_end_idx = int(np.searchsorted(dt_ns, is_end_dt_ns, side="left"))
                min_bars_threshold = min_bars_map.get(tf_l, 300)
                if is_end_idx < min_bars_threshold:
                    _logger.debug("[%s] %s history too short (%d < %d)", sym, tf_l, is_end_idx, min_bars_threshold)
                    insufficient = True
                    break

                temp_is[tf_l] = df.iloc[:is_end_idx].copy()
                is_start_dt_ns = is_start_dt.value
                is_start_idx = int(np.searchsorted(dt_ns[:is_end_idx], is_start_dt_ns, side="left"))
                temp_is[f"is_start_idx_{tf_l}"] = is_start_idx
                temp_oos[tf_l] = df
                oos_start_idx = is_end_idx
                temp_oos[f"oos_start_idx_{tf_l}"] = oos_start_idx

            if insufficient:
                # Arrow pass failed for this sym → route to load_single_symbol_data
                cache_miss_syms.append(sym)
                continue

            # merge_idx requires both tf and "1d" frames
            if tf not in temp_oos or "1d" not in temp_oos:
                cache_miss_syms.append(sym)
                continue

            temp_is[f"merge_idx_{tf}"] = compute_segment_merge_index(temp_is[tf], temp_is["1d"])
            temp_oos[f"merge_idx_{tf}"] = compute_segment_merge_index(temp_oos[tf], temp_oos["1d"])
            temp_is["integrity_audit"] = integrity_audit
            temp_oos["integrity_audit"] = integrity_audit
            tf_df = temp_oos.get(tf)
            if isinstance(tf_df, pd.DataFrame) and not tf_df.empty:
                temp_oos["feature_group_coverage"] = _feature_group_coverage(tf_df, tf_label=tf)

            data_maps[sym] = temp_is
            oos_data_maps[sym] = temp_oos
            valid_symbols.append(sym)

        except Exception as _exc:
            _logger.debug("[%s] Pass-2 failure (%s) — routing to fallback", sym, _exc)
            cache_miss_syms.append(sym)

    # ── cache-miss fallback: original ThreadPool path ─────────────────────────
    if cache_miss_syms:
        _logger.debug("cache-miss fallback: %d symbols", len(cache_miss_syms))
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 2)) as executor:
            futures_map = [
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
                for sym in cache_miss_syms
            ]
            for f in concurrent.futures.as_completed(futures_map):
                sym, t_is, t_oos, insufficient = f.result()
                if not insufficient and t_is and t_oos:
                    data_maps[sym] = t_is
                    oos_data_maps[sym] = t_oos
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
        # OPT-3: merged stage stores {rows, cols} only; integrity cols may be absent
        _audit_agg_cols = [
            c
            for c in ("nan_pct", "inf_count", "gap_count", "duplicate_dt", "nonpositive_price_count")
            if c in audit_df.columns
        ]
        if _audit_agg_cols:
            grp = (
                audit_df.groupby(["stage", "timeframe"], dropna=False)[_audit_agg_cols]
                .mean(numeric_only=True)
                .reset_index()
            )
        else:
            grp = audit_df[["stage", "timeframe"]].drop_duplicates().reset_index(drop=True)
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
        audit_msg = f".. AUDIT: req={requested_count} load={loaded_count} coverage={coverage_val:.2f}"
        if failed_tfs:
            audit_msg += f" | !! FAIL: {', '.join(failed_tfs[:3])}"
        else:
            audit_msg += " | ok"
        _logger.debug(audit_msg)
    else:
        _logger.debug(
            ".. AUDIT: req=%d load=%d coverage=%.2f | ok (0 rows)",
            requested_count,
            loaded_count,
            float(loaded_count / max(requested_count, 1)),
        )

    return data_maps, oos_data_maps, valid_symbols


def load_ltf_exec_1m_frame(
    *,
    symbol: str,
    data_root: Path,
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
    required_columns: tuple[str, ...] = (
        "datetime",
        "high",
        "low",
        "close",
        "volume",
        "taker_buy_base_volume",
        "quote_vol",
        "trades",
    ),
) -> pd.DataFrame | None:
    """[ADR_20260713_TASK_L1_HYBRID_MEMORY_AUDIT] Load bounded 1m LTF data.

    Reads only required columns within [start_datetime, end_datetime].
    Returns None on any error (logged) — never raises.
    """
    path = FuturesStorageLayout.get_ohlcv_path(
        symbol,
        "1m",
        base_dir=data_root / "futures",
    )
    if not path.exists():
        _logger.debug("[LTF_1M] file not found symbol=%s path=%s", symbol, path)
        return None
    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(path), format="parquet")  # type: ignore[no-untyped-call]
        available_columns = set(dataset.schema.names)
        datetime_column = "datetime" if "datetime" in available_columns else "timestamp"
        required_source_columns = {
            datetime_column,
            "high",
            "low",
            "close",
            "volume",
            "taker_buy_base_volume",
        }
        if not required_source_columns.issubset(available_columns):
            _logger.warning("[LTF_1M] required columns missing symbol=%s", symbol)
            return None
        selected_columns = [column for column in required_columns if column in available_columns]
        if datetime_column not in selected_columns:
            selected_columns.append(datetime_column)
        if datetime_column == "timestamp":
            start_value = int(start_datetime.value // 1_000_000)
            end_value = int(end_datetime.value // 1_000_000)
        else:
            start_value = start_datetime
            end_value = end_datetime
        table = dataset.to_table(
            columns=selected_columns,
            filter=(
                (pc.field(datetime_column) >= start_value)  # type: ignore[no-untyped-call]
                & (pc.field(datetime_column) <= end_value)  # type: ignore[no-untyped-call]
            ),
        )
        if table.num_rows == 0:
            return None
        df = table.to_pandas()
        if datetime_column == "timestamp":
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df
    except Exception as exc:
        _logger.warning("[LTF_1M] load failed symbol=%s: %s", symbol, exc)
        return None
