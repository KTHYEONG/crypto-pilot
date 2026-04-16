"""Orchestrates GP → HMM → TBM → meta-labeler and injects ML columns into TF bars."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUTURES_CACHE_DIR, FUTURES_DATA_DIR
from src.domain.futures.data_collector import DataCollector
from src.domain.futures.funding_utils import merge_funding_into_ohlcv
from src.domain.futures.ml_pipeline.feature_engineering import build_gp_input_features
from src.domain.futures.ml_pipeline.gp_alpha_miner import GPAlphaMiner
from src.domain.futures.ml_pipeline.hmm_state_inferrer import HMMStateInferrer
from src.domain.futures.ml_pipeline.meta_labeler import MetaLabeler
from src.domain.futures.ml_pipeline.triple_barrier import label_triple_barrier

_logger = logging.getLogger(__name__)


def _safe_sym(sym: str) -> str:
    return sym.replace("/", "_")


@dataclass
class MLPipelineOutput:
    """Per-symbol calibrated probability series and enriched TF data_maps patch."""

    calib_prob_by_symbol: dict[str, pd.Series] = field(default_factory=dict)
    side_strength_by_symbol: dict[str, pd.Series] = field(default_factory=dict)
    meta_feature_frame_by_symbol: dict[str, pd.DataFrame] = field(default_factory=dict)


def _load_1h_frame(sym: str, fetch_start: str, end: str) -> pd.DataFrame:
    collector = DataCollector()
    raw = collector.collect_and_save(sym, "1h", fetch_start, end)
    return merge_funding_into_ohlcv(sym, raw, FUTURES_DATA_DIR)


def _align_asof_1h_to_tf(df_tf: pd.DataFrame, df_1h_feats: pd.DataFrame) -> pd.DataFrame:
    left = df_tf[["datetime"]].drop_duplicates().sort_values("datetime").copy()
    right = df_1h_feats.copy()
    if "datetime" not in right.columns and right.index.name == "datetime":
        right = right.reset_index()
    right = right.sort_values("datetime")
    # Normalize both keys to UTC: .values on tz-aware series can drop timezone info
    # in some pandas/pyarrow versions, causing MergeError on dtype mismatch.
    left["datetime"] = pd.to_datetime(left["datetime"], utc=True)
    right["datetime"] = pd.to_datetime(right["datetime"], utc=True)
    return pd.merge_asof(left, right, on="datetime", direction="backward")


def _one_symbol(
    sym: str,
    tf: str,
    data_maps: dict[str, dict[str, Any]],
    oos_data_maps: dict[str, dict[str, Any]],
    df_1h: pd.DataFrame,
    fetch_start: str,
    end: str,
    is_end_dt: pd.Timestamp,
    config: dict[str, Any],
    n_jobs: int = 1,
) -> tuple[str, pd.Series | None, pd.Series | None, pd.DataFrame | None]:
    dmap = data_maps.get(sym)
    oos = oos_data_maps.get(sym)
    if dmap is None or oos is None or tf not in dmap or tf not in oos:
        return sym, None, None, None

    df_tf_is = dmap[tf]
    df_tf_full = oos[tf]

    if df_1h.empty or len(df_1h) < 500:
        return sym, None, None, None

    _logger.info("[%s] GP alpha mining start (%d 1h bars)...", sym, len(df_1h))
    gp_feats = build_gp_input_features(df_1h)
    gp_feats["close"] = df_1h["close"].astype(np.float64)

    miner = GPAlphaMiner(
        n_features_to_select=int(config.get("FUTURES_ML_GP_N_ALPHAS", 15)),
        population_size=int(config.get("FUTURES_ML_GP_POPULATION", 2000)),
        generations=int(config.get("FUTURES_ML_GP_GENERATIONS", 20)),
        n_jobs=n_jobs,
    )
    cache_p = Path(FUTURES_CACHE_DIR) / f"{_safe_sym(sym)}_gp_alphas_{tf}.parquet"
    alpha_df = miner.mine_alphas(gp_feats, cache_path=cache_p)
    alpha_df.insert(0, "datetime", df_1h["datetime"].values)
    _logger.info("[%s] GP alpha mining done.", sym)

    _logger.info("[%s] HMM state inference start...", sym)
    hmm = HMMStateInferrer(n_states=int(config.get("FUTURES_ML_HMM_N_STATES", 3)))
    hmm_df = hmm.fit_predict(df_1h, is_end_idx=len(df_1h), symbol=sym, tf=tf)
    hmm_df.insert(0, "datetime", df_1h["datetime"].values)
    _logger.info("[%s] HMM done.", sym)

    wide_1h = alpha_df.merge(hmm_df, on="datetime", how="inner")
    aligned_full = _align_asof_1h_to_tf(df_tf_full, wide_1h)
    gp_cols = [c for c in aligned_full.columns if c.startswith("gp_alpha_")]
    if gp_cols:
        side = aligned_full[gp_cols[:3]].sum(axis=1)
    else:
        side = pd.Series(0.0, index=aligned_full.index)
    med = float(side.abs().median()) if len(side) else 1.0
    aligned_full["ml_side_strength"] = np.tanh(side / (med + 1e-9))

    # 1m fetch window: IS period start ~ is_end + TBM time-stop buffer.
    # Full fetch_start (4+ years) would produce ~2.5M candles and take 30+ min.
    cfg_tbm_h = int(config.get("FUTURES_ML_TBM_TIME_STOP_H", 24))
    if not df_tf_is.empty and "datetime" in df_tf_is.columns:
        is_start_ts = df_tf_is["datetime"].min()
        fetch_start_1m = str(is_start_ts.date()) if hasattr(is_start_ts, "date") else fetch_start
    else:
        fetch_start_1m = fetch_start
    end_1m = str((is_end_dt + pd.Timedelta(hours=cfg_tbm_h + 24)).date())

    _logger.info("[%s] TBM labeling: fetching 1m data (%s ~ %s)...", sym, fetch_start_1m, end_1m)
    collector = DataCollector()
    try:
        df_1m = collector.collect_1m_ohlcv(sym, fetch_start_1m, end_1m)
    except Exception as e:
        _logger.error("[%s] 1m data fetch failed: %s (will use fallback labeling)", sym, e)
        df_1m = pd.DataFrame()
    _logger.info("[%s] TBM labeling: 1m ready (%d bars), fitting barriers...", sym, len(df_1m))
    tbm = (
        label_triple_barrier(
            df_tf_is,
            df_1m,
            tp_atr_mult=float(config.get("FUTURES_ML_TBM_TP_ATR_MULT", 1.0)),
            sl_atr_mult=float(config.get("FUTURES_ML_TBM_SL_ATR_MULT", 1.0)),
            time_stop_bars=max(1, cfg_tbm_h * 60),
        )
        if not df_1m.empty
        else pd.Series(0.5, index=df_tf_is["datetime"])
    )

    feat_cols = [c for c in aligned_full.columns if c.startswith(("gp_alpha_", "hmm_prob"))]
    X_full = aligned_full[feat_cols].copy()
    X_full["ml_side_strength"] = aligned_full["ml_side_strength"]
    y_full = df_tf_full["datetime"].map(tbm.reindex(df_tf_full["datetime"]).to_dict())
    y_full = pd.to_numeric(y_full, errors="coerce").fillna(0.5)

    is_end_idx = int((df_tf_full["datetime"] < is_end_dt).sum())
    is_end_idx = max(1, min(is_end_idx, len(X_full)))

    _logger.info("[%s] MetaLabeler fitting...", sym)
    meta = MetaLabeler()
    meta.fit(X_full, y_full, is_end_idx=is_end_idx)
    calib = meta.predict_proba_calibrated(X_full)
    calib_series = pd.Series(calib, index=df_tf_full["datetime"])
    ss_arr = aligned_full["ml_side_strength"].to_numpy()
    side_series = pd.Series(ss_arr, index=df_tf_full["datetime"])
    _logger.info("[%s] ML pipeline complete.", sym)
    return sym, calib_series, side_series, X_full


def run_ml_pipeline(
    data_maps: dict[str, dict[str, Any]],
    oos_data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    config: dict[str, Any],
    fetch_start: str,
    end: str,
    is_end_date: str,
    is_start: str | None = None,
) -> MLPipelineOutput:
    """Parallel per-symbol ML pipeline; mutates data_maps and oos_data_maps TF frames."""
    out = MLPipelineOutput()
    cfg = {**OPT_FUTURES_CONFIG, **config}
    workers = min(4, max(1, len(symbols)))
    is_end_dt = pd.Timestamp(is_end_date)
    if is_end_dt.tzinfo is None:
        is_end_dt = is_end_dt.tz_localize("UTC")

    # 1h data를 단일 DataCollector로 순차 pre-fetch.
    # 다수 DataCollector 인스턴스가 동시에 exchange.load_markets()를 호출하면
    # Binance API hang이 발생하므로 직렬화하여 제거.
    # fetch_start(IS 시작 700일 전 warm-up 기간)를 그대로 사용해 GP/HMM 품질 유지.
    _ = is_start  # reserved for future use
    _logger.info(
        "ML pipeline: pre-fetching 1h data for %d symbols (sequential) from %s ...",
        len(symbols),
        fetch_start,
    )
    collector_1h = DataCollector()
    prefetched_1h: dict[str, pd.DataFrame] = {}
    failed_symbols: list[str] = []
    for sym in symbols:
        try:
            raw = collector_1h.collect_and_save(sym, "1h", fetch_start, end)
            df_1h = merge_funding_into_ohlcv(sym, raw, FUTURES_DATA_DIR)
            if df_1h.empty or len(df_1h) < 500:
                _logger.error("ML pipeline: 1h insufficient %s (need 500, got %d)", sym, len(df_1h))
                failed_symbols.append(sym)
                prefetched_1h[sym] = pd.DataFrame()
            else:
                prefetched_1h[sym] = df_1h
                _logger.info("ML pipeline: 1h ready %s (%d bars)", sym, len(df_1h))
        except Exception as e:
            _logger.error("ML pipeline: 1h pre-fetch failed %s: %s", sym, e)
            failed_symbols.append(sym)
            prefetched_1h[sym] = pd.DataFrame()

    if failed_symbols:
        _logger.warning("ML pipeline: skipping %d symbols due to data fetch failure: %s",
                       len(failed_symbols), failed_symbols)

    # 수집 성공한 심볼만 processing
    valid_symbols = [s for s in symbols if s not in failed_symbols]
    if not valid_symbols:
        _logger.error("ML pipeline: no valid symbols after 1h pre-fetch. Aborting.")
        return out

    # 1m data(TBM용)를 단일 DataCollector로 순차 pre-fetch.
    # 병렬 처리(ThreadPool) 시 다수 스레드가 동시에 대량의 1m 데이터를 요청하면 
    # Binance 가중치 제한(HTTP 429)에 걸릴 확률이 높으므로 직렬화.
    _logger.info(
        "ML pipeline: pre-fetching 1m data for %d symbols (sequential) for TBM labeling...",
        len(valid_symbols),
    )
    collector_1m = DataCollector()
    cfg_tbm_h = int(cfg.get("FUTURES_ML_TBM_TIME_STOP_H", 24))
    end_1m = str((is_end_dt + pd.Timedelta(hours=cfg_tbm_h + 24)).date())

    for sym in valid_symbols:
        try:
            df_tf_is = data_maps[sym][tf]
            if not df_tf_is.empty and "datetime" in df_tf_is.columns:
                is_start_ts = df_tf_is["datetime"].min()
                if hasattr(is_start_ts, "date"):
                    fetch_start_1m = str(is_start_ts.date())
                else:
                    fetch_start_1m = fetch_start
            else:
                fetch_start_1m = fetch_start
            
            _logger.info("[%s] 1m pre-fetch: %s ~ %s", sym, fetch_start_1m, end_1m)
            collector_1m.collect_1m_ohlcv(sym, fetch_start_1m, end_1m)
        except Exception as e:
            _logger.error("ML pipeline: 1m pre-fetch failed for %s: %s", sym, e)

    import os
    logical_cpus = os.cpu_count() or 1
    n_jobs_per_worker = max(1, logical_cpus // workers)

    with ThreadPoolExecutor(max_workers=min(workers, len(valid_symbols))) as ex:
        futs = [
            ex.submit(
                _one_symbol,
                sym,
                tf,
                data_maps,
                oos_data_maps,
                prefetched_1h.get(sym, pd.DataFrame()),
                fetch_start,
                end,
                is_end_dt,
                cfg,
                n_jobs_per_worker,
            )
            for sym in valid_symbols
        ]
        for f in as_completed(futs):
            try:
                sym, calib_s, side_s, xdf = f.result()
                if calib_s is not None:
                    out.calib_prob_by_symbol[sym] = calib_s
                if side_s is not None:
                    out.side_strength_by_symbol[sym] = side_s
                if xdf is not None:
                    out.meta_feature_frame_by_symbol[sym] = xdf
            except Exception as e:
                _logger.error("ML pipeline: symbol processing failed: %s", e, exc_info=True)

    for sym in symbols:
        cal = out.calib_prob_by_symbol.get(sym)
        side = out.side_strength_by_symbol.get(sym)
        if cal is None:
            continue
        for bucket in (data_maps, oos_data_maps):
            if sym not in bucket or tf not in bucket[sym]:
                continue
            df = bucket[sym][tf]
            # cal/side are indexed by oos_data_maps datetime (full period).
            # data_maps holds IS-only slice (shorter), so re-creating a Series with
            # df["datetime"] as index causes ValueError (length mismatch).
            # Use the original Series directly for index-based mapping.
            bucket[sym][tf] = df.copy()
            bucket[sym][tf]["ml_calib_prob"] = (
                df["datetime"].map(cal).astype(np.float64).fillna(0.5)
            )
            if side is not None:
                bucket[sym][tf]["ml_side_strength"] = (
                    df["datetime"].map(side).astype(np.float64).fillna(0.0)
                )

    # ML pipeline 처리 완료 요약
    successful = len(out.calib_prob_by_symbol)
    total = len(symbols)
    skipped = len(failed_symbols)
    _logger.info(
        "ML pipeline complete: %d/%d symbols processed, %d failed pre-fetch, %d no output",
        successful, total, skipped, total - successful - skipped
    )

    return out
