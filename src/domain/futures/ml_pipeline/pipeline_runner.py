"""ML pipeline execution orchestration for Cross-Sectional Ranking Portfolio."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.isotonic import IsotonicRegression

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUTURES_CACHE_DIR, FUTURES_DATA_DIR

_logger = logging.getLogger(__name__)

# Initialize joblib Memory for disk caching
_memory = Memory(FUTURES_CACHE_DIR, verbose=0)
from src.domain.futures.data_loader import (
    DataCollector,
    merge_funding_into_ohlcv,
    merge_metrics_into_ohlcv,
    summarize_dataframe_integrity,
)
from src.domain.futures.ml_pipeline.alpha.miner import MLAlphaMiner
from src.domain.futures.ml_pipeline.features.cross_sectional import CrossSectionalPipelineUtils
from src.domain.futures.ml_pipeline.features.engineering import (
    ALPHA_ENGINEERED_FEATURE_NAMES,
    GP_FEATURE_SCHEMA_VERSION,
    HMM_SEMANTIC_PROB_COLUMNS,
    build_gp_input_features,
)
from src.domain.futures.ml_pipeline.labels.meta_labeler import MetaLabeler
from src.domain.futures.ml_pipeline.labels.triple_barrier import label_triple_barrier
from src.domain.futures.ml_pipeline.regime.hmm_inferrer import (
    build_hmm_inferrer_from_config,
)
from src.domain.futures.ml_pipeline.regime.tail_overlay import fit_predict_tail_overlay
from src.domain.futures.optimization.optimizer import SignalCalibrator

_logger = logging.getLogger(__name__)

_ALPHA_CACHE_SCHEMA_VERSION = "v1"
_ALPHA_SAFE_DEFAULT_MAX_ITEMS = 2
_alpha_cache_lock = threading.Lock()
_alpha_cache_store: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _hash_jsonable(payload: dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _alpha_data_snapshot_id(panel_df: pd.DataFrame) -> str:
    row_hash = pd.util.hash_pandas_object(panel_df, index=True).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.md5(row_hash.tobytes()).hexdigest()
    shape_sig = f"{panel_df.shape[0]}x{panel_df.shape[1]}"
    cols_sig = hashlib.md5("|".join(str(c) for c in panel_df.columns).encode()).hexdigest()[:12]
    return f"{shape_sig}:{cols_sig}:{digest}"


def _build_alpha_cache_key(
    *,
    panel_df: pd.DataFrame,
    tf: str,
    is_end_date: str | None,
    seed: int | None,
    cfg: dict[str, Any],
    horizons: tuple[int, ...],
    slots_per_theme: int,
    filter_options: dict[str, Any],
) -> str:
    model_cfg = {
        "gp_feature_schema_version": GP_FEATURE_SCHEMA_VERSION,
        "slots_per_theme": int(slots_per_theme),
        "horizons": tuple(int(h) for h in horizons),
        "seed": int(seed or 0),
        "alpha_target_half_life": float(OPT_FUTURES_CONFIG.get("FUTURES_ML_IC_HALF_LIFE", 2.3)),
        "filter_options": filter_options,
        "cfg_subset": {
            "FUTURES_ML_ALPHA_SLOTS_PER_THEME": cfg.get("FUTURES_ML_ALPHA_SLOTS_PER_THEME"),
            "FUTURES_ML_ALPHA_HORIZONS": cfg.get("FUTURES_ML_ALPHA_HORIZONS"),
            "FUTURES_ML_IC_FDR_Q": cfg.get("FUTURES_ML_IC_FDR_Q"),
            "FUTURES_ML_IC_SYMBOL_BALANCE_MAX": cfg.get("FUTURES_ML_IC_SYMBOL_BALANCE_MAX"),
            "FUTURES_ML_IC_REGIME_GATE": cfg.get("FUTURES_ML_IC_REGIME_GATE"),
            "FUTURES_ML_IC_FILTER_USE_HAC": cfg.get("FUTURES_ML_IC_FILTER_USE_HAC"),
            "FUTURES_ML_IC_FILTER_USE_EWMA": cfg.get("FUTURES_ML_IC_FILTER_USE_EWMA"),
            "FUTURES_ML_IC_EWMA_HALF_LIFE": cfg.get("FUTURES_ML_IC_EWMA_HALF_LIFE"),
            "FUTURES_STEP3_REGIME_ALPHA_ENABLED": cfg.get("FUTURES_STEP3_REGIME_ALPHA_ENABLED"),
            "FUTURES_STEP3_CHOP_SUPPORT_MIN": cfg.get("FUTURES_STEP3_CHOP_SUPPORT_MIN"),
            "FUTURES_STEP3_CHOP_IC_MIN": cfg.get("FUTURES_STEP3_CHOP_IC_MIN"),
            "FUTURES_STEP3_CHOP_WEIGHT_MULT": cfg.get("FUTURES_STEP3_CHOP_WEIGHT_MULT"),
            "FUTURES_STEP3_WEIGHT_MULT_FLOOR": cfg.get("FUTURES_STEP3_WEIGHT_MULT_FLOOR"),
        },
    }
    payload = {
        "schema": _ALPHA_CACHE_SCHEMA_VERSION,
        "reference_date": str(is_end_date or ""),
        "tf": str(tf),
        "data_snapshot_id": _alpha_data_snapshot_id(panel_df),
        "model_cfg_fingerprint": _hash_jsonable(model_cfg),
    }
    return _hash_jsonable(payload)


def _resolve_alpha_cache_limits(cfg: dict[str, Any]) -> tuple[bool, int]:
    enabled = bool(cfg.get("FUTURES_ML_ALPHA_CACHE_ENABLED", True))
    max_items = int(cfg.get("FUTURES_ML_ALPHA_CACHE_MAX_ITEMS", _ALPHA_SAFE_DEFAULT_MAX_ITEMS))
    return enabled, max(0, max_items)


def _alpha_cache_get(key: str) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    with _alpha_cache_lock:
        entry = _alpha_cache_store.get(key)
        if entry is None:
            _logger.info("ALPHA_CACHE miss key=%s reason=not_found", key[:12])
            return None, None
        if str(entry.get("key")) != key:
            _logger.warning("ALPHA_CACHE miss key=%s reason=key_mismatch", key[:12])
            return None, None
        _alpha_cache_store.move_to_end(key)
        hit_meta = {
            "cache_state": "hit",
            "cache_key": key,
            "cached_at_epoch_s": float(entry.get("created_at", 0.0)),
            "cache_schema": _ALPHA_CACHE_SCHEMA_VERSION,
        }
        _logger.info("ALPHA_CACHE hit key=%s", key[:12])
        return cast(pd.DataFrame, entry["alpha_panel"]).copy(deep=True), hit_meta


def _alpha_cache_put(
    key: str,
    alpha_panel: pd.DataFrame,
    *,
    max_items: int,
) -> dict[str, Any]:
    evicted_key: str | None = None
    with _alpha_cache_lock:
        _alpha_cache_store[key] = {
            "key": key,
            "alpha_panel": alpha_panel.copy(deep=True),
            "created_at": time.time(),
        }
        _alpha_cache_store.move_to_end(key)
        while max_items >= 0 and len(_alpha_cache_store) > max_items:
            old_key, _ = _alpha_cache_store.popitem(last=False)
            if old_key != key:
                evicted_key = old_key
            else:
                break
    if evicted_key:
        _logger.info("ALPHA_CACHE evict key=%s reason=lru_limit", evicted_key[:12])
    _logger.info("ALPHA_CACHE store key=%s", key[:12])
    return {
        "cache_state": "miss_stored",
        "cache_key": key,
        "cache_schema": _ALPHA_CACHE_SCHEMA_VERSION,
        "evicted_key": evicted_key,
    }


@dataclass
class MLPipelineOutput:
    """Container for ML pipeline results by symbol."""

    calib_prob_by_symbol: dict[str, pd.Series] = field(default_factory=dict)
    calib_prob_long_by_symbol: dict[str, pd.Series] = field(default_factory=dict)
    calib_prob_short_by_symbol: dict[str, pd.Series] = field(default_factory=dict)
    meta_feature_frame_by_symbol: dict[str, pd.DataFrame] = field(default_factory=dict)
    health_metrics_by_symbol: dict[str, dict[str, float]] = field(default_factory=dict)
    alpha_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    hmm_report: dict[str, float] = field(default_factory=dict)
    market_probs: pd.DataFrame = field(default_factory=pd.DataFrame)
    integrity_report: dict[str, Any] = field(default_factory=dict)


def _sorted_hmm_prob_columns(df: pd.DataFrame) -> list[str]:
    sem = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in df.columns]
    if sem:
        return sem
    # Only include columns where the last part is a digit to avoid ValueError (e.g., hmm_prob_0_x)
    legacy = [
        c for c in df.columns
        if str(c).startswith("hmm_prob_") and str(c).split("_")[-1].isdigit()
    ]
    return sorted(legacy, key=lambda x: int(str(x).split("_")[-1]))


def _resolve_hmm_backend_name(hmm_inferrer: Any, cfg: dict[str, Any] | None = None) -> str:
    """Resolve HMM backend name for lightweight reporting."""
    if cfg is not None:
        cfg_backend = cfg.get("FUTURES_HMM_BACKEND")
        if isinstance(cfg_backend, str) and cfg_backend.strip():
            return cfg_backend.strip().lower()
    model = getattr(hmm_inferrer, "_jax_model", None)
    if model is not None:
        name = type(model).__name__.lower()
        if "student" in name and "hmm" in name:
            return "student_t"
    return "jax"


def _resolve_trace_run_id(
    *,
    explicit_run_id: str | None = None,
    tf: str | None = None,
    is_end_date: str | pd.Timestamp | None = None,
    symbol_count: int | None = None,
    seed: int | None = None,
) -> str:
    """Resolve run trace key; fallback to stable execution id when run_id is unavailable."""
    if explicit_run_id and str(explicit_run_id).strip():
        return str(explicit_run_id).strip()
    env_run_id = os.getenv("FUTURES_RUN_ID", "").strip()
    if env_run_id:
        return env_run_id
    stable_payload = {
        "tf": str(tf or "n/a"),
        "is_end_date": str(is_end_date or "n/a"),
        "symbol_count": int(symbol_count or 0),
        "seed": int(seed or 0),
    }
    stable_str = json.dumps(stable_payload, sort_keys=True)
    return f"exec-{hashlib.md5(stable_str.encode()).hexdigest()[:12]}"


def _resolve_cfg_fingerprint(cfg: dict[str, Any] | None) -> str:
    if not isinstance(cfg, dict):
        return "stable_unavailable"
    try:
        return _get_cfg_hash(cfg)
    except Exception:
        return "stable_unavailable"


def _compact_tvtp_snapshot(cfg: dict[str, Any] | None) -> str:
    if not isinstance(cfg, dict):
        return "n/a"

    def _fmt_num(key: str) -> str:
        val = cfg.get(key)
        try:
            return f"{float(val):.3f}"
        except Exception:
            return "n/a"

    enabled_raw = cfg.get("FUTURES_HMM_TVTP_ENABLED")
    if enabled_raw is None:
        enabled = "n/a"
    else:
        enabled = "1" if bool(enabled_raw) else "0"
    return (
        f"enabled={enabled}"
        f",diag_slope={_fmt_num('FUTURES_HMM_TVTP_DIAG_SLOPE')}"
        f",diag_clip={_fmt_num('FUTURES_HMM_TVTP_DIAG_CLIP')}"
        f",sticky_min_mult={_fmt_num('FUTURES_HMM_TVTP_STICKY_PRIOR_MIN_MULT')}"
        f",sticky_max_mult={_fmt_num('FUTURES_HMM_TVTP_STICKY_PRIOR_MAX_MULT')}"
    )


def _build_hmm_traceability(
    *,
    cfg: dict[str, Any] | None,
    tf: str,
    is_end_date: str | pd.Timestamp | None,
    symbol_count: int,
    seed: int | None,
    explicit_run_id: str | None,
    backend: str | None,
    cache_state: str | None,
) -> dict[str, str]:
    return {
        "run_id": _resolve_trace_run_id(
            explicit_run_id=explicit_run_id,
            tf=tf,
            is_end_date=is_end_date,
            symbol_count=symbol_count,
            seed=seed,
        ),
        "config_fingerprint": _resolve_cfg_fingerprint(cfg),
        "hmm_backend": str(backend or "n/a"),
        "tvtp_snapshot": _compact_tvtp_snapshot(cfg),
        "hmm_cache_state": str(cache_state or "n/a"),
    }


def _drop_ml_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Unify removal of existing ML/HMM columns to prevent merge collisions."""
    ml_reserved = [
        "alpha_long_00", "alpha_long", "alpha_short",
        "hmm_modulator_long", "hmm_modulator_short",
        "slot_rank_score", "xs_score_long", "xs_score_short",
        "ml_calib_prob", "ml_calib_prob_long", "ml_calib_prob_short",
        "hmm_prob_bull_trend", "hmm_prob_bear_trend", "hmm_prob_chop", "hmm_prob_crisis"
    ]
    # Drop reserved patterns and any column starting with hmm_
    to_drop = [
        c for c in df.columns 
        if any(p in str(c) for p in ml_reserved) or str(c).startswith("hmm_")
    ]
    if to_drop:
        return df.drop(columns=to_drop)
    return df


_META_EXTRA_FEATS: tuple[str, ...] = (
    "funding_z_72",
    "realized_vol_yz_24",
    "vol_surface_24_168",
    "corr_btc_24",
    "vpin_proxy_12",
    "ret_vol_adj_24",
    "downside_jump_24",
)


def _meta_feature_column_names(wide_1h: pd.DataFrame) -> tuple[str, ...]:
    """Resolve which columns to use as features for the meta-labeler."""
    hmm_cols = _sorted_hmm_prob_columns(wide_1h)
    base: list[str] = []
    if "alpha_long_00" in wide_1h.columns:
        base.append("alpha_long_00")
    base.extend(hmm_cols)
    # 추가 피처: wide_1h에 실제 존재하는 것만 포함
    for c in _META_EXTRA_FEATS:
        if c in wide_1h.columns and c not in base:
            base.append(c)
    return tuple(base)


def _attach_tbm_gp_weights(
    sym: str,
    df_1h: pd.DataFrame,
    label_start: str,
    end: str,
    collector: DataCollector,
    df_1m: pd.DataFrame | None,
) -> pd.DataFrame:
    """Up-weight rows with clear triple-barrier (+1 / -1) hits for GP fitness."""
    out = df_1h.copy()
    lab = _try_tbm_labels_per_1h_row(sym, out, label_start, end, df_1m=df_1m, collector=collector)
    if lab is None or len(lab) != len(out):
        out["tbm_gp_weight"] = 1.0
        return out
    out["tbm_gp_weight"] = np.where(
        np.isfinite(lab) & (np.abs(lab) > 0.9), 1.5, 1.0
    ).astype(np.float64)
    return out


def _enrich_with_gp_features(df: pd.DataFrame, tf: str = "1h") -> pd.DataFrame:
    """Append GP microstructure / momentum columns to OHLCV (sorted by datetime)."""
    out = df.copy()
    if "open" not in out.columns:
        out["open"] = pd.to_numeric(out["close"], errors="coerce").shift(1).fillna(out["close"])
    w = out.sort_values("datetime").reset_index(drop=True)
    idx = pd.DatetimeIndex(pd.to_datetime(w["datetime"], utc=True))
    gp = build_gp_input_features(w.set_index(idx), tf=tf)
    for col in gp.columns:
        w[col] = gp[col].to_numpy()
    return w


def _ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        return df
    out = df.reset_index()
    if "datetime" not in out.columns and len(out.columns) > 0:
        c0 = str(out.columns[0])
        if c0 != "datetime":
            out = out.rename(columns={c0: "datetime"})
    return out


def _run_systemic_hmm_with_causal_split(
    hmm_inferrer: Any,
    market_hmm_feats: pd.DataFrame,
    market_returns: pd.Series,
    is_end_idx_market: int,
    tf: str,
    *,
    symbol: str = "Market",
) -> pd.DataFrame:
    """Run systemic HMM with causal IS/OOS boundary handling in pipeline layer.

    Current inferrer API does not expose frozen-parameter OOS filter, so OOS posterior
    is emitted via last-IS posterior carry-forward to avoid OOS refit leakage.
    """
    n_total = len(market_hmm_feats)
    split = int(np.clip(is_end_idx_market, 0, n_total))
    _logger.info(
        "🧭 HMM causal boundary | total=%d | is_end_idx=%d | is=%d | oos=%d",
        n_total,
        split,
        split,
        max(0, n_total - split),
    )

    if n_total <= 0:
        return pd.DataFrame({"datetime": pd.to_datetime([], utc=True)})

    if split <= 0 or split >= n_total:
        probs = hmm_inferrer.fit_predict_systemic(
            market_hmm_feats,
            market_returns,
            is_end_idx=split,
            symbol=symbol,
            tf=tf,
        )
        probs = _ensure_datetime_column(probs)
        probs["datetime"] = pd.to_datetime(probs["datetime"], utc=True)
        return probs

    is_feats = market_hmm_feats.iloc[:split]
    is_rets = market_returns.iloc[:split]
    is_probs = hmm_inferrer.fit_predict_systemic(
        is_feats,
        is_rets,
        is_end_idx=len(is_feats),
        symbol=symbol,
        tf=tf,
    )
    is_probs = _ensure_datetime_column(is_probs)
    is_probs["datetime"] = pd.to_datetime(is_probs["datetime"], utc=True)

    oos_dt = pd.to_datetime(market_hmm_feats.index[split:], utc=True)
    oos_n = len(oos_dt)
    if oos_n <= 0:
        return is_probs

    tail_row = is_probs.iloc[[-1]].drop(columns=["datetime"], errors="ignore").copy()
    oos_probs = pd.concat([tail_row] * oos_n, ignore_index=True)
    oos_probs.insert(0, "datetime", oos_dt.to_numpy())

    all_cols = list(is_probs.columns)
    combined = pd.concat([is_probs, oos_probs], ignore_index=True)
    combined = combined.reindex(columns=all_cols)
    return combined


def _is_kelly_per_semantic_state(
    p_mat: np.ndarray,
    fwd_ret: np.ndarray,
    is_mask: np.ndarray,
    n_states: int,
    cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
    """James-Stein shrunk Kelly on global variance, plus soft Isotonic blend (tmp v4)."""
    kelly_long: np.ndarray = np.zeros(n_states, dtype=np.float64)
    kelly_short: np.ndarray = np.zeros(n_states, dtype=np.float64)
    state_hard = np.argmax(p_mat, axis=1).astype(np.int64)

    fwd_is = fwd_ret[is_mask]
    if fwd_is.size < 2:
        return kelly_long, kelly_short
    global_mu = float(np.mean(fwd_is))
    global_v = float(np.var(fwd_is, ddof=1)) + 1e-12

    for s in range(n_states):
        m = is_mask & (state_hard == s)
        n_s = int(np.sum(m))
        if n_s < 30:
            continue
        r = fwd_ret[m]
        mu = float(np.mean(r))

        # [REFACTORED] Stronger Relative Kelly (Demeaning)
        # Subtract FULL market average to force alpha-only Longs.
        mu_relative = mu - global_mu

        # James-Stein shrink towards zero (neutral)
        alpha_js = 30.0 / (30.0 + float(n_s))
        mu_shrunk = alpha_js * 0.0 + (1.0 - alpha_js) * mu_relative

        kelly_long[s] = float(np.clip(mu_shrunk / global_v, -1.0, 1.0))
        kelly_short[s] = float(np.clip(-mu_shrunk / global_v, -1.0, 1.0))

    alpha_iso = 0.5
    try:
        ir = IsotonicRegression(increasing=False)

        if (
            len(cols) == 4
            and "hmm_prob_bull_trend" in cols
            and all(c in cols for c in ("hmm_prob_chop", "hmm_prob_bear_trend", "hmm_prob_crisis"))
        ):
            idx_bull = cols.index("hmm_prob_bull_trend")
            idx_chop = cols.index("hmm_prob_chop")
            idx_bear = cols.index("hmm_prob_bear_trend")
            idx_crisis = cols.index("hmm_prob_crisis")
            x = np.array([0, 1, 2, 3], dtype=np.float64)
            y_l = np.array(
                [
                    kelly_long[idx_bull],
                    kelly_long[idx_chop],
                    kelly_long[idx_bear],
                    kelly_long[idx_crisis],
                ],
                dtype=np.float64,
            )
            y_l_adj = ir.fit_transform(x, y_l)
            kelly_long[idx_bull] = (1.0 - alpha_iso) * y_l[0] + alpha_iso * y_l_adj[0]
            kelly_long[idx_chop] = (1.0 - alpha_iso) * y_l[1] + alpha_iso * y_l_adj[1]
            kelly_long[idx_bear] = (1.0 - alpha_iso) * y_l[2] + alpha_iso * y_l_adj[2]
            kelly_long[idx_crisis] = (1.0 - alpha_iso) * y_l[3] + alpha_iso * y_l_adj[3]
            kelly_long[idx_bull] = float(max(0.0, kelly_long[idx_bull]))

            y_s = np.array(
                [
                    kelly_short[idx_bear],
                    kelly_short[idx_chop],
                    kelly_short[idx_bull],
                    kelly_short[idx_crisis],
                ],
                dtype=np.float64,
            )
            y_s_adj = ir.fit_transform(x, y_s)
            kelly_short[idx_bear] = (1.0 - alpha_iso) * y_s[0] + alpha_iso * y_s_adj[0]
            kelly_short[idx_chop] = (1.0 - alpha_iso) * y_s[1] + alpha_iso * y_s_adj[1]
            kelly_short[idx_bull] = (1.0 - alpha_iso) * y_s[2] + alpha_iso * y_s_adj[2]
            kelly_short[idx_crisis] = (1.0 - alpha_iso) * y_s[3] + alpha_iso * y_s_adj[3]
        elif len(cols) == 5 and all(c in cols for c in HMM_SEMANTIC_PROB_COLUMNS):
            order_idx = [cols.index(c) for c in HMM_SEMANTIC_PROB_COLUMNS]
            x = np.arange(5, dtype=np.float64)
            y_l = np.array([kelly_long[i] for i in order_idx], dtype=np.float64)
            y_l_adj = ir.fit_transform(x, y_l)
            for j, i in enumerate(order_idx):
                kelly_long[i] = (1.0 - alpha_iso) * y_l[j] + alpha_iso * y_l_adj[j]
            for j in (0, 1):
                kelly_long[order_idx[j]] = float(max(0.0, kelly_long[order_idx[j]]))

            y_s = np.array(
                [
                    kelly_short[order_idx[2]],
                    kelly_short[order_idx[3]],
                    kelly_short[order_idx[0]],
                    kelly_short[order_idx[1]],
                    kelly_short[order_idx[4]],
                ],
                dtype=np.float64,
            )
            y_s_adj = ir.fit_transform(x, y_s)
            kelly_short[order_idx[2]] = (1.0 - alpha_iso) * y_s[0] + alpha_iso * y_s_adj[0]
            kelly_short[order_idx[3]] = (1.0 - alpha_iso) * y_s[1] + alpha_iso * y_s_adj[1]
            kelly_short[order_idx[0]] = (1.0 - alpha_iso) * y_s[2] + alpha_iso * y_s_adj[2]
            kelly_short[order_idx[1]] = (1.0 - alpha_iso) * y_s[3] + alpha_iso * y_s_adj[3]
            kelly_short[order_idx[4]] = (1.0 - alpha_iso) * y_s[4] + alpha_iso * y_s_adj[4]
    except Exception as e:
        _logger.warning("Isotonic Kelly adjustment failed: %s", e)

    return kelly_long, kelly_short


def _hmm_modulator_kelly_values(
    market_probs: pd.DataFrame,
    market_hmm_feats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """[Advanced HMM Control] Asymmetric Kelly Modulation and Entropy-Based Confidence Shrinkage."""
    cols = _sorted_hmm_prob_columns(market_probs)
    n = len(market_probs)
    
    # Check for 5-state HMM structure
    expected_cols = [
        "hmm_prob_bull_calm",
        "hmm_prob_bull_vol_up",
        "hmm_prob_bear_trend",
        "hmm_prob_chop",
        "hmm_prob_crisis"
    ]
    has_all_cols = all(c in market_probs.columns for c in expected_cols)
    
    if not has_all_cols or not cols:
        _logger.warning("Missing required HMM probability columns for advanced modulation.")
        return pd.DataFrame({
            "hmm_modulator_long": np.ones(n, dtype=np.float64),
            "hmm_modulator_short": np.ones(n, dtype=np.float64),
            "hmm_modulator_base_long": np.ones(n, dtype=np.float64),
            "hmm_modulator_base_short": np.ones(n, dtype=np.float64),
            "btc_trend_vol_adj_24h": np.zeros(n, dtype=np.float64),
        })

    # Posterior probability matrix (N, 5)
    p_mat = market_probs[expected_cols].to_numpy(dtype=np.float64)
    
    # Component 1: Asymmetric Kelly Modulation
    # M_long = 1.0*Bull_Calm + 1.0*Bull_Vol_Up + 0.9*Chop + 0.5*Bear + 0.0*Crisis
    # M_short = 0.0*Bull_Calm + 0.1*Bull_Vol_Up + 0.5*Chop + 1.2*Bear + 0.0*Crisis
    w_long = np.array([1.0, 1.0, 0.5, 0.9, 0.0], dtype=np.float64)
    w_short = np.array([0.0, 0.1, 1.2, 0.5, 0.0], dtype=np.float64)
    
    m_long = p_mat @ w_long
    m_short = p_mat @ w_short
    
    # Component 2: Entropy-Based Confidence Shrinkage
    # E = -sum(Pi * log2(Pi + eps))
    # E_norm = E / log2(5)
    # Penalty = 1.0 - 0.1 * (E_norm)^2  # Minimized to prevent over-protection
    eps = 1e-12
    entropy = -np.sum(p_mat * np.log2(p_mat + eps), axis=1)
    e_norm = entropy / np.log2(5.0)
    penalty = 1.0 - 0.1 * np.square(e_norm)
    
    # Final Modulation
    mod_long = np.clip(m_long * penalty, 0.0, 2.5).astype(np.float64)
    mod_short = np.clip(m_short * penalty, 0.0, 2.5).astype(np.float64)

    # BTC trend for diagnostic report only
    trend_24h: np.ndarray = np.zeros(n, dtype=np.float64)
    if market_hmm_feats is not None and "btc_trend_vol_adj_24h" in market_hmm_feats.columns:
        feat_tmp = market_hmm_feats[["btc_trend_vol_adj_24h"]].copy()
        feat_tmp.index = pd.to_datetime(feat_tmp.index, utc=True)
        feat_tmp = feat_tmp.reset_index()
        idx_col = str(feat_tmp.columns[0])
        if idx_col != "datetime":
            feat_tmp = feat_tmp.rename(columns={idx_col: "datetime"})
        feat_tmp["datetime"] = pd.to_datetime(feat_tmp["datetime"], utc=True)
        dt_df = market_probs[["datetime"]].copy()
        dt_df["datetime"] = pd.to_datetime(dt_df["datetime"], utc=True)
        merged_rec = dt_df.merge(feat_tmp, on="datetime", how="left")
        trend_24h = merged_rec["btc_trend_vol_adj_24h"].fillna(0.0).to_numpy(dtype=np.float64)

    return pd.DataFrame({
        "hmm_modulator_long": mod_long,
        "hmm_modulator_short": mod_short,
        "hmm_modulator_base_long": m_long,
        "hmm_modulator_base_short": m_short,
        "btc_trend_vol_adj_24h": trend_24h,
    })


def _hmm_modulator_kelly_per_symbol(
    market_probs: pd.DataFrame,
    alpha_panel: pd.DataFrame,
    is_end_utc: pd.Timestamp,
    prefetched_1h: dict[str, pd.DataFrame],
    universe_syms: list[str],
    shrink: float,
    crisis_thr: float,
    market_hmm_feats: pd.DataFrame | None,
    anchor_symbol: str | None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """One Kelly modulator per symbol using that symbol's 1h OHLC fwd returns; fallback = anchor (e.g. BTC)."""
    from src.domain.futures.ml_pipeline.regime.per_symbol_overlay import (
        apply_symbol_overlay,
        compute_symbol_overlay,
    )

    cfg_dict: dict[str, Any] = cfg if cfg is not None else dict(OPT_FUTURES_CONFIG)
    overlay_enabled: bool = bool(cfg_dict.get("FUTURES_PER_SYMBOL_OVERLAY_ENABLED", True))

    anchor_df = prefetched_1h.get(anchor_symbol) if anchor_symbol else None
    fallback = _hmm_modulator_kelly_values(
        market_probs,
        market_hmm_feats,
    )
    out: dict[str, pd.DataFrame] = {}

    dt_grid: pd.Series = market_probs["datetime"]

    def _calc_one(sym: str) -> tuple[str, pd.DataFrame]:
        df = prefetched_1h.get(sym)
        # Base modulator is identical for all symbols (market_probs-level signal).
        # Per-symbol differentiation comes from the beta/idio overlay below.
        val = fallback

        # Per-symbol beta scaling + idiosyncratic overlay
        if overlay_enabled and df is not None and len(df) >= 4 and "close" in df.columns:
            try:
                is_anc = sym == anchor_symbol
                ov = compute_symbol_overlay(
                    sym_1h=df,
                    anchor_1h=anchor_df if anchor_df is not None else df,
                    dt_grid=dt_grid,
                    cfg=cfg_dict,
                    is_anchor=is_anc,
                )
                val = apply_symbol_overlay(val, ov, cfg_dict)
            except Exception as exc:
                _logger.warning(
                    "per_symbol_overlay failed for %s (fallback to base mod): %s", sym, exc
                )

        return sym, val

    # [Institutional Quant] Parallel execution for symbol-specific modulators
    with ThreadPoolExecutor(max_workers=min(len(universe_syms), 8)) as executor:
        results = list(executor.map(_calc_one, universe_syms))

    for s, v in results:
        out[s] = v

    return out


def _try_tbm_labels_per_1h_row(
    sym: str,
    df_1h: pd.DataFrame,
    fetch_start: str,
    end: str,
    df_1m: pd.DataFrame | None = None,
    collector: DataCollector | None = None,
) -> np.ndarray | None:
    """Triple-barrier labels aligned to each 1h row."""
    try:
        need = {"open", "high", "low", "close"}
        if not need.issubset(set(df_1h.columns)):
            return None
        if df_1m is None:
            if collector is None:
                return None
            df_1m = collector.collect_1m_ohlcv(sym, fetch_start, end)
        if df_1m is None or len(df_1m) < 200:
            return None

        # [REFACTORED] Use 1h directly for TBM labeling (No 4h resampling)
        d1 = df_1h[[*list(need), "datetime"]].sort_values("datetime").copy()
        d1["datetime"] = pd.to_datetime(d1["datetime"], utc=True)

        if len(d1) < 30:
            return None

        tbm = label_triple_barrier(
            d1,
            df_1m,
            time_stop_bars=int(OPT_FUTURES_CONFIG.get("FUTURES_TBM_TIME_STOP_BARS", 1440)),
            vol_scale_window=int(OPT_FUTURES_CONFIG.get("FUTURES_TBM_VOL_SCALE_WINDOW", 24)),
        )
        if tbm is None or len(tbm) == 0:
            return None

        lab = tbm.rename("tbm_label").reset_index()
        lab["datetime"] = pd.to_datetime(lab["datetime"], utc=True)
        lab = lab.sort_values("datetime")

        tmp = df_1h[["datetime"]].copy()
        tmp["_ord"] = np.arange(len(tmp), dtype=np.int64)
        tmp["datetime"] = pd.to_datetime(tmp["datetime"], utc=True)
        tmp = tmp.sort_values("datetime")

        # Align labels back to 1h rows
        merged = pd.merge_asof(tmp, lab, on="datetime", direction="backward")
        merged = merged.sort_values("_ord")
        return cast(np.ndarray, merged["tbm_label"].to_numpy(dtype=np.float64))
    except Exception:
        return None


def _meta_probs_wf_refit(
    X_w: pd.DataFrame,
    y_ser: pd.Series,
    aligned_tf: pd.DataFrame,
    meta_feats: tuple[str, ...],
    hmm_m_long: np.ndarray,
    hmm_m_short: np.ndarray,
    is_end_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 3: expanding-window MetaLabeler refit on each WF-OOS segment (no softmax)."""
    cfg = OPT_FUTURES_CONFIG
    n_rows = len(X_w)
    feats = list(meta_feats)
    vb = int(cfg.get("FUTURES_META_VERTICAL_BARRIER_BARS", 24))
    mi = int(cfg.get("FUTURES_META_MIN_POS_ISOTONIC", 200))
    wf_on = bool(cfg.get("FUTURES_ML_WF_REFIT_ENABLED", True))
    n_wf = max(1, int(cfg.get("FUTURES_ML_WF_REFIT_LEGS", 3)))

    pl_out: np.ndarray = np.zeros(n_rows, dtype=np.float64)
    ps_out: np.ndarray = np.zeros(n_rows, dtype=np.float64)

    if not wf_on or n_wf <= 1 or is_end_idx >= n_rows - 1:
        meta = MetaLabeler(vertical_barrier_bars=vb, min_pos_isotonic=mi)
        meta.fit(X_w, y_ser, is_end_idx)
        X_a = aligned_tf[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pl, ps = meta.predict_proba_calibrated(X_a)
        rl = np.clip(pl.astype(np.float64) * hmm_m_long, 0.0, 1.0)
        rs = np.clip(ps.astype(np.float64) * hmm_m_short, 0.0, 1.0)
        return rl, rs

    meta = MetaLabeler(vertical_barrier_bars=vb, min_pos_isotonic=mi)
    meta.fit(X_w.iloc[:is_end_idx], y_ser.iloc[:is_end_idx], is_end_idx=is_end_idx)
    X_is = aligned_tf.iloc[:is_end_idx][feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pl_i, ps_i = meta.predict_proba_calibrated(X_is)
    pl_out[:is_end_idx] = np.clip(pl_i.astype(np.float64) * hmm_m_long[:is_end_idx], 0.0, 1.0)
    ps_out[:is_end_idx] = np.clip(ps_i.astype(np.float64) * hmm_m_short[:is_end_idx], 0.0, 1.0)

    n_seg = min(n_wf, max(2, n_rows - is_end_idx))
    edges = np.linspace(is_end_idx, n_rows, n_seg + 1)
    edges_i = np.unique(np.clip(np.round(edges).astype(np.int64), 0, n_rows))

    for k in range(len(edges_i) - 1):
        t0, t1 = int(edges_i[k]), int(edges_i[k + 1])
        if t1 <= t0 or t0 < is_end_idx:
            continue
        train_end = t0
        if train_end < 80:
            continue
        meta.fit(X_w.iloc[:train_end], y_ser.iloc[:train_end], is_end_idx=train_end)
        X_seg = aligned_tf.iloc[t0:t1][feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pl_s, ps_s = meta.predict_proba_calibrated(X_seg)
        pl_out[t0:t1] = np.clip(pl_s.astype(np.float64) * hmm_m_long[t0:t1], 0.0, 1.0)
        ps_out[t0:t1] = np.clip(ps_s.astype(np.float64) * hmm_m_short[t0:t1], 0.0, 1.0)

    return pl_out, ps_out


def _platt_calib_probs_wf(
    aligned_tf: pd.DataFrame,
    gp_base: np.ndarray,
    gp_long: np.ndarray,
    gp_short: np.ndarray,
    is_end_utc: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Platt on IS (alpha vs forward return), then predict p(win) for long/short scores.

    Matches ``precompute_ml_optimization_context`` / ``SignalCalibrator`` usage so WF OOS
    backtests agree with Phase-D precompute when MetaLabeler is off or fails.
    """
    horizon = int(OPT_FUTURES_CONFIG.get("FUTURES_ML_ALPHA_TARGET_HORIZON", 12))
    min_is = 80
    _close_col = "close" if "close" in aligned_tf.columns else "close_x"
    close = aligned_tf[_close_col].astype(np.float64)
    fwd_ret = close.pct_change(horizon).shift(-horizon).to_numpy(dtype=np.float64)
    dt = pd.to_datetime(aligned_tf["datetime"], utc=True)
    is_mask = (dt < is_end_utc).to_numpy()
    valid = is_mask & np.isfinite(gp_base) & np.isfinite(fwd_ret)

    calib = SignalCalibrator()
    if int(valid.sum()) >= min_is:
        y_bin = (fwd_ret[valid] > 0.0001).astype(int)
        if len(np.unique(y_bin)) > 1:
            calib.fit(gp_base[valid], fwd_ret[valid])

    p_long = np.clip(calib.predict_prob(gp_long.astype(np.float64, copy=False)), 0.0, 1.0)
    p_short = np.clip(calib.predict_prob(gp_short.astype(np.float64, copy=False)), 0.0, 1.0)
    return p_long.astype(np.float64), p_short.astype(np.float64)


def _apply_ml_calib_probs(
    aligned_tf: pd.DataFrame,
    wide_1h: pd.DataFrame,
    collector: DataCollector,
    sym: str,
    fetch_start: str,
    end: str,
    is_end_utc: pd.Timestamp,
    use_meta: bool,
    df_1m_prefetch: pd.DataFrame | None = None,
) -> None:
    """Cross-sectional scores (gp x HMM); ml_calib_* via MetaLabeler or Platt (IS WF)."""
    hmm_m_long = (
        aligned_tf["hmm_modulator_long"].to_numpy(dtype=np.float64)
        if "hmm_modulator_long" in aligned_tf.columns
        else np.ones(len(aligned_tf), dtype=np.float64)
    )
    hmm_m_short = (
        aligned_tf["hmm_modulator_short"].to_numpy(dtype=np.float64)
        if "hmm_modulator_short" in aligned_tf.columns
        else np.ones(len(aligned_tf), dtype=np.float64)
    )
    gp_base = (
        aligned_tf["alpha_long_00"].to_numpy(dtype=np.float64)
        if "alpha_long_00" in aligned_tf.columns
        else np.zeros(len(aligned_tf), dtype=np.float64)
    )
    gp_long = (
        aligned_tf["alpha_long"].to_numpy(dtype=np.float64)
        if "alpha_long" in aligned_tf.columns
        else gp_base
    )
    gp_short = (
        aligned_tf["alpha_short"].to_numpy(dtype=np.float64)
        if "alpha_short" in aligned_tf.columns
        else (1.0 - gp_base)
    )
    # [Improvement 1] Friction-Aware EV Hurdle
    hurdle_ratio = float(OPT_FUTURES_CONFIG.get("FUTURES_ML_EV_HURDLE_RATIO", 0.0))
    if hurdle_ratio > 0:
        from config.settings import SLIPPAGE_RATE, TRADING_FEE_RATE
        # Round-trip cost (Taker fee + Slippage) * 2
        rt_cost = (TRADING_FEE_RATE + SLIPPAGE_RATE) * 2.0
        # Heuristic: 0.1% expected return corresponds to ~0.02 deviation from 0.5 in rank space.
        score_hurdle = hurdle_ratio * rt_cost * 10.0
        
        # [NEW] Component 3: Regime-Aware Dynamic EV Hurdle
        # Hurdle_Dyn = BaseHurdle * (1.0 + 2.0 * P(Chop) + 5.0 * P(Crisis))
        p_chop = aligned_tf["hmm_prob_chop"].to_numpy(dtype=np.float64) if "hmm_prob_chop" in aligned_tf.columns else 0.0
        p_crisis = aligned_tf["hmm_prob_crisis"].to_numpy(dtype=np.float64) if "hmm_prob_crisis" in aligned_tf.columns else 0.0
        hurdle_dyn = score_hurdle * (1.0 + 2.0 * p_chop + 5.0 * p_crisis)
        
        # Zero out signals that don't clear the hurdle
        gp_long_mask = (gp_long > (0.5 + hurdle_dyn)).astype(np.float64)
        gp_short_mask = (gp_short > (0.5 + hurdle_dyn)).astype(np.float64)
        
        aligned_tf["xs_score_long"] = gp_long * hmm_m_long * gp_long_mask
        aligned_tf["xs_score_short"] = gp_short * hmm_m_short * gp_short_mask
    else:
        aligned_tf["xs_score_long"] = gp_long * hmm_m_long
        aligned_tf["xs_score_short"] = gp_short * hmm_m_short

    meta_on = bool(use_meta) and bool(OPT_FUTURES_CONFIG.get("FUTURES_USE_META_LABELER", False))

    y_tbm = (
        _try_tbm_labels_per_1h_row(
            sym, wide_1h, fetch_start, end, df_1m=df_1m_prefetch, collector=collector
        )
        if meta_on
        else None
    )
    meta_feats = tuple(c for c in _meta_feature_column_names(wide_1h) if c in wide_1h.columns)
    can_meta = (
        meta_on
        and y_tbm is not None
        and len(y_tbm) == len(wide_1h)
        and len(meta_feats) >= 2
        and len(_sorted_hmm_prob_columns(wide_1h)) > 0
        and all(c in aligned_tf.columns for c in meta_feats)
    )
    n = len(aligned_tf)
    pl = np.ones(n, dtype=np.float64)
    ps = np.ones(n, dtype=np.float64)
    used_meta = False

    if can_meta:
        X_w = wide_1h[list(meta_feats)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_ser = pd.Series(y_tbm, index=wide_1h.index)
        wdt = pd.to_datetime(wide_1h["datetime"], utc=True)
        is_end_idx = int((wdt < is_end_utc).sum())
        if is_end_idx >= 80:
            try:
                pl, ps = _meta_probs_wf_refit(
                    X_w,
                    y_ser,
                    aligned_tf,
                    meta_feats,
                    hmm_m_long,
                    hmm_m_short,
                    is_end_idx,
                )
                used_meta = True
            except Exception as exc:
                _logger.debug("MetaLabeler WF refit skipped: %s", exc)

    if not used_meta:
        pl, ps = _platt_calib_probs_wf(aligned_tf, gp_base, gp_long, gp_short, is_end_utc)

    aligned_tf["ml_calib_prob_long"] = pl
    aligned_tf["ml_calib_prob_short"] = ps
    aligned_tf["ml_calib_prob"] = np.maximum(pl, ps)


class _Step4FusionOutcome(NamedTuple):
    sym: str
    aligned_tf: pd.DataFrame | None
    cp_long: pd.Series | None
    cp_short: pd.Series | None
    error: str | None


def _step4_fusion_one_symbol(
    sym: str,
    tf: str,
    data_maps: dict[str, dict[str, Any]],
    prefetched_1h: dict[str, pd.DataFrame],
    alpha_by_sym: dict[str, pd.DataFrame],
    valid_alpha_set: set[str],
    market_probs: pd.DataFrame,
    hmm_modulator: pd.DataFrame,
    fetch_start: str,
    end: str,
    is_end_utc: pd.Timestamp,
    df_1m_prefetch: pd.DataFrame | None,
    collector: DataCollector,
) -> _Step4FusionOutcome:
    """Per-symbol merge + asof alignment + MetaLabeler (Phase 3: raw x modulator, WF refit)."""
    try:
        _logger.debug("[%s] Step 4 Fusion started.", sym)
        df_1h = prefetched_1h[sym].copy()
        if "datetime" not in df_1h.columns and df_1h.index.name == "datetime":
            df_1h = df_1h.reset_index()
        if "datetime" not in df_1h.columns:
            return _Step4FusionOutcome(sym, None, None, None, f"Missing datetime in df_1h for {sym}")
        df_1h["datetime"] = pd.to_datetime(df_1h["datetime"], utc=True).dt.floor("1s")

        # [Optimization #11] Unify column drop logic
        df_1h = _drop_ml_columns(df_1h)
        
        hmm_cols_ref = _sorted_hmm_prob_columns(market_probs)
        k_fb = len(hmm_cols_ref) if hmm_cols_ref else len(HMM_SEMANTIC_PROB_COLUMNS)

        if sym not in valid_alpha_set:
            wide_1h = df_1h.copy()
            wide_1h["alpha_long_00"] = 0.5
            # [REFACTORED] Merge asymmetric modulators
            wide_1h = pd.merge(wide_1h, hmm_modulator, on="datetime", how="left")
            # [Institutional Quant] Dual-TF Persistence: ffill 4h modulators into 1h bars
            mod_cols = [c for c in hmm_modulator.columns if c != "datetime"]
            wide_1h[mod_cols] = wide_1h[mod_cols].ffill().fillna(1.0)
            
            wide_1h["slot_rank_score"] = 0.0

            # [NEW] Merge all HMM related columns
            hmm_cols_all = [c for c in market_probs.columns if str(c).startswith("hmm_")]
            if hmm_cols_all:
                mp_h = market_probs[["datetime", *hmm_cols_all]]
                wide_1h = wide_1h.merge(mp_h, on="datetime", how="left")
                # [Institutional Quant] Dual-TF Persistence: ffill 4h HMM states into 1h bars
                wide_1h[hmm_cols_all] = wide_1h[hmm_cols_all].ffill()
                
                # Fill probabilities with uniform, others with 0
                prob_cols = [c for c in hmm_cols_all if "prob_" in c]
                other_hmm_cols = [c for c in hmm_cols_all if "prob_" not in c]
                wide_1h[prob_cols] = wide_1h[prob_cols].fillna(1.0 / float(k_fb))
                wide_1h[other_hmm_cols] = wide_1h[other_hmm_cols].fillna(0.0)
            else:
                for c in HMM_SEMANTIC_PROB_COLUMNS:
                    wide_1h[c] = 1.0 / float(len(HMM_SEMANTIC_PROB_COLUMNS))
        else:  # sym in valid_alpha_set: use trained alpha
            sym_alpha = alpha_by_sym[sym].copy()
            if "datetime" not in sym_alpha.columns and sym_alpha.index.name == "datetime":
                sym_alpha = sym_alpha.reset_index()
            if "datetime" in sym_alpha.columns:
                sym_alpha["datetime"] = pd.to_datetime(sym_alpha["datetime"], utc=True).dt.floor("1s")

            if "alpha_long_00" not in sym_alpha.columns:
                _logger.warning(
                    "[%s] alpha_long_00 missing in sym_alpha. cols=%s",
                    sym, list(sym_alpha.columns)
                )
                sym_alpha["alpha_long_00"] = 0.5

            wide_1h = pd.merge(df_1h, sym_alpha, on="datetime", how="left")
            # [Institutional Quant] Dual-TF Alpha persistence: 
            # If sym_alpha is 4h and df_1h is 1h, ffill to maintain the thesis across bars.
            wide_1h["alpha_long_00"] = wide_1h["alpha_long_00"].ffill().fillna(0.5)

            # [REFACTORED] Merge asymmetric modulators
            wide_1h = pd.merge(wide_1h, hmm_modulator, on="datetime", how="left")
            # [Institutional Quant] Dual-TF Persistence: ffill 4h modulators into 1h bars
            mod_cols = [c for c in hmm_modulator.columns if c != "datetime"]
            wide_1h[mod_cols] = wide_1h[mod_cols].ffill().fillna(1.0)

            # Use mean of long/short modulator for ranking score
            # m_avg = (wide_1h["hmm_modulator_long"] + wide_1h["hmm_modulator_short"]) / 2.0
            wide_1h["slot_rank_score"] = wide_1h["alpha_long_00"]

            # [NEW] Merge all HMM related columns
            hmm_cols_all = [c for c in market_probs.columns if str(c).startswith("hmm_")]
            if hmm_cols_all:
                mp_h = market_probs[["datetime", *hmm_cols_all]]
                wide_1h = wide_1h.merge(mp_h, on="datetime", how="left")
                # [Institutional Quant] Dual-TF Persistence: ffill 4h HMM states into 1h bars
                wide_1h[hmm_cols_all] = wide_1h[hmm_cols_all].ffill()

                # Fill probabilities with uniform, others with 0
                prob_cols = [c for c in hmm_cols_all if "prob_" in c]
                other_hmm_cols = [c for c in hmm_cols_all if "prob_" not in c]
                wide_1h[prob_cols] = wide_1h[prob_cols].fillna(1.0 / float(k_fb))
                wide_1h[other_hmm_cols] = wide_1h[other_hmm_cols].fillna(0.0)
            else:
                for c in HMM_SEMANTIC_PROB_COLUMNS:
                    wide_1h[c] = 1.0 / float(len(HMM_SEMANTIC_PROB_COLUMNS))

        df_tf_full = data_maps[sym][tf].copy()
        if "datetime" not in df_tf_full.columns and df_tf_full.index.name == "datetime":
            df_tf_full = df_tf_full.reset_index()
        if "datetime" not in df_tf_full.columns:
            return _Step4FusionOutcome(sym, None, None, None, f"Missing datetime in df_tf_full for {sym}")
        df_tf_full["datetime"] = pd.to_datetime(df_tf_full["datetime"], utc=True).dt.floor("1s")

        # [Optimization #11] Unify column drop logic
        df_tf_full = _drop_ml_columns(df_tf_full)

        # Drop OHLCV columns from wide_1h to prevent _x/_y suffix collision when TF != 1h.
        # We keep the trading-TF OHLCV from df_tf_full; 1h OHLCV is not needed post-merge.
        _ohlcv_base = {"open", "high", "low", "close", "volume"}
        wide_1h_for_merge = wide_1h.drop(
            columns=[c for c in _ohlcv_base if c in wide_1h.columns and c in df_tf_full.columns],
        )

        # [v15.2 Fix] Final asof alignment to trading TF with floor normalization
        wide_1h_for_merge["datetime"] = pd.to_datetime(wide_1h_for_merge["datetime"], utc=True).dt.floor("1s")
        aligned_tf = pd.merge(df_tf_full, wide_1h_for_merge, on="datetime", how="left").fillna(0.0)

        _apply_ml_calib_probs(
            aligned_tf,
            wide_1h,
            collector,
            sym,
            fetch_start,
            end,
            is_end_utc,
            use_meta=(sym in valid_alpha_set),
            df_1m_prefetch=df_1m_prefetch,
        )

        cp_long = aligned_tf.set_index("datetime")["ml_calib_prob_long"]
        cp_short = aligned_tf.set_index("datetime")["ml_calib_prob_short"]
        return _Step4FusionOutcome(sym, aligned_tf, cp_long, cp_short, None)
    except Exception as e:
        return _Step4FusionOutcome(sym, None, None, None, str(e))


def _compute_per_symbol_metrics(
    hmm_modulator_by_sym: dict[str, pd.DataFrame],
    market_probs: pd.DataFrame,
) -> dict[str, float | None]:
    """Compute per-symbol beta/idio overlay metrics for HMM audit report.

    Metrics:
        1. β-adj Protected Exposure (β-adj Prot. Exp.):
               universe mean of (1 - mean_mod_long_i)
               target: 30% ~ 50%
        2. β-Protection Monotonicity:
               Spearman corr(mean_beta_i, 1 - mean_mod_long_i) in risk-off bars.
               target: > 0  (high-β symbols protected more in risk-off)
        3. Idio Crash Capture: N/A at HMM-only stage (no forward returns).

    Args:
        hmm_modulator_by_sym: dict mapping symbol → mod_df (output of
            _hmm_modulator_kelly_per_symbol, with 'hmm_modulator_long' and
            optionally 'beta' columns after apply_symbol_overlay).
        market_probs: market-level HMM posterior DataFrame with 'datetime'
            and hmm_prob_* columns.

    Returns:
        Dict with keys:
            per_sym_beta_adj_protected_exp  (float, %)
            per_sym_beta_monotonicity_corr  (float)
            per_sym_idio_crash_capture      (None — N/A at HMM-only stage)

    Note:
        Time complexity: O(S * N) where S = universe size, N = time bars.

    """
    result: dict[str, float | None] = {
        "per_sym_beta_adj_protected_exp": 0.0,
        "per_sym_beta_monotonicity_corr": 0.0,
        "per_sym_idio_crash_capture": None,
    }

    if not hmm_modulator_by_sym:
        return result

    # --- Metric 1: β-adj Protected Exposure ----------------------------------
    # For each symbol i: protection_i = 1 - mean(mod_long_i)
    # Universe mean of protection_i expressed as %.
    prot_list: list[float] = []
    for mod_df in hmm_modulator_by_sym.values():
        if "hmm_modulator_long" not in mod_df.columns:
            continue
        mean_mod: float = float(
            np.nan_to_num(mod_df["hmm_modulator_long"].to_numpy(dtype=np.float64), nan=1.0).mean()
        )
        prot_list.append(1.0 - mean_mod)

    if prot_list:
        result["per_sym_beta_adj_protected_exp"] = float(np.mean(prot_list) * 100.0)

    # --- Metric 2: β-Protection Monotonicity ---------------------------------
    # Use risk-off bars: dominant regime ∈ {hmm_prob_bear_trend, hmm_prob_crisis}.
    risk_off_cols = {"hmm_prob_bear_trend", "hmm_prob_crisis"}
    prob_cols = [c for c in market_probs.columns if c in risk_off_cols]

    # Build risk-off mask (row-wise max of risk-off regime probs)
    if prob_cols and len(market_probs) > 0:
        risk_off_mask: np.ndarray = (
            market_probs[prob_cols].max(axis=1).to_numpy(dtype=np.float64) > 0.4
        )
    else:
        risk_off_mask = np.ones(len(market_probs), dtype=bool)

    mean_beta_per_sym: list[float] = []
    mean_prot_per_sym: list[float] = []

    for mod_df in hmm_modulator_by_sym.values():
        if "hmm_modulator_long" not in mod_df.columns:
            continue
        mask = risk_off_mask[: len(mod_df)]
        if mask.sum() < 5:
            # Insufficient risk-off bars → skip this metric
            mean_beta_per_sym = []
            break

        mod_long_arr = np.nan_to_num(
            mod_df["hmm_modulator_long"].to_numpy(dtype=np.float64)[: len(mask)], nan=1.0
        )
        prot_ro = float((1.0 - mod_long_arr[mask]).mean())
        mean_prot_per_sym.append(prot_ro)

        if "beta" in mod_df.columns:
            beta_arr = np.nan_to_num(
                mod_df["beta"].to_numpy(dtype=np.float64)[: len(mask)], nan=1.0
            )
            mean_beta_per_sym.append(float(beta_arr[mask].mean()))
        else:
            mean_beta_per_sym.append(1.0)

    if len(mean_beta_per_sym) >= 3:
        # Spearman rank correlation (vectorised via scipy-free rank trick)
        b_arr = np.array(mean_beta_per_sym, dtype=np.float64)
        p_arr = np.array(mean_prot_per_sym, dtype=np.float64)
        # Rank transform
        def _rank(x: np.ndarray) -> np.ndarray:
            order = x.argsort()
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(len(x), dtype=np.float64)
            return ranks

        b_rank = _rank(b_arr)
        p_rank = _rank(p_arr)
        b_c = b_rank - b_rank.mean()
        p_c = p_rank - p_rank.mean()
        denom = float(np.sqrt((b_c**2).sum() * (p_c**2).sum()) + 1e-12)
        spearman_corr = float(np.dot(b_c, p_c) / denom)
        result["per_sym_beta_monotonicity_corr"] = spearman_corr

    # Metric 3: N/A at HMM-only stage
    result["per_sym_idio_crash_capture"] = None
    return result


def _print_hmm_summary(
    market_probs: pd.DataFrame,
    market_hmm_feats: pd.DataFrame,
    hmm_modulator: pd.DataFrame,
    btc_1h: pd.DataFrame | None,
    mode_label: str = "",
    traceability: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Print institutional-grade audit of HMM states and return report metrics (Compact V2)."""
    # ... existing code ...
    report: dict[str, Any] = {}
    regime4_cols = [
        "regime_prob_risk_on_calm",
        "regime_prob_risk_on_volatile",
        "regime_prob_risk_off_trend",
        "regime_prob_chop_liquidity_thin",
    ]
    cols = (
        regime4_cols
        if all(c in market_probs.columns for c in regime4_cols)
        else [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in market_probs.columns]
    )
    if not (cols and len(market_probs) == len(market_hmm_feats)):
        _logger.warning(" [WARN] HMM data alignment failed for audit.")
        _logger.info(" ────────────────────────────────────────────────────────────────────────────")
        return report

    def _series_from_col(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        if col not in frame.columns:
            return pd.Series(default, index=frame.index, dtype=np.float64)
        obj = frame[col]
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, -1]
        return pd.to_numeric(obj, errors="coerce")

    base_cols = ["datetime", *cols]
    if "hmm_tail_risk_8bar" in market_probs.columns:
        base_cols.append("hmm_tail_risk_8bar")
    if "tail_hazard_8h" in market_probs.columns:
        base_cols.append("tail_hazard_8h")
    if "flat_gate" in market_probs.columns:
        base_cols.append("flat_gate")
    if "soft_damp_gate" in market_probs.columns:
        base_cols.append("soft_damp_gate")
    if "hard_damp_gate" in market_probs.columns:
        base_cols.append("hard_damp_gate")
    if "near_flat_gate" in market_probs.columns:
        base_cols.append("near_flat_gate")
    if "gross_cap_mult" in market_probs.columns:
        base_cols.append("gross_cap_mult")
    if "kelly_mult" in market_probs.columns:
        base_cols.append("kelly_mult")
    if "long_mult" in market_probs.columns:
        base_cols.append("long_mult")
    for _sup_col in ("sup_score_q10_h8", "sup_score_q05_h8", "sup_score_q03_h16", "sup_score_soft", "sup_score_hard", "sup_score_near_flat"):
        if _sup_col in market_probs.columns:
            base_cols.append(_sup_col)
    if "gross_cap_mult" in market_probs.columns:
        base_cols.append("gross_cap_mult")
    if "kelly_mult" in market_probs.columns:
        base_cols.append("kelly_mult")
    if "long_mult" in market_probs.columns:
        base_cols.append("long_mult")
    df_eval = market_probs[base_cols].copy()
    df_eval["dominant_state"] = df_eval[cols].idxmax(axis=1)
    pre_crisis_mean = float(
        pd.to_numeric(
            market_probs.get("pre_crisis_hazard", market_probs.get("hmm_prob_pre_crisis", 0.0)),
            errors="coerce",
        ).fillna(0.0).mean()
        * 100.0
    )
    realized_crisis_mean = float(
        pd.to_numeric(
            market_probs.get("realized_crisis_hazard", market_probs.get("hmm_prob_realized_crisis", 0.0)),
            errors="coerce",
        ).fillna(0.0).mean()
        * 100.0
    )
    report["hmm_pre_crisis_mean"] = pre_crisis_mean
    report["hmm_realized_crisis_mean"] = realized_crisis_mean

    mod_cols = ["datetime", "hmm_modulator_long", "hmm_modulator_short"]
    mod_tmp = hmm_modulator[[c for c in mod_cols if c in hmm_modulator.columns]]
    if not mod_tmp.empty and "datetime" in mod_tmp.columns:
        df_eval = pd.merge(df_eval, mod_tmp, on="datetime", how="left")
    else:
        df_eval["hmm_modulator_long"] = 1.0
        df_eval["hmm_modulator_short"] = 1.0

    feat_tmp = market_hmm_feats.copy().reset_index()
    target_feats = ["macro_vol_24h", "macro_cost_168h"]
    feat_cols_to_merge = ["datetime"] + [f for f in target_feats if f in feat_tmp.columns]
    df_eval = pd.merge(df_eval, feat_tmp[feat_cols_to_merge], on="datetime", how="left")

    if btc_1h is not None and not btc_1h.empty:
        btc_tmp = btc_1h[["datetime", "close"]].copy()
        btc_tmp["datetime"] = pd.to_datetime(btc_tmp["datetime"], utc=True)
        btc_tmp["ret"] = btc_tmp["close"].pct_change().fillna(0.0)
        df_eval = pd.merge(df_eval, btc_tmp[["datetime", "ret"]], on="datetime", how="left")

    # [Compact V2] Emoji & Label Mapping
    regime_display: dict[str, str] = {
        "regime_prob_risk_on_calm": "🐂 BULL-CALM ",
        "regime_prob_risk_on_volatile": "🚀 BULL-VOL  ",
        "regime_prob_risk_off_trend": "🐻 BEAR-TREND",
        "regime_prob_chop_liquidity_thin": "🎢 CHOP-THIN ",
        "hmm_prob_bull_calm": "🐂 BULL-CALM ",
        "hmm_prob_bull_vol_up": "🚀 BULL-VOL  ",
        "hmm_prob_bear_trend": "🐻 BEAR-TREND",
        "hmm_prob_chop": "🎢 CHOP-ZONE ",
        "hmm_prob_crisis": "💀 CRISIS    ",
    }

    def _diag_verdict_for_regime(regime_label: str, g_log: float, m_l: float, m_s: float) -> str:
        if (g_log <= -0.04) or ("RISK_OFF" in regime_label) or (m_l <= 0.15 and m_s <= 0.15):
            return "⚠️ Tail-Risk"
        if (g_log <= -0.01) or (m_l < 0.45 and m_s > 0.85):
            return "Adverse"
        if (g_log >= 0.015) and (m_l >= 0.85) and (m_s <= 0.55):
            return "Favorable"
        return "Neutral"

    # Calculate total market vol for Vol-Scale baseline (BTC ret as proxy)
    total_m_sig = 1.0
    if "ret" in df_eval.columns:
        valid_ret = df_eval["ret"].dropna()
        if not valid_ret.empty:
            std_val = float(valid_ret.std() * 100.0)
            if std_val > 1e-6:
                total_m_sig = std_val
    
    # Fallback: if total_m_sig is still 1.0 (no ret or no std), try to use macro_vol if available
    if total_m_sig <= 1.0 and "macro_vol_24h" in df_eval.columns:
        m_vol = df_eval["macro_vol_24h"].mean()
        if m_vol > 0:
            total_m_sig = float(m_vol)

    for state in cols:
        g_df = df_eval[df_eval["dominant_state"] == state]
        pct = len(g_df) / len(df_eval) * 100
        
        # Standardize key to always start with 'hmm_prob_' for the dashboard
        standard_key = state if state.startswith("hmm_prob_") else state.replace("regime_prob_", "hmm_prob_")
        # Specific mappings for the dashboard's expected 5 regimes
        if "risk_on_calm" in standard_key: standard_key = "hmm_prob_bull_calm"
        elif "risk_on_volatile" in standard_key: standard_key = "hmm_prob_bull_vol_up"
        elif "risk_off_trend" in standard_key: standard_key = "hmm_prob_bear_trend"
        elif "chop_liquidity_thin" in standard_key: standard_key = "hmm_prob_chop"
        
        report[f"{standard_key}_share"] = pct / 100.0
        label = regime_display.get(state, state.replace("hmm_prob_", "").upper()[:12])

        if len(g_df) > 0:
            m_l = float(g_df["hmm_modulator_long"].mean())
            m_s = float(g_df["hmm_modulator_short"].mean())
            mu, sig, g_log = 0.0, 0.0, 0.0
            vol_scale = 1.0
            if "ret" in g_df.columns:
                mu = float(g_df["ret"].mean() * 100.0)
                sig = float(g_df["ret"].std() * 100.0)
                g_log = mu - 0.5 * (sig**2 / 100.0)
                vol_scale = sig / total_m_sig
            elif "macro_trend_24h" in g_df.columns and "macro_vol_24h" in g_df.columns:
                # Fallback to macro features if ret is missing
                mu = float(g_df["macro_trend_24h"].mean())
                sig = float(g_df["macro_vol_24h"].mean())
                g_log = mu - 0.5 * (sig**2 / 100.0)
                vol_scale = sig / total_m_sig if total_m_sig > 0 else 1.0
            
            report[f"{standard_key}_vol_scale"] = vol_scale
            report[f"{standard_key}_g_log"] = g_log / 100.0 # Standardized key for dashboard
            
            vol_icon = "🔴" if vol_scale > 1.2 else ("🟡" if vol_scale > 1.0 else ("🟢" if vol_scale < 0.8 else "⚪"))
            
            label_key = state.replace("hmm_prob_", "").replace("regime_prob_", "").upper()
            verd = _diag_verdict_for_regime(label_key, g_log, m_l, m_s)
            if ("RISK-OFF" in label) or ("CRISIS" in label):
                report["hmm_crisis_g_log"] = g_log
            if ("RISK-ON-C" in label) or ("BULL-CALM" in label):
                report["hmm_bull_g_log"] = g_log
        else:
            report[f"{state}_vol_scale"] = 1.0
            report[f"{state}_g_log"] = 0.0

    _logger.info(" ────────────────────────────────────────────────────────────────────────────")

    # Footer metrics
    tail_capture = 0.0
    if "ret" in df_eval.columns:
        q05 = float(df_eval["ret"].quantile(0.05))
        worst_df = df_eval[df_eval["ret"] <= q05]
        if not worst_df.empty:
            tail_capture = float(
                (
                    worst_df["dominant_state"].isin(
                        ["regime_prob_risk_off_trend", "hmm_prob_bear_trend", "hmm_prob_crisis"]
                    )
                ).mean()
                * 100.0
            )

    lead_lag_tail_capture_8bar = 0.0
    realized_crisis_capture = 0.0
    execution_tail_capture = 0.0
    execution_crisis_cap = 0.0
    step2_tail8_tail_lift = 0.0
    step2_tail8_crisis_lift = 0.0
    execution_damp_tail_capture = 0.0
    execution_damp_crisis_cap = 0.0
    execution_damp_precision = 0.0
    execution_protected_exposure_share = 0.0
    execution_soft_damp_tail_capture = 0.0
    execution_soft_damp_crisis_cap = 0.0
    execution_soft_damp_precision = 0.0
    execution_hard_damp_tail_capture = 0.0
    execution_hard_damp_crisis_cap = 0.0
    execution_hard_damp_precision = 0.0
    execution_near_flat_tail_capture = 0.0
    execution_near_flat_crisis_cap = 0.0
    execution_near_flat_precision = 0.0
    false_flat_cost = 0.0
    crisis_precision = 0.0
    flat_gate_precision = 0.0
    if "ret" in df_eval.columns and not df_eval.empty:
        # 8-bar forward realized tail proxy: worst forward return from t+1..t+8
        fwd_worst_8 = pd.Series(np.inf, index=df_eval.index, dtype=np.float64)
        for k in range(1, 9):
            fwd_worst_8 = np.minimum(fwd_worst_8, df_eval["ret"].shift(-k).to_numpy(dtype=np.float64))
        fwd_worst_8 = pd.Series(fwd_worst_8, index=df_eval.index).replace([np.inf, -np.inf], np.nan)
        q10_fwd = float(fwd_worst_8.quantile(0.10)) if fwd_worst_8.notna().any() else np.nan
        realized_tail_mask = fwd_worst_8 <= q10_fwd if np.isfinite(q10_fwd) else pd.Series(False, index=df_eval.index)
        pred_tail_mask = df_eval["dominant_state"].isin(
            ["regime_prob_risk_off_trend", "hmm_prob_bear_trend", "hmm_prob_crisis"]
        )
        if realized_tail_mask.any():
            lead_lag_tail_capture_8bar = float((pred_tail_mask & realized_tail_mask).sum() / max(1, int(realized_tail_mask.sum())) * 100.0)
        if pred_tail_mask.any() and np.isfinite(q10_fwd):
            crisis_precision = float((fwd_worst_8.loc[pred_tail_mask] <= q10_fwd).mean() * 100.0)

        q05_now = float(df_eval["ret"].quantile(0.05))
        # Regime-side crisis coverage should track realized market stress (not execution hazard gates).
        regime_crisis_now = df_eval["ret"] <= q05_now
        if "realized_crisis_hazard" in market_probs.columns:
            hz = pd.to_numeric(market_probs["realized_crisis_hazard"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            thr = float(np.quantile(hz, 0.90))
            execution_crisis_now = pd.Series(hz >= thr, index=df_eval.index)
        else:
            execution_crisis_now = regime_crisis_now
        if regime_crisis_now.any():
            realized_crisis_capture = float(
                (
                    df_eval.loc[regime_crisis_now, "dominant_state"].isin(
                        ["regime_prob_risk_off_trend", "hmm_prob_crisis"]
                    )
                ).mean()
                * 100.0
            )

        if "flat_gate" in df_eval.columns:
            flat_mask = pd.to_numeric(df_eval["flat_gate"], errors="coerce").fillna(0.0) > 0.5
        else:
            flat_mask = df_eval["dominant_state"].isin(["regime_prob_risk_off_trend", "hmm_prob_crisis"])
        if ("tail_hazard_8h" in df_eval.columns) or ("hmm_tail_risk_8bar" in df_eval.columns):
            tcol_exec = "tail_hazard_8h" if "tail_hazard_8h" in df_eval.columns else "hmm_tail_risk_8bar"
            tail_score_exec = _series_from_col(df_eval, tcol_exec, default=0.0).fillna(0.0).clip(0.0, 1.0)
            strong_tail_q = float(np.clip(OPT_FUTURES_CONFIG.get("FUTURES_POLICY_EXEC_STRONG_TAIL_Q", 0.88), 0.80, 0.995))
            strong_tail_abs_thr = float(np.clip(OPT_FUTURES_CONFIG.get("FUTURES_POLICY_EXEC_STRONG_TAIL_ABS_THR", 0.72), 0.0, 1.0))
            strong_tail_thr = max(float(tail_score_exec.quantile(strong_tail_q)), strong_tail_abs_thr)
            strong_tail_mask = tail_score_exec >= strong_tail_thr
            report["hmm_execution_strong_tail_thr"] = strong_tail_thr
            report["hmm_execution_strong_tail_rate"] = float(strong_tail_mask.mean() * 100.0)
        else:
            strong_tail_mask = pd.Series(False, index=df_eval.index)
        exec_gate_mask = flat_mask | strong_tail_mask
        damp_thr = float(np.clip(OPT_FUTURES_CONFIG.get("FUTURES_POLICY_EXEC_DAMP_ACTIVE_THR", 0.82), 0.0, 1.0))
        damp_valid = (
            ("gross_cap_mult" in df_eval.columns)
            and ("kelly_mult" in df_eval.columns)
            and ("long_mult" in df_eval.columns)
        )
        if damp_valid:
            gross_m = _series_from_col(df_eval, "gross_cap_mult", default=1.0).fillna(1.0).clip(0.0, 2.0)
            kelly_m = _series_from_col(df_eval, "kelly_mult", default=1.0).fillna(1.0).clip(0.0, 2.0)
            long_m = _series_from_col(df_eval, "long_mult", default=1.0).fillna(1.0).clip(0.0, 2.0)
            exposure_mult = np.minimum(np.minimum(gross_m.to_numpy(dtype=np.float64), kelly_m.to_numpy(dtype=np.float64)), long_m.to_numpy(dtype=np.float64))
            protected_share = np.clip(1.0 - exposure_mult, 0.0, 1.0)

            # Use actual gate signals from policy_mapper if available
            if "soft_damp_gate" in df_eval.columns:
                soft_damp_mask = _series_from_col(df_eval, "soft_damp_gate", default=0.0).fillna(0.0) > 0.5
            else:
                soft_damp_mask = pd.Series(exposure_mult <= damp_thr, index=df_eval.index)

            if "hard_damp_gate" in df_eval.columns:
                hard_damp_mask = _series_from_col(df_eval, "hard_damp_gate", default=0.0).fillna(0.0) > 0.5
            else:
                hard_damp_mask = pd.Series(exposure_mult <= 0.50, index=df_eval.index)

            if "near_flat_gate" in df_eval.columns:
                near_flat_mask = _series_from_col(df_eval, "near_flat_gate", default=0.0).fillna(0.0) > 0.5
            else:
                near_flat_mask = pd.Series(exposure_mult <= 0.25, index=df_eval.index)

            damp_mask = soft_damp_mask | hard_damp_mask | near_flat_mask

            execution_protected_exposure_share = float(np.mean(protected_share) * 100.0)
            report["hmm_execution_damp_active_rate"] = float(damp_mask.mean() * 100.0)
            report["hmm_execution_soft_damp_rate"] = float(soft_damp_mask.mean() * 100.0)
            report["hmm_execution_hard_damp_rate"] = float(hard_damp_mask.mean() * 100.0)
            report["hmm_execution_near_flat_rate"] = float(near_flat_mask.mean() * 100.0)
            
            # Diagnostic: capture the mean tiers if they exist in the panel
            for k in ["hmm_execution_tier_soft_thr", "hmm_execution_tier_hard_thr", "hmm_execution_tier_near_flat_thr"]:
                if k in df_eval.columns:
                    report[k] = float(df_eval[k].mean())
            if "soft_damp_gate" in df_eval.columns:
                soft_gate_mask = _series_from_col(df_eval, "soft_damp_gate", default=0.0).fillna(0.0) > 0.5
                if soft_gate_mask.any():
                    report["hmm_execution_soft_gate_avg_exposure"] = float(np.mean(exposure_mult[soft_gate_mask.to_numpy(dtype=bool)]))
            if "hard_damp_gate" in df_eval.columns:
                hard_gate_mask = _series_from_col(df_eval, "hard_damp_gate", default=0.0).fillna(0.0) > 0.5
                if hard_gate_mask.any():
                    report["hmm_execution_hard_gate_avg_exposure"] = float(np.mean(exposure_mult[hard_gate_mask.to_numpy(dtype=bool)]))
            if "near_flat_gate" in df_eval.columns:
                near_flat_gate_mask = _series_from_col(df_eval, "near_flat_gate", default=0.0).fillna(0.0) > 0.5
                _nf_arr = exposure_mult[near_flat_gate_mask.to_numpy(dtype=bool)]
                report["hmm_execution_near_flat_gate_avg_exposure"] = float(np.mean(_nf_arr)) if len(_nf_arr) > 0 else 0.0
            if realized_tail_mask.any():
                execution_damp_tail_capture = float(
                    (damp_mask & realized_tail_mask).sum() / max(1, int(realized_tail_mask.sum())) * 100.0
                )
                execution_soft_damp_tail_capture = float(
                    (soft_damp_mask & realized_tail_mask).sum() / max(1, int(realized_tail_mask.sum())) * 100.0
                )
                execution_hard_damp_tail_capture = float(
                    (hard_damp_mask & realized_tail_mask).sum() / max(1, int(realized_tail_mask.sum())) * 100.0
                )
                execution_near_flat_tail_capture = float(
                    (near_flat_mask & realized_tail_mask).sum() / max(1, int(realized_tail_mask.sum())) * 100.0
                )
            if execution_crisis_now.any():
                execution_damp_crisis_cap = float(
                    (damp_mask & execution_crisis_now).sum() / max(1, int(execution_crisis_now.sum())) * 100.0
                )
                execution_soft_damp_crisis_cap = float(
                    (soft_damp_mask & execution_crisis_now).sum() / max(1, int(execution_crisis_now.sum())) * 100.0
                )
                execution_hard_damp_crisis_cap = float(
                    (hard_damp_mask & execution_crisis_now).sum() / max(1, int(execution_crisis_now.sum())) * 100.0
                )
                execution_near_flat_crisis_cap = float(
                    (near_flat_mask & execution_crisis_now).sum() / max(1, int(execution_crisis_now.sum())) * 100.0
                )
            if damp_mask.any() and np.isfinite(q10_fwd):
                execution_damp_precision = float((fwd_worst_8.loc[damp_mask] <= q10_fwd).mean() * 100.0)
            if soft_damp_mask.any() and np.isfinite(q10_fwd):
                execution_soft_damp_precision = float((fwd_worst_8.loc[soft_damp_mask] <= q10_fwd).mean() * 100.0)
            if hard_damp_mask.any() and np.isfinite(q10_fwd):
                execution_hard_damp_precision = float((fwd_worst_8.loc[hard_damp_mask] <= q10_fwd).mean() * 100.0)
            if near_flat_mask.any() and np.isfinite(q10_fwd):
                execution_near_flat_precision = float((fwd_worst_8.loc[near_flat_mask] <= q10_fwd).mean() * 100.0)
        else:
            damp_mask = pd.Series(False, index=df_eval.index)
        if flat_mask.any():
            pos_ret = df_eval.loc[flat_mask, "ret"]
            # Opportunity cost proxy: positive returns lost while flat in CRISIS state vs total market upside
            total_upside = df_eval.loc[df_eval["ret"] > 0, "ret"].sum()
            false_flat_cost = float(pos_ret[pos_ret > 0.0].sum() / max(float(total_upside), 1e-9) * 100.0)
            if np.isfinite(q10_fwd):
                flat_gate_precision = float((fwd_worst_8.loc[flat_mask] <= q10_fwd).mean() * 100.0)
        if realized_tail_mask.any():
            execution_tail_capture = float(
                (exec_gate_mask & realized_tail_mask).sum() / max(1, int(realized_tail_mask.sum())) * 100.0
            )
            if flat_mask.any():
                flat_tail_capture = float(
                    (flat_mask & realized_tail_mask).sum() / max(1, int(realized_tail_mask.sum())) * 100.0
                )
                step2_tail8_tail_lift = max(0.0, execution_tail_capture - flat_tail_capture)
        if execution_crisis_now.any():
            execution_crisis_cap = float(
                (exec_gate_mask & execution_crisis_now).sum() / max(1, int(execution_crisis_now.sum())) * 100.0
            )
            if flat_mask.any():
                flat_crisis_cap = float(
                    (flat_mask & execution_crisis_now).sum() / max(1, int(execution_crisis_now.sum())) * 100.0
                )
                step2_tail8_crisis_lift = max(0.0, execution_crisis_cap - flat_crisis_cap)
            
        try:
            from scipy.stats import spearmanr
            p_crisis = pd.to_numeric(market_probs.get("hmm_prob_crisis", market_probs.get("regime_prob_risk_off_trend", pd.Series(0, index=df_eval.index))), errors="coerce").fillna(0.0).reindex(df_eval.index)
            regime_ic, _ = spearmanr(p_crisis.to_numpy(), fwd_worst_8.fillna(0.0).to_numpy())
            report["hmm_regime_ic"] = float(regime_ic) if np.isfinite(regime_ic) else 0.0
        except Exception:
            pass
    
    if ("hmm_tail_risk_8bar" in df_eval.columns) or ("tail_hazard_8h" in df_eval.columns):
        tcol = "tail_hazard_8h" if "tail_hazard_8h" in df_eval.columns else "hmm_tail_risk_8bar"
        tail_score = pd.to_numeric(df_eval[tcol], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        report["hmm_tail_overlay_mean"] = float(tail_score.mean() * 100.0)
        report["hmm_tail_overlay_p95"] = float(tail_score.quantile(0.95) * 100.0)
        if "ret" in df_eval.columns and len(df_eval) > 10:
            fwd_worst = pd.Series(np.inf, index=df_eval.index, dtype=np.float64)
            for k in range(1, 9):
                fwd_worst = np.minimum(fwd_worst, df_eval["ret"].shift(-k).to_numpy(dtype=np.float64))
            fwd_worst = pd.Series(fwd_worst, index=df_eval.index).replace([np.inf, -np.inf], np.nan)
            q10 = float(fwd_worst.quantile(0.10)) if fwd_worst.notna().any() else np.nan
            if np.isfinite(q10):
                top_mask = tail_score >= float(tail_score.quantile(0.90))
                realized = fwd_worst <= q10
                if bool(np.any(top_mask)):
                    report["hmm_tail_overlay_top_decile_hit_rate"] = float((realized[top_mask]).mean() * 100.0)
        score_diag_specs = (
            ("sup_score_q10_h8", 8, 0.10, "hmm_sup_q10_h8_top_decile_hit"),
            ("sup_score_q05_h8", 8, 0.05, "hmm_sup_q05_h8_top_decile_hit"),
            ("sup_score_q03_h16", 16, 0.03, "hmm_sup_q03_h16_top_decile_hit"),
        )
        for s_col, horizon, qv, out_key in score_diag_specs:
            if s_col not in df_eval.columns:
                continue
            s_val = _series_from_col(df_eval, s_col, default=0.0).fillna(0.0).clip(0.0, 1.0)
            fwd_w = pd.Series(np.inf, index=df_eval.index, dtype=np.float64)
            for k in range(1, int(max(2, horizon)) + 1):
                fwd_w = np.minimum(fwd_w, df_eval["ret"].shift(-k).to_numpy(dtype=np.float64))
            fwd_w = pd.Series(fwd_w, index=df_eval.index).replace([np.inf, -np.inf], np.nan)
            q_thr = float(fwd_w.quantile(qv)) if fwd_w.notna().any() else np.nan
            if not np.isfinite(q_thr):
                continue
            top_dec = s_val >= float(s_val.quantile(0.90))
            if bool(np.any(top_dec)):
                report[out_key] = float((fwd_w[top_dec] <= q_thr).mean() * 100.0)

    report["hmm_regime_tail_capture"] = tail_capture
    report["hmm_regime_crisis_cap"] = realized_crisis_capture
    report["hmm_execution_tail_capture"] = execution_tail_capture
    report["hmm_execution_crisis_cap"] = execution_crisis_cap
    report["hmm_execution_damp_tail_capture"] = execution_damp_tail_capture
    report["hmm_execution_damp_crisis_cap"] = execution_damp_crisis_cap
    report["hmm_execution_damp_precision"] = execution_damp_precision
    report["hmm_execution_protected_exposure_share"] = execution_protected_exposure_share
    report["hmm_execution_soft_damp_tail_capture"] = execution_soft_damp_tail_capture
    report["hmm_execution_soft_damp_crisis_cap"] = execution_soft_damp_crisis_cap
    report["hmm_execution_soft_damp_precision"] = execution_soft_damp_precision
    report["hmm_execution_hard_damp_tail_capture"] = execution_hard_damp_tail_capture
    report["hmm_execution_hard_damp_crisis_cap"] = execution_hard_damp_crisis_cap
    report["hmm_execution_hard_damp_precision"] = execution_hard_damp_precision
    report["hmm_execution_near_flat_tail_capture"] = execution_near_flat_tail_capture
    report["hmm_execution_near_flat_crisis_cap"] = execution_near_flat_crisis_cap
    report["hmm_execution_near_flat_precision"] = execution_near_flat_precision
    report["hmm_tail_capture"] = tail_capture
    report["hmm_lead_lag_tail_capture_8bar"] = lead_lag_tail_capture_8bar
    report["hmm_realized_crisis_capture"] = realized_crisis_capture
    report["hmm_execution_tail8_tail_lift"] = step2_tail8_tail_lift
    report["hmm_execution_tail8_crisis_lift"] = step2_tail8_crisis_lift
    report["hmm_false_flat_cost"] = false_flat_cost
    report["hmm_crisis_precision"] = crisis_precision
    report["hmm_flat_gate_precision"] = flat_gate_precision
    switches = int((df_eval["dominant_state"] != df_eval["dominant_state"].shift(1)).sum())
    avg_dur = float(len(df_eval) / max(1, switches))
    report["hmm_switches"] = float(switches)
    report["hmm_avg_duration"] = avg_dur

    # Unified Practical Thresholds
    rtc_pass = "PASS" if tail_capture > 40.0 else "FAIL"
    rcc_pass = "PASS" if realized_crisis_capture > 40.0 else "FAIL"
    etc_pass = "PASS" if execution_tail_capture > 40.0 else "FAIL"
    ecc_pass = "PASS" if execution_crisis_cap > 40.0 else "FAIL"
    edtc_pass = "PASS" if execution_damp_tail_capture > 80.0 else "FAIL"
    edcc_pass = "PASS" if execution_damp_crisis_cap > 90.0 else "FAIL"
    edp_pass = "OK" if execution_damp_precision > 0.0 else "LOW"
    cp_pass = "OK" if crisis_precision > 0.0 else "LOW"
    fg_pass = "OK" if flat_gate_precision > 0.0 else "LOW"
    ff_pass = "GOOD" if false_flat_cost < 25.0 else "WARN"
    dur_pass = "PASS" if avg_dur > 18 else "SHORT"

    # _logger.info(" [REGIME QUALITY] - Target: Tail/Crisis >40%%, Precision >0%%")
    # _logger.info(f"  > Regime Tail-Capture : {tail_capture:>5.1f}%% [{rtc_pass}]")
    # _logger.info(f"  > Regime Crisis-Cap   : {realized_crisis_capture:>5.1f}%% [{rcc_pass}]")
    # _logger.info(f"  > Crisis-Prec   : {crisis_precision:>5.1f}%% [{cp_pass}]")
    # _logger.info(f"  > Regime IC     : {report.get('hmm_regime_ic', 0.0):>+6.3f}")
    # _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    # _logger.info(" [EXECUTION QUALITY] - Target: Tail/Crisis >80%%/90%%, Precision >0%%")
    # _logger.info(f"  > Execution Tail-Capture : {execution_tail_capture:>5.1f}%% [{etc_pass}]")
    # _logger.info(f"  > Execution Crisis-Cap   : {execution_crisis_cap:>5.1f}%% [{ecc_pass}]")
    # _logger.info(f"  > Damp Tail-Capture      : {execution_damp_tail_capture:>5.1f}%% [{edtc_pass}]")
    # _logger.info(f"  > Damp Crisis-Cap        : {execution_damp_crisis_cap:>5.1f}%% [{edcc_pass}]")
    # _logger.info(f"  > Damp Precision         : {execution_damp_precision:>5.1f}%% [{edp_pass}]")
    # _logger.info(
    #     "  > SoftDamp T/C/P         : %5.1f%% / %5.1f%% / %5.1f%%",
    #     execution_soft_damp_tail_capture,
    #     execution_soft_damp_crisis_cap,
    #     execution_soft_damp_precision,
    # )
    # _logger.info(
    #     "  > HardDamp T/C/P         : %5.1f%% / %5.1f%% / %5.1f%%",
    #     execution_hard_damp_tail_capture,
    #     execution_hard_damp_crisis_cap,
    #     execution_hard_damp_precision,
    # )
    # _logger.info(
    #     "  > NearFlat T/C/P         : %5.1f%% / %5.1f%% / %5.1f%%",
    #     execution_near_flat_tail_capture,
    #     execution_near_flat_crisis_cap,
    #     execution_near_flat_precision,
    # )
    # _logger.info(
    #     "  > GateExp Soft/Hard/NFlat: %5.3f / %5.3f / %5.3f",
    #     float(report.get("hmm_execution_soft_gate_avg_exposure", np.nan)),
    #     float(report.get("hmm_execution_hard_gate_avg_exposure", np.nan)),
    #     float(report.get("hmm_execution_near_flat_gate_avg_exposure", np.nan)),
    # )
    # _logger.info(f"  > Protected Exposure     : {execution_protected_exposure_share:>5.1f}%%")
    # _logger.info(f"  > FlatGate-Prec : {flat_gate_precision:>5.1f}%% [{fg_pass}]")
    # _logger.info(f"  > Step2 Tail8 Lift (Tail/Crisis): {step2_tail8_tail_lift:>5.1f}%% / {step2_tail8_crisis_lift:>5.1f}%%")
    # _logger.info(
    #     "  > SupHit q10/q05/q03    : %5.1f%% / %5.1f%% / %5.1f%%",
    #     float(report.get("hmm_sup_q10_h8_top_decile_hit", np.nan)),
    #     float(report.get("hmm_sup_q05_h8_top_decile_hit", np.nan)),
    #     float(report.get("hmm_sup_q03_h16_top_decile_hit", np.nan)),
    # )
    # _logger.info(f"  > False-Flat    : {false_flat_cost:>+6.3f}%% [{ff_pass}]")
    # _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    # _logger.info(" [OPERATIONAL STABILITY] - Target: >35 bars")
    # _logger.info(f"  > Avg-Duration  : {avg_dur:>5.1f} bars [{dur_pass}]")
    # _logger.info(f"  > Switches      : {switches}")

    # if (
    #     ("pre_crisis_hazard" in market_probs.columns)
    #     or ("realized_crisis_hazard" in market_probs.columns)
    #     or ("hmm_prob_pre_crisis" in market_probs.columns)
    #     or ("hmm_prob_realized_crisis" in market_probs.columns)
    # ):
    #     _logger.info(
    #         "  > AuxCrisis: pre=%5.1f%% | realized=%5.1f%%",
    #         pre_crisis_mean,
    #         realized_crisis_mean,
    #     )
    # _logger.info(" ────────────────────────────────────────────────────────────────────────────\n")

    return report


def _attach_tail_overlay_if_enabled(
    market_probs: pd.DataFrame,
    market_hmm_feats: pd.DataFrame,
    market_returns: pd.Series,
    is_end_idx_market: int,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not bool(cfg.get("FUTURES_HMM_TAIL_OVERLAY_ENABLED", True)):
        return market_probs, {}
    if "hmm_tail_risk_8bar" in market_probs.columns:
        return market_probs, {}

    out_probs = market_probs.copy()
    try:
        ov = fit_predict_tail_overlay(
            market_probs=out_probs,
            market_hmm_feats=market_hmm_feats,
            market_returns=market_returns,
            is_end_idx=is_end_idx_market,
            cfg=cfg,
        )
        out_probs["hmm_tail_risk_8bar"] = ov.risk.reindex(out_probs["datetime"]).fillna(0.0).to_numpy(dtype=np.float64)
        rep = dict(ov.report)
        rep["hmm_tail_overlay_method"] = 1.0 if ov.method == "logistic+isotonic" else (0.5 if ov.method == "logistic" else 0.0)
        return out_probs, rep
    except Exception as e:
        _logger.warning("Tail overlay failed; fallback neutral: %s", e)
        out_probs["hmm_tail_risk_8bar"] = np.clip(
            pd.to_numeric(out_probs.get("hmm_prob_crisis", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
            0.0,
            1.0,
        )
        return out_probs, {"hmm_tail_overlay_method": -1.0}



def _build_panel_with_targets(
    data_maps: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    *,
    skip_targets: bool = False,
) -> pd.DataFrame:
    utils = CrossSectionalPipelineUtils()
    panel_df = utils.build_panel_df(data_maps, tf="1h")
    
    if skip_targets:
        # For HMM inference, we only need systemic features (market breadth, dispersion)
        # We can completely skip the expensive cross-sectional Z-scoring and imputation
        panel_df = utils.add_systemic_features(panel_df)
        return panel_df

    panel_df = utils.add_cross_sectional_features(panel_df)
    
    # [Neutralization] Apply Cross-Sectional Z-Score (Mean/Std) to relevant features
    # to remove market beta and ensure relative strength evaluation.
    cs_neutral_cols = [
        "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", 
        "realized_vol_yz_24", "vol_ratio_24", "vol_ratio_168",
        "mom_proxy_12", "acceleration_24", "beta_neutral_momentum"
    ]
    panel_df = utils.apply_cs_zscore(panel_df, cs_neutral_cols)
    
    panel_df = utils.add_systemic_features(panel_df)
    impute_cols = [c for c in ALPHA_ENGINEERED_FEATURE_NAMES if c in panel_df.columns]
    if impute_cols:
        panel_df = utils.cs_median_impute_panel(panel_df, impute_cols)

    raw_h = cfg.get("FUTURES_ML_ALPHA_HORIZONS", (3, 6, 12, 24))
    default_h = (3, 6, 12, 24)
    h_src = raw_h if isinstance(raw_h, (list, tuple)) else default_h
    horizons = tuple(int(x) for x in h_src)
    _ic_hl = float(OPT_FUTURES_CONFIG.get("FUTURES_ML_IC_HALF_LIFE", 2.3))
    _h_weights = tuple(float(np.exp(-h / _ic_hl)) for h in horizons)
    panel_df["target"] = utils.create_multi_horizon_rank_targets(panel_df, horizons=horizons, weights=_h_weights)
    
    # [v15.2 Fix] Critical: Floor datetime index to 1s to ensure perfect join with original_df
    if "datetime" in panel_df.index.names:
        new_dt = pd.to_datetime(panel_df.index.get_level_values("datetime"), utc=True).floor("1s")
        panel_df.index = panel_df.index.set_levels(new_dt, level="datetime")
        
    return panel_df


def _collect_stage_integrity_from_maps(
    maps: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sym, smap in maps.items():
        recs = smap.get("integrity_audit")
        if isinstance(recs, list):
            for rec in recs:
                if isinstance(rec, dict):
                    rows.append(dict(rec))
        cov = smap.get("feature_group_coverage")
        if isinstance(cov, dict):
            for g, metrics in cov.items():
                if not isinstance(metrics, dict):
                    continue
                rows.append(
                    {
                        "symbol": sym,
                        "timeframe": "n/a",
                        "stage": "feature_group",
                        "feature_group": str(g),
                        **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                    }
                )
    return rows


def _build_integrity_summary(
    data_maps: dict[str, dict[str, Any]],
    panel_df: pd.DataFrame | None,
    tf: str,
    *,
    panel_fillna_cols: list[str] | None = None,
) -> dict[str, Any]:
    stage_rows = _collect_stage_integrity_from_maps(data_maps)
    panel_summary = summarize_dataframe_integrity(panel_df if isinstance(panel_df, pd.DataFrame) else pd.DataFrame(), timeframe=tf)
    summary: dict[str, Any] = {"panel": panel_summary, "panel_pre_fillna_nan_pct": 0.0}
    if isinstance(panel_df, pd.DataFrame) and panel_fillna_cols:
        cols = [c for c in panel_fillna_cols if c in panel_df.columns]
        if cols:
            pre_nan = panel_df[cols].isna().sum().sum()
            summary["panel_pre_fillna_nan_pct"] = float(pre_nan / max(int(len(panel_df) * len(cols)), 1))
    if stage_rows:
        stage_df = pd.DataFrame(stage_rows)
        if {"stage", "timeframe", "nan_pct", "inf_count", "zero_ratio"}.issubset(stage_df.columns):
            agg = (
                stage_df.groupby(["stage", "timeframe"], dropna=False)[
                    ["nan_pct", "inf_count", "zero_ratio", "duplicate_dt", "gap_count", "nonpositive_price_count"]
                ]
                .mean(numeric_only=True)
                .reset_index()
            )
            summary["stages"] = agg.to_dict(orient="records")
        fg_df = stage_df[stage_df.get("stage", "") == "feature_group"] if "stage" in stage_df.columns else pd.DataFrame()
        if not fg_df.empty:
            fg_agg = (
                fg_df.groupby("feature_group", dropna=False)[["col_count", "non_null_coverage", "non_zero_coverage"]]
                .mean(numeric_only=True)
                .reset_index()
            )
            summary["feature_group_coverage"] = fg_agg.to_dict(orient="records")
    return summary


def merge_ml_output_into_data_maps(
    ml_out: MLPipelineOutput,
    maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    *,
    log_tag: str = "",
) -> None:
    """Merge ML fusion columns from ml_out into each symbol's tf DataFrame in maps (in-place)."""
    # [Optimization #12] Parallelize per-symbol merge
    def _merge_one(sym: str) -> None:
        if sym not in maps or tf not in maps[sym]:
            return
            
        original_df = maps[sym][tf].copy()
        original_df = _drop_ml_columns(original_df)
        
        if sym not in ml_out.meta_feature_frame_by_symbol:
            _logger.debug("[%s] ML output missing. Filling with neutral 0.5.", sym)
            original_df["alpha_long_00"] = 0.5
            maps[sym][tf] = original_df
            return
            
        mff = ml_out.meta_feature_frame_by_symbol[sym].copy()
        mff["datetime"] = pd.to_datetime(mff["datetime"], utc=True)
        
        hmm_cols_in_mff = [c for c in mff.columns if str(c).startswith("hmm_")]
        ml_cols = [
            "datetime", "alpha_long_00", "alpha_long", "alpha_short",
            "btc_trend_vol_adj_24h", "hmm_modulator_long", "hmm_modulator_short",
            "slot_rank_score", "ml_calib_prob", "ml_calib_prob_long", "ml_calib_prob_short",
            "xs_score_long", "xs_score_short", *hmm_cols_in_mff,
        ]
        for c in _META_EXTRA_FEATS:
            if c not in ml_cols: ml_cols.append(c)

        unique_ml_cols = []
        seen = set()
        for x in ml_cols:
            if x in mff.columns and x not in seen:
                unique_ml_cols.append(x)
                seen.add(x)
        
        if sym not in maps or tf not in maps[sym]:
            return
            
        ml_features = mff[unique_ml_cols].copy()
        if "datetime" not in ml_features.columns and ml_features.index.name == "datetime":
            ml_features = ml_features.reset_index()
        if "datetime" in ml_features.columns:
            ml_features["datetime"] = pd.to_datetime(ml_features["datetime"], utc=True).dt.floor("1s")

        original_df = maps[sym][tf].copy()
        if "datetime" not in original_df.columns and original_df.index.name == "datetime":
            original_df = original_df.reset_index()
        if "datetime" in original_df.columns:
            original_df["datetime"] = pd.to_datetime(original_df["datetime"], utc=True).dt.floor("1s")

        # Optimization #11
        original_df = _drop_ml_columns(original_df)
        
        # Optimization #12: Use sort=True inside merge instead of separate sort_values
        if "datetime" in original_df.columns and "datetime" in ml_features.columns:
            merged = pd.merge(original_df, ml_features, on="datetime", how="left", sort=True)
        else:
            _logger.warning("[%s] Merge skipped due to missing datetime column.", sym)
            merged = original_df

        # Forward fill for Dual-TF
        ml_non_dt_cols = [c for c in unique_ml_cols if c != "datetime" and c in merged.columns]
        if ml_non_dt_cols:
            merged[ml_non_dt_cols] = merged[ml_non_dt_cols].ffill()

        if "alpha_long_00" not in merged.columns:
            merged["alpha_long_00"] = 0.5
        else:
            merged["alpha_long_00"] = merged["alpha_long_00"].fillna(0.5)
            
        # [Institutional Quant] Prevent total flattening of signal by fillna(0.0)
        # We fill probabilities with uniform, modulators with 1.0, others with 0
        p_cols = [c for c in merged.columns if "prob_" in c]
        mod_cols = [c for c in merged.columns if "modulator" in c]
        k_fb_local = len(p_cols) if p_cols else 5
        merged[p_cols] = merged[p_cols].fillna(1.0 / float(k_fb_local))
        merged[mod_cols] = merged[mod_cols].fillna(1.0)
        
        # [Diagnostic] Final check if alpha is alive
        if "alpha_long_00" in merged.columns and merged["alpha_long_00"].std() < 1e-9:
            _logger.debug("[%s] Signal still dead after merge. Rows=%d", sym, len(merged))

        maps[sym][tf] = merged.fillna(0.0)

    workers = min(len(symbols), 12)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(_merge_one, symbols))


def merge_ml_output_into_is_and_oos(
    ml_out: MLPipelineOutput,
    data_maps: dict[str, dict[str, Any]],
    oos_data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> None:
    merge_ml_output_into_data_maps(ml_out, data_maps, symbols, tf, log_tag=" IS")
    merge_ml_output_into_data_maps(ml_out, oos_data_maps, symbols, tf, log_tag=" OOS")


def copy_data_maps_tf_clone(
    maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> dict[str, dict[str, Any]]:
    """Shallow clone per symbol dict with a copied tf OHLCV frame (for WF leg isolation)."""
    out: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        if sym not in maps or tf not in maps[sym]:
            continue
        inner = dict(maps[sym])
        inner[tf] = maps[sym][tf].copy(deep=False)
        out[sym] = inner
    return out


def run_hmm_fusion_for_is_end(
    symbols: list[str],
    tf: str,
    fetch_start: str,
    end: str,
    cfg: dict[str, Any],
    data_maps: dict[str, dict[str, Any]],
    prefetched_1h: dict[str, pd.DataFrame],
    panel_df: pd.DataFrame | None,
    alpha_panel: pd.DataFrame,
    is_end_date: str | pd.Timestamp,
    collector: DataCollector,
    workers: int = 4,
    n_jobs: int = 4,
    *,
    include_fusion: bool = True,
    summary_mode_label: str = "",
    prefetch_label_start: str | None = None,
    prefetched_1m: dict[str, pd.DataFrame] | None = None,
    prefetched_market_probs: pd.DataFrame | None = None,
    prefetched_market_hmm_feats: pd.DataFrame | None = None,
) -> MLPipelineOutput:
    """Walk-forward leg anchor: retrain systemic HMM with is_end_date cutoff.

    GP alpha_panel frozen.
    When include_fusion=False (hmm_only preview), skips per-symbol fusion.
    """
    from src.domain.futures.ml_pipeline.features.engineering import build_systemic_hmm_features

    is_end_dt = pd.to_datetime(is_end_date)
    is_end_utc = is_end_dt.tz_localize("UTC") if is_end_dt.tzinfo is None else is_end_dt.tz_convert("UTC")
    tail_rep: dict[str, float] = {}
    hmm_inferrer: Any | None = None
    hmm_cache_state = "miss"

    if prefetched_market_probs is not None and prefetched_market_hmm_feats is not None:
        _logger.debug("♻️  HMM cache hit")
        market_probs = prefetched_market_probs
        market_hmm_feats = prefetched_market_hmm_feats
        hmm_cache_state = "hit"
    else:
        if panel_df is None:
            if prefetched_1h:
                tmp_maps = {sym: {"1h": df} for sym, df in prefetched_1h.items()}
                panel_df = _build_panel_with_targets(tmp_maps, cfg, skip_targets=True)
            else:
                panel_df = _build_panel_with_targets(data_maps, cfg, skip_targets=True)
        
        _logger.info("🧠 HMM Inference | is_end=%s%s", is_end_date, summary_mode_label)
        market_hmm_feats = build_systemic_hmm_features(panel_df, None, tf="1h")
        if market_hmm_feats.index.tz is None:
            market_hmm_feats.index = market_hmm_feats.index.tz_localize("UTC")
        else:
            market_hmm_feats.index = market_hmm_feats.index.tz_convert("UTC")

        hmm_k = int(cfg.get("FUTURES_HMM_K_STATES", 5))
        # [Optimization #6] Reduce n_iter and add convergence tolerance
        hmm_n_iter = int(cfg.get("FUTURES_HMM_N_ITER", 200))
        hmm_inferrer = build_hmm_inferrer_from_config(
            cfg,
            n_states=hmm_k,
            n_iter=hmm_n_iter,
            tol=1e-4,
        )
        
        is_end_idx_market = int((market_hmm_feats.index < is_end_utc).sum())
        _btc_anchor = next((s for s in symbols if "BTC" in s), None)
        _btc_df = prefetched_1h.get(_btc_anchor) if _btc_anchor else None
        if _btc_df is not None and "close" in _btc_df.columns:
            _btc_rets = _btc_df.set_index("datetime")["close"].pct_change().reindex(market_hmm_feats.index).fillna(0.0)
        else:
            _btc_rets = market_hmm_feats["macro_trend_168h"]

        market_probs = _run_systemic_hmm_with_causal_split(
            hmm_inferrer=hmm_inferrer,
            market_hmm_feats=market_hmm_feats,
            market_returns=_btc_rets,
            is_end_idx_market=is_end_idx_market,
            tf=tf,
            symbol="Market",
        )
        market_probs, tail_rep = _attach_tail_overlay_if_enabled(
            market_probs=market_probs,
            market_hmm_feats=market_hmm_feats,
            market_returns=_btc_rets,
            is_end_idx_market=is_end_idx_market,
            cfg=cfg,
        )
    if "hmm_tail_risk_8bar" not in market_probs.columns:
        is_end_idx_market = int((pd.to_datetime(market_hmm_feats.index, utc=True) < is_end_utc).sum())
        _btc_anchor = next((s for s in symbols if "BTC" in s), None)
        _btc_df = prefetched_1h.get(_btc_anchor) if _btc_anchor else None
        if _btc_df is not None and "close" in _btc_df.columns:
            _btc_rets = _btc_df.set_index("datetime")["close"].pct_change().reindex(market_hmm_feats.index).fillna(0.0)
        else:
            _btc_rets = market_hmm_feats["macro_trend_168h"]
        market_probs, tail_rep = _attach_tail_overlay_if_enabled(
            market_probs=market_probs,
            market_hmm_feats=market_hmm_feats,
            market_returns=_btc_rets,
            is_end_idx_market=is_end_idx_market,
            cfg=cfg,
        )

    btc_anchor = next((s for s in symbols if "BTC" in s), None)
    btc_1h = prefetched_1h.get(btc_anchor) if btc_anchor else None

    hmm_modulator_by_sym = _hmm_modulator_kelly_per_symbol(
        market_probs,
        alpha_panel,
        is_end_utc,
        prefetched_1h,
        symbols,
        float(cfg.get("FUTURES_HMM_KELLY_SHRINKAGE", 0.4)),
        float(cfg.get("FUTURES_HMM_CRISIS_THRESHOLD", 0.7)),
        market_hmm_feats,
        btc_anchor,
        cfg=dict(cfg),
    )
    dt_series = market_probs["datetime"]
    for _mod_df in hmm_modulator_by_sym.values():
        _mod_df["datetime"] = dt_series

    hmm_modulator_audit = (
        hmm_modulator_by_sym[btc_anchor]
        if btc_anchor and btc_anchor in hmm_modulator_by_sym
        else (next(iter(hmm_modulator_by_sym.values())) if hmm_modulator_by_sym else pd.DataFrame())
    )

    h_rep = _print_hmm_summary(
        market_probs,
        market_hmm_feats,
        hmm_modulator_audit,
        btc_1h,
        mode_label=summary_mode_label,
        traceability=_build_hmm_traceability(
            cfg=cfg,
            tf=tf,
            is_end_date=is_end_date,
            symbol_count=len(symbols),
            seed=None,
            explicit_run_id=None,
            backend=_resolve_hmm_backend_name(hmm_inferrer, cfg),
            cache_state=hmm_cache_state,
        ),
    )
    if tail_rep:
        h_rep.update({k: float(v) for k, v in tail_rep.items()})
    h_rep["hmm_backend"] = _resolve_hmm_backend_name(hmm_inferrer, cfg)

    # Step 5: Per-symbol beta/idio overlay metrics
    per_sym_metrics = _compute_per_symbol_metrics(hmm_modulator_by_sym, market_probs)
    h_rep["per_sym_beta_adj_protected_exp"] = float(
        per_sym_metrics.get("per_sym_beta_adj_protected_exp") or 0.0
    )
    h_rep["per_sym_beta_monotonicity_corr"] = float(
        per_sym_metrics.get("per_sym_beta_monotonicity_corr") or 0.0
    )
    h_rep["per_sym_idio_crash_capture"] = None  # type: ignore[assignment]  # N/A at HMM-only stage

    out = MLPipelineOutput()
    out.hmm_report = h_rep
    out.alpha_panel = alpha_panel
    out.market_probs = market_probs
    if not include_fusion:
        return out

    label_start = prefetch_label_start or fetch_start
    syms_step4 = [s for s in symbols if s in data_maps]
    _logger.debug("🔀 Fusing %d symbols", len(syms_step4))
    valid_alpha_symbols = alpha_panel.index.get_level_values("symbol").unique()
    valid_alpha_set = set(valid_alpha_symbols.tolist())
    need_1m = [s for s in syms_step4 if s in valid_alpha_set]

    def _prefetch_1m(s: str) -> tuple[str, pd.DataFrame | None]:
        try:
            d = collector.collect_1m_ohlcv(s, label_start, end)
            return (s, d if d is not None and len(d) >= 200 else None)
        except Exception:
            return (s, None)

    prefetch_workers = max(1, min(len(need_1m) or 1, 4))
    one_m_cache: dict[str, pd.DataFrame | None] = {}
    
    # [Fix] Use passed cfg instead of global OPT_FUTURES_CONFIG
    meta_on_any = bool(cfg.get("FUTURES_USE_META_LABELER", False))

    if prefetched_1m is not None:
        # Use existing memory cache if provided (Smart bypass)
        one_m_cache = prefetched_1m
    elif need_1m and meta_on_any:
        with ThreadPoolExecutor(max_workers=prefetch_workers) as ex:
            one_m_cache = {s: dm for s, dm in ex.map(_prefetch_1m, need_1m)}

    def _fusion_job(s: str) -> _Step4FusionOutcome:
        mod_df = hmm_modulator_by_sym.get(s, hmm_modulator_audit)
        return _step4_fusion_one_symbol(
            s,
            tf,
            data_maps,
            prefetched_1h,
            alpha_by_sym,
            valid_alpha_set,
            market_probs,
            mod_df,
            fetch_start,
            end,
            is_end_utc,
            one_m_cache.get(s),
            collector,
        )

    alpha_by_sym: dict[str, pd.DataFrame] = {
        s: alpha_panel.xs(s, level="symbol").reset_index()
        for s in syms_step4
        if s in valid_alpha_set
    }

    fusion_workers = max(1, min(len(syms_step4) or 1, max(workers, n_jobs), 12))
    if fusion_workers > 1:
        with ThreadPoolExecutor(max_workers=fusion_workers) as ex:
            fusion_results = list(ex.map(_fusion_job, syms_step4))
    else:
        fusion_results = [_fusion_job(s) for s in syms_step4]

    for res in fusion_results:
        if res.error is not None:
            _logger.error("[%s] R-6 fusion failed: %s", res.sym, res.error)
            continue
        if res.aligned_tf is None or res.cp_long is None or res.cp_short is None:
            _logger.error("[%s] R-6 fusion returned empty.", res.sym)
            continue
        out.meta_feature_frame_by_symbol[res.sym] = res.aligned_tf
        out.calib_prob_long_by_symbol[res.sym] = res.cp_long
        out.calib_prob_short_by_symbol[res.sym] = res.cp_short
        out.calib_prob_by_symbol[res.sym] = res.cp_long

    return out


def _get_cfg_hash(cfg: dict[str, Any]) -> str:
    """Create a stable hash of the configuration dictionary."""
    relevant_cfg = {
        k: v for k, v in cfg.items() 
        if k.startswith("FUTURES_") or k in ("total_trials", "tpe_n_startup_trials")
    }
    cfg_str = json.dumps(relevant_cfg, sort_keys=True, default=str)
    return hashlib.md5(cfg_str.encode()).hexdigest()


@_memory.cache(ignore=["fetch_start_date", "end", "cfg", "workers", "n_jobs", "preloaded_data_maps", "preloaded_1h_maps"])
def _run_ml_pipeline_cached_core(
    tf: str,
    is_end_date: str | None,
    is_start_date: str | None,
    symbols_tuple: tuple[str, ...],
    seed: int,
    cfg_hash: str,
    fetch_start_date: str,
    end: str,
    cfg: dict[str, Any],
    workers: int,
    n_jobs: int,
    gp_only: bool,
    hmm_only: bool,
    preloaded_data_maps: dict[str, dict[str, Any]] | None,
    preloaded_1h_maps: dict[str, pd.DataFrame] | None,
) -> MLPipelineOutput:
    """Core ML pipeline logic that is cached on disk.
    
    The hashing is based only on the first 6 arguments.
    """
    _logger.debug(
        "🚀 [ML CACHE] Cache MISS - Running core pipeline for %s symbols (seed=%s)",
        len(symbols_tuple),
        seed,
    )
    return _run_ml_pipeline_implementation(
        list(symbols_tuple), tf, fetch_start_date, end, cfg, workers, n_jobs,
        is_end_date, is_start_date, gp_only, hmm_only,
        preloaded_data_maps, preloaded_1h_maps, seed=seed
    )


def run_ml_pipeline_for_universe(
    symbols: list[str],
    tf: str,
    fetch_start_date: str,
    end: str,
    cfg: dict[str, Any],
    workers: int = 4,
    n_jobs: int = 4,
    is_end_date: str | None = None,
    is_start_date: str | None = None,
    gp_only: bool = False,
    hmm_only: bool = False,
    preloaded_data_maps: dict[str, dict[str, Any]] | None = None,
    preloaded_1h_maps: dict[str, pd.DataFrame] | None = None,
    seed: int | None = None,
) -> MLPipelineOutput:
    """[Phase 2] Universal Cross-Sectional ML Pipeline with Disk Caching."""
    symbols_tuple = tuple(sorted(symbols))
    # Use provided seed or fallback to first seed in config
    actual_seed = seed if seed is not None else int(cfg.get("FUTURES_LEARNING_SEEDS", [42])[0])
    cfg_hash = _get_cfg_hash(cfg)
    _logger.debug("ML pipeline request | tf=%s symbols=%d seed=%s", tf, len(symbols), actual_seed)
    
    force_retrain = bool(cfg.get("FUTURES_ML_FORCE_RETRAIN_ALPHA", False))
    if force_retrain:
        _logger.debug("🔥 [ML CACHE] Force retrain enabled - bypassing cache.")
        return _run_ml_pipeline_implementation(
            symbols, tf, fetch_start_date, end, cfg, workers, n_jobs,
            is_end_date, is_start_date, gp_only, hmm_only,
            preloaded_data_maps, preloaded_1h_maps, seed=actual_seed
        )

    try:
        return _run_ml_pipeline_cached_core(
            tf, is_end_date, is_start_date, symbols_tuple, actual_seed, cfg_hash,
            fetch_start_date, end, cfg, workers, n_jobs, gp_only, hmm_only,
            preloaded_data_maps, preloaded_1h_maps
        )
    except Exception as e:
        _logger.warning("ML Pipeline Caching failed or bypassed (seed=%s): %s", actual_seed, e)
        return _run_ml_pipeline_implementation(
            symbols, tf, fetch_start_date, end, cfg, workers, n_jobs, 
            is_end_date, is_start_date, gp_only, hmm_only,
            preloaded_data_maps, preloaded_1h_maps, seed=actual_seed
        )


def _run_ml_pipeline_implementation(
    symbols: list[str],
    tf: str,
    fetch_start_date: str,
    end: str,
    cfg: dict[str, Any],
    workers: int = 4,
    n_jobs: int = 4,
    is_end_date: str | None = None,
    is_start_date: str | None = None,
    gp_only: bool = False,
    hmm_only: bool = False,
    preloaded_data_maps: dict[str, dict[str, Any]] | None = None,
    preloaded_1h_maps: dict[str, pd.DataFrame] | None = None,
    seed: int | None = None,
) -> MLPipelineOutput:
    """[Phase 2] Universal Cross-Sectional ML Pipeline implementation."""
    _logger.debug("ML pipeline start | tf=%s symbols=%d seed=%s", tf, len(symbols), seed)

    collector = DataCollector()
    data_maps: dict[str, dict[str, Any]] = preloaded_data_maps or {}
    prefetched_1h: dict[str, pd.DataFrame] = preloaded_1h_maps or {}

    # --- Step 1: Market-Wide Data Collection & Enrichment ---
    _logger.info("  ● STEP 1/4 : Panel & Discovery      [WORKING] symbols=%d", len(symbols))
    
    for sym in symbols:
        # Check if enrichment is needed for target TF
        needs_enrich = False
        if sym in data_maps and tf in data_maps[sym]:
            df_check = data_maps[sym][tf]
            if "ret_1" not in df_check.columns:
                needs_enrich = True
        else:
            needs_enrich = True

        # Check if enrichment is needed for 1h reference
        if tf != "1h" and sym not in prefetched_1h:
            needs_enrich = True
        elif tf != "1h" and sym in prefetched_1h:
            if "ret_1" not in prefetched_1h[sym].columns:
                needs_enrich = True

        if needs_enrich:
            try:
                # Fetch if missing from maps
                if sym not in data_maps or tf not in data_maps[sym]:
                    df_tf = collector.collect_and_save(sym, tf, fetch_start_date, end)
                    if df_tf is not None:
                        df_tf = merge_funding_into_ohlcv(sym, df_tf, Path(FUTURES_DATA_DIR))
                        df_tf = merge_metrics_into_ohlcv(sym, df_tf, Path(FUTURES_DATA_DIR))
                        data_maps.setdefault(sym, {})[tf] = df_tf
                
                if tf != "1h" and sym not in prefetched_1h:
                    df_1h = collector.collect_and_save(sym, "1h", fetch_start_date, end)
                    if df_1h is not None:
                        df_1h = merge_funding_into_ohlcv(sym, df_1h, Path(FUTURES_DATA_DIR))
                        df_1h = merge_metrics_into_ohlcv(sym, df_1h, Path(FUTURES_DATA_DIR))
                        prefetched_1h[sym] = df_1h

                # Enrich if not in hmm_only mode
                if not hmm_only:
                    if sym in data_maps and tf in data_maps[sym]:
                        if "ret_1" not in data_maps[sym][tf].columns:
                            data_maps[sym][tf] = _enrich_with_gp_features(data_maps[sym][tf], tf=tf)
                    
                    if tf != "1h" and sym in prefetched_1h:
                        if "ret_1" not in prefetched_1h[sym].columns:
                            prefetched_1h[sym] = _enrich_with_gp_features(prefetched_1h[sym], tf="1h")
                    elif tf == "1h" and sym in data_maps and "1h" in data_maps[sym]:
                        prefetched_1h[sym] = data_maps[sym]["1h"]

            except Exception as e:
                _logger.debug("[%s] Data fetch/enrich failed: %s", sym, e)
        else:
            # Already enriched, just ensure prefetched_1h is synced for 1h mode
            if tf == "1h" and sym not in prefetched_1h:
                prefetched_1h[sym] = data_maps[sym]["1h"]
    
    _logger.info("  ● STEP 1/4 : Panel & Discovery      [██████████] ✅ DONE")

    # --- Step 2: Systemic HMM Inference (Regime Discovery) ---
    from src.domain.futures.ml_pipeline.features.engineering import build_systemic_hmm_features

    # [Optimization #5] Build panel once and reuse for systemic features
    h_maps = {s: {"1h": prefetched_1h[s]} for s in prefetched_1h}
    h_utils = CrossSectionalPipelineUtils()
    h_panel = h_utils.build_panel_df(h_maps, tf="1h")

    # Check if systemic features already exist (from GP enrichment)
    systemic_cols = ["macro_trend_24h", "macro_vol_24h", "cs_dispersion"]
    if not all(c in h_panel.columns for c in systemic_cols):
        h_panel = h_utils.add_systemic_features(h_panel)

    market_hmm_feats = build_systemic_hmm_features(h_panel, None, tf="1h")
    if market_hmm_feats.index.tz is None:
        market_hmm_feats.index = market_hmm_feats.index.tz_localize("UTC")
    else:
        market_hmm_feats.index = market_hmm_feats.index.tz_convert("UTC")

    hmm_k = int(cfg.get("FUTURES_HMM_K_STATES", 5))
    # [Optimization #6] Reduce n_iter and add convergence tolerance
    hmm_n_iter = int(cfg.get("FUTURES_HMM_N_ITER", 200))
    hmm_inferrer = build_hmm_inferrer_from_config(
        cfg,
        n_states=hmm_k,
        n_iter=hmm_n_iter,
        tol=1e-4,
    )
    _logger.info("  ● STEP 2/4 : HMM Regime Inference   [WORKING] Backend: %s", _resolve_hmm_backend_name(hmm_inferrer, cfg))

    is_end_dt = pd.to_datetime(is_end_date or end)
    is_end_utc = is_end_dt.tz_localize("UTC") if is_end_dt.tzinfo is None else is_end_dt.tz_convert("UTC")
    is_end_idx_market = int((market_hmm_feats.index < is_end_utc).sum())

    _btc_anchor = next((s for s in symbols if "BTC" in s), None)
    _btc_df = prefetched_1h.get(_btc_anchor) if _btc_anchor else None
    if _btc_df is not None and "close" in _btc_df.columns:
        _btc_rets = _btc_df.set_index("datetime")["close"].pct_change().reindex(market_hmm_feats.index).fillna(0.0)
    else:
        _btc_rets = market_hmm_feats["macro_trend_168h"]

    _logger.debug("🧭 HMM causal boundary | total=%d | is_end_idx=%d", len(market_hmm_feats), is_end_idx_market)
    _logger.debug("HMM regime inference | Market %s | backend=%s", tf, _resolve_hmm_backend_name(hmm_inferrer, cfg))

    market_probs = _run_systemic_hmm_with_causal_split(
        hmm_inferrer=hmm_inferrer,
        market_hmm_feats=market_hmm_feats,
        market_returns=_btc_rets,
        is_end_idx_market=is_end_idx_market,
        tf=tf,
        symbol="Market",
    )
    market_probs, tail_rep = _attach_tail_overlay_if_enabled(
        market_probs=market_probs,
        market_hmm_feats=market_hmm_feats,
        market_returns=_btc_rets,
        is_end_idx_market=is_end_idx_market,
        cfg=cfg,
    )
    
    _logger.info("  ● STEP 2/4 : HMM Regime Inference   [██████████] ✅ is=%d, oos=%d", is_end_idx_market, len(market_hmm_feats) - is_end_idx_market)

    if hmm_only:
        _logger.info("✅ HMM Inference complete (HMM-only mode)")
        # Compute per-symbol overlay to populate beta/idio metrics in audit report.
        btc_anchor_hmm = next((s for s in symbols if "BTC" in s), None)
        hmm_mod_by_sym = _hmm_modulator_kelly_per_symbol(
            market_probs,
            pd.DataFrame(),  # alpha_panel not needed for HMM-only
            is_end_utc,
            prefetched_1h,
            symbols,
            float(cfg.get("FUTURES_HMM_KELLY_SHRINKAGE", 0.4)),
            float(cfg.get("FUTURES_HMM_CRISIS_THRESHOLD", 0.7)),
            market_hmm_feats,
            btc_anchor_hmm,
            cfg=dict(cfg),
        )
        dt_series_hmm = market_probs["datetime"]
        for _m in hmm_mod_by_sym.values():
            _m["datetime"] = dt_series_hmm
        hmm_mod_audit = (
            hmm_mod_by_sym[btc_anchor_hmm]
            if btc_anchor_hmm and btc_anchor_hmm in hmm_mod_by_sym
            else (next(iter(hmm_mod_by_sym.values())) if hmm_mod_by_sym else pd.DataFrame())
        )
        integrity_summary = _build_integrity_summary(data_maps, None, tf)
        out = MLPipelineOutput(market_probs=market_probs, integrity_report=integrity_summary)
        out.hmm_report = _print_hmm_summary(
            market_probs,
            market_hmm_feats,
            hmm_mod_audit,
            _btc_df,
            "(HMM-ONLY)",
            traceability=_build_hmm_traceability(
                cfg=cfg,
                tf=tf,
                is_end_date=is_end_date or end,
                symbol_count=len(symbols),
                seed=seed,
                explicit_run_id=None,
                backend=_resolve_hmm_backend_name(hmm_inferrer, cfg),
                cache_state="miss",
            ),
        )
        out.hmm_report.update({k: float(v) for k, v in tail_rep.items()})
        out.hmm_report["hmm_backend"] = _resolve_hmm_backend_name(hmm_inferrer, cfg)
        out.hmm_report["integrity_panel_nan_pct"] = float(integrity_summary.get("panel", {}).get("nan_pct", 0.0))
        out.hmm_report["integrity_panel_prefill_nan_pct"] = float(integrity_summary.get("panel_pre_fillna_nan_pct", 0.0))
        # Step 5: per-symbol overlay metrics
        _per_sym = _compute_per_symbol_metrics(hmm_mod_by_sym, market_probs)
        out.hmm_report["per_sym_beta_adj_protected_exp"] = float(
            _per_sym.get("per_sym_beta_adj_protected_exp") or 0.0
        )
        out.hmm_report["per_sym_beta_monotonicity_corr"] = float(
            _per_sym.get("per_sym_beta_monotonicity_corr") or 0.0
        )
        out.hmm_report["per_sym_idio_crash_capture"] = None  # type: ignore[assignment]
        _logger.info(
            "Integrity summary (HMM-only) panel_nan_pct=%.4f stage_rows=%d",
            float(integrity_summary.get("panel", {}).get("nan_pct", 0.0)),
            len(integrity_summary.get("stages", [])),
        )
        return out

    # --- Step 3: Regime-Aware Alpha Mining ---
    _logger.info("  ● STEP 3/4 : Alpha Signal Mining    [WORKING] %s", "HMM-Aware GPU Loop")
    
    # Build final panel on target TF
    panel_df = h_utils.build_panel_df(data_maps, tf=tf)
    panel_df = h_utils.add_cross_sectional_features(panel_df)
    
    # [Optimization #5] Avoid redundant systemic feature recalculation
    if not all(c in panel_df.columns for c in systemic_cols):
        panel_df = h_utils.add_systemic_features(panel_df)
    
    # Inject HMM features into training panel
    _logger.debug("🔗 HMM → Alpha feature injection")
    hmm_cols_all = [c for c in market_probs.columns if str(c).startswith("hmm_")]
    
    # [Fix] Drop overlapping columns before join
    drop_overlap = [c for c in hmm_cols_all if c in panel_df.columns]
    if drop_overlap:
        panel_df = panel_df.drop(columns=drop_overlap)

    mp_feats = market_probs.set_index("datetime")[hmm_cols_all]
    panel_df = panel_df.join(mp_feats, on="datetime", how="left")
    panel_prefill_nan_pct = 0.0
    if hmm_cols_all:
        panel_prefill_nan_pct = float(panel_df[hmm_cols_all].isna().sum().sum() / max(int(len(panel_df) * len(hmm_cols_all)), 1))
    panel_df[hmm_cols_all] = panel_df[hmm_cols_all].ffill().fillna(1.0 / float(hmm_k))

    # Add targets
    raw_h = cfg.get("FUTURES_ML_ALPHA_HORIZONS", (3, 6, 12, 24))
    horizons = tuple(int(x) for x in (raw_h if isinstance(raw_h, (list, tuple)) else (3, 6, 12, 24)))
    _ic_hl = float(OPT_FUTURES_CONFIG.get("FUTURES_ML_IC_HALF_LIFE", 2.3))
    _h_weights = tuple(float(np.exp(-h / _ic_hl)) for h in horizons)
    panel_df["target"] = h_utils.create_multi_horizon_rank_targets(panel_df, horizons=horizons, weights=_h_weights)

    miner = MLAlphaMiner(
        n_jobs=n_jobs, 
        target_horizons=horizons, 
        slots_per_theme=max(3, min(6, int(cfg.get("FUTURES_ML_ALPHA_SLOTS_PER_THEME", 3))))
    )
    filter_options = {
        # IC filter options are config-driven to avoid drift between config and runtime behavior.
        "fdr_q": float(cfg.get("FUTURES_ML_IC_FDR_Q", OPT_FUTURES_CONFIG.get("FUTURES_ML_IC_FDR_Q", 0.10))),
        "symbol_balance_max": float(
            cfg.get(
                "FUTURES_ML_IC_SYMBOL_BALANCE_MAX",
                OPT_FUTURES_CONFIG.get("FUTURES_ML_IC_SYMBOL_BALANCE_MAX", 3.0),
            )
        ),
        "require_regime_gate": bool(
            cfg.get("FUTURES_ML_IC_REGIME_GATE", OPT_FUTURES_CONFIG.get("FUTURES_ML_IC_REGIME_GATE", True))
        ),
        "use_newey_west": bool(
            cfg.get("FUTURES_ML_IC_FILTER_USE_HAC", OPT_FUTURES_CONFIG.get("FUTURES_ML_IC_FILTER_USE_HAC", True))
        ),
        "use_ewma_ic_stat": bool(
            cfg.get("FUTURES_ML_IC_FILTER_USE_EWMA", OPT_FUTURES_CONFIG.get("FUTURES_ML_IC_FILTER_USE_EWMA", False))
        ),
        "ewma_half_life": float(
            cfg.get("FUTURES_ML_IC_EWMA_HALF_LIFE", OPT_FUTURES_CONFIG.get("FUTURES_ML_IC_EWMA_HALF_LIFE", 540.0))
        ),
        # Keep Step3 regime alpha options intact.
        "step3_regime_alpha_enabled": bool(cfg.get("FUTURES_STEP3_REGIME_ALPHA_ENABLED", False)),
        "step3_chop_support_min": float(cfg.get("FUTURES_STEP3_CHOP_SUPPORT_MIN", 0.25)),
        "step3_chop_ic_min": float(cfg.get("FUTURES_STEP3_CHOP_IC_MIN", -0.01)),
        "step3_chop_weight_mult": float(cfg.get("FUTURES_STEP3_CHOP_WEIGHT_MULT", 0.50)),
        "step3_weight_mult_floor": float(cfg.get("FUTURES_STEP3_WEIGHT_MULT_FLOOR", 0.20)),
    }
    alpha_cache_enabled, alpha_cache_max_items = _resolve_alpha_cache_limits(cfg)
    alpha_cache_meta: dict[str, Any] = {"cache_state": "disabled"}
    alpha_cache_key = _build_alpha_cache_key(
        panel_df=panel_df,
        tf=tf,
        is_end_date=is_end_date,
        seed=seed,
        cfg=cfg,
        horizons=horizons,
        slots_per_theme=int(getattr(miner, "slots_per_theme", cfg.get("FUTURES_ML_ALPHA_SLOTS_PER_THEME", 6))),
        filter_options=filter_options,
    )

    alpha_panel: pd.DataFrame | None = None
    if alpha_cache_enabled and alpha_cache_max_items > 0:
        alpha_panel, hit_meta = _alpha_cache_get(alpha_cache_key)
        if alpha_panel is not None and hit_meta is not None:
            alpha_cache_meta = hit_meta

    if alpha_panel is None:
        alpha_panel = miner.mine_alphas_cs(
            panel_df,
            is_end_date=is_end_date,
            filter_options=filter_options,
        )
        if alpha_cache_enabled and alpha_cache_max_items > 0:
            alpha_cache_meta = _alpha_cache_put(
                alpha_cache_key,
                alpha_panel,
                max_items=alpha_cache_max_items,
            )
        else:
            _logger.info("ALPHA_CACHE bypass reason=disabled_or_zero_capacity")

    if "target" in panel_df.columns:
        alpha_panel["target"] = panel_df["target"]
    alpha_panel.attrs["alpha_cache"] = dict(alpha_cache_meta)
    integrity_summary = _build_integrity_summary(data_maps, panel_df, tf, panel_fillna_cols=hmm_cols_all)
    integrity_summary["panel_pre_fillna_nan_pct"] = panel_prefill_nan_pct
    
    _logger.debug(
        "Integrity summary | panel_nan_pct=%.4f panel_prefill_nan_pct=%.4f stage_rows=%d",
        float(integrity_summary.get("panel", {}).get("nan_pct", 0.0)),
        float(integrity_summary.get("panel_pre_fillna_nan_pct", 0.0)),
        len(integrity_summary.get("stages", [])),
    )

    n_surv = int(getattr(alpha_panel, "attrs", {}).get("alpha_component_filter", {}).get("n_surviving", 0))
    _logger.info("  ● STEP 3/4 : Alpha Signal Mining    [██████████] ✅ Survival: %d elite slots", n_surv)

    # --- Step 4: Signal Fusion & Meta-Labeling ---
    if gp_only:
        _logger.debug("⏭️ Step 4/4 | Fusion bypass (ALPHA-only fast path)")
        hmm_mod_audit = _hmm_modulator_kelly_values(market_probs, market_hmm_feats)
        hmm_mod_audit["datetime"] = market_probs["datetime"]
        out = MLPipelineOutput(
            alpha_panel=alpha_panel,
            market_probs=market_probs,
            integrity_report=integrity_summary,
        )
        out.hmm_report = _print_hmm_summary(
            market_probs,
            market_hmm_feats,
            hmm_mod_audit,
            _btc_df,
            "(ALPHA-ONLY)",
            traceability=_build_hmm_traceability(
                cfg=cfg,
                tf=tf,
                is_end_date=is_end_date or end,
                symbol_count=len(symbols),
                seed=seed,
                explicit_run_id=None,
                backend=_resolve_hmm_backend_name(hmm_inferrer, cfg),
                cache_state="miss",
            ),
        )
        out.hmm_report.update({k: float(v) for k, v in tail_rep.items()})
        out.hmm_report["hmm_backend"] = _resolve_hmm_backend_name(hmm_inferrer, cfg)
        out.hmm_report["integrity_panel_nan_pct"] = float(integrity_summary.get("panel", {}).get("nan_pct", 0.0))
        out.hmm_report["integrity_panel_prefill_nan_pct"] = float(integrity_summary.get("panel_pre_fillna_nan_pct", 0.0))
        alpha_non_empty = not out.alpha_panel.empty
        alpha_component_count = (
            int(out.alpha_panel.index.get_level_values("component").nunique())
            if alpha_non_empty and "component" in out.alpha_panel.index.names
            else 0
        )
        _logger.debug(
            "✅ Alpha Mining complete (ALPHA-only mode) | hmm_report_present=%s alpha_panel_non_empty=%s alpha_component_count=%d",
            bool(out.hmm_report),
            alpha_non_empty,
            alpha_component_count,
        )
        _logger.info("  ● STEP 4/4 : Signal Multi-Fusion    [██████████] ✅ Bypass (Alpha-only)")
        _logger.info(" ──────────────────────────────────────────────────────────────")
        _logger.info(" ✅ ML Pipeline orchestration finished successfully.")
        return out

    _logger.info("  ● STEP 4/4 : Signal Multi-Fusion    [WORKING] Meta-Labeling")
    # Use existing HMM results for fusion to avoid redundant training
    out = run_hmm_fusion_for_is_end(
        list(data_maps.keys()), tf, fetch_start_date, end, cfg, data_maps,
        prefetched_1h, panel_df, alpha_panel, is_end_date or end, collector,
        workers, n_jobs, include_fusion=True,
        prefetched_market_probs=market_probs,
        prefetched_market_hmm_feats=market_hmm_feats
    )
    out.integrity_report = integrity_summary
    
    _logger.info("  ● STEP 4/4 : Signal Multi-Fusion    [██████████] ✅ DONE")
    _logger.info(" ──────────────────────────────────────────────────────────────")
    _logger.info(" ✅ ML Pipeline orchestration finished successfully.")
    return out
