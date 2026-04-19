"""Pre-GP market regime labels (HMM on cross-sectional aggregates; no GP alpha leakage)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

_logger = logging.getLogger(__name__)


def build_market_aggregate_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """One row per datetime: systemic / dispersion proxies already on panel."""
    if panel_df.empty:
        return pd.DataFrame()
    px = panel_df.reset_index()
    if "datetime" not in px.columns:
        return pd.DataFrame()
    agg_kw: dict[str, Any] = {}
    if "cs_dispersion" in px.columns:
        agg_kw["cs_dispersion"] = ("cs_dispersion", "first")
    if "market_breadth" in px.columns:
        agg_kw["market_breadth"] = ("market_breadth", "first")
    if "cross_vol_rank" in px.columns:
        agg_kw["cross_vol_rank_m"] = ("cross_vol_rank", "mean")
    if not agg_kw:
        return pd.DataFrame()
    out = px.groupby("datetime", sort=True).agg(**agg_kw)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def infer_pre_gp_regime_ids(
    panel_df: pd.DataFrame,
    *,
    is_end_date: str | None,
    n_states: int = 3,
    min_train_rows: int = 200,
) -> pd.Series:
    """
    Fit GaussianHMM on IS aggregate features; predict discrete state for all datetimes.
    Returns Series indexed by datetime (tz-aware UTC if possible), int labels 0..K-1.
    """
    feat = build_market_aggregate_features(panel_df)
    if feat.shape[0] < min_train_rows or feat.shape[1] < 1:
        return pd.Series(dtype=np.int64)

    X = feat.to_numpy(dtype=np.float64)
    n = len(feat)
    idx_utc = pd.to_datetime(feat.index, utc=True)
    if is_end_date is not None:
        cut = pd.to_datetime(is_end_date, utc=True)
        is_mask = idx_utc < cut
    else:
        is_mask = np.ones(n, dtype=bool)

    is_end_idx = int(np.sum(is_mask))
    if is_end_idx < min_train_rows:
        is_end_idx = min(max(min_train_rows, int(n * 0.7)), n)
    is_end_idx = min(max(is_end_idx, min(min_train_rows // 2, n)), n)

    if is_end_idx < max(50, int(n_states) * 15):
        return pd.Series(dtype=np.int64)

    X_train = X[:is_end_idx]
    row_std = X_train.std(axis=0)
    row_std[row_std < 1e-9] = 1.0
    Xn = (X - X_train.mean(axis=0)) / row_std

    try:
        k = int(min(max(2, n_states), is_end_idx // 20))
        model = GaussianHMM(
            n_components=k,
            covariance_type="diag",
            n_iter=120,
            random_state=42,
        )
        model.fit(Xn[:is_end_idx])
        states = model.predict(Xn)
    except Exception as e:
        _logger.warning("pre-GP regime HMM failed: %s", e)
        return pd.Series(0, index=feat.index, dtype=np.int64)

    return pd.Series(states.astype(np.int64), index=feat.index, name="regime_pre_hmm")


def attach_regime_pre_to_panel(panel_df: pd.DataFrame, regime_ser: pd.Series) -> pd.DataFrame:
    """Map datetime-level regime ids onto (datetime, symbol) MultiIndex."""
    out = panel_df.copy()
    if regime_ser.empty:
        out["regime_pre_hmm"] = np.int64(0)
        return out
    r = regime_ser.copy()
    r.index = pd.to_datetime(r.index, utc=True)
    dt = pd.to_datetime(out.index.get_level_values("datetime"), utc=True)
    out["regime_pre_hmm"] = dt.map(r).fillna(0).astype(np.int64)
    return out
