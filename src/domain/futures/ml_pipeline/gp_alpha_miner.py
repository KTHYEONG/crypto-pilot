"""Symbolic regression (gplearn) walk-forward alpha mining with NW-adjusted IC fitness."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicTransformer
from numba import njit

_logger = logging.getLogger(__name__)


@njit  # type: ignore
def _fast_rank_average(a: np.ndarray) -> np.ndarray:
    """Numba-accelerated ranking with 'average' method for ties."""
    n = len(a)
    ranks = np.empty(n, dtype=np.float64)
    idx = np.argsort(a)
    i = 0
    while i < n:
        j = i + 1
        while j < n and a[idx[j]] == a[idx[i]]:
            j += 1
        # Average rank for the tie group (1-based)
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[idx[k]] = avg_rank
        i = j
    return ranks


@njit  # type: ignore
def _ols_nw_tstat(y: np.ndarray, x: np.ndarray, lag: int) -> float:
    """Numba-accelerated OLS with Newey-West (HAC) adjusted t-stat for simple regression."""
    n = len(y)
    if n < 2:
        return 0.0

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x * x)
    sum_xy = np.sum(x * y)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-15:
        return 0.0

    b1 = (n * sum_xy - sum_x * sum_y) / denom
    b0 = (sum_y - b1 * sum_x) / n
    resid = y - (b0 + b1 * x)

    # Var(b) = (X'X)^-1 * S * (X'X)^-1
    # For simple regression, we only need the variance of b1 (slope)
    d01 = -sum_x / denom
    d11 = n / denom

    s00, s01, s11 = 0.0, 0.0, 0.0
    for j in range(lag + 1):
        weight = 1.0 - j / (lag + 1.0)
        if j == 0:
            for t in range(n):
                ee = resid[t] * resid[t]
                s00 += ee
                s01 += ee * x[t]
                s11 += ee * x[t] * x[t]
        else:
            g00, g01, g11 = 0.0, 0.0, 0.0
            for t in range(j, n):
                ee = resid[t] * resid[t - j]
                g00 += 2.0 * ee
                g01 += ee * (x[t] + x[t - j])
                g11 += 2.0 * ee * x[t] * x[t - j]
            s00 += weight * g00
            s01 += weight * g01
            s11 += weight * g11

    var_b1 = d01 * d01 * s00 + 2.0 * d01 * d11 * s01 + d11 * d11 * s11
    if var_b1 <= 1e-15:
        return 0.0

    return float(b1 / np.sqrt(var_b1))


def newey_west_ic_fitness(
    y: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None,
) -> float:
    """
    Spearman IC proxy with HAC (Newey-West) t-stat on ranked series.
    Optimized via Numba to bypass Pandas/Statsmodels overhead.
    """
    del sample_weight
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    p_arr = np.asarray(y_pred, dtype=np.float64).ravel()

    mask = np.isfinite(y_arr) & np.isfinite(p_arr)
    n_valid = int(np.sum(mask))
    if n_valid < 40:
        return -1.0

    y_valid = y_arr[mask]
    p_valid = p_arr[mask]

    # 1. Fast Ranking
    yr = _fast_rank_average(y_valid)
    pr = _fast_rank_average(p_valid)

    # 2. HAC Lag Selection
    lag = int(max(1, min(int(4.0 * (n_valid / 100.0) ** (2.0 / 9.0)), 50)))

    # 3. Fast NW t-stat
    try:
        tstat = float(_ols_nw_tstat(yr, pr, lag))
        return tstat if np.isfinite(tstat) else 0.0
    except Exception:
        # Fallback to simple correlation if NW fails
        corr = float(np.corrcoef(yr, pr)[0, 1])
        return corr if np.isfinite(corr) else 0.0


_NW_FITNESS = make_fitness(function=newey_west_ic_fitness, greater_is_better=True, wrap=True)


@dataclass
class GPAlphaMiner:
    n_features_to_select: int = 15
    population_size: int = 2000
    generations: int = 20
    walk_forward_window: int = 8760
    step_size: int = 2190
    target_horizon: int = 24
    n_jobs: int = 1

    def mine_alphas(self, df: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
        """
        Offline walk-forward SymbolicTransformer; outputs gp_alpha_00.. columns on 1h index.
        """
        if df.empty or len(df) < self.walk_forward_window + self.target_horizon + 10:
            _logger.warning("GP: insufficient 1h bars; returning zeros.")
            return self._empty_alpha_df(df)

        if cache_path is not None and cache_path.exists():
            try:
                loaded = pd.read_parquet(cache_path)
                if len(loaded) == len(df) and loaded.index.equals(df.index):
                    _logger.info("GP cache hit: %s (%d rows)", cache_path.name, len(loaded))
                    return loaded
            except Exception as e:
                _logger.debug("GP cache read failed: %s", e)

        feats = df.copy()
        if "close" not in feats.columns:
            return self._empty_alpha_df(df)
        feat_cols = [
            c
            for c in feats.columns
            if c not in ("datetime", "close") and np.issubdtype(feats[c].dtype, np.number)
        ]
        if not feat_cols:
            return self._empty_alpha_df(df)

        x_all = feats[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(
            dtype=np.float64
        )
        close = feats["close"].astype(np.float64).to_numpy()
        n = len(df)
        h = self.target_horizon
        y_all = np.full(n, np.nan, dtype=np.float64)
        for i in range(n - h):
            y_all[i] = float(np.log(max(close[i + h], 1e-12) / max(close[i], 1e-12)))
        out = np.zeros((n, self.n_features_to_select), dtype=np.float64)

        total_folds = max(1, (n - self.step_size - self.walk_forward_window) // self.step_size)
        fold_idx = 0
        for t_end in range(self.walk_forward_window, n - self.step_size, self.step_size):
            fold_idx += 1
            _logger.info(
                "GP fold %d/%d (t_end=%d, pop=%d, gen=%d)...",
                fold_idx, total_folds, t_end, self.population_size, self.generations,
            )
            a0 = t_end - self.walk_forward_window
            x_tr = x_all[a0:t_end]
            y_tr = y_all[a0:t_end]
            m = np.isfinite(y_tr) & np.all(np.isfinite(x_tr), axis=1)
            if int(m.sum()) < 200:
                continue
            x_tr, y_tr = x_tr[m], y_tr[m]
            try:
                st = SymbolicTransformer(
                    population_size=min(self.population_size, len(x_tr)),
                    generations=min(self.generations, 30),
                    hall_of_fame=100,
                    n_components=self.n_features_to_select,
                    metric=_NW_FITNESS,
                    parsimony_coefficient=0.001,
                    max_samples=0.9,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    verbose=0,
                )
                st.fit(x_tr, y_tr)
                x_te = x_all[t_end : t_end + self.step_size]
                part = st.transform(x_te)
                out[t_end : t_end + self.step_size, :] = part
            except Exception as e:
                _logger.debug("GP fold failed at %s: %s", t_end, e)

        cols = [f"gp_alpha_{i:02d}" for i in range(self.n_features_to_select)]
        alpha_df = pd.DataFrame(out, index=df.index, columns=cols)
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                alpha_df.to_parquet(cache_path)
            except Exception as e:
                _logger.warning("GP cache write failed: %s", e)
        return alpha_df

    def _empty_alpha_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [f"gp_alpha_{i:02d}" for i in range(self.n_features_to_select)]
        return pd.DataFrame(0.0, index=df.index, columns=cols)
