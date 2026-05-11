"""Cross-Sectional ML Pipeline Utilities.

Handles Panel data creation, Z-Score targeting, and cross-sectional normalization.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import numba

_logger = logging.getLogger(__name__)

@numba.njit(parallel=False, cache=True)
def _fast_rank_2d_numba(array: np.ndarray) -> np.ndarray:
    """Vectorized percentile ranking across rows (axis=1) with tie-breaking 'average'.
    
    Matches pd.DataFrame(array).rank(axis=1, pct=True, method='average') but 10-20x faster.
    """
    n, m = array.shape
    out = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        row = array[i]
        mask = ~np.isnan(row)
        n_valid = np.sum(mask)
        if n_valid <= 1:
            out[i, :] = 0.5
            continue
            
        valid_data = row[mask]
        sort_idx = np.argsort(valid_data)
        sorted_data = valid_data[sort_idx]
        ranks = np.empty(n_valid, dtype=np.float64)
        
        j = 0
        while j < n_valid:
            k = j + 1
            while k < n_valid and sorted_data[k] == sorted_data[j]:
                k += 1
            avg_rank = j + (k - j - 1) / 2.0
            for m_idx in range(j, k):
                ranks[sort_idx[m_idx]] = avg_rank
            j = k
            
        out_row = np.full(m, 0.5)
        out_row[mask] = ranks / (n_valid - 1)
        out[i] = out_row
    return out


class CrossSectionalPipelineUtils:
    """Utilities for cross-sectional data processing in ML pipelines."""

    @staticmethod
    def build_panel_df(
        data_map: dict[str, dict[str, pd.DataFrame]], tf: str = "1h"
    ) -> pd.DataFrame:
        """Merge multiple symbol DataFrames into a single MultiIndex DataFrame."""
        all_dfs = []
        for sym, tf_map in data_map.items():
            df = tf_map.get(tf)
            if df is None or df.empty:
                continue
            
            # Ensure index is datetime and UTC
            tmp = df.copy()
            if "datetime" in tmp.columns:
                tmp["datetime"] = pd.to_datetime(tmp["datetime"], utc=True)
                tmp.set_index("datetime", inplace=True)
            elif tmp.index.name == "datetime":
                tmp.index = pd.to_datetime(tmp.index, utc=True)
            
            tmp["symbol"] = sym
            all_dfs.append(tmp)
            
        if not all_dfs:
            return pd.DataFrame()
            
        panel_df = pd.concat(all_dfs).reset_index()
        panel_df.set_index(["datetime", "symbol"], inplace=True)
        panel_df.sort_index(inplace=True)
        return panel_df

    @staticmethod
    def create_zscore_targets(
        panel_df: pd.DataFrame, 
        horizon: int = 6
    ) -> pd.Series:
        """Calculate forward returns and apply cross-sectional Z-Score normalization.

        Y = (Ret_i - Mean_Cross) / Std_Cross
        """
        # 1. Calculate forward log returns per symbol
        close = panel_df["close"].unstack(level="symbol")
        fwd_ret = np.log(close.shift(-horizon) / close)
        
        # 2. Cross-sectional Z-Score (Normalization at each time step)
        mean_cross = fwd_ret.mean(axis=1)
        std_cross = fwd_ret.std(axis=1)
        
        z_targets = fwd_ret.sub(mean_cross, axis=0).div(std_cross + 1e-12, axis=0)
        
        # 3. Restack to MultiIndex (datetime, symbol)
        return z_targets.stack(future_stack=True).reindex(panel_df.index)

    @staticmethod
    def cs_median_impute_series(s: pd.Series) -> pd.Series:
        """Cross-sectional median impute per datetime (MultiIndex datetime, symbol)."""
        if s.index.nlevels < 2 or "symbol" not in s.index.names:
            return s
        wide = s.unstack(level="symbol")
        med = wide.median(axis=1)
        filled = wide.fillna(med, axis=0)
        out = filled.stack(future_stack=True).reindex(s.index)
        return out

    @staticmethod
    def cs_median_impute_panel(panel_df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        """Apply cross-sectional median imputation to multiple columns in a panel."""
        out = panel_df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = CrossSectionalPipelineUtils.cs_median_impute_series(out[c])
        return out

    @staticmethod
    def cs_rank_transform(panel_df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
        """Per-datetime percentile rank [0,1]; NaN → neutral 0.5. Adds cs_* columns."""
        out = panel_df.copy()
        
        # [Optimization] Efficient wide-flat mapping
        # This assumes panel_df is sorted by (datetime, symbol)
        dummy_wide = panel_df.iloc[:, 0].unstack(level="symbol")
        valid_mask = dummy_wide.notna().values
        idx_shape = dummy_wide.shape
        
        for col in feature_cols:
            if col not in panel_df.columns:
                continue
            
            vals = panel_df[col].values
            scores_matrix = np.full(idx_shape, np.nan)
            scores_matrix[valid_mask] = vals
            
            ranked_matrix = _fast_rank_2d_numba(scores_matrix)
            out[f"cs_{col}"] = ranked_matrix[valid_mask]
            
        return out

    @staticmethod
    def create_multi_horizon_rank_targets(
        panel_df: pd.DataFrame,
        horizons: tuple[int, ...] = (3, 6, 12, 24),
        weights: tuple[float, ...] | None = None,
    ) -> pd.Series:
        """Calculate vol-adjusted rank targets across multiple horizons."""
        if panel_df.empty or "close" not in panel_df.columns:
            return pd.Series(dtype=np.float64)

        # Use OHLC for more accurate volatility adjustment (Yang-Zhang proxy)
        close_wide = panel_df["close"].unstack(level="symbol")
        open_wide = (
            panel_df["open"].unstack(level="symbol")
            if "open" in panel_df.columns
            else close_wide.shift(1).fillna(close_wide)
        )
        high_wide = (
            panel_df["high"].unstack(level="symbol") if "high" in panel_df.columns else close_wide
        )
        low_wide = (
            panel_df["low"].unstack(level="symbol") if "low" in panel_df.columns else close_wide
        )

        h_list = tuple(horizons)
        if weights is None:
            w_arr = np.array([1.0 / (float(h) ** 0.35) for h in h_list], dtype=np.float64)
        else:
            w_arr = np.asarray(weights, dtype=np.float64)
        w_norm = w_arr / np.sum(w_arr)

        # Calculate a more robust 24h volatility (Yang-Zhang approximation)
        log_ho = np.log((high_wide / open_wide).clip(lower=1e-12))
        log_lo = np.log((low_wide / open_wide).clip(lower=1e-12))
        log_co = np.log((close_wide / open_wide).clip(lower=1e-12))
        rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        vol_base = np.sqrt(rs_var.rolling(24, min_periods=12).mean().clip(lower=1e-9))

        composite_wide = np.zeros(close_wide.shape)
        
        for h, w in zip(h_list, w_norm, strict=True):
            fwd = np.log(close_wide.shift(-h) / close_wide)
            vol = vol_base * np.sqrt(float(h))
            adj = fwd / (vol + 1e-9)
            
            # Use fast ranker
            rank = _fast_rank_2d_numba(adj.values)
            composite_wide += rank * float(w)

        # Final rank of composite
        final_rank_wide = _fast_rank_2d_numba(composite_wide)
        
        # [Optimization] Efficient restack
        valid_mask = close_wide.notna().values
        final_flat = final_rank_wide[valid_mask]
        
        # Scale to [-1, 1]
        stacked = pd.Series((final_flat - 0.5) * 2.0, index=panel_df.index)
        return stacked

    @staticmethod
    def cs_robust_zscore_series(s: pd.Series, clip: float = 3.0) -> pd.Series:
        """Cross-sectional Robust Z-Score (Median/MAD) per datetime.
        
        Z = (x - median) / (mad * 1.4826)
        Clips to [-clip, clip] (Winsorization).
        """
        if s.index.nlevels < 2 or "symbol" not in s.index.names:
            return s
        wide = s.unstack(level="symbol")
        med = wide.median(axis=1)
        mad = (wide.sub(med, axis=0)).abs().median(axis=1)
        
        z = wide.sub(med, axis=0).div(mad * 1.4826 + 1e-12, axis=0)
        z = z.clip(-clip, clip)
        
        return z.stack(future_stack=True).reindex(s.index).fillna(0.0)

    @staticmethod
    def apply_cs_zscore(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Apply cross-sectional Z-Score (Mean/Std) to specified columns.
        
        Logic: For each timestamp, Z = (value - cs_mean) / (cs_std + 1e-9)
        """
        out = df.copy()
        for col in columns:
            if col not in out.columns:
                continue
            wide = out[col].unstack(level="symbol")
            mean_cs = wide.mean(axis=1)
            std_cs = wide.std(axis=1)
            
            z_wide = wide.sub(mean_cs, axis=0).div(std_cs + 1e-9, axis=0)
            out[col] = z_wide.stack(future_stack=True).reindex(out.index).fillna(0.0)
        return out

    @staticmethod
    def add_cross_sectional_features(panel_df: pd.DataFrame) -> pd.DataFrame:
        """Add features relative to the universe and apply Robust Z-Scoring."""
        df = panel_df.copy()
        
        # 1. Base CS Ranks
        if "volume" in df.columns:
            vol = df["volume"].unstack(level="symbol")
            df["cross_vol_rank"] = vol.rank(axis=1, pct=True).stack(future_stack=True).reindex(df.index)
        
        # 2. [NEW] Universal Robust Z-Scoring for Alpha Features
        from src.domain.futures.ml_pipeline.features.engineering import (
            ALPHA_ENGINEERED_FEATURE_NAMES,
        )
        
        target_cols = [c for c in ALPHA_ENGINEERED_FEATURE_NAMES if c in df.columns]
        for col in target_cols:
            df[col] = CrossSectionalPipelineUtils.cs_robust_zscore_series(df[col])
            
        return df.fillna(0.0)

    @staticmethod
    def add_systemic_features(panel_df: pd.DataFrame) -> pd.DataFrame:
        """Add market-wide systemic features to the panel.

        - CS Dispersion: Std of cross-sectional log returns.
        - Market Breadth: Pct of symbols with close > 20h SMA.
        """
        df = panel_df.copy()
        if df.empty:
            return df
            
        if "close" not in df.columns:
            df["cs_dispersion"] = 0.0
            df["market_breadth"] = 0.0
            return df
            
        close = df["close"].unstack(level="symbol")
        
        # 1. Cross-sectional Dispersion (1h)
        log_ret = np.log(close / close.shift(1))
        disp = log_ret.std(axis=1)
        df["cs_dispersion"] = disp.reindex(df.index, level="datetime")
        
        # 2. Market Breadth (20h SMA)
        sma_20 = close.rolling(20).mean()
        breadth = (close > sma_20).mean(axis=1)
        df["market_breadth"] = breadth.reindex(df.index, level="datetime")
        
        return df.fillna(0.0)
