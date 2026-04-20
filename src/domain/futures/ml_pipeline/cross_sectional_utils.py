"""
Cross-Sectional ML Pipeline Utilities.
Handles Panel data creation, Z-Score targeting, and cross-sectional normalization.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


class CrossSectionalPipelineUtils:
    @staticmethod
    def build_panel_df(
        data_map: Dict[str, Dict[str, pd.DataFrame]], tf: str = "1h"
    ) -> pd.DataFrame:
        """
        Merges multiple symbol DataFrames into a single MultiIndex DataFrame (datetime, symbol).
        """
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
        """
        Calculates forward returns and applies cross-sectional Z-Score normalization.
        Y = (Ret_i - Mean_Cross) / Std_Cross
        """
        # 1. Calculate forward log returns per symbol
        # We group by symbol to avoid look-ahead from other symbols during shift
        close = panel_df["close"].unstack(level="symbol")
        fwd_ret = np.log(close.shift(-horizon) / close)
        
        # 2. Cross-sectional Z-Score (Normalization at each time step)
        # axis=1 means compute mean/std across symbols for each row (time)
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
        out = panel_df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = CrossSectionalPipelineUtils.cs_median_impute_series(out[c])
        return out

    @staticmethod
    def cs_rank_transform(panel_df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
        """Per-datetime percentile rank [0,1]; NaN → neutral 0.5. Adds cs_* columns."""
        out = panel_df.copy()
        for col in feature_cols:
            if col not in panel_df.columns:
                continue
            wide = panel_df[col].unstack(level="symbol")
            ranked = wide.rank(axis=1, pct=True, method="average")
            out[f"cs_{col}"] = (
                ranked.stack(future_stack=True).reindex(panel_df.index).fillna(0.5)
            )
        return out

    @staticmethod
    def create_multi_horizon_rank_targets(
        panel_df: pd.DataFrame,
        horizons: tuple[int, ...] = (3, 6, 12, 24),
        weights: tuple[float, ...] | None = None,
    ) -> pd.Series:
        """
        Vol-adjusted forward log returns per horizon, cross-sectional rank blend, mapped to [-1,1].
        """
        close = panel_df["close"].unstack(level="symbol")
        h_list = tuple(horizons)
        if weights is None:
            w_arr = np.array([1.0 / np.sqrt(float(h)) for h in h_list], dtype=np.float64)
        else:
            w_arr = np.asarray(weights, dtype=np.float64)
        w_norm = w_arr / np.sum(w_arr)

        rank_panels: list[pd.DataFrame] = []
        log_ret_1 = np.log(close / close.shift(1))
        vol_base = log_ret_1.rolling(24, min_periods=12).std()

        for h, w in zip(h_list, w_norm, strict=True):
            fwd = np.log(close.shift(-h) / close)
            if fwd.iloc[-h:].notna().any().any():
                _logger.warning(
                    "create_multi_horizon_rank_targets: expected NaN tail not satisfied "
                    "(h=%s); check close grid / lookahead.",
                    h,
                )
            vol = vol_base * np.sqrt(float(h))
            adj = fwd / (vol + 1e-9)
            rank = adj.rank(axis=1, pct=True, method="average")
            rank_panels.append(rank * float(w))

        composite = sum(rank_panels)
        final = composite.rank(axis=1, pct=True, method="average")
        stacked = ((final - 0.5) * 2.0).stack(future_stack=True).reindex(panel_df.index)
        return stacked

    @staticmethod
    def add_cross_sectional_features(panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds features relative to the universe (e.g. Volume Rank, Relative Momentum).
        """
        df = panel_df.copy()
        
        # Example: Cross-sectional Volume Rank (0.0 to 1.0)
        vol = df["volume"].unstack(level="symbol")
        vol_rank = vol.rank(axis=1, pct=True)
        df["cross_vol_rank"] = vol_rank.stack(future_stack=True).reindex(df.index)
        
        # Example: Cross-sectional 24h Return Rank
        close = df["close"].unstack(level="symbol")
        ret_24h = np.log(close / close.shift(24))
        ret_rank = ret_24h.rank(axis=1, pct=True)
        df["cross_ret_24h_rank"] = ret_rank.stack(future_stack=True).reindex(df.index)
        
        return df.fillna(0.0)

    @staticmethod
    def add_systemic_features(panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds market-wide systemic features to the panel.
        - CS Dispersion: Std of cross-sectional log returns.
        - Market Breadth: Pct of symbols with close > 20h SMA.
        """
        df = panel_df.copy()
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
