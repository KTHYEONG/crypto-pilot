"""Cross-Sectional ML Pipeline Utilities.

Handles Panel data creation, Z-Score targeting, and cross-sectional normalization.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


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
        """Calculate vol-adjusted rank targets across multiple horizons."""
        if panel_df.empty or "close" not in panel_df.columns:
            return pd.Series(dtype=np.float64)

        # Use OHLC for more accurate volatility adjustment (Yang-Zhang proxy)
        close = panel_df["close"].unstack(level="symbol")
        open_ = (
            panel_df["open"].unstack(level="symbol")
            if "open" in panel_df.columns
            else close.shift(1).fillna(close)
        )
        high = (
            panel_df["high"].unstack(level="symbol") if "high" in panel_df.columns else close
        )
        low = (
            panel_df["low"].unstack(level="symbol") if "low" in panel_df.columns else close
        )

        h_list = tuple(horizons)
        # Use more balanced weighting (decaying but not as aggressive as 1/sqrt(h))
        # to prioritize medium-term horizons (6h-12h) while keeping 24h as anchor.
        if weights is None:
            w_arr = np.array([1.0 / (float(h) ** 0.35) for h in h_list], dtype=np.float64)
        else:
            w_arr = np.asarray(weights, dtype=np.float64)
        w_norm = w_arr / np.sum(w_arr)

        rank_panels: list[pd.DataFrame] = []
        # Calculate a more robust 24h volatility (Yang-Zhang approximation)
        log_ho = np.log((high / open_).clip(lower=1e-12))
        log_lo = np.log((low / open_).clip(lower=1e-12))
        log_co = np.log((close / open_).clip(lower=1e-12))
        # Rogers-Satchell variance proxy
        rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        vol_base = np.sqrt(rs_var.rolling(24, min_periods=12).mean().clip(lower=1e-9))

        for h, w in zip(h_list, w_norm, strict=True):
            fwd = np.log(close.shift(-h) / close)
            if fwd.iloc[-h:].notna().any().any():
                _logger.warning(
                    "create_multi_horizon_rank_targets: expected NaN tail not satisfied "
                    "(h=%s); check close grid / lookahead.",
                    h,
                )
            # Normalize by horizon-scaled volatility
            vol = vol_base * np.sqrt(float(h))
            adj = fwd / (vol + 1e-9)
            
            # Cross-sectional rank of risk-adjusted returns
            rank = adj.rank(axis=1, pct=True, method="average")
            rank_panels.append(rank * float(w))

        composite = sum(rank_panels)
        final = composite.rank(axis=1, pct=True, method="average")
        # Scale to [-1, 1] for regression objective compatibility
        stacked = ((final - 0.5) * 2.0).stack(future_stack=True).reindex(panel_df.index)
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
        # This implements 개편안 A-1 (전면적 횡단면 정규화)
        from src.domain.futures.ml_pipeline.feature_engineering import (
            ALPHA_ENGINEERED_FEATURE_NAMES,
        )
        
        target_cols = [c for c in ALPHA_ENGINEERED_FEATURE_NAMES if c in df.columns]
        for col in target_cols:
            # Beta-Neutral Momentum (A-2) is handled here naturally as Z-Score removes market-wide trend
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
            # If close is missing (e.g. somehow stripped), add dummy features
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
