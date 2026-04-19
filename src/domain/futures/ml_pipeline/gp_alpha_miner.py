"""
GP Alpha Miner - Cross-Sectional Ranking Edition.
Learns a universal formula across multiple symbols using panel data and CS-IC fitness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer
from numba import njit
from sklearn.preprocessing import QuantileTransformer

from src.domain.futures.ml_pipeline.cross_sectional_utils import CrossSectionalPipelineUtils
from src.domain.futures.ml_pipeline.feature_engineering import GP_ENGINEERED_FEATURE_NAMES
from src.domain.futures.ml_pipeline.gp_alpha_filter import filter_gp_alpha_columns

_logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True)  # type: ignore[untyped-decorator]
def _spearman_fast(ry: np.ndarray, rp: np.ndarray, cnt: int) -> float:
    """O(N) calculation of Spearman correlation using ordinal ranks 1..N"""
    if cnt < 2:
        return 0.0
    sum_d2 = 0.0
    for i in range(cnt):
        d = ry[i] - rp[i]
        sum_d2 += d * d
    den = float(cnt) * (float(cnt) * float(cnt) - 1.0)
    if den < 1e-12:
        return 0.0
    return 1.0 - (6.0 * sum_d2) / den


@njit(cache=True, fastmath=True)  # type: ignore[untyped-decorator]
def _n_valid_required(n_sym: int) -> int:
    if n_sym >= 15:
        frac = 0.3
    else:
        frac = max(0.5, 5.0 / max(float(n_sym), 1.0))
    req = int(np.ceil(frac * float(n_sym)))
    if req < 2:
        req = 2
    if req > n_sym:
        req = n_sym
    return req


@njit(cache=True, fastmath=True)  # type: ignore[untyped-decorator]
def cs_icir_fitness(
    y: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    """
    Spearman CS-IC (time-weighted by sample_weight rows, e.g. TBM) + ICIR-style term.
    sample_weight[0] encodes n_symbols; grid position (t=0,s=0) is hint-only and skipped for IC.
    """
    n_symbols = int(sample_weight[0])
    n_total = len(y)
    if n_total < 100 or n_symbols < 2:
        return -1.0
    n_time = n_total // n_symbols
    if n_time < 1 or n_symbols * n_time != n_total:
        return -1.0

    n_req = _n_valid_required(n_symbols)

    ic_buf = np.empty(n_time, dtype=np.float64)
    wt_buf = np.empty(n_time, dtype=np.float64)
    ic_cnt = 0

    yv = np.empty(n_symbols, dtype=np.float64)
    pv = np.empty(n_symbols, dtype=np.float64)
    wv = np.empty(n_symbols, dtype=np.float64)
    ry_full = np.empty(n_symbols, dtype=np.float64)
    rp_full = np.empty(n_symbols, dtype=np.float64)

    for t in range(n_time):
        s = t * n_symbols
        y_slice = y[s : s + n_symbols]
        p_slice = y_pred[s : s + n_symbols]
        w_slice = sample_weight[s : s + n_symbols]
        cnt = 0
        for i in range(n_symbols):
            if t == 0 and i == 0:
                continue
            wi = w_slice[i]
            if wi > 0.0 and np.isfinite(y_slice[i]) and np.isfinite(p_slice[i]):
                yv[cnt] = y_slice[i]
                pv[cnt] = p_slice[i]
                wv[cnt] = wi
                cnt += 1
        if cnt < n_req:
            continue

        y_valid = yv[:cnt]
        p_valid = pv[:cnt]
        
        order_y = np.argsort(y_valid)
        for pos in range(cnt):
            ry_full[order_y[pos]] = float(pos + 1)
            
        order_p = np.argsort(p_valid)
        for pos in range(cnt):
            rp_full[order_p[pos]] = float(pos + 1)
            
        ic = _spearman_fast(ry_full, rp_full, cnt)
        
        if not np.isfinite(ic):
            continue
        w_bar = 0.0
        for j in range(cnt):
            w_bar += wv[j]
        w_bar /= float(cnt)
        ic_buf[ic_cnt] = ic
        wt_buf[ic_cnt] = w_bar
        ic_cnt += 1

    if ic_cnt < 30:
        return -1.0

    sum_w = 0.0
    sum_wic = 0.0
    for i in range(ic_cnt):
        sum_w += wt_buf[i]
        sum_wic += ic_buf[i] * wt_buf[i]
    mu = sum_wic / (sum_w + 1e-18)

    var_w = 0.0
    for i in range(ic_cnt):
        d = ic_buf[i] - mu
        var_w += wt_buf[i] * d * d
    var_w /= sum_w + 1e-18
    sd = np.sqrt(var_w) + 1e-9
    icir = mu / sd
    icir_sr = icir * np.sqrt(2190.0)
    sign = 1.0 if mu >= 0.0 else -1.0
    capped = min(abs(icir_sr), 5.0) / 5.0
    return 0.4 * mu + 0.6 * sign * capped


_CS_FITNESS = make_fitness(function=cs_icir_fitness, greater_is_better=True, wrap=True)


def _signed_log_arr(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _signed_sqrt_arr(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.sqrt(np.abs(x))


signed_log = make_function(function=_signed_log_arr, name="slog", arity=1, wrap=False)
signed_sqrt = make_function(function=_signed_sqrt_arr, name="ssqrt", arity=1, wrap=False)

_GP_FUNCTION_SET = [
    "add",
    "sub",
    "mul",
    "div",
    "sqrt",
    "log",
    "abs",
    "neg",
    "inv",
    "max",
    "min",
    signed_log,
    signed_sqrt,
]


def _cache_stem_suffix(population_size: int, generations: int, horizons: tuple[int, ...]) -> str:
    hkey = "-".join(str(h) for h in horizons)
    return f"v10_icir_p{population_size}_g{generations}_h{hkey}"


def _pct_uniform_cs_is_fit(unstacked: pd.DataFrame, is_row_mask: np.ndarray) -> pd.DataFrame:
    """
    Replace full-panel rank(pct) with IS-fitted empirical CDF → uniform [0,1] on all rows
    (frozen CDF applied to OOS rows; no cross-time OOS leakage into IS ranks).
    """
    if unstacked.shape[1] < 2 or len(unstacked) != int(is_row_mask.shape[0]):
        return pd.DataFrame(0.5, index=unstacked.index, columns=unstacked.columns)
    mat = unstacked.to_numpy(dtype=np.float64)
    is_rows = np.asarray(is_row_mask, dtype=bool)[: mat.shape[0]]
    is_vals = mat[is_rows, :].ravel()
    is_vals = is_vals[np.isfinite(is_vals)]
    if is_vals.size < 100 or float(np.nanstd(is_vals)) < 1e-14:
        return pd.DataFrame(0.5, index=unstacked.index, columns=unstacked.columns)
    nq = int(min(1000, max(10, is_vals.size)))
    qt = QuantileTransformer(
        n_quantiles=nq,
        output_distribution="uniform",
        random_state=42,
        subsample=min(50_000, int(is_vals.size)),
    )
    qt.fit(is_vals.reshape(-1, 1))
    flat = np.where(np.isfinite(mat), mat, np.nan).reshape(-1, 1)
    out_flat = np.full(flat.shape[0], 0.5, dtype=np.float64)
    m = np.isfinite(flat[:, 0])
    if m.any():
        out_flat[m] = qt.transform(flat[m, :].reshape(-1, 1)).ravel()
    out = out_flat.reshape(mat.shape)
    return pd.DataFrame(out, index=unstacked.index, columns=unstacked.columns)


def _resolve_gp_feature_columns(panel_df: pd.DataFrame) -> list[str]:
    blocked = {"target", "close", "tbm_gp_weight", "regime_pre_hmm"}
    cs_names = sorted(c for c in panel_df.columns if c.startswith("cs_"))
    if cs_names:
        extras = [c for c in ("cross_vol_rank", "cross_ret_24h_rank") if c in panel_df.columns]
        return sorted({c for c in cs_names + extras if c not in blocked})
    return [c for c in panel_df.columns if c not in blocked]


@dataclass
class GPAlphaMiner:
    n_features_to_select: int = 15
    population_size: int = 1000
    generations: int = 20
    target_horizon: int = 6
    target_horizons: tuple[int, ...] = (3, 6, 12, 24)
    n_jobs: int = 1
    parsimony_coefficient: float = 0.001

    def mine_alphas_cs(
        self,
        panel_df: pd.DataFrame,
        cache_path: Path | None = None,
        is_end_date: str | None = None,
        filter_options: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Learns ONE universal formula across symbols using Panel Data.
        panel_df: MultiIndex (datetime, symbol).
        is_end_date: If provided, only data BEFORE this date is used for fitness.
        """
        if panel_df.empty:
            return pd.DataFrame()

        work_panel = panel_df.copy()
        src_cols = [c for c in GP_ENGINEERED_FEATURE_NAMES if c in work_panel.columns]
        if src_cols:
            work_panel = CrossSectionalPipelineUtils.cs_rank_transform(work_panel, src_cols)

        # Cache check
        versioned_cache: Path | None = None
        if cache_path is not None:
            n_syms = work_panel.index.get_level_values("symbol").nunique()
            tag = _cache_stem_suffix(
                self.population_size, self.generations, self.target_horizons
            )
            versioned_cache = cache_path.with_stem(f"{cache_path.stem}_{tag}_s{n_syms}")
            if versioned_cache.exists():
                try:
                    loaded = pd.read_parquet(versioned_cache)
                    if len(loaded) == len(panel_df):
                        _logger.info("GP CS Cache Hit: %s.", versioned_cache.name)
                        return loaded
                except Exception as e:
                    _logger.debug("GP Cache load failed: %s", e)

        # 1. Prepare Rectangular Grid
        unstacked_y = work_panel["target"].unstack(level="symbol")
        unique_times = unstacked_y.index
        unique_symbols = unstacked_y.columns
        n_symbols = len(unique_symbols)

        if is_end_date:
            _cut = pd.to_datetime(is_end_date, utc=True)
            if getattr(unique_times, "tz", None) is None:
                _ut = pd.to_datetime(unique_times, utc=True)
            else:
                _ut = unique_times.tz_convert("UTC")
            is_row_mask_utc = np.asarray(_ut < _cut, dtype=bool)
        else:
            is_row_mask_utc = np.ones(len(unique_times), dtype=bool)

        full_grid_index = pd.MultiIndex.from_product(
            [unique_times, unique_symbols], names=["datetime", "symbol"]
        )

        feat_cols = _resolve_gp_feature_columns(work_panel)
        template = pd.DataFrame(index=full_grid_index)
        join_cols = [*feat_cols, "target"]
        if "tbm_gp_weight" in work_panel.columns:
            join_cols = [*feat_cols, "tbm_gp_weight", "target"]
        aligned_df = template.join(work_panel[join_cols]).fillna(np.nan)

        y_raw = aligned_df["target"].values
        x_grid = aligned_df[feat_cols].values

        # Determine training mask
        sample_weight = np.where(pd.isna(y_raw), 0.0, 1.0)
        if "tbm_gp_weight" in aligned_df.columns:
            tw = aligned_df["tbm_gp_weight"].fillna(1.0).to_numpy(dtype=np.float64)
            sample_weight = sample_weight * np.clip(tw, 0.25, 3.0)
        if is_end_date:
            times = aligned_df.index.get_level_values("datetime")
            if times.tz is None:
                times = times.tz_localize("UTC")
            else:
                times = times.tz_convert("UTC")
            cutoff_dt = pd.to_datetime(is_end_date, utc=True)
            is_mask = np.asarray(times < cutoff_dt)
            sample_weight = sample_weight * is_mask

        # [REFACTORED] Downsampling to 4h is removed to match 1h main architecture.
        # Use full 1h panel for GP fitness calculation.

        # Internal 80/20 chronological holdout + drop NaN rows (CS grid: time x symbol)
        n_time = len(unique_times)
        train_t_limit = max(1, int(n_time * 0.8))
        row_t: np.ndarray = np.arange(len(aligned_df), dtype=np.int64) // n_symbols
        fit_time_ok = row_t < train_t_limit
        nan_row = np.isnan(x_grid).any(axis=1) | np.isnan(y_raw)
        w_fit = fit_time_ok.astype(np.float64)
        w_nan = (~nan_row).astype(np.float64)
        sample_weight = sample_weight * w_fit * w_nan

        sample_weight[0] = float(n_symbols)  # Hint for fitness (row 0 only; see docstring)

        # 2. GP Training
        st = SymbolicTransformer(
            feature_names=feat_cols,
            function_set=_GP_FUNCTION_SET,
            metric=_CS_FITNESS,
            population_size=self.population_size,
            generations=self.generations,
            stopping_criteria=0.08,
            n_components=self.n_features_to_select,
            random_state=42,
            n_jobs=self.n_jobs,
            verbose=1,
            parsimony_coefficient=self.parsimony_coefficient,
            init_depth=(2, 5),
            p_crossover=0.7,
            p_subtree_mutation=0.15,
            p_hoist_mutation=0.05,
            p_point_mutation=0.05,
        )

        x_clean = np.where(np.isfinite(x_grid), x_grid, 0.0)
        y_clean = np.where(np.isfinite(y_raw), y_raw, 0.0)
        st.fit(x_clean, y_clean, sample_weight=sample_weight)

        # 3. Predict and Map back
        self._st = st
        out_grid = st.transform(x_clean)
        cols = [f"gp_alpha_{i:02d}" for i in range(self.n_features_to_select)]
        full_alpha_df = pd.DataFrame(out_grid, index=full_grid_index, columns=cols)

        alpha_series_list = []
        for col in cols:
            unstacked = full_alpha_df[col].unstack(level="symbol")
            if unstacked.std(axis=1).mean() < 1e-12:
                s = pd.Series(0.5, index=full_grid_index, name=col)
            else:
                ranked = _pct_uniform_cs_is_fit(unstacked, is_row_mask_utc)
                s = ranked.stack(future_stack=True)
                s.name = col
            alpha_series_list.append(s)

        alpha_df_all = pd.concat(alpha_series_list, axis=1)
        fo = filter_options or {}
        alpha_df_all, filt_meta = filter_gp_alpha_columns(
            alpha_df_all,
            panel_df,
            is_end_date=is_end_date,
            n_trials=self.n_features_to_select,
            fdr_q=float(fo.get("fdr_q", 0.10)),
            use_newey_west=bool(fo.get("use_newey_west", False)),
            use_ewma_ic_stat=bool(fo.get("use_ewma_ic_stat", False)),
            ewma_half_life=float(fo.get("ewma_half_life", 540.0)),
            symbol_balance_max=float(fo.get("symbol_balance_max", 3.0)),
            require_regime_gate=bool(fo.get("require_regime_gate", True)),
        )
        if float(filt_meta.get("neutralize_primary", 0.0)) > 0.5:
            alpha_df_all["gp_alpha_00"] = 0.5

        alpha_df = alpha_df_all.reindex(panel_df.index).fillna(0.5)

        best_fitness = 0.0
        if hasattr(st, "_best_programs") and len(st._best_programs) > 0:
            best_fitness = st._best_programs[0].fitness_
        alpha_df.attrs["best_fitness"] = float(best_fitness)
        alpha_df.attrs["gp_alpha_filter"] = filt_meta

        if versioned_cache is not None:
            try:
                versioned_cache.parent.mkdir(parents=True, exist_ok=True)
                alpha_df.to_parquet(versioned_cache)
            except Exception as e:
                _logger.warning("GP CS Cache Write Failed: %s", e)

        return alpha_df

    def transform_cs(self, panel_df: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
        """
        Applies a previously trained universal model to a new panel.
        """
        if panel_df.empty:
            return pd.DataFrame()

        work = panel_df.copy()
        src_cols = [c for c in GP_ENGINEERED_FEATURE_NAMES if c in work.columns]
        if src_cols:
            work = CrossSectionalPipelineUtils.cs_rank_transform(work, src_cols)
        feat_cols = _resolve_gp_feature_columns(work)
        x = work[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)

        if hasattr(self, "_st") and self._st is not None:
            out_arr = self._st.transform(x)
            cols = [f"gp_alpha_{i:02d}" for i in range(self.n_features_to_select)]
            return pd.DataFrame(out_arr, index=panel_df.index, columns=cols)

        _logger.warning("transform_cs: _st is missing. Returning neutral features.")
        cols = [f"gp_alpha_{i:02d}" for i in range(self.n_features_to_select)]
        return pd.DataFrame(0.5, index=panel_df.index, columns=cols)
