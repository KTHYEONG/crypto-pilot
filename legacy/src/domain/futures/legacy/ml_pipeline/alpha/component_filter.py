"""Multiple-testing guard for GP cross-sectional alpha components (IS + OOS diagnostics)."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import skew as _scipy_skew

_logger = logging.getLogger(__name__)
_LONG_SLOT_COL_RE = re.compile(r"^alpha_long_(\d{2})$")
_SHORT_SLOT_COL_RE = re.compile(r"^alpha_short_(\d{2})$")


def _norm_sf(x: float) -> float:
    """Calculate survival function 1 - Phi(x) for standard normal.

    Args:
        x: Input value.

    Returns:
        Survival function value.

    """
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _benjamini_hochberg_reject(p_values: Sequence[float], q: float) -> np.ndarray:
    """Determine boolean mask for rejected null hypotheses using Benjamini-Hochberg.

    Args:
        p_values: Sequence of p-values.
        q: False Discovery Rate threshold.

    Returns:
        Boolean mask where True means null rejected.

    """
    p = np.clip(np.asarray(p_values, dtype=np.float64), 1e-15, 1.0)
    m = int(p.size)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    sort_p = p[order]
    rank_idx: np.ndarray = np.arange(1, m + 1, dtype=np.float64)
    thresh = q * rank_idx / float(m)
    ok = sort_p <= thresh
    if not ok.any():
        return np.zeros(m, dtype=bool)
    k = int(np.where(ok)[0].max())
    cutoff = float(sort_p[k])
    out = p <= cutoff
    return out


def _newey_west_se(x: np.ndarray, lag: int | None = None) -> float:
    """Calculate Newey-West HAC standard error of the mean.

    Uses Bartlett kernel.

    Args:
        x: Input data array.
        lag: Number of lags to consider. Defaults to rule-of-thumb if None.

    Returns:
        Standard error of the mean.

    """
    n = len(x)
    if n < 2:
        return float("nan")
    if lag is None:
        lag = max(1, math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = float(np.mean(x))
    e = x - mu
    gamma_0 = float(np.dot(e, e)) / float(n)
    var = gamma_0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        gamma_k = float(np.dot(e[:-k], e[k:])) / float(n)
        var += 2.0 * w * gamma_k
    return float(math.sqrt(max(var, 1e-12) / float(n)))


def _ewma_ic_stats(ic_arr: np.ndarray, half_life: float = 540.0) -> tuple[float, float]:
    """Calculate EWMA mean and volatility of IC.

    Args:
        ic_arr: Array of IC values.
        half_life: Decay half-life in bars.

    Returns:
        Tuple of (EWMA mean, EWMA volatility).

    """
    n = int(ic_arr.size)
    lam = math.log(2.0) / max(half_life, 1e-6)
    w = np.exp(-lam * np.arange(n - 1, -1, -1, dtype=np.float64))
    w = w / (float(np.sum(w)) + 1e-18)
    mu = float(np.dot(w, ic_arr))
    var = float(np.dot(w, (ic_arr - mu) ** 2))
    return mu, math.sqrt(max(var, 1e-18))


def _ic_half_life_bars(ic_arr: np.ndarray) -> float:
    """Calculate AR(1) half-life in bars.

    Returns 0 if non-stationary or ill-defined.

    Args:
        ic_arr: Array of IC values.

    Returns:
        Half-life in bars.

    """
    n = int(ic_arr.size)
    if n < 5:
        return 0.0
    a = ic_arr[:-1]
    b = ic_arr[1:]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    rho = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(rho) or rho <= 0.0 or rho >= 1.0:
        return 0.0
    return float(-math.log(2.0) / math.log(rho))


def _ic_half_life_bars_with_diag(ic_arr: np.ndarray) -> tuple[float, str]:
    """Calculate AR(1) half-life in bars with explicit diagnostic code.

    Args:
        ic_arr: Array of IC values.

    Returns:
        Tuple of (half-life in bars, diagnostic code).

    """
    n = int(ic_arr.size)
    if n < 5:
        return 0.0, "insufficient_samples"
    a = ic_arr[:-1]
    b = ic_arr[1:]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0, "zero_variance"
    rho = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(rho):
        return 0.0, "rho_non_finite"
    if rho <= 0.0 or rho >= 1.0:
        return 0.0, "rho_out_of_bounds"
    hl = float(-math.log(2.0) / math.log(rho))
    if not math.isfinite(hl):
        return 0.0, "half_life_non_finite"
    return hl, "ok"


def _deflated_sharpe_threshold(
    sharpe: float,
    n: int,
    skew: float,
    kurt_excess: float,
    n_trials: int,
) -> bool:
    """Apply DSR-style gate to filter components.

    Bailey & López de Prado (2014) style gate: SR minus expected max H0 SR.

    Args:
        sharpe: Calculated Sharpe Ratio.
        n: Sample size.
        skew: Skewness of returns.
        kurt_excess: Excess kurtosis of returns.
        n_trials: Number of trials (multiplicity).

    Returns:
        True if the component passes the gate.

    """
    if n < 5 or not np.isfinite(sharpe):
        return False
    var_sr = (1.0 - skew * sharpe + (kurt_excess / 4.0) * sharpe * sharpe) / max(n - 1, 1)
    var_sr = max(float(var_sr), 1e-12)
    e_max_h0 = float(math.sqrt(2.0 * math.log(max(n_trials, 2)))) * math.sqrt(var_sr)
    sr_adj = sharpe - e_max_h0
    return bool(sr_adj > 0.0 and sharpe > 0.0)


def _tail_decile_ic_series_fast(
    u_c: pd.DataFrame, u_tgt: pd.DataFrame, min_symbols: int = 8
) -> list[float]:
    """Calculate IC series focusing on tail deciles.

    Args:
        u_c: Unstacked component values.
        u_tgt: Unstacked target values.
        min_symbols: Minimum symbols required per bar.

    Returns:
        List of tail IC values.

    """
    valid_mask = u_c.notna() & u_tgt.notna()
    counts = valid_mask.sum(axis=1)
    mask_min = counts >= min_symbols
    uc_valid = u_c[mask_min]
    ut_valid = u_tgt[mask_min]
    if uc_valid.empty:
        return []
    pct_r = uc_valid.rank(axis=1, pct=True)
    mask_tail = (pct_r >= 0.9) | (pct_r <= 0.1)
    tail_counts = mask_tail.sum(axis=1)
    valid_tails = tail_counts >= 3
    uc_tail = uc_valid[mask_tail][valid_tails]
    ut_tail = ut_valid[mask_tail][valid_tails]
    if uc_tail.empty:
        return []
    r_c = uc_tail.rank(axis=1)
    r_t = ut_tail.rank(axis=1)
    ic = r_c.corrwith(r_t, axis=1).dropna()
    return cast(list[float], ic.tolist())


def _regime_consistency_ok_fast(is_sub: pd.DataFrame, col: str, ic_series: pd.Series) -> bool:
    """Check if IC performance is consistent across different market regimes.

    Args:
        is_sub: In-sample panel data.
        col: Component column name.
        ic_series: Time series of cross-sectional IC.

    Returns:
        True if performance is consistent.

    """
    if "__regime" not in is_sub.columns:
        return True
    regime_s = is_sub.groupby("datetime")["__regime"].first()
    df = pd.DataFrame({"ic": ic_series, "regime": regime_s}).dropna()
    regs = np.sort(np.unique(df["regime"].to_numpy()))
    if len(regs) < 2:
        return True
    mus: list[float] = []
    for r in regs:
        arr = df.loc[df["regime"] == r, "ic"].to_numpy(dtype=np.float64)
        if arr.size >= 5:
            mus.append(float(np.mean(arr)))
    if len(mus) < 2:
        return True
    best = max(mus)
    worst = min(mus)
    all_pos = all(m > 0.0 for m in mus)
    alt = worst > -0.5 * max(best, 1e-9)
    return bool(all_pos or alt)


def _symbol_ic_balance_ok(
    is_sub: pd.DataFrame, col: str, *, max_ratio: float, min_per_symbol: int = 40
) -> bool:
    """Penalize alpha driven by a single symbol.

    Follows institutional IC dispersion guidelines.

    Args:
        is_sub: In-sample panel data.
        col: Component column name.
        max_ratio: Maximum allowed dispersion ratio.
        min_per_symbol: Minimum bars per symbol for inclusion.

    Returns:
        True if the symbol balance is acceptable.

    """
    per: list[float] = []
    for _, g in is_sub.groupby(level="symbol", sort=False):
        if len(g) < min_per_symbol:
            continue
        v = float(g[col].corr(g["target"], method="spearman"))
        if math.isfinite(v):
            per.append(v)
    arr = np.asarray(per, dtype=np.float64)
    if arr.size < 3:
        return True
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    ratio = s / (abs(m) + 1e-9)
    return bool(ratio <= max_ratio)


def filter_alpha_components(
    alpha_wide: pd.DataFrame,
    panel_df: pd.DataFrame,
    *,
    is_end_date: str | None,
    n_trials: int = 15,
    fdr_q: float = 0.10,
    alpha_cols: Sequence[str] | None = None,
    use_newey_west: bool = False,
    use_ewma_ic_stat: bool = False,
    ewma_half_life: float = 540.0,
    symbol_balance_max: float = 3.0,
    require_regime_gate: bool = True,
    step3_regime_alpha_enabled: bool = False,
    step3_chop_support_min: float = 0.25,
    step3_chop_ic_min: float = -0.01,
    step3_chop_weight_mult: float = 0.50,
    step3_weight_mult_floor: float = 0.20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Zero-out GP alpha columns that fail various statistical and practical gates.

    Expects alpha_wide indexed like panel_df (MultiIndex datetime, symbol).

    Args:
        alpha_wide: Wide-form component dataframe.
        panel_df: Input panel dataframe containing target.
        is_end_date: Cutoff date for In-Sample data.
        n_trials: Number of trials for DSR calculation.
        fdr_q: FDR threshold for Benjamini-Hochberg.
        alpha_cols: Subset of columns to filter.
        use_newey_west: Whether to use HAC SE.
        use_ewma_ic_stat: Whether to use EWMA stats for reported t-stat.
        ewma_half_life: EWMA decay half-life.
        symbol_balance_max: Maximum allowed symbol IC dispersion.
        require_regime_gate: Whether to enforce consistency across regimes.

    Returns:
        Tuple of (filtered dataframe, metadata dictionary).

    """
    if use_ewma_ic_stat:
        _logger.debug("use_ewma_ic_stat=True: EWMA mean used for reported t-stat only.")
    if alpha_wide.empty or panel_df.empty or "target" not in panel_df.columns:
        return alpha_wide, {"n_surviving": 0.0, "neutralize_primary": 0.0}

    # --- [Fix] Unified Index Alignment (Standardize names and timezone) ---
    def _standardize_idx(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.MultiIndex):
            return df
        idx = df.index
        # Identify datetime and symbol levels by name (case-insensitive)
        d_idx = -1
        s_idx = -1
        for i, name in enumerate(idx.names):
            if name and name.lower() in ("datetime", "time", "timestamp"):
                d_idx = i
            elif name and name.lower() in ("symbol", "ticker", "asset"):
                s_idx = i
        
        if d_idx == -1 or s_idx == -1:
            return df
            
        # Standardize datetime to UTC aware
        d_vals = idx.get_level_values(d_idx)
        if getattr(d_vals, "tz", None) is None:
            d_vals = pd.to_datetime(d_vals, utc=True)
        else:
            d_vals = d_vals.tz_convert("UTC")
            
        # Reconstruct MultiIndex with standardized names ['datetime', 'symbol']
        # We only keep these two levels to ensure intersection success
        standard_idx = pd.MultiIndex.from_arrays(
            [d_vals, idx.get_level_values(s_idx)],
            names=["datetime", "symbol"]
        )
        # Handle potential duplicates after standardization (unlikely but safe)
        if standard_idx.duplicated().any():
            _logger.warning("Standardized index contains duplicates; dropping.")
            temp_df = df.copy()
            temp_df.index = standard_idx
            return temp_df[~temp_df.index.duplicated(keep='first')]
        
        new_df = df.copy()
        new_df.index = standard_idx
        return new_df

    # Standardize both inputs before any logic
    alpha_wide = _standardize_idx(alpha_wide)
    panel_df = _standardize_idx(panel_df)
    # -----------------------------------------------------------------------

    _logger.debug("filter_alpha_components | alpha_wide index: %s, panel_df index: %s", 
                  alpha_wide.index.names, panel_df.index.names)
    _logger.debug("filter_alpha_components | alpha_wide TZ: %s, panel_df TZ: %s",
                  getattr(alpha_wide.index.get_level_values(0), "tz", "None"),
                  getattr(panel_df.index.get_level_values(0), "tz", "None"))

    # Ensure unique columns in input
    if alpha_wide.columns.duplicated().any():
        _logger.warning("Duplicate columns detected in alpha_wide; dropping duplicates.")
        alpha_wide = alpha_wide.loc[:, ~alpha_wide.columns.duplicated()].copy()

    cols = (
        list(alpha_cols)
        if alpha_cols is not None
        else [
            c
            for c in alpha_wide.columns
            if (c.startswith("alpha_long_") and c[-2:].isdigit())
            or (c.startswith("alpha_short_") and c[-2:].isdigit())
            or c == "alpha_long"
            or c == "alpha_short"
        ]
    )
    if not cols:
        return alpha_wide, {"n_surviving": 0.0, "neutralize_primary": 0.0}

    times = panel_df.index.get_level_values("datetime")
    if is_end_date:
        cut = pd.to_datetime(is_end_date, utc=True)
        # [Fix] Since we standardized to UTC aware above, we can compare directly
        is_ix = np.asarray(times < cut, dtype=bool)
    else:
        is_ix = np.ones(len(panel_df), dtype=bool)

    common = alpha_wide.index.intersection(panel_df.index)
    _logger.debug("filter_alpha_components | Intersection size: %d / %d (alpha) / %d (panel)", 
                 len(common), len(alpha_wide), len(panel_df))
    if len(common) < 50:
        _logger.warning("Index intersection too small: %d rows (alpha=%d, panel=%d)", 
                        len(common), len(alpha_wide), len(panel_df))
        return alpha_wide, {"n_surviving": float(len(cols)), "neutralize_primary": 0.0}

    base = panel_df.loc[common, ["target"]].copy()
    base["__is"] = is_ix[panel_df.index.get_indexer(common)]
    _logger.debug("filter_alpha_components | IS rows: %d, OOS rows: %d", 
                 base["__is"].sum(), len(base) - base["__is"].sum())
    for pcol in ("hmm_prob_bear_trend", "hmm_prob_chop", "hmm_prob_crisis"):
        if pcol in panel_df.columns:
            base[pcol] = panel_df.loc[common, pcol].to_numpy(dtype=np.float64)
    if "regime_pre_hmm" in panel_df.columns:
        base["__regime"] = panel_df.loc[common, "regime_pre_hmm"].to_numpy()
    else:
        base["__regime"] = np.zeros(len(base), dtype=np.int64)
    for c in cols:
        base[c] = alpha_wide.loc[common, c]

    pvals: list[float] = []
    dsr_ok: list[bool] = []
    half_life_ok: list[bool] = []
    tail_ok: list[bool] = []
    oos_ok: list[bool] = []
    short_ok: list[bool] = []
    lso_ok: list[bool] = []  # [G-ALPHA v9.0] Leave-Symbols-Out validation gate
    regime_consistency_ok: list[bool] = []
    step3_chop_ok: list[bool] = []
    step3_weight_mults: list[float] = []
    sym_bal_ok: list[bool] = []
    half_life_diag_codes: list[str] = []
    neutralize_primary = False

    # alpha_long_00 diagnostic metrics
    primary_diagnostic: dict[str, float] = {}
    ic_by_slot: dict[str, float] = {}
    ic_bear_by_slot: dict[str, float] = {}
    ic_chop_by_slot: dict[str, float] = {}
    ic_pos_ratio_by_slot: dict[str, float] = {}
    tail_ic_by_slot: dict[str, float] = {}
    chop_support_by_slot: dict[str, float] = {}
    lso_ic_by_slot: dict[str, float] = {}  # [G-ALPHA v9.0] LSO IC per slot

    def _lso_ic_check(
        u_c: pd.DataFrame,
        u_tgt: pd.DataFrame,
        holdout_frac: float = 0.20,
        seed: int = 42,
    ) -> float:
        """Leave-Symbols-Out IC: symbol의 20%를 holdout하고 held-out set에서 IC 계산.

        Args:
            u_c: wide format alpha DataFrame (rows=datetime, cols=symbols).
            u_tgt: wide format target DataFrame (rows=datetime, cols=symbols).
            holdout_frac: holdout symbol 비율 (default 0.20).
            seed: random seed.

        Returns:
            held-out symbol 집합에서 계산된 mean IC.

        """
        symbols = list(u_c.columns)
        if len(symbols) < 5:
            return 0.0
        n_hold = max(1, int(len(symbols) * holdout_frac))
        rng = np.random.default_rng(seed)
        hold_syms = list(rng.choice(symbols, size=n_hold, replace=False))
        u_hold = u_c[hold_syms]
        t_hold = u_tgt[hold_syms]
        valid = u_hold.notna() & t_hold.notna()
        counts = valid.sum(axis=1)
        rows = counts >= 2
        if not rows.any():
            return 0.0
        r_c = u_hold[rows].rank(axis=1)
        r_t = t_hold[rows].rank(axis=1)
        ics = r_c.corrwith(r_t, axis=1).dropna()
        return float(ics.mean()) if len(ics) > 0 else 0.0
    
    is_sub = base[base["__is"]]
    # [Fix] Use true OOS (dates >= is_end_date) instead of IS tail 20%
    oos_time_set: set[pd.Timestamp] = set()
    oos_rows_times = sorted(
        base[~base["__is"].astype(bool)].index.get_level_values("datetime").unique()
    )
    if len(oos_rows_times) >= 10:
        oos_time_set = set(oos_rows_times)
    else:
        # Fallback: last 20% of IS as pseudo-OOS
        uniq_times = sorted(is_sub.index.get_level_values("datetime").unique())
        if len(uniq_times) >= 10:
            oos_time_set = set(uniq_times[int(len(uniq_times) * 0.8):])

    def _append_failed() -> None:
        pvals.append(1.0)
        dsr_ok.append(False)
        half_life_ok.append(False)
        tail_ok.append(False)
        oos_ok.append(False)
        short_ok.append(False)
        lso_ok.append(False)  # [G-ALPHA v9.0]
        regime_consistency_ok.append(False)
        step3_chop_ok.append(False)
        step3_weight_mults.append(1.0)
        sym_bal_ok.append(False)
        half_life_diag_codes.append("insufficient_ic_samples")

    u_tgt = is_sub["target"].unstack(level="symbol")
    u_tgt_short = 1.0 - u_tgt
    u_tgt_oos = None
    u_tgt_short_oos = None
    r_tgt = u_tgt.rank(axis=1)
    r_tgt_short = u_tgt_short.rank(axis=1)
    r_t_oos = None
    r_t_short_oos = None

    if oos_time_set:
        # [Fix] oos_sub from full base (includes true OOS rows), not is_sub
        oos_sub = base[base.index.get_level_values("datetime").isin(oos_time_set)]
        u_tgt_oos = oos_sub["target"].unstack(level="symbol")
        u_tgt_short_oos = 1.0 - u_tgt_oos
        r_t_oos = u_tgt_oos.rank(axis=1)
        r_t_short_oos = u_tgt_short_oos.rank(axis=1)

    # [Optimization] Unstack all alpha columns at once
    u_alphas = is_sub[cols].unstack(level="symbol")
    u_alphas_oos = (
        oos_sub[cols].unstack(level="symbol") if oos_time_set and not oos_sub.empty else None
    )
    # [Optimization] u_tgt already computed above (line ~387); reuse to avoid double unstack
    valid_bal_syms = [s for s in u_tgt.columns if u_tgt[s].notna().sum() >= 40]
    u_tgt_ranks = {s: u_tgt[s].rank() for s in valid_bal_syms}
    dt_probs: pd.DataFrame | None = None
    hmm_prob_cols = ["hmm_prob_bear_trend", "hmm_prob_chop", "hmm_prob_crisis"]
    if all(c in is_sub.columns for c in hmm_prob_cols):
        try:
            dt_probs = is_sub.groupby("datetime")[hmm_prob_cols].mean()
        except Exception:
            dt_probs = None

    # --- [Optimization ①] 벡터화 IC 사전 계산: pandas corrwith 루프 제거 ---
    # r_tgt numpy 변환 (IS 전체)
    r_tgt_arr = r_tgt.values.astype(np.float64)
    rt_m = np.nanmean(r_tgt_arr, axis=1, keepdims=True)
    rt_c_centered = r_tgt_arr - rt_m
    r_tgt_short_arr = r_tgt_short.values.astype(np.float64)
    rt_short_m = np.nanmean(r_tgt_short_arr, axis=1, keepdims=True)
    rt_short_c_centered = r_tgt_short_arr - rt_short_m

    def _vec_ic_series(u_col: pd.DataFrame, r_tgt_centered: np.ndarray) -> np.ndarray:
        """행별 Spearman IC를 numpy로 벡터화 계산 (pandas corrwith 대체).

        Args:
            u_col: unstack된 alpha 컬럼 DataFrame (T, n_syms).
            r_tgt_centered: 이미 center된 target rank array (T, n_syms).

        Returns:
            IC array shape (T,).

        """
        rc_arr = u_col.rank(axis=1).values.astype(np.float64)
        rc_m = np.nanmean(rc_arr, axis=1, keepdims=True)
        rc_c = rc_arr - rc_m
        num = np.nansum(rc_c * r_tgt_centered, axis=1)
        denom = np.sqrt(
            np.nansum(rc_c**2, axis=1) * np.nansum(r_tgt_centered**2, axis=1)
        ) + 1e-12
        return num / denom  # shape (T,)

    # OOS 분리 계산
    rt_oos_centered: np.ndarray | None = None
    rt_short_oos_centered: np.ndarray | None = None
    if oos_time_set and r_t_oos is not None:
        rt_oos_arr = r_t_oos.values.astype(np.float64)
        rt_oos_m = np.nanmean(rt_oos_arr, axis=1, keepdims=True)
        rt_oos_centered = rt_oos_arr - rt_oos_m
    if oos_time_set and r_t_short_oos is not None:
        rt_short_oos_arr = r_t_short_oos.values.astype(np.float64)
        rt_short_oos_m = np.nanmean(rt_short_oos_arr, axis=1, keepdims=True)
        rt_short_oos_centered = rt_short_oos_arr - rt_short_oos_m

    # 슬롯별 IC 사전 계산
    _precomp_ic: dict[str, np.ndarray] = {}
    _precomp_ic_oos: dict[str, np.ndarray] = {}
    for c in cols:
        is_short_slot = bool(_SHORT_SLOT_COL_RE.match(c))
        tgt_centered = rt_short_c_centered if is_short_slot else rt_c_centered
        _precomp_ic[c] = _vec_ic_series(u_alphas[c], tgt_centered)
        if u_alphas_oos is not None and c in u_alphas_oos.columns:
            oos_tgt_centered = rt_short_oos_centered if is_short_slot else rt_oos_centered
            if oos_tgt_centered is not None:
                _precomp_ic_oos[c] = _vec_ic_series(u_alphas_oos[c], oos_tgt_centered)
    # ------------------------------------------------------------------

    hl_by_col: dict[str, float] = {}  # Fix 1: per-column half-life tracking
    ic_oos_by_slot: dict[str, float] = {}  # Fix 2: per-column OOS CS-IC tracking

    for i, c in enumerate(cols):
        is_short_slot = bool(_SHORT_SLOT_COL_RE.match(c))
        tgt_is = u_tgt_short if is_short_slot else u_tgt
        tgt_oos = u_tgt_short_oos if is_short_slot else u_tgt_oos
        rank_tgt_oos = r_t_short_oos if is_short_slot else r_t_oos
        u_c = u_alphas[c]
        valid_mask = u_c.notna() & tgt_is.notna()
        counts = valid_mask.sum(axis=1)
        # [Optimization ①] 사전 계산된 numpy IC 배열 재사용
        ic_series = pd.Series(_precomp_ic[c], index=u_c.index)
        ic_arr = ic_series[counts >= 3].dropna().to_numpy(dtype=np.float64)

        n_ic = int(ic_arr.size)
        if n_ic < 10:
            hl_by_col[c] = 0.0  # Fix 1: record zero for failed components
            ic_oos_by_slot[c] = 0.0  # Fix 2: record zero for failed components
            _append_failed()
            continue

        mu = float(np.mean(ic_arr))
        sd = float(np.std(ic_arr, ddof=1))

        mu_for_t = mu
        if use_ewma_ic_stat:
            mu_for_t, _sd_ew = _ewma_ic_stats(ic_arr, half_life=ewma_half_life)
            _ = _sd_ew
        if use_newey_west:
            se = _newey_west_se(ic_arr)
            t_stat = mu_for_t / (se + 1e-12) if math.isfinite(se) and se > 0 else 0.0
        else:
            t_stat = mu_for_t / (sd / math.sqrt(n_ic) + 1e-12) if sd > 1e-12 else 0.0
        pvals.append(float(2.0 * min(_norm_sf(abs(t_stat)), 1.0 - 1e-15)))
        # [Optimization #10] Store IC for ensemble reuse
        ic_by_slot[c] = mu
        sharpe = mu / (sd + 1e-12) * math.sqrt(float(max(n_ic, 1)))
        # [Optimization ②] scipy skew/kurtosis: pandas Series 생성 오버헤드 제거
        sk = float(_scipy_skew(ic_arr)) if n_ic > 2 else 0.0
        ku = float(_scipy_kurtosis(ic_arr, fisher=True)) if n_ic > 3 else 0.0
        dsr_gate = _deflated_sharpe_threshold(sharpe, n_ic, sk, ku, n_trials)
        dsr_ok.append(dsr_gate)

        # [Fast-Track] Very high T-Stat bypasses noise/balance gates
        is_fast_track = bool(t_stat > 15.0)
        is_robust = bool(t_stat > 8.0)

        hl, hl_diag = _ic_half_life_bars_with_diag(ic_arr)
        hl_by_col[c] = hl  # Fix 1: store per-column half-life
        half_life_diag_codes.append(hl_diag)

        # [G-ALPHA v9.0] Replace AR(1) half-life gate with robust 3-gate majority vote
        # Gate A: ICIR (annualized Information Ratio) >= 1.3
        _icir = mu / (sd + 1e-12) * math.sqrt(float(n_ic))
        _icir_ok = bool(_icir >= 1.3 or n_ic < 50)

        # Gate B: Positive-bar ratio >= 45% (IC > 0 비율)
        _pos_ratio = float(np.mean(ic_arr > 0.0)) if n_ic > 0 else 0.0
        _pos_ratio_ok = bool(_pos_ratio >= 0.45 or n_ic < 30)

        # Gate C: Sub-period sign consistency — 3등분 각 구간 mean > -0.005 중 2개 이상
        _third = max(1, n_ic // 3)
        _sub_means = [float(np.mean(ic_arr[k * _third:(k + 1) * _third])) for k in range(3)]
        _subperiod_ok = bool(sum(m > -0.005 for m in _sub_means) >= 2 or n_ic < 30)

        # Majority vote: 3개 중 2개 이상 통과
        half_life_ok.append(bool(sum([_icir_ok, _pos_ratio_ok, _subperiod_ok]) >= 2))

        tails = _tail_decile_ic_series_fast(u_c, tgt_is, min_symbols=8)
        tail_mu = float(np.mean(tails)) if tails else mu
        tail_ic_by_slot[c] = tail_mu
        ic_pos_ratio_by_slot[c] = float(np.mean(ic_arr > 0.0)) if n_ic > 0 else 0.0
        # G-ALPHA v8.0: Tail IC must be >= 0.005
        tail_ok.append(bool(tail_mu >= 0.005 or len(tails) < 5))

        # [G-ALPHA v9.0] Leave-Symbols-Out IC gate: full IC 대비 70% 이상 보존 확인
        lso_ic = _lso_ic_check(u_c, tgt_is)
        lso_ic_by_slot[c] = lso_ic
        # n_ic < 50이면 gate 면제 (샘플 부족 시 LSO 신뢰도 낮음)
        lso_ok.append(bool(lso_ic >= 0.70 * abs(mu) or n_ic < 50))

        # Step3 regime-conditional utility diagnostics:
        # use datetime-level IC + market posterior context to measure CHOP/BEAR fragility.
        ic_bear = mu
        ic_chop = mu
        chop_support = 0.0
        if dt_probs is not None:
            ic_df = pd.DataFrame({"ic": ic_series}).join(dt_probs, how="left").dropna()
            if not ic_df.empty:
                bear_mask = (
                    (ic_df["hmm_prob_bear_trend"] >= ic_df["hmm_prob_chop"])
                    & (ic_df["hmm_prob_bear_trend"] >= ic_df["hmm_prob_crisis"])
                    & (ic_df["hmm_prob_bear_trend"] >= 0.40)
                )
                chop_mask = (
                    (ic_df["hmm_prob_chop"] >= ic_df["hmm_prob_bear_trend"])
                    & (ic_df["hmm_prob_chop"] >= ic_df["hmm_prob_crisis"])
                    & (ic_df["hmm_prob_chop"] >= 0.40)
                )
                if bool(bear_mask.any()):
                    ic_bear = float(ic_df.loc[bear_mask, "ic"].mean())
                if bool(chop_mask.any()):
                    ic_chop = float(ic_df.loc[chop_mask, "ic"].mean())
                chop_support = float(chop_mask.mean())
        ic_bear_by_slot[c] = float(ic_bear)
        ic_chop_by_slot[c] = float(ic_chop)
        chop_support_by_slot[c] = float(chop_support)

        if oos_time_set and tgt_oos is not None and rank_tgt_oos is not None and u_alphas_oos is not None:
            u_c_oos = u_alphas_oos[c]
            v_mask_oos = u_c_oos.notna() & tgt_oos.notna()
            c_oos = v_mask_oos.sum(axis=1)
            # [Optimization ①-OOS] 사전 계산된 OOS IC 배열 재사용
            if c in _precomp_ic_oos:
                ic_oos_series = pd.Series(_precomp_ic_oos[c], index=u_c_oos.index)
            else:
                r_c_oos = u_c_oos.rank(axis=1)
                ic_oos_series = r_c_oos.corrwith(rank_tgt_oos, axis=1)
            oos_arr = ic_oos_series[c_oos >= 3].dropna().to_numpy(dtype=np.float64)
            mu_oos = float(np.mean(oos_arr)) if oos_arr.size > 0 else mu
        else:
            mu_oos = mu
        ic_oos_by_slot[c] = mu_oos  # Fix 2: store per-column OOS CS-IC

        if i < 3:
            _logger.debug("  [COL %s] mu_oos=%.4f, ic_bear=%.4f, ic_chop=%.4f", c, mu_oos, ic_bear, ic_chop)

        # G-ALPHA v8.0: Hard OOS Floor >= 0.015. No blending with IS.
        if mu > 1e-6:
            oos_gate = bool(mu_oos >= 0.015)
        else:
            oos_gate = True
        oos_ok.append(oos_gate)

        # Short-side gate must use actual short prediction vs short target IC.
        short_side_ic = mu_oos if is_short_slot else mu_oos
        ic_short_ok = bool(short_side_ic >= 0.015) if is_short_slot else True
        short_ok.append(ic_short_ok)

        primary_col_name = "alpha_long_00" if "alpha_long_00" in cols else ("alpha_long" if "alpha_long" in cols else (cols[0] if cols else None))
        if c == primary_col_name:
            # G-ALPHA v8.0: Retention >= 50% (Decay < 50%)
            if mu > 1e-6 and mu_oos < 0.50 * mu:
                neutralize_primary = True

            # Diagnostic for alpha_long_00
            primary_diagnostic["is_mu"] = mu
            primary_diagnostic["oos_mu"] = mu_oos
            primary_diagnostic["half_life"] = hl
            primary_diagnostic["sharpe"] = sharpe
            primary_diagnostic["t_stat"] = t_stat

        if require_regime_gate:
            regime_consistency_ok.append(_regime_consistency_ok_fast(is_sub, c, ic_series))
        else:
            regime_consistency_ok.append(True)

        if step3_regime_alpha_enabled:
            chop_fragile = bool(chop_support >= step3_chop_support_min and ic_chop < step3_chop_ic_min)
            step3_chop_ok.append(not chop_fragile)
            if chop_fragile:
                mult = max(step3_weight_mult_floor, min(1.0, step3_chop_weight_mult))
                step3_weight_mults.append(float(mult))
            else:
                step3_weight_mults.append(1.0)
        else:
            step3_chop_ok.append(True)
            step3_weight_mults.append(1.0)

        # [Optimization] Symbol balance check using pre-calculated ranks
        bal_ratio = 0.0
        per: list[float] = []
        for s in valid_bal_syms:
            s_c = u_c[s].rank()
            tgt_rank = u_tgt_ranks[s] if not is_short_slot else (1.0 - u_tgt[s]).rank()
            v = float(s_c.corr(tgt_rank))
            if math.isfinite(v):
                per.append(v)
        arr_bal = np.asarray(per, dtype=np.float64)

        # G-ALPHA v8.0: No fast-track bypass for symbol balance.
        if arr_bal.size >= 3:
            m_bal = float(np.mean(arr_bal))
            s_bal = float(np.std(arr_bal, ddof=1))
            bal_ratio = s_bal / (abs(m_bal) + 1e-9)
        sym_bal_ok.append(bool(bal_ratio <= symbol_balance_max))

    reject = _benjamini_hochberg_reject(pvals, fdr_q)
    mag_cols = sorted(c for c in alpha_wide.columns if c.startswith("mag_long_"))
    output_cols = list(
        dict.fromkeys(
            cols
            + mag_cols
            + [c for c in ("alpha_long_raw", "alpha_short_raw") if c in alpha_wide.columns]
        )
    )
    out = alpha_wide[output_cols].copy()
    n_surv = 0

    # 상세 진단을 위한 카운터
    f_fdr, f_dsr, f_hl, f_tail, f_oos, f_short, f_lso, f_reg_consistency, f_bal, f_step3_chop = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ic_weight_by_slot: dict[str, float] = {}
    survived_cols: list[str] = []
    survived_long_cols: list[str] = []
    survived_short_cols: list[str] = []
    gate_fail_reasons_by_col: dict[str, list[str]] = {}
    gate_status_by_col: dict[str, dict[str, bool | str | float]] = {}
    half_life_diag_code_by_col: dict[str, str] = {}

    for i, c in enumerate(cols):
        # 개별 필터 결과 기록
        is_fdr_ok = bool(reject[i])
        is_dsr_ok = dsr_ok[i]
        is_hl_ok = half_life_ok[i]
        is_tail_ok = tail_ok[i]
        is_oos_ok = oos_ok[i]
        is_short_ok = short_ok[i]
        is_lso_ok = lso_ok[i]  # [G-ALPHA v9.0]
        is_reg_ok = regime_consistency_ok[i]
        is_step3_chop_ok = step3_chop_ok[i]
        is_bal_ok = sym_bal_ok[i]
        hl_diag_code = half_life_diag_codes[i] if i < len(half_life_diag_codes) else "unknown"
        half_life_diag_code_by_col[c] = hl_diag_code

        ok = (
            is_fdr_ok
            and is_dsr_ok
            and is_hl_ok
            and is_tail_ok
            and is_oos_ok
            and is_short_ok
            and is_lso_ok  # [G-ALPHA v9.0] LSO IC gate
            and is_reg_ok
            and is_step3_chop_ok
            and is_bal_ok
        )
        ic_weight_by_slot[c] = float(max(0.0, ic_by_slot.get(c, 0.0)) * step3_weight_mults[i])

        if ok:
            n_surv += 1
            survived_cols.append(c)
            if _LONG_SLOT_COL_RE.match(c):
                survived_long_cols.append(c)
            elif _SHORT_SLOT_COL_RE.match(c):
                survived_short_cols.append(c)
            gate_fail_reasons_by_col[c] = []
        else:
            # [Audit Fix] Do not neutralize here so audit report can show raw ICs.
            # Miner.py will handle filtering using survived_cols.
            # out[c] = 0.5
            
            # 탈락 원인 집계
            if not is_fdr_ok: f_fdr += 1
            if not is_dsr_ok: f_dsr += 1
            if not is_hl_ok: f_hl += 1
            if not is_tail_ok: f_tail += 1
            if not is_oos_ok: f_oos += 1
            if not is_short_ok: f_short += 1
            if not is_lso_ok: f_lso += 1  # [G-ALPHA v9.0]
            if not is_reg_ok: f_reg_consistency += 1
            if not is_step3_chop_ok: f_step3_chop += 1
            if not is_bal_ok: f_bal += 1
            reasons: list[str] = []
            if not is_fdr_ok:
                reasons.append("fdr_fail")
            if not is_dsr_ok:
                reasons.append("dsr_fail")
            if not is_hl_ok:
                reasons.append("half_life_fail")
            if not is_tail_ok:
                reasons.append("tail_fail")
            if not is_oos_ok:
                reasons.append("oos_fail")
            if not is_short_ok:
                reasons.append("short_gate_fail")
            if not is_lso_ok:
                reasons.append("lso_fail")  # [G-ALPHA v9.0]
            if not is_reg_ok:
                reasons.append("regime_consistency_fail")
            if not is_step3_chop_ok:
                reasons.append("step3_chop_fail")
            if not is_bal_ok:
                reasons.append("symbol_balance_fail")
            gate_fail_reasons_by_col[c] = reasons

        gate_status_by_col[c] = {
            "fdr_ok": is_fdr_ok,
            "dsr_ok": is_dsr_ok,
            "half_life_ok": is_hl_ok,
            "half_life_bars": float(hl_by_col.get(c, 0.0)),  # Fix 1: per-column half-life
            "tail_ok": is_tail_ok,
            "oos_ok": is_oos_ok,
            "short_ok": is_short_ok,
            "lso_ok": is_lso_ok,  # [G-ALPHA v9.0]
            "lso_ic": float(lso_ic_by_slot.get(c, 0.0)),  # [G-ALPHA v9.0]
            "regime_consistency_ok": is_reg_ok,
            "step3_chop_ok": is_step3_chop_ok,
            "symbol_balance_ok": is_bal_ok,
            "final_selection_ok": ok,
            "half_life_diag_code": hl_diag_code,
        }

    meta: dict[str, Any] = {
        "n_surviving": float(n_surv),
        "survived_cols": survived_cols,
        "survived_long_cols": survived_long_cols,
        "survived_short_cols": survived_short_cols,
        "n_surviving_long": float(len(survived_long_cols)),
        "n_surviving_short": float(len(survived_short_cols)),
        "n_components": float(len(cols)),
        "neutralize_primary": 1.0 if neutralize_primary else 0.0,
        "fail_fdr": float(f_fdr),
        "fail_dsr": float(f_dsr),
        "fail_half_life": float(f_hl),
        "fail_tail": float(f_tail),
        "fail_oos": float(f_oos),
        "fail_short": float(f_short),
        "fail_lso": float(f_lso),  # [G-ALPHA v9.0] LSO gate 탈락 수
        "fail_regime_consistency": float(f_reg_consistency),
        # Backward-compat alias for existing dashboards expecting `fail_regime`.
        "fail_regime": float(f_reg_consistency),
        "fail_step3_chop": float(f_step3_chop),
        "fail_sym_bal": float(f_bal),
        "step3_regime_alpha_enabled": 1.0 if step3_regime_alpha_enabled else 0.0,
        "step3_chop_support_min": float(step3_chop_support_min),
        "step3_chop_ic_min": float(step3_chop_ic_min),
        "step3_chop_weight_mult": float(step3_chop_weight_mult),
        "ic_by_slot": ic_by_slot,
        "ic_oos_by_slot": ic_oos_by_slot,  # Fix 2: OOS CS-IC per slot for dashboard
        "lso_ic_by_slot": lso_ic_by_slot,  # [G-ALPHA v9.0] LSO IC per slot
        "ic_weight_by_slot": ic_weight_by_slot,
        "ic_bear_by_slot": ic_bear_by_slot,
        "ic_chop_by_slot": ic_chop_by_slot,
        "ic_pos_ratio_by_slot": ic_pos_ratio_by_slot,
        "tail_ic_by_slot": tail_ic_by_slot,
        "chop_support_by_slot": chop_support_by_slot,
        "gate_fail_reasons_by_col": gate_fail_reasons_by_col,
        "gate_status_by_col": gate_status_by_col,
        "half_life_diag_code_by_col": half_life_diag_code_by_col,
        "n_final_selected": float(len(survived_cols)),
        "final_selection_fail": float(max(0, len(cols) - len(survived_cols))),
    }

    # alpha_long_00(Primary)의 상세 지표를 meta에 병합
    for k_diag, v_diag in primary_diagnostic.items():
        meta[f"primary_{k_diag}"] = float(v_diag)

    # Direction-aware head diagnostics (minimal gate for long/short raw heads).
    def _direction_head_stats(
        col_name: str,
        target_arr: np.ndarray,
        prefix: str,
    ) -> None:
        if col_name not in base.columns:
            return

        is_mask = base["__is"].to_numpy(dtype=bool)
        pred_all = base[col_name].to_numpy(dtype=np.float64)
        if target_arr.shape[0] != pred_all.shape[0]:
            return

        def _calc_mu(mask: np.ndarray) -> float:
            n_samples = int(mask.sum())
            if n_samples < 20:
                return 0.0
            sub = base.loc[mask]
            u_pred = sub[col_name].unstack(level="symbol")
            u_tgt = pd.Series(target_arr[mask], index=sub.index).unstack(level="symbol")
            
            # [Fix] Mask to focus on the 'head' (top half) of the signal for the given direction.
            # In a [0, 1] rank space, > 0.5 represents the active half.
            u_pred_head = u_pred.where(u_pred > 0.5)
            
            valid = u_pred_head.notna() & u_tgt.notna()
            cnt = valid.sum(axis=1)
            ic_s = u_pred_head.rank(axis=1).corrwith(u_tgt.rank(axis=1), axis=1)
            arr = ic_s[cnt >= 3].dropna().to_numpy(dtype=np.float64)
            mu = float(np.mean(arr)) if arr.size > 0 else 0.0
            _logger.debug(" [_calc_mu] %s | samples=%d | valid_bars=%d | mu=%.4f", prefix, n_samples, int(arr.size), mu)
            return mu

        mu_is = _calc_mu(is_mask)
        mu_oos = 0.0
        if oos_time_set:
            oos_mask = is_mask & base.index.get_level_values("datetime").isin(oos_time_set)
            mu_oos = _calc_mu(oos_mask)
        else:
            mu_oos = mu_is

        # Minimal stable gate: both IS/OOS means should be non-negative.
        head_pass = bool(mu_is >= 0.0 and mu_oos >= 0.0)
        meta[f"{prefix}_is_ic_mean"] = float(mu_is)
        meta[f"{prefix}_oos_ic_mean"] = float(mu_oos)
        meta[f"{prefix}_pass"] = 1.0 if head_pass else 0.0
        meta[f"neutralize_{prefix}"] = 0.0 if head_pass else 1.0

    target_long = base["target"].to_numpy(dtype=np.float64)
    target_short = 1.0 - target_long
    
    # [Fix] Fallback to alpha_long_00 or alpha_long if directional raw heads are missing.
    primary_col = "alpha_long_00" if "alpha_long_00" in base.columns else ("alpha_long" if "alpha_long" in base.columns else (cols[0] if cols else None))
    
    if "alpha_long_raw" in base.columns:
        _direction_head_stats("alpha_long_raw", target_long, "long_head")
    elif primary_col:
        _direction_head_stats(primary_col, target_long, "long_head")
        
    if "alpha_short_raw" in base.columns:
        _direction_head_stats("alpha_short_raw", target_short, "short_head")
    elif primary_col:
        # Create temporary flipped column for short-side IC calculation (1 - alpha vs 1 - target)
        base["__short_proxy"] = 1.0 - base[primary_col]
        _direction_head_stats("__short_proxy", target_short, "short_head")
        base.drop(columns=["__short_proxy"], inplace=True)

    _logger.info("ML alpha FDR+DSR+IC gates: %d / %d columns survive.", n_surv, len(cols))
    return out, meta
