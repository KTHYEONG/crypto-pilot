"""Multiple-testing guard for GP cross-sectional alpha components (IS + OOS diagnostics)."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


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

    Follows tmp.md 4-E guidelines for IC dispersion.

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
) -> tuple[pd.DataFrame, dict[str, float]]:
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

    # Ensure unique columns in input
    if alpha_wide.columns.duplicated().any():
        _logger.warning("Duplicate columns detected in alpha_wide; dropping duplicates.")
        alpha_wide = alpha_wide.loc[:, ~alpha_wide.columns.duplicated()].copy()

    cols = (
        list(alpha_cols)
        if alpha_cols is not None
        else [c for c in alpha_wide.columns if c.startswith("gp_alpha_") and c[-2:].isdigit()]
    )
    if not cols:
        return alpha_wide, {"n_surviving": 0.0, "neutralize_primary": 0.0}

    times = panel_df.index.get_level_values("datetime")
    if is_end_date:
        cut = pd.to_datetime(is_end_date, utc=True)
        if getattr(times, "tz", None) is None:
            times_utc = pd.to_datetime(times, utc=True)
        else:
            times_utc = times.tz_convert("UTC")
        is_ix = np.asarray(times_utc < cut, dtype=bool)
    else:
        is_ix = np.ones(len(panel_df), dtype=bool)

    common = alpha_wide.index.intersection(panel_df.index)
    if len(common) < 50:
        return alpha_wide, {"n_surviving": float(len(cols)), "neutralize_primary": 0.0}

    base = panel_df.loc[common, ["target"]].copy()
    base["__is"] = is_ix[panel_df.index.get_indexer(common)]
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
    regime_ok: list[bool] = []
    sym_bal_ok: list[bool] = []
    neutralize_primary = False

    # gp_alpha_00 diagnostic metrics
    primary_diagnostic: dict[str, float] = {}

    is_sub = base[base["__is"]]
    uniq_times = sorted(is_sub.index.get_level_values("datetime").unique())
    oos_time_set: set[pd.Timestamp] = set()
    if len(uniq_times) >= 10:
        oos_time_set = set(uniq_times[int(len(uniq_times) * 0.8) :])

    def _append_failed() -> None:
        pvals.append(1.0)
        dsr_ok.append(False)
        half_life_ok.append(False)
        tail_ok.append(False)
        oos_ok.append(False)
        regime_ok.append(False)
        sym_bal_ok.append(False)

    u_tgt = is_sub["target"].unstack(level="symbol")
    u_tgt_oos = None
    if oos_time_set:
        oos_sub = is_sub[is_sub.index.get_level_values("datetime").isin(oos_time_set)]
        u_tgt_oos = oos_sub["target"].unstack(level="symbol")

    for _, c in enumerate(cols):
        u_c = is_sub[c].unstack(level="symbol")
        valid_mask = u_c.notna() & u_tgt.notna()
        counts = valid_mask.sum(axis=1)
        r_c = u_c.rank(axis=1)
        r_tgt = u_tgt.rank(axis=1)
        ic_series = r_c.corrwith(r_tgt, axis=1)
        ic_arr = ic_series[counts >= 3].dropna().to_numpy(dtype=np.float64)

        n_ic = int(ic_arr.size)
        if n_ic < 10:
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
        sharpe = mu / (sd + 1e-12) * math.sqrt(float(max(n_ic, 1)))
        sk = float(pd.Series(ic_arr).skew()) if n_ic > 2 else 0.0
        ku = float(pd.Series(ic_arr).kurt()) - 3.0 if n_ic > 3 else 0.0
        dsr_gate = _deflated_sharpe_threshold(sharpe, n_ic, sk, ku, n_trials)
        dsr_ok.append(dsr_gate)

        # [Fast-Track] Very high T-Stat bypasses noise/balance gates
        is_fast_track = bool(t_stat > 15.0)
        is_robust = bool(t_stat > 8.0)

        hl = _ic_half_life_bars(ic_arr)
        if is_fast_track:
            half_life_ok.append(True)
        else:
            # More lenient: 2.0 (for 1h) instead of 4.0; 50% relax if robust
            hl_thresh = 1.0 if is_robust else 2.0
            half_life_ok.append(bool(hl > hl_thresh or n_ic < 50))

        tails = _tail_decile_ic_series_fast(u_c, u_tgt, min_symbols=8)
        tail_mu = float(np.mean(tails)) if tails else mu
        tail_ok.append(bool(tail_mu > -0.05 or len(tails) < 5))

        if oos_time_set and u_tgt_oos is not None:
            oos_sub = is_sub[is_sub.index.get_level_values("datetime").isin(oos_time_set)]
            u_c_oos = oos_sub[c].unstack(level="symbol")
            v_mask_oos = u_c_oos.notna() & u_tgt_oos.notna()
            c_oos = v_mask_oos.sum(axis=1)
            r_c_oos = u_c_oos.rank(axis=1)
            r_t_oos = u_tgt_oos.rank(axis=1)
            ic_oos_series = r_c_oos.corrwith(r_t_oos, axis=1)
            oos_arr = ic_oos_series[c_oos >= 3].dropna().to_numpy(dtype=np.float64)
            mu_oos = float(np.mean(oos_arr)) if oos_arr.size > 0 else mu
        else:
            mu_oos = mu

        # [OOS Blend] weighted average instead of strict ratio
        blend = 0.7 * mu_oos + 0.3 * mu
        if is_fast_track:
            oos_gate = True
        elif mu > 1e-6:
            # proposed: blend > 0.015
            oos_gate = bool(blend > 0.015)
        else:
            oos_gate = True
        oos_ok.append(oos_gate)

        if c == "gp_alpha_00":
            if not is_fast_track and mu > 1e-6 and mu_oos < 0.45 * mu:
                neutralize_primary = True

            # Diagnostic for gp_alpha_00
            primary_diagnostic["is_mu"] = mu
            primary_diagnostic["oos_mu"] = mu_oos
            primary_diagnostic["half_life"] = hl
            primary_diagnostic["sharpe"] = sharpe
            primary_diagnostic["t_stat"] = t_stat

        if require_regime_gate:
            regime_ok.append(_regime_consistency_ok_fast(is_sub, c, ic_series))
        else:
            regime_ok.append(True)

        # Symbol balance check and store for diagnostic if primary
        bal_ratio = 0.0
        per: list[float] = []
        for _, g in is_sub.groupby(level="symbol", sort=False):
            if len(g) >= 40:
                v = float(g[c].corr(g["target"], method="spearman"))
                if math.isfinite(v):
                    per.append(v)
        arr_bal = np.asarray(per, dtype=np.float64)
        if arr_bal.size >= 3:
            m_bal = float(np.mean(arr_bal))
            s_bal = float(np.std(arr_bal, ddof=1))
            bal_ratio = s_bal / (abs(m_bal) + 1e-9)

        if c == "gp_alpha_00":
            primary_diagnostic["sym_dispersion"] = bal_ratio

        if is_fast_track:
            sym_bal_ok.append(True)
        else:
            sym_bal_ok.append(bool(bal_ratio <= symbol_balance_max))

    reject = _benjamini_hochberg_reject(pvals, fdr_q)
    output_cols = list(dict.fromkeys(cols + [c for c in ("gp_alpha_long_raw", "gp_alpha_short_raw") if c in alpha_wide.columns]))
    out = alpha_wide[output_cols].copy()
    n_surv = 0

    # 상세 진단을 위한 카운터
    f_fdr, f_dsr, f_hl, f_tail, f_oos, f_reg, f_bal = 0, 0, 0, 0, 0, 0, 0

    for i, c in enumerate(cols):
        # 개별 필터 결과 기록
        is_fdr_ok = bool(reject[i])
        is_dsr_ok = dsr_ok[i]
        is_hl_ok = half_life_ok[i]
        is_tail_ok = tail_ok[i]
        is_oos_ok = oos_ok[i]
        is_reg_ok = regime_ok[i]
        is_bal_ok = sym_bal_ok[i]

        ok = (
            is_fdr_ok
            and is_dsr_ok
            and is_hl_ok
            and is_tail_ok
            and is_oos_ok
            and is_reg_ok
            and is_bal_ok
        )

        if ok:
            n_surv += 1
        else:
            out[c] = 0.5
            # 탈락 원인 집계 (중복 집계 가능)
            if not is_fdr_ok:
                f_fdr += 1
            if not is_dsr_ok:
                f_dsr += 1
            if not is_hl_ok:
                f_hl += 1
            if not is_tail_ok:
                f_tail += 1
            if not is_oos_ok:
                f_oos += 1
            if not is_reg_ok:
                f_reg += 1
            if not is_bal_ok:
                f_bal += 1

    meta: dict[str, float] = {
        "n_surviving": float(n_surv),
        "n_components": float(len(cols)),
        "neutralize_primary": 1.0 if neutralize_primary else 0.0,
        "fail_fdr": float(f_fdr),
        "fail_dsr": float(f_dsr),
        "fail_half_life": float(f_hl),
        "fail_tail": float(f_tail),
        "fail_oos": float(f_oos),
        "fail_regime": float(f_reg),
        "fail_sym_bal": float(f_bal),
    }

    # gp_alpha_00(Primary)의 상세 지표를 meta에 병합
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
            if int(mask.sum()) < 20:
                return 0.0
            sub = base.loc[mask]
            u_pred = sub[col_name].unstack(level="symbol")
            u_tgt = pd.Series(target_arr[mask], index=sub.index).unstack(level="symbol")
            valid = u_pred.notna() & u_tgt.notna()
            cnt = valid.sum(axis=1)
            ic_s = u_pred.rank(axis=1).corrwith(u_tgt.rank(axis=1), axis=1)
            arr = ic_s[cnt >= 3].dropna().to_numpy(dtype=np.float64)
            return float(np.mean(arr)) if arr.size > 0 else 0.0

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
    _direction_head_stats("gp_alpha_long_raw", target_long, "long_head")
    _direction_head_stats("gp_alpha_short_raw", target_short, "short_head")

    _logger.info("GP alpha FDR+DSR+IC gates: %d / %d columns survive.", n_surv, len(cols))
    return out, meta
