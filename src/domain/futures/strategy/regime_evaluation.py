from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from scipy.stats import kruskal, spearmanr

_logger = logging.getLogger(__name__)
_EPS = 1e-12
_REGIME_NAMES: tuple[str, ...] = (
    "bull_quiet",
    "bull_volatile",
    "bear_quiet",
    "bear_volatile",
    "transition",
    "crash",
)
_N_REGIMES = len(_REGIME_NAMES)


@dataclass(slots=True, frozen=True)
class RegimeScoreCard:
    # C2 — Persistence (weight 0.15)
    c2_dwell_median: float
    c2_transition_rate: float
    c2_entropy_rate: float
    c2_score: float
    # C3 — Distinctness (weight 0.25, binding for allocation)
    c3_kw_pvalue: float
    c3_has_sign_flip: bool
    c3_mutual_info: float
    c3_score: float
    # C3 추가 기준: magnitude separation (sign_flip 대체 완화 조건)
    c3_magnitude_separation: float  # max|group_mean| / pooled_std
    # C4 — OOS Stability (weight 0.20)
    c4_spearman_rho: float
    c4_n_regimes_evaluated: int
    c4_score: float
    # C5 — Coverage (weight 0.10)
    c5_min_occupancy: float
    c5_max_occupancy: float
    c5_effective_regimes: float
    c5_score: float
    # C_overlay — 연속 overlay IC (spearmanr(overlay_mult, risk_adj_edge), NaN if not provided)
    c_overlay_ic: float
    # Weighted partial score covering C2-C5 (C6-C8 require manual audit)
    weighted_c2_to_c5: float
    regime_names: tuple[str, ...]
    occupancy_by_regime: tuple[float, ...]
    # C2 macro-level: direction persistence (bull/bear/crash aggregation, transition excluded from switching)
    c2_macro_dwell_median: float = 0.0
    c2_macro_transition_rate: float = 1.0
    # Pre-signal proxies (bar-level market returns, not event-level strategy edges)
    c3_proxy_pvalue: float = float("nan")
    c3_proxy_sign_flip: bool = False
    c4_proxy_spearman_rho: float = float("nan")


# ---------------------------------------------------------------------------
# C2 — Persistence
# ---------------------------------------------------------------------------

# Macro-class mapping: {0,1}→0(bull), {2,3}→1(bear), {4}→2(transition), {5}→3(crash)
_MACRO_CLASS = np.array([0, 0, 1, 1, 2, 3], dtype=np.int8)
_N_MACRO = 4


def _compute_c2_macro(code_1d: NDArray[np.int8]) -> tuple[float, float]:
    """Return (macro_dwell_median, macro_transition_rate) for direction-level persistence.

    Maps 6-state micro codes to 4 macro classes:
        bull(0,1) -> 0, bear(2,3) -> 1, transition(4) -> 2, crash(5) -> 3.

    This measures how long the *directional* regime (bull/bear/crash) persists,
    which is the operationally relevant metric for strategy allocation decisions.
    The transition state is included as its own class rather than being ignored,
    preserving information while correctly attributing vol-regime flips to the
    quiet/volatile distinction rather than direction changes.

    Args:
        code_1d: Bar-level 6-state regime code sequence [T].

    Returns:
        Tuple of (macro_dwell_median, macro_transition_rate).

    Time complexity: O(T). Space complexity: O(T).
    """
    n = code_1d.shape[0]
    if n < 2:
        return 0.0, 1.0

    macro = np.where(
        (code_1d >= 0) & (code_1d < len(_MACRO_CLASS)),
        _MACRO_CLASS[np.clip(code_1d, 0, len(_MACRO_CLASS) - 1)],
        np.int8(2),  # unknown → transition class
    )

    transitions = int(np.sum(macro[1:] != macro[:-1]))
    macro_tr = float(transitions) / float(n - 1)

    dwell: list[int] = []
    run = 1
    for i in range(1, n):
        if int(macro[i]) == int(macro[i - 1]):
            run += 1
        else:
            dwell.append(run)
            run = 1
    dwell.append(run)
    macro_dwell = float(np.median(np.asarray(dwell, dtype=np.float64)))
    return macro_dwell, macro_tr


def _compute_c2(code_1d: NDArray[np.int8]) -> tuple[float, float, float, float, float, float]:
    """Return (dwell_median, transition_rate, entropy_rate, score/10, macro_dwell_median, macro_transition_rate).

    Score is computed from macro-level dwell (direction persistence) rather than
    micro-level 6-state dwell. The threshold ≥6 is unchanged; only the measurement
    target is corrected from micro to macro.
    """
    n_bars = code_1d.shape[0]
    if n_bars < 2:
        return 0.0, 1.0, 0.0, 0.0, 0.0, 1.0

    transitions = int(np.sum(code_1d[1:] != code_1d[:-1]))
    transition_rate = float(transitions) / float(n_bars - 1)

    dwell: list[int] = []
    run = 1
    for i in range(1, n_bars):
        if int(code_1d[i]) == int(code_1d[i - 1]):
            run += 1
        else:
            dwell.append(run)
            run = 1
    dwell.append(run)
    dwell_median = float(np.median(np.asarray(dwell, dtype=np.float64)))

    # Transition probability matrix → approximate entropy rate
    n = _N_REGIMES
    trans_counts = np.zeros((n, n), dtype=np.float64)
    for i in range(n_bars - 1):
        a, b = int(code_1d[i]), int(code_1d[i + 1])
        if 0 <= a < n and 0 <= b < n:
            trans_counts[a, b] += 1.0
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_probs = np.where(row_sums > 0, trans_counts / np.maximum(row_sums, _EPS), 0.0)

    state_counts = np.zeros(n, dtype=np.float64)
    for c in code_1d:
        if 0 <= int(c) < n:
            state_counts[int(c)] += 1.0
    pi = state_counts / np.maximum(state_counts.sum(), _EPS)
    safe_log = np.where(trans_probs > _EPS, np.log(trans_probs + _EPS), 0.0)
    entropy_rate = float(-np.sum(pi[:, None] * trans_probs * safe_log))

    macro_dwell, macro_tr = _compute_c2_macro(code_1d)

    # Score based on MACRO-level dwell (direction persistence — operationally relevant threshold)
    # Threshold ≥6 unchanged; measurement target corrected from micro to macro.
    score = 0.0
    if macro_dwell >= 8.0:
        score += 6.0
    elif macro_dwell >= 6.0:
        score += 5.0
    elif macro_dwell >= 4.0:
        score += 3.0
    elif macro_dwell >= 2.0:
        score += 1.5

    if transition_rate <= 0.10:
        score += 4.0
    elif transition_rate <= 0.15:
        score += 3.0
    elif transition_rate <= 0.25:
        score += 1.5

    return dwell_median, transition_rate, entropy_rate, min(10.0, score), macro_dwell, macro_tr


# ---------------------------------------------------------------------------
# C3 — Distinctness
# ---------------------------------------------------------------------------

def _compute_c3(
    event_codes: NDArray[np.int8],
    event_edges_bps: NDArray[np.float64],
    min_n_per_group: int,
) -> tuple[float, bool, float, float, float]:
    """Return (kw_pvalue, has_sign_flip, mutual_info, score/10, magnitude_sep).

    Args:
        event_codes: Regime code at entry for each event [N].
        event_edges_bps: Realised forward edge in bps per event [N].
        min_n_per_group: Minimum observations per regime group.

    Returns:
        Tuple of (kw_pvalue, has_sign_flip, mutual_info, score, magnitude_separation).
        magnitude_separation = max|group_mean| / pooled_std — quantifies effect size
        independent of sign-flip, enabling the C3 binding condition to be relaxed for
        risk-overlay evaluation (direction-flip not required for magnitude modulation).

    Time complexity: O(N·R) where R=_N_REGIMES. Space complexity: O(N).
    """
    finite_mask = np.isfinite(event_edges_bps)
    groups: list[NDArray[np.float64]] = []
    group_means: list[float] = []

    for code in range(_N_REGIMES):
        mask = finite_mask & (event_codes == code)
        if int(mask.sum()) < min_n_per_group:
            continue
        g = event_edges_bps[mask]
        groups.append(g)
        group_means.append(float(np.mean(g)))

    if len(groups) < 2:
        return 1.0, False, 0.0, 0.0, 0.0

    try:
        _, kw_pvalue = kruskal(*groups)
        kw_pvalue = float(kw_pvalue)
    except Exception:
        kw_pvalue = 1.0

    has_sign_flip = any(
        group_means[i] * group_means[j] < 0.0
        for i in range(len(group_means))
        for j in range(i + 1, len(group_means))
    )

    # magnitude separation: max(|group_mean|) / pooled_std
    pooled_vals = np.concatenate(groups)
    pooled_std = float(np.std(pooled_vals, ddof=0)) if pooled_vals.size > 1 else _EPS
    magnitude_sep = max(abs(m) for m in group_means) / max(pooled_std, _EPS)

    try:
        from sklearn.metrics import mutual_info_score

        code_f = event_codes[finite_mask].astype(np.int32)
        edge_f = event_edges_bps[finite_mask]
        n_bins = 5
        bin_edges = np.unique(np.quantile(edge_f, np.linspace(0, 1, n_bins + 1)))
        edge_binned = (
            np.searchsorted(bin_edges[1:-1], edge_f).astype(np.int32)
            if len(bin_edges) >= 2
            else np.zeros(code_f.shape[0], dtype=np.int32)
        )
        mi = float(mutual_info_score(code_f, edge_binned))
    except Exception:
        mi = 0.0

    # C3 scoring: has_sign_flip OR magnitude_sep >= 1.5 qualifies as directional
    directional = has_sign_flip or (magnitude_sep >= 1.5)
    if kw_pvalue < 0.01 and directional:
        score = 9.0
    elif kw_pvalue < 0.05 and directional:
        score = 7.0
    elif kw_pvalue < 0.05:
        score = 5.0
    elif kw_pvalue < 0.10:
        score = 3.0
    else:
        score = 1.0
    if mi > 0.02:
        score = min(10.0, score + 1.0)

    return kw_pvalue, has_sign_flip, mi, score, magnitude_sep


# ---------------------------------------------------------------------------
# C4 — OOS Stability
# ---------------------------------------------------------------------------

def _sharpe_by_regime(
    codes: NDArray[np.int8],
    edges_bps: NDArray[np.float64],
    mask: NDArray[np.bool_],
    min_n: int,
) -> dict[int, float]:
    result: dict[int, float] = {}
    valid = mask & np.isfinite(edges_bps)
    for code in range(_N_REGIMES):
        rm = valid & (codes == code)
        n = int(rm.sum())
        if n < min_n:
            continue
        vals = edges_bps[rm]
        mu = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        result[code] = mu / max(std, _EPS)
    return result


def _compute_c4(
    event_codes: NDArray[np.int8],
    event_edges_bps: NDArray[np.float64],
    is_event_mask: NDArray[np.bool_],
    oos_event_mask: NDArray[np.bool_],
    min_n_per_regime: int,
) -> tuple[float, int, float]:
    """Return (spearman_rho, n_common_regimes, score/10)."""
    is_sharpes = _sharpe_by_regime(event_codes, event_edges_bps, is_event_mask, min_n_per_regime)
    oos_sharpes = _sharpe_by_regime(event_codes, event_edges_bps, oos_event_mask, min_n_per_regime)
    common = sorted(set(is_sharpes) & set(oos_sharpes))
    if len(common) < 2:
        return 0.0, len(common), 2.0

    is_vals = np.asarray([is_sharpes[c] for c in common], dtype=np.float64)
    oos_vals = np.asarray([oos_sharpes[c] for c in common], dtype=np.float64)
    try:
        rho_result = spearmanr(is_vals, oos_vals)
        rho = float(rho_result.statistic) if np.isfinite(rho_result.statistic) else 0.0
    except Exception:
        rho = 0.0

    if rho >= 0.7:
        score = 9.0
    elif rho >= 0.5:
        score = 7.0
    elif rho >= 0.0:
        score = 4.0
    else:
        score = 1.0
    return rho, len(common), score


# ---------------------------------------------------------------------------
# C5 — Coverage
# ---------------------------------------------------------------------------

def _compute_c5(code_1d: NDArray[np.int8]) -> tuple[float, float, float, float]:
    """Return (min_occupancy, max_occupancy, n_effective_regimes, score/10)."""
    n_bars = code_1d.shape[0]
    if n_bars == 0:
        return 0.0, 0.0, 0.0, 0.0

    counts = np.zeros(_N_REGIMES, dtype=np.float64)
    for c in code_1d:
        idx = int(c)
        if 0 <= idx < _N_REGIMES:
            counts[idx] += 1.0

    occ = counts / float(n_bars)
    active_occ = occ[counts > 0]
    min_occ = float(active_occ.min()) if active_occ.size > 0 else 0.0
    max_occ = float(occ.max())

    h = -float(np.sum(active_occ * np.log(active_occ + _EPS)))
    n_eff = math.exp(h)

    n_active = int(np.sum(counts > 0))
    if n_active == 0:
        return min_occ, max_occ, n_eff, 0.0
    n_pass = int(np.sum((active_occ >= 0.05) & (active_occ <= 0.60)))
    score = (n_pass / n_active) * 8.0
    if max_occ <= 0.60 and min_occ >= 0.05:
        score = min(10.0, score + 2.0)
    return min_occ, max_occ, n_eff, score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_regime_classifier(
    *,
    all_codes_1d: NDArray[np.int8],
    event_codes: NDArray[np.int8],
    event_edges_bps: NDArray[np.float64],
    is_event_mask: NDArray[np.bool_],
    oos_event_mask: NDArray[np.bool_],
    regime_names: tuple[str, ...] = _REGIME_NAMES,
    min_n_per_group: int = 10,
    overlay_mult_at_entry: NDArray[np.float64] | None = None,
) -> RegimeScoreCard:
    """Evaluate regime classifier quality on C2-C5 axes plus continuous overlay IC.

    Args:
        all_codes_1d: Bar-level regime code sequence [T]. Used for C2 (persistence)
            and C5 (coverage).
        event_codes: Regime code at entry for each event [N]. Must be causal:
            ``event_codes[i] = all_codes_1d[entry_idx[i] - 1]``.
        event_edges_bps: Realised forward edge in bps per event [N].
        is_event_mask: Boolean mask marking in-sample events [N].
        oos_event_mask: Boolean mask marking out-of-sample events [N].
        regime_names: Ordered regime name tuple aligned with code integers.
        min_n_per_group: Minimum observations per regime for KW/Sharpe tests.
        overlay_mult_at_entry: Continuous overlay multiplier at entry for each event [N].
            When provided, Spearman IC between overlay_mult and realised edge is
            computed and stored in ``c_overlay_ic``. NaN when not provided.

    Returns:
        RegimeScoreCard with per-axis scores and sub-metrics.
    """
    all_c = np.asarray(all_codes_1d, dtype=np.int8)
    ev_c = np.asarray(event_codes, dtype=np.int8)
    ev_e = np.asarray(event_edges_bps, dtype=np.float64)
    is_m = np.asarray(is_event_mask, dtype=bool)
    oos_m = np.asarray(oos_event_mask, dtype=bool)

    c2_dwell, c2_tr, c2_ent, c2_score, c2_macro_dwell, c2_macro_tr = _compute_c2(all_c)
    c3_pval, c3_flip, c3_mi, c3_score, magnitude_sep = _compute_c3(ev_c, ev_e, min_n_per_group)
    c4_rho, c4_n, c4_score = _compute_c4(ev_c, ev_e, is_m, oos_m, min_n_per_group)
    c5_min, c5_max, c5_neff, c5_score = _compute_c5(all_c)

    # C_overlay — continuous overlay IC
    c_overlay_ic = float("nan")
    if overlay_mult_at_entry is not None:
        ov = np.asarray(overlay_mult_at_entry, dtype=np.float64)
        valid = np.isfinite(ov) & np.isfinite(ev_e)
        if int(valid.sum()) >= min_n_per_group:
            try:
                from scipy.stats import spearmanr as _spearmanr

                rho_res = _spearmanr(ov[valid], ev_e[valid])
                c_overlay_ic = (
                    float(rho_res.statistic)
                    if np.isfinite(rho_res.statistic)
                    else float("nan")
                )
            except Exception:
                _logger.debug("overlay IC spearmanr failed", exc_info=True)

    # Occupancy per regime
    n_bars = all_c.shape[0]
    counts = np.zeros(_N_REGIMES, dtype=np.float64)
    for c in all_c:
        idx = int(c)
        if 0 <= idx < _N_REGIMES:
            counts[idx] += 1.0
    occ = counts / max(float(n_bars), 1.0)

    # Partial weighted score: C2(0.15) + C3(0.25) + C4(0.20) + C5(0.10) = 0.70 of total
    weighted = (
        0.15 * (c2_score / 10.0)
        + 0.25 * (c3_score / 10.0)
        + 0.20 * (c4_score / 10.0)
        + 0.10 * (c5_score / 10.0)
    )

    return RegimeScoreCard(
        c2_dwell_median=c2_dwell,
        c2_transition_rate=c2_tr,
        c2_entropy_rate=c2_ent,
        c2_score=c2_score,
        c2_macro_dwell_median=c2_macro_dwell,
        c2_macro_transition_rate=c2_macro_tr,
        c3_kw_pvalue=c3_pval,
        c3_has_sign_flip=c3_flip,
        c3_mutual_info=c3_mi,
        c3_score=c3_score,
        c3_magnitude_separation=magnitude_sep,
        c4_spearman_rho=c4_rho,
        c4_n_regimes_evaluated=c4_n,
        c4_score=c4_score,
        c5_min_occupancy=c5_min,
        c5_max_occupancy=c5_max,
        c5_effective_regimes=c5_neff,
        c5_score=c5_score,
        c_overlay_ic=c_overlay_ic,
        weighted_c2_to_c5=weighted,
        regime_names=tuple(regime_names),
        occupancy_by_regime=tuple(float(x) for x in occ),
    )


def evaluate_regime_classifier_proxy(
    *,
    all_codes_1d: NDArray[np.int8],
    market_returns_1d: NDArray[np.float64],
    is_bar_mask: NDArray[np.bool_],
    oos_bar_mask: NDArray[np.bool_],
    regime_names: tuple[str, ...] = _REGIME_NAMES,
    min_n_per_group: int = 10,
) -> tuple[float, bool, float, float]:
    """Compute pre-signal proxy for C3/C4 using bar-level market returns.

    Args:
        all_codes_1d: Bar-level regime code [T].
        market_returns_1d: Bar-level market log-returns scaled to bps [T].
        is_bar_mask: Boolean mask for in-sample bars [T].
        oos_bar_mask: Boolean mask for out-of-sample bars [T].
        regime_names: Regime name tuple.
        min_n_per_group: Minimum bars per regime group.

    Returns:
        Tuple of (c3_proxy_pvalue, c3_proxy_sign_flip, c4_proxy_rho, c3_proxy_score).

    Time complexity: O(T·R) where R=_N_REGIMES. Space complexity: O(T).
    """
    del regime_names  # used externally for display only
    all_c = np.asarray(all_codes_1d, dtype=np.int8)
    mkt = np.asarray(market_returns_1d, dtype=np.float64)
    is_m = np.asarray(is_bar_mask, dtype=bool)
    oos_m = np.asarray(oos_bar_mask, dtype=bool)

    c3_pval, c3_flip, _mi, c3_score, _mag = _compute_c3(all_c, mkt, min_n_per_group)
    c4_rho, _n, _c4_score = _compute_c4(all_c, mkt, is_m, oos_m, min_n_per_group)
    return c3_pval, c3_flip, c4_rho, c3_score


# ---------------------------------------------------------------------------
# P0 — Regime Lift Proof Gate
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class RegimeLiftProofResult:
    """Regime conditioning이 pooled를 이기는지 검증한 결과.

    Attributes:
        proof_passed: True if all gate conditions are satisfied.
        conditioning_path: "regime_conditioned" | "pooled_fallback".
        mean_lift_bps: Mean lift (regime_cond - pooled) in bps.
        n_eff: Effective sample size after AR(1) autocorrelation adjustment.
        nw_tstat: Newey-West HAC t-statistic for mean lift.
        deflated_sharpe: Bailey-LdP DSR corrected for multiple testing.
        fold_pass_ratio: Fraction of folds where fold-level NW tstat passes.
        n_folds_evaluated: Number of folds with sufficient observations.
        nw_tstat_threshold: Configured NW tstat threshold.
        fold_pass_ratio_threshold: Configured fold pass ratio threshold.
    """

    proof_passed: bool
    conditioning_path: str
    mean_lift_bps: float
    n_eff: float
    nw_tstat: float
    deflated_sharpe: float
    fold_pass_ratio: float
    n_folds_evaluated: int
    nw_tstat_threshold: float
    fold_pass_ratio_threshold: float


def newey_west_tstat(
    diff: NDArray[np.float64],
    max_lag: int,
) -> tuple[float, float, float]:
    """Newey-West HAC t-statistic with Bartlett kernel.

    Time complexity: O(T * max_lag). Space complexity: O(T).

    Args:
        diff: 1-D array of demeaned differences [T].
        max_lag: Maximum lag for Bartlett kernel (typically max_holding_bars).

    Returns:
        Tuple of (nw_tstat, nw_se, n_eff).
        n_eff: Effective sample size via AR(1) autocorrelation adjustment.
    """
    arr = np.asarray(diff, dtype=np.float64)
    n = arr.shape[0]
    mu = float(np.mean(arr))
    demeaned = arr - mu

    gamma0 = float(np.dot(demeaned, demeaned)) / n
    nw_var = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_l = float(np.dot(demeaned[lag:], demeaned[:-lag])) / n
        nw_var += 2.0 * weight * gamma_l

    nw_var = max(nw_var, _EPS)
    nw_se = math.sqrt(nw_var / n)
    nw_tstat_val = mu / nw_se

    # N_eff via AR(1) autocorrelation estimate
    if n >= 3:
        rho_raw = float(np.corrcoef(demeaned[1:], demeaned[:-1])[0, 1])
        rho = 0.0 if math.isnan(rho_raw) else float(np.clip(rho_raw, -0.999, 0.999))
    else:
        rho = 0.0
    n_eff = float(n) * (1.0 - rho) / (1.0 + rho)

    return nw_tstat_val, nw_se, n_eff


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    n_trials: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Corrects for multiple testing over n_trials regime cells.
    Time complexity: O(1). Space complexity: O(1).

    Args:
        sr: Observed annualised-equivalent Sharpe ratio of lift series.
        n_obs: Effective number of independent observations.
        n_trials: Number of strategy variants / regime cells tested.
        skewness: Sample skewness of the returns series.
        kurtosis: Excess kurtosis of the returns series.

    Returns:
        Deflated Sharpe Ratio (DSR). Negative → reject at conventional level.
    """
    gamma_euler = 0.5772156649
    e = math.e

    z1 = scipy_stats.norm.ppf(1.0 - 1.0 / max(n_trials, 2))
    z2 = scipy_stats.norm.ppf(1.0 - 1.0 / (max(n_trials, 2) * e))

    e_max = (1.0 - gamma_euler) * z1 + gamma_euler * z2
    v_max_raw = z1**2 - z2**2
    v_max = max(v_max_raw * (1.0 - gamma_euler) ** 2 / 2.0, 0.0)

    n_safe = max(n_obs, 4)
    # Bailey-Lopez de Prado (2014) SR estimation variance:
    #   var(SR) = (1 - g3*SR + (g4-1)/4*SR^2) / (n-1),  g4 = non-excess kurtosis.
    # scipy.stats.kurtosis returns EXCESS kurtosis (k_ex = g4 - 3), so
    #   (g4 - 1)/4 = (k_ex + 2)/4.
    sr_var = (
        1.0
        - skewness * sr
        + 0.25 * (kurtosis + 2.0) * sr**2
    ) / (n_safe - 1.0)
    sr_std = math.sqrt(max(sr_var, _EPS))

    dsr = (sr - e_max) / max(math.sqrt(v_max + sr_std**2), _EPS)
    return float(dsr)


def evaluate_regime_lift_proof(
    *,
    regime_cond_edges: NDArray[np.float64],
    pooled_edges: NDArray[np.float64],
    realized_edges: NDArray[np.float64],
    fold_ids: NDArray[np.int32],
    n_regime_cells: int = 60,
    nw_tstat_threshold: float = 1.5,
    fold_pass_ratio_threshold: float = 0.60,
    max_holding_bars: int = 6,
    proof_enabled: bool = True,
) -> RegimeLiftProofResult:
    """Evaluate whether regime conditioning beats pooled baseline.

    Computes NW HAC t-stat, Bailey-LdP DSR, and per-fold pass ratio
    to determine if regime-conditioned edges are statistically significant.

    Time complexity: O(T * max_holding_bars + F * T/F * max_holding_bars)
        = O(T * max_holding_bars). Space complexity: O(T).

    Args:
        regime_cond_edges: Per-event predicted edge under regime conditioning [N] (bps).
        pooled_edges: Per-event predicted edge under pooled baseline [N] (bps).
        realized_edges: Per-event realized edge used to score each prediction [N] (bps).
        fold_ids: Integer fold assignment per event [N].
        n_regime_cells: Number of regime cells tested (for DSR correction).
        nw_tstat_threshold: Minimum NW t-stat to pass proof.
        fold_pass_ratio_threshold: Minimum fraction of folds that must pass.
        max_holding_bars: Bartlett kernel lag for NW estimator.
        proof_enabled: If False, bypass gate and return proof_passed=True.

    Returns:
        RegimeLiftProofResult with all diagnostic fields populated.
    """
    rc = np.asarray(regime_cond_edges, dtype=np.float64)
    pe = np.asarray(pooled_edges, dtype=np.float64)
    re = np.asarray(realized_edges, dtype=np.float64)
    fi = np.asarray(fold_ids, dtype=np.int32)

    if rc.shape != pe.shape or rc.shape != re.shape or rc.shape != fi.shape:
        raise ValueError("regime_cond_edges, pooled_edges, realized_edges, and fold_ids must align")

    regime_realized = np.sign(rc) * re
    pooled_realized = np.sign(pe) * re
    lift = regime_realized - pooled_realized
    finite_mask = np.isfinite(lift) & np.isfinite(rc) & np.isfinite(pe) & np.isfinite(re)
    lift_finite = lift[finite_mask]

    if lift_finite.size == 0:
        return RegimeLiftProofResult(
            proof_passed=not proof_enabled,
            conditioning_path="regime_conditioned" if not proof_enabled else "pooled_fallback",
            mean_lift_bps=0.0,
            n_eff=0.0,
            nw_tstat=0.0,
            deflated_sharpe=0.0,
            fold_pass_ratio=0.0,
            n_folds_evaluated=0,
            nw_tstat_threshold=nw_tstat_threshold,
            fold_pass_ratio_threshold=fold_pass_ratio_threshold,
        )

    mean_lift = float(np.mean(lift_finite)) if lift_finite.size > 0 else 0.0
    std_lift = float(np.std(lift_finite, ddof=1)) if lift_finite.size > 1 else _EPS
    sr_lift = mean_lift / max(std_lift, _EPS)
    skew_lift = float(scipy_stats.skew(lift_finite)) if lift_finite.size >= 3 else 0.0
    kurt_lift = (
        float(scipy_stats.kurtosis(lift_finite)) if lift_finite.size >= 4 else 0.0
    )

    nw_tstat_val, _nw_se, n_eff = newey_west_tstat(lift_finite, max_holding_bars)
    dsr = deflated_sharpe_ratio(
        sr_lift, int(max(n_eff, 4)), n_regime_cells, skew_lift, kurt_lift
    )

    # Per-fold NW tstat pass count
    unique_folds = np.unique(fi[finite_mask])
    fold_pass_count = 0
    n_folds_evaluated = 0
    fi_finite = fi[finite_mask]
    for fold_id in unique_folds:
        fold_lift = lift_finite[fi_finite == fold_id]
        if fold_lift.shape[0] < 4:
            continue
        n_folds_evaluated += 1
        fold_t, _, _ = newey_west_tstat(fold_lift, max_holding_bars)
        if fold_t >= nw_tstat_threshold:
            fold_pass_count += 1

    fold_pass_ratio = (
        fold_pass_count / n_folds_evaluated if n_folds_evaluated > 0 else 0.0
    )

    if proof_enabled:
        proof_passed = (
            nw_tstat_val >= nw_tstat_threshold
            and dsr >= 0.0
            and fold_pass_ratio >= fold_pass_ratio_threshold
        )
    else:
        proof_passed = True

    conditioning_path = "regime_conditioned" if proof_passed else "pooled_fallback"

    _logger.debug(
        "RegimeLiftProof: passed=%s nw_t=%.3f dsr=%.3f fold_ratio=%.2f/%d folds",
        proof_passed,
        nw_tstat_val,
        dsr,
        fold_pass_ratio,
        n_folds_evaluated,
    )

    return RegimeLiftProofResult(
        proof_passed=proof_passed,
        conditioning_path=conditioning_path,
        mean_lift_bps=mean_lift,
        n_eff=n_eff,
        nw_tstat=nw_tstat_val,
        deflated_sharpe=dsr,
        fold_pass_ratio=fold_pass_ratio,
        n_folds_evaluated=n_folds_evaluated,
        nw_tstat_threshold=nw_tstat_threshold,
        fold_pass_ratio_threshold=fold_pass_ratio_threshold,
    )
