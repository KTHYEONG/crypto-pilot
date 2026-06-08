from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
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
    # C4 — OOS Stability (weight 0.20)
    c4_spearman_rho: float
    c4_n_regimes_evaluated: int
    c4_score: float
    # C5 — Coverage (weight 0.10)
    c5_min_occupancy: float
    c5_max_occupancy: float
    c5_effective_regimes: float
    c5_score: float
    # Weighted partial score covering C2-C5 (C6-C8 require manual audit)
    weighted_c2_to_c5: float
    regime_names: tuple[str, ...]
    occupancy_by_regime: tuple[float, ...]


# ---------------------------------------------------------------------------
# C2 — Persistence
# ---------------------------------------------------------------------------

def _compute_c2(code_1d: NDArray[np.int8]) -> tuple[float, float, float, float]:
    """Return (dwell_median, transition_rate, entropy_rate, score/10)."""
    n_bars = code_1d.shape[0]
    if n_bars < 2:
        return 0.0, 1.0, 0.0, 0.0

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

    # Score: dwell (0-6) + transition_rate (0-4)
    score = 0.0
    if dwell_median >= 8.0:
        score += 6.0
    elif dwell_median >= 6.0:
        score += 5.0
    elif dwell_median >= 4.0:
        score += 3.0
    elif dwell_median >= 2.0:
        score += 1.5

    if transition_rate <= 0.10:
        score += 4.0
    elif transition_rate <= 0.15:
        score += 3.0
    elif transition_rate <= 0.25:
        score += 1.5

    return dwell_median, transition_rate, entropy_rate, min(10.0, score)


# ---------------------------------------------------------------------------
# C3 — Distinctness
# ---------------------------------------------------------------------------

def _compute_c3(
    event_codes: NDArray[np.int8],
    event_edges_bps: NDArray[np.float64],
    min_n_per_group: int,
) -> tuple[float, bool, float, float]:
    """Return (kw_pvalue, has_sign_flip, mutual_info, score/10)."""
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
        return 1.0, False, 0.0, 0.0

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

    if kw_pvalue < 0.01 and has_sign_flip:
        score = 9.0
    elif kw_pvalue < 0.05 and has_sign_flip:
        score = 7.0
    elif kw_pvalue < 0.05:
        score = 5.0
    elif kw_pvalue < 0.10:
        score = 3.0
    else:
        score = 1.0
    if mi > 0.02:
        score = min(10.0, score + 1.0)

    return kw_pvalue, has_sign_flip, mi, score


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
) -> RegimeScoreCard:
    """Evaluate regime classifier quality on C2-C5 axes.

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

    Returns:
        RegimeScoreCard with per-axis scores and sub-metrics.
    """
    all_c = np.asarray(all_codes_1d, dtype=np.int8)
    ev_c = np.asarray(event_codes, dtype=np.int8)
    ev_e = np.asarray(event_edges_bps, dtype=np.float64)
    is_m = np.asarray(is_event_mask, dtype=bool)
    oos_m = np.asarray(oos_event_mask, dtype=bool)

    c2_dwell, c2_tr, c2_ent, c2_score = _compute_c2(all_c)
    c3_pval, c3_flip, c3_mi, c3_score = _compute_c3(ev_c, ev_e, min_n_per_group)
    c4_rho, c4_n, c4_score = _compute_c4(ev_c, ev_e, is_m, oos_m, min_n_per_group)
    c5_min, c5_max, c5_neff, c5_score = _compute_c5(all_c)

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
        c3_kw_pvalue=c3_pval,
        c3_has_sign_flip=c3_flip,
        c3_mutual_info=c3_mi,
        c3_score=c3_score,
        c4_spearman_rho=c4_rho,
        c4_n_regimes_evaluated=c4_n,
        c4_score=c4_score,
        c5_min_occupancy=c5_min,
        c5_max_occupancy=c5_max,
        c5_effective_regimes=c5_neff,
        c5_score=c5_score,
        weighted_c2_to_c5=weighted,
        regime_names=tuple(regime_names),
        occupancy_by_regime=tuple(float(x) for x in occ),
    )
