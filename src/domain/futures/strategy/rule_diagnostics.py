from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from src.core.settings import round_trip_cost_bps
from src.domain.futures.strategy.candidate_labels import label_candidate_events
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.market_regime import RegimeQualityReport, compute_market_regime_context

_logger = logging.getLogger(__name__)


def _newey_west_mean_tstat(values: np.ndarray, max_lag: int) -> float:
    """Return a simple Newey-West t-stat for mean(edge_after_hurdle_bps)."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    n_obs = int(finite.shape[0])
    if n_obs < 2:
        return 0.0
    centered = finite - float(np.mean(finite))
    gamma0 = float(np.dot(centered, centered) / n_obs)
    if gamma0 <= 0.0:
        return float("inf") if float(np.mean(finite)) > 0.0 else 0.0
    lag_cap = min(max(0, int(max_lag)), n_obs - 1)
    long_run_var = gamma0
    for lag in range(1, lag_cap + 1):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n_obs)
        weight = 1.0 - (lag / (lag_cap + 1.0))
        long_run_var += 2.0 * weight * cov
    if long_run_var <= 0.0:
        return 0.0
    se = float(np.sqrt(long_run_var / n_obs))
    if se <= 0.0:
        return 0.0
    return float(np.mean(finite) / se)


def _regime_cell_admission(
    rec_entry_idx: np.ndarray,
    rec_edge: np.ndarray,
    regime_code: np.ndarray,
    regime_names: list[str],
    cfg: CandidateStrategyConfig,
    *,
    arch_prior_bps: float = 0.0,
    global_prior_bps: float = 0.0,
    nw_max_lag: int = 6,
) -> dict[str, Any]:
    """Evaluate per-regime-cell edge via Bayesian posterior probability admission.

    Replaces flat obs/t-stat thresholds with a unified criterion:
      P(μ > δ | data) ≥ cfg.min_admission_posterior_prob
    under a Normal-Normal conjugate model with Newey-West corrected variance.

    τ² (prior variance) is estimated from cross-cell heterogeneity; k0 = Ω_nw/τ² is
    derived from data (James-Stein EB) rather than hard-coded.  cfg.min_regime_cell_oos_obs
    serves only as a NW variance stability floor (default 20), not a domain gate.

    Returns admitted_cells sorted by posterior mean (highest first).
    """
    empty_result: dict[str, Any] = {
        "admitted": False,
        "admitted_cells": [],
        "cell_edges": {},
        "cell_mu_post": {},
        "cell_p_admit": {},
        "tau_estimated_bps": float(cfg.admission_tau_prior_bps),
    }

    valid_idx = (rec_entry_idx >= 0) & (rec_entry_idx < regime_code.shape[0])
    if not np.any(valid_idx):
        return empty_result

    entry_idx_valid = rec_entry_idx[valid_idx]
    edge_valid = rec_edge[valid_idx]
    rec_regime_code = regime_code[entry_idx_valid]

    # Prior mean μ₀: arch preferred, then global, then pooled
    mu0: float
    if math.isfinite(arch_prior_bps) and arch_prior_bps != 0.0:
        mu0 = arch_prior_bps
    elif math.isfinite(global_prior_bps) and global_prior_bps != 0.0:
        mu0 = global_prior_bps
    else:
        mu0 = float(np.mean(edge_valid))

    # Pass 1: collect cell means for τ² estimation (data-derived prior variance)
    first_pass_means: list[float] = []
    for code in range(len(regime_names)):
        mask = rec_regime_code == code
        obs = int(np.sum(mask))
        if obs < cfg.min_regime_cell_oos_obs:
            continue
        first_pass_means.append(float(np.mean(edge_valid[mask])))

    if len(first_pass_means) >= 2:
        tau_sq = max(float(np.var(first_pass_means, ddof=1)), 1e-6)
    else:
        tau_sq = float(cfg.admission_tau_prior_bps) ** 2

    tau_bps = math.sqrt(tau_sq)
    delta = cfg.min_regime_cell_edge_bps

    # Pass 2: Bayesian posterior probability per cell
    passing: list[tuple[float, str]] = []  # (mu_post, name)
    cell_edges: dict[str, float] = {}
    cell_mu_post: dict[str, float] = {}
    cell_p_admit: dict[str, float] = {}

    for code, name in enumerate(regime_names):
        mask = rec_regime_code == code
        obs = int(np.sum(mask))
        if obs < cfg.min_regime_cell_oos_obs:  # NW stability floor
            continue
        cell_vals = edge_valid[mask]
        cell_mean_raw = float(np.mean(cell_vals))

        # Ω_nw: NW long-run variance estimate
        if cfg.admission_use_newey_west:
            t_nw = _newey_west_mean_tstat(cell_vals, max_lag=nw_max_lag)
            se_nw = abs(cell_mean_raw / (t_nw + 1e-12))
            omega_nw = max((se_nw * float(obs) ** 0.5) ** 2, 1e-6)
        else:
            cell_std = float(np.std(cell_vals, ddof=1))
            omega_nw = max(cell_std**2, 1e-6)

        # N-N conjugate posterior
        precision_data = float(obs) / omega_nw
        precision_prior = 1.0 / tau_sq
        sigma2_post = 1.0 / (precision_data + precision_prior)
        mu_post = sigma2_post * (precision_data * cell_mean_raw + precision_prior * mu0)
        sigma_post = math.sqrt(sigma2_post)

        # P(μ > δ | data) via survival function
        p = float(_norm.sf(delta, loc=mu_post, scale=sigma_post))

        cell_edges[name] = cell_mean_raw
        cell_mu_post[name] = mu_post
        cell_p_admit[name] = p

        if p >= cfg.min_admission_posterior_prob:
            passing.append((mu_post, name))

    # Retain top-N by posterior mean (not raw mean)
    passing.sort(key=lambda x: x[0], reverse=True)
    admitted_cells = [name for _, name in passing[: cfg.max_admitted_cells_per_variant]]

    return {
        "admitted": len(admitted_cells) > 0,
        "admitted_cells": admitted_cells,
        "cell_edges": cell_edges,
        "cell_mu_post": cell_mu_post,
        "cell_p_admit": cell_p_admit,
        "tau_estimated_bps": tau_bps,
    }


def log_regime_quality_report(report: RegimeQualityReport) -> None:
    """Emit a compact regime scorecard line for diagnostics."""
    reasons = ",".join(report.reasons) if report.reasons else "none"
    _logger.debug(
        "[REGIME_QUALITY] dwell=%.2f leakage_ok=%s overlay_lift_bps=%.3f "
        "overlay_lift_t=%.3f crisis_ok=%s passed=%s reasons=%s",
        report.persistence_dwell,
        report.leakage_ok,
        report.overlay_lift_bps,
        report.overlay_lift_tstat,
        report.crisis_precision_ok,
        report.passed,
        reasons,
    )


def log_regime_scorecard(scorecard: Any) -> None:
    """Emit [REGIME_SCORECARD] in table format (gate-mode independent)."""
    w = 82
    sep = "-" * w
    has_events = scorecard.c4_n_regimes_evaluated > 0

    def _c3_str() -> str:
        if not has_events:
            return "needs signal phase"
        flip = "Y" if scorecard.c3_has_sign_flip else "N"
        return f"kw_p={scorecard.c3_kw_pvalue:.4f}  flip={flip}  mi={scorecard.c3_mutual_info:.4f}"

    def _c4_str() -> str:
        if not has_events:
            return "needs signal phase"
        return f"rho={scorecard.c4_spearman_rho:.3f}  n_regimes={scorecard.c4_n_regimes_evaluated}"

    def _score(val: float, avail: bool) -> str:
        return f"{val:.1f}/10" if avail else "  n/a  "

    _logger.info("\n" + sep)
    _logger.info(f"| {'[REGIME_SCORECARD]':<{w-4}} |")
    _logger.info(sep)
    _logger.info(f"| {'Axis':<20} | {'Score':^7} | {'Key Metrics':<45} |")
    _logger.info(sep)
    c2_macro_dwell = getattr(scorecard, "c2_macro_dwell_median", float("nan"))
    c2_macro_str = f"{c2_macro_dwell:.2f}" if not __import__("math").isnan(c2_macro_dwell) else "n/a"
    c2_metrics = (
        f"dwell={scorecard.c2_dwell_median:.2f}(micro) macro={c2_macro_str} "
        f"tr={scorecard.c2_transition_rate:.3f} ent={scorecard.c2_entropy_rate:.3f}"
    )
    c5_metrics = (
        f"min={scorecard.c5_min_occupancy:.3f} max={scorecard.c5_max_occupancy:.3f} "
        f"n_eff={scorecard.c5_effective_regimes:.2f}"
    )
    _logger.info(f"| {'C2 Persistence':<20} | {_score(scorecard.c2_score, True):^7} | {c2_metrics:<45} |")
    _logger.info(f"| {'C3 Distinctness':<20} | {_score(scorecard.c3_score, has_events):^7} | {_c3_str():<45} |")
    _logger.info(f"| {'C4 OOS Stability':<20} | {_score(scorecard.c4_score, has_events):^7} | {_c4_str():<45} |")
    _logger.info(f"| {'C5 Coverage':<20} | {_score(scorecard.c5_score, True):^7} | {c5_metrics:<45} |")
    _logger.info(sep)
    _logger.info(
        f"| {'Weighted C2-C5':<20} | {scorecard.weighted_c2_to_c5:^7.3f} | {'C1(hard_gate=pass)  C6-C8(manual)':<45} |"
    )
    _logger.info(sep)
    
    # Occupancy
    occ_pairs = list(zip(scorecard.regime_names, scorecard.occupancy_by_regime, strict=False))
    def _fmt_occ(n: str, v: float) -> str:
        sn = n.replace("volatile", "vol").replace("quiet", "q").replace("transition", "trans")
        return f"{sn}={v:.3f}"
        
    occ_line1 = "  ".join(_fmt_occ(n, v) for n, v in occ_pairs[:3])
    occ_line2 = "  ".join(_fmt_occ(n, v) for n, v in occ_pairs[3:])
    
    _logger.info(f"| {'Occupancy':<20} | {'  n/a  ':^7} | {occ_line1:<45} |")
    if occ_line2:
        _logger.info(f"| {'':<20} | {'':^7} | {occ_line2:<45} |")
        
    # Proxy row
    import math as _math
    c3p = getattr(scorecard, "c3_proxy_pvalue", float("nan"))
    c4p_rho = getattr(scorecard, "c4_proxy_spearman_rho", float("nan"))
    c3p_flip = getattr(scorecard, "c3_proxy_sign_flip", False)
    if not _math.isnan(c3p):
        c4p_str = f"{c4p_rho:.3f}" if not _math.isnan(c4p_rho) else "n/a"
        proxy_metrics = f"kw_p={c3p:.3f} flip={'Y' if c3p_flip else 'N'} rho={c4p_str}"
        _logger.info(f"| {'C3/C4_proxy (mkt)':<20} | {'  n/a  ':^7} | {proxy_metrics:<45} |")
    _logger.info(sep)


@dataclass(slots=True, frozen=True)
class RuleDiagnosticsResult:
    """Rule alpha diagnostics tables."""

    by_family: pd.DataFrame
    by_variant: pd.DataFrame
    by_family_side: pd.DataFrame
    side_flip: pd.DataFrame
    decision: dict[str, float | int | str]
    recommended_keep_variants: tuple[str, ...]
    recommended_flip_variants: tuple[str, ...]
    recommended_keep_signal_cells: tuple[str, ...]
    recommended_flip_signal_cells: tuple[str, ...]
    recommendation_basis: str
    recommendation_split: tuple[int, int]
    report_split: tuple[int, int]
    recommendation_failure_report: dict[str, Any] | None = None


def _safe_spearman(signal: np.ndarray, target: np.ndarray) -> float:
    """Return Spearman IC with finite fallbacks."""
    if signal.shape[0] != target.shape[0]:
        return 0.0

    mask = np.isfinite(signal) & np.isfinite(target)
    if int(mask.sum()) < 2:
        return 0.0
    if np.unique(signal[mask]).size < 2 or np.unique(target[mask]).size < 2:
        return 0.0

    # Fallback to pandas only for the fast spearman corr call on filtered arrays
    ic = float(pd.Series(signal[mask]).corr(pd.Series(target[mask]), method="spearman"))
    return ic if np.isfinite(ic) else 0.0


def _safe_payoff_ratio(mean_mfe_bps: float, mean_mae_bps: float) -> float:
    """Return a stable favorable/adverse excursion ratio."""
    if not np.isfinite(mean_mfe_bps) or not np.isfinite(mean_mae_bps):
        return 0.0
    denom = abs(mean_mae_bps)
    if denom <= 1e-12:
        return 0.0
    return float(mean_mfe_bps / denom)


def _group_key(view: str, row: pd.Series) -> str:
    """Build a stable diagnostic key for a view/group row."""
    if view == "family":
        return f"family={row['family']}"
    if view == "variant":
        return f"variant={row['family']}:{row['variant']}"
    if view == "family_side":
        return f"family_side={row['family']}:{int(row['side'])}"
    raise ValueError(f"unsupported view: {view}")


def _candidate_action(
    *,
    n: int,
    min_obs: int,
    mean_edge_bps: float,
    pct_edge_pos: float,
    payoff_ratio: float,
    q10_shortfall_fail_rate: float,
    min_hit_rate: float,
    min_payoff_ratio: float,
    max_q10_fail_rate: float,
    flip_delta_bps: float | None = None,
    flip_mean_edge_bps: float | None = None,
    train_mean_edge_bps: float | None = None,
) -> str:
    """Classify a diagnostic group."""
    if n < min_obs:
        return "INSUFFICIENT_OBS"
    if np.isfinite(mean_edge_bps) and np.isfinite(pct_edge_pos):
        if (
            mean_edge_bps > 0.0
            and q10_shortfall_fail_rate <= max_q10_fail_rate
            and (pct_edge_pos >= min_hit_rate or payoff_ratio >= min_payoff_ratio)
        ):
            # IS→OOS consistency guard: reject only when IS was positive but OOS turned
            # negative (genuine overfitting signal). IS-negative → OOS-positive is allowed
            # — the strategy improved out-of-sample, which is a desirable regime shift.
            if (
                train_mean_edge_bps is not None
                and np.isfinite(train_mean_edge_bps)
                and train_mean_edge_bps > 0.0
                and mean_edge_bps < 0.0
            ):
                return "DROP_OR_REWORK"
            return "KEEP_CANDIDATE"
        if (
            flip_delta_bps is not None
            and flip_mean_edge_bps is not None
            and mean_edge_bps < 0.0
            and flip_delta_bps >= 25.0
            and flip_mean_edge_bps > mean_edge_bps
        ):
            return "SIDE_FLIP_CANDIDATE"
    return "DROP_OR_REWORK"


def _split_index(n_bars: int) -> int:
    """Return the fixed train/OOS split index used by candidate datasets."""
    return max(1, int(n_bars * 0.8))


def _window_mask(entry_idx: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return a half-open interval mask."""
    return (entry_idx >= start) & (entry_idx < end)


def _resolve_report_window(
    *,
    n_bars: int,
    report_start: int | None,
    report_end: int | None,
) -> tuple[int, int]:
    """Return the report window, defaulting to the legacy 80/20 OOS split."""
    if report_start is None or report_end is None:
        split_idx = _split_index(n_bars)
        return split_idx, n_bars
    return report_start, report_end


def _edge_summary_from_frame(frame: pd.DataFrame, *, cfg: CandidateStrategyConfig) -> dict[str, float]:
    """Compute edge summary metrics for a grouped frame."""
    edge = frame["edge_after_hurdle_bps"].to_numpy(copy=False)
    mae = frame["mae_bps"].to_numpy(copy=False)
    mfe = frame["mfe_bps"].to_numpy(copy=False)
    raw_score = frame["raw_score"].to_numpy(copy=False)
    score_abs = np.abs(raw_score)

    finite_edge = edge[np.isfinite(edge)]
    finite_mae = mae[np.isfinite(mae)]
    finite_mfe = mfe[np.isfinite(mfe)]
    finite_shortfall = finite_mae < -float(cfg.max_expected_shortfall_bps)

    # Phase 2 skeleton — vol-normalised percentile edge.
    # Requires `atr_bps` column in the events frame (not yet emitted by
    # rule_signals.py). When available, normalise: edge_norm = edge / atr_bps
    # using t-1 ATR to prevent look-ahead bias, then derive median/p10 from the
    # normalised distribution.  Until atr_bps is added, cfg.edge_percentile_vol_normalized
    # is a no-op; the flag is reserved for the next spec iteration.
    # TODO(Phase2): implement when atr_bps column is available in labeled events.

    return {
        "mean_edge_bps": float(np.mean(finite_edge)) if finite_edge.size > 0 else float("nan"),
        "median_edge_bps": float(np.median(finite_edge)) if finite_edge.size > 0 else float("nan"),
        "pct_edge_pos": float((finite_edge > 0.0).mean()) if finite_edge.size > 0 else 0.0,
        "p10_edge_bps": float(np.percentile(finite_edge, 10)) if finite_edge.size > 0 else float("nan"),
        "p90_edge_bps": float(np.percentile(finite_edge, 90)) if finite_edge.size > 0 else float("nan"),
        "mean_mae_bps": float(np.mean(finite_mae)) if finite_mae.size > 0 else float("nan"),
        "mean_mfe_bps": float(np.mean(finite_mfe)) if finite_mfe.size > 0 else float("nan"),
        "payoff_ratio": _safe_payoff_ratio(
            float(np.mean(finite_mfe)) if finite_mfe.size > 0 else float("nan"),
            float(np.mean(finite_mae)) if finite_mae.size > 0 else float("nan"),
        ),
        "spearman_score_edge": _safe_spearman(raw_score, edge),
        "spearman_abs_score_edge": _safe_spearman(score_abs, edge),
        "q10_shortfall_fail_rate": float(finite_shortfall.mean()) if finite_shortfall.size > 0 else 0.0,
    }


def _summarize_view(
    *,
    events: pd.DataFrame,
    view: str,
    min_obs: int,
    cfg: CandidateStrategyConfig,
    report_start: int,
    report_end: int,
) -> pd.DataFrame:
    """Summarize a view of rule events."""
    group_cols: list[str]
    if view == "family":
        group_cols = ["family"]
    elif view == "variant":
        group_cols = ["family", "variant"]
    elif view == "family_side":
        group_cols = ["family", "side"]
    else:
        raise ValueError(f"unsupported view: {view}")

    records: list[dict[str, float | int | str]] = []
    grouped = events.groupby(group_cols, sort=False, dropna=False)
    for _, group in grouped:
        row = group.iloc[0]
        key = _group_key(view, row)
        side = group["side"].to_numpy(copy=False)
        entry_idx = group["entry_idx"].to_numpy(copy=False)
        train_group = group.loc[entry_idx < report_start]
        oos_group = group.loc[_window_mask(entry_idx, report_start, report_end)]
        full_metrics = _edge_summary_from_frame(group, cfg=cfg)
        train_metrics = _edge_summary_from_frame(train_group, cfg=cfg) if not train_group.empty else {
            "mean_edge_bps": float("nan"),
            "pct_edge_pos": 0.0,
            "payoff_ratio": 0.0,
            "q10_shortfall_fail_rate": 0.0,
            "spearman_score_edge": 0.0,
        }
        oos_metrics = _edge_summary_from_frame(oos_group, cfg=cfg) if not oos_group.empty else {
            "mean_edge_bps": float("nan"),
            "pct_edge_pos": 0.0,
            "payoff_ratio": 0.0,
            "q10_shortfall_fail_rate": 0.0,
            "spearman_score_edge": 0.0,
        }

        records.append(
            {
                "group": key,
                "n": len(group),
                "long_n": int(np.sum(side > 0.0)),
                "short_n": int(np.sum(side < 0.0)),
                "train_n": len(train_group),
                "oos_n": len(oos_group),
                "train_mean_edge_bps": float(train_metrics["mean_edge_bps"]),
                "oos_mean_edge_bps": float(oos_metrics["mean_edge_bps"]),
                "train_pct_edge_pos": float(train_metrics["pct_edge_pos"]),
                "oos_pct_edge_pos": float(oos_metrics["pct_edge_pos"]),
                "train_payoff_ratio": float(train_metrics["payoff_ratio"]),
                "oos_payoff_ratio": float(oos_metrics["payoff_ratio"]),
                "train_q10_shortfall_fail_rate": float(train_metrics["q10_shortfall_fail_rate"]),
                "oos_q10_shortfall_fail_rate": float(oos_metrics["q10_shortfall_fail_rate"]),
                "train_rank_ic": float(train_metrics["spearman_score_edge"]),
                "oos_rank_ic": float(oos_metrics["spearman_score_edge"]),
                "edge_stability_bps": float(oos_metrics["mean_edge_bps"] - train_metrics["mean_edge_bps"])
                if np.isfinite(train_metrics["mean_edge_bps"]) and np.isfinite(oos_metrics["mean_edge_bps"])
                else float("nan"),
                "mean_edge_bps": float(full_metrics["mean_edge_bps"]),
                "median_edge_bps": float(full_metrics["median_edge_bps"]),
                "pct_edge_pos": float(full_metrics["pct_edge_pos"]),
                "p10_edge_bps": float(full_metrics["p10_edge_bps"]),
                "p90_edge_bps": float(full_metrics["p90_edge_bps"]),
                "mean_mae_bps": float(full_metrics["mean_mae_bps"]),
                "mean_mfe_bps": float(full_metrics["mean_mfe_bps"]),
                "payoff_ratio": float(full_metrics["payoff_ratio"]),
                "spearman_score_edge": float(full_metrics["spearman_score_edge"]),
                "spearman_abs_score_edge": float(full_metrics["spearman_abs_score_edge"]),
                "q10_shortfall_fail_rate": float(full_metrics["q10_shortfall_fail_rate"]),
                "candidate_action": _candidate_action(
                    n=len(group),
                    min_obs=min_obs,
                    mean_edge_bps=float(full_metrics["mean_edge_bps"]),
                    pct_edge_pos=float(full_metrics["pct_edge_pos"]),
                    payoff_ratio=float(full_metrics["payoff_ratio"]),
                    q10_shortfall_fail_rate=float(full_metrics["q10_shortfall_fail_rate"]),
                    min_hit_rate=cfg.min_rule_hit_rate,
                    min_payoff_ratio=cfg.min_variant_oos_payoff_ratio,
                    max_q10_fail_rate=cfg.max_variant_oos_q10_fail_rate,
                ),
            }
        )

    columns = [
        "group",
        "n",
        "long_n",
        "short_n",
        "train_n",
        "oos_n",
        "train_mean_edge_bps",
        "oos_mean_edge_bps",
        "train_pct_edge_pos",
        "oos_pct_edge_pos",
        "train_payoff_ratio",
        "oos_payoff_ratio",
        "train_q10_shortfall_fail_rate",
        "oos_q10_shortfall_fail_rate",
        "train_rank_ic",
        "oos_rank_ic",
        "edge_stability_bps",
        "mean_edge_bps",
        "median_edge_bps",
        "pct_edge_pos",
        "p10_edge_bps",
        "p90_edge_bps",
        "mean_mae_bps",
        "mean_mfe_bps",
        "payoff_ratio",
        "spearman_score_edge",
        "spearman_abs_score_edge",
        "q10_shortfall_fail_rate",
        "candidate_action",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame.from_records(records, columns=columns)
    return out.sort_values(["mean_edge_bps", "group"], ascending=[False, True]).reset_index(drop=True)


def _summarize_recommendation_variants(
    *,
    events: pd.DataFrame,
    aligned: AlignedMarketData,
    min_obs: int,
    cfg: CandidateStrategyConfig,
    recommendation_start: int,
    recommendation_end: int,
    side_flip_lookup: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> pd.DataFrame:
    """Summarize variant metrics for the recommendation window only."""
    # When promotion_level == "variant", group at family:variant granularity
    # regardless of diagnose_signal_cells to avoid obs fragmentation.
    use_signal_cells = bool(
        cfg.diagnose_signal_cells
        and "signal_cell" in events.columns
        and cfg.promotion_level == "signal_cell"
    )
    regime_ctx = compute_market_regime_context(aligned=aligned)
    regime_code = regime_ctx.code_1d
    regime_names = regime_ctx.name_by_code
    records: list[dict[str, float | int | str]] = []
    group_cols = ["signal_cell"] if use_signal_cells else ["family", "variant"]
    grouped = events.groupby(group_cols, sort=False, dropna=False)
    for _key_tuple, group in grouped:
        entry_idx = group["entry_idx"].to_numpy(copy=False)
        rec_group = group.loc[_window_mask(entry_idx, recommendation_start, recommendation_end)]
        if rec_group.empty:
            continue
        rec_metrics = _edge_summary_from_frame(rec_group, cfg=cfg)
        rec_n = int(rec_group.shape[0])
        row0 = group.iloc[0]
        family = str(row0["family"])
        variant = str(row0["variant"])
        key = str(row0["signal_cell"]) if use_signal_cells else f"{family}:{variant}"
        prefix = "cell=" if use_signal_cells else "variant="
        flip_delta, flip_mean = (None, None)
        if side_flip_lookup is not None:
            flip_delta, flip_mean = side_flip_lookup.get(f"{prefix}{key}", (None, None))
        rec_edge = rec_group["edge_after_hurdle_bps"].to_numpy(dtype=np.float64, copy=False)
        breakeven_mean_bps = float(np.mean(rec_edge)) if rec_edge.size > 0 else float("nan")
        breakeven_tstat = _newey_west_mean_tstat(
            rec_edge,
            int(max(1, float(row0.get("expected_holding_bars", 1)) - 1)),
        )
        breakeven_hard_pass = (
            rec_n >= (cfg.min_signal_cell_oos_obs if use_signal_cells else min_obs)
            and np.isfinite(breakeven_mean_bps)
            and breakeven_mean_bps >= max(8.0, float(round_trip_cost_bps()) * 0.6)
            and np.isfinite(breakeven_tstat)
            and breakeven_tstat >= cfg.min_rule_ir_t
        )
        regime_best_name = ""
        regime_best_obs = 0
        regime_best_edge_bps = float("nan")
        regime_eligible_count = 0
        regime_pass = True
        regime_cell_admitted = False
        regime_cell_admitted_cells: list[str] = []
        if cfg.regime_diagnostic_enabled:
            rec_entry_idx = rec_group["entry_idx"].to_numpy(dtype=np.int64, copy=False)
            best_edge = -np.inf
            valid_idx = (rec_entry_idx >= 0) & (rec_entry_idx < regime_code.shape[0])
            if np.any(valid_idx):
                _entry_valid = rec_entry_idx[valid_idx]
                _edge_valid = rec_edge[valid_idx]
                rec_regime_code = regime_code[_entry_valid]
                for code, name in enumerate(regime_names):
                    code_mask = rec_regime_code == code
                    obs = int(np.sum(code_mask))
                    if obs < cfg.min_regime_variant_oos_obs:
                        continue
                    regime_eligible_count += 1
                    mean_edge = float(np.mean(_edge_valid[code_mask]))
                    if mean_edge > best_edge:
                        best_edge = mean_edge
                        regime_best_name = name
                        regime_best_obs = obs
                        regime_best_edge_bps = mean_edge
            regime_pass = regime_eligible_count > 0 and regime_best_edge_bps >= cfg.min_regime_variant_oos_edge_bps
            if cfg.regime_cell_admission_enabled:
                global_prior = float(np.mean(rec_edge)) if rec_edge.size > 0 else 0.0
                cell_result = _regime_cell_admission(
                    rec_entry_idx=rec_entry_idx,
                    rec_edge=rec_edge,
                    regime_code=regime_code,
                    regime_names=list(regime_names),
                    cfg=cfg,
                    global_prior_bps=global_prior,
                )
                regime_cell_admitted = bool(cell_result["admitted"])
                regime_cell_admitted_cells = list(cell_result["admitted_cells"])
        records.append(
            {
                "group": f"{prefix}{key}",
                "family": family,
                "variant": variant,
                "signal_cell": str(row0.get("signal_cell", "")),
                "entry_regime": str(row0.get("entry_regime", "")),
                "exit_policy_id": str(row0.get("exit_policy_id", "")),
                "archetype": str(row0.get("archetype", "")),
                "n": int(group.shape[0]),
                "oos_n": rec_n,
                "oos_mean_edge_bps": float(rec_metrics["mean_edge_bps"]),
                "oos_median_edge_bps": float(rec_metrics["median_edge_bps"]),
                "oos_p10_edge_bps": float(rec_metrics["p10_edge_bps"]),
                "oos_pct_edge_pos": float(rec_metrics["pct_edge_pos"]),
                "oos_payoff_ratio": float(rec_metrics["payoff_ratio"]),
                "oos_q10_shortfall_fail_rate": float(rec_metrics["q10_shortfall_fail_rate"]),
                "event_fraction_per_bar": float(rec_n / float(max(1, recommendation_end - recommendation_start))),
                "breakeven_mean_bps": breakeven_mean_bps,
                "breakeven_tstat": breakeven_tstat,
                "breakeven_hard_pass": breakeven_hard_pass,
                "regime_eligible_count": regime_eligible_count,
                "regime_best_oos_obs": regime_best_obs,
                "regime_best_oos_edge_bps": regime_best_edge_bps,
                "regime_best_name": regime_best_name,
                "regime_pass": regime_pass,
                "regime_cell_admitted": regime_cell_admitted,
                "regime_cell_admitted_cells": ",".join(regime_cell_admitted_cells),
                "oos_rank_ic": float(rec_metrics["spearman_score_edge"]),
                "edge_stability_bps": float("nan"),
                "candidate_action": _candidate_action(
                    n=rec_n,
                    min_obs=cfg.min_signal_cell_oos_obs if use_signal_cells else min_obs,
                    mean_edge_bps=float(rec_metrics["mean_edge_bps"]),
                    pct_edge_pos=float(rec_metrics["pct_edge_pos"]),
                    payoff_ratio=float(rec_metrics["payoff_ratio"]),
                    q10_shortfall_fail_rate=float(rec_metrics["q10_shortfall_fail_rate"]),
                    min_hit_rate=cfg.min_rule_hit_rate,
                    min_payoff_ratio=cfg.min_variant_oos_payoff_ratio,
                    max_q10_fail_rate=cfg.max_variant_oos_q10_fail_rate,
                    flip_delta_bps=flip_delta,
                    flip_mean_edge_bps=flip_mean,
                ),
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "group",
                "family",
                "variant",
                "signal_cell",
                "entry_regime",
                "exit_policy_id",
                "archetype",
                "n",
                "oos_n",
                "oos_mean_edge_bps",
                "oos_median_edge_bps",
                "oos_p10_edge_bps",
                "oos_pct_edge_pos",
                "oos_payoff_ratio",
                "oos_q10_shortfall_fail_rate",
                "oos_rank_ic",
                "event_fraction_per_bar",
                "breakeven_mean_bps",
                "breakeven_tstat",
                "breakeven_hard_pass",
                "regime_eligible_count",
                "regime_best_oos_obs",
                "regime_best_oos_edge_bps",
                "regime_best_name",
                "regime_pass",
                "regime_cell_admitted",
                "regime_cell_admitted_cells",
                "edge_stability_bps",
                "candidate_action",
            ]
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["oos_mean_edge_bps", "group"], ascending=[False, True]
    ).reset_index(drop=True)


def _summarize_side_flip(
    *,
    original: pd.DataFrame,
    flipped: pd.DataFrame,
    view: str,
    min_obs: int,
    cfg: CandidateStrategyConfig,
    report_start: int,
    report_end: int,
) -> pd.DataFrame:
    """Compare original and side-flipped diagnostics for a view."""
    orig_summary = _summarize_view(
        events=original,
        view=view,
        min_obs=min_obs,
        cfg=cfg,
        report_start=report_start,
        report_end=report_end,
    )
    flip_summary = _summarize_view(
        events=flipped,
        view=view,
        min_obs=min_obs,
        cfg=cfg,
        report_start=report_start,
        report_end=report_end,
    )

    flip_map = flip_summary.set_index("group") if not flip_summary.empty else pd.DataFrame()
    records: list[dict[str, float | int | str]] = []
    for row in orig_summary.itertuples(index=False):
        flip_row: pd.Series | None
        if isinstance(flip_map, pd.DataFrame) and row.group in flip_map.index:
            flip_row = flip_map.loc[row.group]
        else:
            flip_row = None

        if flip_row is None or isinstance(flip_row, pd.DataFrame):
            flip_mean = float("nan")
            flip_pct = float("nan")
        else:
            flip_mean = float(flip_row["mean_edge_bps"])
            flip_pct = float(flip_row["pct_edge_pos"])

        delta = flip_mean - float(row.mean_edge_bps) if np.isfinite(flip_mean) else float("nan")
        action = _candidate_action(
            n=int(row.n),
            min_obs=min_obs,
            mean_edge_bps=float(row.mean_edge_bps),
            pct_edge_pos=float(row.pct_edge_pos),
            payoff_ratio=float(row.payoff_ratio),
            q10_shortfall_fail_rate=float(row.q10_shortfall_fail_rate),
            min_hit_rate=cfg.min_rule_hit_rate,
            min_payoff_ratio=cfg.min_variant_oos_payoff_ratio,
            max_q10_fail_rate=cfg.max_variant_oos_q10_fail_rate,
            flip_delta_bps=delta,
            flip_mean_edge_bps=flip_mean,
        )
        records.append(
            {
                "group": row.group,
                "n": int(row.n),
                "orig_mean_edge_bps": float(row.mean_edge_bps),
                "flip_mean_edge_bps": flip_mean,
                "delta_mean_edge_bps": delta,
                "orig_pct_edge_pos": float(row.pct_edge_pos),
                "flip_pct_edge_pos": flip_pct,
                "candidate_action": action,
            }
        )

    columns = [
        "group",
        "n",
        "orig_mean_edge_bps",
        "flip_mean_edge_bps",
        "delta_mean_edge_bps",
        "orig_pct_edge_pos",
        "flip_pct_edge_pos",
        "candidate_action",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame.from_records(records, columns=columns)
    return out.sort_values(["delta_mean_edge_bps", "group"], ascending=[False, True]).reset_index(drop=True)


def _log_summary_block(
    *,
    summary: pd.DataFrame,
    view: str,
) -> None:
    """Emit a compact diagnostic log for a summary table."""
    for row in summary.itertuples(index=False):
        if view == "family":
            family = str(row.group).removeprefix("family=")
            _logger.debug(
                "[DIAG][RULE_FAMILY] family=%s n=%d mean_edge=%.1f pct_pos=%.3f ic=%.4f action=%s",
                family,
                int(row.n),
                float(row.mean_edge_bps),
                float(row.pct_edge_pos),
                float(row.spearman_score_edge),
                str(row.candidate_action),
            )
        elif view == "variant":
            payload = str(row.group).removeprefix("variant=")
            family, variant = payload.split(":", 1)
            _logger.debug(
                "[DIAG][RULE_VARIANT] family=%s variant=%s n=%d mean_edge=%.1f pct_pos=%.3f ic=%.4f action=%s",
                family,
                variant,
                int(row.n),
                float(row.mean_edge_bps),
                float(row.pct_edge_pos),
                float(row.spearman_score_edge),
                str(row.candidate_action),
            )
        elif view == "family_side":
            payload = str(row.group).removeprefix("family_side=")
            family, side = payload.split(":", 1)
            _logger.debug(
                "[DIAG][RULE_FAMILY_SIDE] family=%s side=%s n=%d mean_edge=%.1f pct_pos=%.3f ic=%.4f action=%s",
                family,
                side,
                int(row.n),
                float(row.mean_edge_bps),
                float(row.pct_edge_pos),
                float(row.spearman_score_edge),
                str(row.candidate_action),
            )
        else:
            raise ValueError(f"unsupported view: {view}")


def _log_variant_top_block(
    summary: pd.DataFrame,
    *,
    top_k: int,
    recommended_variants: frozenset[str] | None = None,
) -> None:
    """Emit top variant diagnostics as a formatted table at INFO level."""
    if summary.empty:
        return

    columns = summary.columns
    sort_col = "oos_mean_edge_bps" if "oos_mean_edge_bps" in columns else "mean_edge_bps"
    top = summary.sort_values([sort_col, "group"], ascending=[False, True]).head(top_k)

    header = (
        f"| {'Rank':<4} | {'Strategy Name':<35} | {'Sample (OOS)':<12} | "
        f"{'Profit(bps)':>11} | {'Win Rate':>8} | {'P/L':>6} | {'Score':>6} | {'Action':<6} | {'Rec':<3} |"
    )
    width = len(header)

    title = "[CANDIDATE TOP STRATEGIES] "
    _logger.info("\n" + title + "-" * (width - len(title)))
    _logger.info(header)
    _logger.info(
        f"| {'-'*4:<4} | {'-'*35:<35} | {'-'*12:<12} | {'-'*11:>11} | "
        f"{'-'*8:>8} | {'-'*6:>6} | {'-'*6:>6} | {'-'*6:<6} | {'-'*3:<3} |"
    )

    for idx, row in enumerate(top.itertuples(index=False), start=1):
        full_key = str(row.group).removeprefix("variant=")
        # Match recommendation BEFORE truncating the display string.
        variant_key = f"variant={full_key}"
        rec = "Y" if (recommended_variants is not None and variant_key in recommended_variants) else "N"

        key = full_key if len(full_key) <= 35 else full_key[:32] + "..."

        n_total = int(row.n)
        n_oos = int(getattr(row, "oos_n", 0))
        n_str = f"{n_total} ({n_oos})"

        profit = f"{float(getattr(row, 'oos_mean_edge_bps', row.mean_edge_bps)):>11.1f}"
        win_rate = f"{float(getattr(row, 'oos_pct_edge_pos', 0.0)) * 100:>7.1f}%"
        pl_ratio = f"{float(getattr(row, 'oos_payoff_ratio', 0.0)):>6.2f}"
        score = f"{float(getattr(row, 'oos_rank_ic', row.spearman_score_edge)):>6.3f}"

        status_raw = str(row.candidate_action)
        action = "KEEP" if "KEEP" in status_raw else ("FLIP" if "FLIP" in status_raw else "DROP")

        _logger.info(
            f"| {idx:<4} | {key:<35} | {n_str:<12} | {profit} | "
            f"{win_rate} | {pl_ratio} | {score} | {action:<6} | {rec:<3} |"
        )

    _logger.info("-" * width)


def _ic_tstat(ic: float, n: int) -> float:
    """Return IC t-statistic using Fisher approximation with df correction."""
    if n < 3 or not np.isfinite(ic):
        return 0.0
    denom = 1.0 - ic * ic
    if denom <= 0.0:
        return float("inf") if ic > 0.0 else float("-inf")
    return float(ic * np.sqrt(n - 2) / np.sqrt(denom))


def _recommendation_threshold_checks(row: pd.Series, cfg: CandidateStrategyConfig) -> dict[str, bool]:
    is_signal_cell = str(row.get("group", "")).startswith("cell=")
    min_obs = cfg.min_signal_cell_oos_obs if is_signal_cell else cfg.min_variant_oos_obs
    max_event_fraction = (
        min(cfg.max_signal_cell_event_fraction_per_bar, cfg.max_variant_event_fraction_per_bar)
        if is_signal_cell
        else cfg.max_variant_event_fraction_per_bar
    )
    edge_stability_bps = float(row.get("edge_stability_bps", float("nan")))
    edge_decay_ok = (not np.isfinite(edge_stability_bps)) or edge_stability_bps >= -cfg.max_oos_edge_decay_bps
    hit_or_payoff_ok = (
        float(row.get("oos_pct_edge_pos", 0.0)) >= cfg.min_variant_oos_hit_rate
        or float(row.get("oos_payoff_ratio", 0.0)) >= cfg.min_variant_oos_payoff_ratio
    )
    # Archetype-aware median gate: right-skewed payoff archetypes (trend/momentum)
    # structurally exhibit median<0 + mean>0; exempt from the absolute median floor.
    archetype = str(row.get("archetype", ""))
    median_exempt = archetype in cfg.median_gate_skew_exempt_archetypes
    median_ok = median_exempt or (
        float(row.get("oos_median_edge_bps", float("nan"))) >= cfg.min_variant_oos_median_edge_bps
    )

    # p10 floor: relative to stop size when p10_edge_relative_to_stop is True.
    # avg_stop_loss_bps is derived from stop_atr_mult if present; falls back to
    # the absolute threshold otherwise.
    if cfg.p10_edge_relative_to_stop:
        stop_atr_mult_val = float(row.get("stop_atr_mult", 0.0))
        if stop_atr_mult_val > 0.0:
            # Use stop_atr_mult as a proxy for stop distance in bps units.
            # The multiplier itself is dimensionless; treat it as bps proxy
            # (e.g. stop_atr_mult=1.5 -> floor = -(1.5 * 1.5 * 100) = -225bps).
            # If avg_stop_loss_bps is available use it directly.
            avg_stop_bps = float(row.get("avg_stop_loss_bps", stop_atr_mult_val * 100.0))
            p10_floor = -(cfg.p10_min_fraction_of_stop * abs(avg_stop_bps))
        else:
            p10_floor = cfg.min_variant_oos_p10_edge_bps
    else:
        p10_floor = cfg.min_variant_oos_p10_edge_bps
    p10_ok = float(row.get("oos_p10_edge_bps", float("nan"))) >= p10_floor

    # regime_edge is diagnostic-only when promotion_level == "variant" to avoid
    # triple-penalty (masking + fragmentation + gating).
    regime_ok = (
        bool(row.get("regime_pass", False))
        if (cfg.regime_diagnostic_enabled and cfg.promotion_level == "signal_cell")
        else True
    )
    _oos_rank_ic = float(row.get("oos_rank_ic", float("nan")))
    oos_rank_ic_ok = np.isfinite(_oos_rank_ic) and _oos_rank_ic >= cfg.min_oos_rank_ic
    _oos_n = int(row.get("oos_n", 0))
    # IC t-stat must use the OOS-window N the IC was computed on. `oos_ic_n` is populated
    # when the OOS-window IC was substituted in; fall back to `oos_n` otherwise.
    _oos_ic_n = int(row.get("oos_ic_n", _oos_n))
    ic_tstat_ok = _ic_tstat(_oos_rank_ic, _oos_ic_n) >= cfg.min_ic_tstat

    checks = {
        "min_obs": _oos_n >= min_obs,
        "breakeven_hard_gate": (
            (not cfg.standalone_breakeven_hard_gate_enabled)
            or bool(row.get("breakeven_hard_pass", False))
        ),
        "mean_edge": float(row.get("oos_mean_edge_bps", float("nan"))) >= cfg.min_variant_oos_edge_bps,
        "median_edge": median_ok,
        "p10_edge": p10_ok,
        "q10_fail": float(row.get("oos_q10_shortfall_fail_rate", 1.0)) <= cfg.max_variant_oos_q10_fail_rate,
        "event_density": float(row.get("event_fraction_per_bar", 1.0)) <= max_event_fraction,
        "regime_edge": regime_ok,
        "edge_decay": edge_decay_ok,
        "hit_or_payoff": hit_or_payoff_ok,
        "oos_rank_ic": oos_rank_ic_ok,
        "ic_tstat": ic_tstat_ok,
        "exit_policy": bool(str(row.get("exit_policy_id", ""))) if is_signal_cell else True,
    }
    # Regime-cell OR-path: Bayesian admission already verified per-cell statistical rigor
    # (NW-corrected posterior probability ≥ min_admission_posterior_prob), so
    # global pooled obs and edge gates are overridden.
    # Hard safety gates (q10_fail, event_density) remain mandatory in both paths.
    if cfg.regime_cell_admission_enabled and bool(row.get("regime_cell_admitted", False)):
        for key in ("min_obs", "breakeven_hard_gate", "mean_edge", "median_edge", "p10_edge",
                    "regime_edge", "edge_decay", "hit_or_payoff", "oos_rank_ic", "ic_tstat"):
            checks[key] = True
    return checks


def _meets_recommendation_thresholds(row: pd.Series, cfg: CandidateStrategyConfig) -> bool:
    return all(_recommendation_threshold_checks(row, cfg).values())


def _failed_recommendation_checks(row: pd.Series, cfg: CandidateStrategyConfig) -> tuple[str, ...]:
    checks = _recommendation_threshold_checks(row, cfg)
    return tuple(name for name, passed in checks.items() if not passed)


def summarize_recommendation_gate_failures(
    recommendation_summary: pd.DataFrame,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    """Return one row per recommendation group with pass/fail checks."""
    if recommendation_summary.empty:
        return pd.DataFrame(
            columns=["group", "candidate_action", "recommended", "failed_checks"]
        )

    rows: list[dict[str, Any]] = []
    for row in recommendation_summary.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        checks = _recommendation_threshold_checks(row_series, cfg)
        failed_checks = tuple(name for name, passed in checks.items() if not passed)
        rows.append(
            {
                "group": str(getattr(row, "group", "")),
                "candidate_action": str(getattr(row, "candidate_action", "")),
                "recommended": len(failed_checks) == 0,
                "failed_checks": failed_checks,
                "oos_mean_edge_bps": float(getattr(row, "oos_mean_edge_bps", float("nan"))),
                "variant": str(getattr(row, "variant", "")),
                **checks,
            }
        )
    return pd.DataFrame.from_records(rows)


def _log_recommendation_failure_block(summary: pd.DataFrame, *, cfg: CandidateStrategyConfig, top_k: int) -> None:
    """Emit a single-line summary of why variants failed recommendation thresholds."""
    if summary.empty:
        return

    blocked = summarize_recommendation_gate_failures(summary, cfg)
    blocked = blocked.loc[~blocked["recommended"]].copy()
    if blocked.empty:
        return

    failure_counts: dict[str, int] = {}
    for failed in blocked["failed_checks"]:
        for name in failed:
            failure_counts[name] = failure_counts.get(name, 0) + 1

    ordered_counts = " | ".join(f"{name}x{count}" for name, count in sorted(failure_counts.items()))
    top_blocked = (
        blocked.sort_values(["oos_mean_edge_bps", "group"], ascending=[False, True])
        .head(top_k)
    )
    top_names = ", ".join(
        str(getattr(r, "variant", str(r.group).removeprefix("variant=")))
        for r in top_blocked.itertuples(index=False)
    )
    _logger.debug(
        "[DIAG][RECOMMEND_BLOCKED] n=%d  fail_gates: %s  top: %s",
        len(blocked),
        ordered_counts,
        top_names,
    )


def _variant_group_to_key(group: str) -> str:
    payload = str(group)
    if payload.startswith("cell="):
        return payload
    payload = payload.removeprefix("variant=")
    family, variant = payload.split(":", 1)
    return f"variant={family}:{variant}"


def _build_recommendations(
    *,
    by_variant: pd.DataFrame,
    flipped_by_variant: pd.DataFrame,
    cfg: CandidateStrategyConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return keep and flip recommendations from OOS metrics."""
    keep_groups: list[str] = []
    for _, row in by_variant.iterrows():
        action = str(row.get("candidate_action", ""))
        cell_admitted = cfg.regime_cell_admission_enabled and bool(row.get("regime_cell_admitted", False))
        # Regime-cell admitted signals bypass the action pre-filter: they may be INSUFFICIENT_OBS
        # at the global pooled level but have Bayesian evidence at the cell level.
        if action != "KEEP_CANDIDATE" and not cell_admitted:
            continue
        if _meets_recommendation_thresholds(row, cfg):
            keep_groups.append(_variant_group_to_key(str(row.get("group", ""))))
    recommended_keep = tuple(keep_groups)

    flipped_lookup = flipped_by_variant.set_index("group") if not flipped_by_variant.empty else pd.DataFrame()
    flip_groups: list[str] = []
    for row in by_variant.itertuples(index=False):
        if str(row.candidate_action) != "SIDE_FLIP_CANDIDATE":
            continue
        if str(row.group) not in getattr(flipped_lookup, "index", []):
            continue
        flip_row = flipped_lookup.loc[str(row.group)]
        if isinstance(flip_row, pd.DataFrame):
            continue
        if _meets_recommendation_thresholds(pd.Series(flip_row), cfg):
            flip_groups.append(_variant_group_to_key(str(row.group)))

    return recommended_keep, tuple(flip_groups)


def _split_recommendation_groups(groups: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    signal_cells = tuple(str(group).removeprefix("cell=") for group in groups if str(group).startswith("cell="))
    variants = tuple(str(group).removeprefix("variant=") for group in groups if str(group).startswith("variant="))
    return signal_cells, variants


def _log_side_flip_block(side_flip: pd.DataFrame) -> None:
    """Emit a compact diagnostic log for side-flip comparisons."""
    for row in side_flip.itertuples(index=False):
        _logger.debug(
            "[DIAG][RULE_SIDE_FLIP] group=%s n=%d orig_mean=%.1f flip_mean=%.1f delta=%.1f action=%s",
            str(row.group),
            int(row.n),
            float(row.orig_mean_edge_bps),
            float(row.flip_mean_edge_bps),
            float(row.delta_mean_edge_bps),
            str(row.candidate_action),
        )


def _log_decision_block(decision: dict[str, float | int | str]) -> None:
    """Emit a compact decision summary."""
    _logger.debug(
        "[DIAG][RULE_DECISION] keep=%d flip=%d drop=%d insufficient=%d best_mean_edge=%.1f",
        int(decision.get("keep", 0)),
        int(decision.get("flip", 0)),
        int(decision.get("drop", 0)),
        int(decision.get("insufficient", 0)),
        float(decision.get("best_mean_edge", float("nan"))),
    )


def compute_rule_diagnostics(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    min_obs: int = 100,
    silent: bool = False,
    recommendation_start: int | None = None,
    recommendation_end: int | None = None,
    report_start: int | None = None,
    report_end: int | None = None,
) -> RuleDiagnosticsResult:
    """Compute family/variant/side diagnostics for rule alpha events."""
    if not labeled_events.empty:
        labeled_events = labeled_events.copy()
        numeric_cols_float = ["edge_after_hurdle_bps", "mae_bps", "mfe_bps", "raw_score", "side", "score_z"]
        for col in numeric_cols_float:
            if col in labeled_events.columns:
                labeled_events[col] = pd.to_numeric(labeled_events[col], errors="coerce").astype(np.float64)
        if "entry_idx" in labeled_events.columns:
            labeled_events["entry_idx"] = pd.to_numeric(labeled_events["entry_idx"], errors="coerce").astype(np.int64)

    if labeled_events.empty:
        empty = pd.DataFrame(
            columns=[
                "group",
                "n",
                "long_n",
                "short_n",
                "train_n",
                "oos_n",
                "train_mean_edge_bps",
                "oos_mean_edge_bps",
                "train_pct_edge_pos",
                "oos_pct_edge_pos",
                "train_payoff_ratio",
                "oos_payoff_ratio",
                "train_q10_shortfall_fail_rate",
                "oos_q10_shortfall_fail_rate",
                "edge_stability_bps",
                "mean_edge_bps",
                "median_edge_bps",
                "pct_edge_pos",
                "p10_edge_bps",
                "p90_edge_bps",
                "mean_mae_bps",
                "mean_mfe_bps",
                "payoff_ratio",
                "spearman_score_edge",
                "spearman_abs_score_edge",
                "q10_shortfall_fail_rate",
                "candidate_action",
            ]
        )
        empty_side = pd.DataFrame(
            columns=[
                "group",
                "n",
                "orig_mean_edge_bps",
                "flip_mean_edge_bps",
                "delta_mean_edge_bps",
                "orig_pct_edge_pos",
                "flip_pct_edge_pos",
                "candidate_action",
            ]
        )
        empty_decision: dict[str, float | int | str] = {
            "keep": 0,
            "flip": 0,
            "drop": 0,
            "insufficient": 0,
            "best_group": "",
            "best_mean_edge": float("nan"),
        }
        return RuleDiagnosticsResult(
            empty,
            empty,
            empty,
            empty_side,
            empty_decision,
            (),
            (),
            (),
            (),
            "legacy_oos",
            (0, 0),
            (0, 0),
        )

    required = {
        "family",
        "variant",
        "side",
        "raw_score",
        "score_z",
        "entry_idx",
        "expected_holding_bars",
        "stop_atr_mult",
        "take_profit_atr_mult",
        "edge_after_hurdle_bps",
        "profitable_after_hurdle_label",
        "mae_bps",
        "mfe_bps",
    }
    missing = required.difference(labeled_events.columns)
    if missing:
        raise ValueError(f"missing required diagnostic columns: {sorted(missing)}")

    n_bars = int(aligned.close_2d.shape[0])
    resolved_report_start, resolved_report_end = _resolve_report_window(
        n_bars=n_bars,
        report_start=report_start,
        report_end=report_end,
    )
    resolved_recommendation_start = recommendation_start
    resolved_recommendation_end = recommendation_end
    recommendation_basis = "legacy_oos"
    if resolved_recommendation_start is None or resolved_recommendation_end is None:
        resolved_recommendation_start = resolved_report_start
        resolved_recommendation_end = resolved_report_end
    else:
        recommendation_basis = str(cfg.promotion_decision_split)

    by_family = _summarize_view(
        events=labeled_events,
        view="family",
        min_obs=min_obs,
        cfg=cfg,
        report_start=resolved_report_start,
        report_end=resolved_report_end,
    )
    by_variant = _summarize_view(
        events=labeled_events,
        view="variant",
        min_obs=min_obs,
        cfg=cfg,
        report_start=resolved_report_start,
        report_end=resolved_report_end,
    )
    by_family_side = _summarize_view(
        events=labeled_events,
        view="family_side",
        min_obs=min_obs,
        cfg=cfg,
        report_start=resolved_report_start,
        report_end=resolved_report_end,
    )

    flipped = labeled_events.copy()
    flipped["side"] = -flipped["side"]
    flipped_labeled = label_candidate_events(events=flipped, aligned=aligned, cfg=cfg)

    if not flipped_labeled.empty:
        flipped_labeled = flipped_labeled.copy()
        numeric_cols_float = ["edge_after_hurdle_bps", "mae_bps", "mfe_bps", "raw_score", "side", "score_z"]
        for col in numeric_cols_float:
            if col in flipped_labeled.columns:
                flipped_labeled[col] = pd.to_numeric(flipped_labeled[col], errors="coerce").astype(np.float64)
        if "entry_idx" in flipped_labeled.columns:
            flipped_labeled["entry_idx"] = pd.to_numeric(flipped_labeled["entry_idx"], errors="coerce").astype(np.int64)

    side_flip_frames = [
        _summarize_side_flip(
            original=labeled_events,
            flipped=flipped_labeled,
            view="family",
            min_obs=min_obs,
            cfg=cfg,
            report_start=resolved_report_start,
            report_end=resolved_report_end,
        ),
        _summarize_side_flip(
            original=labeled_events,
            flipped=flipped_labeled,
            view="variant",
            min_obs=min_obs,
            cfg=cfg,
            report_start=resolved_report_start,
            report_end=resolved_report_end,
        ),
        _summarize_side_flip(
            original=labeled_events,
            flipped=flipped_labeled,
            view="family_side",
            min_obs=min_obs,
            cfg=cfg,
            report_start=resolved_report_start,
            report_end=resolved_report_end,
        ),
    ]
    side_flip = pd.concat(side_flip_frames, axis=0, ignore_index=True) if side_flip_frames else pd.DataFrame()
    side_flip_lookup = {
        str(row.group): row
        for row in side_flip.itertuples(index=False)
    } if not side_flip.empty else {}

    def _apply_action(table: pd.DataFrame) -> pd.DataFrame:
        if table.empty:
            return table
        actions: list[str] = []
        for row in table.itertuples(index=False):
            flip_row = side_flip_lookup.get(str(row.group))
            _oos_n = int(getattr(row, "oos_n", 0))
            _full_edge = float(row.mean_edge_bps)
            _full_pct = float(row.pct_edge_pos)
            _full_payoff = float(row.payoff_ratio)
            _full_q10 = float(row.q10_shortfall_fail_rate)

            def _pick(oos_val: float, full_val: float, has_oos: bool) -> float:
                return (oos_val if np.isfinite(oos_val) else full_val) if has_oos else full_val

            _has_oos = _oos_n > 0
            action = _candidate_action(
                n=int(row.n),
                min_obs=min_obs,
                mean_edge_bps=_pick(float(getattr(row, "oos_mean_edge_bps", float("nan"))), _full_edge, _has_oos),
                pct_edge_pos=_pick(float(getattr(row, "oos_pct_edge_pos", float("nan"))), _full_pct, _has_oos),
                payoff_ratio=_pick(float(getattr(row, "oos_payoff_ratio", float("nan"))), _full_payoff, _has_oos),
                q10_shortfall_fail_rate=_pick(
                    float(getattr(row, "oos_q10_shortfall_fail_rate", float("nan"))), _full_q10, _has_oos
                ),
                min_hit_rate=cfg.min_rule_hit_rate,
                min_payoff_ratio=cfg.min_variant_oos_payoff_ratio,
                max_q10_fail_rate=cfg.max_variant_oos_q10_fail_rate,
                flip_delta_bps=float(flip_row.delta_mean_edge_bps) if flip_row is not None else None,
                flip_mean_edge_bps=float(flip_row.flip_mean_edge_bps) if flip_row is not None else None,
                train_mean_edge_bps=float(
                    getattr(row, "train_mean_edge_bps", float("nan"))
                    if recommendation_start is None or recommendation_end is None
                    else float("nan")
                ),
            )
            actions.append(action)
        updated = table.copy()
        updated["candidate_action"] = actions
        return updated

    by_family = _apply_action(by_family)
    by_variant = _apply_action(by_variant)
    by_family_side = _apply_action(by_family_side)
    side_flip_rec_lookup = {
        str(row.group): (float(row.delta_mean_edge_bps), float(row.flip_mean_edge_bps))
        for row in side_flip.itertuples(index=False)
    } if not side_flip.empty else {}
    recommendation_variant_summary = _summarize_recommendation_variants(
        events=labeled_events,
        aligned=aligned,
        min_obs=min_obs,
        cfg=cfg,
        recommendation_start=resolved_recommendation_start,
        recommendation_end=resolved_recommendation_end,
        side_flip_lookup=side_flip_rec_lookup,
    )
    recommendation_flipped_summary = _summarize_recommendation_variants(
        events=flipped_labeled,
        aligned=aligned,
        min_obs=min_obs,
        cfg=cfg,
        recommendation_start=resolved_recommendation_start,
        recommendation_end=resolved_recommendation_end,
    )
    # Replace rec-window IC with OOS-window IC from by_variant (report_start..report_end).
    # Recommendation window = IS+calibration, so its Spearman is in-sample and noisy for
    # pattern signals. by_variant["oos_rank_ic"] is computed on the same OOS window used
    # for all other OOS metrics and is the correct source for the IC gate.
    # We also carry by_variant's OOS sample count as `oos_ic_n` so the IC t-stat uses the
    # N the IC was actually computed on (rec-window `oos_n` is the IS count and would
    # inflate the t-stat).
    if not recommendation_variant_summary.empty and not by_variant.empty and "oos_rank_ic" in by_variant.columns:
        _bv_indexed = by_variant.set_index("group")
        _ic_map: pd.Series = _bv_indexed["oos_rank_ic"]
        recommendation_variant_summary["oos_rank_ic"] = (
            recommendation_variant_summary["group"].map(_ic_map)
            .fillna(recommendation_variant_summary["oos_rank_ic"])
        )
        if "oos_n" in _bv_indexed.columns:
            _n_map: pd.Series = _bv_indexed["oos_n"]
            recommendation_variant_summary["oos_ic_n"] = (
                recommendation_variant_summary["group"].map(_n_map)
                .fillna(recommendation_variant_summary["oos_n"])
                .astype(int)
            )

    recommended_keep_variants, recommended_flip_variants = _build_recommendations(
        by_variant=recommendation_variant_summary,
        flipped_by_variant=recommendation_flipped_summary,
        cfg=cfg,
    )
    recommended_keep_signal_cells, recommended_keep_variants = _split_recommendation_groups(
        recommended_keep_variants
    )
    recommended_flip_signal_cells, recommended_flip_variants = _split_recommendation_groups(
        recommended_flip_variants
    )

    decision_counts = by_variant["candidate_action"].value_counts() if not by_variant.empty else pd.Series(dtype=int)
    best_row = by_variant.iloc[0] if not by_variant.empty else None
    decision: dict[str, float | int | str] = {
        "keep": int(decision_counts.get("KEEP_CANDIDATE", 0)),
        "flip": int(decision_counts.get("SIDE_FLIP_CANDIDATE", 0)),
        "drop": int(decision_counts.get("DROP_OR_REWORK", 0)),
        "insufficient": int(decision_counts.get("INSUFFICIENT_OBS", 0)),
        "best_group": str(best_row.group) if best_row is not None else "",
        "best_mean_edge": float(best_row.mean_edge_bps) if best_row is not None else float("nan"),
    }
    keep_variant_keys = {f"variant={value}" for value in recommended_keep_variants}
    keep_archetypes = sorted(
        {
            str(row.archetype)
            for row in recommendation_variant_summary.itertuples(index=False)
            if str(row.group) in keep_variant_keys
        }
    )
    decision["keep_archetypes"] = ",".join(keep_archetypes)
    decision["keep_archetype_count"] = len(keep_archetypes)
    decision["keep_has_trend"] = int(any(a in {"trend_continuation", "time_series_momentum"} for a in keep_archetypes))
    decision["keep_has_reversion"] = int(any("reversion" in a for a in keep_archetypes))

    recommendation_failure_report: dict[str, Any] | None = None
    if not silent:
        _log_summary_block(summary=by_family, view="family")
        _log_summary_block(summary=by_variant, view="variant")
        _recommended_set = frozenset(
            list(recommended_keep_variants)
            + [f"variant={v}" for v in recommended_keep_variants if not v.startswith("variant=")]
        )
        _log_variant_top_block(
            by_variant,
            top_k=cfg.diagnostic_top_k,
            recommended_variants=_recommended_set,
        )
        failure_summary = summarize_recommendation_gate_failures(recommendation_variant_summary, cfg)
        blocked_df = failure_summary.loc[~failure_summary["recommended"]].copy()
        if not blocked_df.empty:
            failure_counts: dict[str, int] = {}
            for failed in blocked_df["failed_checks"]:
                for name in failed:
                    failure_counts[name] = failure_counts.get(name, 0) + 1
            ordered_counts = " | ".join(f"{name}x{count}" for name, count in sorted(failure_counts.items()))
            top_blocked = (
                blocked_df
                .sort_values(["oos_mean_edge_bps", "group"], ascending=[False, True])
                .head(cfg.diagnostic_top_k)
            )
            top_names = ", ".join(
                str(getattr(r, "variant", str(r.group).removeprefix("variant=")))
                for r in top_blocked.itertuples(index=False)
            )
            recommendation_failure_report = {
                "n_blocked": len(blocked_df),
                "fail_gates_str": ordered_counts,
                "top_blocked_str": top_names,
                "failure_counts": failure_counts,
                "rows": [
                    {
                        "group": str(r.group),
                        "candidate_action": str(r.candidate_action),
                        "failed_checks": list(r.failed_checks),
                    }
                    for r in blocked_df.itertuples(index=False)
                ],
            }
        _log_recommendation_failure_block(
            recommendation_variant_summary,
            cfg=cfg,
            top_k=cfg.diagnostic_top_k,
        )
        _log_side_flip_block(side_flip=side_flip)
        _log_decision_block(decision)
        _logger.debug(
            "[DIAG][RULE_RECOMMEND] basis=%s keep=%s flip=%s",
            recommendation_basis,
            ",".join(recommended_keep_variants) if recommended_keep_variants else "none",
            ",".join(recommended_flip_variants) if recommended_flip_variants else "none",
        )

    return RuleDiagnosticsResult(
        by_family=by_family,
        by_variant=by_variant,
        by_family_side=by_family_side,
        side_flip=side_flip,
        decision=decision,
        recommended_keep_variants=recommended_keep_variants,
        recommended_flip_variants=recommended_flip_variants,
        recommended_keep_signal_cells=recommended_keep_signal_cells,
        recommended_flip_signal_cells=recommended_flip_signal_cells,
        recommendation_basis=recommendation_basis,
        recommendation_split=(resolved_recommendation_start, resolved_recommendation_end),
        report_split=(resolved_report_start, resolved_report_end),
        recommendation_failure_report=recommendation_failure_report,
    )
