from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_ADMISSION_SYMBOLS: int = 3
MAX_TREND_SLEEVE_CORR: float = 0.50
MIN_CORR_OVERLAP_BARS: int = 30


@dataclass(frozen=True)
class FamilyAdmissionVerdict:
    family: str
    n_promoted_symbols: int
    min_lcb_bps: float
    trend_sleeve_corr: float | None
    admitted: bool
    reasons: tuple[str, ...]


def compute_trend_sleeve_corr(
    realized_event_results: pd.DataFrame,
    candidate_family: str,
    trend_families: tuple[str, ...],
    *,
    min_overlap_bars: int = MIN_CORR_OVERLAP_BARS,
) -> float | None:
    candidate = realized_event_results[
        realized_event_results["family"] == candidate_family
    ]
    trend = realized_event_results[
        realized_event_results["family"].isin(trend_families)
    ]

    symbols = candidate["symbol"].unique()
    rhos: list[float] = []
    for sym in symbols:
        sym_candidate = candidate[candidate["symbol"] == sym]
        sym_trend = trend[trend["symbol"] == sym]

        cand_series = sym_candidate.groupby("decision_idx")[
            "realized_side_adjusted_gross_bps"
        ].mean()
        trend_series = sym_trend.groupby("decision_idx")[
            "realized_side_adjusted_gross_bps"
        ].mean()

        common_idx = cand_series.index.intersection(trend_series.index)
        if len(common_idx) < min_overlap_bars:
            continue

        r = float(np.corrcoef(cand_series.loc[common_idx], trend_series.loc[common_idx])[0, 1])
        rhos.append(r)

    if not rhos:
        return None
    return float(np.mean(rhos))


def evaluate_family_admission(
    promotions: pd.DataFrame,
    family: str,
    *,
    trend_sleeve_corr: float | None = None,
    min_symbols: int = MIN_ADMISSION_SYMBOLS,
    min_lcb_bps: float = 0.0,
    max_trend_corr: float = MAX_TREND_SLEEVE_CORR,
) -> FamilyAdmissionVerdict:
    family_rows = promotions[promotions["family"] == family]
    mask = family_rows["lcb_bps"] > min_lcb_bps
    promoted = family_rows.loc[mask, "symbol"].unique()
    n_promoted = len(promoted)

    min_lcb = float(family_rows.loc[mask, "lcb_bps"].min()) if n_promoted > 0 else float(np.nan)

    reasons: list[str] = []
    if n_promoted < min_symbols:
        reasons.append("n_symbols_lt_3")

    if trend_sleeve_corr is None:
        reasons.append("trend_corr_unavailable")
    elif trend_sleeve_corr > max_trend_corr:
        reasons.append("trend_corr_gt_max")

    corr_gate_ok = trend_sleeve_corr is None or trend_sleeve_corr <= max_trend_corr
    admitted = n_promoted >= min_symbols and corr_gate_ok

    return FamilyAdmissionVerdict(
        family=family,
        n_promoted_symbols=n_promoted,
        min_lcb_bps=min_lcb,
        trend_sleeve_corr=trend_sleeve_corr,
        admitted=admitted,
        reasons=tuple(reasons),
    )
