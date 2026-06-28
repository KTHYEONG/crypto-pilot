from __future__ import annotations

from typing import Literal

import numpy as np

from src.domain.futures.strategy.tiered_workflow.dataclasses import RegimeBucketReliability


def build_bucket_reliability(
    *,
    regime: int,
    family: str,
    tf: str,
    fit_edge_bps: float,
    cal_edge_bps: float,
    n_fit: int,
    n_cal: int,
    min_fit_n: int,
    min_cal_n: int,
    min_cal_lift_bps: float,
    min_reliability: float,
    relaxed_reliability_threshold: float = 0.35,
) -> RegimeBucketReliability:
    sign_consistent = (
        fit_edge_bps == 0.0
        or cal_edge_bps == 0.0
        or np.sign(fit_edge_bps) == np.sign(cal_edge_bps)
    )
    fit_ratio = min(float(max(n_fit, 0)) / max(float(min_fit_n), 1.0), 1.0)
    cal_ratio = min(float(max(n_cal, 0)) / max(float(min_cal_n), 1.0), 1.0)
    cal_lift_ratio = min(abs(float(cal_edge_bps)) / max(float(min_cal_lift_bps), 1e-9), 1.0)
    reliability = float(fit_ratio * cal_ratio * cal_lift_ratio)
    if (
        n_fit >= min_fit_n
        and n_cal >= min_cal_n
        and sign_consistent
        and abs(float(cal_edge_bps)) >= float(min_cal_lift_bps)
        and reliability >= min_reliability
    ):
        action: Literal["allow", "downweight", "pool"] = "allow"
    elif not sign_consistent or n_cal < min_cal_n:
        action = "pool"
    else:
        action = "downweight"
    # C: relaxed threshold: downweight → allow if sign-consistent and >= relaxed threshold
    if action == "downweight" and reliability >= relaxed_reliability_threshold and sign_consistent:
        action = "allow"
    return RegimeBucketReliability(
        regime=regime,
        family=family,
        tf=tf,
        fit_edge_bps=float(fit_edge_bps),
        cal_edge_bps=float(cal_edge_bps),
        n_fit=int(n_fit),
        n_cal=int(n_cal),
        sign_consistent=bool(sign_consistent),
        reliability=float(reliability),
        action=action,
    )
