"""Stage 2: data quality checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Stage2Config


def apply_data_quality_stage(
    frame: pd.DataFrame,
    config: Stage2Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter symbols by continuity and data validity constraints."""
    if frame.empty:
        return frame.copy(), pd.DataFrame(columns=["symbol", "stage", "passed", "reason"])
    cfg = config or Stage2Config()

    coverage = frame.get("last_60d_coverage", pd.Series(0.0, index=frame.index)).fillna(0.0)
    zero_bars = frame.get("n_zero_volume_bars_60d", pd.Series(999, index=frame.index)).fillna(999)
    frozen = frame.get("frozen_bars", pd.Series(999, index=frame.index)).fillna(999)
    is_coverage = frame.get("is_coverage", pd.Series(np.nan, index=frame.index))
    n_is_bars = pd.to_numeric(
        frame.get("n_is_bars", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    expected_is_bars = pd.to_numeric(
        frame.get("expected_is_bars", pd.Series(float(cfg.min_is_bars_4h), index=frame.index)),
        errors="coerce",
    ).fillna(float(cfg.min_is_bars_4h))
    gap_count = pd.to_numeric(
        frame.get("n_bar_gaps", pd.Series(0, index=frame.index)),
        errors="coerce",
    ).fillna(0)
    gap_size = pd.to_numeric(
        frame.get("max_gap_bars", pd.Series(0, index=frame.index)),
        errors="coerce",
    ).fillna(0)
    has_nan = frame.get("has_nan", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    has_inf = frame.get("has_inf", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    has_timestamp_issues = frame.get(
        "has_timestamp_issues", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)
    has_kline = frame.get("has_kline", pd.Series(False, index=frame.index)).fillna(False)
    has_funding = frame.get("has_funding", pd.Series(False, index=frame.index)).fillna(False)
    resolved_is_coverage = pd.to_numeric(is_coverage, errors="coerce").where(
        pd.to_numeric(is_coverage, errors="coerce").notna(),
        (n_is_bars / expected_is_bars).clip(lower=0.0, upper=1.0),
    ).fillna(0.0)
    min_is_bars_mask = n_is_bars.fillna(0.0) >= float(cfg.min_is_bars_4h)

    pass_mask = (
        has_kline.astype(bool)
        & has_funding.astype(bool)
        & min_is_bars_mask
        & (resolved_is_coverage >= cfg.min_is_coverage)
        & (coverage >= cfg.min_coverage_60d)
        & (zero_bars <= cfg.max_zero_volume_bars_60d)
        & (frozen <= cfg.max_frozen_bars_60d)
        & (gap_count <= 1)
        & (gap_size <= cfg.max_gap_bars)
        & (~has_nan)
        & (~has_inf)
        & (~has_timestamp_issues)
    )
    reasons = np.where(~has_kline, "missing_kline", "")
    reasons = np.where((reasons == "") & (~has_funding), "missing_funding", reasons)
    reasons = np.where(
        (reasons == "") & (~min_is_bars_mask),
        "insufficient_is_bars",
        reasons,
    )
    reasons = np.where(
        (reasons == "") & (resolved_is_coverage < cfg.min_is_coverage),
        "insufficient_is_coverage",
        reasons,
    )
    reasons = np.where(
        (reasons == "") & (coverage < cfg.min_coverage_60d),
        "insufficient_coverage_60d",
        reasons,
    )
    reasons = np.where(
        (reasons == "") & (zero_bars > cfg.max_zero_volume_bars_60d),
        "too_many_zero_volume_bars",
        reasons,
    )
    reasons = np.where(
        (reasons == "") & (frozen > cfg.max_frozen_bars_60d),
        "too_many_frozen_bars",
        reasons,
    )
    reasons = np.where((reasons == "") & (gap_count > 1), "too_many_gaps", reasons)
    reasons = np.where((reasons == "") & (gap_size > cfg.max_gap_bars), "gap_too_wide", reasons)
    reasons = np.where((reasons == "") & has_nan, "nan_detected", reasons)
    reasons = np.where((reasons == "") & has_inf, "inf_detected", reasons)
    reasons = np.where(
        (reasons == "") & has_timestamp_issues,
        "timestamp_integrity_violation",
        reasons,
    )
    reasons = pd.Series(np.where(reasons == "", "pass", reasons), index=frame.index, dtype="string")

    report = pd.DataFrame(
        {
            "symbol": frame["symbol"].astype("string"),
            "stage": "stage2_data_quality",
            "passed": pass_mask.astype(bool),
            "reason": reasons,
        }
    )
    return frame.loc[pass_mask].copy(), report
